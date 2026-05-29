from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.classifier.dinov2 import (
    UNKNOWN_CLASS_ID,
    ClassPrediction,
    DinoV2KnnClassifier,
    _load_yaml,
    _numeric_label,
)


@dataclass(frozen=True)
class LogisticRegressionRejectSettings:
    enabled: bool
    nearest_percentile: float
    centroid_percentile: float
    min_probability: Optional[float]
    require_both: bool


def _classifier_config(config: dict[str, Any]) -> dict[str, Any]:
    classifier_cfg = config.get("classifier")
    if not isinstance(classifier_cfg, dict):
        raise ValueError("LogisticRegression classifier config section 'classifier' is required")
    return classifier_cfg


def _optional_float(value: Any) -> Optional[float]:
    return float(value) if value is not None else None


def _reject_settings_from_config(config_file: str | Path) -> LogisticRegressionRejectSettings:
    classifier_cfg = _classifier_config(_load_yaml(config_file))
    reject_cfg = classifier_cfg.get("rejector") or {}
    if not isinstance(reject_cfg, dict):
        raise ValueError("LogisticRegression classifier config 'rejector' must be a mapping")
    return LogisticRegressionRejectSettings(
        enabled=bool(reject_cfg.get("enabled", True)),
        nearest_percentile=float(reject_cfg.get("nearest_percentile", 5.0)),
        centroid_percentile=float(reject_cfg.get("centroid_percentile", 5.0)),
        min_probability=_optional_float(reject_cfg.get("min_probability")),
        require_both=bool(reject_cfg.get("require_both", False)),
    )


class LogisticRegressionClassifier(DinoV2KnnClassifier):
    """DINOv2 embedding extractor + supervised LogisticRegression classifier.

    The reference bank is reused as labeled training data. Unknown handling stays
    distance-based because there are no explicit unknown embeddings.
    """

    def __init__(
        self,
        config_file: str | Path,
        device: str = "cuda:0",
        reference_bank: str | Path | None = None,
    ) -> None:
        self.reject_settings = _reject_settings_from_config(config_file)
        self.centroids: dict[int, torch.Tensor] = {}
        self.nearest_thresholds: dict[int, float] = {}
        self.centroid_thresholds: dict[int, float] = {}
        super().__init__(config_file=config_file, device=device, reference_bank=reference_bank)
        self._fit_classifier()
        self._fit_rejector_thresholds()

    def _fit_classifier(self) -> None:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import LabelEncoder
        except ImportError as exc:
            raise RuntimeError(
                "scikit-learn is required for LogisticRegressionClassifier. "
                "Install scikit-learn in the inference environment."
            ) from exc

        embeddings = self.embeddings.detach().cpu().numpy()
        labels = self.labels.detach().cpu().numpy().astype(int)
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(labels)
        self.lr_model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            multi_class="auto",
            solver="lbfgs",
        )
        self.lr_model.fit(embeddings, encoded_labels)

    def _fit_rejector_thresholds(self) -> None:
        labels_cpu = self.labels.detach().cpu()
        embeddings_cpu = self.embeddings.detach().cpu()
        unique_labels = sorted(int(value.item()) for value in torch.unique(labels_cpu))

        for label in unique_labels:
            class_mask = labels_cpu == label
            class_embeddings = F.normalize(embeddings_cpu[class_mask].float(), dim=-1)
            if class_embeddings.shape[0] == 0:
                continue

            centroid = F.normalize(class_embeddings.mean(dim=0, keepdim=True), dim=-1).squeeze(0)
            self.centroids[label] = centroid.to(self.device)

            centroid_similarities = torch.matmul(class_embeddings, centroid).numpy()
            self.centroid_thresholds[label] = float(
                np.percentile(centroid_similarities, self.reject_settings.centroid_percentile)
            )

            if class_embeddings.shape[0] == 1:
                nearest_similarities = centroid_similarities
            else:
                pairwise = torch.matmul(class_embeddings, class_embeddings.T)
                pairwise.fill_diagonal_(-1.0)
                nearest_similarities = pairwise.max(dim=1).values.numpy()
            self.nearest_thresholds[label] = float(
                np.percentile(nearest_similarities, self.reject_settings.nearest_percentile)
            )

    def _class_similarities(self, query: torch.Tensor) -> tuple[dict[int, float], dict[int, float]]:
        query_cpu = query.detach().cpu()
        embeddings_cpu = self.embeddings.detach().cpu()
        labels_cpu = self.labels.detach().cpu()

        centroid_similarities: dict[int, float] = {}
        nearest_similarities: dict[int, float] = {}
        for label in self.centroids:
            centroid = self.centroids[label].detach().cpu()
            centroid_similarities[label] = float(torch.matmul(query_cpu, centroid).item())

            class_embeddings = embeddings_cpu[labels_cpu == label]
            if class_embeddings.shape[0] == 0:
                nearest_similarities[label] = float("-inf")
            else:
                nearest_similarities[label] = float(torch.matmul(query_cpu, class_embeddings.T).max().item())
        return centroid_similarities, nearest_similarities

    def _reject_reason(
        self,
        predicted_label: int,
        probability: float,
        centroid_similarity: float,
        nearest_similarity: float,
    ) -> Optional[str]:
        if not self.reject_settings.enabled:
            return None
        if self.reject_settings.min_probability is not None and probability < self.reject_settings.min_probability:
            return "low_probability"

        nearest_threshold = self.nearest_thresholds.get(predicted_label)
        centroid_threshold = self.centroid_thresholds.get(predicted_label)
        nearest_failed = nearest_threshold is not None and nearest_similarity < nearest_threshold
        centroid_failed = centroid_threshold is not None and centroid_similarity < centroid_threshold

        if self.reject_settings.require_both:
            if nearest_failed and centroid_failed:
                return "distance_reject"
        elif nearest_failed or centroid_failed:
            return "distance_reject"
        return None

    def classify_crop(self, crop_rgb, crop_mask=None) -> ClassPrediction:
        query = self.embed_crop(crop_rgb, crop_mask=crop_mask)
        query_vector = query.reshape(-1)
        query_np = query_vector.detach().cpu().numpy().reshape(1, -1)
        probabilities = self.lr_model.predict_proba(query_np)[0]
        encoded_index = int(np.argmax(probabilities))
        predicted_label = int(self.label_encoder.inverse_transform([encoded_index])[0])
        probability = float(probabilities[encoded_index])

        centroid_similarities, nearest_similarities = self._class_similarities(query_vector)
        centroid_similarity = centroid_similarities.get(predicted_label, float("-inf"))
        nearest_similarity = nearest_similarities.get(predicted_label, float("-inf"))
        sorted_centroid = sorted(centroid_similarities.values(), reverse=True)
        second_centroid = sorted_centroid[1] if len(sorted_centroid) > 1 else float("-inf")
        margin = centroid_similarity - second_centroid if second_centroid != float("-inf") else float("inf")

        reject_reason = self._reject_reason(
            predicted_label,
            probability,
            centroid_similarity,
            nearest_similarity,
        )

        class_index = predicted_label
        class_name = self.label_to_class_name.get(predicted_label, str(predicted_label))
        class_id = _numeric_label(class_name, predicted_label)
        if reject_reason is not None:
            class_index = UNKNOWN_CLASS_ID
            class_name = "unknown"
            class_id = UNKNOWN_CLASS_ID

        neighbor_labels = [int(label) for label in self.label_encoder.inverse_transform(np.argsort(probabilities)[::-1])]
        neighbor_similarities = [float(probabilities[self.label_encoder.transform([label])[0]]) for label in neighbor_labels]

        return ClassPrediction(
            label=class_id,
            class_index=class_index,
            class_name=class_name,
            confidence=probability,
            similarity=float(nearest_similarity),
            vote_ratio=1.0,
            margin=float(margin),
            neighbor_indices=[],
            neighbor_labels=neighbor_labels,
            neighbor_similarities=neighbor_similarities,
            reject_reason=reject_reason,
            class_probabilities={
                int(label): float(probabilities[index])
                for index, label in enumerate(self.label_encoder.inverse_transform(np.arange(len(probabilities))))
            },
            centroid_similarities={int(label): float(value) for label, value in centroid_similarities.items()},
            nearest_similarities={int(label): float(value) for label, value in nearest_similarities.items()},
        )
