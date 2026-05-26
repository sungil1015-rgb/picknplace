from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass(frozen=True)
class DinoV2KnnSettings:
    model_name: str
    reference_bank: str
    top_k: int
    crop_padding: int
    masked_crop: bool
    min_similarity: Optional[float]


@dataclass(frozen=True)
class ClassPrediction:
    label: int
    class_index: int
    class_name: str
    confidence: float
    similarity: float
    neighbor_indices: list[int]
    neighbor_labels: list[int]
    neighbor_similarities: list[float]


def _load_yaml(path: str | Path | None) -> dict[str, Any]:
    if not path:
        raise ValueError("DINOv2 config_file is required")
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"DINOv2 config file not found: {config_path}")

    import yaml

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"DINOv2 config must be a mapping: {config_path}")
    return data


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return Path.cwd() / resolved


def _resolve_device(device: str) -> str:
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device was requested ({device}) but torch.cuda.is_available() is False")
        if ":" in str(device):
            device_index = int(str(device).split(":", 1)[1])
            if device_index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"CUDA device {device} was requested but only {torch.cuda.device_count()} device(s) are available"
                )
    return device


def _classifier_config(config: dict[str, Any]) -> dict[str, Any]:
    classifier_cfg = config.get("classifier")
    if not isinstance(classifier_cfg, dict):
        raise ValueError("DINOv2 classifier config section 'classifier' is required")
    return classifier_cfg


def _settings_from_config(config_file: str | Path) -> DinoV2KnnSettings:
    config = _load_yaml(config_file)
    classifier_cfg = _classifier_config(config)
    required = ("model_name", "reference_bank", "top_k", "crop_padding", "masked_crop")
    missing = [key for key in required if key not in classifier_cfg]
    if missing:
        raise ValueError(f"DINOv2 classifier config missing required key(s): {missing}")

    min_similarity = classifier_cfg.get("min_similarity")
    return DinoV2KnnSettings(
        model_name=str(classifier_cfg["model_name"]),
        reference_bank=str(classifier_cfg["reference_bank"]),
        top_k=int(classifier_cfg["top_k"]),
        crop_padding=int(classifier_cfg["crop_padding"]),
        masked_crop=bool(classifier_cfg["masked_crop"]),
        min_similarity=float(min_similarity) if min_similarity is not None else None,
    )


def _to_pil_rgb(image: np.ndarray | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if image.ndim == 2:
        return Image.fromarray(image).convert("RGB")
    if image.shape[2] == 3:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if image.shape[2] == 4:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGB))
    raise ValueError(f"Unsupported image shape: {image.shape}")


def _numeric_label(class_name: str, class_index: int) -> int:
    try:
        return int(class_name)
    except ValueError:
        return int(class_index)


class DinoV2KnnClassifier:
    """DINOv2 embedding extractor + cosine KNN classifier.

    The reference bank must contain normalized embeddings and integer labels.
    Supported bank format matches weights/dinov2_reference_bank/bank*.pt.
    """

    def __init__(
        self,
        config_file: str | Path,
        device: str = "cuda:0",
        reference_bank: str | Path | None = None,
    ) -> None:
        self.settings = _settings_from_config(config_file)
        if reference_bank is not None:
            self.settings = DinoV2KnnSettings(
                model_name=self.settings.model_name,
                reference_bank=str(reference_bank),
                top_k=self.settings.top_k,
                crop_padding=self.settings.crop_padding,
                masked_crop=self.settings.masked_crop,
                min_similarity=self.settings.min_similarity,
            )
        self.device = _resolve_device(device)

        self.class_names: list[str] = []
        self.rgb_weight = 0.5
        self.gray_weight = 0.5
        self.embeddings: torch.Tensor
        self.labels: torch.Tensor
        self._load_reference_bank()
        self._load_model()

    def _load_reference_bank(self) -> None:
        bank_path = _resolve_path(self.settings.reference_bank)
        if not bank_path.is_file():
            raise FileNotFoundError(f"DINOv2 reference bank not found: {bank_path}")

        bank = torch.load(bank_path, map_location="cpu")
        if not isinstance(bank, dict):
            raise ValueError(f"Unsupported DINOv2 reference bank format: {bank_path}")
        if "embeddings" not in bank or "labels" not in bank:
            raise ValueError("DINOv2 reference bank must contain 'embeddings' and 'labels'")

        embeddings = bank["embeddings"].float()
        labels = bank["labels"].long()
        if embeddings.ndim != 2 or labels.ndim != 1 or embeddings.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Invalid DINOv2 reference bank shapes: embeddings={tuple(embeddings.shape)}, labels={tuple(labels.shape)}"
            )

        self.embeddings = F.normalize(embeddings, dim=-1).to(self.device)
        self.labels = labels.to(self.device)
        self.class_names = [str(value) for value in bank.get("class_names", [])]
        self.rgb_weight = float(bank.get("rgb_weight", self.rgb_weight))
        self.gray_weight = float(bank.get("gray_weight", self.gray_weight))

    def _load_model(self) -> None:
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(self.settings.model_name)
        self.model = AutoModel.from_pretrained(self.settings.model_name).to(self.device)
        self.model.eval()

    def _crop_instance(
        self,
        image_bgr: np.ndarray,
        mask: Optional[np.ndarray] = None,
        bbox: Optional[np.ndarray] = None,
    ) -> Image.Image:
        height, width = image_bgr.shape[:2]
        if bbox is not None:
            x1, y1, x2, y2 = [int(round(float(value))) for value in bbox[:4]]
        elif mask is not None and np.any(mask):
            ys, xs = np.where(mask > 0)
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1
        else:
            x1, y1, x2, y2 = 0, 0, width, height

        pad = self.settings.crop_padding
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(width, x2 + pad)
        y2 = min(height, y2 + pad)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid crop box: {(x1, y1, x2, y2)}")

        crop = image_bgr[y1:y2, x1:x2].copy()
        if self.settings.masked_crop and mask is not None:
            crop_mask = (mask[y1:y2, x1:x2] > 0).astype(np.uint8)
            background = np.full_like(crop, 255)
            crop = np.where(crop_mask[..., None] > 0, crop, background)
        return _to_pil_rgb(crop)

    def _embed_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=list(images), return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state[:, 0]
        return F.normalize(embedding.float(), dim=-1)

    def embed_crop(self, crop_rgb: Image.Image) -> torch.Tensor:
        gray = crop_rgb.convert("L").convert("RGB")
        rgb_embedding, gray_embedding = self._embed_images([crop_rgb, gray])
        combined = (self.rgb_weight * rgb_embedding) + (self.gray_weight * gray_embedding)
        return F.normalize(combined, dim=-1)

    def classify_crop(self, crop_rgb: Image.Image) -> ClassPrediction:
        query = self.embed_crop(crop_rgb)
        similarities = torch.matmul(query, self.embeddings.T).squeeze(0)
        top_k = min(self.settings.top_k, int(similarities.numel()))
        top_values, top_indices = torch.topk(similarities, k=top_k)
        top_labels = self.labels[top_indices]

        scores: dict[int, float] = {}
        counts: dict[int, int] = {}
        for label_tensor, similarity_tensor in zip(top_labels, top_values):
            label = int(label_tensor.item())
            scores[label] = scores.get(label, 0.0) + float(similarity_tensor.item())
            counts[label] = counts.get(label, 0) + 1

        class_index = max(scores, key=lambda label: (scores[label], counts[label]))
        similarity = scores[class_index] / max(counts[class_index], 1)
        if self.settings.min_similarity is not None and similarity < self.settings.min_similarity:
            class_index = -1
            class_name = "unknown"
            class_id = -1
        else:
            class_name = self.class_names[class_index] if 0 <= class_index < len(self.class_names) else str(class_index)
            class_id = _numeric_label(class_name, class_index)

        confidence = scores.get(class_index, 0.0) / max(float(top_values.sum().item()), 1e-6)
        return ClassPrediction(
            label=class_id,
            class_index=class_index,
            class_name=class_name,
            confidence=float(confidence),
            similarity=float(similarity),
            neighbor_indices=[int(index.item()) for index in top_indices],
            neighbor_labels=[int(label.item()) for label in top_labels],
            neighbor_similarities=[float(value.item()) for value in top_values],
        )

    def classify_instance(
        self,
        image_bgr: np.ndarray,
        mask: Optional[np.ndarray] = None,
        bbox: Optional[np.ndarray] = None,
    ) -> ClassPrediction:
        crop = self._crop_instance(image_bgr, mask=mask, bbox=bbox)
        return self.classify_crop(crop)

    def classify_instances(self, image_bgr: np.ndarray, instances: Sequence[Any]) -> list[ClassPrediction]:
        predictions: list[ClassPrediction] = []
        for instance in instances:
            predictions.append(
                self.classify_instance(
                    image_bgr,
                    mask=getattr(instance, "mask", None),
                    bbox=getattr(instance, "bbox", None),
                )
            )
        return predictions
