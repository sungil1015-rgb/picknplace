from __future__ import annotations

from dataclasses import dataclass
import math
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
    masked_crop_background: int
    cls_weight: float
    patch_mean_weight: float
    patch_mean_source: str
    rgb_weight: Optional[float]
    gray_weight: Optional[float]
    knn_temperature: float
    min_class_score: Optional[float]
    min_similarity: Optional[float]
    min_vote_ratio: Optional[float]
    min_margin: Optional[float]


@dataclass(frozen=True)
class ClassPrediction:
    label: int
    class_index: int
    class_name: str
    confidence: float
    similarity: float
    vote_ratio: float
    margin: float
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
    min_vote_ratio = classifier_cfg.get("min_vote_ratio")
    min_margin = classifier_cfg.get("min_margin")
    cls_weight = float(classifier_cfg.get("cls_weight", 1.0))
    patch_mean_weight = float(classifier_cfg.get("patch_mean_weight", 0.0))
    if cls_weight <= 0.0 and patch_mean_weight <= 0.0:
        raise ValueError("DINOv2 classifier config requires cls_weight or patch_mean_weight to be > 0")
    patch_mean_source = str(classifier_cfg.get("patch_mean_source", "masked")).lower()
    if patch_mean_source not in ("masked", "unmasked"):
        raise ValueError("DINOv2 classifier config 'patch_mean_source' must be 'masked' or 'unmasked'")
    return DinoV2KnnSettings(
        model_name=str(classifier_cfg["model_name"]),
        reference_bank=str(classifier_cfg["reference_bank"]),
        top_k=int(classifier_cfg["top_k"]),
        crop_padding=int(classifier_cfg["crop_padding"]),
        masked_crop=bool(classifier_cfg["masked_crop"]),
        masked_crop_background=int(classifier_cfg.get("masked_crop_background", 127)),
        cls_weight=cls_weight,
        patch_mean_weight=patch_mean_weight,
        patch_mean_source=patch_mean_source,
        rgb_weight=_optional_float(classifier_cfg.get("rgb_weight")),
        gray_weight=_optional_float(classifier_cfg.get("gray_weight")),
        knn_temperature=float(classifier_cfg.get("knn_temperature", 0.05)),
        min_class_score=_optional_float(classifier_cfg.get("min_class_score")),
        min_similarity=float(min_similarity) if min_similarity is not None else None,
        min_vote_ratio=float(min_vote_ratio) if min_vote_ratio is not None else None,
        min_margin=float(min_margin) if min_margin is not None else None,
    )


def _optional_float(value: Any) -> Optional[float]:
    return float(value) if value is not None else None


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


def _labels_to_tensor(labels: Any) -> torch.Tensor:
    if isinstance(labels, torch.Tensor):
        return labels.long()
    if isinstance(labels, np.ndarray):
        return torch.as_tensor(labels).long()
    return torch.as_tensor([int(label) for label in labels]).long()


class DinoV2KnnClassifier:
    """DINOv2 embedding extractor + cosine KNN classifier.

    The reference bank must contain normalized embeddings and integer labels.
    Supported bank format matches weights/dinov2_reference_bank/bank*.pt.
    """

    INSTANCE_CROP_PADDING_RATIO = 0.03
    INSTANCE_CROP_BACKGROUND = 127

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
                masked_crop_background=self.settings.masked_crop_background,
                cls_weight=self.settings.cls_weight,
                patch_mean_weight=self.settings.patch_mean_weight,
                patch_mean_source=self.settings.patch_mean_source,
                rgb_weight=self.settings.rgb_weight,
                gray_weight=self.settings.gray_weight,
                knn_temperature=self.settings.knn_temperature,
                min_class_score=self.settings.min_class_score,
                min_similarity=self.settings.min_similarity,
                min_vote_ratio=self.settings.min_vote_ratio,
                min_margin=self.settings.min_margin,
            )
        self.device = _resolve_device(device)

        self.class_names: list[str] = []
        self.label_to_class_name: dict[int, str] = {}
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
        bank_rgb_weight = float(bank.get("rgb_weight", self.rgb_weight))
        bank_gray_weight = float(bank.get("gray_weight", bank.get("shape_weight", self.gray_weight)))
        self.rgb_weight = self.settings.rgb_weight if self.settings.rgb_weight is not None else bank_rgb_weight
        self.gray_weight = self.settings.gray_weight if self.settings.gray_weight is not None else bank_gray_weight

        if self._has_runtime_fusion_embeddings(bank):
            rgb_embeddings = self._fuse_bank_view_embedding(bank, "rgb")
            gray_embeddings = self._fuse_bank_view_embedding(bank, "gray")
            embeddings = (self.rgb_weight * rgb_embeddings) + (self.gray_weight * gray_embeddings)
            if "labels" in bank:
                labels = _labels_to_tensor(bank["labels"])
            elif "classes" in bank:
                labels = _labels_to_tensor(bank["classes"])
            else:
                raise ValueError("DINOv2 reference bank with rgb/gray embeddings must contain 'labels' or 'classes'")
        elif "embeddings" in bank and "labels" in bank:
            embeddings = torch.as_tensor(bank["embeddings"]).float()
            labels = _labels_to_tensor(bank["labels"])
        elif "fused_embeddings" in bank and "classes" in bank:
            embeddings = torch.as_tensor(bank["fused_embeddings"]).float()
            labels = _labels_to_tensor(bank["classes"])
        else:
            raise ValueError(
                "DINOv2 reference bank must contain either "
                "'embeddings' + 'labels' or 'fused_embeddings' + 'classes'"
            )
        if embeddings.ndim != 2 or labels.ndim != 1 or embeddings.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Invalid DINOv2 reference bank shapes: embeddings={tuple(embeddings.shape)}, labels={tuple(labels.shape)}"
            )

        self.embeddings = F.normalize(embeddings, dim=-1).to(self.device)
        self.labels = labels.to(self.device)
        self.class_names = [str(value) for value in bank.get("class_names", [])]
        unique_labels = sorted(int(value.item()) for value in torch.unique(labels))
        if len(self.class_names) == len(unique_labels):
            self.label_to_class_name = {
                label: class_name for label, class_name in zip(unique_labels, self.class_names)
            }
        else:
            self.label_to_class_name = {label: str(label) for label in unique_labels}

    def _has_runtime_fusion_embeddings(self, bank: dict[str, Any]) -> bool:
        if "rgb_cls_embeddings" in bank and "gray_cls_embeddings" in bank:
            if self.settings.patch_mean_source == "masked":
                return "rgb_patch_mean_embeddings" in bank and "gray_patch_mean_embeddings" in bank
            return (
                "rgb_unmasked_patch_mean_embeddings" in bank
                and "gray_unmasked_patch_mean_embeddings" in bank
            )
        return "rgb_embeddings" in bank and "gray_embeddings" in bank

    def _fuse_bank_view_embedding(self, bank: dict[str, Any], prefix: str) -> torch.Tensor:
        cls_key = f"{prefix}_cls_embeddings"
        patch_key = (
            f"{prefix}_patch_mean_embeddings"
            if self.settings.patch_mean_source == "masked"
            else f"{prefix}_unmasked_patch_mean_embeddings"
        )
        if cls_key in bank and patch_key in bank:
            cls_embedding = torch.as_tensor(bank[cls_key]).float()
            patch_embedding = torch.as_tensor(bank[patch_key]).float()
            embedding = (
                self.settings.cls_weight * cls_embedding
                + self.settings.patch_mean_weight * patch_embedding
            )
            return F.normalize(embedding, dim=-1)

        embedding_key = f"{prefix}_embeddings"
        if embedding_key not in bank:
            raise ValueError(f"DINOv2 reference bank missing '{embedding_key}'")
        return F.normalize(torch.as_tensor(bank[embedding_key]).float(), dim=-1)

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
    ) -> tuple[Image.Image, Optional[np.ndarray]]:
        height, width = image_bgr.shape[:2]
        if mask is not None and np.any(mask):
            mask_bool = mask > 0
            ys, xs = np.where(mask_bool)
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1
        elif bbox is not None:
            x1, y1, x2, y2 = [int(round(float(value))) for value in bbox[:4]]
            mask_bool = None
        else:
            x1, y1, x2, y2 = 0, 0, width, height
            mask_bool = None

        bbox_w = x2 - x1
        bbox_h = y2 - y1
        side = int(math.ceil(max(bbox_w, bbox_h) * (1.0 + self.INSTANCE_CROP_PADDING_RATIO * 2.0)))
        if side <= 0:
            raise ValueError(f"Invalid crop box: {(x1, y1, x2, y2)}")

        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        crop_left = int(math.floor(cx - side * 0.5))
        crop_top = int(math.floor(cy - side * 0.5))
        crop_right = crop_left + side
        crop_bottom = crop_top + side

        src_x1 = max(0, crop_left)
        src_y1 = max(0, crop_top)
        src_x2 = min(width, crop_right)
        src_y2 = min(height, crop_bottom)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid crop box: {(x1, y1, x2, y2)}")

        crop = np.full((side, side, 3), self.INSTANCE_CROP_BACKGROUND, dtype=np.uint8)
        crop_mask_bool = np.zeros((side, side), dtype=bool)

        if src_x2 > src_x1 and src_y2 > src_y1:
            dst_x1 = src_x1 - crop_left
            dst_y1 = src_y1 - crop_top
            dst_x2 = dst_x1 + (src_x2 - src_x1)
            dst_y2 = dst_y1 + (src_y2 - src_y1)

            crop[dst_y1:dst_y2, dst_x1:dst_x2] = image_bgr[src_y1:src_y2, src_x1:src_x2]
            if mask_bool is not None:
                crop_mask_bool[dst_y1:dst_y2, dst_x1:dst_x2] = mask_bool[src_y1:src_y2, src_x1:src_x2]
            else:
                crop_mask_bool[dst_y1:dst_y2, dst_x1:dst_x2] = True

        crop[~crop_mask_bool] = self.INSTANCE_CROP_BACKGROUND
        return _to_pil_rgb(crop), crop_mask_bool.astype(np.uint8)

    def _embed_images(
        self,
        images: Sequence[Image.Image],
        masks: Optional[Sequence[Optional[np.ndarray]]] = None,
    ) -> torch.Tensor:
        inputs = self.processor(images=list(images), return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        tokens = outputs.last_hidden_state
        cls_embedding = tokens[:, 0]
        patch_tokens = tokens[:, 1:]
        patch_mean_embedding = self._masked_patch_mean(
            patch_tokens,
            inputs["pixel_values"].shape[-2:],
            masks,
        )
        embedding = (
            self.settings.cls_weight * cls_embedding
            + self.settings.patch_mean_weight * patch_mean_embedding
        )
        return F.normalize(embedding.float(), dim=-1)

    def _masked_patch_mean(
        self,
        patch_tokens: torch.Tensor,
        image_hw: tuple[int, int],
        masks: Optional[Sequence[Optional[np.ndarray]]],
    ) -> torch.Tensor:
        if self.settings.patch_mean_source != "masked" or masks is None:
            return patch_tokens.mean(dim=1)

        patch_count = int(patch_tokens.shape[1])
        image_h, image_w = int(image_hw[0]), int(image_hw[1])
        grid_h, grid_w = self._patch_grid_shape(patch_count, image_h, image_w)
        if grid_h * grid_w != patch_count:
            return patch_tokens.mean(dim=1)

        means: list[torch.Tensor] = []
        for index, mask in enumerate(masks):
            if mask is None or not np.any(mask):
                means.append(patch_tokens[index].mean(dim=0))
                continue

            mask_float = (mask > 0).astype(np.float32)
            coverage = cv2.resize(mask_float, (grid_w, grid_h), interpolation=cv2.INTER_AREA).reshape(-1)
            selected = torch.as_tensor(coverage > 0.1, device=patch_tokens.device)
            if bool(selected.any()):
                means.append(patch_tokens[index][selected].mean(dim=0))
            else:
                means.append(patch_tokens[index].mean(dim=0))
        return torch.stack(means, dim=0)

    def _patch_grid_shape(self, patch_count: int, image_h: int, image_w: int) -> tuple[int, int]:
        patch_size = int(getattr(getattr(self.model, "config", None), "patch_size", 14))
        grid_h = max(1, image_h // patch_size)
        grid_w = max(1, image_w // patch_size)
        if grid_h * grid_w == patch_count:
            return grid_h, grid_w

        side = int(round(patch_count ** 0.5))
        if side * side == patch_count:
            return side, side
        return 1, patch_count

    def embed_crop(self, crop_rgb: Image.Image, crop_mask: Optional[np.ndarray] = None) -> torch.Tensor:
        gray = crop_rgb.convert("L").convert("RGB")
        rgb_embedding, gray_embedding = self._embed_images([crop_rgb, gray], masks=[crop_mask, crop_mask])
        combined = (self.rgb_weight * rgb_embedding) + (self.gray_weight * gray_embedding)
        return F.normalize(combined, dim=-1)

    def classify_crop(self, crop_rgb: Image.Image, crop_mask: Optional[np.ndarray] = None) -> ClassPrediction:
        query = self.embed_crop(crop_rgb, crop_mask=crop_mask)
        similarities = torch.matmul(query, self.embeddings.T).squeeze(0)
        top_k = min(self.settings.top_k, int(similarities.numel()))
        top_values, top_indices = torch.topk(similarities, k=top_k)
        top_labels = self.labels[top_indices]
        top_weights = torch.softmax(top_values / self.settings.knn_temperature, dim=0)

        scores: dict[int, float] = {}
        counts: dict[int, int] = {}
        similarity_sums: dict[int, float] = {}
        for label_tensor, similarity_tensor, weight_tensor in zip(top_labels, top_values, top_weights):
            label = int(label_tensor.item())
            scores[label] = scores.get(label, 0.0) + float(weight_tensor.item())
            similarity_sums[label] = similarity_sums.get(label, 0.0) + float(similarity_tensor.item())
            counts[label] = counts.get(label, 0) + 1

        class_index = max(scores, key=lambda label: (scores[label], similarity_sums[label], counts[label]))
        class_score = scores[class_index]
        similarity = similarity_sums[class_index] / max(counts[class_index], 1)
        vote_ratio = counts[class_index] / max(float(top_k), 1.0)
        second_similarity = max(
            (similarity_sums[label] / max(counts[label], 1) for label in scores if label != class_index),
            default=float("-inf"),
        )
        margin = similarity - second_similarity if second_similarity != float("-inf") else float("inf")
        confidence = class_score

        class_name = self.label_to_class_name.get(class_index, str(class_index))
        class_id = _numeric_label(class_name, class_index)

        is_unknown = False
        if self.settings.min_class_score is not None and class_score < self.settings.min_class_score:
            is_unknown = True
        if self.settings.min_similarity is not None and similarity < self.settings.min_similarity:
            is_unknown = True
        if self.settings.min_vote_ratio is not None and vote_ratio < self.settings.min_vote_ratio:
            is_unknown = True
        if self.settings.min_margin is not None and margin < self.settings.min_margin:
            is_unknown = True

        if is_unknown:
            class_index = -1
            class_name = "unknown"
            class_id = -1

        return ClassPrediction(
            label=class_id,
            class_index=class_index,
            class_name=class_name,
            confidence=float(confidence),
            similarity=float(similarity),
            vote_ratio=float(vote_ratio),
            margin=float(margin),
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
        crop, crop_mask = self._crop_instance(image_bgr, mask=mask, bbox=bbox)
        return self.classify_crop(crop, crop_mask=crop_mask)

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
