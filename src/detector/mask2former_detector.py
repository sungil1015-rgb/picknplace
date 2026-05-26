from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

import cv2
import numpy as np
from PIL import Image

from src.detector import InstanceData
from src.utils.crop import crop_image, normalize_crop_box, uncrop_instance_masks


@dataclass(frozen=True)
class Mask2FormerSettings:
    checkpoint: str
    weights: Optional[str]
    size: dict[str, int]
    crop_enabled: bool
    crop_tolerance: float
    prediction_threshold: float
    mask_threshold: float
    overlap_threshold: float
    min_instance_area: int


def _load_yaml(path: str | Path | None) -> dict[str, Any]:
    if not path:
        raise ValueError("Mask2Former config_file is required")
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Mask2Former config file not found: {config_path}")
    import yaml

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Mask2Former config must be a mapping: {config_path}")
    return data


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Mask2Former config section '{name}' is required")
    return value


def _required(mapping: dict[str, Any], key: str, section_name: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Mask2Former config '{section_name}.{key}' is required")
    return mapping[key]


def _resolve_path(path: str | Path | None) -> Optional[Path]:
    if not path:
        return None
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return Path.cwd() / resolved


def _looks_like_weights(path_value: str | Path | None) -> bool:
    if not path_value:
        return False
    path = _resolve_path(path_value)
    if path is None or not path.exists():
        return False
    if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".safetensors"}:
        return True
    return path.is_dir() and ((path / "model.pt").is_file() or (path / "model.safetensors").is_file())


def _normalize_size(size: Any) -> dict[str, int]:
    if isinstance(size, dict):
        return {"height": int(size["height"]), "width": int(size["width"])}
    if isinstance(size, (list, tuple)):
        if len(size) == 1:
            value = int(size[0])
            return {"height": value, "width": value}
        if len(size) == 2:
            return {"height": int(size[0]), "width": int(size[1])}
    value = int(size)
    return {"height": value, "width": value}


def _load_settings(
    config_file: str | Path | None,
    checkpoint_file: str | None,
) -> Mask2FormerSettings:
    config = _load_yaml(config_file)
    model_cfg = _section(config, "model")
    input_cfg = _section(config, "input")
    crop_cfg = _section(config, "crop")
    post_cfg = _section(config, "postprocess")

    configured_checkpoint = str(_required(model_cfg, "checkpoint", "model"))
    requested_checkpoint = checkpoint_file or configured_checkpoint
    configured_weights = model_cfg.get("weights")

    if configured_weights:
        checkpoint = requested_checkpoint
        weights = str(configured_weights)
    elif _looks_like_weights(requested_checkpoint):
        checkpoint = configured_checkpoint
        weights = str(requested_checkpoint)
    else:
        checkpoint = requested_checkpoint
        weights = None

    crop_enabled = bool(_required(crop_cfg, "enabled", "crop"))
    crop_tolerance = float(_required(crop_cfg, "tolerance", "crop"))

    return Mask2FormerSettings(
        checkpoint=checkpoint,
        weights=weights,
        size=_normalize_size(_required(input_cfg, "size", "input")),
        crop_enabled=crop_enabled,
        crop_tolerance=crop_tolerance,
        prediction_threshold=float(_required(post_cfg, "prediction_threshold", "postprocess")),
        mask_threshold=float(_required(post_cfg, "mask_threshold", "postprocess")),
        overlap_threshold=float(_required(post_cfg, "overlap_threshold", "postprocess")),
        min_instance_area=int(_required(post_cfg, "min_instance_area", "postprocess")),
    )


def _resolve_device(device: str) -> str:
    if str(device).startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device was requested ({device}) but torch.cuda.is_available() is False")
        if ":" in str(device):
            device_index = int(str(device).split(":", 1)[1])
            if device_index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"CUDA device {device} was requested but only {torch.cuda.device_count()} device(s) are available"
                )
    return device


def _image_to_pil_rgb(image: Union[str, Path, np.ndarray, Image.Image]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Unsupported image type: {type(image)!r}")
    if image.ndim == 2:
        return Image.fromarray(image).convert("RGB")
    if image.shape[2] == 3:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    if image.shape[2] == 4:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        return Image.fromarray(rgb)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def _load_state_dict_from_weights(source: Path) -> tuple[dict[str, Any], Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    import torch
    from transformers import Mask2FormerConfig, Mask2FormerImageProcessor

    if source.is_file() and source.suffix.lower() in {".pt", ".pth"}:
        payload = torch.load(source, map_location="cpu")
        if isinstance(payload, dict) and "model_state_dict" in payload:
            return payload["model_state_dict"], payload.get("model_config"), payload.get("processor_config")
        if isinstance(payload, dict):
            return payload, None, None
        raise ValueError(f"Unsupported .pt weights format: {source}")

    if source.is_file() and source.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file as load_safetensors_file

        return load_safetensors_file(str(source)), None, None

    if source.is_dir():
        pt_path = source / "model.pt"
        if pt_path.is_file():
            return _load_state_dict_from_weights(pt_path)
        safetensors_path = source / "model.safetensors"
        if safetensors_path.is_file():
            state_dict, _, _ = _load_state_dict_from_weights(safetensors_path)
            model_config = None
            processor_config = None
            if (source / "config.json").is_file():
                model_config = Mask2FormerConfig.from_pretrained(source).to_dict()
            if (source / "preprocessor_config.json").is_file():
                processor_config = Mask2FormerImageProcessor.from_pretrained(source).to_dict()
            return state_dict, model_config, processor_config

    raise FileNotFoundError(f"No supported Mask2Former weights found in {source}")


def _build_model_from_weights(weights_source: Path, fallback_checkpoint: str, device: str):
    from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

    state_dict, model_config, processor_config = _load_state_dict_from_weights(weights_source)
    if model_config is not None:
        model = Mask2FormerForUniversalSegmentation(Mask2FormerConfig.from_dict(model_config))
        if processor_config is not None:
            processor = Mask2FormerImageProcessor.from_dict(processor_config)
        else:
            processor = Mask2FormerImageProcessor.from_pretrained(fallback_checkpoint, do_reduce_labels=False)
    else:
        processor = Mask2FormerImageProcessor.from_pretrained(fallback_checkpoint, do_reduce_labels=False)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            fallback_checkpoint,
            num_labels=1,
            id2label={0: "object"},
            label2id={"object": 0},
            ignore_mismatched_sizes=True,
        )

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return processor, model


def _as_mask_stack(instance_masks: np.ndarray) -> np.ndarray:
    if instance_masks.ndim == 2:
        instance_masks = instance_masks[np.newaxis, ...]
    if instance_masks.ndim != 3:
        raise ValueError(f"Expected instance mask stack with shape (N, H, W), got {instance_masks.shape}")
    return instance_masks.astype(np.uint8)


def _make_masks_exclusive(instance_masks: np.ndarray, segments_info: list[dict[str, Any]]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    instance_masks = _as_mask_stack(instance_masks)
    if instance_masks.size == 0:
        return instance_masks, segments_info

    order = list(range(instance_masks.shape[0]))
    if len(segments_info) == instance_masks.shape[0]:
        order = sorted(order, key=lambda index: float(segments_info[index].get("score", 0.0)), reverse=True)

    exclusive_masks = np.zeros_like(instance_masks, dtype=np.uint8)
    occupied = np.zeros(instance_masks.shape[1:], dtype=bool)
    for new_index, old_index in enumerate(order):
        current = instance_masks[old_index].astype(bool)
        current &= ~occupied
        exclusive_masks[new_index] = current.astype(np.uint8)
        occupied |= current

    if len(segments_info) == instance_masks.shape[0]:
        segments_info = [segments_info[index] for index in order]
    return exclusive_masks, segments_info


def _remove_small_components(instance_masks: np.ndarray, min_area: int) -> np.ndarray:
    instance_masks = _as_mask_stack(instance_masks)
    if instance_masks.size == 0 or min_area <= 0:
        return instance_masks

    cleaned_masks = np.zeros_like(instance_masks, dtype=np.uint8)
    for index, mask in enumerate(instance_masks):
        if not np.any(mask):
            continue
        component_count, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for component_index in range(1, component_count):
            area = int(component_stats[component_index, cv2.CC_STAT_AREA])
            if area >= min_area:
                cleaned_masks[index][component_labels == component_index] = 1
    return cleaned_masks


def _remove_small_contours(instance_masks: np.ndarray, min_area: int) -> np.ndarray:
    instance_masks = _as_mask_stack(instance_masks)
    if instance_masks.size == 0 or min_area <= 0:
        return instance_masks

    cleaned_masks = np.zeros_like(instance_masks, dtype=np.uint8)
    for index, mask in enumerate(instance_masks):
        binary_mask = (mask > 0).astype(np.uint8) * 255
        if not np.any(binary_mask):
            continue
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) >= float(min_area):
                cv2.drawContours(cleaned_masks[index], [contour], -1, 1, thickness=cv2.FILLED)
    return cleaned_masks


def _filter_instances(
    instance_masks: np.ndarray,
    segments_info: list[dict[str, Any]],
    min_instance_area: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    instance_masks = _as_mask_stack(instance_masks)
    if instance_masks.size == 0:
        return instance_masks, []

    if len(segments_info) < instance_masks.shape[0]:
        segments_info = list(segments_info) + [{} for _ in range(instance_masks.shape[0] - len(segments_info))]
    elif len(segments_info) > instance_masks.shape[0]:
        segments_info = list(segments_info[: instance_masks.shape[0]])

    areas = instance_masks.reshape(instance_masks.shape[0], -1).sum(axis=1)
    keep_indices = [index for index, area in enumerate(areas) if int(area) >= min_instance_area]
    if not keep_indices:
        return instance_masks[:0], []
    return instance_masks[keep_indices], [segments_info[index] for index in keep_indices]


def _bbox_from_mask(mask: np.ndarray, score: float) -> Optional[np.ndarray]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, score], dtype=np.float32)


class Mask2FormerDetector:
    def __init__(
        self,
        config_file: str | None = None,
        checkpoint_file: str | None = None,
        device: str = "cuda:0",
    ) -> None:
        self.settings = _load_settings(config_file, checkpoint_file)
        self.device = _resolve_device(device)

        self.processor, self.model = self._load_model()

    def _load_model(self):
        from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

        weights_path = _resolve_path(self.settings.weights)
        if weights_path is not None and weights_path.exists():
            return _build_model_from_weights(weights_path, self.settings.checkpoint, self.device)

        checkpoint_path = _resolve_path(self.settings.checkpoint)
        checkpoint = str(checkpoint_path) if checkpoint_path is not None and checkpoint_path.exists() else self.settings.checkpoint
        processor = Mask2FormerImageProcessor.from_pretrained(checkpoint, do_reduce_labels=False)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(checkpoint, ignore_mismatched_sizes=True)
        model.to(self.device)
        model.eval()
        return processor, model

    def inference(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        crop_box: Optional[Sequence[float]] = None,
    ) -> List[InstanceData]:
        import torch

        original_image = _image_to_pil_rgb(image)
        image_size = original_image.size
        resolved_crop_box = None
        inference_image = original_image

        if self.settings.crop_enabled and crop_box:
            resolved_crop_box = normalize_crop_box(crop_box, image_size, tolerance=self.settings.crop_tolerance, order="cmes")
            if resolved_crop_box is not None:
                inference_image = crop_image(original_image, resolved_crop_box, order="pil")

        inputs = self.processor(images=inference_image, return_tensors="pt", size=self.settings.size)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        result = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.settings.prediction_threshold,
            mask_threshold=self.settings.mask_threshold,
            overlap_mask_area_threshold=self.settings.overlap_threshold,
            target_sizes=[inference_image.size[::-1]],
            return_binary_maps=True,
        )[0]

        instance_masks = result["segmentation"].detach().cpu().numpy().astype(np.uint8)
        segments_info = list(result.get("segments_info", []))
        instance_masks, segments_info = _make_masks_exclusive(instance_masks, segments_info)
        instance_masks = _remove_small_components(instance_masks, self.settings.min_instance_area)
        instance_masks = _remove_small_contours(instance_masks, self.settings.min_instance_area)
        instance_masks, segments_info = _filter_instances(instance_masks, segments_info, self.settings.min_instance_area)

        if resolved_crop_box is not None:
            instance_masks = uncrop_instance_masks(instance_masks, resolved_crop_box, image_size, order="pil")

        predictions: list[InstanceData] = []
        for index, mask in enumerate(instance_masks):
            info = segments_info[index] if index < len(segments_info) else {}
            score = float(info.get("score", 1.0))
            label = int(info.get("label_id", info.get("category_id", info.get("id", 0))))
            bbox = _bbox_from_mask(mask, score)
            if bbox is None:
                continue
            predictions.append(
                InstanceData(
                    bbox=bbox,
                    label=label,
                    score=score,
                    mask=mask.astype(bool),
                )
            )
        return predictions
