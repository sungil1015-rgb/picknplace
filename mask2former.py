"""Run Mask2Former inference on images in the test folder.

This script is intentionally lightweight: it loads a Mask2Former checkpoint,
runs inference over every image in the shared test directory, and writes an
overlay plus a masked view for each input image.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import colorsys
from copy import deepcopy
import sys
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file as load_safetensors_file
import yaml
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import MASK2FORMER_OUTPUT_DIR, MASK2FORMER_TRAIN2_OUTPUT_DIR, MASK2FORMER_TRAIN3_OUTPUT_DIR, MASK2FORMER_TRAIN_OUTPUT_DIR, TEST_IMAGES_DIR  # noqa: E402
from src.preprocessing.crop import crop_image, load_sample_info, resolve_crop_box_from_info, uncrop_instance_masks  # noqa: E402
 

DEFAULT_CHECKPOINT = None
DEFAULT_WEIGHTS_DIR = REPO_ROOT / "weights" / "mask2former"
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "mask2former.yaml"
DEFAULT_IMAGE_SIZE = (1024, 1224)
DEFAULT_PREDICTION_THRESHOLD = 0.6
DEFAULT_MASK_THRESHOLD = 0.6
DEFAULT_OVERLAP_THRESHOLD = 0.8
DEFAULT_MIN_INSTANCE_AREA = 150
DEFAULT_CONTOUR_SIMPLIFY_RATIO = 0.01
IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def load_config(config_path: str | Path) -> dict[str, Any]:
	path = Path(config_path)
	if not path.is_absolute():
		path = REPO_ROOT / path
	if not path.is_file():
		raise FileNotFoundError(f"Config file not found: {path}")
	with path.open("r", encoding="utf-8") as handle:
		config = yaml.safe_load(handle) or {}
	if not isinstance(config, dict):
		raise ValueError(f"Config must be a mapping: {path}")
	return config


def resolve_path(value: str | Path | None) -> Path | None:
	if value is None:
		return None
	path = Path(value)
	if path.is_absolute():
		return path
	return REPO_ROOT / path


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run Mask2Former on the shared test image folder.")
	parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config file path.")
	parser.add_argument("--checkpoint", default=None, help="Hugging Face model name or local checkpoint path.")
	parser.add_argument("--weights", default=None, help="Optional local weights file or directory. Supports .pt, .safetensors, or a Hugging Face save_pretrained directory.")
	parser.add_argument("--source", default=None, help="Folder containing test images.")
	parser.add_argument("--output", default=None, help="Folder where outputs will be saved.")
	parser.add_argument("--size", type=int, nargs="+", default=None, help="Resize input size before inference. Use one value for square, or two values for HxW.")
	parser.add_argument("--prediction-threshold", type=float, default=None, help="Score threshold for keeping predicted instances.")
	parser.add_argument("--mask-threshold", type=float, default=None, help="Threshold used to binarize predicted instance masks.")
	parser.add_argument("--overlap-threshold", type=float, default=None, help="Overlap threshold for post-processing instances.")
	parser.add_argument("--min-instance-area", type=int, default=None, help="Minimum pixel area required to keep a predicted instance.")
	parser.add_argument("--contour-simplify-ratio", type=float, default=None, help="Contour simplification ratio for overlay outlines.")
	parser.add_argument("--device", default=None, help="Inference device.")
	return parser.parse_args()


def normalize_weight_entries(weights_value: Any) -> list[str]:
	if weights_value is None:
		return []
	if isinstance(weights_value, (list, tuple)):
		return [str(value) for value in weights_value if str(value).strip()]
	return [str(weights_value)]


def list_images(source: Path) -> list[Path]:
	if source.is_file() and source.suffix.lower() in IMAGE_EXTENSIONS:
		return [source]
	if not source.is_dir():
		raise FileNotFoundError(f"Test image folder not found: {source}")
	return [path for path in sorted(source.rglob("*")) if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]


def load_state_dict_from_weights(source: Path) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
	if source.is_file() and source.suffix.lower() == ".pt":
		payload = torch.load(source, map_location="cpu")
		if not isinstance(payload, dict) or "model_state_dict" not in payload:
			raise ValueError(f"Unsupported .pt format: {source}")
		return payload["model_state_dict"], payload.get("model_config"), payload.get("processor_config")

	if source.is_file() and source.suffix.lower() == ".safetensors":
		return load_safetensors_file(str(source)), None, None

	if source.is_dir():
		pt_path = source / "model.pt"
		if pt_path.is_file():
			return load_state_dict_from_weights(pt_path)
		safetensors_path = source / "model.safetensors"
		if safetensors_path.is_file():
			model_state_dict = load_safetensors_file(str(safetensors_path))
			model_config_path = source / "config.json"
			processor_config_path = source / "preprocessor_config.json"
			model_config = None
			processor_config = None
			if model_config_path.is_file():
				model_config = Mask2FormerConfig.from_pretrained(source).to_dict()
			if processor_config_path.is_file():
				processor_config = Mask2FormerImageProcessor.from_pretrained(source).to_dict()
			return model_state_dict, model_config, processor_config

	raise FileNotFoundError(f"No supported weights file found in {source}")


def build_model_from_weights(weights_source: Path, device: str):
	state_dict, model_config, processor_config = load_state_dict_from_weights(weights_source)
	if model_config is not None:
		model = Mask2FormerForUniversalSegmentation(Mask2FormerConfig.from_dict(model_config))
		processor = Mask2FormerImageProcessor.from_dict(processor_config) if processor_config is not None else Mask2FormerImageProcessor.from_pretrained("facebook/mask2former-swin-base-coco-panoptic", do_reduce_labels=False)
	else:
		processor = Mask2FormerImageProcessor.from_pretrained("facebook/mask2former-swin-base-coco-panoptic", do_reduce_labels=False)
		model = Mask2FormerForUniversalSegmentation.from_pretrained(
			"facebook/mask2former-swin-base-coco-panoptic",
			num_labels=1,
			id2label={0: "object"},
			label2id={"object": 0},
			ignore_mismatched_sizes=True,
		)
	missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
	if missing_keys or unexpected_keys:
		print(f"Loaded weights with missing keys: {len(missing_keys)}, unexpected keys: {len(unexpected_keys)}")
	model.to(device)
	model.eval()
	return processor, model


def load_model(checkpoint: str | None, weights: str | None, device: str):
	if checkpoint:
		processor = Mask2FormerImageProcessor.from_pretrained(checkpoint, do_reduce_labels=False)
		model = Mask2FormerForUniversalSegmentation.from_pretrained(checkpoint, ignore_mismatched_sizes=True)
		model.to(device)
		model.eval()
		return processor, model

	weights_source = Path(weights) if weights else DEFAULT_WEIGHTS_DIR
	if weights_source.is_file() or weights_source.is_dir():
		return build_model_from_weights(weights_source, device)

	for candidate in (
		DEFAULT_WEIGHTS_DIR / "model.pt",
		DEFAULT_WEIGHTS_DIR / "model.safetensors",
		MASK2FORMER_TRAIN3_OUTPUT_DIR / "best",
		MASK2FORMER_TRAIN3_OUTPUT_DIR / "best.pt",
		MASK2FORMER_TRAIN3_OUTPUT_DIR / "last",
		MASK2FORMER_TRAIN3_OUTPUT_DIR / "last.pt",
		MASK2FORMER_TRAIN2_OUTPUT_DIR / "best",
		MASK2FORMER_TRAIN2_OUTPUT_DIR / "best.pt",
		MASK2FORMER_TRAIN2_OUTPUT_DIR / "last",
		MASK2FORMER_TRAIN2_OUTPUT_DIR / "last.pt",
		MASK2FORMER_TRAIN_OUTPUT_DIR / "best",
		MASK2FORMER_TRAIN_OUTPUT_DIR / "best.pt",
		MASK2FORMER_TRAIN_OUTPUT_DIR / "last",
		MASK2FORMER_TRAIN_OUTPUT_DIR / "last.pt",
	):
		if candidate.exists():
			return build_model_from_weights(candidate, device)

	checkpoint = checkpoint or "facebook/mask2former-swin-base-coco-panoptic"
	processor = Mask2FormerImageProcessor.from_pretrained(checkpoint, do_reduce_labels=False)
	model = Mask2FormerForUniversalSegmentation.from_pretrained(checkpoint, ignore_mismatched_sizes=True)
	model.to(device)
	model.eval()
	return processor, model


def resolve_checkpoint(checkpoint: str | None) -> str:
	if checkpoint:
		return checkpoint
	for candidate in (
		MASK2FORMER_TRAIN3_OUTPUT_DIR / "best",
		MASK2FORMER_TRAIN3_OUTPUT_DIR / "best.pt",
		MASK2FORMER_TRAIN3_OUTPUT_DIR / "last",
		MASK2FORMER_TRAIN3_OUTPUT_DIR / "last.pt",
		MASK2FORMER_TRAIN2_OUTPUT_DIR / "best",
		MASK2FORMER_TRAIN2_OUTPUT_DIR / "best.pt",
		MASK2FORMER_TRAIN2_OUTPUT_DIR / "last",
		MASK2FORMER_TRAIN2_OUTPUT_DIR / "last.pt",
		MASK2FORMER_TRAIN_OUTPUT_DIR / "best",
		MASK2FORMER_TRAIN_OUTPUT_DIR / "best.pt",
		MASK2FORMER_TRAIN_OUTPUT_DIR / "last",
		MASK2FORMER_TRAIN_OUTPUT_DIR / "last.pt",
	):
		if candidate.exists():
			return str(candidate)
	return "facebook/mask2former-swin-base-coco-panoptic"


def normalize_size(size: Any) -> dict[str, int]:
	if isinstance(size, dict):
		if {"height", "width"}.issubset(size):
			return {"height": int(size["height"]), "width": int(size["width"])}
		raise ValueError(f"Invalid size mapping: {size}")
	if isinstance(size, (list, tuple)):
		if len(size) == 1:
			value = int(size[0])
			return {"height": value, "width": value}
		if len(size) == 2:
			return {"height": int(size[0]), "width": int(size[1])}
		raise ValueError(f"Invalid size sequence: {size}")
	value = int(size)
	return {"height": value, "width": value}


def resolve_output_dir(output_dir: Path) -> Path:
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
	return output_dir.parent / f"{output_dir.name}_{timestamp}"


def image_output_stem(source_root: Path, image_path: Path) -> Path:
	if source_root.is_file():
		return Path(image_path.stem)
	try:
		return image_path.relative_to(source_root).with_suffix("")
	except ValueError:
		return Path(image_path.stem)


def ensure_output_dir(output_dir: Path) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)


def save_resolved_config(output_dir: Path, config: dict[str, Any], config_path: Path, test_cfg: dict[str, Any]) -> Path:
	resolved = deepcopy(config)
	resolved["_meta"] = {
		"config_path": str(config_path),
		"saved_at": datetime.now().isoformat(timespec="seconds"),
	}
	resolved["test"] = test_cfg
	output_path = output_dir / "config.used.yaml"
	with output_path.open("w", encoding="utf-8") as handle:
		yaml.safe_dump(resolved, handle, sort_keys=False, allow_unicode=True)
	return output_path


def palette(index: int) -> tuple[int, int, int]:
	hue = (index * 0.618033988749895) % 1.0
	red, green, blue = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
	return int(blue * 255), int(green * 255), int(red * 255)


def simplify_contour(contour: np.ndarray, ratio: float) -> np.ndarray:
	if contour.shape[0] <= 4:
		return contour
	perimeter = cv2.arcLength(contour, True)
	epsilon = max(1.0, perimeter * ratio)
	return cv2.approxPolyDP(contour, epsilon, True)

def render_instance_overlay(instance_masks: np.ndarray) -> np.ndarray:
	if instance_masks.ndim == 2:
		instance_masks = instance_masks[np.newaxis, ...]
	if instance_masks.ndim != 3:
		raise ValueError(f"Expected instance mask stack with shape (N, H, W), got {instance_masks.shape}")
	colored = np.zeros((instance_masks.shape[1], instance_masks.shape[2], 3), dtype=np.uint8)
	for index, mask in enumerate(instance_masks, start=1):
		colored[mask > 0] = palette(index)
	return colored


def render_prediction_panel(image: Image.Image, instance_masks: np.ndarray, title: str, contour_simplify_ratio: float) -> np.ndarray:
	image_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
	mask_bgr = cv2.cvtColor(render_instance_overlay(instance_masks), cv2.COLOR_RGB2BGR)
	mask_bgr = cv2.resize(mask_bgr, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
	overlay = image_bgr.copy()
	if instance_masks.ndim == 2:
		instance_masks = instance_masks[np.newaxis, ...]
	if instance_masks.size > 0:
		union_mask = instance_masks.max(axis=0) > 0
		if union_mask.shape != image_bgr.shape[:2]:
			union_mask = cv2.resize(union_mask.astype(np.uint8), (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
		blended = cv2.addWeighted(image_bgr, 0.6, mask_bgr, 0.4, 0.0)
		overlay[union_mask] = blended[union_mask]
	overlay = draw_instance_outlines(overlay, instance_masks, contour_simplify_ratio)
	caption_height = 36
	panel = np.full((overlay.shape[0] + caption_height, overlay.shape[1], 3), 255, dtype=np.uint8)
	panel[caption_height:, :, :] = overlay
	cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, caption_height - 1), (20, 20, 20), thickness=-1)
	cv2.putText(panel, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
	return panel


def segmentation_to_instance_masks(segmentation: np.ndarray, segments_info: list[dict[str, object]] | None = None) -> np.ndarray:
	if segmentation.ndim == 3:
		return (segmentation > 0).astype(np.uint8)
	if segmentation.ndim != 2:
		raise ValueError(f"Expected segmentation with shape (H, W) or (N, H, W), got {segmentation.shape}")

	segment_ids: list[int] = []
	for segment in segments_info or []:
		if isinstance(segment, dict) and "id" in segment:
			segment_id = int(segment["id"])
			if segment_id != 0:
				segment_ids.append(segment_id)
	if not segment_ids:
		segment_ids = [int(value) for value in np.unique(segmentation) if int(value) != 0]
	if not segment_ids:
		return np.zeros((0, segmentation.shape[0], segmentation.shape[1]), dtype=np.uint8)
	return np.stack([(segmentation == segment_id).astype(np.uint8) for segment_id in segment_ids], axis=0)


def make_masks_exclusive(instance_masks: np.ndarray, segments_info: list[dict[str, object]]) -> tuple[np.ndarray, list[dict[str, object]]]:
	if instance_masks.ndim == 2:
		instance_masks = instance_masks[np.newaxis, ...]
	if instance_masks.ndim != 3:
		raise ValueError(f"Expected instance mask stack with shape (N, H, W), got {instance_masks.shape}")
	if instance_masks.size == 0:
		return instance_masks, segments_info

	order = list(range(instance_masks.shape[0]))
	if len(segments_info) == instance_masks.shape[0]:
		order = sorted(
			order,
			key=lambda index: float(segments_info[index].get("score", 0.0)) if isinstance(segments_info[index], dict) else 0.0,
			reverse=True,
		)

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


def remove_small_components(instance_masks: np.ndarray, min_area: int) -> np.ndarray:
	if instance_masks.ndim == 2:
		instance_masks = instance_masks[np.newaxis, ...]
	if instance_masks.ndim != 3:
		raise ValueError(f"Expected instance mask stack with shape (N, H, W), got {instance_masks.shape}")
	if instance_masks.size == 0 or min_area <= 0:
		return instance_masks

	cleaned_masks = np.zeros_like(instance_masks, dtype=np.uint8)
	for index, mask in enumerate(instance_masks):
		binary_mask = mask.astype(np.uint8)
		if not np.any(binary_mask):
			continue
		component_count, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
		for component_index in range(1, component_count):
			component_area = int(component_stats[component_index, cv2.CC_STAT_AREA])
			if component_area >= min_area:
				cleaned_masks[index][component_labels == component_index] = 1

	return cleaned_masks


def remove_small_contours(instance_masks: np.ndarray, min_area: int) -> np.ndarray:
	if instance_masks.ndim == 2:
		instance_masks = instance_masks[np.newaxis, ...]
	if instance_masks.ndim != 3:
		raise ValueError(f"Expected instance mask stack with shape (N, H, W), got {instance_masks.shape}")
	if instance_masks.size == 0 or min_area <= 0:
		return instance_masks

	cleaned_masks = np.zeros_like(instance_masks, dtype=np.uint8)
	for index, mask in enumerate(instance_masks):
		binary_mask = (mask > 0).astype(np.uint8) * 255
		if not np.any(binary_mask):
			continue
		contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		for contour in contours:
			contour_area = float(cv2.contourArea(contour))
			if contour_area < float(min_area):
				continue
			cv2.drawContours(cleaned_masks[index], [contour], -1, 1, thickness=cv2.FILLED)

	return cleaned_masks


def draw_instance_outlines(image_bgr: np.ndarray, instance_masks: np.ndarray, contour_simplify_ratio: float) -> np.ndarray:
	outlined = image_bgr.copy()
	if instance_masks.ndim == 2:
		instance_masks = instance_masks[np.newaxis, ...]
	if instance_masks.ndim != 3:
		raise ValueError(f"Expected instance mask stack with shape (N, H, W), got {instance_masks.shape}")
	for index, mask in enumerate(instance_masks, start=1):
		binary_mask = (mask > 0).astype(np.uint8) * 255
		if not np.any(binary_mask):
			continue
		contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		for contour in contours:
			simplified = simplify_contour(contour, contour_simplify_ratio)
			if simplified.shape[0] < 3:
				continue
			cv2.drawContours(outlined, [simplified], -1, (0, 0, 0), 4)
			cv2.drawContours(outlined, [simplified], -1, palette(index), 2)
	return outlined


def save_overlay(image: Image.Image, instance_masks: np.ndarray, output_path: Path, contour_simplify_ratio: float) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	panel = render_prediction_panel(image, instance_masks, "prediction", contour_simplify_ratio)
	cv2.imwrite(str(output_path), panel)


def save_masked_image(image: Image.Image, instance_masks: np.ndarray, output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	image_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
	if instance_masks.ndim == 2:
		instance_masks = instance_masks[np.newaxis, ...]
	if instance_masks.size == 0:
		foreground = np.zeros_like(image_bgr)
	else:
		foreground = np.where(instance_masks.max(axis=0)[..., None] > 0, image_bgr, 0)
	cv2.imwrite(str(output_path), foreground)


def compose_weight_grid(image: Image.Image, weight_results: list[tuple[str, np.ndarray]], contour_simplify_ratio: float) -> np.ndarray:
	if not weight_results:
		raise ValueError("weight_results must not be empty")
	if len(weight_results) > 4:
		raise ValueError("A maximum of 4 weights can be compared at once")
	panels = [render_prediction_panel(image, instance_masks, title, contour_simplify_ratio) for title, instance_masks in weight_results]
	base_height = max(panel.shape[0] for panel in panels)
	base_width = max(panel.shape[1] for panel in panels)
	white = np.full((base_height, base_width, 3), 255, dtype=np.uint8)
	while len(panels) < 4:
		panels.append(white.copy())
	resized_panels: list[np.ndarray] = []
	for panel in panels:
		if panel.shape[0] != base_height or panel.shape[1] != base_width:
			padded = np.full((base_height, base_width, 3), 255, dtype=np.uint8)
			padded[: panel.shape[0], : panel.shape[1]] = panel
			panel = padded
		resized_panels.append(panel)
	row1 = np.concatenate(resized_panels[:2], axis=1)
	row2 = np.concatenate(resized_panels[2:4], axis=1)
	return np.concatenate([row1, row2], axis=0)


def infer_weight_entry(
	weight_entry: str,
	checkpoint: str | None,
	device: str,
) -> tuple[Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation]:
	weight_path = Path(weight_entry)
	if weight_path.exists():
		return build_model_from_weights(weight_path, device)
	return load_model(weight_entry, None, device)


def filter_instances(instance_masks: np.ndarray, segments_info: list[dict[str, object]], min_instance_area: int) -> tuple[np.ndarray, list[dict[str, object]]]:
	if instance_masks.ndim == 2:
		instance_masks = instance_masks[np.newaxis, ...]
	if instance_masks.size == 0:
		return instance_masks, segments_info
	if len(segments_info) < instance_masks.shape[0]:
		segments_info = list(segments_info) + [{} for _ in range(instance_masks.shape[0] - len(segments_info))]
	elif len(segments_info) > instance_masks.shape[0]:
		segments_info = list(segments_info[:instance_masks.shape[0]])
	areas = instance_masks.reshape(instance_masks.shape[0], -1).sum(axis=1)
	keep_indices = [index for index, area in enumerate(areas) if int(area) >= min_instance_area]
	if not keep_indices:
		return instance_masks[:0], []
	filtered_masks = instance_masks[keep_indices]
	filtered_segments = [segments_info[index] for index in keep_indices]
	return filtered_masks, filtered_segments


def infer_image(
	processor,
	model,
	rgb_image: Image.Image,
	device: str,
	size: int,
	prediction_threshold: float,
	mask_threshold: float,
	overlap_threshold: float,
	min_instance_area: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
	inputs = processor(images=rgb_image, return_tensors="pt", size=normalize_size(size))
	inputs = {key: value.to(device) for key, value in inputs.items()}

	with torch.no_grad():
		outputs = model(**inputs)

	if hasattr(processor, "post_process_instance_segmentation"):
		result = processor.post_process_instance_segmentation(
			outputs,
			threshold=prediction_threshold,
			mask_threshold=mask_threshold,
			overlap_mask_area_threshold=overlap_threshold,
			target_sizes=[rgb_image.size[::-1]],
			return_binary_maps=True,
		)[0]
		segments_info = result["segments_info"]
		segmentation = result["segmentation"].detach().cpu().numpy().astype(np.uint8)
		instance_masks = segmentation_to_instance_masks(segmentation, segments_info)
		instance_masks, segments_info = make_masks_exclusive(instance_masks, segments_info)
		instance_masks = remove_small_components(instance_masks, min_instance_area)
		instance_masks = remove_small_contours(instance_masks, min_instance_area)
		return filter_instances(instance_masks, segments_info, min_instance_area)

	raise RuntimeError("Instance segmentation post-processing is not available for this processor.")


def main() -> int:
	args = parse_args()
	config_path = Path(args.config)
	config = load_config(config_path)
	test_cfg = config.get("test", {})
	if not isinstance(test_cfg, dict):
		raise ValueError("test must be a mapping")

	source = Path(args.source or test_cfg.get("source", TEST_IMAGES_DIR))
	output_dir = resolve_output_dir(Path(args.output or test_cfg.get("output", MASK2FORMER_OUTPUT_DIR)))
	ensure_output_dir(output_dir)

	images = list_images(source)
	if not images:
		raise FileNotFoundError(f"No test images found in {source}")

	checkpoint = args.checkpoint or test_cfg.get("checkpoint")
	weights = args.weights or test_cfg.get("weights")
	weight_entries = normalize_weight_entries(weights)
	size_value = args.size if args.size is not None else test_cfg.get("size", DEFAULT_IMAGE_SIZE)
	size = normalize_size(size_value)
	prediction_threshold = float(args.prediction_threshold if args.prediction_threshold is not None else test_cfg.get("prediction_threshold", DEFAULT_PREDICTION_THRESHOLD))
	mask_threshold = float(args.mask_threshold if args.mask_threshold is not None else test_cfg.get("mask_threshold", DEFAULT_MASK_THRESHOLD))
	overlap_threshold = float(args.overlap_threshold if args.overlap_threshold is not None else test_cfg.get("overlap_threshold", DEFAULT_OVERLAP_THRESHOLD))
	min_instance_area = int(args.min_instance_area if args.min_instance_area is not None else test_cfg.get("min_instance_area", DEFAULT_MIN_INSTANCE_AREA))
	contour_simplify_ratio = float(args.contour_simplify_ratio if args.contour_simplify_ratio is not None else test_cfg.get("contour_simplify_ratio", DEFAULT_CONTOUR_SIMPLIFY_RATIO))
	device = str(args.device or test_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
	compare_mode = len(weight_entries) > 1
	if compare_mode:
		loaded_models: list[tuple[str, Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation]] = []
		for weight_entry in weight_entries[:4]:
			processor, model = infer_weight_entry(weight_entry, checkpoint, device)
			loaded_models.append((Path(weight_entry).name, processor, model))
	else:
		single_weight = weight_entries[0] if weight_entries else weights
		processor, model = load_model(checkpoint, single_weight, device)
	crop_cfg = config.get("crop", {})
	if not isinstance(crop_cfg, dict):
		raise ValueError("crop must be a mapping")
	crop_enabled = bool(crop_cfg.get("enabled", False))
	crop_tolerance = float(crop_cfg.get("tolerance", 10.0))
	resolved_test_cfg = {
		"source": str(source),
		"output": str(output_dir),
		"checkpoint": checkpoint,
		"weights": weights,
		"size": [size["height"], size["width"]],
		"crop": {"enabled": crop_enabled, "tolerance": crop_tolerance},
		"prediction_threshold": prediction_threshold,
		"mask_threshold": mask_threshold,
		"overlap_threshold": overlap_threshold,
		"min_instance_area": min_instance_area,
		"contour_simplify_ratio": contour_simplify_ratio,
		"device": device,
	}
	if compare_mode:
		resolved_test_cfg["weights"] = weight_entries[:4]
	saved_config_path = save_resolved_config(output_dir, config, config_path, resolved_test_cfg)
	print(f"Checkpoint: {checkpoint or weights or DEFAULT_WEIGHTS_DIR}")
	print(f"Source: {source}")
	print(f"Output: {output_dir}")
	print(f"Device: {device}")
	print(f"Saved config: {saved_config_path}")

	for image_path in images:
		with Image.open(image_path) as image:
			original_image_rgb = image.convert("RGB")
			cropped_image_rgb = original_image_rgb
			crop_box = None
			if crop_enabled:
				sample_info = load_sample_info(image_path)
				crop_box = resolve_crop_box_from_info(sample_info, tolerance=crop_tolerance, image_size=original_image_rgb.size)
				cropped_image_rgb = crop_image(original_image_rgb, crop_box)
			if compare_mode:
				weight_results: list[tuple[str, np.ndarray]] = []
				for weight_name, weight_processor, weight_model in loaded_models:
					instance_masks, _ = infer_image(
						weight_processor,
						weight_model,
						cropped_image_rgb,
						device,
						size,
						prediction_threshold,
						mask_threshold,
						overlap_threshold,
						min_instance_area,
					)
					if crop_enabled:
						instance_masks = uncrop_instance_masks(instance_masks, crop_box, original_image_rgb.size)
					weight_results.append((weight_name, instance_masks))
			else:
				instance_masks, segments_info = infer_image(
					processor,
					model,
					cropped_image_rgb,
					device,
					size,
					prediction_threshold,
					mask_threshold,
					overlap_threshold,
					min_instance_area,
				)
				if crop_enabled:
					instance_masks = uncrop_instance_masks(instance_masks, crop_box, original_image_rgb.size)

		stem_path = image_output_stem(source, image_path)
		if compare_mode:
			comparison_path = output_dir / stem_path.parent / f"{stem_path.name}_comparison.png"
			comparison_panel = compose_weight_grid(original_image_rgb, weight_results, contour_simplify_ratio)
			comparison_path.parent.mkdir(parents=True, exist_ok=True)
			cv2.imwrite(str(comparison_path), comparison_panel)
			print(f"Saved {comparison_path.name}")
		else:
			overlay_path = output_dir / stem_path.parent / f"{stem_path.name}_overlay.png"
			mask_path = output_dir / stem_path.parent / f"{stem_path.name}_masked.png"
			save_overlay(original_image_rgb, instance_masks, overlay_path, contour_simplify_ratio)
			save_masked_image(original_image_rgb, instance_masks, mask_path)
			print(f"Saved {overlay_path.name} and {mask_path.name}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
