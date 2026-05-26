"""Shared crop helpers for the binpicking dataset.

The crop is selected from the adjacent ``*_info.json`` file using camera focal
length. Training and inference both import these helpers so the same crop rule
is applied everywhere.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_FOCAL_TOLERANCE = 10.0

CROP_BOX_BY_FOCAL_LENGTH: tuple[tuple[float, tuple[float, float, float, float]], ...] = (
	(2013.0, (150.822, 0.0, 1186.162, 717.821)),
	(1242.0, (200.718, 18.144, 961.63, 611.225)),
)
#"2d_roi": ["118","1190","98",813"]

def load_sample_info(image_path: str | Path) -> dict[str, Any]:
	"""Load the ``*_info.json`` file that sits next to an image."""

	path = Path(image_path)
	info_path = path.with_name(f"{path.stem}_info.json")
	if not info_path.is_file():
		return {}
	with info_path.open("r", encoding="utf-8") as handle:
		info = json.load(handle)
	if not isinstance(info, dict):
		raise ValueError(f"Sample info must be a mapping: {info_path}")
	return info


def _coerce_float(value: Any) -> float | None:
	if value is None:
		return None
	if isinstance(value, (int, float)):
		return float(value)
	if isinstance(value, str):
		try:
			return float(value)
		except ValueError:
			return None
	return None


def normalize_crop_box(
	crop_box: tuple[float, float, float, float] | list[float] | None,
	image_size: tuple[int, int] | None = None,
	tolerance: float = 0.0,
	order: str = "pil",
) -> tuple[int, int, int, int] | None:
	if crop_box is None:
		return None
	if len(crop_box) != 4:
		raise ValueError(f"crop_box must have 4 values, got {crop_box}")
	values = [float(value) for value in crop_box]
	if order == "cmes":
		x0, x1, y0, y1 = values
	elif order == "pil":
		x0, y0, x1, y1 = values
	else:
		x0, x1, y0, y1 = values if values[1] > values[2] else (values[0], values[2], values[1], values[3])
	x0 -= tolerance
	y0 -= tolerance
	x1 += tolerance
	y1 += tolerance
	left = int(round(x0))
	top = int(round(y0))
	right = int(round(x1))
	bottom = int(round(y1))
	if image_size is not None:
		width, height = image_size
		left = max(0, min(left, width))
		top = max(0, min(top, height))
		right = max(left + 1, min(right, width))
		bottom = max(top + 1, min(bottom, height))
		if left == 0 and top == 0 and right == width and bottom == height:
			return None
	return left, top, right, bottom


def _normalize_crop_box(crop_box: tuple[float, float, float, float], image_size: tuple[int, int] | None = None) -> tuple[int, int, int, int]:
	normalized = normalize_crop_box(crop_box, image_size=image_size, order="pil")
	if normalized is None and image_size is not None:
		width, height = image_size
		return 0, 0, width, height
	if normalized is None:
		x0, y0, x1, y1 = crop_box
		return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
	return normalized


def resolve_crop_box_from_info(
	info: dict[str, Any],
	*,
	tolerance: float = DEFAULT_FOCAL_TOLERANCE,
	image_size: tuple[int, int] | None = None,
) -> tuple[int, int, int, int] | None:
	"""Pick a crop box from camera focal length metadata.

	The current dataset has two focal-length groups:
	- around 2013 -> wider crop
	- around 1242 -> narrower crop
	"""

	camera_info = info.get("camera_info")
	if not isinstance(camera_info, dict):
		return None
	fx = _coerce_float(camera_info.get("cal.fx"))
	fy = _coerce_float(camera_info.get("cal.fy"))
	if fx is None or fy is None:
		return None
	focal_length = (fx + fy) * 0.5

	for target_focal_length, crop_box in CROP_BOX_BY_FOCAL_LENGTH:
		if abs(focal_length - target_focal_length) <= tolerance:
			return _normalize_crop_box(crop_box, image_size=image_size)

	return None


def crop_image(image: Image.Image, crop_box: tuple[int, int, int, int] | None, order: str = "pil") -> Image.Image:
	"""Crop a PIL image if a crop box is available."""

	if crop_box is None:
		return image
	crop_box = normalize_crop_box(crop_box, image.size, order=order)
	if crop_box is None:
		return image
	return image.crop(crop_box)


def crop_image_and_mask(
	image: Image.Image,
	mask: np.ndarray,
	crop_box: tuple[int, int, int, int] | None,
) -> tuple[Image.Image, np.ndarray]:
	"""Crop an image and its segmentation mask with the same box."""

	if crop_box is None:
		return image, mask
	left, top, right, bottom = crop_box
	return image.crop((left, top, right, bottom)), mask[top:bottom, left:right]


def uncrop_instance_masks(
	instance_masks: np.ndarray,
	crop_box: tuple[int, int, int, int] | None,
	original_size: tuple[int, int],
	order: str = "pil",
) -> np.ndarray:
	"""Place cropped instance masks back onto the original image canvas."""

	if instance_masks.ndim == 2:
		instance_masks = instance_masks[np.newaxis, ...]
	if instance_masks.ndim != 3:
		raise ValueError(f"Expected instance mask stack with shape (N, H, W), got {instance_masks.shape}")

	original_width, original_height = original_size
	crop_box = normalize_crop_box(crop_box, original_size, order=order)
	if crop_box is None:
		if instance_masks.shape[1:] == (original_height, original_width):
			return instance_masks
		uncropped = np.zeros((instance_masks.shape[0], original_height, original_width), dtype=instance_masks.dtype)
		height = min(original_height, instance_masks.shape[1])
		width = min(original_width, instance_masks.shape[2])
		uncropped[:, :height, :width] = instance_masks[:, :height, :width]
		return uncropped

	left, top, right, bottom = crop_box
	uncropped = np.zeros((instance_masks.shape[0], original_height, original_width), dtype=instance_masks.dtype)
	if instance_masks.shape[0] == 0:
		return uncropped
	uncropped[:, top:bottom, left:right] = instance_masks
	return uncropped
