from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class DepthLayerSplit:
    surface: np.ndarray | None
    debug: dict[str, Any]


def valid_depth_values(depth_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = depth_image[: mask.shape[0], : mask.shape[1]][mask]
    return values[np.isfinite(values) & (values > 0)]


def median_valid_depth_at_point(
    depth_image: np.ndarray,
    u: int,
    v: int,
    mask: np.ndarray | None = None,
    window: int = 5,
    fallback_to_mask: bool = True,
) -> float | None:
    if depth_image is None or depth_image.ndim < 2:
        return None

    half = int(window) // 2
    y1 = max(0, int(v) - half)
    y2 = min(depth_image.shape[0], int(v) + half + 1)
    x1 = max(0, int(u) - half)
    x2 = min(depth_image.shape[1], int(u) + half + 1)

    depth_patch = depth_image[y1:y2, x1:x2]
    if mask is None:
        values = depth_patch[depth_patch > 0]
    else:
        mask_patch = mask[y1:y2, x1:x2] > 0
        values = depth_patch[(depth_patch > 0) & mask_patch]
        if values.size == 0 and fallback_to_mask and np.any(mask):
            bounded_mask = mask[: depth_image.shape[0], : depth_image.shape[1]] > 0
            values = depth_image[bounded_mask]
            values = values[values > 0]

    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.median(values.astype(np.float64)))


def split_surface_by_depth_gap(
    depth_image: np.ndarray | None,
    surface_mask: np.ndarray,
    min_depth_gap_mm: float,
    trim_low_percentile: float,
    trim_high_percentile: float,
    cut_band_mm: float,
    bridge_open_kernel_px: int,
    line_cut_enabled: bool,
    line_cut_thickness_px: int,
    line_cut_min_aspect_ratio: float,
    min_layer_area_px: int,
    min_layer_area_ratio: float,
    min_component_area_px: int,
) -> DepthLayerSplit:
    debug: dict[str, Any] = {
        "enabled": True,
        "method": "largest_sorted_depth_gap",
        "min_depth_gap_mm": float(min_depth_gap_mm),
        "trim_low_percentile": float(trim_low_percentile),
        "trim_high_percentile": float(trim_high_percentile),
        "cut_band_mm": float(cut_band_mm),
        "bridge_open_kernel_px": int(bridge_open_kernel_px),
        "line_cut_enabled": bool(line_cut_enabled),
        "line_cut_thickness_px": int(line_cut_thickness_px),
        "line_cut_min_aspect_ratio": float(line_cut_min_aspect_ratio),
        "min_layer_area_px": int(min_layer_area_px),
        "min_layer_area_ratio": float(min_layer_area_ratio),
        "min_component_area_px": int(min_component_area_px),
    }
    if depth_image is None or depth_image.ndim < 2:
        return DepthLayerSplit(None, {**debug, "applied": False, "reason": "missing_depth_image"})

    height = min(depth_image.shape[0], surface_mask.shape[0])
    width = min(depth_image.shape[1], surface_mask.shape[1])
    surface = surface_mask[:height, :width] > 0
    surface_area = int(np.count_nonzero(surface))
    if surface_area <= 0:
        return DepthLayerSplit(None, {**debug, "applied": False, "reason": "empty_surface"})

    depth = np.asarray(depth_image[:height, :width], dtype=np.float64)
    valid = surface & np.isfinite(depth) & (depth > 0.0)
    values = depth[valid]
    debug["surface_area"] = surface_area
    debug["valid_depth_pixels"] = int(values.size)
    if values.size < max(2, int(min_layer_area_px) * 2):
        return DepthLayerSplit(None, {**debug, "applied": False, "reason": "insufficient_valid_depth"})

    low = float(np.percentile(values, float(trim_low_percentile)))
    high = float(np.percentile(values, float(trim_high_percentile)))
    trimmed_valid = valid & (depth >= low) & (depth <= high)
    trimmed_values = depth[trimmed_valid]
    debug["depth_trim_low_mm"] = low
    debug["depth_trim_high_mm"] = high
    debug["trimmed_depth_pixels"] = int(trimmed_values.size)
    if trimmed_values.size < max(2, int(min_layer_area_px) * 2):
        return DepthLayerSplit(None, {**debug, "applied": False, "reason": "insufficient_trimmed_depth"})

    sorted_depth = np.sort(trimmed_values)
    gaps = np.diff(sorted_depth)
    if gaps.size == 0:
        return DepthLayerSplit(None, {**debug, "applied": False, "reason": "no_depth_gap"})

    gap_index = int(np.argmax(gaps))
    max_gap = float(gaps[gap_index])
    split_depth = float((sorted_depth[gap_index] + sorted_depth[gap_index + 1]) * 0.5)
    effective_cut_band = min(max(float(cut_band_mm), 0.0), max_gap * 0.5)
    cut_half = effective_cut_band * 0.5
    near_threshold = split_depth - cut_half
    far_threshold = split_depth + cut_half
    near = trimmed_valid & (depth <= near_threshold)
    far = trimmed_valid & (depth >= far_threshold)
    initial_near_area = int(np.count_nonzero(near))
    initial_far_area = int(np.count_nonzero(far))
    min_area = max(int(min_layer_area_px), int(round(surface_area * float(min_layer_area_ratio))))
    if bool(line_cut_enabled):
        line_cut, line_debug = _straight_line_cut_from_minor_layer(
            surface,
            near,
            far,
            int(line_cut_thickness_px),
            float(line_cut_min_aspect_ratio),
        )
        near = near & ~line_cut
        far = far & ~line_cut
    else:
        line_debug = {
            "line_cut_applied": False,
            "line_cut_reason": "disabled",
            "line_cut_pixels": 0,
        }
    near, near_bridge_debug = _break_narrow_bridges(near, int(bridge_open_kernel_px))
    far, far_bridge_debug = _break_narrow_bridges(far, int(bridge_open_kernel_px))
    near_area = int(np.count_nonzero(near))
    far_area = int(np.count_nonzero(far))
    debug.update(
        {
            "max_depth_gap_mm": max_gap,
            "split_depth_mm": split_depth,
            "effective_cut_band_mm": float(effective_cut_band),
            "near_threshold_mm": float(near_threshold),
            "far_threshold_mm": float(far_threshold),
            "initial_near_layer_area": initial_near_area,
            "initial_far_layer_area": initial_far_area,
            "near_layer_area": near_area,
            "far_layer_area": far_area,
            **line_debug,
            "near_bridge_removed_pixels": near_bridge_debug["bridge_removed_pixels"],
            "far_bridge_removed_pixels": far_bridge_debug["bridge_removed_pixels"],
            "required_layer_area": int(min_area),
        }
    )
    if max_gap < float(min_depth_gap_mm):
        return DepthLayerSplit(None, {**debug, "applied": False, "reason": "depth_gap_too_small"})
    line_cut_applied = bool(line_debug.get("line_cut_applied", False))
    if not line_cut_applied and (initial_near_area < min_area or initial_far_area < min_area):
        return DepthLayerSplit(None, {**debug, "applied": False, "reason": "depth_layer_too_small"})

    selected_mask, selected_layer, component_debug = _largest_depth_layer_component(
        near,
        far,
        int(min_component_area_px),
    )
    if selected_mask is None:
        return DepthLayerSplit(None, {**debug, **component_debug, "applied": False, "reason": "depth_component_too_small"})
    return DepthLayerSplit(
        selected_mask,
        {
            **debug,
            **component_debug,
            "applied": True,
            "reason": None,
            "selected_layer": selected_layer,
            "selected_surface_area": int(np.count_nonzero(selected_mask)),
        },
    )


def _break_narrow_bridges(mask: np.ndarray, kernel_px: int) -> tuple[np.ndarray, dict[str, Any]]:
    kernel_size = int(kernel_px)
    if kernel_size <= 1:
        return mask > 0, {"bridge_removed_pixels": 0}
    if kernel_size % 2 == 0:
        kernel_size += 1
    before = mask > 0
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    opened = cv2.morphologyEx(before.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0
    return opened, {"bridge_removed_pixels": int(np.count_nonzero(before & ~opened))}


def _straight_line_cut_from_minor_layer(
    surface: np.ndarray,
    near: np.ndarray,
    far: np.ndarray,
    thickness_px: int,
    min_aspect_ratio: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    near_area = int(np.count_nonzero(near))
    far_area = int(np.count_nonzero(far))
    minor_name = "near" if near_area <= far_area else "far"
    minor = near if minor_name == "near" else far
    ys, xs = np.where(minor)
    if xs.size < 2:
        return np.zeros(surface.shape[:2], dtype=bool), {
            "line_cut_applied": False,
            "line_cut_reason": "minor_layer_too_small",
            "line_cut_minor_layer": minor_name,
            "line_cut_pixels": 0,
        }

    coords = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    center = coords.mean(axis=0)
    centered = coords - center
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    if singular_values.size < 2 or singular_values[1] < 1e-6:
        aspect = float("inf")
    else:
        aspect = float(singular_values[0] / max(singular_values[1], 1e-6))
    if aspect < float(min_aspect_ratio):
        return np.zeros(surface.shape[:2], dtype=bool), {
            "line_cut_applied": False,
            "line_cut_reason": "minor_layer_not_linear",
            "line_cut_minor_layer": minor_name,
            "line_cut_aspect_ratio": aspect,
            "line_cut_pixels": 0,
        }

    direction = vh[0]
    length = float(np.hypot(surface.shape[1], surface.shape[0]))
    p1 = center - direction * length
    p2 = center + direction * length
    cut = np.zeros(surface.shape[:2], dtype=np.uint8)
    thickness = max(1, int(thickness_px))
    cv2.line(
        cut,
        (int(round(p1[0])), int(round(p1[1]))),
        (int(round(p2[0])), int(round(p2[1]))),
        1,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
    cut_mask = (cut > 0) & (surface > 0)
    return cut_mask, {
        "line_cut_applied": True,
        "line_cut_reason": None,
        "line_cut_minor_layer": minor_name,
        "line_cut_aspect_ratio": aspect,
        "line_cut_center_xy": [float(center[0]), float(center[1])],
        "line_cut_direction_xy": [float(direction[0]), float(direction[1])],
        "line_cut_pixels": int(np.count_nonzero(cut_mask)),
    }


def _largest_depth_layer_component(
    near: np.ndarray,
    far: np.ndarray,
    min_component_area_px: int,
) -> tuple[np.ndarray | None, str | None, dict[str, Any]]:
    best_mask: np.ndarray | None = None
    best_layer: str | None = None
    best_area = 0
    debug: dict[str, Any] = {}
    for layer_name, layer_mask in (("near", near), ("far", far)):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(layer_mask.astype(np.uint8), connectivity=8)
        layer_best_area = 0
        layer_best_label = 0
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area > layer_best_area:
                layer_best_area = area
                layer_best_label = label
        debug[f"{layer_name}_component_count"] = int(max(count - 1, 0))
        debug[f"{layer_name}_largest_component_area"] = int(layer_best_area)
        if layer_best_area > best_area:
            best_area = layer_best_area
            best_layer = layer_name
            best_mask = labels == layer_best_label

    if best_mask is None or best_area < int(min_component_area_px):
        return None, None, {**debug, "selected_component_area": int(best_area)}
    return best_mask, best_layer, {**debug, "selected_component_area": int(best_area)}

