from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.utils.mask import mask_center_point


def normalize_normal_image(normal_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normals = np.asarray(normal_image, dtype=np.float64)
    if normals.ndim != 3 or normals.shape[2] != 3:
        raise ValueError(f"normal_image must have shape (H, W, 3), got {normals.shape}")
    norms = np.linalg.norm(normals, axis=2)
    valid = np.isfinite(norms) & (norms > 1e-6)
    normalized = np.zeros_like(normals, dtype=np.float64)
    normalized[valid] = normals[valid] / norms[valid, np.newaxis]
    return normalized, valid


def local_mean_normal(
    normal_image: np.ndarray,
    mask: np.ndarray,
    u: int,
    v: int,
    window: int,
) -> np.ndarray | None:
    normals, valid = normalize_normal_image(normal_image)
    half = max(0, int(window) // 2)
    y1 = max(0, int(v) - half)
    y2 = min(normals.shape[0], int(v) + half + 1)
    x1 = max(0, int(u) - half)
    x2 = min(normals.shape[1], int(u) + half + 1)

    region = (mask[y1:y2, x1:x2] > 0) & valid[y1:y2, x1:x2]
    if np.any(region):
        values = normals[y1:y2, x1:x2][region]
    elif 0 <= int(v) < normals.shape[0] and 0 <= int(u) < normals.shape[1] and bool(mask[int(v), int(u)]) and bool(valid[int(v), int(u)]):
        values = normals[int(v) : int(v) + 1, int(u) : int(u) + 1].reshape(1, 3)
    else:
        return None

    normal = values.mean(axis=0)
    norm = np.linalg.norm(normal)
    if norm < 1e-6:
        return None
    return normal / norm


def connected_normal_surface(
    normal_image: np.ndarray,
    mask: np.ndarray,
    seed_xy: tuple[int, int],
    seed_window: int,
    angle_threshold_deg: float,
    min_area_ratio: float,
    min_area_px: int = 0,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    seed_u, seed_v = int(seed_xy[0]), int(seed_xy[1])
    seed_normal = local_mean_normal(normal_image, mask, seed_u, seed_v, seed_window)
    if seed_normal is None:
        return None, {"passed": False, "reason": "invalid_seed_normal"}

    normals, valid = normalize_normal_image(normal_image)
    mask_bool = mask > 0
    dots = np.zeros(mask.shape[:2], dtype=np.float64)
    comparable = mask_bool & valid[: mask.shape[0], : mask.shape[1]]
    dots[comparable] = np.clip(
        np.sum(normals[: mask.shape[0], : mask.shape[1]][comparable] * seed_normal, axis=1),
        -1.0,
        1.0,
    )
    angle_map = np.full(mask.shape[:2], np.inf, dtype=np.float64)
    angle_map[comparable] = np.degrees(np.arccos(dots[comparable]))
    same_surface = comparable & (angle_map <= float(angle_threshold_deg))
    if not np.any(same_surface):
        return None, {
            "passed": False,
            "reason": "no_matching_surface",
            "seed_normal": [float(value) for value in seed_normal],
            "angle_threshold_deg": float(angle_threshold_deg),
        }

    seed = _surface_seed(same_surface, seed_u, seed_v)
    if seed is None:
        return None, {
            "passed": False,
            "reason": "no_surface_seed",
            "seed_normal": [float(value) for value in seed_normal],
            "angle_threshold_deg": float(angle_threshold_deg),
        }

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(same_surface.astype(np.uint8), connectivity=8)
    seed_label = int(labels[seed[1], seed[0]])
    if seed_label <= 0 or seed_label >= component_count:
        return None, {
            "passed": False,
            "reason": "invalid_surface_component",
            "seed_normal": [float(value) for value in seed_normal],
            "angle_threshold_deg": float(angle_threshold_deg),
        }

    surface = labels == seed_label
    surface_area = int(stats[seed_label, cv2.CC_STAT_AREA])
    mask_area = max(int(np.count_nonzero(mask_bool)), 1)
    area_ratio = float(surface_area / mask_area)
    center_u, center_v = mask_center_point(surface)
    passed = area_ratio >= float(min_area_ratio) and surface_area >= int(min_area_px)
    reason = None
    if not passed:
        reason = "surface_too_small"
        if surface_area < int(min_area_px):
            reason = "surface_area_px_too_small"
    debug = {
        "passed": bool(passed),
        "reason": reason,
        "seed_xy": [int(seed_u), int(seed_v)],
        "component_seed_xy": [int(seed[0]), int(seed[1])],
        "surface_center_xy": [int(center_u), int(center_v)],
        "surface_area": surface_area,
        "object_area": mask_area,
        "surface_area_ratio": area_ratio,
        "min_surface_region_area_ratio": float(min_area_ratio),
        "min_surface_region_area_px": int(min_area_px),
        "angle_threshold_deg": float(angle_threshold_deg),
        "seed_normal": [float(value) for value in seed_normal],
    }
    return surface, debug


def _surface_seed(surface: np.ndarray, seed_u: int, seed_v: int) -> tuple[int, int] | None:
    if 0 <= seed_v < surface.shape[0] and 0 <= seed_u < surface.shape[1] and bool(surface[seed_v, seed_u]):
        return int(seed_u), int(seed_v)
    ys, xs = np.where(surface)
    if xs.size == 0:
        return None
    distances = (xs.astype(np.float64) - float(seed_u)) ** 2 + (ys.astype(np.float64) - float(seed_v)) ** 2
    index = int(np.argmin(distances))
    return int(xs[index]), int(ys[index])
