from __future__ import annotations

from typing import Any

import numpy as np

from src.utils.geometry import pixel_to_camera
from src.utils.suction_footprint import SuctionFootprint, dual_cup_capsule_mask


def normal_z_score(surface_debug: dict[str, Any]) -> float:
    seed_normal = surface_debug.get("seed_normal")
    if not isinstance(seed_normal, list) or len(seed_normal) != 3:
        return 0.0
    normal = np.asarray(seed_normal, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-9:
        return 0.0
    return float(abs(normal[2] / norm))


def suction_area_coverage(
    mask: np.ndarray,
    surface: np.ndarray,
    footprint: SuctionFootprint | None,
    min_object_coverage: float,
    min_surface_coverage: float,
    suction_area: np.ndarray | None = None,
) -> dict[str, Any]:
    if footprint is None:
        return {"passed": False, "reason": "invalid_footprint"}

    if suction_area is None:
        suction_area = dual_cup_capsule_mask(mask.shape, footprint)
    suction_area_pixels = int(np.count_nonzero(suction_area))
    if suction_area_pixels <= 0:
        return {"passed": False, "reason": "empty_suction_area", "suction_area_pixels": 0}

    object_pixels = int(np.count_nonzero(suction_area & (mask > 0)))
    surface_pixels = int(np.count_nonzero(suction_area & surface))
    object_coverage = float(object_pixels / suction_area_pixels)
    surface_coverage = float(surface_pixels / suction_area_pixels)
    passed = (
        object_coverage >= float(min_object_coverage)
        and surface_coverage >= float(min_surface_coverage)
    )
    reason = None
    if not passed:
        reason = "suction_area_outside_object" if object_coverage < float(min_object_coverage) else "suction_area_crosses_normal_surface"

    return {
        "passed": bool(passed),
        "reason": reason,
        "suction_area_pixels": suction_area_pixels,
        "object_pixels": object_pixels,
        "surface_pixels": surface_pixels,
        "object_coverage": object_coverage,
        "surface_coverage": surface_coverage,
        "min_object_coverage": float(min_object_coverage),
        "min_surface_coverage": float(min_surface_coverage),
    }


def suction_plane_residual_check(
    depth_image: np.ndarray | None,
    suction_area: np.ndarray | None,
    center_xy: tuple[int, int],
    center_depth_mm: float,
    normal_camera: np.ndarray,
    intrinsic: np.ndarray,
    max_bump_mm: float,
    max_dent_mm: float,
    max_abs_p95_mm: float,
    bad_residual_mm: float,
    max_bad_ratio: float,
    min_valid_ratio: float,
) -> dict[str, Any]:
    if depth_image is None or suction_area is None or not np.any(suction_area):
        return {"passed": False, "reason": "missing_depth_or_suction_area"}

    height, width = depth_image.shape[:2]
    area = suction_area[:height, :width]
    area_pixels = int(np.count_nonzero(area))
    if area_pixels <= 0:
        return {"passed": False, "reason": "empty_suction_area", "suction_area_pixels": 0}

    depth = np.asarray(depth_image[:height, :width], dtype=np.float64)
    valid = area & np.isfinite(depth) & (depth > 0.0)
    valid_pixels = int(np.count_nonzero(valid))
    valid_ratio = float(valid_pixels / area_pixels)
    if valid_pixels <= 0 or valid_ratio < float(min_valid_ratio):
        return {
            "passed": False,
            "reason": "insufficient_valid_depth",
            "suction_area_pixels": area_pixels,
            "valid_depth_pixels": valid_pixels,
            "valid_depth_ratio": valid_ratio,
            "min_valid_depth_ratio": float(min_valid_ratio),
        }

    normal = np.asarray(normal_camera, dtype=np.float64).reshape(3)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1e-9:
        return {"passed": False, "reason": "invalid_plane_normal"}
    normal = normal / normal_norm

    center = pixel_to_camera(center_xy[0], center_xy[1], center_depth_mm, intrinsic)
    ys, xs = np.where(valid)
    zs = depth[ys, xs]
    points = _pixels_to_camera(xs.astype(np.float64), ys.astype(np.float64), zs, intrinsic)
    residuals = (points - center.reshape(1, 3)) @ normal
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size == 0:
        return {"passed": False, "reason": "invalid_plane_residuals"}

    max_positive = float(np.max(residuals))
    max_negative = float(np.min(residuals))
    max_bump = max(0.0, max_positive)
    max_dent = max(0.0, -max_negative)
    abs_residual = np.abs(residuals)
    abs_p95 = float(np.percentile(abs_residual, 95.0))
    bad_ratio = float(np.mean(abs_residual > float(bad_residual_mm)))

    passed = (
        max_bump <= float(max_bump_mm)
        and max_dent <= float(max_dent_mm)
        and abs_p95 <= float(max_abs_p95_mm)
        and bad_ratio <= float(max_bad_ratio)
    )
    reason = None
    if not passed:
        if max_bump > float(max_bump_mm):
            reason = "suction_plane_bump"
        elif max_dent > float(max_dent_mm):
            reason = "suction_plane_dent"
        elif abs_p95 > float(max_abs_p95_mm):
            reason = "suction_plane_abs_p95"
        else:
            reason = "suction_plane_bad_ratio"

    return {
        "passed": bool(passed),
        "reason": reason,
        "suction_area_pixels": area_pixels,
        "valid_depth_pixels": valid_pixels,
        "valid_depth_ratio": valid_ratio,
        "max_bump_mm": max_bump,
        "max_dent_mm": max_dent,
        "abs_p95_mm": abs_p95,
        "bad_residual_ratio": bad_ratio,
        "bad_residual_mm": float(bad_residual_mm),
        "max_bump_threshold_mm": float(max_bump_mm),
        "max_dent_threshold_mm": float(max_dent_mm),
        "max_abs_p95_threshold_mm": float(max_abs_p95_mm),
        "max_bad_ratio": float(max_bad_ratio),
    }


def _pixels_to_camera(xs: np.ndarray, ys: np.ndarray, depths_mm: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    z = depths_mm.astype(np.float64)
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    return np.stack((x, y, z), axis=1)
