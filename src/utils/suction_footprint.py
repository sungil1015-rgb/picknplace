from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


DEFAULT_CUP_DIAMETER_MM = 25.0
DEFAULT_CUP_CENTER_SPACING_MM = 35.0
DEFAULT_MIN_CUP_INSIDE_RATIO = 0.85


@dataclass(frozen=True)
class SuctionFootprint:
    feasible: bool
    reason: str | None
    depth_mm: float
    cup_diameter_mm: float
    cup_center_spacing_mm: float
    cup_radius_px: float
    cup_center_spacing_px: float
    axis_xy: list[float]
    cup_centers_xy: list[list[float]]
    cup_inside_ratios: list[float]
    min_cup_inside_ratio: float
    required_min_cup_inside_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def principal_axis_2d(mask: np.ndarray | None) -> np.ndarray:
    if mask is None or not np.any(mask):
        return np.array([1.0, 0.0], dtype=np.float64)
    coords = np.column_stack(np.where(mask > 0))
    if coords.shape[0] < 2:
        return np.array([1.0, 0.0], dtype=np.float64)
    coords_xy = coords[:, ::-1].astype(np.float64)
    centered = coords_xy - coords_xy.mean(axis=0)
    covariance = np.cov(centered, rowvar=False)
    if covariance.shape != (2, 2):
        return np.array([1.0, 0.0], dtype=np.float64)
    _, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, -1]
    norm = np.linalg.norm(axis)
    if norm < 1e-9:
        return np.array([1.0, 0.0], dtype=np.float64)
    axis = axis / norm
    if axis[0] < 0.0 or (abs(axis[0]) < 1e-9 and axis[1] < 0.0):
        axis = -axis
    return axis


def compute_dual_cup_footprint(
    mask: np.ndarray,
    center_xy: tuple[int, int] | np.ndarray,
    depth_mm: float,
    intrinsic: np.ndarray,
    cup_diameter_mm: float = DEFAULT_CUP_DIAMETER_MM,
    cup_center_spacing_mm: float = DEFAULT_CUP_CENTER_SPACING_MM,
    min_cup_inside_ratio: float = DEFAULT_MIN_CUP_INSIDE_RATIO,
    axis_xy: np.ndarray | None = None,
) -> SuctionFootprint | None:
    if intrinsic is None or depth_mm <= 0:
        return None

    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    focal = (fx + fy) * 0.5
    radius_px = (float(cup_diameter_mm) * 0.5) * focal / float(depth_mm)
    spacing_px = float(cup_center_spacing_mm) * focal / float(depth_mm)
    if not np.isfinite(radius_px) or not np.isfinite(spacing_px) or radius_px <= 0.0:
        return None

    axis = principal_axis_2d(mask) if axis_xy is None else np.asarray(axis_xy, dtype=np.float64)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-9:
        axis = np.array([1.0, 0.0], dtype=np.float64)
    else:
        axis = axis / axis_norm

    center = np.asarray(center_xy, dtype=np.float64).reshape(2)
    offset = axis * (spacing_px * 0.5)
    cup_centers = [center - offset, center + offset]
    inside_ratios = [
        cup_inside_ratio(mask, cup_center, radius_px)
        for cup_center in cup_centers
    ]
    min_ratio = min(inside_ratios, default=0.0)
    feasible = min_ratio >= float(min_cup_inside_ratio)
    return SuctionFootprint(
        feasible=bool(feasible),
        reason=None if feasible else "cup_outside_mask",
        depth_mm=float(depth_mm),
        cup_diameter_mm=float(cup_diameter_mm),
        cup_center_spacing_mm=float(cup_center_spacing_mm),
        cup_radius_px=float(radius_px),
        cup_center_spacing_px=float(spacing_px),
        axis_xy=[float(axis[0]), float(axis[1])],
        cup_centers_xy=[[float(point[0]), float(point[1])] for point in cup_centers],
        cup_inside_ratios=[float(value) for value in inside_ratios],
        min_cup_inside_ratio=float(min_ratio),
        required_min_cup_inside_ratio=float(min_cup_inside_ratio),
    )


def cup_inside_ratio(mask: np.ndarray, center_xy: np.ndarray, radius_px: float) -> float:
    radius = float(radius_px)
    x_min = int(np.floor(float(center_xy[0]) - radius))
    x_max = int(np.ceil(float(center_xy[0]) + radius))
    y_min = int(np.floor(float(center_xy[1]) - radius))
    y_max = int(np.ceil(float(center_xy[1]) + radius))
    if x_max < x_min or y_max < y_min:
        return 0.0

    yy, xx = np.mgrid[y_min : y_max + 1, x_min : x_max + 1]
    cup_pixels = ((xx - float(center_xy[0])) ** 2 + (yy - float(center_xy[1])) ** 2) <= radius * radius
    total = int(cup_pixels.sum())
    if total == 0:
        return 0.0

    in_bounds = (yy >= 0) & (yy < mask.shape[0]) & (xx >= 0) & (xx < mask.shape[1])
    inside = np.zeros_like(cup_pixels, dtype=bool)
    valid_y = yy[in_bounds].astype(np.int64)
    valid_x = xx[in_bounds].astype(np.int64)
    inside[in_bounds] = mask[valid_y, valid_x] > 0
    return float((cup_pixels & inside).sum() / total)
