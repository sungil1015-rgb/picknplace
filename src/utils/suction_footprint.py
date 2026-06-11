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


def compute_projected_dual_cup_footprint(
    mask: np.ndarray,
    center_xy: tuple[int, int] | np.ndarray,
    depth_mm: float,
    intrinsic: np.ndarray,
    normal_camera: np.ndarray,
    cup_diameter_mm: float = DEFAULT_CUP_DIAMETER_MM,
    cup_center_spacing_mm: float = DEFAULT_CUP_CENTER_SPACING_MM,
    min_cup_inside_ratio: float = DEFAULT_MIN_CUP_INSIDE_RATIO,
    axis_xy: np.ndarray | None = None,
) -> tuple[SuctionFootprint | None, np.ndarray | None]:
    footprint = compute_dual_cup_footprint(
        mask,
        center_xy,
        depth_mm,
        intrinsic,
        cup_diameter_mm=cup_diameter_mm,
        cup_center_spacing_mm=cup_center_spacing_mm,
        min_cup_inside_ratio=min_cup_inside_ratio,
        axis_xy=axis_xy,
    )
    if footprint is None:
        return None, None

    suction_area, cup_masks = dual_cup_normal_projected_capsule_mask(mask.shape, footprint, normal_camera)
    inside_ratios = [_mask_inside_ratio(mask, cup_mask) for cup_mask in cup_masks]
    min_ratio = min(inside_ratios, default=0.0)
    feasible = min_ratio >= float(min_cup_inside_ratio)

    return (
        SuctionFootprint(
            feasible=bool(feasible),
            reason=None if feasible else "cup_outside_mask",
            depth_mm=footprint.depth_mm,
            cup_diameter_mm=footprint.cup_diameter_mm,
            cup_center_spacing_mm=footprint.cup_center_spacing_mm,
            cup_radius_px=footprint.cup_radius_px,
            cup_center_spacing_px=footprint.cup_center_spacing_px,
            axis_xy=footprint.axis_xy,
            cup_centers_xy=footprint.cup_centers_xy,
            cup_inside_ratios=[float(value) for value in inside_ratios],
            min_cup_inside_ratio=float(min_ratio),
            required_min_cup_inside_ratio=float(min_cup_inside_ratio),
        ),
        suction_area,
    )


def dual_cup_normal_projected_capsule_mask(
    image_shape: tuple[int, int],
    footprint: SuctionFootprint,
    normal_camera: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray]]:
    centers = np.asarray(footprint.cup_centers_xy, dtype=np.float64)
    if centers.shape != (2, 2):
        empty = np.zeros(image_shape[:2], dtype=bool)
        return empty, [empty.copy(), empty.copy()]

    radius = float(footprint.cup_radius_px)
    normal = np.asarray(normal_camera, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(normal)
    normal_z = abs(float(normal[2] / norm)) if norm >= 1e-9 else 1.0
    minor_scale = float(np.clip(normal_z, 0.25, 1.0))

    tilt_axis = np.asarray([normal[0], normal[1]], dtype=np.float64)
    if np.linalg.norm(tilt_axis) < 1e-9:
        minor_axis = np.array([-footprint.axis_xy[1], footprint.axis_xy[0]], dtype=np.float64)
    else:
        minor_axis = tilt_axis / np.linalg.norm(tilt_axis)
    major_axis = np.array([-minor_axis[1], minor_axis[0]], dtype=np.float64)

    suction_area = _ellipse_capsule_mask(image_shape, centers, radius, radius * minor_scale, major_axis, minor_axis)
    cup_masks = [
        _ellipse_mask(image_shape, center, radius, radius * minor_scale, major_axis, minor_axis)
        for center in centers
    ]
    return suction_area, cup_masks


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


def dual_cup_capsule_mask(image_shape: tuple[int, int], footprint: SuctionFootprint) -> np.ndarray:
    centers = np.asarray(footprint.cup_centers_xy, dtype=np.float64)
    if centers.shape != (2, 2):
        return np.zeros(image_shape[:2], dtype=bool)
    radius = float(footprint.cup_radius_px)
    height, width = image_shape[:2]

    x_min = max(0, int(np.floor(float(centers[:, 0].min()) - radius)))
    x_max = min(width - 1, int(np.ceil(float(centers[:, 0].max()) + radius)))
    y_min = max(0, int(np.floor(float(centers[:, 1].min()) - radius)))
    y_max = min(height - 1, int(np.ceil(float(centers[:, 1].max()) + radius)))
    region = np.zeros((height, width), dtype=bool)
    if x_max < x_min or y_max < y_min:
        return region

    yy, xx = np.mgrid[y_min : y_max + 1, x_min : x_max + 1]
    points = np.stack((xx.astype(np.float64), yy.astype(np.float64)), axis=-1)
    start = centers[0]
    end = centers[1]
    segment = end - start
    segment_len_sq = float(np.dot(segment, segment))
    if segment_len_sq < 1e-9:
        capsule = ((points[..., 0] - start[0]) ** 2 + (points[..., 1] - start[1]) ** 2) <= radius * radius
    else:
        t = ((points - start) @ segment) / segment_len_sq
        t = np.clip(t, 0.0, 1.0)
        closest = start + t[..., np.newaxis] * segment
        dist_sq = np.sum((points - closest) ** 2, axis=-1)
        capsule = dist_sq <= radius * radius
    region[y_min : y_max + 1, x_min : x_max + 1] = capsule
    return region


def _ellipse_capsule_mask(
    image_shape: tuple[int, int],
    centers: np.ndarray,
    major_radius: float,
    minor_radius: float,
    major_axis: np.ndarray,
    minor_axis: np.ndarray,
) -> np.ndarray:
    region = np.zeros(image_shape[:2], dtype=bool)
    if centers.shape != (2, 2):
        return region

    pad = float(max(major_radius, minor_radius))
    height, width = image_shape[:2]
    x_min = max(0, int(np.floor(float(centers[:, 0].min()) - pad)))
    x_max = min(width - 1, int(np.ceil(float(centers[:, 0].max()) + pad)))
    y_min = max(0, int(np.floor(float(centers[:, 1].min()) - pad)))
    y_max = min(height - 1, int(np.ceil(float(centers[:, 1].max()) + pad)))
    if x_max < x_min or y_max < y_min:
        return region

    yy, xx = np.mgrid[y_min : y_max + 1, x_min : x_max + 1]
    points = np.stack((xx.astype(np.float64), yy.astype(np.float64)), axis=-1)
    start, end = centers
    segment = end - start
    segment_len_sq = float(np.dot(segment, segment))
    if segment_len_sq < 1e-9:
        local = points - start
    else:
        t = ((points - start) @ segment) / segment_len_sq
        t = np.clip(t, 0.0, 1.0)
        closest = start + t[..., np.newaxis] * segment
        local = points - closest

    major = np.sum(local * major_axis.reshape(1, 1, 2), axis=2) / max(float(major_radius), 1e-6)
    minor = np.sum(local * minor_axis.reshape(1, 1, 2), axis=2) / max(float(minor_radius), 1e-6)
    region[y_min : y_max + 1, x_min : x_max + 1] = (major * major + minor * minor) <= 1.0
    return region


def _ellipse_mask(
    image_shape: tuple[int, int],
    center: np.ndarray,
    major_radius: float,
    minor_radius: float,
    major_axis: np.ndarray,
    minor_axis: np.ndarray,
) -> np.ndarray:
    centers = np.asarray([center, center], dtype=np.float64)
    return _ellipse_capsule_mask(image_shape, centers, major_radius, minor_radius, major_axis, minor_axis)


def _mask_inside_ratio(mask: np.ndarray, region: np.ndarray) -> float:
    total = int(np.count_nonzero(region))
    if total <= 0:
        return 0.0
    return float(np.count_nonzero(region & (mask > 0)) / total)


def single_cup_disk_mask(
    image_shape: tuple[int, int],
    center_xy: tuple[float, float] | np.ndarray,
    radius_px: float,
) -> np.ndarray:
    """단일 흡착 컵 원판의 boolean 마스크.

    suction_collision(비활성 컵 충돌)·single_cup(흡착점 선택)에서 컵 영역
    공통 헬퍼로 사용된다. compute_single_cup_footprint 는 inside-ratio 만
    필요해 기존 cup_inside_ratio 를 쓰므로 이 함수를 호출하지 않는다.
    """
    height, width = image_shape[:2]
    region = np.zeros((height, width), dtype=bool)
    cx = float(center_xy[0])
    cy = float(center_xy[1])
    radius = float(radius_px)
    if not np.isfinite(radius) or radius <= 0.0:
        return region

    x_min = max(0, int(np.floor(cx - radius)))
    x_max = min(width - 1, int(np.ceil(cx + radius)))
    y_min = max(0, int(np.floor(cy - radius)))
    y_max = min(height - 1, int(np.ceil(cy + radius)))
    if x_max < x_min or y_max < y_min:
        return region

    yy, xx = np.mgrid[y_min : y_max + 1, x_min : x_max + 1]
    disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius
    region[y_min : y_max + 1, x_min : x_max + 1] = disk
    return region


def cup_radius_px_at_depth(cup_diameter_mm: float, depth_mm: float, intrinsic: np.ndarray) -> float:
    """깊이 depth_mm 에서 컵 반경의 픽셀 크기."""
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    focal = (fx + fy) * 0.5
    if depth_mm <= 0.0:
        return 0.0
    return (float(cup_diameter_mm) * 0.5) * focal / float(depth_mm)


def compute_single_cup_footprint(
    mask: np.ndarray,
    center_xy: tuple[int, int] | np.ndarray,
    depth_mm: float,
    intrinsic: np.ndarray,
    axis_xy: np.ndarray,
    cup_diameter_mm: float = DEFAULT_CUP_DIAMETER_MM,
    min_cup_inside_ratio: float = DEFAULT_MIN_CUP_INSIDE_RATIO,
) -> SuctionFootprint | None:
    """컵 1개짜리 SuctionFootprint. cup_centers_xy 길이=1, spacing=0."""
    if intrinsic is None or depth_mm <= 0:
        return None
    radius_px = cup_radius_px_at_depth(cup_diameter_mm, depth_mm, intrinsic)
    if not np.isfinite(radius_px) or radius_px <= 0.0:
        return None

    axis = np.asarray(axis_xy, dtype=np.float64).reshape(2)
    axis_norm = np.linalg.norm(axis)
    axis = axis / axis_norm if axis_norm >= 1e-9 else np.array([1.0, 0.0], dtype=np.float64)

    center = np.asarray(center_xy, dtype=np.float64).reshape(2)
    ratio = cup_inside_ratio(mask, center, radius_px)
    feasible = ratio >= float(min_cup_inside_ratio)
    return SuctionFootprint(
        feasible=bool(feasible),
        reason=None if feasible else "cup_outside_mask",
        depth_mm=float(depth_mm),
        cup_diameter_mm=float(cup_diameter_mm),
        cup_center_spacing_mm=0.0,
        cup_radius_px=float(radius_px),
        cup_center_spacing_px=0.0,
        axis_xy=[float(axis[0]), float(axis[1])],
        cup_centers_xy=[[float(center[0]), float(center[1])]],
        cup_inside_ratios=[float(ratio)],
        # 듀얼컵 계약과 동일하게 min_cup_inside_ratio = min(cup_inside_ratios).
        # 단일컵은 컵이 1개라 그 값이 곧 ratio.
        min_cup_inside_ratio=float(ratio),
        required_min_cup_inside_ratio=float(min_cup_inside_ratio),
    )
