from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from src.utils.suction_footprint import single_cup_disk_mask


def _pixels_to_camera(xs: np.ndarray, ys: np.ndarray, depths_mm: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    z = depths_mm.astype(np.float64)
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    return np.stack((x, y, z), axis=1)


def _normal_toward_camera(normal_camera: np.ndarray) -> np.ndarray:
    """카메라 쪽(=-z)을 향하도록 정규화한 법선. 솟음(+) 부호 기준을 맞춘다."""
    normal = np.asarray(normal_camera, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-9:
        return np.array([0.0, 0.0, -1.0], dtype=np.float64)
    normal = normal / norm
    if normal[2] > 0.0:
        normal = -normal
    return normal


def cup_protrusion_collision(
    depth_image: np.ndarray,
    intrinsic: np.ndarray,
    center_xy: tuple[float, float],
    radius_px: float,
    seating_depth_mm: float,
    normal_camera: np.ndarray,
    target_mask: np.ndarray,
    protrusion_tol_mm: float,
    min_valid_ratio: float,
) -> dict[str, Any]:
    """비활성 컵 원판 안에서 '착좌면보다 위로 솟은' 비타깃 표면을 검사.

    솟음 = 접근 법선 방향으로 착좌점보다 카메라 쪽으로 protrusion_tol_mm 초과.
    유효 깊이 비율이 min_valid_ratio 미만이면 불확실 → 충돌로 간주(보수적).
    """
    height, width = depth_image.shape[:2]
    disk = single_cup_disk_mask((height, width), center_xy, radius_px)
    disk_pixels = int(np.count_nonzero(disk))
    if disk_pixels <= 0:
        return {"collision": True, "reason": "empty_disk", "clearance_mm": -np.inf,
                "max_protrusion_mm": np.inf, "valid_ratio": 0.0}

    depth = np.asarray(depth_image, dtype=np.float64)
    target = target_mask > 0
    valid = disk & np.isfinite(depth) & (depth > 0.0) & (~target)
    valid_ratio = float(np.count_nonzero(valid) / disk_pixels)
    if valid_ratio < float(min_valid_ratio):
        return {"collision": True, "reason": "insufficient_valid_depth",
                "clearance_mm": -np.inf, "max_protrusion_mm": np.inf, "valid_ratio": valid_ratio}

    normal = _normal_toward_camera(normal_camera)
    seat = _pixels_to_camera(
        np.array([float(center_xy[0])]), np.array([float(center_xy[1])]),
        np.array([float(seating_depth_mm)]), intrinsic,
    )[0]

    ys, xs = np.where(valid)
    zs = depth[ys, xs]
    points = _pixels_to_camera(xs.astype(np.float64), ys.astype(np.float64), zs, intrinsic)
    residuals = (points - seat.reshape(1, 3)) @ normal  # 양수 = 카메라 쪽으로 솟음
    max_protrusion = float(np.max(residuals)) if residuals.size else -np.inf
    collision = max_protrusion > float(protrusion_tol_mm)
    return {
        "collision": bool(collision),
        "reason": "protrusion_above_seating" if collision else None,
        "max_protrusion_mm": max_protrusion,
        "clearance_mm": float(protrusion_tol_mm) - max_protrusion,
        "valid_ratio": valid_ratio,
    }


def active_cup_grabs_neighbor(
    depth_image: np.ndarray,
    disk_mask: np.ndarray,
    others_mask: np.ndarray,
    seating_depth_mm: float,
    seal_band_mm: float,
    max_neighbor_ratio: float,
) -> dict[str, Any]:
    """활성 컵 원판 안에서 다른 인스턴스가 착좌깊이 ±seal_band 안에 충분히 들어오면 이웃-흡착 위험."""
    disk = disk_mask > 0
    disk_pixels = int(np.count_nonzero(disk))
    if disk_pixels <= 0:
        return {"grabs": False, "neighbor_ratio": 0.0}

    depth = np.asarray(depth_image, dtype=np.float64)
    others = others_mask > 0
    near = (
        disk
        & others
        & np.isfinite(depth)
        & (depth > 0.0)
        & (np.abs(depth - float(seating_depth_mm)) <= float(seal_band_mm))
    )
    neighbor_ratio = float(np.count_nonzero(near) / disk_pixels)
    return {
        "grabs": bool(neighbor_ratio > float(max_neighbor_ratio)),
        "neighbor_ratio": neighbor_ratio,
    }


def union_of_other_masks(
    instances: Sequence[Any],
    target_index: int,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """target_index 를 제외한 모든 인스턴스 마스크의 합집합(boolean)."""
    union = np.zeros(image_shape[:2], dtype=bool)
    for index, instance in enumerate(instances):
        if index == int(target_index):
            continue
        mask = getattr(instance, "mask", None)
        if mask is None:
            continue
        m = np.asarray(mask)
        if m.shape[:2] != union.shape:
            continue
        union |= (m > 0)
    return union
