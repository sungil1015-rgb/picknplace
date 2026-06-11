from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.utils.depth import median_valid_depth_at_point
from src.utils.suction_collision import (
    active_cup_grabs_neighbor,
    cup_protrusion_collision,
)
from src.utils.suction_evaluation import suction_plane_residual_check
from src.utils.suction_footprint import (
    compute_single_cup_footprint,
    cup_radius_px_at_depth,
    single_cup_disk_mask,
)


@dataclass(frozen=True)
class SingleCupParams:
    cup_diameter_mm: float
    min_cup_inside_ratio: float
    depth_window: int
    flat_max_bump_mm: float
    flat_max_dent_mm: float
    flat_max_abs_p95_mm: float
    flat_bad_residual_mm: float
    flat_max_bad_ratio: float
    flat_min_valid_ratio: float
    seal_band_mm: float
    max_neighbor_ratio: float
    cup_center_spacing_mm: float
    protrusion_tol_mm: float
    collision_min_valid_ratio: float
    num_rotations: int


def _mean_normal_in_disk(normal_image: np.ndarray | None, disk: np.ndarray) -> np.ndarray:
    if normal_image is None:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    vectors = np.asarray(normal_image)[disk]
    if vectors.size == 0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1)
    vectors = vectors[norms > 1e-6]
    if vectors.size == 0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    mean = vectors.mean(axis=0).astype(np.float64)
    norm = float(np.linalg.norm(mean))
    return mean / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _rotation_directions(num: int) -> list[np.ndarray]:
    count = max(1, int(num))
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return [np.array([np.cos(a), np.sin(a)], dtype=np.float64) for a in angles]


def fit_disk_plane(
    depth_image: np.ndarray,
    disk_mask: np.ndarray,
    center_xy: tuple[float, float],
    intrinsic: np.ndarray,
) -> tuple[np.ndarray, float] | None:
    """디스크 내 유효 depth 픽셀에 LSQ 평면 피팅.

    평탄성 잔차를 '중심점 anchor + 디스크 평균 법선' 대신 이 평면 기준으로 재면
    anchor 가 럼프 위에 앉거나 법선이 기울 때 잔차가 2~3배 부풀던 편향이 사라진다
    (구겨진 비닐(하리보) 실측에서 dent p90 22.9mm → 11.5mm).

    Returns:
        (normal_camera, depth_mm) — 법선은 카메라향(nz<0, 양수 잔차=솟음),
        depth 는 center_xy 픽셀 ray 와 평면의 교차 깊이. 평면이 그 점을 지나므로
        suction_plane_residual_check 에 그대로 넣으면 잔차가 평면 기준이 된다.
        픽셀 부족/퇴화 시 None.
    """
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    depth = np.asarray(depth_image, dtype=np.float64)
    valid = (disk_mask > 0) & np.isfinite(depth) & (depth > 0.0)
    ys, xs = np.where(valid)
    if ys.size < 16:
        return None
    zs = depth[ys, xs]
    points = np.stack([(xs - cx) * zs / fx, (ys - cy) * zs / fy, zs], axis=1)
    centroid = points.mean(axis=0)
    try:
        _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    normal = vt[2]
    if normal[2] > 0.0:
        normal = -normal
    ray = np.array([(float(center_xy[0]) - cx) / fx,
                    (float(center_xy[1]) - cy) / fy, 1.0], dtype=np.float64)
    denominator = float(normal @ ray)
    if abs(denominator) < 1e-9:
        return None
    depth_at_center = float(normal @ centroid) / denominator
    if not np.isfinite(depth_at_center) or depth_at_center <= 0.0:
        return None
    return normal, depth_at_center


def _best_inactive_direction(
    depth_image: np.ndarray,
    intrinsic: np.ndarray,
    active_xy: tuple[int, int],
    seating_depth_mm: float,
    normal_camera: np.ndarray,
    target_mask: np.ndarray,
    radius_px: float,
    spacing_px: float,
    params: SingleCupParams,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    best_dir: np.ndarray | None = None
    best_clearance = -np.inf
    best_result: dict[str, Any] = {}
    attempts: list[dict[str, Any]] = []
    for direction in _rotation_directions(params.num_rotations):
        inactive_center = (
            float(active_xy[0]) + direction[0] * spacing_px,
            float(active_xy[1]) + direction[1] * spacing_px,
        )
        result = cup_protrusion_collision(
            depth_image, intrinsic, inactive_center, radius_px,
            seating_depth_mm, normal_camera, target_mask,
            protrusion_tol_mm=params.protrusion_tol_mm,
            min_valid_ratio=params.collision_min_valid_ratio,
        )
        attempts.append({"direction": [float(direction[0]), float(direction[1])], **result})
        if not result["collision"] and result["clearance_mm"] > best_clearance:
            best_clearance = result["clearance_mm"]
            best_dir = direction
            best_result = result
    return best_dir, {"collision": best_result if best_dir is not None else {"collision": True},
                      "rotation_attempts": attempts}


def select_single_cup_suction(
    mask: np.ndarray,
    depth_image: np.ndarray,
    normal_image: np.ndarray | None,
    intrinsic: np.ndarray,
    others_mask: np.ndarray,
    candidates: list[dict[str, Any]],
    params: SingleCupParams,
) -> tuple[tuple[int, int, float, Any, dict[str, Any]] | None, dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        u, v = int(candidate["xy"][0]), int(candidate["xy"][1])
        reject: dict[str, Any] = {"candidate_index": index, "candidate_source": candidate.get("source", "unknown")}

        z0 = median_valid_depth_at_point(depth_image, u, v, mask, window=params.depth_window)
        if z0 is None:
            reject["reason"] = "missing_depth"
            attempts.append(reject)
            continue

        radius_px = cup_radius_px_at_depth(params.cup_diameter_mm, float(z0), intrinsic)
        if not np.isfinite(radius_px) or radius_px <= 0.0:
            reject["reason"] = "invalid_radius"
            attempts.append(reject)
            continue

        active_disk = single_cup_disk_mask(depth_image.shape, (u, v), radius_px)
        normal_camera = _mean_normal_in_disk(normal_image, active_disk)

        # 평탄성은 LSQ 피팅 평면 기준으로 측정 (anchor/법선 편향 제거).
        # 피팅 실패 시 기존 anchor+평균법선으로 폴백.
        plane = fit_disk_plane(depth_image, active_disk, (u, v), intrinsic)
        flat_normal, flat_depth = (plane if plane is not None
                                   else (normal_camera, float(z0)))
        flat = suction_plane_residual_check(
            depth_image, active_disk, (u, v), float(flat_depth), flat_normal, intrinsic,
            max_bump_mm=params.flat_max_bump_mm,
            max_dent_mm=params.flat_max_dent_mm,
            max_abs_p95_mm=params.flat_max_abs_p95_mm,
            bad_residual_mm=params.flat_bad_residual_mm,
            max_bad_ratio=params.flat_max_bad_ratio,
            min_valid_ratio=params.flat_min_valid_ratio,
        )
        if not flat.get("passed"):
            reject["reason"] = flat.get("reason", "not_flat")
            reject["flatness"] = flat
            attempts.append(reject)
            continue

        neighbor = active_cup_grabs_neighbor(
            depth_image, active_disk, others_mask, float(z0),
            seal_band_mm=params.seal_band_mm, max_neighbor_ratio=params.max_neighbor_ratio,
        )
        if neighbor["grabs"]:
            reject["reason"] = "active_cup_grabs_neighbor"
            reject["neighbor"] = neighbor
            attempts.append(reject)
            continue

        # cup_radius_px_at_depth(d)=(d/2)*f/z 라서, 중심간 거리 spacing*f/z 를 얻으려면
        # d=2*spacing 을 넣는다(반경 함수를 재활용해 거리(px)를 구하는 것 — 2x컵 아님).
        spacing_px = cup_radius_px_at_depth(2.0 * params.cup_center_spacing_mm, float(z0), intrinsic)
        best_dir, rot_debug = _best_inactive_direction(
            depth_image, intrinsic, (u, v), float(z0), normal_camera, mask,
            radius_px, spacing_px, params,
        )
        if best_dir is None:
            reject["reason"] = "inactive_cup_collision_all_directions"
            reject.update(rot_debug)
            attempts.append(reject)
            continue

        footprint = compute_single_cup_footprint(
            mask, (u, v), float(z0), intrinsic, axis_xy=best_dir,
            cup_diameter_mm=params.cup_diameter_mm, min_cup_inside_ratio=params.min_cup_inside_ratio,
        )
        if footprint is None or not footprint.feasible:
            reject["reason"] = "cup_outside_mask"
            reject["footprint"] = footprint.to_dict() if footprint is not None else None
            attempts.append(reject)
            continue

        surface_debug: dict[str, Any] = {
            "passed": True,
            "reason": None,
            "selected": True,
            "selection_reason": "single_cup_flat_clear",
            "suction_strategy": "single_cup",
            "candidate_index": index,
            "candidate_source": candidate.get("source", "unknown"),
            "surface_center_xy": [u, v],
            "surface_area": int(np.count_nonzero(mask > 0)),
            "object_area": int(np.count_nonzero(mask > 0)),
            "surface_area_ratio": 1.0,
            "seed_normal": [float(value) for value in normal_camera],
            "footprint_axis_xy": [float(best_dir[0]), float(best_dir[1])],
            "flatness": flat,
            "neighbor": neighbor,
            "collision": rot_debug["collision"],
            "rotation_attempts": rot_debug["rotation_attempts"],
            "suction_footprint_check_used": True,
        }
        return (u, v, float(z0), footprint, surface_debug), {"passed": True}

    return None, {"passed": False, "reason": "no_single_cup_candidate_passed",
                  "suction_strategy": "single_cup", "attempts": attempts}
