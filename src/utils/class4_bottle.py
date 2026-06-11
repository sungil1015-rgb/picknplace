from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from src.utils.geometry import pixel_to_camera
from src.utils.mask import largest_component_mask, mask_center_point
from src.utils.normal import normalize_normal_image


@dataclass(frozen=True)
class Class4BottleEstimate:
    passed: bool
    reason: str | None
    point_xy: tuple[int, int] | None
    depth_mm: float | None
    normal_camera: list[float] | None
    cap_mask: np.ndarray | None
    debug: dict[str, Any]

    def to_debug_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("cap_mask", None)
        return data


def estimate_class4_bottle_surface(
    object_mask: np.ndarray,
    depth_image: np.ndarray | None,
    normal_image: np.ndarray | None,
    intrinsic: np.ndarray,
    *,
    cap_depth_percentile: float = 3.0,
    cap_depth_band_mm: float = 15.0,
    cap_anchor_percentile: float = 3.0,
    cap_anchor_band_mm: float = 5.0,
    min_cap_anchor_area_px: int = 30,
    cap_open_kernel_px: int = 3,
    cap_close_kernel_px: int = 7,
    min_cap_area_px: int = 500,
    cap_normal_window_px: int = 20,
    min_cap_normal_pixels: int = 10,
    point_source: str = "mask_center",
    endpoint_fraction: float = 1.0 / 7.0,
    endpoint_valid_ratio_margin: float = 0.05,
    min_endpoint_valid_px: int = 30,
    min_endpoint_valid_ratio: float = 0.6,
) -> Class4BottleEstimate:
    mask = largest_component_mask((object_mask > 0).astype(np.uint8))
    debug: dict[str, Any] = {
        "class4_depth_source": "cap_plane_intersection",
        "cap_depth_percentile": float(cap_depth_percentile),
        "cap_depth_band_mm": float(cap_depth_band_mm),
        "cap_anchor_percentile": float(cap_anchor_percentile),
        "cap_anchor_band_mm": float(cap_anchor_band_mm),
        "min_cap_anchor_area_px": int(min_cap_anchor_area_px),
        "min_cap_area_px": int(min_cap_area_px),
        "cap_normal_window_px": int(cap_normal_window_px),
        "min_cap_normal_pixels": int(min_cap_normal_pixels),
        "point_source": str(point_source),
        "endpoint_fraction": float(endpoint_fraction),
        "endpoint_valid_ratio_margin": float(endpoint_valid_ratio_margin),
        "min_endpoint_valid_px": int(min_endpoint_valid_px),
        "min_endpoint_valid_ratio": float(min_endpoint_valid_ratio),
    }
    if not np.any(mask):
        return _failed("empty_object_mask", debug)
    if depth_image is None or depth_image.ndim < 2:
        return _failed("missing_depth_image", debug)
    if normal_image is None:
        return _failed("missing_normal_image", debug)

    valid_depth = _valid_depth_values(depth_image, mask)
    if valid_depth.size == 0:
        return _failed("no_valid_depth_in_object", debug)

    cap_region_mask, endpoint_debug = _stable_depth_endpoint_mask(
        mask,
        depth_image,
        fraction=float(endpoint_fraction),
        valid_ratio_margin=float(endpoint_valid_ratio_margin),
        min_valid_px=int(min_endpoint_valid_px),
        min_valid_ratio=float(min_endpoint_valid_ratio),
    )
    debug.update(endpoint_debug)
    if cap_region_mask is not None:
        cap_depth_values = _valid_depth_values(depth_image, cap_region_mask)
        cap_depth_source = "stable_endpoint"
    else:
        cap_region_mask = mask
        cap_depth_values = valid_depth
        cap_depth_source = "object_nearest_percentile_fallback"

    cap_depth = float(np.percentile(cap_depth_values.astype(np.float64), float(cap_depth_percentile)))
    cap_mask = _cap_depth_mask(
        cap_region_mask,
        depth_image,
        cap_depth,
        cap_depth_band_mm=float(cap_depth_band_mm),
        open_kernel_px=int(cap_open_kernel_px),
        close_kernel_px=int(cap_close_kernel_px),
    )
    cap_area = int(np.count_nonzero(cap_mask))
    debug["cap_depth_mm"] = cap_depth
    debug["cap_depth_source"] = cap_depth_source
    debug["cap_area_px"] = cap_area
    if cap_area < int(min_cap_area_px):
        return _failed("cap_area_too_small", debug, cap_mask)

    cap_center_xy = mask_center_point(cap_mask)
    point_xy = _bottle_point(mask, point_source)
    cap_depth_at_center = _median_depth(depth_image, cap_mask)
    if cap_depth_at_center is None:
        cap_depth_at_center = cap_depth
    cap_anchor_xy, cap_anchor_depth, cap_anchor_debug = _cap_anchor_point(
        depth_image,
        cap_mask,
        percentile=float(cap_anchor_percentile),
        band_mm=float(cap_anchor_band_mm),
        min_area_px=int(min_cap_anchor_area_px),
    )
    if cap_anchor_xy is None or cap_anchor_depth is None:
        cap_anchor_xy = cap_center_xy
        cap_anchor_depth = cap_depth_at_center
        cap_anchor_debug["cap_anchor_source"] = "cap_center_median_fallback"

    cap_normal, cap_normal_debug = _center_window_normal(
        normal_image,
        cap_mask,
        cap_center_xy,
        window_px=int(cap_normal_window_px),
        min_pixels=int(min_cap_normal_pixels),
    )
    if cap_normal is None:
        return _failed("missing_cap_normal", debug, cap_mask)

    estimated_depth = ray_plane_depth_mm(
        point_xy=point_xy,
        plane_point_xy=cap_anchor_xy,
        plane_point_depth_mm=float(cap_anchor_depth),
        plane_normal_camera=cap_normal,
        intrinsic=intrinsic,
    )
    debug.update(
        {
            "cap_center_xy": [int(cap_center_xy[0]), int(cap_center_xy[1])],
            "cap_depth_at_center_mm": float(cap_depth_at_center),
            "cap_anchor_xy": [int(cap_anchor_xy[0]), int(cap_anchor_xy[1])],
            "cap_anchor_depth_mm": float(cap_anchor_depth),
            "cap_normal": [float(value) for value in cap_normal],
            **cap_normal_debug,
            "bottle_center_xy": [int(point_xy[0]), int(point_xy[1])],
            "estimated_center_depth_mm": float(estimated_depth) if estimated_depth is not None else None,
            "cap_to_center_pixel_distance": float(np.hypot(point_xy[0] - cap_center_xy[0], point_xy[1] - cap_center_xy[1])),
            "cap_anchor_to_center_pixel_distance": float(np.hypot(point_xy[0] - cap_anchor_xy[0], point_xy[1] - cap_anchor_xy[1])),
        }
    )
    debug.update(cap_anchor_debug)
    if estimated_depth is None:
        return _failed("invalid_ray_plane_intersection", debug, cap_mask)

    return Class4BottleEstimate(
        passed=True,
        reason=None,
        point_xy=(int(point_xy[0]), int(point_xy[1])),
        depth_mm=float(estimated_depth),
        normal_camera=[float(value) for value in cap_normal],
        cap_mask=cap_mask,
        debug=debug,
    )


def ray_plane_depth_mm(
    point_xy: tuple[int, int],
    plane_point_xy: tuple[int, int],
    plane_point_depth_mm: float,
    plane_normal_camera: np.ndarray,
    intrinsic: np.ndarray,
) -> float | None:
    normal = np.asarray(plane_normal_camera, dtype=np.float64).reshape(3)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1e-9 or float(plane_point_depth_mm) <= 0.0:
        return None
    normal = normal / normal_norm

    plane_point = pixel_to_camera(
        plane_point_xy[0],
        plane_point_xy[1],
        float(plane_point_depth_mm),
        intrinsic,
    )
    ray = _pixel_ray(point_xy[0], point_xy[1], intrinsic)
    denominator = float(np.dot(normal, ray))
    if abs(denominator) < 1e-9:
        return None

    depth = float(np.dot(normal, plane_point) / denominator)
    if not np.isfinite(depth) or depth <= 0.0:
        return None
    return depth


def _pixel_ray(u: int, v: int, intrinsic: np.ndarray) -> np.ndarray:
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    return np.array([(float(u) - cx) / fx, (float(v) - cy) / fy, 1.0], dtype=np.float64)


def _valid_depth_values(depth_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    height = min(depth_image.shape[0], mask.shape[0])
    width = min(depth_image.shape[1], mask.shape[1])
    bounded_mask = mask[:height, :width] > 0
    values = np.asarray(depth_image[:height, :width][bounded_mask], dtype=np.float64)
    return values[np.isfinite(values) & (values > 0.0)]


def _stable_depth_endpoint_mask(
    mask: np.ndarray,
    depth_image: np.ndarray,
    *,
    fraction: float,
    valid_ratio_margin: float,
    min_valid_px: int,
    min_valid_ratio: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    axis = _principal_axis_2d(mask)
    coords_yx = np.column_stack(np.where(mask > 0))
    debug: dict[str, Any] = {
        "cap_endpoint_axis_xy": [float(axis[0]), float(axis[1])],
        "cap_endpoint_reason": None,
    }
    if coords_yx.size == 0:
        return None, {**debug, "cap_endpoint_reason": "empty_mask"}

    coords_xy = coords_yx[:, ::-1].astype(np.float64)
    projections = coords_xy @ axis
    min_projection = float(np.min(projections))
    max_projection = float(np.max(projections))
    length = max_projection - min_projection
    if not np.isfinite(length) or length < 1.0:
        return None, {**debug, "cap_endpoint_reason": "degenerate_axis"}

    endpoint_width = max(1.0, length * max(0.01, min(0.5, float(fraction))))
    low_mask = np.zeros(mask.shape[:2], dtype=bool)
    high_mask = np.zeros(mask.shape[:2], dtype=bool)
    low_mask[coords_yx[:, 0], coords_yx[:, 1]] = projections <= min_projection + endpoint_width
    high_mask[coords_yx[:, 0], coords_yx[:, 1]] = projections >= max_projection - endpoint_width

    low_stats = _endpoint_depth_stats(depth_image, low_mask)
    high_stats = _endpoint_depth_stats(depth_image, high_mask)
    debug.update(
        {
            "cap_endpoint_fraction": float(fraction),
            "cap_endpoint_length_px": float(length),
            "cap_endpoint_width_px": float(endpoint_width),
            "cap_endpoint_low": low_stats,
            "cap_endpoint_high": high_stats,
        }
    )

    candidates: list[dict[str, Any]] = []
    for name, endpoint_mask, stats in (("low", low_mask, low_stats), ("high", high_mask, high_stats)):
        valid_count = int(stats["valid_depth_count"])
        valid_ratio = float(stats["valid_depth_ratio"])
        raw_std = stats["depth_std_mm"]
        robust_std = stats["depth_robust_std_mm"]
        if (
            valid_count < int(min_valid_px)
            or valid_ratio < float(min_valid_ratio)
            or raw_std is None
            or robust_std is None
        ):
            continue
        candidates.append(
            {
                "name": name,
                "mask": endpoint_mask,
                "valid_ratio": valid_ratio,
                "raw_std": float(raw_std),
                "robust_std": float(robust_std),
                "std_sum": float(raw_std) + float(robust_std),
            }
        )

    if not candidates:
        return None, {**debug, "cap_endpoint_reason": "no_stable_endpoint"}

    if len(candidates) == 1:
        selected = candidates[0]
        selection_reason = "single_valid_endpoint"
    else:
        low_candidate = next((candidate for candidate in candidates if candidate["name"] == "low"), None)
        high_candidate = next((candidate for candidate in candidates if candidate["name"] == "high"), None)
        if low_candidate is not None and high_candidate is not None:
            valid_ratio_diff = abs(float(low_candidate["valid_ratio"]) - float(high_candidate["valid_ratio"]))
            if valid_ratio_diff >= float(valid_ratio_margin):
                selected = max(candidates, key=lambda candidate: float(candidate["valid_ratio"]))
                selection_reason = "higher_valid_ratio"
            else:
                selected = min(candidates, key=lambda candidate: (float(candidate["std_sum"]), -float(candidate["valid_ratio"])))
                selection_reason = "lower_raw_plus_robust_std"
        else:
            selected = candidates[0]
            selection_reason = "single_valid_endpoint"

    return selected["mask"], {
        **debug,
        "cap_endpoint_reason": None,
        "cap_endpoint_valid_ratio_margin": float(valid_ratio_margin),
        "cap_endpoint_selected": str(selected["name"]),
        "cap_endpoint_selected_valid_ratio": float(selected["valid_ratio"]),
        "cap_endpoint_raw_std_score": float(selected["raw_std"]),
        "cap_endpoint_robust_std_score": float(selected["robust_std"]),
        "cap_endpoint_std_sum_score": float(selected["std_sum"]),
        "cap_endpoint_selection": selection_reason,
    }


def _endpoint_depth_stats(depth_image: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    area = int(np.count_nonzero(mask))
    values = _valid_depth_values(depth_image, mask)
    valid_count = int(values.size)
    valid_ratio = float(valid_count / max(area, 1))
    if values.size == 0:
        return {
            "area_px": area,
            "valid_depth_count": valid_count,
            "valid_depth_ratio": valid_ratio,
            "depth_median_mm": None,
            "depth_std_mm": None,
            "depth_robust_std_mm": None,
        }

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return {
        "area_px": area,
        "valid_depth_count": valid_count,
        "valid_depth_ratio": valid_ratio,
        "depth_median_mm": median,
        "depth_std_mm": float(np.std(values)),
        "depth_robust_std_mm": float(1.4826 * mad),
    }


def _principal_axis_2d(mask: np.ndarray) -> np.ndarray:
    coords_yx = np.column_stack(np.where(mask > 0))
    if coords_yx.shape[0] < 2:
        return np.array([1.0, 0.0], dtype=np.float64)
    coords_xy = coords_yx[:, ::-1].astype(np.float64)
    centered = coords_xy - coords_xy.mean(axis=0)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = np.asarray(eigenvectors[:, int(np.argmax(eigenvalues))], dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        return np.array([1.0, 0.0], dtype=np.float64)
    axis = axis / norm
    if axis[0] < 0.0 or (abs(axis[0]) < 1e-9 and axis[1] < 0.0):
        axis = -axis
    return axis


def _cap_depth_mask(
    mask: np.ndarray,
    depth_image: np.ndarray,
    cap_depth_mm: float,
    *,
    cap_depth_band_mm: float,
    open_kernel_px: int,
    close_kernel_px: int,
) -> np.ndarray:
    height = min(depth_image.shape[0], mask.shape[0])
    width = min(depth_image.shape[1], mask.shape[1])
    depth = np.asarray(depth_image[:height, :width], dtype=np.float64)
    bounded_mask = mask[:height, :width] > 0
    valid = np.isfinite(depth) & (depth > 0.0)
    cap = bounded_mask & valid & (np.abs(depth - float(cap_depth_mm)) <= float(cap_depth_band_mm))
    full_cap = np.zeros(mask.shape[:2], dtype=np.uint8)
    full_cap[:height, :width] = cap.astype(np.uint8)
    cap = _morph(full_cap, cv2.MORPH_OPEN, open_kernel_px)
    cap = _morph(cap, cv2.MORPH_CLOSE, close_kernel_px)
    cap = largest_component_mask(cap)
    return cap > 0


def _morph(mask: np.ndarray, operation: int, kernel_px: int) -> np.ndarray:
    kernel_size = int(kernel_px)
    if kernel_size <= 1:
        return mask
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.morphologyEx(mask.astype(np.uint8), operation, kernel)


def _median_depth(depth_image: np.ndarray, mask: np.ndarray) -> float | None:
    values = _valid_depth_values(depth_image, mask)
    if values.size == 0:
        return None
    return float(np.median(values))


def _cap_anchor_point(
    depth_image: np.ndarray,
    mask: np.ndarray,
    *,
    percentile: float,
    band_mm: float,
    min_area_px: int,
) -> tuple[tuple[int, int] | None, float | None, dict[str, Any]]:
    height = min(depth_image.shape[0], mask.shape[0])
    width = min(depth_image.shape[1], mask.shape[1])
    depth = np.asarray(depth_image[:height, :width], dtype=np.float64)
    bounded_mask = mask[:height, :width] > 0
    valid = bounded_mask & np.isfinite(depth) & (depth > 0.0)
    values = depth[valid]
    debug: dict[str, Any] = {
        "cap_anchor_source": "nearest_percentile_region",
        "cap_anchor_valid_depth_count": int(values.size),
    }
    if values.size == 0:
        return None, None, {**debug, "cap_anchor_reason": "no_valid_cap_depth"}

    anchor_threshold = float(np.percentile(values, float(percentile)))
    anchor_region = valid & (depth <= anchor_threshold + float(band_mm))
    anchor_area = int(np.count_nonzero(anchor_region))
    debug.update(
        {
            "cap_anchor_threshold_depth_mm": anchor_threshold,
            "cap_anchor_area_px": anchor_area,
        }
    )
    if anchor_area < int(min_area_px):
        return None, None, {**debug, "cap_anchor_reason": "cap_anchor_area_too_small"}

    ys, xs = np.where(anchor_region)
    anchor_depth = float(np.median(depth[anchor_region]))
    anchor_xy = (int(round(float(np.median(xs)))), int(round(float(np.median(ys)))))
    return anchor_xy, anchor_depth, {**debug, "cap_anchor_reason": None}


def _center_window_normal(
    normal_image: np.ndarray,
    mask: np.ndarray,
    center_xy: tuple[int, int],
    *,
    window_px: int,
    min_pixels: int,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    half = max(1, int(window_px) // 2)
    center_u, center_v = int(center_xy[0]), int(center_xy[1])
    height, width = mask.shape[:2]
    x1 = max(0, center_u - half)
    x2 = min(width, center_u + half + 1)
    y1 = max(0, center_v - half)
    y2 = min(height, center_v + half + 1)

    window_mask = np.zeros(mask.shape[:2], dtype=bool)
    window_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2] > 0
    window_normal, window_valid_count = _median_normal_with_count(normal_image, window_mask)
    debug: dict[str, Any] = {
        "cap_normal_source": "cap_center_window",
        "cap_normal_window_px": int(window_px),
        "cap_normal_min_pixels": int(min_pixels),
        "cap_normal_window_xyxy": [int(x1), int(y1), int(x2), int(y2)],
        "cap_normal_window_valid_pixels": int(window_valid_count),
    }
    if window_normal is not None and window_valid_count >= int(min_pixels):
        return window_normal, debug

    cap_normal, cap_valid_count = _median_normal_with_count(normal_image, mask)
    debug.update(
        {
            "cap_normal_source": "cap_mask_fallback",
            "cap_normal_fallback_reason": "center_window_normal_too_sparse",
            "cap_normal_cap_valid_pixels": int(cap_valid_count),
        }
    )
    return cap_normal, debug


def _median_normal(normal_image: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    normal, _ = _median_normal_with_count(normal_image, mask)
    return normal


def _median_normal_with_count(normal_image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray | None, int]:
    normals, valid = normalize_normal_image(normal_image)
    height = min(normals.shape[0], mask.shape[0])
    width = min(normals.shape[1], mask.shape[1])
    bounded_mask = (mask[:height, :width] > 0) & valid[:height, :width]
    if not np.any(bounded_mask):
        return None, 0
    values = normals[:height, :width][bounded_mask]
    normal = np.median(values, axis=0)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-9:
        return None, int(values.shape[0])
    return normal / norm, int(values.shape[0])



def _bottle_point(mask: np.ndarray, point_source: str) -> tuple[int, int]:
    if str(point_source) == "distance_transform":
        distance = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
        if np.any(distance > 0):
            y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
            return int(x), int(y)
    return mask_center_point(mask)


def _failed(reason: str, debug: dict[str, Any], cap_mask: np.ndarray | None = None) -> Class4BottleEstimate:
    return Class4BottleEstimate(
        passed=False,
        reason=reason,
        point_xy=None,
        depth_mm=None,
        normal_camera=None,
        cap_mask=cap_mask,
        debug={**debug, "reason": reason},
    )
