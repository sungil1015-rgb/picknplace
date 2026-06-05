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
    min_cap_area_px: int = 300,
    point_source: str = "mask_center",
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
        "point_source": str(point_source),
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

    cap_depth = float(np.percentile(valid_depth.astype(np.float64), float(cap_depth_percentile)))
    cap_mask = _cap_depth_mask(
        mask,
        depth_image,
        cap_depth,
        cap_depth_band_mm=float(cap_depth_band_mm),
        open_kernel_px=int(cap_open_kernel_px),
        close_kernel_px=int(cap_close_kernel_px),
    )
    cap_area = int(np.count_nonzero(cap_mask))
    debug["cap_depth_mm"] = cap_depth
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

    cap_normal = _median_normal(normal_image, cap_mask)
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


def _median_normal(normal_image: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    normals, valid = normalize_normal_image(normal_image)
    height = min(normals.shape[0], mask.shape[0])
    width = min(normals.shape[1], mask.shape[1])
    bounded_mask = (mask[:height, :width] > 0) & valid[:height, :width]
    if not np.any(bounded_mask):
        return None
    values = normals[:height, :width][bounded_mask]
    normal = np.median(values, axis=0)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-9:
        return None
    return normal / norm


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
