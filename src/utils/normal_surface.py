from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.utils.mask import mask_center_point
from src.utils.normal import normalize_normal_image


def clustered_normal_surface_candidates(
    normal_image: np.ndarray,
    mask: np.ndarray,
    angle_threshold_deg: float,
    min_area_ratio: float,
    min_area_px: int,
    max_clusters: int,
    open_kernel_px: int,
    close_kernel_px: int,
    fill_holes_max_area_px: int,
    fill_holes_max_aspect_ratio: float,
    center_method: str,
    rect_max_area_ratio: float,
) -> list[tuple[np.ndarray | None, dict[str, Any]]]:
    normals, valid = normalize_normal_image(normal_image)
    mask_bool = mask > 0
    comparable = mask_bool & valid[: mask.shape[0], : mask.shape[1]]
    object_area = max(int(np.count_nonzero(mask_bool)), 1)
    if not np.any(comparable):
        return [
            (
                None,
                {
                    "passed": False,
                    "reason": "no_valid_normal_in_mask",
                    "normal_surface_mode": "clustered",
                    "object_area": object_area,
                },
            )
        ]

    normal_map = normals[: mask.shape[0], : mask.shape[1]]
    remaining = comparable.copy()
    attempts: list[tuple[np.ndarray | None, dict[str, Any]]] = []
    cos_threshold = float(np.cos(np.deg2rad(float(angle_threshold_deg))))
    max_count = max(1, int(max_clusters))

    for cluster_index in range(max_count):
        ys, xs = np.where(remaining)
        if xs.size == 0:
            break

        seed_normal, seed_support = _dominant_normal(normal_map[ys, xs])
        if seed_normal is None:
            break

        dots = np.zeros(mask.shape[:2], dtype=np.float64)
        dots[remaining] = np.sum(normal_map[remaining] * seed_normal, axis=1)
        cluster = remaining & (dots >= cos_threshold)
        raw_pixels = int(np.count_nonzero(cluster))
        if raw_pixels <= 0:
            remaining[ys, xs] = False
            continue

        cleaned, cleanup_debug = _cleanup(
            cluster,
            open_kernel_px,
            close_kernel_px,
            fill_holes_max_area_px,
            fill_holes_max_aspect_ratio,
        )
        remaining[cluster] = False
        if not np.any(cleaned):
            attempts.append(
                (
                    None,
                    {
                        "passed": False,
                        "reason": "empty_cluster_after_cleanup",
                        "normal_surface_mode": "clustered",
                        "cluster_index": int(cluster_index),
                        "cluster_seed_support": int(seed_support),
                        "surface_raw_pixels": raw_pixels,
                        "object_area": object_area,
                        "angle_threshold_deg": float(angle_threshold_deg),
                        "seed_normal": [float(value) for value in seed_normal],
                    },
                )
            )
            continue

        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned.astype(np.uint8), connectivity=8)
        component_ids = list(range(1, component_count))
        component_ids.sort(key=lambda label: int(stats[label, cv2.CC_STAT_AREA]), reverse=True)

        for component_rank, label in enumerate(component_ids):
            surface = labels == label
            surface_area = int(stats[label, cv2.CC_STAT_AREA])
            if surface_area <= 0:
                continue
            surface_normal = _mean_normal(normal_map[surface])
            if surface_normal is None:
                surface_normal = seed_normal
            area_ratio = float(surface_area / object_area)
            center_u, center_v, center_debug = _surface_center(
                surface,
                mask_bool,
                center_method,
                rect_max_area_ratio,
            )
            passed = area_ratio >= float(min_area_ratio) and surface_area >= int(min_area_px)
            reason = None
            if not passed:
                reason = "surface_too_small"
                if surface_area < int(min_area_px):
                    reason = "surface_area_px_too_small"

            attempts.append(
                (
                    surface if passed else None,
                    {
                        "passed": bool(passed),
                        "reason": reason,
                        "normal_surface_mode": "clustered",
                        "cluster_index": int(cluster_index),
                        "component_index": int(label),
                        "component_rank": int(component_rank),
                        "cluster_seed_support": int(seed_support),
                        "seed_xy": [int(center_u), int(center_v)],
                        "component_seed_xy": [int(center_u), int(center_v)],
                        "surface_center_xy": [int(center_u), int(center_v)],
                        "surface_area": surface_area,
                        "object_area": object_area,
                        "surface_area_ratio": area_ratio,
                        "min_surface_region_area_ratio": float(min_area_ratio),
                        "min_surface_region_area_px": int(min_area_px),
                        "angle_threshold_deg": float(angle_threshold_deg),
                        "seed_normal": [float(value) for value in surface_normal],
                        "surface_raw_pixels": raw_pixels,
                        "surface_open_kernel_px": int(open_kernel_px),
                        "surface_close_kernel_px": int(close_kernel_px),
                        **cleanup_debug,
                        "surface_center_method": str(center_method),
                        "surface_center_used_method": center_debug["used_method"],
                        "surface_center_fallback_reason": center_debug.get("fallback_reason"),
                        "surface_rect_area": center_debug.get("rect_area"),
                        "surface_rect_area_ratio": center_debug.get("rect_area_ratio"),
                        "surface_rect_box_xy": center_debug.get("rect_box_xy"),
                        "surface_rect_center_xy": center_debug.get("rect_center_xy"),
                        "surface_rect_max_area_ratio": float(rect_max_area_ratio),
                    },
                )
            )

    if attempts:
        return attempts
    return [
        (
            None,
            {
                "passed": False,
                "reason": "no_clustered_surface",
                "normal_surface_mode": "clustered",
                "object_area": object_area,
            },
        )
    ]


def _dominant_normal(values: np.ndarray) -> tuple[np.ndarray | None, int]:
    if values.size == 0:
        return None, 0
    normals = np.asarray(values, dtype=np.float64).reshape(-1, 3)
    quantized = np.round(normals * 6.0).astype(np.int32)
    _, inverse, counts = np.unique(quantized, axis=0, return_inverse=True, return_counts=True)
    if counts.size == 0:
        return None, 0
    bin_index = int(np.argmax(counts))
    selected = normals[inverse == bin_index]
    normal = _mean_normal(selected)
    if normal is None:
        return None, 0
    return normal, int(counts[bin_index])


def _mean_normal(values: np.ndarray) -> np.ndarray | None:
    normals = np.asarray(values, dtype=np.float64).reshape(-1, 3)
    if normals.size == 0:
        return None
    normal = np.mean(normals, axis=0)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-9:
        return None
    return normal / norm


def _cleanup(
    surface: np.ndarray,
    open_kernel_px: int,
    close_kernel_px: int,
    fill_holes_max_area_px: int,
    fill_holes_max_aspect_ratio: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    cleaned = surface.astype(np.uint8)
    cleaned = _morph(cleaned, cv2.MORPH_OPEN, open_kernel_px)
    filled, fill_debug = _fill_small_enclosed_holes(
        cleaned > 0,
        fill_holes_max_area_px,
        fill_holes_max_aspect_ratio,
    )
    return filled, {
        "surface_close_kernel_used": False,
        "surface_hole_fill_enabled": int(fill_holes_max_area_px) > 0,
        **fill_debug,
    }


def _morph(mask: np.ndarray, operation: int, kernel_px: int) -> np.ndarray:
    kernel_size = int(kernel_px)
    if kernel_size <= 1:
        return mask
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.morphologyEx(mask.astype(np.uint8), operation, kernel)


def _fill_small_enclosed_holes(
    surface: np.ndarray,
    max_area_px: int,
    max_aspect_ratio: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    max_area = int(max_area_px)
    if max_area <= 0:
        return surface > 0, {"surface_holes_filled": 0, "surface_holes_filled_area": 0}

    max_aspect = max(float(max_aspect_ratio), 1.0)
    background = ~(surface > 0)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(background.astype(np.uint8), connectivity=8)
    filled = surface.copy() > 0
    filled_count = 0
    filled_area = 0
    height, width = surface.shape[:2]

    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0 or area > max_area:
            continue
        if x <= 0 or y <= 0 or x + w >= width or y + h >= height:
            continue
        aspect = float(max(w, h) / max(min(w, h), 1))
        if aspect >= max_aspect:
            continue
        filled[labels == label] = True
        filled_count += 1
        filled_area += area

    return filled, {
        "surface_holes_filled": int(filled_count),
        "surface_holes_filled_area": int(filled_area),
        "surface_fill_holes_max_area_px": int(max_area),
        "surface_fill_holes_max_aspect_ratio": float(max_aspect),
    }


def _surface_center(
    surface: np.ndarray,
    object_mask: np.ndarray,
    center_method: str,
    rect_max_area_ratio: float,
) -> tuple[int, int, dict[str, Any]]:
    if str(center_method) == "distance_transform":
        x, y, debug = _distance_transform_center(surface)
        return x, y, debug
    if str(center_method) == "rect_distance_transform":
        center, debug = _rect_distance_transform_center(surface, object_mask, rect_max_area_ratio)
        if center is not None:
            return center[0], center[1], debug
        x, y, fallback_debug = _distance_transform_center(surface)
        return x, y, {**debug, **fallback_debug, "fallback_reason": debug.get("fallback_reason", "invalid_rect_surface")}
    if str(center_method) == "rect_center":
        center, debug = _rect_center(surface, object_mask, rect_max_area_ratio)
        if center is not None:
            return center[0], center[1], debug
        center_xy = mask_center_point(surface)
        return int(center_xy[0]), int(center_xy[1]), {**debug, "used_method": "centroid", "fallback_reason": debug.get("fallback_reason", "invalid_rect_center")}

    center = mask_center_point(surface)
    return int(center[0]), int(center[1]), {"used_method": "centroid"}


def _distance_transform_center(surface: np.ndarray) -> tuple[int, int, dict[str, Any]]:
    distance = cv2.distanceTransform((surface > 0).astype(np.uint8), cv2.DIST_L2, 5)
    if np.any(distance > 0):
        y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
        return int(x), int(y), {"used_method": "distance_transform", "surface_center_distance_px": float(distance[y, x])}
    center = mask_center_point(surface)
    return int(center[0]), int(center[1]), {"used_method": "centroid", "fallback_reason": "empty_distance_transform"}


def _rect_distance_transform_center(
    surface: np.ndarray,
    object_mask: np.ndarray,
    rect_max_area_ratio: float,
) -> tuple[tuple[int, int] | None, dict[str, Any]]:
    rect_mask, rect_box = _min_area_rect_mask_and_box(surface)
    surface_area = max(int(np.count_nonzero(surface)), 1)
    rect_area = int(np.count_nonzero(rect_mask))
    rect_area_ratio = float(rect_area / surface_area)
    debug: dict[str, Any] = {
        "used_method": "rect_distance_transform",
        "rect_area": rect_area,
        "rect_area_ratio": rect_area_ratio,
        "rect_box_xy": rect_box,
    }
    if rect_area <= 0:
        return None, {**debug, "fallback_reason": "empty_rect_surface"}
    if rect_area_ratio > float(rect_max_area_ratio):
        return None, {**debug, "fallback_reason": "rect_area_too_large"}

    rect_surface = rect_mask & (object_mask > 0)
    if not np.any(rect_surface):
        return None, {**debug, "fallback_reason": "empty_rect_object_intersection"}
    x, y, dt_debug = _distance_transform_center(rect_surface)
    return (x, y), {**debug, **dt_debug, "used_method": "rect_distance_transform"}


def _rect_center(
    surface: np.ndarray,
    object_mask: np.ndarray,
    rect_max_area_ratio: float,
) -> tuple[tuple[int, int] | None, dict[str, Any]]:
    rect_mask, rect_box, rect_center = _min_area_rect_mask_box_center(surface)
    surface_area = max(int(np.count_nonzero(surface)), 1)
    rect_area = int(np.count_nonzero(rect_mask))
    rect_area_ratio = float(rect_area / surface_area)
    debug: dict[str, Any] = {
        "used_method": "rect_center",
        "rect_area": rect_area,
        "rect_area_ratio": rect_area_ratio,
        "rect_box_xy": rect_box,
        "rect_center_xy": [float(rect_center[0]), float(rect_center[1])] if rect_center is not None else None,
    }
    if rect_center is None or rect_area <= 0:
        return None, {**debug, "fallback_reason": "empty_rect_surface"}
    center = (int(round(float(rect_center[0]))), int(round(float(rect_center[1]))))
    if _inside_mask(object_mask, center):
        return center, debug

    nearest = _nearest_mask_point(object_mask, np.asarray(rect_center, dtype=np.float64))
    if nearest is None:
        return None, {**debug, "fallback_reason": "rect_center_outside_empty_object"}
    return nearest, {**debug, "used_method": "rect_center_nearest_object", "fallback_reason": "rect_center_outside_object"}


def _min_area_rect_mask_and_box(surface: np.ndarray) -> tuple[np.ndarray, list[list[int]] | None]:
    rect_mask, rect_box, _ = _min_area_rect_mask_box_center(surface)
    return rect_mask, rect_box


def _min_area_rect_mask_box_center(surface: np.ndarray) -> tuple[np.ndarray, list[list[int]] | None, tuple[float, float] | None]:
    binary = (surface > 0).astype(np.uint8)
    region = np.zeros(surface.shape[:2], dtype=np.uint8)
    if not np.any(binary):
        return region > 0, None, None
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return region > 0, None, None
    contour = max(contours, key=cv2.contourArea)
    if contour.shape[0] < 3:
        ys, xs = np.where(binary > 0)
        if xs.size == 0:
            return binary > 0, None, None
        box = np.array([[xs.min(), ys.min()], [xs.max(), ys.min()], [xs.max(), ys.max()], [xs.min(), ys.max()]], dtype=np.int32)
        center = (float(xs.mean()), float(ys.mean()))
    else:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect).astype(np.int32)
        center = (float(rect[0][0]), float(rect[0][1]))
    cv2.fillPoly(region, [box.reshape(-1, 1, 2)], 1)
    return region > 0, [[int(x), int(y)] for x, y in box.reshape(-1, 2)], center


def _inside_mask(mask: np.ndarray, point_xy: tuple[int, int]) -> bool:
    x, y = int(point_xy[0]), int(point_xy[1])
    return 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and bool(mask[y, x])


def _nearest_mask_point(mask: np.ndarray, point_xy: np.ndarray) -> tuple[int, int] | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    points = np.stack((xs.astype(np.float64), ys.astype(np.float64)), axis=1)
    index = int(np.argmin(np.sum((points - point_xy.reshape(1, 2)) ** 2, axis=1)))
    return int(points[index, 0]), int(points[index, 1])
