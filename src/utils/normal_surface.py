from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.utils.mask import mask_center_point


def compact_surface_attempt_debug(debug: dict[str, Any]) -> dict[str, Any]:
    keep_keys = (
        "passed",
        "reason",
        "suction_reject_reason",
        "selected",
        "selection_reason",
        "candidate_index",
        "candidate_source",
        "normal_surface_mode",
        "kmeans_k",
        "kmeans_cluster_index",
        "kmeans_cluster_rank",
        "kmeans_original_cluster_indices",
        "kmeans_merge_angle_deg",
        "kmeans_merged_cluster_count",
        "kmeans_inertia",
        "kmeans_n_init",
        "kmeans_max_iter",
        "kmeans_iterations",
        "component_index",
        "component_rank",
        "surface_center_xy",
        "surface_area",
        "object_area",
        "surface_area_ratio",
        "normal_angular_mean_deg",
        "normal_angular_std_deg",
        "normal_angular_max_deg",
        "normal_dispersion_pixels",
        "grasp_depth_source",
        "grasp_depth_valid_pixels",
        "grasp_depth_window",
        "normal_z_score",
        "robot_z_tilt_deg",
        "normal_robot",
        "normal_robot_xy_dir",
        "normal_robot_xy_norm",
        "max_robot_z_tilt_deg",
        "suction_area_pixels",
        "suction_footprint_check_used",
        "area_dominance_ratio",
        "footprint_axis_source",
        "footprint_axis_xy",
    )
    compact = {key: debug[key] for key in keep_keys if key in debug}
    directional = debug.get("directional_tilt_check")
    if isinstance(directional, dict):
        compact["directional_tilt_check"] = {
            key: directional.get(key)
            for key in (
                "enabled",
                "rejected",
                "reason",
                "allowed_robot_xy",
                "min_tilt_deg",
                "min_allowed_dot",
                "allowed_dot",
            )
            if key in directional
        }
    suction_depth = debug.get("suction_depth_check")
    if isinstance(suction_depth, dict):
        compact["suction_depth_check"] = {
            key: suction_depth.get(key)
            for key in (
                "enabled",
                "passed",
                "reason",
                "max_dual_cup_depth_diff_mm",
                "min_cup_valid_ratio",
                "dual_cup_depth_diff_mm",
                "dual_cup_plane_residual_diff_mm",
                "cup_depth_medians_mm",
                "cup_depth_iqr_mm",
                "cup_plane_residual_medians_mm",
                "cup_plane_residual_iqr_mm",
                "cup_valid_ratios",
                "cup_valid_counts",
                "cup_pixel_counts",
            )
            if key in suction_depth
        }
    split = debug.get("class3_depth_split")
    if isinstance(split, dict):
        compact["class3_depth_split"] = {
            key: split.get(key)
            for key in (
                "enabled",
                "reason",
                "max_depth_gap_mm",
                "split_depth_mm",
                "line_cut_applied",
                "line_cut_reason",
                "near_layer_area",
                "far_layer_area",
                "selected_layer",
            )
            if key in split
        }
    return compact


def surface_center_from_method(
    surface: np.ndarray,
    object_mask: np.ndarray,
    center_method: str,
    rect_max_area_ratio: float,
) -> tuple[int, int, dict[str, Any]]:
    return _surface_center(surface, object_mask, center_method, rect_max_area_ratio)


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
