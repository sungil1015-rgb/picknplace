from __future__ import annotations

from typing import Any, Sequence

import cv2
import numpy as np

from src.utils.collision_guard import CollisionGuardConfig, guarded_normal_near_box_wall
from src.utils.geometry import (
    approach_and_reference_to_quaternion,
    orient_normal_z_up,
    pixel_to_camera,
    project_to_tangent,
    transform_direction,
    transform_normal,
    transform_point,
)
from src.utils.class4_bottle import estimate_class4_bottle_surface
from src.utils.depth import median_valid_depth_at_point, split_surface_by_depth_gap, valid_depth_values
from src.utils.mask import largest_component_mask, mask_center_point
from src.utils.normal_surface import (
    compact_surface_attempt_debug,
    surface_center_from_method,
)
from src.utils.normal_kmeans import kmeans_normal_surface_candidates
from src.utils.suction_evaluation import normal_z_score
from src.utils.suction_footprint import (
    SuctionFootprint,
    compute_dual_cup_footprint,
    compute_projected_dual_cup_footprint,
    dual_cup_capsule_mask,
    dual_cup_normal_projected_capsule_mask,
    principal_axis_2d,
)
from src.utils.suction_config import SuctionConfig


class SuctionPipeline:
    def __init__(self, config: SuctionConfig) -> None:
        self.config = config
        self.depth_window = int(config.depth_window)
        self.normal_window = int(config.normal_window)
        self.normal_outlier_angle_deg = float(config.normal_outlier_angle_deg)
        self.normal_min_valid_pixels = int(config.normal_min_valid_pixels)
        self.normal_surface_candidate_max_count = int(config.normal_surface_candidate_max_count)
        self.cup_diameter_mm = float(config.cup_diameter_mm)
        self.cup_center_spacing_mm = float(config.cup_center_spacing_mm)
        self.collision_guard = CollisionGuardConfig(
            enabled=bool(config.collision_guard_enabled),
            box_roi=tuple(config.collision_guard_box_roi),
            wall_margin_ratio=float(config.collision_guard_wall_margin_ratio),
            min_outward_wall_component=float(config.collision_guard_min_outward_wall_component),
        )
        self.min_cup_inside_ratio = float(config.min_cup_inside_ratio)
        self.suction_strategy_default = self._normalize_suction_strategy(config.suction_strategy_default)
        self.suction_strategy_by_class = {
            int(class_index): self._normalize_suction_strategy(strategy)
            for class_index, strategy in config.suction_strategy_by_class.items()
        }
        self.normal_surface_enabled = bool(config.normal_surface_enabled)
        self.normal_surface_max_robot_z_tilt_deg = float(config.normal_surface_max_robot_z_tilt_deg)
        self.normal_surface_directional_tilt_enabled = bool(config.normal_surface_directional_tilt_enabled)
        self.normal_surface_directional_tilt_allowed_robot_xy = tuple(config.normal_surface_directional_tilt_allowed_robot_xy)
        self.normal_surface_directional_tilt_min_tilt_deg = float(config.normal_surface_directional_tilt_min_tilt_deg)
        self.normal_surface_directional_tilt_min_allowed_dot = float(config.normal_surface_directional_tilt_min_allowed_dot)
        self.normal_surface_suction_depth_check_enabled = bool(config.normal_surface_suction_depth_check_enabled)
        self.normal_surface_suction_depth_check_max_diff_mm = float(config.normal_surface_suction_depth_check_max_diff_mm)
        self.normal_surface_suction_depth_check_min_valid_ratio = float(config.normal_surface_suction_depth_check_min_valid_ratio)
        self.normal_surface_area_dominance_ratio = float(config.normal_surface_area_dominance_ratio)
        self.normal_surface_kmeans_k = int(config.normal_surface_kmeans_k)
        self.normal_surface_kmeans_merge_angle_deg = float(config.normal_surface_kmeans_merge_angle_deg)
        self.normal_surface_kmeans_n_init = int(config.normal_surface_kmeans_n_init)
        self.normal_surface_kmeans_max_iter = int(config.normal_surface_kmeans_max_iter)
        self.normal_surface_kmeans_random_state = int(config.normal_surface_kmeans_random_state)
        self.surface_open_kernel_px = int(config.surface_open_kernel_px)
        self.surface_fill_holes_max_area_px = int(config.surface_fill_holes_max_area_px)
        self.surface_fill_holes_max_aspect_ratio = float(config.surface_fill_holes_max_aspect_ratio)
        self.surface_center_method = str(config.surface_center_method)
        self.surface_rect_max_area_ratio = float(config.surface_rect_max_area_ratio)
        self.min_surface_region_area_ratio = float(config.min_surface_region_area_ratio)
        self.min_surface_region_area_px = int(config.min_surface_region_area_px)
        self.class3_depth_split_enabled = bool(config.class3_depth_split_enabled)
        self.class3_depth_split_min_gap_mm = float(config.class3_depth_split_min_gap_mm)
        self.class3_depth_split_trim_low_percentile = float(config.class3_depth_split_trim_low_percentile)
        self.class3_depth_split_trim_high_percentile = float(config.class3_depth_split_trim_high_percentile)
        self.class3_depth_split_cut_band_mm = float(config.class3_depth_split_cut_band_mm)
        self.class3_depth_split_bridge_open_kernel_px = int(config.class3_depth_split_bridge_open_kernel_px)
        self.class3_depth_split_line_cut_enabled = bool(config.class3_depth_split_line_cut_enabled)
        self.class3_depth_split_line_cut_thickness_px = int(config.class3_depth_split_line_cut_thickness_px)
        self.class3_depth_split_line_cut_min_aspect_ratio = float(config.class3_depth_split_line_cut_min_aspect_ratio)
        self.class3_depth_split_min_layer_area_px = int(config.class3_depth_split_min_layer_area_px)
        self.class3_depth_split_min_layer_area_ratio = float(config.class3_depth_split_min_layer_area_ratio)
        self.class3_depth_split_min_component_area_px = int(config.class3_depth_split_min_component_area_px)
        self.class4_bottle_cap_depth_percentile = float(config.class4_bottle_cap_depth_percentile)
        self.class4_bottle_cap_depth_band_mm = float(config.class4_bottle_cap_depth_band_mm)
        self.class4_bottle_cap_anchor_percentile = float(config.class4_bottle_cap_anchor_percentile)
        self.class4_bottle_cap_anchor_band_mm = float(config.class4_bottle_cap_anchor_band_mm)
        self.class4_bottle_min_cap_anchor_area_px = int(config.class4_bottle_min_cap_anchor_area_px)
        self.class4_bottle_cap_open_kernel_px = int(config.class4_bottle_cap_open_kernel_px)
        self.class4_bottle_cap_close_kernel_px = int(config.class4_bottle_cap_close_kernel_px)
        self.class4_bottle_min_cap_area_px = int(config.class4_bottle_min_cap_area_px)
        self.class4_bottle_cap_normal_window_px = int(config.class4_bottle_cap_normal_window_px)
        self.class4_bottle_min_cap_normal_pixels = int(config.class4_bottle_min_cap_normal_pixels)
        self.class4_bottle_point_source = str(config.class4_bottle_point_source)
        self.class4_bottle_endpoint_fraction = float(config.class4_bottle_endpoint_fraction)
        self.class4_bottle_endpoint_valid_ratio_margin = float(config.class4_bottle_endpoint_valid_ratio_margin)
        self.class4_bottle_min_endpoint_valid_px = int(config.class4_bottle_min_endpoint_valid_px)
        self.class4_bottle_min_endpoint_valid_ratio = float(config.class4_bottle_min_endpoint_valid_ratio)
        self.min_suction_area_object_coverage = float(config.min_suction_area_object_coverage)
        self.min_suction_area_surface_coverage = float(config.min_suction_area_surface_coverage)
        self.axis_min_area_px = int(config.axis_min_area_px)
        self.axis_min_rect_ratio = float(config.axis_min_rect_ratio)
        self.axis_min_pca_ratio = float(config.axis_min_pca_ratio)
        self.axis_reference_step_px = float(config.axis_reference_step_px)

    def compute(
        self,
        instances: Sequence[Any],
        depth_image: np.ndarray | None,
        normal_image: np.ndarray | None,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
    ) -> list[list[list[list[float]]]]:
        if depth_image is None:
            return [[] for _ in instances]

        suction_points: list[list[list[list[float]]]] = []
        for instance in instances:
            mask = getattr(instance, "mask", None)
            if mask is None:
                setattr(instance, "suction_footprint", None)
                setattr(instance, "suction_candidates", [])
                setattr(instance, "suction_surface", None)
                suction_points.append([])
                continue

            point, footprint, candidates, surface_debug = self._compute_one(
                instance,
                mask,
                depth_image,
                normal_image,
                intrinsic,
                extrinsic,
            )
            setattr(instance, "suction_footprint", footprint.to_dict() if footprint is not None else None)
            setattr(instance, "suction_candidates", candidates)
            setattr(instance, "suction_surface", surface_debug)
            suction_points.append([point] if point is not None else [])
        return suction_points

    def prepare_priority_depth_masks(
        self,
        instances: Sequence[Any],
        depth_image: np.ndarray | None,
        normal_image: np.ndarray | None,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
    ) -> None:
        for instance in instances:
            setattr(instance, "priority_depth_mask", None)
            setattr(instance, "priority_depth_center_xy", None)
            setattr(instance, "priority_depth_source", None)
            setattr(instance, "priority_depth_value", None)
            setattr(instance, "priority_depth_surface_debug", None)

            mask = getattr(instance, "mask", None)
            if mask is None:
                continue
            binary_mask = largest_component_mask((mask > 0).astype(np.uint8))
            if binary_mask is None or not np.any(binary_mask):
                continue

            strategy = self._suction_strategy_for_instance(instance)
            if strategy == "mask" or normal_image is None or not self.normal_surface_enabled:
                self._set_priority_depth_mask(instance, binary_mask, "mask")
                continue
            if strategy == "class4_bottle":
                selected = self._select_priority_class4_bottle_depth(
                    instance,
                    binary_mask,
                    depth_image,
                    normal_image,
                    intrinsic,
                )
                if selected is None:
                    self._set_priority_depth_mask(instance, binary_mask, "mask_class4_bottle_fallback")
                else:
                    cap_mask, depth_mm, debug = selected
                    self._set_priority_depth_mask(
                        instance,
                        cap_mask,
                        "class4_bottle_cap",
                        debug,
                        depth_value=depth_mm,
                    )
                continue

            selected = self._select_priority_normal_surface(
                instance,
                binary_mask,
                depth_image,
                normal_image,
                intrinsic,
                extrinsic,
            )
            if selected is None:
                self._set_priority_depth_mask(instance, binary_mask, "mask_fallback")
                continue
            surface, debug = selected
            self._set_priority_depth_mask(instance, surface, "normal_surface", debug)

    def _set_priority_depth_mask(
        self,
        instance: Any,
        mask: np.ndarray,
        source: str,
        debug: dict[str, Any] | None = None,
        depth_value: float | None = None,
    ) -> None:
        setattr(instance, "priority_depth_mask", mask > 0)
        setattr(instance, "priority_depth_source", str(source))
        setattr(instance, "priority_depth_value", depth_value)
        center_xy = None
        if isinstance(debug, dict):
            center_xy = debug.get("surface_center_xy")
        if isinstance(center_xy, (list, tuple)) and len(center_xy) == 2:
            setattr(instance, "priority_depth_center_xy", [int(center_xy[0]), int(center_xy[1])])
        else:
            center = mask_center_point(mask > 0)
            setattr(instance, "priority_depth_center_xy", [int(center[0]), int(center[1])])
        setattr(instance, "priority_depth_surface_debug", compact_surface_attempt_debug(debug) if isinstance(debug, dict) else None)

    def _select_priority_class4_bottle_depth(
        self,
        instance: Any,
        mask: np.ndarray,
        depth_image: np.ndarray | None,
        normal_image: np.ndarray | None,
        intrinsic: np.ndarray,
    ) -> tuple[np.ndarray, float, dict[str, Any]] | None:
        del instance
        estimate = estimate_class4_bottle_surface(
            mask,
            depth_image,
            normal_image,
            intrinsic,
            cap_depth_percentile=self.class4_bottle_cap_depth_percentile,
            cap_depth_band_mm=self.class4_bottle_cap_depth_band_mm,
            cap_anchor_percentile=self.class4_bottle_cap_anchor_percentile,
            cap_anchor_band_mm=self.class4_bottle_cap_anchor_band_mm,
            min_cap_anchor_area_px=self.class4_bottle_min_cap_anchor_area_px,
            cap_open_kernel_px=self.class4_bottle_cap_open_kernel_px,
            cap_close_kernel_px=self.class4_bottle_cap_close_kernel_px,
            min_cap_area_px=self.class4_bottle_min_cap_area_px,
            cap_normal_window_px=self.class4_bottle_cap_normal_window_px,
            min_cap_normal_pixels=self.class4_bottle_min_cap_normal_pixels,
            point_source=self.class4_bottle_point_source,
            endpoint_fraction=self.class4_bottle_endpoint_fraction,
            endpoint_valid_ratio_margin=self.class4_bottle_endpoint_valid_ratio_margin,
            min_endpoint_valid_px=self.class4_bottle_min_endpoint_valid_px,
            min_endpoint_valid_ratio=self.class4_bottle_min_endpoint_valid_ratio,
        )
        debug = {
            "passed": bool(estimate.passed),
            "reason": estimate.reason,
            "normal_surface_mode": "class4_bottle_priority",
            "class4_bottle": estimate.to_debug_dict(),
        }
        if not estimate.passed or estimate.cap_mask is None or estimate.depth_mm is None:
            return None
        debug["surface_center_xy"] = [int(estimate.point_xy[0]), int(estimate.point_xy[1])] if estimate.point_xy is not None else None
        debug["surface_area"] = int(np.count_nonzero(estimate.cap_mask))
        debug["object_area"] = int(np.count_nonzero(mask))
        debug["surface_area_ratio"] = float(debug["surface_area"] / max(debug["object_area"], 1))
        return estimate.cap_mask > 0, float(estimate.depth_mm), debug

    def _select_priority_normal_surface(
        self,
        instance: Any,
        mask: np.ndarray,
        depth_image: np.ndarray | None,
        normal_image: np.ndarray,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]] | None:
        passed: list[tuple[np.ndarray, dict[str, Any]]] = []
        for index, (surface, debug) in enumerate(self._normal_surface_candidates(normal_image, mask)):
            debug["candidate_index"] = int(index)
            debug["candidate_source"] = "kmeans"
            if surface is None or not debug.get("passed"):
                continue
            surface, debug = self._refine_class3_surface_by_depth(
                instance,
                depth_image,
                surface,
                mask,
                debug,
                intrinsic,
            )
            if surface is None or not debug.get("passed"):
                continue
            debug["normal_z_score"] = normal_z_score(debug)
            debug.update(self._robot_z_debug(debug, extrinsic))
            if debug["robot_z_tilt_deg"] > self.normal_surface_max_robot_z_tilt_deg:
                continue
            if self._reject_directional_tilt(debug):
                continue
            passed.append((surface > 0, debug))

        if not passed:
            return None
        selected, _ = self._select_surface_candidate(passed, lambda item: item[1])
        return selected

    def _compute_one(
        self,
        instance: Any,
        mask: np.ndarray,
        depth_image: np.ndarray,
        normal_image: np.ndarray | None,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
    ) -> tuple[list[list[float]] | None, SuctionFootprint | None, list[dict[str, Any]], dict[str, Any] | None]:
        binary_mask = (mask > 0).astype(np.uint8)
        binary_mask = largest_component_mask(binary_mask)
        if not np.any(binary_mask):
            return None, None, [], None

        candidates = self._generate_suction_candidates(binary_mask)
        if not candidates:
            return None, None, [], None

        strategy = self._suction_strategy_for_instance(instance)
        class4_failure_debug = None
        if strategy == "mask":
            selected = self._select_mask_suction(binary_mask, depth_image, intrinsic, candidates)
            if selected is None:
                return None, None, candidates, {"passed": False, "reason": "mask_suction_failed", "suction_strategy": "mask"}
            u, v, depth_mm, footprint, surface_debug = selected
        elif strategy == "class4_bottle":
            selected, class4_failure_debug = self._select_class4_bottle_suction(binary_mask, depth_image, normal_image, intrinsic)
            if selected is not None:
                u, v, depth_mm, footprint, surface_debug = selected
                candidates.insert(1, {"xy": [int(u), int(v)], "source": "class4_bottle_center"})
            elif self.normal_surface_enabled and normal_image is not None:
                selected = None
            else:
                return None, None, candidates, class4_failure_debug
        else:
            selected = None

        if strategy == "normal" or (strategy == "class4_bottle" and selected is None):
            if self.normal_surface_enabled and normal_image is not None:
                selected, failure_debug = self._select_normal_surface_suction(
                    instance,
                    binary_mask,
                    depth_image,
                    normal_image,
                    intrinsic,
                    extrinsic,
                )
                if selected is not None:
                    u, v, depth_mm, footprint, surface_debug = selected
                    candidates.insert(1, {"xy": [int(u), int(v)], "source": "normal_surface_center"})
                    if class4_failure_debug is not None:
                        surface_debug["class4_bottle_fallback"] = class4_failure_debug
                else:
                    if class4_failure_debug is not None and failure_debug is not None:
                        failure_debug["class4_bottle_fallback"] = class4_failure_debug
                    return None, None, candidates, failure_debug
            else:
                selected = self._select_mask_suction(binary_mask, depth_image, intrinsic, candidates)
                if selected is None:
                    return None, None, candidates, {"passed": False, "reason": "mask_suction_failed", "suction_strategy": "mask_fallback"}
                u, v, depth_mm, footprint, surface_debug = selected
                surface_debug["suction_strategy"] = "mask_fallback"

        point_camera = pixel_to_camera(u, v, depth_mm, intrinsic)
        point_robot = transform_point(point_camera, extrinsic)

        normal_camera = self._selected_surface_normal(surface_debug, normal_image, u, v, binary_mask)
        normal_robot = transform_normal(normal_camera, extrinsic)
        normal_robot = orient_normal_z_up(normal_robot)
        normal_robot, collision_debug = guarded_normal_near_box_wall(
            normal_robot,
            u=int(u),
            v=int(v),
            depth_mm=float(depth_mm),
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            config=self.collision_guard,
        )
        surface_debug["collision_guard"] = collision_debug

        reference_camera = self._selected_surface_reference_camera(
            surface_debug,
            binary_mask,
            u,
            v,
            depth_mm,
            intrinsic,
            point_camera,
            normal_camera,
        )
        reference_robot = transform_direction(reference_camera, extrinsic)
        reference_robot = project_to_tangent(reference_robot, normal_robot)
        quaternion = approach_and_reference_to_quaternion(normal_robot, reference_robot)

        return [
            [round(float(value), 3) for value in point_robot],
            [round(float(value), 6) for value in quaternion],
        ], footprint, candidates, surface_debug

    @staticmethod
    def _normalize_suction_strategy(strategy: Any) -> str:
        normalized = str(strategy).strip().lower()
        if normalized in ("mask", "2d", "2d_mask", "2dmask"):
            return "mask"
        if normalized in ("class4", "class4_bottle", "bottle"):
            return "class4_bottle"
        return "normal"

    def _suction_strategy_for_instance(self, instance: Any) -> str:
        class_index = self._class_index(instance)
        if class_index is not None and class_index in self.suction_strategy_by_class:
            return self.suction_strategy_by_class[class_index]
        return self.suction_strategy_default

    @staticmethod
    def _class_index(instance: Any) -> int | None:
        for attr in ("class_index", "label"):
            value = getattr(instance, attr, None)
            try:
                if value is not None and int(value) == 4:
                    return int(value)
                if value is not None:
                    return int(value)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _is_class4_instance(instance: Any) -> bool:
        return SuctionPipeline._class_index(instance) == 4

    def _select_mask_suction(
        self,
        mask: np.ndarray,
        depth_image: np.ndarray,
        intrinsic: np.ndarray,
        candidates: list[dict[str, Any]],
    ) -> tuple[int, int, float, SuctionFootprint | None, dict[str, Any]] | None:
        if not candidates:
            return None
        u, v = candidates[0]["xy"]
        depth_mm = median_valid_depth_at_point(depth_image, u, v, mask, window=self.depth_window)
        if depth_mm is None:
            return None
        footprint = compute_dual_cup_footprint(
            mask,
            (u, v),
            depth_mm,
            intrinsic,
            cup_diameter_mm=self.cup_diameter_mm,
            cup_center_spacing_mm=self.cup_center_spacing_mm,
            min_cup_inside_ratio=self.min_cup_inside_ratio,
        )
        surface_debug: dict[str, Any] = {
            "passed": True,
            "reason": None,
            "selected": True,
            "selection_reason": "2d_mask_candidate",
            "suction_strategy": "mask",
            "candidate_index": 0,
            "candidate_source": str(candidates[0].get("source", "unknown")),
            "surface_center_xy": [int(u), int(v)],
            "surface_area": int(np.count_nonzero(mask)),
            "object_area": int(np.count_nonzero(mask)),
            "surface_area_ratio": 1.0,
            "suction_footprint_check_used": False,
        }
        if footprint is not None:
            suction_area = dual_cup_capsule_mask(mask.shape, footprint) & (mask > 0)
            surface_debug["suction_area_pixels"] = int(np.count_nonzero(suction_area))
        return int(u), int(v), float(depth_mm), footprint, surface_debug

    def _select_class4_bottle_suction(
        self,
        mask: np.ndarray,
        depth_image: np.ndarray,
        normal_image: np.ndarray | None,
        intrinsic: np.ndarray,
    ) -> tuple[tuple[int, int, float, SuctionFootprint | None, dict[str, Any]] | None, dict[str, Any] | None]:
        estimate = estimate_class4_bottle_surface(
            mask,
            depth_image,
            normal_image,
            intrinsic,
            cap_depth_percentile=self.class4_bottle_cap_depth_percentile,
            cap_depth_band_mm=self.class4_bottle_cap_depth_band_mm,
            cap_anchor_percentile=self.class4_bottle_cap_anchor_percentile,
            cap_anchor_band_mm=self.class4_bottle_cap_anchor_band_mm,
            min_cap_anchor_area_px=self.class4_bottle_min_cap_anchor_area_px,
            cap_open_kernel_px=self.class4_bottle_cap_open_kernel_px,
            cap_close_kernel_px=self.class4_bottle_cap_close_kernel_px,
            min_cap_area_px=self.class4_bottle_min_cap_area_px,
            cap_normal_window_px=self.class4_bottle_cap_normal_window_px,
            min_cap_normal_pixels=self.class4_bottle_min_cap_normal_pixels,
            point_source=self.class4_bottle_point_source,
            endpoint_fraction=self.class4_bottle_endpoint_fraction,
            endpoint_valid_ratio_margin=self.class4_bottle_endpoint_valid_ratio_margin,
            min_endpoint_valid_px=self.class4_bottle_min_endpoint_valid_px,
            min_endpoint_valid_ratio=self.class4_bottle_min_endpoint_valid_ratio,
        )
        estimate_debug = estimate.to_debug_dict()
        if not estimate.passed or estimate.point_xy is None or estimate.depth_mm is None or estimate.normal_camera is None:
            return None, {
                "passed": False,
                "reason": estimate.reason or "class4_bottle_estimate_failed",
                "class4_bottle": estimate_debug,
            }

        u, v = estimate.point_xy
        normal_camera = np.asarray(estimate.normal_camera, dtype=np.float64)
        axis_xy, axis_debug = self._class4_object_axis(mask)
        footprint, suction_area = compute_projected_dual_cup_footprint(
            mask,
            (u, v),
            float(estimate.depth_mm),
            intrinsic,
            normal_camera,
            cup_diameter_mm=self.cup_diameter_mm,
            cup_center_spacing_mm=self.cup_center_spacing_mm,
            min_cup_inside_ratio=self.min_cup_inside_ratio,
            axis_xy=axis_xy,
        )
        surface = mask > 0
        surface_debug: dict[str, Any] = {
            "passed": True,
            "reason": None,
            "selected": True,
            "selection_reason": "class4_bottle_cap_plane_intersection",
            "candidate_index": 0,
            "candidate_source": "class4_bottle",
            "seed_xy": [int(u), int(v)],
            "component_seed_xy": [int(u), int(v)],
            "surface_center_xy": [int(u), int(v)],
            "surface_area": int(np.count_nonzero(surface)),
            "object_area": int(np.count_nonzero(mask)),
            "surface_area_ratio": 1.0,
            "seed_normal": [float(value) for value in normal_camera],
            "normal_z_score": normal_z_score({"seed_normal": [float(value) for value in normal_camera]}),
            "footprint_projection": "class4_bottle_normal_projected_ellipse",
            "suction_area_pixels": self._suction_area_pixels(mask.shape, footprint, suction_area),
            "suction_footprint_check_used": False,
            "class4_bottle": estimate_debug,
        }
        surface_debug.update(axis_debug)
        depth_check = self._suction_depth_check(
            depth_image,
            footprint,
            normal_camera,
            intrinsic,
        )
        surface_debug["suction_depth_check"] = depth_check
        if not depth_check.get("passed", True):
            surface_debug["passed"] = False
            surface_debug["suction_reject_reason"] = depth_check.get("reason", "suction_depth_check_failed")
            return None, {
                "passed": False,
                "reason": surface_debug["suction_reject_reason"],
                "class4_bottle": estimate_debug,
                "surface_attempt": compact_surface_attempt_debug(surface_debug),
            }

        return (int(u), int(v), float(estimate.depth_mm), footprint, surface_debug), None

    def _class4_object_axis(self, mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        axis, debug = self._surface_axis_or_fallback(mask, mask)
        source = debug.get("footprint_axis_source")
        if source == "normal_surface_min_area_rect":
            debug["footprint_axis_source"] = "class4_object_mask_min_area_rect"
        elif source == "normal_surface_pca_fallback":
            debug["footprint_axis_source"] = "class4_object_mask_pca_fallback"
        elif source == "object_mask_fallback":
            debug["footprint_axis_source"] = "class4_object_mask_fallback"
        return axis, debug

    def _select_normal_surface_suction(
        self,
        instance: Any,
        mask: np.ndarray,
        depth_image: np.ndarray,
        normal_image: np.ndarray | None,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
    ) -> tuple[tuple[int, int, float, SuctionFootprint | None, dict[str, Any]] | None, dict[str, Any] | None]:
        if not self.normal_surface_enabled or normal_image is None:
            return None, None

        return self._select_kmeans_normal_surface_suction(
            mask,
            depth_image,
            normal_image,
            intrinsic,
            extrinsic,
            instance,
        )

    def _select_kmeans_normal_surface_suction(
        self,
        mask: np.ndarray,
        depth_image: np.ndarray,
        normal_image: np.ndarray,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
        instance: Any | None = None,
    ) -> tuple[tuple[int, int, float, SuctionFootprint | None, dict[str, Any]] | None, dict[str, Any] | None]:
        attempts: list[dict[str, Any]] = []
        passed_options: list[tuple[int, int, float, SuctionFootprint | None, dict[str, Any]]] = []
        surfaces = self._normal_surface_candidates(normal_image, mask)
        for index, (surface, surface_debug) in enumerate(surfaces):
            surface_debug["candidate_index"] = int(index)
            surface_debug["candidate_source"] = "kmeans"
            if surface is None or not surface_debug.get("passed"):
                surface_debug["suction_reject_reason"] = surface_debug.get("reason", "normal_surface_rejected")
                attempts.append(surface_debug)
                continue

            surface, surface_debug = self._refine_class3_surface_by_depth(
                instance,
                depth_image,
                surface,
                mask,
                surface_debug,
                intrinsic,
            )
            if surface is None:
                attempts.append(surface_debug)
                continue

            u, v = surface_debug["surface_center_xy"]
            depth_mm, depth_debug = self._normal_surface_grasp_depth(depth_image, surface, mask, int(u), int(v))
            surface_debug.update(depth_debug)
            if depth_mm is None:
                surface_debug["passed"] = False
                surface_debug["suction_reject_reason"] = "missing_depth_at_normal_surface"
                attempts.append(surface_debug)
                continue

            footprint, suction_area = self._compute_surface_footprint(
                mask,
                surface,
                (u, v),
                depth_mm,
                intrinsic,
                surface_debug,
            )
            surface_debug["suction_area_pixels"] = self._suction_area_pixels(mask.shape, footprint, suction_area)
            surface_debug["suction_footprint_check_used"] = False
            surface_debug["passed"] = True
            surface_debug["reason"] = None
            surface_debug["normal_z_score"] = normal_z_score(surface_debug)
            surface_debug.update(self._robot_z_debug(surface_debug, extrinsic))
            if surface_debug["robot_z_tilt_deg"] > self.normal_surface_max_robot_z_tilt_deg:
                surface_debug["passed"] = False
                surface_debug["suction_reject_reason"] = "robot_z_tilt_too_large"
                surface_debug["max_robot_z_tilt_deg"] = float(self.normal_surface_max_robot_z_tilt_deg)
                attempts.append(surface_debug)
                continue
            if self._reject_directional_tilt(surface_debug):
                surface_debug["passed"] = False
                surface_debug["suction_reject_reason"] = "robot_z_tilt_opposite_direction"
                attempts.append(surface_debug)
                continue
            depth_check = self._suction_depth_check(
                depth_image,
                footprint,
                self._seed_normal_from_debug(surface_debug),
                intrinsic,
            )
            surface_debug["suction_depth_check"] = depth_check
            if not depth_check.get("passed", True):
                surface_debug["passed"] = False
                surface_debug["suction_reject_reason"] = depth_check.get("reason", "suction_depth_check_failed")
                attempts.append(surface_debug)
                continue
            attempts.append(surface_debug)
            passed_options.append((int(u), int(v), float(depth_mm), footprint, surface_debug))

        if passed_options:
            selected, selection_reason = self._select_surface_candidate(passed_options, lambda item: item[4])
            selected_debug = dict(selected[4])
            selected_debug["selected"] = True
            selected_debug["selection_reason"] = selection_reason
            selected_debug["area_dominance_ratio"] = float(self.normal_surface_area_dominance_ratio)
            selected_debug["attempts"] = [
                {
                    **compact_surface_attempt_debug(attempt),
                    "selected": int(attempt.get("candidate_index", -1)) == int(selected_debug.get("candidate_index", -2)),
                }
                for attempt in attempts
            ]
            return (selected[0], selected[1], selected[2], selected[3], selected_debug), None

        return None, {
            "passed": False,
            "reason": "no_normal_surface_passed",
            "normal_surface_mode": "kmeans",
            "attempts": [compact_surface_attempt_debug(attempt) for attempt in attempts],
        }

    def _normal_surface_candidates(
        self,
        normal_image: np.ndarray,
        mask: np.ndarray,
    ) -> list[tuple[np.ndarray | None, dict[str, Any]]]:
        kwargs = dict(
            min_area_ratio=self.min_surface_region_area_ratio,
            min_area_px=self.min_surface_region_area_px,
            open_kernel_px=self.surface_open_kernel_px,
            fill_holes_max_area_px=self.surface_fill_holes_max_area_px,
            fill_holes_max_aspect_ratio=self.surface_fill_holes_max_aspect_ratio,
            center_method=self.surface_center_method,
            rect_max_area_ratio=self.surface_rect_max_area_ratio,
            max_candidates=self.normal_surface_candidate_max_count,
        )
        return kmeans_normal_surface_candidates(
            normal_image,
            mask,
            k=self.normal_surface_kmeans_k,
            merge_angle_deg=self.normal_surface_kmeans_merge_angle_deg,
            n_init=self.normal_surface_kmeans_n_init,
            max_iter=self.normal_surface_kmeans_max_iter,
            random_state=self.normal_surface_kmeans_random_state,
            **kwargs,
        )

    def _refine_class3_surface_by_depth(
        self,
        instance: Any | None,
        depth_image: np.ndarray,
        surface: np.ndarray,
        object_mask: np.ndarray,
        surface_debug: dict[str, Any],
        intrinsic: np.ndarray | None,
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        if not self.class3_depth_split_enabled or self._class_index(instance) != 3:
            return surface, surface_debug

        split = split_surface_by_depth_gap(
            depth_image,
            surface,
            min_depth_gap_mm=self.class3_depth_split_min_gap_mm,
            trim_low_percentile=self.class3_depth_split_trim_low_percentile,
            trim_high_percentile=self.class3_depth_split_trim_high_percentile,
            cut_band_mm=self.class3_depth_split_cut_band_mm,
            bridge_open_kernel_px=self.class3_depth_split_bridge_open_kernel_px,
            line_cut_enabled=self.class3_depth_split_line_cut_enabled,
            line_cut_thickness_px=self.class3_depth_split_line_cut_thickness_px,
            line_cut_min_aspect_ratio=self.class3_depth_split_line_cut_min_aspect_ratio,
            min_layer_area_px=self.class3_depth_split_min_layer_area_px,
            min_layer_area_ratio=self.class3_depth_split_min_layer_area_ratio,
            min_component_area_px=self.class3_depth_split_min_component_area_px,
        )
        refined_debug = {
            **surface_debug,
            "class3_depth_split": split.debug,
        }
        if split.surface is None:
            return surface, refined_debug

        return self._apply_class3_refined_surface(
            split.surface,
            object_mask,
            surface_debug,
            refined_debug,
        )

    def _apply_class3_refined_surface(
        self,
        refined_surface: np.ndarray,
        object_mask: np.ndarray,
        surface_debug: dict[str, Any],
        refined_debug: dict[str, Any],
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        surface_area = int(np.count_nonzero(refined_surface))
        object_area = max(int(np.count_nonzero(object_mask > 0)), 1)
        area_ratio = float(surface_area / object_area)
        center_u, center_v, center_debug = surface_center_from_method(
            refined_surface,
            object_mask > 0,
            self.surface_center_method,
            self.surface_rect_max_area_ratio,
        )
        center_xy = [int(center_u), int(center_v)]
        refined_debug.update(
            {
                "surface_area_before_depth_split": int(surface_debug.get("surface_area", 0)),
                "surface_area_ratio_before_depth_split": surface_debug.get("surface_area_ratio"),
                "surface_area": surface_area,
                "surface_area_ratio": area_ratio,
                "surface_center_xy": center_xy,
                "component_seed_xy": center_xy,
                "seed_xy": center_xy,
                "surface_center_used_method": center_debug["used_method"],
                "surface_center_fallback_reason": center_debug.get("fallback_reason"),
                "surface_rect_area": center_debug.get("rect_area"),
                "surface_rect_area_ratio": center_debug.get("rect_area_ratio"),
                "surface_rect_box_xy": center_debug.get("rect_box_xy"),
                "surface_rect_center_xy": center_debug.get("rect_center_xy"),
            }
        )
        if area_ratio < self.min_surface_region_area_ratio or surface_area < self.min_surface_region_area_px:
            refined_debug["passed"] = False
            refined_debug["reason"] = "class3_depth_surface_too_small"
            refined_debug["suction_reject_reason"] = "class3_depth_surface_too_small"
            return None, refined_debug
        return refined_surface, refined_debug

    @staticmethod
    def _robot_z_tilt_deg(surface_debug: dict[str, Any], extrinsic: np.ndarray) -> float:
        return float(SuctionPipeline._robot_z_debug(surface_debug, extrinsic).get("robot_z_tilt_deg", 180.0))

    @staticmethod
    def _robot_z_debug(surface_debug: dict[str, Any], extrinsic: np.ndarray) -> dict[str, Any]:
        seed_normal = surface_debug.get("seed_normal")
        if not isinstance(seed_normal, list) or len(seed_normal) != 3:
            return {"robot_z_tilt_deg": 180.0}
        normal_camera = np.asarray(seed_normal, dtype=np.float64).reshape(3)
        if np.linalg.norm(normal_camera) < 1e-9:
            return {"robot_z_tilt_deg": 180.0}
        normal_robot = transform_normal(normal_camera, extrinsic)
        normal_robot = orient_normal_z_up(normal_robot)
        normal_robot = normal_robot / max(np.linalg.norm(normal_robot), 1e-9)
        z = float(np.clip(normal_robot[2], -1.0, 1.0))
        tilt_deg = float(np.degrees(np.arccos(z)))
        xy = normal_robot[:2]
        xy_norm = float(np.linalg.norm(xy))
        xy_dir = xy / xy_norm if xy_norm > 1e-9 else np.asarray([0.0, 0.0], dtype=np.float64)
        return {
            "robot_z_tilt_deg": tilt_deg,
            "normal_robot": [float(value) for value in normal_robot],
            "normal_robot_xy_dir": [float(value) for value in xy_dir],
            "normal_robot_xy_norm": xy_norm,
        }

    def _reject_directional_tilt(self, surface_debug: dict[str, Any]) -> bool:
        if not self.normal_surface_directional_tilt_enabled:
            return False
        tilt = float(surface_debug.get("robot_z_tilt_deg", 180.0))
        xy_dir = surface_debug.get("normal_robot_xy_dir")
        if tilt < self.normal_surface_directional_tilt_min_tilt_deg:
            surface_debug["directional_tilt_check"] = self._directional_tilt_debug(None, False, "tilt_below_threshold")
            return False
        if not isinstance(xy_dir, list) or len(xy_dir) != 2:
            surface_debug["directional_tilt_check"] = self._directional_tilt_debug(None, False, "missing_robot_xy_dir")
            return False
        allowed = np.asarray(self.normal_surface_directional_tilt_allowed_robot_xy, dtype=np.float64)
        current = np.asarray(xy_dir, dtype=np.float64)
        allowed_dot = float(np.dot(current, allowed))
        rejected = allowed_dot < self.normal_surface_directional_tilt_min_allowed_dot
        reason = "opposite_direction" if rejected else "allowed_direction"
        surface_debug["directional_tilt_check"] = self._directional_tilt_debug(allowed_dot, rejected, reason)
        return rejected

    def _directional_tilt_debug(self, allowed_dot: float | None, rejected: bool, reason: str) -> dict[str, Any]:
        return {
            "enabled": bool(self.normal_surface_directional_tilt_enabled),
            "rejected": bool(rejected),
            "reason": reason,
            "allowed_robot_xy": [float(value) for value in self.normal_surface_directional_tilt_allowed_robot_xy],
            "min_tilt_deg": float(self.normal_surface_directional_tilt_min_tilt_deg),
            "min_allowed_dot": float(self.normal_surface_directional_tilt_min_allowed_dot),
            "allowed_dot": None if allowed_dot is None else float(allowed_dot),
        }

    def _select_surface_candidate(
        self,
        items: list[Any],
        debug_getter: Any,
    ) -> tuple[Any, str]:
        if len(items) == 1:
            return items[0], "only_passed_surface"

        by_area = sorted(
            items,
            key=lambda item: (
                self._surface_area(debug_getter(item)),
                -float(debug_getter(item).get("robot_z_tilt_deg", 180.0)),
                -int(debug_getter(item).get("candidate_index", 0)),
            ),
            reverse=True,
        )
        best_area = self._surface_area(debug_getter(by_area[0]))
        second_area = self._surface_area(debug_getter(by_area[1])) if len(by_area) > 1 else 0
        dominance_ratio = max(float(self.normal_surface_area_dominance_ratio), 1.0)
        if second_area <= 0 or best_area >= second_area * dominance_ratio:
            return by_area[0], "largest_surface_area_dominates"

        selected = min(
            items,
            key=lambda item: (
                self._normal_angular_std(debug_getter(item)),
                -self._surface_area(debug_getter(item)),
                float(debug_getter(item).get("robot_z_tilt_deg", 180.0)),
                int(debug_getter(item).get("candidate_index", 0)),
            ),
        )
        return selected, "lowest_normal_angular_std_area_tie"

    @staticmethod
    def _surface_area(debug: dict[str, Any]) -> int:
        try:
            return int(debug.get("surface_area", 0))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _normal_angular_std(debug: dict[str, Any]) -> float:
        value = debug.get("normal_angular_std_deg")
        try:
            result = float(value)
        except (TypeError, ValueError):
            return float("inf")
        if not np.isfinite(result):
            return float("inf")
        return result

    def _normal_surface_grasp_depth(
        self,
        depth_image: np.ndarray | None,
        surface: np.ndarray,
        object_mask: np.ndarray,
        u: int,
        v: int,
    ) -> tuple[float | None, dict[str, Any]]:
        if depth_image is not None:
            values = valid_depth_values(depth_image, surface > 0)
            if values.size > 0:
                return float(np.median(values.astype(np.float64))), {
                    "grasp_depth_source": "normal_surface_median",
                    "grasp_depth_valid_pixels": int(values.size),
                    "grasp_depth_window": None,
                }

        fallback = median_valid_depth_at_point(
            depth_image,
            int(u),
            int(v),
            object_mask,
            window=self.depth_window,
        )
        return fallback, {
            "grasp_depth_source": "surface_median_failed_local_window_fallback" if fallback is not None else "missing_depth",
            "grasp_depth_valid_pixels": 0,
            "grasp_depth_window": int(self.depth_window),
        }

    @staticmethod
    def _suction_area_pixels(
        image_shape: tuple[int, ...],
        footprint: SuctionFootprint | None,
        suction_area: np.ndarray | None,
    ) -> int:
        if suction_area is not None:
            return int(np.count_nonzero(suction_area))
        if footprint is None:
            return 0
        return int(np.count_nonzero(dual_cup_capsule_mask(image_shape, footprint)))

    def _suction_depth_check(
        self,
        depth_image: np.ndarray | None,
        footprint: SuctionFootprint | None,
        normal_camera: np.ndarray | None,
        intrinsic: np.ndarray | None,
    ) -> dict[str, Any]:
        debug: dict[str, Any] = {
            "enabled": bool(self.normal_surface_suction_depth_check_enabled),
            "passed": True,
            "reason": None,
            "max_dual_cup_depth_diff_mm": float(self.normal_surface_suction_depth_check_max_diff_mm),
            "min_cup_valid_ratio": float(self.normal_surface_suction_depth_check_min_valid_ratio),
        }
        if not self.normal_surface_suction_depth_check_enabled:
            return debug
        if depth_image is None:
            debug.update({"passed": False, "reason": "missing_depth_image"})
            return debug
        if footprint is None:
            debug.update({"passed": False, "reason": "missing_suction_footprint"})
            return debug
        if intrinsic is None:
            debug.update({"passed": False, "reason": "missing_intrinsic"})
            return debug

        normal = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        if normal_camera is not None:
            candidate = np.asarray(normal_camera, dtype=np.float64).reshape(3)
            candidate_norm = float(np.linalg.norm(candidate))
            if candidate_norm >= 1e-9:
                normal = candidate / candidate_norm

        _, cup_masks = dual_cup_normal_projected_capsule_mask(depth_image.shape[:2], footprint, normal)
        raw_depth_medians: list[float | None] = []
        raw_depth_iqrs: list[float | None] = []
        plane_residual_medians: list[float | None] = []
        plane_residual_iqrs: list[float | None] = []
        cup_valid_ratios: list[float] = []
        cup_valid_counts: list[int] = []
        cup_pixel_counts: list[int] = []
        centers = np.asarray(footprint.cup_centers_xy, dtype=np.float64)
        if centers.shape == (2, 2):
            center_xy = centers.mean(axis=0)
        else:
            center_xy = np.asarray([0.0, 0.0], dtype=np.float64)
        center_camera = pixel_to_camera(
            float(center_xy[0]),
            float(center_xy[1]),
            float(footprint.depth_mm),
            intrinsic,
        )
        for cup_mask in cup_masks[:2]:
            pixels = int(np.count_nonzero(cup_mask))
            cup_pixel_counts.append(pixels)
            if pixels <= 0:
                raw_depth_medians.append(None)
                raw_depth_iqrs.append(None)
                plane_residual_medians.append(None)
                plane_residual_iqrs.append(None)
                cup_valid_ratios.append(0.0)
                cup_valid_counts.append(0)
                continue
            ys, xs = np.where(cup_mask)
            depths = np.asarray(depth_image[ys, xs], dtype=np.float64)
            valid_mask = np.isfinite(depths) & (depths > 0.0)
            valid = depths[valid_mask]
            valid_xs = xs[valid_mask].astype(np.float64)
            valid_ys = ys[valid_mask].astype(np.float64)
            valid_count = int(valid.size)
            cup_valid_counts.append(valid_count)
            cup_valid_ratios.append(float(valid_count / max(pixels, 1)))
            if valid_count <= 0:
                raw_depth_medians.append(None)
                raw_depth_iqrs.append(None)
                plane_residual_medians.append(None)
                plane_residual_iqrs.append(None)
                continue
            raw_q25, raw_q75 = np.percentile(valid, [25.0, 75.0])
            raw_depth_medians.append(float(np.median(valid)))
            raw_depth_iqrs.append(float(raw_q75 - raw_q25))

            points = self._pixels_to_camera_points(valid_xs, valid_ys, valid, intrinsic)
            residuals = (points - center_camera.reshape(1, 3)) @ normal
            residuals = residuals[np.isfinite(residuals)]
            if residuals.size <= 0:
                plane_residual_medians.append(None)
                plane_residual_iqrs.append(None)
                continue
            residual_q25, residual_q75 = np.percentile(residuals, [25.0, 75.0])
            plane_residual_medians.append(float(np.median(residuals)))
            plane_residual_iqrs.append(float(residual_q75 - residual_q25))

        debug.update(
            {
                "cup_depth_medians_mm": raw_depth_medians,
                "cup_depth_iqr_mm": raw_depth_iqrs,
                "cup_plane_residual_medians_mm": plane_residual_medians,
                "cup_plane_residual_iqr_mm": plane_residual_iqrs,
                "cup_valid_ratios": cup_valid_ratios,
                "cup_valid_counts": cup_valid_counts,
                "cup_pixel_counts": cup_pixel_counts,
            }
        )
        if len(plane_residual_medians) < 2 or plane_residual_medians[0] is None or plane_residual_medians[1] is None:
            debug.update({"passed": False, "reason": "missing_cup_depth"})
            return debug
        if min(cup_valid_ratios[:2], default=0.0) < self.normal_surface_suction_depth_check_min_valid_ratio:
            debug.update({"passed": False, "reason": "cup_depth_valid_ratio_too_low"})
            return debug

        residual_diff = abs(float(plane_residual_medians[0]) - float(plane_residual_medians[1]))
        debug["dual_cup_plane_residual_diff_mm"] = float(residual_diff)
        debug["dual_cup_depth_diff_mm"] = float(residual_diff)
        if residual_diff > self.normal_surface_suction_depth_check_max_diff_mm:
            debug.update({"passed": False, "reason": "dual_cup_depth_diff_too_large"})
        return debug

    @staticmethod
    def _pixels_to_camera_points(
        xs: np.ndarray,
        ys: np.ndarray,
        depths_mm: np.ndarray,
        intrinsic: np.ndarray,
    ) -> np.ndarray:
        fx = float(intrinsic[0, 0])
        fy = float(intrinsic[1, 1])
        cx = float(intrinsic[0, 2])
        cy = float(intrinsic[1, 2])
        z = depths_mm.astype(np.float64)
        x = (xs.astype(np.float64) - cx) * z / fx
        y = (ys.astype(np.float64) - cy) * z / fy
        return np.stack((x, y, z), axis=1)

    @staticmethod
    def _seed_normal_from_debug(surface_debug: dict[str, Any]) -> np.ndarray | None:
        seed_normal = surface_debug.get("seed_normal")
        if not isinstance(seed_normal, list) or len(seed_normal) != 3:
            return None
        normal = np.asarray(seed_normal, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            return None
        return normal / norm

    def _compute_surface_footprint(
        self,
        mask: np.ndarray,
        surface: np.ndarray,
        center_xy: tuple[int, int],
        depth_mm: float,
        intrinsic: np.ndarray,
        surface_debug: dict[str, Any],
    ) -> tuple[SuctionFootprint | None, np.ndarray | None]:
        axis_xy, axis_debug = self._surface_axis_or_fallback(mask, surface)
        surface_debug.update(axis_debug)
        seed_normal = surface_debug.get("seed_normal")
        if isinstance(seed_normal, list) and len(seed_normal) == 3:
            footprint, suction_area = compute_projected_dual_cup_footprint(
                mask,
                center_xy,
                depth_mm,
                intrinsic,
                np.asarray(seed_normal, dtype=np.float64),
                cup_diameter_mm=self.cup_diameter_mm,
                cup_center_spacing_mm=self.cup_center_spacing_mm,
                min_cup_inside_ratio=self.min_cup_inside_ratio,
                axis_xy=axis_xy,
            )
            if footprint is not None and suction_area is not None:
                surface_debug["footprint_projection"] = "normal_projected_ellipse"
                return footprint, suction_area

        footprint = compute_dual_cup_footprint(
            mask,
            center_xy,
            depth_mm,
            intrinsic,
            cup_diameter_mm=self.cup_diameter_mm,
            cup_center_spacing_mm=self.cup_center_spacing_mm,
            min_cup_inside_ratio=self.min_cup_inside_ratio,
            axis_xy=axis_xy,
        )
        surface_debug["footprint_projection"] = "fronto_parallel_fallback"
        return footprint, None

    def _surface_axis_or_fallback(self, mask: np.ndarray, surface: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        rect_axis, rect_ratio, rect_area = SuctionPipeline._min_area_rect_axis(surface)
        if rect_axis is not None and rect_area >= self.axis_min_area_px and rect_ratio >= self.axis_min_rect_ratio:
            return rect_axis, {
                "footprint_axis_source": "normal_surface_min_area_rect",
                "footprint_axis_xy": [float(rect_axis[0]), float(rect_axis[1])],
                "footprint_axis_rect_ratio": float(rect_ratio),
                "footprint_axis_area": int(rect_area),
                "footprint_axis_min_area_px": int(self.axis_min_area_px),
                "footprint_axis_min_rect_ratio": float(self.axis_min_rect_ratio),
            }

        axis, ratio, area = SuctionPipeline._principal_axis_with_ratio(surface)
        if area >= self.axis_min_area_px and ratio >= self.axis_min_pca_ratio:
            return axis, {
                "footprint_axis_source": "normal_surface_pca_fallback",
                "footprint_axis_xy": [float(axis[0]), float(axis[1])],
                "footprint_axis_eigen_ratio": float(ratio),
                "footprint_axis_area": int(area),
                "footprint_axis_min_area_px": int(self.axis_min_area_px),
                "footprint_axis_min_pca_ratio": float(self.axis_min_pca_ratio),
            }

        fallback = principal_axis_2d(mask)
        return fallback, {
            "footprint_axis_source": "object_mask_fallback",
            "footprint_axis_xy": [float(fallback[0]), float(fallback[1])],
            "footprint_axis_eigen_ratio": float(ratio),
            "footprint_axis_rect_ratio": float(rect_ratio),
            "footprint_axis_area": int(area),
            "footprint_axis_min_area_px": int(self.axis_min_area_px),
            "footprint_axis_min_rect_ratio": float(self.axis_min_rect_ratio),
            "footprint_axis_min_pca_ratio": float(self.axis_min_pca_ratio),
        }

    @staticmethod
    def _min_area_rect_axis(mask: np.ndarray | None) -> tuple[np.ndarray | None, float, int]:
        if mask is None or not np.any(mask):
            return None, 1.0, 0
        binary = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 1.0, int(np.count_nonzero(binary))
        contour = max(contours, key=cv2.contourArea)
        if contour.shape[0] < 3:
            return None, 1.0, int(np.count_nonzero(binary))

        (_, _), (width, height), angle = cv2.minAreaRect(contour)
        long_side = max(float(width), float(height))
        short_side = max(min(float(width), float(height)), 1e-6)
        if long_side < 1e-6:
            return None, 1.0, int(np.count_nonzero(binary))

        angle_rad = np.deg2rad(float(angle))
        if width >= height:
            axis = np.array([np.cos(angle_rad), np.sin(angle_rad)], dtype=np.float64)
        else:
            axis = np.array([-np.sin(angle_rad), np.cos(angle_rad)], dtype=np.float64)
        norm = np.linalg.norm(axis)
        if norm < 1e-9:
            return None, 1.0, int(np.count_nonzero(binary))
        axis = axis / norm
        if axis[0] < 0.0 or (abs(axis[0]) < 1e-9 and axis[1] < 0.0):
            axis = -axis
        return axis, float(long_side / short_side), int(np.count_nonzero(binary))

    @staticmethod
    def _principal_axis_with_ratio(mask: np.ndarray | None) -> tuple[np.ndarray, float, int]:
        if mask is None or not np.any(mask):
            return np.array([1.0, 0.0], dtype=np.float64), 1.0, 0
        coords = np.column_stack(np.where(mask > 0))
        area = int(coords.shape[0])
        if area < 2:
            return np.array([1.0, 0.0], dtype=np.float64), 1.0, area

        coords_xy = coords[:, ::-1].astype(np.float64)
        centered = coords_xy - coords_xy.mean(axis=0)
        covariance = np.cov(centered, rowvar=False)
        if covariance.shape != (2, 2):
            return np.array([1.0, 0.0], dtype=np.float64), 1.0, area
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        major = max(float(eigenvalues[-1]), 1e-9)
        minor = max(float(eigenvalues[0]), 1e-9)
        axis = eigenvectors[:, -1]
        norm = np.linalg.norm(axis)
        if norm < 1e-9:
            axis = np.array([1.0, 0.0], dtype=np.float64)
        else:
            axis = axis / norm
        if axis[0] < 0.0 or (abs(axis[0]) < 1e-9 and axis[1] < 0.0):
            axis = -axis
        return axis, float(major / minor), area

    def _generate_suction_candidates(self, mask: np.ndarray) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []

        def add_candidate(u: int, v: int, source: str) -> bool:
            if not (0 <= v < mask.shape[0] and 0 <= u < mask.shape[1]):
                return False
            if not bool(mask[v, u]):
                return False
            for candidate in candidates:
                prev_u, prev_v = candidate["xy"]
                if u == prev_u and v == prev_v:
                    return False
            candidates.append({"xy": [int(u), int(v)], "source": source})
            return True

        distance = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
        self._add_best_distance_transform_candidate(distance, add_candidate)

        center_u, center_v = mask_center_point(mask)
        add_candidate(center_u, center_v, "mask_center")
        return candidates

    def _add_best_distance_transform_candidate(self, distance: np.ndarray, add_candidate: Any) -> None:
        if not np.any(distance > 0):
            return
        flat_index = int(np.argmax(distance))
        y, x = np.unravel_index(flat_index, distance.shape)
        if distance[y, x] > 0:
            add_candidate(int(x), int(y), "distance_transform")

    def _principal_reference_camera(
        self,
        mask: np.ndarray,
        u: int,
        v: int,
        depth_mm: float,
        intrinsic: np.ndarray,
        center_camera: np.ndarray,
    ) -> np.ndarray:
        principal = principal_axis_2d(mask)
        step = max(1.0, self.axis_reference_step_px)
        u_ref = int(round(u + principal[0] * step))
        v_ref = int(round(v + principal[1] * step))
        u_ref = int(np.clip(u_ref, 0, mask.shape[1] - 1))
        v_ref = int(np.clip(v_ref, 0, mask.shape[0] - 1))

        ref_depth = self._median_valid_depth_depthless(depth_mm, mask, u_ref, v_ref)
        reference_camera = pixel_to_camera(u_ref, v_ref, ref_depth, intrinsic) - center_camera
        norm = np.linalg.norm(reference_camera)
        if norm < 1e-9:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return reference_camera / norm

    def _selected_surface_reference_camera(
        self,
        surface_debug: dict[str, Any] | None,
        mask: np.ndarray,
        u: int,
        v: int,
        depth_mm: float,
        intrinsic: np.ndarray,
        center_camera: np.ndarray,
        normal_camera: np.ndarray,
    ) -> np.ndarray:
        if isinstance(surface_debug, dict):
            axis = surface_debug.get("footprint_axis_xy")
            if isinstance(axis, list) and len(axis) == 2:
                reference = self._axis_reference_camera(axis, u, v, depth_mm, intrinsic, center_camera)
                if reference is not None:
                    return reference
        long_reference = self._principal_reference_camera(mask, u, v, depth_mm, intrinsic, center_camera)
        return long_reference

    def _short_axis_reference_camera(
        self,
        axis_xy: Any,
        u: int,
        v: int,
        depth_mm: float,
        intrinsic: np.ndarray,
        center_camera: np.ndarray,
        normal_camera: np.ndarray,
    ) -> np.ndarray | None:
        long_reference = self._axis_reference_camera(axis_xy, u, v, depth_mm, intrinsic, center_camera)
        if long_reference is None:
            return None
        return self._short_reference_from_long(long_reference, normal_camera)

    @staticmethod
    def _short_reference_from_long(long_reference: np.ndarray, normal_camera: np.ndarray) -> np.ndarray | None:
        long_axis = np.asarray(long_reference, dtype=np.float64).reshape(3)
        long_norm = float(np.linalg.norm(long_axis))
        normal = np.asarray(normal_camera, dtype=np.float64).reshape(3)
        normal_norm = float(np.linalg.norm(normal))
        if long_norm < 1e-9 or normal_norm < 1e-9:
            return None
        long_axis = long_axis / long_norm
        normal = normal / normal_norm
        short_axis = np.cross(long_axis, normal)
        short_norm = float(np.linalg.norm(short_axis))
        if short_norm < 1e-9:
            return None
        return short_axis / short_norm

    def _axis_reference_camera(
        self,
        axis_xy: Any,
        u: int,
        v: int,
        depth_mm: float,
        intrinsic: np.ndarray,
        center_camera: np.ndarray,
    ) -> np.ndarray | None:
        axis = np.asarray(axis_xy, dtype=np.float64).reshape(2)
        norm = float(np.linalg.norm(axis))
        if norm < 1e-9:
            return None
        axis = axis / norm
        step = max(1.0, self.axis_reference_step_px)
        u_ref = int(round(float(u) + axis[0] * step))
        v_ref = int(round(float(v) + axis[1] * step))
        reference_camera = pixel_to_camera(u_ref, v_ref, depth_mm, intrinsic) - center_camera
        ref_norm = float(np.linalg.norm(reference_camera))
        if ref_norm < 1e-9:
            return None
        return reference_camera / ref_norm

    def _selected_surface_normal(
        self,
        surface_debug: dict[str, Any] | None,
        normal_image: np.ndarray | None,
        u: int,
        v: int,
        mask: np.ndarray,
    ) -> np.ndarray:
        if isinstance(surface_debug, dict):
            seed_normal = surface_debug.get("seed_normal")
            if isinstance(seed_normal, list) and len(seed_normal) == 3:
                normal = np.asarray(seed_normal, dtype=np.float64).reshape(3)
                norm = float(np.linalg.norm(normal))
                if norm >= 1e-9:
                    return normal / norm
        return self._robust_valid_normal(normal_image, u, v, mask)

    def _median_valid_depth_depthless(self, fallback_depth: float, mask: np.ndarray, u: int, v: int) -> float:
        half = self.depth_window // 2
        y1 = max(0, v - half)
        y2 = min(mask.shape[0], v + half + 1)
        x1 = max(0, u - half)
        x2 = min(mask.shape[1], u + half + 1)

        mask_patch = mask[y1:y2, x1:x2] > 0
        if not np.any(mask_patch):
            return float(fallback_depth)
        return float(fallback_depth)

    def _robust_valid_normal(
        self,
        normal_image: np.ndarray | None,
        u: int,
        v: int,
        mask: np.ndarray,
    ) -> np.ndarray:
        fallback = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if normal_image is None:
            return fallback

        half = self.normal_window // 2
        y1 = max(0, v - half)
        y2 = min(normal_image.shape[0], v + half + 1)
        x1 = max(0, u - half)
        x2 = min(normal_image.shape[1], u + half + 1)

        normal_patch = normal_image[y1:y2, x1:x2]
        mask_patch = mask[y1:y2, x1:x2] > 0
        normals = normal_patch[mask_patch]
        normal = self._robust_normal_from_values(normals)
        if normal is not None:
            return normal

        height = min(normal_image.shape[0], mask.shape[0])
        width = min(normal_image.shape[1], mask.shape[1])
        mask_values = normal_image[:height, :width][mask[:height, :width] > 0]
        normal = self._robust_normal_from_values(mask_values)
        return normal if normal is not None else fallback

    def _robust_normal_from_values(self, values: np.ndarray) -> np.ndarray | None:
        normals = np.asarray(values, dtype=np.float64).reshape(-1, 3)
        if normals.shape[0] < max(1, self.normal_min_valid_pixels):
            return None

        finite = np.isfinite(normals).all(axis=1)
        normals = normals[finite]
        if normals.shape[0] < max(1, self.normal_min_valid_pixels):
            return None

        norms = np.linalg.norm(normals, axis=1)
        normals = normals[norms > 1e-6]
        norms = norms[norms > 1e-6]
        if normals.shape[0] < max(1, self.normal_min_valid_pixels):
            return None
        normals = normals / norms.reshape(-1, 1)

        median_normal = np.median(normals, axis=0)
        median_norm = float(np.linalg.norm(median_normal))
        if median_norm < 1e-6:
            return None
        median_normal = median_normal / median_norm

        cos_threshold = float(np.cos(np.deg2rad(self.normal_outlier_angle_deg)))
        inliers = normals[(normals @ median_normal) >= cos_threshold]
        if inliers.shape[0] < max(1, self.normal_min_valid_pixels):
            return None

        normal = inliers.mean(axis=0)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-6:
            return None
        return normal / norm
