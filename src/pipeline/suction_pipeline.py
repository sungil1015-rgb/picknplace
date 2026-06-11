from __future__ import annotations

from typing import Any, Sequence

import cv2
import numpy as np

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
from src.utils.depth import median_valid_depth_at_point, split_surface_by_depth_gap
from src.utils.mask import largest_component_mask, mask_center_point
from src.utils.normal_surface import clustered_normal_surface_candidates
from src.utils.suction_evaluation import (
    normal_z_score,
    suction_area_coverage,
)
from src.utils.suction_footprint import (
    SuctionFootprint,
    compute_dual_cup_footprint,
    compute_projected_dual_cup_footprint,
    dual_cup_capsule_mask,
    principal_axis_2d,
)
from src.utils.suction_config import SuctionConfig
from src.utils.suction_collision import union_of_other_masks
from src.utils.single_cup import SingleCupParams, select_single_cup_suction


class SuctionPipeline:
    def __init__(self, config: SuctionConfig) -> None:
        self.config = config
        self.depth_window = int(config.depth_window)
        self.normal_window = int(config.normal_window)
        self.cup_diameter_mm = float(config.cup_diameter_mm)
        self.cup_center_spacing_mm = float(config.cup_center_spacing_mm)
        self.min_cup_inside_ratio = float(config.min_cup_inside_ratio)
        self.suction_strategy_default = self._normalize_suction_strategy(config.suction_strategy_default)
        self.suction_strategy_by_class = {
            int(class_index): self._normalize_suction_strategy(strategy)
            for class_index, strategy in config.suction_strategy_by_class.items()
        }
        self.candidate_count = int(config.candidate_count)
        self.candidate_min_distance_px = float(config.candidate_min_distance_px)
        self.pca_offset_px = float(config.pca_offset_px)
        self.candidate_min_offset_px = float(config.candidate_min_offset_px)
        self.candidate_clearance_offset_ratio = float(config.candidate_clearance_offset_ratio)
        self.normal_surface_enabled = bool(config.normal_surface_enabled)
        self.surface_angle_threshold_deg = float(config.surface_angle_threshold_deg)
        self.surface_open_kernel_px = int(config.surface_open_kernel_px)
        self.surface_close_kernel_px = int(config.surface_close_kernel_px)
        self.surface_fill_holes_max_area_px = int(config.surface_fill_holes_max_area_px)
        self.surface_fill_holes_max_aspect_ratio = float(config.surface_fill_holes_max_aspect_ratio)
        self.surface_center_method = str(config.surface_center_method)
        self.surface_rect_max_area_ratio = float(config.surface_rect_max_area_ratio)
        self.min_surface_region_area_ratio = float(config.min_surface_region_area_ratio)
        self.min_surface_region_area_px = int(config.min_surface_region_area_px)
        self.normal_cluster_max_count = int(config.normal_cluster_max_count)
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
        self.min_suction_area_object_coverage = float(config.min_suction_area_object_coverage)
        self.min_suction_area_surface_coverage = float(config.min_suction_area_surface_coverage)
        self.axis_min_area_px = int(config.axis_min_area_px)
        self.axis_min_rect_ratio = float(config.axis_min_rect_ratio)
        self.axis_min_pca_ratio = float(config.axis_min_pca_ratio)
        self.axis_reference_step_px = float(config.axis_reference_step_px)
        self.single_cup_enabled = bool(config.single_cup_enabled)
        self.cup_count_default = int(config.cup_count_default)
        self.cup_count_by_class = dict(config.cup_count_by_class)
        self._single_cup_config = config

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
        for target_index, instance in enumerate(instances):
            mask = getattr(instance, "mask", None)
            if mask is None:
                setattr(instance, "suction_footprint", None)
                setattr(instance, "suction_candidates", [])
                setattr(instance, "suction_surface", None)
                suction_points.append([])
                continue

            others_mask = union_of_other_masks(instances, target_index, depth_image.shape)
            point, footprint, candidates, surface_debug = self._compute_one(
                instance,
                mask,
                depth_image,
                normal_image,
                intrinsic,
                extrinsic,
                others_mask,
            )
            setattr(instance, "suction_footprint", footprint.to_dict() if footprint is not None else None)
            setattr(instance, "suction_candidates", candidates)
            setattr(instance, "suction_surface", surface_debug)
            suction_points.append([point] if point is not None else [])
        return suction_points

    def _compute_one(
        self,
        instance: Any,
        mask: np.ndarray,
        depth_image: np.ndarray,
        normal_image: np.ndarray | None,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
        others_mask: np.ndarray | None = None,
    ) -> tuple[list[list[float]] | None, SuctionFootprint | None, list[dict[str, Any]], dict[str, Any] | None]:
        binary_mask = (mask > 0).astype(np.uint8)
        binary_mask = largest_component_mask(binary_mask)
        if not np.any(binary_mask):
            return None, None, [], None

        candidates = self._generate_suction_candidates(binary_mask)
        if not candidates:
            return None, None, [], None

        if self.single_cup_enabled and self._cup_count_for_instance(instance) == 1:
            _others = others_mask if others_mask is not None else np.zeros(depth_image.shape[:2], dtype=bool)
            selected, sc_debug = select_single_cup_suction(
                mask=binary_mask,
                depth_image=depth_image,
                normal_image=normal_image,
                intrinsic=intrinsic,
                others_mask=_others,
                candidates=candidates,
                params=self._single_cup_params(),
            )
            if selected is None:
                return None, None, candidates, sc_debug
            u, v, depth_mm, footprint, surface_debug = selected
            point_camera = pixel_to_camera(u, v, depth_mm, intrinsic)
            point_robot = transform_point(point_camera, extrinsic)
            normal_camera = self._selected_surface_normal(surface_debug, normal_image, u, v, binary_mask)
            normal_robot = orient_normal_z_up(transform_normal(normal_camera, extrinsic))
            reference_camera = self._selected_surface_reference_camera(
                surface_debug, binary_mask, u, v, depth_mm, intrinsic, point_camera, normal_camera,
            )
            reference_robot = project_to_tangent(transform_direction(reference_camera, extrinsic), normal_robot)
            quaternion = approach_and_reference_to_quaternion(normal_robot, reference_robot)
            return [
                [round(float(value), 3) for value in point_robot],
                [round(float(value), 6) for value in quaternion],
            ], footprint, candidates, surface_debug

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

    def _cup_count_for_instance(self, instance: Any) -> int:
        class_index = self._class_index(instance)
        if class_index is not None and class_index in self.cup_count_by_class:
            return int(self.cup_count_by_class[class_index])
        return int(self.cup_count_default)

    def _single_cup_params(self) -> SingleCupParams:
        cfg = self._single_cup_config
        return SingleCupParams(
            cup_diameter_mm=self.cup_diameter_mm,
            min_cup_inside_ratio=cfg.single_cup_min_cup_inside_ratio,
            depth_window=self.depth_window,
            flat_max_bump_mm=cfg.sc_flat_max_bump_mm,
            flat_max_dent_mm=cfg.sc_flat_max_dent_mm,
            flat_max_abs_p95_mm=cfg.sc_flat_max_abs_p95_mm,
            flat_bad_residual_mm=cfg.sc_flat_bad_residual_mm,
            flat_max_bad_ratio=cfg.sc_flat_max_bad_ratio,
            flat_min_valid_ratio=cfg.sc_flat_min_valid_ratio,
            seal_band_mm=cfg.sc_seal_band_mm,
            max_neighbor_ratio=cfg.sc_max_neighbor_ratio,
            cup_center_spacing_mm=self.cup_center_spacing_mm,
            protrusion_tol_mm=cfg.sc_protrusion_tol_mm,
            collision_min_valid_ratio=cfg.sc_collision_min_valid_ratio,
            num_rotations=cfg.sc_num_rotations,
        )

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
        estimate = estimate_class4_bottle_surface(mask, depth_image, normal_image, intrinsic)
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
        coverage = suction_area_coverage(
            mask,
            surface,
            footprint,
            self.min_suction_area_object_coverage,
            self.min_suction_area_surface_coverage,
            suction_area=suction_area,
        )
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
            "suction_area_check": coverage,
            "suction_area_pixels": int(coverage.get("suction_area_pixels", 0)),
            "suction_footprint_check_used": False,
            "class4_bottle": estimate_debug,
        }
        surface_debug.update(axis_debug)

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

        return self._select_clustered_normal_surface_suction(
            mask,
            depth_image,
            normal_image,
            intrinsic,
            extrinsic,
            instance,
        )

    def _select_clustered_normal_surface_suction(
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
        surfaces = clustered_normal_surface_candidates(
            normal_image,
            mask,
            angle_threshold_deg=self.surface_angle_threshold_deg,
            min_area_ratio=self.min_surface_region_area_ratio,
            min_area_px=self.min_surface_region_area_px,
            max_clusters=self.normal_cluster_max_count,
            open_kernel_px=self.surface_open_kernel_px,
            close_kernel_px=self.surface_close_kernel_px,
            fill_holes_max_area_px=self.surface_fill_holes_max_area_px,
            fill_holes_max_aspect_ratio=self.surface_fill_holes_max_aspect_ratio,
            center_method=self.surface_center_method,
            rect_max_area_ratio=self.surface_rect_max_area_ratio,
        )
        for index, (surface, surface_debug) in enumerate(surfaces):
            surface_debug["candidate_index"] = int(index)
            surface_debug["candidate_source"] = "normal_cluster"
            if surface is None or not surface_debug.get("passed"):
                surface_debug["suction_reject_reason"] = surface_debug.get("reason", "normal_cluster_rejected")
                attempts.append(surface_debug)
                continue

            surface, surface_debug = self._refine_class3_surface_by_depth(
                instance,
                depth_image,
                surface,
                mask,
                surface_debug,
            )
            if surface is None:
                attempts.append(surface_debug)
                continue

            u, v = surface_debug["surface_center_xy"]
            depth_mm = median_valid_depth_at_point(depth_image, u, v, mask, window=self.depth_window)
            if depth_mm is None:
                surface_debug["passed"] = False
                surface_debug["suction_reject_reason"] = "missing_depth_at_surface_center"
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
            coverage = suction_area_coverage(
                mask,
                surface,
                footprint,
                self.min_suction_area_object_coverage,
                self.min_suction_area_surface_coverage,
                suction_area=suction_area,
            )
            surface_debug["suction_area_check"] = coverage
            surface_debug["suction_area_pixels"] = int(coverage.get("suction_area_pixels", 0))
            surface_debug["suction_footprint_check_used"] = False
            surface_debug["passed"] = True
            surface_debug["reason"] = None
            surface_debug["normal_z_score"] = normal_z_score(surface_debug)
            surface_debug["robot_z_tilt_deg"] = self._robot_z_tilt_deg(surface_debug, extrinsic)
            attempts.append(surface_debug)
            passed_options.append((int(u), int(v), float(depth_mm), footprint, surface_debug))

        if passed_options:
            selected = max(
                passed_options,
                key=lambda item: (
                    -float(item[4].get("robot_z_tilt_deg", 180.0)),
                    int(item[4].get("surface_area", 0)),
                    -int(item[4].get("candidate_index", 0)),
                ),
            )
            selected_debug = dict(selected[4])
            selected_debug["selected"] = True
            selected_debug["selection_reason"] = "min_robot_z_tilt_among_clustered_surfaces"
            selected_debug["attempts"] = [
                {
                    **dict(attempt),
                    "selected": int(attempt.get("candidate_index", -1)) == int(selected_debug.get("candidate_index", -2)),
                }
                for attempt in attempts
            ]
            return (selected[0], selected[1], selected[2], selected[3], selected_debug), None

        return None, {
            "passed": False,
            "reason": "no_normal_cluster_surface_passed",
            "normal_surface_mode": "clustered",
            "attempts": attempts,
        }

    def _refine_class3_surface_by_depth(
        self,
        instance: Any | None,
        depth_image: np.ndarray,
        surface: np.ndarray,
        object_mask: np.ndarray,
        surface_debug: dict[str, Any],
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
        refined_debug = {**surface_debug, "class3_depth_split": split.debug}
        if split.surface is None:
            return surface, refined_debug

        refined_surface = split.surface
        surface_area = int(np.count_nonzero(refined_surface))
        object_area = max(int(np.count_nonzero(object_mask > 0)), 1)
        area_ratio = float(surface_area / object_area)
        center_xy = list(self._depth_refined_surface_center(refined_surface))
        refined_debug.update(
            {
                "surface_area_before_depth_split": int(surface_debug.get("surface_area", 0)),
                "surface_area_ratio_before_depth_split": surface_debug.get("surface_area_ratio"),
                "surface_area": surface_area,
                "surface_area_ratio": area_ratio,
                "surface_center_xy": center_xy,
                "component_seed_xy": center_xy,
                "seed_xy": center_xy,
            }
        )
        if area_ratio < self.min_surface_region_area_ratio or surface_area < self.min_surface_region_area_px:
            refined_debug["passed"] = False
            refined_debug["reason"] = "class3_depth_surface_too_small"
            refined_debug["suction_reject_reason"] = "class3_depth_surface_too_small"
            return None, refined_debug
        return refined_surface, refined_debug

    @staticmethod
    def _depth_refined_surface_center(surface: np.ndarray) -> tuple[int, int]:
        distance = cv2.distanceTransform((surface > 0).astype(np.uint8), cv2.DIST_L2, 5)
        if np.any(distance > 0):
            y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
            return int(x), int(y)
        center = mask_center_point(surface)
        return int(center[0]), int(center[1])

    @staticmethod
    def _robot_z_tilt_deg(surface_debug: dict[str, Any], extrinsic: np.ndarray) -> float:
        seed_normal = surface_debug.get("seed_normal")
        if not isinstance(seed_normal, list) or len(seed_normal) != 3:
            return 180.0
        normal_camera = np.asarray(seed_normal, dtype=np.float64).reshape(3)
        if np.linalg.norm(normal_camera) < 1e-9:
            return 180.0
        normal_robot = transform_normal(normal_camera, extrinsic)
        normal_robot = orient_normal_z_up(normal_robot)
        z = float(np.clip(normal_robot[2] / max(np.linalg.norm(normal_robot), 1e-9), -1.0, 1.0))
        return float(np.degrees(np.arccos(z)))

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
        max_count = max(1, self.candidate_count)
        min_dist_sq = max(0.0, self.candidate_min_distance_px) ** 2
        candidates: list[dict[str, Any]] = []

        def add_candidate(u: int, v: int, source: str) -> bool:
            if len(candidates) >= max_count:
                return False
            if not (0 <= v < mask.shape[0] and 0 <= u < mask.shape[1]):
                return False
            if not bool(mask[v, u]):
                return False
            for candidate in candidates:
                prev_u, prev_v = candidate["xy"]
                if (u - prev_u) ** 2 + (v - prev_v) ** 2 < min_dist_sq:
                    return False
            candidates.append({"xy": [int(u), int(v)], "source": source})
            return True

        distance = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
        self._add_best_distance_transform_candidate(distance, add_candidate)

        center_u, center_v = mask_center_point(mask)
        add_candidate(center_u, center_v, "mask_center")

        self._add_pca_offset_candidates(mask, distance, center_u, center_v, add_candidate)
        self._add_distance_transform_candidates(distance, add_candidate, lambda: len(candidates) < max_count)
        return candidates[:max_count]

    def _add_best_distance_transform_candidate(self, distance: np.ndarray, add_candidate: Any) -> None:
        if not np.any(distance > 0):
            return
        flat_index = int(np.argmax(distance))
        y, x = np.unravel_index(flat_index, distance.shape)
        if distance[y, x] > 0:
            add_candidate(int(x), int(y), "distance_transform")

    def _add_distance_transform_candidates(self, distance: np.ndarray, add_candidate: Any, has_capacity: Any) -> None:
        if not np.any(distance > 0):
            return
        for flat_index in np.argsort(distance.ravel())[::-1]:
            if not has_capacity():
                break
            y, x = np.unravel_index(int(flat_index), distance.shape)
            if distance[y, x] <= 0:
                break
            add_candidate(int(x), int(y), "distance_transform")

    def _add_pca_offset_candidates(
        self,
        mask: np.ndarray,
        distance: np.ndarray,
        center_u: int,
        center_v: int,
        add_candidate: Any,
    ) -> None:
        axis = principal_axis_2d(mask)
        orthogonal = np.array([-axis[1], axis[0]], dtype=np.float64)
        center_clearance = float(distance[center_v, center_u]) if distance.size else 0.0
        offset = self.pca_offset_px
        if center_clearance > 0.0:
            offset = min(offset, max(self.candidate_min_offset_px, center_clearance * self.candidate_clearance_offset_ratio))
        if offset <= 0.0:
            return

        for source, direction in (
            ("pca_long_axis", axis),
            ("pca_long_axis", -axis),
            ("pca_short_axis", orthogonal),
            ("pca_short_axis", -orthogonal),
        ):
            point = np.array([center_u, center_v], dtype=np.float64) + direction * offset
            u = int(round(float(point[0])))
            v = int(round(float(point[1])))
            if 0 <= v < mask.shape[0] and 0 <= u < mask.shape[1] and bool(mask[v, u]):
                add_candidate(u, v, source)
                continue
            nearest = self._nearest_mask_point(mask, point)
            if nearest is not None:
                add_candidate(nearest[0], nearest[1], source)

    @staticmethod
    def _nearest_mask_point(mask: np.ndarray, point_xy: np.ndarray) -> tuple[int, int] | None:
        coords_yx = np.column_stack(np.where(mask > 0))
        if coords_yx.shape[0] == 0:
            return None
        coords_xy = coords_yx[:, ::-1].astype(np.float64)
        distances = np.sum((coords_xy - point_xy.reshape(2)) ** 2, axis=1)
        nearest = coords_xy[int(np.argmin(distances))]
        return int(round(float(nearest[0]))), int(round(float(nearest[1])))

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
                reference = self._short_axis_reference_camera(axis, u, v, depth_mm, intrinsic, center_camera, normal_camera)
                if reference is not None:
                    return reference
        long_reference = self._principal_reference_camera(mask, u, v, depth_mm, intrinsic, center_camera)
        short_reference = self._short_reference_from_long(long_reference, normal_camera)
        return short_reference if short_reference is not None else long_reference

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
        return self._mean_valid_normal(normal_image, u, v, mask)

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

    def _mean_valid_normal(
        self,
        normal_image: np.ndarray | None,
        u: int,
        v: int,
        mask: np.ndarray,
    ) -> np.ndarray:
        if normal_image is None:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)

        half = self.normal_window // 2
        y1 = max(0, v - half)
        y2 = min(normal_image.shape[0], v + half + 1)
        x1 = max(0, u - half)
        x2 = min(normal_image.shape[1], u + half + 1)

        normal_patch = normal_image[y1:y2, x1:x2]
        mask_patch = mask[y1:y2, x1:x2] > 0
        normals = normal_patch[mask_patch]
        if normals.size == 0:
            normals = normal_image[mask > 0]
        if normals.size == 0:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)

        norms = np.linalg.norm(normals, axis=1)
        normals = normals[norms > 1e-6]
        if normals.size == 0:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)

        normal = normals.mean(axis=0).astype(np.float64)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return normal / norm
