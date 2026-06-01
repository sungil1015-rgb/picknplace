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
from src.utils.depth import median_valid_depth_at_point
from src.utils.mask import largest_component_mask, mask_center_point
from src.utils.normal import connected_normal_surface
from src.utils.suction_evaluation import (
    normal_z_score,
    suction_area_coverage,
)
from src.utils.suction_footprint import (
    DEFAULT_CUP_CENTER_SPACING_MM,
    DEFAULT_CUP_DIAMETER_MM,
    DEFAULT_MIN_CUP_INSIDE_RATIO,
    SuctionFootprint,
    compute_dual_cup_footprint,
    compute_projected_dual_cup_footprint,
    dual_cup_capsule_mask,
    principal_axis_2d,
)


class SuctionPipeline:
    def __init__(
        self,
        depth_window: int = 5,
        normal_window: int = 5,
        cup_diameter_mm: float = DEFAULT_CUP_DIAMETER_MM,
        cup_center_spacing_mm: float = DEFAULT_CUP_CENTER_SPACING_MM,
        min_cup_inside_ratio: float = DEFAULT_MIN_CUP_INSIDE_RATIO,
        candidate_count: int = 12,
        candidate_min_distance_px: float = 20.0,
        pca_offset_px: float = 25.0,
        normal_surface_enabled: bool = True,
        normal_seed_window: int = 11,
        surface_angle_threshold_deg: float = 20.0,
        min_surface_region_area_ratio: float = 0.15,
        min_surface_region_area_px: int = 0,
        min_suction_area_object_coverage: float = 0.90,
        min_suction_area_surface_coverage: float = 0.85,
    ) -> None:
        self.depth_window = depth_window
        self.normal_window = normal_window
        self.cup_diameter_mm = float(cup_diameter_mm)
        self.cup_center_spacing_mm = float(cup_center_spacing_mm)
        self.min_cup_inside_ratio = float(min_cup_inside_ratio)
        self.candidate_count = int(candidate_count)
        self.candidate_min_distance_px = float(candidate_min_distance_px)
        self.pca_offset_px = float(pca_offset_px)
        self.normal_surface_enabled = bool(normal_surface_enabled)
        self.normal_seed_window = int(normal_seed_window)
        self.surface_angle_threshold_deg = float(surface_angle_threshold_deg)
        self.min_surface_region_area_ratio = float(min_surface_region_area_ratio)
        self.min_surface_region_area_px = int(min_surface_region_area_px)
        self.min_suction_area_object_coverage = float(min_suction_area_object_coverage)
        self.min_suction_area_surface_coverage = float(min_suction_area_surface_coverage)

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

            point, footprint, candidates, surface_debug = self._compute_one(mask, depth_image, normal_image, intrinsic, extrinsic)
            setattr(instance, "suction_footprint", footprint.to_dict() if footprint is not None else None)
            setattr(instance, "suction_candidates", candidates)
            setattr(instance, "suction_surface", surface_debug)
            suction_points.append([point] if point is not None else [])
        return suction_points

    def _compute_one(
        self,
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

        if self.normal_surface_enabled and normal_image is not None:
            selected, failure_debug = self._select_normal_surface_suction(binary_mask, depth_image, normal_image, intrinsic, candidates)
            if selected is not None:
                u, v, depth_mm, footprint, surface_debug = selected
                candidates.insert(1, {"xy": [int(u), int(v)], "source": "normal_surface_center"})
            else:
                return None, None, candidates, failure_debug
        else:
            u, v = candidates[0]["xy"]
            surface_debug = self._move_to_center_normal_surface(binary_mask, normal_image, u, v)
            if surface_debug is not None and surface_debug.get("passed"):
                u, v = surface_debug["surface_center_xy"]
                candidates.insert(1, {"xy": [int(u), int(v)], "source": "normal_surface_center"})

            depth_mm = median_valid_depth_at_point(depth_image, u, v, binary_mask, window=self.depth_window)
            if depth_mm is None:
                return None, None, candidates, surface_debug

            footprint = compute_dual_cup_footprint(
                binary_mask,
                (u, v),
                depth_mm,
                intrinsic,
                cup_diameter_mm=self.cup_diameter_mm,
                cup_center_spacing_mm=self.cup_center_spacing_mm,
                min_cup_inside_ratio=self.min_cup_inside_ratio,
            )
            if footprint is not None:
                suction_area = dual_cup_capsule_mask(binary_mask.shape, footprint) & (binary_mask > 0)
                if surface_debug is None:
                    surface_debug = {}
                surface_debug["suction_area_pixels"] = int(np.count_nonzero(suction_area))

        point_camera = pixel_to_camera(u, v, depth_mm, intrinsic)
        point_robot = transform_point(point_camera, extrinsic)

        normal_camera = self._mean_valid_normal(normal_image, u, v, binary_mask)
        normal_robot = transform_normal(normal_camera, extrinsic)
        normal_robot = orient_normal_z_up(normal_robot)

        reference_camera = self._principal_reference_camera(binary_mask, u, v, depth_mm, intrinsic, point_camera)
        reference_robot = transform_direction(reference_camera, extrinsic)
        reference_robot = project_to_tangent(reference_robot, normal_robot)
        quaternion = approach_and_reference_to_quaternion(normal_robot, reference_robot)

        return [
            [round(float(value), 3) for value in point_robot],
            [round(float(value), 6) for value in quaternion],
        ], footprint, candidates, surface_debug

    def _move_to_center_normal_surface(
        self,
        mask: np.ndarray,
        normal_image: np.ndarray | None,
        u: int,
        v: int,
    ) -> dict[str, Any] | None:
        if not self.normal_surface_enabled or normal_image is None:
            return None
        try:
            _, debug = connected_normal_surface(
                normal_image,
                mask,
                (u, v),
                seed_window=self.normal_seed_window,
                angle_threshold_deg=self.surface_angle_threshold_deg,
                min_area_ratio=self.min_surface_region_area_ratio,
                min_area_px=self.min_surface_region_area_px,
            )
            return debug
        except Exception as exc:
            return {"passed": False, "reason": f"normal_surface_error: {exc}"}

    def _select_normal_surface_suction(
        self,
        mask: np.ndarray,
        depth_image: np.ndarray,
        normal_image: np.ndarray | None,
        intrinsic: np.ndarray,
        candidates: list[dict[str, Any]],
    ) -> tuple[tuple[int, int, float, SuctionFootprint, dict[str, Any]] | None, dict[str, Any] | None]:
        if not self.normal_surface_enabled or normal_image is None:
            return None, None

        attempts: list[dict[str, Any]] = []
        passed_options: list[tuple[int, int, float, SuctionFootprint, dict[str, Any]]] = []
        for index, candidate in enumerate(candidates):
            seed_u, seed_v = candidate["xy"]
            surface, surface_debug = self._normal_surface_from_seed(mask, normal_image, seed_u, seed_v)
            surface_debug["candidate_index"] = int(index)
            surface_debug["candidate_source"] = str(candidate.get("source", "unknown"))

            if surface is None or not surface_debug.get("passed"):
                surface_debug["suction_reject_reason"] = surface_debug.get("reason", "normal_surface_rejected")
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

            if footprint is None:
                surface_debug["passed"] = False
                surface_debug["suction_reject_reason"] = "invalid_footprint"
                attempts.append(surface_debug)
                continue
            if not footprint.feasible:
                surface_debug["passed"] = False
                surface_debug["suction_reject_reason"] = footprint.reason or "footprint_rejected"
                attempts.append(surface_debug)
                continue
            if not coverage.get("passed", False):
                surface_debug["passed"] = False
                surface_debug["suction_reject_reason"] = coverage.get("reason", "suction_area_coverage_rejected")
                attempts.append(surface_debug)
                continue

            surface_debug["passed"] = True
            surface_debug["reason"] = None
            surface_debug["normal_z_score"] = normal_z_score(surface_debug)
            attempts.append(surface_debug)
            passed_options.append((int(u), int(v), float(depth_mm), footprint, surface_debug))

        if passed_options:
            selected = max(
                passed_options,
                key=lambda item: (
                    normal_z_score(item[4]),
                    -int(item[4].get("candidate_index", 0)),
                ),
            )
            selected_debug = dict(selected[4])
            selected_debug["selected"] = True
            selected_debug["selection_reason"] = "max_normal_z_among_passed_candidates"
            selected_debug["attempts"] = [
                {
                    **dict(attempt),
                    "selected": int(attempt.get("candidate_index", -1)) == int(selected_debug.get("candidate_index", -2)),
                }
                for attempt in attempts
            ]
            return (selected[0], selected[1], selected[2], selected[3], selected_debug), None

        return None, {"passed": False, "reason": "no_suction_surface_passed", "attempts": attempts}

    def _normal_surface_from_seed(
        self,
        mask: np.ndarray,
        normal_image: np.ndarray,
        u: int,
        v: int,
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        try:
            return connected_normal_surface(
                normal_image,
                mask,
                (u, v),
                seed_window=self.normal_seed_window,
                angle_threshold_deg=self.surface_angle_threshold_deg,
                min_area_ratio=self.min_surface_region_area_ratio,
                min_area_px=self.min_surface_region_area_px,
            )
        except Exception as exc:
            return None, {"passed": False, "reason": f"normal_surface_error: {exc}"}

    def _compute_surface_footprint(
        self,
        mask: np.ndarray,
        center_xy: tuple[int, int],
        depth_mm: float,
        intrinsic: np.ndarray,
        surface_debug: dict[str, Any],
    ) -> tuple[SuctionFootprint | None, np.ndarray | None]:
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
        )
        surface_debug["footprint_projection"] = "fronto_parallel_fallback"
        return footprint, None

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

        center_u, center_v = mask_center_point(mask)
        add_candidate(center_u, center_v, "mask_center")

        distance = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
        self._add_pca_offset_candidates(mask, distance, center_u, center_v, add_candidate)
        self._add_distance_transform_candidates(distance, add_candidate, lambda: len(candidates) < max_count)
        return candidates[:max_count]

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
            offset = min(offset, max(5.0, center_clearance * 0.8))
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
        step = max(5.0, float(self.normal_window))
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
