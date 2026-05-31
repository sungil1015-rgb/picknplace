from __future__ import annotations

from typing import Any, Sequence

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
from src.utils.suction_footprint import (
    DEFAULT_CUP_CENTER_SPACING_MM,
    DEFAULT_CUP_DIAMETER_MM,
    DEFAULT_MIN_CUP_INSIDE_RATIO,
    SuctionFootprint,
    compute_dual_cup_footprint,
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
    ) -> None:
        self.depth_window = depth_window
        self.normal_window = normal_window
        self.cup_diameter_mm = float(cup_diameter_mm)
        self.cup_center_spacing_mm = float(cup_center_spacing_mm)
        self.min_cup_inside_ratio = float(min_cup_inside_ratio)

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
                suction_points.append([])
                continue

            point, footprint = self._compute_one(mask, depth_image, normal_image, intrinsic, extrinsic)
            setattr(instance, "suction_footprint", footprint.to_dict() if footprint is not None else None)
            suction_points.append([point] if point is not None else [])
        return suction_points

    def _compute_one(
        self,
        mask: np.ndarray,
        depth_image: np.ndarray,
        normal_image: np.ndarray | None,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
    ) -> tuple[list[list[float]] | None, SuctionFootprint | None]:
        binary_mask = (mask > 0).astype(np.uint8)
        binary_mask = largest_component_mask(binary_mask)
        if not np.any(binary_mask):
            return None, None

        u, v = mask_center_point(binary_mask)

        depth_mm = median_valid_depth_at_point(depth_image, u, v, binary_mask, window=self.depth_window)
        if depth_mm is None:
            return None, None

        footprint = compute_dual_cup_footprint(
            binary_mask,
            (u, v),
            depth_mm,
            intrinsic,
            cup_diameter_mm=self.cup_diameter_mm,
            cup_center_spacing_mm=self.cup_center_spacing_mm,
            min_cup_inside_ratio=self.min_cup_inside_ratio,
        )

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
        ], footprint

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
