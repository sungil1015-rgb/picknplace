from __future__ import annotations

from typing import Any, Sequence

import cv2
import numpy as np

from src.utils.geometry import (
    approach_and_reference_to_quaternion,
    pixel_to_camera,
    transform_direction,
    transform_normal,
    transform_point,
)


class SuctionPipeline:
    def __init__(self, depth_window: int = 5, normal_window: int = 5) -> None:
        self.depth_window = depth_window
        self.normal_window = normal_window

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
                suction_points.append([])
                continue

            point = self._compute_one(mask, depth_image, normal_image, intrinsic, extrinsic)
            suction_points.append([point] if point is not None else [])
        return suction_points

    def _compute_one(
        self,
        mask: np.ndarray,
        depth_image: np.ndarray,
        normal_image: np.ndarray | None,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
    ) -> list[list[float]] | None:
        binary_mask = (mask > 0).astype(np.uint8)
        binary_mask = self._largest_component_mask(binary_mask)
        if not np.any(binary_mask):
            return None

        u, v = self._mask_center_point(binary_mask)

        depth_mm = self._median_valid_depth(depth_image, u, v, binary_mask)
        if depth_mm is None:
            return None

        point_camera = pixel_to_camera(u, v, depth_mm, intrinsic)
        point_robot = transform_point(point_camera, extrinsic)

        normal_camera = self._mean_valid_normal(normal_image, u, v, binary_mask)
        normal_robot = transform_normal(normal_camera, extrinsic)
        normal_robot = self._orient_normal_z_up(normal_robot)

        reference_camera = self._principal_reference_camera(binary_mask, u, v, depth_mm, intrinsic, point_camera)
        reference_robot = transform_direction(reference_camera, extrinsic)
        reference_robot = self._project_reference_to_tangent(reference_robot, normal_robot)
        quaternion = approach_and_reference_to_quaternion(normal_robot, reference_robot)

        return [
            [round(float(value), 3) for value in point_robot],
            [round(float(value), 6) for value in quaternion],
        ]

    @staticmethod
    def _orient_normal_z_up(normal: np.ndarray) -> np.ndarray:
        normal = np.asarray(normal, dtype=np.float64)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        normal = normal / norm
        if normal[2] < 0.0:
            normal = -normal
        return normal

    @staticmethod
    def _largest_component_mask(mask: np.ndarray) -> np.ndarray:
        binary_mask = (mask > 0).astype(np.uint8)
        if not np.any(binary_mask):
            return binary_mask
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        if component_count <= 2:
            return binary_mask
        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return (labels == largest_label).astype(np.uint8)

    @staticmethod
    def _mask_center_point(mask: np.ndarray) -> tuple[int, int]:
        coords_yx = np.column_stack(np.where(mask > 0))
        if coords_yx.shape[0] == 0:
            return 0, 0

        center_xy = coords_yx[:, ::-1].astype(np.float64).mean(axis=0)
        center_u = int(round(center_xy[0]))
        center_v = int(round(center_xy[1]))
        if (
            0 <= center_v < mask.shape[0]
            and 0 <= center_u < mask.shape[1]
            and mask[center_v, center_u] > 0
        ):
            return center_u, center_v

        coords_xy = coords_yx[:, ::-1].astype(np.float64)
        distances = np.sum((coords_xy - center_xy) ** 2, axis=1)
        nearest_index = int(np.argmin(distances))
        nearest = coords_xy[nearest_index]
        return int(round(nearest[0])), int(round(nearest[1]))

    @staticmethod
    def _project_reference_to_tangent(reference: np.ndarray, normal: np.ndarray) -> np.ndarray:
        normal = np.asarray(normal, dtype=np.float64)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-9:
            normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            normal = normal / normal_norm

        reference = np.asarray(reference, dtype=np.float64)
        tangent = reference - np.dot(reference, normal) * normal
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm >= 1e-9:
            return tangent / tangent_norm

        fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(fallback, normal))) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        tangent = fallback - np.dot(fallback, normal) * normal
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-9:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return tangent / tangent_norm

    def _principal_reference_camera(
        self,
        mask: np.ndarray,
        u: int,
        v: int,
        depth_mm: float,
        intrinsic: np.ndarray,
        center_camera: np.ndarray,
    ) -> np.ndarray:
        coords = np.column_stack(np.where(mask > 0))
        if coords.shape[0] < 2:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)

        coords_xy = coords[:, ::-1].astype(np.float64)
        center_xy = coords_xy.mean(axis=0)
        centered = coords_xy - center_xy
        covariance = np.cov(centered, rowvar=False)
        if covariance.shape != (2, 2):
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)

        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        principal = eigenvectors[:, int(np.argmax(eigenvalues))]
        if principal[0] < 0.0 or (abs(principal[0]) < 1e-9 and principal[1] < 0.0):
            principal = -principal

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

    def _median_valid_depth(self, depth_image: np.ndarray, u: int, v: int, mask: np.ndarray) -> float | None:
        half = self.depth_window // 2
        y1 = max(0, v - half)
        y2 = min(depth_image.shape[0], v + half + 1)
        x1 = max(0, u - half)
        x2 = min(depth_image.shape[1], u + half + 1)

        depth_patch = depth_image[y1:y2, x1:x2]
        mask_patch = mask[y1:y2, x1:x2] > 0
        values = depth_patch[(depth_patch > 0) & mask_patch]
        if values.size == 0:
            values = depth_image[mask > 0]
            values = values[values > 0]
        if values.size == 0:
            return None
        return float(np.median(values))

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
