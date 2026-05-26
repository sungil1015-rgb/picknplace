from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class GraspPriorityScore:
    total: float
    depth_score: float
    center_score: float
    isolation_score: float
    area_score: float
    neighbor_ratio: Optional[float]
    mask_area: Optional[int]
    object_depth: Optional[float]
    center_distance: Optional[float]
    valid_depth: bool


class GraspPriorityScorer:
    def __init__(
        self,
        depth_weight: float = 0.55,
        center_weight: float = 0.20,
        isolation_weight: float = 0.25,
        area_weight: float = 0.0,
        depth_percentile: float = 20.0,
        erosion_kernel: int = 5,
        isolation_dilation_kernel: int = 31,
    ) -> None:
        self.depth_weight = float(depth_weight)
        self.center_weight = float(center_weight)
        self.isolation_weight = float(isolation_weight)
        self.area_weight = float(area_weight)
        self.depth_percentile = float(depth_percentile)
        self.erosion_kernel = int(erosion_kernel)
        self.isolation_dilation_kernel = int(isolation_dilation_kernel)

    def score_instances(
        self,
        instances: Sequence[Any],
        depth_image: np.ndarray | None,
        roi_2d: Optional[Sequence[float]] = None,
    ) -> list[GraspPriorityScore]:
        masks = [self._instance_mask(instance) for instance in instances]
        raw_scores = [
            self._raw_score(instance, mask, masks, depth_image, roi_2d)
            for instance, mask in zip(instances, masks)
        ]
        valid_depths = [
            score.object_depth
            for score in raw_scores
            if score.valid_depth and score.object_depth is not None
        ]
        area_scores = self._area_scores(raw_scores)
        if not valid_depths:
            scores: list[GraspPriorityScore] = []
            for score, area_score in zip(raw_scores, area_scores):
                scores.append(
                    GraspPriorityScore(
                        total=float(
                            (self.center_weight * score.center_score)
                            + (self.isolation_weight * score.isolation_score)
                            + (self.area_weight * area_score)
                        ),
                    depth_score=0.0,
                    center_score=score.center_score,
                    isolation_score=score.isolation_score,
                    area_score=area_score,
                    neighbor_ratio=score.neighbor_ratio,
                    mask_area=score.mask_area,
                    object_depth=score.object_depth,
                    center_distance=score.center_distance,
                    valid_depth=False,
                    )
                )
            return scores

        min_depth = min(valid_depths)
        max_depth = max(valid_depths)
        depth_range = max(max_depth - min_depth, 1e-6)

        scores: list[GraspPriorityScore] = []
        for score, area_score in zip(raw_scores, area_scores):
            if score.object_depth is None or not score.valid_depth:
                depth_score = 0.0
                total = (
                    (self.center_weight * score.center_score)
                    + (self.isolation_weight * score.isolation_score)
                    + (self.area_weight * area_score)
                )
            else:
                depth_score = 1.0 - float(np.clip((score.object_depth - min_depth) / depth_range, 0.0, 1.0))
                total = (
                    (self.depth_weight * depth_score)
                    + (self.center_weight * score.center_score)
                    + (self.isolation_weight * score.isolation_score)
                    + (self.area_weight * area_score)
                )
            scores.append(
                GraspPriorityScore(
                    total=float(total),
                    depth_score=float(depth_score),
                    center_score=score.center_score,
                    isolation_score=score.isolation_score,
                    area_score=area_score,
                    neighbor_ratio=score.neighbor_ratio,
                    mask_area=score.mask_area,
                    object_depth=score.object_depth,
                    center_distance=score.center_distance,
                    valid_depth=score.valid_depth,
                )
            )
        return scores

    def _raw_score(
        self,
        instance: Any,
        mask: Optional[np.ndarray],
        masks: Sequence[Optional[np.ndarray]],
        depth_image: np.ndarray | None,
        roi_2d: Optional[Sequence[float]],
    ) -> GraspPriorityScore:
        if mask is None or not np.any(mask):
            return GraspPriorityScore(0.0, 0.0, 0.0, 0.0, 0.0, None, None, None, None, False)

        center_score, center_distance = self._center_score(mask, roi_2d)
        isolation_score, neighbor_ratio = self._isolation_score(mask, masks)
        object_depth = self._object_depth(mask, depth_image)
        mask_area = int(mask.sum())
        return GraspPriorityScore(
            total=0.0,
            depth_score=0.0,
            center_score=center_score,
            isolation_score=isolation_score,
            area_score=0.0,
            neighbor_ratio=neighbor_ratio,
            mask_area=mask_area,
            object_depth=object_depth,
            center_distance=center_distance,
            valid_depth=object_depth is not None,
        )

    @staticmethod
    def _instance_mask(instance: Any) -> Optional[np.ndarray]:
        mask = getattr(instance, "mask", None)
        if mask is None or not np.any(mask):
            return None
        return mask > 0

    def _object_depth(self, mask: np.ndarray, depth_image: np.ndarray | None) -> Optional[float]:
        if depth_image is None or depth_image.ndim < 2:
            return None

        image_height, image_width = depth_image.shape[:2]
        mask = mask[:image_height, :image_width]
        if not np.any(mask):
            return None

        inner_mask = self._erode_mask(mask)
        values = self._valid_depth_values(depth_image, inner_mask)
        if values.size == 0:
            values = self._valid_depth_values(depth_image, mask)
        if values.size == 0:
            return None
        return float(np.percentile(values.astype(np.float64), self.depth_percentile))

    def _erode_mask(self, mask: np.ndarray) -> np.ndarray:
        if self.erosion_kernel <= 1:
            return mask
        kernel = np.ones((max(1, self.erosion_kernel), max(1, self.erosion_kernel)), dtype=np.uint8)
        eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        return eroded if np.any(eroded) else mask

    def _isolation_score(
        self,
        target_mask: np.ndarray,
        masks: Sequence[Optional[np.ndarray]],
    ) -> tuple[float, Optional[float]]:
        if self.isolation_dilation_kernel <= 1:
            return 1.0, 0.0

        kernel_size = max(1, self.isolation_dilation_kernel)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        dilated = cv2.dilate(target_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        ring = dilated & ~target_mask
        ring_area = int(ring.sum())
        if ring_area == 0:
            return 1.0, 0.0

        other_mask = np.zeros_like(target_mask, dtype=bool)
        for mask in masks:
            if mask is None or mask is target_mask:
                continue
            height = min(other_mask.shape[0], mask.shape[0])
            width = min(other_mask.shape[1], mask.shape[1])
            other_mask[:height, :width] |= mask[:height, :width]

        neighbor_pixels = int((ring & other_mask).sum())
        neighbor_ratio = float(np.clip(neighbor_pixels / max(float(ring_area), 1.0), 0.0, 1.0))
        return 1.0 - neighbor_ratio, neighbor_ratio

    @staticmethod
    def _area_scores(scores: Sequence[GraspPriorityScore]) -> list[float]:
        indexed_areas = [
            (index, score.mask_area)
            for index, score in enumerate(scores)
            if score.mask_area is not None
        ]
        if not indexed_areas:
            return [0.0 for _ in scores]
        if len(indexed_areas) == 1:
            return [1.0 if score.mask_area is not None else 0.0 for score in scores]

        ranked = sorted(indexed_areas, key=lambda item: item[1])
        area_scores = [0.0 for _ in scores]
        denominator = float(len(ranked) - 1)
        start = 0
        while start < len(ranked):
            end = start + 1
            while end < len(ranked) and ranked[end][1] == ranked[start][1]:
                end += 1
            average_rank = (start + end - 1) * 0.5
            score = float(average_rank / denominator)
            for index, _ in ranked[start:end]:
                area_scores[index] = score
            start = end
        return area_scores

    @staticmethod
    def _valid_depth_values(depth_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        values = depth_image[: mask.shape[0], : mask.shape[1]][mask]
        return values[np.isfinite(values) & (values > 0)]

    @staticmethod
    def _center_score(mask: np.ndarray, roi_2d: Optional[Sequence[float]]) -> tuple[float, Optional[float]]:
        ys, xs = np.where(mask)
        if xs.size == 0 or ys.size == 0:
            return 0.0, None

        cx = float(xs.mean())
        cy = float(ys.mean())
        height, width = mask.shape[:2]

        if roi_2d and len(roi_2d) == 4:
            left, right, top, bottom = [float(value) for value in roi_2d]
            roi_width = max(right - left, 1.0)
            roi_height = max(bottom - top, 1.0)
            roi_cx = (left + right) * 0.5
            roi_cy = (top + bottom) * 0.5
        else:
            roi_width = float(width)
            roi_height = float(height)
            roi_cx = width * 0.5
            roi_cy = height * 0.5

        distance = float(np.hypot(cx - roi_cx, cy - roi_cy))
        max_distance = max(float(np.hypot(roi_width * 0.5, roi_height * 0.5)), 1e-6)
        score = 1.0 - float(np.clip(distance / max_distance, 0.0, 1.0))
        return score, distance
