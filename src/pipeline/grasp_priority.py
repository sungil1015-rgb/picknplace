from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from src.utils.depth import median_valid_depth_at_point
from src.utils.mask import mask_center_point


@dataclass(frozen=True)
class GraspPriorityScore:
    total: float
    depth_score: float
    grasp_depth: Optional[float]
    grasp_xy: Optional[tuple[int, int]]
    valid_depth: bool
    class_similarity: Optional[float]
    class_reject_reason: Optional[str]
    mask_area: Optional[int] = None


class GraspPriorityScorer:
    """Minimal priority scorer.

    Priority currently uses only:
    - local depth around the intended grasp point
    - classification similarity, stored for later policy decisions
    - classification reject reason, stored for later policy decisions
    """

    def __init__(self, local_depth_window: int = 11) -> None:
        self.local_depth_window = int(local_depth_window)

    def score_instances(
        self,
        instances: Sequence[Any],
        depth_image: np.ndarray | None,
        roi_2d: Optional[Sequence[float]] = None,
    ) -> list[GraspPriorityScore]:
        del roi_2d

        raw_scores = [self._raw_score(instance, depth_image) for instance in instances]
        valid_depths = [
            score.grasp_depth
            for score in raw_scores
            if score.valid_depth and score.grasp_depth is not None
        ]
        if not valid_depths:
            return raw_scores

        min_depth = min(valid_depths)
        max_depth = max(valid_depths)
        depth_range = max(max_depth - min_depth, 1e-6)

        scores: list[GraspPriorityScore] = []
        for score in raw_scores:
            if score.grasp_depth is None or not score.valid_depth:
                scores.append(score)
                continue
            depth_score = 1.0 - float(np.clip((score.grasp_depth - min_depth) / depth_range, 0.0, 1.0))
            scores.append(
                GraspPriorityScore(
                    total=depth_score,
                    depth_score=depth_score,
                    grasp_depth=score.grasp_depth,
                    grasp_xy=score.grasp_xy,
                    valid_depth=True,
                    class_similarity=score.class_similarity,
                    class_reject_reason=score.class_reject_reason,
                    mask_area=score.mask_area,
                )
            )
        return scores

    def _raw_score(self, instance: Any, depth_image: np.ndarray | None) -> GraspPriorityScore:
        mask = self._instance_mask(instance)
        grasp_xy = self._grasp_point_xy(instance, mask)
        grasp_depth = self._grasp_depth(mask, depth_image, grasp_xy)
        valid_depth = grasp_depth is not None
        return GraspPriorityScore(
            total=0.0,
            depth_score=0.0,
            grasp_depth=grasp_depth,
            grasp_xy=grasp_xy,
            valid_depth=valid_depth,
            class_similarity=self._class_similarity(instance),
            class_reject_reason=self._class_reject_reason(instance),
            mask_area=int(mask.sum()) if mask is not None else None,
        )

    @staticmethod
    def _instance_mask(instance: Any) -> Optional[np.ndarray]:
        mask = getattr(instance, "mask", None)
        if mask is None or not np.any(mask):
            return None
        return mask > 0

    def _grasp_depth(
        self,
        mask: np.ndarray | None,
        depth_image: np.ndarray | None,
        grasp_xy: tuple[int, int] | None,
    ) -> Optional[float]:
        if mask is None or grasp_xy is None:
            return None
        return median_valid_depth_at_point(
            depth_image,
            int(grasp_xy[0]),
            int(grasp_xy[1]),
            mask,
            window=self.local_depth_window,
        )

    @staticmethod
    def _grasp_point_xy(instance: Any, mask: np.ndarray | None) -> tuple[int, int] | None:
        surface_debug = getattr(instance, "suction_surface", None)
        if isinstance(surface_debug, dict):
            center_xy = surface_debug.get("surface_center_xy")
            if isinstance(center_xy, (list, tuple)) and len(center_xy) == 2:
                try:
                    return int(center_xy[0]), int(center_xy[1])
                except (TypeError, ValueError):
                    pass
        if mask is None or not np.any(mask):
            return None
        return mask_center_point(mask)

    @staticmethod
    def _class_similarity(instance: Any) -> Optional[float]:
        value = getattr(instance, "class_similarity", None)
        try:
            if value is not None and np.isfinite(float(value)):
                return float(value)
        except (TypeError, ValueError):
            return None
        return None

    @staticmethod
    def _class_reject_reason(instance: Any) -> Optional[str]:
        value = getattr(instance, "class_reject_reason", None)
        if value is None:
            return None
        text = str(value)
        return text if text else None
