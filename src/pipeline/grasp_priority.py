from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from src.utils.depth import valid_depth_values
from src.utils.mask import mask_center_point


@dataclass(frozen=True)
class GraspPriorityScore:
    total: float
    depth_score: float
    grasp_depth: Optional[float]
    grasp_xy: Optional[tuple[int, int]]
    valid_depth: bool
    class_similarity: Optional[float]
    class_vote_ratio: Optional[float]
    class_reject_reason: Optional[str]
    depth_source: Optional[str] = None
    mask_area: Optional[int] = None
    priority_stage: Optional[dict[str, Any]] = None


class GraspPriorityScorer:
    """Minimal priority scorer.

    Priority currently uses only:
    - median depth of the selected priority surface mask
    - classification similarity, stored for later policy decisions
    - classification reject reason, stored for later policy decisions
    """

    def __init__(
        self,
        known_top_count: int = 4,
        unknown_top_count: int = 2,
        unknown_low_similarity_threshold: float = 0.78,
        unknown_low_vote_threshold: float = 1.0,
    ) -> None:
        self.known_top_count = int(known_top_count)
        self.unknown_top_count = int(unknown_top_count)
        self.unknown_low_similarity_threshold = float(unknown_low_similarity_threshold)
        self.unknown_low_vote_threshold = float(unknown_low_vote_threshold)

    def score_instances(
        self,
        instances: Sequence[Any],
        depth_image: np.ndarray | None,
        roi_2d: Optional[Sequence[float]] = None,
    ) -> list[GraspPriorityScore]:
        del roi_2d

        raw_scores = [self._raw_score(instance, depth_image) for instance in instances]
        return self._stage_scores(raw_scores)

    def _stage_scores(self, raw_scores: list[GraspPriorityScore]) -> list[GraspPriorityScore]:
        valid_indices = [
            index
            for index, score in enumerate(raw_scores)
            if score.valid_depth and score.grasp_depth is not None
        ]
        if not valid_indices:
            return [
                self._replace_score(score, priority_stage=self._stage_debug(index, score, valid_depth_count=0))
                for index, score in enumerate(raw_scores)
            ]

        valid_indices.sort(key=lambda index: float(raw_scores[index].grasp_depth))
        valid_depths = [float(raw_scores[index].grasp_depth) for index in valid_indices if raw_scores[index].grasp_depth is not None]
        min_depth = min(valid_depths)
        max_depth = max(valid_depths)
        depth_range = max(max_depth - min_depth, 1e-6)

        unknown_indices = [
            index
            for index in valid_indices
            if self._is_unknown_estimated(raw_scores[index])
        ]
        known_indices = [
            index
            for index in valid_indices
            if self._is_known_priority_candidate(raw_scores[index])
        ]

        known_pool = sorted(
            known_indices,
            key=lambda index: self._similarity_or_negative(raw_scores[index]),
            reverse=True,
        )[: max(0, self.known_top_count)]
        unknown_pool = sorted(
            unknown_indices,
            key=lambda index: float(raw_scores[index].grasp_depth) if raw_scores[index].grasp_depth is not None else float("inf"),
        )[: max(0, self.unknown_top_count)]

        final_pool = list(dict.fromkeys(known_pool + unknown_pool))
        selected_index = None
        if final_pool:
            selected_index = min(
                final_pool,
                key=lambda index: float(raw_scores[index].grasp_depth) if raw_scores[index].grasp_depth is not None else float("inf"),
            )

        depth_rank_by_index = {index: rank + 1 for rank, index in enumerate(valid_indices)}
        known_set = set(known_pool)
        unknown_set = set(unknown_pool)
        final_set = set(final_pool)

        scores: list[GraspPriorityScore] = []
        for index, score in enumerate(raw_scores):
            depth_score = 0.0
            if score.grasp_depth is not None and score.valid_depth:
                depth_score = 1.0 - float(np.clip((score.grasp_depth - min_depth) / depth_range, 0.0, 1.0))

            total = 0.0
            if index == selected_index:
                total = 1.0
            elif index in final_set:
                total = 0.8

            stage = self._stage_debug(
                index,
                score,
                valid_depth_count=len(valid_indices),
                depth_rank=depth_rank_by_index.get(index),
                known_top_count=self.known_top_count,
                unknown_top_count=self.unknown_top_count,
                in_known_pool=index in known_set,
                in_unknown_estimated_pool=index in unknown_set,
                in_final_pool=index in final_set,
                final_selected=index == selected_index,
            )
            scores.append(self._replace_score(score, total=total, depth_score=depth_score, priority_stage=stage))
        return scores

    def _raw_score(self, instance: Any, depth_image: np.ndarray | None) -> GraspPriorityScore:
        mask = self._instance_mask(instance)
        grasp_xy = self._grasp_point_xy(instance, mask)
        grasp_depth = self._grasp_depth(instance, mask, depth_image, grasp_xy)
        valid_depth = grasp_depth is not None
        return GraspPriorityScore(
            total=0.0,
            depth_score=0.0,
            grasp_depth=grasp_depth,
            grasp_xy=grasp_xy,
            valid_depth=valid_depth,
            class_similarity=self._class_similarity(instance),
            class_vote_ratio=self._class_vote_ratio(instance),
            class_reject_reason=self._class_reject_reason(instance),
            depth_source=self._depth_source(instance),
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
        instance: Any,
        mask: np.ndarray | None,
        depth_image: np.ndarray | None,
        grasp_xy: tuple[int, int] | None,
    ) -> Optional[float]:
        if mask is None or depth_image is None or grasp_xy is None:
            return None
        override_depth = getattr(instance, "priority_depth_value", None)
        try:
            if override_depth is not None and np.isfinite(float(override_depth)) and float(override_depth) > 0:
                return float(override_depth)
        except (TypeError, ValueError):
            pass
        depth_mask = self._priority_depth_mask(instance, mask)
        if depth_mask is None:
            return None
        values = valid_depth_values(depth_image, depth_mask)
        if values.size == 0:
            return None
        return float(np.median(values.astype(np.float64)))

    @staticmethod
    def _priority_depth_mask(
        instance: Any,
        mask: np.ndarray,
    ) -> np.ndarray | None:
        surface_mask = getattr(instance, "priority_depth_mask", None)
        if isinstance(surface_mask, np.ndarray) and np.any(surface_mask):
            return surface_mask > 0
        return mask > 0

    @staticmethod
    def _grasp_point_xy(instance: Any, mask: np.ndarray | None) -> tuple[int, int] | None:
        priority_center = getattr(instance, "priority_depth_center_xy", None)
        if isinstance(priority_center, (list, tuple)) and len(priority_center) == 2:
            try:
                return int(priority_center[0]), int(priority_center[1])
            except (TypeError, ValueError):
                pass
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
    def _class_vote_ratio(instance: Any) -> Optional[float]:
        value = getattr(instance, "class_vote_ratio", None)
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

    def _is_unknown_estimated(self, score: GraspPriorityScore) -> bool:
        similarity = score.class_similarity
        vote_ratio = score.class_vote_ratio
        low_similarity = similarity is not None and similarity < self.unknown_low_similarity_threshold
        low_vote = vote_ratio is not None and vote_ratio < self.unknown_low_vote_threshold
        return bool(low_similarity and low_vote)

    def _is_low_similarity_only(self, score: GraspPriorityScore) -> bool:
        similarity = score.class_similarity
        vote_ratio = score.class_vote_ratio
        low_similarity = similarity is not None and similarity < self.unknown_low_similarity_threshold
        low_vote = vote_ratio is not None and vote_ratio < self.unknown_low_vote_threshold
        return bool(low_similarity and not low_vote)

    def _is_known_priority_candidate(self, score: GraspPriorityScore) -> bool:
        similarity = score.class_similarity
        vote_ratio = score.class_vote_ratio
        low_similarity = similarity is not None and similarity < self.unknown_low_similarity_threshold
        low_vote = vote_ratio is not None and vote_ratio < self.unknown_low_vote_threshold
        return bool(not low_similarity and not low_vote and score.class_reject_reason is None)

    def _stage_debug(
        self,
        index: int,
        score: GraspPriorityScore,
        *,
        valid_depth_count: int,
        depth_rank: int | None = None,
        known_top_count: int = 0,
        unknown_top_count: int = 0,
        in_known_pool: bool = False,
        in_unknown_estimated_pool: bool = False,
        in_final_pool: bool = False,
        final_selected: bool = False,
    ) -> dict[str, Any]:
        low_similarity = score.class_similarity is not None and score.class_similarity < self.unknown_low_similarity_threshold
        low_vote = score.class_vote_ratio is not None and score.class_vote_ratio < self.unknown_low_vote_threshold
        known_priority_candidate = self._is_known_priority_candidate(score)
        unknown_estimated = bool(low_similarity and low_vote)
        excluded_reason = None
        if not known_priority_candidate and not unknown_estimated:
            if low_similarity and not low_vote:
                excluded_reason = "low_similarity_only"
            elif low_vote and not low_similarity:
                excluded_reason = "low_vote_only"
            elif score.class_reject_reason is not None:
                excluded_reason = "class_rejected"
            else:
                excluded_reason = "not_priority_candidate"
        return {
            "index": int(index),
            "valid_depth_count": int(valid_depth_count),
            "depth_rank": None if depth_rank is None else int(depth_rank),
            "known_top_count": int(known_top_count),
            "unknown_top_count": int(unknown_top_count),
            "low_similarity": bool(low_similarity),
            "low_vote": bool(low_vote),
            "known_priority_candidate": bool(known_priority_candidate),
            "unknown_estimated": bool(unknown_estimated),
            "low_similarity_only_excluded": bool(low_similarity and not low_vote),
            "priority_excluded_reason": excluded_reason,
            "unknown_low_similarity_threshold": float(self.unknown_low_similarity_threshold),
            "unknown_low_vote_threshold": float(self.unknown_low_vote_threshold),
            "in_known_pool": bool(in_known_pool),
            "in_unknown_estimated_pool": bool(in_unknown_estimated_pool),
            "in_final_pool": bool(in_final_pool),
            "final_selected": bool(final_selected),
        }

    @staticmethod
    def _similarity_or_negative(score: GraspPriorityScore) -> float:
        return float(score.class_similarity) if score.class_similarity is not None else -1.0

    @staticmethod
    def _replace_score(
        score: GraspPriorityScore,
        *,
        total: float | None = None,
        depth_score: float | None = None,
        priority_stage: dict[str, Any] | None = None,
    ) -> GraspPriorityScore:
        return GraspPriorityScore(
            total=score.total if total is None else float(total),
            depth_score=score.depth_score if depth_score is None else float(depth_score),
            grasp_depth=score.grasp_depth,
            grasp_xy=score.grasp_xy,
            valid_depth=score.valid_depth,
            class_similarity=score.class_similarity,
            class_vote_ratio=score.class_vote_ratio,
            class_reject_reason=score.class_reject_reason,
            depth_source=score.depth_source,
            mask_area=score.mask_area,
            priority_stage=priority_stage if priority_stage is not None else score.priority_stage,
        )

    @staticmethod
    def _depth_source(instance: Any) -> Optional[str]:
        value = getattr(instance, "priority_depth_source", None)
        if value is None:
            return None
        text = str(value)
        return text if text else None
