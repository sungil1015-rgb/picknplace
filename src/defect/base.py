"""DefectDetector ABC + 결과 dataclass.

모든 클래스별 결함 detector는 이 인터페이스 구현. registry가 yaml에 등록된 detector를 lookup.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class DefectResult:
    score: float                                    # image-level scalar (결함 점수)
    is_defect: bool = False                         # threshold 통과 여부
    score_map: Optional[np.ndarray] = None          # (H, W) heatmap (시각화용)
    votes: int = 0                                  # cascade detector 용 (0~4)
    components: dict = field(default_factory=dict)  # debug용 raw signal 값들


class DefectDetector(ABC):
    threshold: float = 0.0
    action: str = "reject"   # "reject" | "priority_down" | "report_only"
    enabled: bool = True

    @abstractmethod
    def score(
        self,
        rgb_bgr: np.ndarray,                            # (H, W, 3) uint8 BGR
        mask: np.ndarray,                               # (H, W) bool 또는 uint8
        normal_map: Optional[np.ndarray] = None,        # (H, W, 3) float32
        dinov2_features: Optional[dict] = None,         # {"patch_layer_8": (n_tokens, dim)}
    ) -> DefectResult:
        """객체 1개에 대한 결함 점수. 모든 detector 통일 인터페이스."""
        raise NotImplementedError
