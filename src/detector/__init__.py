from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class InstanceData:
    """detector 가 반환할 단일 객체 결과 컨테이너."""
    bbox: Optional[np.ndarray] = None     # shape (5,) — [x1, y1, x2, y2, score]
    label: Optional[int] = None
    score: Optional[float] = None
    mask: Optional[np.ndarray] = None     # shape (H, W) bool


from src.detector.mask2former_detector import Mask2FormerDetector  # noqa: E402

__all__ = ["InstanceData", "Mask2FormerDetector"]
