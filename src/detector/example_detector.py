"""
2D instance segmentation detector 예시 스켈레톤.

이 폴더는 RGB 이미지에서 객체를 검출 / 분할하는 모델 wrapper 를 둔다.
실제 구현은 학생이 본인이 선택한 detection framework (e.g. mmdet, detectron2,
ultralytics 등) 에 맞춰 채워 넣으면 된다.

PickNPlace.run() 내부에서 이 클래스를 호출해 InstanceData 리스트를 받는다.
"""

from typing import List, Union
from pathlib import Path

import numpy as np

from src.detector import InstanceData


class ExampleDetector:
    def __init__(self, config_file: str, checkpoint_file: str, device: str = "cuda:0") -> None:
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.device = device
        # TODO: 학생 구현 — 모델 로드

    def inference(self, image: Union[str, Path, np.ndarray]) -> List[InstanceData]:
        """RGB 이미지(또는 경로) 입력 → 객체별 InstanceData 리스트 반환."""
        raise NotImplementedError
