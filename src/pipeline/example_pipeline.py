"""
3D 처리 / grasp point 계산 파이프라인 예시 스켈레톤.

이 폴더는 detector 결과(2D mask)와 depth 를 받아
실제로 로봇이 집을 수 있는 위치(suction / grasp point)를 계산하는
도메인 로직을 둔다.

PickNPlace.run() 안에서 ExampleDetector 결과를 받아 이 파이프라인을 호출하는
구조를 학생이 직접 설계한다.
"""

from typing import List, Tuple

import numpy as np


class ExamplePipeline:
    def __init__(self, c_matrix: np.ndarray) -> None:
        # 카메라 intrinsic 3x3 행렬
        self.c_matrix = c_matrix
        # TODO: 학생 구현 — 표면 추정 / 정렬 / 기타 도메인 모듈 초기화

    def setup_bucket_from_depth(
            self,
            rgb_image: np.ndarray,
            depth_image: np.ndarray,
            roi_2d: List[int],
    ) -> None:
        """depth + 2D ROI 로 bucket 상단 평면의 normal / OBB 셋업."""
        raise NotImplementedError

    def compute_suction_points(
            self,
            masks: List[np.ndarray],
            depth_image: np.ndarray,
    ) -> Tuple[List[List[Tuple[int, int]]], List[int]]:
        """
        객체 mask 별로 suction 가능한 (u, v) 좌표 리스트 반환.

        Returns
        -------
        (suction_points, surface_id_list)
            suction_points  : [N_obj][K_i][2(u,v)]
            surface_id_list : 객체별 surface 타입 식별자
        """
        raise NotImplementedError
