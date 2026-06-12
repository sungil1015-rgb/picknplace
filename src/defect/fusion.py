"""WeightedFusionDetector — PatchCore + normal Laplacian 가중 융합 (mango용).

설계 근거 (2026-06-12 mango 근본 원인 분석):
  - mango bulk 정상 CV 0.126 (tail 제거 후) — rigid급으로 타이트해서 PatchCore 주 신호로 충분.
  - 단 결함 모드가 찢김(tear)이라 기하 신호 (normal 불연속)를 보조로 가중 융합.
  - vote가 아닌 이유: cascade는 threshold 4개 튜닝 필요 + mango는 다색 인쇄라 color 신호 기여 낮음.

수식:
  fused = (1 - w) * z_pc + w * z_lap
  z_pc  = (pc_score - pc_mu) / pc_sigma          (LOO 정상 분포 통계, 오프라인 캘리브)
  z_lap = (lap_ratio - lap_mu) / lap_sigma       (정상 normal map 통계 — 실측 전 기본값)
  is_defect = fused > threshold                  (threshold는 정상 fused p99 + margin 권장)

중요 제약 (정직): 합성 결함은 RGB만 바꾸므로 Laplacian 채널의 TPR 기여는
오프라인 검증 불가. w > 0은 "실제 찢김은 기하 신호를 남긴다"는 사전 지식이며
실물 찢김 테스트로만 검증 가능. FPR 쪽은 정상 데이터로 완전 캘리브 가능.

lap_ratio 정의는 cascade signal 3과 동일 (mask 내 |Laplacian(nz)| > peak_t 비율).
단 fusion에서는 mask 경계 erosion 추가 — 객체 윤곽의 normal 불연속 (정상) 배제.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from src.defect.base import DefectDetector, DefectResult
from src.defect.patchcore_lite import PatchCoreLite


class WeightedFusionDetector(DefectDetector):
    def __init__(
        self,
        memory_bank: str,
        threshold: float,                  # fused z-score threshold (정상 fused p99 + margin)
        pc_mu: float,                      # PatchCore 정상 LOO score 평균 (오프라인 캘리브)
        pc_sigma: float,                   # PatchCore 정상 LOO score 표준편차
        lap_weight: float = 0.25,          # w — Laplacian 가중 (0이면 PatchCore 단독과 동일)
        lap_peak_threshold: float = 0.15,  # |Laplacian(nz)| peak 판정 (cascade와 동일 컨벤션)
        lap_mu: float = 0.0,               # lap_ratio 정상 평균 (실측 전 기본값 — 실물 캘리브로 갱신)
        lap_sigma: float = 1.0,            # lap_ratio 정상 표준편차 (기본값 1 → z_lap = raw ratio)
        lap_erosion_px: int = 7,           # mask 경계 erosion (윤곽 normal 불연속 배제)
        dinov2_layer: int = 8,
        patch_aggregate: str = "top_k_mean",
        top_k: int = 4,
        action: str = "reject",
        enabled: bool = True,
        device: Optional[str] = None,
        model_name: str = "facebook/dinov2-base",
    ):
        self.enabled = bool(enabled)
        self.action = str(action)
        self.threshold = float(threshold)
        self.pc_mu = float(pc_mu)
        self.pc_sigma = max(float(pc_sigma), 1e-6)
        self.lap_weight = float(lap_weight)
        self.lap_peak_threshold = float(lap_peak_threshold)
        self.lap_mu = float(lap_mu)
        self.lap_sigma = max(float(lap_sigma), 1e-6)
        self.lap_erosion_px = int(lap_erosion_px)

        self.patchcore = PatchCoreLite(
            memory_bank=memory_bank,
            threshold=0.0,                 # fusion이 최종 판정 — 내부 threshold 미사용
            dinov2_layer=dinov2_layer,
            patch_aggregate=patch_aggregate,
            top_k=top_k,
            action="reject",
            device=device,
            model_name=model_name,
        )

    def _lap_ratio(self, normal_map: Optional[np.ndarray], mask: np.ndarray) -> Optional[float]:
        """mask 내부 (경계 erosion 후) normal z-채널 Laplacian peak 비율. normal 없으면 None."""
        if normal_map is None or normal_map.ndim != 3 or normal_map.shape[2] < 3:
            return None
        m = (mask > 0) if mask.dtype != bool else mask
        if self.lap_erosion_px > 0:
            k = self.lap_erosion_px
            m = cv2.erode(m.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
        if m.sum() < 50:
            return None
        nz = normal_map[..., 2].astype(np.float32)
        nlap = np.abs(cv2.Laplacian(nz, cv2.CV_32F, ksize=3))
        return float((nlap[m] > self.lap_peak_threshold).mean())

    def score(
        self,
        rgb_bgr: np.ndarray,
        mask: np.ndarray,
        normal_map: Optional[np.ndarray] = None,
        dinov2_features: Optional[dict] = None,
    ) -> DefectResult:
        pc_result = self.patchcore.score(rgb_bgr, mask, normal_map, dinov2_features)
        z_pc = (pc_result.score - self.pc_mu) / self.pc_sigma

        lap_ratio = self._lap_ratio(normal_map, mask)
        if lap_ratio is None:
            # normal 없으면 PatchCore 단독으로 동작 (가중치 재정규화 X — 보수적으로 z_pc만)
            fused = z_pc
            z_lap = None
        else:
            z_lap = (lap_ratio - self.lap_mu) / self.lap_sigma
            fused = (1.0 - self.lap_weight) * z_pc + self.lap_weight * z_lap

        return DefectResult(
            score=float(fused),
            is_defect=fused > self.threshold,
            score_map=pc_result.score_map,
            components={
                "pc_score": pc_result.score,
                "z_pc": float(z_pc),
                "lap_ratio": lap_ratio if lap_ratio is not None else -1.0,
                "z_lap": float(z_lap) if z_lap is not None else -999.0,
            },
        )
