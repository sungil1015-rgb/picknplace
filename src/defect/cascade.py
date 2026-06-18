"""Cascade detector — 4 신호 (PatchCore L2 + color outlier + normal Laplacian + mask topology) AND gate voting.

deformable 객체 (하리보/젤리) 대상. 정상 구겨짐은 PatchCore만 1표 → 통과.
찢어짐은 색/normal/topology 동시 high → 2~3표 → defect.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.defect.base import DefectDetector, DefectResult
from src.defect.color_prior import ColorPrior
from src.defect.patchcore_lite import PatchCoreLite


class CascadeDetector(DefectDetector):
    def __init__(
        self,
        memory_bank: str,
        color_prior: str,
        signals: dict,
        voting_min: int = 2,
        action: str = "reject",
        enabled: bool = True,
        dinov2_layer: int = 8,
        device: Optional[str] = None,
        model_name: str = "facebook/dinov2-large",
    ):
        self.enabled = bool(enabled)
        self.action = str(action)
        self.voting_min = int(voting_min)
        self.signals_cfg = signals

        # signal 1: PatchCore
        self.patchcore = PatchCoreLite(
            memory_bank=memory_bank,
            threshold=float(signals["patchcore"]["threshold"]),
            dinov2_layer=dinov2_layer,
            patch_aggregate=signals["patchcore"].get("aggregate", "max"),
            top_p=float(signals["patchcore"].get("top_p", 0.01)),
            action="reject",
            device=device,
            model_name=model_name,
        )

        # signal 2: color prior
        cp_path = Path(color_prior)
        if not cp_path.is_absolute():
            cp_path = Path.cwd() / cp_path
        self.color_prior = ColorPrior.load(cp_path)
        self.color_z = float(signals["color"]["z_threshold"])
        self.color_ratio_t = float(signals["color"]["outlier_ratio_threshold"])

        # signal 3: normal laplacian
        self.nlap_peak_t = float(signals["normal_laplacian"]["peak_threshold"])
        self.nlap_ratio_t = float(signals["normal_laplacian"]["peak_ratio_threshold"])

        # signal 4: topology
        self.topology_t = float(signals["topology"]["raggedness_threshold"])

        # threshold (DefectDetector ABC 요구) — cascade는 voting이라 unused, 0으로 둠
        self.threshold = 0.0

    def score(
        self,
        rgb_bgr: np.ndarray,
        mask: np.ndarray,
        normal_map: Optional[np.ndarray] = None,
        dinov2_features: Optional[dict] = None,
    ) -> DefectResult:
        m = (mask > 0) if mask.dtype != bool else mask
        components = {}

        # signal 1: PatchCore L2
        pc_result = self.patchcore.score(rgb_bgr, m, normal_map, dinov2_features)
        s1 = pc_result.score
        components["patchcore"] = s1

        # signal 2: color outlier ratio
        lab = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2LAB)
        pixels = lab[m]
        s2 = self.color_prior.outlier_ratio(pixels, z_threshold=self.color_z) if len(pixels) > 0 else 0.0
        components["color"] = s2

        # signal 3: Normal Laplacian peak ratio
        if normal_map is not None and normal_map.ndim == 3 and normal_map.shape[2] >= 3:
            nz = normal_map[..., 2].astype(np.float32)
            nlap = cv2.Laplacian(nz, cv2.CV_32F, ksize=3)
            in_mask = np.abs(nlap)[m]
            s3 = float((in_mask > self.nlap_peak_t).mean()) if len(in_mask) > 0 else 0.0
        else:
            s3 = 0.0
        components["normal_laplacian"] = s3

        # signal 4: Mask topology raggedness
        mu8 = m.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mu8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            perim = cv2.arcLength(cnt, True)
            area = max(cv2.contourArea(cnt), 1.0)
            s4 = float(perim ** 2 / (4.0 * np.pi * area))
        else:
            s4 = 1.0
        components["topology"] = s4

        # voting AND gate
        votes = int(
            (s1 > float(self.signals_cfg["patchcore"]["threshold"]))
            + (s2 > self.color_ratio_t)
            + (s3 > self.nlap_ratio_t)
            + (s4 > self.topology_t)
        )
        is_defect = votes >= self.voting_min

        return DefectResult(
            score=float(max(s1, s2, s3, s4)),
            is_defect=is_defect,
            score_map=pc_result.score_map,
            votes=votes,
            components=components,
        )
