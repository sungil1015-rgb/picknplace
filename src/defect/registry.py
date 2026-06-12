"""DefectRegistry — yaml에 등록된 클래스별 detector lookup."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from src.defect.base import DefectDetector
from src.defect.cascade import CascadeDetector
from src.defect.fusion import WeightedFusionDetector
from src.defect.patchcore_lite import PatchCoreLite


def _build_detector(cfg: dict) -> DefectDetector:
    t = cfg.pop("type")
    if t == "patchcore":
        return PatchCoreLite(**cfg)
    if t == "cascade":
        return CascadeDetector(**cfg)
    if t == "fusion":
        return WeightedFusionDetector(**cfg)
    raise ValueError(f"unknown detector type: {t}")


class DefectRegistry:
    def __init__(self, config_path: str | Path):
        config_path = Path(config_path)
        if not config_path.is_file():
            raise FileNotFoundError(f"defect config not found: {config_path}")
        config = yaml.safe_load(config_path.read_text()) or {}
        self.enabled = bool(config.get("enabled", True))
        self.detectors: dict[int, DefectDetector] = {}
        if not self.enabled:
            return
        for cid_str, det_cfg in (config.get("detectors") or {}).items():
            det_cfg = dict(det_cfg)
            if not det_cfg.get("enabled", True):
                continue
            self.detectors[int(cid_str)] = _build_detector(det_cfg)

    def get(self, class_index: int) -> Optional[DefectDetector]:
        if not self.enabled:
            return None
        return self.detectors.get(int(class_index))

    def __repr__(self):
        return f"DefectRegistry(enabled={self.enabled}, classes={sorted(self.detectors.keys())})"
