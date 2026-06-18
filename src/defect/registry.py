"""DefectRegistry — yaml에 등록된 클래스별 detector lookup.

운영 흐름(서버 작업 최소화): 임계값을 yaml에 수동 기입하지 않아도, threshold 튜닝
스크립트(synth_roc_tune_loo.py)가 산출한 `weights/defect_thresholds.json`이 있으면
detector 빌드 시 자동으로 threshold를 덮어쓴다. 즉 [뱅크 .npz + thresholds.json]만
weights/에 떨궈두면 작동. json 키는 memory_bank 파일명(basename) — 양측이 같은 .npz를
가리키므로 모호성 없음. cascade는 signals.patchcore.threshold에 적용.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import yaml

from src.defect.base import DefectDetector
from src.defect.cascade import CascadeDetector
from src.defect.fusion import WeightedFusionDetector
from src.defect.patchcore_lite import PatchCoreLite

# yaml과 같은 디렉토리 기준(없으면 cwd/weights)으로 찾는 임계 오버라이드 파일명
THRESHOLDS_FILENAME = "defect_thresholds.json"


def _build_detector(cfg: dict) -> DefectDetector:
    t = cfg.pop("type")
    if t == "patchcore":
        return PatchCoreLite(**cfg)
    if t == "cascade":
        return CascadeDetector(**cfg)
    if t == "fusion":
        return WeightedFusionDetector(**cfg)
    raise ValueError(f"unknown detector type: {t}")


def _find_thresholds_file(config_path: Path) -> Optional[Path]:
    """thresholds.json 위치 탐색: config 기준 ../weights, 그다음 cwd/weights."""
    candidates = [
        config_path.resolve().parent.parent / "weights" / THRESHOLDS_FILENAME,
        Path.cwd() / "weights" / THRESHOLDS_FILENAME,
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def _load_threshold_overrides(path: Path) -> dict:
    """thresholds.json 로드. 키=memory_bank basename, 값=τ. 파싱 실패는 시끄럽게 에러."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): float(v) for k, v in data.items()}


def _apply_threshold_override(det_cfg: dict, overrides: dict) -> None:
    """memory_bank basename으로 매칭되는 τ가 있으면 detector cfg에 in-place 적용."""
    mb = det_cfg.get("memory_bank")
    if not mb or not overrides:
        return
    key = Path(mb).name
    if key not in overrides:
        return
    tau = overrides[key]
    if det_cfg.get("type") == "cascade":
        det_cfg.setdefault("signals", {}).setdefault("patchcore", {})["threshold"] = tau
    else:
        det_cfg["threshold"] = tau


class DefectRegistry:
    def __init__(self, config_path: str | Path, strict_thresholds: bool = True):
        """strict_thresholds=True(기본·운영): 활성 클래스의 튜닝 τ가 없으면 ValueError로 차단.
        config의 placeholder τ(base 기준)로 조용히 오판하는 사고를 막는다.
        검증/개발용으로 파일 없이 로드하려면 strict_thresholds=False."""
        config_path = Path(config_path)
        if not config_path.is_file():
            raise FileNotFoundError(f"defect config not found: {config_path}")
        config = yaml.safe_load(config_path.read_text()) or {}
        self.enabled = bool(config.get("enabled", True))
        self.detectors: dict[int, DefectDetector] = {}
        if not self.enabled:
            return

        th_file = _find_thresholds_file(config_path)
        overrides = _load_threshold_overrides(th_file) if th_file is not None else {}
        self.threshold_overrides = overrides
        self.thresholds_path = th_file

        # 활성 detector cfg 수집
        enabled_items: list[tuple[str, dict]] = []
        for cid_str, det_cfg in (config.get("detectors") or {}).items():
            det_cfg = dict(det_cfg)
            if not det_cfg.get("enabled", True):
                continue
            enabled_items.append((cid_str, det_cfg))

        # 튜닝 τ 누락 검증 (detector 빌드 전에 시끄럽게 차단)
        missing = [
            Path(cfg["memory_bank"]).name
            for _, cfg in enabled_items
            if cfg.get("memory_bank") and Path(cfg["memory_bank"]).name not in overrides
        ]
        if missing:
            msg = (
                f"튜닝된 임계(threshold)가 없는 클래스: {missing}. "
                f"config의 placeholder τ(base 기준)로 판정하면 전수 오판 위험. "
            )
            if th_file is None:
                msg += f"weights/{THRESHOLDS_FILENAME}이 없습니다 — synth_roc_tune_loo.py로 생성하세요."
            else:
                msg += f"{th_file}에 해당 키가 없습니다 — 해당 클래스 재튜닝 필요."
            if strict_thresholds:
                raise ValueError(msg)
            print(f"[DefectRegistry] WARNING (strict_thresholds=False): {msg}")

        for cid_str, det_cfg in enabled_items:
            _apply_threshold_override(det_cfg, overrides)
            self.detectors[int(cid_str)] = _build_detector(det_cfg)

    def get(self, class_index: int) -> Optional[DefectDetector]:
        if not self.enabled:
            return None
        return self.detectors.get(int(class_index))

    def __repr__(self):
        return f"DefectRegistry(enabled={self.enabled}, classes={sorted(self.detectors.keys())})"
