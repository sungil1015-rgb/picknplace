"""DefectLogger — detector 호출마다 jsonl 누적 + is_defect=True 시 score_map heatmap PNG 저장.

금요일 false rate 분석용 누적 로그.

설계 근거:
- jsonl 한 줄 = 한 객체 호출. timestamp, source_id, class index/name, score, is_defect, votes, components.
  → false reject/accept를 source_id로 역추적 가능.
- heatmap PNG는 is_defect=True 일 때만 저장 (디스크 절약, 운영 시 결함 객체만 사후 검토 대상).
- matplotlib backend는 모듈 import 시 Agg 강제 — 헤드리스 서버에서 GUI 의존성 차단.
- score_map None인 detector (cascade 일부)는 PNG 스킵, jsonl에는 항상 기록.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.defect.base import DefectResult


class DefectLogger:
    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.heatmap_dir = self.log_dir / "heatmaps"
        self.heatmap_dir.mkdir(parents=True, exist_ok=True)
        # 단일 jsonl 파일 (append-only). 날짜별 회전은 운영 단계에서 cron으로.
        self.jsonl_path = self.log_dir / "defect_log.jsonl"
        self._counter = 0  # 동일 timestamp 내 충돌 방지용 시퀀스

    def log(
        self,
        source_id: str,
        class_index: int,
        class_name: str,
        defect_result: DefectResult,
    ) -> Optional[Path]:
        """한 객체에 대한 detector 결과를 누적. is_defect=True면 heatmap PNG도 저장.

        Returns:
            저장된 heatmap PNG 경로 (저장 안 했으면 None). 테스트에서 검증용.
        """
        self._counter += 1
        ts_iso = datetime.now().isoformat(timespec="milliseconds")
        ts_compact = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

        png_path: Optional[Path] = None
        if defect_result.is_defect and defect_result.score_map is not None:
            png_name = f"{ts_compact}_{self._counter:04d}_cls{class_index}_{class_name}.png"
            png_path = self.heatmap_dir / png_name
            self._save_heatmap(defect_result.score_map, png_path)

        record: dict[str, Any] = {
            "timestamp": ts_iso,
            "source_id": str(source_id),
            "class_index": int(class_index),
            "class_name": str(class_name),
            "score": float(defect_result.score),
            "is_defect": bool(defect_result.is_defect),
            "votes": int(defect_result.votes),
            "components": {k: float(v) for k, v in (defect_result.components or {}).items()},
            "heatmap_path": str(png_path.relative_to(self.log_dir)) if png_path is not None else None,
        }
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return png_path

    @staticmethod
    def _save_heatmap(score_map: np.ndarray, out_path: Path) -> None:
        arr = np.asarray(score_map, dtype=np.float32)
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        im = ax.imshow(arr, cmap="viridis")
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout(pad=0.2)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
