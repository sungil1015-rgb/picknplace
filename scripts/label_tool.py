"""Bin Picking 통합 툴 — 라벨링 + YOLO 추론 시각화.

탭 1: 라벨링 (기존 기능 — 폴리곤 그리기/수정/삭제, 클래스 부여)
탭 2: YOLO 추론 결과 보기 (학습된 모델의 마스크 시각화 + GT 비교)

출력 (탭 1): 20260424_labeled/{images, labels, data_json, dataset.yaml}
"""

import json
import os
import shutil
import sys
import threading
from collections import Counter
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
from PIL import Image, ImageTk

# Modern UI library — industrial dark theme (VS Code / Blender feel)
try:
    import customtkinter as ctk
    ctk.set_appearance_mode("dark")        # always dark — industrial tool feel
    ctk.set_default_color_theme("dark-blue")
    _HAS_CTK = True
except ImportError:
    _HAS_CTK = False


# ──────────────────────────────────────────────
# Theme definitions — selectable at runtime via THEME dropdown
# ──────────────────────────────────────────────
THEMES = {
    "vscode_dark": {
        "name": "VS Code Dark (default)",
        "bg_main": "#1E1E1E", "bg_panel": "#2D2D30", "bg_input": "#3C3C3C",
        "bg_hover": "#3A3D3E", "text": "#D4D4D4", "text_muted": "#9D9D9D",
        "text_header": "#FFFFFF", "border": "#3F3F46",
        "accent": "#F39C12", "accent_hover": "#FFAA1D",
        "info": "#4FC1FF", "ok": "#4EC9B0", "warn": "#DCDCAA", "error": "#F48771",
        "unlabeled": "#5A5A5A",   # darker gray — barely visible on canvas
        "ctk_mode": "dark",
    },
    "github_dim": {
        "name": "GitHub Dim (soft dark)",
        "bg_main": "#22272E", "bg_panel": "#2D333B", "bg_input": "#373E47",
        "bg_hover": "#3D444D", "text": "#ADBAC7", "text_muted": "#768390",
        "text_header": "#CDD9E5", "border": "#444C56",
        "accent": "#F69D50", "accent_hover": "#FFAA60",
        "info": "#6CB6FF", "ok": "#8DDB8C", "warn": "#F69D50", "error": "#FF938A",
        "unlabeled": "#3D444D",   # blends with panel — minimal interference
        "ctk_mode": "dark",
    },
    "gruvbox_dark": {
        "name": "Gruvbox Dark (warm brown)",
        "bg_main": "#282828", "bg_panel": "#3C3836", "bg_input": "#504945",
        "bg_hover": "#665C54", "text": "#EBDBB2", "text_muted": "#A89984",
        "text_header": "#FBF1C7", "border": "#665C54",
        "accent": "#FE8019", "accent_hover": "#FFA040",
        "info": "#83A598", "ok": "#B8BB26", "warn": "#FABD2F", "error": "#FB4934",
        "unlabeled": "#504945",
        "ctk_mode": "dark",
    },
    "solarized_dark": {
        "name": "Solarized Dark (cyan)",
        "bg_main": "#002B36", "bg_panel": "#073642", "bg_input": "#2A4951",
        "bg_hover": "#3A5A62", "text": "#93A1A1", "text_muted": "#657B83",
        "text_header": "#EEE8D5", "border": "#586E75",
        "accent": "#B58900", "accent_hover": "#D4A20E",
        "info": "#268BD2", "ok": "#859900", "warn": "#CB4B16", "error": "#DC322F",
        "unlabeled": "#2A4951",
        "ctk_mode": "dark",
    },
    "nord": {
        "name": "Nord (cool slate)",
        "bg_main": "#2E3440", "bg_panel": "#3B4252", "bg_input": "#434C5E",
        "bg_hover": "#4C566A", "text": "#D8DEE9", "text_muted": "#7B88A1",
        "text_header": "#ECEFF4", "border": "#4C566A",
        "accent": "#D08770", "accent_hover": "#E59E84",
        "info": "#5E81AC", "ok": "#A3BE8C", "warn": "#EBCB8B", "error": "#BF616A",
        "unlabeled": "#434C5E",
        "ctk_mode": "dark",
    },
    "solarized_light": {
        "name": "Solarized Light (beige) ⭐ 눈 피로 적음",
        "bg_main": "#FDF6E3", "bg_panel": "#EEE8D5", "bg_input": "#FFFCEB",
        "bg_hover": "#E5DDC0", "text": "#586E75", "text_muted": "#93A1A1",
        "text_header": "#073642", "border": "#D8C9A6",
        "accent": "#B58900", "accent_hover": "#CB9C0E",
        "info": "#268BD2", "ok": "#859900", "warn": "#CB4B16", "error": "#DC322F",
        "unlabeled": "#A89984",
        "ctk_mode": "light",
    },
}

# Settings path defined first (before _load_theme_name uses it)
SETTINGS_PATH = Path.home() / ".label_tool_state.json"


# ─── Active theme (loaded from settings; default vscode_dark) ───
_DEFAULT_THEME = "github_dim"   # soft dark (user preference)


def _load_theme_name():
    try:
        if SETTINGS_PATH.exists():
            d = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            t = d.get("theme", _DEFAULT_THEME)
            if t in THEMES:
                return t
    except Exception:
        pass
    return _DEFAULT_THEME


ACTIVE_THEME_NAME = _load_theme_name()
T = THEMES[ACTIVE_THEME_NAME]

COLOR_BG_MAIN = T["bg_main"]
COLOR_BG_PANEL = T["bg_panel"]
COLOR_BG_INPUT = T["bg_input"]
COLOR_BG_HOVER = T["bg_hover"]
COLOR_TEXT = T["text"]
COLOR_TEXT_MUTED = T["text_muted"]
COLOR_TEXT_HEADER = T["text_header"]
COLOR_BORDER = T["border"]
COLOR_ACCENT = T["accent"]
COLOR_ACCENT_HOVER = T["accent_hover"]
COLOR_INFO = T["info"]
COLOR_OK = T["ok"]
COLOR_WARN = T["warn"]
COLOR_ERROR = T["error"]
UNLABELED_COLOR = T.get("unlabeled", "#7F8C8D")

# Override ctk mode based on theme
if _HAS_CTK:
    ctk.set_appearance_mode(T["ctk_mode"])

# Side panel layout
SIDE_WIDTH = 380                    # px, right-side panel (DATASET ~ 3D STATS)
TOGGLE_BAR_WIDTH = 22               # px, always-visible side toggle button
LEFT_PANEL_WIDTH = 240              # px, left panel (QUICK KEYS), auto-hides when narrow
LEFT_AUTO_HIDE_BELOW = 1380         # px window width — below this, hide left panel

# label_logic package (scripts/label_logic/)
sys.path.insert(0, str(Path(__file__).parent))
from label_logic.box_roi import save_box_roi, load_box_roi  # noqa: E402
from label_logic.prelabel import run_watershed  # noqa: E402
from label_logic.quality import compute_polygon_stats  # noqa: E402
from label_logic.heightmap import height_map_rgba, normal_map_rgba  # noqa: E402
from label_logic import sam2_predictor  # noqa: E402
from label_logic import auto_classify as _auto_classify  # noqa: E402
from label_logic import picking as _picking  # noqa: E402

CLASSES = [
    ("bottle", "물통", "#3498DB"),
    ("haribo", "하리보", "#E74C3C"),
    ("mango", "망고", "#F39C12"),
    ("metal_case", "캔디케이스", "#9B59B6"),
    ("object", "기타", "#7F8C8D"),
    ("pencil_case", "필통", "#27AE60"),
]
# UNLABELED_COLOR set after THEMES dict (theme-aware)
SELECTED_OUTLINE = "#F1C40F"
DRAW_COLOR = "#FFD700"
HANDLE_FILL = "#FFFFFF"
HANDLE_OUTLINE = "#222222"

ROOT_DIR = Path("C:/Users/dongs/Desktop/260424_1조")
DATA_DIR = ROOT_DIR / "data" / "team1" / "raw"       # mutable — set_active_dataset() updates this
OUT_DIR = ROOT_DIR / "data" / "team1" / "labeled"    # mutable
RUNS_DIR = ROOT_DIR / "runs" / "segment"
IMG_W, IMG_H = 1224, 1024

# OOD anomaly detection (Supersimplenet) — 별도 conda env (py3.11 + anomalib 2.x)
# 호출은 subprocess. score_polygons.py 가 인터페이스.
ANOMALY_ENV_PYTHON = Path("C:/Users/dongs/.conda/envs/anomaly/python.exe")
OOD_SCORE_SCRIPT   = ROOT_DIR / "scripts" / "anomaly" / "score_polygons.py"
OOD_OUTLINE_COLOR  = "#FF1A1A"

# PLY vertex layout for organized Zivid PLY (matches view_3d / probe scripts)
_PLY_VERTEX_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("r", "<u1"), ("g", "<u1"), ("b", "<u1"),
    ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
])


def _read_organized_ply_grid(path: Path) -> np.ndarray:
    """Read an organized Zivid PLY and return (H, W) structured grid."""
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError("PLY header missing end_header")
            if line.strip() == b"end_header":
                break
        n = IMG_W * IMG_H
        buf = f.read(n * _PLY_VERTEX_DTYPE.itemsize)
    return np.frombuffer(buf, dtype=_PLY_VERTEX_DTYPE, count=n).reshape(IMG_H, IMG_W)


def set_active_dataset(data_dir: Path, out_dir: Path) -> None:
    """Switch active dataset. LabelTool.reload() must be called after this."""
    global DATA_DIR, OUT_DIR
    DATA_DIR = Path(data_dir)
    OUT_DIR = Path(out_dir)


def get_drive_root() -> Path | None:
    """Read drive_root from settings.json (or None if not set)."""
    try:
        if SETTINGS_PATH.exists():
            d = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            dr = d.get("drive_root")
            if dr and Path(dr).exists():
                return Path(dr)
    except Exception:
        pass
    return None


def save_drive_root(drive_root: Path) -> None:
    """Save drive_root to settings.json (preserve other keys)."""
    existing = {}
    if SETTINGS_PATH.exists():
        try:
            existing = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    existing["drive_root"] = str(drive_root)
    SETTINGS_PATH.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def discover_datasets(root: Path | None = None):
    """Return list of (display_name, data_dir, out_dir) tuples.

    새 구조: <root>/data/<team>/{raw, labeled, manual_picks}
    각 team 폴더가 하나의 dataset으로 노출됨.

    Drive 폴백은 유지 (다른 PC에서 작업 시).
    """
    out = []
    seen_names = set()

    # 1) Local data/<team>/ (primary — new structure)
    data_root = ROOT_DIR / "data"
    if data_root.exists():
        for team_dir in sorted(data_root.iterdir()):
            if not team_dir.is_dir():
                continue
            raw = team_dir / "raw"
            labeled = team_dir / "labeled"
            if raw.exists():
                out.append((team_dir.name, raw, labeled))
                seen_names.add(team_dir.name)

    # 2) Drive fallback (다른 PC 환경)
    drive_root = root if root else get_drive_root()
    if drive_root and drive_root.exists():
        raw_dir = drive_root / "raw"
        if raw_dir.exists():
            for d in sorted(raw_dir.iterdir()):
                if d.is_dir() and d.name not in seen_names:
                    labeled = drive_root / "labeled" / d.name
                    out.append((f"{d.name} (drive)", d, labeled))
                    seen_names.add(d.name)
    return out
VERTEX_HIT_RADIUS_PX = 10
ZOOM_MIN = 0.25
ZOOM_MAX = 8.0
ZOOM_STEP = 1.25


def dedup_polygons(result_data):
    seen = set()
    out = []
    for entry in result_data:
        if not isinstance(entry, list) or not entry:
            continue
        first = entry[0]
        if not (isinstance(first, list) and first
                and isinstance(first[0], list) and len(first[0]) >= 2):
            continue
        if len(first) < 3:
            continue
        key = tuple((round(p[0]), round(p[1])) for p in first)
        if key in seen:
            continue
        seen.add(key)
        out.append([[float(p[0]), float(p[1])] for p in first])
    return out


def point_in_polygon(x, y, poly):
    n = len(poly)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def polygon_centroid(poly):
    if not poly:
        return 0.0, 0.0
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def closest_edge(px, py, polygon):
    """Return (edge_idx, foot_x, foot_y, dist) for the polygon edge nearest to (px,py).

    edge i connects polygon[i] -> polygon[(i+1) % n]. The new vertex,
    if inserted, goes at index edge_idx+1.
    """
    n = len(polygon)
    if n < 2:
        return None
    best = None
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        dx = x2 - x1
        dy = y2 - y1
        seg_len2 = dx * dx + dy * dy
        if seg_len2 < 1e-12:
            t = 0.0
        else:
            t = ((px - x1) * dx + (py - y1) * dy) / seg_len2
            t = max(0.0, min(1.0, t))
        fx = x1 + t * dx
        fy = y1 + t * dy
        d2 = (px - fx) ** 2 + (py - fy) ** 2
        if best is None or d2 < best[3]:
            best = (i, fx, fy, d2)
    return best


# ============================================================
# Tab 1: LabelTool
# ============================================================

class LabelTool:
    """폴리곤 라벨링 탭."""

    def __init__(self, parent):
        self.parent = parent
        self.shots = self.scan_shots()
        if not self.shots:
            messagebox.showerror("Error", f"No data in {DATA_DIR}")
            return
        self.idx = 0
        self.polys = []
        self.selected = -1
        self.undo_stack = []
        self._listbox_lock = False
        self.base_scale = 1.0
        self.zoom = 1.0
        self.scale = 1.0
        self.tk_image = None
        self.original_image = None
        self.image_id = None
        self.viewport_w = 0
        self.viewport_h = 0
        self.data = {}
        self.mode = "normal"
        self.draw_points = []
        self.draw_preview_ids = []
        self.rubber_band_id = None
        self.editing_idx = -1
        self.edit_handle_ids = []
        self.edit_pre_snapshot = None
        self.dragging_vertex = None
        # Box ROI state (loaded per-dataset from meta.yaml)
        self.box_roi = None  # dict {u_min, u_max, v_min, v_max} or None
        self._box_roi_start = None
        self._box_roi_rect_id = None
        self._box_roi_display_id = None
        # Height/Normal/Heatmap overlay cache (per shot)
        self._height_rgba = None       # full-resolution numpy RGBA (lazy)
        self._normal_rgba = None
        self._heatmap_rgba = None      # 폴리곤 pickability heatmap
        # OOD anomaly overlay (Supersimplenet)
        self.ood_visible = False
        self.ood_threshold = None
        self.ood_normal_stats = None
        self._ood_proc_running = False
        self._height_photo = None      # current scale PhotoImage
        self._normal_photo = None
        self._heatmap_photo = None
        self._height_overlay_id = None
        self._normal_overlay_id = None
        self._heatmap_overlay_id = None
        # Pick (dual-suction) state — computed on demand, cleared on shot change
        self._picks_data = None        # dict: {"shot", "n_ok", "n_fail", "picks":[...]}
        self._pick_canvas_ids = []     # canvas item ids for cleanup
        # YOLO split override (Y key): {shot_stem: "train"|"val"}
        self.split_override = self._load_split_override()
        # config/default.yaml live-reload watcher state
        try:
            self._yaml_path = Path(_picking._CONFIG_PATH)
            self._yaml_mtime = self._yaml_path.stat().st_mtime
        except Exception:
            self._yaml_path = None
            self._yaml_mtime = None
        self.build_ui()
        # Start yaml watcher polling (1s interval)
        if self._yaml_path is not None:
            self.parent.after(1000, self._poll_yaml_change)
        self.load_shot()
        self.refresh_progress()

    # ──────────────────────────────────────────────
    # YOLO split override (Y key)
    # ──────────────────────────────────────────────
    def _split_override_path(self) -> Path:
        return OUT_DIR / "split_override.json"

    def _load_split_override(self) -> dict:
        p = self._split_override_path()
        if not p.exists():
            return {}
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            return {k: v for k, v in d.items() if v in ("train", "val")}
        except Exception:
            return {}

    def _save_split_override(self) -> None:
        p = self._split_override_path()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(
                json.dumps(self.split_override, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            self.status.config(text=f"split_override 저장 실패: {e}")

    def _current_class_set(self) -> set:
        """현재 메모리 polys에서 부여된 class_id 집합 (라벨툴 진행중 반영)."""
        cls = set()
        for p in self.polys:
            cid = p.get("class_id", -1)
            if 0 <= cid < len(CLASSES):
                cls.add(cid)
        return cls

    def _compute_split_for_current(self) -> tuple[str, str]:
        """(split, source) 반환. split ∈ {train, val, empty}."""
        if not self.shots or not (0 <= self.idx < len(self.shots)):
            return "empty", "—"
        shot_name = self.shots[self.idx]["name"]
        if shot_name in self.split_override:
            return self.split_override[shot_name], "manual"
        cls = self._current_class_set()
        if not cls:
            return "empty", "auto"
        if len(cls) == 1:
            return "train", "auto-single"
        return "val", "auto-multi"

    def toggle_split_override(self) -> None:
        """Y 키: AUTO → FORCE-TRAIN → FORCE-VAL → AUTO 순환."""
        if not self.shots or not (0 <= self.idx < len(self.shots)):
            return
        shot_name = self.shots[self.idx]["name"]
        cur = self.split_override.get(shot_name)
        if cur is None:
            self.split_override[shot_name] = "train"
            msg = "✋ YOLO split → TRAIN (수동 override)"
        elif cur == "train":
            self.split_override[shot_name] = "val"
            msg = "✋ YOLO split → VAL (수동 override)"
        else:
            del self.split_override[shot_name]
            msg = "↩ override 해제 → AUTO"
        self._save_split_override()
        self._refresh_split_ui()
        self.status.config(text=msg)

    def _refresh_split_ui(self) -> None:
        """배지 + status_left + dashboard 한 번에 갱신."""
        self._update_split_badge()
        self._update_status_left()
        if hasattr(self, "dash_split_label"):
            split_stats = self._compute_split_distribution()
            self.dash_split_label.config(
                text=f"YOLO: train {split_stats['train']} / val {split_stats['val']} "
                     f"/ skip {split_stats['empty']}  "
                     f"(수동 {split_stats['manual']})"
            )

    def _update_split_badge(self) -> None:
        """shot_label 옆 큰 색상 배지 — TRAIN(초록) / VAL(파랑) / —(회색)."""
        if not hasattr(self, "split_badge"):
            return
        split, src = self._compute_split_for_current()
        if split == "train":
            text, bg = "  TRAIN  ", "#27AE60"
        elif split == "val":
            text, bg = "   VAL   ", "#2980B9"
        else:
            text, bg = "  EMPTY  ", "#7F8C8D"
        if src == "manual":
            text = "✋" + text.strip() + " (수동)"
            text = f"  {text}  "
        self.split_badge.config(text=text, bg=bg)
        self.split_source_label.config(text=f"({src})")

    def _compute_split_distribution(self) -> dict:
        """전체 샷에 대해 train/val/empty 카운트.

        현재 샷은 메모리(self.polys) 기준, 다른 샷은 디스크 라벨 기준.
        """
        counts = {"train": 0, "val": 0, "empty": 0, "manual": 0}
        if not self.shots:
            return counts
        cur_name = self.shots[self.idx]["name"] if (
            0 <= self.idx < len(self.shots)) else None
        for i, shot in enumerate(self.shots):
            name = shot["name"]
            if name in self.split_override:
                counts[self.split_override[name]] += 1
                counts["manual"] += 1
                continue
            if name == cur_name:
                cls = self._current_class_set()
            else:
                lbl = OUT_DIR / "labels" / f"{name}.txt"
                cls = set()
                if lbl.exists():
                    try:
                        for line in lbl.read_text(encoding="utf-8").splitlines():
                            parts = line.strip().split()
                            if parts:
                                cid = int(parts[0])
                                if 0 <= cid < len(CLASSES):
                                    cls.add(cid)
                    except Exception:
                        pass
            if not cls:
                counts["empty"] += 1
            elif len(cls) == 1:
                counts["train"] += 1
            else:
                counts["val"] += 1
        return counts

    def scan_shots(self):
        shots = []
        for bmp in sorted(DATA_DIR.glob("*.bmp")):
            data_json = DATA_DIR / f"{bmp.stem}_data.json"
            if data_json.exists():
                shots.append({"bmp": bmp, "data": data_json, "name": bmp.stem})
        return shots

    # ──────────────────────────────────────────────
    # 3D Stats panel (selected polygon)
    # ──────────────────────────────────────────────
    def refresh_3d_stats(self):
        if not hasattr(self, "stats_text"):
            return
        if not (0 <= self.selected < len(self.polys)):
            self.stats_text.config(text="(폴리곤 선택 시 표시)", fg="#7F8C8D")
            return
        if not self.shots or not (0 <= self.idx < len(self.shots)):
            return
        ply = self.shots[self.idx]["bmp"].with_suffix(".ply")
        if not ply.exists():
            self.stats_text.config(text="(.ply 없음)", fg="#E74C3C")
            return
        try:
            stats = compute_polygon_stats(ply, self.polys[self.selected]["points"])
        except Exception as e:
            self.stats_text.config(text=f"(error: {e})", fg="#E74C3C")
            return
        # interpret status colors
        warn = []
        if stats.norm_cos < 0.7 and not (stats.norm_cos != stats.norm_cos):  # not NaN
            warn.append("normal 일관성 낮음")
        if stats.z_std > 50:
            warn.append("Z std 큼")
        if stats.valid_ratio < 0.6:
            warn.append("3D valid 낮음")
        text = (
            f"#{self.selected:02d}  pts={stats.n_points}\n"
            f"Z mean : {stats.z_mean:7.1f} mm\n"
            f"Z std  : {stats.z_std:7.1f} mm\n"
            f"Norm cos: {stats.norm_cos:6.3f}\n"
            f"Valid % : {stats.valid_ratio*100:5.1f}%\n"
        )
        fg = "#27AE60"
        if warn:
            text += "⚠ " + ", ".join(warn)
            fg = "#E67E22"
        self.stats_text.config(text=text, fg=fg)

    # ──────────────────────────────────────────────
    # Layer toggles (Height / Normal map overlay)
    # ──────────────────────────────────────────────
    def _on_layer_toggle(self):
        # Lazy-compute on first toggle on. ~3s freeze (sync; async in T12).
        if self.show_height_var.get() and self._height_rgba is None:
            self._ensure_height_rgba()
        if self.show_normal_var.get() and self._normal_rgba is None:
            self._ensure_normal_rgba()
        if self.show_heatmap_var.get() and self._heatmap_rgba is None:
            self._ensure_heatmap_rgba()
        self.refresh_canvas()

    def _make_overlay_photo(self, rgba):
        """Convert numpy RGBA (H, W, 4) → PhotoImage at current scale.

        Returns None on failure.
        """
        if rgba is None:
            return None
        try:
            img = Image.fromarray(rgba, "RGBA")
            new_w = max(1, int(IMG_W * self.scale))
            new_h = max(1, int(IMG_H * self.scale))
            img = img.resize((new_w, new_h), Image.NEAREST)
            return ImageTk.PhotoImage(img)
        except Exception as e:
            print(f"overlay photo error: {e}")
            return None

    # ──────────────────────────────────────────────
    # 3D viewer — launch Open3D window in separate process
    # ──────────────────────────────────────────────
    def on_open_3d_view(self):
        if not self.shots or not (0 <= self.idx < len(self.shots)):
            messagebox.showinfo("3D View", "샷이 로드되지 않았습니다.")
            return
        shot = self.shots[self.idx]
        ply = shot["bmp"].with_suffix(".ply")
        if not ply.exists():
            messagebox.showerror("3D View", f".ply not found: {ply}")
            return
        # Save current shot first so polygons reflect latest edits
        try:
            self.save_current()
        except Exception:
            pass
        # Pass labeled data.json (with class_id) for highlight coloring
        labeled_data = OUT_DIR / "data_json" / f"{shot['name']}_data.json"
        if not labeled_data.exists():
            labeled_data = shot["data"]   # fallback to raw
        view_script = Path(__file__).parent / "view_3d.py"
        # open3d only available in Anaconda (Python 3.12), not py (3.13)
        # Prefer Anaconda; fallback to current interpreter.
        anaconda_py = Path("C:/ProgramData/Anaconda3/python.exe")
        py_exec = str(anaconda_py) if anaconda_py.exists() else sys.executable
        cmd = [py_exec, str(view_script), str(ply),
               "--polygons", str(labeled_data),
               "--blend", "0.5",     # equal mix of texture + class color
               "--heatmap",          # K 키로 toggle (class color ↔ heatmap)
               ]
        # If picks were computed for this shot, also pass them to 3D view
        picks_path = self._picks_json_path()
        if picks_path is not None and picks_path.exists():
            cmd.extend(["--picks", str(picks_path)])
        try:
            import subprocess
            subprocess.Popen(cmd)
            self.status.config(text="🌀 3D 윈도우 열림 (별도 process)")
        except Exception as e:
            messagebox.showerror("3D View", str(e))

    # ──────────────────────────────────────────────
    # Pick (dual-suction rectangle) computation + visualization
    # ──────────────────────────────────────────────
    def _picks_json_path(self):
        if not self.shots or not (0 <= self.idx < len(self.shots)):
            return None
        shot = self.shots[self.idx]
        # 새 위치: outputs/<team>/picks/<shot>_picks.json (data/<team>/labeled에서 team 추정)
        try:
            team_name = OUT_DIR.parent.name      # data/<team>/labeled → <team>
            picks_dir = ROOT_DIR / "outputs" / team_name / "picks"
            picks_dir.mkdir(parents=True, exist_ok=True)
            return picks_dir / f"{shot['name']}_picks.json"
        except Exception:
            return ROOT_DIR / "outputs" / "team1" / "picks" / f"{shot['name']}_picks.json"

    # ──────────────────────────────────────────────
    # 현재 샷 완전 삭제 (raw + labeled + manual_picks + outputs)
    # ──────────────────────────────────────────────
    def _collect_shot_files(self, shot_name):
        """현재 샷에 연결된 모든 파일 path. 존재하는 것만 반환.

        명시 candidates + outputs/<team>/ 하위에 shot_name이 stem인 모든 파일
        (미래 viz 추가 대비 fallback glob).
        """
        team_root = DATA_DIR.parent                  # data/<team>/
        team_name = team_root.name
        out_root = ROOT_DIR / "outputs" / team_name  # outputs/<team>/
        candidates = [
            # Raw 4파일
            DATA_DIR / f"{shot_name}.ply",
            DATA_DIR / f"{shot_name}.bmp",
            DATA_DIR / f"{shot_name}_data.json",
            DATA_DIR / f"{shot_name}_info.json",
            # Labeled 출력
            OUT_DIR / "data_json" / f"{shot_name}_data.json",
            OUT_DIR / "labels" / f"{shot_name}.txt",
            OUT_DIR / "images" / f"{shot_name}.bmp",
            OUT_DIR / "images_train" / f"{shot_name}.bmp",
            OUT_DIR / "images_val" / f"{shot_name}.bmp",
            OUT_DIR / "labels_train" / f"{shot_name}.txt",
            OUT_DIR / "labels_val" / f"{shot_name}.txt",
            # Manual GT picks
            team_root / "manual_picks" / f"{shot_name}_manual_picks.json",
            # Outputs (compute_pick / heatmap)
            out_root / "picks" / f"{shot_name}_picks.json",
            out_root / "picks" / f"{shot_name}_picks.png",
            out_root / "heatmaps" / f"{shot_name}_heatmap.png",
            # YOLO derived
            out_root / "yolo_v2" / "images" / "train" / f"{shot_name}.bmp",
            out_root / "yolo_v2" / "images" / "val" / f"{shot_name}.bmp",
            out_root / "yolo_v2" / "labels" / "train" / f"{shot_name}.txt",
            out_root / "yolo_v2" / "labels" / "val" / f"{shot_name}.txt",
        ]
        existing = [p for p in candidates if p.exists()]
        # outputs/<team>/ 하위 fallback glob — 미래 viz 추가 대비
        if out_root.exists():
            for p in out_root.rglob(f"{shot_name}*"):
                if p.is_file() and p not in existing:
                    existing.append(p)
        return existing

    def on_delete_current_shot(self):
        """현재 샷 영구 삭제 — 두 단계 확인."""
        if not self.shots or not (0 <= self.idx < len(self.shots)):
            messagebox.showinfo("샷 삭제", "선택된 샷이 없습니다.")
            return
        shot = self.shots[self.idx]
        name = shot["name"]
        targets = self._collect_shot_files(name)
        if not targets:
            messagebox.showinfo("샷 삭제", "삭제할 파일이 없습니다.")
            return

        # 1단계 확인 — 어떤 파일이 삭제되는지 보여줌
        listing = "\n".join(
            f"  • {p.relative_to(ROOT_DIR)}" for p in targets[:15]
        )
        if len(targets) > 15:
            listing += f"\n  ... 외 {len(targets) - 15}개"
        msg1 = (
            f"샷 [{name}]의 다음 {len(targets)}개 파일을 영구 삭제하시겠습니까?\n\n"
            f"{listing}\n\n"
            f"이 작업은 되돌릴 수 없습니다."
        )
        if not messagebox.askokcancel(
            "⚠ 샷 삭제 확인 (1/2)", msg1, icon="warning",
        ):
            return

        # 2단계 확인 — 마지막 안전장치
        if not messagebox.askyesno(
            "⚠ 최종 확인 (2/2)",
            f"진짜로 삭제하시겠습니까?\n\n"
            f"샷 {name}의 {len(targets)}개 파일이 영구 삭제됩니다.\n"
            f"실수로 누른 거면 [아니오]를 눌러 취소하세요.",
            icon="warning",
        ):
            return

        # 삭제 실행
        deleted, failed = [], []
        for p in targets:
            try:
                p.unlink()
                deleted.append(p)
            except Exception as e:
                failed.append((p, str(e)))

        # 결과 요약 + 다음 샷으로
        result = f"✓ {len(deleted)}개 삭제됨"
        if failed:
            result += f"\n✗ {len(failed)}개 실패:\n"
            result += "\n".join(f"  {p.name}: {err}" for p, err in failed[:5])
        messagebox.showinfo("샷 삭제 결과", result)

        # 샷 목록 재스캔 + 다음 샷 로드
        self.shots = self.scan_shots()
        if self.shots:
            # 같은 인덱스 유지 (이미 사라졌으니 다음 샷이 그 자리에)
            self.idx = min(self.idx, len(self.shots) - 1)
            self.load_shot()
        else:
            messagebox.showinfo("샷 없음",
                                f"team1 데이터셋에 남은 샷이 없습니다.")
            # 빈 상태 유지 (사용자가 다른 dataset으로 전환하거나 종료)
        # 삭제 후 dashboard 갱신
        self.refresh_progress()

    # ──────────────────────────────────────────────
    # Picking thresholds 편집 다이얼로그
    # ──────────────────────────────────────────────
    # 각 게이트의 "비활성화" 값. 🚫 OFF 버튼 클릭 시 이 값으로 설정.
    # None 또는 항목 없음 = 비활성화 불가 (물리적 dim 등).
    PICKING_OFF_VALUES = {
        "suction.min_flat_patch_diameter_mm": 0.0,
        "gates.s1_overlap_defer":             1.0,    # never defer
        "gates.s2_arm_protrusion_mm":         9999.0,
        "gates.s2_cup_protrusion_mm":         9999.0,
        "gates.s2_cup_bump_mm":               9999.0,
        "gates.s2_cup_coverage_min":          0.0,
        "gates.s2_rect_planarity_mm":         9999.0,
        "gates.cup_support_min_points":       0,
        "gates.s3_flatness_min":              0.0,
        "gates.s4_nan_ratio_reject":          1.0,
        "gates.s4_nan_ratio_reject_bottle":   1.0,
        "approach.max_tilt_deg":              90.0,
        "occlusion.clear_hard_reject":        0.0,
        "bin.box_roi_margin_px":              0,
    }

    # (path, label, unit, description) — description은 행 hover 시 tooltip
    PICKING_THRESHOLD_SECTIONS = [
        ("흡착 (Suction tool)", [
            ("suction.cup_od_mm", "컵 외경", "mm",
             "흡착컵의 외경 (직경). 표준 25mm.\n"
             "RECT_SHORT_MM(픽 사각형 짧은 변) = 이 값."),
            ("suction.cup_pitch_mm", "컵 간격 (pitch)", "mm",
             "두 컵 중심 사이 거리. 표준 45mm (50→45 단축).\n"
             "객체 length가 이 값 미만이면 IMPOSSIBLE.\n"
             "RECT_LONG_MM = pitch + cup_od = 70mm."),
            ("suction.min_flat_patch_diameter_mm", "최소 평면 패치 지름", "mm",
             "픽 위치에 요구되는 평평한 영역의 최소 지름.\n"
             "표준 28mm (cup OD 25 + 약간 margin).\n"
             "0으로 두면 사실상 무시 (가장 빨리 풀리는 게이트)."),
            ("suction.safety_offset_mm", "안전 offset", "mm",
             "표면 + 이 거리 위에 픽 위치를 보고. 로봇이 꽉 누르지 않도록.\n"
             "고무 컴플라이언스가 나머지를 흡수. 표준 2mm."),
            ("suction.head_clearance_mm", "헤드 clearance", "mm",
             "흡착 컵 표면 위 헤드 본체까지 높이.\n"
             "Arm 충돌 검사가 이 높이부터 시작 (그 아래는 컵 자체).\n"
             "표준 5mm."),
            ("suction.arm_box_scale", "Arm box scale", "x (1.3)",
             "Arm 충돌 검사 영역 = 픽 사각형 × 이 배율.\n"
             "1.3 = 91×32.5mm. 더 크면 더 많은 충돌 잡힘."),
            ("suction.arm_box_height_mm", "Arm box 높이", "mm",
             "픽 표면에서 위로 이 높이까지 충돌 검사.\n"
             "표준 200mm (팔 길이)."),
        ]),
        ("Approach (탑다운)", [
            ("approach.max_tilt_deg", "최대 tilt 각도", "°",
             "픽 normal과 카메라 -Z 사이 최대 각도.\n"
             "이 이상 기울어지면 RB5 도달 어려워 DEFER.\n"
             "표준 70° (거꾸로는 회전 단계에서 별도 차단)."),
            ("approach.top_surface_percentile", "Top surface", "%",
             "Normal 방향 상위 N%만 표면으로 인정.\n"
             "픽 위치 산출 시 사용. 표준 95 (상위 5%)."),
            ("approach.top_surface_band_mm", "Top surface 폭", "mm",
             "Max에서 이 거리 안 점들도 표면으로 인정.\n"
             "Percentile + 이 값 OR 조건. 표준 5mm."),
        ]),
        ("DEFER 게이트", [
            ("gates.s1_overlap_defer", "S1 폴리곤 overlap", "0~1",
             "다른 클래스 폴리곤이 자기 위 덮은 비율 임계.\n"
             "이상이면 다음 라운드 (위 객체 픽 후 재시도). 표준 0.10."),
            ("gates.s2_arm_protrusion_mm", "S2 팔 충돌", "mm",
             "Arm box 안 다른 물체가 픽 표면 위 이만큼 솟으면 reject.\n"
             "낮을수록 충돌 회피 빡셈. 표준 10mm.\n"
             "현재 데이터 기준 30mm 이상은 88% reject."),
            ("gates.s2_cup_protrusion_mm", "S2 컵 진입 충돌", "mm",
             "컵 위 작은 원기둥 (head clearance 안) 충돌.\n"
             "팔 검사보다 작은 영역. 표준 5mm."),
            ("gates.s2_cup_bump_mm", "S2 컵 표면 돌출", "mm",
             "컵 OD 안 self surface 돌출 (max - p50).\n"
             "이상이면 진공 누설로 흡착 실패.\n"
             "표준 5mm. 컵 컴플라이언스 ~3mm + 노이즈 보정."),
            ("gates.s2_cup_coverage_min", "S2 컵 라벨 coverage", "0~1",
             "컵 OD가 라벨 폴리곤 안에 들어가는 비율.\n"
             "0.8 = 80% 이상이 라벨 안. 가장자리 살짝 빠지는 건 OK.\n"
             "절반 이상 빠지면 reject."),
            ("gates.s2_rect_planarity_mm", "S2 사각형 평면도 std", "mm",
             "픽 사각형 안 self 점들 normal 방향 std.\n"
             "두 컵 사이 영역도 평평해야 함.\n"
             "표준 5mm. 변형체 봉지 자연 곡률 흡수."),
            ("gates.cup_support_min_points", "S2 컵 support 최소점수", "pts",
             "각 cup OD 안 valid 3D 점의 최소 개수.\n"
             "표준 5. 0으로 두면 게이트 완전 비활성.\n"
             "현재 데이터에서 가장 큰 실패 카테고리 (~36%)."),
        ]),
        ("IMPOSSIBLE 게이트", [
            ("gates.s3_flatness_min", "S3 평면 패치 score", "0~1",
             "폴리곤 안 어딘가 normal 정렬 score 이만큼이어야.\n"
             "1.0 = 완벽 평면. 0.9 = 약간 곡률 허용. 표준 0.90.\n"
             "이 미만 = 객체 표면이 너무 거칠어 IMPOSSIBLE."),
            ("gates.s4_nan_ratio_reject", "S4 NaN ratio", "0~1",
             "폴리곤 안 깊이 NaN(측정 실패) 비율 임계.\n"
             "0.25 = 25% 이상 NaN이면 깊이 신뢰X → IMPOSSIBLE.\n"
             "(bottle은 더 빡세게 — 아래 항목)."),
            ("gates.s4_nan_ratio_reject_bottle", "S4 NaN bottle", "0~1",
             "Bottle 클래스 전용 NaN 임계 (더 엄격).\n"
             "투명 영역의 systematic depth bias 때문.\n"
             "표준 0.15."),
        ]),
        ("기타", [
            ("occlusion.clear_hard_reject", "Occlusion clear 임계", "0~1",
             "폴리곤 둘레 occluded 비율 hard reject 임계.\n"
             "0.30 = 70% 이상 가려지면 IMPOSSIBLE.\n"
             "보이는 부분이 너무 작아 진짜 중심 알 수 없음."),
            ("bin.box_roi_margin_px", "Bin 벽 마진", "px",
             "픽 사각형이 빈 벽에서 떨어져야 하는 거리.\n"
             "이 미만이면 box_roi_wall로 reject (충돌 위험).\n"
             "표준 20px."),
            ("confidence.tier_high", "Tier HIGH 임계", "0~1",
             "Confidence 점수 ≥ 이 값 → HIGH (초록).\n"
             "표준 0.70. 라벨툴 캔버스에서 우선 표시."),
            ("confidence.tier_medium", "Tier MEDIUM 임계", "0~1",
             "Confidence ≥ 이 값 → MEDIUM (노랑), 미만 → LOW (주황).\n"
             "표준 0.50."),
        ]),
    ]

    def on_open_thresholds_dialog(self):
        """Picking thresholds 편집 다이얼로그."""
        import importlib
        import yaml as _yaml

        config_path = ROOT_DIR / "config" / "default.yaml"
        if not config_path.exists():
            messagebox.showerror("Thresholds", f"config 없음: {config_path}")
            return

        # 현재 YAML 로드
        with open(config_path, encoding="utf-8") as f:
            current = _yaml.safe_load(f) or {}

        win = tk.Toplevel(self.parent)
        win.title("Picking Thresholds Configuration")
        win.configure(bg=COLOR_BG_PANEL)
        # NOTE: transient/grab_set 제거 → minimize/maximize 버튼 표시 +
        # 다이얼로그 떠 있는 동안 메인 창 조작 가능 (non-modal).
        win.resizable(True, True)
        win.geometry("680x780")

        # ── 1) PROBE 결과 (TOP, 고정 높이) ────────────────────────
        probe_frame = tk.Frame(win, bg=COLOR_BG_PANEL)
        probe_frame.pack(side="top", fill="x", padx=8, pady=(8, 4))
        probe_summary = tk.Label(
            probe_frame, text="📊 Probe: (아직 안 돌림 — ▶ Save & Re-probe 클릭)",
            bg=COLOR_BG_PANEL, fg=COLOR_TEXT, anchor="w",
            font=("Segoe UI", 11, "bold"),
        )
        probe_summary.pack(fill="x")
        probe_text = tk.Text(
            probe_frame, height=10, font=("Consolas", 9),
            bg=COLOR_BG_INPUT, fg=COLOR_TEXT, relief="flat",
        )
        probe_text.pack(fill="x", pady=(2, 0))
        probe_text.insert("1.0", "(probe 실행 후 카테고리 breakdown 표시)")
        probe_text.config(state="disabled")
        last_probe_ok = [None]   # mutable holder for closure (Δ 계산)

        # ── 2) 하단 버튼 (BOTTOM, 고정 높이) ───────────────────────
        btn_frame = tk.Frame(win, bg=COLOR_BG_PANEL)
        btn_frame.pack(side="bottom", fill="x", padx=8, pady=8)

        # ── 3) 임계값 스크롤 영역 (MIDDLE, expand=True) ────────────
        scroll_container = tk.Frame(win, bg=COLOR_BG_PANEL)
        scroll_container.pack(side="top", fill="both", expand=True, padx=4, pady=(0, 4))
        canvas_w = tk.Canvas(scroll_container, bg=COLOR_BG_PANEL,
                              highlightthickness=0)
        scrollbar = ttk.Scrollbar(scroll_container, orient="vertical",
                                   command=canvas_w.yview)
        scrollable = tk.Frame(canvas_w, bg=COLOR_BG_PANEL)
        scrollable.bind("<Configure>", lambda e: canvas_w.configure(
            scrollregion=canvas_w.bbox("all")))
        canvas_w.create_window((0, 0), window=scrollable, anchor="nw")
        canvas_w.configure(yscrollcommand=scrollbar.set)
        canvas_w.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        # 마우스휠 스크롤 — 다이얼로그 안에 마우스 있을 때만 작동
        def _on_mousewheel(event):
            canvas_w.yview_scroll(-int(event.delta / 120), "units")
        canvas_w.bind("<Enter>", lambda e: canvas_w.bind_all("<MouseWheel>", _on_mousewheel))
        canvas_w.bind("<Leave>", lambda e: canvas_w.unbind_all("<MouseWheel>"))
        win.protocol("WM_DELETE_WINDOW",
                     lambda: (canvas_w.unbind_all("<MouseWheel>"), win.destroy()))

        # 입력 필드 생성
        entries = {}        # path → tk.StringVar
        # OFF 토글 상태 추적
        off_state = {}      # path → bool (True = currently OFF)
        saved_values = {}   # path → str (값을 OFF로 바꾸기 직전의 원본)
        row_handles = {}    # path → {"entry": Entry, "button": Button, "toggle": fn}

        def _apply_off_visuals(path, is_off):
            h = row_handles.get(path)
            if not h:
                return
            ent = h["entry"]
            btn = h["button"]
            if is_off:
                btn.config(text="↩ 복원", bg="#16A085")
                ent.config(state="disabled", disabledbackground="#2A2A2A",
                           disabledforeground=COLOR_TEXT_MUTED)
            else:
                btn.config(text="🚫 OFF", bg="#7F1D1D")
                ent.config(state="normal", fg=COLOR_TEXT)

        def _toggle_off(path, off_val):
            var = entries[path]
            if off_state.get(path, False):
                # OFF → 복원
                var.set(saved_values.get(path, str(off_val)))
                off_state[path] = False
                _apply_off_visuals(path, False)
            else:
                # ON → OFF
                saved_values[path] = var.get()
                var.set(str(off_val))
                off_state[path] = True
                _apply_off_visuals(path, True)

        for sect_name, fields in self.PICKING_THRESHOLD_SECTIONS:
            sect_frame = tk.Frame(scrollable, bg=COLOR_BG_PANEL,
                                   pady=4)
            sect_frame.pack(fill="x", padx=8, pady=(8, 0))
            tk.Label(sect_frame, text=sect_name,
                     bg=COLOR_BG_PANEL, fg=COLOR_ACCENT,
                     font=("Segoe UI", 10, "bold"),
                     anchor="w").pack(fill="x")
            for entry in fields:
                # 4-tuple (path, label, unit, desc) — desc는 hover tooltip
                if len(entry) == 4:
                    path, label, unit, desc = entry
                else:
                    path, label, unit = entry
                    desc = ""
                row = tk.Frame(scrollable, bg=COLOR_BG_PANEL)
                row.pack(fill="x", padx=20)
                lbl = tk.Label(row, text=label, bg=COLOR_BG_PANEL,
                               fg=COLOR_TEXT, width=22, anchor="w")
                lbl.pack(side="left")
                # 현재 값 lookup
                cur_val = current
                for key in path.split("."):
                    cur_val = cur_val.get(key, "") if isinstance(cur_val, dict) else ""
                var = tk.StringVar(value=str(cur_val))
                ent = tk.Entry(row, textvariable=var, width=10,
                                bg=COLOR_BG_INPUT, fg=COLOR_TEXT,
                                insertbackground=COLOR_TEXT)
                ent.pack(side="left")
                tk.Label(row, text=" " + unit, bg=COLOR_BG_PANEL,
                         fg=COLOR_TEXT_MUTED, width=8, anchor="w").pack(side="left")
                # (?) 인디케이터 — hover 시 설명 tooltip
                if desc:
                    info = tk.Label(row, text=" ⓘ", bg=COLOR_BG_PANEL,
                                     fg=COLOR_ACCENT, cursor="question_arrow",
                                     font=("Segoe UI", 10, "bold"))
                    info.pack(side="left")
                    # 라벨, 입력칸, 인디케이터 모두 동일 tooltip 부착
                    ToolTip(lbl, desc, delay=300)
                    ToolTip(ent, desc, delay=300)
                    ToolTip(info, desc, delay=200)
                entries[path] = var
                # 🚫 OFF 토글 버튼 (비활성화 가능한 게이트만)
                if path in self.PICKING_OFF_VALUES:
                    off_val = self.PICKING_OFF_VALUES[path]
                    off_btn = tk.Button(
                        row, text="🚫 OFF",
                        command=lambda p=path, ov=off_val: _toggle_off(p, ov),
                        font=("Segoe UI", 8), width=8,
                        bg="#7F1D1D", fg="white", relief="flat",
                    )
                    off_btn.pack(side="left", padx=2)
                    ToolTip(off_btn,
                            f"이 게이트 토글.\n"
                            f"  ▸ OFF: 값 → {off_val} (게이트 무력화)\n"
                            f"  ▸ 다시 클릭: 직전 값 복원\n"
                            f"  OFF 상태에서는 entry가 회색/잠김.",
                            delay=200)
                    row_handles[path] = {"entry": ent, "button": off_btn}

        # btn_frame, probe_frame are already created above (top + bottom layout).

        def _set_path(d, path, value):
            keys = path.split(".")
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value

        def on_save():
            new_config = dict(current)
            for path, var in entries.items():
                raw = var.get().strip()
                if raw == "":
                    continue
                try:
                    val = int(raw) if "." not in raw and "e" not in raw.lower() else float(raw)
                    # px / points 단위는 정수 유지 (_write_yaml_only와 일관)
                    if path.endswith("_px") or path.endswith("_points"):
                        val = int(float(raw))
                except ValueError:
                    messagebox.showerror("Invalid value",
                                         f"{path} = '{raw}' (숫자 입력 필요)")
                    return
                _set_path(new_config, path, val)
            # YAML 저장
            try:
                with open(config_path, "w", encoding="utf-8") as f:
                    _yaml.safe_dump(new_config, f, allow_unicode=True,
                                    sort_keys=False, default_flow_style=False)
            except Exception as e:
                messagebox.showerror("Save failed", str(e))
                return
            # picking 모듈 reload (새 YAML 반영)
            # suction_score는 picking 상수 직접 참조 X지만, label_tool의
            # heatmap 호출이 _picking.HEATMAP_CUP_RADIUS_PX를 쓰니 picking
            # 한 번만 reload해도 OK. suction_score는 함수만 export.
            try:
                from label_logic import suction_score as _suction_score
                importlib.reload(_suction_score)
                importlib.reload(_picking)
            except Exception as e:
                messagebox.showerror("Reload failed", str(e))
                return
            # reload는 picking 모듈 top-level을 재실행해서 FX/FY/CX/CY를 yaml
            # 기본값(MR60)으로 되돌림. 현재 샷이 MR130 등 다른 카메라일 수
            # 있으므로 reload 직후 항상 재주입.
            self._apply_shot_intrinsics()
            self.status.config(text="✓ Thresholds 저장 + reload 완료. Picks 재계산 중…")
            self.parent.update_idletasks()
            win.destroy()
            # Picks 재계산 (라벨된 샷 있으면) — on_compute_picks 안에서 refresh_progress
            try:
                self.on_compute_picks()
            except Exception as e:
                messagebox.showinfo("Picks", f"재계산 실패 (라벨 없음?): {e}")
                self.refresh_progress()   # 실패해도 dashboard는 한 번 더 갱신

        def on_reset():
            # 디스크에서 다시 로드 (편집 폐기) — OFF 토글 상태도 초기화
            with open(config_path, encoding="utf-8") as f:
                fresh = _yaml.safe_load(f) or {}
            for path, var in entries.items():
                cur_val = fresh
                for key in path.split("."):
                    cur_val = cur_val.get(key, "") if isinstance(cur_val, dict) else ""
                # OFF 상태 해제 (entry 활성화 + 버튼 색상 복귀)
                if off_state.get(path, False):
                    off_state[path] = False
                    _apply_off_visuals(path, False)
                saved_values.pop(path, None)
                var.set(str(cur_val))

        def on_all_off():
            """OFF 가능한 모든 게이트 토글 (이미 OFF인 건 그대로 둠)."""
            for path, off_val in self.PICKING_OFF_VALUES.items():
                if path in entries and not off_state.get(path, False):
                    _toggle_off(path, off_val)

        def _write_yaml_only():
            """현재 entry 값을 yaml에 쓰되 모듈 reload / compute_picks는 안 함."""
            new_config = dict(current)
            for path, var in entries.items():
                raw = var.get().strip()
                if raw == "":
                    continue
                try:
                    val = int(raw) if "." not in raw and "e" not in raw.lower() else float(raw)
                    if path.endswith("_px") or path.endswith("_points"):
                        val = int(float(raw))
                except ValueError:
                    messagebox.showerror("Invalid value", f"{path} = '{raw}' (숫자 필요)")
                    return False
                _set_path(new_config, path, val)
            try:
                with open(config_path, "w", encoding="utf-8") as f:
                    _yaml.safe_dump(new_config, f, allow_unicode=True,
                                    sort_keys=False, default_flow_style=False)
            except Exception as e:
                messagebox.showerror("Save failed", str(e))
                return False
            return True

        def _safe_widget(widget):
            """Tk 위젯이 아직 존재하는지 확인 (다이얼로그 닫혔으면 False)."""
            try:
                return widget.winfo_exists()
            except tk.TclError:
                return False

        def _render_probe_result(result):
            """Main thread에서 실행 — probe 결과를 위젯에 표시.
            전체를 try/except로 감싸서 다이얼로그가 도중에 닫힌 race를 안전 처리."""
            if not _safe_widget(probe_summary):
                return  # 다이얼로그 이미 닫힘
            try:
                if result.returncode != 0:
                    err = (result.stderr or "")[:1500]
                    probe_summary.config(text="📊 Probe: ❌ 실패")
                    probe_text.config(state="normal")
                    probe_text.delete("1.0", "end")
                    probe_text.insert("1.0", f"exit={result.returncode}\n\n{err}")
                    probe_text.config(state="disabled")
                    _reprobe_btn_state("normal")
                    return
                import re as _re
                m = _re.search(r"(\d+)\s+polygons,\s+(\d+)\s+ok,\s+(\d+)\s+fail",
                               result.stdout)
                if m:
                    total, ok, fail = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    pct = (ok / total * 100) if total else 0.0
                    delta = ""
                    if last_probe_ok[0] is not None:
                        d = ok - last_probe_ok[0]
                        delta = f"  (Δ {d:+d})"
                    probe_summary.config(
                        text=f"📊 PICKABLE: {ok}/{total} ({pct:.1f}%){delta}"
                    )
                    last_probe_ok[0] = ok
                lines = result.stdout.splitlines()
                cat_lines = []
                in_table = False
                for line in lines:
                    if line.startswith("category"):
                        in_table = True
                    if in_table:
                        if line.startswith("Per-class") or line.startswith("Failure"):
                            break
                        cat_lines.append(line)
                        if line.strip() == "" and len(cat_lines) > 3:
                            break
                probe_text.config(state="normal")
                probe_text.delete("1.0", "end")
                probe_text.insert("1.0", "\n".join(cat_lines) if cat_lines else "(parse 실패)")
                probe_text.config(state="disabled")
                _reprobe_btn_state("normal")
            except tk.TclError:
                # 도중에 widget이 사라진 경우 (사용자가 다이얼로그 닫음) — 조용히 무시
                return

        def _render_probe_error(err_msg):
            if not _safe_widget(probe_summary):
                return
            try:
                probe_summary.config(text=f"📊 Probe: error — {err_msg[:200]}")
                _reprobe_btn_state("normal")
            except tk.TclError:
                return

        reprobe_button_ref = [None]   # 버튼 핸들 보관 (state 토글용)

        def _reprobe_btn_state(state):
            btn = reprobe_button_ref[0]
            if btn is not None and _safe_widget(btn):
                try:
                    btn.configure(state=state)
                except tk.TclError:
                    pass

        # Probe 진행 상태 (elapsed timer용)
        probe_running = [False]
        probe_start_time = [0.0]

        def _tick_elapsed():
            if not probe_running[0]:
                return
            if not _safe_widget(probe_summary):
                return
            import time as _time
            elapsed = int(_time.monotonic() - probe_start_time[0])
            mm = elapsed // 60
            ss = elapsed % 60
            probe_summary.config(
                text=f"📊 Probe: 실행 중 ({mm}:{ss:02d})… 게이트 OFF 많으면 수 분 걸림"
            )
            self.parent.after(1000, _tick_elapsed)

        def on_reprobe():
            """yaml에 쓰고 별 thread에서 probe 실행 → 결과 main thread에서 표시."""
            if not _write_yaml_only():
                return
            import time as _time
            probe_running[0] = True
            probe_start_time[0] = _time.monotonic()
            probe_summary.config(text="📊 Probe: 실행 중 (0:00)…")
            probe_text.config(state="normal")
            probe_text.delete("1.0", "end")
            probe_text.insert(
                "1.0",
                "(probe 실행 중…)\n\n"
                "참고: 게이트가 많이 꺼져 있을수록 시간이 오래 걸립니다.\n"
                "  ▸ 기본 임계값: ~15초\n"
                "  ▸ All OFF 상태: ~3~5분 (415개 폴리곤이 full pipeline 통과)\n"
                "다이얼로그 다른 조작은 자유롭게 가능합니다.",
            )
            probe_text.config(state="disabled")
            _reprobe_btn_state("disabled")
            self.parent.after(1000, _tick_elapsed)

            def _worker():
                try:
                    import subprocess
                    probe_script = ROOT_DIR / "scripts" / "analysis" / "probe_pick_failures.py"
                    result = subprocess.run(
                        [sys.executable, str(probe_script)],
                        cwd=str(ROOT_DIR),
                        capture_output=True, text=True, timeout=600,
                        encoding="utf-8", errors="replace",
                    )
                    probe_running[0] = False
                    self.parent.after(0, lambda: _render_probe_result(result))
                except subprocess.TimeoutExpired:
                    probe_running[0] = False
                    self.parent.after(0, lambda: _render_probe_error("timeout (>10분)"))
                except Exception as e:
                    probe_running[0] = False
                    msg = str(e)
                    self.parent.after(0, lambda: _render_probe_error(msg))

            threading.Thread(target=_worker, daemon=True).start()

        if _HAS_CTK:
            ctk.CTkButton(btn_frame, text="🚫 All OFF", command=on_all_off,
                          height=28, width=90, fg_color="#7F1D1D").pack(side="left", padx=4)
            reprobe_btn = ctk.CTkButton(btn_frame, text="▶ Save & Re-probe",
                                        command=on_reprobe, height=28, width=140)
            reprobe_btn.pack(side="left", padx=4)
            ctk.CTkButton(btn_frame, text="Save & Apply", command=on_save,
                          height=28, width=110).pack(side="right", padx=4)
            ctk.CTkButton(btn_frame, text="Reset", command=on_reset,
                          height=28, width=80).pack(side="right", padx=4)
            ctk.CTkButton(btn_frame, text="Cancel", command=win.destroy,
                          height=28, width=80).pack(side="right", padx=4)
        else:
            tk.Button(btn_frame, text="🚫 All OFF", command=on_all_off,
                      bg="#7F1D1D", fg="white", width=11).pack(side="left", padx=4)
            reprobe_btn = tk.Button(btn_frame, text="▶ Save & Re-probe",
                                    command=on_reprobe,
                                    bg="#3498DB", fg="white", width=18)
            reprobe_btn.pack(side="left", padx=4)
            tk.Button(btn_frame, text="Save & Apply", command=on_save,
                      width=14).pack(side="right", padx=4)
            tk.Button(btn_frame, text="Reset", command=on_reset,
                      width=10).pack(side="right", padx=4)
            tk.Button(btn_frame, text="Cancel", command=win.destroy,
                      width=10).pack(side="right", padx=4)
        reprobe_button_ref[0] = reprobe_btn

        # 창 위치 (부모 중앙)
        win.update_idletasks()
        px = self.parent.winfo_rootx()
        py = self.parent.winfo_rooty()
        pw = self.parent.winfo_width()
        ph = self.parent.winfo_height()
        ww = win.winfo_width()
        wh = win.winfo_height()
        win.geometry(f"+{px + (pw - ww) // 2}+{py + (ph - wh) // 2}")

    def _refresh_shot_jump_list(self):
        """SHOTS Listbox 갱신 — 필터 적용 + 현재 샷 highlight.
        load_shot 끝에서 호출(현재 샷 표시) + 필터 변경 시 trace로 호출됨.

        Lock 플래그: selection_set / delete + insert 가 <<ListboxSelect>> 를
        트리거해서 _on_shot_jump_select가 호출되면 자동으로 다른 샷으로 점프하는
        버그가 있었음. 갱신 중엔 잠가서 핸들러가 즉시 return하게 함.
        """
        if not hasattr(self, "shot_listbox"):
            return
        self._shot_listbox_locked = True
        try:
            filt = (self.shot_filter_var.get() or "").strip().lower() \
                if hasattr(self, "shot_filter_var") else ""
            self.shot_listbox.delete(0, "end")
            self._filtered_shot_idxs = []
            for i, s in enumerate(getattr(self, "shots", [])):
                name = s.get("name", "")
                if filt and filt not in name.lower():
                    continue
                self._filtered_shot_idxs.append(i)
                self.shot_listbox.insert("end", f"{i:3d}  {name}")
            if 0 <= getattr(self, "idx", -1) < len(getattr(self, "shots", [])):
                try:
                    pos = self._filtered_shot_idxs.index(self.idx)
                    self.shot_listbox.selection_clear(0, "end")
                    self.shot_listbox.selection_set(pos)
                    self.shot_listbox.see(pos)
                except ValueError:
                    pass
        finally:
            # tkinter는 selection_set이 발생시키는 ListboxSelect를 다음 idle
            # 사이클에 처리. lock 해제도 그때 풀어야 그 이벤트가 점프 안 함.
            self.parent.after_idle(
                lambda: setattr(self, "_shot_listbox_locked", False))

    def _on_shot_jump_select(self, _event=None):
        """Listbox 클릭/Enter → 선택한 샷으로 jump.
        _refresh_shot_jump_list가 갱신 중일 땐 lock으로 즉시 return.
        """
        if getattr(self, "_shot_listbox_locked", False):
            return
        if not hasattr(self, "shot_listbox"):
            return
        sel = self.shot_listbox.curselection()
        if not sel:
            return
        pos = sel[0]
        if pos >= len(self._filtered_shot_idxs):
            return
        target = self._filtered_shot_idxs[pos]
        if target == self.idx:
            return
        try:
            self.save_current()
        except Exception:
            pass
        self.idx = target
        self.load_shot()

    def _poll_yaml_change(self):
        """config/default.yaml mtime watcher — 외부 편집 감지 시 자동 reload + 재계산.

        IDE/에디터에서 yaml을 직접 저장하면 1초 안에 picking 모듈 reload하고
        현재 샷의 picks를 다시 계산해서 캔버스에 반영.
        """
        try:
            if self._yaml_path is None or not self._yaml_path.exists():
                self.parent.after(1000, self._poll_yaml_change)
                return
            mtime = self._yaml_path.stat().st_mtime
            if self._yaml_mtime is None or mtime > self._yaml_mtime + 0.1:
                self._yaml_mtime = mtime
                # 모듈 reload (picking + suction_score)
                import importlib
                try:
                    from label_logic import suction_score as _ss
                    importlib.reload(_ss)
                    importlib.reload(_picking)
                except Exception as e:
                    self.status.config(text=f"⚠ yaml reload 실패: {e}")
                else:
                    # reload가 picking module의 FX/FY/CX/CY를 yaml 기본(MR60)으로
                    # 되돌렸으므로 현재 샷의 실제 intrinsics를 즉시 재주입.
                    self._apply_shot_intrinsics()
                    # 자동 재계산은 하지 않음 — 사용자가 G 키로 트리거.
                    # (yaml 변경 → reload 까지만 자동, 재계산은 수동.)
                    self.status.config(
                        text="🔄 config 변경 감지 — G 키로 picks 재계산")
        except Exception:
            pass
        # 다음 폴링 예약
        self.parent.after(1000, self._poll_yaml_change)

    def on_compute_picks(self):
        """Run dual-suction pick computation on current shot and show on canvas."""
        if not self.shots or not (0 <= self.idx < len(self.shots)):
            messagebox.showinfo("Picks", "샷이 로드되지 않았습니다.")
            return
        shot = self.shots[self.idx]
        ply = shot["bmp"].with_suffix(".ply")
        if not ply.exists():
            messagebox.showerror("Picks", f".ply not found: {ply}")
            return
        # Save current edits so polygons reflect what we'll compute on
        try:
            self.save_current()
        except Exception:
            pass

        try:
            self.status.config(text="⏳ 픽 계산 중…")
            self.parent.update_idletasks()
            grid = _read_organized_ply_grid(ply)
        except Exception as e:
            messagebox.showerror("Picks", f"PLY 읽기 실패:\n{e}")
            return

        # Manual picks (사용자 GT) 로드 — 있으면 알고리즘 무시하고 GT 사용
        # 새 위치: <team>/manual_picks/<shot>_manual_picks.json
        # 폴백: raw/ 옆 (옛 위치)
        manual_picks = {}
        try:
            mp_dir = ply.parent.parent / "manual_picks"
            mp_path = mp_dir / f"{shot['name']}_manual_picks.json"
            if not mp_path.exists():
                mp_path = ply.parent / f"{shot['name']}_manual_picks.json"
            if mp_path.exists():
                with open(mp_path, encoding="utf-8") as f:
                    manual_picks = json.load(f)
                if manual_picks:
                    self.status.config(
                        text=f"⏳ 픽 계산 중… ({len(manual_picks)} GT manual picks 적용)"
                    )
                    self.parent.update_idletasks()
        except Exception:
            manual_picks = {}

        # 다른 폴리곤 ring들 (S1 게이트용)
        all_other_rings = [pp["points"] for pp in self.polys]

        all_records = []     # every classified polygon (pickable or not)
        n_attempted = 0
        for i, p in enumerate(self.polys):
            cid = p["class_id"]
            # cid 미부여 폴리곤(-1)도 OOD/unknown 일반 흡착 픽 로직으로 시도.
            # picking.py 가 UNKNOWN_CID 받아 bottle 특화 분기 skip + class_prior 중립.
            is_known = 0 <= cid < len(CLASSES)
            pick_cid = cid if is_known else _picking.UNKNOWN_CID
            n_attempted += 1
            scene_others = [r for k, r in enumerate(all_other_rings) if k != i]
            mp = manual_picks.get(str(i))
            try:
                result = _picking.compute_pick(
                    grid, p["points"], pick_cid, box_roi=self.box_roi,
                    scene_polygons=scene_others, manual_pick=mp,
                )
            except Exception as e:
                result = _picking.PickResult(
                    success=False, reason=f"error: {e}", cid=pick_cid,
                )
            all_records.append({"poly_index": i, **result.to_dict()})

        # Tier-aware sort: HIGH > MEDIUM > LOW > UNPICKABLE.
        # 같은 tier 안 → 충돌 마진 (arm_max_protrusion_mm 낮을수록 안전) → depth.
        # Pickable picks get pick_order; UNPICKABLE leave it None.
        # MANUAL GT 픽이 알고리즘 PICKABLE보다 우선. MANUAL은 충돌 마진
        # 정보 없음 → 0.0 (=가장 안전)으로 폴백돼 tier 우선순위 그대로 유지.
        TIER_RANK = {"MANUAL": -1, "HIGH": 0, "MEDIUM": 1, "LOW": 2, "UNPICKABLE": 3}
        successful = [r for r in all_records if r.get("pickable")]
        for r in all_records:
            if not r.get("pickable"):
                r["tier"] = "UNPICKABLE"
        successful.sort(
            key=lambda r: (TIER_RANK.get(r.get("tier", "LOW"), 9),
                           r.get("arm_max_protrusion_mm", 0.0),
                           r["position_mm"][2])
        )
        for order, rec in enumerate(successful, start=1):
            rec["pick_order"] = order

        self._picks_data = {
            "shot": shot["name"],
            "n_ok": len(successful),
            "n_attempted": n_attempted,
            "n_unpickable": n_attempted - len(successful),
            "box_roi": dict(self.box_roi) if self.box_roi else None,
            "picks": all_records,        # contains BOTH pickable + unpickable
        }

        # Persist JSON for 3D viewer + later inspection
        out_path = self._picks_json_path()
        if out_path is not None:
            try:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(self._picks_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                self.status.config(text=f"픽 JSON 저장 실패: {e}")

        self.show_picks_var.set(True)
        self.refresh_canvas()
        # Tier breakdown for status bar
        tier_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "UNPICKABLE": 0}
        for r in all_records:
            t = r.get("tier", "UNPICKABLE")
            tier_counts[t] = tier_counts.get(t, 0) + 1
        msg = (f"✓ 픽 분류: HIGH {tier_counts['HIGH']} · "
               f"MED {tier_counts['MEDIUM']} · LOW {tier_counts['LOW']} · "
               f"UNPICKABLE {tier_counts['UNPICKABLE']}")
        self.status.config(text=msg)
        # Picks 계산 후 dashboard 갱신 (PICKABLE 카운트 변동 반영)
        self.refresh_progress()

    # Tier color map (canvas 2D)
    TIER_COLORS_2D = {
        "HIGH":       "#2ECC71",   # green
        "MEDIUM":     "#F1C40F",   # yellow
        "LOW":        "#E67E22",   # orange
        "UNPICKABLE": "#7F7F7F",   # gray
        # Manual GT picks — 노란색 (3D 뷰의 manual pick 색과 일치)
        "MANUAL":     "#FFD700",   # gold yellow
    }

    def _draw_picks_on_canvas(self):
        """Draw picking rectangles + cup centers + Z-axis on top of polygons.
        Color-coded by tier (HIGH/MED/LOW). UNPICKABLE shown with gray
        dashed outline at the polygon centroid + reason text."""
        # Cleanup previous
        for cid in self._pick_canvas_ids:
            try:
                self.canvas.delete(cid)
            except Exception:
                pass
        self._pick_canvas_ids = []
        if self._picks_data is None or not self.show_picks_var.get():
            return
        Z_AXIS_LEN_MM = 40.0
        picks = self._picks_data.get("picks", [])

        # Pass 1 — UNPICKABLE markers (drawn first so pickable picks layer on top)
        for rec in picks:
            if rec.get("pickable"):
                continue
            poly_idx = rec.get("poly_index", -1)
            if not (0 <= poly_idx < len(self.polys)):
                continue
            poly_pts = self.polys[poly_idx]["points"]
            if not poly_pts:
                continue
            cu = sum(p[0] for p in poly_pts) / len(poly_pts)
            cv = sum(p[1] for p in poly_pts) / len(poly_pts)
            cu_s = cu * self.scale
            cv_s = cv * self.scale
            r = 18
            # Dashed gray circle at centroid
            mark = self.canvas.create_oval(
                cu_s - r, cv_s - r, cu_s + r, cv_s + r,
                outline=self.TIER_COLORS_2D["UNPICKABLE"], width=2, dash=(4, 3),
            )
            self._pick_canvas_ids.append(mark)
            # Diagonal slash through it
            slash = self.canvas.create_line(
                cu_s - r * 0.7, cv_s - r * 0.7,
                cu_s + r * 0.7, cv_s + r * 0.7,
                fill=self.TIER_COLORS_2D["UNPICKABLE"], width=2,
            )
            self._pick_canvas_ids.append(slash)
            # 사유 텍스트는 사용자가 폴리곤을 클릭(selected)했을 때만 표시.
            # 모두 펼쳐두면 겹쳐 어수선 → on-demand.
            if poly_idx != getattr(self, "selected", -1):
                continue
            status = rec.get("status", "")
            status_kr = {
                "DEFER": "보류",
                "IMPOSSIBLE": "불가",
                "MANUAL": "수동",
            }.get(status, status)
            raw_reason = rec.get("reason") or "사유 미상"
            if "←" in raw_reason:
                main, _, hint = raw_reason.partition("←")
                display = f"[{status_kr}] {main.strip()}\n← {hint.strip()}"
            else:
                display = f"[{status_kr}] {raw_reason.strip()}"
            # 가독성용 반투명 배경 박스 (사유 텍스트가 BMP 위에 안 묻히게)
            bg = self.canvas.create_rectangle(
                cu_s - 170, cv_s + r + 8,
                cu_s + 170, cv_s + r + 60,
                fill="#1A1A1A", outline=self.TIER_COLORS_2D["UNPICKABLE"],
                width=1, stipple="gray50",
            )
            self._pick_canvas_ids.append(bg)
            txt = self.canvas.create_text(
                cu_s, cv_s + r + 14, text=display, anchor="n",
                fill="#FFFFFF",
                font=("Malgun Gothic", 9),
                width=320,
            )
            self._pick_canvas_ids.append(txt)

        # Pass 2 — pickable picks (HIGH/MED/LOW)
        for rec in picks:
            if not rec.get("pickable"):
                continue
            tier = rec.get("tier", "LOW")
            tier_color = self.TIER_COLORS_2D.get(tier, "#7F7F7F")
            cid = rec.get("cid", -1)
            # cid 검사 없음 — UNKNOWN_CID(-1) 픽도 일반 흡착 픽 로직 결과로
            # PICKABLE 받았으면 2D 시각화 표시. 색은 tier 기반이라 cid 의존 X.
            corners_2d = rec.get("rect_corners_2d") or []
            if len(corners_2d) != 4:
                continue
            # ── pick_mode: dual = 두 컵 모두 객체 표면 / single = 한 컵 객체
            # 위 + 다른 컵 빈 공간 (haribo fallback). single은 시각적으로 구분.
            pick_mode = rec.get("pick_mode", "dual")
            is_single = (pick_mode == "single")
            scaled = []
            for u, v in corners_2d:
                scaled.extend([u * self.scale, v * self.scale])
            # Single 모드는 사각형 외곽을 dash로 (사각형이 객체보다 큰 의미)
            rect_kwargs = {"outline": tier_color, "fill": "black",
                            "stipple": "gray25", "width": 4}
            if is_single:
                rect_kwargs["dash"] = (8, 4)
            poly_id = self.canvas.create_polygon(scaled, **rect_kwargs)
            self._pick_canvas_ids.append(poly_id)
            # Cup centers + 외경 원. 컵이 폴리곤 안인지 밖인지 확인 — single
            # 모드의 객체 밖 컵은 회색·점선으로 (빈 공간을 잡는 헛흡입 의미).
            cup_r_mm = _picking.SUCTION_CUP_OD_MM / 2.0
            poly_pts_for_mask = None
            if is_single:
                # self.polys 에서 현재 폴리곤 좌표 가져오기
                poly_idx = rec.get("poly_index")
                if (poly_idx is not None and 0 <= poly_idx < len(self.polys)):
                    poly_pts_for_mask = self.polys[poly_idx].get("points")
            for cup_uv, cup_3d in (
                (rec.get("cup_a_2d"), rec.get("cup_a_3d")),
                (rec.get("cup_b_2d"), rec.get("cup_b_3d")),
            ):
                if not cup_uv:
                    continue
                cu = cup_uv[0] * self.scale
                cv = cup_uv[1] * self.scale
                # single 모드: 컵이 폴리곤 밖에 있으면 회색·점선 (빈 컵)
                cup_outside = False
                if is_single and poly_pts_for_mask:
                    try:
                        mask = _picking.polygon_to_mask(poly_pts_for_mask)
                        ui, vi = int(round(cup_uv[0])), int(round(cup_uv[1]))
                        H, W = mask.shape
                        if 0 <= ui < W and 0 <= vi < H:
                            cup_outside = not bool(mask[vi, ui])
                    except Exception:
                        pass
                cup_color = "#7F7F7F" if cup_outside else "#FFEB3B"
                cup_outline_color = "#404040" if cup_outside else "black"
                # 외경 원 — z에 따라 픽셀 반지름 계산 + scale
                if cup_3d and float(cup_3d[2]) > 1e-6:
                    fx_avg = 0.5 * (_picking.FX + _picking.FY)
                    r_px = cup_r_mm * fx_avg / float(cup_3d[2])
                    r_s = r_px * self.scale
                    cup_od = self.canvas.create_oval(
                        cu - r_s, cv - r_s, cu + r_s, cv + r_s,
                        outline=cup_color, width=2, dash=(4, 2),
                    )
                    self._pick_canvas_ids.append(cup_od)
                # 컵 중심 점
                r_center = 6
                cup_id = self.canvas.create_oval(
                    cu - r_center, cv - r_center, cu + r_center, cv + r_center,
                    fill=cup_color, outline=cup_outline_color, width=2,
                )
                self._pick_canvas_ids.append(cup_id)
            # Single 모드 라벨 — 픽 위치 위쪽에 작은 텍스트
            if is_single:
                pos = rec.get("position_mm")
                base_uv = _picking.project_to_image(pos) if pos else None
                if base_uv is not None:
                    lbl_id = self.canvas.create_text(
                        base_uv[0] * self.scale,
                        base_uv[1] * self.scale - 22,
                        text="SINGLE", fill=tier_color, anchor="s",
                        font=("Malgun Gothic", 9, "bold"),
                    )
                    self._pick_canvas_ids.append(lbl_id)
            # Z-axis (approach direction) — line from pick center along normal
            position_3d = rec.get("position_mm")
            normal = rec.get("normal")
            base_uv = None
            if position_3d and normal:
                tip_3d = [
                    position_3d[0] + Z_AXIS_LEN_MM * normal[0],
                    position_3d[1] + Z_AXIS_LEN_MM * normal[1],
                    position_3d[2] + Z_AXIS_LEN_MM * normal[2],
                ]
                base_uv = _picking.project_to_image(position_3d)
                tip_uv = _picking.project_to_image(tip_3d)
                if base_uv and tip_uv:
                    bx, by = base_uv[0] * self.scale, base_uv[1] * self.scale
                    tx, ty = tip_uv[0] * self.scale, tip_uv[1] * self.scale
                    z_line = self.canvas.create_line(
                        bx, by, tx, ty,
                        fill="#2196F3", width=4, arrow="last", arrowshape=(12, 14, 5),
                    )
                    self._pick_canvas_ids.append(z_line)
            # Pick order badge (tier-colored) + confidence text
            order = rec.get("pick_order")
            if order is not None and base_uv is not None:
                lx = base_uv[0] * self.scale
                ly = base_uv[1] * self.scale - 24
                bg = self.canvas.create_oval(
                    lx - 14, ly - 14, lx + 14, ly + 14,
                    fill=tier_color, outline="white", width=2,
                )
                txt = self.canvas.create_text(
                    lx, ly, text=str(order),
                    fill="white", font=("Arial", 13, "bold"),
                )
                self._pick_canvas_ids.append(bg)
                self._pick_canvas_ids.append(txt)
                # Confidence score below badge
                conf = rec.get("confidence", 0.0)
                conf_txt = self.canvas.create_text(
                    lx + 22, ly,
                    text=f"{tier} {conf:.2f}",
                    fill=tier_color, anchor="w",
                    font=("Arial", 9, "bold"),
                )
                self._pick_canvas_ids.append(conf_txt)

    def _ensure_height_rgba(self):
        if self._height_rgba is not None:
            return
        if not self.shots or not (0 <= self.idx < len(self.shots)):
            return
        ply = self.shots[self.idx]["bmp"].with_suffix(".ply")
        if not ply.exists():
            messagebox.showerror("Height map", f".ply not found: {ply}")
            self.show_height_var.set(False)
            return
        try:
            self.status.config(text="⏳ Height map 계산 중... (~3초)")
            self.parent.update_idletasks()
            # height_min=-2.0 to include box bottom plane itself
            # (objects sitting on plane are otherwise invisible)
            self._height_rgba = height_map_rgba(
                ply, alpha=0.55, height_min=-2.0, height_max=200.0,
            )
            self.status.config(text="✓ Height map ready (H 키로 토글)")
        except Exception as e:
            messagebox.showerror("Height map", str(e))
            self.show_height_var.set(False)
            self._height_rgba = None

    def _ensure_normal_rgba(self):
        if self._normal_rgba is not None:
            return
        if not self.shots or not (0 <= self.idx < len(self.shots)):
            return
        ply = self.shots[self.idx]["bmp"].with_suffix(".ply")
        if not ply.exists():
            messagebox.showerror("Normal map", f".ply not found: {ply}")
            self.show_normal_var.set(False)
            return
        try:
            self.status.config(text="⏳ Normal map 계산 중... (~1초)")
            self.parent.update_idletasks()
            self._normal_rgba = normal_map_rgba(ply, alpha=0.5)
            self.status.config(text="✓ Normal map ready (M 키로 토글)")
        except Exception as e:
            messagebox.showerror("Normal map", str(e))
            self.show_normal_var.set(False)
            self._normal_rgba = None

    def _ensure_heatmap_rgba(self):
        """폴리곤 전체 pickability heatmap을 RGBA로 계산.

        Red(low) → Yellow(0.5) → Green(high). 폴리곤 밖은 투명.
        라벨 변경 시 다시 호출되도록 invalidate는 호출자가 책임.
        """
        if self._heatmap_rgba is not None:
            return
        if not self.shots or not (0 <= self.idx < len(self.shots)):
            return
        ply = self.shots[self.idx]["bmp"].with_suffix(".ply")
        if not ply.exists():
            messagebox.showerror("Heatmap", f".ply not found: {ply}")
            self.show_heatmap_var.set(False)
            return
        try:
            self.status.config(text="⏳ Pickability heatmap 계산 중...")
            self.parent.update_idletasks()
            grid = _read_organized_ply_grid(ply)
            valid = ~np.isnan(grid["z"])
            from label_logic.suction_score import pickability_heatmap

            H, W = valid.shape
            combined = np.full((H, W), np.nan, dtype=np.float32)
            for p in self.polys:
                if not p.get("points"):
                    continue
                m = _picking.polygon_to_mask(p["points"], w=W, h=H)
                if not m.any():
                    continue
                hm = pickability_heatmap(
                    grid, valid, m,
                    cup_radius_px=_picking.HEATMAP_CUP_RADIUS_PX,
                )
                combined = np.where(
                    np.isnan(hm), combined,
                    np.where(np.isnan(combined), hm, np.maximum(combined, hm)),
                )
            self._heatmap_rgba = self._heatmap_score_to_rgba(combined, alpha=0.55)
            self.status.config(text="✓ Heatmap ready (K 키로 토글)")
        except Exception as e:
            messagebox.showerror("Heatmap", str(e))
            self.show_heatmap_var.set(False)
            self._heatmap_rgba = None

    @staticmethod
    def _heatmap_score_to_rgba(score: np.ndarray, alpha: float = 0.55):
        """[0,1] 점수 → RGBA. Red(0)→Yellow(0.5)→Green(1). NaN = 투명."""
        H, W = score.shape
        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        valid = ~np.isnan(score)
        s = np.where(valid, np.clip(score, 0.0, 1.0), 0.0).astype(np.float32)
        # Red→Yellow (0..0.5): R=255, G=2*s*255
        # Yellow→Green (0.5..1): R=2*(1-s)*255, G=255
        r = np.where(s < 0.5, 255, ((1.0 - s) * 2.0 * 255).astype(np.int32))
        g = np.where(s >= 0.5, 255, (s * 2.0 * 255).astype(np.int32))
        rgba[..., 0] = np.clip(r, 0, 255)
        rgba[..., 1] = np.clip(g, 0, 255)
        rgba[..., 2] = 0
        rgba[..., 3] = (valid.astype(np.float32) * alpha * 255).astype(np.uint8)
        return rgba

    # ──────────────────────────────────────────────
    # Box ROI mode + Prelabel
    # ──────────────────────────────────────────────
    def _refresh_box_roi_status(self):
        if self.box_roi is None:
            self.box_roi_status.config(
                text="(미설정 — Watershed 전 그려야 함)", fg="#E74C3C"
            )
        else:
            r = self.box_roi
            self.box_roi_status.config(
                text=f"u[{r['u_min']},{r['u_max']}] v[{r['v_min']},{r['v_max']}]",
                fg="#27AE60",
            )

    def on_box_roi_button(self):
        if self.mode in ("draw", "edit"):
            self._exit_all_modes(commit=True)
        self.mode = "box_roi"
        self._box_roi_start = None
        if self._box_roi_rect_id is not None:
            self.canvas.delete(self._box_roi_rect_id)
            self._box_roi_rect_id = None
        self.canvas.config(cursor="crosshair")
        self.mode_label.config(text="Mode: BOX ROI", fg="#8E44AD")
        self.status.config(text="BOX ROI: 좌상단→우하단 드래그하여 박스 영역 지정")

    def on_click_box_roi(self, event):
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        self._box_roi_start = (cx, cy)
        if self._box_roi_rect_id is not None:
            self.canvas.delete(self._box_roi_rect_id)
        self._box_roi_rect_id = self.canvas.create_rectangle(
            cx, cy, cx, cy, outline="#2ECC71", width=3,
        )

    def on_drag_box_roi(self, event):
        if self._box_roi_start is None or self._box_roi_rect_id is None:
            return
        sx, sy = self._box_roi_start
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        self.canvas.coords(self._box_roi_rect_id, sx, sy, cx, cy)

    def on_release_box_roi(self, event):
        if self._box_roi_start is None:
            return
        sx, sy = self._box_roi_start
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        u_min = int(clamp(min(sx, cx) / self.scale, 0, IMG_W - 1))
        u_max = int(clamp(max(sx, cx) / self.scale, 0, IMG_W - 1))
        v_min = int(clamp(min(sy, cy) / self.scale, 0, IMG_H - 1))
        v_max = int(clamp(max(sy, cy) / self.scale, 0, IMG_H - 1))
        if u_max - u_min < 50 or v_max - v_min < 50:
            messagebox.showwarning("Box ROI", "너무 작은 영역. 다시 그려주세요.")
            self._box_roi_start = None
            return
        try:
            save_box_roi(OUT_DIR, u_min=u_min, u_max=u_max,
                         v_min=v_min, v_max=v_max)
        except Exception as e:
            messagebox.showerror("Box ROI 저장 실패", str(e))
            self._box_roi_start = None
            return
        self.box_roi = {"u_min": u_min, "u_max": u_max,
                        "v_min": v_min, "v_max": v_max}
        self._exit_box_roi_mode()
        self._refresh_box_roi_status()
        self.refresh_canvas()
        self.status.config(
            text=f"✓ Box ROI 저장됨 (meta.yaml). 모든 샷에 자동 적용."
        )

    def _exit_box_roi_mode(self):
        self.mode = "normal"
        self._box_roi_start = None
        if self._box_roi_rect_id is not None:
            self.canvas.delete(self._box_roi_rect_id)
            self._box_roi_rect_id = None
        self.canvas.config(cursor="cross")
        self.mode_label.config(text="Mode: NORMAL", fg="#27AE60")

    # ──────────────────────────────────────────────
    # SAM2 click-to-segment mode
    # ──────────────────────────────────────────────
    def on_sam2_button(self):
        if self.mode == "sam2":
            self._exit_sam2_mode()
            return
        if self.mode in ("draw", "edit", "box_roi"):
            self._exit_all_modes(commit=True)
        self.mode = "sam2"
        self.canvas.config(cursor="crosshair")
        self.mode_label.config(text="Mode: SAM2 CLICK", fg="#E67E22")
        if not sam2_predictor.is_loaded():
            self.status.config(
                text="SAM2 CLICK: 객체 위 클릭 → 외곽선 자동. "
                     "(첫 클릭은 모델 로드 ~2초, 이후 빠름)"
            )
        else:
            self.status.config(
                text="SAM2 CLICK: 객체 위 클릭 → 외곽선 자동. ESC로 종료."
            )

    def _exit_sam2_mode(self):
        self.mode = "normal"
        self.canvas.config(cursor="cross")
        self.mode_label.config(text="Mode: NORMAL", fg=COLOR_OK)
        self.status.config(text="SAM2 모드 종료.")

    def on_click_sam2(self, event):
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        u = float(cx) / max(self.scale, 1e-6)
        v = float(cy) / max(self.scale, 1e-6)
        if u < 0 or u >= IMG_W or v < 0 or v >= IMG_H:
            return
        if not self.shots or not (0 <= self.idx < len(self.shots)):
            return
        bmp = self.shots[self.idx]["bmp"]
        # Disable while inferring (prevent rapid double-click)
        self.canvas.config(cursor="watch")
        self.status.config(text=f"⏳ SAM2 추론 중... (u={int(u)}, v={int(v)})")
        self.parent.update_idletasks()

        def worker():
            try:
                poly = sam2_predictor.predict_polygon(bmp, (u, v))
                self.parent.after(0, lambda: self._on_sam2_done(poly, u, v))
            except Exception as e:
                err = str(e)
                self.parent.after(0, lambda: self._on_sam2_error(err))

        threading.Thread(target=worker, daemon=True).start()

    def _on_sam2_done(self, poly, u, v):
        self.canvas.config(cursor="crosshair")
        if not poly:
            self.status.config(
                text=f"SAM2: 마스크 없음 ({int(u)},{int(v)}). 다른 점 시도."
            )
            return
        new_poly = {
            "points": [list(p) for p in poly],
            "class_id": -1,
            "verified": True,    # SAM2 + user click → treat as verified outline
            "canvas_id": None,
            "label_id": None,
        }
        self.polys.append(new_poly)
        self.undo_stack.append(("add", len(self.polys) - 1))
        self.selected = len(self.polys) - 1
        self.refresh_canvas()
        self.status.config(
            text=f"SAM2: ✓ 폴리곤 #{self.selected} ({len(poly)} pts). 0~4 클래스 부여."
        )

    def _on_sam2_error(self, err_msg):
        self.canvas.config(cursor="crosshair")
        messagebox.showerror("SAM2 error", err_msg)
        self.status.config(text=f"✗ SAM2 error: {err_msg}")

    def on_compute_ood(self):
        """현재 샷 폴리곤의 Supersimplenet anomaly score 계산 (subprocess)."""
        if self._ood_proc_running:
            messagebox.showinfo("OOD", "이미 진행 중입니다.")
            return
        if not self.polys:
            messagebox.showinfo("OOD", "폴리곤이 없습니다. 먼저 prelabel 또는 W로 그리세요.")
            return
        if not ANOMALY_ENV_PYTHON.exists():
            messagebox.showerror(
                "OOD",
                f"anomaly env python 못 찾음:\n{ANOMALY_ENV_PYTHON}\n"
                "conda env 'anomaly' 가 만들어졌는지 확인.",
            )
            return
        if not OOD_SCORE_SCRIPT.exists():
            messagebox.showerror("OOD", f"score_polygons.py 없음:\n{OOD_SCORE_SCRIPT}")
            return
        # 현재 shot 정보
        shot = self.shots[self.idx]
        bmp_path = Path(shot["bmp"])
        if not bmp_path.exists():
            messagebox.showerror("OOD", f"BMP 없음: {bmp_path}")
            return
        # 폴리곤 list → 임시 JSON
        import tempfile
        tmp_dir = Path(tempfile.gettempdir()) / "label_tool_ood"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        polys_path  = tmp_dir / f"{shot['name']}_polys.json"
        scores_path = tmp_dir / f"{shot['name']}_scores.json"
        payload = []
        for i, p in enumerate(self.polys):
            pts = p.get("points") or []
            if len(pts) < 3:
                continue
            payload.append({"idx": i, "points": [list(pt) for pt in pts]})
        if not payload:
            messagebox.showinfo("OOD", "유효한 폴리곤 (≥3 vertex) 없음.")
            return
        polys_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        self._ood_proc_running = True
        self.ood_btn.configure(state="disabled")
        self.ood_status.config(text=f"⏳ OOD 점수 계산 중 ({len(payload)} 폴리곤, 최대 30초)...")
        self.parent.update_idletasks()

        import threading, subprocess
        def worker():
            try:
                cmd = [
                    str(ANOMALY_ENV_PYTHON), str(OOD_SCORE_SCRIPT),
                    "--bmp",   str(bmp_path),
                    "--polys", str(polys_path),
                    "--out",   str(scores_path),
                ]
                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"
                res = subprocess.run(
                    cmd, capture_output=True, text=True, env=env,
                    encoding="utf-8", errors="replace",
                    creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
                )
                if res.returncode != 0:
                    err = (res.stderr or res.stdout or "")[-600:]
                    self.parent.after(0, lambda: self._on_compute_ood_error(err))
                    return
                result = json.loads(scores_path.read_text(encoding="utf-8"))
                self.parent.after(0, lambda: self._on_compute_ood_done(result))
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                self.parent.after(0, lambda: self._on_compute_ood_error(msg))

        threading.Thread(target=worker, daemon=True).start()

    def _on_compute_ood_done(self, result):
        self._ood_proc_running = False
        self.ood_btn.configure(state="normal")
        self.ood_threshold = float(result.get("threshold", 0.0))
        self.ood_normal_stats = result.get("normal_stats", {})
        scored = result.get("scores", [])
        applied = 0
        for entry in scored:
            i = entry.get("idx", -1)
            s = entry.get("score")
            if 0 <= i < len(self.polys) and s is not None:
                self.polys[i]["anomaly_score"] = float(s)
                applied += 1
        # 자동으로 overlay ON
        self.ood_visible = True
        n_high = sum(1 for p in self.polys
                      if p.get("anomaly_score") is not None
                      and p["anomaly_score"] > self.ood_threshold)
        self.ood_status.config(
            text=f"✓ {applied} 점수, thr={self.ood_threshold:.3f}, >thr: {n_high}",
        )
        self.refresh_canvas()

    def _on_compute_ood_error(self, err_msg):
        self._ood_proc_running = False
        self.ood_btn.configure(state="normal")
        self.ood_status.config(text=f"✗ {err_msg[:80]}")
        messagebox.showerror("OOD compute error", err_msg)

    def on_prelabel(self):
        source = self.prelabel_source_var.get()
        if source.startswith("Auto-Classify"):
            self._run_auto_classify()
            return
        # Watershed mode (existing flow)
        if self.box_roi is None:
            messagebox.showwarning(
                "Prelabel", "박스 ROI 먼저 그려주세요 (📐 버튼)."
            )
            return
        if self.polys:
            if not messagebox.askyesno(
                "기존 라벨 있음",
                "현재 폴리곤을 모두 지우고 watershed prelabel로 덮어쓸까요?\n"
                "(취소하면 prelabel 진행 안 함)"
            ):
                return
        self.prelabel_status.config(text="⏳ Watershed 중... (최대 15초)")
        self.prelabel_btn.configure(state="disabled")
        self.parent.update_idletasks()

        ply = self.shots[self.idx]["bmp"].with_suffix(".ply")
        box_roi = dict(self.box_roi)

        def worker():
            try:
                polys = run_watershed(ply, box_roi=box_roi)
                self.parent.after(0, lambda: self._on_prelabel_done(polys))
            except Exception as e:
                err_msg = str(e)
                self.parent.after(
                    0, lambda: self._on_prelabel_error(err_msg)
                )

        threading.Thread(target=worker, daemon=True).start()

    def _run_auto_classify(self):
        """Use trained YOLO model to assign class to existing user polygons."""
        if not self.polys:
            messagebox.showwarning(
                "Auto-Classify",
                "사용자 폴리곤이 없습니다.\nW로 외곽선을 먼저 그려주세요.",
            )
            return
        model_name = self.model_var.get()
        if not model_name:
            messagebox.showerror("Auto-Classify", "모델이 선택되지 않았습니다.")
            return
        model_path = ROOT_DIR / "runs" / "segment" / model_name / "weights" / "best.pt"
        if not model_path.exists():
            messagebox.showerror("Auto-Classify", f"best.pt not found:\n{model_path}")
            return

        if not self.shots or not (0 <= self.idx < len(self.shots)):
            return
        bmp = self.shots[self.idx]["bmp"]
        # Snapshot user polygons for matching
        user_polys = [list(p["points"]) for p in self.polys]

        self.prelabel_status.config(
            text="⏳ Auto-Classify 중... (모델 추론 + IoU 매칭)"
        )
        self.prelabel_btn.configure(state="disabled")
        self.parent.update_idletasks()

        def worker():
            try:
                detections = _auto_classify.run_model_inference(
                    bmp, str(model_path),
                )
                assignments = _auto_classify.assign_classes(
                    user_polys, detections, iou_thr=0.3,
                )
                self.parent.after(
                    0, lambda: self._on_auto_classify_done(assignments),
                )
            except Exception as e:
                err = str(e)
                self.parent.after(
                    0, lambda: self._on_auto_classify_error(err),
                )

        threading.Thread(target=worker, daemon=True).start()

    def _on_auto_classify_done(self, assignments):
        """Assign class_id from model match to each user polygon."""
        n_set = 0
        n_unmatched = 0
        for i, (cid, conf, iou) in enumerate(assignments):
            if i >= len(self.polys):
                break
            if 0 <= cid < len(CLASSES):
                self.polys[i]["class_id"] = cid
                # Don't auto-mark verified — user should glance to confirm
                n_set += 1
            else:
                n_unmatched += 1
        self.refresh_canvas()
        self.prelabel_status.config(
            text=f"✓ Auto-Classify: {n_set} 부여, {n_unmatched} 미매칭"
        )
        self.prelabel_btn.configure(state="normal")
        if n_unmatched > 0:
            self.status.config(
                text=f"미매칭 {n_unmatched}개는 회색 (·) — 0~4 키로 직접 부여"
            )

    def _on_auto_classify_error(self, err_msg):
        messagebox.showerror("Auto-Classify error", err_msg)
        self.prelabel_status.config(text=f"✗ {err_msg}")
        self.prelabel_btn.configure(state="normal")

    def open_inference_viewer(self):
        existing = getattr(self, "_infer_viewer_win", None)
        if existing is not None and existing.winfo_exists():
            existing.deiconify()
            existing.lift()
            existing.focus_force()
            return
        win = tk.Toplevel(self.parent)
        win.title("YOLO Inference Viewer")
        win.configure(bg=COLOR_BG_MAIN)
        try:
            InferenceViewer(win)
        except Exception as e:
            win.destroy()
            messagebox.showerror("Inference Viewer", f"실패: {e}")
            return
        self._infer_viewer_win = win

    def open_roboflow_review(self):
        existing = getattr(self, "_rf_review_win", None)
        if existing is not None and existing.winfo_exists():
            existing.deiconify()
            existing.lift()
            existing.focus_force()
            return
        rf_dir = ROOT_DIR / "_external" / "roboflow_260508" / "train" / "labels"
        if not rf_dir.exists():
            messagebox.showerror(
                "Roboflow 라벨 비교",
                f"Roboflow 라벨 폴더 없음:\n{rf_dir}\n\n"
                "ZIP을 _external/roboflow_260508/ 로 풀어주세요."
            )
            return
        win = tk.Toplevel(self.parent)
        win.title("📋 Roboflow 라벨 비교")
        win.configure(bg=COLOR_BG_MAIN)
        try:
            RoboflowReviewViewer(win)
        except Exception as e:
            win.destroy()
            messagebox.showerror("Roboflow 라벨 비교", f"실패: {e}")
            return
        self._rf_review_win = win

    def _on_prelabel_done(self, polys):
        # Delete existing polygon canvas items BEFORE clearing self.polys
        # (otherwise we lose canvas_id references and items remain on canvas)
        for p in self.polys:
            if p["canvas_id"] is not None:
                self.canvas.delete(p["canvas_id"])
            if p["label_id"] is not None:
                self.canvas.delete(p["label_id"])
        self.polys = []
        for poly_pts in polys:
            self.polys.append({
                "points": [list(p) for p in poly_pts],
                "class_id": -1,
                "verified": False,
                "canvas_id": None,
                "label_id": None,
            })
        self.selected = 0 if self.polys else -1
        self.advance_to_unlabeled()
        self.refresh_canvas()
        self.prelabel_status.config(text=f"✓ {len(polys)} 폴리곤 prelabel됨")
        self.prelabel_btn.configure(state="normal")

    def _on_prelabel_error(self, err_msg):
        messagebox.showerror("Prelabel error", err_msg)
        self.prelabel_status.config(text=f"✗ {err_msg}")
        self.prelabel_btn.configure(state="normal")

    def _on_theme_change(self, _event):
        """User selected new theme — save to settings + auto-restart tool."""
        chosen_name = self.theme_var.get()
        chosen_key = self._theme_name_to_key.get(chosen_name)
        if not chosen_key or chosen_key == ACTIVE_THEME_NAME:
            return
        # Save current shot before restart
        if self.shots and 0 <= self.idx < len(self.shots):
            try:
                self.save_current()
            except Exception:
                pass
        # Persist theme choice
        try:
            existing = {}
            if SETTINGS_PATH.exists():
                existing = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            existing["theme"] = chosen_key
            SETTINGS_PATH.write_text(
                json.dumps(existing, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            messagebox.showerror("Theme save error", str(e))
            return
        # Auto-restart with same Python interpreter + script
        try:
            import subprocess
            cmd = [sys.executable] + sys.argv
            subprocess.Popen(cmd)
            self.parent.destroy()
        except Exception as e:
            messagebox.showerror("Restart error", str(e))

    def _on_dataset_change(self, _event):
        name = self.dataset_var.get()
        if name not in self._dataset_map:
            return
        raw, lab = self._dataset_map[name]
        if not raw.exists():
            messagebox.showerror("Error", f"raw folder not found: {raw}")
            return
        # Save current shot before switching
        if self.shots and 0 <= self.idx < len(self.shots):
            try:
                self.save_current()
            except Exception:
                pass
        set_active_dataset(raw, lab)
        self.reload()

    def _toggle_side_panel(self):
        """Hide/show the side panel; toggle bar (▶/◀) always visible."""
        if self._side_visible:
            self.side.pack_forget()
            self.toggle_btn.config(text="▶")
            self._side_visible = False
        else:
            self.side.pack(side="right", fill="y", before=self.toggle_bar)
            self.toggle_btn.config(text="◀")
            self._side_visible = True
        # canvas autofits via on-window-resize handler
        self._refit_canvas_to_window()

    def _toggle_left_panel_section(self, _event=None):
        """Collapse/expand the QUICK KEYS cheat content (header stays visible)."""
        if self._cheat_visible:
            self.cheat_widget.pack_forget()
            self.left_header.config(text="QUICK KEYS  ▶")
            self._cheat_visible = False
        else:
            self.cheat_widget.pack(fill="x", anchor="w")
            self.left_header.config(text="QUICK KEYS  ▼")
            self._cheat_visible = True

    def toggle_left_panel(self):
        """Hide/show entire LEFT panel (QUICK KEYS).

        When hidden, the canvas expands into the freed space.
        Re-open via header click (no — header is hidden too) or Ctrl+L.
        """
        if self._left_visible:
            self.left_panel.pack_forget()
            self._left_visible = False
            self.status.config(
                text="QUICK KEYS 패널 숨김 — Ctrl+L로 다시 보기"
            )
        else:
            # Re-pack: side="right" before the sidebar (DATASET ~ 3D STATS)
            self.left_panel.pack(side="right", fill="y", before=self.side)
            self._left_visible = True
            self.status.config(text="QUICK KEYS 패널 표시")
        self._refit_canvas_to_window()

    def _refit_canvas_to_window(self):
        """Recompute base_scale from current window size + panel visibility.

        Also auto-hides LEFT panel when window is too narrow.
        """
        try:
            ww = self.parent.winfo_width()
            wh = self.parent.winfo_height()
        except Exception:
            return
        if ww < 200 or wh < 200:
            return

        # Auto-hide left panel when window is narrow
        # Only auto-toggle if it differs from current visibility
        should_show_left = ww >= LEFT_AUTO_HIDE_BELOW
        if should_show_left and not self._left_visible:
            self.left_panel.pack(side="right", fill="y", before=self.side)
            self._left_visible = True
        elif not should_show_left and self._left_visible:
            self.left_panel.pack_forget()
            self._left_visible = False

        side_w = TOGGLE_BAR_WIDTH + (SIDE_WIDTH if self._side_visible else 0)
        left_w = LEFT_PANEL_WIDTH if self._left_visible else 0
        avail_w = max(200, ww - side_w - left_w - 30)
        avail_h = max(200, wh - 80)
        new_base = min(avail_w / IMG_W, avail_h / IMG_H, 1.0)
        if abs(new_base - self.base_scale) < 0.005:
            return
        self.base_scale = new_base
        self.scale = self.base_scale * self.zoom
        new_w = max(1, int(IMG_W * self.scale))
        new_h = max(1, int(IMG_H * self.scale))
        self.canvas.config(width=new_w, height=new_h)
        self.viewport_w = int(IMG_W * self.base_scale)
        self.viewport_h = int(IMG_H * self.base_scale)
        if self.original_image is not None:
            self.redraw_image()
            self.refresh_canvas()

    def reload(self):
        """Re-scan shots from current global DATA_DIR/OUT_DIR and load first shot.

        Called after set_active_dataset() switches the active dataset.
        """
        self.shots = self.scan_shots()
        self.idx = 0
        self.polys = []
        self.selected = -1
        self.undo_stack = []
        if not self.shots:
            messagebox.showwarning(
                "Empty dataset", f"No data in {DATA_DIR}",
            )
            return
        self.load_shot()
        self.refresh_progress()
        if hasattr(self, "shot_label"):
            self.shot_label.config(text=self.shots[0]["name"])

    def compute_dataset_progress(self):
        """Walk all shots, count done/partial/untouched.

        - done (✓): all polygons assigned a class AND verified
        - partial (△): at least one polygon has class assigned but not all/not verified
        - untouched (·): no polygons or no class assigned

        Returns (done, partial, total).
        """
        if not self.shots:
            return 0, 0, 0
        done = partial = 0
        for shot in self.shots:
            data_json_path = OUT_DIR / "data_json" / f"{shot['name']}_data.json"
            if not data_json_path.exists():
                data_json_path = shot["data"]
            try:
                with open(data_json_path, encoding="utf-8") as f:
                    d = json.load(f)
            except Exception:
                continue
            cids = d.get("class_id", [])
            verified = d.get("verified", [])
            if not cids:
                continue
            n_assigned = sum(1 for c in cids
                             if isinstance(c, int) and 0 <= c < len(CLASSES))
            if n_assigned == 0:
                continue
            all_assigned = (n_assigned == len(cids))
            all_verified = all(
                verified[i] if i < len(verified) else False
                for i in range(len(cids))
            ) if verified else False
            if all_assigned and all_verified:
                done += 1
            else:
                partial += 1
        return done, partial, len(self.shots)

    def _compute_dashboard_stats(self):
        """폴리곤 / GT pick / PICKABLE 통계 — dashboard용.

        Returns dict:
          n_polys_total — 라벨된 모든 샷의 폴리곤 합 (클래스 부여 무관)
          n_polys_assigned — 그 중 클래스 0~4 부여된 폴리곤 수
          n_gt — manual_picks/ 안 GT 픽 수
          n_pickable — outputs/picks/ 안 status=PICKABLE/MANUAL 수
          n_processed — picks JSON 안 모든 폴리곤 수 (계산 대상)
        """
        # 라벨된 폴리곤 합 (전체 + 클래스 부여)
        labeled_dir = OUT_DIR / "data_json"
        n_polys_total = 0
        n_polys_assigned = 0
        if labeled_dir.exists():
            for f in labeled_dir.glob("*_data.json"):
                try:
                    with open(f, encoding="utf-8") as fh:
                        d = json.load(fh)
                    n_polys_total += len(d.get("result_data", []))
                    cids = d.get("class_id", [])
                    n_polys_assigned += sum(
                        1 for c in cids
                        if isinstance(c, int) and 0 <= c < len(CLASSES)
                    )
                except Exception:
                    pass
        # GT pick (manual_picks/)
        mp_dir = DATA_DIR.parent / "manual_picks"
        n_gt = 0
        if mp_dir.exists():
            for f in mp_dir.glob("*_manual_picks.json"):
                try:
                    with open(f, encoding="utf-8") as fh:
                        n_gt += len(json.load(fh) or {})
                except Exception:
                    pass
        # PICKABLE / processed (outputs/picks/)
        team_name = OUT_DIR.parent.name
        picks_dir = ROOT_DIR / "outputs" / team_name / "picks"
        n_pickable = 0
        n_processed = 0
        if picks_dir.exists():
            for f in picks_dir.glob("*_picks.json"):
                try:
                    with open(f, encoding="utf-8") as fh:
                        d = json.load(fh)
                    for r in d.get("picks", []):
                        n_processed += 1
                        if r.get("status") in ("PICKABLE", "MANUAL"):
                            n_pickable += 1
                except Exception:
                    pass
        return {"n_polys_total": n_polys_total,
                "n_polys_assigned": n_polys_assigned,
                "n_gt": n_gt,
                "n_pickable": n_pickable, "n_processed": n_processed}

    def refresh_progress(self):
        """Update the progress color bar widget + 상세 dashboard."""
        if not hasattr(self, "progress_canvas"):
            return
        done, partial, total = self.compute_dataset_progress()
        c = self.progress_canvas
        c.delete("all")
        w = int(c["width"])
        h = int(c["height"])
        if total == 0:
            return
        done_w = w * done / total
        partial_w = w * partial / total
        c.create_rectangle(0, 0, done_w, h, fill="#27AE60", width=0)
        c.create_rectangle(done_w, 0, done_w + partial_w, h, fill="#F1C40F", width=0)
        c.create_rectangle(done_w + partial_w, 0, w, h, fill="#BDC3C7", width=0)
        if hasattr(self, "progress_text"):
            self.progress_text.config(
                text=f"{done + partial}/{total}  (✓ {done}  △ {partial}  · {total - done - partial})"
            )
        # ── 상세 dashboard ────────────────────────────────────────
        # 위 progress bar는 "샷 단위" (done/partial/total). 여기는 "폴리곤 단위".
        if not hasattr(self, "dash_polys_label"):
            return
        s = self._compute_dashboard_stats()
        n_t = s["n_polys_total"]
        n_a = s["n_polys_assigned"]
        cls_pct = (n_a / n_t * 100) if n_t else 0
        self.dash_polys_label.config(
            text=f"폴리곤(클래스 부여): {n_a}/{n_t} ({cls_pct:.0f}%)"
        )
        gt_pct = (s["n_gt"] / n_a * 100) if n_a else 0
        self.dash_gt_label.config(
            text=f"GT pick: {s['n_gt']}/{n_a} ({gt_pct:.1f}%)"
        )
        pk_pct = (s["n_pickable"] / s["n_processed"] * 100) if s["n_processed"] else 0
        self.dash_pick_label.config(
            text=f"PICKABLE: {s['n_pickable']}/{s['n_processed']} "
                 f"({pk_pct:.0f}%) [pick 계산 후]"
        )
        if hasattr(self, "dash_split_label"):
            split_stats = self._compute_split_distribution()
            self.dash_split_label.config(
                text=f"YOLO: train {split_stats['train']} / val {split_stats['val']} "
                     f"/ skip {split_stats['empty']}  "
                     f"(수동 {split_stats['manual']})"
            )

    def build_ui(self):
        sw = self.parent.winfo_screenwidth()
        sh = self.parent.winfo_screenheight()
        # Canvas should fill space minus side panel + small margin
        max_w = sw - SIDE_WIDTH - 30
        max_h = sh - 130   # small margin top/bottom for status bar
        self.base_scale = min(max_w / IMG_W, max_h / IMG_H, 1.0)
        self.scale = self.base_scale * self.zoom
        self.viewport_w = int(IMG_W * self.base_scale)
        self.viewport_h = int(IMG_H * self.base_scale)

        # Industrial small fonts — info density ↑
        ko_font = ("Malgun Gothic", 9)
        ko_bold = ("Malgun Gothic", 10, "bold")
        ko_big = ("Malgun Gothic", 12, "bold")
        ko_mono = ("Consolas", 9)

        # Helpers for consistent dark widgets
        def _frame(parent, **kw):
            return tk.Frame(parent, bg=COLOR_BG_PANEL, **kw)

        def _label(parent, text="", bold=False, header=False, muted=False,
                   font=None, **kw):
            f = font or (ko_bold if (bold or header) else ko_font)
            fg = COLOR_TEXT_HEADER if header else (
                COLOR_TEXT_MUTED if muted else COLOR_TEXT
            )
            return tk.Label(parent, text=text, bg=COLOR_BG_PANEL,
                             fg=fg, font=f, **kw)

        def _separator(parent):
            return tk.Frame(parent, bg=COLOR_BORDER, height=1)

        # ── LEFT panel (QUICK KEYS) — built but NOT packed yet.
        #    Will be packed later with before=side so it sits LEFT of side panel.
        #    Header click → hide entire panel (canvas takes the space).
        #    Ctrl+L re-opens. Auto-hides when window narrow.
        left_panel = tk.Frame(
            self.parent, padx=8, bg=COLOR_BG_PANEL, width=LEFT_PANEL_WIDTH,
        )
        left_panel.pack_propagate(False)
        self.left_panel = left_panel
        self._left_visible = True

        self.left_header = tk.Label(
            left_panel, text="QUICK KEYS  ▼  (클릭하여 닫기)", font=ko_bold,
            bg=COLOR_BG_PANEL, fg=COLOR_TEXT_HEADER, anchor="w",
            cursor="hand2",
        )
        self.left_header.pack(fill="x", pady=(4, 1))
        # Click header → hide whole panel (canvas expands)
        self.left_header.bind("<Button-1>", lambda _e: self.toggle_left_panel())
        _separator(left_panel).pack(fill="x", pady=(0, 4))
        cheat_text = (
            "■ 라벨\n"
            "0~4   클래스 부여\n"
            "W     새 폴리곤\n"
            "E     꼭지점 수정\n"
            "D     삭제\n"
            "Tab   다음 미라벨\n"
            "\n"
            "■ 모드 (캔버스에서)\n"
            "A     SAM2 Click\n"
            "B     박스 ROI\n"
            "V     3D View 별창\n"
            "G     픽 계산 (3D)\n"
            "\n"
            "■ 작업\n"
            "Ctrl+R  Prelabel\n"
            "Y       YOLO split 토글\n"
            "S       저장 / U  Undo\n"
            "N / P   다음/이전 샷\n"
            "\n"
            "■ 별창\n"
            "Ctrl+I         Inference Viewer\n"
            "Ctrl+Shift+R   Roboflow 비교\n"
            "\n"
            "■ UI / Help\n"
            "Ctrl+B  사이드 토글\n"
            "Ctrl+L  QUICK KEYS 토글\n"
            "+/-/F   줌 / Fit\n"
            "F1      전체 단축키\n"
            "Q       종료"
        )
        self.cheat_widget = tk.Label(
            left_panel, text=cheat_text, justify="left", anchor="w",
            font=ko_mono, bg=COLOR_BG_PANEL, fg=COLOR_TEXT_MUTED,
        )
        self.cheat_widget.pack(fill="x", anchor="w")
        self._cheat_visible = True

        # ── 3D STATS section in LEFT panel (selected polygon, for review) ──
        _label(left_panel, text="3D STATS", header=True, anchor="w").pack(
            fill="x", pady=(10, 1),
        )
        _separator(left_panel).pack(fill="x", pady=(0, 4))
        self.stats_text = tk.Label(
            left_panel,
            text="(폴리곤 선택 시 표시)",
            font=ko_mono, justify="left", anchor="w",
            bg=COLOR_BG_PANEL, fg=COLOR_TEXT_MUTED,
        )
        self.stats_text.pack(fill="x", anchor="w")

        # LAYERS 체크박스 GUI 제거됨 (사용자 요청). BooleanVar는 유지 — H/M/K
        # 단축키 (on_key)가 이 변수를 토글해서 캔버스 오버레이 작동.
        self.show_height_var = tk.BooleanVar(value=False)
        self.show_normal_var = tk.BooleanVar(value=False)
        self.show_heatmap_var = tk.BooleanVar(value=False)
        # 3D viewer (separate window via Open3D)
        if _HAS_CTK:
            self.view3d_btn = ctk.CTkButton(
                left_panel, text="🌀 3D View  (V)",
                command=self.on_open_3d_view, height=30,
            )
        else:
            self.view3d_btn = tk.Button(
                left_panel, text="🌀 3D View  (V)",
                command=self.on_open_3d_view, font=ko_font,
            )
        self.view3d_btn.pack(fill="x", pady=2)
        ToolTip(
            self.view3d_btn,
            "Open3D 별도 윈도우 — 인터랙티브 3D.\n"
            "조작: 좌클릭 회전 / WASD 팬 / 휠 또는 IK 줌 / R 리셋\n"
            "T로 라벨 색 토글, Q/ESC 닫기.",
        )

        # ── 듀얼 흡착 픽포인트 ──
        if _HAS_CTK:
            self.compute_picks_btn = ctk.CTkButton(
                left_panel, text="🤖 Picks 계산  (G)",
                command=self.on_compute_picks, height=30,
            )
        else:
            self.compute_picks_btn = tk.Button(
                left_panel, text="🤖 Picks 계산  (G)",
                command=self.on_compute_picks, font=ko_font,
            )
        self.compute_picks_btn.pack(fill="x", pady=2)
        ToolTip(
            self.compute_picks_btn,
            "현재 샷의 라벨된 객체에 듀얼 흡착 픽포인트 산출.\n"
            "결과는 캔버스에 직사각형(75×25mm) + 컵 센터 2개로 표시.\n"
            "3D View에서도 같이 보이고, 외부 JSON으로도 저장.",
        )
        self.show_picks_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            left_panel, text="픽 직사각형 표시", variable=self.show_picks_var,
            command=self.refresh_canvas, font=ko_font,
            bg=COLOR_BG_PANEL, fg=COLOR_TEXT, activebackground=COLOR_BG_PANEL,
            activeforeground=COLOR_ACCENT, selectcolor=COLOR_BG_INPUT,
        ).pack(anchor="w")
        # ── Picking thresholds 편집 다이얼로그 ──
        if _HAS_CTK:
            self.thresholds_btn = ctk.CTkButton(
                left_panel, text="🎚 Picking Thresholds…",
                command=self.on_open_thresholds_dialog, height=28,
            )
        else:
            self.thresholds_btn = tk.Button(
                left_panel, text="🎚 Picking Thresholds…",
                command=self.on_open_thresholds_dialog, font=ko_font,
            )
        self.thresholds_btn.pack(fill="x", pady=(2, 0))
        ToolTip(
            self.thresholds_btn,
            "임계값 / 상수 편집 (config/default.yaml).\n"
            "Save 누르면 YAML 갱신 + 픽 재계산.",
        )

        # ZOOM 섹션 GUI 제거됨 (사용자 요청). 줌은 단축키로:
        #   Ctrl+휠/+/- = 확대/축소, F = Fit. update_zoom_label은 zoom_label
        #   속성 없으면 자동 skip하므로 안전.

        # ── 현재 이미지 완전 삭제 (안전장치 2단계 확인) ──────────────
        _separator(left_panel).pack(fill="x", pady=(10, 4))
        if _HAS_CTK:
            self.delete_shot_btn = ctk.CTkButton(
                left_panel, text="🗑 현재 이미지 완전 삭제",
                command=self.on_delete_current_shot,
                fg_color="#C0392B", hover_color="#A93226",
                height=30,
            )
        else:
            self.delete_shot_btn = tk.Button(
                left_panel, text="🗑 현재 이미지 완전 삭제",
                command=self.on_delete_current_shot,
                bg="#C0392B", fg="white", activebackground="#A93226",
                font=ko_font, relief="flat",
            )
        self.delete_shot_btn.pack(fill="x", pady=(0, 4))
        ToolTip(
            self.delete_shot_btn,
            "현재 샷의 모든 연결 파일을 영구 삭제\n"
            "(raw PLY/BMP/data/info + labeled + manual_picks + outputs)\n"
            "키 바인딩 없음 — 실수 방지를 위해 버튼만\n"
            "두 번 확인 후 진행됨",
        )

        canvas_container = tk.Frame(self.parent, bg=COLOR_BG_MAIN)
        canvas_container.pack(side="left", padx=2, pady=2, fill="both", expand=True)
        self.canvas_container = canvas_container
        self.canvas = tk.Canvas(
            canvas_container, width=self.viewport_w, height=self.viewport_h,
            bg=COLOR_BG_MAIN, cursor="cross", highlightthickness=0,
        )
        vbar = tk.Scrollbar(canvas_container, orient="vertical", command=self.canvas.yview)
        hbar = tk.Scrollbar(canvas_container, orient="horizontal", command=self.canvas.xview)
        self.canvas.config(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="ew")
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Motion>", self.on_motion_status, add="+")
        self.canvas.bind("<Motion>", self.on_motion)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<Double-Button-1>", self.on_double_click)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind(
            "<Shift-MouseWheel>",
            lambda e: self.canvas.xview_scroll(
                -1 if e.delta > 0 else 1, "units"
            ),
        )
        self.canvas.bind("<Button-2>", lambda e: self.canvas.scan_mark(e.x, e.y))
        self.canvas.bind(
            "<B2-Motion>",
            lambda e: self.canvas.scan_dragto(e.x, e.y, gain=1),
        )
        self.canvas.bind("<Up>", lambda e: self.pan_pixels(0, -50))
        self.canvas.bind("<Down>", lambda e: self.pan_pixels(0, 50))
        self.canvas.bind("<Left>", lambda e: self.pan_pixels(-50, 0))
        self.canvas.bind("<Right>", lambda e: self.pan_pixels(50, 0))

        # Toggle bar (always visible, even when side panel is hidden)
        toggle_bar = tk.Frame(
            self.parent, bg=COLOR_BG_PANEL, width=TOGGLE_BAR_WIDTH,
        )
        toggle_bar.pack(side="right", fill="y")
        toggle_bar.pack_propagate(False)
        self.toggle_bar = toggle_bar
        self.toggle_btn = tk.Button(
            toggle_bar, text="◀", command=self._toggle_side_panel,
            bg=COLOR_BG_PANEL, fg=COLOR_TEXT, activebackground=COLOR_ACCENT,
            activeforeground="#000", font=("Arial", 9, "bold"),
            relief="flat", borderwidth=0, highlightthickness=0,
        )
        self.toggle_btn.pack(fill="x", pady=2)

        # ── 스크롤 가능 side 패널 ─────────────────────────────────────
        # outer = pack_forget 토글용 + 고정 너비. inner = 실제 위젯 컨테이너
        # (canvas의 window). side 변수는 inner를 가리켜서 기존 위젯 코드 그대로.
        side_outer = tk.Frame(
            self.parent, bg=COLOR_BG_PANEL, width=SIDE_WIDTH,
        )
        side_outer.pack(side="right", fill="y")
        side_outer.pack_propagate(False)
        self.side = side_outer   # 외부 토글용
        self._side_visible = True

        side_canvas = tk.Canvas(
            side_outer, bg=COLOR_BG_PANEL, highlightthickness=0,
            borderwidth=0,
        )
        side_sb = tk.Scrollbar(
            side_outer, orient="vertical", command=side_canvas.yview,
        )
        side_canvas.configure(yscrollcommand=side_sb.set)
        side_sb.pack(side="right", fill="y")
        side_canvas.pack(side="left", fill="both", expand=True)

        side = tk.Frame(side_canvas, padx=8, bg=COLOR_BG_PANEL)
        side_win = side_canvas.create_window((0, 0), window=side, anchor="nw")
        # inner frame이 outer 너비에 맞춰 늘어나도록
        def _on_canvas_resize(e):
            side_canvas.itemconfig(side_win, width=e.width)
        side_canvas.bind("<Configure>", _on_canvas_resize)
        # 스크롤 영역을 inner frame 크기에 동기
        side.bind(
            "<Configure>",
            lambda e: side_canvas.configure(scrollregion=side_canvas.bbox("all")),
        )
        # 마우스 휠 — 호버 시에만 활성 (캔버스의 zoom 휠과 충돌 회피)
        def _on_side_wheel(e):
            side_canvas.yview_scroll(-int(e.delta / 120), "units")
        self._side_canvas_for_wheel = side_canvas
        self._side_wheel_handler = _on_side_wheel

        # ── THEME picker (very top of side panel) ──
        _label(side, text="THEME", header=True, anchor="w").pack(fill="x", pady=(4, 1))
        _separator(side).pack(fill="x", pady=(0, 4))
        self.theme_var = tk.StringVar(value=THEMES[ACTIVE_THEME_NAME]["name"])
        theme_names = [THEMES[k]["name"] for k in THEMES]
        self._theme_name_to_key = {THEMES[k]["name"]: k for k in THEMES}
        theme_dropdown = ttk.Combobox(
            side, textvariable=self.theme_var, values=theme_names,
            state="readonly", width=34, font=ko_font,
        )
        theme_dropdown.pack(fill="x", pady=2)
        theme_dropdown.bind("<<ComboboxSelected>>", self._on_theme_change)

        # ── Top header row: small toggle button + Dataset header ──
        ds_header_row = _frame(side)
        ds_header_row.pack(fill="x", pady=(8, 1))
        # Small toggle button — re-opens QUICK KEYS panel when hidden
        self.show_left_btn = tk.Button(
            ds_header_row, text="📋", width=3,
            command=self.toggle_left_panel,
            bg=COLOR_BG_INPUT, fg=COLOR_TEXT,
            activebackground=COLOR_ACCENT, activeforeground="#000",
            font=("Arial", 9), relief="flat", borderwidth=0, highlightthickness=0,
        )
        self.show_left_btn.pack(side="left", padx=(0, 4))
        _label(ds_header_row, text="DATASET", header=True, anchor="w").pack(side="left", fill="x", expand=True)
        _separator(side).pack(fill="x", pady=(0, 4))
        ds_frame = _frame(side)
        ds_frame.pack(fill="x", pady=1)
        self.dataset_var = tk.StringVar()
        ds_entries = discover_datasets(ROOT_DIR)
        self._dataset_map = {name: (raw, lab) for (name, raw, lab) in ds_entries}
        ds_names = list(self._dataset_map.keys())
        self.dataset_dropdown = ttk.Combobox(
            ds_frame, textvariable=self.dataset_var, values=ds_names,
            state="readonly", width=24, font=ko_font,
        )
        self.dataset_dropdown.pack(side="left")
        self.dataset_dropdown.bind("<<ComboboxSelected>>", self._on_dataset_change)
        # Set initial dataset to match current DATA_DIR
        for name, (raw, lab) in self._dataset_map.items():
            if raw.resolve() == DATA_DIR.resolve():
                self.dataset_var.set(name)
                break
        else:
            if ds_names:
                self.dataset_var.set(ds_names[0])
        tk.Frame(side, height=4).pack()

        # ── Progress ──
        _label(side, text="PROGRESS", header=True, anchor="w").pack(fill="x", pady=(6, 1))
        _separator(side).pack(fill="x", pady=(0, 4))
        self.progress_canvas = tk.Canvas(
            side, width=380, height=14, bg=COLOR_BG_INPUT, highlightthickness=0,
        )
        self.progress_canvas.pack(anchor="w")
        self.progress_text = _label(side, text="0/0", anchor="w")
        self.progress_text.pack(fill="x")
        # 상세 통계 dashboard (폴리곤 / GT pick / PICKABLE)
        self.dash_polys_label = _label(side, text="", muted=True, anchor="w")
        self.dash_polys_label.pack(fill="x")
        self.dash_gt_label = _label(side, text="", muted=True, anchor="w")
        self.dash_gt_label.pack(fill="x")
        self.dash_pick_label = _label(side, text="", muted=True, anchor="w")
        self.dash_pick_label.pack(fill="x")
        # YOLO split 전체 분포
        self.dash_split_label = _label(side, text="", muted=True, anchor="w")
        self.dash_split_label.pack(fill="x")

        self.shot_label = _label(side, text="", font=ko_big, header=True, anchor="w")
        self.shot_label.pack(fill="x", pady=(4, 0))
        # YOLO SPLIT 배지 (Y 키로 토글) — 색상으로 train/val 구분
        split_row = tk.Frame(side, bg=COLOR_BG_PANEL)
        split_row.pack(fill="x", pady=(2, 0))
        self.split_badge = tk.Label(
            split_row, text=" — ", font=ko_bold, padx=10, pady=2,
            bg="#7F8C8D", fg="#FFFFFF", relief="flat",
        )
        self.split_badge.pack(side="left")
        self.split_source_label = tk.Label(
            split_row, text="", font=ko_font, padx=6,
            bg=COLOR_BG_PANEL, fg=COLOR_TEXT_MUTED,
        )
        self.split_source_label.pack(side="left")
        self.progress_label = _label(side, text="", muted=True, anchor="w")
        self.progress_label.pack(fill="x")
        self.mode_label = _label(
            side, text="Mode: NORMAL", bold=True, anchor="w",
        )
        self.mode_label.config(fg=COLOR_OK)
        self.mode_label.pack(fill="x", pady=(0, 4))

        # ── SHOTS jump list ─────────────────────────────────────────────
        _label(side, text="SHOTS (필터 입력 → 클릭/Enter로 이동)",
                header=True, anchor="w").pack(fill="x", pady=(6, 1))
        _separator(side).pack(fill="x", pady=(0, 4))
        self.shot_filter_var = tk.StringVar()
        self.shot_filter_entry = tk.Entry(
            side, textvariable=self.shot_filter_var, font=ko_mono,
            bg=COLOR_BG_INPUT, fg=COLOR_TEXT,
            insertbackground=COLOR_TEXT,
            relief="flat", borderwidth=0, highlightthickness=1,
            highlightbackground=COLOR_BG_INPUT,
            highlightcolor=COLOR_ACCENT,
        )
        self.shot_filter_entry.pack(fill="x", pady=(0, 2), ipady=2)
        self.shot_filter_var.trace_add(
            "write", lambda *_: self._refresh_shot_jump_list())
        self.shot_listbox = tk.Listbox(
            side, height=8, font=ko_mono,
            exportselection=False,
            bg=COLOR_BG_INPUT, fg=COLOR_TEXT,
            selectbackground=COLOR_ACCENT, selectforeground="#000000",
            highlightthickness=0, borderwidth=0, relief="flat",
        )
        self.shot_listbox.pack(fill="x")
        # <<ListboxSelect>> 바인딩 의도적 미사용 — selection_set()이 비동기로
        # 이벤트 트리거해 자동 점프 일으킴. 마우스 버튼 release / Enter만 사용.
        # <ButtonRelease-1>은 사용자 명시 클릭만 트리거되므로 selection_set이
        # 부르지 않음 → 자동 점프 X. 단일 클릭으로 jump.
        self.shot_listbox.bind(
            "<ButtonRelease-1>", self._on_shot_jump_select)
        self.shot_listbox.bind(
            "<Double-Button-1>", self._on_shot_jump_select)
        self.shot_listbox.bind(
            "<Return>", self._on_shot_jump_select)
        self._filtered_shot_idxs = []
        self._shot_listbox_locked = False

        # ── Tools (Prelabel + Box ROI + SAM2) ──
        _label(side, text="TOOLS", header=True, anchor="w").pack(fill="x", pady=(6, 1))
        _separator(side).pack(fill="x", pady=(0, 4))
        # Prelabel source selector
        _label(side, text="Prelabel source", muted=True, anchor="w").pack(fill="x")
        prelabel_options = [
            "Watershed (auto outlines)",
            "Auto-Classify (model class)",
        ]
        self.prelabel_source_var = tk.StringVar(value=prelabel_options[1])
        ttk.Combobox(
            side, textvariable=self.prelabel_source_var,
            values=prelabel_options, state="readonly", font=ko_font,
        ).pack(fill="x", pady=(0, 2))
        # Model selector (used when source = Auto-Classify)
        _label(side, text="Model (best.pt)", muted=True, anchor="w").pack(fill="x")
        runs_dir = ROOT_DIR / "runs" / "segment"
        models = []
        if runs_dir.exists():
            for d in sorted(runs_dir.iterdir()):
                if (d / "weights" / "best.pt").exists():
                    models.append(d.name)
        self.model_var = tk.StringVar(value=models[-1] if models else "")
        ttk.Combobox(
            side, textvariable=self.model_var, values=models,
            state="readonly", font=ko_font,
        ).pack(fill="x", pady=(0, 2))
        # Run button row
        tools_row = _frame(side)
        tools_row.pack(fill="x", pady=1)
        if _HAS_CTK:
            self.prelabel_btn = ctk.CTkButton(
                tools_row, text="▶ Prelabel  (Ctrl+R)",
                command=self.on_prelabel,
                width=170, height=28,
            )
        else:
            self.prelabel_btn = tk.Button(
                tools_row, text="▶ Prelabel  (Ctrl+R)",
                command=self.on_prelabel,
                font=ko_bold, bg="#3498DB", fg="white",
            )
        self.prelabel_btn.pack(side="left", padx=4)
        ToolTip(
            self.prelabel_btn,
            "Prelabel ▶ 실행\n"
            "  ▸ Watershed: 박스 안 외곽선 자동 생성\n"
            "    (박스 ROI 먼저 그려야 동작)\n"
            "  ▸ Auto-Classify: 사용자 폴리곤 + 모델 → class 자동 부여\n"
            "    (W로 외곽선 먼저 그리고, 0~4 안 눌러도 됨)\n"
            "  Source dropdown으로 모드 전환.",
        )
        self.prelabel_status = _label(side, text="", muted=True, anchor="w")
        self.prelabel_status.pack(fill="x")
        # Compute OOD anomaly score (Supersimplenet, 별도 env)
        if _HAS_CTK:
            self.ood_btn = ctk.CTkButton(
                side, text="🚨 Compute OOD  (O 토글)",
                command=self.on_compute_ood, height=28,
            )
        else:
            self.ood_btn = tk.Button(
                side, text="🚨 Compute OOD  (O 토글)",
                command=self.on_compute_ood, font=ko_font,
                bg="#E67E22", fg="white",
            )
        self.ood_btn.pack(fill="x", pady=(2, 0))
        ToolTip(
            self.ood_btn,
            "Supersimplenet 으로 각 폴리곤의 anomaly score 계산.\n"
            "별도 conda env (`anomaly`) 의 subprocess 호출.\n"
            "  ▸ 최초 호출은 모델 로드로 5~10초 소요\n"
            "  ▸ 'O' 키로 점수 overlay ON/OFF\n"
            "  ▸ score > threshold 폴리곤은 빨간 대시 외곽선",
        )
        self.ood_status = _label(side, text="", muted=True, anchor="w")
        self.ood_status.pack(fill="x")
        # Box ROI button
        if _HAS_CTK:
            self.box_roi_btn = ctk.CTkButton(
                side, text="📐 박스 ROI 그리기  (B)",
                command=self.on_box_roi_button, height=30,
            )
        else:
            self.box_roi_btn = tk.Button(
                side, text="📐 박스 ROI 그리기  (B)",
                command=self.on_box_roi_button, font=ko_font,
            )
        self.box_roi_btn.pack(fill="x", pady=2)
        ToolTip(
            self.box_roi_btn,
            "캔버스에서 박스 영역 사각형 드래그.\n"
            "한 번 그리면 같은 데이터셋 모든 샷에 자동 적용.\n"
            "meta.yaml에 저장됨.",
        )

        # SAM2 click-to-segment mode
        if _HAS_CTK:
            self.sam2_btn = ctk.CTkButton(
                side, text="🎯 SAM2 Click 모드  (A)",
                command=self.on_sam2_button, height=30,
            )
        else:
            self.sam2_btn = tk.Button(
                side, text="🎯 SAM2 Click 모드  (A)",
                command=self.on_sam2_button, font=ko_font,
            )
        self.sam2_btn.pack(fill="x", pady=2)
        ToolTip(
            self.sam2_btn,
            "SAM2 인터랙티브 분할\n"
            "  객체 위에 점 1개 클릭하면 외곽선 자동 생성.\n"
            "  클릭 후 0~4 키로 클래스 부여.\n"
            "  ESC로 모드 종료.\n"
            "  ※ 첫 호출 시 모델 로드 ~2초, 이후 클릭당 ~0.2초.",
        )

        # Inference Viewer (Toplevel popup) — visualize trained YOLO seg masks
        if _HAS_CTK:
            self.infer_view_btn = ctk.CTkButton(
                side, text="🔍 Inference Viewer  (Ctrl+I)",
                command=self.open_inference_viewer, height=30,
            )
        else:
            self.infer_view_btn = tk.Button(
                side, text="🔍 Inference Viewer  (Ctrl+I)",
                command=self.open_inference_viewer, font=ko_font,
            )
        self.infer_view_btn.pack(fill="x", pady=2)
        ToolTip(
            self.infer_view_btn,
            "학습된 best.pt의 seg 마스크를 별창에서 시각화.\n"
            "  ▸ Model/Image 드롭다운으로 선택\n"
            "  ▸ Confidence 슬라이더로 임계값 조절\n"
            "  ▸ GT 같이 보기 / Bounding Box 토글\n"
            "  현재 라벨링 작업과 독립된 창.",
        )

        # Roboflow Label Review (compare our v2 vs teammate's Roboflow labels)
        if _HAS_CTK:
            self.rf_review_btn = ctk.CTkButton(
                side, text="📋 Roboflow 라벨 비교  (Ctrl+Shift+R)",
                command=self.open_roboflow_review, height=30,
            )
        else:
            self.rf_review_btn = tk.Button(
                side, text="📋 Roboflow 라벨 비교  (Ctrl+Shift+R)",
                command=self.open_roboflow_review, font=ko_font,
            )
        self.rf_review_btn.pack(fill="x", pady=2)
        ToolTip(
            self.rf_review_btn,
            "팀원이 Roboflow에서 재라벨링한 폴리곤과 우리 v2 라벨을\n"
            "같은 이미지 위에 비교 표시.\n"
            "  ▸ Roboflow: 클래스별 색상 실선\n"
            "  ▸ 우리 v2: 흰색 점선\n"
            "  ▸ Prev/Next로 129장 순회\n"
            "  ▸ _external/roboflow_260508/ 에 압축 풀어야 작동.",
        )


        self.box_roi_status = _label(
            side, text="(미설정)", anchor="w",
        )
        self.box_roi_status.config(fg=COLOR_ERROR)
        self.box_roi_status.pack(fill="x")

        _label(side, text="CLASSES", header=True, anchor="w").pack(fill="x", pady=(6, 1))
        _separator(side).pack(fill="x", pady=(0, 4))
        for cid, (en, ko, color) in enumerate(CLASSES):
            f = _frame(side)
            f.pack(fill="x", pady=1)
            tk.Label(
                f, text=f" {cid} ", bg=color, fg="white",
                font=ko_bold, width=3,
            ).pack(side="left")
            _label(f, text=f"  {ko} ({en})").pack(side="left")
        f = _frame(side)
        f.pack(fill="x", pady=1)
        tk.Label(
            f, text=" - ", bg=UNLABELED_COLOR, fg="white",
            font=ko_bold, width=3,
        ).pack(side="left")
        _label(f, text="  미지정", muted=True).pack(side="left")

        # (LAYERS section is in LEFT panel — show_height_var/show_normal_var
        # are defined there before any reference)

        _label(side, text="POLYGONS", header=True, anchor="w").pack(fill="x", pady=(6, 1))
        _separator(side).pack(fill="x", pady=(0, 4))
        self.poly_list = tk.Listbox(
            side, height=8,
            font=ko_mono,
            exportselection=False,
            bg=COLOR_BG_INPUT, fg=COLOR_TEXT,
            selectbackground=COLOR_ACCENT, selectforeground="#000000",
            highlightthickness=0, borderwidth=0, relief="flat",
        )
        self.poly_list.pack(fill="x")
        self.poly_list.bind("<<ListboxSelect>>", self.on_listbox_select)

        # 3D STATS section moved to LEFT panel (alongside QUICK KEYS).
        # See _build_left_panel() — self.stats_text is created there.

        # LAYERS + ZOOM moved to LEFT panel (sidebar was overflowing).
        # QUICK KEYS section is also on the LEFT panel.

        # Status bar (bottom) — 3 segments: left, center (msg), right
        status_bar = tk.Frame(self.parent, bg=COLOR_BG_PANEL, height=22)
        status_bar.pack(side="bottom", fill="x")
        status_bar.pack_propagate(False)
        self.status_left = tk.Label(
            status_bar, text="", anchor="w", font=ko_mono,
            bg=COLOR_BG_PANEL, fg=COLOR_TEXT_MUTED, padx=8,
        )
        self.status_left.pack(side="left", fill="y")
        # Right side: zoom + mouse coords
        self.status_right = tk.Label(
            status_bar, text="", anchor="e", font=ko_mono,
            bg=COLOR_BG_PANEL, fg=COLOR_TEXT_MUTED, padx=8,
        )
        self.status_right.pack(side="right", fill="y")
        # Center: temporary messages (saved, prelabel done, etc.)
        self.status = tk.Label(
            status_bar, text="", anchor="center", font=ko_font,
            bg=COLOR_BG_PANEL, fg=COLOR_TEXT, padx=8,
        )
        self.status.pack(side="left", fill="both", expand=True)

        # Bind keys to canvas (focus required)
        self.canvas.bind("<Key>", self.on_key)
        self.canvas.bind("<FocusIn>", lambda e: None)
        self.canvas.focus_set()

        # Now pack left_panel — must be AFTER side is packed, with before=side
        # so it sits LEFT of the sidebar (DATASET).
        self.left_panel.pack(side="right", fill="y", before=self.side)

        # side 패널 마우스 휠 스크롤 — 자체 휠을 가진 Listbox/Text는 제외.
        # 그 외 위젯(Label, Frame, Button) 위에서 휠 → 사이드 패널 스크롤.
        if hasattr(self, "_side_wheel_handler"):
            def _bind_wheel_recursive(widget):
                try:
                    if widget.winfo_class() not in ("Listbox", "Text",
                                                     "TCombobox", "Combobox"):
                        widget.bind("<MouseWheel>",
                                      self._side_wheel_handler)
                except Exception:
                    pass
                for c in widget.winfo_children():
                    _bind_wheel_recursive(c)
            _bind_wheel_recursive(self.side)

    def _apply_shot_intrinsics(self, shot=None):
        """샷별 _info.json에서 fx/fy/cx/cy를 읽어 picking + suction_score에 주입.

        importlib.reload 후에도 호출 필요 — reload는 picking 모듈 top-level을
        재실행해서 FX/FY/CX/CY를 yaml 기본값(MR60)으로 되돌리기 때문. 이 helper를
        매 reload 직후 호출해서 현재 샷의 실제 카메라 값을 다시 주입한다.

        shot=None이면 현재 self.shots[self.idx] 사용. 라벨/픽 모듈이 아직 준비
        안 됐거나 _info.json이 없으면 silently skip — 기존 동작 보존.
        """
        try:
            if shot is None:
                if not (self.shots and 0 <= self.idx < len(self.shots)):
                    return
                shot = self.shots[self.idx]
            info_path = shot["bmp"].with_name(f"{shot['name']}_info.json")
            if not info_path.exists():
                return
            with open(info_path, encoding="utf-8") as f:
                ci = json.load(f).get("camera_info", {})
            fx = float(ci.get("cal.fx", _picking.FX))
            fy = float(ci.get("cal.fy", _picking.FY))
            cx = float(ci.get("cal.cx", _picking.CX))
            cy = float(ci.get("cal.cy", _picking.CY))
            _picking.set_camera_intrinsics(fx, fy, cx, cy)
            sensor = ci.get("sensor name", "")
            if hasattr(self, "status"):
                self.status.config(
                    text=f"📷 카메라 적용: {sensor}  fx={fx:.0f}",
                )
        except Exception as e:
            if hasattr(self, "status"):
                self.status.config(text=f"⚠ 카메라 intrinsics 로드 실패: {e}")

    def load_shot(self):
        if not (0 <= self.idx < len(self.shots)):
            return
        self._exit_all_modes(commit=False)
        # Invalidate height/normal/heatmap cache (each shot has its own PLY)
        self._height_rgba = None
        self._normal_rgba = None
        self._heatmap_rgba = None
        self._height_photo = None
        self._normal_photo = None
        self._heatmap_photo = None
        # Clear stale picks (each shot has its own pick set)
        self._picks_data = None
        for cid in getattr(self, "_pick_canvas_ids", []):
            try:
                self.canvas.delete(cid)
            except Exception:
                pass
        self._pick_canvas_ids = []
        # Refresh box ROI from meta.yaml of current OUT_DIR
        try:
            self.box_roi = load_box_roi(OUT_DIR)
        except Exception:
            self.box_roi = None
        if hasattr(self, "box_roi_status"):
            self._refresh_box_roi_status()
        shot = self.shots[self.idx]
        self._apply_shot_intrinsics(shot)
        with open(shot["data"], encoding="utf-8") as f:
            self.data = json.load(f)

        out_data = OUT_DIR / "data_json" / f"{shot['name']}_data.json"
        verified_list: list = []
        if out_data.exists():
            with open(out_data, encoding="utf-8") as f:
                saved = json.load(f)
            polygons = dedup_polygons(saved.get("result_data", []))
            classes = saved.get("class_id", [])
            verified_list = saved.get("verified", [])
        else:
            polygons = dedup_polygons(self.data.get("result_data", []))
            classes = [-1] * len(polygons)

        self.polys = []
        for i, poly in enumerate(polygons):
            cid = classes[i] if i < len(classes) else -1
            v = verified_list[i] if i < len(verified_list) else False
            self.polys.append({
                "points": poly,
                "class_id": cid,
                "verified": bool(v),
                "canvas_id": None,
                "label_id": None,
            })

        self.original_image = Image.open(shot["bmp"]).convert("RGB")
        self.canvas.delete("all")
        self.image_id = None
        self.redraw_image()

        self.advance_to_unlabeled()
        self.refresh_canvas()
        self.undo_stack = []
        # SHOTS jump list — 현재 샷 highlight 갱신
        try:
            self._refresh_shot_jump_list()
        except Exception:
            pass
        # Picks JSON 자동 로드 + stale 체크.
        # yaml/picking.py가 picks JSON보다 새것이면 자동 재계산해서 화면에 반영.
        try:
            self._auto_load_or_recompute_picks()
        except Exception as e:
            self.status.config(text=f"⚠ picks 자동 로드 실패: {e}")

    def _auto_load_or_recompute_picks(self):
        """샷 로드 시 picks 표시를 항상 끈다 — G 키로만 표시.

        자동 계산도, 자동 표시도 안 함. 사용자가 G 키를 눌러야
        picks 가 계산되고 canvas 에 표시됨.
        디스크의 picks JSON 도 자동 로드하지 않음 (이전 샷의 잔상도 차단).
        """
        # 이전 샷 잔상 차단 + 표시 off
        self._picks_data = None
        try:
            self.show_picks_var.set(False)
        except Exception:
            pass
        has_label = any(p.get("class_id", -1) >= 0 for p in self.polys)
        if not has_label:
            return
        # 안내만 — 디스크에 picks JSON 이 있든 없든 G 키로 트리거
        picks_path = self._picks_json_path()
        if picks_path is not None and picks_path.exists():
            self.status.config(text="ℹ picks JSON 존재 — G 키로 표시/재계산")
        else:
            self.status.config(text="ℹ picks 없음 — G 키로 계산")

    def refresh_canvas(self):
        for p in self.polys:
            if p["canvas_id"] is not None:
                self.canvas.delete(p["canvas_id"])
            if p["label_id"] is not None:
                self.canvas.delete(p["label_id"])
            if p.get("ood_text_id") is not None:
                self.canvas.delete(p["ood_text_id"])
                p["ood_text_id"] = None
        # Height / Normal / Heatmap overlays (BMP 위, 폴리곤 아래)
        if self._height_overlay_id is not None:
            self.canvas.delete(self._height_overlay_id)
            self._height_overlay_id = None
        if self._normal_overlay_id is not None:
            self.canvas.delete(self._normal_overlay_id)
            self._normal_overlay_id = None
        if self._heatmap_overlay_id is not None:
            self.canvas.delete(self._heatmap_overlay_id)
            self._heatmap_overlay_id = None
        if self.show_normal_var.get() and self._normal_rgba is not None:
            self._normal_photo = self._make_overlay_photo(self._normal_rgba)
            if self._normal_photo is not None:
                self._normal_overlay_id = self.canvas.create_image(
                    0, 0, anchor="nw", image=self._normal_photo,
                )
        if self.show_height_var.get() and self._height_rgba is not None:
            self._height_photo = self._make_overlay_photo(self._height_rgba)
            if self._height_photo is not None:
                self._height_overlay_id = self.canvas.create_image(
                    0, 0, anchor="nw", image=self._height_photo,
                )
        if self.show_heatmap_var.get() and self._heatmap_rgba is not None:
            self._heatmap_photo = self._make_overlay_photo(self._heatmap_rgba)
            if self._heatmap_photo is not None:
                self._heatmap_overlay_id = self.canvas.create_image(
                    0, 0, anchor="nw", image=self._heatmap_photo,
                )
        # Box ROI display rectangle (semi-transparent green outline, no fill)
        if self._box_roi_display_id is not None:
            self.canvas.delete(self._box_roi_display_id)
            self._box_roi_display_id = None
        if self.box_roi is not None and self.mode != "box_roi":
            r = self.box_roi
            self._box_roi_display_id = self.canvas.create_rectangle(
                r["u_min"] * self.scale, r["v_min"] * self.scale,
                r["u_max"] * self.scale, r["v_max"] * self.scale,
                outline="#2ECC71", width=2, dash=(6, 4),
            )
        for i, p in enumerate(self.polys):
            cid = p["class_id"]
            is_assigned = 0 <= cid < len(CLASSES)
            color = CLASSES[cid][2] if is_assigned else UNLABELED_COLOR
            outline = SELECTED_OUTLINE if i == self.selected else color
            width = 4 if i == self.selected else 2
            dash = None
            # OOD overlay: score > threshold → red dashed outline (selected highlight 보다 우선)
            ascore = p.get("anomaly_score")
            is_ood = (self.ood_visible and ascore is not None
                      and self.ood_threshold is not None
                      and ascore > self.ood_threshold)
            if is_ood:
                outline = OOD_OUTLINE_COLOR
                width = 4
                dash = (6, 4)
            # Lighter stipple for unassigned so canvas BMP stays visible
            stipple = "gray25" if is_assigned else "gray12"
            scaled = []
            for x, y in p["points"]:
                scaled.extend([x * self.scale, y * self.scale])
            poly_kwargs = dict(
                outline=outline, fill=color,
                stipple=stipple, width=width,
            )
            if dash is not None:
                poly_kwargs["dash"] = dash
            p["canvas_id"] = self.canvas.create_polygon(scaled, **poly_kwargs)
            cx, cy = polygon_centroid(p["points"])
            p["label_id"] = self.canvas.create_text(
                cx * self.scale, cy * self.scale,
                text=str(i), fill="white", font=("Arial", 14, "bold"),
            )
            # OOD score badge (centroid 아래, 인덱스 라벨과 겹치지 않게)
            if self.ood_visible and ascore is not None:
                badge_color = OOD_OUTLINE_COLOR if is_ood else "#7FB069"
                p["ood_text_id"] = self.canvas.create_text(
                    cx * self.scale, cy * self.scale + 18,
                    text=f"AS {ascore:.3f}",
                    fill=badge_color,
                    font=("Arial", 10, "bold"),
                )
        if self.mode == "edit":
            self.draw_edit_handles()
        elif self.mode == "draw":
            self.update_draw_preview()
        # Pick rectangles overlay (drawn on top of polygons)
        self._draw_picks_on_canvas()
        self.update_panels()

    def on_motion_status(self, event):
        """Update right status with mouse coords (image space) + zoom."""
        if not hasattr(self, "status_right"):
            return
        cx = self.canvas.canvasx(event.x) / max(self.scale, 1e-6)
        cy = self.canvas.canvasy(event.y) / max(self.scale, 1e-6)
        if 0 <= cx < IMG_W and 0 <= cy < IMG_H:
            self.status_right.config(
                text=f"x={int(cx):4d}  y={int(cy):4d}  |  zoom {int(self.zoom*100)}%"
            )
        else:
            self.status_right.config(
                text=f"--           |  zoom {int(self.zoom*100)}%"
            )

    def _update_status_left(self):
        """Update left status: dataset, shot index, progress counts."""
        if not hasattr(self, "status_left"):
            return
        if not self.shots:
            self.status_left.config(text="(no dataset)")
            return
        ds = self.dataset_var.get() if hasattr(self, "dataset_var") else "?"
        shot_no = f"{self.idx + 1}/{len(self.shots)}"
        done, partial, total = (
            self.compute_dataset_progress() if hasattr(self, "compute_dataset_progress")
            else (0, 0, len(self.shots))
        )
        split, src = self._compute_split_for_current()
        split_str = {
            "train": "TRAIN",
            "val":   "VAL",
            "empty": "—",
        }.get(split, "?")
        split_seg = f"split: {split_str} ({src})"
        self.status_left.config(
            text=f"{ds} | shot {shot_no} | ✓{done} △{partial} "
                 f"·{total - done - partial} | {split_seg}"
        )

    def update_panels(self):
        shot = self.shots[self.idx]
        self.shot_label.config(text=shot["name"])
        self._update_status_left()
        self._update_split_badge()
        labeled = sum(
            1 for p in self.polys if 0 <= p["class_id"] < len(CLASSES)
        )
        total = len(self.polys)
        self.progress_label.config(
            text=f"Shot {self.idx+1}/{len(self.shots)}    {labeled}/{total} labeled"
        )
        mode_text = {
            "normal": ("Mode: NORMAL", "#27AE60"),
            "draw": ("Mode: DRAW", "#E67E22"),
            "edit": ("Mode: EDIT", "#2980B9"),
            "box_roi": ("Mode: BOX ROI", "#8E44AD"),
            "sam2": ("Mode: SAM2 CLICK", "#E67E22"),
        }.get(self.mode, ("Mode: ?", "#7F8C8D"))
        self.mode_label.config(text=mode_text[0], fg=mode_text[1])

        self._listbox_lock = True
        self.poly_list.delete(0, tk.END)
        for i, p in enumerate(self.polys):
            cid = p["class_id"]
            verified = p.get("verified", False)
            if cid < 0 or cid >= len(CLASSES):
                mark = "·"   # unassigned
                name = "—"
            else:
                mark = "✓" if verified else "△"
                name = CLASSES[cid][1]   # 한국어만 (영어 제거) — 너비 절약
            self.poly_list.insert(tk.END, f"{mark} {i:02d} {name}")
        if 0 <= self.selected < len(self.polys):
            self.poly_list.selection_set(self.selected)
            self.poly_list.see(self.selected)
        self._listbox_lock = False
        self.refresh_3d_stats()

    def on_click(self, event):
        self.canvas.focus_set()
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        if self.mode == "sam2":
            self.on_click_sam2(event)
            return
        if self.mode == "box_roi":
            self.on_click_box_roi(event)
            return
        if self.mode == "draw":
            self.add_draw_point(cx, cy)
            return
        if self.mode == "edit":
            vi = self.find_vertex_at(cx, cy)
            if vi is not None:
                self.dragging_vertex = vi
            else:
                self.insert_vertex_at(cx, cy)
            return
        x = cx / self.scale
        y = cy / self.scale
        for i in range(len(self.polys) - 1, -1, -1):
            if point_in_polygon(x, y, self.polys[i]["points"]):
                self.selected = i
                self.refresh_canvas()
                return
        self.selected = -1
        self.refresh_canvas()

    def on_motion(self, event):
        if self.mode == "draw" and self.draw_points:
            last_x, last_y = self.draw_points[-1]
            sx = last_x * self.scale
            sy = last_y * self.scale
            ex = self.canvas.canvasx(event.x)
            ey = self.canvas.canvasy(event.y)
            if self.rubber_band_id is not None:
                self.canvas.coords(self.rubber_band_id, sx, sy, ex, ey)
            else:
                self.rubber_band_id = self.canvas.create_line(
                    sx, sy, ex, ey,
                    fill=DRAW_COLOR, width=1, dash=(3, 3),
                )

    def on_drag(self, event):
        if self.mode == "box_roi":
            self.on_drag_box_roi(event)
            return
        if (self.mode == "edit" and self.dragging_vertex is not None
                and 0 <= self.editing_idx < len(self.polys)):
            cx = self.canvas.canvasx(event.x)
            cy = self.canvas.canvasy(event.y)
            x = clamp(cx / self.scale, 0, IMG_W - 1)
            y = clamp(cy / self.scale, 0, IMG_H - 1)
            self.polys[self.editing_idx]["points"][self.dragging_vertex] = [x, y]
            self.refresh_canvas()

    def on_release(self, event):
        if self.mode == "box_roi":
            self.on_release_box_roi(event)
            return
        self.dragging_vertex = None

    def on_double_click(self, _event):
        if self.mode == "draw":
            self.finish_draw_polygon()

    def on_right_click(self, event):
        if self.mode == "draw":
            self.finish_draw_polygon()
            return
        if self.mode == "edit":
            cx = self.canvas.canvasx(event.x)
            cy = self.canvas.canvasy(event.y)
            vi = self.find_vertex_at(cx, cy)
            if vi is not None:
                self.delete_vertex(vi)

    def on_listbox_select(self, _event):
        if self._listbox_lock:
            return
        if self.mode != "normal":
            return
        sel = self.poly_list.curselection()
        if sel:
            self.selected = sel[0]
            self.refresh_canvas()

    def on_key(self, event):
        c = event.char
        k = event.keysym
        if k.lower() == "q":
            self.save_current()
            return

        if k.lower() == "f":
            self.zoom_fit()
            return "break"
        if c in ("+", "="):
            self.zoom_in()
            return "break"
        if c == "-":
            self.zoom_out()
            return "break"

        if self.mode == "sam2":
            if k == "Escape":
                self._exit_sam2_mode()
                return "break"
            if k.lower() == "a":
                self._exit_sam2_mode()
                return "break"
            # Allow class assignment (0~4) directly on selected polygon
            if c in "01234":
                cid = int(c)
                if 0 <= self.selected < len(self.polys):
                    self.undo_stack.append(
                        ("class", self.selected,
                         self.polys[self.selected]["class_id"])
                    )
                    self.polys[self.selected]["class_id"] = cid
                    self.refresh_canvas()
                    self._update_split_badge()
                return "break"
            return

        if self.mode == "draw":
            if k == "Return":
                self.finish_draw_polygon()
                return "break"
            if k == "Escape":
                self.cancel_draw_mode()
                return "break"
            if k.lower() == "w":
                self.cancel_draw_mode()
                return "break"
            return

        if self.mode == "edit":
            if k == "Escape":
                if (self.edit_pre_snapshot is not None
                        and 0 <= self.editing_idx < len(self.polys)):
                    self.polys[self.editing_idx]["points"] = [
                        list(p) for p in self.edit_pre_snapshot
                    ]
                self.exit_edit_mode(commit=False)
                return "break"
            if k.lower() == "e":
                self.exit_edit_mode(commit=True)
                return "break"
            return

        if c in "01234":
            cid = int(c)
            if 0 <= self.selected < len(self.polys):
                self.undo_stack.append(
                    ("class", self.selected, self.polys[self.selected]["class_id"])
                )
                self.polys[self.selected]["class_id"] = cid
                self.advance_to_unlabeled()
                self.refresh_canvas()
                self._update_split_badge()
        elif k == "Tab":
            self.advance_to_unlabeled()
            self.refresh_canvas()
            return "break"
        elif k == "space":
            # toggle verified flag on selected polygon (△ ↔ ✓)
            if 0 <= self.selected < len(self.polys):
                p = self.polys[self.selected]
                if 0 <= p["class_id"] < len(CLASSES):
                    p["verified"] = not p.get("verified", False)
                    self.refresh_canvas()
            return "break"
        elif k.lower() == "h":
            self.show_height_var.set(not self.show_height_var.get())
            self._on_layer_toggle()
            return "break"
        elif k.lower() == "m":
            self.show_normal_var.set(not self.show_normal_var.get())
            self._on_layer_toggle()
            return "break"
        elif k.lower() == "k":
            self.show_heatmap_var.set(not self.show_heatmap_var.get())
            self._on_layer_toggle()
            return "break"
        elif k.lower() == "o":
            self.ood_visible = not self.ood_visible
            n_scored = sum(1 for p in self.polys if p.get("anomaly_score") is not None)
            if self.ood_visible and n_scored == 0:
                self.status.config(text="ℹ OOD: 아직 점수 없음 — 'Compute OOD' 먼저 누르세요.")
            else:
                state = "ON" if self.ood_visible else "OFF"
                self.status.config(text=f"OOD overlay {state} (점수 {n_scored}/{len(self.polys)})")
            self.refresh_canvas()
            return "break"
        elif k.lower() == "n":
            self.save_current()
            if self.idx < len(self.shots) - 1:
                self.idx += 1
                self.load_shot()
        elif k.lower() == "p":
            self.save_current()
            if self.idx > 0:
                self.idx -= 1
                self.load_shot()
        elif k.lower() == "s":
            self.save_current()
            self.status.config(text="✓ Saved")
        elif k.lower() == "w":
            self.enter_draw_mode()
        elif k.lower() == "e":
            self.enter_edit_mode()
        elif k.lower() == "d":
            self.delete_selected()
        elif k.lower() == "u":
            self.undo()
        elif k.lower() == "y":
            self.toggle_split_override()
            return "break"

    def _exit_all_modes(self, commit=True):
        if self.mode == "draw":
            self.cancel_draw_mode()
        elif self.mode == "edit":
            self.exit_edit_mode(commit=commit)

    def enter_draw_mode(self):
        if self.mode == "edit":
            self.exit_edit_mode(commit=True)
        self.mode = "draw"
        self.draw_points = []
        self.clear_draw_preview()
        if self.rubber_band_id is not None:
            self.canvas.delete(self.rubber_band_id)
            self.rubber_band_id = None
        self.canvas.config(cursor="cross")
        self.status.config(
            text="DRAW: click points / Enter or double-click or right-click to close / Esc to cancel"
        )
        self.update_panels()

    def cancel_draw_mode(self):
        self.mode = "normal"
        self.draw_points = []
        self.clear_draw_preview()
        if self.rubber_band_id is not None:
            self.canvas.delete(self.rubber_band_id)
            self.rubber_band_id = None
        self.canvas.config(cursor="cross")
        self.status.config(text="Draw cancelled.")
        self.update_panels()

    def add_draw_point(self, cx, cy):
        x = clamp(cx / self.scale, 0, IMG_W - 1)
        y = clamp(cy / self.scale, 0, IMG_H - 1)
        self.draw_points.append([x, y])
        self.update_draw_preview()

    def finish_draw_polygon(self):
        if self.mode != "draw":
            return
        if len(self.draw_points) < 3:
            self.status.config(
                text=f"Polygon needs ≥3 points (currently {len(self.draw_points)})"
            )
            return
        new_poly = {
            "points": [list(p) for p in self.draw_points],
            "class_id": -1,
            "verified": True,   # human-drawn → auto-verified
            "canvas_id": None,
            "label_id": None,
        }
        self.polys.append(new_poly)
        self.undo_stack.append(("add", len(self.polys) - 1))
        self.selected = len(self.polys) - 1
        self.draw_points = []
        self.clear_draw_preview()
        if self.rubber_band_id is not None:
            self.canvas.delete(self.rubber_band_id)
            self.rubber_band_id = None
        self.mode = "normal"
        self.canvas.config(cursor="cross")
        self.refresh_canvas()
        self.status.config(
            text=f"Added #{self.selected}. Press 0~4 to assign class."
        )

    def update_draw_preview(self):
        self.clear_draw_preview()
        for x, y in self.draw_points:
            sx = x * self.scale
            sy = y * self.scale
            r = 5
            cid = self.canvas.create_oval(
                sx - r, sy - r, sx + r, sy + r,
                fill=DRAW_COLOR, outline="black", width=1,
            )
            self.draw_preview_ids.append(cid)
        if len(self.draw_points) >= 2:
            coords = []
            for x, y in self.draw_points:
                coords.extend([x * self.scale, y * self.scale])
            cid = self.canvas.create_line(
                *coords, fill=DRAW_COLOR, width=2,
            )
            self.draw_preview_ids.append(cid)

    def clear_draw_preview(self):
        for cid in self.draw_preview_ids:
            self.canvas.delete(cid)
        self.draw_preview_ids = []

    def enter_edit_mode(self):
        if not (0 <= self.selected < len(self.polys)):
            self.status.config(text="No polygon selected.")
            return
        if self.mode == "draw":
            self.cancel_draw_mode()
        self.mode = "edit"
        self.editing_idx = self.selected
        self.edit_pre_snapshot = [
            list(p) for p in self.polys[self.editing_idx]["points"]
        ]
        self.canvas.config(cursor="hand2")
        self.refresh_canvas()
        self.status.config(
            text=f"EDIT #{self.editing_idx}: drag vertices / E=commit / Esc=cancel"
        )

    def exit_edit_mode(self, commit=True):
        if self.mode != "edit":
            return
        if (commit and 0 <= self.editing_idx < len(self.polys)
                and self.edit_pre_snapshot is not None):
            current = self.polys[self.editing_idx]["points"]
            if current != self.edit_pre_snapshot:
                self.undo_stack.append(
                    ("edit", self.editing_idx,
                     [list(p) for p in self.edit_pre_snapshot])
                )
                # Editing the outline = user reviewed it → auto-verified
                self.polys[self.editing_idx]["verified"] = True
        self.mode = "normal"
        self.editing_idx = -1
        self.edit_pre_snapshot = None
        self.dragging_vertex = None
        self.clear_edit_handles()
        self.canvas.config(cursor="cross")
        self.refresh_canvas()
        self.status.config(text="Edit committed." if commit else "Edit cancelled.")

    def draw_edit_handles(self):
        self.clear_edit_handles()
        if not (0 <= self.editing_idx < len(self.polys)):
            return
        poly = self.polys[self.editing_idx]["points"]
        for x, y in poly:
            sx = x * self.scale
            sy = y * self.scale
            r = 6
            cid = self.canvas.create_oval(
                sx - r, sy - r, sx + r, sy + r,
                fill=HANDLE_FILL, outline=HANDLE_OUTLINE, width=2,
            )
            self.edit_handle_ids.append(cid)

    def clear_edit_handles(self):
        for cid in self.edit_handle_ids:
            self.canvas.delete(cid)
        self.edit_handle_ids = []

    def find_vertex_at(self, ex, ey):
        if not (0 <= self.editing_idx < len(self.polys)):
            return None
        poly = self.polys[self.editing_idx]["points"]
        best_vi, best_d2 = None, VERTEX_HIT_RADIUS_PX ** 2
        for vi, (x, y) in enumerate(poly):
            sx = x * self.scale
            sy = y * self.scale
            d2 = (ex - sx) ** 2 + (ey - sy) ** 2
            if d2 <= best_d2:
                best_vi, best_d2 = vi, d2
        return best_vi

    def insert_vertex_at(self, cx, cy):
        if not (0 <= self.editing_idx < len(self.polys)):
            return
        poly = self.polys[self.editing_idx]["points"]
        if len(poly) < 2:
            return
        x = cx / self.scale
        y = cy / self.scale
        result = closest_edge(x, y, poly)
        if result is None:
            return
        edge_idx, fx, fy, d2 = result
        canvas_dist_px = (d2 ** 0.5) * self.scale
        if canvas_dist_px > 15:
            return
        fx = clamp(fx, 0, IMG_W - 1)
        fy = clamp(fy, 0, IMG_H - 1)
        poly.insert(edge_idx + 1, [fx, fy])
        self.refresh_canvas()
        self.status.config(
            text=f"꼭지점 추가됨 (edge {edge_idx} 뒤). 총 {len(poly)}개."
        )

    def delete_vertex(self, vi):
        if not (0 <= self.editing_idx < len(self.polys)):
            return
        poly = self.polys[self.editing_idx]["points"]
        if not (0 <= vi < len(poly)):
            return
        if len(poly) <= 3:
            self.status.config(text="폴리곤은 최소 3개 꼭지점이 필요합니다.")
            return
        poly.pop(vi)
        self.refresh_canvas()
        self.status.config(
            text=f"꼭지점 #{vi} 삭제됨. 총 {len(poly)}개."
        )

    def delete_selected(self):
        if self.mode != "normal":
            return
        if not (0 <= self.selected < len(self.polys)):
            return
        idx = self.selected
        poly = self.polys.pop(idx)
        self.undo_stack.append(("delete", idx, poly))
        if not self.polys:
            self.selected = -1
        else:
            self.selected = min(idx, len(self.polys) - 1)
        self.refresh_canvas()
        self._update_split_badge()
        self.status.config(text="Deleted polygon. Press U to undo.")

    def advance_to_unlabeled(self):
        n = len(self.polys)
        if n == 0:
            self.selected = -1
            return
        start = (self.selected + 1) % n if self.selected >= 0 else 0
        for offset in range(n):
            i = (start + offset) % n
            if not (0 <= self.polys[i]["class_id"] < len(CLASSES)):
                self.selected = i
                return
        self.selected = -1

    def undo(self):
        if not self.undo_stack:
            return
        op = self.undo_stack.pop()
        op_type = op[0]
        if op_type == "class":
            _, idx, prev = op
            if 0 <= idx < len(self.polys):
                self.polys[idx]["class_id"] = prev
                self.selected = idx
        elif op_type == "add":
            _, idx = op
            if 0 <= idx < len(self.polys):
                self.polys.pop(idx)
                if self.selected == idx:
                    self.selected = -1
                elif self.selected > idx:
                    self.selected -= 1
        elif op_type == "delete":
            _, idx, poly_data = op
            poly_data["canvas_id"] = None
            poly_data["label_id"] = None
            self.polys.insert(idx, poly_data)
            self.selected = idx
        elif op_type == "edit":
            _, idx, prev_points = op
            if 0 <= idx < len(self.polys):
                self.polys[idx]["points"] = prev_points
                self.selected = idx
        self.refresh_canvas()
        self._update_split_badge()

    def redraw_image(self):
        if self.original_image is None:
            return
        iw = max(1, int(IMG_W * self.scale))
        ih = max(1, int(IMG_H * self.scale))
        img = self.original_image.resize((iw, ih), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(img)
        if self.image_id is None:
            self.image_id = self.canvas.create_image(
                0, 0, anchor="nw", image=self.tk_image,
            )
            self.canvas.tag_lower(self.image_id)
        else:
            self.canvas.itemconfig(self.image_id, image=self.tk_image)
            self.canvas.coords(self.image_id, 0, 0)
            self.canvas.tag_lower(self.image_id)
        self.canvas.config(scrollregion=(0, 0, iw, ih))

    def _viewport_center_anchor(self):
        mx = self.viewport_w / 2
        my = self.viewport_h / 2
        cx = self.canvas.canvasx(mx)
        cy = self.canvas.canvasy(my)
        return (cx, cy, mx, my)

    def zoom_in(self):
        self._apply_zoom(self.zoom * ZOOM_STEP, anchor=self._viewport_center_anchor())

    def zoom_out(self):
        self._apply_zoom(self.zoom / ZOOM_STEP, anchor=self._viewport_center_anchor())

    def zoom_fit(self):
        self._apply_zoom(1.0, anchor=None)

    def _apply_zoom(self, new_zoom, anchor=None):
        new_zoom = clamp(new_zoom, ZOOM_MIN, ZOOM_MAX)
        if abs(new_zoom - self.zoom) < 1e-6:
            return
        old_scale = self.scale
        self.zoom = new_zoom
        self.scale = self.base_scale * self.zoom

        self.redraw_image()
        self.refresh_canvas()
        self.update_zoom_label()

        if anchor is not None:
            ax, ay, mx, my = anchor
            ratio = self.scale / old_scale
            new_ax = ax * ratio
            new_ay = ay * ratio
            new_iw = max(1, IMG_W * self.scale)
            new_ih = max(1, IMG_H * self.scale)
            self.canvas.xview_moveto(
                max(0.0, min(1.0, (new_ax - mx) / new_iw))
            )
            self.canvas.yview_moveto(
                max(0.0, min(1.0, (new_ay - my) / new_ih))
            )

    def update_zoom_label(self):
        if hasattr(self, "zoom_label"):
            self.zoom_label.config(text=f"{int(round(self.zoom * 100))}%")

    def on_mouse_wheel(self, event):
        factor = ZOOM_STEP if event.delta > 0 else 1.0 / ZOOM_STEP
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        self._apply_zoom(self.zoom * factor, anchor=(cx, cy, event.x, event.y))
        return "break"

    def pan_pixels(self, dx, dy):
        iw = max(1, IMG_W * self.scale)
        ih = max(1, IMG_H * self.scale)
        x_lo, _ = self.canvas.xview()
        y_lo, _ = self.canvas.yview()
        cur_x = x_lo * iw
        cur_y = y_lo * ih
        new_x = max(0, min(iw, cur_x + dx))
        new_y = max(0, min(ih, cur_y + dy))
        self.canvas.xview_moveto(new_x / iw)
        self.canvas.yview_moveto(new_y / ih)
        return "break"

    def ensure_dirs(self):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "images").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "data_json").mkdir(parents=True, exist_ok=True)

    def save_current(self):
        self.ensure_dirs()
        shot = self.shots[self.idx]
        target_img = OUT_DIR / "images" / f"{shot['name']}.bmp"
        if not target_img.exists():
            shutil.copy(shot["bmp"], target_img)
        new_data = dict(self.data)
        new_data["function_name"] = "labeled_5class"
        new_data["result_data"] = [[p["points"]] for p in self.polys]
        new_data["class_id"] = [p["class_id"] for p in self.polys]
        new_data["verified"] = [bool(p.get("verified", False)) for p in self.polys]
        with open(
            OUT_DIR / "data_json" / f"{shot['name']}_data.json",
            "w", encoding="utf-8",
        ) as f:
            json.dump(new_data, f, indent=2, ensure_ascii=False)
        lines = []
        for p in self.polys:
            cid = p["class_id"]
            if not (0 <= cid < len(CLASSES)):
                continue
            coords = []
            for x, y in p["points"]:
                coords.append(f"{x/IMG_W:.6f}")
                coords.append(f"{y/IMG_H:.6f}")
            lines.append(f"{cid} " + " ".join(coords))
        with open(
            OUT_DIR / "labels" / f"{shot['name']}.txt",
            "w", encoding="utf-8",
        ) as f:
            f.write("\n".join(lines))
        self.write_dataset_yaml()
        self.refresh_progress()

    def write_dataset_yaml(self):
        text = (
            "# Auto-generated by label_tool.py\n"
            f"path: {OUT_DIR.as_posix()}\n"
            "train: images\n"
            "val: images\n"
            "\n"
            "names:\n"
        )
        for i, (en, ko, _) in enumerate(CLASSES):
            text += f"  {i}: {en}  # {ko}\n"
        with open(OUT_DIR / "dataset.yaml", "w", encoding="utf-8") as f:
            f.write(text)


# ============================================================
# Tab 2: InferenceViewer
# ============================================================

class InferenceViewer:
    """YOLO 추론 결과 시각화 탭."""

    def __init__(self, parent):
        self.parent = parent
        self.model = None
        self._loaded_model_path = None
        self.current_image_path = None
        self.predictions = []
        self.gt_polys = []
        self.tk_image = None
        self.scale = 1.0
        self.canvas_items = []
        # OOD anomaly overlay
        self.ood_visible = False
        self.ood_threshold = None
        self.ood_normal_stats = None
        self._ood_proc_running = False

        self.available_models = self._find_models()
        self.available_images = self._find_images()
        self.build_ui()
        # 'O' 키 토글 (Toplevel focus 일 때만)
        try:
            self.parent.bind("<KeyPress-o>", self._on_ood_key)
            self.parent.bind("<KeyPress-O>", self._on_ood_key)
        except Exception:
            pass
        if self.available_images:
            self.load_image(self.available_images[0])

    def _find_models(self):
        models = []
        if RUNS_DIR.exists():
            for d in sorted(RUNS_DIR.iterdir()):
                best = d / "weights" / "best.pt"
                if best.exists():
                    models.append((d.name, best))
        # Standalone weight files in weights/ (e.g. server에서 받아온 best.pt)
        for p in sorted((ROOT_DIR / "weights").glob("*_best.pt")):
            models.append((p.stem, p))
        return models

    def _find_images(self):
        # YOLO 데이터는 outputs/<team>/yolo_v2/ 로 이동됨
        team_name = OUT_DIR.parent.name
        yolo_root = ROOT_DIR / "outputs" / team_name / "yolo_v2"
        sources = [
            DATA_DIR,
            yolo_root / "images" / "train",
            yolo_root / "images" / "val",
        ]
        seen = set()
        out = []
        for src in sources:
            if not src.exists():
                continue
            for p in sorted(src.glob("*.bmp")):
                if p.stem not in seen:
                    seen.add(p.stem)
                    out.append(p)
        return out

    def _find_gt_label_path(self, image_stem):
        team_name = OUT_DIR.parent.name
        yolo_root = ROOT_DIR / "outputs" / team_name / "yolo_v2"
        candidates = [
            yolo_root / "labels" / "train" / f"{image_stem}.txt",
            yolo_root / "labels" / "val" / f"{image_stem}.txt",
            OUT_DIR / "labels" / f"{image_stem}.txt",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def build_ui(self):
        sw = self.parent.winfo_screenwidth()
        sh = self.parent.winfo_screenheight()
        max_w = sw - 460
        max_h = sh - 180
        self.scale = min(max_w / IMG_W, max_h / IMG_H, 1.0)
        cw = int(IMG_W * self.scale)
        ch = int(IMG_H * self.scale)

        ko_font = ("Malgun Gothic", 10)
        ko_bold = ("Malgun Gothic", 11, "bold")
        ko_big = ("Malgun Gothic", 13, "bold")

        self.canvas = tk.Canvas(
            self.parent, width=cw, height=ch, bg="black", cursor="arrow"
        )
        self.canvas.pack(side="left", padx=4, pady=4)
        self.canvas.bind("<FocusIn>", lambda e: None)

        side = tk.Frame(self.parent, padx=8)
        side.pack(side="right", fill="y")

        tk.Label(side, text="YOLO 추론 시각화", font=ko_big, anchor="w").pack(fill="x")
        tk.Frame(side, height=8).pack()

        # Model selector
        tk.Label(side, text="Model", font=ko_bold, anchor="w").pack(fill="x")
        self.model_var = tk.StringVar()
        model_names = [m[0] for m in self.available_models]
        if model_names:
            self.model_var.set(model_names[-1])  # default = latest run
        self.model_dropdown = ttk.Combobox(
            side, textvariable=self.model_var, values=model_names,
            state="readonly", width=30, font=ko_font,
        )
        self.model_dropdown.pack()
        if not model_names:
            tk.Label(side, text="(학습된 모델 없음)", fg="red", font=ko_font).pack()
        tk.Frame(side, height=6).pack()

        # Image selector
        tk.Label(side, text="Image", font=ko_bold, anchor="w").pack(fill="x")
        self.image_var = tk.StringVar()
        image_names = [p.stem for p in self.available_images]
        if image_names:
            self.image_var.set(image_names[0])
        self.image_dropdown = ttk.Combobox(
            side, textvariable=self.image_var, values=image_names,
            state="readonly", width=30, font=ko_font,
        )
        self.image_dropdown.pack()
        self.image_dropdown.bind("<<ComboboxSelected>>", self.on_image_change)
        tk.Frame(side, height=6).pack()

        # Conf threshold
        tk.Label(side, text="Confidence threshold", font=ko_bold, anchor="w").pack(fill="x")
        self.conf_var = tk.DoubleVar(value=0.25)
        conf_slider = tk.Scale(
            side, from_=0.05, to=0.95, resolution=0.05,
            orient="horizontal", variable=self.conf_var, length=280,
            font=ko_font,
        )
        conf_slider.pack()
        tk.Frame(side, height=6).pack()

        # Run button
        self.run_btn = tk.Button(
            side, text="▶ Run Inference", command=self.run_inference,
            font=ko_bold, bg="#3498DB", fg="white", height=2,
        )
        self.run_btn.pack(fill="x")
        tk.Frame(side, height=6).pack()

        # GT toggle
        self.show_gt_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            side, text="Ground Truth 같이 보기 (흰 점선)",
            variable=self.show_gt_var, command=self.refresh_canvas,
            font=ko_font,
        ).pack(anchor="w")

        # Show boxes toggle
        self.show_box_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            side, text="Bounding Box 표시",
            variable=self.show_box_var, command=self.refresh_canvas,
            font=ko_font,
        ).pack(anchor="w")
        tk.Frame(side, height=6).pack()

        # OOD Compute + toggle
        self.ood_btn = tk.Button(
            side, text="🚨 Compute OOD anomaly score",
            command=self.on_compute_ood,
            font=ko_bold, bg="#E67E22", fg="white",
        )
        self.ood_btn.pack(fill="x")
        self.show_ood_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            side, text="OOD overlay 표시 (단축키 O)",
            variable=self.show_ood_var, command=self._on_show_ood_toggle,
            font=ko_font,
        ).pack(anchor="w")
        self.ood_status = tk.Label(
            side, text="(아직 계산 안 됨)", fg="gray", font=ko_font, anchor="w",
        )
        self.ood_status.pack(fill="x")
        tk.Frame(side, height=6).pack()

        # Detection list
        tk.Label(side, text="Predictions", font=ko_bold, anchor="w").pack(fill="x")
        self.det_list = tk.Listbox(
            side, width=32, height=12, font=("Malgun Gothic", 10),
            exportselection=False,
        )
        self.det_list.pack()
        self.det_list.bind("<<ListboxSelect>>", self.on_det_select)
        tk.Frame(side, height=6).pack()

        # Class palette
        tk.Label(side, text="Class colors", font=ko_bold, anchor="w").pack(fill="x")
        for cid, (en, ko, color) in enumerate(CLASSES):
            f = tk.Frame(side)
            f.pack(anchor="w", pady=1)
            tk.Label(
                f, text=f" {cid} ", bg=color, fg="white",
                font=ko_bold, width=4,
            ).pack(side="left")
            tk.Label(f, text=f"  {ko}", font=ko_font).pack(side="left")

        # Status
        self.inf_status = tk.Label(
            self.parent, text="Ready. Select image and click Run.",
            anchor="w", relief="sunken", font=ko_font,
        )
        self.inf_status.pack(side="bottom", fill="x")

    def on_image_change(self, _event):
        name = self.image_var.get()
        for p in self.available_images:
            if p.stem == name:
                self.load_image(p)
                self.predictions = []  # clear stale
                self.update_det_list()
                break

    def load_image(self, image_path):
        self.current_image_path = image_path
        img = Image.open(image_path).convert("RGB")
        cw = int(IMG_W * self.scale)
        ch = int(IMG_H * self.scale)
        img = img.resize((cw, ch), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(img)
        # Load GT if available
        gt_path = self._find_gt_label_path(image_path.stem)
        self.gt_polys = []
        if gt_path:
            with open(gt_path, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 7:
                        continue
                    cid = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    pts = [(coords[i] * IMG_W, coords[i + 1] * IMG_H)
                           for i in range(0, len(coords) - 1, 2)]
                    self.gt_polys.append((cid, pts))
        self.refresh_canvas()
        self.inf_status.config(
            text=f"Loaded: {image_path.stem} (GT polygons: {len(self.gt_polys)})"
        )

    def run_inference(self):
        if not self.available_models:
            messagebox.showerror("No model", "학습된 모델이 없습니다.")
            return
        # Get selected model
        name = self.model_var.get()
        model_path = None
        for n, p in self.available_models:
            if n == name:
                model_path = p
                break
        if model_path is None:
            messagebox.showerror("Error", "Model not found.")
            return

        self.run_btn.config(state="disabled", text="Loading...")
        self.parent.update()
        try:
            from ultralytics import YOLO
            if self.model is None or self._loaded_model_path != model_path:
                self.inf_status.config(text=f"Loading model: {name}...")
                self.parent.update()
                self.model = YOLO(str(model_path))
                self._loaded_model_path = model_path
            self.run_btn.config(text="Inferring...")
            self.parent.update()
            # imgsz는 학습값을 모델 ckpt에서 자동 추출 (없으면 1024 fallback).
            # 추론 imgsz가 학습 imgsz와 다르면 마스크 품질이 떨어짐.
            from label_logic.auto_classify import (
                extract_clean_polygons, _resolve_imgsz,
            )
            imgsz = _resolve_imgsz(self.model, fallback=1024)
            results = self.model(
                str(self.current_image_path),
                conf=self.conf_var.get(),
                imgsz=imgsz, verbose=False, device=0,
            )[0]
            self.predictions = []
            if results.masks is not None and len(results.boxes) > 0:
                # ★ Ultralytics .xy는 strategy='all' 기본 → 다중 컨투어를
                # merge_multi_segment로 이어붙여 자기 교차하는 별 모양 폴리곤을
                # 만듦. 가장 큰 컨투어 1개만 뽑아 자기 교차를 차단.
                polys_np = extract_clean_polygons(results)
                for i in range(len(results.boxes)):
                    cid = int(results.boxes.cls[i].item())
                    conf = float(results.boxes.conf[i].item())
                    box = results.boxes.xyxy[i].cpu().numpy().tolist()
                    if i < len(polys_np):
                        poly = polys_np[i].tolist()
                        self.predictions.append({
                            "class": cid, "conf": conf,
                            "poly": poly, "box": box,
                        })
            self.refresh_canvas()
            self.update_det_list()
            self.inf_status.config(
                text=f"Inference done: {len(self.predictions)} detections (model: {name})"
            )
        except Exception as e:
            messagebox.showerror("Inference Error", str(e))
            self.inf_status.config(text=f"Error: {e}")
        finally:
            self.run_btn.config(state="normal", text="▶ Run Inference")

    def refresh_canvas(self):
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

        # Draw GT (if toggle on)
        if self.show_gt_var.get():
            for cid, pts in self.gt_polys:
                scaled = []
                for x, y in pts:
                    scaled.extend([x * self.scale, y * self.scale])
                self.canvas.create_polygon(
                    scaled, outline="white", fill="",
                    width=2, dash=(6, 3),
                )

        # Draw predictions
        selected_idx = -1
        sel = self.det_list.curselection()
        if sel:
            selected_idx = sel[0]
        for i, pred in enumerate(self.predictions):
            cid = pred["class"]
            color = CLASSES[cid][2] if 0 <= cid < len(CLASSES) else "#888888"
            poly = pred["poly"]
            scaled = []
            for x, y in poly:
                scaled.extend([x * self.scale, y * self.scale])
            outline = SELECTED_OUTLINE if i == selected_idx else color
            width = 4 if i == selected_idx else 2
            dash = None
            ascore = pred.get("anomaly_score")
            is_ood = (self.ood_visible and ascore is not None
                      and self.ood_threshold is not None
                      and ascore > self.ood_threshold)
            if is_ood:
                outline = OOD_OUTLINE_COLOR
                width = 4
                dash = (6, 4)
            poly_kwargs = dict(outline=outline, fill=color, stipple="gray25", width=width)
            if dash is not None:
                poly_kwargs["dash"] = dash
            self.canvas.create_polygon(scaled, **poly_kwargs)
            # Box (optional)
            if self.show_box_var.get():
                x1, y1, x2, y2 = pred["box"]
                self.canvas.create_rectangle(
                    x1 * self.scale, y1 * self.scale,
                    x2 * self.scale, y2 * self.scale,
                    outline=color, width=1, dash=(2, 2),
                )
            # Label text
            cx = sum(p[0] for p in poly) / len(poly)
            cy = sum(p[1] for p in poly) / len(poly)
            label = f"{CLASSES[cid][0] if 0 <= cid < len(CLASSES) else '?'} {pred['conf']:.2f}"
            self.canvas.create_text(
                cx * self.scale, cy * self.scale,
                text=label, fill="white",
                font=("Arial", 11, "bold"),
            )
            # OOD score badge — class label 아래 (overlay ON + 점수 있을 때만)
            if self.ood_visible and ascore is not None:
                badge_color = OOD_OUTLINE_COLOR if is_ood else "#7FB069"
                self.canvas.create_text(
                    cx * self.scale, cy * self.scale + 18,
                    text=f"AS {ascore:.3f}",
                    fill=badge_color,
                    font=("Arial", 10, "bold"),
                )

    def update_det_list(self):
        self.det_list.delete(0, tk.END)
        for i, pred in enumerate(self.predictions):
            cid = pred["class"]
            name = CLASSES[cid][1] if 0 <= cid < len(CLASSES) else f"?{cid}"
            ascore = pred.get("anomaly_score")
            suffix = f"  AS={ascore:.3f}" if ascore is not None else ""
            self.det_list.insert(
                tk.END,
                f"#{i:02d}  {name:<10}  {pred['conf']:.3f}{suffix}",
            )

    def on_det_select(self, _event):
        self.refresh_canvas()

    # ── OOD anomaly overlay ────────────────────────────────
    def _on_ood_key(self, _event):
        if not self.predictions:
            return
        self.show_ood_var.set(not self.show_ood_var.get())
        self._on_show_ood_toggle()

    def _on_show_ood_toggle(self):
        self.ood_visible = bool(self.show_ood_var.get())
        n_scored = sum(1 for p in self.predictions if p.get("anomaly_score") is not None)
        if self.ood_visible and n_scored == 0:
            self.inf_status.config(text="OOD: 점수 없음 — 'Compute OOD' 먼저 누르세요.")
        else:
            state = "ON" if self.ood_visible else "OFF"
            self.inf_status.config(text=f"OOD overlay {state} (점수 {n_scored}/{len(self.predictions)})")
        self.refresh_canvas()

    def on_compute_ood(self):
        if self._ood_proc_running:
            messagebox.showinfo("OOD", "이미 진행 중입니다.")
            return
        if not self.predictions:
            messagebox.showinfo("OOD", "예측 결과가 없습니다. 먼저 ▶ Run Inference 를 누르세요.")
            return
        if self.current_image_path is None:
            return
        if not ANOMALY_ENV_PYTHON.exists():
            messagebox.showerror(
                "OOD", f"anomaly env python 못 찾음:\n{ANOMALY_ENV_PYTHON}",
            )
            return
        if not OOD_SCORE_SCRIPT.exists():
            messagebox.showerror("OOD", f"score_polygons.py 없음:\n{OOD_SCORE_SCRIPT}")
            return

        import tempfile
        tmp_dir = Path(tempfile.gettempdir()) / "inference_viewer_ood"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        stem = self.current_image_path.stem
        polys_path  = tmp_dir / f"{stem}_polys.json"
        scores_path = tmp_dir / f"{stem}_scores.json"

        payload = []
        for i, pred in enumerate(self.predictions):
            poly = pred.get("poly") or []
            if len(poly) < 3:
                continue
            payload.append({"idx": i, "points": [list(pt) for pt in poly]})
        if not payload:
            messagebox.showinfo("OOD", "유효한 폴리곤 없음.")
            return
        polys_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        self._ood_proc_running = True
        self.ood_btn.config(state="disabled", text="⏳ OOD 계산 중...")
        self.ood_status.config(text=f"⏳ {len(payload)} 폴리곤, 최대 30초", fg="#E67E22")
        self.parent.update_idletasks()

        import threading, subprocess
        def worker():
            try:
                cmd = [
                    str(ANOMALY_ENV_PYTHON), str(OOD_SCORE_SCRIPT),
                    "--bmp",   str(self.current_image_path),
                    "--polys", str(polys_path),
                    "--out",   str(scores_path),
                ]
                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"
                res = subprocess.run(
                    cmd, capture_output=True, text=True, env=env,
                    encoding="utf-8", errors="replace",
                    creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
                )
                if res.returncode != 0:
                    err = (res.stderr or res.stdout or "")[-600:]
                    self.parent.after(0, lambda: self._on_compute_ood_error(err))
                    return
                result = json.loads(scores_path.read_text(encoding="utf-8"))
                self.parent.after(0, lambda: self._on_compute_ood_done(result))
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                self.parent.after(0, lambda: self._on_compute_ood_error(msg))

        threading.Thread(target=worker, daemon=True).start()

    def _on_compute_ood_done(self, result):
        self._ood_proc_running = False
        self.ood_btn.config(state="normal", text="🚨 Compute OOD anomaly score")
        self.ood_threshold = float(result.get("threshold", 0.0))
        self.ood_normal_stats = result.get("normal_stats", {})
        scored = result.get("scores", [])
        applied = 0
        for entry in scored:
            i = entry.get("idx", -1)
            s = entry.get("score")
            if 0 <= i < len(self.predictions) and s is not None:
                self.predictions[i]["anomaly_score"] = float(s)
                applied += 1
        # auto-enable overlay
        self.show_ood_var.set(True)
        self.ood_visible = True
        n_high = sum(1 for p in self.predictions
                      if p.get("anomaly_score") is not None
                      and p["anomaly_score"] > self.ood_threshold)
        self.ood_status.config(
            text=f"✓ {applied} 점수, thr={self.ood_threshold:.3f}, >thr: {n_high}",
            fg="#27AE60",
        )
        self.update_det_list()
        self.refresh_canvas()

    def _on_compute_ood_error(self, err_msg):
        self._ood_proc_running = False
        self.ood_btn.config(state="normal", text="🚨 Compute OOD anomaly score")
        self.ood_status.config(text=f"✗ {err_msg[:80]}", fg="red")
        messagebox.showerror("OOD compute error", err_msg)


# ============================================================
# Help modal — replaces 22-line keymap text in side panel
# ============================================================

KEYMAP_TEXT = (
    "■ 라벨링 — 폴리곤\n"
    "  0~4    클래스 부여 (선택된 폴리곤에)\n"
    "  Click  폴리곤 선택\n"
    "  W      새 폴리곤 그리기 시작\n"
    "  E      꼭지점 수정 모드\n"
    "    └ 점 드래그: 이동\n"
    "    └ 변(edge) 클릭: 점 추가\n"
    "    └ 점 우클릭: 점 삭제\n"
    "  D      선택 폴리곤 삭제\n"
    "  Space  선택 폴리곤 verified 플래그 토글 (△ ↔ ✓)\n"
    "  Enter  그리기 종료\n"
    "  Esc    그리기/모드 취소\n"
    "  Tab    다음 미라벨 폴리곤\n"
    "\n"
    "■ 모드 진입 (캔버스 영역)\n"
    "  A      SAM2 Click 모드 (1점 클릭 → 자동 외곽선)\n"
    "  B      박스 ROI 그리기 (bin 영역 사각형)\n"
    "  V      3D View 별창 (Open3D)\n"
    "\n"
    "■ 캔버스 오버레이 토글\n"
    "  H      Height (높이) 오버레이 ON/OFF\n"
    "  M      Normal (법선) 오버레이 ON/OFF\n"
    "  K      Pickability heatmap 오버레이 ON/OFF\n"
    "  O      OOD anomaly score 오버레이 ON/OFF (Compute OOD 실행 후)\n"
    "\n"
    "■ 작업\n"
    "  Ctrl+R  Prelabel 실행 (Watershed 또는 Auto-Classify)\n"
    "  G       픽 계산 — 수동 트리거 (자동 계산 없음)\n"
    "  🚨 버튼 Compute OOD — Supersimplenet score 계산 (subprocess)\n"
    "\n"
    "■ 저장 / 이동\n"
    "  S        저장\n"
    "  U        Undo\n"
    "  N / P    다음/이전 샷\n"
    "  Tab      다음 미라벨 폴리곤\n"
    "\n"
    "■ 데이터셋 분할\n"
    "  Y     YOLO split 토글 (AUTO → TRAIN → VAL)\n"
    "\n"
    "■ 별창 (Toplevel popup)\n"
    "  Ctrl+I         🔍 Inference Viewer\n"
    "                  (학습된 YOLO best.pt 마스크 시각화)\n"
    "                  내부 단축키: O (OOD overlay 토글), 🚨 버튼 (Compute OOD)\n"
    "  Ctrl+Shift+R   📋 Roboflow 라벨 비교\n"
    "                  (팀원 라벨과 우리 v2 폴리곤 overlay)\n"
    "\n"
    "■ 줌 / Pan\n"
    "  + / -        줌 인/아웃\n"
    "  F            Fit (100%)\n"
    "  Wheel        마우스 위치 기준 줌\n"
    "  Shift+Wheel  수평 스크롤\n"
    "  ← ↑ ↓ →      Pan (이동)\n"
    "  Mid-드래그    Pan\n"
    "\n"
    "■ UI\n"
    "  Ctrl+B  우측 사이드 패널 토글\n"
    "  Ctrl+L  좌측 QUICK KEYS 패널 토글\n"
    "\n"
    "■ 도움말 / 저장\n"
    "  F1 / Ctrl+H   이 키맵 보기\n"
    "  Q             현재 샷 저장 (창은 안 닫힘 — 닫으려면 X 버튼)\n"
    "\n"
    "■ 3D View 별창 (V로 열린 창의 자체 단축키)\n"
    "  P       pivot → 화면 중앙 객체\n"
    "  R       pivot → scene 중앙\n"
    "  T       라벨 하이라이트 ON/OFF\n"
    "  K       하이라이트 색상 모드 토글 (class ↔ heatmap)\n"
    "  M       📐 Recovery 메시 (반투명 표면) ON/OFF\n"
    "  G       🎯 Picks 표시 ON/OFF (초기 OFF — 수동 트리거)\n"
    "  C       🦾 화면 중앙 픽의 충돌 박스 표시 (토글)\n"
    "  L       Manual pick 라벨 모드 진입\n"
    "          └ Click: 픽 배치 / R: 리셋 / V: 검증 / Enter ×2: 저장 / Esc: 취소\n"
    "  Q/Esc   창 닫기"
)


class ToolTip:
    """Lightweight hover tooltip — pure Tkinter (no extra deps)."""

    def __init__(self, widget, text, delay=400):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tip = None
        self._after_id = None
        widget.bind("<Enter>", self._schedule)
        widget.bind("<Leave>", self._hide)
        widget.bind("<ButtonPress>", self._hide)

    def _schedule(self, _e):
        self._cancel()
        self._after_id = self.widget.after(self.delay, self._show)

    def _cancel(self):
        if self._after_id:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self):
        if self.tip:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(
            self.tip, text=self.text,
            bg="#FFFFE0", fg="#000000",
            relief="solid", borderwidth=1,
            font=("Malgun Gothic", 9),
            padx=6, pady=3, justify="left",
        ).pack()

    def _hide(self, _e=None):
        self._cancel()
        if self.tip:
            self.tip.destroy()
            self.tip = None


class HelpModal:
    """Modal window showing keyboard shortcuts."""

    @staticmethod
    def show(parent):
        win = tk.Toplevel(parent)
        win.title("단축키")
        win.transient(parent)
        tk.Label(
            win, text=KEYMAP_TEXT, justify="left",
            font=("Consolas", 10), padx=20, pady=20,
        ).pack()
        tk.Button(win, text="닫기", command=win.destroy,
                  font=("Malgun Gothic", 10)).pack(pady=(0, 12))
        win.focus_set()


# ============================================================
# Roboflow Label Review Viewer — compare our v2 labels vs Roboflow re-labeled
# ============================================================

class RoboflowReviewViewer:
    """Toplevel: side-by-side polygon compare between our labels and Roboflow.

    Reads:
      - Roboflow labels from _external/roboflow_260508/train/labels/<stem>.txt
      - Our labels   from data/team1/labeled/labels/<stem>.txt
      - Images       from data/team1/labeled/images/<stem>.bmp (byte-identical)

    Roboflow has 7 classes (alphabetical + 'object'/'other'); our v2 has 5.
    """

    RF_CLASSES = ["bottle", "haribo", "mango", "object", "other", "pencilcase", "tincase"]
    RF_COLORS = [
        "#E74C3C",  # bottle
        "#F1C40F",  # haribo
        "#FF8C42",  # mango
        "#9B59B6",  # object
        "#7F8C8D",  # other
        "#3498DB",  # pencilcase
        "#16A085",  # tincase
    ]

    def __init__(self, parent):
        self.parent = parent
        self.rf_label_root = ROOT_DIR / "_external" / "roboflow_260508" / "train" / "labels"
        self.our_label_root = OUT_DIR / "labels"
        self.image_root = OUT_DIR / "images"

        self.available_images = self._find_images()
        self.rf_polys = []   # list of (cid_rf, [(x,y),...])
        self.our_polys = []  # list of (cid_ours, [(x,y),...])
        self.tk_image = None
        self.current_image_path = None

        sw = parent.winfo_screenwidth()
        sh = parent.winfo_screenheight()
        max_w = sw - 380
        max_h = sh - 160
        self.scale = min(max_w / IMG_W, max_h / IMG_H, 1.0)

        self.build_ui()
        if self.available_images:
            self.image_var.set(self.available_images[0].stem)
            self.load_image(self.available_images[0])

    def _find_images(self):
        if not self.rf_label_root.exists():
            return []
        stems = sorted(p.stem for p in self.rf_label_root.glob("*.txt"))
        out = []
        for stem in stems:
            img = self.image_root / f"{stem}.bmp"
            if img.exists():
                out.append(img)
        return out

    def build_ui(self):
        ko_font = ("Malgun Gothic", 10)
        ko_bold = ("Malgun Gothic", 11, "bold")
        ko_big = ("Malgun Gothic", 13, "bold")

        cw = int(IMG_W * self.scale)
        ch = int(IMG_H * self.scale)

        self.canvas = tk.Canvas(self.parent, width=cw, height=ch, bg="black")
        self.canvas.pack(side="left", padx=4, pady=4)

        side = tk.Frame(self.parent, padx=8)
        side.pack(side="right", fill="y")

        tk.Label(side, text="📋 Roboflow 라벨 비교", font=ko_big, anchor="w").pack(fill="x")
        tk.Frame(side, height=8).pack()

        tk.Label(side, text="Image", font=ko_bold, anchor="w").pack(fill="x")
        self.image_var = tk.StringVar()
        names = [p.stem for p in self.available_images]
        self.image_dropdown = ttk.Combobox(
            side, textvariable=self.image_var, values=names,
            state="readonly", width=32, font=ko_font,
        )
        self.image_dropdown.pack()
        self.image_dropdown.bind("<<ComboboxSelected>>", self._on_image_change)
        if not names:
            tk.Label(side, text="(Roboflow 라벨 없음)", fg="red", font=ko_font).pack()

        nav = tk.Frame(side)
        nav.pack(fill="x", pady=4)
        tk.Button(nav, text="◀ Prev", command=lambda: self._step(-1),
                  font=ko_font).pack(side="left", expand=True, fill="x")
        tk.Button(nav, text="Next ▶", command=lambda: self._step(1),
                  font=ko_font).pack(side="left", expand=True, fill="x")

        tk.Frame(side, height=8).pack()
        tk.Label(side, text="Overlay", font=ko_bold, anchor="w").pack(fill="x")
        self.show_rf_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            side, text="Roboflow 라벨 (실선 색상)",
            variable=self.show_rf_var, command=self.refresh_canvas,
            font=ko_font,
        ).pack(anchor="w")
        self.show_ours_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            side, text="우리 v2 라벨 (흰 점선)",
            variable=self.show_ours_var, command=self.refresh_canvas,
            font=ko_font,
        ).pack(anchor="w")

        tk.Frame(side, height=8).pack()
        self.count_label = tk.Label(side, text="", font=ko_font, anchor="w", justify="left")
        self.count_label.pack(fill="x")

        tk.Frame(side, height=8).pack()
        tk.Label(side, text="Roboflow 클래스", font=ko_bold, anchor="w").pack(fill="x")
        for cid, name in enumerate(self.RF_CLASSES):
            f = tk.Frame(side)
            f.pack(anchor="w", pady=1)
            tk.Label(
                f, text=f" {cid} ", bg=self.RF_COLORS[cid], fg="white",
                font=ko_bold, width=4,
            ).pack(side="left")
            tk.Label(f, text=f"  {name}", font=ko_font).pack(side="left")

        self.status = tk.Label(
            self.parent, text="", anchor="w", relief="sunken", font=ko_font,
        )
        self.status.pack(side="bottom", fill="x")

    def _on_image_change(self, _event):
        name = self.image_var.get()
        for p in self.available_images:
            if p.stem == name:
                self.load_image(p)
                return

    def _step(self, delta):
        if not self.available_images:
            return
        cur = self.image_var.get()
        idx = 0
        for i, p in enumerate(self.available_images):
            if p.stem == cur:
                idx = i
                break
        idx = (idx + delta) % len(self.available_images)
        self.image_var.set(self.available_images[idx].stem)
        self.load_image(self.available_images[idx])

    def _read_polys(self, path):
        out = []
        if not path.exists():
            return out
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7:
                    continue
                try:
                    cid = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                except ValueError:
                    continue
                pts = [(coords[i] * IMG_W, coords[i + 1] * IMG_H)
                       for i in range(0, len(coords) - 1, 2)]
                if len(pts) >= 3:
                    out.append((cid, pts))
        return out

    def load_image(self, image_path):
        self.current_image_path = image_path
        img = Image.open(image_path).convert("RGB")
        cw = int(IMG_W * self.scale)
        ch = int(IMG_H * self.scale)
        img = img.resize((cw, ch), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(img)

        self.rf_polys = self._read_polys(self.rf_label_root / f"{image_path.stem}.txt")
        self.our_polys = self._read_polys(self.our_label_root / f"{image_path.stem}.txt")

        rf_breakdown = Counter(cid for cid, _ in self.rf_polys)
        rf_lines = ", ".join(f"{self.RF_CLASSES[c]}={n}" for c, n in sorted(rf_breakdown.items()))
        our_breakdown = Counter(cid for cid, _ in self.our_polys)
        our_lines = ", ".join(f"{CLASSES[c][0]}={n}" for c, n in sorted(our_breakdown.items()))

        self.count_label.config(
            text=f"Roboflow: {len(self.rf_polys)} polys\n  {rf_lines or '(none)'}\n"
                 f"우리 v2:  {len(self.our_polys)} polys\n  {our_lines or '(none)'}"
        )
        self.refresh_canvas()
        self.status.config(text=f"Loaded: {image_path.stem}")

    def refresh_canvas(self):
        self.canvas.delete("all")
        if self.tk_image:
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

        if self.show_rf_var.get():
            for cid, pts in self.rf_polys:
                color = self.RF_COLORS[cid] if 0 <= cid < len(self.RF_COLORS) else "#FFFFFF"
                scaled = []
                for x, y in pts:
                    scaled.extend([x * self.scale, y * self.scale])
                self.canvas.create_polygon(
                    scaled, outline=color, fill="", width=2,
                )

        if self.show_ours_var.get():
            for cid, pts in self.our_polys:
                scaled = []
                for x, y in pts:
                    scaled.extend([x * self.scale, y * self.scale])
                self.canvas.create_polygon(
                    scaled, outline="white", fill="", width=2, dash=(6, 3),
                )



# ============================================================
# Main App — single-pane workspace (Notebook removed)
# ============================================================

class App:
    """Single-pane workspace: canvas left + side panel right (collapsible).

    InferenceViewer was a separate tab in v1. It will be absorbed into
    SidePanel's Tools / Layers widgets in upcoming tasks.
    """

    def __init__(self):
        # Use customtkinter if available — gets modern look + system theme
        self.root = ctk.CTk() if _HAS_CTK else tk.Tk()
        self.root.title("Bin Picking Label Tool")
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"{min(sw - 50, 1600)}x{min(sh - 80, 950)}")
        # Industrial dark palette
        self.root.configure(bg=COLOR_BG_MAIN)
        # Configure ttk styles for Combobox to fit dark theme
        # Use 'clam' theme — supports color overrides much better than 'vista'
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            style.theme_use("default")
        style.configure(
            "TCombobox",
            fieldbackground=COLOR_BG_INPUT,
            background=COLOR_BG_INPUT,
            foreground=COLOR_TEXT,
            arrowcolor=COLOR_TEXT,
            bordercolor=COLOR_BORDER,
            lightcolor=COLOR_BG_INPUT,
            darkcolor=COLOR_BG_INPUT,
            selectbackground=COLOR_BG_INPUT,
            selectforeground=COLOR_TEXT,
            insertcolor=COLOR_TEXT,
        )
        # Critical: state map for readonly comboboxes (our dropdowns)
        style.map(
            "TCombobox",
            fieldbackground=[
                ("readonly", COLOR_BG_INPUT),
                ("active", COLOR_BG_HOVER),
            ],
            background=[
                ("readonly", COLOR_BG_INPUT),
                ("active", COLOR_BG_HOVER),
            ],
            foreground=[("readonly", COLOR_TEXT)],
            selectbackground=[("readonly", COLOR_BG_INPUT)],
            selectforeground=[("readonly", COLOR_TEXT)],
            arrowcolor=[("active", COLOR_ACCENT)],
        )
        # Listbox (dropdown popup)
        self.root.option_add("*TCombobox*Listbox.background", COLOR_BG_PANEL)
        self.root.option_add("*TCombobox*Listbox.foreground", COLOR_TEXT)
        self.root.option_add(
            "*TCombobox*Listbox.selectBackground", COLOR_ACCENT,
        )
        self.root.option_add(
            "*TCombobox*Listbox.selectForeground", "#000000",
        )

        self.labeler = LabelTool(self.root)

        # Ctrl+B: toggle right side panel
        self.root.bind_all("<Control-b>", self._toggle_side)
        # Ctrl+L: toggle left QUICK KEYS panel
        self.root.bind_all(
            "<Control-l>",
            lambda _e: self.labeler.toggle_left_panel(),
        )
        # Action shortcuts
        self.root.bind_all(
            "<Control-r>", lambda _e: self.labeler.on_prelabel(),
        )
        # Ctrl+I: open YOLO Inference Viewer (Toplevel)
        self.root.bind_all(
            "<Control-i>", lambda _e: self.labeler.open_inference_viewer(),
        )
        # Ctrl+Shift+R: open Roboflow Label Review (Toplevel)
        # 명시적 syntax — 일부 Tk에서 <Control-R>(대문자)이 Shift+R로 잘 인식 안 됨
        self.root.bind_all(
            "<Control-Shift-KeyPress-R>",
            lambda _e: self.labeler.open_roboflow_review(),
        )
        self.root.bind_all(
            "<KeyPress-b>", self._maybe_box_roi,
        )
        self.root.bind_all(
            "<KeyPress-v>", self._maybe_3d_view,
        )
        self.root.bind_all(
            "<KeyPress-g>", self._maybe_compute_picks,
        )
        self.root.bind_all(
            "<KeyPress-a>", self._maybe_sam2,
        )
        # F1 / Ctrl+H: show keymap
        self.root.bind_all("<F1>", lambda _e: HelpModal.show(self.root))
        self.root.bind_all("<Control-h>", lambda _e: HelpModal.show(self.root))

        # Window resize → canvas auto-fits (responsive)
        self._resize_pending = None
        self.root.bind("<Configure>", self._on_resize)

    def _on_resize(self, event):
        if event.widget != self.root:
            return
        # Throttle: only refit on idle
        if self._resize_pending is not None:
            self.root.after_cancel(self._resize_pending)
        self._resize_pending = self.root.after(120, self._do_refit)

    def _do_refit(self):
        self._resize_pending = None
        if hasattr(self.labeler, "_refit_canvas_to_window"):
            self.labeler._refit_canvas_to_window()

    def _maybe_box_roi(self, event):
        # Single-key shortcuts only fire when canvas has focus, not text inputs
        if str(event.widget).startswith(".") and "entry" in str(event.widget).lower():
            return
        if self.labeler.mode == "draw":
            return   # don't conflict with vertex insertion
        self.labeler.on_box_roi_button()

    def _maybe_3d_view(self, event):
        if str(event.widget).startswith(".") and "entry" in str(event.widget).lower():
            return
        if self.labeler.mode == "draw":
            return
        self.labeler.on_open_3d_view()

    def _maybe_compute_picks(self, event):
        if str(event.widget).startswith(".") and "entry" in str(event.widget).lower():
            return
        if self.labeler.mode in ("draw", "edit", "box_roi", "sam2"):
            return
        self.labeler.on_compute_picks()

    def _maybe_sam2(self, event):
        if str(event.widget).startswith(".") and "entry" in str(event.widget).lower():
            return
        # 'A' inside sam2 mode is handled by labeler.on_key (exit)
        if self.labeler.mode in ("draw", "edit", "box_roi", "sam2"):
            return
        self.labeler.on_sam2_button()

    def _toggle_side(self, _event):
        side = getattr(self.labeler, "side", None)
        if side is None:
            return
        if side.winfo_viewable():
            side.pack_forget()
        else:
            side.pack(side="right", fill="y")

    def run(self):
        self.root.mainloop()


def main():
    app = App()
    app.run()


if __name__ == "__main__":
    main()
