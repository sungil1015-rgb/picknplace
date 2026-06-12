"""CMES picknplace 전체 추론 파이프라인 시각화 — 회사 발표용.

흐름: Zivid 빈 스캔 → 데이터 변환 → gRPC → Mask2Former → DinoV2 KNN
         → DefectRegistry → PatchCore-lite/Cascade → GraspPriority
         → SuctionPipeline → Calibration → 로봇 명령

좌→우 2 row × 5 col. 신경망 박스에는 layer stack 미니 일러스트 포함.

Usage:
    python scripts/plot_pipeline_overview.py
산출물: outputs/figs/pipeline_overview_ko.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Polygon, Circle
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[1]

# ─── 한국어 폰트 등록 ──────────────────────────────────────────────────────────
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
fm.fontManager.addfont(FONT_PATH)
fm.fontManager.addfont(FONT_BOLD_PATH)
FP_REG = FontProperties(fname=FONT_PATH)
FP_BOLD = FontProperties(fname=FONT_BOLD_PATH)
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = FP_REG.get_name()

# ─── 색상 팔레트 (Material design) ────────────────────────────────────────────
C_HW = "#546E7A"        # 회색 — 하드웨어/IO
C_HW_LIGHT = "#CFD8DC"
C_PRE = "#78909C"
C_PRE_LIGHT = "#ECEFF1"
C_NN1 = "#1565C0"       # 파랑 — Mask2Former
C_NN1_LIGHT = "#E3F2FD"
C_NN2 = "#6A1B9A"       # 보라 — DinoV2 classifier
C_NN2_LIGHT = "#F3E5F5"
C_BR = "#F9A825"        # 황색 — DefectRegistry 분기
C_BR_LIGHT = "#FFF8E1"
C_DET = "#E65100"       # 주황 — PatchCore/Cascade
C_DET_LIGHT = "#FFF3E0"
C_LOG = "#00838F"       # 청록 — Grasp/Suction
C_LOG_LIGHT = "#E0F7FA"
C_CAL = "#2E7D32"       # 녹색 — 캘리브
C_CAL_LIGHT = "#E8F5E9"
C_ROBOT = "#C62828"     # 적색 — 로봇
C_ROBOT_LIGHT = "#FFEBEE"
C_TEXT = "#212121"
C_SUB = "#455A64"
C_ARROW = "#37474F"

# ─── 레이아웃 좌표계 ──────────────────────────────────────────────────────────
FIG_W, FIG_H = 22, 13.2        # inches
DPI = 150
XLIM = (0, 22)
YLIM = (0, 13.2)

# 박스 크기/간격
BOX_W = 3.7
BOX_H = 4.7
ROW_GAP = 0.95
COL_GAP = 0.35
HEADER_H = 0.95     # 박스 상단 header strip
FOOTER_H = 0.85     # 박스 하단 footer (in/out 텐서)

START_X = (22 - (5 * BOX_W + 4 * COL_GAP)) / 2

TITLE_H = 0.95
LEGEND_RESERVE = 1.1
START_Y_TOP = YLIM[1] - TITLE_H - 0.5 - BOX_H
START_Y_BOT = START_Y_TOP - ROW_GAP - BOX_H


# ─── Helper: 박스 그리기 ──────────────────────────────────────────────────────
def draw_box(ax, x, y, w, h, color_edge, color_fill, alpha=1.0, lw=1.4):
    """둥근 박스: header strip (진한 색) + body (옅은 색) + footer (옅은 회색).

    설계:
      [y+h - HEADER_H, y+h] : header strip (color_edge 진한 색)
      [y + FOOTER_H, y+h - HEADER_H] : body (color_fill 옅은 색)
      [y, y + FOOTER_H] : footer (옅은 회색, in/out 텐서 표기)
    """
    # 그림자
    shadow = FancyBboxPatch(
        (x + 0.07, y - 0.09), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        linewidth=0, facecolor="#000000", alpha=0.10, zorder=1,
    )
    ax.add_patch(shadow)
    # 외곽 둥근 박스 (전체) — color_edge 진한 색 fill
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        linewidth=lw, edgecolor=color_edge, facecolor=color_edge,
        alpha=alpha, zorder=2,
    ))
    # body 영역 (header 아래 + footer 위) — color_fill 옅은 색으로 덮음
    body_y_bot = y + FOOTER_H
    body_y_top = y + h - HEADER_H
    body_h = body_y_top - body_y_bot
    ax.add_patch(Rectangle(
        (x + 0.06, body_y_bot), w - 0.12, body_h,
        facecolor=color_fill, edgecolor="none", zorder=3,
    ))
    # footer 영역 — 옅은 회색
    ax.add_patch(Rectangle(
        (x + 0.06, y + 0.06), w - 0.12, FOOTER_H - 0.06,
        facecolor="#F5F5F5", edgecolor="none", zorder=3,
    ))
    # 구분선 (footer 위)
    ax.plot([x + 0.18, x + w - 0.18], [body_y_bot, body_y_bot],
            color=color_edge, lw=0.6, alpha=0.5, zorder=4)


def draw_header(ax, x, y_box_top, w, num, title_ko, subtitle_en, color):
    """박스 상단 header strip 위에 번호 + 한국어 제목 + 영문 부제 그리기.

    y_box_top: 박스의 top y 좌표.
    Header strip 영역: [y_box_top - HEADER_H, y_box_top].
    """
    strip_mid_y = y_box_top - HEADER_H * 0.43        # 한국어 제목 라인
    sub_y = y_box_top - HEADER_H * 0.78              # 영문 부제 라인
    # number circle (좌측)
    cx, cy = x + 0.42, y_box_top - HEADER_H * 0.5
    circ = Circle((cx, cy), 0.27, facecolor="white",
                  edgecolor=color, lw=2.0, zorder=4)
    ax.add_patch(circ)
    ax.text(cx, cy, str(num), ha="center", va="center", color=color,
            fontsize=12, fontproperties=FP_BOLD, zorder=5)
    # 한국어 제목 — header strip의 진한 색 위, 흰색 글자
    title_cx = x + (w + 0.7) / 2 + 0.05
    ax.text(title_cx, strip_mid_y, title_ko, ha="center", va="center",
            color="white", fontsize=13.5, fontproperties=FP_BOLD, zorder=5)
    # 영문 부제 — 흰색 옅게
    ax.text(title_cx, sub_y, subtitle_en, ha="center", va="center",
            color="white", fontsize=9.5, fontproperties=FP_REG, zorder=5,
            style="italic", alpha=0.95)


def draw_footer(ax, x, y, w, h, in_label, out_label):
    """박스 하단 footer 영역: 입력 / 출력 텐서 표기."""
    # in
    fy1 = y + FOOTER_H * 0.62
    ax.text(x + 0.22, fy1, "in", ha="left", va="center", color=C_SUB,
            fontsize=8.2, fontproperties=FP_BOLD)
    ax.text(x + 0.55, fy1, in_label, ha="left", va="center", color=C_TEXT,
            fontsize=8.5, family="monospace")
    # out
    fy2 = y + FOOTER_H * 0.22
    ax.text(x + 0.22, fy2, "out", ha="left", va="center", color=C_SUB,
            fontsize=8.2, fontproperties=FP_BOLD)
    ax.text(x + 0.62, fy2, out_label, ha="left", va="center", color=C_TEXT,
            fontsize=8.5, family="monospace")


# ─── Helper: 미니 일러스트들 ──────────────────────────────────────────────────
def illust_zivid(ax, x, y, w, h):
    """카메라 + bin + 객체들."""
    cx = x + w / 2
    # 카메라 본체
    cam_w, cam_h = 1.2, 0.55
    ax.add_patch(Rectangle((cx - cam_w / 2, y + h - 0.7), cam_w, cam_h,
                           facecolor="#37474F", edgecolor="black", lw=0.8, zorder=4))
    # 렌즈
    ax.add_patch(Circle((cx, y + h - 0.45), 0.18, facecolor="#90A4AE",
                        edgecolor="black", lw=0.8, zorder=5))
    # 광선 (적색 + IR)
    for dx in [-0.35, -0.18, 0.0, 0.18, 0.35]:
        ax.plot([cx, cx + dx], [y + h - 0.65, y + 0.4],
                color="#E91E63", lw=0.6, alpha=0.35, zorder=3)
    # bin (trapezoid)
    bin_top_l, bin_top_r = cx - 1.45, cx + 1.45
    bin_bot_l, bin_bot_r = cx - 1.15, cx + 1.15
    bin_top_y, bin_bot_y = y + 0.95, y + 0.2
    bin_poly = Polygon(
        [(bin_top_l, bin_top_y), (bin_top_r, bin_top_y),
         (bin_bot_r, bin_bot_y), (bin_bot_l, bin_bot_y)],
        closed=True, facecolor="#ECEFF1", edgecolor="#37474F", lw=1.2, zorder=3,
    )
    ax.add_patch(bin_poly)
    # 빈 안 객체들
    objs = [
        (cx - 0.75, y + 0.35, 0.5, 0.22, "#FFA726"),  # haribo
        (cx - 0.1, y + 0.32, 0.45, 0.28, "#42A5F5"),  # metal_case
        (cx + 0.45, y + 0.4, 0.38, 0.22, "#9CCC65"),  # pencil
        (cx - 0.55, y + 0.6, 0.32, 0.18, "#EF5350"),  # haribo 2
    ]
    for ox, oy, ow, oh, oc in objs:
        ax.add_patch(Rectangle((ox, oy), ow, oh, facecolor=oc,
                               edgecolor="#263238", lw=0.6, zorder=4))


def illust_convert(ax, x, y, w, h):
    """PLY → 3개 파일 아이콘."""
    cx = x + w / 2
    # PLY input
    ax.add_patch(Rectangle((cx - 1.4, y + h - 0.75), 0.95, 0.55,
                           facecolor="#CFD8DC", edgecolor="#37474F", lw=1.0))
    ax.text(cx - 0.92, y + h - 0.48, ".PLY", ha="center", va="center",
            fontsize=9, fontproperties=FP_BOLD, color=C_TEXT)
    ax.text(cx - 0.92, y + h - 0.93, "organized", ha="center", va="center",
            fontsize=7, color=C_SUB, style="italic")
    # arrow
    ax.annotate("", xy=(cx - 0.05, y + h - 0.45), xytext=(cx - 0.4, y + h - 0.45),
                arrowprops=dict(arrowstyle="->", color=C_ARROW, lw=1.2))
    # 3개 파일
    files = [("rgb.png", "BGR uint8", "#FFAB91"),
             ("depth.png", "uint16 mm", "#80CBC4"),
             ("normal.bin", "fp32 raw", "#CE93D8")]
    fy = y + h - 1.55
    for i, (fname, dtype, color) in enumerate(files):
        fx = cx - 1.2 + i * 1.2
        ax.add_patch(Rectangle((fx, fy), 0.95, 0.45,
                               facecolor=color, edgecolor="#37474F", lw=0.9))
        ax.text(fx + 0.475, fy + 0.28, fname, ha="center", va="center",
                fontsize=7.5, fontproperties=FP_BOLD)
        ax.text(fx + 0.475, fy + 0.08, dtype, ha="center", va="center",
                fontsize=6.5, color=C_SUB, family="monospace")
    # gRPC 라벨
    ax.text(cx, y + 1.05, "gRPC ./img/",
            ha="center", va="center", fontsize=8.5, color=C_SUB,
            family="monospace", style="italic")


def illust_mask2former(ax, x, y, w, h):
    """input → Swin-B backbone (4 stages) → Pixel Decoder → Transformer Decoder → masks."""
    cx = x + w / 2
    body_top = y + h
    # ── 상단: 신경망 구조 (좌→우) ──
    # input RGB
    ax.add_patch(Rectangle((x + 0.2, body_top - 0.85), 0.4, 0.5,
                           facecolor="#FFD180", edgecolor="black", lw=0.8))
    ax.text(x + 0.4, body_top - 0.6, "RGB", ha="center", va="center",
            fontsize=7, color=C_TEXT, fontproperties=FP_BOLD)
    # arrow
    ax.annotate("", xy=(x + 0.75, body_top - 0.6), xytext=(x + 0.62, body_top - 0.6),
                arrowprops=dict(arrowstyle="->", color=C_ARROW, lw=0.9))
    # Swin-B 4 stages (계단식)
    stage_x0 = x + 0.82
    sizes_h = [0.62, 0.55, 0.48, 0.4]
    stage_cy = body_top - 0.6
    sx = stage_x0
    for i, sh in enumerate(sizes_h):
        ax.add_patch(Rectangle((sx, stage_cy - sh / 2), 0.18, sh,
                               facecolor=C_NN1, alpha=0.8, edgecolor="black", lw=0.5))
        sx += 0.22
    ax.text((stage_x0 + sx - 0.04) / 2, body_top - 1.1,
            "Swin-B  ·  4 stages", ha="center",
            fontsize=7.5, fontproperties=FP_BOLD, color=C_NN1)
    # Pixel Decoder
    pd_x = sx + 0.08
    ax.add_patch(FancyBboxPatch(
        (pd_x, body_top - 0.85), 0.42, 0.5,
        boxstyle="round,pad=0.01,rounding_size=0.04",
        facecolor="#90CAF9", edgecolor="black", lw=0.7,
    ))
    ax.text(pd_x + 0.21, body_top - 0.6, "Pixel\nDec.", ha="center", va="center",
            fontsize=6.7, fontproperties=FP_BOLD, color=C_TEXT)
    # Transformer decoder
    td_x = pd_x + 0.5
    ax.add_patch(FancyBboxPatch(
        (td_x, body_top - 0.85), 0.5, 0.5,
        boxstyle="round,pad=0.01,rounding_size=0.04",
        facecolor=C_NN1, edgecolor="black", lw=0.7,
    ))
    ax.text(td_x + 0.25, body_top - 0.6, "Trans.\nDecoder", ha="center", va="center",
            fontsize=6.7, fontproperties=FP_BOLD, color="white")

    # ── 하단: 출력 마스크 ──
    out_label_y = y + h / 2 - 0.4
    ax.text(cx, out_label_y, "per-instance binary masks", ha="center",
            fontsize=8, fontproperties=FP_BOLD, color=C_NN1)
    out_y = y + 0.25
    out_h = out_label_y - out_y - 0.15
    # mini mask grid (3 instances) — 더 크고 더 명확하게
    mg_w, mg_h = 0.72, out_h
    gap = 0.18
    total_w = 3 * mg_w + 2 * gap
    start_gx = cx - total_w / 2
    obj_colors = ["#FFA726", "#42A5F5", "#9CCC65"]
    obj_labels = ["haribo", "metal", "pencil"]
    # 마스크 형상 (의도된 객체 모양)
    shapes = [
        [(0.15, 0.2), (0.55, 0.25), (0.62, 0.45), (0.4, 0.55), (0.18, 0.5)],          # 둥근 형상 (haribo)
        [(0.18, 0.25), (0.58, 0.22), (0.6, 0.55), (0.2, 0.55)],                       # 직사각형 (metal)
        [(0.12, 0.3), (0.6, 0.3), (0.6, 0.45), (0.12, 0.45)],                         # 가로 직사각형 (pencil)
    ]
    for i, (color, lbl, pts) in enumerate(zip(obj_colors, obj_labels, shapes)):
        gx = start_gx + i * (mg_w + gap)
        # 외곽 (스캐닝 결과 흰 캔버스)
        ax.add_patch(Rectangle((gx, out_y), mg_w, mg_h,
                               facecolor="white", edgecolor="#37474F", lw=0.7))
        # mask 형상
        pts_xy = [(gx + px * mg_w / 0.7, out_y + py * mg_h / 0.7) for px, py in pts]
        ax.add_patch(Polygon(pts_xy, closed=True, facecolor=color, alpha=0.9,
                             edgecolor="#263238", lw=0.6))
        # 라벨
        ax.text(gx + mg_w / 2, out_y - 0.13, lbl, ha="center",
                fontsize=6.5, color=C_SUB, family="monospace")
        # 인스턴스 번호
        ax.text(gx + 0.08, out_y + mg_h - 0.1, f"#{i+1}",
                ha="left", va="top", fontsize=6.5,
                color="white", fontproperties=FP_BOLD)


def illust_dinov2_classifier(ax, x, y, w, h):
    """input crop 224 → ViT-B 12 blocks → CLS + patch_mean → reference bank KNN."""
    cx = x + w / 2
    body_top = y + h
    # ── 상단: input crop → ViT blocks ──
    crop_x = x + 0.28
    crop_y = body_top - 0.85
    ax.add_patch(Rectangle((crop_x, crop_y), 0.55, 0.7,
                           facecolor="#E1BEE7", edgecolor="black", lw=0.8))
    ax.text(crop_x + 0.275, crop_y + 0.35, "crop", ha="center", va="center",
            fontsize=7.5, fontproperties=FP_BOLD, color=C_TEXT)
    ax.text(crop_x + 0.275, crop_y - 0.18, "224×224", ha="center",
            fontsize=7, color=C_SUB, family="monospace")
    # ViT 12 blocks — 두께 충분히
    stack_x = crop_x + 0.78
    stack_y_top = body_top - 0.2
    stack_y_bot = body_top - 0.95
    n_blocks = 12
    avail_w = (x + w - 0.32) - stack_x
    block_w = (avail_w - 0.04 * (n_blocks - 1)) / n_blocks
    for i in range(n_blocks):
        bx = stack_x + i * (block_w + 0.04)
        if i == 11:
            ax.add_patch(Rectangle((bx, stack_y_bot), block_w, stack_y_top - stack_y_bot,
                                   facecolor="#9C27B0", edgecolor="white", lw=0.7, zorder=3))
        elif i == 7:  # PatchCore가 쓰는 layer L8 = hidden_states[9] = block index 8 (0-indexed)
            ax.add_patch(Rectangle((bx, stack_y_bot), block_w, stack_y_top - stack_y_bot,
                                   facecolor="#E65100", edgecolor="white", lw=0.7, zorder=3))
        else:
            ax.add_patch(Rectangle((bx, stack_y_bot), block_w, stack_y_top - stack_y_bot,
                                   facecolor=C_NN2, alpha=0.85, edgecolor="black", lw=0.4))
    # 라벨 + 강조 layer 표기 (block 아래)
    ax.text(stack_x + avail_w / 2, stack_y_bot - 0.22,
            "DinoV2-base · ViT 12 transformer blocks", ha="center",
            fontsize=8, fontproperties=FP_BOLD, color=C_NN2)
    # block 8 / block 12 강조 표기 (block 위 작은 dot + 안쪽 라벨)
    for i, label, color in [(7, "L8", "#E65100"), (11, "L12", "#9C27B0")]:
        bx = stack_x + i * (block_w + 0.04) + block_w / 2
        ax.text(bx, (stack_y_top + stack_y_bot) / 2, label,
                ha="center", va="center", fontsize=6.5,
                color="white", fontproperties=FP_BOLD, zorder=4)

    # ── 하단: 사용처 (CLS → KNN, mid-layer → PatchCore) ──
    use_y = y + 0.05
    use_h = stack_y_bot - 0.4 - use_y
    # 좌측: CLS + patch_mean → KNN reference bank
    left_x = x + 0.25
    left_w = (w - 0.5) * 0.55
    ax.add_patch(FancyBboxPatch(
        (left_x, use_y + 0.4), left_w, use_h - 0.4,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor="white", edgecolor=C_NN2, lw=1.0, zorder=3,
    ))
    ax.text(left_x + left_w / 2, use_y + use_h - 0.15,
            "CLS + patch_mean  →  cosine KNN", ha="center",
            fontsize=7.5, fontproperties=FP_BOLD, color=C_NN2)
    # scatter (reference bank)
    rng = np.random.default_rng(7)
    sc_cx = left_x + left_w / 2
    sc_cy = use_y + 0.9
    classes_color = ["#FFA726", "#42A5F5", "#9CCC65", "#EC407A", "#26C6DA", "#7E57C2"]
    for i, color in enumerate(classes_color):
        pts = rng.normal(loc=(sc_cx + (i - 2.5) * 0.17, sc_cy), scale=0.07, size=(7, 2))
        ax.scatter(pts[:, 0], pts[:, 1], s=11, c=color, edgecolors="none", zorder=4)
    # query
    ax.scatter([sc_cx + 0.05], [sc_cy + 0.06], marker="*", s=85, c="red",
               edgecolors="black", lw=0.7, zorder=5)
    ax.text(sc_cx + 0.17, sc_cy + 0.08, "query", fontsize=6.5, color="red",
            fontproperties=FP_BOLD)
    ax.text(sc_cx, use_y + 0.45, "→ class_index ∈ {0..5}", ha="center",
            fontsize=7.2, color=C_TEXT, family="monospace", fontproperties=FP_BOLD)

    # 우측: mid-layer (L8) → PatchCore 신호 표기 (참조용 callout)
    right_x = left_x + left_w + 0.12
    right_w = (x + w - 0.25) - right_x
    ax.add_patch(FancyBboxPatch(
        (right_x, use_y + 0.4), right_w, use_h - 0.4,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor="#FFF3E0", edgecolor="#E65100", lw=1.0, zorder=3,
    ))
    ax.text(right_x + right_w / 2, use_y + use_h - 0.15,
            "mid-layer patches", ha="center",
            fontsize=7.5, fontproperties=FP_BOLD, color="#E65100")
    ax.text(right_x + right_w / 2, use_y + 0.95,
            "hidden_states[9]\n(L8 default)", ha="center", va="center",
            fontsize=6.8, color=C_TEXT, family="monospace")
    ax.text(right_x + right_w / 2, use_y + 0.5,
            "→ PatchCore-lite", ha="center",
            fontsize=7.2, color="#E65100", fontproperties=FP_BOLD,
            family="monospace")


def illust_registry(ax, x, y, w, h):
    """class_id → 3 branches: bypass / Cascade / PatchCore."""
    cx = x + w / 2
    body_top = y + h
    # 상단: class_id 입력 박스
    top_y = body_top - 0.65
    ax.add_patch(FancyBboxPatch(
        (cx - 0.95, top_y - 0.35), 1.9, 0.55,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor="white", edgecolor=C_BR, lw=1.4,
    ))
    ax.text(cx, top_y - 0.07, "class_index ∈ {0..5}",
            ha="center", va="center", fontsize=8.5, fontproperties=FP_BOLD,
            family="monospace", color=C_TEXT)
    # 가운데 라벨 (registry decision)
    mid_y = top_y - 0.65
    ax.text(cx, mid_y, "DefectRegistry.get(idx)",
            ha="center", va="center", fontsize=8, color=C_SUB,
            family="monospace", style="italic")
    # branch lines (decision node → 3 outputs)
    branch_y = y + 0.45
    bb_w = 0.98
    branches = [
        (cx - 1.08, "bypass",        "bottle (4)\nunknown (5)",         "#9E9E9E"),
        (cx,         "Cascade",       "haribo (0)",                     "#F9A825"),
        (cx + 1.08, "PatchCore-lite", "pencil (2)\nmetal (3)",           "#E65100"),
    ]
    for bx, bname, bnote, bcolor in branches:
        # 선 (registry → branch box top)
        ax.plot([cx, bx], [mid_y - 0.2, branch_y + 0.85], color=C_BR,
                lw=1.4, zorder=2)
        # box
        ax.add_patch(FancyBboxPatch(
            (bx - bb_w / 2, branch_y + 0.3), bb_w, 0.55,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor=bcolor, edgecolor="black", lw=0.8,
        ))
        ax.text(bx, branch_y + 0.58, bname, ha="center", va="center",
                fontsize=7.8, fontproperties=FP_BOLD, color="white")
        ax.text(bx, branch_y + 0.05, bnote, ha="center", va="center",
                fontsize=6.8, color=C_SUB, style="italic")


def illust_defect(ax, x, y, w, h):
    """PatchCore (mid-layer patches → memory bank L2 max) + Cascade (4 signals voting)."""
    cx = x + w / 2
    body_top = y + h
    # 좌우 영역 분할 (50:50)
    pad = 0.25
    div_x = x + w / 2
    left_x = x + pad
    left_w = div_x - left_x - 0.05
    right_x = div_x + 0.05
    right_w = (x + w - pad) - right_x

    # ── 좌측: PatchCore-lite ──
    left_cx = left_x + left_w / 2
    ax.text(left_cx, body_top - 0.18, "PatchCore-lite",
            ha="center", fontsize=8.5, fontproperties=FP_BOLD, color=C_DET)
    ax.text(left_cx, body_top - 0.42, "(pencil / metal)",
            ha="center", fontsize=6.8, color=C_SUB, style="italic")
    # patches @ L8 — 좌측, 작은 stack
    pc_w = 0.32
    pc_x = left_x + 0.05
    pc_y = body_top - 1.25
    for i, off in enumerate([0, 0.035, 0.07]):
        ax.add_patch(Rectangle((pc_x + off, pc_y + off), pc_w, 0.42,
                               facecolor=C_DET, alpha=0.78 - 0.15 * i,
                               edgecolor="black", lw=0.5))
    ax.text(pc_x + pc_w / 2 + 0.04, pc_y - 0.16, "patches",
            ha="center", fontsize=6.2, color=C_TEXT, fontproperties=FP_BOLD)
    ax.text(pc_x + pc_w / 2 + 0.04, pc_y - 0.31, "@ L8",
            ha="center", fontsize=5.8, color=C_SUB, family="monospace")
    # memory bank — 우측, 큰 stack
    mb_w = 0.4
    mb_x = left_x + left_w - mb_w - 0.05
    for i, off in enumerate([0, 0.03, 0.06, 0.09]):
        ax.add_patch(Rectangle((mb_x + off, pc_y - 0.05 + off), mb_w, 0.55,
                               facecolor="#FFCC80", alpha=0.92 - 0.1 * i,
                               edgecolor="black", lw=0.5))
    ax.text(mb_x + mb_w / 2 + 0.04, pc_y - 0.16, "memory",
            ha="center", fontsize=6.2, color=C_TEXT, fontproperties=FP_BOLD)
    ax.text(mb_x + mb_w / 2 + 0.04, pc_y - 0.31, "bank",
            ha="center", fontsize=5.8, color=C_SUB, family="monospace")
    # L2 NN max arrow (두 stack 사이)
    arr_x1 = pc_x + pc_w + 0.12
    arr_x2 = mb_x - 0.04
    arr_y = pc_y + 0.28
    ax.annotate("", xy=(arr_x2, arr_y), xytext=(arr_x1, arr_y),
                arrowprops=dict(arrowstyle="<->", color=C_DET, lw=1.3))
    ax.text((arr_x1 + arr_x2) / 2, arr_y + 0.18, "L2 NN max",
            ha="center", fontsize=6.8, color=C_DET, fontproperties=FP_BOLD,
            style="italic")

    # 구분선
    ax.plot([div_x, div_x], [y + 0.95, body_top - 0.55], color=C_SUB, lw=0.6,
            linestyle=":", alpha=0.5)

    # ── 우측: Cascade ──
    right_cx = right_x + right_w / 2
    ax.text(right_cx, body_top - 0.18, "Cascade",
            ha="center", fontsize=8.5, fontproperties=FP_BOLD, color=C_DET)
    ax.text(right_cx, body_top - 0.42, "(haribo)",
            ha="center", fontsize=6.8, color=C_SUB, style="italic")
    signals = ["PatchCore", "Color", "Normal", "Topology"]
    sg_y_start = body_top - 0.85
    for i, sg in enumerate(signals):
        sy = sg_y_start - i * 0.24
        ax.add_patch(Circle((right_x + 0.1, sy), 0.06,
                            facecolor=C_DET, edgecolor="black", lw=0.5))
        ax.text(right_x + 0.22, sy, sg, ha="left", va="center",
                fontsize=7, color=C_TEXT)
    # voting box — 우측 영역 안, 글자 침범 없는 거리
    v_w = 0.62
    v_x = right_x + right_w - v_w - 0.05
    v_y = body_top - 1.4
    ax.add_patch(FancyBboxPatch(
        (v_x, v_y), v_w, 0.65,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=C_DET, edgecolor="black", lw=0.9,
    ))
    ax.text(v_x + v_w / 2, v_y + 0.43, "vote", ha="center", va="center",
            fontsize=7, fontproperties=FP_BOLD, color="white")
    ax.text(v_x + v_w / 2, v_y + 0.18, "≥ 2", ha="center", va="center",
            fontsize=9, fontproperties=FP_BOLD, color="white")
    # 4 signals → voting arrow
    arrow_start_x = right_x + 0.85   # signal 글자 끝 약간 뒤
    ax.annotate("", xy=(v_x - 0.03, v_y + 0.32),
                xytext=(arrow_start_x, v_y + 0.32),
                arrowprops=dict(arrowstyle="->", color=C_DET, lw=1.0))

    # ── 결과 (중앙 하단) ──
    res_y = y + 0.2
    ax.add_patch(FancyBboxPatch(
        (cx - 1.1, res_y), 2.2, 0.5,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor="white", edgecolor=C_DET, lw=1.3,
    ))
    ax.text(cx, res_y + 0.25, "→  is_defect : bool", ha="center", va="center",
            fontsize=8.5, color=C_DET, fontproperties=FP_BOLD,
            family="monospace")


def illust_grasp_priority(ax, x, y, w, h):
    """defect → -inf, else score by area/depth/class."""
    cx = x + w / 2
    # decision diamond (흰색 + 강한 테두리, 상단)
    diamond_top_y = y + h - 0.35
    diamond_bot_y = y + h - 1.4
    diamond_xy = [(cx, diamond_top_y),
                  (cx + 0.65, (diamond_top_y + diamond_bot_y) / 2),
                  (cx, diamond_bot_y),
                  (cx - 0.65, (diamond_top_y + diamond_bot_y) / 2)]
    ax.add_patch(Polygon(diamond_xy, closed=True, facecolor="white",
                         edgecolor=C_LOG, lw=1.8, zorder=4))
    ax.text(cx, (diamond_top_y + diamond_bot_y) / 2, "is_defect ?",
            ha="center", va="center",
            fontsize=8.5, fontproperties=FP_BOLD, color=C_LOG, zorder=5)
    # Y branch (defect, 좌측)
    branch_mid_y = (diamond_top_y + diamond_bot_y) / 2
    pri_box_top = y + 0.85
    ax.annotate("", xy=(cx - 1.4, pri_box_top + 0.05),
                xytext=(cx - 0.5, branch_mid_y - 0.05),
                arrowprops=dict(arrowstyle="->", color="#C62828", lw=1.4))
    ax.text(cx - 0.85, branch_mid_y - 0.32, "yes", fontsize=7.2, color="#C62828",
            fontproperties=FP_BOLD)
    ax.add_patch(FancyBboxPatch(
        (cx - 1.7, y + 0.25), 1.05, 0.6,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor="#FFCDD2", edgecolor="#C62828", lw=1.2,
    ))
    ax.text(cx - 1.175, y + 0.55, "priority = -∞", ha="center", va="center",
            fontsize=7.6, fontproperties=FP_BOLD, color="#C62828",
            family="monospace")
    # N branch (normal, 우측)
    ax.annotate("", xy=(cx + 1.4, pri_box_top + 0.05),
                xytext=(cx + 0.5, branch_mid_y - 0.05),
                arrowprops=dict(arrowstyle="->", color=C_LOG, lw=1.4))
    ax.text(cx + 0.85, branch_mid_y - 0.32, "no", fontsize=7.2, color=C_LOG,
            fontproperties=FP_BOLD)
    # scoring formula box (우측 하단)
    sf_x = cx + 0.3
    sf_y = y + 0.25
    sf_w = 1.4
    sf_h = 0.85
    ax.add_patch(FancyBboxPatch(
        (sf_x - 0.05, sf_y), sf_w, sf_h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor="white", edgecolor=C_LOG, lw=1.2,
    ))
    ax.text(sf_x + sf_w / 2 - 0.05, sf_y + sf_h - 0.18,
            "score(inst)", ha="center", fontsize=7.5,
            color=C_LOG, fontproperties=FP_BOLD, family="monospace")
    ax.text(sf_x + sf_w / 2 - 0.05, sf_y + sf_h - 0.4,
            "= f(area, z, cls)", ha="center",
            fontsize=7.3, color=C_TEXT, family="monospace")
    ax.text(sf_x + sf_w / 2 - 0.05, sf_y + 0.13,
            "argmax → best", ha="center",
            fontsize=7.2, color=C_LOG, fontproperties=FP_BOLD,
            style="italic", family="monospace")


def illust_suction(ax, x, y, w, h):
    """객체 표면 + suction point + normal vector."""
    cx = x + w / 2
    # 객체 표면 (타원)
    ax.add_patch(mpatches.Ellipse((cx, y + h - 1.3), 2.3, 1.1,
                                  facecolor="#B0BEC5", edgecolor="#37474F", lw=1.2,
                                  zorder=3))
    # surface highlight (작은 영역 = candidate)
    ax.add_patch(mpatches.Ellipse((cx + 0.15, y + h - 1.25), 0.6, 0.32,
                                  facecolor="#FFCC80", edgecolor="#E65100", lw=1.2,
                                  alpha=0.9, zorder=4))
    # suction point (별)
    sp_x, sp_y = cx + 0.15, y + h - 1.25
    ax.scatter([sp_x], [sp_y], marker="*", s=140, c="#C62828",
               edgecolors="black", lw=0.8, zorder=6)
    # normal arrow (점→up)
    ax.annotate("", xy=(sp_x + 0.35, sp_y + 0.8), xytext=(sp_x, sp_y),
                arrowprops=dict(arrowstyle="->", color="#1B5E20", lw=2.0))
    ax.text(sp_x + 0.55, sp_y + 0.55, "surface\nnormal",
            ha="left", va="center", fontsize=7, color="#1B5E20",
            fontproperties=FP_BOLD)
    # 출력 표기
    out_y = y + 0.5
    ax.add_patch(Rectangle((cx - 1.4, out_y + 0.1), 2.8, 0.95,
                           facecolor=C_LOG_LIGHT, edgecolor=C_LOG, lw=1.0))
    ax.text(cx, out_y + 0.83, "suction point (camera frame)", ha="center",
            fontsize=7.8, fontproperties=FP_BOLD, color=C_LOG)
    ax.text(cx, out_y + 0.55, "p_cam = (x, y, z) mm", ha="center",
            fontsize=7.5, color=C_TEXT, family="monospace")
    ax.text(cx, out_y + 0.3, "q_cam = (qx, qy, qz, qw)", ha="center",
            fontsize=7.5, color=C_TEXT, family="monospace")
    ax.text(cx, out_y + 0.13, "+ tip offset 6 mm", ha="center",
            fontsize=6.8, color=C_SUB, style="italic")


def illust_calibration(ax, x, y, w, h):
    """카메라 좌표축 + 변환행렬 + 로봇 좌표축."""
    cx = x + w / 2
    body_top = y + h
    # ── 상단: 두 frame + 변환 행렬 ──
    frame_y = body_top - 0.85
    # cam frame (좌측)
    ax_x = x + 0.55
    ax.annotate("", xy=(ax_x + 0.45, frame_y), xytext=(ax_x, frame_y),
                arrowprops=dict(arrowstyle="->", color="#D32F2F", lw=1.5))
    ax.annotate("", xy=(ax_x, frame_y + 0.45), xytext=(ax_x, frame_y),
                arrowprops=dict(arrowstyle="->", color="#388E3C", lw=1.5))
    ax.annotate("", xy=(ax_x - 0.26, frame_y - 0.26), xytext=(ax_x, frame_y),
                arrowprops=dict(arrowstyle="->", color="#1976D2", lw=1.5))
    ax.text(ax_x, frame_y + 0.62, "Camera", ha="center", fontsize=7.5,
            fontproperties=FP_BOLD, color="#37474F")
    # 변환 행렬 박스 (중앙)
    mx_w, mx_h = 0.95, 0.78
    mx_x = cx - mx_w / 2
    mx_y = frame_y - mx_h / 2
    ax.add_patch(FancyBboxPatch(
        (mx_x, mx_y), mx_w, mx_h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor="white", edgecolor=C_CAL, lw=1.3,
    ))
    ax.text(mx_x + mx_w / 2, mx_y + 0.6, "T_robot←cam", ha="center", va="center",
            fontsize=7.5, fontproperties=FP_BOLD, color=C_CAL,
            family="monospace")
    ax.text(mx_x + mx_w / 2, mx_y + 0.36, "[R | t]", ha="center", va="center",
            fontsize=7.8, fontproperties=FP_BOLD, color=C_TEXT,
            family="monospace")
    ax.text(mx_x + mx_w / 2, mx_y + 0.13, "4×4", ha="center", va="center",
            fontsize=6.7, color=C_SUB, family="monospace")
    # cam→matrix arrow
    ax.annotate("", xy=(mx_x - 0.02, mx_y + mx_h / 2),
                xytext=(ax_x + 0.48, frame_y - 0.02),
                arrowprops=dict(arrowstyle="->", color=C_ARROW, lw=1.1))
    # robot frame (우측)
    rax_x = x + w - 0.7
    ax.annotate("", xy=(rax_x + 0.45, frame_y), xytext=(rax_x, frame_y),
                arrowprops=dict(arrowstyle="->", color="#D32F2F", lw=1.5))
    ax.annotate("", xy=(rax_x, frame_y + 0.45), xytext=(rax_x, frame_y),
                arrowprops=dict(arrowstyle="->", color="#388E3C", lw=1.5))
    ax.annotate("", xy=(rax_x - 0.26, frame_y - 0.26), xytext=(rax_x, frame_y),
                arrowprops=dict(arrowstyle="->", color="#1976D2", lw=1.5))
    ax.text(rax_x, frame_y + 0.62, "Robot base", ha="center", fontsize=7.5,
            fontproperties=FP_BOLD, color="#37474F")
    # matrix→robot arrow
    ax.annotate("", xy=(rax_x - 0.06, frame_y - 0.02),
                xytext=(mx_x + mx_w + 0.02, mx_y + mx_h / 2),
                arrowprops=dict(arrowstyle="->", color=C_ARROW, lw=1.1))

    # ── 하단: 캘리브 수치 ──
    info_y = y + 0.15
    info_h = mx_y - 0.45 - info_y
    info_h = max(info_h, 0.9)
    ax.add_patch(FancyBboxPatch(
        (x + 0.3, info_y), w - 0.6, info_h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=C_CAL_LIGHT, edgecolor=C_CAL, lw=1.0,
    ))
    ax.text(cx, info_y + info_h - 0.18, "configs/calibration.yaml", ha="center",
            fontsize=7, color=C_SUB, family="monospace", style="italic")
    ax.text(cx, info_y + info_h - 0.45,
            "trans (-770, 67, 985) mm", ha="center",
            fontsize=7.2, color=C_TEXT, family="monospace")
    ax.text(cx, info_y + info_h - 0.72,
            "rot (179°, 1.1°, -89.4°) xyz", ha="center",
            fontsize=7.2, color=C_TEXT, family="monospace")


def illust_robot(ax, x, y, w, h):
    """로봇 팔 (상단) + 최종 명령 박스 (하단)."""
    cx = x + w / 2
    body_top = y + h
    # ── 상단: robot arm (body 위쪽 ~0.9 영역) — zorder ≥ 4로 body fill 위에 ──
    Z = 4
    base_x = cx - 0.4
    base_y = body_top - 0.95
    # base
    ax.add_patch(Rectangle((base_x - 0.25, base_y - 0.15), 0.5, 0.18,
                           facecolor="#455A64", edgecolor="black", lw=0.8,
                           zorder=Z))
    # link 1 (위로 비스듬)
    j1_x, j1_y = base_x, base_y + 0.03
    j2_x, j2_y = base_x + 0.55, base_y + 0.55
    ax.plot([j1_x, j2_x], [j1_y, j2_y], color="#37474F", lw=4.5,
            solid_capstyle="round", zorder=Z)
    # joint 1
    ax.add_patch(Circle((j1_x, j1_y), 0.08, facecolor="#90A4AE",
                        edgecolor="black", lw=0.6, zorder=Z + 1))
    # link 2 (옆으로)
    j3_x, j3_y = j2_x + 0.55, j2_y - 0.2
    ax.plot([j2_x, j3_x], [j2_y, j3_y], color="#37474F", lw=4.5,
            solid_capstyle="round", zorder=Z)
    # joint 2
    ax.add_patch(Circle((j2_x, j2_y), 0.08, facecolor="#FFC107",
                        edgecolor="black", lw=0.6, zorder=Z + 1))
    # end effector (suction)
    ax.add_patch(Rectangle((j3_x - 0.06, j3_y - 0.22), 0.12, 0.22,
                           facecolor=C_ROBOT, edgecolor="black", lw=0.7,
                           zorder=Z + 1))
    # 잡힌 객체 (파지된 상태)
    ax.add_patch(Rectangle((j3_x - 0.16, j3_y - 0.4), 0.32, 0.18,
                           facecolor="#42A5F5", edgecolor="black", lw=0.7,
                           zorder=Z + 1))
    # ── 하단: cmd 박스 ──
    cmd_h = 1.1
    cmd_y = y + 0.05
    ax.add_patch(FancyBboxPatch(
        (x + 0.3, cmd_y), w - 0.6, cmd_h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor="white", edgecolor=C_ROBOT, lw=1.3,
    ))
    ax.text(cx, cmd_y + cmd_h - 0.22, "robot pose command", ha="center",
            fontsize=8, fontproperties=FP_BOLD, color=C_ROBOT)
    ax.text(cx, cmd_y + cmd_h - 0.48, "(x, y, z) mm", ha="center",
            fontsize=8, color=C_TEXT, family="monospace")
    ax.text(cx, cmd_y + cmd_h - 0.73, "(qx, qy, qz, qw)", ha="center",
            fontsize=8, color=C_TEXT, family="monospace")
    ax.text(cx, cmd_y + 0.15, "gRPC reply → controller", ha="center",
            fontsize=6.8, color=C_SUB, style="italic")


# ─── 화살표 (박스 간) ─────────────────────────────────────────────────────────
def draw_h_arrow(ax, x_from, x_to, y, label=None, label_off=0.32, color=C_ARROW, lw=1.8):
    arrow = FancyArrowPatch(
        (x_from, y), (x_to, y),
        arrowstyle="-|>",
        color=color, lw=lw, zorder=6, mutation_scale=12,
    )
    ax.add_patch(arrow)
    if label:
        mid = (x_from + x_to) / 2
        ax.text(mid, y + label_off, label, ha="center", va="bottom",
                fontsize=7, color=C_SUB, family="monospace", fontweight="bold",
                bbox=dict(facecolor="white", edgecolor=color, lw=0.5,
                          boxstyle="round,pad=0.18", alpha=0.96))


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal")
    ax.axis("off")

    # 배경 (옅은 그라데이션 느낌)
    ax.add_patch(Rectangle((0, 0), XLIM[1], YLIM[1],
                           facecolor="#FAFAFA", edgecolor="none", zorder=0))

    # 타이틀
    fig.suptitle("CMES 빈피킹 추론 파이프라인 — Zivid 스캔에서 로봇 명령까지",
                 fontsize=21, fontproperties=FP_BOLD, color=C_TEXT, y=0.965)
    fig.text(0.5, 0.918,
             "Pick-n-Place gRPC inference flow · 신경망 5단 → 결함 판정 → 파지 계산 → 좌표 변환",
             ha="center", fontsize=12, color=C_SUB, fontproperties=FP_REG,
             style="italic")

    # 박스 배치
    boxes_top = [
        # (illust_fn, edge, fill, num, ko_title, en_subtitle, in_label, out_label)
        (illust_zivid, C_HW, C_HW_LIGHT,
         1, "Zivid 빈 스캔", "Zivid 2+ MR130 hardware",
         "(scene)", "RGB + organized PLY"),
        (illust_convert, C_PRE, C_PRE_LIGHT,
         2, "데이터 변환 · gRPC", "PLY → png/bin · file IO",
         "PLY", "(rgb,depth,normal)"),
        (illust_mask2former, C_NN1, C_NN1_LIGHT,
         3, "인스턴스 세그", "Mask2Former · Swin-B",
         "RGB (1224,1024,3)", "masks (N,1224,1024)"),
        (illust_dinov2_classifier, C_NN2, C_NN2_LIGHT,
         4, "객체 분류", "DinoV2-base · KNN",
         "crop (224,224,3)", "class_index ∈ {0..5}"),
        (illust_registry, C_BR, C_BR_LIGHT,
         5, "결함 detector 분기", "DefectRegistry.get(idx)",
         "class_index", "PatchCore | Cascade | bypass"),
    ]
    boxes_bot = [
        (illust_defect, C_DET, C_DET_LIGHT,
         6, "결함 판정", "PatchCore-lite · Cascade",
         "DinoV2 patches/RGB", "is_defect : bool"),
        (illust_grasp_priority, C_LOG, C_LOG_LIGHT,
         7, "파지 우선순위", "GraspPriority scoring",
         "instances + is_defect", "ranked instances"),
        (illust_suction, C_LOG, C_LOG_LIGHT,
         8, "파지 지점 계산", "SuctionPipeline (cam frame)",
         "best instance + normal", "(p_cam, q_cam)"),
        (illust_calibration, C_CAL, C_CAL_LIGHT,
         9, "좌표 변환", "Calibration (extrinsic)",
         "p_cam, q_cam", "p_robot, q_robot"),
        (illust_robot, C_ROBOT, C_ROBOT_LIGHT,
         10, "로봇 명령", "gRPC reply → controller",
         "(x,y,z,qx,qy,qz,qw)", "robot actuation"),
    ]

    # 그리기 helper — illust 함수에는 body 영역만 넘김
    def draw_one(fn, edge, fill, num, title_ko, sub_en, in_lbl, out_lbl, bx, by):
        draw_box(ax, bx, by, BOX_W, BOX_H, edge, fill)
        draw_header(ax, bx, by + BOX_H, BOX_W, num, title_ko, sub_en, edge)
        body_y = by + FOOTER_H
        body_h = BOX_H - HEADER_H - FOOTER_H
        fn(ax, bx, body_y, BOX_W, body_h)
        draw_footer(ax, bx, by, BOX_W, BOX_H, in_lbl, out_lbl)

    # Row 1
    box_xs = [START_X + i * (BOX_W + COL_GAP) for i in range(5)]
    for i, b in enumerate(boxes_top):
        draw_one(*b, bx=box_xs[i], by=START_Y_TOP)
    # Row 2
    for i, b in enumerate(boxes_bot):
        draw_one(*b, bx=box_xs[i], by=START_Y_BOT)

    # 화살표 — Row 1 (좌→우)
    arrow_labels_top = [
        "(rgb,depth,normal)\nfile IO",
        "RGB tensor",
        "N masks",
        "class_index",
    ]
    for i in range(4):
        x1 = box_xs[i] + BOX_W
        x2 = box_xs[i + 1]
        ay = START_Y_TOP + BOX_H / 2
        draw_h_arrow(ax, x1 + 0.02, x2 - 0.02, ay, arrow_labels_top[i])
    # Row 2 (좌→우)
    arrow_labels_bot = [
        "is_defect",
        "best instance",
        "(p_cam, q_cam)",
        "(p_robot, q_robot)",
    ]
    for i in range(4):
        x1 = box_xs[i] + BOX_W
        x2 = box_xs[i + 1]
        ay = START_Y_BOT + BOX_H / 2
        draw_h_arrow(ax, x1 + 0.02, x2 - 0.02, ay, arrow_labels_bot[i])

    # U-turn: 박스 5 (top, idx 4) 우/하단 → 박스 6 (bot, idx 0) 좌/상단
    # 깔끔하게 박스 사이 간격을 따라가는 L자 경로
    from matplotlib.path import Path as MplPath
    bridge_y = START_Y_TOP - ROW_GAP / 2 - 0.05    # 두 row 사이 중간선
    # 1) box5 우상단 외곽에서 시작 → 우측 박스 밖으로 → 아래로 → 좌측 → box6 위로
    x5_right = box_xs[4] + BOX_W
    x5_mid_y = START_Y_TOP + BOX_H / 2
    x6_left = box_xs[0]
    x6_mid_y = START_Y_BOT + BOX_H / 2
    # 단순 3-segment 경로: 박스5 우측 끝점 → 우상단(밖) → 우하단 → 박스6 좌측 끝점
    elbow_x = box_xs[4] + BOX_W + 0.55
    elbow_x_left = box_xs[0] - 0.55
    verts = [
        (x5_right, x5_mid_y),       # 시작: 박스5 우측 중앙
        (elbow_x, x5_mid_y),         # 우로
        (elbow_x, bridge_y),         # 아래로
        (elbow_x_left, bridge_y),    # 좌로 (좌측 끝)
        (elbow_x_left, x6_mid_y),    # 다시 아래로
        (x6_left, x6_mid_y),         # 우로 (박스6 좌측)
    ]
    # 선만 plot으로
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    ax.plot(xs, ys, color=C_ARROW, lw=1.8, solid_joinstyle="round",
            solid_capstyle="round", zorder=5)
    # 끝점 화살촉
    arr = FancyArrowPatch(
        verts[-2], verts[-1],
        arrowstyle="-|>", color=C_ARROW, lw=1.8,
        mutation_scale=12, zorder=6,
    )
    ax.add_patch(arr)
    # U-turn 라벨 (가운데, 다리 위에)
    ax.text((elbow_x + elbow_x_left) / 2, bridge_y + 0.05,
            "다음 단계  ▶  per-instance loop",
            ha="center", va="bottom", fontsize=10, color=C_SUB,
            fontproperties=FP_BOLD, style="italic",
            bbox=dict(facecolor="white", edgecolor=C_SUB, pad=4,
                      boxstyle="round,pad=0.35"))

    # 범례 (하단 가운데, 가로 한 줄)
    legend_y = 0.35
    legend_items = [
        ("하드웨어 / IO", C_HW),
        ("Mask2Former", C_NN1),
        ("DinoV2 ViT", C_NN2),
        ("분기 (Registry)", C_BR),
        ("결함 detector", C_DET),
        ("Grasp / Suction", C_LOG),
        ("기하 · 캘리브", C_CAL),
        ("로봇 출력", C_ROBOT),
    ]
    # 가로 한 줄 균등 배치
    n_items = len(legend_items)
    item_width = 2.4
    total = n_items * item_width
    start_x = (XLIM[1] - total) / 2
    for i, (name, color) in enumerate(legend_items):
        lx = start_x + i * item_width
        ax.add_patch(FancyBboxPatch(
            (lx, legend_y), 0.34, 0.30,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=color, edgecolor="black", lw=0.6,
        ))
        ax.text(lx + 0.46, legend_y + 0.15, name, va="center", ha="left",
                fontsize=9.5, color=C_TEXT, fontproperties=FP_REG)

    # footer
    fig.text(0.99, 0.02, "CMES picknplace · 2026-06-12",
             ha="right", fontsize=8.5, color=C_SUB, fontproperties=FP_REG,
             style="italic")

    out_dir = ROOT / "outputs" / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pipeline_overview_ko.png"
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"→ {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
