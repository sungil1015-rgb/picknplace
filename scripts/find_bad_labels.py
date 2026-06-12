"""mango 라벨 정제 대상 특정 — (1) 파란 타제품 혼입 폴리곤, (2) 중복 라벨 쌍.

H1/H4 발견 사항의 정확한 (set, stem, polygon_line_idx) 식별.
- 중복: 같은 이미지 내 mango polygon 쌍 중 IoU > 0.9
- 혼입: 0000000455_20260515 부근 이미지의 mango crop 전수 + 파란 픽셀 비율 측정

산출: outputs/eda/mango_exclude_list.txt ("<set>/<stem>:<line_idx>" 줄 단위)
      outputs/figs/mango_rootcause/bad_label_candidates.png (육안 확정용)
"""
from __future__ import annotations
import sys
from pathlib import Path
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANGO_ID = 2

def parse_polygon(line, W, H):
    parts = line.split()
    cid = int(parts[0])
    coords = list(map(float, parts[1:]))
    poly = np.array([[coords[i] * W, coords[i + 1] * H] for i in range(0, len(coords), 2)], dtype=np.int32)
    return cid, poly

def main():
    labeled = ROOT / "data" / "labeled"
    dup_pairs = []      # (set, stem, idx_a, idx_b, iou)
    blue_cands = []     # (set, stem, idx, blue_ratio, crop)

    for set_dir in sorted(labeled.iterdir()):
        if not set_dir.is_dir():
            continue
        labels = set_dir / "train" / "labels"
        images = set_dir / "train" / "images"
        if not labels.exists():
            continue
        for txt in sorted(labels.glob("*.txt")):
            bmp = images / f"{txt.stem}.bmp"
            if not bmp.exists():
                continue
            lines = [l for l in txt.read_text().strip().split("\n") if l.strip()]
            mango_polys = []
            for li, line in enumerate(lines):
                try:
                    cid, poly = parse_polygon(line, 1224, 1024)
                except Exception:
                    continue
                if cid == MANGO_ID:
                    mango_polys.append((li, poly))
            if not mango_polys:
                continue
            img = cv2.imread(str(bmp))
            if img is None:
                continue
            H, W = img.shape[:2]
            masks = []
            for li, poly in mango_polys:
                cid, poly = parse_polygon(lines[li], W, H)
                mask = np.zeros((H, W), dtype=np.uint8)
                cv2.fillPoly(mask, [poly], 1)
                masks.append((li, mask))
            # 중복 검사
            for a in range(len(masks)):
                for b in range(a + 1, len(masks)):
                    inter = (masks[a][1] & masks[b][1]).sum()
                    union = (masks[a][1] | masks[b][1]).sum()
                    iou = inter / max(union, 1)
                    if iou > 0.9:
                        dup_pairs.append((set_dir.name, txt.stem, masks[a][0], masks[b][0], iou))
            # 파란 혼입 검사 (mask 안 파란 픽셀 비율)
            for li, mask in masks:
                m = mask > 0
                if m.sum() < 100:
                    continue
                bgr = img[m].astype(np.float32)
                # 파란색: B가 G,R보다 명확히 큼
                blue = (bgr[:, 0] > bgr[:, 1] + 30) & (bgr[:, 0] > bgr[:, 2] + 30)
                ratio = float(blue.mean())
                if ratio > 0.10:
                    ys, xs = np.where(m)
                    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
                    crop = img[y1:y2, x1:x2].copy()
                    blue_cands.append((set_dir.name, txt.stem, li, ratio, crop))

    print(f"중복 쌍 (IoU>0.9): {len(dup_pairs)}")
    for d in dup_pairs:
        print(f"  {d[0]}/{d[1]} line {d[2]} vs {d[3]} IoU={d[4]:.3f}")
    print(f"파란 혼입 후보 (blue>10%): {len(blue_cands)}")
    for b in blue_cands:
        print(f"  {b[0]}/{b[1]} line {b[2]} blue_ratio={b[3]:.2f}")

    # 육안 확정 figure
    n = len(blue_cands)
    if n > 0:
        fig, axes = plt.subplots(1, max(n, 2), figsize=(4 * max(n, 2), 4))
        if n == 1:
            axes = [axes[0]] if hasattr(axes, '__len__') else [axes]
        for i, (sname, stem, li, ratio, crop) in enumerate(blue_cands):
            ax = axes[i] if hasattr(axes, '__len__') else axes
            ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            ax.set_title(f"{stem}\nline {li}, blue {ratio:.0%}", fontsize=9)
            ax.axis("off")
        out = ROOT / "outputs" / "figs" / "mango_rootcause" / "bad_label_candidates.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=110, bbox_inches="tight")
        print(f"→ {out.relative_to(ROOT)}")

    # exclusion list 저장 (중복은 한쪽만 제외)
    out_list = ROOT / "outputs" / "eda" / "mango_exclude_list.txt"
    out_list.parent.mkdir(parents=True, exist_ok=True)
    with out_list.open("w") as f:
        for sname, stem, li, ratio, _ in blue_cands:
            f.write(f"{sname}/{stem}:{li}  # blue contamination {ratio:.0%}\n")
        for sname, stem, a, b, iou in dup_pairs:
            f.write(f"{sname}/{stem}:{b}  # duplicate of line {a}, IoU {iou:.3f}\n")
    print(f"→ {out_list.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
