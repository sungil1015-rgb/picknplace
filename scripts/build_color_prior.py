"""Color prior(GMM) 빌드 — cascade detector(haribo)의 색 신호(s2)용.

정상 객체 mask 픽셀의 LAB 색 분포를 GMM(3 comp)으로 적합해 저장.
DINOv2 layer와 무관(원시 LAB 픽셀 기반)이므로 base→large 전환·layer 변경과 상관없이
유효하다. 기존 weights/color_gmm_haribo.npz가 있으면 재빌드 불필요(재사용).

Usage:
    python scripts/build_color_prior.py --class_id 1 --class_name haribo
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.defect.color_prior import ColorPrior


def parse_yolo_polygon_line(line: str, W: int, H: int):
    parts = line.split()
    cid = int(parts[0])
    coords = list(map(float, parts[1:]))
    poly = np.array(
        [[coords[i] * W, coords[i + 1] * H] for i in range(0, len(coords), 2)],
        dtype=np.int32,
    )
    return cid, poly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--class_id", type=int, required=True, help="polygon class_id (haribo=1)")
    ap.add_argument("--class_name", type=str, required=True)
    ap.add_argument("--labeled_base", default="data/labeled")
    ap.add_argument("--n_components", type=int, default=3)
    ap.add_argument("--max_pixels", type=int, default=2_000_000, help="GMM 적합용 최대 픽셀(무작위 subsample)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    output = Path(args.output) if args.output else ROOT / "weights" / f"color_gmm_{args.class_name}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)

    labeled_base = ROOT / args.labeled_base
    set_dirs = sorted([d for d in labeled_base.iterdir() if d.is_dir()])
    print(f"[scan] {len(set_dirs)} set dirs")

    pixels = []
    obj_count = 0
    for set_dir in set_dirs:
        labels_dir = set_dir / "train" / "labels"
        images_dir = set_dir / "train" / "images"
        if not labels_dir.exists():
            continue
        for txt in sorted(labels_dir.glob("*.txt")):
            bmp = images_dir / f"{txt.stem}.bmp"
            if not bmp.exists():
                continue
            image = cv2.imread(str(bmp), cv2.IMREAD_COLOR)
            if image is None:
                continue
            H, W = image.shape[:2]
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            for line in txt.read_text().strip().split("\n"):
                if not line.strip():
                    continue
                try:
                    cid, poly = parse_yolo_polygon_line(line, W, H)
                except Exception:
                    continue
                if cid != args.class_id:
                    continue
                mask = np.zeros((H, W), dtype=np.uint8)
                cv2.fillPoly(mask, [poly], 255)
                px = lab[mask > 0]
                if len(px) > 0:
                    pixels.append(px)
                    obj_count += 1

    if not pixels:
        print(f"[error] no objects for class_id={args.class_id}")
        return 1

    pixels = np.concatenate(pixels, axis=0).astype(np.float32)
    if len(pixels) > args.max_pixels:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(pixels), args.max_pixels, replace=False)
        pixels = pixels[idx]
    print(f"[fit] objects={obj_count}, pixels={len(pixels)}, GMM n_components={args.n_components}")

    prior = ColorPrior(n_components=args.n_components)
    prior.fit(pixels)
    prior.save(output)
    print(f"→ {output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
