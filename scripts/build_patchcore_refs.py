"""PatchCore-lite memory bank 빌드 — 라벨링 polygon 데이터에서 객체별로 DinoV2 mid-layer patch features 추출.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/build_patchcore_refs.py --class_id 3 --class_name metal_case
    CUDA_VISIBLE_DEVICES=0 python scripts/build_patchcore_refs.py --class_id 5 --class_name pencil_case
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CROP_PADDING_RATIO = 0.03
CROP_BACKGROUND = 127
DINOV2_LAYER = 8  # mid-layer (0-indexed, 12 transformer blocks 중 8번째 block 출력)
INPUT_SIZE = 224
GRID_SIZE = INPUT_SIZE // 14  # ViT patch size 14 → 16x16 grid


def parse_yolo_polygon_line(line: str, W: int, H: int):
    parts = line.split()
    cid = int(parts[0])
    coords = list(map(float, parts[1:]))
    poly = np.array(
        [[coords[i] * W, coords[i + 1] * H] for i in range(0, len(coords), 2)],
        dtype=np.int32,
    )
    return cid, poly


def crop_with_mask_alpha(image: np.ndarray, mask: np.ndarray, padding_ratio: float = CROP_PADDING_RATIO):
    """기존 classifier (`src/classifier/dinov2.py:_crop_instance`)와 동일 방식: 정사각형 crop, mask 안 외는 gray 127."""
    H, W = image.shape[:2]
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None, None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    side = int(np.ceil(max(bbox_w, bbox_h) * (1.0 + padding_ratio * 2.0)))
    if side <= 0:
        return None, None

    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    crop_left = int(np.floor(cx - side * 0.5))
    crop_top = int(np.floor(cy - side * 0.5))
    crop_right = crop_left + side
    crop_bottom = crop_top + side

    src_x1 = max(0, crop_left)
    src_y1 = max(0, crop_top)
    src_x2 = min(W, crop_right)
    src_y2 = min(H, crop_bottom)

    crop = np.full((side, side, 3), CROP_BACKGROUND, dtype=np.uint8)
    crop_mask = np.zeros((side, side), dtype=bool)
    if src_x2 > src_x1 and src_y2 > src_y1:
        dst_x1 = src_x1 - crop_left
        dst_y1 = src_y1 - crop_top
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        crop[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]
        crop_mask[dst_y1:dst_y2, dst_x1:dst_x2] = mask[src_y1:src_y2, src_x1:src_x2] > 0
    crop[~crop_mask] = CROP_BACKGROUND
    return crop, crop_mask


def extract_patch_features(crop_bgr, crop_mask, processor, model, device):
    """DinoV2 forward (RGB) → mid-layer patch tokens → mask 영역 patches만 select.

    Returns:
        (n_selected, dim) feature array
        n_total: 객체의 grid 안 mask 픽셀 비율
    """
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(crop_rgb)
    inputs = processor(images=pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # HuggingFace transformers: hidden_states[0] = embeddings (pre-block),
    # hidden_states[i+1] = i-th transformer block output.
    # "Layer 8" PatchCore convention = post-8th-block = hidden_states[9].
    patch_tokens = outputs.hidden_states[DINOV2_LAYER + 1][0, 1:, :].cpu().numpy()  # (256, 768) — CLS 제외

    mask_resized = cv2.resize(
        crop_mask.astype(np.uint8), (GRID_SIZE, GRID_SIZE), interpolation=cv2.INTER_AREA
    ) > 0
    selected = patch_tokens[mask_resized.flatten()]
    return selected, int(mask_resized.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--class_id", type=int, required=True, help="라벨링 polygon class_id (data.yaml 매핑)")
    ap.add_argument("--class_name", type=str, required=True, help="output 파일명")
    ap.add_argument("--model", default="facebook/dinov2-base")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--labeled_base", default="data/labeled")
    ap.add_argument("--output", default=None)
    ap.add_argument("--exclude", default=None,
                    help="제외 목록 파일 — '<set>/<stem>:<line_idx>' 줄 단위 (# 주석 허용)")
    args = ap.parse_args()

    excluded = set()
    if args.exclude:
        for line in Path(args.exclude).read_text().splitlines():
            entry = line.split("#")[0].strip()
            if entry:
                excluded.add(entry)
        print(f"[exclude] {len(excluded)} entries: {sorted(excluded)}")

    output = Path(args.output) if args.output else ROOT / "weights" / f"patchcore_{args.class_name}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.model} → {args.device}")
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(args.device).eval()
    print(f"[layer] {DINOV2_LAYER} (hidden_states[{DINOV2_LAYER + 1}]), input {INPUT_SIZE}x{INPUT_SIZE}, grid {GRID_SIZE}x{GRID_SIZE}")

    labeled_base = ROOT / args.labeled_base
    set_dirs = sorted([d for d in labeled_base.iterdir() if d.is_dir()])
    print(f"[scan] {len(set_dirs)} set dirs: {[d.name for d in set_dirs]}")

    memory_bank = []
    patches_per_object = []
    obj_count = 0
    image_count = 0
    skipped_no_patches = 0

    for set_dir in set_dirs:
        labels_dir = set_dir / "train" / "labels"
        images_dir = set_dir / "train" / "images"
        if not labels_dir.exists():
            continue
        for txt in sorted(labels_dir.glob("*.txt")):
            stem = txt.stem
            bmp_path = images_dir / f"{stem}.bmp"
            if not bmp_path.exists():
                continue
            image = cv2.imread(str(bmp_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            H, W = image.shape[:2]

            image_used = False
            for line_idx, line in enumerate(txt.read_text().strip().split("\n")):
                if not line.strip():
                    continue
                if f"{set_dir.name}/{stem}:{line_idx}" in excluded:
                    print(f"  [exclude] {set_dir.name}/{stem}:{line_idx}")
                    continue
                try:
                    cid, poly = parse_yolo_polygon_line(line, W, H)
                except Exception:
                    continue
                if cid != args.class_id:
                    continue

                mask = np.zeros((H, W), dtype=np.uint8)
                cv2.fillPoly(mask, [poly], 255)

                crop, crop_mask = crop_with_mask_alpha(image, mask)
                if crop is None:
                    continue

                patches, n_grid = extract_patch_features(crop, crop_mask, processor, model, args.device)
                if len(patches) == 0:
                    skipped_no_patches += 1
                    continue

                memory_bank.append(patches)
                patches_per_object.append(len(patches))
                obj_count += 1
                image_used = True
            if image_used:
                image_count += 1
                if image_count % 20 == 0:
                    print(f"  [progress] images={image_count}, objects={obj_count}, patches_total={sum(patches_per_object)}")

    if not memory_bank:
        print(f"[error] No objects found for class_id={args.class_id}")
        return 1

    memory_bank = np.concatenate(memory_bank, axis=0).astype(np.float32)
    patches_per_object = np.array(patches_per_object)

    print(f"\n[result] class_id={args.class_id} ({args.class_name})")
    print(f"  images used:        {image_count}")
    print(f"  objects extracted:  {obj_count}")
    print(f"  patches/object:     min={patches_per_object.min()}, mean={patches_per_object.mean():.1f}, max={patches_per_object.max()}")
    print(f"  skipped (no patches inside mask grid): {skipped_no_patches}")
    print(f"  memory bank shape:  {memory_bank.shape}")
    print(f"  size:               {memory_bank.nbytes / 1e6:.1f} MB (fp32)")

    np.savez(
        output,
        memory_bank=memory_bank,
        class_id=args.class_id,
        class_name=args.class_name,
        dinov2_model=args.model,
        dinov2_layer=DINOV2_LAYER,
        input_size=INPUT_SIZE,
        grid_size=GRID_SIZE,
        n_objects=obj_count,
        n_images=image_count,
        patches_per_object=patches_per_object,
    )
    print(f"\n→ {output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
