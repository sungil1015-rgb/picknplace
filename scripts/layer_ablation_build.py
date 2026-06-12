"""Layer ablation — 한 번 forward로 layer 1~11 메모리뱅크 동시 빌드.

목적:
  현재 PatchCore-lite는 layer 8 단일 고정. 이론 근거(PatchCore 원논문 + ViT mid-layer 통설)뿐,
  우리 데이터에서 layer별 실증 비교 없음. EDA로 layer × class AUROC 측정해서 베스트 layer 확인.

전략:
  output_hidden_states=True 한 번 forward → hidden_states[0..12] 13개 동시 추출.
  이 중 layer 1~11 (= hidden_states[2..12]) 의 patches를 동시에 메모리뱅크에 적재.

제외:
  layer 0 (hidden_states[1]): 사실상 patch embedding 후 첫 block 출력. 너무 얕아 결함 신호 약함 — 다만 실증은 함.
  주의: 여기서 "layer L" 은 PatchCore convention의 post-L-th-block 출력. hidden_states[L+1]에 해당.
       hidden_states[0] (pre-block embedding) 과 hidden_states[12] (semantic, classifier 사용 중)는 제외.

산출물: outputs/eda/memory_banks/mb_L{L}_{class}.npz × 11 layer × 3 class = 33 파일

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/layer_ablation_build.py
"""
from __future__ import annotations

import sys
import time
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
INPUT_SIZE = 224
GRID_SIZE = INPUT_SIZE // 14   # 16

LAYERS = list(range(1, 12))    # 1..11 (hidden_states[2..12])
EXCLUDED_LAYERS = {
    0: "hidden_states[1] = post-block-0 출력. 매우 얕은 raw edge/color blob 수준. "
       "정상 분산이 너무 커서 patches 간 거리가 결함/정상 구분에 무의미. 다만 layer 1은 실증 비교 위해 포함.",
    12: "hidden_states[13]은 존재하지 않음 (block 12개). "
        "hidden_states[12] = post-block-11 = 마지막 layer는 DinoV2 KNN classifier가 사용 중. "
        "Semantic 표현이라 'metal_case 정상' vs 'metal_case 스크래치' 가 같은 임베딩으로 수렴 → anomaly 거리 손실.",
}

# 활성 클래스만 (polygon class_id, 출력 이름)
TARGETS = [
    (1, "haribo"),
    (3, "metal_case"),
    (5, "pencil_case"),
]


def parse_yolo_polygon_line(line: str, W: int, H: int):
    parts = line.split()
    cid = int(parts[0])
    coords = list(map(float, parts[1:]))
    poly = np.array(
        [[coords[i] * W, coords[i + 1] * H] for i in range(0, len(coords), 2)],
        dtype=np.int32,
    )
    return cid, poly


def crop_with_mask(image: np.ndarray, mask: np.ndarray, pad_ratio: float = CROP_PADDING_RATIO):
    """build_patchcore_refs.py 와 동일 crop. mask 밖은 gray 127."""
    H, W = image.shape[:2]
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None, None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    bbox_w, bbox_h = x2 - x1, y2 - y1
    side = int(np.ceil(max(bbox_w, bbox_h) * (1.0 + pad_ratio * 2.0)))
    if side <= 0:
        return None, None
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    cl = int(np.floor(cx - side * 0.5))
    ct = int(np.floor(cy - side * 0.5))
    cr, cb = cl + side, ct + side
    sx1, sy1 = max(0, cl), max(0, ct)
    sx2, sy2 = min(W, cr), min(H, cb)
    crop = np.full((side, side, 3), CROP_BACKGROUND, dtype=np.uint8)
    cmask = np.zeros((side, side), dtype=bool)
    if sx2 > sx1 and sy2 > sy1:
        dx1, dy1 = sx1 - cl, sy1 - ct
        dx2, dy2 = dx1 + (sx2 - sx1), dy1 + (sy2 - sy1)
        crop[dy1:dy2, dx1:dx2] = image[sy1:sy2, sx1:sx2]
        cmask[dy1:dy2, dx1:dx2] = mask[sy1:sy2, sx1:sx2] > 0
    crop[~cmask] = CROP_BACKGROUND
    return crop, cmask


def extract_patches_all_layers(crop_bgr, crop_mask, processor, model, device):
    """forward 1번 → 11 layer × patches 동시 추출.

    Returns: dict {L: np.ndarray (n_selected, D)}, n_grid_selected (int)
    """
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(crop_rgb)
    inputs = processor(images=pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    mask_resized = cv2.resize(
        crop_mask.astype(np.uint8), (GRID_SIZE, GRID_SIZE), interpolation=cv2.INTER_AREA
    ) > 0
    sel = mask_resized.flatten()
    n_grid_selected = int(sel.sum())
    if n_grid_selected == 0:
        return None, 0

    out = {}
    for L in LAYERS:
        # PatchCore convention: layer L = post-L-th-block = hidden_states[L+1]
        # 단, hidden_states[0]=embeddings, hidden_states[i]=post-(i-1)-th-block (i>=1)
        # 그래서 "layer L (post-L-th-block)" = hidden_states[L+1]
        # 기존 코드의 DINOV2_LAYER=8 → hidden_states[9] 와 일관
        patch_tokens = outputs.hidden_states[L + 1][0, 1:, :].cpu().numpy()  # (256, D)
        out[L] = patch_tokens[sel]
    return out, n_grid_selected


def build_for_class(target, processor, model, device, output_dir):
    poly_id, name = target
    print(f"\n=== build {name} (polygon {poly_id}) ===")

    labeled_base = ROOT / "data" / "labeled"
    set_dirs = sorted([d for d in labeled_base.iterdir() if d.is_dir()])

    mb_by_layer = {L: [] for L in LAYERS}
    patches_per_object = []
    obj_count = 0
    image_count = 0
    skipped = 0
    t0 = time.time()

    for set_dir in set_dirs:
        labels = set_dir / "train" / "labels"
        images = set_dir / "train" / "images"
        if not labels.exists():
            continue
        for txt in sorted(labels.glob("*.txt")):
            bmp = images / f"{txt.stem}.bmp"
            if not bmp.exists():
                continue
            img = cv2.imread(str(bmp))
            if img is None:
                continue
            H, W = img.shape[:2]
            image_used = False
            for line in txt.read_text().strip().split("\n"):
                if not line.strip():
                    continue
                try:
                    cid, poly = parse_yolo_polygon_line(line, W, H)
                except Exception:
                    continue
                if cid != poly_id:
                    continue
                mask = np.zeros((H, W), dtype=np.uint8)
                cv2.fillPoly(mask, [poly], 255)
                crop, cmask = crop_with_mask(img, mask)
                if crop is None:
                    skipped += 1
                    continue
                patches_dict, n_sel = extract_patches_all_layers(
                    crop, cmask, processor, model, device
                )
                if patches_dict is None or n_sel == 0:
                    skipped += 1
                    continue
                for L in LAYERS:
                    mb_by_layer[L].append(patches_dict[L])
                patches_per_object.append(n_sel)
                obj_count += 1
                image_used = True
            if image_used:
                image_count += 1
                if image_count % 20 == 0:
                    elapsed = time.time() - t0
                    print(f"  [progress] images={image_count}, objects={obj_count}, "
                          f"elapsed={elapsed:.1f}s")

    if obj_count == 0:
        print(f"  no objects found for {name}")
        return

    patches_per_object = np.array(patches_per_object)
    print(f"\n  result: images={image_count}, objects={obj_count}, "
          f"patches/obj mean={patches_per_object.mean():.1f}, skipped={skipped}")

    for L in LAYERS:
        mb = np.concatenate(mb_by_layer[L], axis=0).astype(np.float32)
        out_path = output_dir / f"mb_L{L}_{name}.npz"
        np.savez(
            out_path,
            memory_bank=mb,
            class_name=name,
            polygon_id=poly_id,
            dinov2_layer=L,
            input_size=INPUT_SIZE,
            grid_size=GRID_SIZE,
            n_objects=obj_count,
            n_images=image_count,
            patches_per_object=patches_per_object,
        )
        size_mb = mb.nbytes / 1e6
        print(f"  L{L:>2}: shape={mb.shape}, size={size_mb:.0f}MB → {out_path.relative_to(ROOT)}")


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")
    print(f"[layers] {LAYERS}")
    print(f"[excluded] {sorted(EXCLUDED_LAYERS.keys())}")
    print(f"[targets] {[t[1] for t in TARGETS]}")

    print(f"\n[load] facebook/dinov2-base")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()

    output_dir = ROOT / "outputs" / "eda" / "memory_banks"
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for target in TARGETS:
        build_for_class(target, processor, model, device, output_dir)
    total = time.time() - t0
    print(f"\n[done] total elapsed: {total:.1f}s ({total/60:.1f} min)")
    print(f"[output] {output_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
