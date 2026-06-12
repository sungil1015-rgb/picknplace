"""Multi-layer PatchCore 결합 정량 테스트 — 단일 vs {L4, L8, L11} 조합.

배경: PatchCore 원논문은 multi-layer concat이 기본. 사용자 제안 "4, 8, 12 레이어"
      → 우리 0-indexed 표기로 L4, L8, L11 (L11 = 마지막 block = classifier layer).

결합 방식 2종 (둘 다 per-layer 정규화 필수 — 스케일 30배 차):
  (a) patch-level: d_i = mean_L (d_{L,i} / s_L), object score = max_i d_i  (concat 근사)
  (b) score-level: z_obj = mean_L z_L(obj_score_L)  (객체 score의 z 평균)
  s_L = 정상 객체들의 patch 거리 중앙값 (layer별 고정 스케일).

조합: 단일 {4, 8, 11} + 쌍 {4+8, 8+11, 4+11} + 삼중 {4+8+11}
클래스: mango_v2 + haribo (비율 tear 3~7%, LOO, 80 obj × 3 variants)

Usage: CUDA_VISIBLE_DEVICES=0 python scripts/multilayer_combo.py
산출: outputs/eda/multilayer_combo.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.defect.synthesize import synthesize_tear

CROP_PADDING_RATIO = 0.03
CROP_BACKGROUND = 127
INPUT_SIZE = 224
GRID_SIZE = INPUT_SIZE // 14
USE_LAYERS = [4, 8, 11]
MB_DIR = ROOT / "outputs" / "eda" / "memory_banks"
EXCLUDE_FILE = ROOT / "outputs" / "eda" / "mango_exclude_list.txt"
N_VARIANTS = 3
MAX_OBJ = 80

TARGETS = [(2, "mango_v2", "mango_v2", True), (1, "haribo", "haribo", False)]

COMBOS = [
    ("L4", [4]), ("L8", [8]), ("L11", [11]),
    ("L4+L8", [4, 8]), ("L8+L11", [8, 11]), ("L4+L11", [4, 11]),
    ("L4+L8+L11", [4, 8, 11]),
]


def load_excluded():
    ex = set()
    if EXCLUDE_FILE.is_file():
        for line in EXCLUDE_FILE.read_text().splitlines():
            e = line.split("#")[0].strip()
            if e:
                ex.add(e)
    return ex


def parse_polygon(line, W, H):
    parts = line.split()
    cid = int(parts[0])
    coords = list(map(float, parts[1:]))
    poly = np.array([[coords[i] * W, coords[i + 1] * H] for i in range(0, len(coords), 2)], dtype=np.int32)
    return cid, poly


def crop_with_mask(img, mask):
    H, W = img.shape[:2]
    m = mask > 0 if mask.dtype != bool else mask
    ys, xs = np.where(m)
    if len(ys) == 0:
        return None, None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    side = int(np.ceil(max(x2 - x1, y2 - y1) * (1.0 + CROP_PADDING_RATIO * 2.0)))
    if side <= 0:
        return None, None
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    cl, ct = int(np.floor(cx - side * 0.5)), int(np.floor(cy - side * 0.5))
    sx1, sy1 = max(0, cl), max(0, ct)
    sx2, sy2 = min(W, cl + side), min(H, ct + side)
    crop = np.full((side, side, 3), CROP_BACKGROUND, dtype=np.uint8)
    cmask = np.zeros((side, side), dtype=bool)
    if sx2 > sx1 and sy2 > sy1:
        dx1, dy1 = sx1 - cl, sy1 - ct
        crop[dy1:dy1 + sy2 - sy1, dx1:dx1 + sx2 - sx1] = img[sy1:sy2, sx1:sx2]
        cmask[dy1:dy1 + sy2 - sy1, dx1:dx1 + sx2 - sx1] = m[sy1:sy2, sx1:sx2]
    crop[~cmask] = CROP_BACKGROUND
    return crop, cmask


def collect(poly_id, apply_exclude):
    excluded = load_excluded() if apply_exclude else set()
    samples = []
    gidx = 0
    for set_dir in sorted((ROOT / "data" / "labeled").iterdir()):
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
            for li, line in enumerate(txt.read_text().strip().split("\n")):
                if not line.strip():
                    continue
                if f"{set_dir.name}/{txt.stem}:{li}" in excluded:
                    continue
                try:
                    cid, _ = parse_polygon(line, 1224, 1024)
                except Exception:
                    continue
                if cid != poly_id:
                    continue
                samples.append({"bmp": bmp, "line": line, "gidx": gidx})
                gidx += 1
    return samples


def load_sample(s):
    img = cv2.imread(str(s["bmp"]))
    H, W = img.shape[:2]
    _, poly = parse_polygon(s["line"], W, H)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    return img, mask > 0


def extract_layers(crop, cmask, processor, model, device):
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    inputs = processor(images=Image.fromarray(rgb), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    gm = cv2.resize(cmask.astype(np.uint8), (GRID_SIZE, GRID_SIZE), interpolation=cv2.INTER_AREA) > 0
    sel = np.where(gm.flatten())[0]
    if len(sel) == 0:
        return None
    sel_t = torch.from_numpy(sel).to(device)
    return {L: out.hidden_states[L + 1][0, 1:, :][sel_t] for L in USE_LAYERS}


def loo_dists(p, mb, mb_sq, s, e, device):
    N = mb.shape[0]
    if s > 0 and e < N:
        idx = torch.cat([torch.arange(0, s, device=device), torch.arange(e, N, device=device)])
    elif s == 0:
        idx = torch.arange(e, N, device=device)
    else:
        idx = torch.arange(0, s, device=device)
    rest, rest_sq = mb[idx], mb_sq[idx]
    a2 = (p * p).sum(dim=1, keepdim=True)
    ab = p @ rest.T
    d2 = (a2 + rest_sq.unsqueeze(0) - 2 * ab).clamp(min=0)
    return d2.sqrt().min(dim=1).values.cpu().numpy()


def auroc_of(normal, defect):
    y = np.concatenate([np.zeros(len(normal)), np.ones(len(defect))])
    sc = np.concatenate([normal, defect])
    return float(roc_auc_score(y, sc))


def tpr_at_fpr(normal, defect, f):
    tau = np.quantile(np.asarray(normal), 1 - f)
    return float((np.asarray(defect) > tau).mean())


def main():
    from transformers import AutoImageProcessor, AutoModel
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
    rng = np.random.default_rng(42)

    all_results = {}
    for poly_id, name, prefix, apply_ex in TARGETS:
        print(f"\n=== {name} ===")
        mbs = {}
        cum = None
        for L in USE_LAYERS:
            d = np.load(MB_DIR / f"mb_L{L}_{prefix}.npz")
            t = torch.from_numpy(d["memory_bank"].astype(np.float32)).to(device)
            mbs[L] = (t, (t * t).sum(dim=1))
            if cum is None:
                cum = np.concatenate([[0], np.cumsum(d["patches_per_object"])])
        samples = collect(poly_id, apply_ex)
        assert len(samples) == len(cum) - 1
        rng_s = np.random.default_rng(42)
        if len(samples) > MAX_OBJ:
            idx = rng_s.choice(len(samples), MAX_OBJ, replace=False)
            samples = [samples[i] for i in idx]

        # 전 객체/변형의 per-layer per-patch 거리 저장
        normal_d = []   # [ {L: dists} ]
        defect_d = []
        for i, s in enumerate(samples):
            img, mask = load_sample(s)
            crop, cmask = crop_with_mask(img, mask)
            if crop is None:
                continue
            feats = extract_layers(crop, cmask, processor, model, device)
            if feats is None:
                continue
            se = int(cum[s["gidx"]]), int(cum[s["gidx"] + 1])
            normal_d.append({L: loo_dists(feats[L], *mbs[L], *se, device) for L in USE_LAYERS})
            for _ in range(N_VARIANTS):
                synth = synthesize_tear(img, mask, rng, area_mode="mask_ratio",
                                        area_ratio_range=(0.03, 0.07))
                scp, _ = crop_with_mask(synth, mask)
                if scp is None:
                    continue
                df = extract_layers(scp, cmask, processor, model, device)
                if df is None:
                    continue
                defect_d.append({L: loo_dists(df[L], *mbs[L], *se, device) for L in USE_LAYERS})
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(samples)}]")

        # per-layer 정규화 스케일: 정상 patch 거리 중앙값
        s_L = {L: float(np.median(np.concatenate([d[L] for d in normal_d]))) for L in USE_LAYERS}
        # score-level용: 정상 객체 score (max) 통계
        obj_stats = {}
        for L in USE_LAYERS:
            ns = np.array([d[L].max() for d in normal_d])
            obj_stats[L] = (float(ns.mean()), float(ns.std() + 1e-9))
        print(f"  scale s_L: { {L: round(v, 2) for L, v in s_L.items()} }")

        def patch_combo_score(d, layers):
            norm = np.stack([d[L] / s_L[L] for L in layers])   # (n_layers, n_patches)
            return float(norm.mean(axis=0).max())

        def score_combo_score(d, layers):
            zs = [(d[L].max() - obj_stats[L][0]) / obj_stats[L][1] for L in layers]
            return float(np.mean(zs))

        rows = []
        for label, layers in COMBOS:
            for method, fn in (("patch", patch_combo_score), ("score", score_combo_score)):
                if len(layers) == 1 and method == "score":
                    continue   # 단일은 patch=원래 max와 동치 (스케일 무관)
                ns = [fn(d, layers) for d in normal_d]
                ds = [fn(d, layers) for d in defect_d]
                au = auroc_of(ns, ds)
                t5 = tpr_at_fpr(ns, ds, 0.05)
                rows.append({"combo": label, "method": method, "auroc": au, "tpr_at_fpr5": t5})
                print(f"  {label:>10} [{method}]: AUROC={au:.3f} TPR@FPR5={t5:.2f}")
        all_results[name] = {"scale": s_L, "rows": rows}

    out = ROOT / "outputs" / "eda" / "multilayer_combo.json"
    with out.open("w") as f:
        json.dump(all_results, f, indent=1, ensure_ascii=False)
    print(f"\n→ {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
