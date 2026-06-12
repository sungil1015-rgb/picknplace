"""mango layer ablation — L1~L11 비교 (bank v2 정제 + 비율 tear 3~7%).

haribo도 동일 조건 (비율 tear)으로 재평가 — 공정 비교.
(haribo 11-layer 메모리뱅크는 기존 outputs/eda/memory_banks/mb_L{L}_haribo.npz 재사용,
 mango는 오염 제외 반영해 새로 빌드 → mb_L{L}_mango_v2.npz)

metal/pencil은 scratch (불변)라 기존 layer ablation 결과 그대로 유효.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/layer_ablation_mango.py --stage build
    CUDA_VISIBLE_DEVICES=0 python scripts/layer_ablation_mango.py --stage eval
산출: outputs/eda/layer_ablation_mango.json, outputs/figs/layer_ablation/auroc_vs_layer_v2.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.defect.synthesize import synthesize_tear

CROP_PADDING_RATIO = 0.03
CROP_BACKGROUND = 127
INPUT_SIZE = 224
GRID_SIZE = INPUT_SIZE // 14
LAYERS = list(range(1, 12))
EXCLUDE_FILE = ROOT / "outputs" / "eda" / "mango_exclude_list.txt"
MB_DIR = ROOT / "outputs" / "eda" / "memory_banks"
N_VARIANTS = 3
MAX_OBJ = 80

# (poly_id, name, bank_prefix, apply_exclude)
TARGETS = [
    (2, "mango_v2", "mango_v2", True),
    (1, "haribo", "haribo", False),
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


def extract_all_layers(crop, cmask, processor, model, device, to_numpy=False):
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
    res = {}
    for L in LAYERS:
        t = out.hidden_states[L + 1][0, 1:, :][sel_t]
        res[L] = t.cpu().numpy() if to_numpy else t
    return res


def stage_build(processor, model, device):
    """mango v2 — 11 layer 메모리뱅크 동시 빌드 (오염 제외)."""
    samples = collect(2, apply_exclude=True)
    print(f"[build] mango_v2: {len(samples)} objects")
    mb = {L: [] for L in LAYERS}
    ppo = []
    t0 = time.time()
    for i, s in enumerate(samples):
        img, mask = load_sample(s)
        crop, cmask = crop_with_mask(img, mask)
        if crop is None:
            continue
        feats = extract_all_layers(crop, cmask, processor, model, device, to_numpy=True)
        if feats is None:
            continue
        for L in LAYERS:
            mb[L].append(feats[L])
        ppo.append(len(feats[LAYERS[0]]))
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(samples)}] {time.time()-t0:.0f}s")
    ppo = np.array(ppo)
    MB_DIR.mkdir(parents=True, exist_ok=True)
    for L in LAYERS:
        arr = np.concatenate(mb[L], axis=0).astype(np.float32)
        np.savez(MB_DIR / f"mb_L{L}_mango_v2.npz",
                 memory_bank=arr, dinov2_layer=L, input_size=INPUT_SIZE,
                 grid_size=GRID_SIZE, n_objects=len(ppo), patches_per_object=ppo)
    print(f"[build] done — {len(ppo)} obj × 11 layers, {time.time()-t0:.0f}s")


def loo_score(p, mb, mb_sq, s, e, device):
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
    return float(d2.sqrt().min(dim=1).values.max())


def stage_eval(processor, model, device):
    rng = np.random.default_rng(42)
    results = {}
    for poly_id, name, prefix, apply_ex in TARGETS:
        print(f"\n=== eval {name} (ratio tear 3~7%) ===")
        mbs = {}
        cum = None
        for L in LAYERS:
            d = np.load(MB_DIR / f"mb_L{L}_{prefix}.npz")
            t = torch.from_numpy(d["memory_bank"].astype(np.float32)).to(device)
            mbs[L] = (t, (t * t).sum(dim=1))
            if cum is None:
                cum = np.concatenate([[0], np.cumsum(d["patches_per_object"])])
        samples = collect(poly_id, apply_ex)
        n_bank = len(cum) - 1
        assert len(samples) == n_bank, f"{name}: collect {len(samples)} != bank {n_bank}"
        rng_s = np.random.default_rng(42)
        if len(samples) > MAX_OBJ:
            idx = rng_s.choice(len(samples), MAX_OBJ, replace=False)
            samples = [samples[i] for i in idx]
        normal = {L: [] for L in LAYERS}
        defect = {L: [] for L in LAYERS}
        for i, s in enumerate(samples):
            img, mask = load_sample(s)
            crop, cmask = crop_with_mask(img, mask)
            if crop is None:
                continue
            feats = extract_all_layers(crop, cmask, processor, model, device)
            if feats is None:
                continue
            se = int(cum[s["gidx"]]), int(cum[s["gidx"] + 1])
            for L in LAYERS:
                normal[L].append(loo_score(feats[L], *mbs[L], *se, device))
            for _ in range(N_VARIANTS):
                synth = synthesize_tear(img, mask, rng, area_mode="mask_ratio",
                                        area_ratio_range=(0.03, 0.07))
                scp, _ = crop_with_mask(synth, mask)
                if scp is None:
                    continue
                dfeats = extract_all_layers(scp, cmask, processor, model, device)
                if dfeats is None:
                    continue
                for L in LAYERS:
                    defect[L].append(loo_score(dfeats[L], *mbs[L], *se, device))
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(samples)}]")
        rows = []
        for L in LAYERS:
            ns, ds = normal[L], defect[L]
            y = np.concatenate([np.zeros(len(ns)), np.ones(len(ds))])
            sc = np.concatenate([ns, ds])
            au = float(roc_auc_score(y, sc))
            fpr, tpr, thr = roc_curve(y, sc)
            f1 = 2 * tpr * (1 - fpr) / np.maximum(tpr + (1 - fpr), 1e-9)
            bi = int(np.argmax(f1))
            rows.append({"layer": L, "auroc": au, "tau": float(thr[bi]),
                         "tpr": float(tpr[bi]), "fpr": float(fpr[bi])})
            print(f"  L{L:>2}: AUROC={au:.3f} TPR={tpr[bi]:.2f} FPR={fpr[bi]:.2f}")
        results[name] = rows

    out = ROOT / "outputs" / "eda" / "layer_ablation_mango.json"
    with out.open("w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    print(f"→ {out.relative_to(ROOT)}")

    # figure: mango_v2 + haribo (ratio tear) 곡선
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, rows in results.items():
        ax.plot([r["layer"] for r in rows], [r["auroc"] for r in rows], "o-",
                label=f"{name} (ratio tear)", linewidth=2, markersize=7)
        best = max(rows, key=lambda r: r["auroc"])
        ax.annotate(f"best L{best['layer']}", xy=(best["layer"], best["auroc"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.axvline(8, color="gray", linestyle="--", alpha=0.5, label="current (L8)")
    ax.set_xticks(LAYERS)
    ax.set_xlabel("DinoV2 layer")
    ax.set_ylabel("AUROC (LOO, ratio tear 3~7%)")
    ax.set_title("Layer ablation v2 — mango (정제 bank) + haribo, 비율 tear 동일 조건")
    ax.grid(alpha=0.3)
    ax.set_ylim(0.5, 1.02)
    ax.legend(loc="lower center")
    fig_out = ROOT / "outputs" / "figs" / "layer_ablation" / "auroc_vs_layer_v2.png"
    fig_out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_out, dpi=120, bbox_inches="tight")
    print(f"→ {fig_out.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["build", "eval", "all"])
    args = ap.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
    if args.stage in ("build", "all"):
        stage_build(processor, model, device)
    if args.stage in ("eval", "all"):
        stage_eval(processor, model, device)


if __name__ == "__main__":
    main()
