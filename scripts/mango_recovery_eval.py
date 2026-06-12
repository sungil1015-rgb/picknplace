"""mango 복구 LOO ROC 재평가 — tear 비율화 + top-k 통계 + 라벨 정제 반영.

변경점 (2026-06-12 근본 원인 분석 후속):
  1. tear 합성: 절대픽셀 (400~1500) → mask 면적 비율 3~7% (haribo 현행 ~4.5% 기준 회귀 호환)
  2. score 통계: max 와 top_k_mean(k=4) 동시 산출 비교 (글레어 단일 patch 민감도)
  3. mango: 라벨 오염 1건 제외 (outputs/eda/mango_exclude_list.txt), 재빌드 bank 사용
  4. fusion 캘리브용 정상 통계 (mu/sigma/p95/p99) json 출력

클래스별 별도 프로세스 실행 (GPU 분산):
    CUDA_VISIBLE_DEVICES=0 python scripts/mango_recovery_eval.py --target mango
    CUDA_VISIBLE_DEVICES=0 python scripts/mango_recovery_eval.py --target haribo
    CUDA_VISIBLE_DEVICES=2 python scripts/mango_recovery_eval.py --target metal_case
    CUDA_VISIBLE_DEVICES=2 python scripts/mango_recovery_eval.py --target pencil_case

산출: outputs/eda/recovery_eval_<class>.json, outputs/figs/mango_recovery/roc_<class>.png
"""
from __future__ import annotations

import argparse
import json
import sys
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

from src.defect.synthesize import synthesize_scratch, synthesize_tear

CROP_PADDING_RATIO = 0.03
CROP_BACKGROUND = 127
DINOV2_LAYER = 8
INPUT_SIZE = 224
GRID_SIZE = INPUT_SIZE // 14
N_SYNTH_VARIANTS = 5
MAX_OBJ = 80
TOP_K = 4

# name → (polygon_id, synth_type, bank_file, prev_yaml_tau)
TARGETS = {
    "haribo":      (1, "tear",    "patchcore_haribo.npz",      8.6),
    "mango":       (2, "tear",    "patchcore_mango_v2.npz",    None),   # v2 = 정제 재빌드
    "metal_case":  (3, "scratch", "patchcore_metal_case.npz",  9.0),
    "pencil_case": (5, "scratch", "patchcore_pencil_case.npz", 9.2),
}
EXCLUDE_FILE = ROOT / "outputs" / "eda" / "mango_exclude_list.txt"


def load_excluded():
    excluded = set()
    if EXCLUDE_FILE.is_file():
        for line in EXCLUDE_FILE.read_text().splitlines():
            entry = line.split("#")[0].strip()
            if entry:
                excluded.add(entry)
    return excluded


def parse_polygon(line, W, H):
    parts = line.split()
    cid = int(parts[0])
    coords = list(map(float, parts[1:]))
    poly = np.array([[coords[i] * W, coords[i + 1] * H] for i in range(0, len(coords), 2)], dtype=np.int32)
    return cid, poly


def crop_with_mask(img, mask):
    H, W = img.shape[:2]
    m = mask > 0
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


def extract_patches(crop_bgr, crop_mask, processor, model, device):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    inputs = processor(images=pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    patch_tokens = outputs.hidden_states[DINOV2_LAYER + 1][0, 1:, :]
    grid_mask = cv2.resize(crop_mask.astype(np.uint8), (GRID_SIZE, GRID_SIZE), interpolation=cv2.INTER_AREA) > 0
    sel = np.where(grid_mask.flatten())[0]
    if len(sel) == 0:
        return None
    return patch_tokens[torch.from_numpy(sel).to(device)]


def loo_nn_dists(patches, mb, mb_sq, ex_start, ex_end):
    """LOO patch별 NN 거리 (max/top-k 둘 다 계산 위해 벡터 반환)."""
    N = mb.shape[0]
    if ex_start > 0 and ex_end < N:
        idx = torch.cat([
            torch.arange(0, ex_start, device=mb.device),
            torch.arange(ex_end, N, device=mb.device),
        ])
    elif ex_start == 0:
        idx = torch.arange(ex_end, N, device=mb.device)
    else:
        idx = torch.arange(0, ex_start, device=mb.device)
    rest = mb[idx]
    rest_sq = mb_sq[idx]
    a2 = (patches * patches).sum(dim=1, keepdim=True)
    ab = patches @ rest.T
    d2 = (a2 + rest_sq.unsqueeze(0) - 2 * ab).clamp(min=0)
    return d2.sqrt().min(dim=1).values   # (n_patches,)


def agg_scores(nn_dist):
    s_max = float(nn_dist.max())
    k = min(TOP_K, len(nn_dist))
    s_topk = float(torch.topk(nn_dist, k).values.mean())
    return s_max, s_topk


def collect_with_idx(target_class, labeled_base, excluded, apply_exclude):
    """빌드와 동일 순서로 (img, mask, global_idx) 수집. 제외 항목은 빌드와 동일하게 skip."""
    samples = []
    global_idx = 0
    for set_dir in sorted(labeled_base.iterdir()):
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
            img = None
            for line_idx, line in enumerate(txt.read_text().strip().split("\n")):
                if not line.strip():
                    continue
                if apply_exclude and f"{set_dir.name}/{txt.stem}:{line_idx}" in excluded:
                    continue
                try:
                    cid, poly = parse_polygon(line, 1224, 1024)
                except Exception:
                    continue
                if cid != target_class:
                    continue
                if img is None:
                    img = cv2.imread(str(bmp))
                    if img is None:
                        break
                H, W = img.shape[:2]
                cid, poly = parse_polygon(line, W, H)
                mask = np.zeros((H, W), dtype=np.uint8)
                cv2.fillPoly(mask, [poly], 255)
                samples.append((img, mask > 0, global_idx))
                global_idx += 1
    return samples


def roc_stats(normal_scores, defect_scores):
    ns, ds = np.array(normal_scores), np.array(defect_scores)
    y = np.concatenate([np.zeros(len(ns)), np.ones(len(ds))])
    s = np.concatenate([ns, ds])
    auroc = roc_auc_score(y, s)
    fpr, tpr, thr = roc_curve(y, s)
    f1 = 2 * tpr * (1 - fpr) / np.maximum(tpr + (1 - fpr), 1e-9)
    bi = int(np.argmax(f1))
    return {
        "auroc": float(auroc),
        "f1_best_tau": float(thr[bi]),
        "f1": float(f1[bi]),
        "tpr": float(tpr[bi]),
        "fpr": float(fpr[bi]),
        "normal_mu": float(ns.mean()),
        "normal_sigma": float(ns.std()),
        "normal_p95": float(np.percentile(ns, 95)),
        "normal_p99": float(np.percentile(ns, 99)),
        "defect_mu": float(ds.mean()),
        "separated": bool(np.percentile(ds, 1) >= np.percentile(ns, 99)),
        "fpr_curve": fpr.tolist(),
        "tpr_curve": tpr.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=list(TARGETS.keys()))
    args = ap.parse_args()

    name = args.target
    poly_id, synth_type, bank_file, prev_tau = TARGETS[name]
    apply_exclude = (name == "mango")
    excluded = load_excluded() if apply_exclude else set()

    from transformers import AutoImageProcessor, AutoModel
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[{name}] device={device}, synth={synth_type} "
          f"({'ratio 3~7%' if synth_type == 'tear' else 'absolute (불변)'}), "
          f"exclude={len(excluded)}")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()

    bank_path = ROOT / "weights" / bank_file
    data = np.load(bank_path)
    mb = torch.from_numpy(data["memory_bank"].astype(np.float32)).to(device)
    mb_sq = (mb * mb).sum(dim=1)
    ppo = data["patches_per_object"]
    cum = np.concatenate([[0], np.cumsum(ppo)])
    print(f"[{name}] bank {bank_file}: {mb.shape}, n_obj={len(ppo)}")

    samples = collect_with_idx(poly_id, ROOT / "data" / "labeled", excluded, apply_exclude)
    print(f"[{name}] collected {len(samples)} objects (bank n_obj={len(ppo)} — 일치해야 함)")
    assert len(samples) == len(ppo), f"collect({len(samples)}) != bank({len(ppo)}) — 순서/제외 불일치"

    rng = np.random.default_rng(42)
    if len(samples) > MAX_OBJ:
        idx_sub = rng.choice(len(samples), MAX_OBJ, replace=False)
        samples = [samples[i] for i in idx_sub]

    res = {"max": {"normal": [], "defect": []},
           "top_k": {"normal": [], "defect": []}}

    for i, (img, mask, gidx) in enumerate(samples):
        ex_s, ex_e = int(cum[gidx]), int(cum[gidx + 1])
        crop, cmask = crop_with_mask(img, mask)
        if crop is None:
            continue
        p = extract_patches(crop, cmask, processor, model, device)
        if p is None:
            continue
        nn = loo_nn_dists(p, mb, mb_sq, ex_s, ex_e)
        s_max, s_topk = agg_scores(nn)
        res["max"]["normal"].append(s_max)
        res["top_k"]["normal"].append(s_topk)

        for v in range(N_SYNTH_VARIANTS):
            if synth_type == "tear":
                synth = synthesize_tear(img, mask, rng,
                                        area_mode="mask_ratio",
                                        area_ratio_range=(0.03, 0.07))
            else:
                synth = synthesize_scratch(img, mask, rng)
            sc, _ = crop_with_mask(synth, mask)
            if sc is None:
                continue
            dp = extract_patches(sc, cmask, processor, model, device)
            if dp is None:
                continue
            dn = loo_nn_dists(dp, mb, mb_sq, ex_s, ex_e)
            d_max, d_topk = agg_scores(dn)
            res["max"]["defect"].append(d_max)
            res["top_k"]["defect"].append(d_topk)

        if (i + 1) % 20 == 0:
            print(f"[{name}] [{i+1}/{len(samples)}]")

    out = {"class": name, "bank": bank_file, "synth": synth_type,
           "tear_mode": "mask_ratio 3~7%" if synth_type == "tear" else "absolute",
           "prev_yaml_tau": prev_tau, "top_k": TOP_K}
    for agg in ("max", "top_k"):
        st = roc_stats(res[agg]["normal"], res[agg]["defect"])
        out[agg] = {k: v for k, v in st.items() if not k.endswith("_curve")}
        out[agg + "_curves"] = {"fpr": st["fpr_curve"], "tpr": st["tpr_curve"]}
        print(f"[{name}] {agg:>6}: AUROC={st['auroc']:.3f} tau*={st['f1_best_tau']:.2f} "
              f"TPR={st['tpr']:.2f} FPR={st['fpr']:.2f} "
              f"normal mu={st['normal_mu']:.2f} sd={st['normal_sigma']:.2f} "
              f"p99={st['normal_p99']:.2f} {'SEP' if st['separated'] else 'OVER'}")

    eda_dir = ROOT / "outputs" / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    with (eda_dir / f"recovery_eval_{name}.json").open("w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    # ROC figure (max vs top_k)
    fig_dir = ROOT / "outputs" / "figs" / "mango_recovery"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    for agg, color in (("max", "tab:blue"), ("top_k", "tab:orange")):
        c = out[agg + "_curves"]
        ax.plot(c["fpr"], c["tpr"], color=color, lw=2,
                label=f"{agg} (AUROC={out[agg]['auroc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"{name} — recovery LOO ROC ({out['tear_mode']})")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / f"roc_{name}.png", dpi=110, bbox_inches="tight")
    print(f"[{name}] done → recovery_eval_{name}.json, roc_{name}.png")


if __name__ == "__main__":
    main()
