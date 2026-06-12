"""mango fusion detector 전방위 정량평가 — 9종 테스트 배터리.

bank: weights/patchcore_mango_v2.npz (오염 정제, 309 obj). 모든 score는 LOO (자기 patch 제외).
기준 운영점: τ_ref = 8.72 (80샘플 p95, recovery_eval) — drift 비교용 고정 참조.

Usage (테스트별 별도 프로세스, GPU 분산):
    CUDA_VISIBLE_DEVICES=0 python scripts/mango_full_eval.py --test severity
    CUDA_VISIBLE_DEVICES=2 python scripts/mango_full_eval.py --test photometric
    ... (loso, geometric, defect_types, stress, aggregate, fulldata, latency)

산출: outputs/eda/fulleval_<test>.json + outputs/figs/mango_fulleval/<test>.png
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
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.defect.synthesize import synthesize_scratch, synthesize_tear, _create_random_blob

CROP_PADDING_RATIO = 0.03
CROP_BACKGROUND = 127
DINOV2_LAYER = 8
INPUT_SIZE = 224
GRID_SIZE = INPUT_SIZE // 14
MANGO_ID = 2
TAU_REF = 8.72          # 80샘플 p95 (recovery_eval 기준)
BANK = "patchcore_mango_v2.npz"
EXCLUDE_FILE = ROOT / "outputs" / "eda" / "mango_exclude_list.txt"

EDA = ROOT / "outputs" / "eda"
FIGS = ROOT / "outputs" / "figs" / "mango_fulleval"


# ─── 공통 유틸 ────────────────────────────────────────────────────────────────
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


def collect(labeled_base, excluded):
    """빌드 v2와 동일 순서 — (img_path, mask_poly_line, set_name, global_idx). lazy load."""
    samples = []
    gidx = 0
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
            for li, line in enumerate(txt.read_text().strip().split("\n")):
                if not line.strip():
                    continue
                if f"{set_dir.name}/{txt.stem}:{li}" in excluded:
                    continue
                try:
                    cid, _ = parse_polygon(line, 1224, 1024)
                except Exception:
                    continue
                if cid != MANGO_ID:
                    continue
                samples.append({"bmp": bmp, "line": line, "set": set_dir.name, "gidx": gidx})
                gidx += 1
    return samples


def load_sample(s):
    img = cv2.imread(str(s["bmp"]))
    H, W = img.shape[:2]
    _, poly = parse_polygon(s["line"], W, H)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    return img, mask > 0


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


class Scorer:
    def __init__(self, device):
        from transformers import AutoImageProcessor, AutoModel
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
        data = np.load(ROOT / "weights" / BANK)
        self.mb = torch.from_numpy(data["memory_bank"].astype(np.float32)).to(device)
        self.mb_sq = (self.mb * self.mb).sum(dim=1)
        self.cum = np.concatenate([[0], np.cumsum(data["patches_per_object"])])
        self.n_obj = len(data["patches_per_object"])

    def patches(self, crop, cmask):
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=Image.fromarray(rgb), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        tokens = out.hidden_states[DINOV2_LAYER + 1][0, 1:, :]
        gm = cv2.resize(cmask.astype(np.uint8), (GRID_SIZE, GRID_SIZE), interpolation=cv2.INTER_AREA) > 0
        sel = np.where(gm.flatten())[0]
        if len(sel) == 0:
            return None
        return tokens[torch.from_numpy(sel).to(self.device)]

    def nn_dists(self, p, gidx=None, bank_idx=None):
        """LOO (gidx 지정 시 자기 patch 제외) 또는 임의 bank subset (bank_idx)."""
        if bank_idx is not None:
            mb, mb_sq = self.mb[bank_idx], self.mb_sq[bank_idx]
        elif gidx is not None:
            s, e = int(self.cum[gidx]), int(self.cum[gidx + 1])
            idx = torch.cat([torch.arange(0, s, device=self.device),
                             torch.arange(e, self.mb.shape[0], device=self.device)])
            mb, mb_sq = self.mb[idx], self.mb_sq[idx]
        else:
            mb, mb_sq = self.mb, self.mb_sq
        a2 = (p * p).sum(dim=1, keepdim=True)
        ab = p @ mb.T
        d2 = (a2 + mb_sq.unsqueeze(0) - 2 * ab).clamp(min=0)
        return d2.sqrt().min(dim=1).values

    def score_max(self, crop, cmask, gidx=None, bank_idx=None):
        p = self.patches(crop, cmask)
        if p is None:
            return None
        return float(self.nn_dists(p, gidx, bank_idx).max())


def subsample(samples, n=80, seed=42):
    rng = np.random.default_rng(seed)
    if len(samples) <= n:
        return samples
    idx = rng.choice(len(samples), n, replace=False)
    return [samples[i] for i in idx]


def auroc_of(normal, defect):
    y = np.concatenate([np.zeros(len(normal)), np.ones(len(defect))])
    s = np.concatenate([normal, defect])
    return float(roc_auc_score(y, s))


def tpr_at_fpr(normal, defect, fpr_target):
    ns = np.sort(np.asarray(normal))
    tau = np.quantile(ns, 1 - fpr_target)
    return float((np.asarray(defect) > tau).mean()), float(tau)


def save_json(name, obj):
    EDA.mkdir(parents=True, exist_ok=True)
    with (EDA / f"fulleval_{name}.json").open("w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    print(f"[{name}] → fulleval_{name}.json")


# ─── 합성 결함 추가 종류 ──────────────────────────────────────────────────────
def synth_hole(img, mask, rng):
    """구멍/펑크 — 작은 어두운 원."""
    out = img.copy()
    m = mask > 0 if mask.dtype != bool else mask
    er = cv2.erode(m.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    ys, xs = np.where(er if er.sum() > 50 else m)
    if len(ys) == 0:
        return out
    i = rng.integers(len(ys))
    r = int(np.sqrt(m.sum() * rng.uniform(0.01, 0.03) / np.pi))
    cv2.circle(out, (int(xs[i]), int(ys[i])), max(r, 4), (15, 15, 20), -1, cv2.LINE_AA)
    return out


def synth_stain(img, mask, rng, dark=True):
    """얼룩 — 반투명 blob (누액/오염 시뮬)."""
    out = img.copy()
    m = mask > 0 if mask.dtype != bool else mask
    ys, xs = np.where(m)
    if len(ys) == 0:
        return out
    i = rng.integers(len(ys))
    area = int(m.sum() * rng.uniform(0.04, 0.10))
    blob = _create_random_blob(out.shape[:2], (int(xs[i]), int(ys[i])), area, rng) & m
    if blob.sum() == 0:
        return out
    if dark:
        out[blob] = (out[blob].astype(np.float32) * 0.45).astype(np.uint8)
    else:
        out[blob] = np.clip(out[blob].astype(np.float32) * 1.0 + 90, 0, 255).astype(np.uint8)
    return out


def synth_tear_jelly(img, mask, rng):
    """tear 변형 — 속살색을 젤리색 (밝은 주황)으로."""
    return synthesize_tear(img, mask, rng, inner_color_bgr=(60, 150, 235),
                           area_mode="mask_ratio", area_ratio_range=(0.03, 0.07))


# ─── 증강 (정상 객체 robustness) ──────────────────────────────────────────────
def aug_photometric(crop, cmask, kind, rng):
    c = crop.astype(np.float32)
    m3 = np.repeat(cmask[:, :, None], 3, axis=2)
    if kind == "bright+20":   c = np.where(m3, c * 1.2, c)
    elif kind == "bright-20": c = np.where(m3, c * 0.8, c)
    elif kind == "bright+40": c = np.where(m3, c * 1.4, c)
    elif kind == "bright-40": c = np.where(m3, c * 0.6, c)
    elif kind == "gamma0.7":  c = np.where(m3, ((c / 255) ** 0.7) * 255, c)
    elif kind == "gamma1.4":  c = np.where(m3, ((c / 255) ** 1.4) * 255, c)
    elif kind == "wb_warm":   c = np.where(m3, c * np.array([0.92, 1.0, 1.10]), c)
    elif kind == "wb_cool":   c = np.where(m3, c * np.array([1.10, 1.0, 0.92]), c)
    elif kind == "noise5":    c = np.where(m3, c + rng.normal(0, 5, c.shape), c)
    elif kind == "noise12":   c = np.where(m3, c + rng.normal(0, 12, c.shape), c)
    elif kind == "blur3":
        b = cv2.GaussianBlur(crop, (3, 3), 0).astype(np.float32); c = np.where(m3, b, c)
    elif kind == "blur7":
        b = cv2.GaussianBlur(crop, (7, 7), 0).astype(np.float32); c = np.where(m3, b, c)
    elif kind.startswith("jpeg"):
        q = int(kind[4:])
        _, enc = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, q])
        c = cv2.imdecode(enc, cv2.IMREAD_COLOR).astype(np.float32)
        c = np.where(m3, c, crop.astype(np.float32))
    out = np.clip(c, 0, 255).astype(np.uint8)
    out[~cmask] = CROP_BACKGROUND
    return out, cmask


def aug_geometric(crop, cmask, kind):
    if kind == "rot90":
        return np.rot90(crop, 1).copy(), np.rot90(cmask, 1).copy()
    if kind == "rot180":
        return np.rot90(crop, 2).copy(), np.rot90(cmask, 2).copy()
    if kind == "rot270":
        return np.rot90(crop, 3).copy(), np.rot90(cmask, 3).copy()
    if kind == "hflip":
        return crop[:, ::-1].copy(), cmask[:, ::-1].copy()
    if kind == "vflip":
        return crop[::-1].copy(), cmask[::-1].copy()
    if kind in ("scale0.8", "scale1.2"):
        f = 0.8 if kind == "scale0.8" else 1.2
        side = crop.shape[0]
        ns = max(8, int(side * f))
        rc = cv2.resize(crop, (ns, ns))
        rm = cv2.resize(cmask.astype(np.uint8), (ns, ns), interpolation=cv2.INTER_NEAREST) > 0
        out = np.full_like(crop, CROP_BACKGROUND)
        om = np.zeros_like(cmask)
        if f < 1:
            o = (side - ns) // 2
            out[o:o + ns, o:o + ns] = rc
            om[o:o + ns, o:o + ns] = rm
        else:
            o = (ns - side) // 2
            out = rc[o:o + side, o:o + side]
            om = rm[o:o + side, o:o + side]
        out[~om] = CROP_BACKGROUND
        return out, om
    raise ValueError(kind)


def aug_stress(crop, cmask, kind, rng):
    out = crop.copy()
    m = cmask.copy()
    if kind == "glare":
        ys, xs = np.where(m)
        if len(ys) == 0:
            return out, m
        i = rng.integers(len(ys))
        cy, cx = int(ys[i]), int(xs[i])
        side = crop.shape[0]
        rad = int(side * rng.uniform(0.10, 0.18))
        yy, xx = np.mgrid[0:side, 0:side]
        d2 = (yy - cy) ** 2 + (xx - cx) ** 2
        alpha = np.exp(-d2 / (2 * (rad * 0.6) ** 2))
        alpha[~m] = 0
        out = np.clip(out.astype(np.float32) + alpha[:, :, None] * 200, 0, 255).astype(np.uint8)
        return out, m
    if kind in ("occl20", "occl40"):
        frac = 0.2 if kind == "occl20" else 0.4
        ys, xs = np.where(m)
        if len(ys) == 0:
            return out, m
        cut_y = np.quantile(ys, frac)
        m2 = m.copy()
        m2[ys[ys <= cut_y], xs[ys <= cut_y]] = False
        out[~m2] = CROP_BACKGROUND
        return out, m2
    raise ValueError(kind)


# ─── 테스트들 ────────────────────────────────────────────────────────────────
def test_severity(sc, samples, rng):
    bins = [(0.005, 0.01), (0.01, 0.02), (0.02, 0.03), (0.03, 0.05),
            (0.05, 0.07), (0.07, 0.10), (0.10, 0.15)]
    sub = subsample(samples, 80)
    normals = []
    per_bin = {f"{a*100:.1f}-{b*100:.0f}%": [] for a, b in bins}
    for i, s in enumerate(sub):
        img, mask = load_sample(s)
        crop, cmask = crop_with_mask(img, mask)
        if crop is None:
            continue
        v = sc.score_max(crop, cmask, gidx=s["gidx"])
        if v is None:
            continue
        normals.append(v)
        for (a, b) in bins:
            for _ in range(3):
                synth = synthesize_tear(img, mask, rng, area_mode="mask_ratio",
                                        area_ratio_range=(a, b))
                scp, _ = crop_with_mask(synth, mask)
                d = sc.score_max(scp, cmask, gidx=s["gidx"]) if scp is not None else None
                if d is not None:
                    per_bin[f"{a*100:.1f}-{b*100:.0f}%"].append(d)
        if (i + 1) % 20 == 0:
            print(f"[severity] {i+1}/{len(sub)}")
    rows = []
    for k, ds in per_bin.items():
        au = auroc_of(normals, ds)
        tpr5, tau5 = tpr_at_fpr(normals, ds, 0.05)
        rows.append({"bin": k, "auroc": au, "tpr_at_fpr5": tpr5, "n": len(ds)})
        print(f"[severity] {k}: AUROC={au:.3f} TPR@FPR5%={tpr5:.2f}")
    save_json("severity", {"normal_n": len(normals), "rows": rows})
    FIGS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([r["bin"] for r in rows], [r["auroc"] for r in rows], "o-", label="AUROC")
    ax.plot([r["bin"] for r in rows], [r["tpr_at_fpr5"] for r in rows], "s--", label="TPR@FPR5%")
    ax.set_xlabel("tear size (% of mask area)"); ax.set_ylim(0, 1.05)
    ax.set_title("mango — tear severity sweep (detection limit)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.xticks(rotation=30); plt.tight_layout()
    plt.savefig(FIGS / "severity.png", dpi=110, bbox_inches="tight")


def test_defect_types(sc, samples, rng):
    types = {
        "tear_3-7%":   lambda i, m: synthesize_tear(i, m, rng, area_mode="mask_ratio", area_ratio_range=(0.03, 0.07)),
        "tear_jelly":  lambda i, m: synth_tear_jelly(i, m, rng),
        "scratch":     lambda i, m: synthesize_scratch(i, m, rng),
        "hole":        lambda i, m: synth_hole(i, m, rng),
        "stain_dark":  lambda i, m: synth_stain(i, m, rng, dark=True),
        "stain_light": lambda i, m: synth_stain(i, m, rng, dark=False),
    }
    sub = subsample(samples, 80)
    normals = []
    per_type = {k: [] for k in types}
    for i, s in enumerate(sub):
        img, mask = load_sample(s)
        crop, cmask = crop_with_mask(img, mask)
        if crop is None:
            continue
        v = sc.score_max(crop, cmask, gidx=s["gidx"])
        if v is None:
            continue
        normals.append(v)
        for tname, fn in types.items():
            for _ in range(3):
                synth = fn(img, mask)
                scp, _ = crop_with_mask(synth, mask)
                d = sc.score_max(scp, cmask, gidx=s["gidx"]) if scp is not None else None
                if d is not None:
                    per_type[tname].append(d)
        if (i + 1) % 20 == 0:
            print(f"[defect_types] {i+1}/{len(sub)}")
    rows = []
    for k, ds in per_type.items():
        au = auroc_of(normals, ds)
        tpr5, _ = tpr_at_fpr(normals, ds, 0.05)
        rows.append({"type": k, "auroc": au, "tpr_at_fpr5": tpr5, "n": len(ds)})
        print(f"[defect_types] {k}: AUROC={au:.3f} TPR@FPR5%={tpr5:.2f}")
    save_json("defect_types", {"normal_n": len(normals), "rows": rows})
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [r["type"] for r in rows]
    ax.bar(names, [r["auroc"] for r in rows], alpha=0.7, label="AUROC")
    ax.plot(names, [r["tpr_at_fpr5"] for r in rows], "ro--", label="TPR@FPR5%")
    ax.set_ylim(0, 1.05); ax.set_title("mango — defect type battery")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    plt.xticks(rotation=25); plt.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGS / "defect_types.png", dpi=110, bbox_inches="tight")


def test_photometric(sc, samples, rng):
    kinds = ["bright+20", "bright-20", "bright+40", "bright-40", "gamma0.7", "gamma1.4",
             "wb_warm", "wb_cool", "noise5", "noise12", "blur3", "blur7", "jpeg70", "jpeg40"]
    sub = subsample(samples, 80)
    base = []
    per_kind = {k: [] for k in kinds}
    for i, s in enumerate(sub):
        img, mask = load_sample(s)
        crop, cmask = crop_with_mask(img, mask)
        if crop is None:
            continue
        v = sc.score_max(crop, cmask, gidx=s["gidx"])
        if v is None:
            continue
        base.append(v)
        for k in kinds:
            ac, am = aug_photometric(crop, cmask, k, rng)
            d = sc.score_max(ac, am, gidx=s["gidx"])
            if d is not None:
                per_kind[k].append(d)
        if (i + 1) % 20 == 0:
            print(f"[photometric] {i+1}/{len(sub)}")
    base = np.array(base)
    rows = []
    for k, vs in per_kind.items():
        vs = np.array(vs)
        rows.append({"aug": k, "d_median": float(np.median(vs) - np.median(base)),
                     "fpr_at_tau_ref": float((vs > TAU_REF).mean())})
        print(f"[photometric] {k}: Δmedian={rows[-1]['d_median']:+.2f} FPR@τ={rows[-1]['fpr_at_tau_ref']:.2f}")
    save_json("photometric", {"base_median": float(np.median(base)),
                              "base_fpr_at_tau_ref": float((base > TAU_REF).mean()),
                              "tau_ref": TAU_REF, "rows": rows})
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [r["aug"] for r in rows]
    ax.bar(names, [r["fpr_at_tau_ref"] for r in rows], alpha=0.75, color="tab:red")
    ax.axhline((base > TAU_REF).mean(), color="k", linestyle="--", label="base FPR")
    ax.set_ylabel(f"FPR @ τ={TAU_REF}"); ax.set_title("mango — photometric robustness (normal FPR drift)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    plt.xticks(rotation=35); plt.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGS / "photometric.png", dpi=110, bbox_inches="tight")


def test_geometric(sc, samples, rng):
    kinds = ["rot90", "rot180", "rot270", "hflip", "vflip", "scale0.8", "scale1.2"]
    sub = subsample(samples, 80)
    base = []
    per_kind = {k: [] for k in kinds}
    for i, s in enumerate(sub):
        img, mask = load_sample(s)
        crop, cmask = crop_with_mask(img, mask)
        if crop is None:
            continue
        v = sc.score_max(crop, cmask, gidx=s["gidx"])
        if v is None:
            continue
        base.append(v)
        for k in kinds:
            ac, am = aug_geometric(crop, cmask, k)
            d = sc.score_max(ac, am, gidx=s["gidx"])
            if d is not None:
                per_kind[k].append(d)
        if (i + 1) % 20 == 0:
            print(f"[geometric] {i+1}/{len(sub)}")
    base = np.array(base)
    rows = []
    for k, vs in per_kind.items():
        vs = np.array(vs)
        rows.append({"aug": k, "d_median": float(np.median(vs) - np.median(base)),
                     "fpr_at_tau_ref": float((vs > TAU_REF).mean())})
        print(f"[geometric] {k}: Δmedian={rows[-1]['d_median']:+.2f} FPR@τ={rows[-1]['fpr_at_tau_ref']:.2f}")
    save_json("geometric", {"base_median": float(np.median(base)),
                            "base_fpr_at_tau_ref": float((base > TAU_REF).mean()),
                            "rows": rows})


def test_stress(sc, samples, rng):
    kinds = ["glare", "occl20", "occl40", "mask_dilate10", "mask_erode10"]
    sub = subsample(samples, 80)
    base = []
    per_kind = {k: [] for k in kinds}
    for i, s in enumerate(sub):
        img, mask = load_sample(s)
        crop, cmask = crop_with_mask(img, mask)
        if crop is None:
            continue
        v = sc.score_max(crop, cmask, gidx=s["gidx"])
        if v is None:
            continue
        base.append(v)
        for k in kinds:
            if k.startswith("mask_"):
                kk = 10
                m2 = (cv2.dilate if "dilate" in k else cv2.erode)(
                    mask.astype(np.uint8), np.ones((kk, kk), np.uint8)) > 0
                c2, cm2 = crop_with_mask(img, m2)
                d = sc.score_max(c2, cm2, gidx=s["gidx"]) if c2 is not None else None
            else:
                ac, am = aug_stress(crop, cmask, k, rng)
                d = sc.score_max(ac, am, gidx=s["gidx"]) if am.sum() > 50 else None
            if d is not None:
                per_kind[k].append(d)
        if (i + 1) % 20 == 0:
            print(f"[stress] {i+1}/{len(sub)}")
    base = np.array(base)
    rows = []
    for k, vs in per_kind.items():
        vs = np.array(vs)
        rows.append({"aug": k, "d_median": float(np.median(vs) - np.median(base)),
                     "fpr_at_tau_ref": float((vs > TAU_REF).mean()), "n": len(vs)})
        print(f"[stress] {k}: Δmedian={rows[-1]['d_median']:+.2f} FPR@τ={rows[-1]['fpr_at_tau_ref']:.2f}")
    save_json("stress", {"base_fpr_at_tau_ref": float((base > TAU_REF).mean()), "rows": rows})


def test_loso(sc, samples, rng):
    """leave-one-set-out — 신규 세션 FPR 리스크. bank는 v2에서 set별 patch slice 제거."""
    sets = sorted({s["set"] for s in samples})
    set_of_gidx = {s["gidx"]: s["set"] for s in samples}
    rows = []
    for held in sets:
        held_samples = [s for s in samples if s["set"] == held]
        keep_gidx = [g for g, st in set_of_gidx.items() if st != held]
        # bank index 구성 (held set patch 제외)
        idx_list = []
        for g in keep_gidx:
            s0, e0 = int(sc.cum[g]), int(sc.cum[g + 1])
            idx_list.append(torch.arange(s0, e0, device=sc.device))
        bank_idx = torch.cat(idx_list)
        # held normals + 합성 tear
        normals, defects = [], []
        for s in held_samples:
            img, mask = load_sample(s)
            crop, cmask = crop_with_mask(img, mask)
            if crop is None:
                continue
            v = sc.score_max(crop, cmask, bank_idx=bank_idx)
            if v is None:
                continue
            normals.append(v)
            for _ in range(2):
                synth = synthesize_tear(img, mask, rng, area_mode="mask_ratio",
                                        area_ratio_range=(0.03, 0.07))
                scp, _ = crop_with_mask(synth, mask)
                d = sc.score_max(scp, cmask, bank_idx=bank_idx) if scp is not None else None
                if d is not None:
                    defects.append(d)
        ns = np.array(normals)
        row = {"held_set": held, "n_normal": len(ns),
               "normal_median": float(np.median(ns)),
               "fpr_at_tau_ref": float((ns > TAU_REF).mean()),
               "auroc": auroc_of(normals, defects) if defects else None}
        rows.append(row)
        print(f"[loso] {held}: n={len(ns)} median={row['normal_median']:.2f} "
              f"FPR@τ={row['fpr_at_tau_ref']:.2f} AUROC={row['auroc']:.3f}")
    save_json("loso", {"tau_ref": TAU_REF, "rows": rows})


def test_aggregate(sc, samples, rng):
    """max / top-k / top-p sweep + clustered bootstrap CI (base tear 3~7%)."""
    sub = subsample(samples, 80)
    per_obj = []   # (normal nn_dists, [defect nn_dists ×5])
    for i, s in enumerate(sub):
        img, mask = load_sample(s)
        crop, cmask = crop_with_mask(img, mask)
        if crop is None:
            continue
        p = sc.patches(crop, cmask)
        if p is None:
            continue
        nd = sc.nn_dists(p, gidx=s["gidx"]).cpu().numpy()
        dds = []
        for _ in range(5):
            synth = synthesize_tear(img, mask, rng, area_mode="mask_ratio",
                                    area_ratio_range=(0.03, 0.07))
            scp, _ = crop_with_mask(synth, mask)
            if scp is None:
                continue
            dp = sc.patches(scp, cmask)
            if dp is None:
                continue
            dds.append(sc.nn_dists(dp, gidx=s["gidx"]).cpu().numpy())
        per_obj.append((nd, dds))
        if (i + 1) % 20 == 0:
            print(f"[aggregate] {i+1}/{len(sub)}")

    def agg(nn, mode, param):
        nn_sorted = np.sort(nn)[::-1]
        if mode == "max":
            return float(nn_sorted[0])
        if mode == "topk":
            return float(nn_sorted[:min(param, len(nn_sorted))].mean())
        if mode == "topp":
            k = max(1, int(len(nn_sorted) * param))
            return float(nn_sorted[:k].mean())
        if mode == "mean":
            return float(nn_sorted.mean())

    configs = [("max", None)] + [("topk", k) for k in (2, 3, 4, 6, 8)] + \
              [("topp", p) for p in (0.01, 0.02, 0.05)] + [("mean", None)]
    rows = []
    for mode, param in configs:
        normals = [agg(nd, mode, param) for nd, _ in per_obj]
        defects = [agg(d, mode, param) for _, dds in per_obj for d in dds]
        au = auroc_of(normals, defects)
        tpr5, _ = tpr_at_fpr(normals, defects, 0.05)
        label = mode if param is None else f"{mode}{param}"
        rows.append({"agg": label, "auroc": au, "tpr_at_fpr5": tpr5})
        print(f"[aggregate] {label}: AUROC={au:.3f} TPR@FPR5%={tpr5:.2f}")

    # clustered bootstrap (객체 단위 재표집) — max 기준 CI
    rng2 = np.random.default_rng(0)
    boots = []
    n = len(per_obj)
    for _ in range(1000):
        pick = rng2.integers(0, n, n)
        normals = [agg(per_obj[j][0], "max", None) for j in pick]
        defects = [agg(d, "max", None) for j in pick for d in per_obj[j][1]]
        if len(set([0, 1]) - set()) and defects:
            boots.append(auroc_of(normals, defects))
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    print(f"[aggregate] max AUROC bootstrap 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
    save_json("aggregate", {"rows": rows, "max_auroc_ci95": ci, "n_boot": len(boots)})


def test_fulldata(sc, samples, rng):
    """전수 309 정상 LOO — 정밀 통계 (τ 캘리브) + set별 분해."""
    scores, sets = [], []
    for i, s in enumerate(samples):
        img, mask = load_sample(s)
        crop, cmask = crop_with_mask(img, mask)
        if crop is None:
            continue
        v = sc.score_max(crop, cmask, gidx=s["gidx"])
        if v is None:
            continue
        scores.append(v)
        sets.append(s["set"])
        if (i + 1) % 50 == 0:
            print(f"[fulldata] {i+1}/{len(samples)}")
    a = np.array(scores)
    out = {
        "n": len(a), "mu": float(a.mean()), "sigma": float(a.std()),
        "cv": float(a.std() / a.mean()),
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)), "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)), "p995": float(np.percentile(a, 99.5)),
        "max": float(a.max()),
        "top5": sorted(a.tolist(), reverse=True)[:5],
        "per_set": {st: {"n": int((np.array(sets) == st).sum()),
                         "median": float(np.median(a[np.array(sets) == st]))}
                    for st in sorted(set(sets))},
    }
    print(f"[fulldata] n={out['n']} mu={out['mu']:.2f} cv={out['cv']:.3f} "
          f"p95={out['p95']:.2f} p99={out['p99']:.2f} max={out['max']:.2f}")
    save_json("fulldata", out)


def test_latency(sc, samples, rng):
    sub = subsample(samples, 20)
    pairs = []
    for s in sub:
        img, mask = load_sample(s)
        crop, cmask = crop_with_mask(img, mask)
        if crop is not None:
            pairs.append((crop, cmask, s["gidx"]))
    # warmup
    for crop, cmask, g in pairs[:3]:
        sc.score_max(crop, cmask, gidx=g)
    torch.cuda.synchronize()
    times = []
    for rep in range(5):
        for crop, cmask, g in pairs:
            t0 = time.time()
            sc.score_max(crop, cmask, gidx=g)
            torch.cuda.synchronize()
            times.append((time.time() - t0) * 1000)
    t = np.array(times)
    out = {"n": len(t), "mean_ms": float(t.mean()), "p50_ms": float(np.median(t)),
           "p95_ms": float(np.percentile(t, 95)),
           "note": "A5000 기준. RTX 4080은 대략 동급~1.2x 추정 (架 측정 필요)"}
    print(f"[latency] mean={out['mean_ms']:.1f}ms p95={out['p95_ms']:.1f}ms")
    save_json("latency", out)


TESTS = {
    "severity": test_severity, "defect_types": test_defect_types,
    "photometric": test_photometric, "geometric": test_geometric,
    "stress": test_stress, "loso": test_loso,
    "aggregate": test_aggregate, "fulldata": test_fulldata,
    "latency": test_latency,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, choices=list(TESTS.keys()))
    args = ap.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[{args.test}] device={device}")
    sc = Scorer(device)
    samples = collect(ROOT / "data" / "labeled", load_excluded())
    assert len(samples) == sc.n_obj, f"collect {len(samples)} != bank {sc.n_obj}"
    rng = np.random.default_rng(42)
    TESTS[args.test](sc, samples, rng)


if __name__ == "__main__":
    main()
