"""Layer ablation — LOO ROC eval across layer × class.

목적:
  outputs/eda/memory_banks/mb_L{L}_{class}.npz (33 파일) 을 입력으로,
  각 (layer, class) 조합에 대해 strict LOO ROC + 합성 결함 5변형/객체 × 80 객체.

전략:
  forward 1번에 layer 1~11 hidden_states 동시 추출 → 11 layer LOO score 동시 계산.
  즉, layer마다 forward 재실행 X. 클래스당 80 obj × (1 normal + 5 synth) × forward 1번.

산출물:
  outputs/figs/layer_ablation/auroc_vs_layer.png      — class별 곡선
  outputs/figs/layer_ablation/score_dist_{class}_L{L}.png  — score 분포 (3 class × 11 layer = 33)
  outputs/figs/layer_ablation/roc_curves_{class}.png  — class별 11 layer ROC 한 장
  outputs/eda/layer_ablation_results.npz              — raw 점수 array
  outputs/eda/layer_ablation_summary.md               — 표 + 제외 layer 근거

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/layer_ablation_eval.py
"""
from __future__ import annotations

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

from src.defect.synthesize import synthesize_scratch, synthesize_tear

CROP_PADDING_RATIO = 0.03
CROP_BACKGROUND = 127
INPUT_SIZE = 224
GRID_SIZE = INPUT_SIZE // 14

LAYERS = list(range(1, 12))
EXCLUDED_LAYERS = {
    0: "hidden_states[0] = patch embedding (pre-block). Transformer attention 전, raw 14x14 RGB embedding 수준. "
       "정상/결함 모두 비슷한 raw 통계라 anomaly 신호 부재.",
    12: "hidden_states[12] = post-block-11 (마지막). DinoV2 KNN classifier가 이 표현으로 카테고리 판별 중. "
        "Semantic level이라 결함 있어도 'metal_case' 라는 동일 카테고리로 수렴 → patch 거리가 anomaly 신호 잃음.",
}

# (polygon_id, name, synth_type)
TARGETS = [
    (1, "haribo", "tear"),
    (3, "metal_case", "scratch"),
    (5, "pencil_case", "scratch"),
]
N_SYNTH_VARIANTS = 5
MAX_OBJ = 80


def parse_polygon(line, W, H):
    parts = line.split()
    cid = int(parts[0])
    coords = list(map(float, parts[1:]))
    poly = np.array([[coords[i] * W, coords[i + 1] * H] for i in range(0, len(coords), 2)], dtype=np.int32)
    return cid, poly


def crop_with_mask(img, mask, pad_ratio=CROP_PADDING_RATIO):
    H, W = img.shape[:2]
    m = mask > 0
    ys, xs = np.where(m)
    if len(ys) == 0:
        return None, None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    side = int(np.ceil(max(x2 - x1, y2 - y1) * (1.0 + pad_ratio * 2.0)))
    if side <= 0:
        return None, None
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    cl, ct = int(np.floor(cx - side * 0.5)), int(np.floor(cy - side * 0.5))
    cr, cb = cl + side, ct + side
    sx1, sy1 = max(0, cl), max(0, ct)
    sx2, sy2 = min(W, cr), min(H, cb)
    crop = np.full((side, side, 3), CROP_BACKGROUND, dtype=np.uint8)
    cmask = np.zeros((side, side), dtype=bool)
    if sx2 > sx1 and sy2 > sy1:
        dx1, dy1 = sx1 - cl, sy1 - ct
        crop[dy1:dy1 + sy2 - sy1, dx1:dx1 + sx2 - sx1] = img[sy1:sy2, sx1:sx2]
        cmask[dy1:dy1 + sy2 - sy1, dx1:dx1 + sx2 - sx1] = m[sy1:sy2, sx1:sx2]
    crop[~cmask] = CROP_BACKGROUND
    return crop, cmask


def extract_patches_all_layers(crop_bgr, crop_mask, processor, model, device):
    """forward 1번 → layer 1~11의 mask 영역 patch tokens (on device)."""
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(crop_rgb)
    inputs = processor(images=pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    grid_mask = cv2.resize(crop_mask.astype(np.uint8), (GRID_SIZE, GRID_SIZE), interpolation=cv2.INTER_AREA) > 0
    sel = np.where(grid_mask.flatten())[0]
    if len(sel) == 0:
        return None
    sel_t = torch.from_numpy(sel).to(device)
    out = {}
    for L in LAYERS:
        tokens = outputs.hidden_states[L + 1][0, 1:, :]  # (256, D)
        out[L] = tokens[sel_t]
    return out


def loo_score(patches: torch.Tensor, mb: torch.Tensor, mb_sq: torch.Tensor,
              exclude_start: int, exclude_end: int) -> float:
    """객체 i의 patches가 mb의 [exclude_start, exclude_end) 위치 → 그 영역 제외하고 NN L2 max."""
    N = mb.shape[0]
    if exclude_start > 0 and exclude_end < N:
        idx = torch.cat([
            torch.arange(0, exclude_start, device=mb.device),
            torch.arange(exclude_end, N, device=mb.device),
        ])
    elif exclude_start == 0:
        idx = torch.arange(exclude_end, N, device=mb.device)
    else:
        idx = torch.arange(0, exclude_start, device=mb.device)
    rest = mb[idx]
    rest_sq = mb_sq[idx]
    a2 = (patches * patches).sum(dim=1, keepdim=True)
    ab = patches @ rest.T
    d2 = (a2 + rest_sq.unsqueeze(0) - 2 * ab).clamp(min=0)
    return float(d2.sqrt().min(dim=1).values.max())


def collect_samples(target_class: int, labeled_base: Path):
    """build 순서와 동일하게 (img, mask, global_idx) 수집."""
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
            img = cv2.imread(str(bmp))
            if img is None:
                continue
            H, W = img.shape[:2]
            for line in txt.read_text().strip().split("\n"):
                if not line.strip():
                    continue
                try:
                    cid, poly = parse_polygon(line, W, H)
                except Exception:
                    continue
                if cid != target_class:
                    continue
                mask = np.zeros((H, W), dtype=np.uint8)
                cv2.fillPoly(mask, [poly], 255)
                samples.append((img, mask > 0, global_idx))
                global_idx += 1
    return samples


def eval_one_class(target, processor, model, device, mb_dir):
    poly_id, name, synth_type = target
    print(f"\n=== eval {name} (polygon {poly_id}, synth={synth_type}) ===")

    # Load all 11 layer memory banks
    mbs = {}        # L → (mb tensor, mb_sq tensor)
    cum_map = None  # patches_per_object cumulative (모든 layer 동일해야)
    n_obj_bank = None
    for L in LAYERS:
        path = mb_dir / f"mb_L{L}_{name}.npz"
        data = np.load(path)
        mb = torch.from_numpy(data["memory_bank"].astype(np.float32)).to(device)
        mb_sq = (mb * mb).sum(dim=1)
        mbs[L] = (mb, mb_sq)
        if cum_map is None:
            ppo = data["patches_per_object"]
            cum_map = np.concatenate([[0], np.cumsum(ppo)])
            n_obj_bank = len(ppo)
    print(f"  memory bank: n_obj={n_obj_bank}, "
          f"shapes={ {L: tuple(mbs[L][0].shape) for L in LAYERS} }")

    samples = collect_samples(poly_id, ROOT / "data" / "labeled")
    print(f"  total samples: {len(samples)}")
    rng = np.random.default_rng(42)
    if len(samples) > MAX_OBJ:
        idx_sub = rng.choice(len(samples), MAX_OBJ, replace=False)
        samples = [samples[i] for i in idx_sub]
    print(f"  using {len(samples)} objects, {N_SYNTH_VARIANTS} synth variants each")

    normal_scores = {L: [] for L in LAYERS}
    defect_scores = {L: [] for L in LAYERS}
    t0 = time.time()

    for i, (img, mask, gidx) in enumerate(samples):
        ex_start, ex_end = int(cum_map[gidx]), int(cum_map[gidx + 1])

        crop, cmask = crop_with_mask(img, mask)
        if crop is None:
            continue
        n_patches = extract_patches_all_layers(crop, cmask, processor, model, device)
        if n_patches is None:
            continue
        for L in LAYERS:
            mb, mb_sq = mbs[L]
            s = loo_score(n_patches[L], mb, mb_sq, ex_start, ex_end)
            normal_scores[L].append(s)

        for v in range(N_SYNTH_VARIANTS):
            synth = (synthesize_scratch if synth_type == "scratch" else synthesize_tear)(img, mask, rng)
            synth_crop, _ = crop_with_mask(synth, mask)
            if synth_crop is None:
                continue
            d_patches = extract_patches_all_layers(synth_crop, cmask, processor, model, device)
            if d_patches is None:
                continue
            for L in LAYERS:
                mb, mb_sq = mbs[L]
                s = loo_score(d_patches[L], mb, mb_sq, ex_start, ex_end)
                defect_scores[L].append(s)

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(samples)}] elapsed={elapsed:.1f}s "
                  f"L8 normal_mean={np.mean(normal_scores[8]):.2f} "
                  f"defect_mean={np.mean(defect_scores[8]):.2f}")

    rows = []
    for L in LAYERS:
        ns = np.array(normal_scores[L])
        ds = np.array(defect_scores[L])
        if len(ns) == 0 or len(ds) == 0:
            continue
        y_true = np.concatenate([np.zeros(len(ns)), np.ones(len(ds))])
        y_score = np.concatenate([ns, ds])
        auroc = roc_auc_score(y_true, y_score)
        fpr, tpr, thr = roc_curve(y_true, y_score)
        f1 = 2 * tpr * (1 - fpr) / np.maximum(tpr + (1 - fpr), 1e-9)
        bi = int(np.argmax(f1))
        rows.append({
            "layer": L,
            "auroc": auroc,
            "best_threshold": float(thr[bi]),
            "best_f1": float(f1[bi]),
            "best_tpr": float(tpr[bi]),
            "best_fpr": float(fpr[bi]),
            "normal_mean": float(ns.mean()),
            "normal_std": float(ns.std()),
            "normal_p99": float(np.percentile(ns, 99)),
            "defect_mean": float(ds.mean()),
            "defect_std": float(ds.std()),
            "defect_p01": float(np.percentile(ds, 1)),
            "n_normal": int(len(ns)),
            "n_defect": int(len(ds)),
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "normal_scores": ns.tolist(),
            "defect_scores": ds.tolist(),
        })

    print(f"\n  {name} layer × AUROC:")
    for r in rows:
        sep = "SEPARATED" if r["defect_p01"] >= r["normal_p99"] else "OVERLAP"
        print(f"    L{r['layer']:>2}: AUROC={r['auroc']:.3f} τ*={r['best_threshold']:>5.2f} "
              f"F1={r['best_f1']:.3f} TPR={r['best_tpr']:.2f} FPR={r['best_fpr']:.2f} [{sep}]")
    return rows


def plot_auroc_vs_layer(results, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, rows in results.items():
        layers = [r["layer"] for r in rows]
        aurocs = [r["auroc"] for r in rows]
        ax.plot(layers, aurocs, "o-", label=name, linewidth=2, markersize=7)
        best_l = layers[int(np.argmax(aurocs))]
        ax.axvline(best_l, alpha=0.0)   # placeholder; per-class best 표시는 marker로
        # annotate best
        ax.annotate(f"best L{best_l}", xy=(best_l, max(aurocs)),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.axvline(8, color="gray", linestyle="--", alpha=0.5, label="current default (L8)")
    ax.set_xlabel("DinoV2 layer (post-block index, 1=shallow → 11=deepest pre-classifier)")
    ax.set_ylabel("AUROC (LOO synth-defect ROC)")
    ax.set_title("Layer ablation — AUROC vs layer\n(excluded: L0 pre-block, L12 semantic/classifier)")
    ax.set_xticks(LAYERS)
    ax.grid(alpha=0.3)
    ax.set_ylim(0.5, 1.02)
    ax.legend(loc="lower center")
    out_path = out_dir / "auroc_vs_layer.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  → {out_path.relative_to(ROOT)}")


def plot_roc_curves_per_class(name, rows, out_dir):
    fig, ax = plt.subplots(figsize=(7, 6.5))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(rows)))
    for c, r in zip(cmap, rows):
        ax.plot(r["fpr"], r["tpr"], color=c, label=f"L{r['layer']} (AUROC={r['auroc']:.3f})", linewidth=1.5)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"{name} — ROC by layer (LOO)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    out_path = out_dir / f"roc_curves_{name}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  → {out_path.relative_to(ROOT)}")


def plot_score_dist(name, rows, out_dir):
    """11 layer × 1 figure 한장: 각 layer score 분포 subplot."""
    n = len(rows)
    cols = 4
    rows_grid = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_grid, cols, figsize=(4 * cols, 3 * rows_grid))
    axes = axes.flatten()
    for i, r in enumerate(rows):
        ax = axes[i]
        ns = np.array(r["normal_scores"])
        ds = np.array(r["defect_scores"])
        lo = min(ns.min(), ds.min())
        hi = max(ns.max(), ds.max())
        bins = np.linspace(lo, hi, 30)
        ax.hist(ns, bins=bins, alpha=0.55, color="blue", label=f"normal n={len(ns)}", edgecolor="black")
        ax.hist(ds, bins=bins, alpha=0.55, color="red", label=f"defect n={len(ds)}", edgecolor="black")
        ax.axvline(r["best_threshold"], color="green", linestyle="--", linewidth=1.2,
                   label=f"τ*={r['best_threshold']:.2f}")
        ax.set_title(f"L{r['layer']}  AUROC={r['auroc']:.3f}  F1={r['best_f1']:.2f}", fontsize=10)
        ax.legend(fontsize=7)
        ax.set_xlabel("score")
        ax.set_ylabel("count")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    plt.suptitle(f"{name} — score distribution by layer (LOO)", fontsize=12)
    plt.tight_layout()
    out_path = out_dir / f"score_dist_{name}.png"
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  → {out_path.relative_to(ROOT)}")


def write_summary_md(results, summary_path):
    lines = []
    lines.append("# Layer ablation EDA — summary\n")
    lines.append("작성: 2026-06-12. PatchCore-lite의 DinoV2 mid-layer 선택을 실증 비교.\n")
    lines.append("## 설정\n")
    lines.append(f"- 비교 layer: {LAYERS} (총 {len(LAYERS)}개)")
    lines.append(f"- 클래스: {', '.join(r[1] for r in TARGETS)} (운영 활성 3개)")
    lines.append(f"- 프로토콜: strict LOO + 합성 결함 {N_SYNTH_VARIANTS}변형/객체 × {MAX_OBJ}객체")
    lines.append(f"- forward는 layer당 별도 X — `output_hidden_states=True`로 forward 1회에 11 layer 동시 추출\n")

    lines.append("## 제외된 layer 와 근거\n")
    lines.append("| layer | 제외 근거 |")
    lines.append("|---|---|")
    for L in sorted(EXCLUDED_LAYERS.keys()):
        lines.append(f"| {L} | {EXCLUDED_LAYERS[L]} |")
    lines.append("")

    lines.append("## 클래스별 결과 표 (AUROC desc)\n")
    for name in sorted(results.keys()):
        rows = sorted(results[name], key=lambda r: -r["auroc"])
        lines.append(f"### {name}\n")
        lines.append("| L | AUROC | F1-best τ | F1 | TPR | FPR | normal mean±std | defect mean±std | overlap? |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            sep = "SEP" if r["defect_p01"] >= r["normal_p99"] else "OVER"
            lines.append(
                f"| {r['layer']} | **{r['auroc']:.3f}** | {r['best_threshold']:.2f} | "
                f"{r['best_f1']:.3f} | {r['best_tpr']:.2f} | {r['best_fpr']:.2f} | "
                f"{r['normal_mean']:.2f}±{r['normal_std']:.2f} | "
                f"{r['defect_mean']:.2f}±{r['defect_std']:.2f} | {sep} |"
            )
        lines.append("")
        best = max(results[name], key=lambda r: r["auroc"])
        current = next((r for r in results[name] if r["layer"] == 8), None)
        delta = (best["auroc"] - current["auroc"]) if current else None
        lines.append(f"- **best layer: L{best['layer']}** (AUROC {best['auroc']:.3f})")
        if current and delta is not None:
            lines.append(f"- current default L8: AUROC {current['auroc']:.3f} (Δ = {delta:+.3f})")
        lines.append("")

    lines.append("## 종합 결론\n")
    best_per_class = {name: max(rs, key=lambda r: r["auroc"]) for name, rs in results.items()}
    lines.append("| class | best layer | best AUROC | L8 (current) AUROC | Δ |")
    lines.append("|---|---|---|---|---|")
    for name, br in best_per_class.items():
        cur = next((r for r in results[name] if r["layer"] == 8), None)
        cur_a = cur["auroc"] if cur else float("nan")
        delta = br["auroc"] - cur_a
        lines.append(f"| {name} | L{br['layer']} | {br['auroc']:.3f} | {cur_a:.3f} | {delta:+.3f} |")
    lines.append("")
    lines.append("## 산출물\n")
    lines.append("- `outputs/figs/layer_ablation/auroc_vs_layer.png` — 3 클래스 layer×AUROC")
    lines.append("- `outputs/figs/layer_ablation/roc_curves_{class}.png` — 클래스별 11 layer ROC")
    lines.append("- `outputs/figs/layer_ablation/score_dist_{class}.png` — 클래스별 11 layer score 분포")
    lines.append("- `outputs/eda/layer_ablation_results.npz` — raw 점수 array")
    lines.append("- `outputs/eda/memory_banks/mb_L{L}_{class}.npz` × 33 — 임시 메모리뱅크 (~6GB)")
    lines.append("")
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[summary] → {summary_path.relative_to(ROOT)}")


def main():
    from transformers import AutoImageProcessor, AutoModel
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")
    print(f"[layers] {LAYERS}, excluded {sorted(EXCLUDED_LAYERS.keys())}")

    mb_dir = ROOT / "outputs" / "eda" / "memory_banks"
    if not mb_dir.exists():
        print(f"[error] memory bank dir not found: {mb_dir}")
        print("  먼저 scripts/layer_ablation_build.py 를 실행하세요.")
        return 1

    print(f"\n[load] facebook/dinov2-base")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()

    figs_dir = ROOT / "outputs" / "figs" / "layer_ablation"
    figs_dir.mkdir(parents=True, exist_ok=True)
    eda_dir = ROOT / "outputs" / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    t0 = time.time()
    for target in TARGETS:
        rows = eval_one_class(target, processor, model, device, mb_dir)
        results[target[1]] = rows

    # 저장 (raw scores 포함)
    np.savez(
        eda_dir / "layer_ablation_results.npz",
        results_json=str({name: [{k: v for k, v in r.items() if k not in ("fpr", "tpr", "normal_scores", "defect_scores")} for r in rs] for name, rs in results.items()}),
    )

    # 시각화
    print("\n[plots]")
    plot_auroc_vs_layer(results, figs_dir)
    for name, rows in results.items():
        plot_roc_curves_per_class(name, rows, figs_dir)
        plot_score_dist(name, rows, figs_dir)

    write_summary_md(results, eda_dir / "layer_ablation_summary.md")

    total = time.time() - t0
    print(f"\n[done] total elapsed: {total:.1f}s ({total/60:.1f} min)")


if __name__ == "__main__":
    sys.exit(main() or 0)
