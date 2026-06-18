"""LOO 방식 ROC tune — proper validation.

각 정상 객체에 대해:
  - LOO 정상 score: 자기 patches를 memory bank에서 제외하고 자기 patches NN distance max
  - 객체당 N개 합성 변형 → 각 합성 LOO score (자기 정상 patches 제외)

LOO 정상 vs 합성 결함 score → ROC + AUROC + F1 최적 threshold.

이건 self-match 함정 회피한 proper validation.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/synth_roc_tune_loo.py
"""
from __future__ import annotations

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
DINOV2_LAYER = 8   # fallback only. 실제 layer는 각 뱅크의 dinov2_layer 메타에서 읽음 (tune_one_class)
INPUT_SIZE = 224
GRID_SIZE = INPUT_SIZE // 14

# polygon class_id → (이름, 합성 종류, memory_bank 파일명, 표시용 기존 threshold)
# 주의: mango는 fusion 기반이라 이 plain-patchcore LOO τ는 운영값이 아님(참고용). 뱅크 없으면 자동 skip.
# layer는 config 변경(haribo/pencil L5, metal L7)에 맞춰 재빌드된 뱅크의 메타에서 자동 인식됨.
TARGETS = [
    (1, "haribo", "tear", "patchcore_haribo.npz", 11.0),
    (2, "mango", "tear", "patchcore_mango_v2.npz", 18.0),
    (3, "metal_case", "scratch", "patchcore_metal_case.npz", 10.0),
    (5, "pencil_case", "scratch", "patchcore_pencil_case.npz", 16.0),
]

N_SYNTH_VARIANTS = 5  # 객체당 합성 결함 변형 수


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


def extract_patches(crop_bgr, crop_mask, processor, model, device, layer=DINOV2_LAYER):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    inputs = processor(images=pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    patch_tokens = outputs.hidden_states[layer + 1][0, 1:, :]
    grid_mask = cv2.resize(crop_mask.astype(np.uint8), (GRID_SIZE, GRID_SIZE), interpolation=cv2.INTER_AREA) > 0
    sel = np.where(grid_mask.flatten())[0]
    if len(sel) == 0:
        return None
    return patch_tokens[torch.from_numpy(sel).to(device)]


def loo_score(patches: torch.Tensor, mb: torch.Tensor, mb_sq: torch.Tensor,
              exclude_start: int, exclude_end: int) -> float:
    """객체 i의 patches가 memory bank의 [exclude_start, exclude_end) 위치에 있음.
    그 범위를 제외하고 nearest L2 distance 계산."""
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


def parse_polygon(line, W, H):
    parts = line.split()
    cid = int(parts[0])
    coords = list(map(float, parts[1:]))
    poly = np.array([[coords[i] * W, coords[i + 1] * H] for i in range(0, len(coords), 2)], dtype=np.int32)
    return cid, poly


def collect_samples_with_idx(target_class: int, labeled_base: Path):
    """라벨링 데이터에서 target_class polygon에 해당하는 (img, mask, global_obj_idx) 수집.
    global_obj_idx = build 시 memory bank에 들어간 순서 (build_patchcore_refs.py와 동일 순서)."""
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
                cid, poly = parse_polygon(line, W, H)
                if cid != target_class:
                    continue
                mask = np.zeros((H, W), dtype=np.uint8)
                cv2.fillPoly(mask, [poly], 255)
                samples.append((img, mask > 0, global_idx))
                global_idx += 1
    return samples


def tune_one_class(target, processor, model, device, output_dir, max_obj=80):
    poly_id, name, synth_type, mb_file, prev_th = target
    print(f"\n=== {name} (polygon {poly_id}, synth={synth_type}, prev τ={prev_th}) ===")

    mb_path = ROOT / "weights" / mb_file
    if not mb_path.is_file():
        print(f"  skip — memory bank 없음: {mb_path.name} (재빌드 필요 or 이 클래스 제외)")
        return None
    mb_data = np.load(mb_path)
    mb = torch.from_numpy(mb_data["memory_bank"].astype(np.float32)).to(device)
    mb_sq = (mb * mb).sum(dim=1)
    patches_per_object = mb_data["patches_per_object"]
    cum = np.concatenate([[0], np.cumsum(patches_per_object)])
    # 테스트 patch는 반드시 뱅크와 같은 layer에서 뽑아야 거리가 유효 (config layer 변경 대응).
    # 뱅크에 저장된 dinov2_layer를 source of truth로 사용.
    bank_layer = int(mb_data["dinov2_layer"]) if "dinov2_layer" in mb_data else DINOV2_LAYER
    print(f"  memory bank: {mb.shape}, n_objects in bank: {len(patches_per_object)}, layer=L{bank_layer}")

    samples = collect_samples_with_idx(poly_id, ROOT / "data" / "labeled")
    print(f"  total label objects: {len(samples)}")
    if len(samples) < 10:
        print(f"  skip (not enough)")
        return None

    # subsample to max_obj
    rng = np.random.default_rng(42)
    if len(samples) > max_obj:
        idx_sub = rng.choice(len(samples), max_obj, replace=False)
        samples = [samples[i] for i in idx_sub]
    print(f"  using {len(samples)} objects for ROC, {N_SYNTH_VARIANTS} synth variants each")

    normal_scores = []
    defect_scores = []
    for i, (img, mask, global_idx) in enumerate(samples):
        # LOO: exclude this object's patches from memory bank
        ex_start, ex_end = int(cum[global_idx]), int(cum[global_idx + 1])

        # normal LOO score
        crop, cmask = crop_with_mask(img, mask)
        if crop is None:
            continue
        n_patches = extract_patches(crop, cmask, processor, model, device, bank_layer)
        if n_patches is None:
            continue
        normal_scores.append(loo_score(n_patches, mb, mb_sq, ex_start, ex_end))

        # synthetic defect — N variants
        for v in range(N_SYNTH_VARIANTS):
            if synth_type == "scratch":
                synth = synthesize_scratch(img, mask, rng)
            else:
                synth = synthesize_tear(img, mask, rng)
            synth_crop, _ = crop_with_mask(synth, mask)
            if synth_crop is None:
                continue
            d_patches = extract_patches(synth_crop, cmask, processor, model, device, bank_layer)
            if d_patches is None:
                continue
            defect_scores.append(loo_score(d_patches, mb, mb_sq, ex_start, ex_end))

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(samples)}] normal_mean={np.mean(normal_scores):.2f} defect_mean={np.mean(defect_scores):.2f} n_defect={len(defect_scores)}")

    normal_scores = np.array(normal_scores)
    defect_scores = np.array(defect_scores)

    if len(normal_scores) == 0 or len(defect_scores) == 0:
        print("  no scores")
        return None

    y_true = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(defect_scores))])
    y_score = np.concatenate([normal_scores, defect_scores])
    auroc = roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    f1 = 2 * tpr * (1 - fpr) / np.maximum(tpr + (1 - fpr), 1e-9)
    best_idx = int(np.argmax(f1))
    best_th = float(thresholds[best_idx])
    best_f1 = float(f1[best_idx])
    best_tpr = float(tpr[best_idx])
    best_fpr = float(fpr[best_idx])

    normal_p99 = float(np.percentile(normal_scores, 99))
    normal_p995 = float(np.percentile(normal_scores, 99.5))
    defect_p01 = float(np.percentile(defect_scores, 1))
    overlap = "OVERLAP" if defect_p01 < normal_p99 else "SEPARATED"

    # 운영 τ — mango만 보수적(higher τ): 합성 TPR 검증 불가 → 신뢰 가능한 정상분포 분위수로
    # 오거부(FPR)를 직접 통제. max()는 분위수가 비정상적으로 낮을 때의 하한 보호. 나머지는 F1-best 유지.
    if name == "mango":
        operating_th = float(max(best_th, normal_p995))
        th_source = "max(F1-best,p99.5)"
    else:
        operating_th = best_th
        th_source = "F1-best"

    print(f"\n  AUROC: {auroc:.3f}  [{overlap}]")
    print(f"  normal LOO: n={len(normal_scores)}, mean={normal_scores.mean():.2f}, std={normal_scores.std():.2f}, p95={np.percentile(normal_scores, 95):.2f}, p99={normal_p99:.2f}")
    print(f"  defect:     n={len(defect_scores)}, mean={defect_scores.mean():.2f}, std={defect_scores.std():.2f}, p01={defect_p01:.2f}, p05={np.percentile(defect_scores, 5):.2f}")
    print(f"  F1-best τ={best_th:.2f}, F1={best_f1:.3f}, TPR={best_tpr:.3f}, FPR={best_fpr:.3f}")
    print(f"  → operating τ={operating_th:.2f} ({th_source}) [p99={normal_p99:.2f}, p99.5={normal_p995:.2f}]")

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bins = np.linspace(min(normal_scores.min(), defect_scores.min()),
                       max(normal_scores.max(), defect_scores.max()), 40)
    axes[0].hist(normal_scores, bins=bins, alpha=0.6, color="blue", label=f"normal LOO (n={len(normal_scores)})", edgecolor="black")
    axes[0].hist(defect_scores, bins=bins, alpha=0.6, color="red", label=f"defect LOO (n={len(defect_scores)})", edgecolor="black")
    axes[0].axvline(best_th, color="green", linestyle="--", label=f"F1-best τ={best_th:.2f}")
    axes[0].axvline(prev_th, color="orange", linestyle=":", label=f"yaml τ={prev_th}")
    axes[0].set_xlabel("PatchCore LOO score")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"{name} score distribution (LOO)")
    axes[0].legend()

    axes[1].plot(fpr, tpr, "b-", linewidth=2, label=f"AUROC={auroc:.3f}")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[1].scatter(best_fpr, best_tpr, color="green", s=100, zorder=5, label=f"F1-best (FPR={best_fpr:.2f}, TPR={best_tpr:.2f})")
    axes[1].set_xlabel("FPR")
    axes[1].set_ylabel("TPR")
    axes[1].set_title(f"{name} ROC ({overlap})")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle(f"{name} - {synth_type} (LOO validation, {N_SYNTH_VARIANTS} variants/obj)")
    plt.tight_layout()
    out_png = output_dir / f"roc_loo_{name}.png"
    plt.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  → {out_png.relative_to(ROOT)}")

    return {
        "name": name, "polygon_id": poly_id, "auroc": auroc,
        "mb_file": mb_file,
        "best_threshold": best_th, "best_f1": best_f1,
        "normal_p995": normal_p995, "operating_threshold": operating_th, "th_source": th_source,
        "best_tpr": best_tpr, "best_fpr": best_fpr,
        "normal_p99": normal_p99, "defect_p01": defect_p01,
        "overlap": overlap, "prev_threshold": prev_th,
    }


def main():
    from transformers import AutoImageProcessor, AutoModel

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    MODEL = "facebook/dinov2-large"   # 2026-06-19 large 전환 (config와 일치)
    print(f"[device] {device}")
    print(f"[load] {MODEL}")
    processor = AutoImageProcessor.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(device).eval()

    output_dir = ROOT / "outputs" / "figs" / "synth_roc"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for target in TARGETS:
        r = tune_one_class(target, processor, model, device, output_dir, max_obj=80)
        if r is not None:
            results.append(r)

    print("\n\n=== 최종 권장 threshold 표 (LOO validation) ===")
    print(f"  {'class':<12} {'AUROC':>7} {'sep?':>10} {'prev τ':>7} {'F1-best τ':>10} {'F1':>6} {'TPR':>6} {'FPR':>6} {'norm p99':>9} {'def p01':>8}")
    for r in results:
        print(f"  {r['name']:<12} {r['auroc']:>7.3f} {r['overlap']:>10} {r['prev_threshold']:>7.1f} {r['best_threshold']:>10.2f} {r['best_f1']:>6.3f} {r['best_tpr']:>6.3f} {r['best_fpr']:>6.3f} {r['normal_p99']:>9.2f} {r['defect_p01']:>8.2f}")

    # ── 임계 오버라이드 파일 생성 (DefectRegistry가 자동 적용) ──
    # 키 = memory_bank basename. 이 파일을 weights/에 두면 config 수동 편집 없이 τ 적용됨.
    # defect_thresholds.json은 operating_threshold 채택 (mango=보수적 분위수, 나머지=F1-best)
    out = {r["mb_file"]: round(float(r["operating_threshold"]), 4) for r in results}
    th_path = ROOT / "weights" / "defect_thresholds.json"
    th_path.parent.mkdir(parents=True, exist_ok=True)
    th_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n→ threshold override 저장: {th_path.relative_to(ROOT)}")
    print(f"   {out}")
    print("   (이 파일 + 재빌드된 뱅크를 weights/에 두면 registry가 자동 적용 → 작동)")

    # 완전성 검증 — 일부 클래스가 스킵(뱅크 없음/샘플 부족)되면 thresholds.json이 불완전 → 시끄럽게 실패
    expected = {t[1] for t in TARGETS}
    done = {r["name"] for r in results}
    missing = sorted(expected - done)
    if missing:
        print(f"\n[ERROR] 튜닝 실패/스킵 클래스: {missing} (뱅크 없음/샘플 부족). "
              f"thresholds.json 불완전 — 해당 뱅크 재빌드 후 재실행 필요.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
