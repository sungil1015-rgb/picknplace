"""mango 정상 분산 (CV 0.851) 원인 확인 — 객체 embedding 클러스터링 + 군집별 crop 몽타주.

질문: mango reference 안에 외형이 갈리는 축이 실제로 있나? 있다면 무엇인가?
방법: 객체별 DinoV2 h[9] masked patch-mean embedding → k-means (k=2..4, silhouette 선택)
      → PCA 2D scatter + 군집별 crop 몽타주 → 눈으로 판정.

비교 기준: haribo 동일 분석 (CV 0.191 — 건강한 클래스가 어떻게 보이는지 대조).

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/mango_subtype_check.py
산출: outputs/figs/mango_subtype_clusters.png, outputs/figs/haribo_subtype_clusters.png (대조)
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CROP_PADDING_RATIO = 0.03
CROP_BACKGROUND = 127
INPUT_SIZE = 224
GRID_SIZE = INPUT_SIZE // 14
LAYER = 8   # h[9] = post-block-8, PatchCore와 동일

# raw polygon class_id (6-class data.yaml alphabet: 0=bottle,1=haribo,2=mango,3=metal_case,4=object,5=pencil_case)
TARGETS = [(2, "mango"), (1, "haribo")]
MAX_OBJ = 300       # 클래스당 최대 객체 (속도)
MONTAGE_PER_CLUSTER = 10
CROP_VIEW = 128     # 몽타주 셀 px


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


def collect(target_class, labeled_base, max_obj):
    samples = []
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
            H = W = 0
            for line in txt.read_text().strip().split("\n"):
                if not line.strip():
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
                crop, cmask = crop_with_mask(img, mask)
                if crop is None:
                    continue
                samples.append((crop, cmask, set_dir.name))
                if len(samples) >= max_obj:
                    return samples
    return samples


def embed_objects(samples, processor, model, device):
    embs = []
    for i, (crop, cmask, _) in enumerate(samples):
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        inputs = processor(images=pil, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        tokens = out.hidden_states[LAYER + 1][0, 1:, :]   # (256, 768)
        gm = cv2.resize(cmask.astype(np.uint8), (GRID_SIZE, GRID_SIZE), interpolation=cv2.INTER_AREA) > 0
        sel = np.where(gm.flatten())[0]
        if len(sel) == 0:
            embs.append(None)
            continue
        e = tokens[torch.from_numpy(sel).to(device)].mean(dim=0)
        embs.append(e.cpu().numpy())
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(samples)}]")
    keep = [i for i, e in enumerate(embs) if e is not None]
    return np.stack([embs[i] for i in keep]), [samples[i] for i in keep]


def analyze_class(name, samples, processor, model, device, out_dir):
    print(f"\n=== {name}: {len(samples)} objects ===")
    X, samples = embed_objects(samples, processor, model, device)
    print(f"  embeddings: {X.shape}")

    # 정규화 없이 raw L2 공간 (PatchCore와 동일 거리 공간)
    # k 선택: silhouette
    best_k, best_score, best_labels = 2, -1, None
    for k in (2, 3, 4):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        s = silhouette_score(X, km.labels_)
        print(f"  k={k}: silhouette={s:.3f}, sizes={np.bincount(km.labels_).tolist()}")
        if s > best_score:
            best_k, best_score, best_labels = k, s, km.labels_
    labels = best_labels
    print(f"  → best k={best_k} (silhouette {best_score:.3f})")

    # 군집 간/내 거리 비율 (섬이 떨어져 있는 정도)
    centroids = np.stack([X[labels == c].mean(axis=0) for c in range(best_k)])
    intra = np.mean([np.linalg.norm(X[labels == c] - centroids[c], axis=1).mean() for c in range(best_k)])
    if best_k >= 2:
        inter = np.mean([np.linalg.norm(centroids[a] - centroids[b])
                         for a in range(best_k) for b in range(a + 1, best_k)])
    else:
        inter = 0.0
    print(f"  intra-cluster mean dist={intra:.2f}, inter-centroid dist={inter:.2f}, ratio={inter/max(intra,1e-6):.2f}")

    # PCA 2D
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)

    # figure: 상단 scatter + 하단 군집별 몽타주
    n_rows_montage = best_k
    fig = plt.figure(figsize=(16, 4.5 + 2.1 * n_rows_montage))
    gs = fig.add_gridspec(1 + n_rows_montage, MONTAGE_PER_CLUSTER,
                          height_ratios=[2.2] + [1] * n_rows_montage,
                          hspace=0.35, wspace=0.05)

    ax_sc = fig.add_subplot(gs[0, :])
    cmap = plt.cm.tab10
    for c in range(best_k):
        pts = X2[labels == c]
        ax_sc.scatter(pts[:, 0], pts[:, 1], s=22, color=cmap(c),
                      label=f"cluster {c} (n={int((labels==c).sum())})", alpha=0.75)
    ax_sc.set_title(f"{name} — object embeddings (DinoV2 h[9] masked patch-mean), "
                    f"k={best_k}, silhouette={best_score:.3f}, inter/intra={inter/max(intra,1e-6):.2f}")
    ax_sc.legend()
    ax_sc.grid(alpha=0.3)

    rng = np.random.default_rng(0)
    for c in range(best_k):
        idx_c = np.where(labels == c)[0]
        # centroid에서 가까운 순으로 대표 샘플
        d = np.linalg.norm(X[idx_c] - centroids[c], axis=1)
        order = idx_c[np.argsort(d)]
        pick = order[:MONTAGE_PER_CLUSTER]
        for j in range(MONTAGE_PER_CLUSTER):
            ax = fig.add_subplot(gs[1 + c, j])
            ax.axis("off")
            if j < len(pick):
                crop = samples[pick[j]][0]
                view = cv2.resize(crop, (CROP_VIEW, CROP_VIEW))
                ax.imshow(cv2.cvtColor(view, cv2.COLOR_BGR2RGB))
                if j == 0:
                    ax.set_ylabel(f"c{c}")
                    ax.axis("on")
                    ax.set_xticks([]); ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_color(cmap(c)); spine.set_linewidth(3)

    out_path = out_dir / f"{name}_subtype_clusters.png"
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  → {out_path.relative_to(ROOT)}")
    return {"name": name, "k": best_k, "silhouette": best_score,
            "inter_intra": inter / max(intra, 1e-6)}


def main():
    from transformers import AutoImageProcessor, AutoModel
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()

    out_dir = ROOT / "outputs" / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    labeled = ROOT / "data" / "labeled"

    results = []
    for cid, name in TARGETS:
        samples = collect(cid, labeled, MAX_OBJ)
        if len(samples) < 20:
            print(f"[skip] {name}: {len(samples)} objects")
            continue
        results.append(analyze_class(name, samples, processor, model, device, out_dir))

    print("\n=== summary ===")
    for r in results:
        print(f"  {r['name']:<8} k={r['k']} silhouette={r['silhouette']:.3f} inter/intra={r['inter_intra']:.2f}")


if __name__ == "__main__":
    main()
