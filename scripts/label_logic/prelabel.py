"""Watershed prelabel — extracted from probe_3d_watershed.py.

Returns list of polygons (each polygon = list of (u, v) points in image pixels).
Reused by:
  - label_tool GUI (Prelabel button)
  - probe_3d_watershed.py CLI (visualization)
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import cv2
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.filters import gaussian


IMG_W, IMG_H = 1224, 1024
VERTEX_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("r", "<u1"), ("g", "<u1"), ("b", "<u1"),
    ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
])


def _read_organized_ply(path: Path):
    """Read CMES-format organized PLY (header + binary 9-channel records)."""
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError("PLY header missing end_header")
            if line.strip() == b"end_header":
                break
        n = IMG_W * IMG_H
        buf = f.read(n * VERTEX_DTYPE.itemsize)
    return np.frombuffer(buf, dtype=VERTEX_DTYPE, count=n).reshape(IMG_H, IMG_W)


def _ransac_plane(xyz, valid, n_iter=800, dist_thr=3.0, seed=42):
    """Simple RANSAC plane fit. Returns (normal, d) of n·x + d = 0,
    oriented so d > 0 (camera-side positive)."""
    rng = np.random.default_rng(seed)
    pts = xyz[valid]
    if len(pts) < 3:
        raise RuntimeError("not enough valid points for RANSAC")
    best_inl = -1
    best = None
    for _ in range(n_iter):
        idx = rng.choice(len(pts), 3, replace=False)
        p0, p1, p2 = pts[idx]
        n = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(n)
        if nn < 1e-6:
            continue
        n = n / nn
        d = -float(np.dot(n, p0))
        inliers = int((np.abs(pts @ n + d) < dist_thr).sum())
        if inliers > best_inl:
            best_inl = inliers
            best = (n, d)
    n, d = best
    if d < 0:
        n, d = -n, -d
    return n, float(d)


def run_watershed(
    ply_path: Path,
    box_roi: dict,
    height_min: float = 5.0,
    height_max: float = 300.0,
    sidewall_dot_max: float = 0.5,
    sigma: float = 2.0,
    peak_dist: int = 25,
    peak_thr: float = 8.0,
    min_region_px: int = 1500,
    plane_thr: float = 3.0,
) -> List[List[Tuple[float, float]]]:
    """Run watershed prelabel on one organized PLY.

    Args:
        ply_path: organized PLY (1224x1024).
        box_roi: dict {u_min, u_max, v_min, v_max} (axis-aligned, required).
        height_min/max: mm above box bottom plane (object band).
        sidewall_dot_max: sidewall removal threshold (|n_pixel · n_plane|<this).
        sigma: height map smoothing.
        peak_dist: min distance between watershed peaks (px).
        peak_thr: min peak height above plane (mm).
        min_region_px: drop regions smaller than this.
        plane_thr: RANSAC inlier distance (mm).

    Returns:
        List of polygons. Each polygon = list of (u, v) float tuples (>=3 points).
    """
    grid = _read_organized_ply(Path(ply_path))
    xyz = np.stack([grid["x"], grid["y"], grid["z"]], -1).astype(np.float32)
    valid = ~np.isnan(xyz[..., 2])

    n, d = _ransac_plane(xyz, valid, dist_thr=plane_thr)

    # Box ROI mask (axis-aligned)
    roi_mask = np.zeros(valid.shape, dtype=bool)
    u0, u1 = max(0, box_roi["u_min"]), min(IMG_W - 1, box_roi["u_max"])
    v0, v1 = max(0, box_roi["v_min"]), min(IMG_H - 1, box_roi["v_max"])
    roi_mask[v0:v1 + 1, u0:u1 + 1] = True

    # Height above plane
    flat = xyz.reshape(-1, 3)
    h = (flat @ n + d).reshape(xyz.shape[:2])
    h[~valid] = np.nan

    # Sidewall removal via surface normals
    nx = grid["nx"].astype(np.float32)
    ny = grid["ny"].astype(np.float32)
    nz = grid["nz"].astype(np.float32)
    valid_n = ~(np.isnan(nx) | np.isnan(ny) | np.isnan(nz))
    dot = np.abs(nx * n[0] + ny * n[1] + nz * n[2])
    sidewall = valid_n & (dot < sidewall_dot_max)

    object_mask = (valid & roi_mask & ~sidewall
                   & (h > height_min) & (h < height_max))

    # Smooth height map within mask
    h_clean = np.where(object_mask, h, 0.0).astype(np.float32)
    h_smooth = gaussian(h_clean, sigma=sigma, preserve_range=True)
    h_smooth = np.where(object_mask, h_smooth, 0.0).astype(np.float32)

    # Find peaks → markers → watershed
    peaks = peak_local_max(
        h_smooth,
        min_distance=peak_dist,
        threshold_abs=peak_thr,
        labels=object_mask.astype(np.uint8),
    )
    if len(peaks) == 0:
        return []

    markers = np.zeros(h_smooth.shape, dtype=np.int32)
    for i, (r, c) in enumerate(peaks):
        markers[r, c] = i + 1

    ws = watershed(-h_smooth, markers=markers, mask=object_mask)

    # Extract polygon contours per region
    polygons: List[List[Tuple[float, float]]] = []
    n_regions = int(ws.max())
    for cid in range(1, n_regions + 1):
        region = (ws == cid).astype(np.uint8)
        if int(region.sum()) < min_region_px:
            continue
        contours, _ = cv2.findContours(
            region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.005 * peri, True)
        pts = [(float(p[0][0]), float(p[0][1])) for p in approx]
        if len(pts) >= 3:
            polygons.append(pts)
    return polygons
