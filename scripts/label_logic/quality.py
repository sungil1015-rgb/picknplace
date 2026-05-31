"""Per-polygon 3D quality stats — extracted from probe_3d_quality.py.

Used by GUI's "3D Stats" side panel widget and CLI scripts.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .prelabel import _read_organized_ply, IMG_W, IMG_H


@dataclass
class PolygonStats:
    n_points: int             # total pixels inside polygon
    n_valid: int              # 3D-valid pixels (z not NaN)
    valid_ratio: float        # 0..1
    z_mean: float             # mm; NaN if no valid points
    z_std: float
    norm_cos: float           # 1.0 = perfectly aligned normals; NaN if no valid


def _polygon_mask(polygon: List[Tuple[float, float]]) -> np.ndarray:
    img = Image.new("L", (IMG_W, IMG_H), 0)
    pts = [(float(p[0]), float(p[1])) for p in polygon]
    if len(pts) < 3:
        return np.zeros((IMG_H, IMG_W), dtype=bool)
    ImageDraw.Draw(img).polygon(pts, fill=1)
    return np.array(img, dtype=bool)


def compute_polygon_stats(ply_path: Path,
                          polygon: List[Tuple[float, float]]) -> PolygonStats:
    """Compute 3D quality stats for points inside a 2D polygon (image pixels)."""
    grid = _read_organized_ply(Path(ply_path))
    mask = _polygon_mask(polygon)
    if not mask.any():
        return PolygonStats(0, 0, 0.0,
                            float("nan"), float("nan"), float("nan"))
    pts = grid[mask]
    z = pts["z"]
    valid = ~np.isnan(z)
    n_total = int(len(pts))
    n_valid = int(valid.sum())
    if n_valid == 0:
        return PolygonStats(n_total, 0, 0.0,
                            float("nan"), float("nan"), float("nan"))
    z_v = z[valid]
    z_mean = float(np.mean(z_v))
    z_std = float(np.std(z_v))
    nx = pts["nx"][valid]
    ny = pts["ny"][valid]
    nz = pts["nz"][valid]
    valid_n = ~(np.isnan(nx) | np.isnan(ny) | np.isnan(nz))
    if int(valid_n.sum()) < 5:
        norm_cos = float("nan")
    else:
        nx, ny, nz = nx[valid_n], ny[valid_n], nz[valid_n]
        mean_n = np.array([nx.mean(), ny.mean(), nz.mean()])
        norm = float(np.linalg.norm(mean_n))
        if norm < 1e-6:
            norm_cos = float("nan")
        else:
            mean_n = mean_n / norm
            dots = nx * mean_n[0] + ny * mean_n[1] + nz * mean_n[2]
            norm_cos = float(np.mean(dots))
    return PolygonStats(
        n_points=n_total, n_valid=n_valid,
        valid_ratio=n_valid / n_total,
        z_mean=z_mean, z_std=z_std, norm_cos=norm_cos,
    )
