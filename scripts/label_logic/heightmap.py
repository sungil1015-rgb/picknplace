"""Height map / Normal map RGBA overlays for canvas H/M toggle layers.

Used by GUI's Layers section — H/M keys toggle these on top of BMP image.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib

from .prelabel import _read_organized_ply, _ransac_plane


def height_map_rgba(ply_path: Path,
                    alpha: float = 0.6,
                    height_min: float = 5.0,
                    height_max: float = 200.0,
                    cmap: str = "viridis") -> np.ndarray:
    """Return (H, W, 4) uint8 RGBA. Alpha = 0 where invalid or out of range.

    height_min/max in mm above box bottom plane.
    """
    grid = _read_organized_ply(Path(ply_path))
    xyz = np.stack([grid["x"], grid["y"], grid["z"]], -1).astype(np.float32)
    valid = ~np.isnan(xyz[..., 2])
    n, d = _ransac_plane(xyz, valid)
    flat = xyz.reshape(-1, 3)
    h = (flat @ n + d).reshape(xyz.shape[:2])
    in_band = valid & (h > height_min) & (h < height_max)

    h_norm = np.zeros_like(h)
    if in_band.any():
        lo = float(h[in_band].min())
        hi = float(h[in_band].max())
        if hi > lo:
            h_norm = np.clip((h - lo) / (hi - lo), 0.0, 1.0)

    rgba = (matplotlib.colormaps[cmap](h_norm) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(in_band, int(alpha * 255), 0).astype(np.uint8)
    return rgba


def normal_map_rgba(ply_path: Path, alpha: float = 0.5) -> np.ndarray:
    """Return (H, W, 4) uint8 RGBA. nx/ny/nz mapped to RGB (-1..1 → 0..255)."""
    grid = _read_organized_ply(Path(ply_path))
    nx = grid["nx"].astype(np.float32)
    ny = grid["ny"].astype(np.float32)
    nz = grid["nz"].astype(np.float32)
    valid = ~(np.isnan(nx) | np.isnan(ny) | np.isnan(nz))
    rgba = np.zeros((nx.shape[0], nx.shape[1], 4), dtype=np.uint8)
    rgba[..., 0] = np.clip((nx + 1) * 127.5, 0, 255).astype(np.uint8)
    rgba[..., 1] = np.clip((ny + 1) * 127.5, 0, 255).astype(np.uint8)
    rgba[..., 2] = np.clip((nz + 1) * 127.5, 0, 255).astype(np.uint8)
    rgba[..., 3] = np.where(valid, int(alpha * 255), 0).astype(np.uint8)
    return rgba
