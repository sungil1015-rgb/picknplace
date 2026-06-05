from __future__ import annotations

import numpy as np


def normalize_normal_image(normal_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normals = np.asarray(normal_image, dtype=np.float64)
    if normals.ndim != 3 or normals.shape[2] != 3:
        raise ValueError(f"normal_image must have shape (H, W, 3), got {normals.shape}")
    norms = np.linalg.norm(normals, axis=2)
    valid = np.isfinite(norms) & (norms > 1e-6)
    normalized = np.zeros_like(normals, dtype=np.float64)
    normalized[valid] = normals[valid] / norms[valid, np.newaxis]
    return normalized, valid
