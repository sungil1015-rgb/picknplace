from __future__ import annotations

import numpy as np


def valid_depth_values(depth_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = depth_image[: mask.shape[0], : mask.shape[1]][mask]
    return values[np.isfinite(values) & (values > 0)]


def median_valid_depth_at_point(
    depth_image: np.ndarray,
    u: int,
    v: int,
    mask: np.ndarray | None = None,
    window: int = 5,
    fallback_to_mask: bool = True,
) -> float | None:
    if depth_image is None or depth_image.ndim < 2:
        return None

    half = int(window) // 2
    y1 = max(0, int(v) - half)
    y2 = min(depth_image.shape[0], int(v) + half + 1)
    x1 = max(0, int(u) - half)
    x2 = min(depth_image.shape[1], int(u) + half + 1)

    depth_patch = depth_image[y1:y2, x1:x2]
    if mask is None:
        values = depth_patch[depth_patch > 0]
    else:
        mask_patch = mask[y1:y2, x1:x2] > 0
        values = depth_patch[(depth_patch > 0) & mask_patch]
        if values.size == 0 and fallback_to_mask and np.any(mask):
            bounded_mask = mask[: depth_image.shape[0], : depth_image.shape[1]] > 0
            values = depth_image[bounded_mask]
            values = values[values > 0]

    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.median(values.astype(np.float64)))
