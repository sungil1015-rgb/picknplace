"""Color prior — 정상 객체 mask 픽셀의 LAB 분포 (GMM) + Mahalanobis distance.

cascade detector 신호 2 (색 outlier ratio) 에서 사용.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class ColorPrior:
    def __init__(self, n_components: int = 3):
        self.n_components = int(n_components)
        self.means_: np.ndarray | None = None       # (K, D)
        self.covariances_: np.ndarray | None = None # (K, D, D)
        self.precisions_: np.ndarray | None = None  # (K, D, D)
        self.weights_: np.ndarray | None = None     # (K,)

    def fit(self, pixels_lab: np.ndarray):
        """pixels_lab: (N, 3) 정상 픽셀의 LAB 값."""
        from sklearn.mixture import GaussianMixture

        gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type="full",
            random_state=0,
            n_init=3,
        )
        gmm.fit(pixels_lab.astype(np.float64))
        self.means_ = gmm.means_.astype(np.float32)
        self.covariances_ = gmm.covariances_.astype(np.float32)
        self.weights_ = gmm.weights_.astype(np.float32)
        self.precisions_ = np.array(
            [np.linalg.inv(c).astype(np.float32) for c in self.covariances_]
        )

    def mahalanobis(self, pixels_lab: np.ndarray) -> np.ndarray:
        """각 픽셀 vs 가장 가까운 component의 Mahalanobis distance. shape: (N,)"""
        assert self.means_ is not None, "fit() 먼저 호출"
        x = pixels_lab.astype(np.float32)
        N = x.shape[0]
        min_d = np.full(N, np.inf, dtype=np.float32)
        for k in range(self.n_components):
            diff = x - self.means_[k]                            # (N, 3)
            d = np.sqrt(np.einsum("ij,jk,ik->i", diff, self.precisions_[k], diff))
            min_d = np.minimum(min_d, d)
        return min_d

    def outlier_ratio(self, pixels_lab: np.ndarray, z_threshold: float = 3.0) -> float:
        if pixels_lab.shape[0] == 0:
            return 0.0
        d = self.mahalanobis(pixels_lab)
        return float((d > z_threshold).mean())

    def save(self, path: str | Path):
        np.savez(
            path,
            means=self.means_,
            covariances=self.covariances_,
            precisions=self.precisions_,
            weights=self.weights_,
            n_components=self.n_components,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ColorPrior":
        data = np.load(path)
        obj = cls(n_components=int(data["n_components"]))
        obj.means_ = data["means"]
        obj.covariances_ = data["covariances"]
        obj.precisions_ = data["precisions"]
        obj.weights_ = data["weights"]
        return obj
