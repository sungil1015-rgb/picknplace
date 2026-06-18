"""PatchCore-lite — DinoV2 mid-layer patch features + memory bank + L2 nearest neighbor.

memory bank는 미리 빌드된 .npz 파일 로드 (scripts/build_patchcore_refs.py 산출).
dinov2_features dict가 주어지면 backbone forward 재사용, 없으면 직접 forward.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

from src.defect.base import DefectDetector, DefectResult


class PatchCoreLite(DefectDetector):
    def __init__(
        self,
        memory_bank: str,
        threshold: float,
        dinov2_layer: int = 8,
        patch_aggregate: str = "max",   # "max" | "top_p_mean" | "top_k_mean"
        top_p: float = 0.01,
        top_k: int = 4,                 # top_k_mean용 — 글레어 등 단일 patch outlier 민감도 완화
        action: str = "reject",
        enabled: bool = True,
        device: Optional[str] = None,
        model_name: str = "facebook/dinov2-base",
        input_size: int = 224,
        crop_padding_ratio: float = 0.03,
        crop_background: int = 127,
    ):
        self.memory_bank_path = Path(memory_bank)
        if not self.memory_bank_path.is_absolute():
            self.memory_bank_path = Path.cwd() / self.memory_bank_path
        if not self.memory_bank_path.is_file():
            raise FileNotFoundError(f"memory bank not found: {self.memory_bank_path}")

        data = np.load(self.memory_bank_path)
        self.memory_bank_np = data["memory_bank"].astype(np.float32)   # (N, D)
        # layer는 config(dinov2_layer)가 source of truth. 단 메모리뱅크가 다른 layer로
        # 빌드돼 있으면 테스트 patch(L_config) vs 뱅크 벡터(L_bank) 비교가 무의미해지므로
        # 불일치 시 즉시 에러 (조용히 틀린 결과 방지). 뱅크에 layer 메타 없으면 config 신뢰(legacy).
        bank_layer = int(data["dinov2_layer"]) if "dinov2_layer" in data else None
        if bank_layer is not None and bank_layer != int(dinov2_layer):
            raise ValueError(
                f"layer mismatch: config는 L{dinov2_layer}를 요청했으나 memory bank "
                f"'{self.memory_bank_path.name}'는 L{bank_layer}로 빌드됨. "
                f"L{dinov2_layer}로 재빌드하거나(scripts/build_patchcore_refs.py --layer {dinov2_layer}) "
                f"config dinov2_layer를 {bank_layer}로 되돌리세요."
            )
        self.layer = int(dinov2_layer)
        self.input_size = int(data["input_size"]) if "input_size" in data else input_size
        self.grid_size = self.input_size // 14

        self.threshold = float(threshold)
        self.patch_aggregate = str(patch_aggregate)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.action = str(action)
        self.enabled = bool(enabled)
        self.crop_padding_ratio = float(crop_padding_ratio)
        self.crop_background = int(crop_background)
        self.model_name = str(model_name)

        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.memory_bank = torch.from_numpy(self.memory_bank_np).to(self.device)
        self._mb_sq = (self.memory_bank * self.memory_bank).sum(dim=1)   # (N,) precomputed

        # backbone (dinov2_features 캐시 없을 때 직접 forward용)
        self._processor = None
        self._model = None

    def _ensure_backbone(self):
        if self._model is None:
            from transformers import AutoImageProcessor, AutoModel
            self._processor = AutoImageProcessor.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()

    def _crop_with_mask(self, rgb_bgr: np.ndarray, mask: np.ndarray):
        H, W = rgb_bgr.shape[:2]
        m = mask > 0 if mask.dtype != bool else mask
        ys, xs = np.where(m)
        if len(ys) == 0:
            return None, None
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        side = int(np.ceil(max(x2 - x1, y2 - y1) * (1.0 + self.crop_padding_ratio * 2.0)))
        if side <= 0:
            return None, None
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        crop_left = int(np.floor(cx - side * 0.5))
        crop_top = int(np.floor(cy - side * 0.5))
        crop_right = crop_left + side
        crop_bottom = crop_top + side
        sx1 = max(0, crop_left); sy1 = max(0, crop_top)
        sx2 = min(W, crop_right); sy2 = min(H, crop_bottom)
        crop = np.full((side, side, 3), self.crop_background, dtype=np.uint8)
        crop_mask = np.zeros((side, side), dtype=bool)
        if sx2 > sx1 and sy2 > sy1:
            dx1, dy1 = sx1 - crop_left, sy1 - crop_top
            dx2, dy2 = dx1 + (sx2 - sx1), dy1 + (sy2 - sy1)
            crop[dy1:dy2, dx1:dx2] = rgb_bgr[sy1:sy2, sx1:sx2]
            crop_mask[dy1:dy2, dx1:dx2] = m[sy1:sy2, sx1:sx2]
        crop[~crop_mask] = self.crop_background
        return crop, crop_mask

    def _extract_patches_via_forward(self, rgb_bgr: np.ndarray, mask: np.ndarray):
        crop, crop_mask = self._crop_with_mask(rgb_bgr, mask)
        if crop is None:
            return None
        self._ensure_backbone()
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(crop_rgb)
        inputs = self._processor(images=pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=True)
        patch_tokens = outputs.hidden_states[self.layer + 1][0, 1:, :]   # (n_tokens, D)
        grid_mask = cv2.resize(crop_mask.astype(np.uint8), (self.grid_size, self.grid_size), interpolation=cv2.INTER_AREA) > 0
        selected_idx = np.where(grid_mask.flatten())[0]
        if len(selected_idx) == 0:
            return None
        return patch_tokens[torch.from_numpy(selected_idx).to(self.device)], grid_mask

    def _extract_patches_from_cache(self, mask: np.ndarray, dinov2_features: dict):
        """classifier가 캐시한 patch tokens 재사용 (forward 추가 0)."""
        key = f"patch_layer_{self.layer}"
        if key not in dinov2_features:
            return None
        crop_mask = dinov2_features.get("crop_mask")
        if crop_mask is None:
            # fallback: assume input mask is already crop-aligned
            return None
        patch_tokens = dinov2_features[key]   # (n_tokens, D) torch tensor on device
        grid_mask = cv2.resize(crop_mask.astype(np.uint8), (self.grid_size, self.grid_size), interpolation=cv2.INTER_AREA) > 0
        selected_idx = np.where(grid_mask.flatten())[0]
        if len(selected_idx) == 0:
            return None
        return patch_tokens[torch.from_numpy(selected_idx).to(patch_tokens.device)], grid_mask

    def score(
        self,
        rgb_bgr: np.ndarray,
        mask: np.ndarray,
        normal_map: Optional[np.ndarray] = None,
        dinov2_features: Optional[dict] = None,
    ) -> DefectResult:
        # patch tokens 확보
        result = None
        if dinov2_features is not None:
            result = self._extract_patches_from_cache(mask, dinov2_features)
        if result is None:
            result = self._extract_patches_via_forward(rgb_bgr, mask)
        if result is None:
            return DefectResult(score=0.0, is_defect=False)
        test_patches, grid_mask = result   # (n_q, D), (grid, grid)

        # L2 NN via (a-b)^2 = a^2 + b^2 - 2 a.b
        if test_patches.device != self.memory_bank.device:
            test_patches = test_patches.to(self.memory_bank.device)
        a2 = (test_patches * test_patches).sum(dim=1, keepdim=True)   # (n_q, 1)
        ab = test_patches @ self.memory_bank.T                         # (n_q, N)
        d2 = (a2 + self._mb_sq.unsqueeze(0) - 2 * ab).clamp(min=0)
        nn_dist = d2.sqrt().min(dim=1).values                          # (n_q,)

        if self.patch_aggregate == "max":
            score = float(nn_dist.max())
        elif self.patch_aggregate == "top_k_mean":
            k = min(self.top_k, len(nn_dist))
            score = float(torch.topk(nn_dist, k).values.mean())
        else:   # top_p_mean
            k = max(1, int(len(nn_dist) * self.top_p))
            score = float(torch.topk(nn_dist, k).values.mean())

        # heatmap scatter
        score_map_flat = np.zeros(self.grid_size * self.grid_size, dtype=np.float32)
        flat_idx = np.where(grid_mask.flatten())[0]
        score_map_flat[flat_idx] = nn_dist.detach().cpu().numpy()
        score_map_grid = score_map_flat.reshape(self.grid_size, self.grid_size)
        H, W = mask.shape[:2]
        score_map = cv2.resize(score_map_grid, (W, H), interpolation=cv2.INTER_LINEAR)

        return DefectResult(
            score=score,
            is_defect=score > self.threshold,
            score_map=score_map,
        )
