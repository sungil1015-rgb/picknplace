"""합성 결함 generator — threshold ROC tune용.

스크래치 (rigid 표면): random line + dark color + blur.
찢어짐 (deformable 표면): random irregular patch + 속살 색 노출.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def synthesize_scratch(
    rgb_bgr: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    n_lines_range: tuple = (3, 8),
    length_range: tuple = (40, 180),
    thickness_choices: tuple = (2, 3, 4, 5, 6),
    color_range: tuple = (0, 40),       # 매우 어둡게 (스크래치 가시화)
    blur_ksize: int = 0,                # blur X (sharp edge 유지)
) -> np.ndarray:
    """mask 영역 안에 random scratch lines 합성."""
    out = rgb_bgr.copy()
    m = mask > 0 if mask.dtype != bool else mask
    ys, xs = np.where(m)
    if len(ys) == 0:
        return out
    n_lines = rng.integers(n_lines_range[0], n_lines_range[1])
    for _ in range(n_lines):
        i = rng.integers(len(ys))
        x1, y1 = int(xs[i]), int(ys[i])
        length = rng.integers(length_range[0], length_range[1])
        angle = rng.uniform(0, 2 * np.pi)
        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))
        thickness = int(rng.choice(thickness_choices))
        color = tuple(int(c) for c in rng.integers(color_range[0], color_range[1], 3))
        cv2.line(out, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    if blur_ksize >= 3:
        out = cv2.GaussianBlur(out, (blur_ksize, blur_ksize), 0)
    return out


def _create_random_blob(shape: tuple, center: tuple, area: int, rng: np.random.Generator) -> np.ndarray:
    """random walk으로 mask 내 불규칙 blob 생성."""
    H, W = shape
    mask = np.zeros((H, W), dtype=bool)
    cx, cy = int(center[0]), int(center[1])
    visited = {(cx, cy)}
    frontier = [(cx, cy)]
    while frontier and len(visited) < area:
        i = int(rng.integers(len(frontier)))
        x, y = frontier.pop(i)
        if 0 <= x < W and 0 <= y < H:
            mask[y, x] = True
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H and (nx, ny) not in visited and rng.random() < 0.65:
                visited.add((nx, ny))
                frontier.append((nx, ny))
    return mask


def synthesize_tear(
    rgb_bgr: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    inner_color_bgr: tuple = (30, 60, 90),       # 더 어두운 갈색 (속살, 정상 비닐 색과 명확 구분)
    area_range: tuple = (400, 1500),             # area_mode="absolute"일 때만 사용
    noise_range: tuple = (-15, 15),
    near_edge: bool = True,
    area_mode: str = "absolute",                 # "absolute" | "mask_ratio"
    area_ratio_range: tuple = (0.03, 0.07),      # mask 면적 대비 비율 (mask_ratio 모드)
) -> np.ndarray:
    """deformable 객체 mask 안에 찢어짐 시뮬 (random irregular blob에 속살 색 덮음).

    near_edge=True면 mask 경계 근처에서 시작 (실제 찢어짐 위치 패턴).

    area_mode:
      "absolute"  — tear 면적을 절대 픽셀로 샘플 (구버전, 고정 물리 크기 결함에 대응).
      "mask_ratio" — mask 면적의 비율로 샘플. 객체 크기가 클래스마다 달라도
                     crop resize 후 tear 점유율이 일정해짐 (mango 스케일 희석 문제 해결).
                     비율 3~7%는 haribo 절대픽셀 현행 조건 (~4.5%) 기준 — 회귀 호환.
    """
    out = rgb_bgr.copy()
    m = mask > 0 if mask.dtype != bool else mask
    ys, xs = np.where(m)
    if len(ys) == 0:
        return out

    if near_edge:
        # erode → edge 후보
        m_u8 = (m.astype(np.uint8) * 255)
        eroded = cv2.erode(m_u8, np.ones((7, 7), np.uint8))
        edge_band = (m_u8 > 0) & (eroded == 0)
        ey, ex = np.where(edge_band)
        if len(ey) > 0:
            i = int(rng.integers(len(ey)))
            cx, cy = int(ex[i]), int(ey[i])
        else:
            i = int(rng.integers(len(ys)))
            cx, cy = int(xs[i]), int(ys[i])
    else:
        i = int(rng.integers(len(ys)))
        cx, cy = int(xs[i]), int(ys[i])

    if area_mode == "mask_ratio":
        mask_area = int(m.sum())
        lo = max(50, int(mask_area * area_ratio_range[0]))
        hi = max(lo + 1, int(mask_area * area_ratio_range[1]))
        area = int(rng.integers(lo, hi))
    else:
        area = int(rng.integers(area_range[0], area_range[1]))
    blob_mask = _create_random_blob(out.shape[:2], (cx, cy), area, rng)
    blob_mask &= m

    n = int(blob_mask.sum())
    if n == 0:
        return out
    noise = rng.integers(noise_range[0], noise_range[1], size=(n, 3))
    base = np.array(inner_color_bgr, dtype=np.int32)
    new_color = np.clip(base + noise, 0, 255).astype(np.uint8)
    out[blob_mask] = new_color
    return out
