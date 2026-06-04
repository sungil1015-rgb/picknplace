"""
bottle_suction.py

물병(투명) 전용 흡착점 결정 모듈.

핵심 로직:
  마스크 bounding box의 r_obs = h_px / w_px 로 기울기 판단
  r_obs > LID_THRESHOLD  → 뚜껑 흡착  (세워진 or 앞뒤 기울기)
  r_obs < SIDE_THRESHOLD → 측면 흡착  (좌우로 많이 누운 상태)
  그 사이               → 뚜껑 시도 후 실패시 측면 fallback

※ arccos 역산 대신 r_obs 직접 비교:
   실제 Zivid 마스크에서 r_obs를 몇 장 찍어서 threshold 튜닝 권장.
   현재 기본값은 합성 마스크 실험 기준.
"""

import cv2
import numpy as np

# ── 분기 threshold (r_obs = h_px / w_px) ──────────────────
LID_THRESHOLD  = 1.5   # 이 이상 → 뚜껑 흡착
SIDE_THRESHOLD = 0.8   # 이 이하 → 측면 흡착

# 뚜껑 탐색 밴드: 마스크 상단에서 몇 % 이내
LID_BAND_RATIO = 0.20


def compute_r_obs(mask: np.ndarray) -> float:
    """마스크 bounding box 비율 r_obs = h_px / w_px."""
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return 0.0
    h = float(ys.max() - ys.min() + 1)
    w = float(xs.max() - xs.min() + 1)
    if w < 4:
        return 0.0
    return h / w


def find_lid_suction_point(
    mask: np.ndarray,
    depth_image: np.ndarray,
    window: int = 7,
) -> tuple:
    """
    뚜껑 흡착점 탐색.
    마스크 상단 LID_BAND_RATIO 영역에서 valid depth 탐색.

    Returns: (u, v) | None, depth_mm | None
    """
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None, None

    y_min = int(ys.min())
    y_max = int(ys.max())
    band_h = max(1, int((y_max - y_min) * LID_BAND_RATIO))
    y_top  = y_min + band_h

    top_mask = mask.copy()
    top_mask[y_top:, :] = 0

    return _best_valid_depth_point(top_mask, depth_image, window)


def find_side_suction_point(
    mask: np.ndarray,
    depth_image: np.ndarray,
    window: int = 7,
) -> tuple:
    """
    측면 흡착점 탐색 (물병이 많이 눕혀진 경우).
    원통 최상단 모선 근처가 depth가 잡힐 가능성이 높음.
    마스크 전체에서 distance transform 최대점 기준 탐색.

    Returns: (u, v) | None, depth_mm | None
    """
    return _best_valid_depth_point(mask, depth_image, window)


def select_bottle_suction_point(
    mask: np.ndarray,
    depth_image: np.ndarray,
    window: int = 7,
) -> dict:
    """
    메인 진입점.

    Parameters
    ----------
    mask        : binary mask (H x W, uint8)
    depth_image : depth map in mm (H x W, float32 or uint16)
    window      : depth median 탐색 윈도우 크기

    Returns
    -------
    dict:
        "mode"   : "lid" | "side" | "failed"
        "uv"     : (u, v) 픽셀 좌표 또는 None
        "depth_mm" : float 또는 None
        "r_obs"  : 마스크 비율 (디버그용)
    """
    r_obs = compute_r_obs(mask)

    if r_obs >= LID_THRESHOLD:
        # ── 세워진 상태 → 뚜껑 우선 ──
        uv, depth = find_lid_suction_point(mask, depth_image, window)
        if uv is not None:
            return {"mode": "lid", "uv": uv, "depth_mm": depth, "r_obs": r_obs}
        # fallback: 측면
        uv, depth = find_side_suction_point(mask, depth_image, window)
        mode = "side" if uv is not None else "failed"
        return {"mode": mode, "uv": uv, "depth_mm": depth, "r_obs": r_obs}

    elif r_obs <= SIDE_THRESHOLD:
        # ── 누운 상태 → 측면 우선 ──
        uv, depth = find_side_suction_point(mask, depth_image, window)
        if uv is not None:
            return {"mode": "side", "uv": uv, "depth_mm": depth, "r_obs": r_obs}
        # fallback: 뚜껑
        uv, depth = find_lid_suction_point(mask, depth_image, window)
        mode = "lid" if uv is not None else "failed"
        return {"mode": mode, "uv": uv, "depth_mm": depth, "r_obs": r_obs}

    else:
        # ── 중간 (0.8 ~ 1.5) → 뚜껑 시도 후 측면 fallback ──
        uv, depth = find_lid_suction_point(mask, depth_image, window)
        if uv is not None:
            return {"mode": "lid", "uv": uv, "depth_mm": depth, "r_obs": r_obs}
        uv, depth = find_side_suction_point(mask, depth_image, window)
        mode = "side" if uv is not None else "failed"
        return {"mode": mode, "uv": uv, "depth_mm": depth, "r_obs": r_obs}


# ── 내부 유틸 ────────────────────────────────────────────────

def _best_valid_depth_point(
    mask: np.ndarray,
    depth_image: np.ndarray,
    window: int,
) -> tuple:
    """
    mask 내부에서 valid depth가 가장 안정적인 점 반환.
    distance transform 내림차순으로 후보를 뽑고
    윈도우 내 valid depth median이 있는 첫 번째 후보 반환.
    """
    binary = (mask > 0).astype(np.uint8)
    if not np.any(binary):
        return None, None

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    flat_idx = np.argsort(dist.ravel())[::-1][:30]
    h, w = mask.shape
    candidates = [(int(i % w), int(i // w)) for i in flat_idx]

    half = window // 2
    min_valid = max(1, (window * window) // 4)

    for u, v in candidates:
        v0 = max(0, v - half); v1 = min(h, v + half + 1)
        u0 = max(0, u - half); u1 = min(w, u + half + 1)

        region = depth_image[v0:v1, u0:u1].astype(np.float32)
        rmask  = binary[v0:v1, u0:u1]
        valid  = region[(rmask > 0) & (region > 0)]

        if len(valid) < min_valid:
            continue

        depth_mm = float(np.median(valid))
        if depth_mm <= 0:
            continue

        return (u, v), depth_mm

    return None, None
