"""teammate(picknplace) suction_pipeline 입력 → 우리 병 파지점 → main suction point.

teammate `SuctionPipeline.compute` 가 가진 입력(mask, depth_image, normal_image,
intrinsic 3x3, extrinsic 4x4)을 그대로 받아, 병(bottle) 인스턴스의 파지점을
우리 로직(`compute_pick`)으로 계산하고 main suction point 형식으로 돌려준다.
teammate 파일은 건드리지 않고, 병일 때 아래 한 호출만 추가하면 된다.

── 사용법 (src/pipeline/suction_pipeline.py 의 compute() 인스턴스 루프 안) ──

    from label_logic.main_bridge import bottle_suction_point  # sys.path에 scripts/

    for instance in instances:
        label = getattr(instance, "label", None)
        if label is not None and int(label) == BOTTLE_LABEL:      # 병이면 우리 경로
            point = bottle_suction_point(
                instance.mask, depth_image, normal_image,
                intrinsic, extrinsic, rgb_image=rgb_image)        # rgb는 선택
            suction_points.append([point] if point is not None else [])
            continue
        ... 기존 generic 경로 ...

반환: [[x,y,z](소수3), [qx,qy,qz,qw](소수6)]  (robot frame) — main과 동일 형식.
      파지 불가/검출 실패 시 None.

주의:
  - depth_image 는 mm, normal_image 는 (H,W,3) 카메라 좌표 법선, intrinsic 은 3x3.
  - 이미지 해상도는 config/default.yaml 의 camera.resolution 과 같아야 한다
    (같은 카메라면 자동 일치). 다르면 명확한 에러를 던진다.
  - rgb_image(H,W,3 uint8)를 주면 병 캡 밝기 매칭/복원 정확도가 올라간다(선택).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from . import picking as _pk
from .picking import (
    compute_pick, set_camera_intrinsics, BOTTLE_CID, UNKNOWN_CID,
)
from .main_suction_adapter import bottle_pick_to_suction_point

# grid dtype = 우리 organized PLY 와 동일 (label_logic/prelabel.VERTEX_DTYPE).
_GRID_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("r", "<u1"), ("g", "<u1"), ("b", "<u1"),
    ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
])


def build_grid(depth_image, normal_image, intrinsic, rgb_image=None) -> np.ndarray:
    """depth(mm,HxW) + normal(HxWx3) + intrinsic(3x3) → 우리 grid (structured, HxW).

    - z: 유효하지 않은 깊이(≤0 또는 비유한)는 NaN (우리 로직이 invalid로 처리)
    - x,y: 핀홀 역투영  x=(u-cx)z/fx, y=(v-cy)z/fy  (teammate pixel_to_camera 동일)
    - nx,ny,nz: normal_image 그대로. 없으면 nz=1 임시
    - r,g,b: rgb_image 있으면 채움. 없으면 0 (캡 밝기 prior만 약해짐)
    """
    depth = np.asarray(depth_image, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError(f"depth_image 는 (H,W) 여야 함, got {depth.shape}")
    H, W = depth.shape
    K = np.asarray(intrinsic, dtype=np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    z = depth.copy()
    z[~np.isfinite(z) | (z <= 0.0)] = np.nan

    uu, vv = np.meshgrid(np.arange(W, dtype=np.float64),
                         np.arange(H, dtype=np.float64))
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy

    grid = np.zeros((H, W), dtype=_GRID_DTYPE)
    grid["x"] = x.astype("<f4")
    grid["y"] = y.astype("<f4")
    grid["z"] = z.astype("<f4")
    if normal_image is not None:
        n = np.asarray(normal_image, dtype=np.float64)
        grid["nx"] = n[..., 0].astype("<f4")
        grid["ny"] = n[..., 1].astype("<f4")
        grid["nz"] = n[..., 2].astype("<f4")
    else:
        grid["nz"] = np.float32(1.0)
    if rgb_image is not None:
        rgb = np.asarray(rgb_image)
        grid["r"] = rgb[..., 0].astype("<u1")
        grid["g"] = rgb[..., 1].astype("<u1")
        grid["b"] = rgb[..., 2].astype("<u1")
    return grid


def _mask_to_polygon(mask) -> Optional[list]:
    """이진 mask → 최대 외곽 컨투어 폴리곤 [[x,y], ...]."""
    if cv2 is None:
        raise RuntimeError("cv2(OpenCV) 필요 — mask→polygon 변환")
    m = (np.asarray(mask) > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    eps = 0.01 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps, True)
    poly = approx.reshape(-1, 2).astype(float).tolist()
    return poly if len(poly) >= 3 else None


def _check_resolution(depth) -> None:
    """이미지 해상도가 config camera.resolution 과 같은지 확인."""
    if depth.ndim != 2:
        raise ValueError(f"depth_image 는 (H,W) 여야 함, got {depth.shape}")
    H, W = depth.shape
    if (H, W) != (_pk.IMG_H, _pk.IMG_W):
        raise ValueError(
            f"이미지 해상도 {(W, H)} != config camera.resolution "
            f"{(_pk.IMG_W, _pk.IMG_H)}. 같은 카메라를 쓰거나 "
            f"config/default.yaml 의 camera.resolution 을 맞추세요.")


def _pick_point(grid, mask, cid, extrinsic) -> Optional[list]:
    """grid + mask + cid → main suction point [[x,y,z],[qx,qy,qz,qw]] 또는 None.

    cid: 병이면 BOTTLE_CID(병 경로), 그 외는 UNKNOWN_CID(일반 흡착 경로) 권장.
    """
    polygon = _mask_to_polygon(mask)
    if polygon is None:
        return None
    result = compute_pick(grid, polygon, cid=cid)
    if not result.success:
        return None
    # reference(흡착컵 x축 정렬축): 병이면 원기둥 축(bottle_fit.axis_dir), 아니면 PCA 장축.
    long_axis = result.long_axis
    if result.bottle_fit and result.bottle_fit.get("axis_dir"):
        long_axis = result.bottle_fit["axis_dir"]
    return bottle_pick_to_suction_point(
        result.position_mm, result.normal, long_axis,
        np.asarray(extrinsic, dtype=np.float64))


def bottle_suction_point(mask, depth_image, normal_image, intrinsic, extrinsic,
                         rgb_image=None) -> Optional[list]:
    """병 인스턴스 1개 → main suction point [[x,y,z],[qx,qy,qz,qw]] 또는 None.

    teammate `SuctionPipeline.compute` 에서 label==bottle 일 때 호출.
    """
    depth = np.asarray(depth_image)
    _check_resolution(depth)
    K = np.asarray(intrinsic, dtype=np.float64)
    set_camera_intrinsics(K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    grid = build_grid(depth_image, normal_image, intrinsic, rgb_image)
    return _pick_point(grid, mask, BOTTLE_CID, extrinsic)


def compute_suction_points(instances, depth_image, normal_image, intrinsic,
                           extrinsic, rgb_image=None, bottle_label=4):
    """teammate SuctionPipeline.compute 와 동일 출력 — 전 인스턴스를 우리 로직으로.

    각 인스턴스의 흡착 파지점을 우리 compute_pick 으로 계산해 main 형식으로 돌려준다.
      - label == bottle_label(기본 4) → 병 경로(BOTTLE_CID)
      - 그 외 → 일반 흡착 경로(UNKNOWN_CID; class_prior 중립, 병 특화 skip)

    Returns: list[list[suction_point]]  (= [[point] or [] for each instance])
             teammate compute() 반환과 동일 구조.
    """
    if depth_image is None:
        return [[] for _ in instances]
    depth = np.asarray(depth_image)
    _check_resolution(depth)
    K = np.asarray(intrinsic, dtype=np.float64)
    set_camera_intrinsics(K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    # grid 는 샷당 1회만 생성(인스턴스마다 재계산 X)
    grid = build_grid(depth_image, normal_image, intrinsic, rgb_image)

    out = []
    for inst in instances:
        mask = getattr(inst, "mask", None)
        if mask is None:
            out.append([])
            continue
        label = getattr(inst, "label", None)
        try:
            is_bottle = label is not None and int(label) == int(bottle_label)
        except (TypeError, ValueError):
            is_bottle = False
        cid = BOTTLE_CID if is_bottle else UNKNOWN_CID
        point = _pick_point(grid, mask, cid, extrinsic)
        out.append([point] if point is not None else [])
    return out
