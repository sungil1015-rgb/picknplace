# -*- coding: utf-8 -*-
"""동욱 picking.py 를 팀 파이프라인에 연결하는 어댑터 (SuctionPipeline 대체).

흐름:
  depth(HxW, mm) + intrinsic(3x3) → organized XYZ grid (camera mm)
  → label_logic.picking.compute_pick (PickResult, camera frame mm)
  → 팀 geometry 헬퍼로 robot 좌표 [[x,y,z],[qx,qy,qz,qw]] 변환.

PICKABLE 이 아닌 인스턴스는 빈 리스트([])를 반환 (팀 포맷 동일).
진단값(pick_status/confidence/tier)은 instance 속성으로도 부착한다.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# label_logic.picking 임포트 — 이 파일은 src/pipeline/ → repo root = parents[2], scripts/ 하위
_REPO = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from label_logic import picking as _picking  # noqa: E402

from src.utils.geometry import (  # noqa: E402
    transform_point,
    transform_normal,
    transform_direction,
    orient_normal_z_up,
    project_to_tangent,
    approach_and_reference_to_quaternion,
)


# label_logic.picking 이 기대하는 organized PLY grid dtype (Zivid CMES 포맷)
GRID_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("r", "<u1"), ("g", "<u1"), ("b", "<u1"),
    ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
])


def build_grid(
    depth_image: np.ndarray,
    rgb_image: np.ndarray | None,
    normal_image: np.ndarray | None,
    intrinsic: np.ndarray,
) -> np.ndarray:
    """depth(mm)+rgb(BGR)+normal → picking 이 기대하는 구조화 organized grid (HxW).

    필드: x,y,z(camera mm) / r,g,b(uint8) / nx,ny,nz(camera 법선).
    무효 깊이(0/NaN/Inf)는 x,y,z,nx,ny,nz 를 NaN 으로 둔다 (picking 의 valid 마스크용).
    """
    h, w = depth_image.shape[:2]
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    z = depth_image.astype(np.float32)
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    invalid = ~np.isfinite(z) | (z <= 0)

    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    for arr in (x, y, z):
        arr[invalid] = np.nan

    grid = np.zeros((h, w), dtype=GRID_DTYPE)
    grid["x"] = x
    grid["y"] = y
    grid["z"] = z

    if rgb_image is not None:
        # 팀 rgb_image 는 BGR(cv2) → picking 은 r,g,b
        grid["r"] = rgb_image[..., 2]
        grid["g"] = rgb_image[..., 1]
        grid["b"] = rgb_image[..., 0]

    if normal_image is not None:
        nx = normal_image[..., 0].astype(np.float32).copy()
        ny = normal_image[..., 1].astype(np.float32).copy()
        nz = normal_image[..., 2].astype(np.float32).copy()
        for arr in (nx, ny, nz):
            arr[invalid] = np.nan
        grid["nx"] = nx
        grid["ny"] = ny
        grid["nz"] = nz
    else:
        grid["nx"] = np.nan
        grid["ny"] = np.nan
        grid["nz"] = np.nan

    return grid


class DongukPicking:
    """label_logic.picking.compute_pick 기반 파지점 계산기.

    팀 SuctionPipeline.compute 와 같은 역할이되, picking 이 폴리곤+cid+organized grid
    를 쓰므로 compute() 에 polygons / class_ids 를 추가로 받는다.
    """

    def __init__(self, logger: Any = None) -> None:
        self.logger = logger

    def compute(
        self,
        instances: Sequence[Any],
        polygons: Sequence[Sequence[Sequence[float]]],
        class_ids: Sequence[int],
        rgb_image: np.ndarray | None,
        depth_image: np.ndarray | None,
        normal_image: np.ndarray | None,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
    ) -> list[list[list[list[float]]]]:
        if depth_image is None:
            return [[] for _ in instances]

        fx = float(intrinsic[0, 0])
        fy = float(intrinsic[1, 1])
        cx = float(intrinsic[0, 2])
        cy = float(intrinsic[1, 2])
        _picking.set_camera_intrinsics(fx, fy, cx, cy)
        grid = build_grid(depth_image, rgb_image, normal_image, intrinsic)

        all_polygons = [list(p) for p in polygons]
        out: list[list[list[list[float]]]] = []

        for i, (inst, poly, cid) in enumerate(zip(instances, polygons, class_ids)):
            # 가림(S1) 검사용 scene = 자기 자신을 뺀 나머지 폴리곤
            scene_others = [p for j, p in enumerate(all_polygons) if j != i]
            try:
                res = _picking.compute_pick(
                    grid, poly, int(cid), scene_polygons=scene_others,
                )
            except Exception as exc:  # noqa: BLE001
                if self.logger is not None:
                    self.logger.exception(f"[DongukPicking] compute_pick 실패 (cid={cid}): {exc}")
                setattr(inst, "pick_status", "ERROR")
                out.append([])
                continue

            setattr(inst, "pick_status", res.status)
            setattr(inst, "pick_confidence", float(res.confidence))
            setattr(inst, "pick_tier", res.tier)

            if not res.success or res.status != _picking.STATUS_PICKABLE:
                out.append([])
                continue

            pos_robot = transform_point(np.asarray(res.position_mm, dtype=float), extrinsic)
            normal_robot = orient_normal_z_up(
                transform_normal(np.asarray(res.normal, dtype=float), extrinsic)
            )
            ref_robot = project_to_tangent(
                transform_direction(np.asarray(res.long_axis, dtype=float), extrinsic),
                normal_robot,
            )
            quaternion = approach_and_reference_to_quaternion(normal_robot, ref_robot)
            out.append([
                [round(float(v), 3) for v in pos_robot],
                [round(float(q), 6) for q in quaternion],
            ])

        return out
