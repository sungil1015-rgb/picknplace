"""우리 병 픽(picking.py 출력) → main(picknplace-main) suction_points 형식 어댑터.

main의 SuctionPipeline._compute_one 출력과 **동일한 형식/규약**으로 변환한다:

    suction point = [ [x, y, z], [qx, qy, qz, qw] ]
      - 위치: extrinsic(4x4) 적용한 로봇 베이스 좌표(mm), 소수 3자리
      - 쿼터니언: z축=접근방향(법선, 로봇좌표 z-up 보정),
                  x축=reference(병 장축)를 접평면에 투영, [qx,qy,qz,qw], 소수 6자리

기하 함수는 picknplace-main/src/utils/geometry.py 및
src/pipeline/suction_pipeline.py 와 **동일 구현**(verbatim 이식)이라, 추후
SuctionPipeline에 병(cid=0) 분기를 넣을 때 이 모듈의 `bottle_pick_to_suction_point`
호출 한 줄로 병합 가능하다. main 트리 안에서 import 가능하면 main 원본을 쓴다.

병합 시 통합 지점 (suction_pipeline.py SuctionPipeline.compute):
    for instance in instances:
        if getattr(instance, "label", None) == BOTTLE_CID:   # 병이면 우리 경로
            pick = <label_logic picking 병 경로 실행>          # position_mm, normal, long_axis (카메라)
            point = bottle_pick_to_suction_point(
                pick["position_mm"], pick["normal"], pick["long_axis"], extrinsic)
            suction_points.append([point] if point else [])
            continue
        ...기존 경로...
"""
from __future__ import annotations

import numpy as np

# ── main 원본이 import 가능하면 그것을 사용 (병합 후 자동으로 단일 구현) ──
try:
    from src.utils.geometry import (            # type: ignore
        transform_point, transform_normal, transform_direction,
        approach_and_reference_to_quaternion,
    )
    _USING_MAIN_GEOMETRY = True
except Exception:
    _USING_MAIN_GEOMETRY = False

    # ↓↓ picknplace-main/src/utils/geometry.py 와 동일 구현 (verbatim) ↓↓
    def transform_point(point_xyz, transform_4x4):
        point_h = np.ones(4, dtype=np.float64)
        point_h[:3] = np.asarray(point_xyz, dtype=np.float64)
        return (np.asarray(transform_4x4, dtype=np.float64) @ point_h)[:3]

    def transform_normal(normal_xyz, transform_4x4):
        rotation = np.asarray(transform_4x4, dtype=np.float64)[:3, :3]
        normal = rotation @ np.asarray(normal_xyz, dtype=np.float64)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return normal / norm

    def transform_direction(direction_xyz, transform_4x4):
        rotation = np.asarray(transform_4x4, dtype=np.float64)[:3, :3]
        direction = rotation @ np.asarray(direction_xyz, dtype=np.float64)
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return direction / norm

    def _rotation_matrix_to_quaternion(rotation):
        r = np.asarray(rotation, dtype=np.float64)
        trace = float(np.trace(r))
        if trace > 0.0:
            scale = np.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * scale
            qx = (r[2, 1] - r[1, 2]) / scale
            qy = (r[0, 2] - r[2, 0]) / scale
            qz = (r[1, 0] - r[0, 1]) / scale
        else:
            index = int(np.argmax(np.diag(r)))
            if index == 0:
                scale = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
                qw = (r[2, 1] - r[1, 2]) / scale
                qx = 0.25 * scale
                qy = (r[0, 1] + r[1, 0]) / scale
                qz = (r[0, 2] + r[2, 0]) / scale
            elif index == 1:
                scale = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
                qw = (r[0, 2] - r[2, 0]) / scale
                qx = (r[0, 1] + r[1, 0]) / scale
                qy = 0.25 * scale
                qz = (r[1, 2] + r[2, 1]) / scale
            else:
                scale = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
                qw = (r[1, 0] - r[0, 1]) / scale
                qx = (r[0, 2] + r[2, 0]) / scale
                qy = (r[1, 2] + r[2, 1]) / scale
                qz = 0.25 * scale
        quaternion = np.array([qx, qy, qz, qw], dtype=np.float64)
        return quaternion / max(np.linalg.norm(quaternion), 1e-9)

    def approach_and_reference_to_quaternion(approach_xyz, reference_xyz):
        z_axis = np.asarray(approach_xyz, dtype=np.float64)
        z_norm = np.linalg.norm(z_axis)
        if z_norm < 1e-9:
            z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            z_axis = z_axis / z_norm

        x_axis = np.asarray(reference_xyz, dtype=np.float64)
        x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-9:
            fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(z_axis[0]) > 0.9:
                fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            x_axis = fallback - np.dot(fallback, z_axis) * z_axis
            x_norm = np.linalg.norm(x_axis)
            if x_norm < 1e-9:
                x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
                x_norm = np.linalg.norm(x_axis)
        x_axis = x_axis / max(x_norm, 1e-9)

        y_axis = np.cross(z_axis, x_axis)
        y_norm = np.linalg.norm(y_axis)
        if y_norm < 1e-9:
            y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            y_axis = y_axis - np.dot(y_axis, z_axis) * z_axis
            y_axis = y_axis / max(np.linalg.norm(y_axis), 1e-9)
            x_axis = np.cross(y_axis, z_axis)
            x_axis = x_axis / max(np.linalg.norm(x_axis), 1e-9)
        else:
            y_axis = y_axis / y_norm

        rotation = np.column_stack((x_axis, y_axis, z_axis))
        return _rotation_matrix_to_quaternion(rotation)


# ── suction_pipeline.py 와 동일 구현 (staticmethod들 verbatim) ──

def orient_normal_z_up(normal):
    """main SuctionPipeline._orient_normal_z_up 동일."""
    normal = np.asarray(normal, dtype=np.float64)
    norm = np.linalg.norm(normal)
    if norm < 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    normal = normal / norm
    if normal[2] < 0.0:
        normal = -normal
    return normal


def project_reference_to_tangent(reference, normal):
    """main SuctionPipeline._project_reference_to_tangent 동일."""
    normal = np.asarray(normal, dtype=np.float64)
    normal_norm = np.linalg.norm(normal)
    if normal_norm < 1e-9:
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        normal = normal / normal_norm

    reference = np.asarray(reference, dtype=np.float64)
    tangent = reference - np.dot(reference, normal) * normal
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm >= 1e-9:
        return tangent / tangent_norm

    fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(fallback, normal))) > 0.9:
        fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    tangent = fallback - np.dot(fallback, normal) * normal
    return tangent / max(np.linalg.norm(tangent), 1e-9)


# ── 우리 병 픽 → main suction point ────────────────────────────────────────

def bottle_pick_to_suction_point(position_mm_cam, normal_cam, long_axis_cam,
                                 extrinsic=None):
    """picking.py 병 픽 결과 1개 → main suction point [[x,y,z],[qx,qy,qz,qw]].

    Args:
      position_mm_cam: 파지점(카메라 좌표 mm) — picks의 position_mm
      normal_cam:      접근 법선(카메라 좌표) — picks의 normal
      long_axis_cam:   병 장축(카메라 좌표) — bottle_fit.axis_dir (reference로 사용:
                       흡착컵 x축을 병 길이 방향에 정렬 = main의 주축 reference와 동일 취지)
      extrinsic:       4x4 카메라→로봇 변환. None이면 항등(=카메라 좌표 그대로).

    Returns: [ [x,y,z](소수3), [qx,qy,qz,qw](소수6) ]  (main과 동일 라운딩)
    """
    if extrinsic is None:
        extrinsic = np.eye(4, dtype=np.float64)

    point_robot = transform_point(np.asarray(position_mm_cam, float), extrinsic)

    normal_robot = transform_normal(np.asarray(normal_cam, float), extrinsic)
    normal_robot = orient_normal_z_up(normal_robot)

    reference_robot = transform_direction(np.asarray(long_axis_cam, float), extrinsic)
    reference_robot = project_reference_to_tangent(reference_robot, normal_robot)

    quaternion = approach_and_reference_to_quaternion(normal_robot, reference_robot)

    return [
        [round(float(v), 3) for v in point_robot],
        [round(float(v), 6) for v in quaternion],
    ]


# main PickNPlace.FIXED_SEGMENTATION_ROI 와 동일 (pick_n_place.py 참조).
# main은 입력 roi_2d 와 무관하게 항상 이 고정 ROI를 result["2d_roi"]로 반환한다.
MAIN_FIXED_SEGMENTATION_ROI = [150.822, 1186.162, 0.0, 866.0]


def picks_to_main_result(picks_data, label_lines, extrinsic=None, img_w=1224, img_h=1024,
                         roi_2d=None):
    """샷 하나의 picks.json(dict) + 라벨 폴리곤 → main run() result dict 형식.

    main 계약:
      result_data:    List[List[[x,y]]] 객체별 폴리곤(픽셀 int)
      state:          1 정상 / -2 객체 없음
      class_id:       List[int]
      pts_per_object: List[int]
      suction_points: List[List[[ [x,y,z],[qx,qy,qz,qw] ]]]
      2d_roi:         List[float] [x0,x1,y0,y1] — main 고정 ROI(기본값) 또는 인자값
    정렬: suction 보유 객체 우선 (main은 +grasp_priority; 여기선 confidence 사용).
    """
    roi_out = list(roi_2d) if roi_2d is not None else list(MAIN_FIXED_SEGMENTATION_ROI)
    items = []
    for p in picks_data.get("picks", []):
        li = p.get("poly_index")
        if li is None or li >= len(label_lines):
            continue
        t = label_lines[li].split()
        v = np.array(t[1:], float).reshape(-1, 2)
        v[:, 0] *= img_w
        v[:, 1] *= img_h
        polygon = [[int(round(x)), int(round(y))] for x, y in v]

        pts = []
        if p.get("pickable") and p.get("position_mm") and p.get("normal"):
            bf = p.get("bottle_fit") or {}
            long_axis = bf.get("axis_dir") or p.get("long_axis")
            if long_axis:
                pts.append(bottle_pick_to_suction_point(
                    p["position_mm"], p["normal"], long_axis, extrinsic))
        items.append((polygon, int(p.get("cid", 0)), pts,
                      float(p.get("confidence") or 0.0)))

    if not items:
        return {"result_data": [], "state": -2, "class_id": [],
                "pts_per_object": [], "suction_points": [], "2d_roi": roi_out}

    items.sort(key=lambda it: (len(it[2]) > 0, it[3]), reverse=True)
    return {
        "result_data":    [it[0] for it in items],
        "state":          1,
        "class_id":       [it[1] for it in items],
        "pts_per_object": [len(it[2]) for it in items],
        "suction_points": [it[2] for it in items],
        "2d_roi":         roi_out,
    }
