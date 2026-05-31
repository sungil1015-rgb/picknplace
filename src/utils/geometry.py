from __future__ import annotations

import numpy as np


def make_intrinsic(cx: float, cy: float, fx: float, fy: float) -> np.ndarray:
    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def rotation_matrix_x(angle_deg: float) -> np.ndarray:
    angle_rad = np.deg2rad(angle_deg)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float64,
    )


def rotation_matrix_y(angle_deg: float) -> np.ndarray:
    angle_rad = np.deg2rad(angle_deg)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )


def rotation_matrix_z(angle_deg: float) -> np.ndarray:
    angle_rad = np.deg2rad(angle_deg)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def euler_xyz_to_rotation(rx_deg: float, ry_deg: float, rz_deg: float, order: str = "xyz") -> np.ndarray:
    order = order.lower()
    matrices = {
        "x": rotation_matrix_x(rx_deg),
        "y": rotation_matrix_y(ry_deg),
        "z": rotation_matrix_z(rz_deg),
    }
    if set(order) != {"x", "y", "z"} or len(order) != 3:
        raise ValueError(f"Unsupported euler order: {order}")

    rotation = np.eye(3, dtype=np.float64)
    for axis in order:
        rotation = matrices[axis] @ rotation
    return rotation


def extrinsic_from_translation_and_euler(
    tx: float,
    ty: float,
    tz: float,
    rx_deg: float,
    ry_deg: float,
    rz_deg: float,
    order: str = "zyx",
) -> np.ndarray:
    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3, :3] = euler_xyz_to_rotation(rx_deg, ry_deg, rz_deg, order=order)
    extrinsic[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
    return extrinsic


def pixel_to_camera(u: float, v: float, depth_mm: float, intrinsic: np.ndarray) -> np.ndarray:
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    z = float(depth_mm)
    x = (float(u) - cx) * z / fx
    y = (float(v) - cy) * z / fy
    return np.array([x, y, z], dtype=np.float64)


def transform_point(point_xyz: np.ndarray, transform_4x4: np.ndarray) -> np.ndarray:
    point_h = np.ones(4, dtype=np.float64)
    point_h[:3] = np.asarray(point_xyz, dtype=np.float64)
    return (np.asarray(transform_4x4, dtype=np.float64) @ point_h)[:3]


def transform_normal(normal_xyz: np.ndarray, transform_4x4: np.ndarray) -> np.ndarray:
    rotation = np.asarray(transform_4x4, dtype=np.float64)[:3, :3]
    normal = rotation @ np.asarray(normal_xyz, dtype=np.float64)
    norm = np.linalg.norm(normal)
    if norm < 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return normal / norm


def transform_direction(direction_xyz: np.ndarray, transform_4x4: np.ndarray) -> np.ndarray:
    rotation = np.asarray(transform_4x4, dtype=np.float64)[:3, :3]
    direction = rotation @ np.asarray(direction_xyz, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return direction / norm


def orient_normal_z_up(normal_xyz: np.ndarray) -> np.ndarray:
    normal = np.asarray(normal_xyz, dtype=np.float64)
    norm = np.linalg.norm(normal)
    if norm < 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    normal = normal / norm
    if normal[2] < 0.0:
        normal = -normal
    return normal


def project_to_tangent(reference_xyz: np.ndarray, normal_xyz: np.ndarray) -> np.ndarray:
    normal = np.asarray(normal_xyz, dtype=np.float64)
    normal_norm = np.linalg.norm(normal)
    if normal_norm < 1e-9:
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        normal = normal / normal_norm

    reference = np.asarray(reference_xyz, dtype=np.float64)
    tangent = reference - np.dot(reference, normal) * normal
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm >= 1e-9:
        return tangent / tangent_norm

    fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(fallback, normal))) > 0.9:
        fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    tangent = fallback - np.dot(fallback, normal) * normal
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm < 1e-9:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return tangent / tangent_norm


def align_vectors_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source = source / max(np.linalg.norm(source), 1e-9)
    target = target / max(np.linalg.norm(target), 1e-9)

    cross = np.cross(source, target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    cross_norm = np.linalg.norm(cross)

    if cross_norm < 1e-9:
        if dot > 0.0:
            return np.eye(3, dtype=np.float64)
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(source[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = np.cross(source, axis)
        axis = axis / max(np.linalg.norm(axis), 1e-9)
        return _axis_angle_to_rotation(axis, np.pi)

    axis = cross / cross_norm
    angle = np.arctan2(cross_norm, dot)
    return _axis_angle_to_rotation(axis, angle)


def _axis_angle_to_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
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


def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if q.size != 4:
        raise ValueError(f"Quaternion must have 4 values, got {q.size}")
    qx, qy, qz, qw = q / max(np.linalg.norm(q), 1e-9)
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def normal_to_quaternion(normal_xyz: np.ndarray) -> np.ndarray:
    approach = -np.asarray(normal_xyz, dtype=np.float64)
    return rotation_matrix_to_quaternion(align_vectors_rotation(np.array([0.0, 0.0, 1.0]), approach))


def approach_and_reference_to_quaternion(approach_xyz: np.ndarray, reference_xyz: np.ndarray) -> np.ndarray:
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
    return rotation_matrix_to_quaternion(rotation)
