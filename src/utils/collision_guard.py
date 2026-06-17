from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from src.utils.geometry import pixel_to_camera, transform_direction


@dataclass(frozen=True)
class CollisionGuardConfig:
    enabled: bool
    box_roi: tuple[float, float, float, float]
    wall_margin_ratio: float
    min_outward_wall_component: float


def guarded_normal_near_box_wall(
    normal_robot: np.ndarray,
    *,
    u: int,
    v: int,
    depth_mm: float,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    config: CollisionGuardConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    debug: dict[str, Any] = {
        "enabled": bool(config.enabled),
        "applied": False,
        "reason": None,
        "box_roi": [float(value) for value in config.box_roi],
        "wall_margin_ratio": float(config.wall_margin_ratio),
        "min_outward_wall_component": float(config.min_outward_wall_component),
    }
    normal = _unit(normal_robot)
    if not config.enabled:
        return normal, {**debug, "reason": "disabled"}

    wall_xy, wall_debug = _nearest_wall_direction(u, v, config.box_roi, config.wall_margin_ratio)
    debug.update(wall_debug)
    if wall_xy is None:
        return normal, {**debug, "reason": "not_near_box_wall"}

    wall_camera = _image_direction_to_camera(u, v, wall_xy, depth_mm, intrinsic)
    wall_robot = _unit(transform_direction(wall_camera, extrinsic))
    component = float(np.dot(normal, wall_robot))
    debug.update(
        {
            "wall_direction_camera": [float(value) for value in wall_camera],
            "wall_direction_robot": [float(value) for value in wall_robot],
            "outward_wall_component": component,
            "normal_robot_before": [float(value) for value in normal],
        }
    )
    if component >= float(config.min_outward_wall_component):
        return normal, {**debug, "reason": "normal_already_outward"}

    target_component = float(np.clip(config.min_outward_wall_component, 0.0, 0.95))
    tangent = normal - component * wall_robot
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm < 1e-9:
        adjusted = wall_robot
    else:
        outward_scale = target_component * tangent_norm / max(float(np.sqrt(1.0 - target_component**2)), 1e-9)
        adjusted = tangent + outward_scale * wall_robot
    adjusted = _unit(adjusted)
    return adjusted, {
        **debug,
        "applied": True,
        "reason": "added_outward_wall_component",
        "normal_robot_after": [float(value) for value in adjusted],
    }


def _nearest_wall_direction(
    u: int,
    v: int,
    box_roi: Sequence[float],
    margin_ratio: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    left, right, top, bottom = [float(value) for value in box_roi]
    width = max(right - left, 1.0)
    height = max(bottom - top, 1.0)
    margin_x = width * float(margin_ratio)
    margin_y = height * float(margin_ratio)
    distances = {
        "left": float(u - left),
        "right": float(right - u),
        "top": float(v - top),
        "bottom": float(bottom - v),
    }
    near: list[tuple[str, np.ndarray]] = []
    if distances["left"] <= margin_x:
        near.append(("left", np.array([-1.0, 0.0], dtype=np.float64)))
    if distances["right"] <= margin_x:
        near.append(("right", np.array([1.0, 0.0], dtype=np.float64)))
    if distances["top"] <= margin_y:
        near.append(("top", np.array([0.0, -1.0], dtype=np.float64)))
    if distances["bottom"] <= margin_y:
        near.append(("bottom", np.array([0.0, 1.0], dtype=np.float64)))

    debug: dict[str, Any] = {
        "wall_margin_px": [float(margin_x), float(margin_y)],
        "wall_distances_px": distances,
        "near_walls": [name for name, _ in near],
    }
    if not near:
        return None, debug

    direction = _unit(np.sum([item[1] for item in near], axis=0))
    return direction, {**debug, "wall_direction_xy": [float(direction[0]), float(direction[1])]}


def _image_direction_to_camera(
    u: int,
    v: int,
    direction_xy: np.ndarray,
    depth_mm: float,
    intrinsic: np.ndarray,
) -> np.ndarray:
    center = pixel_to_camera(float(u), float(v), float(depth_mm), intrinsic)
    shifted = pixel_to_camera(float(u) + float(direction_xy[0]), float(v) + float(direction_xy[1]), float(depth_mm), intrinsic)
    return _unit(shifted - center)


def _unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(value))
    if norm < 1e-9:
        if value.size == 2:
            return np.array([1.0, 0.0], dtype=np.float64)
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return value / norm
