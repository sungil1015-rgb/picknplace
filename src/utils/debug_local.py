from __future__ import annotations

import argparse
import json
import logging
import resource
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runner.modes.utils.option_manager import load_option_file
from src.pick_n_place import PickNPlace
from src.utils.depth import median_valid_depth_at_point
from src.utils.geometry import quaternion_to_rotation_matrix
from src.utils.mask import largest_component_mask, mask_center_point
from src.utils.suction_evaluation import normal_z_score
from src.utils.suction_footprint import (
    compute_dual_cup_footprint,
)

def _build_logger() -> logging.Logger:
    logger = logging.getLogger("debug_local")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def _rss_mb() -> float | None:
    status_path = Path("/proc/self/status")
    if not status_path.is_file():
        return None
    for line in status_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2:
                return float(parts[1]) / 1024.0
    return None


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / 1024.0


def _memory_text() -> str:
    rss = _rss_mb()
    rss_text = f"{rss:.1f} MB" if rss is not None else "n/a"
    return f"rss={rss_text}, peak={_peak_rss_mb():.1f} MB"


def _selected_model(option_file: str) -> tuple[dict[str, Any], str]:
    _, _, model_name, model_list, gpu_id, _ = load_option_file(option_file)
    for model_cfg in model_list:
        if model_cfg.get("NAME") == model_name:
            return model_cfg, str(gpu_id)
    raise ValueError(f"Model config not found in {option_file}: {model_name}")


def _load_intrinsic(info_path: Path | None) -> tuple[float, float, float, float]:
    if info_path is None or not info_path.is_file():
        return 590.99, 514.788, 2013.0, 2013.0

    with info_path.open("r", encoding="utf-8") as handle:
        info = json.load(handle)
    camera_info = info.get("camera_info", {})
    return (
        float(camera_info["cal.cx"]),
        float(camera_info["cal.cy"]),
        float(camera_info["cal.fx"]),
        float(camera_info["cal.fy"]),
    )


def _load_normal(path: Path, image_shape: tuple[int, int]) -> np.ndarray | None:
    if not path.is_file():
        return None
    height, width = image_shape
    data = np.fromfile(path, dtype=np.float32)
    expected = height * width * 3
    if data.size < expected:
        raise ValueError(f"Invalid normal.bin size: {path} has {data.size}, expected at least {expected}")
    if data.size > expected:
        data = data[:expected]
    return data.reshape(height, width, 3)


def _sample_dirs(input_dir: Path) -> list[Path]:
    if (input_dir / "rgb.png").is_file():
        return [input_dir]
    return sorted({path.parent for path in input_dir.rglob("rgb.png")})


def _grasp_pixel(prediction: Any, polygon: list[list[int]]) -> tuple[int, int]:
    surface_debug = getattr(prediction, "suction_surface", None)
    if isinstance(surface_debug, dict):
        center_xy = surface_debug.get("surface_center_xy")
        if isinstance(center_xy, list) and len(center_xy) == 2:
            return int(center_xy[0]), int(center_xy[1])

    mask = getattr(prediction, "mask", None)
    if mask is not None and np.any(mask):
        binary = largest_component_mask((mask > 0).astype(np.uint8))
        return mask_center_point(binary)

    points = np.asarray(polygon, dtype=np.float64)
    center = np.round(points.mean(axis=0)).astype(np.int32)
    return int(center[0]), int(center[1])


def _draw_xyz_axes(
    image: np.ndarray,
    anchor: tuple[int, int],
    quaternion_xyzw: Any,
    intrinsic: np.ndarray | None = None,
    extrinsic: np.ndarray | None = None,
    origin_robot_xyz: Any = None,
    scale: int = 36,
) -> None:
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(4)
    rotation = quaternion_to_rotation_matrix(quaternion)
    origin = np.asarray(anchor, dtype=np.int32)

    colors = [(0, 0, 255), (0, 180, 0), (255, 0, 0)]
    labels = ["X", "Y", "Z"]
    for axis, color, label in zip(rotation.T, colors, labels):
        projected = _project_robot_axis(anchor, axis, intrinsic, extrinsic, origin_robot_xyz, scale_mm=float(scale))
        if projected is None:
            direction = np.asarray([axis[0], axis[1]], dtype=np.float64)
            norm = np.linalg.norm(direction)
            if norm < 1e-9:
                continue
            end = origin + np.round(direction / norm * scale).astype(np.int32)
            origin_xy = tuple(origin)
            end_xy = (int(end[0]), int(end[1]))
        else:
            origin_xy, end_xy = projected
        cv2.arrowedLine(image, origin_xy, end_xy, color, 2, cv2.LINE_AA, tipLength=0.25)
        cv2.putText(image, label, end_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)


def _draw_z_axis(
    image: np.ndarray,
    anchor: tuple[int, int],
    quaternion_xyzw: Any,
    intrinsic: np.ndarray | None = None,
    extrinsic: np.ndarray | None = None,
    origin_robot_xyz: Any = None,
    scale: int = 30,
) -> None:
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(4)
    rotation = quaternion_to_rotation_matrix(quaternion)
    z_axis = rotation.T[2]
    projected = _project_robot_axis(anchor, z_axis, intrinsic, extrinsic, origin_robot_xyz, scale_mm=float(scale))
    if projected is None:
        origin = np.asarray(anchor, dtype=np.int32)
        direction = np.asarray([z_axis[0], z_axis[1]], dtype=np.float64)
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            return
        end = origin + np.round(direction / norm * scale).astype(np.int32)
        origin_xy = tuple(origin)
        end_xy = (int(end[0]), int(end[1]))
    else:
        origin_xy, end_xy = projected
    cv2.arrowedLine(image, origin_xy, end_xy, (255, 0, 0), 2, cv2.LINE_AA, tipLength=0.25)
    cv2.putText(image, "Z", end_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 0, 0), 2, cv2.LINE_AA)


def _project_robot_axis(
    anchor: tuple[int, int],
    axis_robot: np.ndarray,
    intrinsic: np.ndarray | None,
    extrinsic: np.ndarray | None,
    origin_robot_xyz: Any,
    scale_mm: float,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if intrinsic is None or extrinsic is None or origin_robot_xyz is None:
        return None
    try:
        origin_robot = np.asarray(origin_robot_xyz, dtype=np.float64).reshape(3)
        axis = np.asarray(axis_robot, dtype=np.float64).reshape(3)
    except (TypeError, ValueError):
        return None
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-9:
        return None

    camera_from_robot = np.linalg.inv(np.asarray(extrinsic, dtype=np.float64))
    origin_camera = _transform_robot_point_to_camera(origin_robot, camera_from_robot)
    end_camera = _transform_robot_point_to_camera(origin_robot + (axis / axis_norm) * scale_mm, camera_from_robot)
    origin_xy = _project_camera_point(origin_camera, intrinsic)
    end_xy = _project_camera_point(end_camera, intrinsic)
    if origin_xy is None or end_xy is None:
        return None
    if np.sum((np.asarray(origin_xy) - np.asarray(anchor)) ** 2) > 30.0 ** 2:
        offset = np.asarray(anchor, dtype=np.int32) - np.asarray(origin_xy, dtype=np.int32)
        end_xy = tuple((np.asarray(end_xy, dtype=np.int32) + offset).tolist())
        origin_xy = (int(anchor[0]), int(anchor[1]))
    return origin_xy, end_xy


def _transform_robot_point_to_camera(point_robot: np.ndarray, camera_from_robot: np.ndarray) -> np.ndarray:
    point_h = np.ones(4, dtype=np.float64)
    point_h[:3] = point_robot
    return (camera_from_robot @ point_h)[:3]


def _project_camera_point(point_camera: np.ndarray, intrinsic: np.ndarray) -> tuple[int, int] | None:
    z = float(point_camera[2])
    if abs(z) < 1e-9:
        return None
    u = (float(point_camera[0]) * float(intrinsic[0, 0]) / z) + float(intrinsic[0, 2])
    v = (float(point_camera[1]) * float(intrinsic[1, 1]) / z) + float(intrinsic[1, 2])
    if not np.isfinite(u) or not np.isfinite(v):
        return None
    return int(round(u)), int(round(v))


def _class_color(class_id: Any) -> tuple[int, int, int]:
    palette = [
        (0, 255, 0),
        (0, 180, 255),
        (255, 80, 80),
        (255, 0, 255),
        (255, 180, 0),
        (0, 255, 255),
        (180, 80, 255),
        (80, 220, 120),
    ]
    if class_id == 5:
        return (160, 160, 160)
    try:
        return palette[int(class_id) % len(palette)]
    except (TypeError, ValueError):
        return palette[0]


def _draw_class4_cap_marker(
    image: np.ndarray,
    prediction: Any,
    color: tuple[int, int, int],
) -> None:
    if _prediction_class_id(prediction) != 4:
        return
    cap_xy = _class4_cap_center_xy(prediction)
    if cap_xy is None:
        return
    x, y = cap_xy
    half = 14
    top_left = (max(0, x - half), max(0, y - half))
    bottom_right = (min(image.shape[1] - 1, x + half), min(image.shape[0] - 1, y + half))
    cv2.rectangle(image, top_left, bottom_right, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.rectangle(image, top_left, bottom_right, color, 2, cv2.LINE_AA)


def _prediction_class_id(prediction: Any) -> int | None:
    if prediction is None:
        return None
    for attr in ("class_index", "label"):
        value = getattr(prediction, attr, None)
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _class4_cap_center_xy(prediction: Any) -> tuple[int, int] | None:
    surface_debug = getattr(prediction, "suction_surface", None)
    if not isinstance(surface_debug, dict):
        return None
    class4_debug = surface_debug.get("class4_bottle")
    if not isinstance(class4_debug, dict):
        fallback = surface_debug.get("class4_bottle_fallback")
        if isinstance(fallback, dict):
            class4_debug = fallback.get("class4_bottle")
    if not isinstance(class4_debug, dict):
        return None
    debug = class4_debug.get("debug")
    if not isinstance(debug, dict):
        return None
    cap_xy = debug.get("cap_center_xy") or debug.get("cap_anchor_xy")
    if not isinstance(cap_xy, list) or len(cap_xy) != 2:
        return None
    try:
        return int(cap_xy[0]), int(cap_xy[1])
    except (TypeError, ValueError):
        return None


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _classification_debug(prediction: Any, object_number: int | None = None) -> dict[str, Any]:
    if prediction is None:
        return {"object_number": object_number} if object_number is not None else {}
    bbox = getattr(prediction, "bbox", None)
    bbox_values = [] if bbox is None else list(np.asarray(bbox).reshape(-1))
    return {
        "object_number": object_number,
        "label": getattr(prediction, "label", None),
        "class_index": getattr(prediction, "class_index", None),
        "class_name": getattr(prediction, "class_name", None),
        "reject_reason": getattr(prediction, "class_reject_reason", None),
        "confidence": _finite_float(getattr(prediction, "class_confidence", None)),
        "similarity": _finite_float(getattr(prediction, "class_similarity", None)),
        "vote_ratio": _finite_float(getattr(prediction, "class_vote_ratio", None)),
        "margin": _finite_float(getattr(prediction, "class_margin", None)),
        "neighbor_indices": list(getattr(prediction, "class_neighbor_indices", []) or []),
        "neighbor_labels": list(getattr(prediction, "class_neighbor_labels", []) or []),
        "neighbor_similarities": [
            _finite_float(value)
            for value in (getattr(prediction, "class_neighbor_similarities", []) or [])
        ],
        "detector_score": _finite_float(getattr(prediction, "score", None)),
        "bbox": [_finite_float(value) for value in bbox_values],
    }


def _priority_debug(prediction: Any) -> dict[str, Any]:
    if prediction is None:
        return {}
    footprint = getattr(prediction, "suction_footprint", None)
    return {
        "priority": _finite_float(getattr(prediction, "grasp_priority", None)),
        "depth_score": _finite_float(getattr(prediction, "grasp_depth_score", None)),
        "grasp_depth": _finite_float(getattr(prediction, "grasp_depth", None)),
        "grasp_xy": getattr(prediction, "grasp_xy", None),
        "class_similarity": _finite_float(getattr(prediction, "grasp_class_similarity", None)),
        "class_reject_reason": getattr(prediction, "grasp_class_reject_reason", None),
        "valid_depth": bool(getattr(prediction, "grasp_valid_depth", False)),
        "mask_area": getattr(prediction, "grasp_mask_area", None),
        "suction_normal_z_score": _finite_float(getattr(prediction, "suction_normal_z_score", None)),
        "suction_footprint": footprint if isinstance(footprint, dict) else None,
        "suction_candidates": list(getattr(prediction, "suction_candidates", []) or []),
        "suction_surface": getattr(prediction, "suction_surface", None),
    }


def _draw_grasp_target_label(
    image: np.ndarray,
    points: np.ndarray,
    prediction: Any,
    color: tuple[int, int, int],
) -> None:
    priority = _finite_float(getattr(prediction, "grasp_priority", None)) if prediction is not None else None
    depth = _finite_float(getattr(prediction, "grasp_depth", None)) if prediction is not None else None
    text = "GRASP #1"
    if priority is not None:
        text += f" P:{priority:.2f}"
    if depth is not None:
        text += f" D:{depth:.0f}"

    origin = np.maximum(points.min(axis=0) + np.array([0, -12]), np.array([0, 24])).astype(int)
    origin_xy = (int(origin[0]), int(origin[1]))
    cv2.putText(image, text, origin_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 5, cv2.LINE_AA)
    cv2.putText(image, text, origin_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)


def _draw_object_number(
    image: np.ndarray,
    points: np.ndarray,
    object_number: int,
    color: tuple[int, int, int],
) -> None:
    origin = np.maximum(points.min(axis=0) + np.array([4, 24]), np.array([4, 24])).astype(int)
    origin_xy = (int(origin[0]), int(origin[1]))
    text = f"#{object_number}"
    cv2.putText(image, text, origin_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 5, cv2.LINE_AA)
    cv2.putText(image, text, origin_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)


def _local_depth_mm(
    depth_image: np.ndarray | None,
    grasp_xy: tuple[int, int],
    mask: np.ndarray | None,
    window: int,
) -> float | None:
    u, v = grasp_xy
    return median_valid_depth_at_point(depth_image, u, v, mask, window=window)


def _draw_dual_cup_footprint(
    image: np.ndarray,
    grasp_xy: tuple[int, int],
    prediction: Any,
    depth_image: np.ndarray | None,
    intrinsic: np.ndarray | None,
    color: tuple[int, int, int],
    suction_pipeline: Any | None,
) -> dict[str, Any]:
    stored_footprint = getattr(prediction, "suction_footprint", None) if prediction is not None else None
    if isinstance(stored_footprint, dict):
        radius_px = _finite_float(stored_footprint.get("cup_radius_px"))
        cup_centers_xy = stored_footprint.get("cup_centers_xy")
        if radius_px is not None and isinstance(cup_centers_xy, list) and len(cup_centers_xy) == 2:
            radius_int = max(1, int(round(radius_px)))
            center_points = [
                (int(round(float(point[0]))), int(round(float(point[1]))))
                for point in cup_centers_xy
                if isinstance(point, list) and len(point) == 2
            ]
            if len(center_points) == 2:
                cv2.line(image, center_points[0], center_points[1], (0, 255, 0), 2, cv2.LINE_AA)
                for point in center_points:
                    cv2.circle(image, point, radius_int, (255, 255, 255), 4, cv2.LINE_AA)
                    cv2.circle(image, point, radius_int, color, 2, cv2.LINE_AA)
                    cv2.circle(image, point, 3, color, -1, cv2.LINE_AA)

                data = dict(stored_footprint)
                data.update({
                    "drawn": True,
                    "source": "pipeline_suction_footprint",
                    "cup_centers_xy": [[int(x), int(y)] for x, y in center_points],
                })
                return data

    mask = getattr(prediction, "mask", None) if prediction is not None else None
    depth_window = int(getattr(suction_pipeline, "depth_window", 7))
    depth_mm = _local_depth_mm(depth_image, grasp_xy, mask, window=depth_window)
    if depth_mm is None or depth_mm <= 0 or intrinsic is None:
        return {"drawn": False, "reason": "missing_depth_or_intrinsic"}

    cup_diameter_mm = float(getattr(suction_pipeline, "cup_diameter_mm", 25.0))
    cup_center_spacing_mm = float(getattr(suction_pipeline, "cup_center_spacing_mm", 35.0))
    min_cup_inside_ratio = float(getattr(suction_pipeline, "min_cup_inside_ratio", 0.85))
    footprint = compute_dual_cup_footprint(
        mask if mask is not None else np.zeros(image.shape[:2], dtype=np.uint8),
        grasp_xy,
        depth_mm,
        intrinsic,
        cup_diameter_mm=cup_diameter_mm,
        cup_center_spacing_mm=cup_center_spacing_mm,
        min_cup_inside_ratio=min_cup_inside_ratio,
    )
    if footprint is None:
        return {"drawn": False, "reason": "invalid_projection"}

    radius_int = max(1, int(round(footprint.cup_radius_px)))
    center_points = [
        tuple(np.round(point).astype(int).tolist())
        for point in footprint.cup_centers_xy
    ]

    cv2.line(image, center_points[0], center_points[1], (0, 255, 0), 2, cv2.LINE_AA)
    for point in center_points:
        cv2.circle(image, point, radius_int, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.circle(image, point, radius_int, color, 2, cv2.LINE_AA)
        cv2.circle(image, point, 3, color, -1, cv2.LINE_AA)

    data = footprint.to_dict()
    data.update({
        "drawn": True,
        "cup_centers_xy": [[int(x), int(y)] for x, y in center_points],
    })
    return data


def _render_debug(
    image: np.ndarray,
    depth_image: np.ndarray | None,
    intrinsic: np.ndarray | None,
    extrinsic: np.ndarray | None,
    result: dict[str, Any],
    predictions: list[Any],
    suction_pipeline: Any | None,
    suction_debug_top_k: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    overlay = image.copy()
    polygons = result.get("result_data", [])
    class_ids = result.get("class_id", [])
    suction_points = result.get("suction_points", [])
    summaries: list[dict[str, Any]] = []

    for index, polygon in enumerate(polygons):
        object_number = index + 1
        points = np.asarray(polygon, dtype=np.int32)
        if points.ndim != 2 or points.shape[0] < 3:
            continue

        class_id = class_ids[index] if index < len(class_ids) else None
        is_grasp_target = index == 0
        is_suction_debug_target = index < suction_debug_top_k
        color = (0, 0, 255) if is_grasp_target else _class_color(class_id)
        thickness = 4 if is_grasp_target else 2
        cv2.polylines(overlay, [points], True, color, thickness, cv2.LINE_AA)
        _draw_object_number(overlay, points, object_number, color)

        prediction = predictions[index] if index < len(predictions) else None
        grasp_xy = _grasp_pixel(prediction, polygon) if prediction is not None else _grasp_pixel(None, polygon)
        point_list = suction_points[index] if index < len(suction_points) else []
        has_suction_point = bool(point_list)
        footprint_debug = {}
        if is_suction_debug_target and has_suction_point:
            cv2.circle(overlay, grasp_xy, 8 if is_grasp_target else 6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.drawMarker(
                overlay,
                grasp_xy,
                color,
                markerType=cv2.MARKER_CROSS,
                markerSize=20 if is_grasp_target else 16,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
            footprint_debug = _draw_dual_cup_footprint(
                overlay,
                grasp_xy,
                prediction,
                depth_image,
                intrinsic,
                color,
                suction_pipeline,
            )
            if is_grasp_target:
                _draw_grasp_target_label(overlay, points, prediction, color)
        else:
            cv2.circle(overlay, grasp_xy, 8, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(overlay, grasp_xy, 5, color, -1, cv2.LINE_AA)

        grasp_xyz = None
        if has_suction_point:
            point = point_list[0]
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                grasp_xyz = point[0]
                if is_suction_debug_target:
                    _draw_z_axis(overlay, grasp_xy, point[1], intrinsic=intrinsic, extrinsic=extrinsic, origin_robot_xyz=grasp_xyz)

        _draw_class4_cap_marker(overlay, prediction, color)

        summaries.append(
            {
                "rank": index + 1,
                "object_number": object_number,
                "is_grasp_target": is_grasp_target,
                "is_suction_debug_target": is_suction_debug_target,
                "class_id": class_id,
                "polygon": [[int(x), int(y)] for x, y in points.tolist()],
                "grasp_xy": [int(grasp_xy[0]), int(grasp_xy[1])],
                "grasp_xyz": [float(value) for value in grasp_xyz[:3]] if grasp_xyz is not None else None,
                "footprint": footprint_debug,
                "priority": _priority_debug(prediction),
                "classification": _classification_debug(prediction, object_number),
            }
        )

    return overlay, summaries


def _save_debug_result(
    sample_dir: Path,
    output_root: Path,
    image: np.ndarray,
    depth_image: np.ndarray | None,
    intrinsic: np.ndarray | None,
    extrinsic: np.ndarray | None,
    result: dict[str, Any],
    predictions: list[Any],
    suction_pipeline: Any | None,
    suction_debug_top_k: int,
) -> None:
    relative_name = "_".join(sample_dir.parts[-2:]) if len(sample_dir.parts) >= 2 else sample_dir.name
    overlay, summaries = _render_debug(
        image,
        depth_image,
        intrinsic,
        extrinsic,
        result,
        predictions,
        suction_pipeline,
        suction_debug_top_k,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    image_path = output_root / f"{relative_name}_debug.png"
    json_path = output_root / f"{relative_name}_result.json"

    cv2.imwrite(str(image_path), overlay)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "sample": str(sample_dir),
                "state": result.get("state"),
                "objects": summaries,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )


def _compute_debug_suction_top_k(
    model: PickNPlace,
    result: dict[str, Any],
    predictions: list[Any],
    depth_image: np.ndarray | None,
    normal_image: np.ndarray | None,
    top_k: int,
) -> None:
    if top_k <= 0 or not predictions:
        result["suction_points"] = [[] for _ in predictions]
        result["pts_per_object"] = [0 for _ in predictions]
        return

    for prediction in predictions:
        prediction.suction_footprint = None
        prediction.suction_candidates = []
        prediction.suction_surface = None
        prediction.suction_normal_z_score = None

    target_count = min(int(top_k), len(predictions))
    suction_points = [[] for _ in predictions]
    computed_points = model.suction_pipeline.compute(
        predictions[:target_count],
        depth_image,
        normal_image,
        model.c_matrix,
        model.extrinsic,
    )
    for index, point_list in enumerate(computed_points):
        suction_points[index] = point_list
        surface_debug = getattr(predictions[index], "suction_surface", None)
        predictions[index].suction_normal_z_score = normal_z_score(surface_debug) if isinstance(surface_debug, dict) else None

    result["suction_points"] = suction_points
    result["pts_per_object"] = [len(points) for points in suction_points]


def run_debug(args: argparse.Namespace) -> None:
    logger = _build_logger()
    model_cfg, gpu_id = _selected_model(args.option)
    cuda = args.cuda if args.cuda is not None else gpu_id

    options = dict(model_cfg.get("OPTIONS", {}))
    model = PickNPlace(
        logger=logger,
        config_name=str(model_cfg.get("CONFIG", "")),
        checkpoint=str(model_cfg.get("WEIGHT", "")),
        options=options,
        cuda=str(cuda),
    )

    cx, cy, fx, fy = _load_intrinsic(Path(args.info) if args.info else None)
    model.set_intrinsic(cx, cy, fx, fy)

    sample_dirs = _sample_dirs(Path(args.input))
    if args.limit is not None:
        sample_dirs = sample_dirs[: args.limit]

    logger.info(f"debug samples: {len(sample_dirs)}")
    for sample_dir in sample_dirs:
        sample_start = time.perf_counter()
        rgb_path = sample_dir / "rgb.png"
        depth_path = sample_dir / "depth.png"
        normal_path = sample_dir / "normal.bin"

        load_start = time.perf_counter()
        image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {rgb_path}")
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED) if depth_path.is_file() else None
        normal = _load_normal(normal_path, image.shape[:2])
        load_elapsed = time.perf_counter() - load_start

        run_start = time.perf_counter()
        result, predictions = model.run(
            image,
            depth_image=depth,
            normal_image=normal,
            compute_suction_pts=False,
            roi_2d=list(getattr(model, "segmentation_roi", [])),
        )
        _compute_debug_suction_top_k(
            model,
            result,
            predictions,
            depth,
            normal,
            top_k=args.suction_top_k,
        )
        run_elapsed = time.perf_counter() - run_start

        save_start = time.perf_counter()
        _save_debug_result(
            sample_dir,
            Path(args.output),
            image,
            depth,
            model.c_matrix,
            model.extrinsic,
            result,
            predictions,
            model.suction_pipeline,
            args.suction_top_k,
        )
        save_elapsed = time.perf_counter() - save_start
        total_elapsed = time.perf_counter() - sample_start

        logger.info(
            "saved: %s | time load=%.3fs run=%.3fs save=%.3fs total=%.3fs | %s",
            sample_dir,
            load_elapsed,
            run_elapsed,
            save_elapsed,
            total_elapsed,
            _memory_text(),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local PickNPlace debug images.")
    parser.add_argument("--input", default="debug_img", help="debug image root or one sample directory")
    parser.add_argument("--output", default="debug_result", help="debug result output directory")
    parser.add_argument("--option", default="inference.opt", help="inference option file")
    parser.add_argument("--info", default="img/info.json", help="camera info json for intrinsic values")
    parser.add_argument("--cuda", default=None, help="GPU id, or -1/cpu for CPU")
    parser.add_argument("--limit", type=int, default=None, help="process only the first N samples")
    parser.add_argument("--suction-top-k", type=int, default=5, help="compute and draw suction for the top N priority objects")
    return parser.parse_args()


if __name__ == "__main__":
    run_debug(parse_args())
