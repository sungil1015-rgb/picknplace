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
from src.utils.crop import DEFAULT_SEGMENTATION_ROI_2013
from src.utils.depth import median_valid_depth_at_point
from src.utils.geometry import quaternion_to_rotation_matrix
from src.utils.mask import largest_component_mask, mask_center_point
from src.utils.suction_footprint import (
    DEFAULT_CUP_CENTER_SPACING_MM,
    DEFAULT_CUP_DIAMETER_MM,
    compute_dual_cup_footprint,
)


DEBUG_ROI_2013 = DEFAULT_SEGMENTATION_ROI_2013
CUP_DIAMETER_MM = DEFAULT_CUP_DIAMETER_MM
CUP_CENTER_SPACING_MM = DEFAULT_CUP_CENTER_SPACING_MM


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
    scale: int = 36,
) -> None:
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(4)
    rotation = quaternion_to_rotation_matrix(quaternion)
    origin = np.asarray(anchor, dtype=np.int32)

    colors = [(0, 0, 255), (0, 180, 0), (255, 0, 0)]
    labels = ["X", "Y", "Z"]
    for axis, color, label in zip(rotation.T, colors, labels):
        direction = np.asarray([axis[0], axis[1]], dtype=np.float64)
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            continue
        end = origin + np.round(direction / norm * scale).astype(np.int32)
        end_xy = (int(end[0]), int(end[1]))
        cv2.arrowedLine(image, tuple(origin), end_xy, color, 2, cv2.LINE_AA, tipLength=0.25)
        cv2.putText(image, label, end_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)


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


def _draw_class_label(
    image: np.ndarray,
    class_id: Any,
    xy: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    x = int(max(0, xy[0]))
    y = int(max(18, xy[1]))
    text = f"class {class_id}"
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _classification_debug(prediction: Any) -> dict[str, Any]:
    if prediction is None:
        return {}
    bbox = getattr(prediction, "bbox", None)
    bbox_values = [] if bbox is None else list(np.asarray(bbox).reshape(-1))
    return {
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
        "support_score": _finite_float(getattr(prediction, "grasp_support_score", None)),
        "depth_score": _finite_float(getattr(prediction, "grasp_depth_score", None)),
        "center_score": _finite_float(getattr(prediction, "grasp_center_score", None)),
        "isolation_score": _finite_float(getattr(prediction, "grasp_isolation_score", None)),
        "clearance_score": _finite_float(getattr(prediction, "grasp_clearance_score", None)),
        "clearance_distance": _finite_float(getattr(prediction, "grasp_clearance_distance", None)),
        "area_score": _finite_float(getattr(prediction, "grasp_area_score", None)),
        "mask_area": getattr(prediction, "grasp_mask_area", None),
        "object_depth": _finite_float(getattr(prediction, "grasp_object_depth", None)),
        "center_distance": _finite_float(getattr(prediction, "grasp_center_distance", None)),
        "depth_candidate": bool(getattr(prediction, "grasp_depth_candidate", False)),
        "valid_depth": bool(getattr(prediction, "grasp_valid_depth", False)),
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
    depth = _finite_float(getattr(prediction, "grasp_object_depth", None)) if prediction is not None else None
    text = "GRASP #1"
    if priority is not None:
        text += f" P:{priority:.2f}"
    if depth is not None:
        text += f" D:{depth:.0f}"

    origin = np.maximum(points.min(axis=0) + np.array([0, -12]), np.array([0, 24])).astype(int)
    origin_xy = (int(origin[0]), int(origin[1]))
    cv2.putText(image, text, origin_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 5, cv2.LINE_AA)
    cv2.putText(image, text, origin_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)


def _local_depth_mm(
    depth_image: np.ndarray | None,
    grasp_xy: tuple[int, int],
    mask: np.ndarray | None,
    window: int = 7,
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
                cv2.line(image, center_points[0], center_points[1], color, 2, cv2.LINE_AA)
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
    depth_mm = _local_depth_mm(depth_image, grasp_xy, mask)
    if depth_mm is None or depth_mm <= 0 or intrinsic is None:
        return {"drawn": False, "reason": "missing_depth_or_intrinsic"}

    footprint = compute_dual_cup_footprint(
        mask if mask is not None else np.zeros(image.shape[:2], dtype=np.uint8),
        grasp_xy,
        depth_mm,
        intrinsic,
        cup_diameter_mm=CUP_DIAMETER_MM,
        cup_center_spacing_mm=CUP_CENTER_SPACING_MM,
    )
    if footprint is None:
        return {"drawn": False, "reason": "invalid_projection"}

    radius_int = max(1, int(round(footprint.cup_radius_px)))
    center_points = [
        tuple(np.round(point).astype(int).tolist())
        for point in footprint.cup_centers_xy
    ]

    cv2.line(image, center_points[0], center_points[1], color, 2, cv2.LINE_AA)
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
    result: dict[str, Any],
    predictions: list[Any],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    overlay = image.copy()
    polygons = result.get("result_data", [])
    class_ids = result.get("class_id", [])
    suction_points = result.get("suction_points", [])
    summaries: list[dict[str, Any]] = []

    for index, polygon in enumerate(polygons):
        points = np.asarray(polygon, dtype=np.int32)
        if points.ndim != 2 or points.shape[0] < 3:
            continue

        class_id = class_ids[index] if index < len(class_ids) else None
        is_grasp_target = index == 0
        color = (0, 0, 255) if is_grasp_target else _class_color(class_id)
        thickness = 5 if is_grasp_target else 3
        cv2.polylines(overlay, [points], True, color, thickness, cv2.LINE_AA)

        prediction = predictions[index] if index < len(predictions) else None
        grasp_xy = _grasp_pixel(prediction, polygon) if prediction is not None else _grasp_pixel(None, polygon)
        footprint_debug = {}
        if is_grasp_target:
            cv2.circle(overlay, grasp_xy, 15, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.drawMarker(
                overlay,
                grasp_xy,
                color,
                markerType=cv2.MARKER_CROSS,
                markerSize=34,
                thickness=4,
                line_type=cv2.LINE_AA,
            )
            footprint_debug = _draw_dual_cup_footprint(
                overlay,
                grasp_xy,
                prediction,
                depth_image,
                intrinsic,
                color,
            )
            _draw_grasp_target_label(overlay, points, prediction, color)
        else:
            cv2.circle(overlay, grasp_xy, 8, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(overlay, grasp_xy, 5, color, -1, cv2.LINE_AA)

        grasp_xyz = None
        point_list = suction_points[index] if index < len(suction_points) else []
        if point_list:
            point = point_list[0]
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                grasp_xyz = point[0]
                _draw_xyz_axes(overlay, grasp_xy, point[1])

        if not is_grasp_target:
            _draw_class_label(overlay, class_id, tuple(points[0].tolist()), color)

        summaries.append(
            {
                "rank": index + 1,
                "is_grasp_target": is_grasp_target,
                "class_id": class_id,
                "polygon": [[int(x), int(y)] for x, y in points.tolist()],
                "grasp_xy": [int(grasp_xy[0]), int(grasp_xy[1])],
                "grasp_xyz": [float(value) for value in grasp_xyz[:3]] if grasp_xyz is not None else None,
                "footprint": footprint_debug,
                "priority": _priority_debug(prediction),
                "classification": _classification_debug(prediction),
            }
        )

    return overlay, summaries


def _save_debug_result(
    sample_dir: Path,
    output_root: Path,
    image: np.ndarray,
    depth_image: np.ndarray | None,
    intrinsic: np.ndarray | None,
    result: dict[str, Any],
    predictions: list[Any],
) -> None:
    relative_name = "_".join(sample_dir.parts[-2:]) if len(sample_dir.parts) >= 2 else sample_dir.name
    overlay, summaries = _render_debug(image, depth_image, intrinsic, result, predictions)

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
            compute_suction_pts=True,
            roi_2d=DEBUG_ROI_2013,
        )
        run_elapsed = time.perf_counter() - run_start

        save_start = time.perf_counter()
        _save_debug_result(sample_dir, Path(args.output), image, depth, model.c_matrix, result, predictions)
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
    return parser.parse_args()


if __name__ == "__main__":
    run_debug(parse_args())
