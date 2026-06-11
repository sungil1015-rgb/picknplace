"""
PickNPlace 추론 파이프라인 (학생 구현 대상).

grpc_mode 가 이 클래스를 다음 4개 API 로만 호출한다.
더미 구현이 들어 있으므로 서버를 띄우면 바로 동작을 확인할 수 있다.
TODO 주석이 달린 부분을 본인 모델 / 파이프라인으로 교체하면 된다.

    1) PickNPlace(logger, config, weight, options, cuda)   - 생성자
    2) set_intrinsic(cx, cy, fx, fy)                        - 카메라 intrinsic 적용
    3) run(rgb, depth, normal, ...)                         - 추론 메인
    4) save_result(rgb, predictions, polygons, ...)         - 시각화 결과 저장

run() 의 반환 형태와 reply JSON 스키마는 README.md 의 "응답 형식" 절 참고.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import logging
import os
from pathlib import Path

import cv2
import numpy as np

from src.pipeline.grasp_priority import GraspPriorityScorer
from src.pipeline.suction_pipeline import SuctionPipeline
from src.utils.geometry import extrinsic_from_translation_and_euler, make_intrinsic, quaternion_to_rotation_matrix
from src.utils.mask import largest_component_mask, mask_to_polygon
from src.utils.suction_config import SuctionConfig, as_bool
from src.utils.suction_evaluation import normal_z_score


def _load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"PickNPlace config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"PickNPlace config must be a mapping: {config_path}")
    return data


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"PickNPlace config section '{name}' is required")
    return value


def _required(mapping: dict[str, Any], key: str, section_name: str) -> Any:
    if key not in mapping:
        raise ValueError(f"PickNPlace config '{section_name}.{key}' is required")
    return mapping[key]


def _class_float_mapping(mapping: Any) -> dict[int, float]:
    if not isinstance(mapping, dict):
        return {}
    parsed: dict[int, float] = {}
    for key, value in mapping.items():
        try:
            parsed[int(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return parsed


class PickNPlace:
    def __init__(
            self,
            logger: logging.Logger,
            config_name: str,
            checkpoint: str,
            options: Dict[str, Any],
            cuda: str = "0",
    ) -> None:
        self.logger = logger
        self.name = "PickNPlace"
        self.options = options
        self.config_name = config_name
        self.checkpoint = checkpoint
        self.cuda = cuda
        self.config = _load_yaml(config_name)

        pipeline_cfg = _section(self.config, "pipeline")
        polygon_cfg = _section(self.config, "polygon")
        self.fixed_focal_length = float(_required(pipeline_cfg, "fixed_focal_length", "pipeline"))
        self.use_fixed_focal_length = as_bool(pipeline_cfg.get("use_fixed_focal_length", True))
        self.segmentation_roi = [float(value) for value in _required(pipeline_cfg, "segmentation_roi", "pipeline")]
        if len(self.segmentation_roi) != 4:
            raise ValueError("PickNPlace config 'pipeline.segmentation_roi' must contain 4 values")
        self.roi_source = str(pipeline_cfg.get("roi_source", "config")).lower()
        if self.roi_source not in ("config", "input"):
            raise ValueError("PickNPlace config 'pipeline.roi_source' must be 'config' or 'input'")
        self.polygon_approx_ratio = float(_required(polygon_cfg, "approx_ratio", "polygon"))
        self.polygon_min_epsilon = float(_required(polygon_cfg, "min_epsilon", "polygon"))

        self.detector_cfg = _section(self.config, "detector")
        self.classifier_cfg = _section(self.config, "classifier")
        self.priority_cfg = _section(self.config, "priority")
        self.calibration_cfg = _section(self.config, "calibration")
        self.suction_cfg = _section(self.config, "suction")

        self.c_matrix = np.eye(3, dtype=np.float64)

        self.extrinsic = self._load_extrinsic()

        self.detector = self._load_detector()
        self.classifier = self._load_classifier()
        self.suction_pipeline = self._load_suction_pipeline()
        self.priority_scorer = self._load_priority_scorer()
        if self.detector is None:
            self.logger.info(f"[{self.name}] 더미 모드로 초기화 (detector=None)")
        self.logger.info(f"[{self.name}] config={config_name}, weight={checkpoint}")

    def _device(self) -> str:
        return f"cuda:{self.cuda}" if str(self.cuda).lower() not in ("", "cpu", "-1") else "cpu"

    def _load_detector(self):
        detector_type = str(_required(self.detector_cfg, "type", "detector")).lower()
        if detector_type != "mask2former":
            return None

        try:
            from src.detector.mask2former_detector import Mask2FormerDetector

            device = self._device()
            detector = Mask2FormerDetector(
                config_file=str(_required(self.detector_cfg, "config", "detector")),
                checkpoint_file=self.checkpoint,
                device=device,
            )
            self.logger.info(f"[{self.name}] Mask2FormerDetector 로드 완료: device={device}")
            return detector
        except Exception as exc:
            self.logger.exception(f"[{self.name}] Mask2FormerDetector 로드 실패: {exc}")
            raise

    def _load_classifier(self):
        classifier_name = str(_required(self.classifier_cfg, "type", "classifier")).lower()
        classifier_config = str(_required(self.classifier_cfg, "config", "classifier"))
        if classifier_name not in ("dinov2", "dino_v2") or not classifier_config:
            return None
        try:
            config = _load_yaml(classifier_config)
            classifier_cfg = config.get("classifier", {})
            if not isinstance(classifier_cfg, dict) or not bool(classifier_cfg.get("enabled", False)):
                return None

            from src.classifier import DinoV2KnnClassifier

            classifier = DinoV2KnnClassifier(config_file=classifier_config, device=self._device())
            self.logger.info(f"[{self.name}] DinoV2KnnClassifier 로드 완료: device={self._device()}")
            return classifier
        except Exception as exc:
            self.logger.exception(f"[{self.name}] DinoV2KnnClassifier 로드 실패: {exc}")
            raise

    def _load_priority_scorer(self) -> GraspPriorityScorer:
        priority_config = str(_required(self.priority_cfg, "config", "priority"))
        try:
            config = _load_yaml(priority_config)
            priority_cfg = config.get("priority", {})
            if not isinstance(priority_cfg, dict):
                raise ValueError("priority config section 'priority' must be a mapping")
            scorer = GraspPriorityScorer(
                depth_weight=float(_required(priority_cfg, "depth_weight", "priority")),
                center_weight=float(_required(priority_cfg, "center_weight", "priority")),
                isolation_weight=float(_required(priority_cfg, "clearance_weight", "priority")),
                area_weight=float(_required(priority_cfg, "area_weight", "priority")),
                mode=str(_required(priority_cfg, "mode", "priority")),
                depth_candidate_score_margin=float(_required(priority_cfg, "depth_candidate_score_margin", "priority")),
                depth_percentile=float(_required(priority_cfg, "depth_percentile", "priority")),
                depth_percentile_by_class=_class_float_mapping(priority_cfg.get("depth_percentile_by_class", {})),
                erosion_kernel=int(_required(priority_cfg, "erosion_kernel", "priority")),
                clearance_max_distance=float(_required(priority_cfg, "clearance_max_distance", "priority")),
            )
            self.logger.info(f"[{self.name}] GraspPriorityScorer 로드 완료: {priority_config}")
            return scorer
        except Exception as exc:
            self.logger.exception(f"[{self.name}] GraspPriorityScorer 로드 실패: {exc}")
            raise

    def _load_suction_pipeline(self) -> SuctionPipeline:
        suction_config = str(_required(self.suction_cfg, "config", "suction"))
        try:
            config = _load_yaml(suction_config)
            suction_cfg = config.get("suction", {})
            if not isinstance(suction_cfg, dict):
                raise ValueError("suction config section 'suction' must be a mapping")
            self.logger.info(f"[{self.name}] SuctionPipeline config 로드 완료: {suction_config}")
        except Exception as exc:
            self.logger.exception(f"[{self.name}] SuctionPipeline config 로드 실패: {exc}")
            raise

        return SuctionPipeline(SuctionConfig.from_mapping(suction_cfg))

    def _load_extrinsic(self) -> np.ndarray:
        calibration_config = str(_required(self.calibration_cfg, "config", "calibration"))

        try:
            config = _load_yaml(calibration_config)
            extrinsic_cfg = config.get("extrinsic", {})
            matrix = extrinsic_cfg.get("robot_from_camera")
            if matrix is not None:
                extrinsic = np.asarray(matrix, dtype=np.float64)
                if extrinsic.shape != (4, 4):
                    raise ValueError(f"robot_from_camera must be 4x4, got {extrinsic.shape}")
                if not np.allclose(extrinsic[3], np.array([0.0, 0.0, 0.0, 1.0])):
                    raise ValueError("robot_from_camera last row must be [0, 0, 0, 1]")
            else:
                required_keys = ["cal_x", "cal_y", "cal_z", "cal_rx", "cal_ry", "cal_rz"]
                missing_keys = [key for key in required_keys if key not in extrinsic_cfg]
                if missing_keys:
                    raise ValueError(f"extrinsic config missing keys: {missing_keys}")
                rotation_order = str(extrinsic_cfg.get("rotation_order", "xyz"))
                extrinsic = extrinsic_from_translation_and_euler(
                    float(extrinsic_cfg["cal_x"]),
                    float(extrinsic_cfg["cal_y"]),
                    float(extrinsic_cfg["cal_z"]),
                    float(extrinsic_cfg["cal_rx"]),
                    float(extrinsic_cfg["cal_ry"]),
                    float(extrinsic_cfg["cal_rz"]),
                    order=rotation_order,
                )
            self.logger.info(f"[{self.name}] extrinsic 로드 완료: {calibration_config}")
            return extrinsic
        except Exception as exc:
            self.logger.exception(f"[{self.name}] extrinsic 로드 실패: {exc}")
            raise

    # ─── 카메라 intrinsic ───────────────────────────────────────────────

    def set_intrinsic(self, cx: float, cy: float, fx: float, fy: float) -> None:
        """카메라 intrinsic을 적용한다. config에서 고정 focal length 정책을 선택할 수 있다."""
        if self.use_fixed_focal_length:
            fx = self.fixed_focal_length
            fy = self.fixed_focal_length
        self.c_matrix = make_intrinsic(cx, cy, fx, fy)

    def _resolve_segmentation_roi(
            self,
            rgb_image: np.ndarray,
            roi_2d: Optional[List[float]],
    ) -> List[float]:
        if self.roi_source == "input" and roi_2d is not None:
            input_roi = [float(value) for value in roi_2d]
            if len(input_roi) != 4:
                raise ValueError("roi_2d must contain 4 values")
            self.logger.info(f"[{self.name}] using input 2D ROI: {input_roi}")
            return input_roi
        fixed_roi = list(self.segmentation_roi)
        self.logger.info(f"[{self.name}] using fixed 2D ROI: {fixed_roi}")
        return fixed_roi

    # ─── 추론 메인 ──────────────────────────────────────────────────────

    def run(
            self,
            rgb_image: np.ndarray,
            depth_image: Optional[np.ndarray] = None,
            normal_image: Optional[np.ndarray] = None,
            depth_scale: float = 1000,
            compute_suction_pts: bool = False,
            vis_pcd: bool = False,
            save_ply: bool = False,
            roi_2d: Optional[List[float]] = None,
    ) -> Tuple[Dict[str, Any], List[Any]]:
        """
        추론 메인. 반환은 (result, predictions) 튜플.

        ── 더미 구현 ──
        detector 가 None 이면 입력 이미지 shape 만 로깅하고
        '객체 없음(state=-2)' 결과를 반환한다.

        TODO: 아래를 본인 detection + 3D pipeline 으로 교체.
        """
        self.logger.info(f"[{self.name}] run() 호출")
        self.logger.info(f"[{self.name}]   rgb   : {rgb_image.shape if rgb_image is not None else None}")
        self.logger.info(f"[{self.name}]   depth : {depth_image.shape if depth_image is not None else None}")
        self.logger.info(f"[{self.name}]   normal: {normal_image.shape if normal_image is not None else None}")
        self.logger.info(f"[{self.name}]   roi_2d: {roi_2d}")
        resolved_roi_2d = self._resolve_segmentation_roi(rgb_image, roi_2d)

        # ── 더미: detector 가 없으면 예시 결과 반환 ──
        # generate_example_data.py 의 사각형 2개를 검출한 것처럼 동작한다.
        # 학생이 출력 형태를 확인하는 용도.
        if self.detector is None:
            self.logger.info(f"[{self.name}] detector=None → 더미 예시 결과 반환")

            # 객체 A: 빨간 사각형 (150,120)-(300,280), depth=500mm
            # 객체 B: 초록 사각형 (350,200)-(520,380), depth=600mm
            polygons = [
                [[150, 120], [300, 120], [300, 280], [150, 280]],
                [[350, 200], [520, 200], [520, 380], [350, 380]],
            ]
            # suction_points: 객체별 picking 포인트 리스트
            # 각 포인트는 ((x,y,z), (qx,qy,qz,qw)) — 로봇 베이스 좌표(mm) + 자세(quaternion)
            suction_points = [
                [   # 객체 A
                    ((-44.899, -72.088, 1192.901), (0.597, 0.801, -0.006, -0.052)),
                ],
                [   # 객체 B
                    ((-91.016, 165.696, 1138.743), (0.212, 0.969, -0.124, 0.036)),
                ],
            ]

            result = {
                "result_data": polygons,            # List[Polygon] — 객체별 polygon 좌표
                "state": 1,                         # 1 = 정상 검출
                "class_id": [0, 0],                 # List[int] — 객체별 surface id
                "pts_per_object": [1, 1],           # List[int] — 객체별 suction 후보 개수
                "suction_points": suction_points,   # List[List[((x,y,z),(qx,qy,qz,qw))]] — 로봇 좌표
                "2d_roi": resolved_roi_2d,
            }
            return result, []

        predictions = self.detector.inference(rgb_image, crop_box=resolved_roi_2d)

        polygons: List[List[List[int]]] = []
        class_ids: List[int] = []
        suction_points: List[List[Any]] = []
        kept_predictions = []

        for prediction in predictions:
            if prediction.mask is None:
                continue
            mask = largest_component_mask(prediction.mask, empty_as_none=True)
            if mask is None:
                continue
            prediction.mask = mask
            polygon = mask_to_polygon(mask, self.polygon_approx_ratio, self.polygon_min_epsilon)
            if not polygon:
                continue
            polygons.append(polygon)
            suction_points.append([])
            kept_predictions.append(prediction)

        if not polygons:
            result = {
                "result_data": [],
                "state": -2,
                "class_id": [],
                "pts_per_object": [],
                "suction_points": [],
                "2d_roi": resolved_roi_2d,
            }
            return result, kept_predictions

        if self.classifier is not None:
            class_predictions = self.classifier.classify_instances(rgb_image, kept_predictions)
            class_ids = [prediction.label for prediction in class_predictions]
            for instance, class_prediction in zip(kept_predictions, class_predictions):
                instance.label = class_prediction.label
                instance.class_index = class_prediction.class_index
                instance.class_name = class_prediction.class_name
                instance.class_confidence = class_prediction.confidence
                instance.class_similarity = class_prediction.similarity
                instance.class_vote_ratio = class_prediction.vote_ratio
                instance.class_margin = class_prediction.margin
                instance.class_reject_reason = class_prediction.reject_reason
                instance.class_neighbor_indices = class_prediction.neighbor_indices
                instance.class_neighbor_labels = class_prediction.neighbor_labels
                instance.class_neighbor_similarities = class_prediction.neighbor_similarities
        else:
            class_ids = [int(prediction.label) if prediction.label is not None else 0 for prediction in kept_predictions]

        priority_scores = self.priority_scorer.score_instances(
            kept_predictions,
            depth_image,
            resolved_roi_2d,
        )
        for instance, priority_score in zip(kept_predictions, priority_scores):
            instance.grasp_priority = priority_score.total
            instance.grasp_support_score = priority_score.support_score
            instance.grasp_depth_score = priority_score.depth_score
            instance.grasp_center_score = priority_score.center_score
            instance.grasp_isolation_score = priority_score.isolation_score
            instance.grasp_clearance_score = priority_score.clearance_score
            instance.grasp_area_score = priority_score.area_score
            instance.grasp_clearance_distance = priority_score.clearance_distance
            instance.grasp_mask_area = priority_score.mask_area
            instance.grasp_object_depth = priority_score.object_depth
            instance.grasp_center_distance = priority_score.center_distance
            instance.grasp_depth_candidate = priority_score.depth_candidate
            instance.grasp_valid_depth = priority_score.valid_depth

        if compute_suction_pts:
            for instance in kept_predictions:
                instance.suction_footprint = None
                instance.suction_candidates = []
                instance.suction_surface = None
                instance.suction_normal_z_score = None

            target_index = self._select_priority_target_index(kept_predictions, priority_scores)
            if target_index is not None:
                target_suction_points = self.suction_pipeline.compute(
                    [kept_predictions[target_index]],
                    depth_image,
                    normal_image,
                    self.c_matrix,
                    self.extrinsic,
                )
                kept_predictions[target_index].suction_normal_z_score = self._suction_normal_z_score(kept_predictions[target_index])
                suction_points[target_index] = target_suction_points[0] if target_suction_points else []

        items = list(zip(polygons, class_ids, suction_points, kept_predictions, priority_scores))
        items.sort(
            key=lambda item: (
                len(item[2]) > 0,
                float(item[4].total),
                float(getattr(item[3], "class_similarity", 0.0)),
                float(getattr(item[3], "score", 0.0) or 0.0),
            ),
            reverse=True,
        )
        polygons = [item[0] for item in items]
        class_ids = [item[1] for item in items]
        suction_points = [item[2] for item in items]
        kept_predictions = [item[3] for item in items]

        result = {
            "result_data": polygons,
            "state": 1,
            "class_id": class_ids,
            "pts_per_object": [len(points) for points in suction_points],
            "suction_points": suction_points,
            "2d_roi": resolved_roi_2d,
        }
        return result, kept_predictions

    @staticmethod
    def _select_priority_target_index(
            predictions: List[Any],
            priority_scores: List[Any],
    ) -> Optional[int]:
        if not predictions or not priority_scores:
            return None
        count = min(len(predictions), len(priority_scores))
        if count <= 0:
            return None
        return max(
            range(count),
            key=lambda index: (
                float(priority_scores[index].total),
                float(getattr(predictions[index], "class_similarity", 0.0)),
                float(getattr(predictions[index], "score", 0.0) or 0.0),
            ),
        )

    @staticmethod
    def _suction_normal_z_score(prediction: Any) -> float:
        surface_debug = getattr(prediction, "suction_surface", None)
        if not isinstance(surface_debug, dict) or not surface_debug.get("passed", False):
            return 0.0
        return normal_z_score(surface_debug)

    # ─── 결과 시각화 저장 ───────────────────────────────────────────────

    def save_result(
            self,
            rgb_img: np.ndarray,
            predictions: Any,
            polygons: List[Any],
            suction_pts: Optional[Any] = None,
            save_path: str = ".",
            save_name: str = "temp.png",
            compute_suction_pts: bool = False,
    ) -> None:
        """
        추론 결과를 이미지로 시각화해서 디스크에 저장.

        ── 더미 구현 ──
        detection 결과 없으면 원본 RGB 를 그대로 저장.

        TODO: polygon, suction point 를 오버레이해서 시각화.
        """
        os.makedirs(save_path, exist_ok=True)
        full_path = os.path.join(save_path, f"{save_name}.png")

        if not polygons:
            # 결과 없음 — 원본 이미지만 저장
            cv2.imwrite(full_path, rgb_img)
            self.logger.info(f"[{self.name}] save_result: 객체 없음 → 원본 저장 ({full_path})")
            return

        vis_img = rgb_img.copy()
        for index, polygon in enumerate(polygons):
            pts = np.array(polygon, dtype=np.int32)
            if pts.ndim != 2 or pts.shape[0] < 3:
                continue
            is_grasp_target = index == 0
            color = (0, 0, 255) if is_grasp_target else (0, 180, 0)
            thickness = 4 if is_grasp_target else 2
            cv2.polylines(vis_img, [pts], True, color, thickness)

            anchor_xy = np.mean(pts, axis=0).astype(int)
            if index < len(predictions):
                surface_debug = getattr(predictions[index], "suction_surface", None)
                if isinstance(surface_debug, dict):
                    center_xy = surface_debug.get("surface_center_xy")
                    if isinstance(center_xy, list) and len(center_xy) == 2:
                        anchor_xy = np.asarray([int(center_xy[0]), int(center_xy[1])], dtype=np.int32)
            if is_grasp_target:
                priority = getattr(predictions[index], "grasp_priority", None) if index < len(predictions) else None
                depth = getattr(predictions[index], "grasp_object_depth", None) if index < len(predictions) else None
                target_text = "GRASP #1"
                if priority is not None:
                    target_text += f" P:{float(priority):.2f}"
                if depth is not None:
                    target_text += f" D:{float(depth):.0f}"
                text_origin = tuple(np.maximum(pts.min(axis=0) + np.array([0, -10]), np.array([0, 20])).astype(int))
                cv2.putText(vis_img, target_text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 4, cv2.LINE_AA)
                cv2.putText(vis_img, target_text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                cv2.drawMarker(
                    vis_img,
                    tuple(anchor_xy),
                    color,
                    markerType=cv2.MARKER_CROSS,
                    markerSize=28,
                    thickness=3,
                    line_type=cv2.LINE_AA,
                )
            else:
                cv2.circle(vis_img, tuple(anchor_xy), 5, (0, 255, 255), -1, cv2.LINE_AA)

            if index < len(predictions):
                self._draw_class4_cap_marker(vis_img, predictions[index], color)

            if suction_pts and index < len(suction_pts) and suction_pts[index]:
                point_info = suction_pts[index][0]
                if isinstance(point_info, (list, tuple)) and len(point_info) == 2:
                    position = point_info[0]
                    quaternion = point_info[1]
                    self._draw_suction_debug(vis_img, anchor_xy, position, quaternion)

        cv2.imwrite(full_path, vis_img)
        self.logger.info(f"[{self.name}] save_result: {full_path}")

    @staticmethod
    def _draw_class4_cap_marker(
            image: np.ndarray,
            prediction: Any,
            color: tuple[int, int, int],
    ) -> None:
        if PickNPlace._prediction_class_id(prediction) != 4:
            return
        cap_xy = PickNPlace._class4_cap_center_xy(prediction)
        if cap_xy is None:
            return
        x, y = cap_xy
        half = 14
        top_left = (max(0, x - half), max(0, y - half))
        bottom_right = (min(image.shape[1] - 1, x + half), min(image.shape[0] - 1, y + half))
        cv2.rectangle(image, top_left, bottom_right, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.rectangle(image, top_left, bottom_right, color, 2, cv2.LINE_AA)

    @staticmethod
    def _prediction_class_id(prediction: Any) -> Optional[int]:
        for attr in ("class_index", "label"):
            value = getattr(prediction, attr, None)
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _class4_cap_center_xy(prediction: Any) -> Optional[tuple[int, int]]:
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

    def _draw_suction_debug(
            self,
            vis_img: np.ndarray,
            anchor_xy: np.ndarray,
            position_xyz: Any,
            quaternion_xyzw: Any,
    ) -> None:
        try:
            quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
            rotation = quaternion_to_rotation_matrix(quaternion)
            axis_x = rotation[:, 0]
            axis_y = rotation[:, 1]
            axis_z = rotation[:, 2]

            anchor = np.asarray(anchor_xy, dtype=np.int32).reshape(2)
            scale = 35

            def _project(axis: np.ndarray) -> tuple[int, int]:
                offset = np.array([axis[0], axis[1]], dtype=np.float64)
                norm = np.linalg.norm(offset)
                if norm < 1e-9:
                    offset = np.array([1.0, 0.0], dtype=np.float64)
                    norm = 1.0
                offset = offset / norm
                end = anchor + np.round(offset * scale).astype(np.int32)
                return int(end[0]), int(end[1])

            x_end = _project(axis_x)
            y_end = _project(axis_y)
            z_end = _project(axis_z)

            cv2.arrowedLine(vis_img, tuple(anchor), x_end, (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.2)
            cv2.arrowedLine(vis_img, tuple(anchor), y_end, (0, 255, 0), 2, cv2.LINE_AA, tipLength=0.2)
            cv2.arrowedLine(vis_img, tuple(anchor), z_end, (255, 0, 0), 2, cv2.LINE_AA, tipLength=0.2)

            pos = np.asarray(position_xyz, dtype=np.float64).reshape(-1)
            text = f"P[{pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}] Q[{quaternion[0]:.2f},{quaternion[1]:.2f},{quaternion[2]:.2f},{quaternion[3]:.2f}]"
            text_origin = (int(anchor[0] + 8), int(max(20, anchor[1] - 8)))
            cv2.putText(vis_img, text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis_img, text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        except Exception:
            # 디버그 오버레이 실패는 저장 전체를 막지 않도록 무시한다.
            return
