"""Auto-classify user-drawn polygons via a trained YOLO seg model.

Workflow:
  1. User draws polygons (outlines) with W mode
  2. Run trained model on the BMP to get (poly, class_id, conf) detections
  3. For each user polygon, find best-matching detection by IoU (>= iou_thr)
  4. Return matched class_id (or -1 if no match)
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw


IMG_W, IMG_H = 1224, 1024


def _polygon_mask(polygon: List[Tuple[float, float]],
                  w: int = IMG_W, h: int = IMG_H) -> np.ndarray:
    img = Image.new("L", (w, h), 0)
    pts = [(float(p[0]), float(p[1])) for p in polygon]
    if len(pts) < 3:
        return np.zeros((h, w), dtype=bool)
    ImageDraw.Draw(img).polygon(pts, fill=1)
    return np.array(img, dtype=bool)


def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = int((mask_a & mask_b).sum())
    union = int((mask_a | mask_b).sum())
    if union == 0:
        return 0.0
    return float(inter) / float(union)


# Model cache (singleton per path)
_model_cache: dict = {}


def get_model(model_path: str):
    if model_path in _model_cache:
        return _model_cache[model_path]
    from ultralytics import YOLO
    m = YOLO(model_path)
    _model_cache[model_path] = m
    return m


def extract_clean_polygons(results_obj) -> List[np.ndarray]:
    """Extract single-largest-contour polygons in original image coordinates.

    Ultralytics 8.4의 기본 ``Masks.xy``는 ``masks2segments(strategy='all')``로
    호출돼서 마스크가 여러 컨투어로 끊긴 경우 ``merge_multi_segment``로
    이어붙인다. 그 연결선이 자기 교차하는 별 모양 폴리곤을 만들어 결과
    시각화가 망가짐. 여기선 ``strategy='largest'``로 가장 큰 컨투어 1개만
    뽑아 원본 해상도로 스케일한다.

    Returns list of (N, 2) numpy arrays, same order/length as
    ``results_obj.masks``. masks가 없으면 빈 리스트.
    """
    if results_obj.masks is None:
        return []
    from ultralytics.utils import ops
    segs = ops.masks2segments(results_obj.masks.data, strategy="largest")
    return [
        ops.scale_coords(
            results_obj.masks.data.shape[1:], s,
            results_obj.orig_shape, normalize=False,
        )
        for s in segs
    ]


def _resolve_imgsz(model, fallback: int = 640) -> int:
    """모델 체크포인트에 저장된 학습 imgsz를 우선 사용 (없으면 fallback).

    학습 imgsz와 다른 값으로 추론하면 마스크 품질이 떨어짐.
    """
    try:
        train_args = getattr(model, "ckpt", None) or {}
        if isinstance(train_args, dict):
            sz = train_args.get("train_args", {}).get("imgsz")
            if isinstance(sz, int) and sz > 0:
                return sz
        a = getattr(model, "args", None) or {}
        sz = a.get("imgsz") if isinstance(a, dict) else None
        if isinstance(sz, int) and sz > 0:
            return sz
    except Exception:
        pass
    return fallback


def run_model_inference(
    bmp_path: Path, model_path: str, conf: float = 0.25,
) -> List[Tuple[List[Tuple[float, float]], int, float]]:
    """Return list of (polygon, class_id, conf) from YOLO seg inference."""
    model = get_model(model_path)
    imgsz = _resolve_imgsz(model, fallback=640)
    results = model.predict(
        str(bmp_path), conf=conf, imgsz=imgsz,
        verbose=False, device=0,
    )
    out = []
    if not results or results[0].masks is None:
        return out
    r = results[0]
    polys_np = extract_clean_polygons(r)
    for i in range(len(r.boxes)):
        cid = int(r.boxes.cls[i].item())
        cf = float(r.boxes.conf[i].item())
        if i < len(polys_np):
            poly = [(float(p[0]), float(p[1])) for p in polys_np[i]]
            if len(poly) >= 3:
                out.append((poly, cid, cf))
    return out


def assign_classes(
    user_polys: List[List[Tuple[float, float]]],
    detections: List[Tuple[List[Tuple[float, float]], int, float]],
    iou_thr: float = 0.3,
) -> List[Tuple[int, float, float]]:
    """For each user polygon, find best-matching detection by IoU.

    Returns list of (class_id, model_conf, iou) — same length as user_polys.
    class_id = -1 if no detection above iou_thr.
    """
    if not detections:
        return [(-1, 0.0, 0.0)] * len(user_polys)
    det_masks = [_polygon_mask(d[0]) for d in detections]
    out: List[Tuple[int, float, float]] = []
    for upoly in user_polys:
        umask = _polygon_mask(upoly)
        if not umask.any():
            out.append((-1, 0.0, 0.0))
            continue
        best_iou = 0.0
        best_idx = -1
        for j, dm in enumerate(det_masks):
            iou = _iou(umask, dm)
            if iou > best_iou:
                best_iou = iou
                best_idx = j
        if best_idx >= 0 and best_iou >= iou_thr:
            cid = detections[best_idx][1]
            cf = detections[best_idx][2]
            out.append((cid, cf, best_iou))
        else:
            out.append((-1, 0.0, best_iou))
    return out
