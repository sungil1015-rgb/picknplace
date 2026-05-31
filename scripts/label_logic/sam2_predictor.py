"""SAM2 wrapper — singleton lazy-loaded model + point-prompt → polygon.

First call downloads the model (~75MB for sam2_t.pt) and is slow (~1.5s).
Subsequent calls take ~0.1-0.2s on RTX 4050.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple


_model = None
_model_name = None


def get_model(model_name: str = "sam2_t.pt"):
    """Get singleton SAM2 model (loads on first call)."""
    global _model, _model_name
    if _model is not None and _model_name == model_name:
        return _model
    from ultralytics import SAM
    _model = SAM(model_name)
    _model_name = model_name
    return _model


def is_loaded() -> bool:
    return _model is not None


def predict_polygon(
    image_path: Path,
    point_xy: Tuple[float, float],
    model_name: str = "sam2_t.pt",
) -> Optional[List[Tuple[float, float]]]:
    """Run SAM2 with a single positive point prompt.

    Returns polygon (list of (u, v) float tuples) or None if no mask.
    """
    model = get_model(model_name)
    px, py = float(point_xy[0]), float(point_xy[1])
    results = model.predict(
        str(image_path),
        points=[[[px, py]]],
        labels=[[1]],
        verbose=False,
    )
    if not results:
        return None
    masks = results[0].masks
    if masks is None or masks.xy is None or len(masks.xy) == 0:
        return None
    pts = masks.xy[0]
    if len(pts) < 3:
        return None
    return [(float(p[0]), float(p[1])) for p in pts]
