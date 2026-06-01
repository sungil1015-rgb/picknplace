from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _required(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"suction config missing required key: {key}")
    return mapping[key]


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1", "yes", "y", "on"):
            return True
        if normalized in ("false", "0", "no", "n", "off"):
            return False
    return bool(value)


@dataclass(frozen=True)
class SuctionConfig:
    depth_window: int
    normal_window: int
    candidate_count: int
    candidate_min_distance_px: float
    pca_offset_px: float
    candidate_min_offset_px: float
    candidate_clearance_offset_ratio: float
    cup_diameter_mm: float
    cup_center_spacing_mm: float
    min_cup_inside_ratio: float
    normal_surface_enabled: bool
    normal_seed_window: int
    surface_angle_threshold_deg: float
    min_surface_region_area_ratio: float
    min_surface_region_area_px: int
    min_suction_area_object_coverage: float
    min_suction_area_surface_coverage: float
    axis_min_area_px: int
    axis_min_rect_ratio: float
    axis_min_pca_ratio: float
    axis_reference_step_px: float

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "SuctionConfig":
        return cls(
            depth_window=int(_required(mapping, "depth_window")),
            normal_window=int(_required(mapping, "normal_window")),
            candidate_count=int(_required(mapping, "candidate_count")),
            candidate_min_distance_px=float(_required(mapping, "candidate_min_distance_px")),
            pca_offset_px=float(_required(mapping, "pca_offset_px")),
            candidate_min_offset_px=float(_required(mapping, "candidate_min_offset_px")),
            candidate_clearance_offset_ratio=float(_required(mapping, "candidate_clearance_offset_ratio")),
            cup_diameter_mm=float(_required(mapping, "cup_diameter_mm")),
            cup_center_spacing_mm=float(_required(mapping, "cup_center_spacing_mm")),
            min_cup_inside_ratio=float(_required(mapping, "min_cup_inside_ratio")),
            normal_surface_enabled=as_bool(_required(mapping, "normal_surface_enabled")),
            normal_seed_window=int(_required(mapping, "normal_seed_window")),
            surface_angle_threshold_deg=float(_required(mapping, "surface_angle_threshold_deg")),
            min_surface_region_area_ratio=float(_required(mapping, "min_surface_region_area_ratio")),
            min_surface_region_area_px=int(_required(mapping, "min_surface_region_area_px")),
            min_suction_area_object_coverage=float(_required(mapping, "min_suction_area_object_coverage")),
            min_suction_area_surface_coverage=float(_required(mapping, "min_suction_area_surface_coverage")),
            axis_min_area_px=int(_required(mapping, "axis_min_area_px")),
            axis_min_rect_ratio=float(_required(mapping, "axis_min_rect_ratio")),
            axis_min_pca_ratio=float(_required(mapping, "axis_min_pca_ratio")),
            axis_reference_step_px=float(_required(mapping, "axis_reference_step_px")),
        )
