from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _required(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"suction config missing required key: {key}")
    return mapping[key]


def _section(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key, {})
    return value if isinstance(value, dict) else {}


def _get(mapping: dict[str, Any], section: dict[str, Any], section_key: str, flat_key: str, default: Any = None) -> Any:
    if section_key in section:
        return section[section_key]
    if flat_key in mapping:
        return mapping[flat_key]
    if default is not None:
        return default
    raise ValueError(f"suction config missing required key: {flat_key}")


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
    suction_strategy_default: str
    suction_strategy_by_class: dict[int, str]
    normal_surface_enabled: bool
    surface_angle_threshold_deg: float
    surface_open_kernel_px: int
    surface_close_kernel_px: int
    surface_fill_holes_max_area_px: int
    surface_fill_holes_max_aspect_ratio: float
    surface_center_method: str
    surface_rect_max_area_ratio: float
    min_surface_region_area_ratio: float
    min_surface_region_area_px: int
    normal_cluster_max_count: int
    class3_depth_split_enabled: bool
    class3_depth_split_min_gap_mm: float
    class3_depth_split_trim_low_percentile: float
    class3_depth_split_trim_high_percentile: float
    class3_depth_split_cut_band_mm: float
    class3_depth_split_bridge_open_kernel_px: int
    class3_depth_split_line_cut_enabled: bool
    class3_depth_split_line_cut_thickness_px: int
    class3_depth_split_line_cut_min_aspect_ratio: float
    class3_depth_split_min_layer_area_px: int
    class3_depth_split_min_layer_area_ratio: float
    class3_depth_split_min_component_area_px: int
    min_suction_area_object_coverage: float
    min_suction_area_surface_coverage: float
    axis_min_area_px: int
    axis_min_rect_ratio: float
    axis_min_pca_ratio: float
    axis_reference_step_px: float

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "SuctionConfig":
        strategy_cfg = mapping.get("strategy", mapping.get("suction_strategy", {}))
        cup_cfg = _section(mapping, "cup")
        candidate_cfg = _section(mapping, "candidates")
        normal_cfg = _section(mapping, "normal_surface")
        cluster_cfg = _section(normal_cfg, "cluster")
        morphology_cfg = _section(normal_cfg, "morphology")
        center_cfg = _section(normal_cfg, "center")
        region_cfg = _section(normal_cfg, "min_region")
        class3_depth_cfg = _section(normal_cfg, "class3_depth_split")
        advanced_cfg = _section(mapping, "advanced")
        footprint_cfg = _section(advanced_cfg, "footprint")
        axis_cfg = _section(advanced_cfg, "axis")
        return cls(
            depth_window=int(_get(mapping, advanced_cfg, "depth_window", "depth_window")),
            normal_window=int(_get(mapping, advanced_cfg, "normal_window", "normal_window")),
            candidate_count=int(_get(mapping, candidate_cfg, "count", "candidate_count")),
            candidate_min_distance_px=float(_get(mapping, candidate_cfg, "min_distance_px", "candidate_min_distance_px")),
            pca_offset_px=float(_get(mapping, candidate_cfg, "pca_offset_px", "pca_offset_px")),
            candidate_min_offset_px=float(_get(mapping, advanced_cfg, "candidate_min_offset_px", "candidate_min_offset_px")),
            candidate_clearance_offset_ratio=float(_get(mapping, advanced_cfg, "candidate_clearance_offset_ratio", "candidate_clearance_offset_ratio")),
            cup_diameter_mm=float(_get(mapping, cup_cfg, "diameter_mm", "cup_diameter_mm")),
            cup_center_spacing_mm=float(_get(mapping, cup_cfg, "center_spacing_mm", "cup_center_spacing_mm")),
            min_cup_inside_ratio=float(_get(mapping, footprint_cfg, "min_cup_inside_ratio", "min_cup_inside_ratio")),
            suction_strategy_default=str(strategy_cfg.get("default", "normal")) if isinstance(strategy_cfg, dict) else "normal",
            suction_strategy_by_class=_strategy_by_class(strategy_cfg),
            normal_surface_enabled=as_bool(_get(mapping, normal_cfg, "enabled", "normal_surface_enabled")),
            surface_angle_threshold_deg=float(_get(mapping, normal_cfg, "angle_threshold_deg", "surface_angle_threshold_deg")),
            surface_open_kernel_px=int(_get(mapping, morphology_cfg, "open_kernel_px", "surface_open_kernel_px")),
            surface_close_kernel_px=int(_get(mapping, morphology_cfg, "close_kernel_px", "surface_close_kernel_px")),
            surface_fill_holes_max_area_px=int(_get(mapping, morphology_cfg, "fill_holes_max_area_px", "surface_fill_holes_max_area_px", 0)),
            surface_fill_holes_max_aspect_ratio=float(_get(mapping, morphology_cfg, "fill_holes_max_aspect_ratio", "surface_fill_holes_max_aspect_ratio", 4.0)),
            surface_center_method=str(_get(mapping, center_cfg, "method", "surface_center_method")),
            surface_rect_max_area_ratio=float(_get(mapping, center_cfg, "rect_max_area_ratio", "surface_rect_max_area_ratio")),
            min_surface_region_area_ratio=float(_get(mapping, region_cfg, "area_ratio", "min_surface_region_area_ratio")),
            min_surface_region_area_px=int(_get(mapping, region_cfg, "area_px", "min_surface_region_area_px")),
            normal_cluster_max_count=int(_get(mapping, cluster_cfg, "max_count", "normal_cluster_max_count", 5)),
            class3_depth_split_enabled=as_bool(class3_depth_cfg.get("enabled", False)),
            class3_depth_split_min_gap_mm=float(class3_depth_cfg.get("min_gap_mm", 1.0)),
            class3_depth_split_trim_low_percentile=float(class3_depth_cfg.get("trim_low_percentile", 1.0)),
            class3_depth_split_trim_high_percentile=float(class3_depth_cfg.get("trim_high_percentile", 99.0)),
            class3_depth_split_cut_band_mm=float(class3_depth_cfg.get("cut_band_mm", 0.0)),
            class3_depth_split_bridge_open_kernel_px=int(class3_depth_cfg.get("bridge_open_kernel_px", 1)),
            class3_depth_split_line_cut_enabled=as_bool(class3_depth_cfg.get("line_cut_enabled", False)),
            class3_depth_split_line_cut_thickness_px=int(class3_depth_cfg.get("line_cut_thickness_px", 5)),
            class3_depth_split_line_cut_min_aspect_ratio=float(class3_depth_cfg.get("line_cut_min_aspect_ratio", 3.0)),
            class3_depth_split_min_layer_area_px=int(class3_depth_cfg.get("min_layer_area_px", 300)),
            class3_depth_split_min_layer_area_ratio=float(class3_depth_cfg.get("min_layer_area_ratio", 0.08)),
            class3_depth_split_min_component_area_px=int(class3_depth_cfg.get("min_component_area_px", 300)),
            min_suction_area_object_coverage=float(_get(mapping, footprint_cfg, "min_object_coverage", "min_suction_area_object_coverage")),
            min_suction_area_surface_coverage=float(_get(mapping, footprint_cfg, "min_surface_coverage", "min_suction_area_surface_coverage")),
            axis_min_area_px=int(_get(mapping, axis_cfg, "min_area_px", "axis_min_area_px")),
            axis_min_rect_ratio=float(_get(mapping, axis_cfg, "min_rect_ratio", "axis_min_rect_ratio")),
            axis_min_pca_ratio=float(_get(mapping, axis_cfg, "min_pca_ratio", "axis_min_pca_ratio")),
            axis_reference_step_px=float(_get(mapping, axis_cfg, "reference_step_px", "axis_reference_step_px")),
        )


def _strategy_by_class(strategy_cfg: Any) -> dict[int, str]:
    if not isinstance(strategy_cfg, dict):
        return {}
    by_class = strategy_cfg.get("by_class", {})
    if not isinstance(by_class, dict):
        return {}
    parsed: dict[int, str] = {}
    for key, value in by_class.items():
        try:
            class_index = int(key)
        except (TypeError, ValueError):
            continue
        parsed[class_index] = str(value)
    return parsed
