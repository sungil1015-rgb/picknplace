"""Picking-point computation for dual-suction head.

Output: 6-DOF pose in Zivid camera frame (mm) + visualization rectangle.
Coordinate frame: Zivid camera (mm, +X right, +Y down, +Z forward).

Pipeline (haribo / mango / pencil_case / metal_case):
  1) polygon → 2D mask
  2) PLY grid + valid mask → 3D points
  3) AABB center → position
  4) PCA → long_axis / mid_axis / normal
  5) cup A/B = position ± (PITCH/2) * long_axis
  6) validate both cup patches have surface support
  7) build rectangle corners (long × short) for visualization

bottle is handled by the same path but caller may want to filter further
(e.g., trust valid mask only). Unknown class is rejected upstream.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import yaml
from PIL import Image, ImageDraw

# ─── Config 로드 — config/default.yaml에서 모든 임계값 / 상수 ───────────
# 변경은 YAML 한 곳에서. 코드 수정 없이 튜닝 가능.
# 데이터셋별 override는 config/<team>.yaml 만들어서 (TODO: 자동 로드).
_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"
with open(_CONFIG_PATH, encoding="utf-8") as _f:
    CONFIG = yaml.safe_load(_f)

# ─── Camera intrinsics — config 기본값(MR60 가정).
# Runtime에 다른 카메라(MR130 등) 샷 처리할 땐 set_camera_intrinsics()로
# 갱신. label_tool이 샷의 *_info.json에서 fx/fy/cx/cy 읽어 적용.
FX = float(CONFIG["camera"]["intrinsics"]["fx"])
FY = float(CONFIG["camera"]["intrinsics"]["fy"])
CX = float(CONFIG["camera"]["intrinsics"]["cx"])
CY = float(CONFIG["camera"]["intrinsics"]["cy"])
IMG_W = int(CONFIG["camera"]["resolution"]["width"])
IMG_H = int(CONFIG["camera"]["resolution"]["height"])


def set_camera_intrinsics(fx: float, fy: float, cx: float, cy: float) -> None:
    """샷별 카메라 intrinsics를 runtime에 갱신.
    picking.py + suction_score.py 둘 다 동기 (둘이 별도 변환 함수 가짐).
    호출 후엔 다음 compute_pick부터 새 값 적용.
    """
    global FX, FY, CX, CY
    FX, FY, CX, CY = float(fx), float(fy), float(cx), float(cy)
    try:
        from . import suction_score as _ss
        _ss.set_camera_intrinsics(fx, fy, cx, cy)
    except Exception:
        pass

# ─── Suction tool (흡착 헤드) ─────────────────────────────────────
SUCTION_CUP_OD_MM = float(CONFIG["suction"]["cup_od_mm"])
SUCTION_CUP_PITCH_MM = float(CONFIG["suction"]["cup_pitch_mm"])
MIN_FLAT_PATCH_DIAMETER_MM = float(CONFIG["suction"]["min_flat_patch_diameter_mm"])
RECT_LONG_MM = SUCTION_CUP_PITCH_MM + SUCTION_CUP_OD_MM
RECT_SHORT_MM = SUCTION_CUP_OD_MM
SAFETY_OFFSET_MM = float(CONFIG["suction"]["safety_offset_mm"])
HEAD_CLEARANCE_MM = float(CONFIG["suction"]["head_clearance_mm"])
ARM_BOX_SCALE = float(CONFIG["suction"]["arm_box_scale"])
ARM_BOX_HEIGHT_MM = float(CONFIG["suction"]["arm_box_height_mm"])
ARM_RADIUS_MM = float(CONFIG["suction"]["arm_radius_mm"])

# ─── Approach (탑다운 + tilt 한계) ────────────────────────────────
MAX_TILT_DEG = float(CONFIG["approach"]["max_tilt_deg"])
MAX_TILT_COS = float(np.cos(np.radians(MAX_TILT_DEG)))
TOP_SURFACE_PERCENTILE = float(CONFIG["approach"]["top_surface_percentile"])
TOP_SURFACE_BAND_MM = float(CONFIG["approach"]["top_surface_band_mm"])
# Refine hard cap (mm) — heatmap argmax가 initial position에서 이만큼 이상
# 벗어나면 refine 결과 버리고 initial pose 유지. 곡면 꼭대기 같은 거짓 최적점
# 으로 끌려가는 걸 방지. <= 0이면 cap 비활성.
REFINE_MAX_OFFSET_MM = float(CONFIG["approach"].get("refine_max_offset_mm", 0.0))

# ─── NaN 비율 (S4) — bottle은 더 빡세게 ──────────────────────────
NAN_RATIO_REJECT = float(CONFIG["gates"]["s4_nan_ratio_reject"])
NAN_RATIO_REJECT_BOTTLE = float(CONFIG["gates"]["s4_nan_ratio_reject_bottle"])

# ─── Heatmap visualization (2D/3D 뷰 + argmax 검색 모두 통일) ────
HEATMAP_CUP_RADIUS_PX = float(
    CONFIG.get("heatmap", {}).get("cup_radius_px", 25.0)
)

# ─── Classes ────────────────────────────────────────────────────
CLASSES = list(CONFIG["classes"]["names"])
BOTTLE_CID = int(CONFIG["classes"]["bottle_cid"])
EXCLUDE_CIDS = set(CONFIG["classes"]["exclude_cids"])
# Unknown / OOD — 학습된 5종 어디에도 속하지 않는 객체. compute_pick 호출 시
# cid=UNKNOWN_CID 로 보내면 일반 흡착 픽 로직 적용 (bottle 특화 분기 skip,
# class_prior 0.5 중립, size_margin은 객체 길이 기반이라 클래스 무관).
UNKNOWN_CID = -1
# CLASS_PRIORS: cid 정수 키 (이름 → cid 매핑은 CLASSES.index)
CLASS_PRIORS = {CLASSES.index(_name): float(_p)
                for _name, _p in CONFIG["classes"]["priors"].items()
                if _name in CLASSES}

# ─── Confidence scoring ─────────────────────────────────────────
SCORE_WEIGHTS = {k: float(v) for k, v in CONFIG["confidence"]["weights"].items()}
TIER_HIGH = float(CONFIG["confidence"]["tier_high"])
TIER_MEDIUM = float(CONFIG["confidence"]["tier_medium"])

# Tier / Status names — code-level constants (튜닝 대상 X, 코드에서 참조)
TIER_HIGH_NAME = "HIGH"
TIER_MEDIUM_NAME = "MEDIUM"
TIER_LOW_NAME = "LOW"
TIER_UNPICKABLE_NAME = "UNPICKABLE"
STATUS_PICKABLE = "PICKABLE"
STATUS_DEFER = "DEFER"
STATUS_IMPOSSIBLE = "IMPOSSIBLE"
STATUS_MANUAL = "MANUAL"     # 사용자가 3D 뷰에서 직접 라벨링한 GT 픽

# ─── Gate thresholds ────────────────────────────────────────────
GATE_S1_OVERLAP_DEFER = float(CONFIG["gates"]["s1_overlap_defer"])
GATE_S2_ARM_PROTRUSION_MM = float(CONFIG["gates"]["s2_arm_protrusion_mm"])
GATE_S2_CUP_PROTRUSION_MM = float(CONFIG["gates"]["s2_cup_protrusion_mm"])
GATE_S2_CUP_BUMP_MM = float(CONFIG["gates"]["s2_cup_bump_mm"])
GATE_S2_CUP_COVERAGE_MIN = float(CONFIG["gates"]["s2_cup_coverage_min"])
# 컵 둘레(rim) sealing — 흡착 물리상 둘레 ring이 self surface에 모두 닿아야
# 진공 형성. 면적 기반 cup_coverage(0.55)는 disk 절반만 덮여도 통과시키므로
# rim 비율을 별도 게이트로 검사. 객체 강성에 따라 threshold 다름:
#   - 강체 (pencil_case, metal_case): 0.90 (라벨 정확, 표면 평평)
#   - 비닐 (haribo, mango): 0.70 (구김·주름으로 라벨 가장자리 노이즈)
#   - bottle: 곡면 측면이라 rim 게이트 자체 skip
GATE_S2_CUP_RIM_MIN = float(CONFIG["gates"].get("s2_cup_rim_min", 0.95))
_RIM_PER_CLASS = CONFIG["gates"].get("s2_cup_rim_min_per_class", {}) or {}
_CLASS_NAMES = CONFIG["classes"]["names"]
# {cid: threshold} — 이름 → cid 변환
GATE_S2_CUP_RIM_MIN_BY_CID = {}
for _name, _thr in _RIM_PER_CLASS.items():
    if _name in _CLASS_NAMES:
        GATE_S2_CUP_RIM_MIN_BY_CID[_CLASS_NAMES.index(_name)] = float(_thr)
# rim 검사 시 self_mask를 N픽셀 dilate해서 라벨 가장자리 노이즈 흡수.
GATE_S2_CUP_RIM_DILATE_PX = int(CONFIG["gates"].get("s2_cup_rim_dilate_px", 0))
GATE_S2_RECT_PLANARITY_MM = float(CONFIG["gates"]["s2_rect_planarity_mm"])
GATE_S2_CORNER_PROTRUSION_MM = float(CONFIG["gates"].get("s2_corner_protrusion_mm", 20.0))
GATE_S2_CORNER_NAN_RATIO = float(CONFIG["gates"].get("s2_corner_nan_ratio", 0.7))
GATE_S2_CORNER_MIN_FAILS = int(CONFIG["gates"].get("s2_corner_min_fails", 2))
# Single-cup pick 허용용 neighbor 검사 — 객체보다 짧은 폴리곤에서 한 컵이
# 객체 밖에 있을 때, 그 컵 주변 반경 안에 "다른 객체 표면"이 있는지 확인.
# 빈 바닥/벽처럼 깊이 차이 큰 점은 무시(빈 공간으로 간주), 컵 평면 ±band 안
# self-mask 밖 점이 max_points 이상이면 다른 객체로 판단 → reject.
GATE_S2_NEIGHBOR_RADIUS_MM = float(CONFIG["gates"].get("s2_neighbor_radius_mm", 40.0))
GATE_S2_NEIGHBOR_BAND_MM = float(CONFIG["gates"].get("s2_neighbor_band_mm", 10.0))
GATE_S2_NEIGHBOR_MAX_POINTS = int(CONFIG["gates"].get("s2_neighbor_max_points", 20))
# length: True면 객체 < pitch도 IMPOSSIBLE 대신 일반 흐름 진행 (single-cup
# 가능성을 neighbor_clean 게이트가 판정).
GATE_LENGTH_DEFER = bool(CONFIG["gates"].get("length_defer", False))
# single_cup_cids: 객체 < pitch인 작은 폴리곤에서 dual-cup 가정 게이트
# (cup_support / cup_coverage / rect_planarity / arm_protrusion / cup_bump)를
# skip하고 neighbor_clean만 적용. haribo 같은 작은 사탕 전용.
SINGLE_CUP_CIDS = set(CONFIG["gates"].get("single_cup_cids", []))
GATE_S3_FLATNESS_MIN = float(CONFIG["gates"]["s3_flatness_min"])
GATE_CUP_SUPPORT_MIN_POINTS = int(CONFIG["gates"].get("cup_support_min_points", 5))

# ─── Occlusion 검사 (둘레 가려짐) ──────────────────────────────
OCCLUSION_DEPTH_JUMP_MM = float(CONFIG["occlusion"]["depth_jump_mm"])
OCCLUSION_NORMAL_OFFSET_PX = int(CONFIG["occlusion"]["normal_offset_px"])
OCCLUSION_SAMPLE_STEP_PX = int(CONFIG["occlusion"]["sample_step_px"])
OCCLUSION_CLEAR_HARD_REJECT = float(CONFIG["occlusion"]["clear_hard_reject"])

# ─── Depth recovery (transparent/reflective objects) ────────────
_REC_CFG = CONFIG.get("recovery", {})
RECOVERY_ENABLE = bool(_REC_CFG.get("enable", False))
RECOVERY_NAN_TRIGGER = float(_REC_CFG.get("nan_ratio_trigger", 0.10))
RECOVERY_NAN_MAX_AFTER = float(_REC_CFG.get("max_nan_ratio_after", 0.50))
RECOVERY_BOTTLE_USE_CYL = bool(_REC_CFG.get("bottle_use_cylinder", True))
RECOVERY_CYL_R_MIN = float(_REC_CFG.get("cylinder_radius_min_mm", 5.0))
RECOVERY_CYL_R_MAX = float(_REC_CFG.get("cylinder_radius_max_mm", 60.0))
RECOVERY_CYL_MIN_PTS = int(_REC_CFG.get("cylinder_min_valid_points", 50))

# ─── Bottle cylinder-pose pipeline ─────────────────────────────
_BOT_CFG = CONFIG.get("bottle", {})
BOTTLE_CYLINDER_POSE_ENABLE = bool(_BOT_CFG.get("enable_cylinder_pose", False))
BOTTLE_AXIS_Z_MAX = float(_BOT_CFG.get("axis_z_max", 0.5))
BOTTLE_MIN_VALID_FOR_FIT = int(_BOT_CFG.get("min_valid_for_fit", 30))
# known_radius_mm: None이면 free fit, 값 있으면 fix (sparse 데이터 강건성↑)
_BOT_R = _BOT_CFG.get("known_radius_mm")
BOTTLE_KNOWN_RADIUS_MM = float(_BOT_R) if _BOT_R is not None else None
BOTTLE_RADIUS_TOLERANCE_MM = float(_BOT_CFG.get("radius_tolerance_mm", 5.0))
# 실험: cap PCA + 본체 offset 단순화 모드 (docs/병_뚜껑_PCA_offset_고려사항.md)
BOTTLE_USE_CAP_PCA = bool(_BOT_CFG.get("use_cap_pca_offset", False))
BOTTLE_BODY_OFFSET_MM = float(_BOT_CFG.get("body_offset_mm", 30.0))


def _classify_tier(confidence: float) -> str:
    if confidence >= TIER_HIGH:
        return TIER_HIGH_NAME
    if confidence >= TIER_MEDIUM:
        return TIER_MEDIUM_NAME
    return TIER_LOW_NAME


# ─── Occlusion boundary detection (Phase 1D) ────────────────────────────────
# Detect when an object is partially covered by another object in front.
# We walk the polygon perimeter and compare depth on each side of the edge:
#   - Inside depth ~ object surface
#   - Outside depth significantly SMALLER (closer to camera) → an occluder
#     sits in front of where the edge runs → this edge is hidden by something.
# A high fraction of occluded boundary means we only see a partial slice of
# the object. The visible centroid does not represent the true mass center.

# 위 OCCLUSION_* 상수들은 모듈 상단에서 config/default.yaml 로드 결과로 정의됨.
# (재정의하지 않고 함수 default 인자에서 그대로 참조)


def _compute_occlusion_clear(
    grid: np.ndarray, polygon_2d, valid_mask: np.ndarray,
    jump_mm: float = OCCLUSION_DEPTH_JUMP_MM,
    normal_offset: int = OCCLUSION_NORMAL_OFFSET_PX,
    sample_step: int = OCCLUSION_SAMPLE_STEP_PX,
) -> float:
    """Fraction of polygon perimeter NOT occluded by foreground objects.

    Returns value in [0, 1]:
        1.0  = boundary is clean (free space outside everywhere)
        0.0  = entire boundary is occluded (object almost fully hidden)
    """
    pts = [(float(p[0]), float(p[1])) for p in polygon_2d]
    if len(pts) < 3:
        return 1.0

    # Polygon centroid for outward-normal disambiguation
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)

    H, W = valid_mask.shape
    z_field = grid["z"]

    n_total = 0
    n_occluded = 0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        dx, dy = x2 - x1, y2 - y1
        seg_len = float((dx * dx + dy * dy) ** 0.5)
        if seg_len < 1e-6:
            continue
        # Edge direction (unit)
        ex, ey = dx / seg_len, dy / seg_len
        # Two candidate perpendicular normals
        nx, ny = ey, -ex
        # Ensure (nx, ny) points OUTWARD: dot(midpoint→centroid, n) should be < 0
        mx = 0.5 * (x1 + x2)
        my = 0.5 * (y1 + y2)
        if (cx - mx) * nx + (cy - my) * ny > 0:
            nx, ny = -nx, -ny

        n_samples = max(1, int(seg_len // sample_step))
        for k in range(n_samples):
            t = (k + 0.5) / n_samples
            sx = x1 + t * dx
            sy = y1 + t * dy
            u_in = int(round(sx - nx * normal_offset))
            v_in = int(round(sy - ny * normal_offset))
            u_out = int(round(sx + nx * normal_offset))
            v_out = int(round(sy + ny * normal_offset))
            if not (0 <= u_in < W and 0 <= v_in < H
                    and 0 <= u_out < W and 0 <= v_out < H):
                continue
            z_in = float(z_field[v_in, u_in])
            z_out = float(z_field[v_out, u_out])
            # Skip if either side has invalid depth
            if not (np.isfinite(z_in) and np.isfinite(z_out)):
                continue
            n_total += 1
            # Outside is closer than inside by jump_mm → foreground occluder
            if z_out + jump_mm < z_in:
                n_occluded += 1

    if n_total == 0:
        return 1.0   # no usable boundary samples → assume clear
    return float(1.0 - (n_occluded / n_total))


def _compute_confidence(
    cid: int,
    valid_ratio: float,
    cup_a_support: int,
    cup_b_support: int,
    normal_std_score: float,
    obj_long_length_mm: float,
    occlusion_clear: float,
) -> tuple:
    """Combine 6 signals into confidence in [0, 1]. Returns (score, breakdown_dict).

    wall_margin 신호는 box_roi 폐기와 함께 제거(2026-05-13). 빈 가장자리 회피는
    S2_corner_depth 게이트가 담당하며, 가중치는 나머지 6개 신호에 비례 재분배.
    """
    s_valid = float(np.clip(valid_ratio, 0.0, 1.0))
    s_cup = float(np.clip(min(cup_a_support, cup_b_support) / 50.0, 0.0, 1.0))
    s_normal = float(np.clip(normal_std_score, 0.0, 1.0))
    s_size = float(np.clip(
        (obj_long_length_mm - SUCTION_CUP_PITCH_MM) / SUCTION_CUP_PITCH_MM,
        0.0, 1.0,
    ))
    s_prior = float(CLASS_PRIORS.get(cid, 0.5))
    s_occ = float(np.clip(occlusion_clear, 0.0, 1.0))

    breakdown = {
        "valid_ratio": s_valid,
        "cup_support": s_cup,
        "normal_std": s_normal,
        "size_margin": s_size,
        "class_prior": s_prior,
        "occlusion_clear": s_occ,
    }
    score = sum(SCORE_WEIGHTS[k] * v for k, v in breakdown.items())
    return float(np.clip(score, 0.0, 1.0)), breakdown


@dataclass
class PickResult:
    """Top-down dual-suction pick pose + visualization geometry.

    All 3D coordinates are in Zivid camera frame (mm).
    Approach axis: always camera -Z (top-down — confirmed 2026-05-09).
    """
    success: bool
    reason: str = ""
    cid: int = -1

    # 3-tier classification (PICKABLE / DEFER / IMPOSSIBLE)
    status: str = STATUS_IMPOSSIBLE
    # Gate names that failed (informative — empty for PICKABLE)
    gate_failures: List[str] = field(default_factory=list)

    # 6-DOF pose (camera frame)
    position_mm: np.ndarray = field(default_factory=lambda: np.zeros(3))
    long_axis: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    mid_axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -1.0]))
    yaw_rad: float = 0.0   # 4-DOF top-down approximation

    # Rectangle (camera frame, mm)
    rect_corners_3d: List[np.ndarray] = field(default_factory=list)
    # Rectangle in image pixels (projected via intrinsics)
    rect_corners_2d: List[Tuple[float, float]] = field(default_factory=list)

    # Cup centers (camera frame)
    cup_a_3d: Optional[np.ndarray] = None
    cup_b_3d: Optional[np.ndarray] = None
    cup_a_2d: Optional[Tuple[float, float]] = None
    cup_b_2d: Optional[Tuple[float, float]] = None

    # Diagnostics
    n_valid_points: int = 0
    eigenvalues: np.ndarray = field(default_factory=lambda: np.zeros(3))
    cup_a_support: int = 0
    cup_b_support: int = 0

    # Confidence scoring (Phase 1A) — populated for successful picks.
    confidence: float = 0.0
    tier: str = TIER_UNPICKABLE_NAME    # HIGH / MEDIUM / LOW / UNPICKABLE
    score_breakdown: dict = field(default_factory=dict)
    # Bottle cylinder fit 정보 (시각화용) — bottle 픽에만 채워짐
    bottle_fit: Optional[dict] = None
    # Pick mode — dual: 두 컵 모두 객체 표면 (정상). single: 한 컵 객체 위 +
    # 다른 컵 빈 공간 (작은 객체 fallback, haribo 한정). dual-cup 실패 시
    # haribo만 single로 retry한 결과면 "single". 그 외는 "dual".
    pick_mode: str = "dual"
    # Single-cup retry 진단 — dual fail 후 single도 fail한 경우, single이
    # 어느 게이트에서 잡혔는지 보존. PICKABLE이거나 single retry 자체가
    # 시도 안 된 경우엔 빈 값.
    single_retry_status: str = ""
    single_retry_gate_failures: List[str] = field(default_factory=list)
    single_retry_reason: str = ""
    # Arm 박스 안 최대 표면 protrusion (mm). 낮을수록 충돌 마진 큼 (안전).
    # PICKABLE 픽에만 채워짐. 정렬 보조 키로 사용 (label_tool).
    arm_max_protrusion_mm: float = 0.0

    def to_dict(self) -> dict:
        """Serializable dict for JSON output (used by probe + view_3d)."""
        d = {
            "success": self.success,
            "pickable": bool(self.success),
            "status": self.status,
            "gate_failures": list(self.gate_failures),
            "reason": self.reason,
            "cid": int(self.cid),
            "tier": self.tier,
            "confidence": float(self.confidence),
            "score_breakdown": dict(self.score_breakdown),
            "bottle_fit": dict(self.bottle_fit) if self.bottle_fit else None,
            "n_valid_points": int(self.n_valid_points),
            "cup_a_support": int(self.cup_a_support),
            "cup_b_support": int(self.cup_b_support),
            "pick_mode": self.pick_mode,
            "single_retry_status": self.single_retry_status,
            "single_retry_gate_failures": list(self.single_retry_gate_failures),
            "single_retry_reason": self.single_retry_reason,
        }
        if self.success:
            d.update({
                "position_mm": self.position_mm.tolist(),
                "long_axis": self.long_axis.tolist(),
                "mid_axis": self.mid_axis.tolist(),
                "normal": self.normal.tolist(),
                "yaw_rad": float(self.yaw_rad),
                "rect_corners_3d": [c.tolist() for c in self.rect_corners_3d],
                "rect_corners_2d": [list(c) for c in self.rect_corners_2d],
                "cup_a_3d": self.cup_a_3d.tolist() if self.cup_a_3d is not None else None,
                "cup_b_3d": self.cup_b_3d.tolist() if self.cup_b_3d is not None else None,
                "cup_a_2d": list(self.cup_a_2d) if self.cup_a_2d else None,
                "cup_b_2d": list(self.cup_b_2d) if self.cup_b_2d else None,
                "eigenvalues": self.eigenvalues.tolist(),
                "arm_max_protrusion_mm": float(self.arm_max_protrusion_mm),
            })
        return d


def project_to_image(p_camera, fx=None, fy=None, cx=None, cy=None):
    """Project a 3D point (camera frame) to image pixel (u, v).

    Returns None if behind camera.
    fx/fy/cx/cy = None → 모듈 글로벌(set_camera_intrinsics로 갱신) 사용.
    """
    z = float(p_camera[2])
    if z <= 1e-6:
        return None
    fx = FX if fx is None else float(fx)
    fy = FY if fy is None else float(fy)
    cx = CX if cx is None else float(cx)
    cy = CY if cy is None else float(cy)
    u = fx * float(p_camera[0]) / z + cx
    v = fy * float(p_camera[1]) / z + cy
    return (u, v)


def polygon_to_mask(poly_pts, w=IMG_W, h=IMG_H) -> np.ndarray:
    img = Image.new("L", (w, h), 0)
    pts = [(float(p[0]), float(p[1])) for p in poly_pts]
    if len(pts) < 3:
        return np.zeros((h, w), dtype=bool)
    ImageDraw.Draw(img).polygon(pts, fill=1)
    return np.array(img, dtype=bool)


def _orient_normal_toward_camera(normal: np.ndarray) -> np.ndarray:
    """Flip normal so it points toward the camera (-Z in Zivid frame)."""
    if normal[2] > 0:
        return -normal
    return normal


def _orient_axis_consistently(axis: np.ndarray) -> np.ndarray:
    """Make eigenvector sign deterministic — pick the one with +X dominant.

    PCA eigenvectors are sign-ambiguous. For visualization stability we
    flip so that the largest |component| is positive.
    """
    i = int(np.argmax(np.abs(axis)))
    if axis[i] < 0:
        return -axis
    return axis


def _from_manual_pick(mp: dict, cid: int) -> PickResult:
    """사용자 GT manual pick → PickResult.

    validation.all_pass = True면 PICKABLE 등급 부여 (HIGH tier),
    실패면 MANUAL 그대로 (CMES에 GT임을 알림).
    """
    position = np.asarray(mp.get("position_mm", [0, 0, 0]), dtype=np.float64)
    normal = np.asarray(mp.get("normal", [0, 0, -1]), dtype=np.float64)
    long_axis = np.asarray(mp.get("long_axis", [1, 0, 0]), dtype=np.float64)
    mid_axis = np.asarray(mp.get("mid_axis", [0, 1, 0]), dtype=np.float64)
    cup_a = np.asarray(mp.get("cup_a_3d", position + (SUCTION_CUP_PITCH_MM/2) * long_axis),
                        dtype=np.float64)
    cup_b = np.asarray(mp.get("cup_b_3d", position - (SUCTION_CUP_PITCH_MM/2) * long_axis),
                        dtype=np.float64)

    # Rectangle corners (long × short)
    half_long = RECT_LONG_MM / 2.0
    half_short = RECT_SHORT_MM / 2.0
    corners_3d = [
        position + half_long * long_axis + half_short * mid_axis,
        position + half_long * long_axis - half_short * mid_axis,
        position - half_long * long_axis - half_short * mid_axis,
        position - half_long * long_axis + half_short * mid_axis,
    ]
    corners_2d = []
    for c in corners_3d:
        uv = project_to_image(c)
        if uv is not None:
            corners_2d.append(uv)
    cup_a_2d = project_to_image(cup_a)
    cup_b_2d = project_to_image(cup_b)
    yaw = float(np.arctan2(long_axis[1], long_axis[0]))

    val = mp.get("validation", {})
    all_pass = bool(val.get("all_pass", True))
    # MANUAL은 사용자가 GT로 명시 → 항상 PICKABLE로 받기 (검증 결과는 보존)
    confidence = 1.0 if all_pass else 0.5
    tier = TIER_HIGH_NAME if all_pass else TIER_LOW_NAME

    return PickResult(
        success=True, status=STATUS_MANUAL, cid=cid,
        position_mm=position, normal=normal,
        long_axis=long_axis, mid_axis=mid_axis,
        yaw_rad=yaw,
        rect_corners_3d=corners_3d, rect_corners_2d=corners_2d,
        cup_a_3d=cup_a, cup_b_3d=cup_b,
        cup_a_2d=cup_a_2d, cup_b_2d=cup_b_2d,
        confidence=confidence, tier=tier,
        score_breakdown={"manual_pick": 1.0,
                          "validation_pass": float(all_pass)},
        reason="manual GT pick (3D label)",
    )


def _impossible(reason: str, cid: int, gates=None, **extra) -> PickResult:
    return PickResult(success=False, status=STATUS_IMPOSSIBLE,
                      gate_failures=list(gates or []), reason=reason,
                      cid=cid, **extra)


def _defer(reason: str, cid: int, gates=None, **extra) -> PickResult:
    return PickResult(success=False, status=STATUS_DEFER,
                      gate_failures=list(gates or []), reason=reason,
                      cid=cid, **extra)


def _scene_other_mask(scene_polygons, w=IMG_W, h=IMG_H) -> np.ndarray:
    """Union mask of OTHER polygons in the same shot (S1 occluder check)."""
    if not scene_polygons:
        return np.zeros((h, w), dtype=bool)
    union = np.zeros((h, w), dtype=bool)
    for poly in scene_polygons:
        union |= polygon_to_mask(poly, w, h)
    return union


def _has_flat_patch(grid: np.ndarray, valid: np.ndarray,
                    self_mask: np.ndarray, threshold: float = 0.90) -> bool:
    """폴리곤 안에 flatness ≥ threshold인 disk 후보가 1개 이상 있는가."""
    from .suction_score import normal_flatness_at
    ys, xs = np.where(self_mask)
    if len(xs) == 0:
        return False
    u_min, u_max = int(xs.min()), int(xs.max())
    v_min, v_max = int(ys.min()), int(ys.max())
    H, W = self_mask.shape
    for gi in range(5):
        for gj in range(5):
            cu = u_min + (u_max - u_min) * (gi + 0.5) / 5
            cv = v_min + (v_max - v_min) * (gj + 0.5) / 5
            ui, vi = int(round(cu)), int(round(cv))
            if not (0 <= ui < W and 0 <= vi < H):
                continue
            if not self_mask[vi, ui]:
                continue
            s = normal_flatness_at(grid, valid, (cu, cv), cup_radius_px=12.0)
            if s is not None and s >= threshold:
                return True
    return False


# ════════════════════════════════════════════════════════════════════════
# compute_pick = 6개 helper로 분할:
#   _check_class_and_depth  — 클래스 + 마스크 + 깊이 NaN (IMPOSSIBLE 조기 종료)
#   _compute_initial_pose   — Zivid 평균 normal + tilt + top-surface position
#                             + PCA long_axis (DEFER on tilt fail)
#   _refine_pose_with_heatmap — heatmap argmax × 다중 yaw + arm 사후 필터
#   _check_structural       — 길이 / S3 평면 / occlusion (IMPOSSIBLE)
#   _check_defer_gates      — S1 overlap / cup support / cup coverage / box ROI /
#                             rect planarity / S2 arm + cup protrusion / cup bump
#   _build_pickable_result  — confidence + tier + PickResult 구성
#
# 모든 helper는 ctx (dict)를 in-place 갱신. 메인 compute_pick은 흐름 제어만.
# ════════════════════════════════════════════════════════════════════════


def _check_class_and_depth(ctx: dict) -> Optional[PickResult]:
    """클래스 / 폴리곤 마스크 / 깊이 NaN ratio 검사. IMPOSSIBLE이면 PickResult 반환.

    cid 허용:
      - 0 ~ len(CLASSES)-1: 학습된 5종. EXCLUDE_CIDS 검사 + 클래스별 분기.
      - UNKNOWN_CID (-1): OOD/unknown. 일반 흡착 픽 로직으로 진행
        (bottle 특화 분기 skip, class_prior 중립).
      - 그 외: 비정상 → IMPOSSIBLE.
    """
    cid = ctx["cid"]
    is_known = 0 <= cid < len(CLASSES)
    if not is_known and cid != UNKNOWN_CID:
        return _impossible(f"비정상 클래스 id={cid}", cid)
    if is_known and cid in EXCLUDE_CIDS:
        return _impossible(
            f"클래스 '{CLASSES[cid]}' 제외 대상 (투명/곡면)  "
            f"← classes.exclude_cids", cid,
            gates=["class_excluded"])

    mask = polygon_to_mask(ctx["polygon_2d"])
    mask_total = int(mask.sum())
    if mask_total == 0:
        return _impossible("폴리곤 마스크가 비어있음 (점 부족)", cid)

    grid = ctx["grid"]
    xyz = np.stack([grid["x"], grid["y"], grid["z"]], axis=-1)
    valid = ~np.isnan(xyz[..., 2])
    use = mask & valid
    n_valid = int(use.sum())
    nan_ratio_before = 1.0 - (n_valid / mask_total)

    # ── Depth recovery: NaN 비율이 trigger 이상이면 보간/원기둥 fit으로 채움
    recovered_from = None
    recovery_method = None
    if RECOVERY_ENABLE and nan_ratio_before >= RECOVERY_NAN_TRIGGER:
        from . import depth_recovery as _dr
        new_grid, n_filled = grid, 0
        # bottle + cylinder 옵션 → Method C 먼저 시도
        if cid == BOTTLE_CID and RECOVERY_BOTTLE_USE_CYL:
            new_grid_c, n_c, info_c = _dr.recover_xyz_in_mask_cylinder(
                grid, mask, FX, FY, CX, CY,
                radius_min_mm=RECOVERY_CYL_R_MIN,
                radius_max_mm=RECOVERY_CYL_R_MAX,
                min_valid_for_fit=RECOVERY_CYL_MIN_PTS,
            )
            if n_c > 0:
                new_grid, n_filled = new_grid_c, n_c
                recovery_method = f"cylinder(r={info_c['radius_mm']:.1f}mm)"
        # Method C 실패 또는 다른 클래스 → Method A (선형 보간)
        if n_filled == 0:
            new_grid_a, n_a = _dr.recover_xyz_in_mask(
                grid, mask, FX, FY, CX, CY,
            )
            if n_a > 0:
                new_grid, n_filled = new_grid_a, n_a
                recovery_method = "linear"
        if n_filled > 0:
            grid = new_grid
            xyz = np.stack([grid["x"], grid["y"], grid["z"]], axis=-1)
            valid = ~np.isnan(xyz[..., 2])
            use = mask & valid
            n_valid = int(use.sum())
            recovered_from = nan_ratio_before
            ctx["grid"] = grid

    nan_ratio = 1.0 - (n_valid / mask_total)
    nan_thr = NAN_RATIO_REJECT_BOTTLE if cid == BOTTLE_CID else NAN_RATIO_REJECT
    # 복원 시도했으면 max_nan_ratio_after 게이트도 적용
    if recovered_from is not None and nan_ratio > RECOVERY_NAN_MAX_AFTER:
        return _impossible(
            f"복원 후에도 깊이 신뢰 부족 "
            f"(NaN {nan_ratio:.0%} > {RECOVERY_NAN_MAX_AFTER:.0%})  "
            f"← recovery.max_nan_ratio_after",
            cid, gates=["S4_valid_ratio"], n_valid_points=n_valid)
    if nan_ratio > nan_thr:
        key = "s4_nan_ratio_reject_bottle" if cid == BOTTLE_CID else "s4_nan_ratio_reject"
        return _impossible(
            f"깊이 신뢰 부족 (NaN {nan_ratio:.0%} > {nan_thr:.0%})  "
            f"← gates.{key}",
            cid, gates=["S4_valid_ratio"], n_valid_points=n_valid)

    points = xyz[use].astype(np.float64)
    n = len(points)
    if n < 30:
        return _impossible(f"유효 3D 점 부족 ({n} < 30개)", cid,
                           gates=["S4_valid_ratio"], n_valid_points=n)

    ctx.update({
        "mask": mask, "mask_total": mask_total, "valid": valid,
        "use": use, "points": points, "n": n, "nan_ratio": nan_ratio,
        "recovered_from": recovered_from,   # None이면 복원 안 함
        "recovery_method": recovery_method, # 'cylinder(...)' / 'linear' / None
    })
    return None


def _detect_cap_plane(
    mask: np.ndarray, valid: np.ndarray, grid: np.ndarray, polygon_2d,
    end_fraction: float = 0.25,
    ransac_inlier_mm: float = 2.0,
    ransac_iters: int = 5,
    max_plane_thickness_mm: float = 2.5,
    min_inliers: int = 50,
    min_ply_normal_consistency: float = 0.75,  # cap normal과 PLY normal cos 최소
    min_valid_density: float = 0.5,             # end-region valid 픽셀 비율 최소
) -> Optional[dict]:
    """Bottle 폴리곤의 한쪽 끝(뚜껑) 평면 detect.

    물병은 강체 + 뚜껑은 불투명 plastic → 깊이 정확. 사용자 의도: 뚜껑 평면을
    찾아서 그 normal을 cylinder axis로, 평면 위 ellipse minor axis를 radius로
    사용. 측면이 투명해도 prior로 cylinder 본체 재구성 가능.

    흐름:
      1) polygon 2D long axis 추출
      2) long axis 양 끝 end_fraction 영역에서 valid 3D 점 cluster 2개
      3) 각 cluster RANSAC 평면 fit (±ransac_inlier_mm 잔차)
      4) 두 cluster 신뢰도 점수화:
         - plane_thickness_mm < max_plane_thickness_mm  (평면다움)
         - PLY normal과 fit normal cos > min_ply_normal_consistency  (평면 영역)
           ← cylinder 측면 곡면은 normal이 방사형 → cos 낮음 → 거름
         - valid_density > min_valid_density  (불투명 영역)
           ← bottle 측면은 투명/반사로 NaN 많음. cap만 100% valid.
      5) 통과한 후보 중 (normal_consistency × valid_density / plane_thickness)
         점수 가장 높은 쪽 = 뚜껑
      6) 둘 다 fail → None

    Returns:
      dict {axis_dir, centroid, radius_mm, n_inliers, plane_thickness_mm,
            inlier_yx, normal_consistency, valid_density}
      None: 뚜껑 detect 실패 (이 경우 호출자가 IMPOSSIBLE 처리)
    """
    use = mask & valid
    ys_use, xs_use = np.where(use)
    if len(ys_use) < min_inliers * 2:
        return None

    # polygon 2D long axis (PCA)
    poly_pts = np.array(polygon_2d, dtype=np.float64)
    if len(poly_pts) < 3:
        return None
    poly_centroid = poly_pts.mean(axis=0)
    pc = poly_pts - poly_centroid
    cov_2d = pc.T @ pc
    _, evecs_2d = np.linalg.eigh(cov_2d)
    long_axis_2d = evecs_2d[:, -1]   # 큰 eigval = long axis

    # valid pixel을 long axis로 project
    pix = np.stack([xs_use.astype(np.float64),
                    ys_use.astype(np.float64)], axis=1)
    proj_long = (pix - poly_centroid) @ long_axis_2d
    p_lo = float(np.percentile(proj_long, end_fraction * 100))
    p_hi = float(np.percentile(proj_long, (1 - end_fraction) * 100))

    def _ransac_plane(idx_mask):
        if int(idx_mask.sum()) < min_inliers:
            return None
        yc = ys_use[idx_mask]; xc = xs_use[idx_mask]
        pts = np.stack(
            [grid["x"][yc, xc], grid["y"][yc, xc], grid["z"][yc, xc]],
            axis=-1,
        ).astype(np.float64)
        # 각 점의 PLY surface normal (cap 평면 normal과 일치해야)
        ply_nx = grid["nx"][yc, xc].astype(np.float64)
        ply_ny = grid["ny"][yc, xc].astype(np.float64)
        ply_nz = grid["nz"][yc, xc].astype(np.float64)
        # 초기 PCA + RANSAC 정제
        P0 = pts.mean(0)
        centered = pts - P0
        cov = (centered.T @ centered) / max(len(pts) - 1, 1)
        _, evecs = np.linalg.eigh(cov)
        normal = evecs[:, 0]
        if normal[2] > 0:
            normal = -normal
        # RANSAC 정제 — 평면 잔차만으로 inlier (PLY normal noise 있어 점별 검증
        # 안 함, 최종에 평균만 검증)
        inlier = np.abs((pts - P0) @ normal) < ransac_inlier_mm
        for _ in range(ransac_iters):
            if int(inlier.sum()) < min_inliers:
                return None
            in_pts = pts[inlier]
            P0 = in_pts.mean(0)
            c_in = in_pts - P0
            cov_in = (c_in.T @ c_in) / max(len(in_pts) - 1, 1)
            evals_in, evecs_in = np.linalg.eigh(cov_in)
            normal = evecs_in[:, 0]
            if normal[2] > 0:
                normal = -normal
            inlier = np.abs((pts - P0) @ normal) < ransac_inlier_mm
        if int(inlier.sum()) < min_inliers:
            return None
        in_pts = pts[inlier]
        c_in = in_pts - P0
        cov_in = (c_in.T @ c_in) / max(len(in_pts) - 1, 1)
        evals_in, evecs_in = np.linalg.eigh(cov_in)
        normal = evecs_in[:, 0]
        if normal[2] > 0:
            normal = -normal
        thickness = float(np.std(c_in @ normal))
        # PLY normal 평균 vs fit normal — 평면 영역이면 cos≈1, 측면 곡면이면 작음.
        # 항상 계산해서 candidate ranking에 사용. min_ply_normal_consistency 미달이면
        # cluster 자체 탈락.
        cos_match = 0.0
        in_nx = ply_nx[inlier]; in_ny = ply_ny[inlier]; in_nz = ply_nz[inlier]
        f_mask = np.isfinite(in_nx) & np.isfinite(in_ny) & np.isfinite(in_nz)
        if int(f_mask.sum()) >= 10:
            mnx = float(in_nx[f_mask].mean())
            mny = float(in_ny[f_mask].mean())
            mnz = float(in_nz[f_mask].mean())
            mmag = (mnx*mnx + mny*mny + mnz*mnz) ** 0.5
            if mmag > 1e-6:
                cos_match = abs((mnx/mmag)*normal[0] + (mny/mmag)*normal[1]
                                  + (mnz/mmag)*normal[2])
        if min_ply_normal_consistency > 0 and cos_match < min_ply_normal_consistency:
            return None
        # ellipse minor axis = perspective 안 받은 실제 반지름
        a1 = evecs_in[:, 2]
        a2 = evecs_in[:, 1]
        r1 = c_in @ a1
        r2 = c_in @ a2
        std1, std2 = float(np.std(r1)), float(np.std(r2))
        minor_p95 = float(np.percentile(np.abs(r1 if std1 < std2 else r2), 95))
        # inlier 픽셀 좌표 (view_3d 시각화용 — 진짜 뚜껑으로 판정된 점들)
        in_y = yc[inlier].tolist()
        in_x = xc[inlier].tolist()
        return {
            "axis_dir": normal, "centroid": P0,
            "radius_mm": minor_p95,
            "n_inliers": int(len(in_pts)),
            "plane_thickness_mm": thickness,
            "inlier_yx": [in_y, in_x],
            "normal_consistency": cos_match,
        }

    lo_mask = proj_long <= p_lo
    hi_mask = proj_long >= p_hi
    lo = _ransac_plane(lo_mask)
    hi = _ransac_plane(hi_mask)

    # end-region valid density 계산 — bottle 측면은 투명/반사로 NaN 많음, cap은
    # 100% valid. end_fraction 영역의 polygon-pixel 중 valid 비율을 cap 신뢰도
    # 신호로. (use는 mask & valid라 valid 픽셀만 들었지만, 우리는 폴리곤 전체
    # 영역의 density가 필요 → poly_mask와 valid_full 별도 계산.)
    def _end_valid_density(end_pix_idx):
        """end region 픽셀들의 valid 비율 (이미 use=mask&valid라 별도 계산 필요)."""
        # use indices가 valid한 픽셀들. end region 안 use 점 수.
        n_use_in_end = int(end_pix_idx.sum())
        # poly 전체 픽셀 중 같은 end region (long axis 투영 비교)에 들어가는 픽셀 수
        # 빠른 근사 — poly_mask 안 픽셀들의 long axis 분포로 동일 percentile 영역
        ys_p, xs_p = np.where(mask)
        if len(ys_p) == 0: return 0.0
        pix_p = np.stack([xs_p.astype(np.float64),
                          ys_p.astype(np.float64)], axis=1)
        proj_p = (pix_p - poly_centroid) @ long_axis_2d
        if (proj_long <= p_lo).any() and end_pix_idx is lo_mask:
            poly_end = proj_p <= p_lo
        else:
            poly_end = proj_p >= p_hi
        n_poly_in_end = int(poly_end.sum())
        if n_poly_in_end == 0: return 0.0
        return float(n_use_in_end / n_poly_in_end)

    if lo is not None:
        lo["valid_density"] = _end_valid_density(lo_mask)
    if hi is not None:
        hi["valid_density"] = _end_valid_density(hi_mask)

    # 측면 top line(누운 cylinder의 카메라 쪽 곡면)이 평면처럼 fit되는 케이스
    # 제거: plane normal이 카메라 시선(+Z) 방향에 너무 가까우면(|z|>0.85)
    # cylinder 측면이지 cap이 아님. 진짜 cap normal은 cylinder axis 방향이라
    # 누운 자세면 |z|<0.5, 세운 자세면 |z|>0.85 (=세운 자세는 별도 IMPOSSIBLE).
    def _is_real_cap(c):
        if c is None: return False
        if c["plane_thickness_mm"] >= max_plane_thickness_mm: return False
        if abs(float(c["axis_dir"][2])) > 0.85: return False
        if c.get("valid_density", 1.0) < min_valid_density: return False
        return True
    candidates = [c for c in (lo, hi) if _is_real_cap(c)]
    if not candidates:
        return None

    # 점수 기반 선택: normal_consistency × valid_density / max(plane_thickness, 0.3)
    # — 평면답고, 픽셀 잘 보이고, 두께 얇을수록 cap에 가까움.
    def _cap_score(c):
        nc = float(c.get("normal_consistency", 0.0))
        vd = float(c.get("valid_density", 1.0))
        th = max(float(c["plane_thickness_mm"]), 0.3)
        return nc * vd / th
    return max(candidates, key=_cap_score)


def _fit_disc_plane(
    mask: np.ndarray, valid: np.ndarray, grid: np.ndarray,
    ransac_inlier_mm: float = 2.0,
    ransac_iters: int = 5,
    min_inliers: int = 100,
) -> Optional[dict]:
    """폴리곤이 위에서 본 disc 모양일 때 — 폴리곤 전체 valid 3D 점에 RANSAC
    평면 fit. cap detect와 다른 점: 양 끝 25% 영역 아닌 전체 사용.

    bottle이 수직으로 서있거나 거꾸로 놓여서 cap top 또는 bottom이 카메라에
    원형 disc로 보이는 경우 — 그 disc 평면을 직접 fit하면 정확.

    Returns:
      dict {axis_dir, centroid, radius_mm, n_inliers, plane_thickness_mm,
            inlier_yx, long_axis, mid_axis}
      None: fit 실패
    """
    use = mask & valid
    ys, xs = np.where(use)
    if len(ys) < min_inliers:
        return None
    pts = np.stack([grid["x"][ys, xs], grid["y"][ys, xs], grid["z"][ys, xs]],
                    axis=-1).astype(np.float64)
    P0 = pts.mean(0)
    centered = pts - P0
    cov = (centered.T @ centered) / max(len(pts) - 1, 1)
    _, evecs = np.linalg.eigh(cov)
    normal = evecs[:, 0]    # 가장 작은 eigval = plane normal
    if normal[2] > 0:
        normal = -normal    # 카메라 향함 (-Z)
    inlier = np.abs((pts - P0) @ normal) < ransac_inlier_mm
    for _ in range(ransac_iters):
        if int(inlier.sum()) < min_inliers:
            return None
        in_pts = pts[inlier]
        P0 = in_pts.mean(0)
        c_in = in_pts - P0
        cov_in = (c_in.T @ c_in) / max(len(in_pts) - 1, 1)
        evals_in, evecs_in = np.linalg.eigh(cov_in)
        normal = evecs_in[:, 0]
        if normal[2] > 0:
            normal = -normal
        inlier = np.abs((pts - P0) @ normal) < ransac_inlier_mm
    if int(inlier.sum()) < min_inliers:
        return None
    in_pts = pts[inlier]
    c_in = in_pts - P0
    cov_in = (c_in.T @ c_in) / max(len(in_pts) - 1, 1)
    evals_in, evecs_in = np.linalg.eigh(cov_in)
    normal = evecs_in[:, 0]
    if normal[2] > 0:
        normal = -normal
    thickness = float(np.std(c_in @ normal))
    # disc 평면 안의 long/mid axis (PCA eigvec 큰 두 개)
    long_axis = evecs_in[:, 2]
    mid_axis = evecs_in[:, 1]
    # disc radius (95-percentile larger direction)
    r_large = c_in @ long_axis
    radius_mm = float(np.percentile(np.abs(r_large), 95))
    return {
        "axis_dir": normal, "centroid": P0,
        "radius_mm": radius_mm,
        "n_inliers": int(len(in_pts)),
        "plane_thickness_mm": thickness,
        "inlier_yx": [ys[inlier].tolist(), xs[inlier].tolist()],
        "long_axis": long_axis, "mid_axis": mid_axis,
    }


def _compute_bottle_pose(ctx: dict) -> Optional[PickResult]:
    """Bottle 전용 — 뚜껑(cap) 평면 detect 기반 cylinder pose.

    사용자 의도: 뚜껑은 불투명 plastic → 깊이 정확. 그 평면 detect로
    cylinder axis + radius 신뢰 있게 추출. 측면이 투명해도 prior로 본체 재구성.

    흐름:
      1) _detect_cap_plane으로 뚜껑 detect — 못 찾으면 IMPOSSIBLE
      2) cap normal = cylinder axis 방향
      3) cap inlier ellipse minor axis = 실제 radius (perspective 안 받은)
      4) axis 부호 = cylinder body가 뚜껑 어느 쪽에 있는지 (mask 점 분포)
      5) axis·Z > BOTTLE_AXIS_Z_MAX → 세운 자세 (측면 흡착 불가, IMPOSSIBLE)
      6) Pick pose: cylinder body 중간 측면, 카메라 쪽 표면
    """
    cid, n = ctx["cid"], ctx["n"]
    points = ctx["points"]
    if n < BOTTLE_MIN_VALID_FOR_FIT:
        return _impossible(
            f"bottle cylinder fit 위한 3D 점 부족 "
            f"({n} < {BOTTLE_MIN_VALID_FOR_FIT})",
            cid, gates=["bottle_cylinder_fit"], n_valid_points=n)

    # ─── 자세 자동 분기 — polygon aspect로 disc(top-down) vs 측면 누운 자세 판별
    # bottle이 수직 또는 거꾸로 놓여서 cap top/bottom이 위에서 둥근 disc로 보이면
    # polygon aspect ≈ 1. 누운 cylinder는 길쭉한 직사각형 (aspect > 2).
    poly_pts = np.array(ctx["polygon_2d"], dtype=np.float64)
    if len(poly_pts) >= 3:
        pc2 = poly_pts - poly_pts.mean(axis=0)
        evals2, _ = np.linalg.eigh(pc2.T @ pc2)
        poly_aspect = float(np.sqrt(max(evals2[1], 1e-9)
                                     / max(evals2[0], 1e-9)))
    else:
        poly_aspect = 999.0
    DISC_ASPECT_MAX = 2.0
    if poly_aspect < DISC_ASPECT_MAX:
        # 위에서 본 disc 자세 — 폴리곤 전체에 평면 fit, cap이든 bottom이든 평면
        # disc에 single cup top-down.
        disc = _fit_disc_plane(ctx["mask"], ctx["valid"], ctx["grid"])
        if disc is None:
            return _impossible(
                f"bottle disc 평면 fit 실패 (aspect={poly_aspect:.2f}, "
                f"valid 점 부족 또는 평면 아님)",
                cid, gates=["bottle_disc_fit"], n_valid_points=n)
        # disc normal cos_z — top-down 접근 가능해야. cos_z = |normal[2]|.
        # 1 = 정확히 카메라 향함 (이상), 0 = 직각 (불가).
        cos_z = abs(float(disc["axis_dir"][2]))
        if cos_z < 0.7:
            return _impossible(
                f"bottle disc 비스듬 (cos_z={cos_z:.2f} < 0.7) — "
                f"top-down 흡착 어려움",
                cid, gates=["bottle_disc_tilt"], n_valid_points=n)
        normal_v = disc["axis_dir"] / max(float(np.linalg.norm(disc["axis_dir"])), 1e-12)
        long_axis = disc["long_axis"] / max(float(np.linalg.norm(disc["long_axis"])), 1e-12)
        mid_axis = disc["mid_axis"] / max(float(np.linalg.norm(disc["mid_axis"])), 1e-12)
        position = disc["centroid"] + SAFETY_OFFSET_MM * normal_v
        ctx.update({
            "position": position, "normal": normal_v,
            "long_axis": long_axis, "mid_axis": mid_axis,
            "eigvals": np.array([0.0, 0.0, float(disc["radius_mm"])]),
            "initial_position": position.copy(),
            "bottle_fit": {
                "method": "disc",                    # 자세 분기 표시
                "radius_mm": float(disc["radius_mm"]),
                "axis_z": cos_z,
                "cap_radius_mm": float(disc["radius_mm"]),
                "cap_thickness_mm": float(disc["plane_thickness_mm"]),
                "cap_inliers": int(disc["n_inliers"]),
                "cap_centroid": disc["centroid"].tolist(),
                "cap_inlier_yx": disc.get("inlier_yx"),
                "axis_dir": normal_v.tolist(),
                "poly_aspect": poly_aspect,
            },
        })
        return None

    # ─── 측면 cylinder 자세 (길쭉한 폴리곤) — 기존 cap detect 로직
    # 1) Cap detect — 신뢰 게이트. 뚜껑 못 찾으면 IMPOSSIBLE.
    # (사용자 의도: 뚜껑 = 신뢰의 근거. 없으면 잡지 말기.)
    cap = _detect_cap_plane(
        ctx["mask"], ctx["valid"], ctx["grid"], ctx["polygon_2d"],
    )
    if cap is None:
        return _impossible(
            "bottle 뚜껑 평면을 찾을 수 없음 "
            "(폴리곤 양 끝 모두 평면 fit 실패)  ← bottle.enable_cylinder_pose",
            cid, gates=["bottle_cap_detect"], n_valid_points=n)

    # 2) 자세 검증 — cap normal Z 성분으로 누운/세운 판단
    if abs(float(cap["axis_dir"][2])) > BOTTLE_AXIS_Z_MAX:
        return _impossible(
            f"bottle 세운 자세 (axis·Z={abs(cap['axis_dir'][2]):.2f} > "
            f"{BOTTLE_AXIS_Z_MAX}) — 측면 흡착 불가  ← bottle.axis_z_max",
            cid, gates=["bottle_upright"], n_valid_points=n)

    # 3) cap normal과 free cylinder fit axis 일치성 검증.
    # 둘이 일치하면(cosine > 0.85) 진짜 cylinder axis. 안 일치하면 둘 중 하나
    # 잘못된 거 → 신뢰 못함 → IMPOSSIBLE.
    try:
        from .depth_recovery import _fit_cylinder_pca
        P0, free_axis, free_radius = _fit_cylinder_pca(points)
    except Exception as e:
        return _impossible(
            f"bottle cylinder fit 실패 ({e})", cid,
            gates=["bottle_cylinder_fit"], n_valid_points=n)
    cap_axis = cap["axis_dir"]
    cos_consistency = abs(float(cap_axis @ free_axis))
    if cos_consistency < 0.75:
        return _impossible(
            f"bottle cap normal과 free fit axis 불일치 "
            f"(cos={cos_consistency:.2f} < 0.75) — 신뢰 부족",
            cid, gates=["bottle_axis_disagree"], n_valid_points=n)
    cap_centroid = cap["centroid"]

    # A안 — polygon long axis hint로 cap normal 강제 정렬.
    # 누운 cylinder의 카메라 가시 영역(최대 반원)으로는 axis가 axis 회전에
    # 자유도 1 남음 → cap fit normal이 outlier로 살짝 기울 수 있음.
    # 폴리곤은 사용자가 그린 정확한 외곽선이므로 long axis가 진짜 cylinder axis
    # 의 image plane 투영. cap normal의 (nx, ny) 방향을 그쪽으로 강제,
    # nz(누움 정도)는 cap fit 그대로 유지.
    arr2d_p = np.array(ctx["polygon_2d"], dtype=np.float64)
    if len(arr2d_p) >= 3:
        pc_p = arr2d_p - arr2d_p.mean(axis=0)
        _, evecs_pp = np.linalg.eigh(pc_p.T @ pc_p)
        long_2d_p = evecs_pp[:, -1]
        long_2d_p = long_2d_p / max(float(np.linalg.norm(long_2d_p)), 1e-12)
        nx, ny, nz = float(cap_axis[0]), float(cap_axis[1]), float(cap_axis[2])
        xy_mag = float(np.sqrt(nx * nx + ny * ny))
        if xy_mag > 1e-6:
            # cap normal의 xy 방향과 polygon long_2d_p 부호 맞춤
            sign = 1.0 if (nx * long_2d_p[0] + ny * long_2d_p[1]) >= 0 else -1.0
            new_nx = long_2d_p[0] * xy_mag * sign
            new_ny = long_2d_p[1] * xy_mag * sign
            aligned = np.array([new_nx, new_ny, nz], dtype=np.float64)
            aligned = aligned / max(float(np.linalg.norm(aligned)), 1e-12)
            cap_axis = aligned

    # 사용자 의도: 뚜껑 평면이 모든 기준.
    #   long_axis = polygon-aligned cap normal (= cylinder axis)
    #   cap centroid = cylinder의 한쪽 끝 (뚜껑 위치)
    long_axis = cap_axis.copy()
    # axis 부호: mask 점들이 cap centroid 어느 쪽에 더 많이 분포 = cylinder body
    proj_axis = (points - cap_centroid) @ long_axis
    if int((proj_axis > 0).sum()) < int((proj_axis < 0).sum()):
        long_axis = -long_axis
        proj_axis = -proj_axis
    long_axis = long_axis / max(float(np.linalg.norm(long_axis)), 1e-12)
    cos_axis_z = abs(float(long_axis[2]))
    if cos_axis_z > BOTTLE_AXIS_Z_MAX:
        return _impossible(
            f"bottle 세운 자세 (axis·Z={cos_axis_z:.2f})  "
            f"← bottle.axis_z_max",
            cid, gates=["bottle_upright"], n_valid_points=n)
    # radius = cap detect의 ellipse minor axis (perspective 안 받은 진짜 반경)
    radius = float(cap["radius_mm"])
    if not (RECOVERY_CYL_R_MIN <= radius <= RECOVERY_CYL_R_MAX):
        return _impossible(
            f"bottle 반경 prior 밖 ({radius:.0f}mm)  "
            f"← recovery.cylinder_radius_min/max_mm",
            cid, gates=["bottle_radius"], n_valid_points=n)
    # cylinder length: 폴리곤 2D long axis 픽셀 거리 × z/fx로 추정
    # (mask valid 점 분포는 outlier로 inflate. 폴리곤 길이가 더 robust.)
    arr2d = np.array(ctx["polygon_2d"], dtype=np.float64)
    pc2 = arr2d - arr2d.mean(0)
    _, evecs2 = np.linalg.eigh(pc2.T @ pc2)
    long_2d = evecs2[:, -1]
    poly_long_px = float(np.percentile(pc2 @ long_2d, 95) -
                         np.percentile(pc2 @ long_2d, 5))
    z_med = float(np.median(points[:, 2]))
    fx_avg = 0.5 * (FX + FY)
    cyl_length = poly_long_px * z_med / fx_avg
    if cyl_length < SUCTION_CUP_PITCH_MM:
        return _impossible(
            f"bottle body 길이 부족 ({cyl_length:.0f}mm < "
            f"{SUCTION_CUP_PITCH_MM}mm pitch)",
            cid, gates=["bottle_length"], n_valid_points=n)
    # 사용자 의도: 뚜껑이 cylinder의 한 끝, body는 axis 따라 평행이동.
    # 흡착 위치 = cylinder body 측면 (cap에서 axis 따라 안쪽).
    #
    # 정확한 cylinder model로 position 잡으면 length 측정 오차에 민감 →
    # axis 따라 멀리 가면 polygon 영역 밖으로. 대신:
    #   normal = cap normal에 수직 카메라 쪽 (cylinder 측면 surface normal)
    #   position = 실측 top surface mean (mask 안 카메라 쪽 5%, 즉 카메라가
    #              실제로 본 cylinder 측면 위 점들)
    # 이 방법은 long_axis = cap normal과 어울리고, 실측 surface라 정확.
    # cylinder 측면 perp 방향은 cap centroid에서 카메라 쪽:
    along_cap = float(cap_centroid @ long_axis)
    perp_cap = cap_centroid - along_cap * long_axis
    perp_n = float(np.linalg.norm(perp_cap))
    if perp_n < 1e-6:
        return _impossible(
            "bottle cap이 카메라 ray와 일치 (degenerate)",
            cid, gates=["bottle_geom"], n_valid_points=n)
    cam_to_axis_dir = perp_cap / perp_n
    normal = -cam_to_axis_dir   # 표면 → 카메라 쪽

    # 실측 top surface mean (mask 안 카메라 쪽 5%, body 영역 우선)
    body_pts = points[proj_axis > 0] if int((proj_axis > 0).sum()) > 30 else points
    proj_to_cam = body_pts @ normal
    top_thr = float(np.percentile(proj_to_cam, 95))
    top_band = proj_to_cam >= (top_thr - 5.0)
    top_pts = body_pts[top_band]
    if len(top_pts) < 5:
        top_pts = body_pts
    surface_anchor_real = top_pts.mean(axis=0)
    # axis 방향은 mask centroid 정렬 (cylinder body 중간 부근)
    centroid_pts = points.mean(axis=0)
    delta_along = float((centroid_pts - surface_anchor_real) @ long_axis)
    surface_aligned = surface_anchor_real + delta_along * long_axis
    position = surface_aligned + SAFETY_OFFSET_MM * normal
    mid_axis = np.cross(normal, long_axis)
    mid_axis = mid_axis / max(float(np.linalg.norm(mid_axis)), 1e-12)

    eigvals = np.array([0.0, 0.0, radius])
    # cylinder body 추정 (시각화용) — cap centroid에서 axis 따라 polygon long
    # axis 픽셀 거리만큼 평행이동한 cylinder.
    cyl_end = cap_centroid + cyl_length * long_axis
    ctx.update({
        "position": position, "normal": normal,
        "long_axis": long_axis, "mid_axis": mid_axis,
        "eigvals": eigvals,
        "initial_position": position.copy(),
        "bottle_fit": {
            "radius_mm": radius,
            "axis_z": cos_axis_z,
            "cap_radius_mm": float(cap["radius_mm"]),
            "cap_thickness_mm": float(cap["plane_thickness_mm"]),
            "cap_inliers": int(cap["n_inliers"]),
            "cap_normal_consistency": float(cap.get("normal_consistency", 0.0)),
            "cap_valid_density": float(cap.get("valid_density", 1.0)),
            "cap_centroid": cap_centroid.tolist(),
            "cap_inlier_yx": cap.get("inlier_yx"),   # view_3d 짙은 색 paint용
            "cyl_length": float(cyl_length),
            "cyl_end": cyl_end.tolist(),    # 시각화용
            "axis_dir": long_axis.tolist(),
        },
    })
    return None


def _compute_bottle_pose_pca(ctx: dict) -> Optional[PickResult]:
    """실험 — bottle: cap PCA + 본체 방향 30mm offset 단순화 픽 자세.

    docs/병_뚜껑_PCA_offset_고려사항.md 참조.
    `bottle.use_cap_pca_offset: true` 일 때 _compute_initial_pose 가 라우팅.

    흐름 (5줄):
      1) _detect_cap_plane → cap 평면 (centroid, normal)
      2) 자세 검증 (axis_z_max)
      3) body 방향 부호 = mask 점 더 많은 쪽
      4) pick anchor = cap_centroid + BODY_OFFSET * body_dir
      5) surface normal = anchor 에서 axis 수직 + 카메라 쪽
    """
    cid, n = ctx["cid"], ctx["n"]
    points = ctx["points"]
    if n < BOTTLE_MIN_VALID_FOR_FIT:
        return _impossible(
            f"bottle cap PCA: 3D 점 부족 ({n} < {BOTTLE_MIN_VALID_FOR_FIT})",
            cid, gates=["bottle_cylinder_fit"], n_valid_points=n)

    cap = _detect_cap_plane(
        ctx["mask"], ctx["valid"], ctx["grid"], ctx["polygon_2d"],
    )
    if cap is None:
        return _impossible(
            "bottle cap PCA: 뚜껑 평면 detect 실패  "
            "← bottle.use_cap_pca_offset",
            cid, gates=["bottle_cap_detect"], n_valid_points=n)

    cap_centroid = np.asarray(cap["centroid"], dtype=np.float64)
    cap_axis = np.asarray(cap["axis_dir"], dtype=np.float64)
    cap_axis = cap_axis / max(float(np.linalg.norm(cap_axis)), 1e-12)

    if abs(float(cap_axis[2])) > BOTTLE_AXIS_Z_MAX:
        return _impossible(
            f"bottle cap PCA: 세운 자세 (|axis·Z|={abs(cap_axis[2]):.2f} > "
            f"{BOTTLE_AXIS_Z_MAX}) — 측면 흡착 불가  ← bottle.axis_z_max",
            cid, gates=["bottle_upright"], n_valid_points=n)

    # Body 방향 부호: mask 점들 axis 투영 분포로 결정 (cap 바깥 vs cylinder body)
    proj = (points - cap_centroid) @ cap_axis
    body_dir = cap_axis if int((proj > 0).sum()) >= int((proj < 0).sum()) else -cap_axis

    pick_anchor = cap_centroid + BOTTLE_BODY_OFFSET_MM * body_dir

    # Cylinder 측면 surface normal at anchor = axis 수직 + 카메라 쪽.
    along = float(pick_anchor @ body_dir)
    perp = pick_anchor - along * body_dir
    perp_n = float(np.linalg.norm(perp))
    if perp_n < 1e-6:
        return _impossible(
            "bottle cap PCA: anchor 가 카메라 ray 와 일치 (degenerate)",
            cid, gates=["bottle_geom"], n_valid_points=n)
    normal = -perp / perp_n   # 표면 → 카메라

    position = pick_anchor + SAFETY_OFFSET_MM * normal
    long_axis = body_dir
    mid_axis = np.cross(normal, long_axis)
    m_n = float(np.linalg.norm(mid_axis))
    if m_n < 1e-6:
        return _impossible(
            "bottle cap PCA: mid_axis degenerate",
            cid, gates=["bottle_geom"], n_valid_points=n)
    mid_axis = mid_axis / m_n

    radius = float(cap["radius_mm"])
    if not (RECOVERY_CYL_R_MIN <= radius <= RECOVERY_CYL_R_MAX):
        return _impossible(
            f"bottle cap PCA: 반경 prior 밖 ({radius:.0f}mm)  "
            f"← recovery.cylinder_radius_min/max_mm",
            cid, gates=["bottle_radius"], n_valid_points=n)

    eigvals = np.array([0.0, 0.0, radius])
    # 시각화용 cyl_end: anchor 에서 body 방향으로 100mm (overlay 길이)
    cyl_end = cap_centroid + 100.0 * body_dir
    ctx.update({
        "position": position, "normal": normal,
        "long_axis": long_axis, "mid_axis": mid_axis,
        "eigvals": eigvals,
        "initial_position": position.copy(),
        "bottle_fit": {
            "method": "cap_pca_offset",
            "radius_mm": radius,
            "cap_centroid": cap_centroid.tolist(),
            "body_offset_mm": BOTTLE_BODY_OFFSET_MM,
            "axis_dir": body_dir.tolist(),
            "cyl_end": cyl_end.tolist(),
            "pick_anchor": pick_anchor.tolist(),
        },
    })
    return None


def _compute_initial_pose(ctx: dict) -> Optional[PickResult]:
    """Zivid 평균 normal + tilt + top-surface position + PCA long_axis.
    Bottle 클래스는 cylinder fit pipeline으로 분기 (구조적으로 다른 자세)."""
    cid, n = ctx["cid"], ctx["n"]
    # Bottle: cylinder prior로 측면 dual-cup 자세
    if cid == BOTTLE_CID and cid not in EXCLUDE_CIDS:
        # 실험 모드 — cap PCA + body offset (단순화). 기본 OFF.
        if BOTTLE_USE_CAP_PCA:
            return _compute_bottle_pose_pca(ctx)
        if BOTTLE_CYLINDER_POSE_ENABLE:
            return _compute_bottle_pose(ctx)
    grid, use, points = ctx["grid"], ctx["use"], ctx["points"]

    nx_pix = grid["nx"][use]; ny_pix = grid["ny"][use]; nz_pix = grid["nz"][use]
    finite_n = np.isfinite(nx_pix) & np.isfinite(ny_pix) & np.isfinite(nz_pix)
    if finite_n.sum() < 30:
        return _impossible(f"유효 법선 부족 ({finite_n.sum()} < 30개)",
                           cid, gates=["normals"], n_valid_points=n)
    nx_avg = float(nx_pix[finite_n].mean())
    ny_avg = float(ny_pix[finite_n].mean())
    nz_avg = float(nz_pix[finite_n].mean())
    n_mag = float(np.sqrt(nx_avg ** 2 + ny_avg ** 2 + nz_avg ** 2))
    if n_mag < 1e-6:
        return _defer("평균 법선 계산 실패 (벡터 크기~0)", cid,
                      gates=["normals"], n_valid_points=n)
    normal = np.array([nx_avg, ny_avg, nz_avg]) / n_mag
    if normal[2] > 0:        # 카메라(-Z) 쪽 향하게 부호 보정
        normal = -normal

    # Tilt 한계
    cos_tilt = float(-normal[2])
    if cos_tilt < MAX_TILT_COS:
        tilt_deg = float(np.degrees(np.arccos(max(-1.0, min(1.0, cos_tilt)))))
        return _defer(
            f"표면 기울기 한계 초과 ({tilt_deg:.1f}° > {MAX_TILT_DEG}°)  "
            f"← approach.max_tilt_deg",
            cid, gates=["tilt_limit"], n_valid_points=n, normal=normal)

    # Top-surface points — outlier에 강건하게 잡기.
    # 객체 표면 z를 median으로 robust 추정 → 표면에서 ±50mm 밖 점은 outlier로
    # 제거(노이즈 스파이크 / 반사로 인한 거짓 가까움). 그 후 top percentile.
    proj_n = points @ normal
    surface_proj = float(np.median(proj_n))    # 객체 표면 normal-projection
    SURFACE_OUTLIER_MM = 50.0
    clean = (np.abs(proj_n - surface_proj) <= SURFACE_OUTLIER_MM)
    proj_n_clean = proj_n[clean]
    points_clean = points[clean]
    if len(proj_n_clean) < 10:    # 거의 다 outlier 취급되면 원본 사용 (안전망)
        proj_n_clean = proj_n
        points_clean = points
    cutoff = max(
        float(np.percentile(proj_n_clean, TOP_SURFACE_PERCENTILE)),
        float(proj_n_clean.max() - TOP_SURFACE_BAND_MM),
    )
    top_mask_clean = proj_n_clean >= cutoff
    if int(top_mask_clean.sum()) < 10:
        cutoff = float(np.percentile(proj_n_clean, 80))
        top_mask_clean = proj_n_clean >= cutoff
    top_points = points_clean[top_mask_clean]
    # Position: 2D mask centroid의 3D lookup — BMP 화면 객체 중앙에 픽 위치.
    # (3D points.mean을 쓰면 perspective projection 비선형으로 인해 재투영
    # 좌표가 객체 시각 중심에서 크게 벗어남. fx·X/Z + cx는 Z로 나누므로
    # mean(X/Z) ≠ mean(X)/mean(Z). 객체가 기울어져 z 분포 있을 때 어긋남.)
    use_all = ctx["use"]
    ys_u, xs_u = np.where(use_all)
    if len(xs_u) >= 5:
        u_c = float(xs_u.mean())
        v_c = float(ys_u.mean())
        # 그 픽셀의 3D 좌표. NaN이면 inverse projection으로 합성 (z=median).
        ui = int(round(u_c)); vi = int(round(v_c))
        H_g, W_g = use_all.shape
        if 0 <= vi < H_g and 0 <= ui < W_g and use_all[vi, ui]:
            X = float(grid["x"][vi, ui])
            Y = float(grid["y"][vi, ui])
            Z = float(grid["z"][vi, ui])
        else:
            Z = float(np.median(points_clean[:, 2]))
            X = (u_c - CX) * Z / FX
            Y = (v_c - CY) * Z / FY
        center_xy = np.array([X, Y, Z], dtype=np.float64)
    else:
        center_xy = points_clean.mean(axis=0)
    # normal 방향만 표면(top) 평면으로 살짝 끌어올림 + SAFETY_OFFSET
    top_z_along_n = top_points.mean(axis=0) @ normal
    cur_z_along_n = center_xy @ normal
    position = center_xy + (top_z_along_n - cur_z_along_n) * normal + SAFETY_OFFSET_MM * normal

    # PCA on top points (평면 내) → long_axis
    centered = top_points - top_points.mean(axis=0)
    centered_in_plane = centered - np.outer(centered @ normal, normal)
    cov = (centered_in_plane.T @ centered_in_plane) / max(len(top_points) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    long_axis = _orient_axis_consistently(eigvecs[:, 2])
    long_axis = long_axis - np.dot(long_axis, normal) * normal
    long_axis = long_axis / max(np.linalg.norm(long_axis), 1e-12)
    mid_axis = np.cross(normal, long_axis)

    ctx.update({
        "position": position, "normal": normal,
        "long_axis": long_axis, "mid_axis": mid_axis,
        "eigvals": eigvals,
        # Top-surface mean (안전한 기준점 — refine cap 비교용).
        # polygon 2D centroid보다 강건: 부분 마스크에서도 "보이는 위쪽 표면의
        # 무게중심"으로, 거짓 중심 끌림이 적음.
        "initial_position": position.copy(),
    })
    return None


def _refine_pose_with_heatmap(ctx: dict) -> None:
    """Heatmap argmax + 다중 yaw로 픽 위치/yaw 정제. 실패 시 원래 pose 유지.
    Bottle은 cylinder fit이 이미 최적 pose라 refine skip."""
    if ctx.get("cid") == BOTTLE_CID:
        return
    try:
        from .suction_score import (pickability_heatmap,
                                      approach_box_check as _arm_box_chk_fn)
        heatmap = pickability_heatmap(ctx["grid"], ctx["valid"], ctx["mask"],
                                       cup_radius_px=HEATMAP_CUP_RADIUS_PX,
                                       use_arm_score=False)
    except Exception:
        return  # heatmap 실패 시 원래 pose 사용

    mask = ctx["mask"]
    grid = ctx["grid"]
    valid = ctx["valid"]
    normal, long_axis = ctx["normal"], ctx["long_axis"]
    ys_m, xs_m = np.where(mask)
    if len(xs_m) == 0:
        return
    u_min_m, u_max_m = int(xs_m.min()), int(xs_m.max())
    v_min_m, v_max_m = int(ys_m.min()), int(ys_m.max())
    cup_offset = SUCTION_CUP_PITCH_MM / 2.0
    H_m, W_m = mask.shape

    # 다중 yaw 후보 — 8방향(22.5° 간격). PCA가 부분 마스크에서 long_axis를
    # 잘못 잡는 케이스(Q2: 한 컵은 중앙, 다른 컵은 가장자리 빨강)에 강건.
    yaw_options = [long_axis.copy()]
    for yaw_deg in (22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5):
        a = np.radians(yaw_deg)
        cos_a, sin_a = np.cos(a), np.sin(a)
        rotated = (long_axis * cos_a + np.cross(normal, long_axis) * sin_a
                   + normal * np.dot(normal, long_axis) * (1 - cos_a))
        rotated = rotated - np.dot(rotated, normal) * normal
        rotated = rotated / max(np.linalg.norm(rotated), 1e-12)
        yaw_options.append(rotated)

    # 추가 후보: polygon 전체 점(top-surface 무관)으로 평면 내 PCA major.
    # _compute_initial_pose의 PCA는 top-surface 95% 위쪽 점만 써서, 마름모/
    # 사다리꼴 폴리곤에선 진짜 major axis와 어긋날 수 있음. polygon 전체에
    # 기반한 axis + 그 기준 회전 7개를 추가 후보로 넣어 PCA 회전 오류에 강건.
    # (마름모 케이스: polygon minor가 양 끝 넓은 변에 컵이 닿게 함)
    try:
        use_all = mask & valid
        n_all = int(use_all.sum())
        if n_all >= 30:
            # 다운샘플 (큰 폴리곤에서 비용 절감) — 최대 ~500점
            step = max(1, n_all // 500)
            ys_all, xs_all = np.where(use_all)
            ys_all = ys_all[::step]; xs_all = xs_all[::step]
            pts_all = np.stack([grid["x"][ys_all, xs_all],
                                 grid["y"][ys_all, xs_all],
                                 grid["z"][ys_all, xs_all]], axis=-1).astype(np.float64)
            centered_all = pts_all - pts_all.mean(axis=0)
            in_plane = centered_all - np.outer(centered_all @ normal, normal)
            cov_p = (in_plane.T @ in_plane) / max(len(pts_all) - 1, 1)
            _, evecs_p = np.linalg.eigh(cov_p)
            poly_major = _orient_axis_consistently(evecs_p[:, 2])
            poly_major = poly_major - np.dot(poly_major, normal) * normal
            mag = float(np.linalg.norm(poly_major))
            if mag > 1e-9:
                poly_major = poly_major / mag
                yaw_options.append(poly_major)
                # polygon major 기준 22.5° 간격 회전 (minor=+90° 포함)
                for yaw_deg in (22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5):
                    a = np.radians(yaw_deg)
                    cos_a, sin_a = np.cos(a), np.sin(a)
                    rotated = (poly_major * cos_a
                               + np.cross(normal, poly_major) * sin_a
                               + normal * np.dot(normal, poly_major) * (1 - cos_a))
                    rotated = rotated - np.dot(rotated, normal) * normal
                    m = float(np.linalg.norm(rotated))
                    if m > 1e-9:
                        yaw_options.append(rotated / m)
    except Exception:
        pass

    # (u,v,yaw) grid 후보 score 평가.
    # 두 컵 균형: 한쪽만 좋은 후보(예: 컵 A=초록, B=빨강)는 reject. 양쪽 다
    # 일정 score 이상 + 차이가 작은 후보 우선.
    # CUP_MIN_EACH 0.10: heatmap이 가장자리 빨강이라도 cup_coverage 사전 체크가
    # 뒤에서 거르므로 여기선 느슨하게. (마름모 객체에서 양 끝이 살짝 가장자리에
    # 가까워 score 0.2~0.3인 경우도 후보로 받기 위함)
    CUP_MIN_EACH = 0.10
    BALANCE_PENALTY = 0.5   # |sa-sb| × 이 가중치만큼 score에서 깎음
    candidates = []  # (score, pos, long_axis_cand)
    for uu in range(u_min_m, u_max_m + 1, 8):
        for vv in range(v_min_m, v_max_m + 1, 8):
            if not (0 <= uu < W_m and 0 <= vv < H_m) or not mask[vv, uu]:
                continue
            z_uv = float(grid["z"][vv, uu])
            if not np.isfinite(z_uv):
                continue
            x_uv = (uu - CX) * z_uv / FX
            y_uv = (vv - CY) * z_uv / FY
            cand = np.array([x_uv, y_uv, z_uv]) + SAFETY_OFFSET_MM * normal
            for cand_long in yaw_options:
                cup_a_c = cand + cup_offset * cand_long
                cup_b_c = cand - cup_offset * cand_long
                ua = project_to_image(cup_a_c)
                ub = project_to_image(cup_b_c)
                if ua is None or ub is None:
                    continue
                ua_i, va_i = int(round(ua[0])), int(round(ua[1]))
                ub_i, vb_i = int(round(ub[0])), int(round(ub[1]))
                if not (0 <= ua_i < W_m and 0 <= va_i < H_m
                        and 0 <= ub_i < W_m and 0 <= vb_i < H_m):
                    continue
                sa = float(heatmap[va_i, ua_i])
                sb = float(heatmap[vb_i, ub_i])
                if not (np.isfinite(sa) and np.isfinite(sb)):
                    continue
                # 한쪽 컵이라도 임계 미만이면 후보 자격 박탈
                if sa < CUP_MIN_EACH or sb < CUP_MIN_EACH:
                    continue
                # 두 컵 균형 페널티: score = min - α·|차이|
                score = min(sa, sb) - BALANCE_PENALTY * abs(sa - sb)
                if score > 0.0:
                    candidates.append((score, cand, cand_long))

    # Score 내림차순 → arm + cup coverage + 거리 cap 통과 첫 후보 채택.
    # cup coverage 사전 체크가 핵심: refine이 yaw를 바꿔도 cup_coverage 게이트가
    # 뒤에서 reject하면 결국 initial pose로 fallback돼서 의미 없음. 여기서
    # 미리 걸러야 polygon minor axis 같은 유리한 yaw가 채택됨.
    from .suction_score import cup_self_coverage as _ccov
    candidates.sort(key=lambda x: -x[0])
    arm_hl = (RECT_LONG_MM / 2.0) * ARM_BOX_SCALE
    arm_hs = (RECT_SHORT_MM / 2.0) * ARM_BOX_SCALE
    initial_pos = ctx.get("initial_position")
    cup_r_mm = SUCTION_CUP_OD_MM / 2.0
    # Score에 거리 페널티 추가 후 재정렬. cap을 hard로 두는 대신 soft penalty:
    # 곡면 꼭대기처럼 score 비슷한 멀리 후보는 페널티로 자연스레 reject되고,
    # polygon minor axis처럼 score 분명 더 좋은 후보는 멀어도 채택.
    # DIST_PENALTY 0.015 = 1mm 당 0.015 깎임 → 25mm = -0.375 페널티.
    DIST_PENALTY = 0.015
    if initial_pos is not None:
        adjusted = []
        for s, p, l in candidates:
            d = float(np.linalg.norm(p - initial_pos))
            adjusted.append((s - DIST_PENALTY * d, p, l, d))
        adjusted.sort(key=lambda x: -x[0])
        candidates = [(s, p, l) for (s, p, l, _d) in adjusted]
    # Hard cap: 너무 멀면 (예: 50mm+) 무조건 reject — 안전망
    HARD_CAP_MM = max(REFINE_MAX_OFFSET_MM * 3.0, 50.0) if REFINE_MAX_OFFSET_MM > 0 else 50.0
    for _, cand_pos, cand_long in candidates[:50]:
        if initial_pos is not None:
            if float(np.linalg.norm(cand_pos - initial_pos)) > HARD_CAP_MM:
                continue
        cup_a_c = cand_pos + cup_offset * cand_long
        cup_b_c = cand_pos - cup_offset * cand_long
        cov_a = _ccov(mask, cup_a_c, radius_mm=cup_r_mm)
        cov_b = _ccov(mask, cup_b_c, radius_mm=cup_r_mm)
        if (cov_a is not None and cov_a < GATE_S2_CUP_COVERAGE_MIN) or \
           (cov_b is not None and cov_b < GATE_S2_CUP_COVERAGE_MIN):
            continue
        cand_mid = np.cross(normal, cand_long)
        arm_chk = _arm_box_chk_fn(
            grid, valid, cand_pos, normal, cand_long, cand_mid,
            half_long_mm=arm_hl, half_short_mm=arm_hs,
            t_min_mm=HEAD_CLEARANCE_MM, t_max_mm=ARM_BOX_HEIGHT_MM,
            self_mask=mask,
        )
        if arm_chk["n_outside"] > 0 and arm_chk["max_t_mm"] > GATE_S2_ARM_PROTRUSION_MM:
            continue
        ctx["position"] = cand_pos
        ctx["long_axis"] = cand_long
        ctx["mid_axis"] = cand_mid
        return


def _check_structural(ctx: dict) -> Optional[PickResult]:
    """구조적 IMPOSSIBLE: 길이 < pitch / S3 flatness / occlusion perimeter."""
    cid, n = ctx["cid"], ctx["n"]
    points, position, long_axis = ctx["points"], ctx["position"], ctx["long_axis"]
    normal, mid_axis = ctx["normal"], ctx["mid_axis"]
    eigvals = ctx["eigvals"]

    # 길이 (모든 폴리곤 점에서 long_axis 방향 5-95 percentile).
    # 객체 < pitch이면 dual-cup 불가지만 length_defer 모드에서는 일반 흐름
    # 진행 → 뒤의 S2_neighbor_clean이 single-cup 가능성을 판정.
    proj_long_all = (points - position) @ long_axis
    obj_long_length = float(np.percentile(proj_long_all, 95)
                            - np.percentile(proj_long_all, 5))
    if obj_long_length < SUCTION_CUP_PITCH_MM and not GATE_LENGTH_DEFER:
        return _impossible(
            f"객체 길이 부족 ({obj_long_length:.1f}mm < "
            f"{SUCTION_CUP_PITCH_MM}mm pitch)  ← suction.cup_pitch_mm",
            cid, gates=["length"], n_valid_points=n, eigenvalues=eigvals,
            position_mm=position, long_axis=long_axis, mid_axis=mid_axis,
            normal=normal,
        )

    # S3 평면 패치 존재 (bottle은 곡면 = 평면 패치 없는 게 정상 → skip)
    if (cid != BOTTLE_CID and
            not _has_flat_patch(ctx["grid"], ctx["valid"], ctx["mask"],
                                  threshold=GATE_S3_FLATNESS_MIN)):
        return _impossible(
            f"평면 패치 없음 (S3 flatness < {GATE_S3_FLATNESS_MIN})  "
            f"← gates.s3_flatness_min", cid,
            gates=["S3_flatness"], n_valid_points=n, position_mm=position,
            long_axis=long_axis, mid_axis=mid_axis, normal=normal)

    # Occlusion perimeter
    try:
        occ_clear = _compute_occlusion_clear(ctx["grid"], ctx["polygon_2d"],
                                              ctx["valid"])
    except Exception:
        occ_clear = 1.0
    if occ_clear < OCCLUSION_CLEAR_HARD_REJECT:
        return _impossible(
            f"둘레 가려짐 ({(1.0 - occ_clear):.0%} 가림 → 일부만 보임)  "
            f"← occlusion.clear_hard_reject",
            cid, gates=["occlusion_perimeter"], n_valid_points=n,
            position_mm=position, long_axis=long_axis, mid_axis=mid_axis,
            normal=normal,
        )

    ctx["obj_long_length"] = obj_long_length
    ctx["occ_clear"] = occ_clear
    return None


def _check_defer_gates(ctx: dict) -> Optional[PickResult]:
    """1차 dual-cup 시도, 실패 시 SINGLE_CUP_CIDS 클래스만 single-cup fallback.

    1) bottle은 cylinder 측면 곡면이라 dual cup sealing 물리적 불가
       (cup pitch 50mm > cap radius 12-15mm) → dual skip하고 바로 single
    2) 그 외 dual-cup 모드(single_cup=False)로 모든 게이트 검사
    3) 통과하면 dual-cup PICKABLE
    4) 실패 + cid in SINGLE_CUP_CIDS → single-cup mode로 재시도
       (dual-cup 가정 게이트 skip, neighbor_clean만 빈 컵 안전성 검사)
    5) 둘 다 실패면 dual-cup 실패 사유 반환 + single retry 사유를
       PickResult에 별도 필드로 inject (진단용).
    """
    # bottle: 측면 흡착은 single cup만 가능. dual cup 시도 자체 skip.
    if ctx["cid"] == BOTTLE_CID:
        single_result = _check_defer_gates_inner(ctx, single_cup=True)
        if single_result is None:
            ctx["pick_mode"] = "single"
            return None
        return single_result
    result = _check_defer_gates_inner(ctx, single_cup=False)
    if result is None:
        ctx["pick_mode"] = "dual"
        return None
    if ctx["cid"] in SINGLE_CUP_CIDS:
        single_result = _check_defer_gates_inner(ctx, single_cup=True)
        if single_result is None:
            ctx["pick_mode"] = "single"
            return None
        # single retry도 실패 → dual fail PickResult에 single 결과 inject.
        # 분석할 때 어느 게이트가 single mode에서 잡고 있는지 알 수 있음.
        result.single_retry_status = single_result.status
        result.single_retry_gate_failures = list(single_result.gate_failures)
        result.single_retry_reason = single_result.reason
        # reason 문자열도 합쳐서 UI에 보이게
        result.reason = (
            f"{result.reason}  "
            f"[single retry도 실패: "
            f"{','.join(single_result.gate_failures) or '?'} — "
            f"{single_result.reason}]"
        )
    return result


def _check_defer_gates_inner(ctx: dict, single_cup: bool) -> Optional[PickResult]:
    """DEFER 게이트 일괄. single_cup=True면 dual-cup 가정 게이트 skip."""
    cid, n = ctx["cid"], ctx["n"]
    grid, valid, mask = ctx["grid"], ctx["valid"], ctx["mask"]
    points = ctx["points"]
    position, normal = ctx["position"], ctx["normal"]
    long_axis, mid_axis = ctx["long_axis"], ctx["mid_axis"]
    eigvals = ctx["eigvals"]

    # ─── S1: 다른 폴리곤이 자기 위 덮음 ────────────────────────────
    if ctx.get("scene_polygons"):
        other_mask = _scene_other_mask(ctx["scene_polygons"])
        n_overlap = int((mask & other_mask).sum())
        overlap_ratio = n_overlap / ctx["mask_total"]
        if overlap_ratio > GATE_S1_OVERLAP_DEFER:
            return _defer(
                f"다른 객체에 덮임 ({overlap_ratio:.0%} > "
                f"{GATE_S1_OVERLAP_DEFER:.0%})  ← gates.s1_overlap_defer",
                cid, gates=["S1_overlap"], n_valid_points=n,
                position_mm=position, long_axis=long_axis,
                mid_axis=mid_axis, normal=normal)

    # ─── 컵 위치 + cup support (3D 구) ─────────────────────────────
    # single_cup mode: 흡착의 주체는 cup_a(=position)이지만 dead cup(cup_b)도
    # 진공 라인이 연결돼 있어서 옆 객체에 닿으면 흡입함 → "안전한 거리"
    # 필요. 두 컵의 sign(±long_axis)은 양쪽 후보를 평가해서 arm_box 충돌과
    # neighbor(B컵 주변 다른 객체 surface) 둘 다 최소화하는 방향 채택.
    cup_radius = SUCTION_CUP_OD_MM / 2.0
    if single_cup:
        from .suction_score import approach_box_check as _abc_q
        arm_hl_q = (RECT_LONG_MM / 2.0) * ARM_BOX_SCALE
        arm_hs_q = (RECT_SHORT_MM / 2.0) * ARM_BOX_SCALE
        # neighbor 검사용 다른 객체 후보 점 — self_mask 밖 valid 영역
        other_for_nb = (~mask) & valid
        if other_for_nb.any():
            xyz_other_nb = np.stack(
                [grid["x"][other_for_nb], grid["y"][other_for_nb],
                 grid["z"][other_for_nb]], axis=-1).astype(np.float64)
        else:
            xyz_other_nb = None

        candidates = []
        for sign in (-1.0, +1.0):
            rc_cand = position + sign * (SUCTION_CUP_PITCH_MM / 2.0) * long_axis
            cup_b_cand = position + sign * SUCTION_CUP_PITCH_MM * long_axis
            arm_c = _abc_q(grid, valid, rc_cand, normal, long_axis, mid_axis,
                           half_long_mm=arm_hl_q, half_short_mm=arm_hs_q,
                           t_min_mm=HEAD_CLEARANCE_MM, t_max_mm=ARM_BOX_HEIGHT_MM,
                           self_mask=mask)
            arm_t = float(arm_c["max_t_mm"])
            arm_n_out = int(arm_c["n_outside"])
            # B컵 주변 다른 객체 surface 점 개수 (반경 R, 평면 ±band)
            nb_n = 0
            if xyz_other_nb is not None:
                d = xyz_other_nb - cup_b_cand[None, :]
                in_sphere = (d * d).sum(axis=1) <= (GATE_S2_NEIGHBOR_RADIUS_MM ** 2)
                if in_sphere.any():
                    signed_b = (cup_b_cand[None, :] - xyz_other_nb[in_sphere]) @ normal
                    nb_n = int((np.abs(signed_b) <= GATE_S2_NEIGHBOR_BAND_MM).sum())
            arm_ok = arm_n_out == 0 or arm_t <= GATE_S2_ARM_PROTRUSION_MM
            nb_ok = nb_n <= GATE_S2_NEIGHBOR_MAX_POINTS
            # 우선순위: 둘 다 통과 > arm만 통과 > nb만 통과 > 둘 다 fail.
            # 같은 등급 안에서는 arm_t / nb_n 작은 쪽.
            score = (-(int(arm_ok) + int(nb_ok)), arm_t, nb_n)
            candidates.append((score, sign))
        candidates.sort()
        single_sign = candidates[0][1]
        cup_a = position
        cup_b = position + single_sign * SUCTION_CUP_PITCH_MM * long_axis
        n_a = int((np.linalg.norm(points - cup_a, axis=1) < cup_radius).sum())
        n_b = 0
    else:
        cup_offset = SUCTION_CUP_PITCH_MM / 2.0
        cup_a = position + cup_offset * long_axis
        cup_b = position - cup_offset * long_axis
        n_a = int((np.linalg.norm(points - cup_a, axis=1) < cup_radius).sum())
        n_b = int((np.linalg.norm(points - cup_b, axis=1) < cup_radius).sum())
    # bottle은 cylinder prior로 가상 surface 계산이라 실제 valid 점 없는 게
    # 정상 (측면 투명). cap detect로 이미 신뢰 확보 → cup_support 게이트 skip.
    if (cid != BOTTLE_CID and not single_cup and GATE_CUP_SUPPORT_MIN_POINTS > 0
            and (n_a < GATE_CUP_SUPPORT_MIN_POINTS
                 or n_b < GATE_CUP_SUPPORT_MIN_POINTS)):
        return _defer(
            f"컵 패치 3D 점 부족 (A={n_a}, B={n_b}, "
            f"최소 {GATE_CUP_SUPPORT_MIN_POINTS})  ← gates.cup_support_min_points",
            cid, gates=["cup_support"], n_valid_points=n, eigenvalues=eigvals,
            position_mm=position, long_axis=long_axis, mid_axis=mid_axis,
            normal=normal, cup_a_3d=cup_a, cup_b_3d=cup_b,
            cup_a_support=n_a, cup_b_support=n_b)

    # ─── Cup self-coverage (컵 OD 면적이 라벨 안에 들어가나) — single skip
    from .suction_score import cup_self_coverage, cup_rim_coverage
    for label, cup_3d in (("A", cup_a), ("B", cup_b)):
        if single_cup:
            break
        cov = cup_self_coverage(mask, cup_3d, radius_mm=cup_radius)
        if cov is not None and cov < GATE_S2_CUP_COVERAGE_MIN:
            return _defer(
                f"컵 {label} 라벨 밖으로 빠짐 "
                f"({cov:.0%} 안쪽 < {GATE_S2_CUP_COVERAGE_MIN:.0%})  "
                f"← gates.s2_cup_coverage_min",
                cid, gates=[f"S2_cup_{label}_coverage"],
                n_valid_points=n, eigenvalues=eigvals, position_mm=position,
                long_axis=long_axis, mid_axis=mid_axis, normal=normal,
                cup_a_3d=cup_a, cup_b_3d=cup_b,
                cup_a_support=n_a, cup_b_support=n_b)

    # ─── Cup rim sealing (둘레가 폴리곤 안에 모두 들어가나)
    # 면적 게이트와 별개 — 면적은 충분해도 둘레 한쪽이 끊기면 진공 안 잡힘.
    # dual: A·B 모두 검사. single: cup_a(=position)만 검사. dead cup_b는
    # 의도적 off-target이라 polygon-안 비율 의미 없음. bottle은 곡면 측면이라
    # 폴리곤 자체가 cylinder 옆면 일부만 표현해서 rim도 부분 빠짐이 정상 → skip.
    # threshold는 클래스별 (비닐: 관대, 강체: 엄격). mask 2px dilate로
    # 라벨 가장자리 노이즈 흡수.
    rim_iter = (("A", cup_a),) if single_cup else (("A", cup_a), ("B", cup_b))
    if cid != BOTTLE_CID:
        rim_thr = GATE_S2_CUP_RIM_MIN_BY_CID.get(cid, GATE_S2_CUP_RIM_MIN)
        # dilation: scipy 사용 (이미 의존성에 있음). 없으면 fallback (no dilate).
        rim_mask = mask
        if GATE_S2_CUP_RIM_DILATE_PX > 0:
            try:
                from scipy.ndimage import binary_dilation
                rim_mask = binary_dilation(mask, iterations=GATE_S2_CUP_RIM_DILATE_PX)
            except Exception:
                pass
        for label, cup_3d in rim_iter:
            rim = cup_rim_coverage(rim_mask, cup_3d, radius_mm=cup_radius)
            if rim is not None and rim < rim_thr:
                return _defer(
                    f"컵 {label} 둘레 일부가 폴리곤 밖 "
                    f"({rim:.0%} sealed < {rim_thr:.0%}) "
                    f"— 흡착 시 바람샘 위험  ← gates.s2_cup_rim_min",
                    cid, gates=[f"S2_cup_{label}_rim"],
                    n_valid_points=n, eigenvalues=eigvals, position_mm=position,
                    long_axis=long_axis, mid_axis=mid_axis, normal=normal,
                    cup_a_3d=cup_a, cup_b_3d=cup_b,
                    cup_a_support=n_a, cup_b_support=n_b)

    # ─── Rectangle 코너 + box ROI ─────────────────────────────────
    # single_cup mode: 사각형은 그리퍼 tool 전체 footprint를 표현해야 하므로
    # 두 컵의 중점(= cup_a와 dead cup_b의 중점)에 재중심. dead cup 방향은
    # 위 cup geometry 블록에서 정한 single_sign과 동일하게 사용.
    half_long = RECT_LONG_MM / 2.0
    half_short = RECT_SHORT_MM / 2.0
    if single_cup:
        rect_center = position + single_sign * (SUCTION_CUP_PITCH_MM / 2.0) * long_axis
    else:
        rect_center = position
    corners_3d = [
        rect_center + half_long * long_axis + half_short * mid_axis,
        rect_center + half_long * long_axis - half_short * mid_axis,
        rect_center - half_long * long_axis - half_short * mid_axis,
        rect_center - half_long * long_axis + half_short * mid_axis,
    ]
    corners_2d = []
    for c in corners_3d:
        uv = project_to_image(c)
        if uv is None:
            return _defer("사각형 코너가 카메라 뒤로 투영됨", cid,
                          gates=["projection"], n_valid_points=n,
                          position_mm=position, long_axis=long_axis,
                          mid_axis=mid_axis, normal=normal)
        corners_2d.append(uv)
    # ─── S2 corner_depth — 사각형 4코너 depth로 빈 벽/빈 밖 회피 ─────
    # box_roi(2D 사용자 그림) 대신 depth로 검사. 각 코너 위치의 8px disk에서:
    #   - NaN 비율 ≥ s2_corner_nan_ratio       → 빈 밖 / 카메라 시야 밖
    #   - 객체 표면보다 normal 반대로 s2_corner_protrusion_mm+ 솟음 → 빈 벽
    # 단일 코너 false positive 방지: 4코너 모두 평가 후 fail 카운트 ≥
    # s2_corner_min_fails 면 reject.
    # bottle은 곡면 측면 흡착이라 사각형 코너가 cylinder 옆구리 너머
    # (빈 공간/NaN)으로 나가는 게 정상 → 검사 skip.
    # single-cup mode는 사각형이 객체보다 커서 코너가 옆 객체 위에 자주 가는 게
    # 정상. 옆 객체 검출은 neighbor_clean이 담당하므로 corner_depth skip.
    if cid != BOTTLE_CID and not single_cup:
        z_field = grid["z"]
        H_g, W_g = z_field.shape
        CORNER_DISK_PX = 8
        corner_fails = []  # (ci, reason_str, gate_name)
        for ci, ((u, v), c3d) in enumerate(zip(corners_2d, corners_3d)):
            u_lo = max(0, int(round(u - CORNER_DISK_PX)))
            u_hi = min(W_g, int(round(u + CORNER_DISK_PX)) + 1)
            v_lo = max(0, int(round(v - CORNER_DISK_PX)))
            v_hi = min(H_g, int(round(v + CORNER_DISK_PX)) + 1)
            if u_lo >= u_hi or v_lo >= v_hi:
                corner_fails.append((ci, "out-of-image"))
                continue
            patch = z_field[v_lo:v_hi, u_lo:u_hi]
            n_patch = patch.size
            finite_mask = np.isfinite(patch)
            n_finite = int(finite_mask.sum())
            nan_ratio = 1.0 - (n_finite / n_patch) if n_patch else 1.0
            if nan_ratio >= GATE_S2_CORNER_NAN_RATIO:
                corner_fails.append((ci, f"NaN {nan_ratio:.0%}"))
                continue
            if n_finite == 0:
                continue
            finite_z = patch[finite_mask]
            vv, uu = np.mgrid[v_lo:v_hi, u_lo:u_hi]
            finite_u = uu[finite_mask].astype(np.float64)
            finite_v = vv[finite_mask].astype(np.float64)
            xs = (finite_u - CX) * finite_z / FX
            ys = (finite_v - CY) * finite_z / FY
            pts_corner = np.stack([xs, ys, finite_z.astype(np.float64)], axis=-1)
            # 픽 평면에서의 부호 있는 거리: +면 객체 표면 위(카메라 쪽 = 빈 벽)
            signed = (position[None, :] - pts_corner) @ normal
            protrusion = float(signed.max())
            if protrusion > GATE_S2_CORNER_PROTRUSION_MM:
                corner_fails.append((ci, f"{protrusion:.0f}mm 솟음"))

        if len(corner_fails) >= GATE_S2_CORNER_MIN_FAILS:
            reasons = ", ".join(f"코너{ci}({r})" for ci, r in corner_fails)
            return _defer(
                f"사각형 코너 {len(corner_fails)}개 실패 [{reasons}]  "
                f"← gates.s2_corner_protrusion_mm / s2_corner_nan_ratio "
                f"(min_fails={GATE_S2_CORNER_MIN_FAILS})",
                cid, gates=["S2_corner_depth"],
                n_valid_points=n, eigenvalues=eigvals, position_mm=position,
                long_axis=long_axis, mid_axis=mid_axis, normal=normal)

    cup_a_2d = project_to_image(cup_a)
    cup_b_2d = project_to_image(cup_b)

    # ─── S2 neighbor_clean — 자기 폴리곤 밖에 있는 컵 주변 검사 ─────
    # dual·single 공통: dead cup도 진공 라인 연결돼 있어 다른 객체에 닿으면
    # 흡입 가능 → A·B 둘 다 검사. 컵이 자기 폴리곤 안이면 cup_coverage가
    # 담당하므로 skip. 폴리곤 밖에 있을 때만 주변에 다른 객체 surface가
    # 있는지 본다 (빈 바닥/벽은 깊이 차이로 자동 무시).
    # bottle은 측면 흡착이라 컵이 cylinder 옆구리에 있어야 정상 → skip.
    if cid != BOTTLE_CID:
        for label, cup_3d, cup_uv in (("A", cup_a, cup_a_2d),
                                        ("B", cup_b, cup_b_2d)):
            if cup_uv is None:
                continue
            ui, vi = int(round(cup_uv[0])), int(round(cup_uv[1]))
            H_g, W_g = mask.shape
            if not (0 <= ui < W_g and 0 <= vi < H_g):
                continue
            if mask[vi, ui]:
                continue  # 컵이 자기 폴리곤 안 → skip (cup_coverage 등이 담당)
            # 컵이 폴리곤 밖 → 주변 3D 점 검사
            # 사용자 폴리곤 마스크를 제외한 다른 valid 3D 점 중,
            # 컵으로부터 반경 GATE_S2_NEIGHBOR_RADIUS_MM 안에 있고,
            # 컵 평면(normal 기준) ±GATE_S2_NEIGHBOR_BAND_MM 안에 있는 점.
            other = (~mask) & valid
            if not other.any():
                continue
            xyz_other = np.stack([grid["x"][other], grid["y"][other],
                                    grid["z"][other]], axis=-1).astype(np.float64)
            d = xyz_other - cup_3d[None, :]
            dist2 = (d * d).sum(axis=1)
            in_sphere = dist2 <= (GATE_S2_NEIGHBOR_RADIUS_MM ** 2)
            if not in_sphere.any():
                continue
            # 컵 평면 부호 거리: signed 작을수록 표면 평면에 가까움
            signed = (cup_3d[None, :] - xyz_other) @ normal
            in_band = np.abs(signed) <= GATE_S2_NEIGHBOR_BAND_MM
            n_other_surface = int((in_sphere & in_band).sum())
            if n_other_surface > GATE_S2_NEIGHBOR_MAX_POINTS:
                return _defer(
                    f"컵 {label} 주변에 다른 객체 (반경 "
                    f"{GATE_S2_NEIGHBOR_RADIUS_MM:.0f}mm, ±"
                    f"{GATE_S2_NEIGHBOR_BAND_MM:.0f}mm 평면 안 점 "
                    f"{n_other_surface}개 > {GATE_S2_NEIGHBOR_MAX_POINTS}) "
                    f"— single-cup 픽 시 다른 객체 잡힘 가능  "
                    f"← gates.s2_neighbor_*",
                    cid, gates=[f"S2_neighbor_{label}_clean"],
                    n_valid_points=n, eigenvalues=eigvals, position_mm=position,
                    long_axis=long_axis, mid_axis=mid_axis, normal=normal,
                    cup_a_3d=cup_a, cup_b_3d=cup_b,
                    cup_a_2d=cup_a_2d, cup_b_2d=cup_b_2d)

    # ─── Rectangle 평면도 (사각형 안 self surface 굴곡) ───────────
    # bottle은 곡면 측면 흡착, single-cup mode는 사각형이 객체보다 커서 정상.
    use_self = mask & valid
    if cid != BOTTLE_CID and not single_cup and use_self.any():
        xyz_self = np.stack([grid["x"][use_self], grid["y"][use_self],
                              grid["z"][use_self]], axis=-1).astype(np.float64)
        rel = xyz_self - position[None, :]
        x_p = rel @ long_axis
        y_p = rel @ mid_axis
        z_p = rel @ normal
        in_rect = (np.abs(x_p) <= half_long) & (np.abs(y_p) <= half_short)
        if int(in_rect.sum()) >= 30:
            rect_std = float(z_p[in_rect].std())
            if rect_std > GATE_S2_RECT_PLANARITY_MM:
                return _defer(
                    f"두 컵 사이 굴곡 큼 (std {rect_std:.1f}mm > "
                    f"{GATE_S2_RECT_PLANARITY_MM}mm)  ← gates.s2_rect_planarity_mm",
                    cid, gates=["S2_rect_planarity"],
                    n_valid_points=n, eigenvalues=eigvals, position_mm=position,
                    long_axis=long_axis, mid_axis=mid_axis, normal=normal,
                    cup_a_3d=cup_a, cup_b_3d=cup_b,
                    cup_a_2d=cup_a_2d, cup_b_2d=cup_b_2d,
                    cup_a_support=n_a, cup_b_support=n_b)

    # ─── S2 arm box collision ─────────────────────────────────────
    # 충돌 박스는 그리퍼 tool footprint 중심에 둠. single 모드는 position이
    # cup_a로 옮겨갔으므로 박스도 rect_center(= 두 컵 중점)로 시프트해야
    # 한쪽으로 쏠리지 않음. dual은 rect_center == position.
    from .suction_score import (approach_box_check, approach_cylinder_check,
                                  cup_max_bump)
    arm_hl = (RECT_LONG_MM / 2.0) * ARM_BOX_SCALE
    arm_hs = (RECT_SHORT_MM / 2.0) * ARM_BOX_SCALE
    arm_chk = approach_box_check(
        grid, valid, rect_center, normal, long_axis, mid_axis,
        half_long_mm=arm_hl, half_short_mm=arm_hs,
        t_min_mm=HEAD_CLEARANCE_MM, t_max_mm=ARM_BOX_HEIGHT_MM,
        self_mask=mask)
    # 정렬용 충돌 마진: arm 박스 안 최대 protrusion (mm). 낮을수록 안전.
    # n_outside==0이면 0.0. PICKABLE 경로에서 _build_pickable_result가 읽음.
    ctx["arm_max_protrusion_mm"] = float(arm_chk["max_t_mm"])
    # arm 충돌 게이트는 single/dual 공통으로 적용 — approach_box_check가
    # self_mask로 자기 폴리곤 제외하니까 단지 옆 객체가 솟아있을 때만 fire함.
    # (이전 버전은 single에서 skip했지만 옆 haribo가 100mm 솟은 케이스를
    # 통과시키는 버그가 있었음 — neighbor_clean은 반경 40mm·±10mm로 범위가
    # 좁아 catch 못 함.)
    if (arm_chk["n_outside"] > 0
            and arm_chk["max_t_mm"] > GATE_S2_ARM_PROTRUSION_MM):
        return _defer(
            f"팔 진입 경로에 장애물 ({arm_chk['max_t_mm']:.1f}mm 솟음, "
            f"{arm_chk['n_outside']}점 > {GATE_S2_ARM_PROTRUSION_MM}mm)  "
            f"← gates.s2_arm_protrusion_mm",
            cid, gates=["S2_arm_protrusion"], n_valid_points=n,
            eigenvalues=eigvals, position_mm=position, long_axis=long_axis,
            mid_axis=mid_axis, normal=normal,
            cup_a_3d=cup_a, cup_b_3d=cup_b,
            cup_a_2d=cup_a_2d, cup_b_2d=cup_b_2d,
            cup_a_support=n_a, cup_b_support=n_b)

    # ─── S2 컵 진입 경로 — single-cup mode skip
    for label, cup_3d in (("A", cup_a), ("B", cup_b)):
        if single_cup:
            break
        cup_chk = approach_cylinder_check(
            grid, valid, cup_3d, normal,
            radius_mm=cup_radius, t_min_mm=0.0, t_max_mm=HEAD_CLEARANCE_MM,
            self_mask=mask)
        if cup_chk["n_outside"] > 0 and cup_chk["max_t_mm"] > GATE_S2_CUP_PROTRUSION_MM:
            return _defer(
                f"컵 {label} 진입 경로 막힘 "
                f"(장애물 {cup_chk['max_t_mm']:.1f}mm, "
                f"{cup_chk['n_outside']}점)  ← gates.s2_cup_protrusion_mm",
                cid, gates=[f"S2_cup_{label}_protrusion"],
                n_valid_points=n, eigenvalues=eigvals, position_mm=position,
                long_axis=long_axis, mid_axis=mid_axis, normal=normal,
                cup_a_3d=cup_a, cup_b_3d=cup_b,
                cup_a_2d=cup_a_2d, cup_b_2d=cup_b_2d,
                cup_a_support=n_a, cup_b_support=n_b)

    # ─── S2 컵 밀착 (self surface 돌출) — bottle/single_cup skip
    if cid == BOTTLE_CID or single_cup:
        cup_pairs_for_bump = []
    else:
        cup_pairs_for_bump = (("A", cup_a), ("B", cup_b))
    for label, cup_3d in cup_pairs_for_bump:
        bump = cup_max_bump(grid, valid, cup_3d, normal,
                             radius_mm=cup_radius, self_mask=mask)
        if bump is not None and bump > GATE_S2_CUP_BUMP_MM:
            return _defer(
                f"컵 {label} 밀착 실패 (표면 굴곡 {bump:.1f}mm > "
                f"{GATE_S2_CUP_BUMP_MM}mm)  ← gates.s2_cup_bump_mm",
                cid, gates=[f"S2_cup_{label}_bump"],
                n_valid_points=n, eigenvalues=eigvals, position_mm=position,
                long_axis=long_axis, mid_axis=mid_axis, normal=normal,
                cup_a_3d=cup_a, cup_b_3d=cup_b,
                cup_a_2d=cup_a_2d, cup_b_2d=cup_b_2d,
                cup_a_support=n_a, cup_b_support=n_b)

    # 모든 게이트 통과 → ctx에 cup/rect 저장 (PICKABLE 결과 빌드용)
    ctx.update({
        "cup_a": cup_a, "cup_b": cup_b,
        "cup_a_2d": cup_a_2d, "cup_b_2d": cup_b_2d,
        "n_a": n_a, "n_b": n_b,
        "corners_3d": corners_3d, "corners_2d": corners_2d,
    })
    return None


def _build_pickable_result(ctx: dict) -> PickResult:
    """모든 게이트 통과 → confidence + tier + PickResult 구성.

    _compute_all_signals가 미리 호출됐으면 그 결과 재사용; 아니면 새로 계산.
    """
    cid = ctx["cid"]
    long_axis = ctx["long_axis"]
    yaw = float(np.arctan2(long_axis[1], long_axis[0]))
    pick_mode = ctx.get("pick_mode", "dual")
    # single 모드는 _compute_all_signals이 사전 호출된 dual-cup 값과 어긋남 →
    # 재계산해서 cup_b 없는 실제 기하 반영.
    if pick_mode == "single":
        res = _compute_all_signals(ctx, single_cup=True)
        if res is None:
            confidence, breakdown = 0.0, {}
        else:
            confidence, breakdown = res
    elif "_signals_confidence" in ctx and "_signals_breakdown" in ctx:
        confidence = ctx["_signals_confidence"]
        breakdown = ctx["_signals_breakdown"]
    else:
        # fallback (정상 흐름이면 _compute_all_signals가 미리 호출됨)
        res = _compute_all_signals(ctx)
        if res is None:
            confidence, breakdown = 0.0, {}
        else:
            confidence, breakdown = res
    tier = _classify_tier(confidence)
    return PickResult(
        success=True, status=STATUS_PICKABLE, cid=cid,
        position_mm=ctx["position"], long_axis=long_axis,
        mid_axis=ctx["mid_axis"], normal=ctx["normal"], yaw_rad=yaw,
        rect_corners_3d=ctx["corners_3d"], rect_corners_2d=ctx["corners_2d"],
        cup_a_3d=ctx["cup_a"], cup_b_3d=ctx["cup_b"],
        cup_a_2d=ctx["cup_a_2d"], cup_b_2d=ctx["cup_b_2d"],
        n_valid_points=ctx["n"], eigenvalues=ctx["eigvals"],
        cup_a_support=ctx["n_a"], cup_b_support=ctx["n_b"],
        confidence=confidence, tier=tier, score_breakdown=breakdown,
        bottle_fit=ctx.get("bottle_fit"),
        pick_mode=pick_mode,
        arm_max_protrusion_mm=float(ctx.get("arm_max_protrusion_mm", 0.0)),
    )


def _compute_all_signals(ctx: dict, single_cup: bool = False) -> Optional[tuple]:
    """6 신호 + confidence 일괄 계산. 자세까지 계산된 폴리곤에 한해 가능.

    PICKABLE과 fail 폴리곤 모두에 신호를 채워주기 위한 공용 함수.
    single_cup=True면 cup_b는 존재 X, 신호는 cup_a (=position) 한 점 기준.
    Returns (confidence, breakdown) 또는 None (계산 불가 시).
    """
    if "position" not in ctx or "long_axis" not in ctx:
        return None
    cid = ctx["cid"]
    points = ctx.get("points")
    if points is None:
        return None

    cup_radius = SUCTION_CUP_OD_MM / 2.0
    if single_cup:
        cup_a = ctx["position"]
        cup_b = None
        n_a = int((np.linalg.norm(points - cup_a, axis=1) < cup_radius).sum())
        n_b = 0
    else:
        cup_offset = SUCTION_CUP_PITCH_MM / 2.0
        cup_a = ctx["position"] + cup_offset * ctx["long_axis"]
        cup_b = ctx["position"] - cup_offset * ctx["long_axis"]
        n_a = int((np.linalg.norm(points - cup_a, axis=1) < cup_radius).sum())
        n_b = int((np.linalg.norm(points - cup_b, axis=1) < cup_radius).sum())

    # obj_long_length — _check_structural에서 계산됐을 수도 있고 아닐 수도
    if "obj_long_length" in ctx:
        obj_long_length = ctx["obj_long_length"]
    else:
        proj_long_all = (points - ctx["position"]) @ ctx["long_axis"]
        obj_long_length = float(np.percentile(proj_long_all, 95)
                                - np.percentile(proj_long_all, 5))
        ctx["obj_long_length"] = obj_long_length

    # occ_clear — 같은 식으로 lazy 계산
    if "occ_clear" in ctx:
        occ_clear = ctx["occ_clear"]
    else:
        try:
            occ_clear = _compute_occlusion_clear(
                ctx["grid"], ctx["polygon_2d"], ctx["valid"])
        except Exception:
            occ_clear = 1.0
        ctx["occ_clear"] = occ_clear

    # normal_std (컵 위치 normal flatness) — PICKABLE때만 정확하게 계산 가능
    # single_cup: cup_a 한 점만 평가 (normal_flatness_at).
    try:
        cup_a_2d = project_to_image(cup_a)
        if single_cup:
            from .suction_score import normal_flatness_at
            ns_score = normal_flatness_at(ctx["grid"], ctx["valid"],
                                            cup_a_2d, cup_radius_px=12.0)
            if ns_score is None:
                ns_score = 0.0
        else:
            cup_b_2d = project_to_image(cup_b)
            from .suction_score import normal_flatness_dual
            ns_score = normal_flatness_dual(ctx["grid"], ctx["valid"],
                                              cup_a_2d, cup_b_2d,
                                              cup_radius_px=12.0)
    except Exception:
        ns_score = 0.0

    valid_ratio = 1.0 - ctx.get("nan_ratio", 0.0)

    # single_cup mode: cup_b 존재 X → cup_support 신호는 cup_a만 보고 계산.
    # _compute_confidence는 min(a, b)/50을 쓰므로 b에 a 값을 넘겨 a 단독 평가가
    # 되게 함 (n_b 자체는 진단용으로 0 보존).
    cup_b_for_signal = n_a if single_cup else n_b
    confidence, breakdown = _compute_confidence(
        cid=cid,
        valid_ratio=valid_ratio,
        cup_a_support=n_a, cup_b_support=cup_b_for_signal,
        normal_std_score=ns_score,
        obj_long_length_mm=obj_long_length,
        occlusion_clear=occ_clear,
    )
    # ctx에 저장 (PICKABLE이 _build_pickable_result에서 재계산 안 하도록)
    ctx["_signals_confidence"] = confidence
    ctx["_signals_breakdown"] = breakdown
    ctx["_signals_n_a"] = n_a
    ctx["_signals_n_b"] = n_b
    return confidence, breakdown


def _inject_signals_into_fail(pr: PickResult, ctx: dict) -> PickResult:
    """fail PickResult에 6 신호 + tier inject (logistic regression 학습용)."""
    res = _compute_all_signals(ctx)
    if res is None:
        return pr
    confidence, breakdown = res
    pr.confidence = float(confidence)
    pr.tier = _classify_tier(confidence)
    pr.score_breakdown = dict(breakdown)
    pr.cup_a_support = ctx.get("_signals_n_a", pr.cup_a_support)
    pr.cup_b_support = ctx.get("_signals_n_b", pr.cup_b_support)
    return pr


def compute_pick(grid: np.ndarray, polygon_2d, cid: int,
                 box_roi=None, scene_polygons=None,
                 manual_pick=None) -> PickResult:
    """Top-down dual-suction pick. Returns PickResult with 3-tier status.

    Status:
      MANUAL     — 사용자가 3D에서 직접 라벨링한 GT (있으면 알고리즘 무시)
      PICKABLE   — 모든 게이트 통과
      DEFER      — 일시적 실패 (다음 라운드 재시도)
      IMPOSSIBLE — 영구 불가 (제외 클래스, 길이/평면/깊이 부족)

    분할 흐름:
      0. manual_pick GT override
      1. _check_class_and_depth   (IMPOSSIBLE 조기)
      2. _compute_initial_pose    (IMPOSSIBLE/DEFER on tilt)
      3. _refine_pose_with_heatmap (silent — pose 갱신만)
      4. _check_structural        (IMPOSSIBLE — 길이/평면/occlusion)
      5. _check_defer_gates       (DEFER — overlap/cup/rect/arm/bump)
      6. _build_pickable_result   (PICKABLE)
    """
    if manual_pick is not None:
        return _from_manual_pick(manual_pick, cid)

    ctx: dict = {
        "cid": cid, "polygon_2d": polygon_2d, "grid": grid,
        "scene_polygons": scene_polygons, "box_roi": box_roi,
    }
    for stage in (_check_class_and_depth, _compute_initial_pose):
        err = stage(ctx)
        if err is not None:
            return err  # 자세 없음 — 신호 계산 불가

    _refine_pose_with_heatmap(ctx)   # in-place; 실패해도 이전 pose 사용

    # 자세 결정 완료 — 게이트와 무관하게 6 신호 계산 (통계 모델용)
    _compute_all_signals(ctx)

    for stage in (_check_structural, _check_defer_gates):
        err = stage(ctx)
        if err is not None:
            # fail이지만 자세는 있음 → 신호를 PickResult에 inject
            return _inject_signals_into_fail(err, ctx)
    return _build_pickable_result(ctx)
