"""Suction-quality scoring + footprint clearance helpers.

Two scoring paths:
  1) `normal_flatness_at` — uses PLY's nx/ny/nz fields directly. No open3d.
     Returns flatness in [0, 1]: 1.0 = local normals are aligned (= 평면).
  2) `normal_std_score_at` — original SuctionNet-1Billion adaptation via Open3D.
     Kept for reference; only runs if open3d available.

Footprint clearance:
  `disk_protrusion_max` — max(z_pick - z) within a pixel disk; used by the
  collision gates (arm, cup A, cup B) in compute_pick.

References
  - docs/picking_research/02_suction_methods_survey.md
  - Cao et al., "SuctionNet-1Billion" RA-L 2021
"""
from __future__ import annotations
from typing import Optional, Tuple

import numpy as np

try:
    import open3d as o3d
    _HAS_OPEN3D = True
except Exception:
    _HAS_OPEN3D = False


# ─── Camera intrinsics (runtime 갱신) ─────────────────────────────────────
# 함수 default 인자가 None이면 이 글로벌을 사용. picking.set_camera_intrinsics
# 가 이걸 동기 갱신함. label_tool이 샷별 *_info.json 보고 자동 적용.
_FX = 1242.55     # default MR60
_FY = 1242.77
_CX = 605.119
_CY = 502.016


def set_camera_intrinsics(fx: float, fy: float, cx: float, cy: float) -> None:
    global _FX, _FY, _CX, _CY
    _FX, _FY, _CX, _CY = float(fx), float(fy), float(cx), float(cy)


# ─── Path 1: PLY-field-based flatness (preferred — no open3d) ────────────
def normal_flatness_at(
    grid: np.ndarray,
    valid_mask: np.ndarray,
    cup_uv: Tuple[float, float],
    cup_radius_px: float = 12.0,
) -> Optional[float]:
    """Local normal alignment within a pixel disk.

    Uses Zivid's per-pixel normals (PLY nx/ny/nz fields) directly.
    Score = mean of dot(n_i, n_bar) where n_bar is the patch mean normal.
    1.0 = perfect plane. 0.0 = normals are random.

    Returns None if disk has too few valid pixels.
    """
    H, W = valid_mask.shape
    cu, cv = float(cup_uv[0]), float(cup_uv[1])
    u0 = max(0, int(np.floor(cu - cup_radius_px)))
    u1 = min(W, int(np.ceil(cu + cup_radius_px)) + 1)
    v0 = max(0, int(np.floor(cv - cup_radius_px)))
    v1 = min(H, int(np.ceil(cv + cup_radius_px)) + 1)
    if u1 <= u0 or v1 <= v0:
        return None
    sub_v = valid_mask[v0:v1, u0:u1]
    yy, xx = np.mgrid[v0:v1, u0:u1]
    in_disk = (xx - cu) ** 2 + (yy - cv) ** 2 <= cup_radius_px ** 2
    elig = sub_v & in_disk
    if elig.sum() < 5:
        return None
    nx = grid["nx"][v0:v1, u0:u1][elig]
    ny = grid["ny"][v0:v1, u0:u1][elig]
    nz = grid["nz"][v0:v1, u0:u1][elig]
    finite = np.isfinite(nx) & np.isfinite(ny) & np.isfinite(nz)
    if finite.sum() < 5:
        return None
    nx, ny, nz = nx[finite], ny[finite], nz[finite]
    mx, my, mz = nx.mean(), ny.mean(), nz.mean()
    mag = float(np.sqrt(mx * mx + my * my + mz * mz))
    if mag < 1e-6:
        return 0.0
    mx, my, mz = mx / mag, my / mag, mz / mag
    align = float((nx * mx + ny * my + nz * mz).mean())
    return float(np.clip(align, 0.0, 1.0))


def normal_flatness_dual(
    grid: np.ndarray,
    valid_mask: np.ndarray,
    cup_a_uv: Optional[Tuple[float, float]],
    cup_b_uv: Optional[Tuple[float, float]],
    cup_radius_px: float = 12.0,
) -> float:
    """Min flatness across the two cups (single bad cup pulls score down)."""
    if cup_a_uv is None or cup_b_uv is None:
        return 0.0
    sa = normal_flatness_at(grid, valid_mask, cup_a_uv, cup_radius_px)
    sb = normal_flatness_at(grid, valid_mask, cup_b_uv, cup_radius_px)
    if sa is None or sb is None:
        return 0.0
    return float(min(sa, sb))


# ─── Approach cylinder check — proper 3D collision along surface normal ──
def approach_cylinder_check(
    grid: np.ndarray,
    valid_mask: np.ndarray,
    position: np.ndarray,       # (3,) pick center, camera frame mm
    normal: np.ndarray,         # (3,) approach axis (toward camera, unit)
    radius_mm: float,
    t_min_mm: float = 0.0,      # 시작 (픽 표면 = 0)
    t_max_mm: float = 200.0,    # 끝 (팔 길이 위)
    self_mask: Optional[np.ndarray] = None,
    fx: Optional[float] = None,
    fy: Optional[float] = None,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
) -> dict:
    """픽 좌표계 Z축(=normal) 방향 원기둥 안 collision check.

    Cylinder: axis = position + t * normal,  t ∈ [t_min_mm, t_max_mm]
              radius = radius_mm

    Returns dict with:
        n_in_cylinder: 원기둥 안 유효 점 개수
        n_outside:     self_mask 제외한 점 개수 (진짜 충돌)
        max_t_mm:      outside 점 중 normal 방향 최대 거리 (= 가장 가까이 솟음)
    """
    if fx is None: fx = _FX
    if fy is None: fy = _FY
    if cx is None: cx = _CX
    if cy is None: cy = _CY
    H, W = valid_mask.shape
    # Image-space bounding box로 검사 영역 제한 (모든 픽셀 검사 X)
    # 30° tilt에서 t_max=200mm면 옆방향 100mm 빠짐. 보수적으로 radius+0.5*t_max.
    z_pick = float(position[2])
    if z_pick <= 1e-6:
        return {"n_in_cylinder": 0, "n_outside": 0, "max_t_mm": 0.0}
    fx_avg = 0.5 * (fx + fy)
    margin_px = (radius_mm + 0.5 * t_max_mm) * fx_avg / z_pick
    pu = fx * float(position[0]) / z_pick + cx
    pv = fy * float(position[1]) / z_pick + cy
    u0 = max(0, int(np.floor(pu - margin_px)))
    u1 = min(W, int(np.ceil(pu + margin_px)) + 1)
    v0 = max(0, int(np.floor(pv - margin_px)))
    v1 = min(H, int(np.ceil(pv + margin_px)) + 1)
    if u1 <= u0 or v1 <= v0:
        return {"n_in_cylinder": 0, "n_outside": 0, "max_t_mm": 0.0}

    sub_x = grid["x"][v0:v1, u0:u1]
    sub_y = grid["y"][v0:v1, u0:u1]
    sub_z = grid["z"][v0:v1, u0:u1]
    sub_v = valid_mask[v0:v1, u0:u1]

    # 각 픽셀 3D 위치 - position
    vx = sub_x - position[0]
    vy = sub_y - position[1]
    vz = sub_z - position[2]
    # Normal 방향 성분 (= t)
    t = vx * normal[0] + vy * normal[1] + vz * normal[2]
    # Perpendicular 거리 제곱
    perp_x = vx - t * normal[0]
    perp_y = vy - t * normal[1]
    perp_z = vz - t * normal[2]
    r2 = perp_x ** 2 + perp_y ** 2 + perp_z ** 2

    in_cyl = sub_v & (t > t_min_mm) & (t < t_max_mm) & (r2 < radius_mm ** 2)
    n_in = int(in_cyl.sum())

    if self_mask is not None:
        sub_self = self_mask[v0:v1, u0:u1]
        outside = in_cyl & ~sub_self
    else:
        outside = in_cyl
    n_outside = int(outside.sum())
    if n_outside == 0:
        return {"n_in_cylinder": n_in, "n_outside": 0, "max_t_mm": 0.0}
    return {
        "n_in_cylinder": n_in,
        "n_outside": n_outside,
        "max_t_mm": float(t[outside].max()),
    }


# ─── Pickability heatmap — 폴리곤 전체 per-pixel score ────────────────────
def pickability_heatmap(
    grid: np.ndarray,
    valid_mask: np.ndarray,
    self_mask: np.ndarray,
    cup_radius_px: float = 25.0,    # ≈12.5mm at z=600 (cup OD/2 in pixels)
    arm_radius_px: float = 100.0,   # ≈50mm at z=600 (arm radius in pixels) — 정보용
    bump_max_mm: float = 5.0,       # std-from-local-plane (tilt 영향 X)
    arm_max_mm: float = 30.0,       # arm score 정보용. score 곱연산엔 미사용
    use_arm_score: bool = False,    # True면 arm을 score에 곱함. 빈 환경에선 거의 0이라 default False
) -> np.ndarray:
    """폴리곤 안 각 픽셀의 cup-as-center 픽 적합도 (3D 기반) [0..1].

    score = flatness × bump_score × arm_score × edge_score × valid_neighborhood

    각 항 모두 [0..1]:
      flatness    cup window 안 normal 정렬 정도 (3D)
      bump_score  cup window 안 z 변동 / bump_max_mm (3D — 흡착 밀착 검증)
                   1.0 = 완벽 평면, 0 = 5mm 이상 굴곡 → 진공 새 못 잡음
      arm_score   arm window 안 위에 솟은 픽셀 (3D — 옆 물체 충돌 검증)
                   1.0 = 위에 아무것도 없음, 0 = 10mm 이상 위로 솟음
      edge_score  폴리곤 가장자리 거리 / cup_radius_px (2D — cup이 라벨 밖으로 빠짐 방지)
      valid_nbhd  깊이 valid 픽셀 비율 ≥ 0.5

    폴리곤 밖과 self_mask 안 invalid 픽셀은 NaN.

    [최적화] self_mask의 bbox + margin만 잘라서 ROI에서 계산. 결과는 전체
    grid 크기로 padding back(NaN). 폴리곤이 grid보다 훨씬 작아 큰 속도 향상.
    """
    H, W = valid_mask.shape
    cup_win = int(2 * cup_radius_px + 1)
    if cup_win % 2 == 0:
        cup_win += 1
    arm_win = int(2 * arm_radius_px + 1)
    if arm_win % 2 == 0:
        arm_win += 1

    # ── ROI = self_mask bbox + (cup_win + arm_win)/2 margin ─────────────
    ys_m, xs_m = np.where(self_mask)
    if len(xs_m) == 0:
        return np.full((H, W), np.nan, dtype=np.float32)
    margin = (cup_win + arm_win) // 2 + 2   # box filter edge effect 여유
    v0 = max(0, int(ys_m.min()) - margin)
    v1 = min(H, int(ys_m.max()) + margin + 1)
    u0 = max(0, int(xs_m.min()) - margin)
    u1 = min(W, int(xs_m.max()) + margin + 1)

    # ROI slice
    grid_r = grid[v0:v1, u0:u1]
    valid_r = valid_mask[v0:v1, u0:u1]
    self_r = self_mask[v0:v1, u0:u1]

    # ── Flatness — normal alignment in cup window ─────────────────────
    nx = grid_r["nx"].astype(np.float32)
    ny = grid_r["ny"].astype(np.float32)
    nz = grid_r["nz"].astype(np.float32)
    v_f = valid_r.astype(np.float32)
    nx_v = np.where(valid_r, nx, 0.0)
    ny_v = np.where(valid_r, ny, 0.0)
    nz_v = np.where(valid_r, nz, 0.0)
    valid_frac = _box_uniform_filter(v_f, cup_win)
    eps = 1e-6
    mean_nx = _box_uniform_filter(nx_v, cup_win) / np.maximum(valid_frac, eps)
    mean_ny = _box_uniform_filter(ny_v, cup_win) / np.maximum(valid_frac, eps)
    mean_nz = _box_uniform_filter(nz_v, cup_win) / np.maximum(valid_frac, eps)
    flatness = np.sqrt(mean_nx ** 2 + mean_ny ** 2 + mean_nz ** 2)
    flatness = np.clip(flatness, 0.0, 1.0)

    # ── Bump + Arm clearance ──────────────────────────────────────────
    z = grid_r["z"].astype(np.float32)
    self_valid = self_r & valid_r
    outside_valid = (~self_r) & valid_r
    try:
        from scipy.ndimage import minimum_filter
        sv_f = self_valid.astype(np.float32)
        z_self0 = np.where(self_valid, z, 0.0)
        cnt = _box_uniform_filter(sv_f, cup_win)
        z_smooth = _box_uniform_filter(z_self0, cup_win) / np.maximum(cnt, eps)
        residual = np.where(self_valid, z - z_smooth, 0.0)
        var_resid = (
            _box_uniform_filter(residual * residual, cup_win)
            / np.maximum(cnt, eps)
        )
        std_resid = np.sqrt(np.maximum(var_resid, 0.0))
        bump_score = np.clip(1.0 - std_resid / bump_max_mm, 0.0, 1.0)

        z_outside_min = minimum_filter(np.where(outside_valid, z, np.inf),
                                        size=arm_win)
        obstacle_mm = np.where(
            np.isfinite(z_outside_min) & np.isfinite(z),
            np.maximum(z - z_outside_min, 0.0),
            0.0,
        )
        arm_score = np.clip(1.0 - obstacle_mm / arm_max_mm, 0.0, 1.0)
    except Exception:
        bump_score = np.ones_like(flatness)
        arm_score = np.ones_like(flatness)

    # ── Edge distance (ROI 안에서만 distance transform) ───────────────
    try:
        from scipy.ndimage import distance_transform_edt
        edge_dist = distance_transform_edt(self_r).astype(np.float32)
    except Exception:
        edge_dist = self_r.astype(np.float32) * cup_radius_px
    edge_score = np.clip(edge_dist / cup_radius_px, 0.0, 1.0)

    # ── Combine + padding back to full grid ───────────────────────────
    valid_nbhd = (valid_frac >= 0.5).astype(np.float32)
    score_roi = flatness * bump_score * edge_score * valid_nbhd
    if use_arm_score:
        score_roi = score_roi * arm_score
    score_roi = np.where(self_r, score_roi, np.nan)

    out = np.full((H, W), np.nan, dtype=np.float32)
    out[v0:v1, u0:u1] = score_roi
    return out


# ─── Cup self-coverage — cup이 폴리곤을 얼마나 덮나 ───────────────────────
def cup_self_coverage(
    self_mask: np.ndarray,
    cup_3d: np.ndarray,           # (3,) cup 중심 (camera frame mm)
    radius_mm: float = 12.5,
    fx: Optional[float] = None,
    fy: Optional[float] = None,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
) -> Optional[float]:
    """컵 OD 디스크(2D 이미지)가 self 폴리곤 안에 들어있는 비율.

    < 1.0이면 컵 일부가 폴리곤 밖 = 라벨 객체 표면이 아닌 곳에 닿음.
    가장자리 모서리만 살짝 빠지는 건 OK (~0.85+), 절반 이상 빠지면 흡착 실패.

    Returns ratio in [0, 1] or None if cup projects off-image.
    """
    if fx is None: fx = _FX
    if fy is None: fy = _FY
    if cx is None: cx = _CX
    if cy is None: cy = _CY
    z_pick = float(cup_3d[2])
    if z_pick <= 1e-6:
        return None
    fx_avg = 0.5 * (fx + fy)
    r_px = radius_mm * fx_avg / z_pick
    cu = fx * float(cup_3d[0]) / z_pick + cx
    cv = fy * float(cup_3d[1]) / z_pick + cy
    H, W = self_mask.shape
    u0 = max(0, int(np.floor(cu - r_px - 1)))
    u1 = min(W, int(np.ceil(cu + r_px + 2)))
    v0 = max(0, int(np.floor(cv - r_px - 1)))
    v1 = min(H, int(np.ceil(cv + r_px + 2)))
    if u1 <= u0 or v1 <= v0:
        return None
    yy, xx = np.mgrid[v0:v1, u0:u1]
    in_disk = (xx - cu) ** 2 + (yy - cv) ** 2 <= r_px ** 2
    n_disk = int(in_disk.sum())
    if n_disk < 1:
        return None
    n_self = int((in_disk & self_mask[v0:v1, u0:u1]).sum())
    return float(n_self / n_disk)


# ─── Cup rim coverage — 둘레 sample이 폴리곤 안에 들어가는 비율 ────────
def cup_rim_coverage(
    self_mask: np.ndarray,
    cup_3d: np.ndarray,
    radius_mm: float = 12.5,
    n_samples: int = 24,
    fx: Optional[float] = None,
    fy: Optional[float] = None,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
) -> Optional[float]:
    """컵 OD 둘레(rim) 점들이 self 폴리곤 안에 들어있는 비율.

    면적 기반 cup_self_coverage는 disk 면적의 절반이 폴리곤 안이면 0.5지만,
    흡착 물리는 둘레 ring 전체가 surface에 닿아야 진공이 형성됨 — 둘레 한쪽이
    끊기면 그 부분으로 공기 유입 → 못 잡음. 본 함수는 둘레를 n_samples점으로
    discretize해서 self_mask 안 비율을 반환한다. 0.95+면 거의 완전 sealed.

    Returns ratio in [0, 1] or None if cup projects off-image.
    """
    if fx is None: fx = _FX
    if fy is None: fy = _FY
    if cx is None: cx = _CX
    if cy is None: cy = _CY
    z_pick = float(cup_3d[2])
    if z_pick <= 1e-6:
        return None
    fx_avg = 0.5 * (fx + fy)
    r_px = radius_mm * fx_avg / z_pick
    cu = fx * float(cup_3d[0]) / z_pick + cx
    cv = fy * float(cup_3d[1]) / z_pick + cy
    H, W = self_mask.shape
    angles = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    us = cu + r_px * np.cos(angles)
    vs = cv + r_px * np.sin(angles)
    ui = np.round(us).astype(int)
    vi = np.round(vs).astype(int)
    in_image = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    if not in_image.any():
        return None
    in_poly = np.zeros(n_samples, dtype=bool)
    in_poly[in_image] = self_mask[vi[in_image], ui[in_image]]
    return float(in_poly.sum() / n_samples)


# ─── Cup seal — within-cup-disk surface roughness ────────────────────────
def cup_max_bump(
    grid: np.ndarray,
    valid_mask: np.ndarray,
    cup_3d: np.ndarray,           # (3,) cup 중심 (3D, mm, camera frame)
    normal: np.ndarray,           # (3,) 표면 법선 (toward camera, unit)
    radius_mm: float = 12.5,      # 컵 OD/2 = 흡착 접촉 반경
    self_mask: Optional[np.ndarray] = None,
    fx: Optional[float] = None,
    fy: Optional[float] = None,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
) -> Optional[float]:
    """컵 OD 안 표면이 얼마나 울퉁불퉁한지 (mm).

    컵이 surface에 밀착하려면 컵 둘레가 평평한 ring 위에 닿아야 함.
    안에 돌출이 있으면 컵이 그 위에 얹혀서 둘레는 들뜸 → 진공 새 → 못 잡음.

    Returns:
        max bump (mm) — disk 안 normal 방향 정사영의 (max − median).
        None if disk has too few points.
    """
    if fx is None: fx = _FX
    if fy is None: fy = _FY
    if cx is None: cx = _CX
    if cy is None: cy = _CY
    z_pick = float(cup_3d[2])
    if z_pick <= 1e-6:
        return None
    fx_avg = 0.5 * (fx + fy)
    r_px = radius_mm * fx_avg / z_pick

    # 컵 중심 image projection
    cu = fx * float(cup_3d[0]) / z_pick + cx
    cv = fy * float(cup_3d[1]) / z_pick + cy

    H, W = valid_mask.shape
    u0 = max(0, int(np.floor(cu - r_px)))
    u1 = min(W, int(np.ceil(cu + r_px)) + 1)
    v0 = max(0, int(np.floor(cv - r_px)))
    v1 = min(H, int(np.ceil(cv + r_px)) + 1)
    if u1 <= u0 or v1 <= v0:
        return None

    # 그 영역에서 cup_3d로부터 normal 방향 정사영 t = (point - cup_3d) @ normal
    sub_x = grid["x"][v0:v1, u0:u1]
    sub_y = grid["y"][v0:v1, u0:u1]
    sub_z = grid["z"][v0:v1, u0:u1]
    sub_v = valid_mask[v0:v1, u0:u1]

    vx = sub_x - cup_3d[0]
    vy = sub_y - cup_3d[1]
    vz = sub_z - cup_3d[2]
    t = vx * normal[0] + vy * normal[1] + vz * normal[2]
    perp_x = vx - t * normal[0]
    perp_y = vy - t * normal[1]
    perp_z = vz - t * normal[2]
    r2 = perp_x ** 2 + perp_y ** 2 + perp_z ** 2

    in_disk_3d = sub_v & (r2 <= radius_mm ** 2)
    if self_mask is not None:
        # Self surface 안 점만 — 외부 점은 진입 충돌 검사에서 따로 처리
        sub_self = self_mask[v0:v1, u0:u1]
        in_disk_3d = in_disk_3d & sub_self
    if int(in_disk_3d.sum()) < 5:
        return None
    t_vals = t[in_disk_3d]
    return float(np.percentile(t_vals, 95) - np.percentile(t_vals, 50))


# ─── Approach box check — 직사각형 footprint collision ────────────────────
def approach_box_check(
    grid: np.ndarray,
    valid_mask: np.ndarray,
    position: np.ndarray,
    normal: np.ndarray,
    long_axis: np.ndarray,
    mid_axis: np.ndarray,
    half_long_mm: float,
    half_short_mm: float,
    t_min_mm: float = 0.0,
    t_max_mm: float = 200.0,
    self_mask: Optional[np.ndarray] = None,
    fx: Optional[float] = None,
    fy: Optional[float] = None,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
) -> dict:
    """픽 frame의 직사각형 box 안 collision check.

    Box (픽 좌표계):
        x (long_axis): [-half_long_mm, +half_long_mm]
        y (mid_axis):  [-half_short_mm, +half_short_mm]
        z (normal):    [t_min_mm, t_max_mm]

    self_mask 안 점은 self surface로 무시. outside만 obstacle로 카운트.

    Returns: {n_outside, max_t_mm}
    """
    if fx is None: fx = _FX
    if fy is None: fy = _FY
    if cx is None: cx = _CX
    if cy is None: cy = _CY
    H, W = valid_mask.shape
    z_pick = float(position[2])
    if z_pick <= 1e-6:
        return {"n_outside": 0, "max_t_mm": 0.0}
    fx_avg = 0.5 * (fx + fy)
    # 보수적 image bbox: max(half_long, half_short) + 0.5 * t_max (tilt 보정)
    margin_mm = max(half_long_mm, half_short_mm) + 0.5 * t_max_mm
    margin_px = margin_mm * fx_avg / z_pick
    pu = fx * float(position[0]) / z_pick + cx
    pv = fy * float(position[1]) / z_pick + cy
    u0 = max(0, int(np.floor(pu - margin_px)))
    u1 = min(W, int(np.ceil(pu + margin_px)) + 1)
    v0 = max(0, int(np.floor(pv - margin_px)))
    v1 = min(H, int(np.ceil(pv + margin_px)) + 1)
    if u1 <= u0 or v1 <= v0:
        return {"n_outside": 0, "max_t_mm": 0.0}

    sub_x = grid["x"][v0:v1, u0:u1]
    sub_y = grid["y"][v0:v1, u0:u1]
    sub_z = grid["z"][v0:v1, u0:u1]
    sub_v = valid_mask[v0:v1, u0:u1]

    vx = sub_x - position[0]
    vy = sub_y - position[1]
    vz = sub_z - position[2]
    # Pick frame projections
    t = vx * normal[0] + vy * normal[1] + vz * normal[2]
    px = vx * long_axis[0] + vy * long_axis[1] + vz * long_axis[2]
    py = vx * mid_axis[0] + vy * mid_axis[1] + vz * mid_axis[2]

    in_box = (sub_v
              & (t > t_min_mm) & (t < t_max_mm)
              & (np.abs(px) < half_long_mm)
              & (np.abs(py) < half_short_mm))
    if self_mask is not None:
        sub_self = self_mask[v0:v1, u0:u1]
        outside = in_box & ~sub_self
    else:
        outside = in_box
    n_outside = int(outside.sum())
    if n_outside == 0:
        return {"n_outside": 0, "max_t_mm": 0.0}
    return {"n_outside": n_outside, "max_t_mm": float(t[outside].max())}


# ─── Footprint clearance — used by collision gates ────────────────────────
def disk_protrusion_max(
    z_field: np.ndarray,
    valid_mask: np.ndarray,
    cu: float, cv: float, r_px: float,
    z_pick: float,
    self_mask: Optional[np.ndarray] = None,
) -> dict:
    """Max protrusion (= z_pick - z_other) inside a pixel disk.

    A pixel "protrudes" if it sits closer to camera than the pick surface
    (smaller z in Zivid frame). Used to detect adjacent-object collision.

    If self_mask is given, splits pixels into inside/outside polygon —
    only outside protrusions are real obstacles (inside = self curvature).

    Returns dict with at minimum {n_eligible, max_protrusion_mm}.
    """
    H, W = z_field.shape
    u0 = max(0, int(np.floor(cu - r_px)))
    u1 = min(W, int(np.ceil(cu + r_px)) + 1)
    v0 = max(0, int(np.floor(cv - r_px)))
    v1 = min(H, int(np.ceil(cv + r_px)) + 1)
    if u1 <= u0 or v1 <= v0:
        return {"n_eligible": 0, "max_protrusion_mm": 0.0,
                "outside_max_protrusion_mm": 0.0}
    yy, xx = np.mgrid[v0:v1, u0:u1]
    in_disk = (xx - cu) ** 2 + (yy - cv) ** 2 <= r_px ** 2
    sub_valid = valid_mask[v0:v1, u0:u1]
    eligible = in_disk & sub_valid
    if not eligible.any():
        return {"n_eligible": 0, "max_protrusion_mm": 0.0,
                "outside_max_protrusion_mm": 0.0}
    z_vals = z_field[v0:v1, u0:u1][eligible]
    protrusion = z_pick - z_vals
    out = {
        "n_eligible": int(eligible.sum()),
        "max_protrusion_mm": float(max(0.0, protrusion.max())),
    }
    if self_mask is not None:
        sub_self = self_mask[v0:v1, u0:u1]
        outside = eligible & ~sub_self
        if outside.any():
            z_out = z_field[v0:v1, u0:u1][outside]
            out["outside_max_protrusion_mm"] = float(max(0.0, (z_pick - z_out).max()))
        else:
            out["outside_max_protrusion_mm"] = 0.0
    else:
        out["outside_max_protrusion_mm"] = out["max_protrusion_mm"]
    return out


def _box_uniform_filter(arr: np.ndarray, size: int) -> np.ndarray:
    """2D box (uniform) filter via cumulative-sum trick — no scipy dependency.
    Pads with edge replication. `arr` is (H, W). `size` is window side (odd).
    """
    if size <= 1:
        return arr.copy()
    h, w = arr.shape
    r = size // 2
    # Edge-replicate padding
    padded = np.pad(arr, ((r + 1, r), (r + 1, r)), mode="edge")
    csum = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    # Sum over window via inclusion-exclusion on cumulative sum
    s = (
        csum[size:, size:]
        - csum[:-size, size:]
        - csum[size:, :-size]
        + csum[:-size, :-size]
    )
    return s / float(size * size)


def normal_std_score_at(
    grid: np.ndarray,
    valid_mask: np.ndarray,
    cup_uv: Tuple[float, float],
    cup_radius_px: float = 12.0,
    knn: int = 64,
    window: int = 25,
) -> float:
    """Average Normal-STD-based flatness score (0..1) within a small disk
    around `cup_uv` on the image grid.

    Args:
        grid: organized PLY (H,W) with x/y/z fields
        valid_mask: (H,W) bool — True where 3D is valid
        cup_uv: (u, v) image-pixel center of the cup
        cup_radius_px: search radius around cup_uv in pixels
        knn: KDTree neighbors for normal estimation
        window: STD-filter window size (odd)

    Returns:
        Score in [0, 1]. Higher = flatter / better suction surface.
        Returns 0 if Open3D unavailable, no valid points near cup, or
        any error occurs (defensive default = "not confident").
    """
    if not _HAS_OPEN3D:
        return 0.0
    h, w = valid_mask.shape
    cu, cv = float(cup_uv[0]), float(cup_uv[1])
    # Pixel disk around cup center
    u0 = max(0, int(np.floor(cu - cup_radius_px)))
    u1 = min(w, int(np.ceil(cu + cup_radius_px)) + 1)
    v0 = max(0, int(np.floor(cv - cup_radius_px)))
    v1 = min(h, int(np.ceil(cv + cup_radius_px)) + 1)
    if u1 <= u0 or v1 <= v0:
        return 0.0

    sub_valid = valid_mask[v0:v1, u0:u1]
    if not sub_valid.any():
        return 0.0

    sub_grid = grid[v0:v1, u0:u1]
    xyz = np.stack([sub_grid["x"], sub_grid["y"], sub_grid["z"]], -1).astype(np.float64)
    pts = xyz[sub_valid]
    if len(pts) < knn // 2:
        return 0.0

    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        eff_knn = min(knn, len(pts))
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=eff_knn))
        pcd.orient_normals_towards_camera_location([0.0, 0.0, 0.0])
        normals = np.asarray(pcd.normals)
    except Exception:
        return 0.0

    # Build local normal map (sub-region) — invalid pixels filled with 0
    sh, sw = sub_valid.shape
    nmap = np.zeros((sh, sw, 3), dtype=np.float64)
    nmap[sub_valid] = normals

    # 2D STD filter on each component, then mean across components
    stds = []
    for c in range(3):
        ch = nmap[..., c]
        m = _box_uniform_filter(ch, window)
        m2 = _box_uniform_filter(ch * ch, window)
        stds.append(np.sqrt(np.maximum(m2 - m * m, 0)))
    mean_std = np.mean(stds, axis=0)

    # Score per pixel: 1 - normalized std. Use within-region max for normalization.
    max_std = float(mean_std.max())
    if max_std < 1e-9:
        return 1.0   # uniformly flat region — perfect score
    score_map = 1.0 - mean_std / max_std

    # Restrict to disk around cup center
    yy, xx = np.mgrid[0:sh, 0:sw]
    dx = xx + u0 - cu
    dy = yy + v0 - cv
    disk = (dx * dx + dy * dy) <= (cup_radius_px * cup_radius_px)
    eligible = disk & sub_valid
    if not eligible.any():
        return 0.0
    return float(np.clip(score_map[eligible].mean(), 0.0, 1.0))


def normal_std_score_dual(
    grid: np.ndarray,
    valid_mask: np.ndarray,
    cup_a_uv: Optional[Tuple[float, float]],
    cup_b_uv: Optional[Tuple[float, float]],
    cup_radius_px: float = 12.0,
) -> float:
    """Worst of the two cups (so a single bad cup pulls the score down)."""
    if cup_a_uv is None or cup_b_uv is None:
        return 0.0
    sa = normal_std_score_at(grid, valid_mask, cup_a_uv, cup_radius_px)
    sb = normal_std_score_at(grid, valid_mask, cup_b_uv, cup_radius_px)
    return float(min(sa, sb))
