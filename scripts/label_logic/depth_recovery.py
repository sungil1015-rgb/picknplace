"""Polygon 내부의 NaN depth 복원 — 두 방법 제공.

Method A: scipy 2D 선형 보간 (recover_xyz_in_mask)
  - 모든 클래스 적용 가능
  - 평면 보간이라 곡면 부정확
  - 단순/빠름

Method C: RANSAC 원기둥 fit + 광선 교차 (recover_xyz_in_mask_cylinder)
  - bottle 같은 원기둥 객체 전용
  - 유효 점에 원기둥 fit → 축/반경 → NaN 픽셀 광선-원기둥 교차로 z 산출
  - 곡률 반영 → cup_support / coverage / planarity 게이트와 호환

사용 위치: picking.py의 _check_class_and_depth — NaN 비율이
recovery.nan_ratio_trigger 이상이면 호출. bottle 클래스는 cylinder method,
다른 클래스는 linear method (config로 선택).
"""
from __future__ import annotations

import numpy as np

try:
    from scipy.interpolate import griddata
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    from scipy.optimize import least_squares as _least_squares
    _HAS_LSQ = True
except ImportError:
    _HAS_LSQ = False


def _fit_cylinder_lsq(pts, R, init_P0, init_dir, n_trim=3, trim_mm=12.0):
    """반경 R 고정 최소제곱 원기둥 fit (axis 위치+방향 5-DOF).

    앞·뒤 벽이 둘 다 축에서 R만큼 떨어져야 한다는 제약이 축 위치와 tilt를
    강하게 묶음 (PCA보다 tilt 신뢰도 높음). inlier trimming으로 노이즈 강건성.
    Returns (P0, axis_dir, resid_abs) — resid_abs = |perp_dist - R| 전체 점 기준.
    """
    def dvec(th, ph):
        return np.array([np.sin(th) * np.cos(ph),
                         np.sin(th) * np.sin(ph), np.cos(th)])

    def perp(P, P0, d):
        v = P - P0
        return np.linalg.norm(v - np.outer(v @ d, d), axis=1)

    d0 = init_dir / (np.linalg.norm(init_dir) + 1e-12)
    th = float(np.arccos(np.clip(d0[2], -1, 1)))
    ph = float(np.arctan2(d0[1], d0[0]))
    P0 = np.asarray(init_P0, dtype=np.float64)
    cur = pts
    for _ in range(n_trim + 1):
        x0 = np.r_[P0, th, ph]
        sol = _least_squares(
            lambda x: perp(cur, x[:3], dvec(x[3], x[4])) - R,
            x0, method="lm", max_nfev=2000)
        P0 = sol.x[:3]
        d = dvec(sol.x[3], sol.x[4])
        th, ph = sol.x[3], sol.x[4]
        rr_all = perp(pts, P0, d)
        inl = np.abs(rr_all - R) < trim_mm
        if int(inl.sum()) < 50:
            break
        cur = pts[inl]
    return P0, d, np.abs(perp(pts, P0, d) - R)


def recover_xyz_in_mask(
    grid: np.ndarray,
    mask: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
) -> tuple[np.ndarray, int]:
    """polygon mask 안 NaN depth를 보간으로 채움.

    Args:
        grid: PLY structured array (H, W), fields x/y/z/r/g/b/nx/ny/nz
        mask: bool (H, W) — 폴리곤 영역
        fx, fy, cx, cy: 카메라 intrinsic (x, y 재계산용)

    Returns:
        (new_grid, n_filled). new_grid는 mask 안 NaN xyz가 채워진 복사본.
        n_filled는 채워진 픽셀 개수 (호출자 로깅용).
    """
    if not _HAS_SCIPY:
        return grid, 0

    z = grid["z"].astype(np.float64)
    valid = ~np.isnan(z)
    nan_in_mask = mask & ~valid
    valid_in_mask = mask & valid

    n_nan = int(nan_in_mask.sum())
    n_valid = int(valid_in_mask.sum())
    if n_nan == 0 or n_valid < 4:
        return grid, 0

    valid_uv = np.argwhere(valid_in_mask)   # (N, 2) row=v, col=u
    valid_z = z[valid_in_mask]
    nan_uv = np.argwhere(nan_in_mask)

    # linear 보간 (convex hull 안에서만 작동)
    filled = griddata(valid_uv, valid_z, nan_uv, method="linear")
    # hull 밖 점들은 nearest로 채움
    if np.isnan(filled).any():
        nn = griddata(valid_uv, valid_z, nan_uv, method="nearest")
        filled = np.where(np.isnan(filled), nn, filled)

    # 새 grid 복사 (원본 보호)
    new_grid = grid.copy()
    z_orig_dtype = grid["z"].dtype
    new_z = z.copy()
    new_z[nan_in_mask] = filled

    # x, y 재계산: intrinsic 역투영
    H, W = z.shape
    vv, uu = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    nan_u = uu[nan_in_mask].astype(np.float64)
    nan_v = vv[nan_in_mask].astype(np.float64)
    new_x_vals = (nan_u - cx) * filled / fx
    new_y_vals = (nan_v - cy) * filled / fy

    new_x = grid["x"].astype(np.float64).copy()
    new_y = grid["y"].astype(np.float64).copy()
    new_x[nan_in_mask] = new_x_vals
    new_y[nan_in_mask] = new_y_vals

    new_grid["x"] = new_x.astype(grid["x"].dtype)
    new_grid["y"] = new_y.astype(grid["y"].dtype)
    new_grid["z"] = new_z.astype(z_orig_dtype)
    # normal은 복원 안 함 — picking.py의 normal 평균은 finite 체크로 알아서 건너뜀
    return new_grid, n_nan


# ============================================================================
# Method C — RANSAC 원기둥 fit + 광선 교차 (bottle 전용)
# ============================================================================

def _fit_cylinder_pca(
    points: np.ndarray,
    known_radius_mm: float | None = None,
    radius_tolerance_mm: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """3D 점군에 원기둥 fit (PCA + 가시 표면 bias 보정).

    카메라가 원기둥의 일부 표면만 보면 centroid가 진짜 축에서 벗어남
    (가시 표면 쪽으로 ~2r/π 만큼). 이를 보정해서 진짜 축 위치 추정.

    Args:
        points: (N, 3) 유효 3D 점 (카메라 좌표계, +Z = 카메라 정면)
        known_radius_mm: 알려진 반경 (있으면 radius 강제 + axis도 inlier로 정제).
                         누운 sparse 데이터에서 강건성↑.
        radius_tolerance_mm: known_radius ± 이 범위 안 점만 axis 재계산 inlier.

    Returns:
        (P0, axis_dir, radius)
          P0: 축 위 한 점 (보정된 centroid)
          axis_dir: 단위 방향 벡터 (PCA 최대 고유벡터)
          radius: known_radius 주어지면 그 값, 아니면 점들의 축까지 수직거리 median
    """
    P0_raw = points.mean(axis=0)
    centered = points - P0_raw
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigvals ascending → 마지막이 가장 큼
    axis_dir = eigvecs[:, -1]
    axis_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-12)

    # 가시 bias 보정: centroid → 축 방향으로 ~2r/π 만큼 카메라 쪽으로 치우침.
    # radius와 P0가 서로 의존이라 1-shot 보정으론 부족 → 수렴까지 iterate.
    P0 = P0_raw.copy()
    radius = float(known_radius_mm) if known_radius_mm is not None else 0.0
    for _ in range(8):
        c = points - P0
        proj = c @ axis_dir
        perp = c - np.outer(proj, axis_dir)
        if known_radius_mm is None:
            radius = float(np.median(np.linalg.norm(perp, axis=1)))
        # 카메라에서 P0까지 벡터의 축 수직 성분 → 보정 방향
        cam_to_P0 = P0  # 카메라가 원점
        along = float(np.dot(cam_to_P0, axis_dir))
        cam_perp = cam_to_P0 - along * axis_dir
        norm_p = float(np.linalg.norm(cam_perp))
        if norm_p < 1e-6:
            break
        # P0_new = P0_raw + (2r/π) * unit(cam→raw centroid_perp)
        # → raw centroid가 anchor, 매 iter마다 offset 크기만 갱신
        offset_size = 2.0 * radius / np.pi
        # raw centroid의 perp 방향이 정답에 가까움
        raw_perp = P0_raw - float(np.dot(P0_raw, axis_dir)) * axis_dir
        raw_norm = float(np.linalg.norm(raw_perp))
        if raw_norm < 1e-6:
            break
        P0_new = P0_raw + offset_size * (raw_perp / raw_norm)
        if np.linalg.norm(P0_new - P0) < 0.1:  # 0.1mm 수렴
            P0 = P0_new
            break
        P0 = P0_new

    # known_radius가 있으면 axis도 inlier 기반 재계산 (sparse 데이터 강건성).
    # inlier = |perp_dist - known_radius| < tolerance
    if known_radius_mm is not None and len(points) >= 30:
        c = points - P0
        proj = c @ axis_dir
        perp = c - np.outer(proj, axis_dir)
        rad_dist = np.linalg.norm(perp, axis=1)
        inlier_mask = np.abs(rad_dist - known_radius_mm) < radius_tolerance_mm
        if int(inlier_mask.sum()) >= 20:
            inliers = points[inlier_mask]
            P0_in = inliers.mean(axis=0)
            cov_in = np.cov((inliers - P0_in).T)
            ev_in, evec_in = np.linalg.eigh(cov_in)
            new_axis = evec_in[:, -1]
            new_axis = new_axis / (np.linalg.norm(new_axis) + 1e-12)
            # 부호 일관성 — 원래 방향과 같은 쪽
            if float(new_axis @ axis_dir) < 0:
                new_axis = -new_axis
            axis_dir = new_axis
            # P0 한 번 더 갱신 (axis 바뀌면 가시 bias 다시 적용)
            along = float(P0_in @ axis_dir)
            raw_perp = P0_in - along * axis_dir
            raw_norm = float(np.linalg.norm(raw_perp))
            if raw_norm > 1e-6:
                offset_size = 2.0 * known_radius_mm / np.pi
                P0 = P0_in + offset_size * (raw_perp / raw_norm)

    return P0, axis_dir, radius


def _ray_cylinder_intersect(
    ray_dir: np.ndarray,
    P0: np.ndarray, axis_dir: np.ndarray, radius: float,
) -> float | None:
    """카메라 원점에서 ray_dir 방향 광선 vs 무한 원기둥 표면 교차.

    Args:
        ray_dir: 단위 벡터 (카메라 좌표계, +Z 정면 가정)
        P0: 축 위 한 점
        axis_dir: 축 단위 방향
        radius: 원기둥 반경 (mm)

    Returns:
        가까운 양의 t (있으면), 없으면 None.
        교차점 = t * ray_dir (카메라 원점 기준 mm).
    """
    # 광선 = t * ray_dir
    # 원기둥: 점 Q가 축에서 수직거리 = radius
    # axis 방향 성분 제거한 ray, 원점-P0 벡터 → 2차방정식
    r_along = float(np.dot(ray_dir, axis_dir))
    r_perp = ray_dir - r_along * axis_dir
    w = -P0
    w_along = float(np.dot(w, axis_dir))
    w_perp = w - w_along * axis_dir

    a = float(np.dot(r_perp, r_perp))
    b = 2.0 * float(np.dot(r_perp, w_perp))
    c = float(np.dot(w_perp, w_perp)) - radius * radius

    if a < 1e-9:
        return None  # 광선이 축과 거의 평행
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return None  # 광선이 원기둥 빗나감
    sq = np.sqrt(disc)
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    # 카메라 정면 (t>0) 중 가까운 값
    candidates = [t for t in (t1, t2) if t > 0]
    if not candidates:
        return None
    return float(min(candidates))


def segment_front_shell(
    z_vals: np.ndarray, bw: float = 5.0, valley_frac: float = 0.5,
) -> float | None:
    """투명 병에서 앞벽+뒤(see-through) 깊이가 bimodal일 때 앞쪽 덩어리의 상한
    z(cut, mm)를 반환. unimodal/분리 약하면 None.

    2026-05-29: 투명 병은 앞면 통과해 뒷벽·바닥까지 찍혀 z 히스토그램이 두 봉우리.
    가장 가까운(앞벽) 봉우리만 떼면 흡착이 닿는 진짜 표면 = 픽 대상.
    """
    z_vals = np.asarray(z_vals, dtype=np.float64)
    zmin, zmax = float(z_vals.min()), float(z_vals.max())
    if zmax - zmin < 3 * bw:
        return None
    edges = np.arange(zmin, zmax + bw, bw)
    cnt, _ = np.histogram(z_vals, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    k = np.array([1, 2, 3, 2, 1], float); k /= k.sum()
    cs = np.convolve(cnt.astype(float), k, mode="same")
    gmax = cs.max()
    if gmax <= 0:
        return None
    thr = 0.15 * gmax
    near_peak = None
    for i in range(1, len(cs) - 1):
        if cs[i] >= thr and cs[i] >= cs[i - 1] and cs[i] >= cs[i + 1]:
            near_peak = i
            break
    if near_peak is None:
        return None
    after = cs[near_peak:]
    if len(after) < 3:
        return None
    vidx = near_peak + int(np.argmin(after))
    if vidx <= near_peak or vidx >= len(cs) - 1:
        return None
    valley = cs[vidx]
    back_mass = cs[vidx:].max()
    if valley < valley_frac * min(cs[near_peak], back_mass) and back_mass > thr:
        return float(centers[vidx])
    return None


def _fit_axis_2d_locked(pts, polygon_2d, fx, fy, cx, cy, R,
                        n_beta=41, beta_max=3.0):
    """2D 폴리곤 장축으로 in-plane 방향을 고정하고 깊이 기울기(1-DOF)만 fit.

    투명·bimodal 깊이는 noise라 3D PCA/LSQ가 깊이 tilt를 과대평가(검증: 352#1
    |axisZ| 0.63 과함). 반면 **2D 실루엣 장축은 신뢰** → 그 방향으로 투영되게
    축을 묶고, 깊이 성분 beta만 sweep해 원기둥(R) 잔차 최소인 걸 선택.
    Returns (P0, axis_dir).
    """
    poly = np.asarray(polygon_2d, dtype=np.float64)
    pc = poly - poly.mean(0)
    _, ev = np.linalg.eigh(pc.T @ pc)
    la = ev[:, -1]                       # 2D 장축 (픽셀)
    Z = float(np.median(pts[:, 2]))
    d_in = np.array([la[0] * Z / fx, la[1] * Z / fy, 0.0])
    d_in /= np.linalg.norm(d_in) + 1e-12

    # ── 깊이 tilt = 앞면(카메라쪽 표면) 깊이 변화율 ───────────────
    # 투명 병은 see-through 뒷벽 노이즈로 자유 fit이 tilt를 과/오추정. 반면
    # **카메라쪽 표면(near-depth) 능선은 축과 평행**이라 그 깊이 기울기가 곧
    # 진짜 축 tilt. (검증 2026-05-29: 평평0.00 vs 기울음0.37 정확 구분, 352#1도
    # 사실 0.37 기울음이었음.) 2D 픽셀 장축으로 bin(클러터의 3D 깊이 왜곡 회피),
    # bin별 near-z(p15) 선형 fit → mm/mm slope.
    u_px = fx * pts[:, 0] / pts[:, 2] + cx
    v_px = fy * pts[:, 1] / pts[:, 2] + cy
    proj2d = (np.stack([u_px, v_px], 1) - poly.mean(0)) @ la   # 픽셀
    slope = 0.0
    edges = np.linspace(float(proj2d.min()), float(proj2d.max()), 12)
    aa, zz = [], []
    for b in range(len(edges) - 1):
        sel = (proj2d >= edges[b]) & (proj2d < edges[b + 1])
        if int(sel.sum()) >= 20:
            aa.append(0.5 * (edges[b] + edges[b + 1]) * Z / fx)   # 픽셀→mm
            zz.append(float(np.percentile(pts[sel, 2], 15)))
    if len(aa) >= 4:
        slope = float(np.polyfit(aa, zz, 1)[0])
        slope = float(np.clip(slope, -1.5, 1.5))
    axis = d_in + slope * np.array([0.0, 0.0, 1.0])
    axis /= np.linalg.norm(axis) + 1e-12

    # 위치(축선)는 반경 R 원 fit (앞·뒤 벽 모두 R 위)
    a = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(axis, a); u /= np.linalg.norm(u) + 1e-12
    w = np.cross(axis, u)
    pu = pts @ u; pw = pts @ w
    if _HAS_LSQ:
        sol = _least_squares(lambda c: np.hypot(pu - c[0], pw - c[1]) - R,
                             [float(pu.mean()), float(pw.mean())],
                             method="lm", max_nfev=400)
        cu, cw = sol.x
    else:
        cu, cw = float(pu.mean()), float(pw.mean())
    P0 = cu * u + cw * w
    return P0, axis


def _fit_upright_bottle(pts, known_radius_mm, length_mm):
    """세워진 병 — 카메라 향한 캡 디스크만 보임. 디스크 평면 법선=축(카메라 향함),
    디스크 중심=캡 top, 몸통은 length_mm만큼 빈 안쪽(-axis)으로 연장.
    Returns (P0=disc center, axis_dir(카메라 향함), radius, amin, amax) 또는 None.
    """
    z = pts[:, 2]
    near = z <= np.percentile(z, 35)        # 카메라 쪽 = 캡/바닥 디스크
    Pn = pts[near]
    if len(Pn) < 20:
        return None
    c0 = Pn.mean(0)
    cc = Pn - c0
    _, evec = np.linalg.eigh(cc.T @ cc / len(cc))
    n = evec[:, 0]                          # 디스크 평면 법선
    if n[2] > 0:                            # 카메라(-Z) 향하게
        n = -n
    n /= np.linalg.norm(n) + 1e-12
    R = float(known_radius_mm) if known_radius_mm else \
        float(np.percentile(np.linalg.norm(cc - np.outer(cc @ n, n), axis=1), 90))
    # ★ 디스크 중심: centroid는 디스크가 가려/부분이면 치우침(114#1 27mm 오차).
    # 디스크 평면에서 반경 R 원 fit → 진짜 중심(부분 호에도 강건).
    P0 = c0
    if known_radius_mm and _HAS_LSQ:
        a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
        u = np.cross(n, a); u /= np.linalg.norm(u) + 1e-12
        w = np.cross(n, u)
        pu = Pn @ u; pw = Pn @ w
        try:
            sol = _least_squares(lambda c: np.hypot(pu - c[0], pw - c[1]) - R,
                                 [float(pu.mean()), float(pw.mean())],
                                 method="lm", max_nfev=400)
            P0 = sol.x[0] * u + sol.x[1] * w + float((Pn @ n).mean()) * n
        except Exception:
            P0 = c0
    # 캡/바닥은 axis=0, 몸통은 -length 방향(카메라 반대=빈 안쪽)
    return P0, n, R, -float(length_mm), 0.0


def fit_bottle_cylinder(
    grid: np.ndarray,
    mask: np.ndarray,
    known_radius_mm: float | None = None,
    radius_tolerance_mm: float = 8.0,
    min_valid_for_fit: int = 50,
    radius_min_mm: float = 5.0,
    radius_max_mm: float = 70.0,
    use_front_shell: bool = False,
    polygon_2d=None,
    fx: float | None = None, fy: float | None = None,
    cx: float | None = None, cy: float | None = None,
    length_mm: float = 145.0,
    upright_aspect_max: float = 1.7,
) -> dict:
    """폴리곤 안 valid 점에 원기둥을 fit해 축·중심·반경·길이를 반환 (NaN 채우기와 무관).

    NOTE: use_front_shell 기본 OFF — 앞벽만 남기면 호(arc) 모양이라 PCA 축이
    길이 방향이 아닌 단면으로 돌아가 망가짐(검증 2026-05-29). 앞·뒤 벽 모두
    같은 원기둥 위라 축 fit엔 둘 다 필요. front-shell은 픽 표면 선택용 별도 단계.

    투명 병은 깊이가 bimodal(앞벽+see-through 뒷벽)이라 free fit이 반경을 부풀림
    (실측 r=54.9 vs 실제 30). 핵심: 앞·뒤 벽이 둘 다 보이면 **centroid가 이미
    축 위**에 있으므로, _fit_cylinder_pca의 가시-표면 bias 보정(2r/π, 한쪽 면만
    보이는 가정)을 쓰면 축이 한쪽으로 밀려버림(검증: shot104#2 medRad 60→33).
    → centroid + PCA long axis + inlier 기반 axis 정제만 사용. known_radius_mm
    주면 반경 고정(없으면 inlier perp median).

    Returns dict {fit_ok, P0, axis_dir, radius_mm, center, axial_min, axial_max,
                  n_valid, inlier_ratio, front_cut, radius_src}.
    """
    z = grid["z"].astype(np.float64)
    x = grid["x"].astype(np.float64)
    y = grid["y"].astype(np.float64)
    valid = ~np.isnan(z)
    vm = mask & valid
    info = {"fit_ok": False, "n_valid": int(vm.sum()), "front_cut": None}
    if int(vm.sum()) < min_valid_for_fit:
        return info

    ys, xs = np.where(vm)
    zsel = z[ys, xs]
    # 앞벽 segment (bimodal일 때만)
    front_cut = segment_front_shell(zsel) if use_front_shell else None
    if front_cut is not None:
        keep = zsel <= front_cut
        if int(keep.sum()) >= min_valid_for_fit:
            ys, xs = ys[keep], xs[keep]
            info["front_cut"] = float(front_cut)
        else:
            front_cut = None

    pts = np.stack([x[ys, xs], y[ys, xs], z[ys, xs]], axis=1)

    # ── 세워진 병 감지: 폴리곤 종횡비 ~1(원형) + 점 법선이 카메라 향함 ──
    # 세움은 캡/바닥 디스크가 카메라 정면이라 평균 |N·cam| 높음. 가려진 누운병은
    # 종횡비는 작아도(104#6 1.34) 옆면이라 법선이 방사형→낮음(0.73). 캡up/바닥up
    # 모두 디스크가 정면이라 잡힘. (검증: 세움 0.86~0.98 vs 가려진누움 0.73 → 0.80)
    orientation = "lying"
    if polygon_2d is not None and len(np.asarray(polygon_2d)) >= 3:
        pp = np.asarray(polygon_2d, np.float64); pp -= pp.mean(0)
        evp = np.linalg.eigvalsh(pp.T @ pp)
        aspect = float(np.sqrt(max(evp) / max(min(evp), 1e-9)))
        if aspect <= upright_aspect_max:
            nx = grid["nx"][ys, xs].astype(np.float64)
            ny = grid["ny"][ys, xs].astype(np.float64)
            nz = grid["nz"][ys, xs].astype(np.float64)
            fm = np.isfinite(nx) & np.isfinite(ny) & np.isfinite(nz)
            cen = pts.mean(0); cam = -cen / (np.linalg.norm(cen) + 1e-12)
            if int(fm.sum()) > 10:
                ndotc = np.abs(nx[fm] * cam[0] + ny[fm] * cam[1] + nz[fm] * cam[2])
                if float(ndotc.mean()) >= 0.80:   # 디스크가 카메라 정면 = 세움
                    orientation = "upright"
    if orientation == "upright":
        up = _fit_upright_bottle(pts, known_radius_mm, length_mm)
        if up is not None:
            P0u, axu, radu, aminu, amaxu = up
            cen = P0u + 0.5 * (aminu + amaxu) * axu
            info.update({
                "fit_ok": True, "P0": P0u.tolist(), "axis_dir": axu.tolist(),
                "radius_mm": radu, "radius_src": "upright_disc",
                "center": cen.tolist(), "axial_min": aminu, "axial_max": amaxu,
                "n_valid": int(len(pts)), "inlier_ratio": 1.0,
                "orientation": "upright"})
            return info
        # 디스크 fit 실패 시 일반 경로로 폴백

    # init: centroid + PCA long axis
    P0 = pts.mean(axis=0)
    cc = pts - P0
    cov = cc.T @ cc / max(len(pts) - 1, 1)
    _, evec = np.linalg.eigh(cov)
    axis_dir = evec[:, -1]
    axis_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-12)

    have_2d = (polygon_2d is not None and fx and fy and cx and cy
               and len(np.asarray(polygon_2d)) >= 3)
    if known_radius_mm is not None and have_2d and _HAS_LSQ:
        # ★ 2D 실루엣 장축으로 in-plane 방향 고정 + 깊이 tilt만 fit. 자유 3D fit
        # (LSQ/PCA)은 투명·bimodal 깊이 노이즈로 tilt를 과대평가(검증: 352#1
        # |axisZ| 0.63→과함). 2D는 신뢰 → 이게 가장 강건(352#1 inlier 45%→90%).
        radius = float(known_radius_mm)
        radius_src = "2d_locked"
        P0, axis_dir = _fit_axis_2d_locked(pts, polygon_2d, fx, fy, cx, cy, radius)
    elif known_radius_mm is not None and _HAS_LSQ:
        # 폴리곤/intrinsic 없을 때 — 반경 고정 자유 LSQ (앞·뒤 벽 R 제약).
        radius = float(known_radius_mm)
        radius_src = "known_lsq"
        P0, axis_dir, _ = _fit_cylinder_lsq(
            pts, radius, P0, axis_dir, trim_mm=radius_tolerance_mm + 4.0)
    else:
        # known_radius 없거나 scipy 부재 → PCA inlier 정제 + median 반경
        for _ in range(5):
            c = pts - P0
            proj = c @ axis_dir
            perp = c - np.outer(proj, axis_dir)
            rad = np.linalg.norm(perp, axis=1)
            r_ref = (float(known_radius_mm) if known_radius_mm is not None
                     else float(np.median(rad)))
            inl = np.abs(rad - r_ref) < radius_tolerance_mm
            if int(inl.sum()) < 20:
                break
            Pi = pts[inl]
            P0 = Pi.mean(axis=0)
            ci = Pi - P0
            covi = ci.T @ ci / max(len(Pi) - 1, 1)
            _, eveci = np.linalg.eigh(covi)
            na = eveci[:, -1]
            axis_dir = (-na if float(na @ axis_dir) < 0 else na)
            axis_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-12)
        c = pts - P0
        proj = c @ axis_dir
        perp = c - np.outer(proj, axis_dir)
        rad = np.linalg.norm(perp, axis=1)
        radius = (float(known_radius_mm) if known_radius_mm is not None
                  else float(np.median(rad)))
        radius_src = "known_prior" if known_radius_mm is not None else "free_median"

    if not (radius_min_mm <= radius <= radius_max_mm):
        info.update({"radius_mm": radius, "radius_src": radius_src})
        return info

    # 최종 통계 (proj, perp을 최종 axis로 재계산)
    c = pts - P0
    proj = c @ axis_dir
    perp = c - np.outer(proj, axis_dir)
    rad = np.linalg.norm(perp, axis=1)
    inlier_ratio = float(np.mean(np.abs(rad - radius) < radius_tolerance_mm))
    amin = float(np.percentile(proj, 2))
    amax = float(np.percentile(proj, 98))
    center = P0 + 0.5 * (amin + amax) * axis_dir
    info.update({
        "fit_ok": True, "P0": P0.tolist(), "axis_dir": axis_dir.tolist(),
        "radius_mm": radius, "radius_src": radius_src,
        "center": center.tolist(), "axial_min": amin, "axial_max": amax,
        "n_valid": int(len(pts)), "inlier_ratio": inlier_ratio,
        "orientation": "lying",
    })
    return info


def detect_cap_end(
    grid: np.ndarray,
    mask: np.ndarray,
    fit: dict,
    cap_len_mm: float = 15.0,
    n_bins: int = 8,
) -> dict:
    """fit된 원기둥의 두 끝 중 어느 쪽이 뚜껑인지 식별.

    핵심 단서 — RGB 밝기 (2026-05-29 실데이터 5/5 검증):
      뚜껑 = 금속 은회색(불투명) → 밝기 105~142.
      몸통 = 투명 → 흰 배경/내용물 비쳐 밝음 105~142보다 밝은 141~174.
      → **두 끝 중 더 어두운(낮은 brightness) 끝 = 뚜껑.** 채도(낮을수록 cap)도 보조.
    기하 단서(법선 축정렬, see-through depth spread)는 참고용으로만 같이 리턴 —
    뚜껑·바닥 둘 다 불투명/곡면이라 기하만으론 거의 동점이었음.

    Returns dict {ok, cap_end('min'/'max'), cap_axial_lo, cap_axial_hi(=axial 컷),
                  val_min, val_max, sat_min, sat_max, margin,
                  ndota_min, ndota_max, spread_min, spread_max, spread_mid}.
    """
    out = {"ok": False}
    if not fit.get("fit_ok"):
        return out
    axis = np.asarray(fit["axis_dir"], dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    center = np.asarray(fit["center"], dtype=np.float64)

    z = grid["z"].astype(np.float64)
    valid = ~np.isnan(z)
    vm = mask & valid
    ys, xs = np.where(vm)
    if len(ys) < 50:
        return out
    P = np.stack([grid["x"][ys, xs], grid["y"][ys, xs], z[ys, xs]],
                 axis=1).astype(np.float64)
    N = np.stack([grid["nx"][ys, xs], grid["ny"][ys, xs], grid["nz"][ys, xs]],
                 axis=1).astype(np.float64)
    rr = grid["r"][ys, xs].astype(np.float64)
    gg = grid["g"][ys, xs].astype(np.float64)
    bb = grid["b"][ys, xs].astype(np.float64)
    cmax = np.maximum.reduce([rr, gg, bb])
    cmin = np.minimum.reduce([rr, gg, bb])
    sat_px = np.where(cmax > 0, (cmax - cmin) / cmax, 0.0)
    val_px = cmax  # 밝기 0..255
    t = (P - center) @ axis
    tmin, tmax = float(t.min()), float(t.max())
    if tmax - tmin < 2 * cap_len_mm:
        return out

    span = tmax - tmin
    tn = (t - tmin) / (span + 1e-9)   # 0(min끝)..1(max끝)

    # ── 몸통 기준 밝기 = 축 중앙 50% 의 median (투명 몸통은 밝음) ──────
    midm = (tn > 0.25) & (tn < 0.75)
    body_ref = float(np.median(val_px[midm])) if int(midm.sum()) >= 20 \
        else float(np.median(val_px))

    # ── 뚜껑 픽셀 = 몸통보다 DARK_MARGIN 이상 어두운 픽셀 (금속 회색 캡) ──
    DARK_MARGIN = 25.0
    dark = val_px < (body_ref - DARK_MARGIN)
    n_dark = int(dark.sum())
    dark_frac = n_dark / float(len(val_px))

    # 뚜껑 후보가 한쪽 끝에 뭉쳤나 — 어두운 픽셀의 축위치 분포로 판정.
    # 뚜껑 길이는 추정하지 않고 cap_len_mm(실측 prior) 고정 → 검출된 뚜껑
    # 위치(어두운 픽셀 중심)에 그 길이를 중앙정렬해 배치(2026-05-29 사용자 지시).
    # 병 끝(robust 극단). 캡은 어느 쪽 끝이든 이 극단에 cap_len로 배치.
    t_lo = float(np.percentile(t, 2))
    t_hi = float(np.percentile(t, 98))
    if n_dark >= 20:
        dtn = tn[dark]
        med = float(np.median(dtn))
        cap_end = "min" if med < 0.5 else "max"
        conc = float((dtn < 0.25).mean()) if cap_end == "min" \
            else float((dtn > 0.75).mean())
        # 캡은 병의 끝 → 어두운픽셀 median(클러터에 안쪽으로 끌림) 대신 검출된
        # 끝의 극단에 cap_len 배치 (2026-05-29 사용자: 클러터서 캡이 안쪽으로 내려옴).
        if cap_end == "min":
            cap_lo, cap_hi = t_lo, t_lo + cap_len_mm
        else:
            cap_lo, cap_hi = t_hi - cap_len_mm, t_hi
        cap_center = 0.5 * (cap_lo + cap_hi)
    else:
        cap_end, conc, med = "min", 0.0, 0.5
        cap_center = t_lo + 0.5 * cap_len_mm
        cap_lo, cap_hi = t_lo, t_lo + cap_len_mm

    # ── 가림/미가시 판정 ─────────────────────────────────────────
    # 진짜 뚜껑: 어두운 픽셀이 충분(≥15%) + 한쪽 끝에 강하게 뭉침(≥0.4).
    # 둘 다 아니면 뚜껑이 가려졌거나 시야에 없음 → occluded=True.
    occluded = (dark_frac < 0.15) or (conc < 0.40)
    confidence = float(min(dark_frac / 0.23, 1.0) * min(conc / 0.75, 1.0))

    out.update({
        "ok": True, "cap_end": cap_end,
        "occluded": bool(occluded), "confidence": confidence,
        "dark_frac": dark_frac, "concentration": conc, "body_ref": body_ref,
        "cap_center": cap_center, "cap_len_mm": cap_len_mm,
        "cap_axial_lo": cap_lo, "cap_axial_hi": cap_hi,
    })
    return out


def refine_cylinder_with_cap(grid, mask, fit, cap, known_radius_mm=None,
                             dark_margin=25.0):
    """뚜껑(노이즈 적음) 법선으로 축 방향을 잡고 위치는 몸통에 맞춰 fit dict을 갱신.

    뚜껑이 신뢰될 때만 동작. 갱신: fit["axis_dir","P0","center","axial_min/max",
    "axis_src"]. 반환 verify dict {applied, cap_axis_agree_deg, body_inlier,
    cap_surf_resid_mm} — 뚜껑↔원기둥 일치도(사용자 검증 아이디어).
    """
    verify = {"applied": False}
    if not cap.get("ok") or cap.get("occluded", False):
        verify["reason"] = "no_cap"
        return verify
    radius = float(fit["radius_mm"])
    axis0 = np.asarray(fit["axis_dir"], np.float64)
    axis0 = axis0 / (np.linalg.norm(axis0) + 1e-12)
    center0 = np.asarray(fit["center"], np.float64)

    z = grid["z"].astype(np.float64)
    vm = mask & ~np.isnan(z)
    ys, xs = np.where(vm)
    if len(ys) < 50:
        verify["reason"] = "few_pts"
        return verify
    P = np.stack([grid["x"][ys, xs], grid["y"][ys, xs], z[ys, xs]], 1).astype(np.float64)
    val = np.maximum.reduce([grid["r"][ys, xs], grid["g"][ys, xs],
                             grid["b"][ys, xs]]).astype(np.float64)
    t0 = (P - center0) @ axis0
    capm = ((val < cap["body_ref"] - dark_margin)
            & (t0 >= cap["cap_axial_lo"]) & (t0 <= cap["cap_axial_hi"]))
    if int(capm.sum()) < 20:
        verify["reason"] = "few_cap_px"
        return verify

    # 뚜껑 평면 법선 (최소분산 방향)
    Pc = P[capm]
    cc = Pc - Pc.mean(0)
    _, evec = np.linalg.eigh(cc.T @ cc / len(cc))
    n_cap = evec[:, 0]
    n_cap = n_cap / (np.linalg.norm(n_cap) + 1e-12)
    if float(n_cap @ axis0) < 0:
        n_cap = -n_cap
    agree = float(np.degrees(np.arccos(np.clip(abs(n_cap @ axis0), 0, 1))))

    # 방향=n_cap 고정, 위치(축선)도 뚜껑으로 — 뚜껑 점은 반경 R 원(rim) 위에
    # 있으므로 뚜껑 평면에 투영해 반경R 원을 fit하면 그 중심 = 진짜 축점.
    # (뚜껑 centroid는 보이는 면쪽으로 치우쳐 틀림 → 원fit으로 bias 보정.)
    # 노이즈 없는 뚜껑만으로 축을 완결 → 투명·bimodal 몸통 노이즈에 의존 안 함.
    body = ~capm
    Pb = P[body]
    a = np.array([1.0, 0, 0]) if abs(n_cap[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(n_cap, a); u /= np.linalg.norm(u) + 1e-12
    w = np.cross(n_cap, u)
    cpu = Pc @ u; cpw = Pc @ w
    if _HAS_LSQ:
        sol = _least_squares(
            lambda c: np.hypot(cpu - c[0], cpw - c[1]) - radius,
            [float(cpu.mean()), float(cpw.mean())], method="lm", max_nfev=1000)
        cu, cw = sol.x
    else:
        cu, cw = float(cpu.mean()), float(cpw.mean())
    P0new = cu * u + cw * w + float((Pc @ n_cap).mean()) * n_cap

    # 검증: 몸통이 이 뚜껑축 원기둥에 얼마나 맞나 (낮아도 몸통 노이즈 탓일 수 있음)
    v = Pb - P0new
    perp = np.linalg.norm(v - np.outer(v @ n_cap, n_cap), axis=1)
    body_inlier = float(np.mean(np.abs(perp - radius) < 8.0))
    vc = Pc - P0new
    cap_perp = np.linalg.norm(vc - np.outer(vc @ n_cap, n_cap), axis=1)
    cap_surf_resid = float(np.median(np.abs(cap_perp - radius)))

    proj_all = (P - P0new) @ n_cap
    amin = float(np.percentile(proj_all, 2))
    amax = float(np.percentile(proj_all, 98))
    center_new = P0new + 0.5 * (amin + amax) * n_cap

    fit["axis_dir"] = n_cap.tolist()
    fit["P0"] = P0new.tolist()
    fit["center"] = center_new.tolist()
    fit["axial_min"] = amin
    fit["axial_max"] = amax
    fit["axis_src"] = "cap_normal"

    # cap dict도 새 축/center 프레임으로 갱신 (뚜껑 길이 15mm 중앙정렬 유지)
    cap_t_new = (Pc - center_new) @ n_cap
    cap_center_new = float(np.median(cap_t_new))
    cap_len = float(cap.get("cap_len_mm", 15.0))
    t_all_new = (P - center_new) @ n_cap
    cap["cap_end"] = "min" if cap_center_new < 0 else "max"
    cap["cap_center"] = cap_center_new
    cap["cap_axial_lo"] = max(float(t_all_new.min()), cap_center_new - 0.5 * cap_len)
    cap["cap_axial_hi"] = min(float(t_all_new.max()), cap_center_new + 0.5 * cap_len)

    verify.update({"applied": True, "cap_axis_agree_deg": agree,
                   "body_inlier": body_inlier, "cap_surf_resid_mm": cap_surf_resid})
    return verify


def verify_cap_consistency(grid, mask, fit, cap, dark_margin=25.0):
    """축을 바꾸지 않고 뚜껑↔원기둥 일치도만 측정 (사용자 검증 아이디어).

    노이즈 없는 뚜껑 점이 fit된 원기둥 표면(perp≈R)에 얼마나 맞는지 + 뚜껑
    평면 법선이 축과 얼마나 정렬되는지. 안 맞으면 신뢰 낮음 플래그.
    Returns {applied, cap_axis_agree_deg, cap_surf_resid_mm}.
    """
    v = {"applied": False}
    if not cap.get("ok") or cap.get("occluded", False):
        v["reason"] = "no_cap"
        return v
    axis = np.asarray(fit["axis_dir"], np.float64); axis /= np.linalg.norm(axis) + 1e-12
    center = np.asarray(fit["center"], np.float64); R = float(fit["radius_mm"])
    z = grid["z"].astype(np.float64); vm = mask & ~np.isnan(z)
    ys, xs = np.where(vm)
    if len(ys) < 50:
        v["reason"] = "few_pts"; return v
    P = np.stack([grid["x"][ys, xs], grid["y"][ys, xs], z[ys, xs]], 1).astype(np.float64)
    val = np.maximum.reduce([grid["r"][ys, xs], grid["g"][ys, xs], grid["b"][ys, xs]]).astype(np.float64)
    t = (P - center) @ axis
    capm = ((val < cap["body_ref"] - dark_margin)
            & (t >= cap["cap_axial_lo"]) & (t <= cap["cap_axial_hi"]))
    if int(capm.sum()) < 20:
        v["reason"] = "few_cap_px"; return v
    Pc = P[capm]; cc = Pc - Pc.mean(0)
    _, evec = np.linalg.eigh(cc.T @ cc / len(cc)); n_cap = evec[:, 0]
    agree = float(np.degrees(np.arccos(np.clip(abs(n_cap @ axis), 0, 1))))
    vc = Pc - center
    cap_perp = np.linalg.norm(vc - np.outer(vc @ axis, axis), axis=1)
    v.update({"applied": True, "cap_axis_agree_deg": agree,
              "cap_surf_resid_mm": float(np.median(np.abs(cap_perp - R)))})
    return v


def compute_bottle_pick_pose(
    grid: np.ndarray,
    mask: np.ndarray,
    known_radius_mm: float | None = None,
    cap_clear_mm: float = 30.0,
    cap_len_mm: float = 15.0,
    polygon_2d=None,
    fx: float | None = None, fy: float | None = None,
    cx: float | None = None, cy: float | None = None,
    length_mm: float = 145.0,
    upright_aspect_max: float = 1.7,
) -> dict:
    """재구성된 원기둥 + 뚜껑 방향으로 병 흡착 픽 자세 산출.

    누운 병: 2D 실루엣 장축 고정+깊이 tilt fit, 측면 흡착(몸통 중앙 표면).
    세워진 병: 캡 디스크가 카메라 향함 → 축=디스크 법선, **탑다운 흡착(캡 윗면)**.

    Returns dict {ok, contact, approach, axis_dir, center, radius_mm, cap_end,
                  orientation, body_axial_lo, body_axial_hi, fit, cap, verify}.
    """
    out = {"ok": False}
    fit = fit_bottle_cylinder(grid, mask, known_radius_mm=known_radius_mm,
                              polygon_2d=polygon_2d, fx=fx, fy=fy, cx=cx, cy=cy,
                              length_mm=length_mm,
                              upright_aspect_max=upright_aspect_max)
    if not fit.get("fit_ok"):
        out["fit"] = fit
        return out

    axis = np.asarray(fit["axis_dir"], dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    center = np.asarray(fit["center"], dtype=np.float64)
    radius = float(fit["radius_mm"])

    # ── 세워진 병: 캡 디스크가 카메라 향함 → 탑다운 흡착 ──
    if fit.get("orientation") == "upright":
        cap_center = np.asarray(fit["P0"], np.float64)  # 디스크 중심 = 캡 top
        approach = axis                                 # 캡 바깥 법선(카메라 향함, axis=-Z)
        # cap_axial은 center 기준(viewer가 off=0.5(amin+amax)로 P0-frame 변환).
        # 캡(디스크)은 P0(=proj 0) 위치 → center 기준 -off.
        off = 0.5 * (fit["axial_min"] + fit["axial_max"])
        out.update({
            "ok": True, "orientation": "upright",
            "contact": cap_center.tolist(), "approach": approach.tolist(),
            "axis_dir": axis.tolist(), "center": center.tolist(),
            "radius_mm": radius, "cap_end": "max",
            "fit": fit, "cap": {"ok": True, "occluded": False, "cap_end": "max",
                                "cap_axial_lo": -off - cap_len_mm,
                                "cap_axial_hi": -off},
            "verify": {"applied": False, "reason": "upright"},
        })
        return out

    cap = detect_cap_end(grid, mask, fit, cap_len_mm=cap_len_mm)
    out["verify"] = verify_cap_consistency(grid, mask, fit, cap)
    out["orientation"] = "lying"

    # 점들을 center 기준 축좌표로 → 전체 범위
    z = grid["z"].astype(np.float64)
    valid = ~np.isnan(z)
    vm = mask & valid
    ys, xs = np.where(vm)
    P = np.stack([grid["x"][ys, xs], grid["y"][ys, xs], z[ys, xs]],
                 axis=1).astype(np.float64)
    t = (P - center) @ axis
    tmin, tmax = float(t.min()), float(t.max())

    # 몸통 구간 = 전체에서 뚜껑쪽 cap_clear_mm 제외.
    # 단 뚜껑이 가려/불확실(occluded)하면 어느 끝이 뚜껑인지 못 믿으므로 제외 안 함.
    body_lo, body_hi = tmin, tmax
    cap_trusted = cap.get("ok") and not cap.get("occluded", False)
    if cap_trusted:
        if cap["cap_end"] == "min":
            body_lo = tmin + cap_clear_mm
        else:
            body_hi = tmax - cap_clear_mm
    if body_hi - body_lo < 5.0:
        body_lo, body_hi = tmin, tmax   # 너무 짧으면 전체 사용
    a_pick = 0.5 * (body_lo + body_hi)

    # 축 위 픽 위치 Q, 카메라(원점) 쪽 방사 방향
    Q = center + a_pick * axis
    to_cam = -Q                       # Q→카메라(원점)
    radial = to_cam - float(to_cam @ axis) * axis
    rn = float(np.linalg.norm(radial))
    if rn < 1e-6:
        out["fit"] = fit; out["cap"] = cap
        return out
    radial /= rn
    contact = Q + radius * radial     # 카메라-쪽 표면점
    approach = radial                 # 표면 바깥 법선(카메라 향함). 흡착은 -approach로 진입.

    out.update({
        "ok": True, "contact": contact.tolist(), "approach": approach.tolist(),
        "axis_dir": axis.tolist(), "center": center.tolist(),
        "radius_mm": radius, "cap_end": cap.get("cap_end"),
        "body_axial_lo": body_lo, "body_axial_hi": body_hi,
        "fit": fit, "cap": cap,
    })
    return out


def recover_xyz_in_mask_cylinder(
    grid: np.ndarray,
    mask: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    radius_min_mm: float = 5.0,
    radius_max_mm: float = 60.0,
    min_valid_for_fit: int = 50,
) -> tuple[np.ndarray, int, dict]:
    """원기둥 fit + 광선 교차로 polygon 안 NaN xyz 채움 (bottle 전용).

    Returns:
        (new_grid, n_filled, info)
          info = {'radius_mm', 'axis_dir', 'P0', 'fit_ok'} — 디버깅용
        n_filled = 0 이면 복원 실패 (fit 실패 / 광선 교차 0건)
    """
    z = grid["z"].astype(np.float64)
    x = grid["x"].astype(np.float64)
    y = grid["y"].astype(np.float64)
    valid = ~np.isnan(z)
    valid_in_mask = mask & valid
    nan_in_mask = mask & ~valid

    info = {"fit_ok": False, "radius_mm": None, "n_valid": int(valid_in_mask.sum())}
    if nan_in_mask.sum() == 0 or valid_in_mask.sum() < min_valid_for_fit:
        return grid, 0, info

    pts = np.stack([x[valid_in_mask], y[valid_in_mask], z[valid_in_mask]], axis=1)
    P0, axis_dir, radius = _fit_cylinder_pca(pts)
    info.update({"radius_mm": float(radius),
                 "axis_dir": axis_dir.tolist(),
                 "P0": P0.tolist()})
    if not (radius_min_mm <= radius <= radius_max_mm):
        # 원기둥 형상 아닌 듯 — 복원 포기 (호출자가 fallback 처리)
        return grid, 0, info
    info["fit_ok"] = True

    # 각 NaN 픽셀에 대해 광선 발사 + 교차 계산
    H, W = z.shape
    new_z = z.copy()
    new_x = x.copy()
    new_y = y.copy()
    n_filled = 0
    for v_pix, u_pix in np.argwhere(nan_in_mask):
        rx = (float(u_pix) - cx) / fx
        ry = (float(v_pix) - cy) / fy
        ray = np.array([rx, ry, 1.0])
        ray = ray / np.linalg.norm(ray)
        t = _ray_cylinder_intersect(ray, P0, axis_dir, radius)
        if t is None:
            continue
        new_x[v_pix, u_pix] = t * ray[0]
        new_y[v_pix, u_pix] = t * ray[1]
        new_z[v_pix, u_pix] = t * ray[2]
        n_filled += 1

    if n_filled == 0:
        return grid, 0, info

    new_grid = grid.copy()
    new_grid["x"] = new_x.astype(grid["x"].dtype)
    new_grid["y"] = new_y.astype(grid["y"].dtype)
    new_grid["z"] = new_z.astype(grid["z"].dtype)
    return new_grid, n_filled, info


# ============================================================================
# Method C' — 뚜껑(cap) 앵커 원기둥 extrusion (bottle 전용, body-fit 대체)
# ============================================================================

def recover_xyz_in_mask_cylinder_from_cap(
    grid: np.ndarray,
    mask: np.ndarray,
    polygon_2d,
    fx: float, fy: float, cx: float, cy: float,
    radius_min_mm: float = 5.0,
    radius_max_mm: float = 60.0,
    min_valid_for_fit: int = 50,
    known_radius_mm: float | None = None,
) -> tuple[np.ndarray, int, dict]:
    """뚜껑 평면에서 축·중심·반경을 뽑아 원기둥을 extrusion → NaN 측면 복원.

    사용자 의도 (2026-05-29): 물병 뚜껑은 불투명 plastic → 깊이 신뢰. 위에서
    보면 원, 기울어지면 타원. 뚜껑에서 (a) 평면 normal = 원기둥 축(=extrusion
    직선 방향), (b) 평면 중심 = 축 위 한 점, (c) ellipse minor axis = 반경을
    얻고, 그 원을 축 방향으로 늘린 원기둥 표면에 광선을 쏴 투명 측면을 채운다.
    뚜껑 지름 = 본체 지름 가정 (현재 물병 형상에서 성립).

    body-point RANSAC fit(recover_xyz_in_mask_cylinder, Method C)과 차이:
      - noise 원천인 투명 측면 점에 의존 안 함 → 견고
      - 가시 표면 bias(2r/π) 보정 같은 fragile 휴리스틱 불필요
      - 파지 자세(_compute_bottle_pose)의 cap-based 축과 동일 prior로 일관

    실패(cap 미검출 / 반경 범위 밖 / 광선 교차 0건) 시 n_filled=0 반환 →
    호출자가 기존 method로 fallback.

    Returns:
        (new_grid, n_filled, info)
          info = {'fit_ok', 'radius_mm', 'axis_dir', 'P0', 'method', 'reason'}
    """
    info = {"fit_ok": False, "radius_mm": None, "method": "cap",
            "n_valid": 0, "reason": None}

    # 순환 import 회피: picking이 depth_recovery를 import 하므로 lazy import
    try:
        from .picking import _detect_cap_plane
    except Exception as e:  # pragma: no cover
        info["reason"] = f"import:{e}"
        return grid, 0, info

    z = grid["z"].astype(np.float64)
    x = grid["x"].astype(np.float64)
    y = grid["y"].astype(np.float64)
    valid = ~np.isnan(z)
    valid_in_mask = mask & valid
    nan_in_mask = mask & ~valid
    info["n_valid"] = int(valid_in_mask.sum())

    if nan_in_mask.sum() == 0 or valid_in_mask.sum() < min_valid_for_fit:
        info["reason"] = "insufficient_valid"
        return grid, 0, info

    cap = _detect_cap_plane(mask, valid, grid, polygon_2d)
    if cap is None:
        info["reason"] = "no_cap"
        return grid, 0, info

    P0 = np.asarray(cap["centroid"], dtype=np.float64)
    axis_dir = np.asarray(cap["axis_dir"], dtype=np.float64)
    axis_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-12)

    # ── P0(축 위 한 점) 보정 ──────────────────────────────────────
    # cap["centroid"]는 long-axis 25% end-slice inlier 평균이라 valid 점이
    # 적거나 cap이 그 영역 대부분일 때 진짜 disc 중심(=축)에서 벗어남. P0가
    # 축에서 벗어나면 반경이 맞아도 원기둥이 통째로 어긋남. → 검출된 cap
    # 평면과 coplanar 한 valid 점 전체의 평균(보이는 disc 중심)으로 재설정.
    pts_all = np.stack([x[valid_in_mask], y[valid_in_mask],
                        z[valid_in_mask]], axis=1)
    plane_tol = max(3.0 * float(cap.get("plane_thickness_mm", 1.0)), 5.0)
    coplanar = np.abs((pts_all - P0) @ axis_dir) < plane_tol
    n_cop = int(coplanar.sum())
    cap_pts = pts_all[coplanar] if n_cop >= 20 else None
    if cap_pts is not None:
        P0 = cap_pts.mean(axis=0)   # 보이는 뚜껑 disc 중심 = 축 위 점

    # ── 반경 결정 ────────────────────────────────────────────────
    # known_radius_mm(실측 prior)가 최우선·최신뢰. 없으면 coplanar disc 점들의
    # 축까지 수직거리 p90 (cap["radius_mm"]은 25% slice minor라 과소추정 경향).
    if known_radius_mm is not None:
        radius = float(known_radius_mm)
        radius_src = "known_prior"
    elif cap_pts is not None:
        cc = cap_pts - P0
        perp = cc - np.outer(cc @ axis_dir, axis_dir)
        radius = float(np.percentile(np.linalg.norm(perp, axis=1), 90))
        radius_src = "cap_coplanar"
    else:
        radius = float(cap["radius_mm"])
        radius_src = "cap_slice"

    info.update({"radius_mm": radius, "radius_src": radius_src,
                 "axis_dir": axis_dir.tolist(), "P0": P0.tolist(),
                 "n_coplanar": n_cop})
    if not (radius_min_mm <= radius <= radius_max_mm):
        info["reason"] = f"radius_oor:{radius:.1f}"
        return grid, 0, info
    info["fit_ok"] = True

    # 각 NaN 픽셀에 광선 발사 + 무한 원기둥(축=cap normal, 반경=cap radius) 교차
    new_z = z.copy()
    new_x = x.copy()
    new_y = y.copy()
    n_filled = 0
    for v_pix, u_pix in np.argwhere(nan_in_mask):
        rx = (float(u_pix) - cx) / fx
        ry = (float(v_pix) - cy) / fy
        ray = np.array([rx, ry, 1.0])
        ray = ray / np.linalg.norm(ray)
        t = _ray_cylinder_intersect(ray, P0, axis_dir, radius)
        if t is None:
            continue
        new_x[v_pix, u_pix] = t * ray[0]
        new_y[v_pix, u_pix] = t * ray[1]
        new_z[v_pix, u_pix] = t * ray[2]
        n_filled += 1

    if n_filled == 0:
        info["reason"] = "no_intersect"
        return grid, 0, info

    new_grid = grid.copy()
    new_grid["x"] = new_x.astype(grid["x"].dtype)
    new_grid["y"] = new_y.astype(grid["y"].dtype)
    new_grid["z"] = new_z.astype(grid["z"].dtype)
    return new_grid, n_filled, info
