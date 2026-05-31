"""3D viewer with rotation gizmo — Open3D new GUI (open3d.visualization.gui).

Run as a separate subprocess from label_tool.py to avoid Tkinter conflicts:
    py scripts/view_3d.py <path/to/shot.ply> [--polygons <data.json>]

Highlights:
- Pivot point sits at the labeled object center (via P key).
- 3 colored rotation rings drawn at the pivot:
    X = red   (rotates around world X)
    Y = green (rotates around world Y)
    Z = blue  (rotates around world Z)
  Click and drag a ring → camera orbits around that axis through the pivot.
- Default left-drag (empty space): free rotate around the pivot.

Keys:
    P       pivot to labeled object closest to screen center
    R       reset pivot to scene center
    T       toggle label highlights ON/OFF
    M       toggle depth-recovery meshes (translucent surface) ON/OFF
    Q/ESC   close
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


IMG_W, IMG_H = 1224, 1024
VERTEX_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("r", "<u1"), ("g", "<u1"), ("b", "<u1"),
    ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
])


def _hex_to_rgb01(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


CLASS_COLORS = [
    _hex_to_rgb01("#3498DB"),   # 0 bottle      - blue
    _hex_to_rgb01("#E74C3C"),   # 1 haribo      - red
    _hex_to_rgb01("#F39C12"),   # 2 mango       - orange
    _hex_to_rgb01("#9B59B6"),   # 3 metal_case  - purple
    _hex_to_rgb01("#7F8C8D"),   # 4 object      - gray (OOD/기타)
    _hex_to_rgb01("#27AE60"),   # 5 pencil_case - green
]
CLASS_NAMES = ["bottle", "haribo", "mango", "metal_case", "object", "pencil_case"]

# 흡착컵 OD (외경) + arm box 형상 — config에서 로드.
# 컵 외경 ring + arm 스윕 박스 시각화 ('C' 키 토글) 에 사용.
try:
    import yaml as _yaml
    from pathlib import Path as _Path
    _cfg = _yaml.safe_load(
        open(_Path(__file__).resolve().parents[1] / "config" / "default.yaml",
             encoding="utf-8"))
    CUP_OD_MM = float(_cfg["suction"]["cup_od_mm"])
    CUP_PITCH_MM = float(_cfg["suction"]["cup_pitch_mm"])
    HEAD_CLEARANCE_MM = float(_cfg["suction"]["head_clearance_mm"])
    ARM_BOX_SCALE = float(_cfg["suction"]["arm_box_scale"])
    ARM_BOX_HEIGHT_MM = float(_cfg["suction"]["arm_box_height_mm"])
    S2_ARM_PROTRUSION_MM = float(_cfg["gates"]["s2_arm_protrusion_mm"])
    _bkr = _cfg.get("bottle", {}).get("known_radius_mm", None)
    BOTTLE_KNOWN_RADIUS_MM = float(_bkr) if _bkr is not None else None
    BOTTLE_CAP_LEN_MM = float(_cfg.get("bottle", {}).get("cap_len_mm", 15.0))
    BOTTLE_LENGTH_MM = float(_cfg.get("bottle", {}).get("length_mm", 145.0))
    BOTTLE_UPRIGHT_ASPECT_MAX = float(_cfg.get("bottle", {}).get("upright_aspect_max", 1.7))
except Exception:
    CUP_OD_MM = 26.0
    CUP_PITCH_MM = 50.0
    HEAD_CLEARANCE_MM = 30.0
    ARM_BOX_SCALE = 1.3
    ARM_BOX_HEIGHT_MM = 150.0
    S2_ARM_PROTRUSION_MM = 10.0
    BOTTLE_KNOWN_RADIUS_MM = None
    BOTTLE_CAP_LEN_MM = 15.0
    BOTTLE_LENGTH_MM = 145.0
    BOTTLE_UPRIGHT_ASPECT_MAX = 1.7

# 픽 사각형 = pitch + cup_od (long), cup_od (short). picking.py와 동일 정의.
RECT_LONG_MM = CUP_PITCH_MM + CUP_OD_MM
RECT_SHORT_MM = CUP_OD_MM
ARM_HALF_LONG_MM = (RECT_LONG_MM / 2.0) * ARM_BOX_SCALE
ARM_HALF_SHORT_MM = (RECT_SHORT_MM / 2.0) * ARM_BOX_SCALE


def _make_cup_ring(center, normal, radius_mm, n_seg=48, rgb=(1.0, 0.95, 0.10)):
    """Normal에 수직 평면의 원형 LineSet. 흡착컵 외경 시각화."""
    n = np.asarray(normal, dtype=np.float64)
    n = n / max(float(np.linalg.norm(n)), 1e-12)
    # n에 수직인 임의 perpendicular vector
    if abs(n[0]) < 0.9:
        u = np.array([1.0, 0.0, 0.0])
    else:
        u = np.array([0.0, 1.0, 0.0])
    u = u - np.dot(u, n) * n
    u = u / max(float(np.linalg.norm(u)), 1e-12)
    v = np.cross(n, u)
    c = np.asarray(center, dtype=np.float64)
    pts = np.array(
        [c + radius_mm * (np.cos(2 * np.pi * i / n_seg) * u
                          + np.sin(2 * np.pi * i / n_seg) * v)
         for i in range(n_seg)],
        dtype=np.float64,
    )
    lines = np.array(
        [[i, (i + 1) % n_seg] for i in range(n_seg)], dtype=np.int32,
    )
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(np.tile(np.array(rgb), (n_seg, 1)))
    return ls


def _make_arm_box_wireframe(position, normal, long_axis, mid_axis,
                              half_long_mm, half_short_mm,
                              t_min_mm, t_max_mm, rgb=(1.0, 0.3, 0.3)):
    """팔이 픽 자세대로 진입할 때 휩쓰는 박스를 12 edge LineSet으로.

    박스 (pick frame):
        x (long_axis): [-half_long_mm, +half_long_mm]
        y (mid_axis):  [-half_short_mm, +half_short_mm]
        z (normal):    [t_min_mm, t_max_mm]

    picking.py의 approach_box_check와 동일한 영역 — 박스 안 다른 객체 점이
    있으면 충돌. 사용자가 3D에서 직접 보고 확인 가능.
    """
    p = np.asarray(position, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64)
    lx = np.asarray(long_axis, dtype=np.float64)
    my = np.asarray(mid_axis, dtype=np.float64)
    # 8 corners: (sx*half_long, sy*half_short, t) 조합
    corners = []
    for t in (t_min_mm, t_max_mm):
        for sx in (-1, +1):
            for sy in (-1, +1):
                c = (p + sx * half_long_mm * lx
                       + sy * half_short_mm * my
                       + t * n)
                corners.append(c)
    pts = np.array(corners, dtype=np.float64)
    # 인덱스: 0..3 = bottom face (t_min), 4..7 = top face (t_max).
    # bottom: (-,-)=0 (-,+)=1 (+,-)=2 (+,+)=3 — sx 먼저, sy 안.
    # 위 루프 순서: t→sx→sy 이므로 [0:(-,-),1:(-,+),2:(+,-),3:(+,+)]
    edges = np.array([
        # bottom face
        [0, 1], [1, 3], [3, 2], [2, 0],
        # top face
        [4, 5], [5, 7], [7, 6], [6, 4],
        # vertical edges (sweep direction)
        [0, 4], [1, 5], [2, 6], [3, 7],
    ], dtype=np.int32)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(edges)
    ls.colors = o3d.utility.Vector3dVector(
        np.tile(np.array(rgb, dtype=np.float64), (edges.shape[0], 1))
    )
    return ls


def read_organized_ply(path: Path):
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError("PLY header missing end_header")
            if line.strip() == b"end_header":
                break
        n = IMG_W * IMG_H
        buf = f.read(n * VERTEX_DTYPE.itemsize)
    return np.frombuffer(buf, dtype=VERTEX_DTYPE, count=n).reshape(IMG_H, IMG_W)


def grid_to_pointcloud(grid):
    xyz = np.stack([grid["x"], grid["y"], grid["z"]], -1)
    valid = ~np.isnan(xyz[..., 2])
    pts = xyz[valid].astype(np.float32)
    rgb = np.stack([grid["r"], grid["g"], grid["b"]], -1)[valid].astype(np.float32) / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd, valid


def _score_to_rgb01(scores):
    """[0,1] scores → (N, 3) RGB in [0,1]. R(0) → Y(0.5) → G(1). NaN → black."""
    s = np.clip(np.where(np.isnan(scores), 0.0, scores), 0.0, 1.0).astype(np.float32)
    r = np.where(s < 0.5, 1.0, (1.0 - s) * 2.0)
    g = np.where(s >= 0.5, 1.0, s * 2.0)
    b = np.zeros_like(r)
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def polygons_to_highlights(grid, valid, polygons_data: dict, blend: float = 0.75,
                           compute_heatmap: bool = False):
    """Convert image-space polygons to 3D highlight point clouds.

    Returns (highlights, centroids, cids, class_colors_list, heatmap_colors_list,
             masks_list, avg_normals, polygons_2d_list, original_indices).

    original_indices[k] = highlight k에 대응하는 result_data 원본 index.
    polygons_to_highlights가 invalid 폴리곤을 skip하면서 인덱스가 어긋나면,
    manual_pick GT가 다른 객체에 매핑되는 버그가 있음 → original_indices로 매핑.
    """
    from PIL import Image, ImageDraw
    highlights, centroids, cids_drawn = [], [], []
    class_colors_list, heatmap_colors_list = [], []
    masks_list, avg_normals, polygons_2d_list = [], [], []
    original_indices = []   # highlight 순서 → result_data 원본 i

    pickability_heatmap = None
    heatmap_cup_radius_px = 25.0   # default, picking config로 갱신
    if compute_heatmap:
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from label_logic.suction_score import pickability_heatmap
            from label_logic.picking import HEATMAP_CUP_RADIUS_PX as _hcrp
            heatmap_cup_radius_px = float(_hcrp)
            print("  ✓ pickability_heatmap imported — K key will toggle colors")
        except Exception as e:
            print(f"  ✗ pickability_heatmap import failed: {e}")
            print(f"    → K key will NOT work. Check scipy/numpy in this env.")
            pickability_heatmap = None

    result_data = polygons_data.get("result_data", [])
    class_ids = polygons_data.get("class_id", [])
    xyz = np.stack([grid["x"], grid["y"], grid["z"]], -1)
    rgb = np.stack([grid["r"], grid["g"], grid["b"]], -1).astype(np.float32) / 255.0

    n_total = len(result_data)
    n_skip = 0
    n_drawn = 0
    per_class = [0] * len(CLASS_NAMES)

    for i, entry in enumerate(result_data):
        if not isinstance(entry, list) or not entry:
            n_skip += 1
            continue
        first = entry[0]
        if not isinstance(first, list) or not first or not isinstance(first[0], list):
            n_skip += 1
            continue
        cid = class_ids[i] if i < len(class_ids) else -1
        if not (0 <= cid < len(CLASS_COLORS)):
            n_skip += 1
            continue
        pts2d = [(float(p[0]), float(p[1])) for p in first]
        if len(pts2d) < 3:
            n_skip += 1
            continue
        img = Image.new("L", (IMG_W, IMG_H), 0)
        ImageDraw.Draw(img).polygon(pts2d, fill=1)
        poly_mask = np.array(img, dtype=bool)
        mask = poly_mask & valid
        if not mask.any():
            n_skip += 1
            continue
        pts3d = xyz[mask].copy()
        pts3d[:, 2] -= 1.5   # offset toward camera to avoid z-fighting
        orig_rgb = rgb[mask]
        class_rgb = np.array(CLASS_COLORS[cid], dtype=np.float32)
        blended = np.clip((1 - blend) * orig_rgb + blend * class_rgb, 0, 1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts3d)
        pcd.colors = o3d.utility.Vector3dVector(blended)
        highlights.append(pcd)
        class_colors_list.append(blended.astype(np.float32))

        # Heatmap colors (parallel to pts3d)
        heat_rgb = None
        if pickability_heatmap is not None:
            try:
                hm = pickability_heatmap(grid, valid, poly_mask,
                                           cup_radius_px=heatmap_cup_radius_px)
                # hm is (H, W) with NaN outside polygon. mask & valid pts → look up.
                scores_at_pts = hm[mask]
                heat_rgb = _score_to_rgb01(scores_at_pts)
            except Exception as e:
                print(f"  (heatmap compute failed for poly {i}: {e})")
                heat_rgb = None
        heatmap_colors_list.append(heat_rgb)

        bbox_min = pts3d.min(axis=0)
        bbox_max = pts3d.max(axis=0)
        centroids.append(((bbox_min + bbox_max) * 0.5).astype(np.float64))
        cids_drawn.append(cid)
        n_drawn += 1
        per_class[cid] += 1

        # 추가 저장 (manual pick 라벨링 / 검증용)
        masks_list.append(poly_mask)
        polygons_2d_list.append(np.asarray(pts2d, dtype=np.float32))
        original_indices.append(i)   # result_data 원본 인덱스 보존
        # Polygon 평균 normal — Zivid PLY에서 직접
        nx_p = grid["nx"][mask]
        ny_p = grid["ny"][mask]
        nz_p = grid["nz"][mask]
        nf = np.isfinite(nx_p) & np.isfinite(ny_p) & np.isfinite(nz_p)
        if nf.any():
            n_avg = np.array([nx_p[nf].mean(), ny_p[nf].mean(),
                              nz_p[nf].mean()], dtype=np.float64)
            mag = float(np.linalg.norm(n_avg))
            if mag > 1e-6:
                n_avg = n_avg / mag
            else:
                n_avg = np.array([0.0, 0.0, -1.0])
            # 카메라 -Z 쪽으로
            if n_avg[2] > 0:
                n_avg = -n_avg
        else:
            n_avg = np.array([0.0, 0.0, -1.0])
        avg_normals.append(n_avg)

    print(f"\n=== Highlight summary ===")
    print(f"  Total polygons              : {n_total}")
    print(f"  Drawn (highlighted)         : {n_drawn}")
    if n_skip:
        print(f"  Skipped (invalid/no-3D/etc) : {n_skip}")
    if n_drawn:
        per_class_str = ", ".join(
            f"{name}={per_class[i]}" for i, name in enumerate(CLASS_NAMES)
            if per_class[i] > 0
        )
        print(f"  Per class                   : {per_class_str}")
    return (highlights, centroids, cids_drawn, class_colors_list,
            heatmap_colors_list, masks_list, avg_normals, polygons_2d_list,
            original_indices)


def _get_screen_size_windows():
    try:
        import ctypes
        u = ctypes.windll.user32
        return u.GetSystemMetrics(0), u.GetSystemMetrics(1)
    except Exception:
        return 1920, 1080


# ─────────────────────────────────────────────────────────────────────────────
# Viewer
# ─────────────────────────────────────────────────────────────────────────────


def _rodrigues(axis, angle):
    """Rodrigues rotation matrix (3x3) for `angle` rad around unit `axis`."""
    a = np.asarray(axis, dtype=np.float64)
    a = a / (np.linalg.norm(a) + 1e-12)
    c, s = np.cos(angle), np.sin(angle)
    ux, uy, uz = a
    K = np.array([[0, -uz, uy], [uz, 0, -ux], [-uy, ux, 0]], dtype=np.float64)
    return np.eye(3) * c + s * K + (1 - c) * np.outer(a, a)


class Viewer3D:
    AXIS_VECTORS = np.eye(3, dtype=np.float64)            # X, Y, Z
    AXIS_COLORS = [(0.95, 0.20, 0.20),                    # X red
                   (0.30, 0.85, 0.30),                    # Y green
                   (0.30, 0.55, 1.00)]                    # Z blue
    AXIS_NAMES = ["X", "Y", "Z"]

    def __init__(self, ply_path: Path, polygons_data=None, blend=0.75,
                 dim_bg=False, picks_data=None, heatmap_mode=False,
                 win_w=1200, win_h=800, win_left=200, win_top=60):
        self.ply_path = ply_path
        self.picks_data = picks_data
        self.heatmap_mode = bool(heatmap_mode)
        self.heatmap_active = False    # 시작 시 class color (T로 toggle)

        # ── Load geometry ──
        grid = read_organized_ply(ply_path)
        pcd, valid = grid_to_pointcloud(grid)
        print(f"  {len(pcd.points):,} valid points")

        self.highlights, self.centroids, self.cids = [], [], []
        self.class_colors_list, self.heatmap_colors_list = [], []
        self.masks_list, self.avg_normals, self.polygons_2d_list = [], [], []
        self.original_indices = []   # highlight k → result_data 원본 i
        if polygons_data is not None:
            (self.highlights, self.centroids, self.cids,
             self.class_colors_list, self.heatmap_colors_list,
             self.masks_list, self.avg_normals, self.polygons_2d_list,
             self.original_indices) = (
                polygons_to_highlights(
                    grid, valid, polygons_data, blend=blend,
                    compute_heatmap=self.heatmap_mode,
                )
            )
        # 검증 / 수동 라벨링용 grid + valid_mask 보관
        self.grid_data = grid
        self.valid_2d = valid

        # Bottle cap inlier 픽셀 — RANSAC이 잡은 진짜 뚜껑 점들. 별도 PointCloud로
        # 큰 point size + 짙은 파랑으로 그려서 base highlight 위로 덮음. 사용자
        # 시각 검증: 진짜 뚜껑 위에 색이 칠해지는지 / 측면을 잘못 잡았는지.
        self.cap_inlier_pcds = []   # 별도 트래킹 (T 토글 시 같이 hide)
        if self.picks_data is not None and self.original_indices:
            DEEP_BLUE = np.array([0.05, 0.10, 0.85], dtype=np.float32)
            idx_to_hl = {oi: k for k, oi in enumerate(self.original_indices)}
            for p in self.picks_data.get("picks", []):
                bf = p.get("bottle_fit") or {}
                cap_yx = bf.get("cap_inlier_yx")
                if not cap_yx or len(cap_yx) != 2: continue
                poly_idx = p.get("poly_index")
                if poly_idx is None: continue
                k = idx_to_hl.get(int(poly_idx))
                if k is None: continue
                cap_ys = np.asarray(cap_yx[0], dtype=np.int64)
                cap_xs = np.asarray(cap_yx[1], dtype=np.int64)
                H, W = grid["z"].shape
                in_bounds = ((cap_ys >= 0) & (cap_ys < H)
                              & (cap_xs >= 0) & (cap_xs < W))
                cap_ys = cap_ys[in_bounds]; cap_xs = cap_xs[in_bounds]
                if len(cap_ys) == 0: continue
                zs = grid["z"][cap_ys, cap_xs].astype(np.float64)
                xs = grid["x"][cap_ys, cap_xs].astype(np.float64)
                ys = grid["y"][cap_ys, cap_xs].astype(np.float64)
                fin = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs)
                if not fin.any(): continue
                pts3d_cap = np.stack([xs[fin], ys[fin], zs[fin]], axis=-1)
                # 카메라 쪽 1.5mm offset — base highlight와 동일 (z-fighting 회피)
                pts3d_cap[:, 2] -= 2.5   # base보다 1mm 더 앞 (확실히 보이게)
                cap_pcd = o3d.geometry.PointCloud()
                cap_pcd.points = o3d.utility.Vector3dVector(pts3d_cap)
                cap_pcd.colors = o3d.utility.Vector3dVector(
                    np.tile(DEEP_BLUE, (len(pts3d_cap), 1)).astype(np.float64))
                self.cap_inlier_pcds.append(cap_pcd)
        # data.json 경로 (manual_picks 저장 시 같은 디렉토리)
        self.polygons_data = polygons_data

        # ── Camera intrinsics 해석 (샷별 _info.json 우선, yaml fallback) ───
        # MR60 / MR130 등 다른 카메라 샷에서 recovery 메시·드래그 mm 변환이
        # 어긋나지 않도록 init 시 1회 resolve해서 self에 저장.
        (self.fx, self.fy, self.cx, self.cy,
         self._bottle_cid_resolved, _intr_src) = self._resolve_intrinsics(ply_path)
        if self.fx is not None:
            print(f"  intrinsics: fx={self.fx:.1f} fy={self.fy:.1f} "
                  f"cx={self.cx:.1f} cy={self.cy:.1f} ({_intr_src})")
        if dim_bg and self.highlights:
            colors = np.asarray(pcd.colors) * 0.35
            pcd.colors = o3d.utility.Vector3dVector(colors)
        self.pcd = pcd

        # ── Scene metrics ──
        pts = np.asarray(pcd.points)
        if len(pts) > 0:
            self.scene_center = pts.mean(axis=0).astype(np.float64)
            self.scene_extent = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        else:
            self.scene_center = np.array([0.0, 0.0, 800.0], dtype=np.float64)
            self.scene_extent = 500.0
        self.pivot = self.scene_center.copy()

        # Gizmo ring sizing — visible at default zoom but not too chunky
        self.gizmo_radius = float(max(20.0, min(80.0, self.scene_extent * 0.07)))
        self.gizmo_tube = float(self.gizmo_radius * 0.022)
        self.gizmo_alpha = 0.55

        # ── GUI ──
        self.app = gui.Application.instance
        self.app.initialize()
        self.window = self.app.create_window(
            f"3D View — {ply_path.stem}", win_w, win_h, win_left, win_top,
        )

        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        # Mid blue-gray background (Blender-like) — easier on eyes than pure black
        # and keeps coloured highlights readable.
        self.scene_widget.scene.set_background([0.55, 0.58, 0.62, 1.0])

        # Materials
        self.mat_pc = rendering.MaterialRecord()
        self.mat_pc.shader = "defaultUnlit"
        self.mat_pc.point_size = 2.0

        self.mat_unlit = rendering.MaterialRecord()
        self.mat_unlit.shader = "defaultUnlit"

        # Add geometries
        self.scene_widget.scene.add_geometry("pointcloud", pcd, self.mat_pc)
        for i, h in enumerate(self.highlights):
            self.scene_widget.scene.add_geometry(f"hi_{i}", h, self.mat_pc)
        # Bottle cap inlier — base highlight보다 더 큰 point로 그려서 위로 띄움.
        mat_cap = rendering.MaterialRecord()
        mat_cap.shader = "defaultUnlit"
        mat_cap.point_size = 6.0
        for ci, cap_pcd in enumerate(self.cap_inlier_pcds):
            self.scene_widget.scene.add_geometry(f"cap_inlier_{ci}", cap_pcd, mat_cap)
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50.0)
        coord.compute_vertex_normals()
        self.scene_widget.scene.add_geometry("coord", coord, self.mat_unlit)

        # ── Depth recovery meshes (translucent surface for filled NaN regions)
        # bottle → RANSAC cylinder fit + ray intersection (Method C)
        # 다른 클래스 → scipy linear interp (Method A)
        self.recovery_mesh_names: list[str] = []
        self.recovery_visible = True
        self._build_recovery_meshes()

        # ── Gizmo (3 rings) + pivot marker — added at origin, moved via transform ──
        self._add_gizmo_geometries()
        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=self.gizmo_tube * 1.6, resolution=10,
        )
        sphere.paint_uniform_color([1.0, 1.0, 0.0])
        sphere.compute_vertex_normals()
        self.scene_widget.scene.add_geometry("pivot_marker", sphere, self.mat_unlit)

        # ── Picking rectangles (if --picks given) ──
        # 자동 표시 안 함 — G 키로 토글. picks_data는 로드만 해두고 geom은 보류.
        self.pick_geom_names = []
        # 3D 라벨 (Open3D Label3D 객체) — show_geometry 안 먹어서 따로 추적.
        # G OFF 시 remove_3d_label로 제거, ON 시 다시 만들어 추가.
        self.pick_labels = []
        # cap_inlier_* point cloud — _init_scene 에서 추가됨. G 토글에 같이 묶여야
        # 하므로 별도 이름 리스트로 추적 (위에서 cap_inlier_pcds 생성 직후 채워짐).
        self.cap_inlier_geom_names = []
        # Collision viz: pick idx → [arm_box geom name, object_sweep geom name].
        # 'C' 키 누르면 화면 중앙에 가장 가까운 픽만 표시 (P 키와 같은 로직).
        # 같은 픽이 이미 보이는 상태면 hide (toggle off).
        self.pick_box_geom_names = {}
        # pick idx → 3D 위치 (mm). 화면 proximity 계산용.
        self.pick_box_centers = {}
        self.collision_viz_pick_idx = None   # 현재 표시 중인 픽 idx (None=숨김)
        # pick idx → [(highlight_k, np.array of point idx)] — sweep 부피 안
        # 들어오는 다른 객체 점들. C 키로 검정 칠해서 충돌 시각화.
        self.pick_collision_targets = {}
        # 검정 칠하기 전 원본 색 백업: [(highlight_k, idx_array, colors_copy)].
        # 복구 시 그대로 되돌림 (heatmap 토글 등과 무관).
        self.collision_color_backup = None
        # G 키 토글 상태 — 초기 OFF (자동 표시 안 함).
        self.picks_visible = False
        self.picks_added = False   # _add_pick_geometries가 한 번이라도 호출됐는지
        # cap_inlier 는 _init_scene 에서 add_geometry 됐는데 초기엔 안 보여야 하므로
        # 여기서 이름을 채우고 hide. (실제 add_geometry 는 위 _init_scene 에서 이미 수행)
        for ci in range(len(self.cap_inlier_pcds)):
            name = f"cap_inlier_{ci}"
            self.cap_inlier_geom_names.append(name)
            self.scene_widget.scene.show_geometry(name, False)

        # ── Camera ──
        bbox = self.scene_widget.scene.bounding_box
        self.scene_widget.setup_camera(60.0, bbox, self.scene_center)
        # Match legacy view orientation: front=[0,-0.3,-1], up=[0,-1,0]
        # In Open3D, "front" points from the scene back toward the camera, so
        # eye = lookat + front * distance (NOT minus).
        cam_dist = self.scene_extent * 0.9
        self._look_dir_default = np.array([0.0, -0.3, -1.0], dtype=np.float64)
        self._up_default = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        eye = self.scene_center + self._look_dir_default * cam_dist
        self.scene_widget.scene.camera.look_at(
            self.scene_center.astype(np.float32),
            eye.astype(np.float32),
            self._up_default.astype(np.float32),
        )
        self.scene_widget.center_of_rotation = self.pivot.astype(np.float32)
        self._refresh_gizmo_pose()

        # ── Drag state ──
        self.active_axis = None
        self.drag_initial_hit = None
        self.drag_initial_eye = None
        self.drag_initial_up = None
        self.drag_initial_view = None
        self.drag_initial_proj = None
        self.drag_view_w = 0
        self.drag_view_h = 0
        self.highlights_visible = True

        # Manual pick labeling state ─────────────────────────
        self.label_mode = False
        # manual_pick: dict per polygon_idx — {position, normal, long_axis, mid_axis,
        #                                       polygon_idx, validation}
        self.manual_pick = None
        self.manual_geom_names = []   # 3D geometry 이름 (cleanup용)
        self.manual_picks_save_path = self._derive_manual_picks_path()
        self._load_manual_picks_from_disk()

        # Pick gizmo (회전 링 + 이동 화살표)
        self.pick_gizmo_ring_radius = 50.0    # mm — 카메라 gizmo보다 크게
        self.pick_gizmo_ring_tube = 2.5       # mm — 두껍게
        self.pick_gizmo_arrow_len = 70.0      # mm — 링 바깥까지
        self.pick_gizmo_arrow_radius = 3.0
        self.pick_gizmo_hit_tol = 12.0        # 링 hit 허용 오차 (mm) — 후하게
        self.pick_axis_colors = [
            (1.0, 0.25, 0.25),   # X = long_axis = red
            (0.30, 0.95, 0.30),  # Y = mid_axis  = green
            (0.30, 0.55, 1.00),  # Z = normal    = blue
        ]
        # Active handle: ('ring', axis_idx) or ('arrow', axis_idx)
        self.manual_active_handle = None
        self.manual_drag_init_pose = None    # pose snapshot at drag start
        self.manual_drag_init_hit = None     # 3D hit point at drag start
        # Modifier-key drag (Shift = rotate, Ctrl = translate XY, Ctrl+Shift = translate Z)
        self.manual_mod_drag_kind = None     # "rotate" / "translate_xy" / "translate_z"
        self.manual_mod_drag_last_xy = None  # (mx, my) of previous drag step
        # 저장 안전장치: 첫 Enter는 검증만, 두 번째 Enter로 실제 저장
        self.manual_pending_save = False

        # Layout / callbacks
        self.window.add_child(self.scene_widget)
        self.window.set_on_layout(self._on_layout)
        self.scene_widget.set_on_mouse(self._on_mouse)
        self.scene_widget.set_on_key(self._on_key)

        self._print_help()

    # ── Layout ──
    def _on_layout(self, _ctx):
        self.scene_widget.frame = self.window.content_rect

    # ── Help ──
    def _print_help(self):
        print("\n=== Controls ===")
        print("  Drag a colored ring     : rotate around X(red)/Y(green)/Z(blue)")
        print("                            axis through pivot")
        print("  Left-drag (empty space) : free rotate around pivot")
        print("  Right-drag / Mid-drag   : pan")
        print("  Wheel                   : zoom")
        print("  P                       : pivot to labeled object nearest")
        print("                            to screen center")
        print("  R                       : reset pivot to scene center")
        print("  T                       : toggle label highlights ON/OFF")
        print("  K                       : toggle highlight color mode")
        print("                            (class colors ↔ pickability heatmap)")
        print("  M                       : toggle recovery meshes ON/OFF")
        print("  G                       : toggle picks ON/OFF (initial=OFF)")
        print("  C                       : show collision box for pick nearest")
        print("                            to screen center (toggle off if same pick)")
        print("  L                       : enter manual pick label mode")
        print("                            ─ first click on polygon: place pick")
        print("                            ─ drag R/G/B rings: rotate around X/Y/Z axes")
        print("                            ─ drag R/G/B arrows: translate along X/Y/Z")
        print("                            ─ W/S: move along Z (normal: away/into surface)")
        print("                            ─ A/D: move along X (long_axis: left/right)")
        print("                            ─ Q/E: move along Y (mid_axis)")
        print("                            ─ R: reset pick (re-place on next click)")
        print("                            ─ V: validate, Enter: save, ESC: cancel")
        print("                            ─ B: save current pose as bottle cap")
        print("                                 (data/<team>/manual_cap/)")
        print("  Q / ESC                 : close")

    # ── Picks (dual-suction rectangles + cup centers + Z axis) ──
    @staticmethod
    def _rotation_align(from_vec, to_vec):
        """Rotation matrix that maps `from_vec` onto `to_vec` (unit vectors)."""
        a = np.asarray(from_vec, dtype=np.float64)
        a = a / max(np.linalg.norm(a), 1e-12)
        b = np.asarray(to_vec, dtype=np.float64)
        b = b / max(np.linalg.norm(b), 1e-12)
        v = np.cross(a, b)
        s = float(np.linalg.norm(v))
        c = float(np.dot(a, b))
        if s < 1e-9:
            return np.eye(3) if c > 0 else -np.eye(3)
        K = np.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]], dtype=np.float64)
        return np.eye(3) + K + (K @ K) * ((1.0 - c) / (s * s))

    # Tier color map (3D, 0..1 RGB)
    TIER_COLORS_3D = {
        "HIGH":       (0.18, 0.80, 0.44),   # green
        "MEDIUM":     (0.95, 0.77, 0.06),   # yellow
        "LOW":        (0.90, 0.49, 0.13),   # orange
        "UNPICKABLE": (0.50, 0.50, 0.50),   # gray
        "MANUAL":     (1.00, 0.84, 0.00),   # gold yellow — GT 명시
    }

    def _add_pick_geometries(self):
        """Render each pick — HIGH/MED/LOW filled rectangle + cups + Z-arrow.
        UNPICKABLE objects rendered as gray dashed marker."""
        picks = self.picks_data.get("picks", [])
        Z_AXIS_LEN = 40.0   # mm
        # poly_index → highlight idx 역매핑 (object bbox 산출용).
        # masks_list[k]는 highlight k의 2D 마스크, original_indices[k]는
        # 원본 polygon index. picks[i]["poly_index"]로 lookup.
        poly_to_hl = {int(oi): k for k, oi in enumerate(self.original_indices)}
        for idx, p in enumerate(picks):
            if not p.get("pickable"):
                continue   # UNPICKABLE handled in 2D / next pass
            tier = p.get("tier", "LOW")
            tier_rgb = self.TIER_COLORS_3D.get(tier, (0.5, 0.5, 0.5))
            corners = np.asarray(p["rect_corners_3d"], dtype=np.float64)
            if corners.shape != (4, 3):
                continue

            # Filled rectangle = 2 triangles, translucent BLACK with tier outline
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(corners)
            mesh.triangles = o3d.utility.Vector3iVector(
                np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
            )
            mesh.compute_vertex_normals()
            mat_fill = rendering.MaterialRecord()
            mat_fill.shader = "defaultUnlitTransparency"
            mat_fill.base_color = [0.05, 0.05, 0.05, 0.55]
            mat_fill.has_alpha = True
            name_fill = f"pick_fill_{idx}"
            self.scene_widget.scene.add_geometry(name_fill, mesh, mat_fill)
            self.pick_geom_names.append(name_fill)

            # Outline — tier-colored line set
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(corners)
            ls.lines = o3d.utility.Vector2iVector(
                np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32)
            )
            ls.colors = o3d.utility.Vector3dVector(
                np.tile(np.array(tier_rgb), (4, 1))
            )
            mat_line = rendering.MaterialRecord()
            mat_line.shader = "unlitLine"
            mat_line.line_width = 5.0
            name_line = f"pick_line_{idx}"
            self.scene_widget.scene.add_geometry(name_line, ls, mat_line)
            self.pick_geom_names.append(name_line)

            # Cup centers — 중심 sphere + 외경 ring (흡착 접촉 영역 시각화).
            # ring 반지름 = SUCTION_CUP_OD_MM/2. normal에 수직 평면 위 원.
            # single 모드: cup_a만 흡입(active=노랑), cup_b는 dead cup(회색)로
            # 표시 — 그리퍼 물리적 위치는 듀얼과 같지만 한쪽만 흡입 켜진 상태.
            cup_radius_marker = 4.0
            cup_od_radius_mm = CUP_OD_MM / 2.0
            normal_vec = np.asarray(p["normal"], dtype=np.float64)
            is_single = (p.get("pick_mode") == "single")
            ACTIVE_RGB = [1.0, 0.95, 0.10]   # yellow
            DEAD_RGB = [0.45, 0.45, 0.45]    # grey
            for cup_key in ("cup_a_3d", "cup_b_3d"):
                cup_pos = p.get(cup_key)
                if cup_pos is None:
                    continue
                is_dead = is_single and cup_key == "cup_b_3d"
                cup_rgb = DEAD_RGB if is_dead else ACTIVE_RGB
                # 중심 점 (작은 sphere)
                s = o3d.geometry.TriangleMesh.create_sphere(
                    radius=cup_radius_marker, resolution=10,
                )
                s.translate(np.asarray(cup_pos), relative=False)
                s.paint_uniform_color(cup_rgb)
                s.compute_vertex_normals()
                mat_s = rendering.MaterialRecord()
                mat_s.shader = "defaultUnlit"
                name_s = f"pick_cup_{idx}_{cup_key}"
                self.scene_widget.scene.add_geometry(name_s, s, mat_s)
                self.pick_geom_names.append(name_s)
                # 외경 ring (normal 수직)
                ring = _make_cup_ring(
                    np.asarray(cup_pos), normal_vec, cup_od_radius_mm,
                    rgb=tuple(cup_rgb),
                )
                mat_ring = rendering.MaterialRecord()
                mat_ring.shader = "unlitLine"
                mat_ring.line_width = 3.0
                name_ring = f"pick_cup_ring_{idx}_{cup_key}"
                self.scene_widget.scene.add_geometry(name_ring, ring, mat_ring)
                self.pick_geom_names.append(name_ring)

            # ── Arm sweep box (C 키 토글) ─────────────────────────────
            # 픽 자세대로 진입 시 팔이 휩쓰는 부피. 다른 객체 점이 이 박스
            # 안에 보이면 실제 충돌. arm_max_protrusion_mm으로 색상 등급화:
            # 0mm = green, threshold = red.
            # single 모드는 position == cup_a(객체 centroid)라 박스를 그대로
            # 두면 한쪽으로 쏠림. 그리퍼 tool 중점(= cup_a와 dead cup_b의 중점
            # = position - PITCH/2·long_axis)으로 시프트.
            position = np.asarray(p["position_mm"], dtype=np.float64)
            long_vec = np.asarray(p["long_axis"], dtype=np.float64)
            mid_vec = np.asarray(p["mid_axis"], dtype=np.float64)
            # 축은 unit이어야 하지만 안전하게 정규화 (안 그러면 박스 스케일 깨짐).
            long_vec = long_vec / max(float(np.linalg.norm(long_vec)), 1e-12)
            mid_vec = mid_vec / max(float(np.linalg.norm(mid_vec)), 1e-12)
            normal_vec = normal_vec / max(float(np.linalg.norm(normal_vec)), 1e-12)
            if p.get("pick_mode") == "single":
                # cup_b는 picking 모듈이 동적으로 ±long_axis 중 안전한 쪽으로
                # 결정 → arm box는 cup_a(=position)와 cup_b 중점에 둠.
                cup_b_for_arm = p.get("cup_b_3d")
                if cup_b_for_arm is not None:
                    arm_center = 0.5 * (position
                                         + np.asarray(cup_b_for_arm, dtype=np.float64))
                else:
                    arm_center = position - (CUP_PITCH_MM / 2.0) * long_vec
            else:
                arm_center = position
            prot = float(p.get("arm_max_protrusion_mm", 0.0))
            ratio = max(0.0, min(1.0,
                prot / max(S2_ARM_PROTRUSION_MM, 1e-6)
            ))
            box_rgb = (0.2 + 0.8 * ratio,            # R
                       0.8 * (1.0 - ratio) + 0.2,    # G
                       0.2)                           # B
            arm_box = _make_arm_box_wireframe(
                arm_center, normal_vec, long_vec, mid_vec,
                half_long_mm=ARM_HALF_LONG_MM,
                half_short_mm=ARM_HALF_SHORT_MM,
                t_min_mm=HEAD_CLEARANCE_MM,
                t_max_mm=ARM_BOX_HEIGHT_MM,
                rgb=box_rgb,
            )
            mat_arm = rendering.MaterialRecord()
            mat_arm.shader = "unlitLine"
            mat_arm.line_width = 2.5
            name_arm = f"pick_armbox_{idx}"
            self.scene_widget.scene.add_geometry(name_arm, arm_box, mat_arm)
            self.scene_widget.scene.show_geometry(name_arm, False)
            self.pick_box_geom_names.setdefault(idx, []).append(name_arm)
            self.pick_box_centers[idx] = arm_center.copy()

            # ── Object lift sweep (seg-based prism) ─────────────────────
            # bbox 대신 polygon(seg) 자체를 normal 방향으로 압출한 prism.
            # 화면에 보이는 초록 영역 그대로의 모양으로 위/아래 두 layer를
            # 각각 카메라 ray로 normal 수직 plane에 투영해 그림.
            # 충돌 검사: 다른 점을 카메라로 투영(u,v) → mask 안인지 + normal
            # 방향 깊이가 [z_bot, z_top] 안인지.
            poly_idx = p.get("poly_index")
            hl_idx = poly_to_hl.get(int(poly_idx)) if poly_idx is not None else None
            obj_prism = None   # (mask2d, z_bot, z_top) for 충돌 체크
            if (hl_idx is not None and hl_idx < len(self.masks_list)
                    and hl_idx < len(self.polygons_2d_list)
                    and self.fx is not None):
                mask2d = self.masks_list[hl_idx]
                poly2d = self.polygons_2d_list[hl_idx]   # (N,2) pixel coords
                use = mask2d & self.valid_2d
                if int(use.sum()) >= 30 and len(poly2d) >= 3:
                    # 객체 body 3D 점을 normal 수직 평면(x_lat, y_lat)으로
                    # 투영 → convex hull. "픽 방향에서 본 객체 윤곽".
                    # 카메라 silhouette 대신 pick 방향 silhouette을 써서 lift
                    # 방향과 박스 모양이 일치.
                    xs = self.grid_data["x"][use]
                    ys = self.grid_data["y"][use]
                    zs = self.grid_data["z"][use]
                    pts = np.stack([xs, ys, zs], axis=-1).astype(np.float64)
                    rel = pts - position[None, :]
                    pz = rel @ normal_vec
                    # body cluster 필터 (mask 경계 floor 픽셀 제거)
                    pz_med = float(np.median(pz))
                    OBJECT_DEPTH_WINDOW = 60.0
                    near = np.abs(pz - pz_med) < OBJECT_DEPTH_WINDOW
                    if int(near.sum()) < 20:
                        near = np.ones_like(pz, dtype=bool)
                    # Normal 수직 plane 축 — world X/Y를 normal 수직으로 투영
                    x_world = np.array([1.0, 0.0, 0.0])
                    x_lat = x_world - np.dot(x_world, normal_vec) * normal_vec
                    if float(np.linalg.norm(x_lat)) < 0.1:
                        x_world = np.array([0.0, 1.0, 0.0])
                        x_lat = x_world - np.dot(x_world, normal_vec) * normal_vec
                    x_lat = x_lat / max(float(np.linalg.norm(x_lat)), 1e-12)
                    y_lat = np.cross(normal_vec, x_lat)
                    y_lat = y_lat / max(float(np.linalg.norm(y_lat)), 1e-12)
                    px_b = rel[near] @ x_lat
                    py_b = rel[near] @ y_lat
                    pts_2d = np.stack([px_b, py_b], axis=-1)
                    # Convex hull (pick 방향 silhouette)
                    hull_2d = None
                    if len(pts_2d) >= 4:
                        try:
                            from scipy.spatial import ConvexHull
                            ch = ConvexHull(pts_2d)
                            hull_2d = pts_2d[ch.vertices]   # (M,2) CCW
                        except Exception:
                            hull_2d = None
                    if hull_2d is None:
                        # fallback: bbox 4 corners
                        x_lo = float(pts_2d[:, 0].min())
                        x_hi = float(pts_2d[:, 0].max())
                        y_lo = float(pts_2d[:, 1].min())
                        y_hi = float(pts_2d[:, 1].max())
                        hull_2d = np.array([[x_lo, y_lo], [x_hi, y_lo],
                                             [x_hi, y_hi], [x_lo, y_hi]])
                    # z 범위: 앞(lift 방향)만. 뒤(객체 body 안)는 제외.
                    z_bot = 0.0
                    z_top = ARM_BOX_HEIGHT_MM
                    # Hull 정점 → 3D base on pick plane
                    base_arr = (position[None, :]
                                 + hull_2d[:, 0:1] * x_lat[None, :]
                                 + hull_2d[:, 1:2] * y_lat[None, :])
                    bot_pts = base_arr + z_bot * normal_vec
                    top_pts = base_arr + z_top * normal_vec
                    if len(bot_pts) >= 3:
                        Nv = len(bot_pts)
                        all_pts = np.vstack([bot_pts, top_pts])
                        edges = []
                        for i in range(Nv):
                            j = (i + 1) % Nv
                            edges.append([i, j])           # bottom
                            edges.append([Nv + i, Nv + j])  # top
                            edges.append([i, Nv + i])      # vertical
                        edges = np.array(edges, dtype=np.int32)
                        prism_ls = o3d.geometry.LineSet()
                        prism_ls.points = o3d.utility.Vector3dVector(all_pts)
                        prism_ls.lines = o3d.utility.Vector2iVector(edges)
                        prism_rgb = (max(0.0, tier_rgb[0] * 0.7),
                                     max(0.0, tier_rgb[1] * 0.7),
                                     max(0.0, tier_rgb[2] * 0.7))
                        prism_ls.colors = o3d.utility.Vector3dVector(
                            np.tile(np.array(prism_rgb), (edges.shape[0], 1))
                        )
                        mat_pr = rendering.MaterialRecord()
                        mat_pr.shader = "unlitLine"
                        mat_pr.line_width = 2.0
                        name_pr = f"pick_objprism_{idx}"
                        self.scene_widget.scene.add_geometry(name_pr, prism_ls, mat_pr)
                        self.scene_widget.scene.show_geometry(name_pr, False)
                        self.pick_box_geom_names.setdefault(idx, []).append(name_pr)
                        # 충돌 검사용: hull 2D + z 범위 + lat 축
                        obj_prism = (hull_2d, x_lat, y_lat, z_bot, z_top)

            # ── 충돌 점 인덱스 사전 계산 ─────────────────────────────────
            # arm box: PCA frame (long/mid/normal) 안 점.
            # object prism: 점을 카메라로 (u,v) 투영 → mask2d 안 + normal
            # 방향 깊이 [z_bot, z_top] 안.
            targets = []
            H_m, W_m = (self.masks_list[0].shape if self.masks_list else (IMG_H, IMG_W))
            for k, hl_pcd in enumerate(self.highlights):
                if k == hl_idx:
                    continue
                pts_o = np.asarray(hl_pcd.points, dtype=np.float64)
                if pts_o.size == 0:
                    continue
                rel_o = pts_o - position[None, :]
                # Arm frame (PCA)
                a_x = rel_o @ long_vec
                a_y = rel_o @ mid_vec
                a_z = rel_o @ normal_vec
                in_arm = ((np.abs(a_x) < ARM_HALF_LONG_MM)
                          & (np.abs(a_y) < ARM_HALF_SHORT_MM)
                          & (a_z > HEAD_CLEARANCE_MM)
                          & (a_z < ARM_BOX_HEIGHT_MM))
                hit = in_arm
                if obj_prism is not None:
                    hull_2d_p, x_lat_p, y_lat_p, z_bot_p, z_top_p = obj_prism
                    # 점을 (x_lat, y_lat, normal) 축으로 투영 → hull 안 + z 범위
                    o_x = rel_o @ x_lat_p
                    o_y = rel_o @ y_lat_p
                    # convex hull point-in-polygon: CCW hull → 모든 edge에 대해
                    # 2D cross product (a→b) × (a→p) >= 0이면 좌측 = inside.
                    inside_hull = np.ones(len(pts_o), dtype=bool)
                    Mh = len(hull_2d_p)
                    for hi in range(Mh):
                        a = hull_2d_p[hi]
                        b = hull_2d_p[(hi + 1) % Mh]
                        ex, ey = float(b[0] - a[0]), float(b[1] - a[1])
                        cross = ex * (o_y - a[1]) - ey * (o_x - a[0])
                        inside_hull &= (cross >= 0.0)
                    in_obj = inside_hull & (a_z > z_bot_p) & (a_z < z_top_p)
                    hit = in_arm | in_obj
                if hit.any():
                    targets.append((k, np.where(hit)[0]))
            if targets:
                self.pick_collision_targets[idx] = targets

            # Bottle cylinder model 시각화 (bottle_fit dict 있을 때만)
            bf = p.get("bottle_fit")
            if bf and bf.get("cap_centroid") and bf.get("cyl_end"):
                cap_c = np.asarray(bf["cap_centroid"], dtype=np.float64)
                cyl_e = np.asarray(bf["cyl_end"], dtype=np.float64)
                axis_d = np.asarray(bf["axis_dir"], dtype=np.float64)
                r_cyl = float(bf["radius_mm"])
                # Cap disc (뚜껑 평면)
                cap_ring = _make_cup_ring(cap_c, axis_d, r_cyl,
                                            n_seg=64, rgb=(0.0, 0.7, 1.0))
                mat_cap = rendering.MaterialRecord()
                mat_cap.shader = "unlitLine"
                mat_cap.line_width = 4.0
                self.scene_widget.scene.add_geometry(
                    f"bottle_cap_{idx}", cap_ring, mat_cap)
                self.pick_geom_names.append(f"bottle_cap_{idx}")
                # Cylinder body end ring (반대편)
                end_ring = _make_cup_ring(cyl_e, axis_d, r_cyl,
                                            n_seg=64, rgb=(0.2, 0.4, 0.8))
                self.scene_widget.scene.add_geometry(
                    f"bottle_endring_{idx}", end_ring, mat_cap)
                self.pick_geom_names.append(f"bottle_endring_{idx}")
                # Cap → end axis 라인
                axis_line = o3d.geometry.LineSet()
                axis_line.points = o3d.utility.Vector3dVector(
                    np.array([cap_c, cyl_e]))
                axis_line.lines = o3d.utility.Vector2iVector(
                    np.array([[0, 1]], dtype=np.int32))
                axis_line.colors = o3d.utility.Vector3dVector(
                    np.array([[0.0, 0.5, 1.0]]))
                self.scene_widget.scene.add_geometry(
                    f"bottle_axis_{idx}", axis_line, mat_cap)
                self.pick_geom_names.append(f"bottle_axis_{idx}")

            # Z-axis arrow at pick center pointing along normal (approach dir)
            position = np.asarray(p["position_mm"], dtype=np.float64)
            normal = np.asarray(p["normal"], dtype=np.float64)
            normal = normal / max(np.linalg.norm(normal), 1e-12)
            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=1.0,
                cone_radius=2.5,
                cylinder_height=Z_AXIS_LEN * 0.78,
                cone_height=Z_AXIS_LEN * 0.22,
            )
            # create_arrow points along +Z by default; rotate to align with normal
            R = self._rotation_align([0.0, 0.0, 1.0], normal)
            arrow.rotate(R, center=(0, 0, 0))
            arrow.translate(position, relative=False)
            arrow.paint_uniform_color([0.13, 0.59, 0.95])   # blue (Z color)
            arrow.compute_vertex_normals()
            mat_z = rendering.MaterialRecord()
            mat_z.shader = "defaultUnlit"
            name_z = f"pick_zaxis_{idx}"
            self.scene_widget.scene.add_geometry(name_z, arrow, mat_z)
            self.pick_geom_names.append(name_z)

            # Pick order text label floating above the arrow tip
            order = p.get("pick_order")
            conf = p.get("confidence", 0.0)
            tip = position + (Z_AXIS_LEN + 6.0) * normal
            label_text = f"#{order} {tier} {conf:.2f}" if order else f"{tier} {conf:.2f}"
            lbl = self.scene_widget.add_3d_label(tip.tolist(), label_text)
            if lbl is not None:
                self.pick_labels.append(lbl)

        # ── 두번째 패스: 못 잡은 폴리곤의 사유 라벨 ─────────────────────
        # 사용자 요청: "왜 못 잡았는지 3D에 안 보여" → 각 DEFER/IMPOSSIBLE 폴리곤
        # 위치에 짧은 사유 라벨 추가.
        gate_short = {
            "S2_arm_protrusion": "ARM",
            "S2_rect_planarity": "BUMPY",
            "S2_cup_A_coverage": "CUP_A_EDGE",
            "S2_cup_B_coverage": "CUP_B_EDGE",
            "S2_cup_A_bump": "CUP_A_BUMP",
            "S2_cup_B_bump": "CUP_B_BUMP",
            "S2_cup_A_protrusion": "CUP_A_HIT",
            "S2_cup_B_protrusion": "CUP_B_HIT",
            "tilt_limit": "TILT",
            "length": "SHORT",
            "cup_support": "NO_SUPPORT",
            "occlusion_perimeter": "OCCLUDED",
            "S1_overlap": "COVERED",
            "class_excluded": "EXCLUDED",
            "S4_valid_ratio": "NO_DEPTH",
            "box_roi_wall": "WALL",
            "S3_flatness": "NO_FLAT",
        }
        n_unpickable_labels = 0
        for idx, p in enumerate(picks):
            if p.get("pickable"):
                continue
            pos = p.get("position_mm")
            if not pos or all(abs(v) < 1e-6 for v in pos):
                continue   # no 3D position computed (early reject)
            gates = p.get("gate_failures", []) or []
            if not gates:
                continue
            short = gate_short.get(gates[0], gates[0][:8])
            status = p.get("status", "?")
            label_text = f"X {short}" if status == "IMPOSSIBLE" else f"! {short}"
            pos_arr = np.asarray(pos, dtype=np.float64)
            # 라벨 위치는 픽 위치 살짝 위 (toward camera, normal 방향)
            normal = p.get("normal", [0, 0, -1])
            n = np.asarray(normal, dtype=np.float64)
            n = n / max(np.linalg.norm(n), 1e-12)
            label_pos = pos_arr + 15.0 * n
            lbl = self.scene_widget.add_3d_label(label_pos.tolist(), label_text)
            if lbl is not None:
                self.pick_labels.append(lbl)
            n_unpickable_labels += 1

        if self.pick_geom_names:
            n_ok = sum(1 for p in picks if p.get("success"))
            print(f"  loaded {n_ok} pick rectangles "
                  f"({len(self.pick_geom_names)} geoms)")
        if n_unpickable_labels:
            print(f"  showed {n_unpickable_labels} reject reason labels")

    def _rebuild_pick_labels(self):
        """G OFF 시 제거된 Label3D 들을 다시 만들어 추가.

        geom 은 show_geometry 로 토글 가능하지만 label 은 add/remove 만 됨.
        G OFF 가 라벨을 모두 제거하므로 G ON 으로 돌릴 때 라벨만 재생성.
        geom 자체는 _add_pick_geometries 에서 한 번만 만들어진 채 show_geometry
        로만 토글되므로 라벨 부분만 다시 만든다.
        """
        if self.picks_data is None:
            return
        picks = self.picks_data.get("picks", [])
        Z_AXIS_LEN = 40.0   # _add_pick_geometries 와 동일

        # ── PICKABLE: pick_order / tier / conf 라벨 ──────────────────
        for p in picks:
            if not p.get("pickable"):
                continue
            position = np.asarray(p.get("position_mm", [0, 0, 0]), dtype=np.float64)
            normal = np.asarray(p.get("normal", [0, 0, -1]), dtype=np.float64)
            n = normal / max(np.linalg.norm(normal), 1e-12)
            tip = position + (Z_AXIS_LEN + 6.0) * n
            order = p.get("pick_order")
            tier = p.get("tier", "?")
            conf = p.get("confidence", 0.0)
            label_text = f"#{order} {tier} {conf:.2f}" if order else f"{tier} {conf:.2f}"
            lbl = self.scene_widget.add_3d_label(tip.tolist(), label_text)
            if lbl is not None:
                self.pick_labels.append(lbl)

        # ── UNPICKABLE: 실패 사유 라벨 ────────────────────────────────
        gate_short = {
            "S2_arm_protrusion": "ARM",
            "S2_rect_planarity": "BUMPY",
            "S2_cup_A_coverage": "CUP_A_EDGE",
            "S2_cup_B_coverage": "CUP_B_EDGE",
            "S2_cup_A_bump": "CUP_A_BUMP",
            "S2_cup_B_bump": "CUP_B_BUMP",
            "S2_cup_A_protrusion": "CUP_A_HIT",
            "S2_cup_B_protrusion": "CUP_B_HIT",
            "tilt_limit": "TILT",
            "length": "SHORT",
            "cup_support": "NO_SUPPORT",
            "occlusion_perimeter": "OCCLUDED",
            "S1_overlap": "COVERED",
            "class_excluded": "EXCLUDED",
            "S4_valid_ratio": "NO_DEPTH",
            "box_roi_wall": "WALL",
            "S3_flatness": "NO_FLAT",
        }
        for p in picks:
            if p.get("pickable"):
                continue
            pos = p.get("position_mm")
            if not pos or all(abs(v) < 1e-6 for v in pos):
                continue
            gates = p.get("gate_failures", []) or []
            if not gates:
                continue
            short = gate_short.get(gates[0], gates[0][:8])
            status = p.get("status", "?")
            label_text = f"X {short}" if status == "IMPOSSIBLE" else f"! {short}"
            pos_arr = np.asarray(pos, dtype=np.float64)
            normal = p.get("normal", [0, 0, -1])
            n = np.asarray(normal, dtype=np.float64)
            n = n / max(np.linalg.norm(n), 1e-12)
            label_pos = pos_arr + 15.0 * n
            lbl = self.scene_widget.add_3d_label(label_pos.tolist(), label_text)
            if lbl is not None:
                self.pick_labels.append(lbl)

    @staticmethod
    def _resolve_intrinsics(ply_path):
        """샷별 camera intrinsics 해석.

        Priority: <shot>_info.json > config/default.yaml.
        반환: (fx, fy, cx, cy, bottle_cid, source_label).
        하나라도 못 구하면 그 값은 None. caller가 None 체크 후 fallback.
        """
        fx = fy = cx = cy = None
        bottle_cid = 0  # 2026-05-17 nc=6 마이그레이션 후 새 기본값
        src = "yaml"
        try:
            import yaml as _yaml
            cfg_path = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
            with open(cfg_path, encoding="utf-8") as f:
                cfg = _yaml.safe_load(f)
            intr = cfg["camera"]["intrinsics"]
            fx = float(intr["fx"]); fy = float(intr["fy"])
            cx = float(intr["cx"]); cy = float(intr["cy"])
            bottle_cid = int(cfg["classes"]["bottle_cid"])
        except Exception:
            pass
        # 샷별 _info.json override — MR60/MR130 등 카메라가 다르면 여기서 덮어씀
        try:
            info_path = Path(ply_path).with_name(
                f"{Path(ply_path).stem}_info.json"
            )
            if info_path.exists():
                with open(info_path, encoding="utf-8") as f:
                    ci = json.load(f).get("camera_info", {})
                if "cal.fx" in ci:
                    fx = float(ci["cal.fx"])
                    fy = float(ci.get("cal.fy", fy))
                    cx = float(ci.get("cal.cx", cx))
                    cy = float(ci.get("cal.cy", cy))
                    sensor = ci.get("sensor name", "")
                    src = sensor or "info.json"
        except Exception:
            pass
        return fx, fy, cx, cy, bottle_cid, src

    def _make_cyl_segment(self, axis, P0, radius, a0, a1, with_endcap=False):
        """축(axis) 위 P0 기준 axial [a0,a1] 구간의 원기둥 표면 메시.

        a0,a1 은 P0 기준 축방향 좌표(mm). with_endcap이면 양 끝면도 채움(뚜껑용).
        반환: o3d TriangleMesh (카메라 좌표계) 또는 None.
        """
        axis = np.asarray(axis, dtype=np.float64)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        P0 = np.asarray(P0, dtype=np.float64)
        height = float(a1 - a0)
        if height < 2.0 or radius <= 0:
            return None
        mesh = o3d.geometry.TriangleMesh.create_cylinder(
            radius=float(radius), height=height, resolution=40,
            split=4, create_uv_map=False)
        if not with_endcap:
            # create_cylinder는 양 끝 disc 포함 — 옆면만 원하면 끝 삼각형 제거.
            # (간단히 두되, 뚜껑 segment는 endcap 유지해 disc 보이게.)
            pass
        # +Z 축, 원점 중심 → Z→axis 회전, 구간 중심으로 이동.
        z = np.array([0.0, 0.0, 1.0])
        v = np.cross(z, axis)
        s = float(np.linalg.norm(v))
        c = float(z @ axis)
        if s < 1e-8:
            R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
        else:
            vx = np.array([[0, -v[2], v[1]],
                           [v[2], 0, -v[0]],
                           [-v[1], v[0], 0]], dtype=np.float64)
            R = np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))
        mesh.rotate(R, center=(0, 0, 0))
        mesh.translate(P0 + 0.5 * (a0 + a1) * axis)
        try:
            mesh.compute_vertex_normals()
        except Exception:
            pass
        return mesh

    def _make_pick_gizmo(self, contact, approach):
        """병 픽 자세 표시 — 접촉점 구슬 + 접근축 화살표. 반환 [(mesh, rgb), ...]."""
        contact = np.asarray(contact, dtype=np.float64)
        approach = np.asarray(approach, dtype=np.float64)
        approach = approach / (np.linalg.norm(approach) + 1e-12)
        cyan = (0.1, 0.9, 0.9)
        out = []
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=6.0, resolution=12)
        sph.translate(contact)
        sph.compute_vertex_normals()
        out.append((sph, cyan))
        try:
            arr = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=2.5, cone_radius=5.0,
                cylinder_height=32.0, cone_height=12.0, resolution=12)
            z = np.array([0.0, 0.0, 1.0])
            v = np.cross(z, approach)
            s = float(np.linalg.norm(v))
            c = float(z @ approach)
            if s < 1e-8:
                R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
            else:
                vx = np.array([[0, -v[2], v[1]],
                               [v[2], 0, -v[0]],
                               [-v[1], v[0], 0]], dtype=np.float64)
                R = np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))
            arr.rotate(R, center=(0, 0, 0))
            arr.translate(contact)
            arr.compute_vertex_normals()
            out.append((arr, cyan))
        except Exception:
            pass
        return out

    # ── Depth recovery visualization ────────────────────────────
    def _build_recovery_meshes(self):
        """폴리곤별 NaN 영역을 복원해서 반투명 메시로 시각화.

        bottle (cid=BOTTLE_CID) → Method C (RANSAC 원기둥 fit + 광선 교차)
        다른 클래스 → Method A (scipy 선형 보간)
        실패 시 Method A로 fallback.

        각 메시는 클래스 색상 + alpha=0.4 으로 렌더링. 키 'M'으로 토글.
        """
        if not self.polygons_2d_list or not self.masks_list:
            return
        # Intrinsics는 __init__에서 self.fx/fy/cx/cy로 resolve됨 (샷별 _info.json 우선)
        fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy
        bottle_cid = self._bottle_cid_resolved
        if fx is None or fy is None or cx is None or cy is None:
            print("  recovery 시각화 skip — intrinsics 미설정")
            return
        # Imports
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from label_logic.depth_recovery import (
                recover_xyz_in_mask,
                recover_xyz_in_mask_cylinder,
                recover_xyz_in_mask_cylinder_from_cap,
                fit_bottle_cylinder,
                detect_cap_end,
                compute_bottle_pick_pose,
            )
            from scipy.spatial import Delaunay
        except ImportError as e:
            print(f"  recovery 시각화 skip — 모듈 import 실패: {e}")
            return

        n_built = 0
        for i, mask_ in enumerate(self.masks_list):
            if i >= len(self.cids):
                break
            mask = mask_.astype(bool)
            if mask.sum() == 0:
                continue
            cid = self.cids[i]
            # cid 유효성 가드 (미라벨 -1 / out-of-range 처리)
            if not (0 <= cid < len(CLASS_COLORS)):
                continue

            # ── Bottle: fit된 원기둥(반경30 prior)을 통째로 반투명 메시로 ──
            # 투명 병은 깊이가 bimodal(앞벽+see-through 뒷벽)이라 채울 NaN이 거의
            # 없어 구멍-채우기로는 안 보임. 대신 양쪽 벽 점에 반경30 고정 원기둥을
            # fit해서 복원된 병 전체를 메시로 그림. (cap detect는 실제 병에서 실패.)
            if cid == bottle_cid:
                try:
                    poly2d = (self.polygons_2d_list[i]
                              if i < len(self.polygons_2d_list) else None)
                    pose = compute_bottle_pick_pose(
                        self.grid_data, mask,
                        known_radius_mm=BOTTLE_KNOWN_RADIUS_MM,
                        cap_len_mm=BOTTLE_CAP_LEN_MM,
                        polygon_2d=poly2d, fx=fx, fy=fy, cx=cx, cy=cy,
                        length_mm=BOTTLE_LENGTH_MM,
                        upright_aspect_max=BOTTLE_UPRIGHT_ASPECT_MAX)
                except Exception as e:
                    print(f"  poly#{i} bottle pose error: {e}")
                    pose = {"ok": False, "fit": {"fit_ok": False}}
                fitc = pose.get("fit", {"fit_ok": False})
                if fitc.get("fit_ok"):
                    cap = pose.get("cap", {"ok": False})
                    axis = fitc["axis_dir"]; P0 = fitc["P0"]
                    radius = fitc["radius_mm"]
                    amin, amax = fitc["axial_min"], fitc["axial_max"]
                    cls_color = (CLASS_COLORS[cid] if cid < len(CLASS_COLORS)
                                 else (0.7, 0.7, 0.9))
                    cap_color = (0.95, 0.15, 0.15)   # 뚜껑(확신) = 빨강
                    # 뚜껑 가림/불확실(occluded)이면 빨강 표시 안 함 — 신뢰 못 하므로.
                    cap_trusted = cap.get("ok") and not cap.get("occluded", False)
                    segs = []   # (a0, a1, color, alpha, endcap)
                    if cap_trusted:
                        cl, ch = cap["cap_axial_lo"], cap["cap_axial_hi"]
                        # cap_axial_*는 center 기준, _make_cyl_segment는 P0 기준 →
                        # center = P0 + 0.5*(amin+amax)*axis 이므로 offset 보정.
                        off = 0.5 * (amin + amax)
                        cl_p, ch_p = cl + off, ch + off
                        if cap["cap_end"] == "min":
                            segs = [(cl_p, ch_p, cap_color, 0.7, True),
                                    (ch_p, amax, cls_color, 0.4, False)]
                        else:
                            segs = [(amin, cl_p, cls_color, 0.4, False),
                                    (cl_p, ch_p, cap_color, 0.7, True)]
                    else:
                        segs = [(amin, amax, cls_color, 0.4, False)]
                    added = 0
                    for si, (a0, a1, col, al, ec) in enumerate(segs):
                        m2 = self._make_cyl_segment(axis, P0, radius, a0, a1,
                                                    with_endcap=ec)
                        if m2 is None:
                            continue
                        mat = rendering.MaterialRecord()
                        mat.shader = "defaultLitTransparency"
                        mat.base_color = [float(col[0]), float(col[1]),
                                          float(col[2]), al]
                        name = f"recovery_{i}_{si}"
                        self.scene_widget.scene.add_geometry(name, m2, mat)
                        self.recovery_mesh_names.append(name)
                        added += 1
                    # ── 픽 자세 gizmo (접촉점 구슬 + 접근축 화살표) ──
                    if pose.get("ok"):
                        for gi, gm in enumerate(self._make_pick_gizmo(
                                pose["contact"], pose["approach"])):
                            gmesh, gcol = gm
                            gmat = rendering.MaterialRecord()
                            gmat.shader = "defaultLit"
                            gmat.base_color = [gcol[0], gcol[1], gcol[2], 1.0]
                            gname = f"recovery_{i}_pick{gi}"
                            self.scene_widget.scene.add_geometry(gname, gmesh, gmat)
                            self.recovery_mesh_names.append(gname)
                    if added:
                        n_built += 1
                        if pose.get("orientation") == "upright":
                            capmsg = "세워짐(탑다운 흡착)"
                        elif cap.get("ok") and "dark_frac" in cap:
                            capmsg = (f"cap={cap['cap_end']} dark={cap['dark_frac']:.0%} "
                                      f"conc={cap['concentration']:.2f} "
                                      f"conf={cap['confidence']:.2f}"
                                      + (" [가림/불확실]" if cap.get("occluded") else ""))
                        else:
                            capmsg = "cap=미검출"
                        vf = pose.get("verify", {}) if pose.get("ok") else {}
                        src = fitc.get("radius_src", "?")
                        if vf.get("applied"):
                            pickmsg = (f"축={src} 뚜껑↔축{vf['cap_axis_agree_deg']:.0f}° "
                                       f"뚜껑↔원기둥={vf['cap_surf_resid_mm']:.0f}mm")
                        else:
                            pickmsg = f"축={src}"
                        print(f"  poly#{i} cid={cid} bottle cyl: "
                              f"r={fitc['radius_mm']:.1f}({fitc['radius_src']}) "
                              f"len={amax-amin:.0f}mm inlier={fitc['inlier_ratio']:.0%} "
                              f"|axisZ|={abs(fitc['axis_dir'][2]):.2f} | {capmsg} | {pickmsg}")
                        continue
                print(f"  poly#{i} bottle cyl fit 실패 → NaN-fill fallback")

            # NaN 비율 확인 (5% 미만이면 시각화 의미 없음)
            valid_in = mask & self.valid_2d
            nan_ratio = 1.0 - (valid_in.sum() / mask.sum())
            if nan_ratio < 0.05:
                continue

            # 복원 실행 (non-bottle, 또는 bottle 원기둥 fit 실패 시 fallback)
            method_name = None
            new_grid = None
            n_filled = 0
            try:
                if cid == bottle_cid:
                    new_grid_c, n_c, info_c = recover_xyz_in_mask_cylinder(
                        self.grid_data, mask, fx, fy, cx, cy,
                    )
                    if n_c > 0:
                        new_grid, n_filled = new_grid_c, n_c
                        method_name = f"cylinder r={info_c['radius_mm']:.1f}"
                if n_filled == 0:
                    new_grid_a, n_a = recover_xyz_in_mask(
                        self.grid_data, mask, fx, fy, cx, cy,
                    )
                    if n_a > 0:
                        new_grid, n_filled = new_grid_a, n_a
                        method_name = "linear"
            except Exception as e:
                print(f"  poly#{i} recovery error: {e}")
                continue
            if n_filled < 10 or new_grid is None:
                continue

            # 복원된 픽셀(원래 NaN, 지금 valid) 위치 추출
            new_z = new_grid["z"]
            recovered = mask & np.isnan(self.grid_data["z"]) & ~np.isnan(new_z)
            if int(recovered.sum()) < 10:
                continue

            H, W = mask.shape
            vv, uu = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
            u_arr = uu[recovered].astype(np.float64)
            v_arr = vv[recovered].astype(np.float64)
            x_arr = new_grid["x"][recovered].astype(np.float64)
            y_arr = new_grid["y"][recovered].astype(np.float64)
            z_arr = new_z[recovered].astype(np.float64)

            # 2D Delaunay → 3D triangle mesh
            pts2d = np.column_stack([u_arr, v_arr])
            if len(pts2d) < 3:
                continue
            try:
                tri = Delaunay(pts2d)
            except Exception as e:
                print(f"  poly#{i} Delaunay error: {e}")
                continue

            vertices = np.column_stack([x_arr, y_arr, z_arr])
            triangles = tri.simplices
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(triangles)
            try:
                mesh.compute_vertex_normals()
            except Exception:
                pass

            # 반투명 머티리얼 — 클래스 색상 + alpha 0.4
            color = (CLASS_COLORS[cid] if cid < len(CLASS_COLORS)
                     else (0.7, 0.7, 0.9))
            mat = rendering.MaterialRecord()
            mat.shader = "defaultLitTransparency"
            mat.base_color = [float(color[0]), float(color[1]),
                              float(color[2]), 0.4]

            name = f"recovery_{i}"
            self.scene_widget.scene.add_geometry(name, mesh, mat)
            self.recovery_mesh_names.append(name)
            n_built += 1
            print(f"  poly#{i} cid={cid} recovery mesh: "
                  f"{n_filled} px filled, {len(triangles)} tri ({method_name})")

        if n_built > 0:
            print(f"  📐 recovery 메시 {n_built}개 추가 (M 키로 토글)")

    # ── Gizmo ──
    def _add_gizmo_geometries(self):
        for axis_idx, color in enumerate(self.AXIS_COLORS):
            ring = o3d.geometry.TriangleMesh.create_torus(
                torus_radius=self.gizmo_radius,
                tube_radius=self.gizmo_tube,
                radial_resolution=64,
                tubular_resolution=8,
            )
            # Default torus is in XY plane → rotates around Z. Reorient for X / Y.
            if axis_idx == 0:           # X axis: ring lies in YZ plane
                R = o3d.geometry.get_rotation_matrix_from_xyz((0.0, np.pi / 2, 0.0))
                ring.rotate(R, center=(0, 0, 0))
            elif axis_idx == 1:         # Y axis: ring lies in XZ plane
                R = o3d.geometry.get_rotation_matrix_from_xyz((np.pi / 2, 0.0, 0.0))
                ring.rotate(R, center=(0, 0, 0))
            ring.paint_uniform_color(list(color))
            ring.compute_vertex_normals()

            mat = rendering.MaterialRecord()
            mat.shader = "defaultUnlitTransparency"
            mat.base_color = [color[0], color[1], color[2], self.gizmo_alpha]
            mat.has_alpha = True
            self.scene_widget.scene.add_geometry(f"gizmo_{axis_idx}", ring, mat)

    def _refresh_gizmo_pose(self):
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = self.pivot
        for i in range(3):
            self.scene_widget.scene.set_geometry_transform(f"gizmo_{i}", T)
        self.scene_widget.scene.set_geometry_transform("pivot_marker", T)

    def _set_pivot(self, new_pivot):
        self.pivot = np.asarray(new_pivot, dtype=np.float64)
        self._refresh_gizmo_pose()
        # Default orbit (left-drag in empty space) now revolves around pivot.
        self.scene_widget.center_of_rotation = self.pivot.astype(np.float32)

    # ── Ray casting ──
    def _ray_through_pixel(self, x, y, view_mat=None, proj_mat=None,
                          view_w=None, view_h=None):
        """Return (origin_world, direction_world) for ray through pixel (x, y).

        If view_mat/proj_mat/view_w/view_h are None, use current camera state.
        """
        cam = self.scene_widget.scene.camera
        if view_mat is None:
            # Use camera.unproject (current camera state)
            frame = self.scene_widget.frame
            w = float(frame.width)
            h = float(frame.height)
            try:
                near = np.asarray(cam.unproject(float(x), float(y), 0.0, w, h),
                                 dtype=np.float64)
                far = np.asarray(cam.unproject(float(x), float(y), 1.0, w, h),
                                dtype=np.float64)
            except Exception:
                return None, None
            if not (np.all(np.isfinite(near)) and np.all(np.isfinite(far))):
                return None, None
            d = far - near
            n = float(np.linalg.norm(d))
            if not np.isfinite(n) or n < 1e-9:
                return None, None
            return near, d / n
        # Manual unproject through saved matrices (for drag consistency)
        w, h = float(view_w), float(view_h)
        ndc_x = 2.0 * float(x) / w - 1.0
        ndc_y = 1.0 - 2.0 * float(y) / h
        try:
            inv = np.linalg.inv(proj_mat @ view_mat)
        except np.linalg.LinAlgError:
            return None, None
        near = inv @ np.array([ndc_x, ndc_y, -1.0, 1.0], dtype=np.float64)
        far = inv @ np.array([ndc_x, ndc_y, 1.0, 1.0], dtype=np.float64)
        if abs(near[3]) < 1e-9 or abs(far[3]) < 1e-9:
            return None, None
        near = near[:3] / near[3]
        far = far[:3] / far[3]
        if not (np.all(np.isfinite(near)) and np.all(np.isfinite(far))):
            return None, None
        d = far - near
        n = float(np.linalg.norm(d))
        if not np.isfinite(n) or n < 1e-9:
            return None, None
        return near, d / n

    def _ray_axis_plane_hit(self, origin, direction, axis):
        """Intersect ray with plane perp to `axis` through `pivot`.

        Returns (t, hit) or (None, None).
        """
        denom = float(np.dot(direction, axis))
        if abs(denom) < 1e-6:
            return None, None
        t = float(np.dot(self.pivot - origin, axis) / denom)
        if t <= 0:
            return None, None
        return t, origin + t * direction

    def _pick_ring(self, mx, my):
        """Return (axis_idx, hit_world) if a ring is hit, else (None, None)."""
        origin, direction = self._ray_through_pixel(mx, my)
        if origin is None:
            return None, None
        # Pick threshold: 4× tube radius for a forgiving grab
        thresh = self.gizmo_tube * 4.0
        best = None
        for axis_idx in range(3):
            axis = self.AXIS_VECTORS[axis_idx]
            t, hit = self._ray_axis_plane_hit(origin, direction, axis)
            if t is None:
                continue
            d_to_pivot = float(np.linalg.norm(hit - self.pivot))
            d_to_ring = abs(d_to_pivot - self.gizmo_radius)
            if d_to_ring < thresh:
                if best is None or t < best[0]:
                    best = (t, axis_idx, hit)
        if best is None:
            return None, None
        return best[1], best[2]

    # ─────────────────────────────────────────────────────────────────
    # Manual pick labeling — 3D 클릭으로 픽 배치, 키로 변환, V로 검증, Enter로 저장
    # ─────────────────────────────────────────────────────────────────
    def _derive_manual_picks_path(self):
        """새 위치: <team>/manual_picks/<shot>_manual_picks.json. 폴더 자동 생성.
        폴백: PLY 옆 (옛 위치, 호환성)."""
        try:
            ply_path = Path(self.ply_path)
            ply_stem = ply_path.stem
            # raw/ 의 부모 = team_dir; team_dir/manual_picks/
            mp_dir = ply_path.parent.parent / "manual_picks"
            mp_dir.mkdir(parents=True, exist_ok=True)
            return mp_dir / f"{ply_stem}_manual_picks.json"
        except Exception:
            try:
                return Path(self.ply_path).parent / f"{Path(self.ply_path).stem}_manual_picks.json"
            except Exception:
                return Path("manual_picks.json")

    def _load_manual_picks_from_disk(self):
        """기존 manual_picks 파일이 있으면 로드해서 표시."""
        self.saved_manual_picks = {}   # poly_idx (str) -> dict
        if self.manual_picks_save_path and self.manual_picks_save_path.exists():
            try:
                with open(self.manual_picks_save_path, encoding="utf-8") as f:
                    self.saved_manual_picks = json.load(f)
                print(f"  ✓ loaded {len(self.saved_manual_picks)} manual picks from "
                      f"{self.manual_picks_save_path.name}")
            except Exception as e:
                print(f"  manual picks load failed: {e}")

    def _save_manual_pick_as_cap(self):
        """B 키 — 현재 manual_pick pose 를 bottle cap pose 로 저장.

        manual_pick 은 6-DOF (position + normal + long_axis + mid_axis) 를 갖는데
        bottle cap 매칭에도 같은 6-DOF 가 필요:
          cap_center = position           (cup A/B 중점 = 픽 위치)
          cap_axis   = long_axis          (두 cup 잇는 방향 = cap 원기둥 축)
          cap_normal = normal             (radial outward — cup 접근 방향)

        사용자 flow:
          1. L 키로 manual pick 모드 진입
          2. 폴리곤 (bottle) 클릭 → 픽 자동 배치
          3. gizmo 링/화살표 드래그로 두 cup 위치를 cap axis 위에 맞춤
             (두 cup 이 cap 원기둥 양 끝 또는 cap+body 경계 위에 가도록)
          4. B 키 → 즉시 manual_cap JSON 저장
        """
        if self.manual_pick is None:
            print("  (cap 저장 — 먼저 폴리곤 클릭해서 픽 자세 배치)")
            return

        try:
            # 저장 경로: <team>/manual_cap/<shot>_poly<idx>_manual_cap.json
            ply_path = Path(self.ply_path)
            shot_name = ply_path.stem
            cap_dir = ply_path.parent.parent / "manual_cap"
            cap_dir.mkdir(parents=True, exist_ok=True)
            poly_idx = int(self.manual_pick.get("polygon_idx", -1))
            out_path = cap_dir / f"{shot_name}_poly{poly_idx}_manual_cap.json"

            data = {
                "shot": shot_name,
                "polygon_idx": poly_idx,
                "center_mm": np.asarray(self.manual_pick["position"]).tolist(),
                "axis_dir": np.asarray(self.manual_pick["long_axis"]).tolist(),
                "approach_normal": np.asarray(self.manual_pick["normal"]).tolist(),
                "mid_axis": np.asarray(self.manual_pick["mid_axis"]).tolist(),
                "source": "view_3d manual_pick as cap (B key)",
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"\n  ✓ saved cap pose → {out_path.relative_to(ply_path.parent.parent.parent)}")
            print(f"    center : {data['center_mm']}")
            print(f"    axis   : {data['axis_dir']}")
            print(f"    normal : {data['approach_normal']}")
        except Exception as e:
            print(f"  cap pose 저장 실패: {e}")

    def _toggle_label_mode(self):
        if self.label_mode:
            self._exit_label_mode(save=False)
        else:
            self._enter_label_mode()

    def _enter_label_mode(self):
        self.label_mode = True
        # 라벨 모드 동안 카메라 gizmo 숨김 (manual pick gizmo와 색깔 충돌 방지)
        try:
            for i in range(3):
                self.scene_widget.scene.show_geometry(f"gizmo_{i}", False)
            self.scene_widget.scene.show_geometry("pivot_marker", False)
        except Exception:
            pass
        print("\n=== Manual pick label mode ON ===")
        print("  ▶ Click on polygon highlight: place pick center")
        print("  ▶ Shift + drag       : rotate pick (trackball)")
        print("  ▶ Ctrl + drag        : translate pick in screen plane")
        print("  ▶ Ctrl + Shift + drag: translate pick along normal (depth)")
        print("  ▶ Drag R/G/B rings   : rotate around X/Y/Z (alternative)")
        print("  ▶ Drag R/G/B arrows  : translate along X/Y/Z (alternative)")
        print("  R     : reset pick (re-place on next click)")
        print("  V     : validate (run all gates)")
        print("  Enter (1st): validate + ask confirmation")
        print("  Enter (2nd): SAVE manual pick to file")
        print("  ESC   : cancel save (if pending) / exit label mode")
        print("  L     : exit label mode (no save)")

    def _exit_label_mode(self, save=False):
        if save:
            self._save_manual_pick()
        self._clear_manual_geometry()
        self.manual_pick = None
        self.manual_pending_save = False
        self.label_mode = False
        # 카메라 gizmo 복원
        try:
            for i in range(3):
                self.scene_widget.scene.show_geometry(f"gizmo_{i}", True)
            self.scene_widget.scene.show_geometry("pivot_marker", True)
        except Exception:
            pass
        print("=== Manual pick label mode OFF ===")

    def _clear_manual_geometry(self):
        for name in self.manual_geom_names:
            try:
                self.scene_widget.scene.remove_geometry(name)
            except Exception:
                pass
        self.manual_geom_names = []

    def _find_nearest_highlight_point(self, screen_x, screen_y):
        """화면 좌표에 가장 가까운 highlight 점 → (point_3d, polygon_idx)."""
        if not self.highlights:
            return None, -1
        cam = self.scene_widget.scene.camera
        view = np.asarray(cam.get_view_matrix(), dtype=np.float64)
        proj = np.asarray(cam.get_projection_matrix(), dtype=np.float64)
        frame = self.scene_widget.frame
        w, h = float(frame.width), float(frame.height)
        best_d2, best_pt, best_poly = float("inf"), None, -1
        for poly_idx, hi in enumerate(self.highlights):
            pts = np.asarray(hi.points)
            if len(pts) == 0:
                continue
            stride = max(1, len(pts) // 5000)
            pts_sub = pts[::stride]
            n = len(pts_sub)
            h4 = np.hstack([pts_sub, np.ones((n, 1))])
            cam_pts = (view @ h4.T).T
            front = cam_pts[:, 2] < 0
            if not front.any():
                continue
            cam_f = cam_pts[front]
            pts_f = pts_sub[front]
            clip = (proj @ cam_f.T).T
            wv = np.abs(clip[:, 3]) > 1e-9
            if not wv.any():
                continue
            clip = clip[wv]
            pts_f = pts_f[wv]
            ndc = clip[:, :3] / clip[:, 3:4]
            sx = (ndc[:, 0] + 1) * 0.5 * w
            sy = (1 - ndc[:, 1]) * 0.5 * h
            d2 = (sx - screen_x) ** 2 + (sy - screen_y) ** 2
            idx = int(np.argmin(d2))
            if d2[idx] < best_d2:
                best_d2 = d2[idx]
                best_pt = pts_f[idx]
                best_poly = poly_idx
        if best_pt is None or best_d2 > 50 ** 2:
            return None, -1
        return best_pt, best_poly

    def _place_manual_pick(self, point_3d, polygon_idx):
        """클릭된 3D 점 위치에 픽 배치.

        polygon_idx는 highlight 리스트 안 index (k). 저장 시 result_data 원본
        index로 변환해서 manual_picks JSON 키로 사용 — label_tool/probe와 매핑 일치.
        """
        self.manual_pending_save = False   # 새 픽 배치 → 저장 대기 취소
        if polygon_idx < 0 or polygon_idx >= len(self.avg_normals):
            return
        # highlight idx → result_data 원본 idx (skip된 폴리곤 보정)
        if polygon_idx < len(self.original_indices):
            poly_idx_original = int(self.original_indices[polygon_idx])
        else:
            poly_idx_original = polygon_idx   # fallback (기대보다 안 발생)
        normal = self.avg_normals[polygon_idx].copy()
        # long_axis: 임의 (X-axis projected to surface plane)
        ref = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        long_axis = ref - np.dot(ref, normal) * normal
        long_axis = long_axis / max(np.linalg.norm(long_axis), 1e-12)
        mid_axis = np.cross(normal, long_axis)
        # 표면에서 살짝 뜸 (2mm safety along normal — toward camera)
        position = np.asarray(point_3d, dtype=np.float64) + 2.0 * normal
        cid = self.cids[polygon_idx] if polygon_idx < len(self.cids) else -1
        self.manual_pick = {
            "position": position,
            "normal": normal,
            "long_axis": long_axis,
            "mid_axis": mid_axis,
            "polygon_idx": poly_idx_original,    # ★ result_data 원본 idx
            "highlight_idx": polygon_idx,         # 시각화용 (디버그)
            "cid": cid,
        }
        cname = CLASS_NAMES[cid] if 0 <= cid < len(CLASS_NAMES) else "?"
        print(f"  placed manual pick on polygon #{poly_idx_original} ({cname}) "
              f"at [{position[0]:.0f}, {position[1]:.0f}, {position[2]:.0f}]")
        self._refresh_manual_geometry()

    def _translate_manual_pick(self, dx, dy, dz):
        """픽 좌표계에서 long/mid/normal 방향으로 mm 이동."""
        if not self.manual_pick:
            return
        delta = (dx * self.manual_pick["long_axis"]
                 + dy * self.manual_pick["mid_axis"]
                 + dz * self.manual_pick["normal"])
        self.manual_pick["position"] = self.manual_pick["position"] + delta
        self._refresh_manual_geometry()

    def _rotate_manual_pick_yaw(self, angle_deg):
        if not self.manual_pick:
            return
        a = np.radians(angle_deg)
        n = self.manual_pick["normal"]
        la = self.manual_pick["long_axis"]
        cos_a, sin_a = np.cos(a), np.sin(a)
        rot = (la * cos_a + np.cross(n, la) * sin_a
               + n * np.dot(n, la) * (1 - cos_a))
        rot = rot - np.dot(rot, n) * n
        rot = rot / max(np.linalg.norm(rot), 1e-12)
        self.manual_pick["long_axis"] = rot
        self.manual_pick["mid_axis"] = np.cross(n, rot)
        self._refresh_manual_geometry()

    def _refresh_manual_geometry(self):
        self._clear_manual_geometry()
        if not self.manual_pick:
            return
        position = self.manual_pick["position"]
        long_a = self.manual_pick["long_axis"]
        mid_a = self.manual_pick["mid_axis"]
        normal = self.manual_pick["normal"]
        # 현재 picking.py 상수에 맞춤 (pitch 45mm, OD 25)
        half_long = (45.0 + 25.0) / 2.0
        half_short = 25.0 / 2.0
        cup_offset = 45.0 / 2.0
        corners = [
            position + half_long * long_a + half_short * mid_a,
            position + half_long * long_a - half_short * mid_a,
            position - half_long * long_a - half_short * mid_a,
            position - half_long * long_a + half_short * mid_a,
        ]
        # Outline (yellow line)
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(corners)
        ls.lines = o3d.utility.Vector2iVector(
            np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32))
        ls.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([1.0, 1.0, 0.0]), (4, 1)))
        mat_line = rendering.MaterialRecord()
        mat_line.shader = "unlitLine"
        mat_line.line_width = 6.0
        self.scene_widget.scene.add_geometry("manual_rect", ls, mat_line)
        self.manual_geom_names.append("manual_rect")
        # Cup spheres (yellow)
        for i, sign in enumerate([+1, -1]):
            cup_pos = position + sign * cup_offset * long_a
            s = o3d.geometry.TriangleMesh.create_sphere(radius=5.0, resolution=10)
            s.translate(np.asarray(cup_pos))
            s.paint_uniform_color([1.0, 1.0, 0.0])
            s.compute_vertex_normals()
            mat = rendering.MaterialRecord()
            mat.shader = "defaultUnlit"
            name = f"manual_cup_{i}"
            self.scene_widget.scene.add_geometry(name, s, mat)
            self.manual_geom_names.append(name)
        # ── 3 회전 링 (R/G/B) — 클릭+드래그로 그 축 둘레 회전 ───────
        # 노란 approach arrow는 제거 (gizmo의 파란 Z arrow가 normal 방향 표시 = +normal)
        for axis_idx, axis_dir in enumerate([long_a, mid_a, normal]):
            ring = o3d.geometry.TriangleMesh.create_torus(
                torus_radius=self.pick_gizmo_ring_radius,
                tube_radius=self.pick_gizmo_ring_tube,
                radial_resolution=64,
                tubular_resolution=8,
            )
            # default torus axis = Z. Align to axis_dir.
            R = self._rotation_align(np.array([0.0, 0.0, 1.0]), axis_dir)
            ring.rotate(R, center=(0, 0, 0))
            ring.translate(np.asarray(position))
            color = self.pick_axis_colors[axis_idx]
            ring.paint_uniform_color(list(color))
            ring.compute_vertex_normals()
            mat_r = rendering.MaterialRecord()
            mat_r.shader = "defaultUnlitTransparency"
            mat_r.base_color = [color[0], color[1], color[2], 0.85]
            mat_r.has_alpha = True
            name = f"manual_ring_{axis_idx}"
            self.scene_widget.scene.add_geometry(name, ring, mat_r)
            self.manual_geom_names.append(name)

        # ── 3 이동 화살표 (R/G/B) — 클릭+드래그로 그 축 따라 이동 ──
        for axis_idx, axis_dir in enumerate([long_a, mid_a, normal]):
            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=self.pick_gizmo_arrow_radius,
                cone_radius=self.pick_gizmo_arrow_radius * 2.5,
                cylinder_height=self.pick_gizmo_arrow_len * 0.78,
                cone_height=self.pick_gizmo_arrow_len * 0.22,
            )
            # default arrow axis = +Z. Rotate to axis_dir.
            R = self._rotation_align(np.array([0.0, 0.0, 1.0]), axis_dir)
            arrow.rotate(R, center=(0, 0, 0))
            arrow.translate(np.asarray(position))
            color = self.pick_axis_colors[axis_idx]
            arrow.paint_uniform_color(list(color))
            arrow.compute_vertex_normals()
            mat_ar = rendering.MaterialRecord()
            mat_ar.shader = "defaultUnlit"
            name = f"manual_axis_arrow_{axis_idx}"
            self.scene_widget.scene.add_geometry(name, arrow, mat_ar)
            self.manual_geom_names.append(name)

    def _pick_manual_handle(self, screen_x, screen_y):
        """클릭 위치에서 manual pick gizmo handle 검출.

        Returns ('ring', axis_idx, hit_world) or ('arrow', axis_idx, hit_world)
        or (None, None, None).
        """
        if not self.manual_pick:
            return None, None, None
        origin, direction = self._ray_through_pixel(screen_x, screen_y)
        if origin is None:
            print(f"  [hit-test] ray failed for ({screen_x}, {screen_y})")
            return None, None, None
        position = self.manual_pick["position"]
        axes = [self.manual_pick["long_axis"],
                self.manual_pick["mid_axis"],
                self.manual_pick["normal"]]

        # 1) 회전 링 검사 — 평면 (axis 법선) 위 ring radius 근처에 hit
        best_kind = None
        best_axis = None
        best_t = float("inf")
        best_hit = None
        ring_r = self.pick_gizmo_ring_radius
        ring_tol = self.pick_gizmo_hit_tol
        ring_debug = []
        for axis_idx, axis_dir in enumerate(axes):
            denom = float(np.dot(direction, axis_dir))
            if abs(denom) < 1e-6:
                ring_debug.append(f"axis{axis_idx}: parallel (denom={denom:.4f})")
                continue   # ray parallel to ring plane
            t = float(np.dot(position - origin, axis_dir) / denom)
            if t < 0:
                ring_debug.append(f"axis{axis_idx}: t<0 ({t:.1f})")
                continue
            hit = origin + t * direction
            r = float(np.linalg.norm(hit - position))
            ring_debug.append(f"axis{axis_idx}: r={r:.1f} (ring_r={ring_r}±{ring_tol})")
            if abs(r - ring_r) > ring_tol:
                continue
            if t < best_t:
                best_t = t
                best_kind = "ring"
                best_axis = axis_idx
                best_hit = hit

        # 2) 이동 화살표 검사 — ray와 axis line 사이 최단 거리가 작으면 hit
        arrow_len = self.pick_gizmo_arrow_len
        arrow_tol = self.pick_gizmo_arrow_radius * 5.0   # 후하게
        for axis_idx, axis_dir in enumerate(axes):
            # closest point on ray to line (position → position + axis_dir)
            d1 = direction
            d2 = axis_dir
            r0 = origin - position
            a = float(np.dot(d1, d1))
            b = float(np.dot(d1, d2))
            c = float(np.dot(d2, d2))
            d = float(np.dot(d1, r0))
            e = float(np.dot(d2, r0))
            denom = a * c - b * b
            if abs(denom) < 1e-9:
                continue
            t_ray = (b * e - c * d) / denom        # along ray
            t_line = (a * e - b * d) / denom       # along axis line
            if t_ray < 0:
                continue
            if t_line < 0 or t_line > arrow_len:
                continue
            p_ray = origin + t_ray * d1
            p_line = position + t_line * d2
            if float(np.linalg.norm(p_ray - p_line)) > arrow_tol:
                continue
            if t_ray < best_t:
                best_t = t_ray
                best_kind = "arrow"
                best_axis = axis_idx
                best_hit = p_ray

        if best_kind is None:
            # 디버그 출력 — 어느 축도 hit 못 함
            print(f"  [hit-test miss] ({screen_x}, {screen_y}) → no ring/arrow")
            for d in ring_debug:
                print(f"    {d}")
        return best_kind, best_axis, best_hit

    def _begin_manual_drag(self, kind, axis_idx, hit):
        """드래그 시작 — 현재 pose snapshot 저장."""
        if not self.manual_pick:
            return
        self.manual_pending_save = False   # 편집 → 저장 대기 취소
        self.manual_active_handle = (kind, axis_idx)
        self.manual_drag_init_pose = {
            "position": self.manual_pick["position"].copy(),
            "long_axis": self.manual_pick["long_axis"].copy(),
            "mid_axis": self.manual_pick["mid_axis"].copy(),
            "normal": self.manual_pick["normal"].copy(),
        }
        self.manual_drag_init_hit = np.asarray(hit, dtype=np.float64)

    def _update_manual_drag(self, screen_x, screen_y):
        """드래그 중 — 현재 마우스 위치로부터 회전/이동 적용."""
        if self.manual_active_handle is None or not self.manual_drag_init_pose:
            return
        origin, direction = self._ray_through_pixel(screen_x, screen_y)
        if origin is None:
            return
        kind, axis_idx = self.manual_active_handle
        init = self.manual_drag_init_pose
        axes_init = [init["long_axis"], init["mid_axis"], init["normal"]]
        axis_dir = axes_init[axis_idx]
        position_init = init["position"]

        if kind == "ring":
            # Plane (perpendicular to axis_dir, through position) → ray 교차
            denom = float(np.dot(direction, axis_dir))
            if abs(denom) < 1e-6:
                return
            t = float(np.dot(position_init - origin, axis_dir) / denom)
            if t < 0:
                return
            hit_now = origin + t * direction
            v_init = self.manual_drag_init_hit - position_init
            v_now = hit_now - position_init
            # 평면 내 (axis_dir 성분 제거)
            v_init_p = v_init - np.dot(v_init, axis_dir) * axis_dir
            v_now_p = v_now - np.dot(v_now, axis_dir) * axis_dir
            n_init = float(np.linalg.norm(v_init_p))
            n_now = float(np.linalg.norm(v_now_p))
            if n_init < 1e-6 or n_now < 1e-6:
                return
            cos_a = float(np.dot(v_init_p, v_now_p)) / (n_init * n_now)
            cos_a = max(-1.0, min(1.0, cos_a))
            cross = np.cross(v_init_p, v_now_p)
            sin_a = float(np.dot(cross, axis_dir)) / (n_init * n_now)
            angle = float(np.arctan2(sin_a, cos_a))
            # Rodrigues — pose의 3축에 적용
            cos_r, sin_r = np.cos(angle), np.sin(angle)
            def rot(v):
                return (v * cos_r
                        + np.cross(axis_dir, v) * sin_r
                        + axis_dir * np.dot(axis_dir, v) * (1 - cos_r))
            new_long = rot(init["long_axis"])
            new_mid = rot(init["mid_axis"])
            new_normal = rot(init["normal"])
            # 정규화
            new_long /= max(np.linalg.norm(new_long), 1e-12)
            new_mid /= max(np.linalg.norm(new_mid), 1e-12)
            new_normal /= max(np.linalg.norm(new_normal), 1e-12)
            # Normal이 카메라 반대편(z>0)으로 가면 거부
            if new_normal[2] > 0:
                return
            self.manual_pick["long_axis"] = new_long
            self.manual_pick["mid_axis"] = new_mid
            self.manual_pick["normal"] = new_normal
        elif kind == "arrow":
            # 이동 — ray와 axis line 사이 최단점 → axis_dir 방향 1D 변위
            d1 = direction
            d2 = axis_dir
            r0_init = self.manual_drag_init_hit - position_init
            r0_now = origin - position_init
            a = float(np.dot(d1, d1))
            b = float(np.dot(d1, d2))
            c = float(np.dot(d2, d2))
            denom = a * c - b * b
            if abs(denom) < 1e-9:
                return
            d_init = float(np.dot(d1, r0_init))
            e_init = float(np.dot(d2, r0_init))
            t_line_init = (a * e_init - b * d_init) / denom
            d_now = float(np.dot(d1, r0_now))
            e_now = float(np.dot(d2, r0_now))
            t_line_now = (a * e_now - b * d_now) / denom
            delta = t_line_now - t_line_init
            self.manual_pick["position"] = position_init + delta * axis_dir

        self._refresh_manual_geometry()

    def _end_manual_drag(self):
        self.manual_active_handle = None
        self.manual_drag_init_pose = None
        self.manual_drag_init_hit = None

    # ── Shift/Ctrl + drag — modifier 기반 픽 조작 ──────────────────────
    @staticmethod
    def _rodrigues_axis_angle(axis, angle):
        a = np.asarray(axis, dtype=np.float64)
        a = a / max(np.linalg.norm(a), 1e-12)
        c, s = np.cos(angle), np.sin(angle)
        K = np.array([[0, -a[2], a[1]],
                      [a[2], 0, -a[0]],
                      [-a[1], a[0], 0]], dtype=np.float64)
        return np.eye(3) + s * K + (1 - c) * (K @ K)

    def _camera_axes_world(self):
        """카메라 기준 right (world space x), up (world space y), forward."""
        cam = self.scene_widget.scene.camera
        view = np.asarray(cam.get_view_matrix(), dtype=np.float64)
        # view matrix rows = camera axes in world space
        cam_right = view[0, :3]
        cam_up = view[1, :3]
        cam_forward = -view[2, :3]
        return cam_right, cam_up, cam_forward

    def _apply_shift_rotation(self, dx, dy):
        """Trackball-style: 드래그 방향 → 화면 직교축 회전.

        Normal이 카메라 반대편(z>0)으로 가면 회전 거부 — 픽이 뒤집히는 것 방지.
        """
        if not self.manual_pick:
            return
        cam_right, cam_up, _ = self._camera_axes_world()
        sensitivity = 0.003   # rad/pixel — 너무 빠른 회전 방지
        R_yaw = self._rodrigues_axis_angle(cam_up, dx * sensitivity)
        R_pitch = self._rodrigues_axis_angle(cam_right, dy * sensitivity)
        R = R_pitch @ R_yaw
        new_long = R @ self.manual_pick["long_axis"]
        new_mid = R @ self.manual_pick["mid_axis"]
        new_normal = R @ self.manual_pick["normal"]
        new_normal = new_normal / max(np.linalg.norm(new_normal), 1e-12)
        # Normal이 카메라(-Z) 반대편이면 거부 (픽이 빈 안쪽 향하면 흡착 불가)
        if new_normal[2] > 0:
            print(f"  [rotation blocked] normal would flip away from camera (z={new_normal[2]:.2f})")
            return
        # 90°에 가까우면 한번 더 경고만 (tilt limit 검증과 별개)
        self.manual_pick["long_axis"] = new_long / max(np.linalg.norm(new_long), 1e-12)
        self.manual_pick["mid_axis"] = new_mid / max(np.linalg.norm(new_mid), 1e-12)
        self.manual_pick["normal"] = new_normal
        self._refresh_manual_geometry()

    def _apply_ctrl_translation_xy(self, dx, dy):
        """화면 평면 이동 — drag pixel을 mm 환산해서 픽 위치 이동."""
        if not self.manual_pick:
            return
        cam_right, cam_up, _ = self._camera_axes_world()
        z_pick = float(self.manual_pick["position"][2])
        if z_pick <= 1e-6:
            return
        # 샷별 intrinsics 사용 — MR60/MR130 등 카메라 다르면 픽셀→mm 환산도 다름
        if self.fx is not None and self.fy is not None:
            fx_avg = 0.5 * (self.fx + self.fy)
        else:
            fx_avg = 0.5 * (1242.55 + 1242.77)   # safety fallback (MR60)
        mm_per_pixel = z_pick / fx_avg
        delta = (dx * mm_per_pixel) * cam_right + (-dy * mm_per_pixel) * cam_up
        self.manual_pick["position"] = self.manual_pick["position"] + delta
        self._refresh_manual_geometry()

    def _apply_ctrl_translation_z(self, dy):
        """깊이 이동 — drag y에 비례해 normal 방향 이동."""
        if not self.manual_pick:
            return
        delta_mm = -dy * 0.2   # 위로 드래그 = +normal (카메라 쪽). 0.2mm/pixel
        normal = self.manual_pick["normal"]
        self.manual_pick["position"] = (
            self.manual_pick["position"] + delta_mm * normal
        )
        self._refresh_manual_geometry()

    def _begin_mod_drag(self, kind, mx, my):
        self.manual_pending_save = False   # 편집 → 저장 대기 취소
        self.manual_mod_drag_kind = kind
        self.manual_mod_drag_last_xy = (mx, my)
        print(f"  [mod-drag start] {kind}")

    def _update_mod_drag(self, mx, my):
        if not self.manual_mod_drag_kind or not self.manual_mod_drag_last_xy:
            return
        dx = mx - self.manual_mod_drag_last_xy[0]
        dy = my - self.manual_mod_drag_last_xy[1]
        self.manual_mod_drag_last_xy = (mx, my)
        if self.manual_mod_drag_kind == "rotate":
            self._apply_shift_rotation(dx, dy)
        elif self.manual_mod_drag_kind == "translate_xy":
            self._apply_ctrl_translation_xy(dx, dy)
        elif self.manual_mod_drag_kind == "translate_z":
            self._apply_ctrl_translation_z(dy)

    def _end_mod_drag(self):
        self.manual_mod_drag_kind = None
        self.manual_mod_drag_last_xy = None

    def _validate_manual_pick(self):
        """현재 manual pick 자세에 모든 게이트 적용 → 콘솔에 결과 출력."""
        if not self.manual_pick:
            print("  (no manual pick to validate)")
            return
        sys.path.insert(0, str(Path(__file__).parent))
        try:
            from label_logic.suction_score import (
                cup_self_coverage, cup_max_bump, approach_box_check,
            )
            from label_logic.picking import (
                GATE_S2_CUP_COVERAGE_MIN, GATE_S2_CUP_BUMP_MM,
                GATE_S2_RECT_PLANARITY_MM, GATE_S2_ARM_PROTRUSION_MM,
                MAX_TILT_DEG, MAX_TILT_COS, HEAD_CLEARANCE_MM,
                ARM_BOX_SCALE, ARM_BOX_HEIGHT_MM,
                SUCTION_CUP_OD_MM, SUCTION_CUP_PITCH_MM, RECT_LONG_MM, RECT_SHORT_MM,
            )
        except Exception as e:
            print(f"  validation imports failed: {e}")
            return

        position = self.manual_pick["position"]
        normal = self.manual_pick["normal"]
        long_axis = self.manual_pick["long_axis"]
        mid_axis = self.manual_pick["mid_axis"]
        poly_idx = self.manual_pick["polygon_idx"]
        if poly_idx < 0 or poly_idx >= len(self.masks_list):
            print("  (no polygon mask for validation)")
            return
        mask = self.masks_list[poly_idx]
        valid = self.valid_2d
        grid = self.grid_data

        cup_offset = SUCTION_CUP_PITCH_MM / 2.0
        cup_a = position + cup_offset * long_axis
        cup_b = position - cup_offset * long_axis

        results = {}
        # Tilt
        cos_tilt = float(-normal[2])
        tilt_deg = float(np.degrees(np.arccos(max(-1.0, min(1.0, cos_tilt)))))
        results["tilt_deg"] = tilt_deg
        results["tilt_pass"] = tilt_deg <= MAX_TILT_DEG
        # Cup coverage
        cov_a = cup_self_coverage(mask, cup_a, radius_mm=SUCTION_CUP_OD_MM/2.0)
        cov_b = cup_self_coverage(mask, cup_b, radius_mm=SUCTION_CUP_OD_MM/2.0)
        results["cup_a_coverage"] = cov_a
        results["cup_b_coverage"] = cov_b
        results["cup_coverage_pass"] = (cov_a is not None and cov_a >= GATE_S2_CUP_COVERAGE_MIN
                                         and cov_b is not None and cov_b >= GATE_S2_CUP_COVERAGE_MIN)
        # Cup bump
        bump_a = cup_max_bump(grid, valid, cup_a, normal,
                               radius_mm=SUCTION_CUP_OD_MM/2.0, self_mask=mask)
        bump_b = cup_max_bump(grid, valid, cup_b, normal,
                               radius_mm=SUCTION_CUP_OD_MM/2.0, self_mask=mask)
        results["cup_a_bump_mm"] = bump_a
        results["cup_b_bump_mm"] = bump_b
        bump_pass = ((bump_a is None or bump_a <= GATE_S2_CUP_BUMP_MM)
                      and (bump_b is None or bump_b <= GATE_S2_CUP_BUMP_MM))
        results["cup_bump_pass"] = bump_pass
        # Rect planarity
        use_self = mask & valid
        if use_self.any():
            xyz = np.stack([grid["x"][use_self], grid["y"][use_self],
                             grid["z"][use_self]], axis=-1).astype(np.float64)
            rel = xyz - position[None, :]
            x_p = rel @ long_axis
            y_p = rel @ mid_axis
            z_p = rel @ normal
            in_rect = (np.abs(x_p) <= RECT_LONG_MM/2.0) & (np.abs(y_p) <= RECT_SHORT_MM/2.0)
            n_in = int(in_rect.sum())
            if n_in >= 30:
                rect_std = float(z_p[in_rect].std())
                results["rect_planarity_mm"] = rect_std
                results["rect_planarity_pass"] = rect_std <= GATE_S2_RECT_PLANARITY_MM
            else:
                results["rect_planarity_mm"] = None
                results["rect_planarity_pass"] = False
        else:
            results["rect_planarity_mm"] = None
            results["rect_planarity_pass"] = False
        # Arm clearance (3D 직사각형 — 픽 사각형 × ARM_BOX_SCALE)
        arm_half_long = (RECT_LONG_MM / 2.0) * ARM_BOX_SCALE
        arm_half_short = (RECT_SHORT_MM / 2.0) * ARM_BOX_SCALE
        arm_chk = approach_box_check(
            grid, valid, position, normal, long_axis, mid_axis,
            half_long_mm=arm_half_long, half_short_mm=arm_half_short,
            t_min_mm=HEAD_CLEARANCE_MM, t_max_mm=ARM_BOX_HEIGHT_MM,
            self_mask=mask,
        )
        results["arm_outside_max_mm"] = arm_chk["max_t_mm"] if arm_chk["n_outside"] else 0.0
        results["arm_pass"] = (arm_chk["n_outside"] == 0
                                or arm_chk["max_t_mm"] <= GATE_S2_ARM_PROTRUSION_MM)

        all_pass = (results["tilt_pass"] and results["cup_coverage_pass"]
                     and results["cup_bump_pass"]
                     and results["rect_planarity_pass"]
                     and results["arm_pass"])
        results["all_pass"] = all_pass
        self.manual_pick["validation"] = results

        print("\n  --- Manual pick validation ---")
        print(f"  tilt: {tilt_deg:.1f}° / max {MAX_TILT_DEG}° → {'PASS' if results['tilt_pass'] else 'FAIL'}")
        print(f"  cup A coverage: {cov_a:.0%} / min {GATE_S2_CUP_COVERAGE_MIN:.0%} → {'PASS' if (cov_a or 0) >= GATE_S2_CUP_COVERAGE_MIN else 'FAIL'}"
              if cov_a is not None else "  cup A coverage: N/A")
        print(f"  cup B coverage: {cov_b:.0%} / min {GATE_S2_CUP_COVERAGE_MIN:.0%} → {'PASS' if (cov_b or 0) >= GATE_S2_CUP_COVERAGE_MIN else 'FAIL'}"
              if cov_b is not None else "  cup B coverage: N/A")
        print(f"  cup A bump: {bump_a:.1f}mm / max {GATE_S2_CUP_BUMP_MM}mm" if bump_a is not None else "  cup A bump: N/A")
        print(f"  cup B bump: {bump_b:.1f}mm / max {GATE_S2_CUP_BUMP_MM}mm" if bump_b is not None else "  cup B bump: N/A")
        if results["rect_planarity_mm"] is not None:
            print(f"  rect planarity: {results['rect_planarity_mm']:.1f}mm std / max {GATE_S2_RECT_PLANARITY_MM}mm → {'PASS' if results['rect_planarity_pass'] else 'FAIL'}")
        print(f"  arm clearance: max {results['arm_outside_max_mm']:.1f}mm / max {GATE_S2_ARM_PROTRUSION_MM}mm → {'PASS' if results['arm_pass'] else 'FAIL'}")
        print(f"  → OVERALL: {'✓ PASS' if all_pass else '✗ FAIL'}")

    def _save_manual_pick(self):
        if not self.manual_pick:
            print("  (nothing to save)")
            return
        # 검증 한 번 강제
        self._validate_manual_pick()
        position = self.manual_pick["position"]
        normal = self.manual_pick["normal"]
        long_axis = self.manual_pick["long_axis"]
        mid_axis = self.manual_pick["mid_axis"]
        cup_offset = 45.0 / 2.0
        cup_a = (position + cup_offset * long_axis).tolist()
        cup_b = (position - cup_offset * long_axis).tolist()
        record = {
            "polygon_idx": int(self.manual_pick["polygon_idx"]),
            "cid": int(self.manual_pick.get("cid", -1)),
            "position_mm": position.tolist(),
            "normal": normal.tolist(),
            "long_axis": long_axis.tolist(),
            "mid_axis": mid_axis.tolist(),
            "cup_a_3d": cup_a,
            "cup_b_3d": cup_b,
            "validation": self.manual_pick.get("validation", {}),
        }
        key = str(self.manual_pick["polygon_idx"])
        self.saved_manual_picks[key] = record
        try:
            self.manual_picks_save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.manual_picks_save_path, "w", encoding="utf-8") as f:
                json.dump(self.saved_manual_picks, f, ensure_ascii=False, indent=2)
            print(f"  ✓ saved manual pick (polygon #{key}) → "
                  f"{self.manual_picks_save_path.name}")
        except Exception as e:
            print(f"  save failed: {e}")

    # ── Mouse callback ──
    def _on_mouse(self, event):
        # Manual pick 라벨 모드 — gizmo 우선 / 픽 없으면 첫 클릭으로 배치
        # 좌표는 widget-local (event.x/y 직접 사용 — 기존 카메라 gizmo와 동일)
        if self.label_mode:
            mx, my = event.x, event.y
            # Modifier 키 상태 (Open3D MouseEvent.modifiers는 bit flags)
            try:
                mods = int(event.modifiers)
                shift_down = bool(mods & int(gui.KeyModifier.SHIFT))
                ctrl_down = bool(mods & int(gui.KeyModifier.CTRL))
            except Exception:
                shift_down = ctrl_down = False

            if event.type == gui.MouseEvent.Type.BUTTON_DOWN:
                if event.is_button_down(gui.MouseButton.LEFT):
                    # 픽 있음 + modifier 누름 → modifier drag 시작
                    if self.manual_pick is not None and (shift_down or ctrl_down):
                        if ctrl_down and shift_down:
                            self._begin_mod_drag("translate_z", mx, my)
                        elif shift_down:
                            self._begin_mod_drag("rotate", mx, my)
                        elif ctrl_down:
                            self._begin_mod_drag("translate_xy", mx, my)
                        return gui.Widget.EventCallbackResult.CONSUMED
                    # 픽 있음 + 일반 클릭 → gizmo handle 우선 검사
                    if self.manual_pick is not None:
                        kind, axis_idx, hit = self._pick_manual_handle(mx, my)
                        if kind is not None:
                            print(f"  [drag start] {kind} axis={axis_idx}")
                            self._begin_manual_drag(kind, axis_idx, hit)
                            return gui.Widget.EventCallbackResult.CONSUMED
                        # gizmo 아니고 픽 있음 → 그냥 fallthrough (카메라 회전)
                    else:
                        # 픽 없음 → 첫 클릭으로 배치
                        pt, poly_idx = self._find_nearest_highlight_point(mx, my)
                        if pt is not None:
                            self._place_manual_pick(pt, poly_idx)
                            return gui.Widget.EventCallbackResult.CONSUMED
                        else:
                            print("  (no highlight point near click — try clicking on colored region)")
                            return gui.Widget.EventCallbackResult.CONSUMED
            elif event.type == gui.MouseEvent.Type.DRAG:
                if self.manual_mod_drag_kind is not None:
                    self._update_mod_drag(mx, my)
                    return gui.Widget.EventCallbackResult.CONSUMED
                if self.manual_active_handle is not None:
                    self._update_manual_drag(mx, my)
                    return gui.Widget.EventCallbackResult.CONSUMED
            elif event.type == gui.MouseEvent.Type.BUTTON_UP:
                if self.manual_mod_drag_kind is not None:
                    self._end_mod_drag()
                    return gui.Widget.EventCallbackResult.CONSUMED
                if self.manual_active_handle is not None:
                    self._end_manual_drag()
                    return gui.Widget.EventCallbackResult.CONSUMED
        et = event.type

        # 라벨 모드면 카메라 gizmo 무시 (manual pick gizmo와 충돌 방지)
        if self.label_mode and et == gui.MouseEvent.Type.BUTTON_DOWN:
            return gui.Widget.EventCallbackResult.IGNORED

        if et == gui.MouseEvent.Type.BUTTON_DOWN:
            if event.is_button_down(gui.MouseButton.LEFT):
                axis, hit = self._pick_ring(event.x, event.y)
                if axis is not None:
                    cam = self.scene_widget.scene.camera
                    view = np.asarray(cam.get_view_matrix(), dtype=np.float64)
                    proj = np.asarray(cam.get_projection_matrix(), dtype=np.float64)
                    R = view[:3, :3]
                    t = view[:3, 3]
                    eye = -R.T @ t
                    up = R[1, :].astype(np.float64)   # row 1 of R = up direction in world
                    self.active_axis = axis
                    self.drag_initial_hit = hit.astype(np.float64)
                    self.drag_initial_eye = eye
                    self.drag_initial_up = up
                    self.drag_initial_view = view
                    self.drag_initial_proj = proj
                    frame = self.scene_widget.frame
                    self.drag_view_w = int(frame.width)
                    self.drag_view_h = int(frame.height)
                    print(f"  grabbed {self.AXIS_NAMES[axis]}-axis ring")
                    return gui.Widget.EventCallbackResult.CONSUMED

        elif et == gui.MouseEvent.Type.DRAG and self.active_axis is not None:
            origin, direction = self._ray_through_pixel(
                event.x, event.y,
                view_mat=self.drag_initial_view,
                proj_mat=self.drag_initial_proj,
                view_w=self.drag_view_w,
                view_h=self.drag_view_h,
            )
            if origin is None:
                return gui.Widget.EventCallbackResult.CONSUMED
            axis = self.AXIS_VECTORS[self.active_axis]
            t, hit = self._ray_axis_plane_hit(origin, direction, axis)
            if t is None:
                return gui.Widget.EventCallbackResult.CONSUMED
            v0 = self.drag_initial_hit - self.pivot
            v1 = hit - self.pivot
            # Project onto plane perp to axis (drop axis component)
            v0p = v0 - np.dot(v0, axis) * axis
            v1p = v1 - np.dot(v1, axis) * axis
            n0 = float(np.linalg.norm(v0p))
            n1 = float(np.linalg.norm(v1p))
            if n0 < 1e-6 or n1 < 1e-6:
                return gui.Widget.EventCallbackResult.CONSUMED
            v0p /= n0
            v1p /= n1
            sin_a = float(np.dot(np.cross(v0p, v1p), axis))
            cos_a = float(np.dot(v0p, v1p))
            angle = float(np.arctan2(sin_a, cos_a))
            # Rotate camera by -angle so the model appears to rotate by +angle
            Rcam = _rodrigues(axis, -angle)
            new_eye = self.pivot + Rcam @ (self.drag_initial_eye - self.pivot)
            new_up = Rcam @ self.drag_initial_up
            self.scene_widget.scene.camera.look_at(
                self.pivot.astype(np.float32),
                new_eye.astype(np.float32),
                new_up.astype(np.float32),
            )
            return gui.Widget.EventCallbackResult.CONSUMED

        elif et == gui.MouseEvent.Type.BUTTON_UP and self.active_axis is not None:
            print(f"  released {self.AXIS_NAMES[self.active_axis]}-axis ring")
            self.active_axis = None
            self.drag_initial_hit = None
            self.drag_initial_view = None
            return gui.Widget.EventCallbackResult.CONSUMED

        return gui.Widget.EventCallbackResult.IGNORED

    # ── Collision color helpers (C 키) ──────────────────────────────
    def _apply_collision_colors(self, pick_idx: int) -> int:
        """pick_idx의 sweep 부피 안 점들을 검정으로 칠하고 원본 색 백업.
        반환: 검정으로 바뀐 총 점 개수."""
        targets = self.pick_collision_targets.get(pick_idx, [])
        if not targets:
            self.collision_color_backup = []
            return 0
        backup = []
        n_total = 0
        for hl_k, hit_idx in targets:
            if hl_k >= len(self.highlights):
                continue
            h = self.highlights[hl_k]
            colors = np.asarray(h.colors, dtype=np.float64)
            if colors.size == 0:
                continue
            # 현재 색 스냅샷 → 백업 (heatmap 토글 등 상태 그대로 복구)
            cols_snap = colors[hit_idx].copy()
            backup.append((hl_k, hit_idx, cols_snap))
            colors[hit_idx] = 0.0   # black
            h.colors = o3d.utility.Vector3dVector(colors)
            self.scene_widget.scene.remove_geometry(f"hi_{hl_k}")
            self.scene_widget.scene.add_geometry(f"hi_{hl_k}", h, self.mat_pc)
            self.scene_widget.scene.show_geometry(
                f"hi_{hl_k}", self.highlights_visible)
            n_total += int(len(hit_idx))
        self.collision_color_backup = backup
        return n_total

    def _restore_collision_colors(self):
        """이전 _apply_collision_colors가 검정 칠한 점들을 원본 색으로 복구."""
        if not self.collision_color_backup:
            self.collision_color_backup = None
            return
        for hl_k, hit_idx, cols_snap in self.collision_color_backup:
            if hl_k >= len(self.highlights):
                continue
            h = self.highlights[hl_k]
            colors = np.asarray(h.colors, dtype=np.float64)
            if colors.size == 0:
                continue
            colors[hit_idx] = cols_snap
            h.colors = o3d.utility.Vector3dVector(colors)
            self.scene_widget.scene.remove_geometry(f"hi_{hl_k}")
            self.scene_widget.scene.add_geometry(f"hi_{hl_k}", h, self.mat_pc)
            self.scene_widget.scene.show_geometry(
                f"hi_{hl_k}", self.highlights_visible)
        self.collision_color_backup = None

    # ── Key callback ──
    def _on_key(self, event):
        if event.type != gui.KeyEvent.Type.DOWN:
            return gui.Widget.EventCallbackResult.IGNORED
        k = event.key

        # ── Manual pick 라벨 모드 키 ──────────────────────────────
        # V (검증) / Enter 두 번 (저장) / ESC (취소) / R (픽 리셋)
        # 저장은 안전장치로 Enter 2회 — 첫 번째는 검증만, 두 번째로 실제 저장
        if self.label_mode:
            if k == gui.KeyName.ESCAPE:
                if self.manual_pending_save:
                    # 저장 대기 중에 ESC → 저장만 취소 (라벨 모드는 유지)
                    self.manual_pending_save = False
                    print("  ⊘ save cancelled — still editing pick")
                else:
                    self._exit_label_mode(save=False)
                return gui.Widget.EventCallbackResult.CONSUMED
            if k == gui.KeyName.L:
                self._exit_label_mode(save=False)
                return gui.Widget.EventCallbackResult.CONSUMED
            if k == gui.KeyName.ENTER:
                if not self.manual_pick:
                    print("  (no pick to save — click on highlight first)")
                    return gui.Widget.EventCallbackResult.CONSUMED
                if not self.manual_pending_save:
                    # 첫 Enter — 검증 + 저장 대기 상태로 전환
                    self._validate_manual_pick()
                    self.manual_pending_save = True
                    print("\n  ━━━ Press Enter AGAIN to SAVE,"
                          " or ESC to cancel save ━━━")
                else:
                    # 두 번째 Enter — 실제 저장
                    self.manual_pending_save = False
                    self._exit_label_mode(save=True)
                return gui.Widget.EventCallbackResult.CONSUMED
            if k == gui.KeyName.V:
                self.manual_pending_save = False   # 검증 = 다른 행동 → pending 취소
                self._validate_manual_pick()
                return gui.Widget.EventCallbackResult.CONSUMED
            if k == gui.KeyName.R:
                # R = 픽 리셋 → 다음 클릭으로 새 위치 배치
                # 라벨 모드에서는 픽 없어도 카메라 reset으로 fallthrough 차단
                # (사용자가 의도치 않게 카메라가 reset되는 일 방지)
                if self.manual_pick is not None:
                    self.manual_pending_save = False
                    self._clear_manual_geometry()
                    self.manual_pick = None
                    print("  manual pick reset — click highlight to re-place")
                else:
                    print("  (no manual pick to reset — click highlight first)")
                return gui.Widget.EventCallbackResult.CONSUMED
            if k == gui.KeyName.B:
                # B = 현재 manual_pick pose 를 cap pose 로 저장.
                # cap_center = position, cap_axis = long_axis (두 cup 잇는 방향),
                # cap_normal = normal. 위치/방향 정렬은 기존 gizmo 로.
                self._save_manual_pick_as_cap()
                return gui.Widget.EventCallbackResult.CONSUMED
            # WASD + QE: 픽 좌표계 기준 이동 (mm 단위)
            #   W/S = normal 방향 (Z축: 표면에서 뜨기/파고들기)
            #   A/D = long_axis 방향 (X축: 좌우)
            #   Q/E = mid_axis 방향 (Y축: 옆)
            _MOVE_STEP = 2.0  # mm per key press
            if self.manual_pick is not None:
                if k == gui.KeyName.W:
                    self._translate_manual_pick(0, 0, _MOVE_STEP)
                    print(f"  ↑ Z+{_MOVE_STEP}mm (away from surface)")
                    return gui.Widget.EventCallbackResult.CONSUMED
                if k == gui.KeyName.S:
                    self._translate_manual_pick(0, 0, -_MOVE_STEP)
                    print(f"  ↓ Z-{_MOVE_STEP}mm (into surface)")
                    return gui.Widget.EventCallbackResult.CONSUMED
                if k == gui.KeyName.A:
                    self._translate_manual_pick(-_MOVE_STEP, 0, 0)
                    print(f"  ← X-{_MOVE_STEP}mm")
                    return gui.Widget.EventCallbackResult.CONSUMED
                if k == gui.KeyName.D:
                    self._translate_manual_pick(_MOVE_STEP, 0, 0)
                    print(f"  → X+{_MOVE_STEP}mm")
                    return gui.Widget.EventCallbackResult.CONSUMED
                if k == gui.KeyName.Q:
                    self._translate_manual_pick(0, -_MOVE_STEP, 0)
                    print(f"  Y-{_MOVE_STEP}mm")
                    return gui.Widget.EventCallbackResult.CONSUMED
                if k == gui.KeyName.E:
                    self._translate_manual_pick(0, _MOVE_STEP, 0)
                    print(f"  Y+{_MOVE_STEP}mm")
                    return gui.Widget.EventCallbackResult.CONSUMED
            # 그 외 키는 일반 모드처럼 처리 (camera control 등)

        # ── 일반 모드 ─────────────────────────────────────────────────
        if k == gui.KeyName.L:
            self._toggle_label_mode()
            return gui.Widget.EventCallbackResult.CONSUMED
        if k == gui.KeyName.ESCAPE or k == gui.KeyName.Q:
            self.window.close()
            return gui.Widget.EventCallbackResult.CONSUMED
        if k == gui.KeyName.R:
            self._set_pivot(self.scene_center)
            cam_dist = self.scene_extent * 0.9
            eye = self.scene_center + self._look_dir_default * cam_dist
            self.scene_widget.scene.camera.look_at(
                self.scene_center.astype(np.float32),
                eye.astype(np.float32),
                self._up_default.astype(np.float32),
            )
            print("  reset view; pivot → scene center")
            return gui.Widget.EventCallbackResult.CONSUMED
        if k == gui.KeyName.T:
            self.highlights_visible = not self.highlights_visible
            for i in range(len(self.highlights)):
                self.scene_widget.scene.show_geometry(
                    f"hi_{i}", self.highlights_visible,
                )
            print(f"  highlights {'ON' if self.highlights_visible else 'OFF'}")
            return gui.Widget.EventCallbackResult.CONSUMED
        if k == gui.KeyName.M:
            # 복원 메시 (반투명 표면) 토글
            self.recovery_visible = not self.recovery_visible
            for name in self.recovery_mesh_names:
                self.scene_widget.scene.show_geometry(name, self.recovery_visible)
            print(f"  📐 recovery 메시 {'ON' if self.recovery_visible else 'OFF'}")
            return gui.Widget.EventCallbackResult.CONSUMED
        if k == gui.KeyName.G:
            # Picks 표시 토글 — 초기 OFF, G 키로만 ON. 한 번에 묶이는 것:
            #   1) pick_geom_names      — 사각형 / 컵 / Z arrow / bottle viz (show_geometry)
            #   2) cap_inlier_geom_names — bottle cap RANSAC inlier point cloud
            #   3) pick_labels          — Label3D 객체 (show_geometry 안 먹음, add/remove)
            #   4) pick_box_geom_names  — collision box (OFF 시만 강제 숨김)
            if self.picks_data is None:
                print("  (no picks data — launch with --picks)")
                return gui.Widget.EventCallbackResult.CONSUMED
            if not self.picks_added:
                # 첫 G — geom + label 한 번에 생성
                self._add_pick_geometries()
                self.picks_added = True
                self.picks_visible = True
                # cap_inlier 같이 켜기
                for name in self.cap_inlier_geom_names:
                    self.scene_widget.scene.show_geometry(name, True)
                print(f"  🎯 picks ON ({len(self.pick_geom_names)} geoms, "
                      f"{len(self.pick_labels)} labels)")
            else:
                self.picks_visible = not self.picks_visible
                # 1) geom show/hide
                for name in self.pick_geom_names:
                    self.scene_widget.scene.show_geometry(name, self.picks_visible)
                # 2) cap_inlier show/hide
                for name in self.cap_inlier_geom_names:
                    self.scene_widget.scene.show_geometry(name, self.picks_visible)
                # 3) labels — show_geometry 안 먹어서 remove/re-add
                if self.picks_visible:
                    # 다시 켜기: label list 비었으면 (이전 OFF 에서 제거됨) 재생성.
                    # _add_pick_labels 가 없으면 전체 _add_pick_geometries 재호출은
                    # geom 중복 추가로 에러 → 별도 함수가 없으니 OFF 시점에 labels 만 비웠다는 가정.
                    if not self.pick_labels:
                        self._rebuild_pick_labels()
                else:
                    # 끄기: 라벨 제거
                    for lbl in self.pick_labels:
                        try:
                            self.scene_widget.remove_3d_label(lbl)
                        except Exception:
                            pass
                    self.pick_labels = []
                # 4) collision box / colors — OFF 시 함께 정리 (혼란 방지)
                if not self.picks_visible:
                    for names in self.pick_box_geom_names.values():
                        for name in names:
                            self.scene_widget.scene.show_geometry(name, False)
                    if self.collision_viz_pick_idx is not None:
                        self._restore_collision_colors()
                        self.collision_viz_pick_idx = None
                print(f"  🎯 picks {'ON' if self.picks_visible else 'OFF'}")
            return gui.Widget.EventCallbackResult.CONSUMED
        if k == gui.KeyName.C:
            # 화면 중앙에 가장 가까운 픽만 collision box 표시.
            # 카메라를 대상 객체 위로 이동 후 C → 그 픽의 arm + object sweep
            # 박스만 표시. 같은 픽이 이미 표시 중이면 hide (toggle off).
            # 단, picks 자체가 OFF(G 안 누른 상태)면 collision box 만 띄우는 건
            # 사각형/컵 없이 박스만 떠서 혼란 → picks ON 상태에서만 동작.
            if not self.picks_visible:
                print("  (press G first to show picks before C)")
                return gui.Widget.EventCallbackResult.CONSUMED
            if not self.pick_box_geom_names:
                print("  (no pick boxes — load with --picks)")
                return gui.Widget.EventCallbackResult.CONSUMED
            cam = self.scene_widget.scene.camera
            view = np.asarray(cam.get_view_matrix(), dtype=np.float64)
            best_idx, best_d2 = None, float("inf")
            for pidx, pos in self.pick_box_centers.items():
                p = view @ np.array([pos[0], pos[1], pos[2], 1.0], dtype=np.float64)
                if p[2] >= 0:
                    continue   # behind camera
                ax = p[0] / -p[2]
                ay = p[1] / -p[2]
                d2 = ax * ax + ay * ay
                if d2 < best_d2:
                    best_d2 = d2
                    best_idx = pidx
            if best_idx is None:
                print("  (no picks in front of camera)")
                return gui.Widget.EventCallbackResult.CONSUMED
            # 토글 로직: 같은 픽이 이미 보이면 hide, 아니면 다른 거 다 hide + 새 픽 show.
            if self.collision_viz_pick_idx == best_idx:
                for name in self.pick_box_geom_names[best_idx]:
                    self.scene_widget.scene.show_geometry(name, False)
                self._restore_collision_colors()
                self.collision_viz_pick_idx = None
                print(f"  🦾 collision box (pick #{best_idx}) OFF")
            else:
                # 이전 픽 hide + 색 복구
                if self.collision_viz_pick_idx is not None:
                    prev = self.pick_box_geom_names.get(
                        self.collision_viz_pick_idx, [])
                    for name in prev:
                        self.scene_widget.scene.show_geometry(name, False)
                    self._restore_collision_colors()
                for name in self.pick_box_geom_names[best_idx]:
                    self.scene_widget.scene.show_geometry(name, True)
                # 새 픽의 충돌 점 검정 칠하기
                n_hit_total = self._apply_collision_colors(best_idx)
                self.collision_viz_pick_idx = best_idx
                picks_list = self.picks_data.get("picks", [])
                if 0 <= best_idx < len(picks_list):
                    rec = picks_list[best_idx]
                    order = rec.get("pick_order")
                    tier = rec.get("tier", "?")
                    prot = rec.get("arm_max_protrusion_mm", 0.0)
                    print(f"  🦾 collision box ON — pick #{best_idx} "
                          f"(order={order}, tier={tier}, "
                          f"arm_protrusion={prot:.1f}mm, "
                          f"collision_pts={n_hit_total})")
                else:
                    print(f"  🦾 collision box ON — pick #{best_idx} "
                          f"(collision_pts={n_hit_total})")
            return gui.Widget.EventCallbackResult.CONSUMED
        if k == gui.KeyName.K:
            # Toggle 색상 모드: class color vs pickability heatmap
            if not self.heatmap_colors_list or all(
                c is None for c in self.heatmap_colors_list
            ):
                print("  (heatmap not computed — restart with --heatmap)")
                return gui.Widget.EventCallbackResult.CONSUMED
            self.heatmap_active = not self.heatmap_active
            for i, h in enumerate(self.highlights):
                src_colors = (
                    self.heatmap_colors_list[i] if self.heatmap_active
                    else self.class_colors_list[i]
                )
                if src_colors is None:
                    continue
                h.colors = o3d.utility.Vector3dVector(src_colors)
                # Open3D scene: 변경된 색을 반영하려면 geometry 재추가
                self.scene_widget.scene.remove_geometry(f"hi_{i}")
                self.scene_widget.scene.add_geometry(f"hi_{i}", h, self.mat_pc)
                self.scene_widget.scene.show_geometry(
                    f"hi_{i}", self.highlights_visible,
                )
            mode_name = "heatmap (R/Y/G)" if self.heatmap_active else "class colors"
            print(f"  highlight color → {mode_name}")
            return gui.Widget.EventCallbackResult.CONSUMED
        if k == gui.KeyName.P:
            if not self.centroids:
                print("  (no labeled objects to pivot to)")
                return gui.Widget.EventCallbackResult.CONSUMED
            cam = self.scene_widget.scene.camera
            view = np.asarray(cam.get_view_matrix(), dtype=np.float64)
            best_i, best_d2 = -1, float("inf")
            for i, c3d in enumerate(self.centroids):
                p = view @ np.array([c3d[0], c3d[1], c3d[2], 1.0], dtype=np.float64)
                # OpenGL: camera looks down -Z, so in front means p[2] < 0
                if p[2] >= 0:
                    continue
                ax = p[0] / -p[2]
                ay = p[1] / -p[2]
                d2 = ax * ax + ay * ay
                if d2 < best_d2:
                    best_d2 = d2
                    best_i = i
            if best_i < 0:
                print("  (all labeled objects are behind the camera)")
                return gui.Widget.EventCallbackResult.CONSUMED
            c3d = self.centroids[best_i]
            cid = self.cids[best_i]
            cname = CLASS_NAMES[cid] if 0 <= cid < len(CLASS_NAMES) else "?"
            self._set_pivot(c3d)
            print(f"  pivot → object #{best_i} ({cname}) at "
                  f"({c3d[0]:.1f}, {c3d[1]:.1f}, {c3d[2]:.1f})")
            return gui.Widget.EventCallbackResult.CONSUMED
        return gui.Widget.EventCallbackResult.IGNORED

    def run(self):
        self.app.run()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply", type=str, help="path to organized PLY")
    ap.add_argument("--polygons", type=str, default=None,
                    help="path to data.json with polygons + class_id (highlights)")
    ap.add_argument("--picks", type=str, default=None,
                    help="path to <shot>_picks.json (dual-suction rectangles)")
    ap.add_argument("--no-highlights", action="store_true",
                    help="show point cloud only (no polygon highlights)")
    ap.add_argument("--blend", type=float, default=0.75,
                    help="class color blend strength 0..1 (default 0.75)")
    ap.add_argument("--dim-bg", action="store_true",
                    help="dim background (non-labeled) points to make highlights pop")
    # heatmap default: ON. Pass --no-heatmap to disable.
    ap.add_argument("--heatmap", action="store_true", default=True,
                    help="Compute pickability heatmap (default: on)")
    ap.add_argument("--no-heatmap", dest="heatmap", action="store_false",
                    help="Disable heatmap computation")
    ap.add_argument("--left", type=int, default=None,
                    help="window left position (default: screen right half)")
    ap.add_argument("--top", type=int, default=60)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    args = ap.parse_args()

    ply_path = Path(args.ply)
    if not ply_path.exists():
        print(f"!! PLY not found: {ply_path}", file=sys.stderr)
        sys.exit(1)
    print(f"Loading {ply_path.name}...")

    pdata = None
    if args.polygons and not args.no_highlights:
        pp = Path(args.polygons)
        if pp.exists():
            with open(pp, encoding="utf-8") as f:
                pdata = json.load(f)
        else:
            print(f"  (polygons file not found: {pp})")

    picks_data = None
    if args.picks:
        pk = Path(args.picks)
        if pk.exists():
            with open(pk, encoding="utf-8") as f:
                picks_data = json.load(f)
            n_ok = picks_data.get("n_ok", 0)
            print(f"  Picks: {n_ok} successful from {pk.name}")
        else:
            print(f"  (picks file not found: {pk})")

    sw, sh = _get_screen_size_windows()
    width = args.width if args.width else max(900, int(sw * 0.45))
    height = args.height if args.height else max(700, int(sh * 0.85))
    left = args.left if args.left is not None else (sw - width - 30)
    top = args.top

    viewer = Viewer3D(
        ply_path, polygons_data=pdata, blend=args.blend, dim_bg=args.dim_bg,
        picks_data=picks_data, heatmap_mode=args.heatmap,
        win_w=width, win_h=height, win_left=left, win_top=top,
    )
    viewer.run()


if __name__ == "__main__":
    main()
