from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from sklearn.cluster import KMeans

from src.utils.normal import normalize_normal_image
from src.utils.normal_surface import surface_center_from_method


def kmeans_normal_surface_candidates(
    normal_image: np.ndarray,
    mask: np.ndarray,
    min_area_ratio: float,
    min_area_px: int,
    open_kernel_px: int,
    fill_holes_max_area_px: int,
    fill_holes_max_aspect_ratio: float,
    center_method: str,
    rect_max_area_ratio: float,
    k: int = 3,
    merge_angle_deg: float = 3.0,
    n_init: int = 10,
    max_iter: int = 100,
    random_state: int = 0,
    max_candidates: int = 5,
) -> list[tuple[np.ndarray | None, dict[str, Any]]]:
    normals, valid = normalize_normal_image(normal_image)
    mask_bool = mask > 0
    height, width = mask_bool.shape[:2]
    normal_map = normals[:height, :width]
    comparable = mask_bool & valid[:height, :width]
    object_area = max(int(np.count_nonzero(mask_bool)), 1)
    if not np.any(comparable):
        return [
            (
                None,
                {
                    "passed": False,
                    "reason": "no_valid_normal_in_mask",
                    "normal_surface_mode": "kmeans",
                    "object_area": object_area,
                    "kmeans_k": int(k),
                },
            )
        ]

    ys, xs = np.where(comparable)
    features = normal_map[ys, xs].astype(np.float32)
    cluster_count = min(max(1, int(k)), int(features.shape[0]))
    if cluster_count <= 0:
        return [
            (
                None,
                {
                    "passed": False,
                    "reason": "empty_kmeans_features",
                    "normal_surface_mode": "kmeans",
                    "object_area": object_area,
                    "kmeans_k": int(k),
                },
            )
        ]

    kmeans = KMeans(
        n_clusters=cluster_count,
        init="k-means++",
        n_init=max(1, int(n_init)),
        max_iter=max(1, int(max_iter)),
        random_state=int(random_state),
    )
    labels = kmeans.fit_predict(
        features,
    )
    labels = labels.reshape(-1)
    centers = np.asarray(kmeans.cluster_centers_, dtype=np.float64)
    inertia = float(kmeans.inertia_)
    iterations = int(kmeans.n_iter_)
    merged_labels, merged_centers, merge_groups = _merge_similar_clusters(
        labels,
        centers,
        float(merge_angle_deg),
    )

    attempts: list[tuple[np.ndarray | None, dict[str, Any]]] = []
    max_candidate_count = max(1, int(max_candidates))
    cluster_ids = list(range(len(merge_groups)))
    cluster_ids.sort(key=lambda cluster_id: int(np.count_nonzero(merged_labels == cluster_id)), reverse=True)

    for cluster_rank, cluster_id in enumerate(cluster_ids):
        if len(attempts) >= max_candidate_count:
            break
        cluster_mask = np.zeros(mask_bool.shape[:2], dtype=bool)
        selected = merged_labels == cluster_id
        cluster_mask[ys[selected], xs[selected]] = True
        raw_pixels = int(np.count_nonzero(cluster_mask))
        if raw_pixels <= 0:
            continue

        cleaned, cleanup_debug = _cleanup(
            cluster_mask,
            open_kernel_px,
            fill_holes_max_area_px,
            fill_holes_max_aspect_ratio,
        )
        if not np.any(cleaned):
            attempts.append(
                (
                    None,
                    {
                        "passed": False,
                        "reason": "empty_cluster_after_cleanup",
                        "normal_surface_mode": "kmeans",
                        "kmeans_k": int(cluster_count),
                        "kmeans_cluster_index": int(cluster_id),
                        "kmeans_cluster_rank": int(cluster_rank),
                        "kmeans_original_cluster_indices": [int(value) for value in merge_groups[cluster_id]],
                        "kmeans_merge_angle_deg": float(merge_angle_deg),
                        "kmeans_merged_cluster_count": int(len(merge_groups)),
                        "kmeans_inertia": inertia,
                        "kmeans_n_init": int(n_init),
                        "kmeans_max_iter": int(max_iter),
                        "kmeans_iterations": iterations,
                        "surface_raw_pixels": raw_pixels,
                        "object_area": object_area,
                        "seed_normal": _normal_list(merged_centers[cluster_id]),
                        **cleanup_debug,
                    },
                )
            )
            continue

        component_count, component_labels, stats, _ = cv2.connectedComponentsWithStats(cleaned.astype(np.uint8), connectivity=8)
        component_ids = list(range(1, component_count))
        component_ids.sort(key=lambda label: int(stats[label, cv2.CC_STAT_AREA]), reverse=True)
        for component_rank, component_id in enumerate(component_ids):
            if len(attempts) >= max_candidate_count:
                break
            surface_area = int(stats[component_id, cv2.CC_STAT_AREA])
            if surface_area <= 0:
                continue
            area_ratio = float(surface_area / object_area)
            base_debug: dict[str, Any] = {
                "normal_surface_mode": "kmeans",
                "kmeans_k": int(cluster_count),
                "kmeans_cluster_index": int(cluster_id),
                "kmeans_cluster_rank": int(cluster_rank),
                "kmeans_original_cluster_indices": [int(value) for value in merge_groups[cluster_id]],
                "kmeans_merge_angle_deg": float(merge_angle_deg),
                "kmeans_merged_cluster_count": int(len(merge_groups)),
                "kmeans_inertia": inertia,
                "kmeans_n_init": int(n_init),
                "kmeans_max_iter": int(max_iter),
                "kmeans_iterations": iterations,
                "component_index": int(component_id),
                "component_rank": int(component_rank),
                "surface_area": surface_area,
                "object_area": object_area,
                "surface_area_ratio": area_ratio,
                "min_surface_region_area_ratio": float(min_area_ratio),
                "min_surface_region_area_px": int(min_area_px),
                "seed_normal": _normal_list(merged_centers[cluster_id]),
                "surface_raw_pixels": raw_pixels,
                "surface_open_kernel_px": int(open_kernel_px),
                **cleanup_debug,
            }
            passed = area_ratio >= float(min_area_ratio) and surface_area >= int(min_area_px)
            if not passed:
                reason = "surface_area_px_too_small" if surface_area < int(min_area_px) else "surface_too_small"
                attempts.append((None, {"passed": False, "reason": reason, **base_debug}))
                break

            surface = component_labels == component_id
            center_u, center_v, center_debug = surface_center_from_method(
                surface,
                mask_bool,
                center_method,
                rect_max_area_ratio,
            )
            attempts.append(
                (
                    surface,
                    {
                        "passed": True,
                        "reason": None,
                        "surface_center_xy": [int(center_u), int(center_v)],
                        "seed_xy": [int(center_u), int(center_v)],
                        "component_seed_xy": [int(center_u), int(center_v)],
                        "surface_center_method": str(center_method),
                        "surface_center_used_method": center_debug["used_method"],
                        "surface_center_fallback_reason": center_debug.get("fallback_reason"),
                        "surface_rect_area": center_debug.get("rect_area"),
                        "surface_rect_area_ratio": center_debug.get("rect_area_ratio"),
                        "surface_rect_box_xy": center_debug.get("rect_box_xy"),
                        "surface_rect_center_xy": center_debug.get("rect_center_xy"),
                        "surface_rect_max_area_ratio": float(rect_max_area_ratio),
                        **base_debug,
                    },
                )
            )

    if attempts:
        return attempts
    return [
        (
            None,
            {
                "passed": False,
                "reason": "no_kmeans_surface",
                "normal_surface_mode": "kmeans",
                "object_area": object_area,
                "kmeans_k": int(cluster_count),
                "kmeans_merge_angle_deg": float(merge_angle_deg),
                "kmeans_merged_cluster_count": int(len(merge_groups)),
            },
        )
    ]


def _merge_similar_clusters(
    labels: np.ndarray,
    centers: np.ndarray,
    merge_angle_deg: float,
) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    center_count = int(centers.shape[0])
    if center_count <= 1 or merge_angle_deg <= 0:
        normalized = np.asarray([_normal_array(center) for center in centers], dtype=np.float64)
        return labels.astype(np.int32), normalized, [[index] for index in range(center_count)]

    parent = list(range(center_count))
    normalized_centers = np.asarray([_normal_array(center) for center in centers], dtype=np.float64)
    cos_threshold = float(np.cos(np.deg2rad(float(merge_angle_deg))))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left in range(center_count):
        for right in range(left + 1, center_count):
            dot = float(np.dot(normalized_centers[left], normalized_centers[right]))
            if dot >= cos_threshold:
                union(left, right)

    roots = [find(index) for index in range(center_count)]
    unique_roots = sorted(set(roots), key=lambda root: int(np.count_nonzero(np.isin(labels, [i for i, value in enumerate(roots) if value == root]))), reverse=True)
    root_to_group = {root: group_index for group_index, root in enumerate(unique_roots)}
    label_to_group = np.asarray([root_to_group[root] for root in roots], dtype=np.int32)
    merged_labels = label_to_group[labels]

    merge_groups: list[list[int]] = []
    merged_centers: list[np.ndarray] = []
    for root in unique_roots:
        group = [index for index, value in enumerate(roots) if value == root]
        merge_groups.append(group)
        weights = np.asarray([np.count_nonzero(labels == index) for index in group], dtype=np.float64)
        values = normalized_centers[group]
        merged_center = np.average(values, axis=0, weights=weights) if np.any(weights > 0) else values.mean(axis=0)
        merged_centers.append(_normal_array(merged_center))

    return merged_labels.astype(np.int32), np.asarray(merged_centers, dtype=np.float64), merge_groups


def _normal_array(normal: np.ndarray) -> np.ndarray:
    values = np.asarray(normal, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(values))
    if norm <= 1e-9:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    return values / norm


def _normal_list(normal: np.ndarray) -> list[float]:
    values = np.asarray(normal, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(values))
    if norm > 1e-9:
        values = values / norm
    return [float(value) for value in values]


def _cleanup(
    surface: np.ndarray,
    open_kernel_px: int,
    fill_holes_max_area_px: int,
    fill_holes_max_aspect_ratio: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    cleaned = _morph(surface.astype(np.uint8), cv2.MORPH_OPEN, int(open_kernel_px)) > 0
    filled, fill_debug = _fill_small_enclosed_holes(
        cleaned,
        int(fill_holes_max_area_px),
        float(fill_holes_max_aspect_ratio),
    )
    return filled, {
        "surface_close_kernel_used": False,
        "surface_hole_fill_enabled": int(fill_holes_max_area_px) > 0,
        **fill_debug,
    }


def _morph(mask: np.ndarray, operation: int, kernel_px: int) -> np.ndarray:
    kernel_size = int(kernel_px)
    if kernel_size <= 1:
        return mask
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.morphologyEx(mask.astype(np.uint8), operation, kernel)


def _fill_small_enclosed_holes(
    surface: np.ndarray,
    max_area_px: int,
    max_aspect_ratio: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if max_area_px <= 0:
        return surface > 0, {"surface_holes_filled": 0, "surface_holes_filled_area": 0}

    background = ~(surface > 0)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(background.astype(np.uint8), connectivity=8)
    filled = surface.copy() > 0
    filled_count = 0
    filled_area = 0
    height, width = surface.shape[:2]
    max_aspect = max(float(max_aspect_ratio), 1.0)

    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0 or area > int(max_area_px):
            continue
        if x <= 0 or y <= 0 or x + w >= width or y + h >= height:
            continue
        aspect = float(max(w, h) / max(min(w, h), 1))
        if aspect >= max_aspect:
            continue
        filled[labels == label] = True
        filled_count += 1
        filled_area += area

    return filled, {
        "surface_holes_filled": int(filled_count),
        "surface_holes_filled_area": int(filled_area),
        "surface_fill_holes_max_area_px": int(max_area_px),
        "surface_fill_holes_max_aspect_ratio": float(max_aspect),
    }
