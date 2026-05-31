from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def largest_component_mask(mask: np.ndarray, empty_as_none: bool = False) -> np.ndarray | None:
    binary = (mask > 0).astype(np.uint8)
    if not np.any(binary):
        return None if empty_as_none else binary
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if component_count <= 1:
        return None if empty_as_none else binary
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest_label


def mask_center_point(mask: np.ndarray) -> tuple[int, int]:
    coords_yx = np.column_stack(np.where(mask > 0))
    if coords_yx.shape[0] == 0:
        return 0, 0

    center_xy = coords_yx[:, ::-1].astype(np.float64).mean(axis=0)
    center_u = int(round(center_xy[0]))
    center_v = int(round(center_xy[1]))
    if (
        0 <= center_v < mask.shape[0]
        and 0 <= center_u < mask.shape[1]
        and mask[center_v, center_u] > 0
    ):
        return center_u, center_v

    coords_xy = coords_yx[:, ::-1].astype(np.float64)
    distances = np.sum((coords_xy - center_xy) ** 2, axis=1)
    nearest = coords_xy[int(np.argmin(distances))]
    return int(round(nearest[0])), int(round(nearest[1]))


def mask_to_polygon(
    mask: np.ndarray,
    polygon_approx_ratio: float,
    polygon_min_epsilon: float,
) -> list[list[int]]:
    binary_mask = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    epsilon = max(float(polygon_min_epsilon), perimeter * float(polygon_approx_ratio))
    polygon = cv2.approxPolyDP(contour, epsilon, True)
    if polygon.shape[0] < 3:
        x, y, w, h = cv2.boundingRect(contour)
        polygon = np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]], dtype=np.int32)
    return polygon.reshape(-1, 2).astype(int).tolist()


def erode_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return mask
    kernel = np.ones((max(1, kernel_size), max(1, kernel_size)), dtype=np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return eroded if np.any(eroded) else mask


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool)
    return mask & ~eroded
