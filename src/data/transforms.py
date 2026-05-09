from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def rotation_matrix(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.asarray([[c, -s], [s, c]], dtype=np.float32)


def world_to_local_points(points_xy: np.ndarray, origin_xy: np.ndarray, origin_heading: float) -> np.ndarray:
    rotation = rotation_matrix(-origin_heading)
    flat = points_xy.reshape(-1, 2) - origin_xy[None, :]
    return (flat @ rotation.T).reshape(points_xy.shape)


def local_to_world_points(points_xy: np.ndarray, origin_xy: np.ndarray, origin_heading: float) -> np.ndarray:
    rotation = rotation_matrix(origin_heading)
    flat = points_xy.reshape(-1, 2) @ rotation.T + origin_xy[None, :]
    return (flat).reshape(points_xy.shape)


def transform_states_to_local(states: np.ndarray, origin_xy: np.ndarray, origin_heading: float) -> np.ndarray:
    local = states.copy()
    local[..., 0:2] = world_to_local_points(states[..., 0:2], origin_xy, origin_heading)
    local[..., 2:4] = world_to_local_points(states[..., 2:4], np.zeros(2, dtype=np.float32), origin_heading)
    local[..., 4] = wrap_angle(states[..., 4] - origin_heading)
    return local


def transform_map_to_local(polyline: np.ndarray, origin_xy: np.ndarray, origin_heading: float) -> np.ndarray:
    local = polyline.copy()
    local[..., 0:2] = world_to_local_points(polyline[..., 0:2], origin_xy, origin_heading)
    if polyline.shape[-1] > 2:
        local[..., 2] = wrap_angle(polyline[..., 2] - origin_heading)
    return local


def resample_polyline(points: np.ndarray, target_points: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return np.zeros((target_points, points.shape[-1] if points.ndim == 2 else 2), dtype=np.float32), np.zeros(
            target_points, dtype=bool
        )
    if len(points) == 1:
        tiled = np.repeat(points.astype(np.float32), target_points, axis=0)
        mask = np.zeros(target_points, dtype=bool)
        mask[0] = True
        return tiled, mask
    deltas = np.linalg.norm(np.diff(points[:, 0:2], axis=0), axis=-1)
    arc = np.concatenate([[0.0], np.cumsum(deltas)])
    total = arc[-1]
    if total < 1e-6:
        tiled = np.repeat(points[:1].astype(np.float32), target_points, axis=0)
        mask = np.zeros(target_points, dtype=bool)
        mask[0] = True
        return tiled, mask
    query = np.linspace(0.0, total, target_points)
    resampled = []
    for dim in range(points.shape[-1]):
        resampled.append(np.interp(query, arc, points[:, dim]))
    mask = np.ones(target_points, dtype=bool)
    return np.stack(resampled, axis=-1).astype(np.float32), mask
