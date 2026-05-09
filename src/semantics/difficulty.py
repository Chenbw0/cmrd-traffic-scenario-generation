from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from isgen import save_json
from isgen.semantics.kinematics import compute_trajectory_features, masked_mean
from isgen.semantics.normalization import empirical_cdf_transform, empirical_cdf_values, robust_scale, robust_stats

LOGGER = logging.getLogger(__name__)
_FALLBACK_WARNING_EMITTED = False
DIFFICULTY_CODE_VERSION = "2026-04-19-difficulty-debug-v4"
EXPECTED_DIFFICULTY_FEATURES = [
    "min_pairwise_distance_future",
    "min_distance_to_focal_future",
    "ttc_proxy",
    "conflict_count",
    "interaction_density",
    "required_deceleration_proxy",
    "relative_speed_mean",
    "relative_speed_max",
    "mean_speed",
    "max_speed",
    "mean_abs_accel",
    "p95_abs_accel",
    "mean_abs_jerk",
    "mean_abs_yaw_rate",
    "curvature",
    "num_agents",
    "map_polyline_density_near_focal",
    "route_curvature_proxy",
]
INVERTED_DIFFICULTY_FEATURES = {"min_pairwise_distance_future", "min_distance_to_focal_future", "ttc_proxy"}


def _safe_feature_value(value: Any, default: float = 0.0) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(scalar):
        return float(default)
    return scalar


def _sanitize_difficulty_features(features: Dict[str, Any]) -> Dict[str, float]:
    return {key: _safe_feature_value(features.get(key, 0.0), 0.0) for key in EXPECTED_DIFFICULTY_FEATURES}


def _difficulty_feature_matrix(features_list_or_matrix: Any) -> tuple[np.ndarray, int]:
    if isinstance(features_list_or_matrix, np.ndarray):
        matrix = np.asarray(features_list_or_matrix, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != len(EXPECTED_DIFFICULTY_FEATURES):
            raise ValueError(
                f"Difficulty feature matrix must have shape [N, {len(EXPECTED_DIFFICULTY_FEATURES)}], got {tuple(matrix.shape)}."
            )
        missing_feature_count = int(np.sum(~np.isfinite(matrix)))
        matrix = np.where(np.isfinite(matrix), matrix, 0.0)
        return matrix.astype(np.float32, copy=False), missing_feature_count
    if isinstance(features_list_or_matrix, dict):
        matrix = np.stack(
            [
                np.asarray(features_list_or_matrix.get(key, []), dtype=np.float32)
                for key in EXPECTED_DIFFICULTY_FEATURES
            ],
            axis=1,
        )
        missing_feature_count = int(np.sum(~np.isfinite(matrix)))
        matrix = np.where(np.isfinite(matrix), matrix, 0.0)
        return matrix.astype(np.float32, copy=False), missing_feature_count
    features_list = list(features_list_or_matrix)
    if not features_list:
        return np.zeros((0, len(EXPECTED_DIFFICULTY_FEATURES)), dtype=np.float32), 0
    if isinstance(features_list[0], dict):
        matrix = np.asarray(
            [
                [_safe_feature_value(feature_row.get(key, 0.0), 0.0) for key in EXPECTED_DIFFICULTY_FEATURES]
                for feature_row in features_list
            ],
            dtype=np.float32,
        )
    else:
        matrix = np.asarray(features_list, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != len(EXPECTED_DIFFICULTY_FEATURES):
            raise ValueError(
                f"Difficulty feature matrix must have shape [N, {len(EXPECTED_DIFFICULTY_FEATURES)}], got {tuple(matrix.shape)}."
            )
    missing_feature_count = int(np.sum(~np.isfinite(matrix)))
    matrix = np.where(np.isfinite(matrix), matrix, 0.0)
    return matrix.astype(np.float32, copy=False), missing_feature_count


def _to_tensor(array: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(array, torch.Tensor):
        return array.float()
    return torch.as_tensor(array, dtype=torch.float32)


def _wrap_angle_torch(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _masked_softmin(values: torch.Tensor, mask: torch.Tensor, dim: int | tuple[int, ...], temperature: float, fallback: float) -> torch.Tensor:
    if isinstance(dim, int):
        dim = (dim,)
    values = torch.nan_to_num(values.float(), nan=0.0, posinf=1e3, neginf=-1e3)
    reduce_dims = tuple(sorted({(d if d >= 0 else values.ndim + d) for d in dim}))
    mask = mask.bool()
    if not reduce_dims:
        return values
    permute_order = [axis for axis in range(values.ndim) if axis not in reduce_dims] + list(reduce_dims)
    values_perm = values.permute(*permute_order)
    mask_perm = mask.permute(*permute_order)
    kept_shape = values_perm.shape[: values.ndim - len(reduce_dims)]
    values_flat = values_perm.reshape(*kept_shape, -1)
    mask_flat = mask_perm.reshape(*kept_shape, -1)
    masked_logits = torch.where(mask_flat, -values_flat / max(temperature, 1e-4), torch.full_like(values_flat, -1e9))
    any_valid = mask_flat.any(dim=-1, keepdim=True)
    safe_logits = torch.where(any_valid, masked_logits, torch.zeros_like(masked_logits))
    weights = torch.softmax(safe_logits, dim=-1) * mask_flat.float()
    weights = weights / torch.clamp(weights.sum(dim=-1, keepdim=True), min=1e-6)
    weighted = (weights * torch.where(mask_flat, values_flat, torch.zeros_like(values_flat))).sum(dim=-1)
    return torch.where(any_valid.squeeze(-1), weighted, torch.full_like(weighted, float(fallback)))


def _masked_softmax_value(values: torch.Tensor, mask: torch.Tensor, dim: int | tuple[int, ...], temperature: float, fallback: float) -> torch.Tensor:
    if isinstance(dim, int):
        dim = (dim,)
    values = torch.nan_to_num(values.float(), nan=0.0, posinf=1e3, neginf=-1e3)
    reduce_dims = tuple(sorted({(d if d >= 0 else values.ndim + d) for d in dim}))
    if not reduce_dims:
        return values
    permute_order = [axis for axis in range(values.ndim) if axis not in reduce_dims] + list(reduce_dims)
    values_perm = values.permute(*permute_order)
    mask_perm = mask.permute(*permute_order)
    kept_shape = values_perm.shape[: values.ndim - len(reduce_dims)]
    values_flat = values_perm.reshape(*kept_shape, -1)
    mask_flat = mask_perm.reshape(*kept_shape, -1)
    masked_logits = torch.where(mask_flat, values_flat / max(temperature, 1e-4), torch.full_like(values_flat, -1e9))
    any_valid = mask_flat.any(dim=-1, keepdim=True)
    safe_logits = torch.where(any_valid, masked_logits, torch.zeros_like(masked_logits))
    weights = torch.softmax(safe_logits, dim=-1) * mask_flat.float()
    weights = weights / torch.clamp(weights.sum(dim=-1, keepdim=True), min=1e-6)
    weighted = (weights * torch.where(mask_flat, values_flat, torch.zeros_like(values_flat))).sum(dim=-1)
    result = torch.where(any_valid.squeeze(-1), weighted, torch.full_like(weighted, float(fallback)))
    return result


def _masked_mean_tensor(values: torch.Tensor, mask: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
    if isinstance(dim, int):
        dim = (dim,)
    values = torch.nan_to_num(values.float(), nan=0.0, posinf=1e3, neginf=-1e3)
    masked_values = torch.where(mask, values, torch.zeros_like(values))
    mask_f = mask.float()
    numerator = masked_values.sum(dim=dim)
    denominator = torch.clamp(mask_f.sum(dim=dim), min=1.0)
    return numerator / denominator


def _safe_corrcoef(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.float()
    y = y.float()
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = torch.clamp(x_centered.square().sum().sqrt() * y_centered.square().sum().sqrt(), min=1e-6)
    return (x_centered * y_centered).sum() / denom


def _soft_empirical_cdf(raw_score: torch.Tensor, sorted_values: List[float], config: Dict) -> torch.Tensor:
    if not sorted_values:
        return torch.full_like(raw_score, 0.5)
    raw_score = torch.nan_to_num(raw_score.float(), nan=0.0, posinf=1e3, neginf=-1e3)
    sorted_tensor = torch.as_tensor(sorted_values, dtype=raw_score.dtype, device=raw_score.device)
    q75 = sorted_tensor[int(0.75 * max(len(sorted_tensor) - 1, 0))]
    q25 = sorted_tensor[int(0.25 * max(len(sorted_tensor) - 1, 0))]
    raw_iqr = torch.clamp(q75 - q25, min=1e-3)
    temperature_scale = float(config.get("difficulty", {}).get("soft_cdf_temperature", 0.1))
    temperature = torch.clamp(raw_iqr * temperature_scale, min=1e-3)
    logits = (raw_score.unsqueeze(-1) - sorted_tensor.unsqueeze(0)) / temperature
    return torch.sigmoid(logits).mean(dim=-1)


def inverse_empirical_cdf_tensor(score: torch.Tensor, sorted_values: List[float]) -> torch.Tensor:
    if not sorted_values:
        return torch.zeros_like(score)
    sorted_tensor = torch.as_tensor(sorted_values, dtype=score.dtype, device=score.device)
    if sorted_tensor.numel() == 1:
        return torch.full_like(score, float(sorted_tensor.item()))
    clamped_score = torch.clamp(score, 0.0, 1.0)
    position = clamped_score * float(sorted_tensor.numel() - 1)
    lower = torch.floor(position).long()
    upper = torch.clamp(lower + 1, max=sorted_tensor.numel() - 1)
    frac = position - lower.float()
    lower_values = sorted_tensor[lower]
    upper_values = sorted_tensor[upper]
    return lower_values * (1.0 - frac) + upper_values * frac


def _difficulty_scale(value: float, stats: Dict[str, float], epsilon: float = 1e-3) -> float:
    spread = max(float(stats.get("iqr", 0.0)), abs(float(stats.get("max", 0.0)) - float(stats.get("min", 0.0))))
    if spread < epsilon:
        return 0.0
    return float((value - float(stats.get("median", 0.0))) / spread)


def _difficulty_scale_tensor(value: torch.Tensor, stats: Dict[str, float], epsilon: float = 1e-3) -> torch.Tensor:
    value = torch.nan_to_num(value.float(), nan=0.0, posinf=1e3, neginf=-1e3)
    spread = max(float(stats.get("iqr", 0.0)), abs(float(stats.get("max", 0.0)) - float(stats.get("min", 0.0))))
    if spread < epsilon:
        return torch.zeros_like(value)
    return (value - float(stats.get("median", 0.0))) / spread


def _apply_bounded_z_scalar(z_value: float, z_cap: float | None) -> float:
    if z_cap is None or z_cap <= 0.0:
        return float(z_value)
    return float(z_cap * np.tanh(z_value / z_cap))


def _apply_bounded_z_tensor(z_value: torch.Tensor, z_cap: float | None) -> torch.Tensor:
    if z_cap is None or z_cap <= 0.0:
        return z_value
    return float(z_cap) * torch.tanh(z_value / float(z_cap))


def _collect_normalized_components(
    raw_features: Dict[str, torch.Tensor],
    normalizer: "DifficultyNormalizer",
    config: Dict,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, bool]]:
    raw_score = torch.zeros_like(next(iter(raw_features.values()))) if raw_features else torch.zeros(1)
    normalized_components: Dict[str, torch.Tensor] = {}
    missing_features: Dict[str, bool] = {}
    component_temperature = float(config.get("difficulty", {}).get("soft_component_temperature", 2.0))
    for key in EXPECTED_DIFFICULTY_FEATURES:
        value = raw_features.get(key)
        stats = normalizer.metric_stats.get(key)
        if value is None or stats is None:
            missing_features[key] = True
            continue
        value = torch.nan_to_num(value.float(), nan=0.0, posinf=1e3, neginf=-1e3)
        corrected_value = -value if key in INVERTED_DIFFICULTY_FEATURES else value
        z_value = _difficulty_scale_tensor(corrected_value, stats)
        bounded_z = _apply_bounded_z_tensor(z_value, normalizer.z_score_cap)
        normalized_components[key] = torch.sigmoid(bounded_z / max(component_temperature, 1e-4))
        if key in normalizer.metric_weights:
            raw_score = raw_score + float(normalizer.metric_weights[key]) * bounded_z
        missing_features[key] = False
    return raw_score, normalized_components, missing_features


def _pairwise_distance_features(future_states: torch.Tensor, future_mask: torch.Tensor, agent_mask: torch.Tensor, focal_only: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    xy = future_states[..., 0:2]
    diffs = xy.unsqueeze(2) - xy.unsqueeze(1)
    distances = torch.linalg.norm(diffs, dim=-1)
    valid = future_mask.unsqueeze(2) & future_mask.unsqueeze(1)
    valid = valid & agent_mask.unsqueeze(-1).unsqueeze(-1)
    valid = valid & agent_mask.unsqueeze(1).unsqueeze(-1)
    eye = torch.eye(distances.shape[1], dtype=torch.bool, device=distances.device).unsqueeze(0).unsqueeze(-1)
    valid = valid & (~eye)
    distances = torch.where(valid, distances, torch.full_like(distances, 1e6))
    if focal_only:
        focal_dist = distances[:, 0]
        return focal_dist.min(dim=0).values.min(), distances
    return distances.min(), distances


def _ttc_proxy(current_states: torch.Tensor, future_states: torch.Tensor, future_mask: torch.Tensor, agent_mask: torch.Tensor, ttc_max_sec: float) -> torch.Tensor:
    current_xy = current_states[..., 0:2]
    current_v = current_states[..., 2:4]
    rel_pos = current_xy.unsqueeze(2) - current_xy.unsqueeze(1)
    rel_vel = current_v.unsqueeze(2) - current_v.unsqueeze(1)
    closing_rate = -(rel_pos * rel_vel).sum(dim=-1)
    rel_speed_sq = torch.clamp(torch.sum(rel_vel**2, dim=-1), min=1e-3)
    ttc = torch.where(
        closing_rate > 0.0,
        torch.clamp(closing_rate / rel_speed_sq, min=0.0, max=ttc_max_sec),
        torch.full_like(closing_rate, ttc_max_sec),
    )
    valid_pairs = agent_mask.unsqueeze(1) & agent_mask.unsqueeze(2)
    eye = torch.eye(ttc.shape[-1], dtype=torch.bool, device=ttc.device).unsqueeze(0)
    ttc = torch.where(valid_pairs & (~eye), ttc, torch.full_like(ttc, ttc_max_sec))
    if future_mask.sum() == 0:
        return torch.tensor(ttc_max_sec, device=ttc.device)
    return ttc.min()


def _required_decel_proxy(current_states: torch.Tensor, future_states: torch.Tensor, future_mask: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
    current_xy = current_states[..., 0:2]
    speed = torch.linalg.norm(current_states[..., 2:4], dim=-1)
    pair_dist = torch.cdist(current_xy, current_xy)
    valid_pairs = agent_mask.unsqueeze(1) & agent_mask.unsqueeze(2)
    eye = torch.eye(pair_dist.shape[-1], dtype=torch.bool, device=pair_dist.device).unsqueeze(0)
    pair_dist = torch.where(valid_pairs & (~eye), pair_dist, torch.full_like(pair_dist, 1e6))
    min_dist = pair_dist.min(dim=-1).values
    required = (speed**2) / torch.clamp(2.0 * min_dist, min=1.0)
    required = required * agent_mask.float()
    return required.max()


def _masked_mean_np(values: np.ndarray, mask: np.ndarray) -> float:
    if mask.size == 0 or not mask.any():
        return 0.0
    return float(values[mask].mean())


def _safe_pairwise_current_metrics(current_states: np.ndarray, agent_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid_idx = np.flatnonzero(agent_mask)
    if len(valid_idx) <= 1:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)
    valid_states = current_states[valid_idx]
    rel_xy = valid_states[:, None, 0:2] - valid_states[None, :, 0:2]
    rel_v = valid_states[:, None, 2:4] - valid_states[None, :, 2:4]
    distances = np.linalg.norm(rel_xy, axis=-1)
    rel_speed = np.linalg.norm(rel_v, axis=-1)
    return distances.astype(np.float32), rel_speed.astype(np.float32)


def _derive_future_kinematics_np(
    current_states: np.ndarray,
    future_states: np.ndarray,
    future_valid: np.ndarray,
    agent_mask: np.ndarray,
    dt: float,
    heading_speed_eps: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    current_xy = current_states[:, 0:2]
    current_vel = current_states[:, 2:4]
    current_speed = np.linalg.norm(current_vel, axis=-1)
    current_heading = current_states[:, 4]

    prev_xy = np.concatenate([current_xy[:, None, :], future_states[:, :-1, 0:2]], axis=1)
    velocity = (future_states[..., 0:2] - prev_xy) / dt
    velocity = velocity * future_valid[..., None]
    speed = np.linalg.norm(velocity, axis=-1)

    prev_vel = np.concatenate([current_vel[:, None, :], velocity[:, :-1, :]], axis=1)
    prev_vel_valid = np.concatenate([agent_mask[:, None], future_valid[:, :-1]], axis=1)
    accel_valid = future_valid & prev_vel_valid
    accel_vec = (velocity - prev_vel) / dt
    accel = np.linalg.norm(accel_vec, axis=-1) * accel_valid

    prev_accel = np.concatenate([np.zeros_like(accel_vec[:, :1, :]), accel_vec[:, :-1, :]], axis=1)
    prev_accel_valid = np.concatenate([np.zeros_like(future_valid[:, :1]), accel_valid[:, :-1]], axis=1)
    jerk_valid = accel_valid & prev_accel_valid
    jerk_vec = (accel_vec - prev_accel) / dt
    jerk = np.linalg.norm(jerk_vec, axis=-1) * jerk_valid

    heading = np.arctan2(velocity[..., 1], velocity[..., 0] + 1e-6)
    heading_valid = future_valid & (speed > heading_speed_eps)
    current_heading_valid = agent_mask & (current_speed > heading_speed_eps)
    prev_heading = np.concatenate([current_heading[:, None], heading[:, :-1]], axis=1)
    prev_heading_valid = np.concatenate([current_heading_valid[:, None], heading_valid[:, :-1]], axis=1)
    yaw_valid = heading_valid & prev_heading_valid
    delta_heading = (heading - prev_heading + np.pi) % (2 * np.pi) - np.pi
    yaw_rate = np.abs(delta_heading / dt) * yaw_valid
    curvature_valid = yaw_valid & (speed > heading_speed_eps)
    curvature = np.zeros_like(speed)
    curvature[curvature_valid] = yaw_rate[curvature_valid] / np.clip(speed[curvature_valid], 1e-3, None)
    return speed, accel, jerk, yaw_rate, curvature


def _derive_future_kinematics_torch(
    current_states: torch.Tensor,
    future_states: torch.Tensor,
    future_valid: torch.Tensor,
    agent_mask: torch.Tensor,
    dt: float,
    heading_speed_eps: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    current_xy = current_states[..., 0:2]
    current_vel = current_states[..., 2:4]
    current_speed = torch.linalg.norm(current_vel, dim=-1)
    current_heading = current_states[..., 4]

    prev_xy = torch.cat([current_xy.unsqueeze(2), future_states[:, :, :-1, 0:2]], dim=2)
    velocity = (future_states[..., 0:2] - prev_xy) / dt
    velocity = velocity * future_valid.unsqueeze(-1).float()
    speed = torch.linalg.norm(velocity, dim=-1)

    prev_vel = torch.cat([current_vel.unsqueeze(2), velocity[:, :, :-1, :]], dim=2)
    prev_vel_valid = torch.cat([agent_mask.unsqueeze(-1), future_valid[:, :, :-1]], dim=2)
    accel_valid = future_valid & prev_vel_valid
    accel_vec = (velocity - prev_vel) / dt
    accel = torch.linalg.norm(accel_vec, dim=-1) * accel_valid.float()

    prev_accel = torch.cat([torch.zeros_like(accel_vec[:, :, :1, :]), accel_vec[:, :, :-1, :]], dim=2)
    prev_accel_valid = torch.cat([torch.zeros_like(future_valid[:, :, :1]), accel_valid[:, :, :-1]], dim=2)
    jerk_valid = accel_valid & prev_accel_valid
    jerk_vec = (accel_vec - prev_accel) / dt
    jerk = torch.linalg.norm(jerk_vec, dim=-1) * jerk_valid.float()

    heading = torch.atan2(velocity[..., 1], velocity[..., 0] + 1e-6)
    heading_valid = future_valid & (speed > heading_speed_eps)
    current_heading_valid = agent_mask & (current_speed > heading_speed_eps)
    prev_heading = torch.cat([current_heading.unsqueeze(2), heading[:, :, :-1]], dim=2)
    prev_heading_valid = torch.cat([current_heading_valid.unsqueeze(-1), heading_valid[:, :, :-1]], dim=2)
    yaw_valid = heading_valid & prev_heading_valid
    delta_heading = _wrap_angle_torch(heading - prev_heading)
    yaw_rate = torch.abs(delta_heading / dt) * yaw_valid.float()
    curvature_valid = yaw_valid & (speed > heading_speed_eps)
    curvature = torch.where(curvature_valid, yaw_rate / torch.clamp(speed, min=1e-3), torch.zeros_like(speed))
    return speed, accel, jerk, yaw_rate, curvature


def compute_difficulty_features(slice_item: Dict, config: Dict) -> Dict[str, float]:
    global _FALLBACK_WARNING_EMITTED
    current_states = np.asarray(slice_item["current_states"], dtype=np.float32)
    future_states = np.asarray(slice_item["future_states"], dtype=np.float32)
    future_mask = np.asarray(slice_item["future_mask"], dtype=bool)
    agent_mask = np.asarray(slice_item["agent_mask"], dtype=bool)
    map_polylines = np.asarray(slice_item["map_polylines"], dtype=np.float32)
    map_polyline_mask = np.asarray(slice_item["map_polyline_mask"], dtype=bool)
    map_point_mask = np.asarray(slice_item.get("map_point_mask", np.zeros(map_polylines.shape[:2], dtype=bool)), dtype=bool)
    dt = float(config["data"]["timestep_sec"])
    interaction_radius = float(config["difficulty"]["interaction_radius_m"])
    conflict_distance = float(config["difficulty"]["conflict_distance_m"])
    ttc_max_sec = float(config["difficulty"]["ttc_max_sec"])
    valid_agents = np.flatnonzero(agent_mask)
    num_agents = float(len(valid_agents))
    if len(valid_agents) == 0:
        return _sanitize_difficulty_features({
            "min_pairwise_distance_future": 100.0,
            "min_distance_to_focal_future": 100.0,
            "ttc_proxy": ttc_max_sec,
            "conflict_count": 0.0,
            "interaction_density": 0.0,
            "relative_speed_mean": 0.0,
            "relative_speed_max": 0.0,
            "required_deceleration_proxy": 0.0,
            "mean_speed": 0.0,
            "max_speed": 0.0,
            "mean_abs_accel": 0.0,
            "p95_abs_accel": 0.0,
            "mean_abs_jerk": 0.0,
            "mean_abs_yaw_rate": 0.0,
            "curvature": 0.0,
            "num_agents": 0.0,
            "map_polyline_density_near_focal": float(map_polyline_mask.sum()),
            "route_curvature_proxy": 0.0,
        })
    valid_current = current_states[valid_agents]
    current_speed = np.linalg.norm(valid_current[:, 2:4], axis=-1)
    density = float(np.sum(np.linalg.norm(valid_current[:, 0:2], axis=-1) <= interaction_radius))
    current_pair_dist, relative_speed = _safe_pairwise_current_metrics(current_states, agent_mask)
    if len(valid_agents) > 1:
        eye = np.eye(len(valid_agents), dtype=bool)
        rel_pos = valid_current[:, None, 0:2] - valid_current[None, :, 0:2]
        rel_vel = valid_current[:, None, 2:4] - valid_current[None, :, 2:4]
        closing_rate = -(rel_pos * rel_vel).sum(axis=-1)
        rel_speed_sq = np.clip(np.sum(rel_vel**2, axis=-1), 1e-3, None)
        ttc_matrix = np.where(
            closing_rate > 0.0,
            np.clip(closing_rate / rel_speed_sq, 0.0, ttc_max_sec),
            ttc_max_sec,
        )
        ttc_matrix[eye] = ttc_max_sec
        min_current_dist = np.where(eye, np.inf, current_pair_dist).min(axis=-1)
        required_decel = float(np.max((current_speed**2) / np.clip(2.0 * min_current_dist, 1.0, None)))
        relative_speed_values = relative_speed[~eye]
        ttc_proxy = float(ttc_matrix.min()) if ttc_matrix.size else ttc_max_sec
        relative_speed_mean = float(relative_speed_values.mean()) if relative_speed_values.size else 0.0
        relative_speed_max = float(relative_speed_values.max()) if relative_speed_values.size else 0.0
    else:
        required_decel = 0.0
        ttc_proxy = ttc_max_sec
        relative_speed_mean = 0.0
        relative_speed_max = 0.0
    future_valid = future_mask & agent_mask[:, None]
    speed = np.linalg.norm(future_states[..., 2:4], axis=-1)
    accel = np.zeros_like(speed)
    yaw_rate = np.zeros_like(speed)
    curvature = np.zeros_like(speed)
    if future_states.shape[1] > 1:
        delta_v = np.diff(future_states[..., 2:4], axis=1) / dt
        accel_mask = future_valid[:, 1:] & future_valid[:, :-1]
        accel[:, 1:] = np.linalg.norm(delta_v, axis=-1) * accel_mask
        delta_h = (np.diff(future_states[..., 4], axis=1) + np.pi) % (2 * np.pi) - np.pi
        yaw_rate[:, 1:] = np.abs(delta_h / dt) * accel_mask
        curvature[:, 1:] = yaw_rate[:, 1:] / np.clip(speed[:, 1:], 1e-3, None)
    jerk = np.zeros_like(speed)
    if accel.shape[1] > 2:
        delta_a = np.diff(accel[:, 1:], axis=1) / dt
        jerk_mask = future_valid[:, 2:] & future_valid[:, 1:-1] & future_valid[:, :-2]
        jerk[:, 2:] = np.abs(delta_a) * jerk_mask
    min_pairwise = 100.0
    min_focal = 100.0
    conflict_pairs: set[tuple[int, int]] = set()
    if future_valid.any() and len(valid_agents) > 1:
        for t_idx in range(future_states.shape[1]):
            present = np.flatnonzero(future_valid[:, t_idx])
            if len(present) <= 1:
                continue
            xy = future_states[present, t_idx, 0:2]
            pair_dist = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
            eye_t = np.eye(len(present), dtype=bool)
            pair_dist[eye_t] = np.inf
            min_pairwise = min(min_pairwise, float(pair_dist.min()))
            if 0 in present:
                focal_local = int(np.where(present == 0)[0][0])
                focal_dist = pair_dist[focal_local]
                if focal_dist.size:
                    min_focal = min(min_focal, float(focal_dist.min()))
            conflict_local = np.argwhere(pair_dist < conflict_distance)
            for i_local, j_local in conflict_local:
                i_track = int(present[i_local])
                j_track = int(present[j_local])
                if i_track != j_track:
                    conflict_pairs.add(tuple(sorted((i_track, j_track))))
    if not future_valid.any() or len(valid_agents) <= 1:
        if not _FALLBACK_WARNING_EMITTED:
            LOGGER.warning("Difficulty computation fell back to dynamics-only mode for slices with insufficient future/agent context.")
            _FALLBACK_WARNING_EMITTED = True
        min_pairwise = 100.0
        min_focal = 100.0
    valid_future_values = future_valid
    p95_abs_accel = float(np.quantile(accel[valid_future_values], 0.95)) if valid_future_values.any() else 0.0
    map_valid = map_polyline_mask[:, None] & map_point_mask
    route_curvature_proxy = float(np.abs(map_polylines[..., 2])[map_valid].mean()) if map_valid.any() and map_polylines.shape[-1] > 2 else 0.0
    return _sanitize_difficulty_features({
        "min_pairwise_distance_future": float(min_pairwise),
        "min_distance_to_focal_future": float(min_focal),
        "ttc_proxy": float(ttc_proxy),
        "conflict_count": float(len(conflict_pairs)),
        "interaction_density": density,
        "relative_speed_mean": relative_speed_mean,
        "relative_speed_max": relative_speed_max,
        "required_deceleration_proxy": required_decel,
        "mean_speed": _masked_mean_np(speed, valid_future_values),
        "max_speed": float(speed[valid_future_values].max()) if valid_future_values.any() else 0.0,
        "mean_abs_accel": _masked_mean_np(accel, valid_future_values),
        "p95_abs_accel": p95_abs_accel,
        "mean_abs_jerk": _masked_mean_np(jerk, valid_future_values),
        "mean_abs_yaw_rate": _masked_mean_np(yaw_rate, valid_future_values),
        "curvature": _masked_mean_np(curvature, valid_future_values),
        "num_agents": num_agents,
        "map_polyline_density_near_focal": float(map_polyline_mask.sum()),
        "route_curvature_proxy": route_curvature_proxy,
    })


@dataclass
class DifficultyNormalizer:
    metric_stats: Dict[str, Dict[str, float]]
    raw_scores_sorted: List[float]
    quantile_thresholds: Dict[str, float]
    metric_weights: Dict[str, float]
    z_score_cap: float | None = None

    def transform_difficulty(self, features: Dict[str, float]) -> Tuple[float, float]:
        scores, raw_scores, _, _ = self.transform_batch([features])
        if scores.size == 0:
            return 0.5, 0.0
        return float(scores[0]), float(raw_scores[0])

    def transform_batch(self, features_list_or_matrix: Any) -> tuple[np.ndarray, np.ndarray, list[str], int]:
        matrix, missing_feature_count = _difficulty_feature_matrix(features_list_or_matrix)
        if matrix.shape[0] == 0:
            return (
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                [],
                int(missing_feature_count),
            )
        corrected = matrix.copy()
        for feature_name in INVERTED_DIFFICULTY_FEATURES:
            feature_idx = EXPECTED_DIFFICULTY_FEATURES.index(feature_name)
            corrected[:, feature_idx] *= -1.0
        raw_scores = np.zeros((corrected.shape[0],), dtype=np.float32)
        for key, weight in self.metric_weights.items():
            if key not in EXPECTED_DIFFICULTY_FEATURES:
                continue
            stats = self.metric_stats.get(key, {"median": 0.0, "iqr": 1.0, "min": 0.0, "max": 1.0})
            idx = EXPECTED_DIFFICULTY_FEATURES.index(key)
            spread = max(float(stats.get("iqr", 0.0)), abs(float(stats.get("max", 0.0)) - float(stats.get("min", 0.0))), 1e-3)
            z_values = (corrected[:, idx] - float(stats.get("median", 0.0))) / spread
            bounded = z_values if not self.z_score_cap or self.z_score_cap <= 0.0 else float(self.z_score_cap) * np.tanh(z_values / float(self.z_score_cap))
            raw_scores += float(weight) * bounded.astype(np.float32)
        if self.raw_scores_sorted:
            sorted_values = np.asarray(self.raw_scores_sorted, dtype=np.float32)
            valid_sorted = sorted_values[np.isfinite(sorted_values)]
            if valid_sorted.size == 0:
                scores = np.full_like(raw_scores, 0.5, dtype=np.float32)
            else:
                ranks = np.searchsorted(valid_sorted, raw_scores, side="right")
                scores = (ranks.astype(np.float32) / float(valid_sorted.size)).astype(np.float32)
        else:
            scores = np.full_like(raw_scores, 0.5, dtype=np.float32)
        scores = np.clip(scores, 0.0, 1.0).astype(np.float32)
        levels = [self.assign_difficulty_level(float(score)) for score in scores.tolist()]
        return scores, raw_scores.astype(np.float32), levels, int(missing_feature_count)

    def assign_difficulty_level(self, score: float) -> str:
        low = self.quantile_thresholds["low"]
        mid = self.quantile_thresholds["mid"]
        if score < low:
            return "low"
        if score < mid:
            return "mid"
        return "high"


def _missing_cached_feature_fields(features: Dict[str, Any]) -> list[str]:
    return [
        feature_name
        for feature_name in EXPECTED_DIFFICULTY_FEATURES
        if feature_name not in features or not np.isfinite(_safe_feature_value(features.get(feature_name), np.nan))
    ]


def _get_slice_feature_dict(
    slice_item: Dict,
    feature_key: str,
    config: Dict,
    strict_cached_features: bool = False,
) -> Dict[str, float]:
    if feature_key in slice_item and slice_item[feature_key]:
        features = dict(slice_item[feature_key])
        features.pop("raw_difficulty_score", None)
        missing_fields = _missing_cached_feature_fields(features)
        if strict_cached_features and missing_fields:
            raise KeyError(
                f"Slice {slice_item.get('slice_id', '<unknown>')} is missing cached difficulty features for '{feature_key}': {missing_fields}"
            )
        return _sanitize_difficulty_features(features)
    if strict_cached_features:
        raise KeyError(
            f"Slice {slice_item.get('slice_id', '<unknown>')} is missing cached difficulty feature block '{feature_key}'."
        )
    return _sanitize_difficulty_features(compute_difficulty_features(slice_item, config))


def fit_difficulty_normalizer(train_slices: Iterable[Dict], config: Dict, feature_key: str = "difficulty_features_selected_agents") -> DifficultyNormalizer:
    train_slices = list(train_slices)
    metric_weights = dict(config["difficulty"]["metric_weights"])
    z_score_cap = float(config.get("difficulty", {}).get("z_score_cap", 0.0)) or None
    features_per_metric: Dict[str, List[float]] = {}
    raw_scores: List[float] = []
    direction_corrected_list: List[Dict[str, float]] = []
    for slice_item in tqdm(train_slices, desc="Fitting difficulty normalizer", unit="slice"):
        features = _get_slice_feature_dict(slice_item, feature_key, config, strict_cached_features=False)
        corrected = {}
        for key, value in features.items():
            corrected_value = _safe_feature_value(value, 0.0)
            if key in {"min_pairwise_distance_future", "min_distance_to_focal_future", "ttc_proxy"}:
                corrected_value = -corrected_value
            corrected[key] = corrected_value
            features_per_metric.setdefault(key, []).append(corrected_value)
        direction_corrected_list.append(corrected)
    metric_stats = {key: robust_stats(values) for key, values in features_per_metric.items()}
    for corrected in direction_corrected_list:
        raw_scores.append(
            sum(
                metric_weights.get(key, 0.0) * _apply_bounded_z_scalar(_difficulty_scale(corrected.get(key, 0.0), metric_stats[key]), z_score_cap)
                for key in metric_weights
            )
        )
    raw_scores_sorted = empirical_cdf_values(raw_scores)["sorted_values"]
    quantiles = config["difficulty"]["quantiles"]
    thresholds = {
        "low": float(np.quantile(raw_scores, quantiles[0])) if raw_scores else 0.33,
        "mid": float(np.quantile(raw_scores, quantiles[1])) if raw_scores else 0.67,
    }
    normalizer = DifficultyNormalizer(
        metric_stats=metric_stats,
        raw_scores_sorted=raw_scores_sorted,
        quantile_thresholds=thresholds,
        metric_weights=metric_weights,
        z_score_cap=z_score_cap,
    )
    return normalizer


def attach_difficulty_to_slices(
    slices: Iterable[Dict],
    normalizer: DifficultyNormalizer,
    config: Dict,
    feature_key: str = "difficulty_features_selected_agents",
    score_key: str = "difficulty_score_selected_agents",
    level_key: str = "difficulty_level_selected_agents",
) -> List[Dict]:
    slices = list(slices)
    strict_cached = bool(config.get("processing", {}).get("strict_cached_features_for_scoring", False))
    start_time = perf_counter()
    features_list: list[Dict[str, float]] = []
    updated = []
    for slice_item in slices:
        features_list.append(_get_slice_feature_dict(slice_item, feature_key, config, strict_cached_features=strict_cached))
    scores, raw_scores, levels, missing_feature_count = normalizer.transform_batch(features_list)
    for slice_item, features, score, raw_score, level in tqdm(
        zip(slices, features_list, scores.tolist(), raw_scores.tolist(), levels),
        desc="Scoring slice difficulty",
        unit="slice",
        leave=False,
        total=len(slices),
    ):
        slice_item[feature_key] = {**features, "raw_difficulty_score": raw_score}
        slice_item[score_key] = score
        slice_item[level_key] = level
        if score_key == "difficulty_score_selected_agents":
            slice_item["difficulty_features"] = slice_item[feature_key]
            slice_item["difficulty_score"] = score
            slice_item["difficulty_level"] = slice_item[level_key]
        updated.append(slice_item)
    elapsed = max(perf_counter() - start_time, 1e-9)
    LOGGER.info(
        "Scored difficulty: vectorized=%s num_slices=%d scoring_time_sec=%.4f slices_per_sec=%.2f missing_feature_count=%d feature_key=%s",
        True,
        len(updated),
        elapsed,
        len(updated) / elapsed,
        int(missing_feature_count),
        feature_key,
    )
    return updated


def difficulty_report(slices_by_split: Dict[str, List[Dict]], score_key: str = "difficulty_score_selected_agents") -> Dict[str, Dict[str, float]]:
    report: Dict[str, Dict[str, float]] = {}
    for split, items in slices_by_split.items():
        scores = np.asarray([float(item.get(score_key, 0.0)) for item in items], dtype=np.float32)
        if len(scores) == 0:
            report[split] = {"count": 0}
            continue
        report[split] = {
            "count": int(len(scores)),
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "min": float(scores.min()),
            "max": float(scores.max()),
            "low_count": int(np.sum(scores < 0.333)),
            "mid_count": int(np.sum((scores >= 0.333) & (scores < 0.667))),
            "high_count": int(np.sum(scores >= 0.667)),
        }
    return report


def save_difficulty_stats(normalizers: Dict[str, DifficultyNormalizer], path: str | Path) -> None:
    payload = {}
    for name, normalizer in normalizers.items():
        payload[name] = {
            "metric_stats": normalizer.metric_stats,
            "raw_scores_sorted": normalizer.raw_scores_sorted,
            "quantile_thresholds": normalizer.quantile_thresholds,
            "metric_weights": normalizer.metric_weights,
            "z_score_cap": normalizer.z_score_cap,
        }
    save_json(payload, path)


def load_difficulty_normalizer(path: str | Path, label_space: str = "selected_agents") -> DifficultyNormalizer:
    from isgen import load_json

    payload = load_json(path)
    if "metric_stats" in payload:
        normalizer_payload = payload
    else:
        if label_space not in payload:
            raise KeyError(f"Difficulty normalizer '{label_space}' not found in {path}. Available keys: {list(payload.keys())}")
        normalizer_payload = payload[label_space]
    return DifficultyNormalizer(
        metric_stats=normalizer_payload["metric_stats"],
        raw_scores_sorted=normalizer_payload["raw_scores_sorted"],
        quantile_thresholds=normalizer_payload["quantile_thresholds"],
        metric_weights=normalizer_payload["metric_weights"],
        z_score_cap=normalizer_payload.get("z_score_cap"),
    )


def differentiable_difficulty_score(
    current_states: torch.Tensor,
    future_states: torch.Tensor,
    future_mask: torch.Tensor,
    agent_mask: torch.Tensor,
    normalizer: DifficultyNormalizer,
    config: Dict,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    dt = float(config["data"]["timestep_sec"])
    current_states = torch.nan_to_num(current_states.float(), nan=0.0, posinf=0.0, neginf=0.0)
    future_states = torch.nan_to_num(future_states.float(), nan=0.0, posinf=0.0, neginf=0.0)
    interaction_radius = float(config["difficulty"]["interaction_radius_m"])
    conflict_distance = float(config["difficulty"]["conflict_distance_m"])
    ttc_max_sec = float(config["difficulty"]["ttc_max_sec"])
    distance_temperature = float(config.get("difficulty", {}).get("softmin_temperature_m", 2.0))
    ttc_temperature = float(config.get("difficulty", {}).get("soft_ttc_temperature_sec", distance_temperature))
    conflict_temperature = float(config.get("difficulty", {}).get("soft_conflict_temperature_m", 1.0))
    density_temperature = float(config.get("difficulty", {}).get("soft_density_temperature_m", 2.0))
    topk_temperature = float(config.get("difficulty", {}).get("soft_topk_temperature", 1.0))

    batch_size, num_agents, horizon, _ = future_states.shape
    future_valid = future_mask & agent_mask.unsqueeze(-1)
    traj_features = compute_trajectory_features(future_states, future_valid, dt)
    speed = torch.nan_to_num(traj_features["speed"], nan=0.0, posinf=0.0, neginf=0.0)
    accel = torch.nan_to_num(traj_features["accel"], nan=0.0, posinf=0.0, neginf=0.0)
    jerk = torch.nan_to_num(traj_features["jerk"], nan=0.0, posinf=0.0, neginf=0.0)
    yaw_rate = torch.nan_to_num(traj_features["yaw_rate"], nan=0.0, posinf=0.0, neginf=0.0)
    curvature = torch.nan_to_num(traj_features["curvature"], nan=0.0, posinf=0.0, neginf=0.0)

    current_xy = current_states[..., 0:2]
    current_vel = current_states[..., 2:4]
    current_speed = torch.nan_to_num(torch.linalg.norm(current_vel, dim=-1), nan=0.0, posinf=0.0, neginf=0.0)

    eye = torch.eye(num_agents, dtype=torch.bool, device=current_states.device).unsqueeze(0)
    upper_triangle = torch.triu(torch.ones(num_agents, num_agents, dtype=torch.bool, device=current_states.device), diagonal=1).unsqueeze(0)
    pair_valid = agent_mask.unsqueeze(1) & agent_mask.unsqueeze(2) & (~eye)
    upper_pair_valid = pair_valid & upper_triangle

    current_rel_xy = current_xy.unsqueeze(2) - current_xy.unsqueeze(1)
    current_pair_dist = torch.nan_to_num(torch.linalg.norm(current_rel_xy, dim=-1), nan=0.0, posinf=1e3, neginf=0.0)
    current_rel_vel = current_vel.unsqueeze(2) - current_vel.unsqueeze(1)
    current_pair_rel_speed = torch.nan_to_num(torch.linalg.norm(current_rel_vel, dim=-1), nan=0.0, posinf=1e3, neginf=0.0)
    closing_projection = -(current_rel_xy * current_rel_vel).sum(dim=-1)
    rel_speed_sq = torch.clamp(current_rel_vel.square().sum(dim=-1), min=1e-3)
    ttc_matrix = torch.clamp(closing_projection / rel_speed_sq, min=0.0, max=ttc_max_sec)
    ttc_valid = upper_pair_valid & (closing_projection > 0.0)
    ttc_proxy = _masked_softmin(ttc_matrix, ttc_valid, dim=(1, 2), temperature=max(ttc_temperature, 1e-3), fallback=ttc_max_sec)

    focal_current_dist = torch.nan_to_num(torch.linalg.norm(current_xy - current_xy[:, :1, :], dim=-1), nan=0.0, posinf=1e3, neginf=0.0)
    density_logits = torch.sigmoid((interaction_radius - focal_current_dist) / max(density_temperature, 1e-4))
    interaction_density = (density_logits * agent_mask.float()).sum(dim=-1)

    nearest_neighbor_dist = _masked_softmin(
        current_pair_dist,
        pair_valid,
        dim=2,
        temperature=distance_temperature,
        fallback=max(interaction_radius, 30.0),
    )
    required_decel_per_agent = (current_speed.square() / torch.clamp(2.0 * nearest_neighbor_dist, min=1.0)) * agent_mask.float()
    required_decel = _masked_softmax_value(required_decel_per_agent, agent_mask, dim=1, temperature=topk_temperature, fallback=0.0)
    relative_speed_mean = _masked_mean_tensor(current_pair_rel_speed, upper_pair_valid, dim=(1, 2))
    relative_speed_max = _masked_softmax_value(current_pair_rel_speed, upper_pair_valid, dim=(1, 2), temperature=topk_temperature, fallback=0.0)

    future_xy = future_states[..., 0:2]
    future_pair_dist = torch.nan_to_num(
        torch.linalg.norm(future_xy.unsqueeze(2) - future_xy.unsqueeze(1), dim=-1),
        nan=0.0,
        posinf=1e3,
        neginf=0.0,
    )
    future_pair_valid = future_valid.unsqueeze(2) & future_valid.unsqueeze(1) & (~eye.unsqueeze(-1))
    upper_future_pair_valid = future_pair_valid & upper_triangle.unsqueeze(-1)
    min_pairwise = _masked_softmin(
        future_pair_dist,
        upper_future_pair_valid,
        dim=(1, 2, 3),
        temperature=distance_temperature,
        fallback=100.0,
    )
    focal_future_dist = future_pair_dist[:, 0, 1:, :]
    focal_future_valid = future_pair_valid[:, 0, 1:, :]
    min_distance_to_focal = _masked_softmin(
        focal_future_dist,
        focal_future_valid,
        dim=(1, 2),
        temperature=distance_temperature,
        fallback=100.0,
    )

    conflict_prob_t = torch.sigmoid((conflict_distance - future_pair_dist) / max(conflict_temperature, 1e-4))
    conflict_prob_t = torch.where(upper_future_pair_valid, conflict_prob_t, torch.zeros_like(conflict_prob_t))
    valid_pair_time_counts = upper_future_pair_valid.any(dim=-1)
    conflict_any = _masked_softmax_value(
        conflict_prob_t,
        upper_future_pair_valid,
        dim=3,
        temperature=max(conflict_temperature, 1e-4),
        fallback=0.0,
    )
    conflict_any = torch.where(valid_pair_time_counts, conflict_any, torch.zeros_like(conflict_any))
    conflict_count = conflict_any.sum(dim=(1, 2))

    speed_valid = future_valid
    mean_speed = masked_mean(speed, speed_valid, dim=(-1, -2))
    max_speed = _masked_softmax_value(speed, speed_valid, dim=(1, 2), temperature=topk_temperature, fallback=0.0)
    mean_abs_accel = masked_mean(accel, speed_valid, dim=(-1, -2))
    p95_abs_accel = _masked_softmax_value(accel, speed_valid, dim=(1, 2), temperature=topk_temperature, fallback=0.0)
    mean_abs_jerk = masked_mean(jerk, speed_valid, dim=(-1, -2))
    mean_abs_yaw_rate = masked_mean(yaw_rate, speed_valid, dim=(-1, -2))
    curvature_mean = masked_mean(curvature, speed_valid, dim=(-1, -2))
    num_agents_feature = agent_mask.float().sum(dim=-1)

    raw_features: Dict[str, torch.Tensor] = {
        "min_pairwise_distance_future": min_pairwise,
        "min_distance_to_focal_future": min_distance_to_focal,
        "ttc_proxy": ttc_proxy,
        "conflict_count": conflict_count,
        "interaction_density": interaction_density,
        "required_deceleration_proxy": required_decel,
        "relative_speed_mean": relative_speed_mean,
        "relative_speed_max": relative_speed_max,
        "mean_speed": mean_speed,
        "max_speed": max_speed,
        "mean_abs_accel": mean_abs_accel,
        "p95_abs_accel": p95_abs_accel,
        "mean_abs_jerk": mean_abs_jerk,
        "mean_abs_yaw_rate": mean_abs_yaw_rate,
        "curvature": curvature_mean,
        "num_agents": num_agents_feature,
    }
    raw_features = {
        key: torch.nan_to_num(value.float(), nan=0.0, posinf=1e3, neginf=-1e3) for key, value in raw_features.items()
    }
    raw_score, normalized_components, missing_features = _collect_normalized_components(raw_features, normalizer, config)
    weighted_contributions: Dict[str, torch.Tensor] = {}
    z_components: Dict[str, torch.Tensor] = {}
    for key in EXPECTED_DIFFICULTY_FEATURES:
        tensor = raw_features.get(key)
        stats = normalizer.metric_stats.get(key)
        if tensor is None or stats is None:
            continue
        corrected = -tensor if key in INVERTED_DIFFICULTY_FEATURES else tensor
        corrected = torch.nan_to_num(corrected.float(), nan=0.0, posinf=1e3, neginf=-1e3)
        z_value = _difficulty_scale_tensor(corrected, stats)
        bounded_z = _apply_bounded_z_tensor(z_value, normalizer.z_score_cap)
        z_components[key] = bounded_z
        if key in normalizer.metric_weights:
            weighted_contributions[key] = float(normalizer.metric_weights[key]) * bounded_z
    raw_score = torch.nan_to_num(raw_score, nan=0.0, posinf=1e3, neginf=-1e3)
    difficulty = torch.nan_to_num(_soft_empirical_cdf(raw_score, normalizer.raw_scores_sorted, config), nan=0.5, posinf=1.0, neginf=0.0)
    sorted_tensor = torch.as_tensor(normalizer.raw_scores_sorted, dtype=raw_score.dtype, device=raw_score.device)
    debug_payload = {
        "raw_features": raw_features,
        "normalized_components": normalized_components,
        "missing_features": missing_features,
        "raw_score": raw_score,
        "z_components": z_components,
        "weighted_contributions": weighted_contributions,
        "raw_score_support": {
            "sorted_min": sorted_tensor.min() if sorted_tensor.numel() > 0 else torch.zeros_like(raw_score),
            "sorted_max": sorted_tensor.max() if sorted_tensor.numel() > 0 else torch.zeros_like(raw_score),
            "sorted_p95": torch.quantile(sorted_tensor, 0.95) if sorted_tensor.numel() > 0 else torch.zeros_like(raw_score),
            "raw_minus_support_max": raw_score - (sorted_tensor.max() if sorted_tensor.numel() > 0 else torch.zeros_like(raw_score)),
        },
        "map_features_missing": {
            "map_polyline_density_near_focal": True,
            "route_curvature_proxy": True,
        },
        "metadata": {
            "batch_size": batch_size,
            "num_agents": num_agents,
            "horizon": horizon,
        },
    }
    debug_payload = {
        key: {
            inner_key: torch.nan_to_num(inner_value, nan=0.0, posinf=1e3, neginf=-1e3)
            if isinstance(inner_value, torch.Tensor)
            else inner_value
            for inner_key, inner_value in value.items()
        }
        if isinstance(value, dict)
        else value
        for key, value in debug_payload.items()
    }
    return difficulty, debug_payload
