from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch

from isgen import load_json, save_json
from isgen.semantics.kinematics import compute_trajectory_features, masked_mean
from isgen.semantics.normalization import empirical_cdf_transform, empirical_cdf_values, robust_stats

BEHAVIOR_CODE_VERSION = "2026-04-19-behavior-v2"
EXPECTED_BEHAVIOR_FEATURES = [
    "mean_abs_accel",
    "p95_abs_accel",
    "mean_abs_jerk",
    "mean_abs_yaw_rate",
    "speed_gain",
    "final_displacement",
    "mean_speed",
    "max_speed",
    "hard_accel_rate",
    "hard_brake_rate",
    "hard_yaw_rate_rate",
]
DEFAULT_BEHAVIOR_WEIGHTS = {
    "mean_abs_accel": 1.0,
    "p95_abs_accel": 0.8,
    "mean_abs_jerk": 0.6,
    "mean_abs_yaw_rate": 0.9,
    "speed_gain": 0.4,
    "final_displacement": 0.3,
    "mean_speed": 0.4,
    "max_speed": 0.4,
    "hard_accel_rate": 0.8,
    "hard_brake_rate": 0.8,
    "hard_yaw_rate_rate": 0.8,
}

BEHAVIOR_TARGET_KEYS = {
    "behavior_aggressiveness_score_selected_agents",
    "behavior_quantile_score_selected_agents",
    "behavior_raw_score_selected_agents",
}
BEHAVIOR_GENERATED_SCORE_SPACES = {"quantile", "aggressiveness", "raw"}


def _safe_feature_value(value: Any, default: float = 0.0) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(scalar):
        return float(default)
    return scalar


def _sanitize_behavior_features(features: Dict[str, Any]) -> Dict[str, float]:
    return {key: _safe_feature_value(features.get(key, 0.0), 0.0) for key in EXPECTED_BEHAVIOR_FEATURES}


def _behavior_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    return config.get("behavior", {})


def resolve_behavior_target_key(config: Dict[str, Any]) -> str:
    target_key = str(_behavior_cfg(config).get("target_score_key", "behavior_quantile_score_selected_agents"))
    if target_key not in BEHAVIOR_TARGET_KEYS:
        raise ValueError(
            f"Unsupported behavior.target_score_key: {target_key}. "
            f"Expected one of {sorted(BEHAVIOR_TARGET_KEYS)}."
        )
    return target_key


def behavior_target_key_flags(target_key: str) -> Dict[str, bool]:
    return {
        "uses_behavior_aggressiveness_score_selected_agents": target_key == "behavior_aggressiveness_score_selected_agents",
        "uses_behavior_quantile_score_selected_agents": target_key == "behavior_quantile_score_selected_agents",
        "uses_retrieved_behavior_aggressiveness": False,
        "fallback_to_point_five": False,
    }


def resolve_generator_condition_score_key(config: Dict[str, Any]) -> str:
    condition_key = str(_behavior_cfg(config).get("generator_condition_score_key", resolve_behavior_target_key(config)))
    if condition_key not in BEHAVIOR_TARGET_KEYS:
        raise ValueError(
            f"Unsupported behavior.generator_condition_score_key: {condition_key}. "
            f"Expected one of {sorted(BEHAVIOR_TARGET_KEYS)}."
        )
    return condition_key


def resolve_generated_score_space(config: Dict[str, Any]) -> str:
    score_space = str(_behavior_cfg(config).get("generated_score_space", "quantile")).lower()
    if score_space not in BEHAVIOR_GENERATED_SCORE_SPACES:
        raise ValueError(
            f"Unsupported behavior.generated_score_space: {score_space}. "
            f"Expected one of {sorted(BEHAVIOR_GENERATED_SCORE_SPACES)}."
        )
    return score_space


def _metric_weights(config: Dict[str, Any]) -> Dict[str, float]:
    return dict(_behavior_cfg(config).get("metric_weights", DEFAULT_BEHAVIOR_WEIGHTS))


def _z_score_cap(config: Dict[str, Any]) -> float | None:
    value = float(_behavior_cfg(config).get("z_score_cap", 4.0))
    return value if value > 0.0 else None


def _difficulty_scale(value: float, stats: Dict[str, float], epsilon: float = 1e-3) -> float:
    spread = max(float(stats.get("iqr", 0.0)), abs(float(stats.get("max", 0.0)) - float(stats.get("min", 0.0))))
    if spread < epsilon:
        return 0.0
    return float((value - float(stats.get("median", 0.0))) / spread)


def _difficulty_scale_tensor(value: torch.Tensor, stats: Dict[str, float], epsilon: float = 1e-3) -> torch.Tensor:
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


def _soft_empirical_cdf(raw_score: torch.Tensor, sorted_values: List[float], temperature_scale: float) -> torch.Tensor:
    if not sorted_values:
        return torch.full_like(raw_score, 0.5)
    sorted_tensor = torch.as_tensor(sorted_values, dtype=raw_score.dtype, device=raw_score.device)
    q75 = sorted_tensor[int(0.75 * max(sorted_tensor.numel() - 1, 0))]
    q25 = sorted_tensor[int(0.25 * max(sorted_tensor.numel() - 1, 0))]
    iqr = torch.clamp(q75 - q25, min=1e-3)
    temperature = torch.clamp(iqr * float(temperature_scale), min=1e-3)
    logits = (raw_score.unsqueeze(-1) - sorted_tensor.unsqueeze(0)) / temperature
    return torch.sigmoid(logits).mean(dim=-1)


def _interpolated_empirical_cdf(raw_score: torch.Tensor, sorted_values: List[float]) -> torch.Tensor:
    if not sorted_values:
        return torch.full_like(raw_score, 0.5)
    if len(sorted_values) == 1:
        return torch.full_like(raw_score, 0.5)
    sorted_tensor = torch.as_tensor(sorted_values, dtype=raw_score.dtype, device=raw_score.device)
    upper_idx = torch.searchsorted(sorted_tensor, raw_score.detach(), right=False)
    upper_idx = torch.clamp(upper_idx, 0, sorted_tensor.numel() - 1)
    lower_idx = torch.clamp(upper_idx - 1, 0, sorted_tensor.numel() - 1)
    lower = sorted_tensor[lower_idx]
    upper = sorted_tensor[upper_idx]
    denom = torch.clamp(upper - lower, min=1e-6)
    frac = torch.where(
        upper_idx == 0,
        torch.zeros_like(raw_score),
        torch.where(
            upper_idx == sorted_tensor.numel() - 1,
            torch.where(raw_score >= upper, torch.ones_like(raw_score), torch.clamp((raw_score - lower) / denom, 0.0, 1.0)),
            torch.clamp((raw_score - lower) / denom, 0.0, 1.0),
        ),
    )
    rank = lower_idx.float() + frac
    return torch.clamp(rank / float(sorted_tensor.numel() - 1), 0.0, 1.0)


def _inverse_empirical_cdf_tensor(score: torch.Tensor, sorted_values: List[float]) -> torch.Tensor:
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


def _signed_speed_accel_np(speed: np.ndarray, future_valid: np.ndarray, current_speed: np.ndarray, dt: float) -> np.ndarray:
    signed = np.zeros_like(speed)
    if speed.shape[1] == 0:
        return signed
    signed[:, 0] = ((speed[:, 0] - current_speed) / dt) * future_valid[:, 0]
    if speed.shape[1] > 1:
        valid = future_valid[:, 1:] & future_valid[:, :-1]
        signed[:, 1:] = ((speed[:, 1:] - speed[:, :-1]) / dt) * valid
    return signed


def _signed_speed_accel_torch(speed: torch.Tensor, future_valid: torch.Tensor, current_speed: torch.Tensor, dt: float) -> torch.Tensor:
    speed = torch.nan_to_num(speed.float(), nan=0.0, posinf=0.0, neginf=0.0)
    current_speed = torch.nan_to_num(current_speed.float(), nan=0.0, posinf=0.0, neginf=0.0)
    signed = torch.zeros_like(speed)
    if speed.shape[-1] == 0:
        return signed
    first_delta = (speed[:, :, 0] - current_speed) / dt
    signed[:, :, 0] = torch.where(future_valid[:, :, 0], first_delta, torch.zeros_like(first_delta))
    if speed.shape[-1] > 1:
        valid = future_valid[:, :, 1:] & future_valid[:, :, :-1]
        later_delta = (speed[:, :, 1:] - speed[:, :, :-1]) / dt
        signed[:, :, 1:] = torch.where(valid, later_delta, torch.zeros_like(later_delta))
    return signed


def _scalar_jerk_from_accel_torch(accel: torch.Tensor, future_valid: torch.Tensor, dt: float) -> torch.Tensor:
    accel = torch.nan_to_num(accel.float(), nan=0.0, posinf=0.0, neginf=0.0)
    jerk = torch.zeros_like(accel)
    if accel.shape[-1] <= 2:
        return jerk
    valid = future_valid[:, :, 2:] & future_valid[:, :, 1:-1] & future_valid[:, :, :-2]
    jerk_delta = torch.abs((accel[:, :, 2:] - accel[:, :, 1:-1]) / dt)
    jerk[:, :, 2:] = torch.where(valid, jerk_delta, torch.zeros_like(jerk_delta))
    return jerk


def _masked_quantile_torch(values: torch.Tensor, mask: torch.Tensor, quantile: float) -> torch.Tensor:
    batch_size = values.shape[0]
    flat_values = torch.nan_to_num(values.float(), nan=0.0, posinf=1e3, neginf=-1e3).reshape(batch_size, -1)
    flat_mask = mask.reshape(batch_size, -1)
    sorted_values, _ = torch.sort(torch.where(flat_mask, flat_values, torch.full_like(flat_values, float("inf"))), dim=-1)
    valid_counts = flat_mask.sum(dim=-1)
    quantile_index = torch.clamp(torch.ceil(valid_counts.float() * float(quantile)).long() - 1, min=0)
    gathered = sorted_values.gather(dim=-1, index=quantile_index.unsqueeze(-1)).squeeze(-1)
    fallback = torch.zeros_like(gathered)
    return torch.nan_to_num(torch.where(valid_counts > 0, gathered, fallback), nan=0.0, posinf=0.0, neginf=0.0)


def _numpy_behavior_features(
    future_states: np.ndarray,
    future_mask: np.ndarray,
    current_states: np.ndarray,
    agent_mask: np.ndarray,
    config: Dict[str, Any],
) -> Dict[str, float]:
    dt = float(config["data"]["timestep_sec"])
    future_valid = future_mask & agent_mask[:, None]
    if not future_valid.any():
        return {key: 0.0 for key in EXPECTED_BEHAVIOR_FEATURES}
    velocity = future_states[..., 2:4]
    speed = np.linalg.norm(velocity, axis=-1)
    current_speed = np.linalg.norm(current_states[..., 2:4], axis=-1)
    accel = np.zeros_like(speed)
    if speed.shape[1] > 1:
        accel[:, 1:] = np.linalg.norm(np.diff(velocity, axis=1) / dt, axis=-1) * (future_valid[:, 1:] & future_valid[:, :-1])
    jerk = np.zeros_like(speed)
    if accel.shape[1] > 2:
        jerk[:, 2:] = np.abs(np.diff(accel[:, 1:], axis=1) / dt) * (future_valid[:, 2:] & future_valid[:, 1:-1] & future_valid[:, :-2])
    heading_delta = np.zeros_like(speed)
    if speed.shape[1] > 1:
        delta_h = (np.diff(future_states[..., 4], axis=1) + np.pi) % (2 * np.pi) - np.pi
        heading_delta[:, 1:] = np.abs(delta_h / dt) * (future_valid[:, 1:] & future_valid[:, :-1])
    signed_accel = _signed_speed_accel_np(speed, future_valid, current_speed, dt)
    hard_accel_threshold = float(_behavior_cfg(config).get("hard_accel_threshold_mps2", 2.0))
    hard_brake_threshold = float(_behavior_cfg(config).get("hard_brake_threshold_mps2", 2.0))
    hard_yaw_threshold = float(_behavior_cfg(config).get("hard_yaw_rate_threshold_radps", 0.3))
    valid_values = future_valid
    final_indices = np.maximum(np.sum(future_valid, axis=1) - 1, 0)
    final_xy = np.zeros((future_states.shape[0], 2), dtype=np.float32)
    final_speed = np.zeros((future_states.shape[0],), dtype=np.float32)
    for agent_idx in range(future_states.shape[0]):
        if agent_mask[agent_idx] and future_valid[agent_idx].any():
            final_xy[agent_idx] = future_states[agent_idx, final_indices[agent_idx], 0:2]
            final_speed[agent_idx] = speed[agent_idx, final_indices[agent_idx]]
    displacement = np.linalg.norm(final_xy - current_states[:, 0:2], axis=-1)
    valid_agent_float = np.maximum(agent_mask.astype(np.float32).sum(), 1.0)
    return {
        "mean_abs_accel": float(accel[valid_values].mean()) if valid_values.any() else 0.0,
        "p95_abs_accel": float(np.quantile(accel[valid_values], 0.95)) if valid_values.any() else 0.0,
        "mean_abs_jerk": float(jerk[valid_values].mean()) if valid_values.any() else 0.0,
        "mean_abs_yaw_rate": float(heading_delta[valid_values].mean()) if valid_values.any() else 0.0,
        "speed_gain": float(((final_speed - current_speed) * agent_mask).sum() / valid_agent_float),
        "final_displacement": float((displacement * agent_mask).sum() / valid_agent_float),
        "mean_speed": float(speed[valid_values].mean()) if valid_values.any() else 0.0,
        "max_speed": float(speed[valid_values].max()) if valid_values.any() else 0.0,
        "hard_accel_rate": float(((signed_accel > hard_accel_threshold) & valid_values).sum() / max(valid_values.sum(), 1)),
        "hard_brake_rate": float(((signed_accel < -hard_brake_threshold) & valid_values).sum() / max(valid_values.sum(), 1)),
        "hard_yaw_rate_rate": float(((heading_delta > hard_yaw_threshold) & valid_values).sum() / max(valid_values.sum(), 1)),
    }


def compute_behavior_aggressiveness_features(
    future_states: np.ndarray,
    future_mask: np.ndarray,
    current_states: np.ndarray,
    agent_mask: np.ndarray,
    config: Dict[str, Any],
) -> Dict[str, float]:
    return _sanitize_behavior_features(
        _numpy_behavior_features(
        future_states=np.asarray(future_states, dtype=np.float32),
        future_mask=np.asarray(future_mask, dtype=bool),
        current_states=np.asarray(current_states, dtype=np.float32),
        agent_mask=np.asarray(agent_mask, dtype=bool),
        config=config,
        )
    )


def compute_behavior_aggressiveness_features_from_slice(slice_item: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, float]:
    return compute_behavior_aggressiveness_features(
        future_states=slice_item["future_states"],
        future_mask=slice_item["future_mask"],
        current_states=slice_item["current_states"],
        agent_mask=slice_item["agent_mask"],
        config=config,
    )


@dataclass
class BehaviorNormalizer:
    metric_stats: Dict[str, Dict[str, float]]
    raw_scores_sorted: List[float]
    quantile_thresholds: Dict[str, float]
    metric_weights: Dict[str, float]
    raw_score_stats: Dict[str, float] | None = None
    z_score_cap: float | None = None
    score_temperature: float = 1.0
    soft_cdf_temperature: float = 0.01

    def _raw_score_stats(self) -> Dict[str, float]:
        if self.raw_score_stats is not None:
            return self.raw_score_stats
        if self.raw_scores_sorted:
            self.raw_score_stats = robust_stats(self.raw_scores_sorted)
        else:
            self.raw_score_stats = {"median": 0.0, "iqr": 1.0, "min": 0.0, "max": 1.0}
        return self.raw_score_stats

    def transform_raw_score(self, raw_score: float) -> float:
        scaled = _difficulty_scale(float(raw_score), self._raw_score_stats())
        bounded = _apply_bounded_z_scalar(scaled, self.z_score_cap)
        return float(1.0 / (1.0 + np.exp(-bounded / max(float(self.score_temperature), 1e-3))))

    def transform_raw_score_tensor(self, raw_score: torch.Tensor) -> torch.Tensor:
        raw_score = torch.nan_to_num(raw_score.float(), nan=0.0, posinf=1e3, neginf=-1e3)
        scaled = _difficulty_scale_tensor(raw_score, self._raw_score_stats())
        bounded = _apply_bounded_z_tensor(scaled, self.z_score_cap)
        return torch.sigmoid(bounded / max(float(self.score_temperature), 1e-3))

    def transform_quantile_score(self, raw_score: float) -> float:
        return float(empirical_cdf_transform(float(raw_score), self.raw_scores_sorted))

    def transform_quantile_score_tensor(self, raw_score: torch.Tensor) -> torch.Tensor:
        safe_raw_score = torch.nan_to_num(raw_score.float(), nan=0.0, posinf=1e3, neginf=-1e3)
        return _soft_empirical_cdf(safe_raw_score, self.raw_scores_sorted, self.soft_cdf_temperature)

    def inverse_score_tensor(self, score: torch.Tensor) -> torch.Tensor:
        score = torch.nan_to_num(score.float(), nan=0.5, posinf=1.0 - 1e-4, neginf=1e-4)
        stats = self._raw_score_stats()
        spread = max(float(stats.get("iqr", 0.0)), abs(float(stats.get("max", 0.0)) - float(stats.get("min", 0.0))), 1e-3)
        safe_score = torch.clamp(score, 1e-4, 1.0 - 1e-4)
        z = torch.log(safe_score / (1.0 - safe_score)) * max(float(self.score_temperature), 1e-3)
        if self.z_score_cap is not None and self.z_score_cap > 0.0:
            z = torch.clamp(z, -float(self.z_score_cap), float(self.z_score_cap))
        return z * spread + float(stats.get("median", 0.0))

    def inverse_quantile_score_tensor(self, score: torch.Tensor) -> torch.Tensor:
        safe_score = torch.nan_to_num(score.float(), nan=0.5, posinf=1.0, neginf=0.0)
        return _inverse_empirical_cdf_tensor(safe_score, self.raw_scores_sorted)

    def transform_score(self, features: Dict[str, float]) -> Tuple[float, float]:
        raw_score = 0.0
        for key, weight in self.metric_weights.items():
            stats = self.metric_stats.get(key, {"median": 0.0, "iqr": 1.0, "min": 0.0, "max": 1.0})
            scaled = _difficulty_scale(_safe_feature_value(features.get(key, 0.0), 0.0), stats)
            raw_score += float(weight) * _apply_bounded_z_scalar(scaled, self.z_score_cap)
        return self.transform_raw_score(raw_score), float(raw_score)

    def transform_scores(self, features: Dict[str, float]) -> Tuple[float, float, float]:
        aggressiveness_score, raw_score = self.transform_score(features)
        quantile_score = self.transform_quantile_score(raw_score)
        return aggressiveness_score, quantile_score, raw_score

    def assign_level(self, score: float) -> str:
        if score < self.quantile_thresholds["low"]:
            return "low"
        if score < self.quantile_thresholds["mid"]:
            return "mid"
        return "high"


def fit_behavior_normalizer(
    train_slices: Iterable[Dict[str, Any]],
    config: Dict[str, Any],
    feature_key: str = "behavior_features_selected_agents",
) -> BehaviorNormalizer:
    train_slices = list(train_slices)
    metric_weights = _metric_weights(config)
    z_score_cap = _z_score_cap(config)
    features_per_metric: Dict[str, List[float]] = {key: [] for key in EXPECTED_BEHAVIOR_FEATURES}
    per_slice_features: List[Dict[str, float]] = []
    for slice_item in train_slices:
        features = _sanitize_behavior_features(slice_item.get(feature_key) or compute_behavior_aggressiveness_features_from_slice(slice_item, config))
        per_slice_features.append(features)
        for key in EXPECTED_BEHAVIOR_FEATURES:
            features_per_metric[key].append(_safe_feature_value(features.get(key, 0.0), 0.0))
    metric_stats = {key: robust_stats(values) for key, values in features_per_metric.items()}
    raw_scores = []
    for features in per_slice_features:
        raw_scores.append(
            sum(
                float(metric_weights.get(key, 0.0))
                * _apply_bounded_z_scalar(_difficulty_scale(_safe_feature_value(features.get(key, 0.0), 0.0), metric_stats[key]), z_score_cap)
                for key in metric_weights
            )
        )
    raw_scores_sorted = empirical_cdf_values(raw_scores)["sorted_values"]
    raw_score_stats = robust_stats(raw_scores)
    raw_scores_array = np.asarray(raw_scores, dtype=np.float32)
    score_temperature = float(_behavior_cfg(config).get("score_temperature", 1.0))
    transformed_scores = 1.0 / (1.0 + np.exp(-np.asarray([
        _apply_bounded_z_scalar(_difficulty_scale(float(raw), raw_score_stats), z_score_cap) / max(score_temperature, 1e-3)
        for raw in raw_scores_array
    ], dtype=np.float32)))
    quantiles = _behavior_cfg(config).get("quantiles", [0.333, 0.667])
    return BehaviorNormalizer(
        metric_stats=metric_stats,
        raw_scores_sorted=raw_scores_sorted,
        quantile_thresholds={
            "low": float(np.quantile(transformed_scores, quantiles[0])) if raw_scores else 0.333,
            "mid": float(np.quantile(transformed_scores, quantiles[1])) if raw_scores else 0.667,
        },
        metric_weights=metric_weights,
        raw_score_stats=raw_score_stats,
        z_score_cap=z_score_cap,
        score_temperature=score_temperature,
        soft_cdf_temperature=float(_behavior_cfg(config).get("soft_cdf_temperature", 0.01)),
    )


def attach_behavior_to_slices(
    slices: Iterable[Dict[str, Any]],
    normalizer: BehaviorNormalizer,
    config: Dict[str, Any],
    feature_key: str = "behavior_features_selected_agents",
    score_key: str = "behavior_aggressiveness_score_selected_agents",
    level_key: str = "behavior_aggressiveness_level_selected_agents",
) -> List[Dict[str, Any]]:
    updated: List[Dict[str, Any]] = []
    for slice_item in slices:
        features = _sanitize_behavior_features(slice_item.get(feature_key) or compute_behavior_aggressiveness_features_from_slice(slice_item, config))
        score, quantile_score, raw_score = normalizer.transform_scores(features)
        slice_item[feature_key] = {**features, "raw_behavior_score": raw_score}
        slice_item["behavior_raw_score_selected_agents"] = raw_score
        slice_item[score_key] = score
        slice_item["behavior_quantile_score_selected_agents"] = quantile_score
        slice_item[level_key] = normalizer.assign_level(score)
        updated.append(slice_item)
    return updated


def save_behavior_stats(normalizer: BehaviorNormalizer, path: str | Path) -> None:
    save_json(
        {
            "metric_stats": normalizer.metric_stats,
            "raw_scores_sorted": normalizer.raw_scores_sorted,
            "quantile_thresholds": normalizer.quantile_thresholds,
            "metric_weights": normalizer.metric_weights,
            "raw_score_stats": normalizer.raw_score_stats,
            "z_score_cap": normalizer.z_score_cap,
            "score_temperature": normalizer.score_temperature,
            "soft_cdf_temperature": normalizer.soft_cdf_temperature,
        },
        path,
    )


def load_behavior_normalizer(path: str | Path) -> BehaviorNormalizer:
    payload = load_json(path)
    return BehaviorNormalizer(
        metric_stats=payload["metric_stats"],
        raw_scores_sorted=payload["raw_scores_sorted"],
        quantile_thresholds=payload["quantile_thresholds"],
        metric_weights=payload["metric_weights"],
        raw_score_stats=payload.get("raw_score_stats"),
        z_score_cap=payload.get("z_score_cap"),
        score_temperature=float(payload.get("score_temperature", 1.0)),
        soft_cdf_temperature=float(payload.get("soft_cdf_temperature", 0.01)),
    )


def differentiable_behavior_aggressiveness_score(
    current_states: torch.Tensor,
    future_states: torch.Tensor,
    future_mask: torch.Tensor,
    agent_mask: torch.Tensor,
    normalizer: BehaviorNormalizer,
    config: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    dt = float(config["data"]["timestep_sec"])
    current_states = torch.nan_to_num(current_states.float(), nan=0.0, posinf=0.0, neginf=0.0)
    future_states = torch.nan_to_num(future_states.float(), nan=0.0, posinf=0.0, neginf=0.0)
    future_valid = future_mask & agent_mask.unsqueeze(-1)
    traj = compute_trajectory_features(future_states, future_valid, dt)
    speed = torch.nan_to_num(traj["speed"], nan=0.0, posinf=0.0, neginf=0.0)
    accel = torch.nan_to_num(traj["accel"], nan=0.0, posinf=0.0, neginf=0.0)
    jerk = _scalar_jerk_from_accel_torch(accel, future_valid, dt)
    yaw_rate = torch.nan_to_num(traj["yaw_rate"], nan=0.0, posinf=0.0, neginf=0.0)
    current_speed = torch.nan_to_num(torch.linalg.norm(current_states[..., 2:4], dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
    signed_accel = _signed_speed_accel_torch(speed, future_valid, current_speed, dt)
    hard_accel_threshold = float(_behavior_cfg(config).get("hard_accel_threshold_mps2", 2.0))
    hard_brake_threshold = float(_behavior_cfg(config).get("hard_brake_threshold_mps2", 2.0))
    hard_yaw_threshold = float(_behavior_cfg(config).get("hard_yaw_rate_threshold_radps", 0.3))

    valid_count = torch.clamp(future_valid.float().sum(dim=(1, 2)), min=1.0)
    final_index = torch.clamp(future_valid.float().sum(dim=-1).long() - 1, min=0)
    safe_future_states = torch.where(future_valid.unsqueeze(-1), future_states, torch.zeros_like(future_states))
    gather_index = final_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, safe_future_states.shape[-1])
    final_states = torch.gather(safe_future_states, dim=2, index=gather_index).squeeze(2)
    final_speed = torch.gather(speed, dim=2, index=final_index.unsqueeze(-1)).squeeze(-1)
    displacement = torch.nan_to_num(
        torch.linalg.norm(final_states[..., 0:2] - current_states[..., 0:2], dim=-1),
        nan=0.0,
        posinf=1e3,
        neginf=0.0,
    )
    agent_denominator = torch.clamp(agent_mask.float().sum(dim=1), min=1.0)
    hard_accel_prob = torch.sigmoid((signed_accel - hard_accel_threshold) / 0.05)
    hard_brake_prob = torch.sigmoid((-signed_accel - hard_brake_threshold) / 0.05)
    hard_yaw_prob = torch.sigmoid((yaw_rate - hard_yaw_threshold) / 0.025)
    raw_features: Dict[str, torch.Tensor] = {
        "mean_abs_accel": masked_mean(accel, future_valid, dim=(-1, -2)),
        "p95_abs_accel": _masked_quantile_torch(accel, future_valid, 0.95),
        "mean_abs_jerk": masked_mean(jerk, future_valid, dim=(-1, -2)),
        "mean_abs_yaw_rate": masked_mean(yaw_rate, future_valid, dim=(-1, -2)),
        "speed_gain": torch.where(agent_mask, final_speed - current_speed, torch.zeros_like(final_speed)).sum(dim=1) / agent_denominator,
        "final_displacement": torch.where(agent_mask, displacement, torch.zeros_like(displacement)).sum(dim=1) / agent_denominator,
        "mean_speed": masked_mean(speed, future_valid, dim=(-1, -2)),
        "max_speed": torch.where(future_valid, speed, torch.zeros_like(speed)).amax(dim=(1, 2)),
        "hard_accel_rate": torch.where(future_valid, hard_accel_prob, torch.zeros_like(hard_accel_prob)).sum(dim=(1, 2)) / valid_count,
        "hard_brake_rate": torch.where(future_valid, hard_brake_prob, torch.zeros_like(hard_brake_prob)).sum(dim=(1, 2)) / valid_count,
        "hard_yaw_rate_rate": torch.where(future_valid, hard_yaw_prob, torch.zeros_like(hard_yaw_prob)).sum(dim=(1, 2)) / valid_count,
    }
    raw_features = {
        key: torch.nan_to_num(value.float(), nan=0.0, posinf=1e3, neginf=-1e3) for key, value in raw_features.items()
    }
    raw_score = torch.zeros_like(next(iter(raw_features.values())))
    normalized_components: Dict[str, torch.Tensor] = {}
    weighted_contributions: Dict[str, torch.Tensor] = {}
    z_components: Dict[str, torch.Tensor] = {}
    missing_features: Dict[str, bool] = {}
    component_temperature = float(_behavior_cfg(config).get("soft_component_temperature", 2.0))
    for key in EXPECTED_BEHAVIOR_FEATURES:
        value = raw_features.get(key)
        stats = normalizer.metric_stats.get(key)
        if value is None or stats is None:
            missing_features[key] = True
            continue
        value = torch.nan_to_num(value.float(), nan=0.0, posinf=1e3, neginf=-1e3)
        z_value = _difficulty_scale_tensor(value, stats)
        bounded_z = _apply_bounded_z_tensor(z_value, normalizer.z_score_cap)
        z_components[key] = bounded_z
        normalized_components[key] = torch.sigmoid(bounded_z / max(component_temperature, 1e-4))
        if key in normalizer.metric_weights:
            weighted = float(normalizer.metric_weights[key]) * bounded_z
            weighted_contributions[key] = weighted
            raw_score = raw_score + weighted
        missing_features[key] = False
    raw_score = torch.nan_to_num(raw_score, nan=0.0, posinf=1e3, neginf=-1e3)
    behavior_score = normalizer.transform_raw_score_tensor(raw_score)
    raw_scores_sorted = normalizer.raw_scores_sorted
    raw_score_support = {
        "sorted_min": float(raw_scores_sorted[0]) if raw_scores_sorted else 0.0,
        "sorted_max": float(raw_scores_sorted[-1]) if raw_scores_sorted else 0.0,
        "sorted_p95": float(np.quantile(np.asarray(raw_scores_sorted, dtype=np.float32), 0.95)) if raw_scores_sorted else 0.0,
        "raw_minus_support_max": raw_score - (float(raw_scores_sorted[-1]) if raw_scores_sorted else 0.0),
    }
    debug_payload = {
        "raw_features": raw_features,
        "normalized_components": normalized_components,
        "weighted_contributions": weighted_contributions,
        "z_components": z_components,
        "missing_features": missing_features,
        "raw_score": raw_score,
        "raw_score_support": raw_score_support,
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
    return behavior_score, debug_payload


def behavior_score_from_raw_tensor(
    raw_score: torch.Tensor,
    normalizer: BehaviorNormalizer,
    score_key: str,
) -> torch.Tensor:
    if score_key == "behavior_aggressiveness_score_selected_agents":
        return normalizer.transform_raw_score_tensor(raw_score)
    if score_key == "behavior_quantile_score_selected_agents":
        return normalizer.transform_quantile_score_tensor(raw_score)
    if score_key == "behavior_raw_score_selected_agents":
        return raw_score
    raise ValueError(f"Unsupported behavior score key: {score_key}")


def inverse_behavior_score_tensor(
    score: torch.Tensor,
    normalizer: BehaviorNormalizer,
    score_key: str,
) -> torch.Tensor:
    if score_key == "behavior_aggressiveness_score_selected_agents":
        return normalizer.inverse_score_tensor(score)
    if score_key == "behavior_quantile_score_selected_agents":
        return normalizer.inverse_quantile_score_tensor(score)
    if score_key == "behavior_raw_score_selected_agents":
        return score
    raise ValueError(f"Unsupported behavior score key: {score_key}")


def behavior_score_bundle_from_raw(
    raw_score: float,
    normalizer: BehaviorNormalizer,
) -> Dict[str, float]:
    aggressiveness = normalizer.transform_raw_score(float(raw_score))
    quantile = normalizer.transform_quantile_score(float(raw_score))
    return {
        "raw": float(raw_score),
        "aggressiveness": float(aggressiveness),
        "quantile": float(quantile),
    }


def behavior_score_bundle_from_features(
    features: Dict[str, float],
    normalizer: BehaviorNormalizer,
) -> Dict[str, float]:
    aggressiveness, quantile, raw_score = normalizer.transform_scores(features)
    return {
        "raw": float(raw_score),
        "aggressiveness": float(aggressiveness),
        "quantile": float(quantile),
    }
