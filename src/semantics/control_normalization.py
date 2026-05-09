from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch

from isgen import load_json, save_json

CONTROL_CHANNEL_NAMES = ["accel", "yaw_rate"]
CONTROL_NORMALIZER_VERSION = "2026-04-19-control-v1"


def _wrap_angle_np(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def _wrap_angle_torch(values: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(values), torch.cos(values))


def compute_raw_controls_numpy(
    current_states: np.ndarray,
    future_states: np.ndarray,
    future_mask: np.ndarray,
    agent_mask: np.ndarray,
    dt: float,
    accel_scale_mps2: float,
    yaw_rate_scale_radps: float,
) -> np.ndarray:
    future_valid = np.asarray(future_mask, dtype=bool) & np.asarray(agent_mask, dtype=bool)[:, None]
    future_states = np.asarray(future_states, dtype=np.float32)
    current_states = np.asarray(current_states, dtype=np.float32)
    speed = np.linalg.norm(future_states[..., 2:4], axis=-1)
    current_speed = np.linalg.norm(current_states[..., 2:4], axis=-1)
    accel = np.zeros_like(speed, dtype=np.float32)
    if speed.shape[1] > 0:
        accel[:, 0] = ((speed[:, 0] - current_speed) / dt) * future_valid[:, 0]
    if speed.shape[1] > 1:
        valid = future_valid[:, 1:] & future_valid[:, :-1]
        accel[:, 1:] = ((speed[:, 1:] - speed[:, :-1]) / dt) * valid
    headings = future_states[..., 4]
    current_heading = current_states[..., 4]
    yaw_rate = np.zeros_like(speed, dtype=np.float32)
    if headings.shape[1] > 0:
        yaw_rate[:, 0] = (_wrap_angle_np(headings[:, 0] - current_heading) / dt) * future_valid[:, 0]
    if headings.shape[1] > 1:
        valid = future_valid[:, 1:] & future_valid[:, :-1]
        yaw_rate[:, 1:] = (_wrap_angle_np(headings[:, 1:] - headings[:, :-1]) / dt) * valid
    raw = np.stack(
        [
            accel / max(float(accel_scale_mps2), 1e-6),
            yaw_rate / max(float(yaw_rate_scale_radps), 1e-6),
        ],
        axis=-1,
    )
    return raw * future_valid[..., None].astype(np.float32)


def compute_raw_controls_torch(
    current_states: torch.Tensor,
    future_states: torch.Tensor,
    future_mask: torch.Tensor,
    agent_mask: torch.Tensor,
    dt: float,
    accel_scale_mps2: float,
    yaw_rate_scale_radps: float,
) -> torch.Tensor:
    future_valid = future_mask & agent_mask.unsqueeze(-1)
    speed = torch.linalg.norm(future_states[..., 2:4], dim=-1)
    current_speed = torch.linalg.norm(current_states[..., 2:4], dim=-1)
    accel = torch.zeros_like(speed)
    if speed.shape[-1] > 0:
        accel[..., 0] = ((speed[..., 0] - current_speed) / dt) * future_valid[..., 0].float()
    if speed.shape[-1] > 1:
        valid = future_valid[..., 1:] & future_valid[..., :-1]
        accel[..., 1:] = ((speed[..., 1:] - speed[..., :-1]) / dt) * valid.float()
    headings = future_states[..., 4]
    current_heading = current_states[..., 4]
    yaw_rate = torch.zeros_like(speed)
    if headings.shape[-1] > 0:
        yaw_rate[..., 0] = _wrap_angle_torch(headings[..., 0] - current_heading) / dt * future_valid[..., 0].float()
    if headings.shape[-1] > 1:
        valid = future_valid[..., 1:] & future_valid[..., :-1]
        yaw_rate[..., 1:] = _wrap_angle_torch(headings[..., 1:] - headings[..., :-1]) / dt * valid.float()
    raw = torch.stack(
        [
            accel / max(float(accel_scale_mps2), 1e-6),
            yaw_rate / max(float(yaw_rate_scale_radps), 1e-6),
        ],
        dim=-1,
    )
    return raw * future_valid.unsqueeze(-1).float()


def _np_stats(values: np.ndarray, clamp_quantile: float) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "mean": 0.0,
            "std": 1.0,
            "median": 0.0,
            "p01": 0.0,
            "p05": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "abs_p99": 0.0,
            "abs_q_clamp": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "mean": float(values.mean()),
        "std": float(max(values.std(), 1e-6)),
        "median": float(np.median(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "abs_p99": float(np.quantile(np.abs(values), 0.99)),
        "abs_q_clamp": float(np.quantile(np.abs(values), float(clamp_quantile))),
        "min": float(values.min()),
        "max": float(values.max()),
    }


@dataclass
class ControlNormalizer:
    channel_names: List[str]
    channel_stats: Dict[str, Dict[str, float]]
    enabled: bool = True

    def normalize_controls(self, raw_controls: torch.Tensor) -> torch.Tensor:
        raw_controls = torch.nan_to_num(raw_controls, nan=0.0, posinf=0.0, neginf=0.0)
        if not self.enabled:
            return raw_controls
        outputs = []
        for channel_idx, channel_name in enumerate(self.channel_names):
            stats = self.channel_stats[channel_name]
            outputs.append((raw_controls[..., channel_idx] - float(stats["mean"])) / float(max(stats["std"], 1e-6)))
        return torch.nan_to_num(torch.stack(outputs, dim=-1), nan=0.0, posinf=0.0, neginf=0.0)

    def denormalize_controls(self, normalized_controls: torch.Tensor) -> torch.Tensor:
        normalized_controls = torch.nan_to_num(normalized_controls, nan=0.0, posinf=0.0, neginf=0.0)
        if not self.enabled:
            return normalized_controls
        outputs = []
        for channel_idx, channel_name in enumerate(self.channel_names):
            stats = self.channel_stats[channel_name]
            outputs.append(normalized_controls[..., channel_idx] * float(max(stats["std"], 1e-6)) + float(stats["mean"]))
        return torch.nan_to_num(torch.stack(outputs, dim=-1), nan=0.0, posinf=0.0, neginf=0.0)

    def clamp_raw_controls_to_train_quantiles(
        self,
        raw_controls: torch.Tensor,
        mode: str,
        margin: float,
    ) -> torch.Tensor:
        raw_controls = torch.nan_to_num(raw_controls, nan=0.0, posinf=0.0, neginf=0.0)
        if not self.enabled or mode == "none":
            return raw_controls
        outputs = []
        for channel_idx, channel_name in enumerate(self.channel_names):
            stats = self.channel_stats[channel_name]
            abs_limit = float(stats.get("abs_q_clamp", stats.get("abs_p99", 0.0))) * float(max(margin, 1.0))
            if abs_limit <= 0.0:
                outputs.append(raw_controls[..., channel_idx])
            else:
                outputs.append(torch.clamp(raw_controls[..., channel_idx], -abs_limit, abs_limit))
        return torch.nan_to_num(torch.stack(outputs, dim=-1), nan=0.0, posinf=0.0, neginf=0.0)

    def summary(self) -> Dict[str, Dict[str, float]]:
        return {key: dict(value) for key, value in self.channel_stats.items()}


def fit_control_normalizer(train_slices: Iterable[Dict[str, Any]], config: Dict[str, Any]) -> ControlNormalizer:
    model_cfg = config["model"]
    dt = float(config["data"]["timestep_sec"])
    accel_scale = float(model_cfg.get("accel_scale_mps2", 4.0))
    yaw_scale = float(model_cfg.get("yaw_rate_scale_radps", 1.5))
    clamp_quantile = float(model_cfg.get("control_clamp_quantile", 0.995))
    channel_values: Dict[str, List[np.ndarray]] = {name: [] for name in CONTROL_CHANNEL_NAMES}
    for slice_item in train_slices:
        raw_controls = compute_raw_controls_numpy(
            current_states=slice_item["current_states"],
            future_states=slice_item["future_states"],
            future_mask=slice_item["future_mask"],
            agent_mask=slice_item["agent_mask"],
            dt=dt,
            accel_scale_mps2=accel_scale,
            yaw_rate_scale_radps=yaw_scale,
        )
        valid = np.asarray(slice_item["future_mask"], dtype=bool) & np.asarray(slice_item["agent_mask"], dtype=bool)[:, None]
        for channel_idx, channel_name in enumerate(CONTROL_CHANNEL_NAMES):
            channel_values[channel_name].append(raw_controls[..., channel_idx][valid].astype(np.float32))
    stats = {
        channel_name: _np_stats(np.concatenate(values) if values else np.zeros((0,), dtype=np.float32), clamp_quantile)
        for channel_name, values in channel_values.items()
    }
    return ControlNormalizer(channel_names=list(CONTROL_CHANNEL_NAMES), channel_stats=stats, enabled=True)


def save_control_stats(normalizer: ControlNormalizer, path: str | Path) -> None:
    save_json(
        {
            "version": CONTROL_NORMALIZER_VERSION,
            "channel_names": normalizer.channel_names,
            "channel_stats": normalizer.channel_stats,
            "enabled": normalizer.enabled,
        },
        path,
    )


def load_control_normalizer(path: str | Path) -> ControlNormalizer:
    payload = load_json(path)
    return ControlNormalizer(
        channel_names=list(payload["channel_names"]),
        channel_stats={key: dict(value) for key, value in payload["channel_stats"].items()},
        enabled=bool(payload.get("enabled", True)),
    )


def summarize_control_tensor(raw_controls: torch.Tensor, valid_mask: torch.Tensor, channel_names: List[str] | None = None) -> Dict[str, Dict[str, float]]:
    channel_names = channel_names or CONTROL_CHANNEL_NAMES
    stats: Dict[str, Dict[str, float]] = {}
    valid_mask = valid_mask.bool()
    for channel_idx, channel_name in enumerate(channel_names):
        values = raw_controls[..., channel_idx][valid_mask].detach().float()
        if values.numel() == 0:
            stats[channel_name] = {"mean": 0.0, "std": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
            continue
        stats[channel_name] = {
            "mean": float(values.mean().item()),
            "std": float(values.std(unbiased=False).item()),
            "p95": float(torch.quantile(values, 0.95).item()),
            "p99": float(torch.quantile(values, 0.99).item()),
            "min": float(values.min().item()),
            "max": float(values.max().item()),
        }
    return stats
