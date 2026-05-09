from __future__ import annotations

import math
from typing import Dict

import torch


def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return numerator / torch.clamp(denominator, min=1e-6)


def speed_from_velocity(states: torch.Tensor) -> torch.Tensor:
    states = torch.nan_to_num(states.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return torch.linalg.norm(states[..., 2:4], dim=-1)


def finite_difference(values: torch.Tensor, mask: torch.Tensor, dt: float) -> torch.Tensor:
    values = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)
    safe_values = torch.where(mask.unsqueeze(-1), values, torch.zeros_like(values))
    diffs = (safe_values[..., 1:, :] - safe_values[..., :-1, :]) / dt
    valid = mask[..., 1:] & mask[..., :-1]
    padded = torch.zeros_like(values)
    padded[..., 1:, :] = torch.where(valid.unsqueeze(-1), diffs, torch.zeros_like(diffs))
    return padded


def acceleration_from_states(states: torch.Tensor, mask: torch.Tensor, dt: float) -> torch.Tensor:
    velocities = states[..., 2:4]
    return finite_difference(velocities, mask, dt)


def jerk_from_states(states: torch.Tensor, mask: torch.Tensor, dt: float) -> torch.Tensor:
    accel = acceleration_from_states(states, mask, dt)
    accel_mask = mask
    return finite_difference(accel, accel_mask, dt)


def yaw_rate_from_states(states: torch.Tensor, mask: torch.Tensor, dt: float) -> torch.Tensor:
    states = torch.nan_to_num(states.float(), nan=0.0, posinf=0.0, neginf=0.0)
    headings = states[..., 4]
    delta = (headings[..., 1:] - headings[..., :-1] + math.pi) % (2 * math.pi) - math.pi
    valid = mask[..., 1:] & mask[..., :-1]
    yaw_rate = torch.zeros_like(headings)
    safe_delta = torch.where(valid, delta, torch.zeros_like(delta))
    yaw_rate[..., 1:] = _safe_divide(safe_delta, torch.full_like(delta, dt))
    return yaw_rate


def curvature_from_states(states: torch.Tensor, mask: torch.Tensor, dt: float) -> torch.Tensor:
    speed = speed_from_velocity(states)
    yaw_rate = yaw_rate_from_states(states, mask, dt)
    return _safe_divide(torch.abs(yaw_rate), speed + 1e-3)


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim=None) -> torch.Tensor:
    if dim is None:
        dim = tuple(range(mask.ndim))
    values = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)
    mask_f = mask.float()
    numerator = torch.where(mask, values, torch.zeros_like(values)).sum(dim=dim)
    denominator = torch.clamp(mask_f.sum(dim=dim), min=1.0)
    return numerator / denominator


def compute_trajectory_features(states: torch.Tensor, mask: torch.Tensor, dt: float) -> Dict[str, torch.Tensor]:
    speed = speed_from_velocity(states)
    accel = torch.linalg.norm(acceleration_from_states(states, mask, dt), dim=-1)
    jerk = torch.linalg.norm(jerk_from_states(states, mask, dt), dim=-1)
    yaw_rate = torch.abs(yaw_rate_from_states(states, mask, dt))
    curvature = curvature_from_states(states, mask, dt)
    return {
        "speed": speed,
        "accel": accel,
        "jerk": jerk,
        "yaw_rate": yaw_rate,
        "curvature": curvature,
    }
