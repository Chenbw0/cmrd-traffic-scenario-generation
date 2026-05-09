from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F


def knot_segment_bounds(future_frames: int, num_control_knots: int) -> list[tuple[int, int]]:
    if future_frames <= 0:
        return []
    if num_control_knots <= 0:
        raise ValueError("num_control_knots must be positive.")
    bounds: list[tuple[int, int]] = []
    edges = torch.linspace(0, future_frames, num_control_knots + 1).round().long().tolist()
    for knot_idx in range(num_control_knots):
        start = min(int(edges[knot_idx]), future_frames - 1)
        end = min(max(int(edges[knot_idx + 1]), start + 1), future_frames)
        bounds.append((start, end))
    return bounds


def controls_to_knots(
    controls: torch.Tensor,
    future_mask: torch.Tensor,
    agent_mask: torch.Tensor,
    num_control_knots: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if controls.ndim != 4 or controls.shape[-1] != 2:
        raise ValueError("controls_to_knots expects controls with shape [B, A, T, 2].")
    controls = torch.nan_to_num(controls, nan=0.0, posinf=0.0, neginf=0.0)
    batch_size, num_agents, future_frames, control_dim = controls.shape
    valid = future_mask & agent_mask.unsqueeze(-1)
    knots = torch.zeros(
        (batch_size, num_agents, num_control_knots, control_dim),
        device=controls.device,
        dtype=controls.dtype,
    )
    knot_mask = torch.zeros(
        (batch_size, num_agents, num_control_knots),
        device=controls.device,
        dtype=torch.bool,
    )
    for knot_idx, (start, end) in enumerate(knot_segment_bounds(future_frames, num_control_knots)):
        segment_controls = controls[:, :, start:end]
        segment_valid = valid[:, :, start:end]
        count = torch.clamp(segment_valid.float().sum(dim=2, keepdim=True), min=1.0)
        averaged = (segment_controls * segment_valid.unsqueeze(-1).float()).sum(dim=2) / count
        knots[:, :, knot_idx] = averaged
        knot_mask[:, :, knot_idx] = segment_valid.any(dim=2)
    knots = knots * knot_mask.unsqueeze(-1).float()
    return knots, knot_mask


def _linear_interpolate_knots(knots: torch.Tensor, future_frames: int) -> torch.Tensor:
    knots = torch.nan_to_num(knots, nan=0.0, posinf=0.0, neginf=0.0)
    if knots.shape[2] == future_frames:
        return knots
    if knots.shape[2] == 1:
        return knots.expand(-1, -1, future_frames, -1)
    reshaped = knots.permute(0, 1, 3, 2).reshape(-1, knots.shape[-1], knots.shape[2])
    interpolated = F.interpolate(reshaped, size=future_frames, mode="linear", align_corners=True)
    return interpolated.reshape(knots.shape[0], knots.shape[1], knots.shape[-1], future_frames).permute(0, 1, 3, 2)


def _catmull_rom_point(
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    tau2 = tau * tau
    tau3 = tau2 * tau
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * tau
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * tau2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * tau3
    )


def _cubic_interpolate_knots(knots: torch.Tensor, future_frames: int) -> torch.Tensor:
    knots = torch.nan_to_num(knots, nan=0.0, posinf=0.0, neginf=0.0)
    if knots.shape[2] == future_frames:
        return knots
    if knots.shape[2] == 1:
        return knots.expand(-1, -1, future_frames, -1)
    batch_size, num_agents, num_knots, control_dim = knots.shape
    knot_positions = torch.linspace(0.0, float(max(future_frames - 1, 0)), num_knots, device=knots.device, dtype=knots.dtype)
    frame_positions = torch.linspace(0.0, float(max(future_frames - 1, 0)), future_frames, device=knots.device, dtype=knots.dtype)
    outputs = []
    for frame_pos in frame_positions:
        right = int(torch.searchsorted(knot_positions, frame_pos, right=False).item())
        if right <= 0:
            outputs.append(knots[:, :, 0])
            continue
        if right >= num_knots:
            outputs.append(knots[:, :, -1])
            continue
        left = right - 1
        span = torch.clamp(knot_positions[right] - knot_positions[left], min=1e-6)
        tau = (frame_pos - knot_positions[left]) / span
        tau_tensor = torch.full((batch_size, num_agents, control_dim), float(tau.item()), device=knots.device, dtype=knots.dtype)
        p0 = knots[:, :, max(left - 1, 0)]
        p1 = knots[:, :, left]
        p2 = knots[:, :, right]
        p3 = knots[:, :, min(right + 1, num_knots - 1)]
        outputs.append(_catmull_rom_point(p0, p1, p2, p3, tau_tensor))
    return torch.stack(outputs, dim=2)


def interpolate_control_knots(
    knots: torch.Tensor,
    future_frames: int,
    mode: str = "linear",
) -> torch.Tensor:
    if future_frames <= 0:
        raise ValueError("future_frames must be positive.")
    if knots.ndim != 4 or knots.shape[-1] != 2:
        raise ValueError("interpolate_control_knots expects knots with shape [B, A, K, 2].")
    if mode == "linear":
        return _linear_interpolate_knots(knots, future_frames)
    if mode == "cubic":
        return _cubic_interpolate_knots(knots, future_frames)
    raise ValueError(f"Unsupported control_interpolation mode: {mode}")


def knot_control_delta_loss(
    interpolated_controls: torch.Tensor,
    future_mask: torch.Tensor,
    agent_mask: torch.Tensor,
    jerk_weight: float,
    yaw_accel_weight: float,
) -> torch.Tensor:
    interpolated_controls = torch.nan_to_num(interpolated_controls, nan=0.0, posinf=0.0, neginf=0.0)
    valid = future_mask & agent_mask.unsqueeze(-1)
    if interpolated_controls.shape[2] <= 1:
        return torch.zeros((), device=interpolated_controls.device, dtype=interpolated_controls.dtype)
    safe_controls = torch.where(valid.unsqueeze(-1), interpolated_controls, torch.zeros_like(interpolated_controls))
    delta_valid = valid[:, :, 1:] & valid[:, :, :-1]
    accel_delta = safe_controls[..., 0][:, :, 1:] - safe_controls[..., 0][:, :, :-1]
    yaw_delta = safe_controls[..., 1][:, :, 1:] - safe_controls[..., 1][:, :, :-1]
    denom = torch.clamp(delta_valid.float().sum(), min=1.0)
    jerk_term = torch.where(delta_valid, accel_delta.square(), torch.zeros_like(accel_delta)).sum() / denom
    yaw_term = torch.where(delta_valid, yaw_delta.square(), torch.zeros_like(yaw_delta)).sum() / denom
    return float(jerk_weight) * jerk_term + float(yaw_accel_weight) * yaw_term
