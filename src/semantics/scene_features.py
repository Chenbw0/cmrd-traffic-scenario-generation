from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from isgen.semantics.kinematics import compute_trajectory_features, masked_mean


def _expand_reduced_tensor(values: torch.Tensor, reduced: torch.Tensor, dim) -> torch.Tensor:
    dims = dim if isinstance(dim, tuple) else (dim,)
    positive_dims = sorted({axis if axis >= 0 else values.ndim + axis for axis in dims})
    expanded = reduced
    for axis in positive_dims:
        expanded = expanded.unsqueeze(axis)
    return expanded


def masked_std(values: torch.Tensor, mask: torch.Tensor, dim=None) -> torch.Tensor:
    if dim is None:
        dim = tuple(range(mask.ndim))
    values = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)
    mean = masked_mean(values, mask, dim=dim)
    mean_expanded = _expand_reduced_tensor(values, mean, dim)
    mask_f = mask.float()
    centered_sq = torch.where(mask, (values - mean_expanded).square(), torch.zeros_like(values))
    variance = centered_sq.sum(dim=dim) / torch.clamp(mask_f.sum(dim=dim), min=1.0)
    return variance.sqrt()


def flatten_behavior_features(future_states: torch.Tensor, future_mask: torch.Tensor, current_states: torch.Tensor, agent_mask: torch.Tensor, dt: float) -> torch.Tensor:
    traj_features = compute_trajectory_features(future_states, future_mask, dt)
    speed = masked_mean(traj_features["speed"], future_mask, dim=(-1, -2))
    accel = masked_mean(traj_features["accel"], future_mask, dim=(-1, -2))
    jerk = masked_mean(traj_features["jerk"], future_mask, dim=(-1, -2))
    yaw_rate = masked_mean(traj_features["yaw_rate"], future_mask, dim=(-1, -2))
    final_displacement = torch.linalg.norm(future_states[:, :, -1, 0:2] - current_states[:, :, 0:2], dim=-1)
    final_mask = future_mask[:, :, -1] & agent_mask
    final_disp_mean = masked_mean(final_displacement, final_mask, dim=-1)
    return torch.stack([speed, accel, jerk, yaw_rate, final_disp_mean], dim=-1)


def flatten_realism_features(
    future_states: torch.Tensor,
    future_mask: torch.Tensor,
    current_states: torch.Tensor,
    agent_mask: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    future_states = torch.nan_to_num(future_states.float(), nan=0.0, posinf=0.0, neginf=0.0)
    current_states = torch.nan_to_num(current_states.float(), nan=0.0, posinf=0.0, neginf=0.0)
    future_valid = future_mask & agent_mask.unsqueeze(-1)
    traj_features = compute_trajectory_features(future_states, future_valid, dt)
    speed_mean = masked_mean(traj_features["speed"], future_valid, dim=(-1, -2))
    speed_std = masked_std(traj_features["speed"], future_valid, dim=(-1, -2))
    accel_mean = masked_mean(traj_features["accel"], future_valid, dim=(-1, -2))
    accel_std = masked_std(traj_features["accel"], future_valid, dim=(-1, -2))
    jerk_mean = masked_mean(traj_features["jerk"], future_valid, dim=(-1, -2))
    jerk_std = masked_std(traj_features["jerk"], future_valid, dim=(-1, -2))
    yaw_rate_mean = masked_mean(traj_features["yaw_rate"], future_valid, dim=(-1, -2))
    yaw_rate_std = masked_std(traj_features["yaw_rate"], future_valid, dim=(-1, -2))
    final_mask = future_valid[:, :, -1]
    final_index = torch.clamp(future_valid.float().sum(dim=-1).long() - 1, min=0)
    gather_index = final_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, future_states.shape[-1])
    final_states = torch.gather(future_states, dim=2, index=gather_index).squeeze(2)
    final_displacement = torch.linalg.norm(final_states[..., 0:2] - current_states[..., 0:2], dim=-1)
    final_disp_mean = masked_mean(final_displacement, agent_mask, dim=-1)
    final_disp_std = masked_std(final_displacement, agent_mask, dim=-1)

    final_xy = final_states[..., 0:2]
    pairwise = torch.cdist(final_xy, final_xy)
    eye = torch.eye(pairwise.shape[-1], dtype=torch.bool, device=pairwise.device).unsqueeze(0)
    pair_valid = final_mask.unsqueeze(1) & final_mask.unsqueeze(2) & (~eye)
    pairwise_valid = torch.where(pair_valid, pairwise, torch.full_like(pairwise, 1e3))
    min_pairwise_distance = pairwise_valid.amin(dim=(-1, -2))
    any_pairs = pair_valid.any(dim=(-1, -2))
    min_pairwise_distance = torch.where(any_pairs, min_pairwise_distance, torch.full_like(min_pairwise_distance, 1e3))
    close_pair_rate = torch.where(
        pair_valid,
        torch.sigmoid((5.0 - pairwise) / 0.5),
        torch.zeros_like(pairwise),
    ).sum(dim=(-1, -2)) / torch.clamp(pair_valid.float().sum(dim=(-1, -2)), min=1.0)

    features = torch.stack(
        [
            speed_mean,
            speed_std,
            accel_mean,
            accel_std,
            jerk_mean,
            jerk_std,
            yaw_rate_mean,
            yaw_rate_std,
            final_disp_mean,
            final_disp_std,
            min_pairwise_distance,
            close_pair_rate,
        ],
        dim=-1,
    )
    return torch.nan_to_num(features, nan=0.0, posinf=1e3, neginf=-1e3)


def slice_scene_summary(slice_item: Dict) -> Dict[str, float]:
    current_states = np.asarray(slice_item["current_states"], dtype=np.float32)
    agent_mask = np.asarray(slice_item["agent_mask"], dtype=bool)
    valid_states = current_states[agent_mask]
    if len(valid_states) == 0:
        return {"num_agents": 0.0, "mean_speed": 0.0, "density": 0.0}
    speed = np.linalg.norm(valid_states[:, 2:4], axis=-1)
    density = float(np.sum(np.linalg.norm(valid_states[:, 0:2], axis=-1) <= 30.0))
    return {"num_agents": float(len(valid_states)), "mean_speed": float(speed.mean()), "density": density}
