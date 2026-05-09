from __future__ import annotations

from typing import Sequence

import torch

from isgen.semantics.anchors import extract_anchor_targets


STATIC_ORACLE_PAIR_FEATURE_NAMES: tuple[str, ...] = (
    "future_min_distance_ij",
    "time_to_min_distance_norm",
    "relative_speed_at_closest_approach",
    "relative_dx_at_closest_approach",
    "relative_dy_at_closest_approach",
    "final_pair_distance_ij",
    "distance_change",
    "soft_conflict_score",
)


def _pairwise_valid_mask(future_mask: torch.Tensor, agent_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    overlap = future_mask.unsqueeze(2) & future_mask.unsqueeze(1)
    num_agents = int(agent_mask.shape[1])
    eye = torch.eye(num_agents, device=agent_mask.device, dtype=torch.bool).unsqueeze(0)
    pair_valid = agent_mask.unsqueeze(2) & agent_mask.unsqueeze(1) & (~eye) & overlap.any(dim=-1)
    return pair_valid, overlap


def _gather_time(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    expanded = indices.unsqueeze(-1).unsqueeze(-1).expand(*indices.shape, 1, values.shape[-1])
    return torch.take_along_dim(values, expanded, dim=3).squeeze(3)


def interaction_oracle_feature_dim(mode: str, num_trace_points: int = 5) -> int:
    mode = str(mode)
    if mode == "none":
        return 0
    if mode == "static_summary":
        return len(STATIC_ORACLE_PAIR_FEATURE_NAMES)
    if mode == "dynamic_trace":
        return int(num_trace_points) * 4
    if mode == "teacher_forced_neighbors":
        return 6
    raise ValueError(f"Unsupported interaction_oracle mode: {mode}")


def extract_static_oracle_pair_features(
    current_states: torch.Tensor,
    future_states: torch.Tensor,
    future_mask: torch.Tensor,
    agent_mask: torch.Tensor,
    conflict_distance_m: float = 5.0,
    conflict_temperature_m: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    safe_current = torch.nan_to_num(current_states, nan=0.0, posinf=0.0, neginf=0.0)
    safe_future = torch.nan_to_num(future_states, nan=0.0, posinf=0.0, neginf=0.0)
    pair_valid, overlap = _pairwise_valid_mask(future_mask, agent_mask)
    batch_size, num_agents, horizon = future_mask.shape

    rel_xy = safe_future[:, :, None, :, 0:2] - safe_future[:, None, :, :, 0:2]
    pair_distance = torch.linalg.norm(rel_xy, dim=-1)
    inf = torch.full_like(pair_distance, float("inf"))
    masked_distance = torch.where(overlap, pair_distance, inf)
    future_min_distance, min_idx = masked_distance.min(dim=-1)
    future_min_distance = torch.where(pair_valid, future_min_distance, torch.zeros_like(future_min_distance))
    min_idx = torch.where(pair_valid, min_idx, torch.zeros_like(min_idx))

    horizon_scale = max(horizon - 1, 1)
    time_to_min_distance = min_idx.float() / float(horizon_scale)
    time_to_min_distance = torch.where(pair_valid, time_to_min_distance, torch.zeros_like(time_to_min_distance))

    rel_xy_at_min = _gather_time(rel_xy, min_idx)
    rel_vel = safe_future[:, :, None, :, 2:4] - safe_future[:, None, :, :, 2:4]
    rel_vel_at_min = _gather_time(rel_vel, min_idx)
    relative_speed_at_min = torch.linalg.norm(rel_vel_at_min, dim=-1)

    time_index = torch.arange(horizon, device=future_mask.device, dtype=torch.long).view(1, 1, 1, horizon)
    final_idx = torch.where(overlap, time_index, torch.full_like(time_index, -1)).max(dim=-1).values
    final_idx = torch.where(pair_valid, final_idx, torch.zeros_like(final_idx))
    final_rel_xy = _gather_time(rel_xy, final_idx)
    final_pair_distance = torch.linalg.norm(final_rel_xy, dim=-1)

    current_rel_xy = safe_current[:, :, None, 0:2] - safe_current[:, None, :, 0:2]
    current_pair_distance = torch.linalg.norm(current_rel_xy, dim=-1)
    distance_change = final_pair_distance - current_pair_distance

    temperature = max(float(conflict_temperature_m), 1e-3)
    soft_conflict_score = torch.sigmoid((float(conflict_distance_m) - future_min_distance) / temperature)

    features = torch.stack(
        [
            future_min_distance,
            time_to_min_distance,
            relative_speed_at_min,
            rel_xy_at_min[..., 0],
            rel_xy_at_min[..., 1],
            final_pair_distance,
            distance_change,
            soft_conflict_score,
        ],
        dim=-1,
    )
    features = torch.where(pair_valid.unsqueeze(-1), features, torch.zeros_like(features))
    return torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0), pair_valid


def extract_dynamic_oracle_pair_features(
    current_states: torch.Tensor,
    future_states: torch.Tensor,
    future_mask: torch.Tensor,
    agent_mask: torch.Tensor,
    num_trace_points: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    safe_future = torch.nan_to_num(future_states, nan=0.0, posinf=0.0, neginf=0.0)
    pair_valid, overlap = _pairwise_valid_mask(future_mask, agent_mask)
    horizon = int(future_mask.shape[-1])
    num_trace_points = max(int(num_trace_points), 1)
    trace_indices = torch.linspace(0, max(horizon - 1, 0), steps=num_trace_points, device=future_states.device).round().long()

    pair_distance = torch.linalg.norm(
        safe_future[:, :, None, :, 0:2] - safe_future[:, None, :, :, 0:2],
        dim=-1,
    )
    rel_xy = safe_future[:, :, None, :, 0:2] - safe_future[:, None, :, :, 0:2]
    rel_vel = safe_future[:, :, None, :, 2:4] - safe_future[:, None, :, :, 2:4]
    rel_speed = torch.linalg.norm(rel_vel, dim=-1)

    sampled_distance = pair_distance.index_select(dim=3, index=trace_indices)
    sampled_rel_xy = rel_xy.index_select(dim=3, index=trace_indices)
    sampled_rel_speed = rel_speed.index_select(dim=3, index=trace_indices)
    sampled_overlap = overlap.index_select(dim=3, index=trace_indices)

    feature_blocks = []
    for trace_idx in range(num_trace_points):
        distance_t = sampled_distance[..., trace_idx]
        rel_xy_t = sampled_rel_xy[..., trace_idx, :]
        rel_speed_t = sampled_rel_speed[..., trace_idx]
        valid_t = sampled_overlap[..., trace_idx] & pair_valid
        block = torch.stack(
            [
                torch.where(valid_t, distance_t, torch.zeros_like(distance_t)),
                torch.where(valid_t, rel_xy_t[..., 0], torch.zeros_like(distance_t)),
                torch.where(valid_t, rel_xy_t[..., 1], torch.zeros_like(distance_t)),
                torch.where(valid_t, rel_speed_t, torch.zeros_like(distance_t)),
            ],
            dim=-1,
        )
        feature_blocks.append(block)
    features = torch.cat(feature_blocks, dim=-1)
    features = torch.where(pair_valid.unsqueeze(-1), features, torch.zeros_like(features))
    return torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0), pair_valid


def extract_teacher_forced_neighbor_summaries(
    current_states: torch.Tensor,
    future_states: torch.Tensor,
    future_mask: torch.Tensor,
    agent_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    anchor_target, anchor_valid_mask = extract_anchor_targets(
        current_states=current_states,
        future_states=future_states,
        future_mask=future_mask,
        agent_mask=agent_mask,
    )
    summaries = torch.where(anchor_valid_mask.unsqueeze(-1), anchor_target, torch.zeros_like(anchor_target))
    return torch.nan_to_num(summaries, nan=0.0, posinf=0.0, neginf=0.0), anchor_valid_mask


def oracle_mode_uses_pair_features(mode: str) -> bool:
    return str(mode) in {"static_summary", "dynamic_trace"}


def oracle_mode_uses_teacher_forced_neighbors(mode: str) -> bool:
    return str(mode) == "teacher_forced_neighbors"

