from __future__ import annotations

from typing import Tuple

import torch


ANCHOR_CHANNEL_NAMES = [
    "mid_x",
    "mid_y",
    "final_x",
    "final_y",
    "final_speed",
    "final_heading_delta",
]


def _wrap_heading_delta(delta: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(delta), torch.cos(delta))


def _agent_local_xy(
    current_states: torch.Tensor,
    xy_world: torch.Tensor,
) -> torch.Tensor:
    current_xy = current_states[..., 0:2]
    current_heading = current_states[..., 4]
    delta = xy_world - current_xy
    cos_h = torch.cos(current_heading)
    sin_h = torch.sin(current_heading)
    local_x = cos_h * delta[..., 0] + sin_h * delta[..., 1]
    local_y = -sin_h * delta[..., 0] + cos_h * delta[..., 1]
    return torch.stack([local_x, local_y], dim=-1)


def extract_anchor_targets(
    current_states: torch.Tensor,
    future_states: torch.Tensor,
    future_mask: torch.Tensor,
    agent_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    current_states = torch.nan_to_num(current_states, nan=0.0, posinf=0.0, neginf=0.0)
    future_states = torch.nan_to_num(future_states, nan=0.0, posinf=0.0, neginf=0.0)
    future_frames = int(future_states.shape[2])
    if future_frames <= 0:
        raise ValueError("extract_anchor_targets requires future_states with positive horizon.")
    mid_idx = max((future_frames - 1) // 2, 0)
    final_idx = future_frames - 1
    mid_state = future_states[:, :, mid_idx]
    final_state = future_states[:, :, final_idx]
    mid_local_xy = _agent_local_xy(current_states, mid_state[..., 0:2])
    final_local_xy = _agent_local_xy(current_states, final_state[..., 0:2])
    final_speed = torch.linalg.norm(final_state[..., 2:4], dim=-1, keepdim=True)
    final_heading_delta = _wrap_heading_delta(final_state[..., 4:5] - current_states[..., 4:5])
    anchor_target = torch.cat(
        [
            mid_local_xy,
            final_local_xy,
            final_speed,
            final_heading_delta,
        ],
        dim=-1,
    )
    anchor_valid_mask = agent_mask & future_mask[:, :, mid_idx] & future_mask[:, :, final_idx]
    anchor_target = torch.where(anchor_valid_mask.unsqueeze(-1), anchor_target, torch.zeros_like(anchor_target))
    anchor_target = torch.nan_to_num(anchor_target, nan=0.0, posinf=0.0, neginf=0.0)
    return anchor_target, anchor_valid_mask
