from __future__ import annotations

import torch


CURRENT_PAIR_FEATURE_NAMES: tuple[str, ...] = (
    "current_rel_dx",
    "current_rel_dy",
    "current_pair_distance",
    "current_rel_vx",
    "current_rel_vy",
    "current_relative_speed",
    "heading_delta_sin",
    "heading_delta_cos",
    "approach_rate",
)


def _safe_norm(value: torch.Tensor, dim: int, eps: float = 1e-6) -> torch.Tensor:
    return torch.sqrt(value.square().sum(dim=dim) + eps)


def extract_current_pair_features(
    current_states: torch.Tensor,
    agent_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    safe_current = torch.nan_to_num(current_states, nan=0.0, posinf=0.0, neginf=0.0)
    num_agents = int(agent_mask.shape[1])
    eye = torch.eye(num_agents, device=agent_mask.device, dtype=torch.bool).unsqueeze(0)
    pair_valid = agent_mask.unsqueeze(2) & agent_mask.unsqueeze(1) & (~eye)

    rel_xy = safe_current[:, :, None, 0:2] - safe_current[:, None, :, 0:2]
    pair_distance = _safe_norm(rel_xy, dim=-1)
    rel_vel = safe_current[:, :, None, 2:4] - safe_current[:, None, :, 2:4]
    relative_speed = _safe_norm(rel_vel, dim=-1)

    heading_delta = safe_current[:, :, None, 4] - safe_current[:, None, :, 4]
    heading_delta_sin = torch.sin(heading_delta)
    heading_delta_cos = torch.cos(heading_delta)

    safe_distance = torch.clamp(pair_distance, min=1e-3)
    approach_rate = -((rel_xy * rel_vel).sum(dim=-1) / safe_distance)

    features = torch.stack(
        [
            rel_xy[..., 0],
            rel_xy[..., 1],
            pair_distance,
            rel_vel[..., 0],
            rel_vel[..., 1],
            relative_speed,
            heading_delta_sin,
            heading_delta_cos,
            approach_rate,
        ],
        dim=-1,
    )
    features = torch.where(pair_valid.unsqueeze(-1), features, torch.zeros_like(features))
    return torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0), pair_valid
