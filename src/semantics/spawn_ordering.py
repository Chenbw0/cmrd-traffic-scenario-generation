from __future__ import annotations

import math
from typing import Tuple

import torch


def safe_norm(value: torch.Tensor, dim: int, eps: float = 1e-6) -> torch.Tensor:
    return torch.sqrt(value.square().sum(dim=dim) + eps)


def canonical_sort_scene(
    states: torch.Tensor,
    valid_mask: torch.Tensor,
    scores: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    positions = states[..., 0:2]
    radius = safe_norm(positions, dim=-1)
    angle = torch.atan2(positions[..., 1], positions[..., 0])
    angle_norm = torch.remainder(angle + math.pi, 2.0 * math.pi) / (2.0 * math.pi)
    speed = safe_norm(states[..., 2:4], dim=-1)

    # Deterministic geometric order for unordered current-scene sets.
    key = radius * 1_000.0 + angle_norm * 10.0 + speed
    key = torch.where(valid_mask, key, torch.full_like(key, 1e9))
    order = torch.argsort(key, dim=-1)
    gather_state = order.unsqueeze(-1).expand(-1, -1, states.shape[-1])
    sorted_states = torch.gather(states, dim=1, index=gather_state)
    sorted_mask = torch.gather(valid_mask, dim=1, index=order)
    if scores is None:
        return sorted_states, sorted_mask, None
    sorted_scores = torch.gather(scores, dim=1, index=order)
    return sorted_states, sorted_mask, sorted_scores
