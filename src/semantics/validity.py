from __future__ import annotations

import torch


def pairwise_distances(xy: torch.Tensor) -> torch.Tensor:
    diff = xy.unsqueeze(2) - xy.unsqueeze(1)
    return torch.linalg.norm(diff, dim=-1)


def collision_penalty(future_states: torch.Tensor, future_mask: torch.Tensor, agent_mask: torch.Tensor, overlap_threshold: float = 1.5) -> torch.Tensor:
    xy = torch.nan_to_num(future_states[..., 0:2].float(), nan=0.0, posinf=0.0, neginf=0.0)
    distances = pairwise_distances(xy)
    valid = future_mask.unsqueeze(2) & future_mask.unsqueeze(1)
    valid = valid & agent_mask.unsqueeze(1).unsqueeze(-1) & agent_mask.unsqueeze(2).unsqueeze(-1)
    eye = torch.eye(distances.size(1), dtype=torch.bool, device=distances.device).unsqueeze(0).unsqueeze(-1)
    distances = distances.masked_fill(eye, 1e6)
    penalties = torch.where(valid, torch.relu(overlap_threshold - distances), torch.zeros_like(distances))
    return penalties.mean()


def offroad_distance_proxy(future_states: torch.Tensor, future_mask: torch.Tensor, map_polylines: torch.Tensor, map_point_mask: torch.Tensor) -> torch.Tensor:
    if map_polylines.numel() == 0 or not map_point_mask.any():
        return torch.zeros((), device=future_states.device)
    future_xy = torch.nan_to_num(future_states[..., 0:2].float(), nan=0.0, posinf=0.0, neginf=0.0).reshape(future_states.shape[0], -1, 2)
    future_valid = future_mask.reshape(future_mask.shape[0], -1)
    map_xy = torch.nan_to_num(map_polylines[..., 0:2].float(), nan=0.0, posinf=0.0, neginf=0.0).reshape(map_polylines.shape[0], -1, 2)
    map_valid = map_point_mask.reshape(map_point_mask.shape[0], -1)
    batch_penalties = []
    for batch_idx in range(future_states.shape[0]):
        if not future_valid[batch_idx].any() or not map_valid[batch_idx].any():
            batch_penalties.append(torch.zeros((), device=future_states.device))
            continue
        points = future_xy[batch_idx][future_valid[batch_idx]]
        refs = map_xy[batch_idx][map_valid[batch_idx]]
        distances = torch.cdist(points, refs)
        batch_penalties.append(torch.relu(distances.min(dim=-1).values - 4.0).mean())
    return torch.stack(batch_penalties).mean()
