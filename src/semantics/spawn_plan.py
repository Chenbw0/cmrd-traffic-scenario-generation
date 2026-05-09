from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn.functional as F


SUMMARY_FEATURE_NAMES = [
    "mean_speed_norm",
    "mean_radius_norm",
    "mean_pair_distance_norm",
    "min_pairwise_distance_norm",
    "conflict_ratio",
    "heading_std",
]


def _safe_norm(value: torch.Tensor, dim: int, eps: float = 1e-6) -> torch.Tensor:
    return torch.sqrt(value.square().sum(dim=dim) + eps)


def spawn_plan_feature_names(radial_bins: int, angular_bins: int) -> List[str]:
    names = list(SUMMARY_FEATURE_NAMES)
    for radial_idx in range(radial_bins):
        for angular_idx in range(angular_bins):
            names.append(f"occupancy_r{radial_idx}_a{angular_idx}")
    return names


def spawn_plan_feature_dim(radial_bins: int, angular_bins: int) -> int:
    return len(spawn_plan_feature_names(radial_bins, angular_bins))


def extract_spawn_plan_targets(
    current_states: torch.Tensor,
    agent_mask: torch.Tensor,
    map_radius_m: float,
    max_speed_mps: float,
    conflict_distance_m: float,
    radial_bins: int,
    angular_bins: int,
) -> Dict[str, torch.Tensor]:
    if radial_bins <= 0 or angular_bins <= 0:
        raise ValueError("radial_bins and angular_bins must be positive.")
    device = current_states.device
    dtype = current_states.dtype
    mask_f = agent_mask.float()
    count = mask_f.sum(dim=-1)
    safe_count = count.clamp(min=1.0)

    positions = current_states[..., 0:2]
    speed = _safe_norm(current_states[..., 2:4], dim=-1)
    radius = _safe_norm(positions, dim=-1)

    mean_speed = (speed * mask_f).sum(dim=-1) / safe_count
    mean_radius = (radius * mask_f).sum(dim=-1) / safe_count

    pair_distance = _safe_norm(positions.unsqueeze(2) - positions.unsqueeze(1), dim=-1)
    eye = torch.eye(agent_mask.shape[1], device=device, dtype=torch.bool).unsqueeze(0)
    pair_valid = agent_mask.unsqueeze(2) & agent_mask.unsqueeze(1) & ~eye
    pair_count = pair_valid.float().sum(dim=(-1, -2)).clamp(min=1.0)
    mean_pair_distance = torch.where(pair_valid, pair_distance, torch.zeros_like(pair_distance)).sum(dim=(-1, -2)) / pair_count
    inf = torch.full_like(pair_distance, 1e3)
    min_pair_distance = torch.where(pair_valid, pair_distance, inf).amin(dim=(-1, -2))
    has_pair = pair_valid.any(dim=(-1, -2))
    min_pair_distance = torch.where(has_pair, min_pair_distance, torch.full_like(min_pair_distance, float(map_radius_m)))
    conflict_pairs = (pair_valid & (pair_distance < float(conflict_distance_m))).float().sum(dim=(-1, -2)) * 0.5
    max_conflict_pairs = (count * (count - 1.0) * 0.5).clamp(min=1.0)
    conflict_ratio = conflict_pairs / max_conflict_pairs

    heading = current_states[..., 4]
    heading_float = heading.float()
    heading_complex = torch.polar(torch.ones_like(heading_float), heading_float)
    heading_mean = (heading_complex * mask_f).sum(dim=-1) / safe_count
    # Use a smooth circular-dispersion proxy. The previous sqrt-based form had
    # an infinite derivative at zero and could produce non-finite gradients.
    heading_std = torch.clamp(1.0 - torch.abs(heading_mean), min=0.0, max=1.0).to(dtype=dtype)

    radius_norm = torch.clamp(radius / max(float(map_radius_m), 1e-6), min=0.0, max=0.999999)
    angle = torch.atan2(positions[..., 1], positions[..., 0])
    angle_norm = torch.remainder(angle + math.pi, 2.0 * math.pi) / (2.0 * math.pi)
    radial_bin = torch.clamp((radius_norm * float(radial_bins)).long(), min=0, max=radial_bins - 1)
    angular_bin = torch.clamp((angle_norm * float(angular_bins)).long(), min=0, max=angular_bins - 1)
    combined_bin = radial_bin * angular_bins + angular_bin
    occupancy = F.one_hot(combined_bin, num_classes=radial_bins * angular_bins).to(dtype=dtype)
    occupancy = occupancy * mask_f.unsqueeze(-1)
    occupancy = occupancy.sum(dim=1) / safe_count.unsqueeze(-1)

    summary = torch.stack(
        [
            torch.clamp(mean_speed / max(float(max_speed_mps), 1e-6), min=0.0, max=1.0),
            torch.clamp(mean_radius / max(float(map_radius_m), 1e-6), min=0.0, max=1.0),
            torch.clamp(mean_pair_distance / max(float(map_radius_m) * 2.0, 1e-6), min=0.0, max=1.0),
            torch.clamp(min_pair_distance / max(float(map_radius_m), 1e-6), min=0.0, max=1.0),
            torch.clamp(conflict_ratio, min=0.0, max=1.0),
            torch.clamp(heading_std, min=0.0, max=1.0),
        ],
        dim=-1,
    ).to(dtype=dtype)
    plan_features = torch.cat([summary, occupancy.to(dtype=dtype)], dim=-1)
    return {
        "count_targets": count.long(),
        "summary_features": summary,
        "occupancy_features": occupancy.to(dtype=dtype),
        "plan_features": plan_features,
    }
