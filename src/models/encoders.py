from __future__ import annotations

import torch
from torch import nn

from isgen.models.blocks import FourierEmbedding, MLP


class AgentHistoryEncoder(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, num_layers: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.current_proj = nn.Linear(state_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, history_states: torch.Tensor, history_mask: torch.Tensor, current_states: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
        bsz, num_agents, hist_len, state_dim = history_states.shape
        safe_history_states = torch.nan_to_num(history_states, nan=0.0, posinf=0.0, neginf=0.0)
        safe_history_states = torch.where(history_mask.unsqueeze(-1), safe_history_states, torch.zeros_like(safe_history_states))
        safe_current_states = torch.nan_to_num(current_states, nan=0.0, posinf=0.0, neginf=0.0)
        safe_current_states = torch.where(agent_mask.unsqueeze(-1), safe_current_states, torch.zeros_like(safe_current_states))
        history_flat = safe_history_states.reshape(bsz * num_agents, hist_len, state_dim)
        mask_flat = history_mask.reshape(bsz * num_agents, hist_len)
        safe_mask = mask_flat.clone()
        empty_rows = safe_mask.sum(dim=1) == 0
        safe_mask[empty_rows, 0] = True
        encoded = self.state_proj(history_flat)
        transformed = self.transformer(encoded, src_key_padding_mask=~safe_mask)
        pooled = (transformed * safe_mask.unsqueeze(-1).float()).sum(dim=1) / torch.clamp(safe_mask.sum(dim=1, keepdim=True), min=1.0)
        current = self.current_proj(safe_current_states.reshape(bsz * num_agents, state_dim))
        fused = self.out_proj(torch.cat([pooled, current], dim=-1))
        fused = fused.reshape(bsz, num_agents, -1)
        return fused * agent_mask.unsqueeze(-1).float()


class CurrentStateEncoder(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.current_proj = MLP(state_dim, hidden_dim, hidden_dim, dropout)

    def forward(self, current_states: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
        safe_current_states = torch.nan_to_num(current_states, nan=0.0, posinf=0.0, neginf=0.0)
        safe_current_states = torch.where(agent_mask.unsqueeze(-1), safe_current_states, torch.zeros_like(safe_current_states))
        encoded = self.current_proj(safe_current_states)
        return encoded * agent_mask.unsqueeze(-1).float()


class MapPolylineEncoder(nn.Module):
    def __init__(self, map_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.point_mlp = MLP(map_dim, hidden_dim, hidden_dim, dropout)
        self.polyline_mlp = MLP(hidden_dim, hidden_dim, hidden_dim, dropout)

    def forward(self, map_polylines: torch.Tensor, map_point_mask: torch.Tensor, map_polyline_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, num_polylines, num_points, map_dim = map_polylines.shape
        safe_map_polylines = torch.nan_to_num(map_polylines, nan=0.0, posinf=0.0, neginf=0.0)
        safe_map_polylines = torch.where(
            map_point_mask.unsqueeze(-1),
            safe_map_polylines,
            torch.zeros_like(safe_map_polylines),
        )
        flat_polylines = safe_map_polylines.reshape(bsz * num_polylines, num_points, map_dim)
        point_mask = map_point_mask.reshape(bsz * num_polylines, num_points)
        valid_polyline_rows = point_mask.any(dim=1)

        hidden_dim = self.point_mlp.net[-1].out_features
        encoded_points = torch.zeros(
            (bsz * num_polylines, num_points, hidden_dim),
            device=map_polylines.device,
            dtype=map_polylines.dtype,
        )
        if bool(valid_polyline_rows.any()):
            encoded_points_valid = self.point_mlp(flat_polylines[valid_polyline_rows])
            encoded_points[valid_polyline_rows] = encoded_points_valid.to(dtype=encoded_points.dtype)

        pooled_points = torch.zeros(
            (bsz * num_polylines, hidden_dim),
            device=map_polylines.device,
            dtype=map_polylines.dtype,
        )
        if bool(valid_polyline_rows.any()):
            valid_point_mask = point_mask[valid_polyline_rows]
            valid_encoded_points = encoded_points[valid_polyline_rows]
            pooled_points_valid = (valid_encoded_points * valid_point_mask.unsqueeze(-1).float()).sum(dim=1) / torch.clamp(
                valid_point_mask.sum(dim=1, keepdim=True),
                min=1.0,
            )
            pooled_points[valid_polyline_rows] = pooled_points_valid.to(dtype=pooled_points.dtype)

        polyline_features = torch.zeros(
            (bsz * num_polylines, hidden_dim),
            device=map_polylines.device,
            dtype=map_polylines.dtype,
        )
        if bool(valid_polyline_rows.any()):
            polyline_features_valid = self.polyline_mlp(pooled_points[valid_polyline_rows])
            polyline_features[valid_polyline_rows] = polyline_features_valid.to(dtype=polyline_features.dtype)
        polyline_features = polyline_features.reshape(bsz, num_polylines, -1)
        scene_map_context = (polyline_features * map_polyline_mask.unsqueeze(-1).float()).sum(dim=1) / torch.clamp(
            map_polyline_mask.sum(dim=1, keepdim=True), min=1.0
        )
        return polyline_features, scene_map_context


class InteractionGraphEncoder(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        # Edge inputs concatenate:
        #   src node feature [hidden_dim]
        #   relative position [2]
        #   relative velocity [2]
        #   pairwise distance [1]
        #   bearing [1]
        # so the edge feature dimension is hidden_dim + 6.
        self.edge_mlp = nn.ModuleList([MLP(hidden_dim + 6, hidden_dim, hidden_dim, dropout) for _ in range(num_layers)])
        self.node_mlp = nn.ModuleList([MLP(hidden_dim * 2, hidden_dim, hidden_dim, dropout) for _ in range(num_layers)])

    def forward(self, agent_features: torch.Tensor, current_states: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
        feats = torch.nan_to_num(agent_features, nan=0.0, posinf=0.0, neginf=0.0)
        feats = torch.where(agent_mask.unsqueeze(-1), feats, torch.zeros_like(feats))
        safe_current_states = torch.nan_to_num(current_states, nan=0.0, posinf=0.0, neginf=0.0)
        safe_current_states = torch.where(agent_mask.unsqueeze(-1), safe_current_states, torch.zeros_like(safe_current_states))
        xy = safe_current_states[..., 0:2]
        vel = safe_current_states[..., 2:4]
        rel_xy = xy.unsqueeze(2) - xy.unsqueeze(1)
        rel_vel = vel.unsqueeze(2) - vel.unsqueeze(1)
        pair_mask = agent_mask.unsqueeze(1) & agent_mask.unsqueeze(2)
        self_mask = torch.eye(agent_mask.shape[1], device=agent_mask.device, dtype=torch.bool).unsqueeze(0)
        edge_mask = pair_mask & ~self_mask
        safe_rel_xy = torch.where(edge_mask.unsqueeze(-1), rel_xy, torch.zeros_like(rel_xy))
        dist_sq = safe_rel_xy.square().sum(dim=-1, keepdim=True)
        dist = torch.sqrt(dist_sq + 1e-12)
        bearing_xy = safe_rel_xy.clone()
        bearing_xy[..., 0] = torch.where(edge_mask, bearing_xy[..., 0], torch.ones_like(bearing_xy[..., 0]))
        bearing_xy[..., 1] = torch.where(edge_mask, bearing_xy[..., 1], torch.zeros_like(bearing_xy[..., 1]))
        bearing = torch.atan2(bearing_xy[..., 1], bearing_xy[..., 0]).unsqueeze(-1)
        for edge_mlp, node_mlp in zip(self.edge_mlp, self.node_mlp):
            src = feats.unsqueeze(2).expand(-1, -1, feats.shape[1], -1)
            edge_inputs = torch.cat([src, rel_xy, rel_vel, dist, bearing], dim=-1)
            edge_features = edge_mlp(edge_inputs)
            messages = (edge_features * edge_mask.unsqueeze(-1).float()).sum(dim=2) / torch.clamp(
                edge_mask.sum(dim=2, keepdim=True), min=1.0
            )
            feats = node_mlp(torch.cat([feats, messages], dim=-1))
            feats = feats * agent_mask.unsqueeze(-1).float()
        return feats


class DifficultyEmbedding(nn.Module):
    def __init__(self, hidden_dim: int, dropout_prob: float) -> None:
        super().__init__()
        self.fourier = FourierEmbedding(1)
        self.mlp = MLP(self.fourier.out_dim, hidden_dim, hidden_dim, dropout_prob)
        self.dropout_prob = dropout_prob

    def forward(self, difficulty: torch.Tensor, training: bool) -> torch.Tensor:
        difficulty = torch.nan_to_num(difficulty.float(), nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        embedding = self.mlp(self.fourier(difficulty.unsqueeze(-1)))
        if training and self.dropout_prob > 0.0:
            keep_mask = (torch.rand(difficulty.shape[0], device=difficulty.device) > self.dropout_prob).float().unsqueeze(-1)
            embedding = embedding * keep_mask
        return embedding
