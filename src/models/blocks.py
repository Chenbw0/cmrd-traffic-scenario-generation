from __future__ import annotations

import math

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FourierEmbedding(nn.Module):
    def __init__(self, in_dim: int, num_frequencies: int = 16) -> None:
        super().__init__()
        self.register_buffer("frequencies", 2.0 ** torch.arange(num_frequencies, dtype=torch.float32), persistent=False)
        self.out_dim = in_dim * num_frequencies * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        scaled = x.unsqueeze(-1) * self.frequencies
        return torch.cat([torch.sin(2 * math.pi * scaled), torch.cos(2 * math.pi * scaled)], dim=-1).flatten(start_dim=-2)


class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.embedding_dim // 2
        exponent = torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
        exponent = torch.exp(-math.log(10000.0) * exponent / max(half_dim - 1, 1))
        args = timesteps.float().unsqueeze(-1) * exponent.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.embedding_dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
