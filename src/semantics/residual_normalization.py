from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from isgen import load_json, save_json

RESIDUAL_NORMALIZER_VERSION = "2026-04-21-residual-control-knots-v1"


def _default_channel_stats() -> Dict[str, float]:
    return {
        "mean": 0.0,
        "std": 1.0,
        "p95": 0.0,
        "p99": 0.0,
        "abs_p99": 0.0,
    }


@dataclass
class ResidualNormalizer:
    channel_names: List[str]
    channel_stats: Dict[str, Dict[str, float]]
    enabled: bool = True

    def normalize_residual(self, residual: torch.Tensor) -> torch.Tensor:
        residual = torch.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)
        if not self.enabled:
            return residual
        outputs = []
        for channel_idx, channel_name in enumerate(self.channel_names):
            stats = self.channel_stats.get(channel_name, _default_channel_stats())
            outputs.append((residual[..., channel_idx] - float(stats["mean"])) / float(max(stats["std"], 1e-6)))
        return torch.nan_to_num(torch.stack(outputs, dim=-1), nan=0.0, posinf=0.0, neginf=0.0)

    def denormalize_residual(self, residual_norm: torch.Tensor) -> torch.Tensor:
        residual_norm = torch.nan_to_num(residual_norm, nan=0.0, posinf=0.0, neginf=0.0)
        if not self.enabled:
            return residual_norm
        outputs = []
        for channel_idx, channel_name in enumerate(self.channel_names):
            stats = self.channel_stats.get(channel_name, _default_channel_stats())
            outputs.append(residual_norm[..., channel_idx] * float(max(stats["std"], 1e-6)) + float(stats["mean"]))
        return torch.nan_to_num(torch.stack(outputs, dim=-1), nan=0.0, posinf=0.0, neginf=0.0)

    def clip_normalized_residual(self, residual_norm: torch.Tensor, clip_std: float) -> torch.Tensor:
        residual_norm = torch.nan_to_num(residual_norm, nan=0.0, posinf=0.0, neginf=0.0)
        if clip_std <= 0.0:
            return residual_norm
        return torch.clamp(residual_norm, min=-float(clip_std), max=float(clip_std))

    def summary(self) -> Dict[str, Dict[str, float]]:
        return {key: dict(value) for key, value in self.channel_stats.items()}


class RunningResidualStats:
    def __init__(
        self,
        channel_names: List[str],
        max_samples_per_channel: int = 200000,
        per_update_sample_limit: int = 8192,
    ) -> None:
        self.channel_names = list(channel_names)
        self.max_samples_per_channel = int(max(max_samples_per_channel, 1024))
        self.per_update_sample_limit = int(max(per_update_sample_limit, 256))
        self.counts = {name: 0 for name in self.channel_names}
        self.means = {name: 0.0 for name in self.channel_names}
        self.m2 = {name: 0.0 for name in self.channel_names}
        self.samples = {name: np.zeros((0,), dtype=np.float32) for name in self.channel_names}

    def _update_running_moments(self, channel_name: str, values: np.ndarray) -> None:
        count = int(values.size)
        if count == 0:
            return
        batch_mean = float(values.mean())
        batch_var = float(values.var())
        batch_m2 = batch_var * count
        current_count = int(self.counts[channel_name])
        current_mean = float(self.means[channel_name])
        current_m2 = float(self.m2[channel_name])
        if current_count == 0:
            self.counts[channel_name] = count
            self.means[channel_name] = batch_mean
            self.m2[channel_name] = batch_m2
            return
        total_count = current_count + count
        delta = batch_mean - current_mean
        self.means[channel_name] = current_mean + delta * count / max(total_count, 1)
        self.m2[channel_name] = current_m2 + batch_m2 + delta * delta * current_count * count / max(total_count, 1)
        self.counts[channel_name] = total_count

    def _update_reservoir(self, channel_name: str, values: np.ndarray) -> None:
        if values.size == 0:
            return
        if values.size > self.per_update_sample_limit:
            indices = np.random.choice(values.size, size=self.per_update_sample_limit, replace=False)
            values = values[indices]
        combined = np.concatenate([self.samples[channel_name], values.astype(np.float32, copy=False)], axis=0)
        if combined.size > self.max_samples_per_channel:
            indices = np.random.choice(combined.size, size=self.max_samples_per_channel, replace=False)
            combined = combined[indices]
        self.samples[channel_name] = combined.astype(np.float32, copy=False)

    def update(self, residual: torch.Tensor, mask: torch.Tensor) -> None:
        residual = torch.nan_to_num(residual.detach(), nan=0.0, posinf=0.0, neginf=0.0).float().cpu()
        mask = mask.detach().bool().cpu()
        while mask.ndim < residual.ndim:
            mask = mask.unsqueeze(-1)
        if mask.shape != residual.shape:
            mask = mask.expand_as(residual)
        for channel_idx, channel_name in enumerate(self.channel_names):
            values = residual[..., channel_idx][mask[..., channel_idx]]
            if values.numel() == 0:
                continue
            finite_values = values[torch.isfinite(values)]
            if finite_values.numel() == 0:
                continue
            np_values = finite_values.numpy().astype(np.float32, copy=False)
            self._update_running_moments(channel_name, np_values)
            self._update_reservoir(channel_name, np_values)

    def build_normalizer(self, enabled: bool = True) -> ResidualNormalizer:
        channel_stats: Dict[str, Dict[str, float]] = {}
        for channel_name in self.channel_names:
            count = int(self.counts[channel_name])
            if count <= 0:
                channel_stats[channel_name] = _default_channel_stats()
                continue
            mean = float(self.means[channel_name])
            std = float(np.sqrt(max(self.m2[channel_name] / max(count, 1), 1e-12)))
            sample_values = self.samples[channel_name]
            if sample_values.size == 0:
                p95 = p99 = abs_p99 = 0.0
            else:
                p95 = float(np.quantile(sample_values, 0.95))
                p99 = float(np.quantile(sample_values, 0.99))
                abs_p99 = float(np.quantile(np.abs(sample_values), 0.99))
            channel_stats[channel_name] = {
                "mean": mean,
                "std": max(std, 1e-6),
                "p95": p95,
                "p99": p99,
                "abs_p99": abs_p99,
            }
        return ResidualNormalizer(
            channel_names=list(self.channel_names),
            channel_stats=channel_stats,
            enabled=bool(enabled),
        )


def save_residual_stats(normalizer: ResidualNormalizer, path: str | Path) -> None:
    save_json(
        {
            "version": RESIDUAL_NORMALIZER_VERSION,
            "channel_names": list(normalizer.channel_names),
            "channel_stats": normalizer.summary(),
            "enabled": bool(normalizer.enabled),
        },
        path,
    )


def load_residual_normalizer(path: str | Path) -> ResidualNormalizer:
    payload = load_json(path)
    return ResidualNormalizer(
        channel_names=list(payload["channel_names"]),
        channel_stats={key: dict(value) for key, value in payload["channel_stats"].items()},
        enabled=bool(payload.get("enabled", True)),
    )
