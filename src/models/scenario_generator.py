from __future__ import annotations

import logging
import inspect
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn

from isgen.models.blocks import MLP, SinusoidalTimestepEmbedding
from isgen.models.control_knots import (
    controls_to_knots,
    interpolate_control_knots,
)
from isgen.models.diffusion import DiffusionScheduleConfig, DiffusionScheduler
from isgen.models.encoders import AgentHistoryEncoder, CurrentStateEncoder, DifficultyEmbedding, InteractionGraphEncoder, MapPolylineEncoder
from isgen.semantics.anchors import ANCHOR_CHANNEL_NAMES, extract_anchor_targets
from isgen.semantics.generator_condition import (
    generator_consumes_behavior,
    generator_consumes_difficulty,
    resolve_generator_condition_mode,
)
from isgen.semantics.interaction_oracle import (
    extract_dynamic_oracle_pair_features,
    extract_static_oracle_pair_features,
    extract_teacher_forced_neighbor_summaries,
    interaction_oracle_feature_dim,
    oracle_mode_uses_pair_features,
    oracle_mode_uses_teacher_forced_neighbors,
)
from isgen.semantics.interaction_features import (
    CURRENT_PAIR_FEATURE_NAMES,
    extract_current_pair_features,
)
from isgen.semantics.control_normalization import (
    CONTROL_CHANNEL_NAMES,
    ControlNormalizer,
    compute_raw_controls_torch,
    summarize_control_tensor,
)
from isgen.semantics.residual_normalization import (
    ResidualNormalizer,
    RunningResidualStats,
)

LOGGER = logging.getLogger(__name__)


def _assert_finite_on_mask(name: str, tensor: torch.Tensor, mask: torch.Tensor) -> None:
    expanded_mask = mask
    while expanded_mask.ndim < tensor.ndim:
        expanded_mask = expanded_mask.unsqueeze(-1)
    if expanded_mask.shape != tensor.shape:
        expanded_mask = expanded_mask.expand_as(tensor)
    valid_values = tensor[expanded_mask]
    if valid_values.numel() == 0:
        return
    finite_mask = torch.isfinite(valid_values)
    if bool(finite_mask.all()):
        return
    non_finite_count = int((~finite_mask).sum().item())
    total_count = int(valid_values.numel())
    raise RuntimeError(
        f"{name} contains non-finite values on valid positions: "
        f"{non_finite_count}/{total_count} are NaN/Inf."
    )


def _masked_std(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded_mask = mask
    while expanded_mask.ndim < tensor.ndim:
        expanded_mask = expanded_mask.unsqueeze(-1)
    if expanded_mask.shape != tensor.shape:
        expanded_mask = expanded_mask.expand_as(tensor)
    values = tensor[expanded_mask]
    if values.numel() == 0:
        return torch.zeros((), device=tensor.device, dtype=tensor.dtype)
    values = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return values.std(unbiased=False).to(dtype=tensor.dtype)


def _clip_fraction(before: torch.Tensor, after: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded_mask = mask
    while expanded_mask.ndim < before.ndim:
        expanded_mask = expanded_mask.unsqueeze(-1)
    if expanded_mask.shape != before.shape:
        expanded_mask = expanded_mask.expand_as(before)
    if int(expanded_mask.sum().item()) <= 0:
        return torch.zeros((), device=before.device, dtype=before.dtype)
    changed = torch.abs(before - after) > 1e-6
    clipped = (changed & expanded_mask).float().sum()
    denom = torch.clamp(expanded_mask.float().sum(), min=1.0)
    return (clipped / denom).to(dtype=before.dtype)


class TemporalDenoiser(nn.Module):
    def __init__(self, future_dim: int, hidden_dim: int, num_layers: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.input_proj = nn.Linear(future_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_dim, future_dim)

    def forward(self, noisy_future: torch.Tensor, future_mask: torch.Tensor, agent_mask: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        bsz, num_agents, horizon, future_dim = noisy_future.shape
        valid_mask = future_mask & agent_mask.unsqueeze(-1)
        safe_noisy_future = torch.where(valid_mask.unsqueeze(-1), noisy_future, torch.zeros_like(noisy_future))
        tokens = self.input_proj(safe_noisy_future) + context.unsqueeze(2)
        tokens = tokens.reshape(bsz * num_agents, horizon, -1)
        mask = valid_mask.reshape(bsz * num_agents, horizon)
        safe_mask = mask.clone()
        empty_rows = safe_mask.sum(dim=1) == 0
        safe_mask[empty_rows, 0] = True
        encoded = self.transformer(tokens, src_key_padding_mask=~safe_mask)
        denoised = self.output_proj(encoded).reshape(bsz, num_agents, horizon, future_dim)
        return denoised * valid_mask.unsqueeze(-1).float()


class ControlKnotBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_control_knots: int, control_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_control_knots = num_control_knots
        self.control_dim = control_dim
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_control_knots * control_dim),
        )

    def forward(self, context: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
        knots = self.head(context).reshape(context.shape[0], context.shape[1], self.num_control_knots, self.control_dim)
        return knots * agent_mask.unsqueeze(-1).unsqueeze(-1).float()


class AnchorPredictor(nn.Module):
    def __init__(self, hidden_dim: int, anchor_dim: int, dropout: float) -> None:
        super().__init__()
        self.anchor_dim = anchor_dim
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, anchor_dim),
        )

    def forward(self, context: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
        anchors = self.head(context)
        return anchors * agent_mask.unsqueeze(-1).float()


class AnchorResidualDenoiser(nn.Module):
    def __init__(self, anchor_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.anchor_dim = anchor_dim
        self.net = nn.Sequential(
            nn.Linear(anchor_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, anchor_dim),
        )

    def forward(
        self,
        noisy_anchor: torch.Tensor,
        anchor_mask: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        safe_noisy_anchor = torch.where(anchor_mask.unsqueeze(-1), noisy_anchor, torch.zeros_like(noisy_anchor))
        safe_context = torch.where(anchor_mask.unsqueeze(-1), context, torch.zeros_like(context))
        output = self.net(torch.cat([safe_noisy_anchor, safe_context], dim=-1))
        return torch.where(anchor_mask.unsqueeze(-1), output, torch.zeros_like(output))


class DifficultyConditionedScenarioDiffusion(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        model_cfg = config["model"]
        diffusion_cfg = config["diffusion"]
        self.config = config
        self.hidden_dim = int(model_cfg["hidden_dim"])
        self.future_state_dim = int(model_cfg["future_state_dim"])
        self.local_future_dim = int(model_cfg["local_future_dim"])
        self.decoder_type = str(model_cfg.get("decoder_type", "additive_local_state"))
        self.control_dim = int(model_cfg.get("control_dim", 2))
        self.control_representation = str(model_cfg.get("control_representation", "per_step"))
        self.num_control_knots = int(model_cfg.get("num_control_knots", 6))
        self.control_interpolation = str(model_cfg.get("control_interpolation", "linear"))
        self.use_anchor_latent = bool(model_cfg.get("use_anchor_latent", False))
        self.anchor_dim = int(model_cfg.get("anchor_dim", len(ANCHOR_CHANNEL_NAMES)))
        self.anchor_channel_names = list(ANCHOR_CHANNEL_NAMES[: self.anchor_dim])
        interaction_oracle_cfg = config.get("interaction_oracle", {})
        self.interaction_oracle_mode = str(interaction_oracle_cfg.get("mode", "none"))
        self.interaction_oracle_num_trace_points = int(interaction_oracle_cfg.get("num_trace_points", 5))
        self.interaction_oracle_conflict_distance_m = float(interaction_oracle_cfg.get("conflict_distance_m", 5.0))
        self.interaction_oracle_conflict_temperature_m = float(interaction_oracle_cfg.get("conflict_temperature_m", 1.0))
        self.interaction_oracle_feature_dim = int(
            interaction_oracle_feature_dim(self.interaction_oracle_mode, self.interaction_oracle_num_trace_points)
        )
        interaction_field_cfg = config.get("interaction_field", {})
        self.learned_interaction_field_enabled = bool(interaction_field_cfg.get("enabled", False))
        self.learned_interaction_field_target = str(interaction_field_cfg.get("target_type", "static_summary"))
        self.learned_interaction_field_hidden_dim = int(interaction_field_cfg.get("hidden_dim", self.hidden_dim))
        self.learned_interaction_field_latent_dim = int(interaction_field_cfg.get("latent_dim", self.hidden_dim))
        self.learned_interaction_num_message_passing_steps = int(interaction_field_cfg.get("num_message_passing_steps", 2))
        self.learned_interaction_field_condition_residual = bool(interaction_field_cfg.get("condition_residual_diffusion", True))
        if self.learned_interaction_field_enabled and self.interaction_oracle_mode != "none":
            raise ValueError("interaction_field.enabled=true cannot be combined with interaction_oracle.mode!=none.")
        if self.learned_interaction_field_target != "static_summary":
            raise ValueError(
                f"Unsupported interaction_field.target_type: {self.learned_interaction_field_target}. "
                "Only 'static_summary' is currently implemented."
            )
        self.learned_interaction_feature_dim = int(
            interaction_oracle_feature_dim(self.learned_interaction_field_target, self.interaction_oracle_num_trace_points)
        )
        self.current_pair_feature_dim = len(CURRENT_PAIR_FEATURE_NAMES)
        residual_cfg = config.get("residual_diffusion", {})
        anchor_residual_cfg = config.get("anchor_residual_diffusion", {})
        self.residual_diffusion_enabled = bool(residual_cfg.get("enabled", model_cfg.get("residual_diffusion_enabled", True)))
        self.anchor_residual_diffusion_enabled = bool(anchor_residual_cfg.get("enabled", False))
        self.accel_scale_mps2 = float(model_cfg.get("accel_scale_mps2", 4.0))
        self.yaw_rate_scale_radps = float(model_cfg.get("yaw_rate_scale_radps", 1.5))
        self.max_speed_mps = float(model_cfg.get("max_speed_mps", 20.0))
        self.min_speed_mps = float(model_cfg.get("min_speed_mps", 0.0))
        self.control_soft_clamp = bool(model_cfg.get("control_soft_clamp", True))
        self.use_control_normalizer = bool(model_cfg.get("use_control_normalizer", True))
        self.control_clamp_mode = str(model_cfg.get("control_clamp_mode", "quantile"))
        self.control_clamp_quantile = float(model_cfg.get("control_clamp_quantile", 0.995))
        self.control_clamp_margin = float(model_cfg.get("control_clamp_margin", 1.25))
        self.control_squash = bool(model_cfg.get("control_squash", False))
        generator_input_cfg = config.get("generator_input", {})
        self.use_history = bool(generator_input_cfg.get("use_history", True))
        self.generator_input_history_mode = str(generator_input_cfg.get("history_mode", "full_history"))
        self.require_history_at_sample = bool(generator_input_cfg.get("require_history_at_sample", True))
        self.encoder_type = "full_history_encoder" if self.use_history else "current_state_encoder"
        self.generator_condition_mode = resolve_generator_condition_mode(config)
        self.generator_consumes_behavior = generator_consumes_behavior(config)
        self.generator_consumes_difficulty = generator_consumes_difficulty(config)
        self.condition_dropout_prob = float(diffusion_cfg["condition_dropout_prob"])
        self.diffusion_target_type = str(residual_cfg.get("target_type", diffusion_cfg.get("target_type", "epsilon")))
        self.anchor_diffusion_target_type = str(anchor_residual_cfg.get("target_type", "x0"))
        self.min_snr_gamma = float(diffusion_cfg.get("min_snr_gamma", 0.0))
        self.timestep_sampling = str(diffusion_cfg.get("timestep_sampling", "uniform"))
        self.normalize_residual = bool(residual_cfg.get("normalize_residual", True))
        self.normalize_anchor_residual = bool(anchor_residual_cfg.get("normalize_residual", True))
        self.residual_sample_scale = float(residual_cfg.get("residual_scale", config.get("sampling", {}).get("residual_scale", 1.0)))
        self.anchor_residual_sample_scale = float(anchor_residual_cfg.get("residual_scale", config.get("sampling", {}).get("anchor_residual_scale", 0.5)))
        self.residual_clip_std = float(residual_cfg.get("residual_clip_std", 3.0))
        self.anchor_residual_clip_std = float(anchor_residual_cfg.get("residual_clip_std", 3.0))
        self.residual_sample_steps = int(residual_cfg.get("sample_steps", config.get("sampling", {}).get("sample_steps", diffusion_cfg.get("num_steps", 100))))
        self.anchor_residual_sample_steps = int(anchor_residual_cfg.get("sample_steps", config.get("sampling", {}).get("sample_steps", diffusion_cfg.get("num_steps", 100))))
        self.model_future_dim = self.control_dim if self.decoder_type == "kinematic_controls" else self.local_future_dim
        self.control_channel_names = list(CONTROL_CHANNEL_NAMES)
        self.control_normalizer: ControlNormalizer | None = None
        self.residual_normalizer: ResidualNormalizer | None = None
        self.anchor_residual_normalizer: ResidualNormalizer | None = None
        self.residual_stats_tracker = RunningResidualStats(list(self.control_channel_names))
        self.anchor_residual_stats_tracker = RunningResidualStats(list(self.anchor_channel_names))
        self.last_sample_debug: Dict[str, object] = {}
        if self.use_history:
            self.agent_encoder = AgentHistoryEncoder(
                state_dim=5,
                hidden_dim=self.hidden_dim,
                num_layers=int(model_cfg["num_agent_layers"]),
                num_heads=int(model_cfg["num_heads"]),
                dropout=float(model_cfg["dropout"]),
            )
        else:
            self.agent_encoder = CurrentStateEncoder(
                state_dim=5,
                hidden_dim=self.hidden_dim,
                dropout=float(model_cfg["dropout"]),
            )
        self.map_encoder = MapPolylineEncoder(map_dim=4, hidden_dim=int(model_cfg["map_hidden_dim"]), dropout=float(model_cfg["dropout"]))
        self.graph_encoder = InteractionGraphEncoder(hidden_dim=self.hidden_dim, num_layers=int(model_cfg["num_graph_layers"]), dropout=float(model_cfg["dropout"]))
        self.difficulty_embedding = DifficultyEmbedding(hidden_dim=self.hidden_dim, dropout_prob=self.condition_dropout_prob)
        self.time_embedding = nn.Sequential(
            SinusoidalTimestepEmbedding(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.context_proj = MLP(self.hidden_dim * 3, self.hidden_dim, self.hidden_dim, dropout=float(model_cfg["dropout"]))
        if oracle_mode_uses_pair_features(self.interaction_oracle_mode):
            self.interaction_pair_encoder = nn.Sequential(
                nn.Linear(self.interaction_oracle_feature_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(float(model_cfg["dropout"])),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            self.interaction_pair_score = nn.Linear(self.hidden_dim, 1)
        else:
            self.interaction_pair_encoder = None
            self.interaction_pair_score = None
        if oracle_mode_uses_teacher_forced_neighbors(self.interaction_oracle_mode):
            self.teacher_forced_neighbor_encoder = nn.Sequential(
                nn.Linear(self.interaction_oracle_feature_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(float(model_cfg["dropout"])),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        else:
            self.teacher_forced_neighbor_encoder = None
        if self.learned_interaction_field_enabled:
            self.learned_interaction_pair_input = nn.Sequential(
                nn.Linear(self.hidden_dim * 2 + self.current_pair_feature_dim, self.learned_interaction_field_hidden_dim),
                nn.ReLU(),
                nn.Dropout(float(model_cfg["dropout"])),
                nn.Linear(self.learned_interaction_field_hidden_dim, self.learned_interaction_field_hidden_dim),
                nn.ReLU(),
                nn.Dropout(float(model_cfg["dropout"])),
                nn.Linear(self.learned_interaction_field_hidden_dim, self.learned_interaction_field_latent_dim),
            )
            self.learned_interaction_pair_to_agent = nn.Sequential(
                nn.Linear(self.learned_interaction_field_latent_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(float(model_cfg["dropout"])),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            self.learned_interaction_pair_to_agent_score = nn.Linear(self.learned_interaction_field_latent_dim, 1)
            self.learned_interaction_agent_update = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.learned_interaction_field_hidden_dim),
                nn.ReLU(),
                nn.Dropout(float(model_cfg["dropout"])),
                nn.Linear(self.learned_interaction_field_hidden_dim, self.hidden_dim),
            )
            self.learned_interaction_pair_update = nn.Sequential(
                nn.Linear(self.learned_interaction_field_latent_dim + self.hidden_dim * 2 + self.current_pair_feature_dim, self.learned_interaction_field_hidden_dim),
                nn.ReLU(),
                nn.Dropout(float(model_cfg["dropout"])),
                nn.Linear(self.learned_interaction_field_hidden_dim, self.learned_interaction_field_latent_dim),
            )
            self.learned_interaction_pair_projector = nn.Sequential(
                nn.Linear(self.learned_interaction_field_latent_dim, self.learned_interaction_field_hidden_dim),
                nn.ReLU(),
                nn.Dropout(float(model_cfg["dropout"])),
                nn.Linear(self.learned_interaction_field_hidden_dim, self.learned_interaction_feature_dim),
            )
            self.learned_interaction_context_proj = nn.Sequential(
                nn.Linear(self.learned_interaction_field_latent_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(float(model_cfg["dropout"])),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            self.learned_interaction_context_score = nn.Linear(self.learned_interaction_field_latent_dim, 1)
        else:
            self.learned_interaction_pair_input = None
            self.learned_interaction_pair_to_agent = None
            self.learned_interaction_pair_to_agent_score = None
            self.learned_interaction_agent_update = None
            self.learned_interaction_pair_update = None
            self.learned_interaction_pair_projector = None
            self.learned_interaction_context_proj = None
            self.learned_interaction_context_score = None
        self.anchor_predictor = AnchorPredictor(
            hidden_dim=self.hidden_dim,
            anchor_dim=self.anchor_dim,
            dropout=float(model_cfg["dropout"]),
        )
        self.anchor_embedding = nn.Sequential(
            nn.Linear(self.anchor_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(model_cfg["dropout"])),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.control_knot_backbone = ControlKnotBackbone(
            input_dim=self.hidden_dim * (2 if self.use_anchor_latent else 1),
            hidden_dim=self.hidden_dim,
            num_control_knots=self.num_control_knots,
            control_dim=self.control_dim,
            dropout=float(model_cfg["dropout"]),
        )
        self.anchor_residual_denoiser = AnchorResidualDenoiser(
            anchor_dim=self.anchor_dim,
            hidden_dim=self.hidden_dim,
            dropout=float(model_cfg["dropout"]),
        )
        self.denoiser = TemporalDenoiser(
            future_dim=self.model_future_dim,
            hidden_dim=self.hidden_dim,
            num_layers=int(model_cfg["num_temporal_layers"]),
            num_heads=int(model_cfg["num_heads"]),
            dropout=float(model_cfg["dropout"]),
        )
        self.scheduler = DiffusionScheduler(
            DiffusionScheduleConfig(
                num_steps=int(diffusion_cfg["num_steps"]),
                beta_start=float(diffusion_cfg["beta_start"]),
                beta_end=float(diffusion_cfg["beta_end"]),
            )
        )
        if self.decoder_type == "kinematic_controls":
            assert self.model_future_dim == self.control_dim, "Kinematic control decoder must predict control_dim channels."

    def set_control_normalizer(self, normalizer: ControlNormalizer | None) -> None:
        self.control_normalizer = normalizer

    def set_residual_normalizer(self, normalizer: ResidualNormalizer | None) -> None:
        self.residual_normalizer = normalizer

    def set_anchor_residual_normalizer(self, normalizer: ResidualNormalizer | None) -> None:
        self.anchor_residual_normalizer = normalizer

    def update_residual_stats(self, residual_target_raw: torch.Tensor, knot_mask: torch.Tensor) -> None:
        if not self.normalize_residual:
            return
        self.residual_stats_tracker.update(residual_target_raw, knot_mask)
        self.residual_normalizer = self.residual_stats_tracker.build_normalizer(enabled=True)

    def residual_stats_summary(self) -> Dict[str, Dict[str, float]] | None:
        if self.residual_normalizer is None:
            return None
        return self.residual_normalizer.summary()

    def update_anchor_residual_stats(self, residual_target_raw: torch.Tensor, anchor_mask: torch.Tensor) -> None:
        if not self.normalize_anchor_residual:
            return
        self.anchor_residual_stats_tracker.update(residual_target_raw, anchor_mask)
        self.anchor_residual_normalizer = self.anchor_residual_stats_tracker.build_normalizer(enabled=True)

    def anchor_residual_stats_summary(self) -> Dict[str, Dict[str, float]] | None:
        if self.anchor_residual_normalizer is None:
            return None
        return self.anchor_residual_normalizer.summary()

    def normalize_controls(self, raw_controls: torch.Tensor) -> torch.Tensor:
        raw_controls = torch.nan_to_num(raw_controls, nan=0.0, posinf=0.0, neginf=0.0)
        if self.use_control_normalizer:
            if self.control_normalizer is None:
                raise RuntimeError("Control normalizer is enabled but not set on the model.")
            return self.control_normalizer.normalize_controls(raw_controls)
        return raw_controls

    def denormalize_controls(self, normalized_controls: torch.Tensor) -> torch.Tensor:
        normalized_controls = torch.nan_to_num(normalized_controls, nan=0.0, posinf=0.0, neginf=0.0)
        if self.use_control_normalizer:
            if self.control_normalizer is None:
                raise RuntimeError("Control normalizer is enabled but not set on the model.")
            return self.control_normalizer.denormalize_controls(normalized_controls)
        return normalized_controls

    def clamp_raw_controls_to_train_quantiles(self, raw_controls: torch.Tensor) -> torch.Tensor:
        raw_controls = torch.nan_to_num(raw_controls, nan=0.0, posinf=0.0, neginf=0.0)
        if self.control_normalizer is None:
            return raw_controls
        return torch.nan_to_num(self.control_normalizer.clamp_raw_controls_to_train_quantiles(
            raw_controls,
            mode=self.control_clamp_mode,
            margin=self.control_clamp_margin,
        ), nan=0.0, posinf=0.0, neginf=0.0)

    def normalize_residual_target(self, residual_target_raw: torch.Tensor) -> torch.Tensor:
        residual_target_raw = torch.nan_to_num(residual_target_raw, nan=0.0, posinf=0.0, neginf=0.0)
        if not self.normalize_residual or self.residual_normalizer is None:
            return residual_target_raw
        return self.residual_normalizer.normalize_residual(residual_target_raw)

    def denormalize_residual_target(self, residual_target_norm: torch.Tensor) -> torch.Tensor:
        residual_target_norm = torch.nan_to_num(residual_target_norm, nan=0.0, posinf=0.0, neginf=0.0)
        if not self.normalize_residual or self.residual_normalizer is None:
            return residual_target_norm
        return self.residual_normalizer.denormalize_residual(residual_target_norm)

    def clip_residual_target_norm(self, residual_target_norm: torch.Tensor) -> torch.Tensor:
        residual_target_norm = torch.nan_to_num(residual_target_norm, nan=0.0, posinf=0.0, neginf=0.0)
        if self.residual_clip_std <= 0.0:
            return residual_target_norm
        if not self.normalize_residual or self.residual_normalizer is None:
            return torch.clamp(residual_target_norm, -self.residual_clip_std, self.residual_clip_std)
        return self.residual_normalizer.clip_normalized_residual(residual_target_norm, self.residual_clip_std)

    def normalize_anchor_residual_target(self, residual_target_raw: torch.Tensor) -> torch.Tensor:
        residual_target_raw = torch.nan_to_num(residual_target_raw, nan=0.0, posinf=0.0, neginf=0.0)
        if not self.normalize_anchor_residual or self.anchor_residual_normalizer is None:
            return residual_target_raw
        return self.anchor_residual_normalizer.normalize_residual(residual_target_raw)

    def denormalize_anchor_residual_target(self, residual_target_norm: torch.Tensor) -> torch.Tensor:
        residual_target_norm = torch.nan_to_num(residual_target_norm, nan=0.0, posinf=0.0, neginf=0.0)
        if not self.normalize_anchor_residual or self.anchor_residual_normalizer is None:
            return residual_target_norm
        return self.anchor_residual_normalizer.denormalize_residual(residual_target_norm)

    def clip_anchor_residual_target_norm(self, residual_target_norm: torch.Tensor) -> torch.Tensor:
        residual_target_norm = torch.nan_to_num(residual_target_norm, nan=0.0, posinf=0.0, neginf=0.0)
        if self.anchor_residual_clip_std <= 0.0:
            return residual_target_norm
        if not self.normalize_anchor_residual or self.anchor_residual_normalizer is None:
            return torch.clamp(residual_target_norm, -self.anchor_residual_clip_std, self.anchor_residual_clip_std)
        return self.anchor_residual_normalizer.clip_normalized_residual(residual_target_norm, self.anchor_residual_clip_std)

    def get_runtime_metadata(self) -> Dict[str, object]:
        return {
            "decoder_type": self.decoder_type,
            "local_future_dim": self.local_future_dim,
            "control_dim": self.control_dim,
            "control_representation": self.control_representation,
            "num_control_knots": self.num_control_knots,
            "control_interpolation": self.control_interpolation,
            "use_anchor_latent": self.use_anchor_latent,
            "anchor_dim": self.anchor_dim,
            "anchor_channel_names": list(self.anchor_channel_names),
            "residual_diffusion_enabled": self.residual_diffusion_enabled,
            "anchor_residual_diffusion_enabled": self.anchor_residual_diffusion_enabled,
            "normalize_residual": self.normalize_residual,
            "normalize_anchor_residual": self.normalize_anchor_residual,
            "residual_clip_std": self.residual_clip_std,
            "anchor_residual_clip_std": self.anchor_residual_clip_std,
            "residual_sample_steps": self.residual_sample_steps,
            "anchor_residual_sample_steps": self.anchor_residual_sample_steps,
            "control_channel_names": list(self.control_channel_names),
            "forward_path_name": "encode_future_controls_to_knots" if self.control_representation == "knots" else ("encode_future_controls" if self.decoder_type == "kinematic_controls" else "encode_future_local"),
            "sample_decode_path_name": "decode_control_knots_to_future" if self.control_representation == "knots" else ("decode_controls_to_future" if self.decoder_type == "kinematic_controls" else "decode_future_local"),
            "control_normalizer_enabled": bool(self.use_control_normalizer),
            "control_normalizer_loaded": self.control_normalizer is not None,
            "control_clamp_mode": self.control_clamp_mode,
            "control_clamp_quantile": self.control_clamp_quantile,
            "control_clamp_margin": self.control_clamp_margin,
            "control_soft_clamp": self.control_soft_clamp,
            "uses_encode_future_controls": self.decoder_type == "kinematic_controls",
            "uses_decode_controls_to_future": self.decoder_type == "kinematic_controls",
            "uses_encode_future_local": self.decoder_type != "kinematic_controls",
            "uses_decode_future_local": self.decoder_type != "kinematic_controls",
            "generator_input_use_history": self.use_history,
            "generator_input_history_mode": self.generator_input_history_mode,
            "require_history_at_sample": self.require_history_at_sample,
            "encoder_type": self.encoder_type,
            "history_used_for_generation": self.use_history,
            "generator_condition_mode": self.generator_condition_mode,
            "generator_consumes_behavior": self.generator_consumes_behavior,
            "generator_consumes_difficulty": self.generator_consumes_difficulty,
            "interaction_oracle_mode": self.interaction_oracle_mode,
            "interaction_oracle_feature_dim": self.interaction_oracle_feature_dim,
            "interaction_oracle_num_trace_points": self.interaction_oracle_num_trace_points,
            "interaction_oracle_conflict_distance_m": self.interaction_oracle_conflict_distance_m,
            "interaction_oracle_conflict_temperature_m": self.interaction_oracle_conflict_temperature_m,
            "learned_interaction_field_enabled": self.learned_interaction_field_enabled,
            "learned_interaction_field_target": self.learned_interaction_field_target,
            "learned_interaction_field_feature_dim": self.learned_interaction_feature_dim,
            "learned_interaction_field_latent_dim": self.learned_interaction_field_latent_dim,
            "learned_interaction_num_message_passing_steps": self.learned_interaction_num_message_passing_steps,
            "learned_interaction_field_condition_residual": self.learned_interaction_field_condition_residual,
            "diffusion_target_type": self.diffusion_target_type,
            "anchor_diffusion_target_type": self.anchor_diffusion_target_type,
            "timestep_sampling": self.timestep_sampling,
            "min_snr_gamma": self.min_snr_gamma,
            "residual_sample_scale": self.residual_sample_scale,
            "anchor_residual_sample_scale": self.anchor_residual_sample_scale,
            "residual_normalizer_loaded": self.residual_normalizer is not None,
            "anchor_residual_normalizer_loaded": self.anchor_residual_normalizer is not None,
            "runtime_source_file": str(Path(inspect.getsourcefile(self.__class__) or __file__).resolve()),
        }

    def compute_condition_embedding(
        self,
        target_difficulty: torch.Tensor | None = None,
        target_behavior: torch.Tensor | None = None,
        force_drop: bool = False,
    ) -> torch.Tensor:
        if self.generator_condition_mode == "none":
            if target_behavior is not None:
                base = target_behavior
            elif target_difficulty is not None:
                base = target_difficulty
            else:
                base = torch.zeros(1, device=self.time_embedding[1].weight.device, dtype=torch.float32)
            return torch.zeros((base.shape[0], self.hidden_dim), device=base.device, dtype=torch.float32)
        condition_value = self._resolve_condition(target_difficulty=target_difficulty, target_behavior=target_behavior)
        condition_value = torch.nan_to_num(condition_value.float(), nan=0.0, posinf=0.0, neginf=0.0)
        embedding = self.difficulty_embedding(condition_value, training=self.training and not force_drop)
        embedding = torch.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)
        if force_drop:
            embedding = torch.zeros_like(embedding)
        return embedding

    @staticmethod
    def encode_future_local(current_states: torch.Tensor, future_states: torch.Tensor) -> torch.Tensor:
        safe_current_states = torch.nan_to_num(current_states, nan=0.0, posinf=0.0, neginf=0.0)
        safe_future_states = torch.nan_to_num(future_states, nan=0.0, posinf=0.0, neginf=0.0)
        current_xy = safe_current_states[..., None, 0:2]
        current_vel = safe_current_states[..., None, 2:4]
        current_heading = safe_current_states[..., None, 4:5]
        delta_xy = safe_future_states[..., 0:2] - current_xy
        delta_vel = safe_future_states[..., 2:4] - current_vel
        heading_delta = safe_future_states[..., 4:5] - current_heading
        delta_heading = torch.atan2(torch.sin(heading_delta), torch.cos(heading_delta))
        return torch.cat([delta_xy, delta_vel, delta_heading], dim=-1)

    @staticmethod
    def decode_future_local(current_states: torch.Tensor, encoded_future: torch.Tensor) -> torch.Tensor:
        safe_current_states = torch.nan_to_num(current_states, nan=0.0, posinf=0.0, neginf=0.0)
        safe_encoded_future = torch.nan_to_num(encoded_future, nan=0.0, posinf=0.0, neginf=0.0)
        future_xy = safe_encoded_future[..., 0:2] + safe_current_states[..., None, 0:2]
        future_vel = safe_encoded_future[..., 2:4] + safe_current_states[..., None, 2:4]
        future_heading = torch.atan2(
            torch.sin(safe_encoded_future[..., 4:5] + safe_current_states[..., None, 4:5]),
            torch.cos(safe_encoded_future[..., 4:5] + safe_current_states[..., None, 4:5]),
        )
        return torch.cat([future_xy, future_vel, future_heading], dim=-1)

    def _control_to_physical(self, normalized_controls: torch.Tensor) -> torch.Tensor:
        normalized_controls = torch.nan_to_num(normalized_controls, nan=0.0, posinf=0.0, neginf=0.0)
        accel_norm = normalized_controls[..., 0]
        yaw_rate_norm = normalized_controls[..., 1]
        if self.control_soft_clamp:
            accel_norm = torch.clamp(accel_norm, -1.0, 1.0)
            yaw_rate_norm = torch.clamp(yaw_rate_norm, -1.0, 1.0)
        if self.control_squash:
            accel_norm = torch.tanh(accel_norm)
            yaw_rate_norm = torch.tanh(yaw_rate_norm)
        accel = accel_norm * self.accel_scale_mps2
        yaw_rate = yaw_rate_norm * self.yaw_rate_scale_radps
        return torch.stack([accel, yaw_rate], dim=-1)

    def encode_future_controls(
        self,
        current_states: torch.Tensor,
        future_states: torch.Tensor,
        future_mask: torch.Tensor,
        agent_mask: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        return compute_raw_controls_torch(
            current_states=current_states,
            future_states=future_states,
            future_mask=future_mask,
            agent_mask=agent_mask,
            dt=dt,
            accel_scale_mps2=self.accel_scale_mps2,
            yaw_rate_scale_radps=self.yaw_rate_scale_radps,
        )

    def encode_future_controls_to_knots(
        self,
        current_states: torch.Tensor,
        future_states: torch.Tensor,
        future_mask: torch.Tensor,
        agent_mask: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        per_step_raw_controls = self.encode_future_controls(
            current_states=current_states,
            future_states=future_states,
            future_mask=future_mask,
            agent_mask=agent_mask,
            dt=dt,
        )
        per_step_encoded_controls = self.normalize_controls(per_step_raw_controls)
        knots, knot_mask = controls_to_knots(
            per_step_encoded_controls,
            future_mask=future_mask,
            agent_mask=agent_mask,
            num_control_knots=self.num_control_knots,
        )
        return knots, knot_mask, per_step_raw_controls

    def interpolate_control_knots(
        self,
        control_knots: torch.Tensor,
        future_frames: int,
    ) -> torch.Tensor:
        return interpolate_control_knots(
            control_knots,
            future_frames=future_frames,
            mode=self.control_interpolation,
        )

    def decode_control_knots_to_future(
        self,
        current_states: torch.Tensor,
        control_knots: torch.Tensor,
        future_mask: torch.Tensor,
        agent_mask: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        per_step_controls = self.interpolate_control_knots(control_knots, future_frames=future_mask.shape[-1])
        per_step_controls = torch.nan_to_num(per_step_controls, nan=0.0, posinf=0.0, neginf=0.0)
        raw_controls = self.denormalize_controls(per_step_controls)
        raw_controls_clamped = self.clamp_raw_controls_to_train_quantiles(raw_controls)
        decoded_future, decoded_controls_physical = self.decode_controls_to_future(
            current_states=current_states,
            controls=raw_controls_clamped,
            future_mask=future_mask,
            agent_mask=agent_mask,
            dt=dt,
        )
        return decoded_future, raw_controls_clamped, decoded_controls_physical

    def decode_controls_to_future(
        self,
        current_states: torch.Tensor,
        controls: torch.Tensor,
        future_mask: torch.Tensor,
        agent_mask: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        safe_controls = torch.nan_to_num(controls, nan=0.0, posinf=0.0, neginf=0.0)
        physical_controls = self._control_to_physical(safe_controls)
        batch_size, num_agents, horizon, _ = controls.shape
        safe_current_states = torch.nan_to_num(current_states, nan=0.0, posinf=0.0, neginf=0.0)
        safe_current_states = torch.where(agent_mask.unsqueeze(-1), safe_current_states, torch.zeros_like(safe_current_states))
        position = safe_current_states[..., 0:2]
        speed = torch.linalg.norm(safe_current_states[..., 2:4], dim=-1)
        heading = safe_current_states[..., 4]
        decoded_steps = []
        for step_idx in range(horizon):
            valid = future_mask[..., step_idx] & agent_mask
            accel_t = physical_controls[..., step_idx, 0]
            yaw_rate_t = physical_controls[..., step_idx, 1]
            next_speed = torch.clamp(speed + accel_t * dt, min=self.min_speed_mps, max=self.max_speed_mps)
            next_heading = torch.atan2(torch.sin(heading + yaw_rate_t * dt), torch.cos(heading + yaw_rate_t * dt))
            next_position = torch.stack(
                [
                    position[..., 0] + next_speed * torch.cos(next_heading) * dt,
                    position[..., 1] + next_speed * torch.sin(next_heading) * dt,
                ],
                dim=-1,
            )
            next_velocity = torch.stack(
                [
                    next_speed * torch.cos(next_heading),
                    next_speed * torch.sin(next_heading),
                ],
                dim=-1,
            )
            next_state = torch.cat([next_position, next_velocity, next_heading.unsqueeze(-1)], dim=-1)
            decoded_steps.append(torch.where(valid.unsqueeze(-1), next_state, torch.zeros_like(next_state)))
            position = torch.where(valid.unsqueeze(-1), next_position, position)
            speed = torch.where(valid, next_speed, speed)
            heading = torch.where(valid, next_heading, heading)
        decoded_future = torch.stack(decoded_steps, dim=2)
        valid_mask = future_mask & agent_mask.unsqueeze(-1)
        decoded_future = torch.where(valid_mask.unsqueeze(-1), decoded_future, torch.zeros_like(decoded_future))
        physical_controls = torch.where(valid_mask.unsqueeze(-1), physical_controls, torch.zeros_like(physical_controls))
        decoded_future = torch.nan_to_num(decoded_future, nan=0.0, posinf=0.0, neginf=0.0)
        physical_controls = torch.nan_to_num(physical_controls, nan=0.0, posinf=0.0, neginf=0.0)
        return decoded_future, physical_controls

    def _resolve_condition(
        self,
        target_difficulty: torch.Tensor | None = None,
        target_behavior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.generator_condition_mode == "none":
            if target_behavior is not None:
                return torch.zeros_like(target_behavior)
            if target_difficulty is not None:
                return torch.zeros_like(target_difficulty)
            raise ValueError("Generator condition mode is none and no fallback tensor was provided.")
        if self.generator_condition_mode == "behavior_only":
            if target_behavior is None:
                raise ValueError("generator_condition.mode=behavior_only requires target_behavior.")
            return target_behavior
        if self.generator_condition_mode == "difficulty_only":
            if target_difficulty is None:
                raise ValueError("generator_condition.mode=difficulty_only requires target_difficulty.")
            return target_difficulty
        if self.generator_condition_mode == "behavior_plus_difficulty":
            if target_behavior is None or target_difficulty is None:
                raise ValueError(
                    "generator_condition.mode=behavior_plus_difficulty requires both target_behavior and target_difficulty."
                )
            return 0.5 * (target_behavior + target_difficulty)
        raise ValueError(f"Unsupported generator_condition_mode: {self.generator_condition_mode}")

    def _build_scene_context(
        self,
        history_states: torch.Tensor,
        history_mask: torch.Tensor,
        current_states: torch.Tensor,
        agent_mask: torch.Tensor,
        map_polylines: torch.Tensor,
        map_point_mask: torch.Tensor,
        map_polyline_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_history:
            agent_context = self.agent_encoder(history_states, history_mask, current_states, agent_mask)
        else:
            agent_context = self.agent_encoder(current_states, agent_mask)
        graph_context = self.graph_encoder(agent_context, current_states, agent_mask)
        _, map_context = self.map_encoder(map_polylines, map_point_mask, map_polyline_mask)
        agent_context = torch.nan_to_num(agent_context, nan=0.0, posinf=0.0, neginf=0.0)
        graph_context = torch.nan_to_num(graph_context, nan=0.0, posinf=0.0, neginf=0.0)
        map_context = torch.nan_to_num(map_context, nan=0.0, posinf=0.0, neginf=0.0)
        scene_context = map_context.unsqueeze(1).expand_as(graph_context)
        combined = torch.cat([agent_context, graph_context, scene_context], dim=-1)
        combined = torch.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
        return self.context_proj(combined)

    def _aggregate_pairwise_interaction_context(
        self,
        pair_features: torch.Tensor,
        pair_valid: torch.Tensor,
    ) -> torch.Tensor:
        if self.interaction_pair_encoder is None or self.interaction_pair_score is None:
            return torch.zeros(
                pair_features.shape[0],
                pair_features.shape[1],
                self.hidden_dim,
                device=pair_features.device,
                dtype=pair_features.dtype,
            )
        safe_features = torch.where(pair_valid.unsqueeze(-1), pair_features, torch.zeros_like(pair_features))
        pair_embedding = self.interaction_pair_encoder(safe_features)
        pair_embedding = torch.where(pair_valid.unsqueeze(-1), pair_embedding, torch.zeros_like(pair_embedding))
        scores = self.interaction_pair_score(pair_embedding).squeeze(-1)
        if pair_features.shape[-1] >= 8:
            scores = scores + safe_features[..., -1]
        scores = torch.where(pair_valid, scores, torch.full_like(scores, -1e4))
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(pair_valid, weights, torch.zeros_like(weights))
        weight_sum = weights.sum(dim=-1, keepdim=True)
        weights = torch.where(weight_sum > 0.0, weights / torch.clamp(weight_sum, min=1e-6), torch.zeros_like(weights))
        context = torch.einsum("bij,bijh->bih", weights, pair_embedding)
        return torch.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)

    def predict_learned_interaction_field(
        self,
        scene_context: torch.Tensor,
        current_states: torch.Tensor,
        agent_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        if (
            not self.learned_interaction_field_enabled
            or self.learned_interaction_pair_input is None
            or self.learned_interaction_pair_to_agent is None
            or self.learned_interaction_pair_to_agent_score is None
            or self.learned_interaction_agent_update is None
            or self.learned_interaction_pair_update is None
            or self.learned_interaction_pair_projector is None
            or self.learned_interaction_context_proj is None
            or self.learned_interaction_context_score is None
        ):
            return (
                torch.zeros(
                    current_states.shape[0],
                    current_states.shape[1],
                    current_states.shape[1],
                    self.learned_interaction_feature_dim,
                    device=current_states.device,
                    dtype=current_states.dtype,
                ),
                torch.zeros(
                    current_states.shape[0],
                    current_states.shape[1],
                    current_states.shape[1],
                    device=current_states.device,
                    dtype=torch.bool,
                ),
                None,
                torch.zeros(
                    current_states.shape[0],
                    current_states.shape[1],
                    current_states.shape[1],
                    self.learned_interaction_field_latent_dim,
                    device=current_states.device,
                    dtype=current_states.dtype,
                ),
            )
        pair_raw_features, pair_valid = extract_current_pair_features(current_states=current_states, agent_mask=agent_mask)
        agent_state = torch.nan_to_num(scene_context, nan=0.0, posinf=0.0, neginf=0.0)
        context_i = agent_state.unsqueeze(2).expand(-1, -1, current_states.shape[1], -1)
        context_j = agent_state.unsqueeze(1).expand(-1, current_states.shape[1], -1, -1)
        pair_input = torch.cat([context_i, context_j, pair_raw_features], dim=-1)
        pair_latent = self.learned_interaction_pair_input(pair_input)
        pair_latent = torch.where(pair_valid.unsqueeze(-1), pair_latent, torch.zeros_like(pair_latent))
        pair_latent = torch.nan_to_num(pair_latent, nan=0.0, posinf=0.0, neginf=0.0)

        for _ in range(max(self.learned_interaction_num_message_passing_steps, 0)):
            pair_messages = self.learned_interaction_pair_to_agent(pair_latent)
            pair_messages = torch.where(pair_valid.unsqueeze(-1), pair_messages, torch.zeros_like(pair_messages))
            pair_scores = self.learned_interaction_pair_to_agent_score(pair_latent).squeeze(-1)
            pair_scores = pair_scores - pair_raw_features[..., 2]
            pair_scores = torch.where(pair_valid, pair_scores, torch.full_like(pair_scores, -1e4))
            pair_weights = torch.softmax(pair_scores, dim=-1)
            pair_weights = torch.where(pair_valid, pair_weights, torch.zeros_like(pair_weights))
            pair_weight_sum = pair_weights.sum(dim=-1, keepdim=True)
            pair_weights = torch.where(
                pair_weight_sum > 0.0,
                pair_weights / torch.clamp(pair_weight_sum, min=1e-6),
                torch.zeros_like(pair_weights),
            )
            aggregated_agent_message = torch.einsum("bij,bijh->bih", pair_weights, pair_messages)
            updated_agent_state = self.learned_interaction_agent_update(torch.cat([agent_state, aggregated_agent_message], dim=-1))
            updated_agent_state = torch.nan_to_num(updated_agent_state, nan=0.0, posinf=0.0, neginf=0.0)
            agent_state = torch.where(agent_mask.unsqueeze(-1), agent_state + updated_agent_state, torch.zeros_like(agent_state))

            context_i = agent_state.unsqueeze(2).expand(-1, -1, current_states.shape[1], -1)
            context_j = agent_state.unsqueeze(1).expand(-1, current_states.shape[1], -1, -1)
            pair_update_input = torch.cat([pair_latent, context_i, context_j, pair_raw_features], dim=-1)
            pair_delta = self.learned_interaction_pair_update(pair_update_input)
            pair_delta = torch.where(pair_valid.unsqueeze(-1), pair_delta, torch.zeros_like(pair_delta))
            pair_latent = torch.where(pair_valid.unsqueeze(-1), pair_latent + pair_delta, torch.zeros_like(pair_latent))
            pair_latent = torch.nan_to_num(pair_latent, nan=0.0, posinf=0.0, neginf=0.0)

        pair_prediction = self.learned_interaction_pair_projector(pair_latent)
        pair_prediction = torch.where(pair_valid.unsqueeze(-1), pair_prediction, torch.zeros_like(pair_prediction))
        pair_prediction = torch.nan_to_num(pair_prediction, nan=0.0, posinf=0.0, neginf=0.0)

        context_embedding = self.learned_interaction_context_proj(pair_latent)
        context_embedding = torch.where(pair_valid.unsqueeze(-1), context_embedding, torch.zeros_like(context_embedding))
        context_scores = self.learned_interaction_context_score(pair_latent).squeeze(-1) + pair_prediction[..., 7]
        context_scores = torch.where(pair_valid, context_scores, torch.full_like(context_scores, -1e4))
        context_weights = torch.softmax(context_scores, dim=-1)
        context_weights = torch.where(pair_valid, context_weights, torch.zeros_like(context_weights))
        context_weight_sum = context_weights.sum(dim=-1, keepdim=True)
        context_weights = torch.where(
            context_weight_sum > 0.0,
            context_weights / torch.clamp(context_weight_sum, min=1e-6),
            torch.zeros_like(context_weights),
        )
        interaction_context = torch.einsum("bij,bijh->bih", context_weights, context_embedding)
        interaction_context = torch.nan_to_num(interaction_context, nan=0.0, posinf=0.0, neginf=0.0)
        return pair_prediction, pair_valid, interaction_context, pair_latent

    def _aggregate_teacher_forced_neighbor_context(
        self,
        current_states: torch.Tensor,
        agent_mask: torch.Tensor,
        neighbor_features: torch.Tensor,
        neighbor_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.teacher_forced_neighbor_encoder is None:
            return torch.zeros(
                current_states.shape[0],
                current_states.shape[1],
                self.hidden_dim,
                device=current_states.device,
                dtype=current_states.dtype,
            )
        safe_features = torch.where(neighbor_valid_mask.unsqueeze(-1), neighbor_features, torch.zeros_like(neighbor_features))
        neighbor_embedding = self.teacher_forced_neighbor_encoder(safe_features)
        num_agents = int(agent_mask.shape[1])
        eye = torch.eye(num_agents, device=current_states.device, dtype=torch.bool).unsqueeze(0)
        pair_valid = agent_mask.unsqueeze(2) & neighbor_valid_mask.unsqueeze(1) & (~eye)
        current_rel = torch.nan_to_num(current_states[:, :, None, 0:2] - current_states[:, None, :, 0:2], nan=0.0, posinf=0.0, neginf=0.0)
        current_distance = torch.linalg.norm(current_rel, dim=-1)
        scores = -current_distance
        scores = torch.where(pair_valid, scores, torch.full_like(scores, -1e4))
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(pair_valid, weights, torch.zeros_like(weights))
        weight_sum = weights.sum(dim=-1, keepdim=True)
        weights = torch.where(weight_sum > 0.0, weights / torch.clamp(weight_sum, min=1e-6), torch.zeros_like(weights))
        context = torch.einsum("bij,bjh->bih", weights, neighbor_embedding)
        return torch.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)

    def build_interaction_oracle_context(
        self,
        current_states: torch.Tensor,
        agent_mask: torch.Tensor,
        future_states: torch.Tensor | None = None,
        future_mask: torch.Tensor | None = None,
        override_pair_features: torch.Tensor | None = None,
        override_pair_valid: torch.Tensor | None = None,
        override_neighbor_features: torch.Tensor | None = None,
        override_neighbor_valid_mask: torch.Tensor | None = None,
        mode_override: str | None = None,
    ) -> torch.Tensor | None:
        mode = str(mode_override or self.interaction_oracle_mode)
        if mode == "none":
            return None
        if oracle_mode_uses_pair_features(mode):
            if override_pair_features is not None:
                pair_features = torch.nan_to_num(override_pair_features, nan=0.0, posinf=0.0, neginf=0.0)
                if override_pair_valid is None:
                    num_agents = int(agent_mask.shape[1])
                    eye = torch.eye(num_agents, device=agent_mask.device, dtype=torch.bool).unsqueeze(0)
                    pair_valid = agent_mask.unsqueeze(2) & agent_mask.unsqueeze(1) & (~eye)
                else:
                    pair_valid = override_pair_valid.bool()
            elif future_states is not None and future_mask is not None:
                if mode == "static_summary":
                    pair_features, pair_valid = extract_static_oracle_pair_features(
                        current_states=current_states,
                        future_states=future_states,
                        future_mask=future_mask,
                        agent_mask=agent_mask,
                        conflict_distance_m=self.interaction_oracle_conflict_distance_m,
                        conflict_temperature_m=self.interaction_oracle_conflict_temperature_m,
                    )
                elif mode == "dynamic_trace":
                    pair_features, pair_valid = extract_dynamic_oracle_pair_features(
                        current_states=current_states,
                        future_states=future_states,
                        future_mask=future_mask,
                        agent_mask=agent_mask,
                        num_trace_points=self.interaction_oracle_num_trace_points,
                    )
                else:
                    raise ValueError(f"Unsupported interaction_oracle mode: {mode}")
            else:
                return None
            return self._aggregate_pairwise_interaction_context(pair_features, pair_valid)
        if oracle_mode_uses_teacher_forced_neighbors(mode):
            if override_neighbor_features is not None:
                neighbor_features = torch.nan_to_num(override_neighbor_features, nan=0.0, posinf=0.0, neginf=0.0)
                neighbor_valid_mask = override_neighbor_valid_mask.bool() if override_neighbor_valid_mask is not None else agent_mask
            elif future_states is not None and future_mask is not None:
                neighbor_features, neighbor_valid_mask = extract_teacher_forced_neighbor_summaries(
                    current_states=current_states,
                    future_states=future_states,
                    future_mask=future_mask,
                    agent_mask=agent_mask,
                )
            else:
                return None
            return self._aggregate_teacher_forced_neighbor_context(
                current_states=current_states,
                agent_mask=agent_mask,
                neighbor_features=neighbor_features,
                neighbor_valid_mask=neighbor_valid_mask,
            )
        raise ValueError(f"Unsupported interaction_oracle mode: {mode}")

    def predict_anchor(
        self,
        history_states: torch.Tensor,
        history_mask: torch.Tensor,
        current_states: torch.Tensor,
        agent_mask: torch.Tensor,
        map_polylines: torch.Tensor,
        map_point_mask: torch.Tensor,
        map_polyline_mask: torch.Tensor,
        scene_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if scene_context is None:
            scene_context = self._build_scene_context(
                history_states=history_states,
                history_mask=history_mask,
                current_states=current_states,
                agent_mask=agent_mask,
                map_polylines=map_polylines,
                map_point_mask=map_point_mask,
                map_polyline_mask=map_polyline_mask,
            )
        anchor_mu = self.anchor_predictor(scene_context, agent_mask)
        anchor_mu = torch.nan_to_num(anchor_mu, nan=0.0, posinf=0.0, neginf=0.0)
        _assert_finite_on_mask("anchor_predictor output", anchor_mu, agent_mask)
        return anchor_mu, scene_context

    def predict_anchor_residual(
        self,
        noisy_future: torch.Tensor,
        timesteps: torch.Tensor,
        history_states: torch.Tensor,
        history_mask: torch.Tensor,
        current_states: torch.Tensor,
        agent_mask: torch.Tensor,
        map_polylines: torch.Tensor,
        map_point_mask: torch.Tensor,
        map_polyline_mask: torch.Tensor,
        scene_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if scene_context is None:
            scene_context = self._build_scene_context(
                history_states=history_states,
                history_mask=history_mask,
                current_states=current_states,
                agent_mask=agent_mask,
                map_polylines=map_polylines,
                map_point_mask=map_point_mask,
                map_polyline_mask=map_polyline_mask,
            )
        time_context = self.time_embedding(timesteps)
        anchor_context = scene_context + time_context.unsqueeze(1)
        anchor_context = torch.nan_to_num(anchor_context, nan=0.0, posinf=0.0, neginf=0.0)
        prediction = self.anchor_residual_denoiser(
            noisy_anchor=noisy_future,
            anchor_mask=agent_mask,
            context=anchor_context,
        )
        prediction = torch.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0)
        _assert_finite_on_mask("anchor_residual_denoiser output", prediction, agent_mask)
        return prediction

    def predict_control_knots(
        self,
        history_states: torch.Tensor,
        history_mask: torch.Tensor,
        current_states: torch.Tensor,
        agent_mask: torch.Tensor,
        map_polylines: torch.Tensor,
        map_point_mask: torch.Tensor,
        map_polyline_mask: torch.Tensor,
        target_difficulty: torch.Tensor | None = None,
        target_behavior: torch.Tensor | None = None,
        anchor_features: torch.Tensor | None = None,
        scene_context: torch.Tensor | None = None,
        interaction_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if scene_context is None:
            scene_context = self._build_scene_context(
                history_states=history_states,
                history_mask=history_mask,
                current_states=current_states,
                agent_mask=agent_mask,
                map_polylines=map_polylines,
                map_point_mask=map_point_mask,
                map_polyline_mask=map_polyline_mask,
            )
        if self.generator_condition_mode != "none":
            conditioning = self.compute_condition_embedding(
                target_difficulty=target_difficulty if self.generator_consumes_difficulty else None,
                target_behavior=target_behavior if self.generator_consumes_behavior else None,
            )
            scene_context = scene_context + conditioning.unsqueeze(1)
        if interaction_context is not None:
            interaction_context = torch.nan_to_num(interaction_context, nan=0.0, posinf=0.0, neginf=0.0)
            scene_context = scene_context + interaction_context
        if self.use_anchor_latent:
            if anchor_features is None:
                raise ValueError("predict_control_knots requires anchor_features when model.use_anchor_latent=true.")
            anchor_features = torch.nan_to_num(anchor_features, nan=0.0, posinf=0.0, neginf=0.0)
            anchor_embedding = self.anchor_embedding(anchor_features)
            anchor_embedding = torch.nan_to_num(anchor_embedding, nan=0.0, posinf=0.0, neginf=0.0)
            scene_context = torch.cat([scene_context, anchor_embedding], dim=-1)
        scene_context = torch.nan_to_num(scene_context, nan=0.0, posinf=0.0, neginf=0.0)
        knots = self.control_knot_backbone(scene_context, agent_mask)
        _assert_finite_on_mask(
            "control_knot_backbone output",
            knots,
            agent_mask.unsqueeze(-1).expand(-1, -1, self.num_control_knots),
        )
        return knots

    def _build_context(
        self,
        history_states: torch.Tensor,
        history_mask: torch.Tensor,
        current_states: torch.Tensor,
        agent_mask: torch.Tensor,
        map_polylines: torch.Tensor,
        map_point_mask: torch.Tensor,
        map_polyline_mask: torch.Tensor,
        target_difficulty: torch.Tensor | None,
        target_behavior: torch.Tensor | None,
        timesteps: torch.Tensor,
        training: bool,
        condition_force_drop: bool = False,
        interaction_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scene_context = self._build_scene_context(
            history_states=history_states,
            history_mask=history_mask,
            current_states=current_states,
            agent_mask=agent_mask,
            map_polylines=map_polylines,
            map_point_mask=map_point_mask,
            map_polyline_mask=map_polyline_mask,
        )
        difficulty_context = self.compute_condition_embedding(
            target_difficulty=target_difficulty,
            target_behavior=target_behavior,
            force_drop=condition_force_drop,
        )
        time_context = self.time_embedding(timesteps)
        scene_context = torch.nan_to_num(scene_context, nan=0.0, posinf=0.0, neginf=0.0)
        difficulty_context = torch.nan_to_num(difficulty_context, nan=0.0, posinf=0.0, neginf=0.0)
        time_context = torch.nan_to_num(time_context, nan=0.0, posinf=0.0, neginf=0.0)
        if interaction_context is not None:
            interaction_context = torch.nan_to_num(interaction_context, nan=0.0, posinf=0.0, neginf=0.0)
            scene_context = scene_context + interaction_context
        context = scene_context + difficulty_context.unsqueeze(1) + time_context.unsqueeze(1)
        return torch.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)

    def predict_noise(
        self,
        noisy_future: torch.Tensor,
        timesteps: torch.Tensor,
        history_states: torch.Tensor,
        history_mask: torch.Tensor,
        current_states: torch.Tensor,
        future_mask: torch.Tensor,
        agent_mask: torch.Tensor,
        map_polylines: torch.Tensor,
        map_point_mask: torch.Tensor,
        map_polyline_mask: torch.Tensor,
        target_difficulty: torch.Tensor | None = None,
        target_behavior: torch.Tensor | None = None,
        condition_force_drop: bool = False,
        interaction_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = self._build_context(
            history_states=history_states,
            history_mask=history_mask,
            current_states=current_states,
            agent_mask=agent_mask,
            map_polylines=map_polylines,
            map_point_mask=map_point_mask,
            map_polyline_mask=map_polyline_mask,
            target_difficulty=target_difficulty,
            target_behavior=target_behavior,
            timesteps=timesteps,
            training=self.training,
            condition_force_drop=condition_force_drop,
            interaction_context=interaction_context,
        )
        noisy_future = torch.nan_to_num(noisy_future, nan=0.0, posinf=0.0, neginf=0.0)
        context = torch.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)
        predicted = self.denoiser(noisy_future, future_mask, agent_mask, context)
        return torch.nan_to_num(predicted, nan=0.0, posinf=0.0, neginf=0.0)

    def _sample_training_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.timestep_sampling == "uniform":
            return self.scheduler.sample_timesteps(batch_size)
        if self.timestep_sampling == "uniform_low_bias":
            values = torch.rand(batch_size, device=device)
            return torch.clamp((values.square() * float(self.scheduler.config.num_steps)).long(), max=self.scheduler.config.num_steps - 1)
        raise ValueError(f"Unsupported timestep_sampling strategy: {self.timestep_sampling}")

    def forward(
        self,
        history_states: torch.Tensor,
        history_mask: torch.Tensor,
        current_states: torch.Tensor,
        future_states: torch.Tensor,
        future_mask: torch.Tensor,
        agent_mask: torch.Tensor,
        map_polylines: torch.Tensor,
        map_point_mask: torch.Tensor,
        map_polyline_mask: torch.Tensor,
        target_difficulty: torch.Tensor | None = None,
        target_behavior: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        device = future_states.device
        self.scheduler = self.scheduler.to(device)
        valid_mask = future_mask & agent_mask.unsqueeze(-1)
        if self.decoder_type == "kinematic_controls" and self.control_representation == "knots":
            gt_knots, knot_mask, raw_target_controls = self.encode_future_controls_to_knots(
                current_states=current_states,
                future_states=future_states,
                future_mask=future_mask,
                agent_mask=agent_mask,
                dt=float(self.config["data"]["timestep_sec"]),
            )
            _assert_finite_on_mask("gt_knots", gt_knots, knot_mask)
            gt_knots_decoded_future, gt_knots_raw_controls, gt_knots_controls_physical = self.decode_control_knots_to_future(
                current_states=current_states,
                control_knots=gt_knots,
                future_mask=future_mask,
                agent_mask=agent_mask,
                dt=float(self.config["data"]["timestep_sec"]),
            )
            scene_context = self._build_scene_context(
                history_states=history_states,
                history_mask=history_mask,
                current_states=current_states,
                agent_mask=agent_mask,
                map_polylines=map_polylines,
                map_point_mask=map_point_mask,
                map_polyline_mask=map_polyline_mask,
            )
            if self.learned_interaction_field_enabled:
                predicted_interaction_pair_features, interaction_pair_valid, interaction_context, predicted_interaction_pair_latent = self.predict_learned_interaction_field(
                    scene_context=scene_context,
                    current_states=current_states,
                    agent_mask=agent_mask,
                )
                oracle_interaction_pair_features, oracle_pair_valid = extract_static_oracle_pair_features(
                    current_states=current_states,
                    future_states=future_states,
                    future_mask=future_mask,
                    agent_mask=agent_mask,
                    conflict_distance_m=self.interaction_oracle_conflict_distance_m,
                    conflict_temperature_m=self.interaction_oracle_conflict_temperature_m,
                )
                interaction_pair_valid = interaction_pair_valid & oracle_pair_valid
            else:
                interaction_context = self.build_interaction_oracle_context(
                    current_states=current_states,
                    future_states=future_states,
                    future_mask=future_mask,
                    agent_mask=agent_mask,
                )
                predicted_interaction_pair_features = torch.zeros(
                    current_states.shape[0],
                    current_states.shape[1],
                    current_states.shape[1],
                    self.learned_interaction_feature_dim,
                    device=current_states.device,
                    dtype=current_states.dtype,
                )
                oracle_interaction_pair_features = torch.zeros_like(predicted_interaction_pair_features)
                interaction_pair_valid = torch.zeros(
                    current_states.shape[0],
                    current_states.shape[1],
                    current_states.shape[1],
                    device=current_states.device,
                    dtype=torch.bool,
                )
                predicted_interaction_pair_latent = torch.zeros(
                    current_states.shape[0],
                    current_states.shape[1],
                    current_states.shape[1],
                    self.learned_interaction_field_latent_dim,
                    device=current_states.device,
                    dtype=current_states.dtype,
                )
            anchor_target, anchor_valid_mask = extract_anchor_targets(
                current_states=current_states,
                future_states=future_states,
                future_mask=future_mask,
                agent_mask=agent_mask,
            )
            if self.use_anchor_latent:
                anchor_mu, scene_context = self.predict_anchor(
                    history_states=history_states,
                    history_mask=history_mask,
                    current_states=current_states,
                    agent_mask=agent_mask,
                    map_polylines=map_polylines,
                    map_point_mask=map_point_mask,
                    map_polyline_mask=map_polyline_mask,
                    scene_context=scene_context,
                )
                anchor_mu = torch.where(anchor_valid_mask.unsqueeze(-1), anchor_mu, torch.zeros_like(anchor_mu))
                _assert_finite_on_mask("anchor_mu", anchor_mu, agent_mask)
            else:
                anchor_mu = torch.zeros_like(anchor_target)
            if self.training and self.use_anchor_latent and self.anchor_residual_diffusion_enabled and self.normalize_anchor_residual:
                anchor_residual_target_raw = torch.where(
                    anchor_valid_mask.unsqueeze(-1),
                    anchor_target - anchor_mu.detach(),
                    torch.zeros_like(anchor_target),
                )
                self.update_anchor_residual_stats(anchor_residual_target_raw, anchor_valid_mask)
            else:
                anchor_residual_target_raw = torch.where(
                    anchor_valid_mask.unsqueeze(-1),
                    anchor_target - anchor_mu.detach(),
                    torch.zeros_like(anchor_target),
                )
            anchor_residual_target_norm_unclipped = self.normalize_anchor_residual_target(anchor_residual_target_raw)
            anchor_residual_target = self.clip_anchor_residual_target_norm(anchor_residual_target_norm_unclipped)
            if self.use_anchor_latent:
                _assert_finite_on_mask("anchor_residual_target", anchor_residual_target, anchor_valid_mask)
            if self.use_anchor_latent and self.anchor_residual_diffusion_enabled:
                anchor_timesteps = self._sample_training_timesteps(future_states.shape[0], device)
                anchor_noise = torch.randn_like(anchor_residual_target)
                anchor_noise = torch.where(anchor_valid_mask.unsqueeze(-1), anchor_noise, torch.zeros_like(anchor_noise))
                noisy_anchor_residual = self.scheduler.q_sample(anchor_residual_target, anchor_timesteps, anchor_noise)
                noisy_anchor_residual = torch.where(
                    anchor_valid_mask.unsqueeze(-1),
                    noisy_anchor_residual,
                    torch.zeros_like(noisy_anchor_residual),
                )
                anchor_model_prediction = self.predict_anchor_residual(
                    noisy_future=noisy_anchor_residual,
                    timesteps=anchor_timesteps,
                    history_states=history_states,
                    history_mask=history_mask,
                    current_states=current_states,
                    agent_mask=agent_mask,
                    map_polylines=map_polylines,
                    map_point_mask=map_point_mask,
                    map_polyline_mask=map_polyline_mask,
                    scene_context=scene_context,
                )
                anchor_model_prediction = torch.where(
                    anchor_valid_mask.unsqueeze(-1),
                    anchor_model_prediction,
                    torch.zeros_like(anchor_model_prediction),
                )
                anchor_pred_x0 = self.scheduler.predict_start_from_model_output(
                    noisy_anchor_residual,
                    anchor_timesteps,
                    anchor_model_prediction,
                    self.anchor_diffusion_target_type,
                )
                anchor_pred_x0 = torch.where(anchor_valid_mask.unsqueeze(-1), anchor_pred_x0, torch.zeros_like(anchor_pred_x0))
                anchor_pred_x0 = self.clip_anchor_residual_target_norm(anchor_pred_x0)
                _assert_finite_on_mask("pred_x0 anchor residual", anchor_pred_x0, anchor_valid_mask)
                sampled_anchor_residual = self.denormalize_anchor_residual_target(anchor_pred_x0)
                sampled_anchor = torch.where(
                    anchor_valid_mask.unsqueeze(-1),
                    anchor_mu + sampled_anchor_residual,
                    torch.zeros_like(anchor_mu),
                )
            else:
                anchor_timesteps = self._sample_training_timesteps(future_states.shape[0], device)
                anchor_noise = torch.zeros_like(anchor_residual_target)
                noisy_anchor_residual = anchor_residual_target
                anchor_model_prediction = torch.zeros_like(anchor_residual_target)
                anchor_pred_x0 = torch.zeros_like(anchor_residual_target)
                sampled_anchor_residual = torch.zeros_like(anchor_residual_target_raw)
                sampled_anchor = anchor_mu
            anchor_features_for_knots = anchor_mu if self.use_anchor_latent else None
            sampled_anchor_features_for_knots = sampled_anchor if self.use_anchor_latent else None
            mu_knots = self.predict_control_knots(
                history_states=history_states,
                history_mask=history_mask,
                current_states=current_states,
                agent_mask=agent_mask,
                map_polylines=map_polylines,
                map_point_mask=map_point_mask,
                map_polyline_mask=map_polyline_mask,
                target_difficulty=target_difficulty if self.generator_consumes_difficulty else None,
                target_behavior=target_behavior if self.generator_consumes_behavior else None,
                anchor_features=anchor_features_for_knots,
                scene_context=scene_context,
                interaction_context=interaction_context,
            )
            mu_knots = torch.where(knot_mask.unsqueeze(-1), mu_knots, torch.zeros_like(mu_knots))
            _assert_finite_on_mask("mu_knots", mu_knots, knot_mask)
            mu_decoded_future, mu_raw_controls, mu_controls_physical = self.decode_control_knots_to_future(
                current_states=current_states,
                control_knots=mu_knots,
                future_mask=future_mask,
                agent_mask=agent_mask,
                dt=float(self.config["data"]["timestep_sec"]),
            )
            residual_target_raw = torch.where(knot_mask.unsqueeze(-1), gt_knots - mu_knots.detach(), torch.zeros_like(gt_knots))
            _assert_finite_on_mask("residual_target_raw", residual_target_raw, knot_mask)
            if self.training and self.residual_diffusion_enabled and self.normalize_residual:
                self.update_residual_stats(residual_target_raw, knot_mask)
            residual_target_norm_unclipped = self.normalize_residual_target(residual_target_raw)
            residual_target = self.clip_residual_target_norm(residual_target_norm_unclipped)
            _assert_finite_on_mask("residual_target", residual_target, knot_mask)
            timesteps = self._sample_training_timesteps(future_states.shape[0], device)
            noise = torch.randn_like(residual_target)
            noise = torch.where(knot_mask.unsqueeze(-1), noise, torch.zeros_like(noise))
            if self.residual_diffusion_enabled:
                noisy_future = self.scheduler.q_sample(residual_target, timesteps, noise)
                noisy_future = torch.where(knot_mask.unsqueeze(-1), noisy_future, torch.zeros_like(noisy_future))
                model_prediction = self.predict_noise(
                    noisy_future=noisy_future,
                    timesteps=timesteps,
                    history_states=history_states,
                    history_mask=history_mask,
                    current_states=current_states,
                    future_mask=knot_mask,
                    agent_mask=agent_mask,
                    map_polylines=map_polylines,
                    map_point_mask=map_point_mask,
                    map_polyline_mask=map_polyline_mask,
                    target_difficulty=target_difficulty if self.generator_consumes_difficulty else None,
                    target_behavior=target_behavior if self.generator_consumes_behavior else None,
                    interaction_context=interaction_context if self.learned_interaction_field_enabled and self.learned_interaction_field_condition_residual else None,
                )
                model_prediction = torch.where(knot_mask.unsqueeze(-1), model_prediction, torch.zeros_like(model_prediction))
                pred_x0 = self.scheduler.predict_start_from_model_output(
                    noisy_future,
                    timesteps,
                    model_prediction,
                    self.diffusion_target_type,
                )
                pred_x0 = torch.where(knot_mask.unsqueeze(-1), pred_x0, torch.zeros_like(pred_x0))
                pred_x0 = self.clip_residual_target_norm(pred_x0)
                _assert_finite_on_mask("pred_x0 residual knots", pred_x0, knot_mask)
                sampled_residual_knots = self.denormalize_residual_target(pred_x0)
            else:
                noisy_future = residual_target
                model_prediction = torch.zeros_like(residual_target)
                pred_x0 = torch.zeros_like(residual_target)
                sampled_residual_knots = torch.zeros_like(residual_target_raw)
            _assert_finite_on_mask("sampled_residual_knots", sampled_residual_knots, knot_mask)
            if self.use_anchor_latent and self.anchor_residual_diffusion_enabled:
                sampled_anchor_conditioned_knots = self.predict_control_knots(
                    history_states=history_states,
                    history_mask=history_mask,
                    current_states=current_states,
                    agent_mask=agent_mask,
                    map_polylines=map_polylines,
                    map_point_mask=map_point_mask,
                    map_polyline_mask=map_polyline_mask,
                    target_difficulty=target_difficulty if self.generator_consumes_difficulty else None,
                    target_behavior=target_behavior if self.generator_consumes_behavior else None,
                    anchor_features=sampled_anchor_features_for_knots,
                    scene_context=scene_context,
                    interaction_context=interaction_context,
                )
                sampled_anchor_conditioned_knots = torch.where(
                    knot_mask.unsqueeze(-1),
                    sampled_anchor_conditioned_knots,
                    torch.zeros_like(sampled_anchor_conditioned_knots),
                )
            else:
                sampled_anchor_conditioned_knots = mu_knots
            sampled_knots = (
                mu_knots + sampled_residual_knots
                if self.residual_diffusion_enabled
                else sampled_anchor_conditioned_knots
            )
            _assert_finite_on_mask("sampled_knots", sampled_knots, knot_mask)
            sampled_decoded_future, sampled_raw_controls, sampled_controls_physical = self.decode_control_knots_to_future(
                current_states=current_states,
                control_knots=sampled_knots,
                future_mask=future_mask,
                agent_mask=agent_mask,
                dt=float(self.config["data"]["timestep_sec"]),
            )
            if self.use_anchor_latent and self.anchor_residual_diffusion_enabled and not self.residual_diffusion_enabled:
                encoded_target = anchor_residual_target
                noise = anchor_noise
                noisy_future = noisy_anchor_residual
                model_prediction = anchor_model_prediction
                pred_x0 = anchor_pred_x0
            else:
                encoded_target = residual_target
            pred_x0_raw_controls = sampled_raw_controls
            pred_x0_raw_controls_clamped = sampled_raw_controls
            decoded_future = mu_decoded_future
            decoded_controls_physical = mu_controls_physical
        elif self.decoder_type == "kinematic_controls":
            raw_target_controls = self.encode_future_controls(
                current_states=current_states,
                future_states=future_states,
                future_mask=future_mask,
                agent_mask=agent_mask,
                dt=float(self.config["data"]["timestep_sec"]),
            )
            encoded_target = self.normalize_controls(raw_target_controls)
            encoded_target = torch.where(valid_mask.unsqueeze(-1), encoded_target, torch.zeros_like(encoded_target))
            timesteps = self._sample_training_timesteps(future_states.shape[0], device)
            noise = torch.randn_like(encoded_target)
            noise = torch.where(valid_mask.unsqueeze(-1), noise, torch.zeros_like(noise))
            noisy_future = self.scheduler.q_sample(encoded_target, timesteps, noise)
            noisy_future = torch.where(valid_mask.unsqueeze(-1), noisy_future, torch.zeros_like(noisy_future))
            model_prediction = self.predict_noise(
                noisy_future=noisy_future,
                timesteps=timesteps,
                history_states=history_states,
                history_mask=history_mask,
                current_states=current_states,
                future_mask=future_mask,
                agent_mask=agent_mask,
                map_polylines=map_polylines,
                map_point_mask=map_point_mask,
                map_polyline_mask=map_polyline_mask,
                target_difficulty=target_difficulty if self.generator_consumes_difficulty else None,
                target_behavior=target_behavior if self.generator_consumes_behavior else None,
            )
            predicted_interaction_pair_features = torch.zeros(
                current_states.shape[0],
                current_states.shape[1],
                current_states.shape[1],
                self.learned_interaction_feature_dim,
                device=current_states.device,
                dtype=current_states.dtype,
            )
            oracle_interaction_pair_features = torch.zeros_like(predicted_interaction_pair_features)
            interaction_pair_valid = torch.zeros(
                current_states.shape[0],
                current_states.shape[1],
                current_states.shape[1],
                device=current_states.device,
                dtype=torch.bool,
            )
            predicted_interaction_pair_latent = torch.zeros(
                current_states.shape[0],
                current_states.shape[1],
                current_states.shape[1],
                self.learned_interaction_field_latent_dim,
                device=current_states.device,
                dtype=current_states.dtype,
            )
            model_prediction = torch.where(valid_mask.unsqueeze(-1), model_prediction, torch.zeros_like(model_prediction))
            pred_x0 = self.scheduler.predict_start_from_model_output(
                noisy_future,
                timesteps,
                model_prediction,
                self.diffusion_target_type,
            )
            pred_x0 = torch.where(valid_mask.unsqueeze(-1), pred_x0, torch.zeros_like(pred_x0))
            pred_x0_raw_controls = self.denormalize_controls(pred_x0)
            pred_x0_raw_controls_clamped = self.clamp_raw_controls_to_train_quantiles(pred_x0_raw_controls)
            decoded_future, decoded_controls_physical = self.decode_controls_to_future(
                current_states=current_states,
                controls=pred_x0_raw_controls_clamped,
                future_mask=future_mask,
                agent_mask=agent_mask,
                dt=float(self.config["data"]["timestep_sec"]),
            )
            gt_knots = encoded_target
            knot_mask = future_mask
            mu_knots = pred_x0
            mu_decoded_future = decoded_future
            mu_raw_controls = pred_x0_raw_controls_clamped
            mu_controls_physical = decoded_controls_physical
            residual_target_raw = encoded_target
            residual_target_norm_unclipped = encoded_target
            residual_target = encoded_target
            sampled_knots = pred_x0
            sampled_residual_knots = pred_x0
            sampled_decoded_future = decoded_future
            sampled_controls_physical = decoded_controls_physical
            gt_knots_decoded_future = decoded_future
            gt_knots_raw_controls = pred_x0_raw_controls_clamped
            gt_knots_controls_physical = decoded_controls_physical
        else:
            encoded_target = self.encode_future_local(current_states, future_states)
            encoded_target = torch.where(valid_mask.unsqueeze(-1), encoded_target, torch.zeros_like(encoded_target))
            raw_target_controls = torch.zeros(
                (*encoded_target.shape[:-1], self.control_dim),
                dtype=encoded_target.dtype,
                device=encoded_target.device,
            )
            timesteps = self._sample_training_timesteps(future_states.shape[0], device)
            noise = torch.randn_like(encoded_target)
            noise = torch.where(valid_mask.unsqueeze(-1), noise, torch.zeros_like(noise))
            noisy_future = self.scheduler.q_sample(encoded_target, timesteps, noise)
            noisy_future = torch.where(valid_mask.unsqueeze(-1), noisy_future, torch.zeros_like(noisy_future))
            model_prediction = self.predict_noise(
                noisy_future=noisy_future,
                timesteps=timesteps,
                history_states=history_states,
                history_mask=history_mask,
                current_states=current_states,
                future_mask=future_mask,
                agent_mask=agent_mask,
                map_polylines=map_polylines,
                map_point_mask=map_point_mask,
                map_polyline_mask=map_polyline_mask,
                target_difficulty=target_difficulty if self.generator_consumes_difficulty else None,
                target_behavior=target_behavior if self.generator_consumes_behavior else None,
            )
            model_prediction = torch.where(valid_mask.unsqueeze(-1), model_prediction, torch.zeros_like(model_prediction))
            pred_x0 = self.scheduler.predict_start_from_model_output(
                noisy_future,
                timesteps,
                model_prediction,
                self.diffusion_target_type,
            )
            pred_x0 = torch.where(valid_mask.unsqueeze(-1), pred_x0, torch.zeros_like(pred_x0))
            decoded_future = self.decode_future_local(current_states, pred_x0)
            decoded_controls_physical = torch.zeros(
                (*pred_x0.shape[:-1], self.control_dim),
                dtype=pred_x0.dtype,
                device=pred_x0.device,
            )
            pred_x0_raw_controls = torch.zeros_like(decoded_controls_physical)
            pred_x0_raw_controls_clamped = torch.zeros_like(decoded_controls_physical)
            gt_knots = encoded_target
            knot_mask = future_mask
            mu_knots = pred_x0
            mu_decoded_future = decoded_future
            mu_raw_controls = pred_x0_raw_controls_clamped
            mu_controls_physical = decoded_controls_physical
            residual_target_raw = encoded_target
            residual_target_norm_unclipped = encoded_target
            residual_target = encoded_target
            sampled_knots = pred_x0
            sampled_residual_knots = pred_x0
            sampled_decoded_future = decoded_future
            sampled_controls_physical = decoded_controls_physical
            gt_knots_decoded_future = decoded_future
            gt_knots_raw_controls = pred_x0_raw_controls_clamped
            gt_knots_controls_physical = decoded_controls_physical
        model_debug = {
            **self.get_runtime_metadata(),
            "encoded_target_shape": list(encoded_target.shape),
            "encoded_target_channel_names": list(self.control_channel_names if self.decoder_type == "kinematic_controls" else ["dx", "dy", "dvx", "dvy", "dheading"]),
        }
        anchor_zero = torch.zeros(
            (*current_states.shape[:-1], self.anchor_dim),
            device=current_states.device,
            dtype=current_states.dtype,
        )
        return {
            "target_encoded_future": encoded_target,
            "target_raw_controls": raw_target_controls,
            "noise": noise,
            "predicted_noise": model_prediction,
            "pred_x0": pred_x0,
            "pred_x0_raw_controls": pred_x0_raw_controls,
            "pred_x0_raw_controls_clamped": pred_x0_raw_controls_clamped,
            "decoded_future": decoded_future,
            "decoded_controls_physical": decoded_controls_physical,
            "mu_knots": mu_knots,
            "gt_knots": gt_knots,
            "knot_mask": knot_mask,
            "anchor_target": anchor_target if self.decoder_type == "kinematic_controls" and self.control_representation == "knots" else anchor_zero,
            "anchor_valid_mask": anchor_valid_mask if self.decoder_type == "kinematic_controls" and self.control_representation == "knots" else agent_mask,
            "anchor_mu": anchor_mu if self.decoder_type == "kinematic_controls" and self.control_representation == "knots" else anchor_zero,
            "sampled_anchor": sampled_anchor if self.decoder_type == "kinematic_controls" and self.control_representation == "knots" else anchor_zero,
            "anchor_residual_target_raw": anchor_residual_target_raw if self.decoder_type == "kinematic_controls" and self.control_representation == "knots" else anchor_zero,
            "anchor_residual_target_norm_unclipped": anchor_residual_target_norm_unclipped if self.decoder_type == "kinematic_controls" and self.control_representation == "knots" else anchor_zero,
            "anchor_residual_target": anchor_residual_target if self.decoder_type == "kinematic_controls" and self.control_representation == "knots" else anchor_zero,
            "anchor_noise": anchor_noise if self.decoder_type == "kinematic_controls" and self.control_representation == "knots" else anchor_zero,
            "anchor_noisy_future": noisy_anchor_residual if self.decoder_type == "kinematic_controls" and self.control_representation == "knots" else anchor_zero,
            "anchor_predicted_noise": anchor_model_prediction if self.decoder_type == "kinematic_controls" and self.control_representation == "knots" else anchor_zero,
            "anchor_pred_x0": anchor_pred_x0 if self.decoder_type == "kinematic_controls" and self.control_representation == "knots" else anchor_zero,
            "sampled_anchor_residual": sampled_anchor_residual if self.decoder_type == "kinematic_controls" and self.control_representation == "knots" else anchor_zero,
            "mu_decoded_future": mu_decoded_future,
            "mu_raw_controls": mu_raw_controls,
            "mu_controls_physical": mu_controls_physical,
            "gt_knots_decoded_future": gt_knots_decoded_future,
            "gt_knots_raw_controls": gt_knots_raw_controls,
            "gt_knots_controls_physical": gt_knots_controls_physical,
            "residual_target_raw": residual_target_raw,
            "residual_target_norm_unclipped": residual_target_norm_unclipped,
            "residual_target": residual_target,
            "sampled_residual_knots": sampled_residual_knots,
            "final_knots": sampled_knots,
            "sampled_knots": sampled_knots,
            "sampled_decoded_future": sampled_decoded_future,
            "sampled_controls_physical": sampled_controls_physical,
            "predicted_interaction_pair_features": predicted_interaction_pair_features,
            "oracle_interaction_pair_features": oracle_interaction_pair_features,
            "interaction_pair_valid": interaction_pair_valid,
            "predicted_interaction_pair_latent": predicted_interaction_pair_latent,
            "decoder_type": self.decoder_type,
            "timesteps": timesteps,
            "anchor_timesteps": anchor_timesteps if self.decoder_type == "kinematic_controls" and self.control_representation == "knots" else timesteps,
            "model_debug": model_debug,
        }

    @torch.no_grad()
    def sample(
        self,
        history_states: torch.Tensor,
        history_mask: torch.Tensor,
        current_states: torch.Tensor,
        future_mask: torch.Tensor,
        agent_mask: torch.Tensor,
        map_polylines: torch.Tensor,
        map_point_mask: torch.Tensor,
        map_polyline_mask: torch.Tensor,
        sample_steps: int,
        guidance_scale: float,
        sampler: str = "ddpm",
        ddim_eta: float = 0.0,
        initial_noise: torch.Tensor | None = None,
        target_difficulty: torch.Tensor | None = None,
        target_behavior: torch.Tensor | None = None,
        use_residual_diffusion: bool | None = None,
        use_anchor_residual_diffusion: bool | None = None,
        override_anchor_features: torch.Tensor | None = None,
        override_interaction_pair_features: torch.Tensor | None = None,
        override_interaction_pair_valid: torch.Tensor | None = None,
        override_teacher_forced_neighbor_features: torch.Tensor | None = None,
        override_teacher_forced_neighbor_valid_mask: torch.Tensor | None = None,
        override_interaction_mode: str | None = None,
        override_rollout_sampling_mode: str | None = None,
    ) -> torch.Tensor:
        device = current_states.device
        self.scheduler = self.scheduler.to(device)
        effective_guidance_scale = 1.0 if self.generator_condition_mode == "none" else float(guidance_scale)
        valid_mask = future_mask & agent_mask.unsqueeze(-1)
        use_residual = (
            bool(self.config.get("sampling", {}).get("use_residual_diffusion", True)) and self.residual_diffusion_enabled
            if use_residual_diffusion is None
            else bool(use_residual_diffusion)
        )
        use_anchor_residual = (
            bool(self.config.get("sampling", {}).get("use_anchor_residual_diffusion", False))
            and self.anchor_residual_diffusion_enabled
            if use_anchor_residual_diffusion is None
            else bool(use_anchor_residual_diffusion)
        )
        effective_sample_steps = int(sample_steps)
        config_sampling_steps = int(self.config.get("sampling", {}).get("sample_steps", effective_sample_steps))
        if use_residual and int(self.residual_sample_steps) > 0 and effective_sample_steps == config_sampling_steps:
            effective_sample_steps = int(self.residual_sample_steps)
        condition_embedding = self.compute_condition_embedding(
            target_difficulty=target_difficulty if self.generator_consumes_difficulty else None,
            target_behavior=target_behavior if self.generator_consumes_behavior else None,
            force_drop=False,
        )
        if self.decoder_type == "kinematic_controls" and self.control_representation == "knots":
            scene_context = self._build_scene_context(
                history_states=history_states,
                history_mask=history_mask,
                current_states=current_states,
                agent_mask=agent_mask,
                map_polylines=map_polylines,
                map_point_mask=map_point_mask,
                map_polyline_mask=map_polyline_mask,
            )
            interaction_context = self.build_interaction_oracle_context(
                current_states=current_states,
                agent_mask=agent_mask,
                override_pair_features=override_interaction_pair_features,
                override_pair_valid=override_interaction_pair_valid,
                override_neighbor_features=override_teacher_forced_neighbor_features,
                override_neighbor_valid_mask=override_teacher_forced_neighbor_valid_mask,
                mode_override=override_interaction_mode,
            )
            if self.learned_interaction_field_enabled:
                _, _, learned_interaction_context, _ = self.predict_learned_interaction_field(
                    scene_context=scene_context,
                    current_states=current_states,
                    agent_mask=agent_mask,
                )
                interaction_context = learned_interaction_context
            if self.use_anchor_latent:
                anchor_mu, scene_context = self.predict_anchor(
                    history_states=history_states,
                    history_mask=history_mask,
                    current_states=current_states,
                    agent_mask=agent_mask,
                    map_polylines=map_polylines,
                    map_point_mask=map_point_mask,
                    map_polyline_mask=map_polyline_mask,
                    scene_context=scene_context,
                )
            else:
                anchor_mu = torch.zeros(
                    (current_states.shape[0], current_states.shape[1], self.anchor_dim),
                    device=device,
                    dtype=current_states.dtype,
                )
            anchor_mask = agent_mask
            anchor_sampling_mode = "deterministic_anchor"
            if self.use_anchor_latent and override_anchor_features is not None:
                if tuple(override_anchor_features.shape) != tuple(anchor_mu.shape):
                    raise ValueError(
                        "override_anchor_features shape does not match expected anchor shape: "
                        f"got={tuple(override_anchor_features.shape)} expected={tuple(anchor_mu.shape)}"
                    )
                sampled_anchor = torch.where(
                    anchor_mask.unsqueeze(-1),
                    torch.nan_to_num(override_anchor_features, nan=0.0, posinf=0.0, neginf=0.0),
                    torch.zeros_like(anchor_mu),
                )
                sampled_anchor_residual = sampled_anchor - anchor_mu
                anchor_clip_fraction = torch.zeros((), device=device, dtype=current_states.dtype)
                anchor_sampling_mode = str(override_rollout_sampling_mode or "oracle_gt_anchor")
            elif self.use_anchor_latent and self.anchor_residual_diffusion_enabled and use_anchor_residual:
                if self.normalize_anchor_residual and self.anchor_residual_normalizer is None:
                    raise RuntimeError("Anchor residual diffusion is enabled for sampling but anchor residual normalizer is not loaded.")
                anchor_shape = (current_states.shape[0], current_states.shape[1], self.anchor_dim)
                anchor_initial_noise = initial_noise if initial_noise is not None and tuple(initial_noise.shape) == anchor_shape else None
                sampled_anchor_encoded = self.scheduler.sample_loop(
                    model=self.predict_anchor_residual,
                    shape=anchor_shape,
                    model_kwargs={
                        "history_states": history_states,
                        "history_mask": history_mask,
                        "current_states": current_states,
                        "agent_mask": agent_mask,
                        "map_polylines": map_polylines,
                        "map_point_mask": map_point_mask,
                        "map_polyline_mask": map_polyline_mask,
                        "scene_context": scene_context,
                    },
                    sample_steps=int(self.anchor_residual_sample_steps if self.anchor_residual_sample_steps > 0 else effective_sample_steps),
                    guidance_scale=1.0,
                    sampler=sampler,
                    ddim_eta=ddim_eta,
                    initial_noise=anchor_initial_noise,
                    target_type=self.anchor_diffusion_target_type,
                )
                sampled_anchor_encoded = torch.where(anchor_mask.unsqueeze(-1), sampled_anchor_encoded, torch.zeros_like(sampled_anchor_encoded))
                sampled_anchor_encoded_unclipped = sampled_anchor_encoded
                sampled_anchor_encoded = self.clip_anchor_residual_target_norm(sampled_anchor_encoded)
                sampled_anchor_residual = self.denormalize_anchor_residual_target(sampled_anchor_encoded)
                sampled_anchor = anchor_mu + self.anchor_residual_sample_scale * sampled_anchor_residual
                anchor_clip_fraction = _clip_fraction(sampled_anchor_encoded_unclipped, sampled_anchor_encoded, anchor_mask)
                anchor_sampling_mode = f"anchor_residual_{self.anchor_diffusion_target_type}_scale_{self.anchor_residual_sample_scale:.1f}"
            else:
                sampled_anchor_residual = torch.zeros_like(anchor_mu)
                sampled_anchor = anchor_mu
                anchor_clip_fraction = torch.zeros((), device=device, dtype=current_states.dtype)
            knot_anchor_features = sampled_anchor if self.use_anchor_latent else None
            mu_knots = self.predict_control_knots(
                history_states=history_states,
                history_mask=history_mask,
                current_states=current_states,
                agent_mask=agent_mask,
                map_polylines=map_polylines,
                map_point_mask=map_point_mask,
                map_polyline_mask=map_polyline_mask,
                target_difficulty=target_difficulty if self.generator_consumes_difficulty else None,
                target_behavior=target_behavior if self.generator_consumes_behavior else None,
                anchor_features=knot_anchor_features,
                scene_context=scene_context,
                interaction_context=interaction_context,
            )
            residual_mask = (
                torch.ones(
                    (current_states.shape[0], current_states.shape[1], self.num_control_knots),
                    device=device,
                    dtype=torch.bool,
                )
                & agent_mask.unsqueeze(-1)
            )
            residual_sampling_mode = "deterministic_backbone"
            if use_residual:
                if self.normalize_residual and self.residual_normalizer is None:
                    raise RuntimeError("Residual diffusion is enabled for sampling but residual normalizer is not loaded.")
                residual_shape = (current_states.shape[0], current_states.shape[1], self.num_control_knots, self.control_dim)
                residual_initial_noise = initial_noise if initial_noise is not None and tuple(initial_noise.shape) == residual_shape else None
                encoded = self.scheduler.sample_loop(
                    model=self.predict_noise,
                    shape=residual_shape,
                    model_kwargs={
                        "history_states": history_states,
                        "history_mask": history_mask,
                        "current_states": current_states,
                        "future_mask": residual_mask,
                        "agent_mask": agent_mask,
                        "map_polylines": map_polylines,
                        "map_point_mask": map_point_mask,
                        "map_polyline_mask": map_polyline_mask,
                        "target_difficulty": target_difficulty if self.generator_consumes_difficulty else None,
                        "target_behavior": target_behavior if self.generator_consumes_behavior else None,
                        "interaction_context": interaction_context if self.learned_interaction_field_enabled and self.learned_interaction_field_condition_residual else None,
                    },
                    sample_steps=effective_sample_steps,
                    guidance_scale=effective_guidance_scale,
                    sampler=sampler,
                    ddim_eta=ddim_eta,
                    initial_noise=residual_initial_noise,
                    target_type=self.diffusion_target_type,
                )
                encoded = torch.where(residual_mask.unsqueeze(-1), encoded, torch.zeros_like(encoded))
                encoded_unclipped = encoded
                encoded = self.clip_residual_target_norm(encoded)
                sampled_residual_knots = self.denormalize_residual_target(encoded)
                _assert_finite_on_mask("sampled_residual_knots", sampled_residual_knots, residual_mask)
                final_knots = mu_knots + self.residual_sample_scale * sampled_residual_knots
                residual_clip_fraction = _clip_fraction(encoded_unclipped, encoded, residual_mask)
                residual_sampling_mode = f"residual_{self.diffusion_target_type}_scale_{self.residual_sample_scale:.1f}"
            else:
                encoded = torch.zeros_like(mu_knots)
                sampled_residual_knots = torch.zeros_like(mu_knots)
                final_knots = mu_knots
                residual_clip_fraction = torch.zeros((), device=device, dtype=mu_knots.dtype)
            _assert_finite_on_mask(
                "final_knots",
                final_knots,
                agent_mask.unsqueeze(-1).expand(-1, -1, self.num_control_knots),
            )
            decoded_future, raw_controls_clamped, _ = self.decode_control_knots_to_future(
                current_states=current_states,
                control_knots=final_knots,
                future_mask=future_mask,
                agent_mask=agent_mask,
                dt=float(self.config["data"]["timestep_sec"]),
            )
            self.last_sample_debug = {
                **self.get_runtime_metadata(),
                "sample_steps": int(sample_steps),
                "guidance_scale": float(effective_guidance_scale),
                "sampler": str(sampler),
                "ddim_eta": float(ddim_eta),
                "use_residual_diffusion": bool(use_residual),
                "use_anchor_residual_diffusion": bool(self.use_anchor_latent and self.anchor_residual_diffusion_enabled and use_anchor_residual),
                "effective_sample_steps": int(effective_sample_steps),
                "rollout_sampling_mode": residual_sampling_mode if use_residual else anchor_sampling_mode,
                "residual_scale": float(self.residual_sample_scale),
                "anchor_residual_scale": float(self.anchor_residual_sample_scale),
                "sampled_residual_std": float(_masked_std(sampled_residual_knots, residual_mask).item()),
                "final_knot_std": float(_masked_std(final_knots, residual_mask).item()),
                "mu_knot_std": float(_masked_std(mu_knots, residual_mask).item()),
                "residual_clip_fraction": float(residual_clip_fraction.item()),
                "sampled_anchor_residual_std": float(_masked_std(sampled_anchor_residual, anchor_mask).item()),
                "anchor_mu_std": float(_masked_std(anchor_mu, anchor_mask).item()),
                "final_anchor_std": float(_masked_std(sampled_anchor, anchor_mask).item()),
                "anchor_residual_clip_fraction": float(anchor_clip_fraction.item()),
                "interaction_oracle_mode_used": str(override_interaction_mode or self.interaction_oracle_mode),
                "interaction_mode_used": "learned_static_summary" if self.learned_interaction_field_enabled else str(override_interaction_mode or self.interaction_oracle_mode),
                "guidance_semantics": (
                    "generator_condition.mode=none: conditional and unconditional branches are identical; "
                    "guidance is effectively disabled."
                    if self.generator_condition_mode == "none"
                    else "standard_cfg: prediction = uncond + guidance_scale * (cond - uncond); guidance_scale=1.0 equals the plain conditional prediction, guidance_scale=0.0 equals the unconditional prediction."
                ),
                "sampled_control_stats_raw": summarize_control_tensor(raw_controls_clamped, valid_mask, self.control_channel_names),
                "condition_embedding_norm_mean": float(condition_embedding.norm(dim=-1).mean().item()),
            }
            return decoded_future
        shape = (*future_mask.shape, self.model_future_dim)
        encoded = self.scheduler.sample_loop(
            model=self.predict_noise,
            shape=shape,
            model_kwargs={
                "history_states": history_states,
                "history_mask": history_mask,
                "current_states": current_states,
                "future_mask": future_mask,
                "agent_mask": agent_mask,
                "map_polylines": map_polylines,
                "map_point_mask": map_point_mask,
                "map_polyline_mask": map_polyline_mask,
                "target_difficulty": target_difficulty if self.generator_consumes_difficulty else None,
                "target_behavior": target_behavior if self.generator_consumes_behavior else None,
                "interaction_context": None,
            },
            sample_steps=sample_steps,
            guidance_scale=effective_guidance_scale,
            sampler=sampler,
            ddim_eta=ddim_eta,
            initial_noise=initial_noise,
            target_type=self.diffusion_target_type,
        )
        encoded = torch.where(valid_mask.unsqueeze(-1), encoded, torch.zeros_like(encoded))
        if self.decoder_type == "kinematic_controls":
            raw_controls = self.denormalize_controls(encoded)
            raw_controls_clamped = self.clamp_raw_controls_to_train_quantiles(raw_controls)
            decoded_future, _ = self.decode_controls_to_future(
                current_states=current_states,
                controls=raw_controls_clamped,
                future_mask=future_mask,
                agent_mask=agent_mask,
                dt=float(self.config["data"]["timestep_sec"]),
            )
            valid_mask = future_mask & agent_mask.unsqueeze(-1)
            self.last_sample_debug = {
                **self.get_runtime_metadata(),
                "sample_steps": int(sample_steps),
                "guidance_scale": float(effective_guidance_scale),
                "sampler": str(sampler),
                "ddim_eta": float(ddim_eta),
                "use_residual_diffusion": bool(use_residual),
                "guidance_semantics": (
                    "generator_condition.mode=none: conditional and unconditional branches are identical; "
                    "guidance is effectively disabled."
                    if self.generator_condition_mode == "none"
                    else "standard_cfg: prediction = uncond + guidance_scale * (cond - uncond); guidance_scale=1.0 equals the plain conditional prediction, guidance_scale=0.0 equals the unconditional prediction."
                ),
                "sampled_control_stats_raw": summarize_control_tensor(raw_controls_clamped, valid_mask, self.control_channel_names),
                "condition_embedding_norm_mean": float(condition_embedding.norm(dim=-1).mean().item()),
            }
            return decoded_future
        self.last_sample_debug = {
            **self.get_runtime_metadata(),
            "sample_steps": int(sample_steps),
            "guidance_scale": float(effective_guidance_scale),
            "sampler": str(sampler),
            "ddim_eta": float(ddim_eta),
            "use_residual_diffusion": bool(use_residual),
            "guidance_semantics": (
                "generator_condition.mode=none: conditional and unconditional branches are identical; "
                "guidance is effectively disabled."
                if self.generator_condition_mode == "none"
                else "standard_cfg: prediction = uncond + guidance_scale * (cond - uncond); guidance_scale=1.0 equals the plain conditional prediction, guidance_scale=0.0 equals the unconditional prediction."
            ),
            "condition_embedding_norm_mean": float(condition_embedding.norm(dim=-1).mean().item()),
        }
        return self.decode_future_local(current_states, encoded)
