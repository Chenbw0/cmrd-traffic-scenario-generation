from __future__ import annotations

import math
from typing import Dict, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from isgen.models.blocks import MLP
from isgen.models.encoders import DifficultyEmbedding, MapPolylineEncoder
from isgen.semantics.spawn_ordering import canonical_sort_scene
from isgen.semantics.spawn_plan import SUMMARY_FEATURE_NAMES, spawn_plan_feature_dim


def _wrap_heading(heading: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(heading), torch.cos(heading))


def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)


class SceneSpawnGenerator(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        spawn_cfg = config.get("spawn", {})
        prototype_cfg = config.get("spawn_prototypes", {})
        model_cfg = config["model"]
        data_cfg = config["data"]
        self.config = config
        self.architecture = str(spawn_cfg.get("architecture", "interaction_autoregressive"))
        self.hidden_dim = int(spawn_cfg.get("hidden_dim", model_cfg.get("hidden_dim", 128)))
        self.noise_dim = int(spawn_cfg.get("noise_dim", 64))
        self.num_ar_layers = int(spawn_cfg.get("num_ar_layers", spawn_cfg.get("num_slot_layers", 2)))
        self.num_ar_heads = int(spawn_cfg.get("num_ar_heads", spawn_cfg.get("num_slot_heads", model_cfg.get("num_heads", 4))))
        self.dropout = float(spawn_cfg.get("dropout", model_cfg.get("dropout", 0.1)))
        self.max_agents = int(data_cfg.get("max_agents", 24))
        self.state_dim = 5
        self.map_dim = 4
        self.plan_radial_bins = int(spawn_cfg.get("plan_radial_bins", 4))
        self.plan_angular_bins = int(spawn_cfg.get("plan_angular_bins", 8))
        self.plan_feature_dim = int(spawn_plan_feature_dim(self.plan_radial_bins, self.plan_angular_bins))
        self.summary_feature_dim = int(len(SUMMARY_FEATURE_NAMES))
        self.map_radius_m = float(data_cfg.get("map_radius_m", 80.0))
        self.max_speed_mps = float(spawn_cfg.get("max_speed_mps", 20.0))
        self.prototype_sampling_temperature = float(prototype_cfg.get("sampling_temperature", 1.0))
        self.prototype_rerank_topk = int(prototype_cfg.get("rerank_topk", 8))
        self.prototype_mixture_enabled = bool(prototype_cfg.get("mixture_enabled", False))
        self.prototype_mixture_topk = int(prototype_cfg.get("mixture_topk", 4))
        self.prototype_mixture_temperature = float(
            prototype_cfg.get("mixture_temperature", self.prototype_sampling_temperature)
        )
        self.prototype_mixture_target_logit_boost = float(
            prototype_cfg.get("mixture_target_logit_boost", 2.0)
        )
        self.prototype_support_bias_weight = float(prototype_cfg.get("support_bias_weight", 1.0))
        self.prototype_count_compat_weight = float(prototype_cfg.get("count_compat_weight", 2.0))
        self.prototype_difficulty_compat_weight = float(prototype_cfg.get("difficulty_compat_weight", 2.0))
        self.prototype_plan_compat_weight = float(prototype_cfg.get("plan_compat_weight", 2.0))
        self.prototype_plan_summary_compat_weight = float(
            prototype_cfg.get("plan_summary_compat_weight", 2.0)
        )
        self.prototype_plan_occupancy_compat_weight = float(
            prototype_cfg.get("plan_occupancy_compat_weight", 1.0)
        )
        self.prototype_count_residual_max_delta = int(prototype_cfg.get("count_residual_max_delta", 6))
        self.prototype_residual_position_scale_m = float(
            prototype_cfg.get("residual_position_scale_m", 12.0)
        )
        self.prototype_residual_speed_scale_mps = float(
            prototype_cfg.get("residual_speed_scale_mps", 3.0)
        )
        self.prototype_residual_heading_scale_rad = float(
            prototype_cfg.get("residual_heading_scale_rad", 0.35)
        )
        self.prototype_memory_enabled = bool(prototype_cfg.get("memory_cross_attention_enabled", True))
        self.structured_adaptation_enabled = bool(prototype_cfg.get("structured_adaptation_enabled", False))
        self.structured_anchor_stride = max(int(prototype_cfg.get("structured_anchor_stride", 2)), 1)
        self.structured_max_map_candidates = max(int(prototype_cfg.get("structured_max_map_candidates", 256)), 32)
        self.structured_shift_scale_m = float(
            prototype_cfg.get("structured_shift_scale_m", min(self.prototype_residual_position_scale_m, 8.0))
        )
        self.structured_scene_speed_scale_max_delta = float(
            prototype_cfg.get("structured_scene_speed_scale_max_delta", 0.4)
        )
        self.structured_agent_speed_residual_scale_mps = float(
            prototype_cfg.get("structured_agent_speed_residual_scale_mps", 2.0)
        )
        self.structured_pair_relaxation_enabled = bool(
            prototype_cfg.get("structured_pair_relaxation_enabled", False)
        )
        self.structured_pair_relaxation_iterations = max(
            int(prototype_cfg.get("structured_pair_relaxation_iterations", 2)),
            0,
        )
        self.structured_pair_relaxation_min_spacing_m = float(
            prototype_cfg.get("structured_pair_relaxation_min_spacing_m", 4.0)
        )
        self.structured_pair_relaxation_strength = float(
            prototype_cfg.get("structured_pair_relaxation_strength", 0.75)
        )
        self.structured_pair_relaxation_max_step_m = float(
            prototype_cfg.get("structured_pair_relaxation_max_step_m", 2.0)
        )
        self.structured_spacing_selection_enabled = bool(
            prototype_cfg.get("structured_spacing_selection_enabled", False)
        )
        self.structured_spacing_selection_min_spacing_m = float(
            prototype_cfg.get("structured_spacing_selection_min_spacing_m", 4.0)
        )
        self.structured_spacing_selection_fallback_spacing_m = float(
            prototype_cfg.get("structured_spacing_selection_fallback_spacing_m", 2.0)
        )
        self.structured_spacing_selection_final_fill_spacing_m = float(
            prototype_cfg.get("structured_spacing_selection_final_fill_spacing_m", 1.2)
        )
        self.structured_spacing_selection_fill_to_count = bool(
            prototype_cfg.get("structured_spacing_selection_fill_to_count", True)
        )
        self.hybrid_compiler_enabled = bool(prototype_cfg.get("hybrid_compiler_enabled", False))
        self.compiler_anchor_stride = max(int(prototype_cfg.get("compiler_anchor_stride", 2)), 1)
        self.compiler_max_map_candidates = max(int(prototype_cfg.get("compiler_max_map_candidates", 256)), 32)
        self.compiler_min_spacing_m = float(prototype_cfg.get("compiler_min_spacing_m", 1.5))
        self.compiler_proto_score_bias = float(prototype_cfg.get("compiler_proto_score_bias", 2.0))
        self.compiler_occupancy_weight = float(prototype_cfg.get("compiler_occupancy_weight", 2.0))
        self.compiler_radius_weight = float(prototype_cfg.get("compiler_radius_weight", 1.0))
        self.compiler_speed_weight = float(prototype_cfg.get("compiler_speed_weight", 1.0))
        self.compiler_target_speed_floor = float(prototype_cfg.get("compiler_target_speed_floor", 1.5))
        self.prototype_enabled = self.architecture == "support_guided_prototype"
        self.map_encoder = MapPolylineEncoder(self.map_dim, self.hidden_dim, self.dropout)
        self.difficulty_embedding = DifficultyEmbedding(self.hidden_dim, dropout_prob=0.0)
        self.noise_proj = MLP(self.noise_dim, self.hidden_dim, self.hidden_dim, self.dropout)
        self.scene_proj = MLP(self.hidden_dim * 3, self.hidden_dim, self.hidden_dim, self.dropout)
        self.plan_head = MLP(self.hidden_dim, self.hidden_dim, self.plan_feature_dim, self.dropout)
        self.plan_embedding = MLP(self.plan_feature_dim, self.hidden_dim, self.hidden_dim, self.dropout)
        self.count_prior_head = MLP(self.hidden_dim, self.hidden_dim, self.max_agents + 1, self.dropout)
        count_output_dim = (
            self.max_agents + 1
            if not self.prototype_enabled
            else (self.prototype_count_residual_max_delta * 2 + 1)
        )
        self.count_head = MLP(self.hidden_dim, self.hidden_dim, count_output_dim, self.dropout)
        self.count_context_proj = MLP(1, self.hidden_dim, self.hidden_dim, self.dropout)
        self.prototype_context_proj = MLP(self.plan_feature_dim + 1, self.hidden_dim, self.hidden_dim, self.dropout)
        self.prototype_state_proj = MLP(self.state_dim + 1, self.hidden_dim, self.hidden_dim, self.dropout)
        self.prototype_memory_proj = MLP(self.state_dim + 1, self.hidden_dim, self.hidden_dim, self.dropout)
        self.prototype_support_query_proj = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, self.dropout)
        self.prototype_support_key_proj = MLP(self.plan_feature_dim + 2, self.hidden_dim, self.hidden_dim, self.dropout)
        self.prototype_prior_head: nn.Module | None = None

        self.prev_state_proj = MLP(self.state_dim + 1, self.hidden_dim, self.hidden_dim, self.dropout)
        self.step_embedding = nn.Embedding(self.max_agents, self.hidden_dim)
        self.context_proj = MLP(self.hidden_dim * 4, self.hidden_dim, self.hidden_dim, self.dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_ar_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            batch_first=True,
        )
        self.autoregressive_decoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_ar_layers)
        self.prototype_memory_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_ar_heads,
            dropout=self.dropout,
            batch_first=True,
        )
        self.prototype_memory_norm = nn.LayerNorm(self.hidden_dim)
        self.prototype_memory_ffn = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, self.dropout)
        self.prototype_memory_ffn_norm = nn.LayerNorm(self.hidden_dim)
        self.bos_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.state_head = MLP(self.hidden_dim, self.hidden_dim, self.state_dim, self.dropout)
        self.presence_head = MLP(self.hidden_dim, self.hidden_dim, 1, self.dropout)
        self.prototype_keep_head = MLP(self.hidden_dim, self.hidden_dim, 1, self.dropout)
        self.prototype_shift_head = MLP(self.hidden_dim, self.hidden_dim, 1, self.dropout)
        self.prototype_speed_head = MLP(self.hidden_dim, self.hidden_dim, 1, self.dropout)
        self.prototype_scene_speed_scale_head = MLP(self.hidden_dim, self.hidden_dim, 1, self.dropout)
        self.register_buffer("prototype_states", torch.zeros(0, self.max_agents, self.state_dim), persistent=False)
        self.register_buffer("prototype_masks", torch.zeros(0, self.max_agents, dtype=torch.bool), persistent=False)
        self.register_buffer("prototype_plan_features", torch.zeros(0, self.plan_feature_dim), persistent=False)
        self.register_buffer("prototype_difficulty_mean", torch.zeros(0), persistent=False)
        self.register_buffer("prototype_counts", torch.zeros(0), persistent=False)
        self.prototype_location_ids: list[str] = []
        self.prototype_slice_ids: list[str] = []

    def set_support_prototype_bank(self, bank: Dict) -> None:
        self.prototype_states = bank["prototype_states"].to(device=self.bos_token.device, dtype=self.bos_token.dtype)
        self.prototype_masks = bank["prototype_masks"].to(device=self.bos_token.device)
        self.prototype_plan_features = bank["prototype_plan_features"].to(device=self.bos_token.device, dtype=self.bos_token.dtype)
        self.prototype_difficulty_mean = bank.get("prototype_difficulty_mean", torch.zeros(self.prototype_states.shape[0])).to(
            device=self.bos_token.device,
            dtype=self.bos_token.dtype,
        )
        self.prototype_counts = self.prototype_masks.float().sum(dim=-1).to(device=self.bos_token.device, dtype=self.bos_token.dtype)
        self.prototype_location_ids = list(bank.get("prototype_location_ids", []))
        self.prototype_slice_ids = list(bank.get("prototype_slice_ids", []))
        if int(self.prototype_states.shape[1]) != int(self.max_agents):
            bank_agents = int(self.prototype_states.shape[1])
            if bank_agents > self.max_agents:
                self.prototype_states = self.prototype_states[:, : self.max_agents]
                self.prototype_masks = self.prototype_masks[:, : self.max_agents]
            else:
                pad_agents = self.max_agents - bank_agents
                self.prototype_states = F.pad(self.prototype_states, (0, 0, 0, pad_agents))
                self.prototype_masks = F.pad(self.prototype_masks, (0, pad_agents), value=False)
        num_prototypes = int(self.prototype_states.shape[0])
        if num_prototypes <= 0:
            raise ValueError("Support-guided prototype bank is empty.")
        self.prototype_prior_head = MLP(self.hidden_dim, self.hidden_dim, num_prototypes, self.dropout).to(self.bos_token.device)

    def _prototype_support_logits(self, scene_context: torch.Tensor) -> torch.Tensor:
        if not self.prototype_enabled or int(self.prototype_plan_features.shape[0]) <= 0:
            return torch.zeros(scene_context.shape[0], 0, device=scene_context.device, dtype=scene_context.dtype)
        prototype_feature_bank = torch.cat(
            [
                self.prototype_plan_features,
                (self.prototype_counts / max(float(self.max_agents), 1.0)).unsqueeze(-1),
                self.prototype_difficulty_mean.unsqueeze(-1),
            ],
            dim=-1,
        )
        query = F.normalize(self.prototype_support_query_proj(scene_context), dim=-1)
        key = F.normalize(self.prototype_support_key_proj(prototype_feature_bank), dim=-1)
        return (query @ key.transpose(0, 1)) / math.sqrt(max(float(self.hidden_dim), 1.0))

    def _prototype_plan_compat_logits(self, scene_context: torch.Tensor) -> torch.Tensor:
        if not self.prototype_enabled or int(self.prototype_plan_features.shape[0]) <= 0:
            return torch.zeros(scene_context.shape[0], 0, device=scene_context.device, dtype=scene_context.dtype)
        query_plan = torch.sigmoid(self.plan_head(scene_context))
        query_summary = query_plan[:, : self.summary_feature_dim]
        query_occupancy = query_plan[:, self.summary_feature_dim :]
        prototype_summary = self.prototype_plan_features[:, : self.summary_feature_dim]
        prototype_occupancy = self.prototype_plan_features[:, self.summary_feature_dim :]
        summary_gap = torch.abs(query_summary.unsqueeze(1) - prototype_summary.unsqueeze(0)).mean(dim=-1)
        occupancy_gap = torch.abs(query_occupancy.unsqueeze(1) - prototype_occupancy.unsqueeze(0)).mean(dim=-1)
        return -(
            self.prototype_plan_summary_compat_weight * summary_gap
            + self.prototype_plan_occupancy_compat_weight * occupancy_gap
        )

    def _select_support_prototype(
        self,
        scene_context: torch.Tensor,
        target_difficulty: torch.Tensor,
        forced_prototype_ids: torch.Tensor | None = None,
        location_ids: Sequence[str] | None = None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        if not self.prototype_enabled:
            return None, None, None, None, None, None, None
        if self.prototype_prior_head is None or int(self.prototype_states.shape[0]) <= 0:
            raise RuntimeError("Prototype-guided spawn requires a loaded support prototype bank.")
        del location_ids
        base_logits = self.prototype_prior_head(scene_context)
        support_logits = self._prototype_support_logits(scene_context)
        plan_compat_logits = self._prototype_plan_compat_logits(scene_context)
        count_prior_logits = self.count_prior_head(scene_context)
        count_prior_probs = F.softmax(count_prior_logits, dim=-1)
        count_support = torch.arange(self.max_agents + 1, device=scene_context.device, dtype=scene_context.dtype)
        expected_count = (count_prior_probs * count_support.unsqueeze(0)).sum(dim=-1)
        count_compat = -torch.abs(
            self.prototype_counts.unsqueeze(0) - expected_count.unsqueeze(-1)
        ) / max(float(self.max_agents), 1.0)
        difficulty_compat = -torch.abs(
            self.prototype_difficulty_mean.unsqueeze(0) - target_difficulty.to(device=scene_context.device, dtype=scene_context.dtype).unsqueeze(-1)
        )
        prior_logits = (
            base_logits
            + self.prototype_support_bias_weight * support_logits
            + self.prototype_count_compat_weight * count_compat
            + self.prototype_difficulty_compat_weight * difficulty_compat
            + self.prototype_plan_compat_weight * plan_compat_logits
        )
        num_prototypes = int(prior_logits.shape[-1])
        if self.prototype_mixture_enabled:
            topk = max(1, min(int(self.prototype_mixture_topk), num_prototypes))
            base_topk_scores, base_topk_ids = torch.topk(prior_logits, k=topk, dim=-1)
            topk_scores = base_topk_scores
            topk_ids = base_topk_ids
            if forced_prototype_ids is not None:
                forced_ids = forced_prototype_ids.to(device=scene_context.device, dtype=torch.long)
                forced_scores = prior_logits.gather(1, forced_ids.unsqueeze(-1)).squeeze(-1)
                missing_mask = ~base_topk_ids.eq(forced_ids.unsqueeze(-1)).any(dim=-1)
                if bool(missing_mask.any()):
                    row_mask = missing_mask.unsqueeze(-1)
                    forced_ids_expanded = forced_ids.unsqueeze(-1)
                    forced_scores_expanded = forced_scores.unsqueeze(-1)
                    replaced_topk_ids = torch.cat([base_topk_ids[:, :-1], forced_ids_expanded], dim=-1)
                    replaced_topk_scores = torch.cat([base_topk_scores[:, :-1], forced_scores_expanded], dim=-1)
                    topk_ids = torch.where(row_mask, replaced_topk_ids, base_topk_ids)
                    topk_scores = torch.where(row_mask, replaced_topk_scores, base_topk_scores)
                teacher_mask = topk_ids.eq(forced_ids.unsqueeze(-1))
                topk_scores = topk_scores + teacher_mask.float() * self.prototype_mixture_target_logit_boost
            mixture_weights = F.softmax(
                topk_scores / max(self.prototype_mixture_temperature, 1e-3),
                dim=-1,
            )
            flat_ids = topk_ids.reshape(-1)
            topk_plan = self.prototype_plan_features.index_select(0, flat_ids).view(
                scene_context.shape[0], topk, self.plan_feature_dim
            )
            topk_states = self.prototype_states.index_select(0, flat_ids).view(
                scene_context.shape[0], topk, self.max_agents, self.state_dim
            )
            topk_masks = self.prototype_masks.index_select(0, flat_ids).view(
                scene_context.shape[0], topk, self.max_agents
            )
            topk_counts = self.prototype_counts.index_select(0, flat_ids).view(scene_context.shape[0], topk)
            topk_difficulties = self.prototype_difficulty_mean.index_select(0, flat_ids).view(scene_context.shape[0], topk)
            selected_ids = topk_ids.gather(1, mixture_weights.argmax(dim=-1, keepdim=True)).squeeze(-1)
            selected_plan = (topk_plan * mixture_weights.unsqueeze(-1)).sum(dim=1)
            selected_states = (topk_states * mixture_weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
            presence_scores = (topk_masks.float() * mixture_weights.unsqueeze(-1)).sum(dim=1)
            selected_count = (topk_counts * mixture_weights).sum(dim=1)
            selected_count_long = selected_count.round().clamp(min=0.0, max=float(self.max_agents)).to(dtype=torch.long)
            sorted_idx = torch.argsort(presence_scores, dim=-1, descending=True)
            rank = torch.arange(self.max_agents, device=scene_context.device, dtype=torch.long).unsqueeze(0).expand_as(sorted_idx)
            keep_sorted = rank < selected_count_long.unsqueeze(-1)
            selected_masks = torch.zeros_like(presence_scores, dtype=torch.bool)
            selected_masks.scatter_(1, sorted_idx, keep_sorted)
            selected_masks = selected_masks & (presence_scores > 1e-4)
            selected_difficulty_mean = (topk_difficulties * mixture_weights).sum(dim=1)
            return (
                prior_logits,
                selected_ids,
                selected_plan,
                selected_states,
                selected_masks,
                selected_count,
                selected_difficulty_mean,
            )
        if forced_prototype_ids is None:
            topk = max(1, min(int(self.prototype_rerank_topk), int(prior_logits.shape[-1])))
            topk_scores, topk_ids = torch.topk(prior_logits, k=topk, dim=-1)
            if self.prototype_sampling_temperature > 0.0:
                probs = F.softmax(topk_scores / max(self.prototype_sampling_temperature, 1e-3), dim=-1)
                selected_pos = torch.argmax(probs, dim=-1)
            else:
                selected_pos = torch.argmax(topk_scores, dim=-1)
            selected_ids = topk_ids.gather(1, selected_pos.unsqueeze(-1)).squeeze(-1)
        else:
            selected_ids = forced_prototype_ids.to(device=scene_context.device, dtype=torch.long)
        selected_plan = self.prototype_plan_features.index_select(0, selected_ids)
        selected_states = self.prototype_states.index_select(0, selected_ids)
        selected_masks = self.prototype_masks.index_select(0, selected_ids)
        selected_count = selected_masks.float().sum(dim=-1)
        selected_difficulty_mean = self.prototype_difficulty_mean.index_select(0, selected_ids)
        return prior_logits, selected_ids, selected_plan, selected_states, selected_masks, selected_count, selected_difficulty_mean

    def _scene_context(
        self,
        map_polylines: torch.Tensor,
        map_point_mask: torch.Tensor,
        map_polyline_mask: torch.Tensor,
        target_difficulty: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        _, map_context = self.map_encoder(map_polylines, map_point_mask, map_polyline_mask)
        difficulty_context = self.difficulty_embedding(target_difficulty.float(), training=self.training)
        noise_context = self.noise_proj(noise)
        return self.scene_proj(torch.cat([map_context, difficulty_context, noise_context], dim=-1))

    def _decode_state(
        self,
        hidden: torch.Tensor,
        prototype_state: torch.Tensor | None = None,
        prototype_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raw_state = self.state_head(hidden)
        absolute_state = torch.cat(
            [
                torch.tanh(raw_state[..., 0:2]) * self.map_radius_m,
                torch.tanh(raw_state[..., 2:4]) * self.max_speed_mps,
                _wrap_heading(raw_state[..., 4:5]),
            ],
            dim=-1,
        )
        if prototype_state is None or prototype_mask is None:
            return absolute_state
        prototype_state = prototype_state.to(device=hidden.device, dtype=hidden.dtype)
        prototype_mask = prototype_mask.to(device=hidden.device, dtype=torch.bool)
        refined_state = torch.cat(
            [
                torch.clamp(
                    prototype_state[..., 0:2] + torch.tanh(raw_state[..., 0:2]) * self.prototype_residual_position_scale_m,
                    min=-self.map_radius_m,
                    max=self.map_radius_m,
                ),
                torch.clamp(
                    prototype_state[..., 2:4] + torch.tanh(raw_state[..., 2:4]) * self.prototype_residual_speed_scale_mps,
                    min=-self.max_speed_mps,
                    max=self.max_speed_mps,
                ),
                _wrap_heading(
                    prototype_state[..., 4:5]
                    + torch.tanh(raw_state[..., 4:5]) * self.prototype_residual_heading_scale_rad
                ),
            ],
            dim=-1,
        )
        while prototype_mask.ndim < refined_state.ndim:
            prototype_mask = prototype_mask.unsqueeze(-1)
        return torch.where(prototype_mask, refined_state, absolute_state)

    def _global_context(
        self,
        scene_context: torch.Tensor,
        target_difficulty: torch.Tensor,
        override_scene_plan: torch.Tensor | None = None,
        override_count: torch.Tensor | None = None,
        forced_prototype_ids: torch.Tensor | None = None,
        location_ids: Sequence[str] | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        tuple[torch.Tensor | None, torch.Tensor | None],
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        (
            prior_logits,
            selected_prototype_ids,
            selected_prototype_plan,
            selected_prototype_states,
            selected_prototype_masks,
            prototype_count,
            prototype_difficulty_mean,
        ) = self._select_support_prototype(
            scene_context,
            target_difficulty=target_difficulty,
            forced_prototype_ids=forced_prototype_ids,
            location_ids=location_ids,
        )
        prototype_context = (
            self.prototype_context_proj(
                torch.cat(
                    [
                        selected_prototype_plan,
                        (prototype_count / max(float(self.max_agents), 1.0)).unsqueeze(-1),
                    ],
                    dim=-1,
                )
            )
            if selected_prototype_plan is not None and prototype_count is not None
            else scene_context * 0.0
        )
        predicted_plan = torch.sigmoid(self.plan_head(scene_context + prototype_context))
        conditioned_plan = predicted_plan if override_scene_plan is None else override_scene_plan.to(
            device=scene_context.device,
            dtype=scene_context.dtype,
        )
        plan_context = self.plan_embedding(conditioned_plan)
        count_logits = self.count_head(scene_context + prototype_context)
        count_probs = F.softmax(count_logits, dim=-1)
        if prototype_count is not None:
            residual_support = torch.arange(
                -self.prototype_count_residual_max_delta,
                self.prototype_count_residual_max_delta + 1,
                device=scene_context.device,
                dtype=scene_context.dtype,
            )
            expected_residual = (count_probs * residual_support.unsqueeze(0)).sum(dim=-1)
            predicted_residual = residual_support[count_probs.argmax(dim=-1)]
            expected_count = torch.clamp(prototype_count.to(dtype=scene_context.dtype) + expected_residual, min=0.0, max=float(self.max_agents))
            predicted_count = torch.clamp(prototype_count.to(dtype=scene_context.dtype) + predicted_residual, min=0.0, max=float(self.max_agents))
        else:
            count_support = torch.arange(self.max_agents + 1, device=scene_context.device, dtype=scene_context.dtype)
            expected_count = (count_probs * count_support.unsqueeze(0)).sum(dim=-1)
            predicted_count = count_probs.argmax(dim=-1).to(dtype=scene_context.dtype)
        if override_count is None:
            conditioned_count = predicted_count
        else:
            conditioned_count = override_count.to(device=scene_context.device, dtype=scene_context.dtype)
        count_context = self.count_context_proj(
            (conditioned_count / max(float(self.max_agents), 1.0)).unsqueeze(-1)
        )
        global_context = self.context_proj(torch.cat([scene_context, plan_context, count_context, prototype_context], dim=-1))
        return (
            predicted_plan,
            count_logits,
            count_probs,
            expected_count,
            predicted_count,
            conditioned_count,
            global_context,
            prior_logits,
            selected_prototype_ids,
            (selected_prototype_states, selected_prototype_masks),
            prototype_count,
            prototype_difficulty_mean,
        )

    def _occupancy_bin_index(self, position_xy: torch.Tensor) -> torch.Tensor:
        radius = torch.linalg.norm(position_xy, dim=-1)
        radius_norm = torch.clamp(radius / max(self.map_radius_m, 1e-6), min=0.0, max=0.999999)
        angle = torch.atan2(position_xy[..., 1], position_xy[..., 0])
        angle_norm = torch.remainder(angle + math.pi, 2.0 * math.pi) / (2.0 * math.pi)
        radial_bin = torch.clamp((radius_norm * float(self.plan_radial_bins)).long(), min=0, max=self.plan_radial_bins - 1)
        angular_bin = torch.clamp((angle_norm * float(self.plan_angular_bins)).long(), min=0, max=self.plan_angular_bins - 1)
        return radial_bin * self.plan_angular_bins + angular_bin

    def _flatten_map_candidates(
        self,
        map_polylines: torch.Tensor,
        map_point_mask: torch.Tensor,
        map_polyline_mask: torch.Tensor,
        batch_idx: int,
    ) -> torch.Tensor:
        valid = map_point_mask[batch_idx] & map_polyline_mask[batch_idx].unsqueeze(-1)
        flat = map_polylines[batch_idx][valid]
        if int(flat.shape[0]) <= 0:
            return flat.new_zeros((0, map_polylines.shape[-1]))
        stride = max(self.compiler_anchor_stride, 1)
        flat = flat[::stride]
        if int(flat.shape[0]) > self.compiler_max_map_candidates:
            sample_idx = torch.linspace(
                0,
                int(flat.shape[0]) - 1,
                steps=self.compiler_max_map_candidates,
                device=flat.device,
            ).round().long()
            flat = flat.index_select(0, sample_idx)
        return flat

    def _batched_structured_map_candidates(
        self,
        map_polylines: torch.Tensor,
        map_point_mask: torch.Tensor,
        map_polyline_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_candidates: list[torch.Tensor] = []
        max_count = 0
        for batch_idx in range(int(map_polylines.shape[0])):
            valid = map_point_mask[batch_idx] & map_polyline_mask[batch_idx].unsqueeze(-1)
            flat = map_polylines[batch_idx][valid]
            if int(flat.shape[0]) > 0:
                flat = flat[:: self.structured_anchor_stride]
                if int(flat.shape[0]) > self.structured_max_map_candidates:
                    sample_idx = torch.linspace(
                        0,
                        int(flat.shape[0]) - 1,
                        steps=self.structured_max_map_candidates,
                        device=flat.device,
                    ).round().long()
                    flat = flat.index_select(0, sample_idx)
            batch_candidates.append(flat)
            max_count = max(max_count, int(flat.shape[0]))
        max_count = max(max_count, 1)
        padded_candidates: list[torch.Tensor] = []
        candidate_masks: list[torch.Tensor] = []
        for flat in batch_candidates:
            valid_count = int(flat.shape[0])
            padded = flat.new_zeros((max_count, map_polylines.shape[-1]))
            mask = torch.zeros(max_count, device=flat.device, dtype=torch.bool)
            if valid_count > 0:
                padded[:valid_count] = flat
                mask[:valid_count] = True
            padded_candidates.append(padded)
            candidate_masks.append(mask)
        stacked_candidates = torch.stack(padded_candidates, dim=0)
        stacked_masks = torch.stack(candidate_masks, dim=0)
        return stacked_candidates, stacked_masks

    def _apply_structured_pair_relaxation(
        self,
        adapted_state: torch.Tensor,
        generated_mask: torch.Tensor,
        tangent: torch.Tensor,
    ) -> torch.Tensor:
        if (
            not self.structured_pair_relaxation_enabled
            or self.structured_pair_relaxation_iterations <= 0
            or self.structured_pair_relaxation_min_spacing_m <= 0.0
            or self.structured_pair_relaxation_strength <= 0.0
        ):
            return adapted_state
        valid = generated_mask.to(dtype=torch.bool)
        if int(valid.sum().item()) <= 1:
            return adapted_state

        pos = adapted_state[..., 0:2]
        min_spacing = float(self.structured_pair_relaxation_min_spacing_m)
        strength = float(self.structured_pair_relaxation_strength)
        max_step = max(float(self.structured_pair_relaxation_max_step_m), 1e-6)
        agent_ids = torch.arange(self.max_agents, device=pos.device, dtype=pos.dtype)
        fallback_sign = torch.sign(agent_ids.view(1, -1, 1) - agent_ids.view(1, 1, -1))
        fallback_sign = torch.where(fallback_sign == 0, torch.ones_like(fallback_sign), fallback_sign)
        fallback_sign = fallback_sign.unsqueeze(-1)
        eye = torch.eye(self.max_agents, device=pos.device, dtype=torch.bool).unsqueeze(0)

        for _ in range(self.structured_pair_relaxation_iterations):
            delta = pos.unsqueeze(2) - pos.unsqueeze(1)
            distance = torch.linalg.norm(delta, dim=-1).clamp_min(1e-4)
            pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1) & ~eye
            gap = F.relu(min_spacing - distance)
            gap = torch.where(pair_valid, gap, torch.zeros_like(gap))
            tangent_i = tangent.unsqueeze(2)
            signed_along = (delta * tangent_i).sum(dim=-1)
            sign = torch.sign(signed_along).unsqueeze(-1)
            sign = torch.where(sign == 0, fallback_sign, sign)
            pair_push = tangent_i * sign * gap.unsqueeze(-1) * (0.5 * strength)
            denom = pair_valid.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
            step = pair_push.sum(dim=2) / denom
            step_norm = torch.linalg.norm(step, dim=-1, keepdim=True).clamp_min(1e-6)
            step = step * torch.clamp(max_step / step_norm, max=1.0)
            pos = torch.clamp(
                pos + torch.where(valid.unsqueeze(-1), step, torch.zeros_like(step)),
                min=-self.map_radius_m,
                max=self.map_radius_m,
            )
        return torch.cat([pos, adapted_state[..., 2:]], dim=-1)

    def _select_structured_slots(
        self,
        keep_scores: torch.Tensor,
        valid_proto: torch.Tensor,
        adapted_pos: torch.Tensor,
        target_count: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(keep_scores.shape[0])
        device = keep_scores.device
        order = torch.argsort(keep_scores, dim=-1, descending=True)
        if (
            not self.structured_spacing_selection_enabled
            or self.structured_spacing_selection_min_spacing_m <= 0.0
        ):
            rank = torch.arange(self.max_agents, device=device, dtype=torch.long).unsqueeze(0).expand_as(order)
            keep_sorted = rank < target_count.unsqueeze(-1)
            generated_mask = torch.zeros(batch_size, self.max_agents, device=device, dtype=torch.bool)
            generated_mask.scatter_(1, order, keep_sorted)
            return generated_mask & valid_proto

        sorted_valid = torch.gather(valid_proto, dim=1, index=order)
        gather_pos = order.unsqueeze(-1).expand(-1, -1, 2)
        sorted_pos = torch.gather(adapted_pos, dim=1, index=gather_pos)
        selected_sorted = torch.zeros(batch_size, self.max_agents, device=device, dtype=torch.bool)
        selected_count = torch.zeros(batch_size, device=device, dtype=torch.long)
        target_count = target_count.to(device=device, dtype=torch.long).clamp(min=0, max=self.max_agents)

        def spacing_pass(selected: torch.Tensor, count: torch.Tensor, threshold_m: float) -> tuple[torch.Tensor, torch.Tensor]:
            if threshold_m <= 0.0:
                return selected, count
            inf = torch.full(
                (batch_size, self.max_agents),
                1e6,
                device=device,
                dtype=sorted_pos.dtype,
            )
            for rank_idx in range(self.max_agents):
                candidate_valid = sorted_valid[:, rank_idx] & ~selected[:, rank_idx] & (count < target_count)
                distances = torch.linalg.norm(
                    sorted_pos[:, rank_idx : rank_idx + 1, :] - sorted_pos,
                    dim=-1,
                )
                nearest_selected = torch.where(selected, distances, inf).amin(dim=-1)
                spacing_ok = (count == 0) | (nearest_selected >= float(threshold_m))
                accept = candidate_valid & spacing_ok
                selected[:, rank_idx] = selected[:, rank_idx] | accept
                count = count + accept.to(dtype=torch.long)
            return selected, count

        primary_spacing = max(float(self.structured_spacing_selection_min_spacing_m), 0.0)
        fallback_spacing = max(float(self.structured_spacing_selection_fallback_spacing_m), 0.0)
        selected_sorted, selected_count = spacing_pass(selected_sorted, selected_count, primary_spacing)
        selected_sorted, selected_count = spacing_pass(selected_sorted, selected_count, fallback_spacing)
        if self.structured_spacing_selection_fill_to_count:
            final_fill_spacing = max(float(self.structured_spacing_selection_final_fill_spacing_m), 0.0)
            if final_fill_spacing > 0.0:
                selected_sorted, selected_count = spacing_pass(
                    selected_sorted,
                    selected_count,
                    final_fill_spacing,
                )
            else:
                for rank_idx in range(self.max_agents):
                    accept = sorted_valid[:, rank_idx] & ~selected_sorted[:, rank_idx] & (selected_count < target_count)
                    selected_sorted[:, rank_idx] = selected_sorted[:, rank_idx] | accept
                    selected_count = selected_count + accept.to(dtype=torch.long)

        generated_mask = torch.zeros(batch_size, self.max_agents, device=device, dtype=torch.bool)
        generated_mask.scatter_(1, order, selected_sorted)
        return generated_mask & valid_proto

    def _decode_structured_adaptation(
        self,
        global_context: torch.Tensor,
        prototype_states: torch.Tensor,
        prototype_mask: torch.Tensor,
        conditioned_count: torch.Tensor,
        map_polylines: torch.Tensor,
        map_point_mask: torch.Tensor,
        map_polyline_mask: torch.Tensor,
        teacher_forcing_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = int(global_context.shape[0])
        device = global_context.device
        dtype = global_context.dtype
        map_candidates, map_candidate_mask = self._batched_structured_map_candidates(
            map_polylines,
            map_point_mask,
            map_polyline_mask,
        )
        map_candidates = map_candidates.to(device=device, dtype=dtype)
        map_candidate_mask = map_candidate_mask.to(device=device)
        proto_token = torch.cat([prototype_states, prototype_mask.unsqueeze(-1).float()], dim=-1)
        proto_hidden = self.prototype_state_proj(proto_token)
        step_ids = torch.arange(self.max_agents, device=device, dtype=torch.long)
        proto_hidden = proto_hidden + self.step_embedding(step_ids).unsqueeze(0) + global_context.unsqueeze(1)
        proto_hidden = self.autoregressive_decoder(proto_hidden)

        keep_logits = self.prototype_keep_head(proto_hidden).squeeze(-1)
        longitudinal_shift = (
            torch.tanh(self.prototype_shift_head(proto_hidden).squeeze(-1)) * self.structured_shift_scale_m
        )
        speed_residual = (
            torch.tanh(self.prototype_speed_head(proto_hidden).squeeze(-1))
            * self.structured_agent_speed_residual_scale_mps
        )
        scene_speed_scale = 1.0 + (
            torch.tanh(self.prototype_scene_speed_scale_head(global_context).squeeze(-1))
            * self.structured_scene_speed_scale_max_delta
        )

        proto_positions = prototype_states[..., 0:2]
        proto_heading = prototype_states[..., 4]
        proto_speed = torch.linalg.norm(prototype_states[..., 2:4], dim=-1)
        anchor_xy = map_candidates[..., 0:2]
        anchor_heading = map_candidates[..., 2]
        pair_distance = torch.cdist(proto_positions, anchor_xy)
        pair_distance = torch.where(
            map_candidate_mask.unsqueeze(1),
            pair_distance,
            torch.full_like(pair_distance, 1e6),
        )
        nearest_idx = pair_distance.argmin(dim=-1)
        gather_xy = nearest_idx.unsqueeze(-1).expand(-1, -1, 2)
        snapped_pos = torch.gather(anchor_xy, dim=1, index=gather_xy)
        snapped_heading = torch.gather(anchor_heading, dim=1, index=nearest_idx)
        has_candidates = map_candidate_mask.any(dim=-1)
        snapped_pos = torch.where(has_candidates[:, None, None], snapped_pos, proto_positions)
        snapped_heading = torch.where(has_candidates[:, None], snapped_heading, proto_heading)

        tangent = torch.stack([torch.cos(snapped_heading), torch.sin(snapped_heading)], dim=-1)
        adapted_pos = torch.clamp(
            snapped_pos + tangent * longitudinal_shift.unsqueeze(-1),
            min=-self.map_radius_m,
            max=self.map_radius_m,
        )
        adapted_speed = torch.clamp(
            proto_speed * scene_speed_scale.unsqueeze(-1) + speed_residual,
            min=0.0,
            max=self.max_speed_mps,
        )
        adapted_velocity = tangent * adapted_speed.unsqueeze(-1)
        adapted_state = torch.cat(
            [adapted_pos, adapted_velocity, _wrap_heading(snapped_heading).unsqueeze(-1)],
            dim=-1,
        )

        valid_proto = prototype_mask.to(dtype=torch.bool)
        if teacher_forcing_mask is not None:
            target_count = teacher_forcing_mask.sum(dim=-1).to(device=device, dtype=torch.long)
        else:
            target_count = conditioned_count.round().clamp(min=0.0, max=float(self.max_agents)).to(dtype=torch.long)
        prototype_count = valid_proto.sum(dim=-1)
        target_count = torch.minimum(target_count, prototype_count)
        keep_scores = torch.where(valid_proto, keep_logits, torch.full_like(keep_logits, -1e6))
        generated_mask = self._select_structured_slots(
            keep_scores=keep_scores,
            valid_proto=valid_proto,
            adapted_pos=adapted_state[..., 0:2],
            target_count=target_count,
        )
        adapted_state = self._apply_structured_pair_relaxation(
            adapted_state=adapted_state,
            generated_mask=generated_mask,
            tangent=tangent,
        )
        generated_states = torch.where(
            generated_mask.unsqueeze(-1),
            adapted_state,
            torch.zeros_like(adapted_state),
        )
        presence_logits = torch.where(
            valid_proto,
            keep_logits,
            torch.full_like(keep_logits, -6.0),
        )
        return generated_states, presence_logits, generated_mask

    def _build_compiler_candidate_states(
        self,
        proto_states: torch.Tensor,
        map_candidates: torch.Tensor,
        target_speed_mps: float,
        target_radius_norm: float,
        occupancy_need: torch.Tensor,
    ) -> list[dict[str, torch.Tensor | float | int | bool]]:
        candidates: list[dict[str, torch.Tensor | float | int | bool]] = []
        if int(proto_states.shape[0]) <= 0:
            return candidates
        proto_positions = proto_states[:, 0:2]
        proto_speed = torch.linalg.norm(proto_states[:, 2:4], dim=-1)
        if int(map_candidates.shape[0]) > 0:
            anchor_pos = map_candidates[:, 0:2]
            anchor_heading = map_candidates[:, 2]
            distance = torch.cdist(proto_positions, anchor_pos)
            nearest_idx = distance.argmin(dim=-1)
            snapped_pos = anchor_pos.index_select(0, nearest_idx)
            snapped_heading = anchor_heading.index_select(0, nearest_idx)
        else:
            snapped_pos = proto_positions
            snapped_heading = proto_states[:, 4]
        snapped_bin = self._occupancy_bin_index(snapped_pos)
        radius_norm = torch.linalg.norm(snapped_pos, dim=-1) / max(self.map_radius_m, 1e-6)
        speed_norm = proto_speed / max(self.max_speed_mps, 1e-6)
        target_speed_norm = target_speed_mps / max(self.max_speed_mps, 1e-6)
        for idx in range(int(proto_states.shape[0])):
            bin_idx = int(snapped_bin[idx].item())
            occ_score = float(occupancy_need[bin_idx].item())
            radius_gap = abs(float(radius_norm[idx].item()) - float(target_radius_norm))
            speed_gap = abs(float(speed_norm[idx].item()) - float(target_speed_norm))
            candidates.append(
                {
                    "position": snapped_pos[idx],
                    "heading": snapped_heading[idx],
                    "speed": proto_speed[idx],
                    "bin_idx": bin_idx,
                    "score": self.compiler_proto_score_bias
                    + self.compiler_occupancy_weight * occ_score
                    - self.compiler_radius_weight * radius_gap
                    - self.compiler_speed_weight * speed_gap,
                    "is_proto": True,
                }
            )
        return candidates

    def _fill_map_only_candidates(
        self,
        candidates: list[dict[str, torch.Tensor | float | int | bool]],
        map_candidates: torch.Tensor,
        target_speed_mps: float,
        target_radius_norm: float,
        occupancy_need: torch.Tensor,
    ) -> list[dict[str, torch.Tensor | float | int | bool]]:
        if int(map_candidates.shape[0]) <= 0:
            return candidates
        map_bin = self._occupancy_bin_index(map_candidates[:, 0:2])
        map_radius_norm = torch.linalg.norm(map_candidates[:, 0:2], dim=-1) / max(self.map_radius_m, 1e-6)
        for idx in range(int(map_candidates.shape[0])):
            bin_idx = int(map_bin[idx].item())
            occ_score = float(occupancy_need[bin_idx].item())
            radius_gap = abs(float(map_radius_norm[idx].item()) - float(target_radius_norm))
            candidates.append(
                {
                    "position": map_candidates[idx, 0:2],
                    "heading": map_candidates[idx, 2],
                    "speed": map_candidates.new_tensor(float(target_speed_mps)),
                    "bin_idx": bin_idx,
                    "score": self.compiler_occupancy_weight * occ_score - self.compiler_radius_weight * radius_gap,
                    "is_proto": False,
                }
            )
        return candidates

    def _compile_scene_from_prototype(
        self,
        prototype_states: torch.Tensor,
        prototype_mask: torch.Tensor,
        conditioned_count: torch.Tensor,
        conditioned_plan: torch.Tensor,
        map_polylines: torch.Tensor,
        map_point_mask: torch.Tensor,
        map_polyline_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = int(conditioned_plan.shape[0])
        device = conditioned_plan.device
        dtype = conditioned_plan.dtype
        generated_states = torch.zeros(batch_size, self.max_agents, self.state_dim, device=device, dtype=dtype)
        presence_logits = torch.full((batch_size, self.max_agents), -6.0, device=device, dtype=dtype)
        generated_mask = torch.zeros(batch_size, self.max_agents, device=device, dtype=torch.bool)
        summary = conditioned_plan[:, : self.summary_feature_dim]
        occupancy = conditioned_plan[:, self.summary_feature_dim :]
        for batch_idx in range(batch_size):
            target_count = int(
                conditioned_count[batch_idx]
                .round()
                .clamp(min=0.0, max=float(self.max_agents))
                .item()
            )
            if target_count <= 0:
                continue
            target_speed_mps = max(
                float(summary[batch_idx, 0].item()) * self.max_speed_mps,
                self.compiler_target_speed_floor,
            )
            target_radius_norm = float(summary[batch_idx, 1].item())
            target_min_pair_m = float(summary[batch_idx, 3].item()) * self.map_radius_m
            spacing_threshold = max(
                self.compiler_min_spacing_m,
                min(6.0, 0.2 * max(target_min_pair_m, 0.0)),
            )
            occupancy_need = occupancy[batch_idx].clone() * float(target_count)
            map_candidates = self._flatten_map_candidates(
                map_polylines,
                map_point_mask,
                map_polyline_mask,
                batch_idx,
            ).to(device=device, dtype=dtype)
            valid_proto_states = prototype_states[batch_idx][prototype_mask[batch_idx]]
            candidates = self._build_compiler_candidate_states(
                valid_proto_states,
                map_candidates,
                target_speed_mps=target_speed_mps,
                target_radius_norm=target_radius_norm,
                occupancy_need=occupancy_need,
            )
            candidates = self._fill_map_only_candidates(
                candidates,
                map_candidates,
                target_speed_mps=target_speed_mps,
                target_radius_norm=target_radius_norm,
                occupancy_need=occupancy_need,
            )
            if not candidates:
                continue
            candidates.sort(key=lambda item: float(item["score"]), reverse=True)
            selected: list[dict[str, torch.Tensor | float | int | bool]] = []
            for candidate in candidates:
                position = candidate["position"]
                if selected:
                    selected_pos = torch.stack([item["position"] for item in selected], dim=0)
                    min_distance = torch.linalg.norm(selected_pos - position.unsqueeze(0), dim=-1).amin()
                    if float(min_distance.item()) < spacing_threshold:
                        continue
                selected.append(candidate)
                bin_idx = int(candidate["bin_idx"])
                occupancy_need[bin_idx] = occupancy_need[bin_idx] - 1.0
                if len(selected) >= target_count:
                    break
            if not selected:
                continue
            selected_pos = torch.stack([item["position"] for item in selected], dim=0)
            selected_heading = torch.stack([item["heading"] for item in selected], dim=0)
            selected_speed = torch.stack([item["speed"] for item in selected], dim=0).to(dtype=dtype)
            current_mean_speed = float(selected_speed.mean().item()) if int(selected_speed.numel()) > 0 else 0.0
            if current_mean_speed > 1e-3:
                speed_scale = max(0.5, min(1.5, target_speed_mps / current_mean_speed))
                selected_speed = selected_speed * speed_scale
            heading_vec = torch.stack([torch.cos(selected_heading), torch.sin(selected_heading)], dim=-1)
            velocity = heading_vec * selected_speed.unsqueeze(-1)
            num_selected = min(len(selected), self.max_agents)
            selected_state = torch.cat(
                [selected_pos[:num_selected], velocity[:num_selected], _wrap_heading(selected_heading[:num_selected]).unsqueeze(-1)],
                dim=-1,
            )
            generated_states[batch_idx, :num_selected] = selected_state
            generated_mask[batch_idx, :num_selected] = True
            presence_logits[batch_idx, :num_selected] = 6.0
        return generated_states, presence_logits, generated_mask

    def _build_training_tokens(
        self,
        sorted_states: torch.Tensor,
        sorted_mask: torch.Tensor,
        global_context: torch.Tensor,
        prototype_states: torch.Tensor | None = None,
        prototype_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prev_states = torch.zeros_like(sorted_states)
        prev_states[:, 1:] = sorted_states[:, :-1]
        prev_active = torch.zeros_like(sorted_mask, dtype=sorted_states.dtype)
        prev_active[:, 1:] = sorted_mask[:, :-1].float()
        token_features = torch.cat([prev_states, prev_active.unsqueeze(-1)], dim=-1)
        token_emb = self.prev_state_proj(token_features)
        if prototype_states is not None and prototype_mask is not None:
            prototype_token = torch.cat([prototype_states, prototype_mask.unsqueeze(-1).float()], dim=-1)
            token_emb = token_emb + self.prototype_state_proj(prototype_token)
        step_ids = torch.arange(self.max_agents, device=sorted_states.device, dtype=torch.long)
        token_emb = token_emb + self.step_embedding(step_ids).unsqueeze(0) + global_context.unsqueeze(1)
        token_emb[:, 0] = token_emb[:, 0] + self.bos_token.squeeze(0)
        return token_emb

    def _build_prototype_memory(
        self,
        prototype_states: torch.Tensor | None,
        prototype_mask: torch.Tensor | None,
        global_context: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if (
            not self.prototype_enabled
            or not self.prototype_memory_enabled
            or prototype_states is None
            or prototype_mask is None
        ):
            return None, None
        prototype_token = torch.cat([prototype_states, prototype_mask.unsqueeze(-1).float()], dim=-1)
        memory = self.prototype_memory_proj(prototype_token)
        step_ids = torch.arange(self.max_agents, device=prototype_states.device, dtype=torch.long)
        memory = memory + self.step_embedding(step_ids).unsqueeze(0) + global_context.unsqueeze(1)
        return memory, ~prototype_mask.to(dtype=torch.bool)

    def _apply_prototype_memory(
        self,
        hidden: torch.Tensor,
        prototype_memory: torch.Tensor | None,
        prototype_key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if prototype_memory is None or prototype_key_padding_mask is None:
            return hidden
        attn_output, _ = self.prototype_memory_attention(
            query=hidden,
            key=prototype_memory,
            value=prototype_memory,
            key_padding_mask=prototype_key_padding_mask,
            need_weights=False,
        )
        hidden = self.prototype_memory_norm(hidden + attn_output)
        ffn_output = self.prototype_memory_ffn(hidden)
        hidden = self.prototype_memory_ffn_norm(hidden + ffn_output)
        return hidden

    def _decode_teacher_forced(
        self,
        teacher_forcing_states: torch.Tensor,
        teacher_forcing_mask: torch.Tensor,
        global_context: torch.Tensor,
        prototype_states: torch.Tensor | None = None,
        prototype_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sorted_states, sorted_mask, _ = canonical_sort_scene(teacher_forcing_states, teacher_forcing_mask)
        token_emb = self._build_training_tokens(sorted_states, sorted_mask, global_context, prototype_states=prototype_states, prototype_mask=prototype_mask)
        hidden = self.autoregressive_decoder(token_emb, mask=_causal_mask(self.max_agents, token_emb.device))
        prototype_memory, prototype_key_padding_mask = self._build_prototype_memory(
            prototype_states,
            prototype_mask,
            global_context,
        )
        hidden = self._apply_prototype_memory(hidden, prototype_memory, prototype_key_padding_mask)
        states = self._decode_state(hidden, prototype_state=prototype_states, prototype_mask=prototype_mask)
        presence_logits = self.presence_head(hidden).squeeze(-1)
        return states, presence_logits

    def _decode_free_running(
        self,
        global_context: torch.Tensor,
        predicted_count: torch.Tensor,
        prototype_states: torch.Tensor | None = None,
        prototype_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(global_context.shape[0])
        device = global_context.device
        dtype = global_context.dtype
        generated_states = torch.zeros(batch_size, self.max_agents, self.state_dim, device=device, dtype=dtype)
        presence_logits = torch.zeros(batch_size, self.max_agents, device=device, dtype=dtype)
        active_mask = torch.zeros(batch_size, self.max_agents, device=device, dtype=torch.bool)
        count_long = predicted_count.round().clamp(min=0.0, max=float(self.max_agents)).to(dtype=torch.long)
        step_ids_all = torch.arange(self.max_agents, device=device, dtype=torch.long)
        prototype_memory, prototype_key_padding_mask = self._build_prototype_memory(
            prototype_states,
            prototype_mask,
            global_context,
        )
        for step in range(self.max_agents):
            seq_len = step + 1
            prev_states = torch.zeros(batch_size, seq_len, self.state_dim, device=device, dtype=dtype)
            prev_active = torch.zeros(batch_size, seq_len, 1, device=device, dtype=dtype)
            if step > 0:
                prev_states[:, 1:] = generated_states[:, :step]
                prev_active[:, 1:, 0] = active_mask[:, :step].float()
            token_features = torch.cat([prev_states, prev_active], dim=-1)
            token_emb = self.prev_state_proj(token_features)
            if prototype_states is not None and prototype_mask is not None:
                prototype_token = torch.cat(
                    [prototype_states[:, :seq_len], prototype_mask[:, :seq_len].unsqueeze(-1).float()],
                    dim=-1,
                )
                token_emb = token_emb + self.prototype_state_proj(prototype_token)
            token_emb = token_emb + self.step_embedding(step_ids_all[:seq_len]).unsqueeze(0) + global_context.unsqueeze(1)
            token_emb[:, 0] = token_emb[:, 0] + self.bos_token.squeeze(0)
            hidden = self.autoregressive_decoder(token_emb, mask=_causal_mask(seq_len, device))
            hidden = self._apply_prototype_memory(hidden, prototype_memory, prototype_key_padding_mask)
            current_hidden = hidden[:, -1]
            current_state = self._decode_state(
                current_hidden,
                prototype_state=prototype_states[:, step] if prototype_states is not None else None,
                prototype_mask=prototype_mask[:, step] if prototype_mask is not None else None,
            )
            current_presence = self.presence_head(current_hidden).squeeze(-1)
            is_active = step < count_long
            generated_states[:, step] = current_state
            presence_logits[:, step] = current_presence
            active_mask[:, step] = is_active
            generated_states[:, step] = torch.where(is_active.unsqueeze(-1), generated_states[:, step], torch.zeros_like(generated_states[:, step]))
        return generated_states, presence_logits

    def forward(
        self,
        map_polylines: torch.Tensor,
        map_point_mask: torch.Tensor,
        map_polyline_mask: torch.Tensor,
        target_difficulty: torch.Tensor,
        noise: torch.Tensor | None = None,
        teacher_forcing_states: torch.Tensor | None = None,
        teacher_forcing_mask: torch.Tensor | None = None,
        override_scene_plan: torch.Tensor | None = None,
        override_count: torch.Tensor | None = None,
        forced_prototype_ids: torch.Tensor | None = None,
        location_ids: Sequence[str] | None = None,
        use_hybrid_compiler: bool | None = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = int(map_polylines.shape[0])
        device = map_polylines.device
        dtype = map_polylines.dtype
        if noise is None:
            noise = torch.randn(batch_size, self.noise_dim, device=device, dtype=dtype)
        else:
            noise = noise.to(device=device, dtype=dtype)
        scene_context = self._scene_context(
            map_polylines=map_polylines,
            map_point_mask=map_point_mask,
            map_polyline_mask=map_polyline_mask,
            target_difficulty=target_difficulty,
            noise=noise,
        )
        (
            predicted_plan,
            count_logits,
            count_probs,
            expected_count,
            predicted_count,
            conditioned_count,
            global_context,
            prior_logits,
            selected_prototype_ids,
            prototype_bundle,
            selected_prototype_count,
            prototype_difficulty_mean,
        ) = self._global_context(
            scene_context,
            target_difficulty=target_difficulty,
            override_scene_plan=override_scene_plan,
            override_count=override_count,
            forced_prototype_ids=forced_prototype_ids,
            location_ids=location_ids,
        )
        prototype_states, prototype_mask = prototype_bundle if prototype_bundle is not None else (None, None)
        prototype_count = (
            selected_prototype_count.to(device=device, dtype=dtype)
            if selected_prototype_count is not None
            else (
                prototype_mask.float().sum(dim=-1)
                if prototype_mask is not None
                else torch.zeros(batch_size, device=device, dtype=dtype)
            )
        )
        prototype_difficulty_mean = (
            prototype_difficulty_mean.to(device=device, dtype=dtype)
            if prototype_difficulty_mean is not None
            else torch.zeros(batch_size, device=device, dtype=dtype)
        )
        selected_prototype_plan = (
            self.prototype_plan_features.index_select(0, selected_prototype_ids).to(device=device, dtype=dtype)
            if selected_prototype_ids is not None
            else None
        )
        apply_structured_adaptation = bool(
            self.prototype_enabled
            and self.structured_adaptation_enabled
            and prototype_states is not None
            and prototype_mask is not None
        )
        apply_hybrid_compiler = bool(
            not apply_structured_adaptation
            and self.hybrid_compiler_enabled
            and (use_hybrid_compiler if use_hybrid_compiler is not None else teacher_forcing_states is None)
        )
        hybrid_train_proxy = bool(False)
        if apply_structured_adaptation:
            current_states, presence_logits, agent_mask = self._decode_structured_adaptation(
                global_context=global_context,
                prototype_states=prototype_states,
                prototype_mask=prototype_mask,
                conditioned_count=conditioned_count,
                map_polylines=map_polylines,
                map_point_mask=map_point_mask,
                map_polyline_mask=map_polyline_mask,
                teacher_forcing_mask=teacher_forcing_mask,
            )
        elif apply_hybrid_compiler and prototype_states is not None and prototype_mask is not None:
            current_states, presence_logits, agent_mask = self._compile_scene_from_prototype(
                prototype_states=prototype_states,
                prototype_mask=prototype_mask,
                conditioned_count=conditioned_count,
                conditioned_plan=override_scene_plan.to(device=device, dtype=dtype) if override_scene_plan is not None else predicted_plan,
                map_polylines=map_polylines,
                map_point_mask=map_point_mask,
                map_polyline_mask=map_polyline_mask,
            )
        elif teacher_forcing_states is not None and teacher_forcing_mask is not None:
            current_states, presence_logits = self._decode_teacher_forced(
                teacher_forcing_states=teacher_forcing_states,
                teacher_forcing_mask=teacher_forcing_mask,
                global_context=global_context,
                prototype_states=prototype_states,
                prototype_mask=prototype_mask,
            )
            step_index = torch.arange(self.max_agents, device=device).unsqueeze(0)
            agent_mask = step_index < conditioned_count.round().clamp(min=0.0, max=float(self.max_agents)).to(dtype=torch.long).unsqueeze(1)
        else:
            current_states, presence_logits = self._decode_free_running(
                global_context=global_context,
                predicted_count=conditioned_count,
                prototype_states=prototype_states,
                prototype_mask=prototype_mask,
            )
            step_index = torch.arange(self.max_agents, device=device).unsqueeze(0)
            agent_mask = step_index < conditioned_count.round().clamp(min=0.0, max=float(self.max_agents)).to(dtype=torch.long).unsqueeze(1)
        presence_probs = torch.sigmoid(presence_logits)
        current_states = torch.where(agent_mask.unsqueeze(-1), current_states, torch.zeros_like(current_states))
        actual_generated_count = agent_mask.sum(dim=-1).to(dtype=dtype)
        return {
            "generated_current_states": current_states,
            "generated_presence_logits": presence_logits,
            "generated_presence_probs": presence_probs,
            "generated_count_logits": count_logits,
            "generated_count_probs": count_probs,
            "generated_count": actual_generated_count,
            "predicted_count_head": predicted_count,
            "conditioned_count": conditioned_count,
            "generated_expected_count": expected_count,
            "generated_agent_mask": agent_mask,
            "predicted_scene_plan": predicted_plan,
            "conditioned_scene_plan": override_scene_plan.to(device=device, dtype=dtype) if override_scene_plan is not None else predicted_plan,
            "prototype_prior_logits": prior_logits,
            "selected_prototype_ids": selected_prototype_ids,
            "selected_prototype_count": prototype_count,
            "selected_prototype_difficulty_mean": prototype_difficulty_mean,
            "selected_prototype_plan_features": selected_prototype_plan,
            "selected_prototype_states": prototype_states,
            "selected_prototype_mask": prototype_mask,
            "scene_context": scene_context,
            "global_context": global_context,
            "structured_adaptation_applied": torch.tensor(
                1.0 if apply_structured_adaptation else 0.0,
                device=device,
                dtype=dtype,
            ),
            "hybrid_compiler_applied": apply_hybrid_compiler,
            "hybrid_train_proxy": hybrid_train_proxy,
            "spawn_architecture": torch.tensor(0.0, device=device, dtype=dtype),
        }

    @torch.no_grad()
    def sample(
        self,
        map_polylines: torch.Tensor,
        map_point_mask: torch.Tensor,
        map_polyline_mask: torch.Tensor,
        target_difficulty: torch.Tensor,
        noise: torch.Tensor | None = None,
        presence_threshold: float | None = None,
        override_scene_plan: torch.Tensor | None = None,
        override_count: torch.Tensor | None = None,
        forced_prototype_ids: torch.Tensor | None = None,
        location_ids: Sequence[str] | None = None,
    ) -> Dict[str, torch.Tensor]:
        del presence_threshold
        return self(
            map_polylines=map_polylines,
            map_point_mask=map_point_mask,
            map_polyline_mask=map_polyline_mask,
            target_difficulty=target_difficulty,
            noise=noise,
            teacher_forcing_states=None,
            teacher_forcing_mask=None,
            override_scene_plan=override_scene_plan,
            override_count=override_count,
            forced_prototype_ids=forced_prototype_ids,
            location_ids=location_ids,
            use_hybrid_compiler=not self.structured_adaptation_enabled,
        )
