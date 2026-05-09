from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F

from isgen.models.control_knots import knot_control_delta_loss
from isgen.models.diffusion import DiffusionScheduleConfig, DiffusionScheduler
from isgen.semantics.control_normalization import summarize_control_tensor
from isgen.semantics.generator_condition import (
    behavior_control_loss_in_total,
    generator_consumes_behavior,
    resolve_generator_condition_mode,
)
from isgen.semantics.scene_features import flatten_realism_features
from isgen.semantics.behavior import (
    BehaviorNormalizer,
    behavior_score_from_raw_tensor,
    differentiable_behavior_aggressiveness_score,
    inverse_behavior_score_tensor,
    resolve_behavior_target_key,
    resolve_generator_condition_score_key,
)


def active_training_losses(config: Dict) -> list[str]:
    losses = ["supervised_control", "supervised_rollout"]
    if bool(config.get("interaction_field", {}).get("enabled", False)) and float(config["losses"].get("interaction_field_weight", 0.0)) > 0.0:
        losses.append("interaction_field")
    if bool(config.get("model", {}).get("use_anchor_latent", False)) and float(config["losses"].get("anchor_weight", 1.0)) > 0.0:
        losses.append("anchor")
    residual_cfg = config.get("residual_diffusion", {})
    if float(config["losses"].get("residual_diffusion_weight", 1.0)) > 0.0 and bool(
        residual_cfg.get("enabled", config["model"].get("residual_diffusion_enabled", True))
    ):
        losses.append("residual_diffusion")
    anchor_residual_cfg = config.get("anchor_residual_diffusion", {})
    if bool(config.get("model", {}).get("use_anchor_latent", False)) and float(
        config["losses"].get("anchor_residual_diffusion_weight", 1.0)
    ) > 0.0 and bool(anchor_residual_cfg.get("enabled", False)):
        losses.append("anchor_residual_diffusion")
    if float(config["losses"].get("control_delta_weight", 0.0)) > 0.0:
        losses.append("control_delta")
    if float(config["losses"].get("realism_mmd_weight", 0.0)) > 0.0:
        losses.append("realism")
    if behavior_control_loss_in_total(config):
        losses.append("behavior_control")
    return losses


def _tensor_percentile(values: torch.Tensor, quantile: float) -> torch.Tensor:
    values = values.detach().float().reshape(-1)
    if values.numel() == 0:
        return torch.tensor(0.0, dtype=torch.float32)
    sorted_values, _ = torch.sort(values)
    index = min(int(round((sorted_values.numel() - 1) * float(quantile))), sorted_values.numel() - 1)
    return sorted_values[index]


def _masked_mean(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < loss.ndim:
        mask = mask.unsqueeze(-1)
    if mask.shape != loss.shape:
        mask = mask.expand_as(loss)
    loss = torch.nan_to_num(loss.float(), nan=0.0, posinf=0.0, neginf=0.0)
    masked = torch.where(mask, loss, torch.zeros_like(loss))
    return masked.sum() / torch.clamp(mask.float().sum(), min=1.0)


def _rbf_mmd(real: torch.Tensor, fake: torch.Tensor, gamma: float = 0.5) -> torch.Tensor:
    if real.shape[0] == 0 or fake.shape[0] == 0:
        return torch.zeros((), device=real.device if real.numel() else fake.device, dtype=torch.float32)
    real_sq = torch.cdist(real, real).square()
    fake_sq = torch.cdist(fake, fake).square()
    cross_sq = torch.cdist(real, fake).square()
    xx = torch.exp(-gamma * real_sq).mean()
    yy = torch.exp(-gamma * fake_sq).mean()
    xy = torch.exp(-gamma * cross_sq).mean()
    return torch.nan_to_num(xx + yy - 2.0 * xy, nan=0.0, posinf=0.0, neginf=0.0)


def supervised_control_loss(mu_knots: torch.Tensor, gt_knots: torch.Tensor, knot_mask: torch.Tensor) -> torch.Tensor:
    return _masked_mean(F.smooth_l1_loss(mu_knots, gt_knots, reduction="none"), knot_mask)


def anchor_loss(
    anchor_mu: torch.Tensor,
    anchor_target: torch.Tensor,
    anchor_valid_mask: torch.Tensor,
    config: Dict,
) -> Dict[str, torch.Tensor]:
    zero = torch.zeros((), device=anchor_mu.device, dtype=anchor_mu.dtype)
    if not bool(config.get("model", {}).get("use_anchor_latent", False)):
        return {
            "loss": zero,
            "mid_mae": zero,
            "final_mae": zero,
            "heading_mae": zero,
        }
    element_loss = F.smooth_l1_loss(anchor_mu, anchor_target, reduction="none")
    heading_delta = torch.atan2(
        torch.sin(anchor_mu[..., 5:6] - anchor_target[..., 5:6]),
        torch.cos(anchor_mu[..., 5:6] - anchor_target[..., 5:6]),
    )
    element_loss = torch.cat(
        [element_loss[..., 0:5], F.smooth_l1_loss(heading_delta, torch.zeros_like(heading_delta), reduction="none")],
        dim=-1,
    )
    weights = torch.ones((anchor_mu.shape[-1],), device=anchor_mu.device, dtype=anchor_mu.dtype)
    final_weight = float(config["losses"].get("final_anchor_weight", 2.0))
    if weights.numel() >= 6:
        weights[2:] = final_weight
    weighted_loss = element_loss * weights.view(*([1] * (element_loss.ndim - 1)), -1)
    mid_mae = _masked_mean(torch.abs(anchor_mu[..., 0:2] - anchor_target[..., 0:2]), anchor_valid_mask)
    final_mae = _masked_mean(torch.abs(anchor_mu[..., 2:5] - anchor_target[..., 2:5]), anchor_valid_mask)
    heading_mae = _masked_mean(torch.abs(heading_delta), anchor_valid_mask)
    return {
        "loss": _masked_mean(weighted_loss, anchor_valid_mask),
        "mid_mae": mid_mae.detach(),
        "final_mae": final_mae.detach(),
        "heading_mae": heading_mae.detach(),
    }


def interaction_field_loss(
    predicted_pair_features: torch.Tensor,
    target_pair_features: torch.Tensor,
    predicted_pair_latent: torch.Tensor,
    pair_valid: torch.Tensor,
    config: Dict,
) -> Dict[str, torch.Tensor]:
    zero = torch.zeros((), device=predicted_pair_features.device, dtype=predicted_pair_features.dtype)
    if not bool(config.get("interaction_field", {}).get("enabled", False)):
        return {
            "loss": zero,
            "min_distance_mae": zero,
            "conflict_mae": zero,
            "distance_change_mae": zero,
            "final_distance_mae": zero,
            "pair_latent_std": zero,
        }
    if not bool(pair_valid.any()):
        return {
            "loss": zero,
            "min_distance_mae": zero,
            "conflict_mae": zero,
            "distance_change_mae": zero,
            "final_distance_mae": zero,
            "pair_latent_std": zero,
        }
    expanded_mask = pair_valid.unsqueeze(-1).expand_as(target_pair_features)
    valid_target = target_pair_features[expanded_mask].reshape(-1, target_pair_features.shape[-1]).float()
    scale = valid_target.std(dim=0, unbiased=False)
    scale = torch.clamp(scale, min=0.1)
    scale = scale.view(*([1] * (target_pair_features.ndim - 1)), -1).to(device=predicted_pair_features.device, dtype=predicted_pair_features.dtype)
    normalized_pred = predicted_pair_features / scale
    normalized_target = target_pair_features / scale
    feature_weights = config["losses"].get("interaction_field_feature_weights", [2.0, 0.5, 0.5, 0.5, 0.5, 1.5, 2.0, 2.0])
    weights_tensor = torch.tensor(feature_weights, device=predicted_pair_features.device, dtype=predicted_pair_features.dtype)
    if weights_tensor.numel() < predicted_pair_features.shape[-1]:
        pad = torch.ones(
            predicted_pair_features.shape[-1] - weights_tensor.numel(),
            device=predicted_pair_features.device,
            dtype=predicted_pair_features.dtype,
        )
        weights_tensor = torch.cat([weights_tensor, pad], dim=0)
    weights_tensor = weights_tensor[: predicted_pair_features.shape[-1]]
    weighted_loss = F.smooth_l1_loss(normalized_pred, normalized_target, reduction="none") * weights_tensor.view(
        *([1] * (predicted_pair_features.ndim - 1)),
        -1,
    )
    loss = _masked_mean(weighted_loss, pair_valid)
    min_distance_mae = _masked_mean(torch.abs(predicted_pair_features[..., 0] - target_pair_features[..., 0]), pair_valid)
    distance_change_mae = _masked_mean(torch.abs(predicted_pair_features[..., 6] - target_pair_features[..., 6]), pair_valid)
    conflict_mae = _masked_mean(torch.abs(predicted_pair_features[..., 7] - target_pair_features[..., 7]), pair_valid)
    final_distance_mae = _masked_mean(torch.abs(predicted_pair_features[..., 5] - target_pair_features[..., 5]), pair_valid)
    expanded_latent_mask = pair_valid.unsqueeze(-1).expand_as(predicted_pair_latent)
    pair_latent_std = predicted_pair_latent[expanded_latent_mask].float().std(unbiased=False) if bool(expanded_latent_mask.any()) else zero
    return {
        "loss": loss,
        "min_distance_mae": min_distance_mae.detach(),
        "conflict_mae": conflict_mae.detach(),
        "distance_change_mae": distance_change_mae.detach(),
        "final_distance_mae": final_distance_mae.detach(),
        "pair_latent_std": pair_latent_std.detach(),
    }


def diffusion_loss(predicted_noise: torch.Tensor, training_target: torch.Tensor, future_mask: torch.Tensor) -> torch.Tensor:
    return _masked_mean((predicted_noise - training_target).square(), future_mask)


def supervised_rollout_loss(
    predicted_future: torch.Tensor,
    gt_future: torch.Tensor,
    future_mask: torch.Tensor,
    config: Dict,
) -> torch.Tensor:
    position = _masked_mean(F.smooth_l1_loss(predicted_future[..., 0:2], gt_future[..., 0:2], reduction="none"), future_mask)
    velocity = _masked_mean(F.smooth_l1_loss(predicted_future[..., 2:4], gt_future[..., 2:4], reduction="none"), future_mask)
    heading_delta = torch.atan2(
        torch.sin(predicted_future[..., 4:5] - gt_future[..., 4:5]),
        torch.cos(predicted_future[..., 4:5] - gt_future[..., 4:5]),
    )
    heading = _masked_mean(F.smooth_l1_loss(heading_delta, torch.zeros_like(heading_delta), reduction="none"), future_mask)
    return (
        float(config["losses"]["position_weight"]) * position
        + float(config["losses"]["velocity_weight"]) * velocity
        + float(config["losses"]["heading_weight"]) * heading
    )


def reconstruction_loss(
    predicted_future: torch.Tensor,
    gt_future: torch.Tensor,
    future_mask: torch.Tensor,
    timesteps: torch.Tensor,
    config: Dict,
) -> torch.Tensor:
    del timesteps
    return supervised_rollout_loss(predicted_future=predicted_future, gt_future=gt_future, future_mask=future_mask, config=config)


def residual_diffusion_loss(
    model_output: Dict[str, torch.Tensor],
    model,
    config: Dict,
) -> Dict[str, torch.Tensor]:
    residual_target = model_output["residual_target"]
    knot_mask = model_output["knot_mask"]
    timesteps = model_output["timesteps"]
    device = residual_target.device
    zero = torch.zeros((), device=device, dtype=residual_target.dtype)
    residual_cfg = config.get("residual_diffusion", {})
    if not bool(residual_cfg.get("enabled", config["model"].get("residual_diffusion_enabled", True))):
        return {
            "loss": zero,
            "unweighted": zero,
            "target_std": zero,
        }
    scheduler = (
        model.scheduler
        if model is not None
        else DiffusionScheduler(
            DiffusionScheduleConfig(
                num_steps=int(config["diffusion"]["num_steps"]),
                beta_start=float(config["diffusion"]["beta_start"]),
                beta_end=float(config["diffusion"]["beta_end"]),
            ),
            device=device,
        )
    )
    target_type = str(residual_cfg.get("target_type", config["diffusion"].get("target_type", "epsilon")))
    training_target = scheduler.training_target(
        residual_target,
        model_output["noise"],
        timesteps,
        target_type,
    )
    if target_type == "x0":
        element_loss = F.smooth_l1_loss(model_output["predicted_noise"], training_target, reduction="none")
    else:
        element_loss = (model_output["predicted_noise"] - training_target).square()
    weight = scheduler.min_snr_weight(
        timesteps,
        float(config["diffusion"].get("min_snr_gamma", 0.0)),
        target_type,
    ).view(-1, 1, 1, 1)
    weighted_loss = element_loss * weight
    return {
        "loss": _masked_mean(weighted_loss, knot_mask),
        "unweighted": _masked_mean(element_loss, knot_mask),
        "target_std": residual_target[knot_mask.unsqueeze(-1).expand_as(residual_target)].std(unbiased=False)
        if knot_mask.any()
        else zero,
    }


def anchor_residual_diffusion_loss(
    model_output: Dict[str, torch.Tensor],
    model,
    config: Dict,
) -> Dict[str, torch.Tensor]:
    anchor_target = model_output["anchor_residual_target"]
    anchor_mask = model_output["anchor_valid_mask"]
    timesteps = model_output["anchor_timesteps"]
    device = anchor_target.device
    zero = torch.zeros((), device=device, dtype=anchor_target.dtype)
    anchor_residual_cfg = config.get("anchor_residual_diffusion", {})
    if not bool(config.get("model", {}).get("use_anchor_latent", False)) or not bool(anchor_residual_cfg.get("enabled", False)):
        return {
            "loss": zero,
            "unweighted": zero,
            "target_std": zero,
        }
    scheduler = (
        model.scheduler
        if model is not None
        else DiffusionScheduler(
            DiffusionScheduleConfig(
                num_steps=int(config["diffusion"]["num_steps"]),
                beta_start=float(config["diffusion"]["beta_start"]),
                beta_end=float(config["diffusion"]["beta_end"]),
            ),
            device=device,
        )
    )
    target_type = str(anchor_residual_cfg.get("target_type", "x0"))
    training_target = scheduler.training_target(
        anchor_target,
        model_output["anchor_noise"],
        timesteps,
        target_type,
    )
    if target_type == "x0":
        element_loss = F.smooth_l1_loss(model_output["anchor_predicted_noise"], training_target, reduction="none")
    else:
        element_loss = (model_output["anchor_predicted_noise"] - training_target).square()
    weight = scheduler.min_snr_weight(
        timesteps,
        float(config["diffusion"].get("min_snr_gamma", 0.0)),
        target_type,
    ).view(-1, 1, 1)
    weighted_loss = element_loss * weight
    expanded_mask = anchor_mask.unsqueeze(-1).expand_as(anchor_target)
    return {
        "loss": _masked_mean(weighted_loss, anchor_mask),
        "unweighted": _masked_mean(element_loss, anchor_mask),
        "target_std": anchor_target[expanded_mask].std(unbiased=False) if anchor_mask.any() else zero,
    }


def check_behavior_label_consistency(
    batch: Dict[str, torch.Tensor],
    normalizer: BehaviorNormalizer,
    config: Dict,
) -> Dict[str, Any]:
    target_score_key = resolve_behavior_target_key(config)
    recomputed_aggressiveness, behavior_details = differentiable_behavior_aggressiveness_score(
        current_states=batch["current_states"],
        future_states=batch["future_states"],
        future_mask=batch["future_mask"],
        agent_mask=batch["agent_mask"],
        normalizer=normalizer,
        config=config,
    )
    recomputed_behavior = behavior_score_from_raw_tensor(
        behavior_details["raw_score"],
        normalizer=normalizer,
        score_key=target_score_key,
    )
    cached_behavior = batch[target_score_key]
    abs_error = torch.abs(recomputed_behavior - cached_behavior)
    return {
        "behavior_label_mae": float(abs_error.mean().item()),
        "recomputed_behavior": recomputed_behavior,
        "recomputed_behavior_aggressiveness": recomputed_aggressiveness,
        "behavior_target_key": target_score_key,
        "behavior_details": behavior_details,
    }


def behavior_control_loss(
    current_states: torch.Tensor,
    predicted_future: torch.Tensor,
    future_mask: torch.Tensor,
    agent_mask: torch.Tensor,
    target_behavior: torch.Tensor,
    timesteps: torch.Tensor,
    normalizer: BehaviorNormalizer,
    config: Dict,
) -> Dict[str, Any]:
    generated_metrics = generated_behavior_metrics(
        current_states=current_states,
        predicted_future=predicted_future,
        future_mask=future_mask,
        agent_mask=agent_mask,
        normalizer=normalizer,
        config=config,
    )
    generated_behavior_aggressiveness = generated_metrics["generated_behavior_aggressiveness"]
    generated_details = generated_metrics["generated_details"]
    generated_behavior = generated_metrics["generated_behavior"]
    target_score_key = generated_metrics["behavior_target_key"]
    scalar_loss = F.smooth_l1_loss(generated_behavior, target_behavior, reduction="none")
    target_raw_score = inverse_behavior_score_tensor(target_behavior, normalizer=normalizer, score_key=target_score_key)
    raw_score_loss = F.smooth_l1_loss(generated_details["raw_score"], target_raw_score, reduction="none")
    num_steps = int(config["diffusion"]["num_steps"])
    active_mask = torch.ones_like(target_behavior, dtype=torch.bool)
    if bool(config["losses"].get("behavior_low_noise_only", False)):
        active_mask = timesteps <= int(float(config["losses"].get("behavior_max_timestep_ratio", 0.3)) * num_steps)
    timestep_ratio = timesteps.float() / max(num_steps, 1)
    timestep_weight = torch.exp(-float(config["losses"].get("behavior_timestep_alpha", 0.0)) * timestep_ratio)
    timestep_weight = timestep_weight * active_mask.float()
    active_denominator = torch.clamp(timestep_weight.sum(), min=1e-6)
    loss = ((scalar_loss + 0.25 * raw_score_loss) * timestep_weight).sum() / active_denominator
    if not torch.isfinite(loss):
        raise RuntimeError(
            "Behavior control loss became non-finite after aggregation. "
            f"target_score_key={target_score_key}"
        )
    return {
        **generated_metrics,
        "loss": loss,
        "active_fraction": active_mask.float().mean(),
    }


def generated_behavior_metrics(
    current_states: torch.Tensor,
    predicted_future: torch.Tensor,
    future_mask: torch.Tensor,
    agent_mask: torch.Tensor,
    normalizer: BehaviorNormalizer,
    config: Dict,
) -> Dict[str, Any]:
    target_score_key = resolve_behavior_target_key(config)
    generated_behavior_aggressiveness, generated_details = differentiable_behavior_aggressiveness_score(
        current_states=current_states,
        future_states=predicted_future,
        future_mask=future_mask,
        agent_mask=agent_mask,
        normalizer=normalizer,
        config=config,
    )
    generated_behavior = behavior_score_from_raw_tensor(
        generated_details["raw_score"],
        normalizer=normalizer,
        score_key=target_score_key,
    )
    non_finite_raw_features = [
        key for key, value in generated_details["raw_features"].items() if not torch.isfinite(value).all()
    ]
    if non_finite_raw_features:
        raise RuntimeError(
            "Behavior control produced non-finite raw features: "
            + ", ".join(sorted(non_finite_raw_features))
        )
    if not torch.isfinite(generated_details["raw_score"]).all():
        raise RuntimeError("Behavior control produced non-finite raw_score.")
    if not torch.isfinite(generated_behavior).all():
        raise RuntimeError(
            f"Behavior control produced non-finite generated_behavior in score space '{target_score_key}'."
        )
    if not torch.isfinite(generated_behavior_aggressiveness).all():
        raise RuntimeError("Behavior control produced non-finite generated_behavior_aggressiveness.")
    return {
        "generated_behavior": generated_behavior,
        "generated_behavior_aggressiveness": generated_behavior_aggressiveness,
        "generated_details": generated_details,
        "behavior_target_key": target_score_key,
    }


def control_delta_loss(
    decoded_controls_physical: torch.Tensor,
    future_mask: torch.Tensor,
    agent_mask: torch.Tensor,
    config: Dict,
) -> Dict[str, torch.Tensor]:
    weight = float(config["losses"].get("control_delta_weight", 0.0))
    if weight <= 0.0:
        zero = torch.zeros((), device=decoded_controls_physical.device, dtype=decoded_controls_physical.dtype)
        return {
            "loss": zero,
            "jerk_term": zero,
            "yaw_accel_term": zero,
        }
    loss = knot_control_delta_loss(
        decoded_controls_physical,
        future_mask=future_mask,
        agent_mask=agent_mask,
        jerk_weight=float(config["losses"].get("control_delta_jerk_weight", 1.0)),
        yaw_accel_weight=float(config["losses"].get("control_delta_yaw_accel_weight", 0.5)),
    )
    valid = future_mask & agent_mask.unsqueeze(-1)
    accel = decoded_controls_physical[..., 0]
    yaw_rate = decoded_controls_physical[..., 1]
    delta_valid = valid[..., 1:] & valid[..., :-1]
    jerk_delta = accel[..., 1:] - accel[..., :-1]
    yaw_accel_delta = yaw_rate[..., 1:] - yaw_rate[..., :-1]
    jerk_term = _masked_mean(jerk_delta.square(), delta_valid)
    yaw_accel_term = _masked_mean(yaw_accel_delta.square(), delta_valid)
    return {
        "loss": loss,
        "jerk_term": jerk_term.detach(),
        "yaw_accel_term": yaw_accel_term.detach(),
    }


def realism_distribution_loss(
    predicted_future: torch.Tensor,
    gt_future: torch.Tensor,
    future_mask: torch.Tensor,
    current_states: torch.Tensor,
    agent_mask: torch.Tensor,
    config: Dict,
) -> Dict[str, torch.Tensor | str]:
    weight = float(config["losses"].get("realism_mmd_weight", 0.0))
    if weight <= 0.0:
        with torch.no_grad():
            pred_features = flatten_realism_features(
                future_states=predicted_future.detach(),
                future_mask=future_mask,
                current_states=current_states.detach(),
                agent_mask=agent_mask,
                dt=float(config["data"]["timestep_sec"]),
            )
            gt_features = flatten_realism_features(
                future_states=gt_future.detach(),
                future_mask=future_mask,
                current_states=current_states.detach(),
                agent_mask=agent_mask,
                dt=float(config["data"]["timestep_sec"]),
            )
            pred_features = torch.nan_to_num(pred_features, nan=0.0, posinf=0.0, neginf=0.0)
            gt_features = torch.nan_to_num(gt_features, nan=0.0, posinf=0.0, neginf=0.0)
            valid_rows = (future_mask & agent_mask.unsqueeze(-1)).any(dim=(-1, -2))
            pred_valid = pred_features[valid_rows]
            gt_valid = gt_features[valid_rows]
            zero = torch.zeros((), device=predicted_future.device, dtype=predicted_future.dtype)
            if pred_valid.shape[0] == 0 or gt_valid.shape[0] == 0:
                return {
                    "loss": zero,
                    "weighted": zero,
                    "enabled": torch.zeros((), device=predicted_future.device, dtype=predicted_future.dtype),
                    "weight": torch.as_tensor(weight, device=predicted_future.device, dtype=predicted_future.dtype),
                    "feature_dim": torch.zeros((), device=predicted_future.device, dtype=predicted_future.dtype),
                    "valid_count": torch.zeros((), device=predicted_future.device, dtype=predicted_future.dtype),
                    "pred_feature_std": zero,
                    "gt_feature_std": zero,
                    "disabled_reason": "weight_zero",
                }
            pred_center = pred_valid - pred_valid.mean(dim=0, keepdim=True)
            gt_center = gt_valid - gt_valid.mean(dim=0, keepdim=True)
            pred_std = pred_valid.std(dim=0, unbiased=False)
            gt_std = gt_valid.std(dim=0, unbiased=False)
            denom = torch.clamp(gt_std, min=1e-4)
            pred_standardized = pred_center / denom
            gt_standardized = gt_center / denom
            unweighted = _rbf_mmd(gt_standardized, pred_standardized, gamma=0.5).detach()
            return {
                "loss": unweighted,
                "weighted": zero,
                "enabled": torch.zeros((), device=predicted_future.device, dtype=predicted_future.dtype),
                "weight": torch.as_tensor(weight, device=predicted_future.device, dtype=predicted_future.dtype),
                "feature_dim": torch.as_tensor(float(pred_valid.shape[-1]), device=predicted_future.device, dtype=predicted_future.dtype),
                "valid_count": torch.as_tensor(float(pred_valid.shape[0]), device=predicted_future.device, dtype=predicted_future.dtype),
                "pred_feature_std": pred_std.mean().detach(),
                "gt_feature_std": gt_std.mean().detach(),
                "disabled_reason": "weight_zero",
            }
    pred_features = flatten_realism_features(
        future_states=predicted_future,
        future_mask=future_mask,
        current_states=current_states,
        agent_mask=agent_mask,
        dt=float(config["data"]["timestep_sec"]),
    )
    gt_features = flatten_realism_features(
        future_states=gt_future,
        future_mask=future_mask,
        current_states=current_states,
        agent_mask=agent_mask,
        dt=float(config["data"]["timestep_sec"]),
    )
    pred_features = torch.nan_to_num(pred_features, nan=0.0, posinf=0.0, neginf=0.0)
    gt_features = torch.nan_to_num(gt_features, nan=0.0, posinf=0.0, neginf=0.0)
    valid_rows = (future_mask & agent_mask.unsqueeze(-1)).any(dim=(-1, -2))
    pred_valid = pred_features[valid_rows]
    gt_valid = gt_features[valid_rows]
    if pred_valid.shape[0] == 0 or gt_valid.shape[0] == 0:
        zero = torch.zeros((), device=predicted_future.device, dtype=predicted_future.dtype)
        return {
            "loss": zero,
            "weighted": zero,
            "enabled": torch.zeros((), device=predicted_future.device, dtype=predicted_future.dtype),
            "weight": torch.as_tensor(weight, device=predicted_future.device, dtype=predicted_future.dtype),
            "feature_dim": torch.zeros((), device=predicted_future.device, dtype=predicted_future.dtype),
            "valid_count": torch.zeros((), device=predicted_future.device, dtype=predicted_future.dtype),
            "pred_feature_std": zero,
            "gt_feature_std": zero,
            "disabled_reason": "no_valid_rows",
        }
    pred_center = pred_valid - pred_valid.mean(dim=0, keepdim=True)
    gt_center = gt_valid - gt_valid.mean(dim=0, keepdim=True)
    pred_std = pred_valid.std(dim=0, unbiased=False)
    gt_std = gt_valid.std(dim=0, unbiased=False)
    denom = torch.clamp(gt_std, min=1e-4)
    pred_standardized = pred_center / denom
    gt_standardized = gt_center / denom
    unweighted = _rbf_mmd(gt_standardized, pred_standardized, gamma=0.5)
    weighted = unweighted * weight
    return {
        "loss": unweighted,
        "weighted": weighted,
        "enabled": torch.as_tensor(1.0 if weight > 0.0 else 0.0, device=predicted_future.device, dtype=predicted_future.dtype),
        "weight": torch.as_tensor(weight, device=predicted_future.device, dtype=predicted_future.dtype),
        "feature_dim": torch.as_tensor(float(pred_valid.shape[-1]), device=predicted_future.device, dtype=predicted_future.dtype),
        "valid_count": torch.as_tensor(float(pred_valid.shape[0]), device=predicted_future.device, dtype=predicted_future.dtype),
        "pred_feature_std": pred_std.mean().detach(),
        "gt_feature_std": gt_std.mean().detach(),
        "disabled_reason": "weight_zero" if weight <= 0.0 else "",
    }


def _sample_path_ddim_rollout(
    model,
    batch: Dict[str, torch.Tensor],
    sample_steps: int,
) -> Dict[str, torch.Tensor]:
    device = batch["current_states"].device
    future_mask = batch["future_mask"]
    agent_mask = batch["agent_mask"]
    valid_mask = future_mask & agent_mask.unsqueeze(-1)
    shape = (*future_mask.shape, model.model_future_dim)
    encoded = torch.randn(shape, device=device)
    encoded = torch.where(valid_mask.unsqueeze(-1), encoded, torch.zeros_like(encoded))
    indices = model.scheduler._sampling_indices(sample_steps)
    for step_idx, timestep in enumerate(indices):
        timesteps = torch.full((shape[0],), int(timestep.item()), device=device, dtype=torch.long)
        predicted_noise = model.predict_noise(
            noisy_future=encoded,
            timesteps=timesteps,
            history_states=batch["history_states"],
            history_mask=batch["history_mask"],
            current_states=batch["current_states"],
            future_mask=batch["future_mask"],
            agent_mask=batch["agent_mask"],
            map_polylines=batch["map_polylines"],
            map_point_mask=batch["map_point_mask"],
            map_polyline_mask=batch["map_polyline_mask"],
            target_difficulty=batch.get("target_difficulty"),
            target_behavior=batch["generator_condition_behavior"],
        )
        predicted_noise = torch.where(valid_mask.unsqueeze(-1), predicted_noise, torch.zeros_like(predicted_noise))
        prev_timestep = int(indices[step_idx + 1].item()) if step_idx + 1 < len(indices) else -1
        alpha_bar_prev = (
            torch.tensor(1.0, device=device, dtype=encoded.dtype)
            if prev_timestep < 0
            else model.scheduler.alpha_bars[prev_timestep]
        )
        pred_x0 = model.scheduler.predict_start_from_noise(encoded, timesteps, predicted_noise)
        pred_x0 = torch.where(valid_mask.unsqueeze(-1), pred_x0, torch.zeros_like(pred_x0))
        direction = torch.sqrt(torch.clamp(1 - alpha_bar_prev, min=0.0)) * predicted_noise
        encoded = torch.sqrt(torch.clamp(alpha_bar_prev, min=1e-6)) * pred_x0 + direction
        encoded = torch.where(valid_mask.unsqueeze(-1), encoded, torch.zeros_like(encoded))
    if model.decoder_type == "kinematic_controls":
        raw_controls = model.denormalize_controls(encoded)
        raw_controls = model.clamp_raw_controls_to_train_quantiles(raw_controls)
        decoded_future, decoded_controls_physical = model.decode_controls_to_future(
            current_states=batch["current_states"],
            controls=raw_controls,
            future_mask=batch["future_mask"],
            agent_mask=batch["agent_mask"],
            dt=float(model.config["data"]["timestep_sec"]),
        )
    else:
        decoded_future = model.decode_future_local(batch["current_states"], encoded)
        decoded_controls_physical = torch.zeros(
            (*encoded.shape[:-1], 2),
            device=device,
            dtype=encoded.dtype,
        )
    return {
        "decoded_future": decoded_future,
        "decoded_controls_physical": decoded_controls_physical,
    }


def sample_path_loss(
    model,
    batch: Dict[str, torch.Tensor],
    behavior_normalizer: BehaviorNormalizer,
    config: Dict,
    batch_idx: int,
) -> Dict[str, torch.Tensor]:
    enabled = bool(config["losses"].get("sample_path_loss_enabled", False)) and generator_consumes_behavior(config)
    device = batch["current_states"].device
    zero = torch.zeros((), device=device, dtype=batch["current_states"].dtype)
    if not enabled:
        return {
            "loss": zero,
            "behavior_mae": zero,
            "behavior_bias": zero,
            "motion_guardrail": zero,
            "active_fraction": zero,
        }
    every_n_batches = max(int(config["losses"].get("sample_path_every_n_batches", 10)), 1)
    if batch_idx % every_n_batches != 0:
        return {
            "loss": zero,
            "behavior_mae": zero,
            "behavior_bias": zero,
            "motion_guardrail": zero,
            "active_fraction": zero,
        }
    fraction = float(config["losses"].get("sample_path_batch_fraction", 0.25))
    batch_size = batch["current_states"].shape[0]
    subset_size = max(1, min(batch_size, int(round(batch_size * fraction))))
    indices = torch.randperm(batch_size, device=device)[:subset_size]
    subset = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.shape[0] == batch_size:
            subset[key] = value[indices]
        else:
            subset[key] = value
    rollout = _sample_path_ddim_rollout(
        model=model,
        batch=subset,
        sample_steps=max(int(config["losses"].get("sample_path_steps", 8)), 2),
    )
    behavior_result = behavior_control_loss(
        current_states=subset["current_states"],
        predicted_future=rollout["decoded_future"],
        future_mask=subset["future_mask"],
        agent_mask=subset["agent_mask"],
        target_behavior=subset["generator_condition_behavior"],
        timesteps=torch.zeros(subset["current_states"].shape[0], dtype=torch.long, device=device),
        normalizer=behavior_normalizer,
        config=config,
    )
    accel = rollout["decoded_controls_physical"][..., 0]
    yaw_rate = rollout["decoded_controls_physical"][..., 1]
    valid = subset["future_mask"] & subset["agent_mask"].unsqueeze(-1)
    motion_guardrail = zero
    if model.control_normalizer is not None:
        accel_limit = float(model.control_normalizer.channel_stats["accel"].get("abs_p99", 0.0))
        yaw_limit = float(model.control_normalizer.channel_stats["yaw_rate"].get("abs_p99", 0.0))
        if accel_limit > 0.0:
            motion_guardrail = motion_guardrail + _masked_mean(torch.relu(accel.abs() - accel_limit), valid)
        if yaw_limit > 0.0:
            motion_guardrail = motion_guardrail + 0.5 * _masked_mean(torch.relu(yaw_rate.abs() - yaw_limit), valid)
    generated_behavior = behavior_result["generated_behavior"]
    target_behavior = subset["generator_condition_behavior"]
    behavior_bias = (generated_behavior - target_behavior).mean().detach()
    behavior_mae = torch.abs(generated_behavior - target_behavior).mean().detach()
    loss = behavior_result["loss"] + 0.25 * motion_guardrail
    return {
        "loss": loss,
        "behavior_mae": behavior_mae,
        "behavior_bias": behavior_bias,
        "motion_guardrail": motion_guardrail.detach(),
        "active_fraction": torch.as_tensor(float(subset_size) / float(max(batch_size, 1)), device=device, dtype=batch["current_states"].dtype),
    }


def compute_total_loss(
    model_output: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    behavior_normalizer: BehaviorNormalizer,
    config: Dict,
    model=None,
    training: bool = False,
    batch_idx: int = 0,
    compute_debug_metrics: bool = True,
) -> Dict[str, Any]:
    target_behavior_key = resolve_behavior_target_key(config)
    generator_condition_behavior_key = resolve_generator_condition_score_key(config)
    condition_mode = resolve_generator_condition_mode(config)
    behavior_loss_enabled = behavior_control_loss_in_total(config)
    interaction_result = interaction_field_loss(
        model_output["predicted_interaction_pair_features"],
        model_output["oracle_interaction_pair_features"],
        model_output["predicted_interaction_pair_latent"],
        model_output["interaction_pair_valid"],
        config,
    )
    anchor_result = anchor_loss(
        model_output["anchor_mu"],
        model_output["anchor_target"],
        model_output["anchor_valid_mask"],
        config,
    )
    supervised_control_term = supervised_control_loss(
        model_output["mu_knots"],
        model_output["gt_knots"],
        model_output["knot_mask"],
    )
    supervised_rollout_term = supervised_rollout_loss(
        predicted_future=model_output["mu_decoded_future"],
        gt_future=batch["future_states"],
        future_mask=batch["future_mask"],
        config=config,
    )
    oracle_knot_rollout_term = supervised_rollout_loss(
        predicted_future=model_output["gt_knots_decoded_future"],
        gt_future=batch["future_states"],
        future_mask=batch["future_mask"],
        config=config,
    )
    residual_diffusion_result = residual_diffusion_loss(
        model_output=model_output,
        model=model,
        config=config,
    )
    anchor_residual_result = anchor_residual_diffusion_loss(
        model_output=model_output,
        model=model,
        config=config,
    )
    behavior_zero = torch.zeros((), device=batch["target_behavior"].device, dtype=batch["target_behavior"].dtype)
    behavior_result = (
        behavior_control_loss(
            current_states=batch["current_states"],
            predicted_future=model_output["decoded_future"],
            future_mask=batch["future_mask"],
            agent_mask=batch["agent_mask"],
            target_behavior=batch["target_behavior"],
            timesteps=model_output["timesteps"],
            normalizer=behavior_normalizer,
            config=config,
        )
        if behavior_loss_enabled
        else (
            lambda generated_behavior_result: {
                **generated_behavior_result,
                "loss": behavior_zero,
                "active_fraction": torch.zeros_like(behavior_zero),
            }
        )(
            generated_behavior_metrics(
                current_states=batch["current_states"].detach(),
                predicted_future=model_output["sampled_decoded_future"].detach(),
                future_mask=batch["future_mask"],
                agent_mask=batch["agent_mask"],
                normalizer=behavior_normalizer,
                config=config,
            )
        )
    )
    control_delta_result = control_delta_loss(
        decoded_controls_physical=model_output["mu_controls_physical"],
        future_mask=batch["future_mask"],
        agent_mask=batch["agent_mask"],
        config=config,
    )
    realism_result = realism_distribution_loss(
        predicted_future=model_output["sampled_decoded_future"],
        gt_future=batch["future_states"],
        future_mask=batch["future_mask"],
        current_states=batch["current_states"],
        agent_mask=batch["agent_mask"],
        config=config,
    )
    zero = torch.zeros((), device=batch["target_behavior"].device, dtype=batch["target_behavior"].dtype)
    mu_knot_mae = _masked_mean(
        torch.abs(model_output["mu_knots"] - model_output["gt_knots"]),
        model_output["knot_mask"],
    )
    knot_abs_error = torch.abs(model_output["mu_knots"] - model_output["gt_knots"])
    knot_abs_error = knot_abs_error[model_output["knot_mask"].unsqueeze(-1).expand_as(knot_abs_error)]
    knot_abs_error = knot_abs_error[torch.isfinite(knot_abs_error)]
    if knot_abs_error.numel() > 0:
        knot_abs_error = knot_abs_error.float()
        mu_knot_abs_p95 = torch.quantile(knot_abs_error, 0.95)
        mu_knot_abs_p99 = torch.quantile(knot_abs_error, 0.99)
    else:
        mu_knot_abs_p95 = zero
        mu_knot_abs_p99 = zero
    mu_rollout_gap_to_oracle = torch.clamp(supervised_rollout_term - oracle_knot_rollout_term, min=0.0)
    expanded_knot_mask = model_output["knot_mask"].unsqueeze(-1).expand_as(model_output["gt_knots"])
    residual_target_raw = torch.nan_to_num(model_output["residual_target_raw"], nan=0.0, posinf=0.0, neginf=0.0)
    sampled_residual_knots = torch.nan_to_num(model_output["sampled_residual_knots"], nan=0.0, posinf=0.0, neginf=0.0)
    final_knots = torch.nan_to_num(model_output["final_knots"], nan=0.0, posinf=0.0, neginf=0.0)
    gt_knots = torch.nan_to_num(model_output["gt_knots"], nan=0.0, posinf=0.0, neginf=0.0)
    residual_target_norm_unclipped = torch.nan_to_num(model_output["residual_target_norm_unclipped"], nan=0.0, posinf=0.0, neginf=0.0)
    residual_target_clipped = torch.nan_to_num(model_output["residual_target"], nan=0.0, posinf=0.0, neginf=0.0)
    residual_knot_std = (
        residual_target_raw[expanded_knot_mask].float().std(unbiased=False)
        if bool(expanded_knot_mask.any())
        else zero
    )
    sampled_residual_std = (
        sampled_residual_knots[expanded_knot_mask].float().std(unbiased=False)
        if bool(expanded_knot_mask.any())
        else zero
    )
    final_knot_std = final_knots[expanded_knot_mask].float().std(unbiased=False) if bool(expanded_knot_mask.any()) else zero
    gt_knot_std = gt_knots[expanded_knot_mask].float().std(unbiased=False) if bool(expanded_knot_mask.any()) else zero
    clip_delta = torch.abs(residual_target_norm_unclipped - residual_target_clipped) > 1e-6
    residual_clip_fraction = (
        (clip_delta[expanded_knot_mask].float().mean() if bool(expanded_knot_mask.any()) else zero)
        .to(dtype=zero.dtype)
    )
    mu_energy = torch.sqrt(torch.clamp((model_output["mu_knots"][expanded_knot_mask].float().square().sum()), min=1e-12))
    residual_energy = torch.sqrt(torch.clamp((residual_target_raw[expanded_knot_mask].float().square().sum()), min=0.0))
    residual_energy_ratio = residual_energy / torch.clamp(mu_energy, min=1e-6)
    anchor_expanded_mask = model_output["anchor_valid_mask"].unsqueeze(-1).expand_as(model_output["anchor_target"])
    anchor_residual_raw = torch.nan_to_num(model_output["anchor_residual_target_raw"], nan=0.0, posinf=0.0, neginf=0.0)
    sampled_anchor_residual = torch.nan_to_num(model_output["sampled_anchor_residual"], nan=0.0, posinf=0.0, neginf=0.0)
    anchor_target_tensor = torch.nan_to_num(model_output["anchor_target"], nan=0.0, posinf=0.0, neginf=0.0)
    anchor_mu_tensor = torch.nan_to_num(model_output["anchor_mu"], nan=0.0, posinf=0.0, neginf=0.0)
    sampled_anchor_tensor = torch.nan_to_num(model_output["sampled_anchor"], nan=0.0, posinf=0.0, neginf=0.0)
    anchor_residual_std = (
        anchor_residual_raw[anchor_expanded_mask].float().std(unbiased=False)
        if bool(anchor_expanded_mask.any())
        else zero
    )
    sampled_anchor_residual_std = (
        sampled_anchor_residual[anchor_expanded_mask].float().std(unbiased=False)
        if bool(anchor_expanded_mask.any())
        else zero
    )
    anchor_mu_std = anchor_mu_tensor[anchor_expanded_mask].float().std(unbiased=False) if bool(anchor_expanded_mask.any()) else zero
    anchor_target_std = anchor_target_tensor[anchor_expanded_mask].float().std(unbiased=False) if bool(anchor_expanded_mask.any()) else zero
    final_anchor_std = sampled_anchor_tensor[anchor_expanded_mask].float().std(unbiased=False) if bool(anchor_expanded_mask.any()) else zero
    behavior_label_debug = check_behavior_label_consistency(batch, behavior_normalizer, config) if compute_debug_metrics else None
    target_behavior_raw = inverse_behavior_score_tensor(
        batch["target_behavior"],
        normalizer=behavior_normalizer,
        score_key=target_behavior_key,
    )
    target_behavior_quantile = (
        batch["target_behavior"]
        if target_behavior_key == "behavior_quantile_score_selected_agents"
        else behavior_score_from_raw_tensor(target_behavior_raw, behavior_normalizer, "behavior_quantile_score_selected_agents")
    )
    target_behavior_aggressiveness = behavior_score_from_raw_tensor(
        target_behavior_raw, behavior_normalizer, "behavior_aggressiveness_score_selected_agents"
    )
    generator_condition_raw = inverse_behavior_score_tensor(
        batch["generator_condition_behavior"],
        normalizer=behavior_normalizer,
        score_key=generator_condition_behavior_key,
    )
    generator_condition_quantile = (
        batch["generator_condition_behavior"]
        if generator_condition_behavior_key == "behavior_quantile_score_selected_agents"
        else behavior_score_from_raw_tensor(generator_condition_raw, behavior_normalizer, "behavior_quantile_score_selected_agents")
    )
    generated_behavior_raw = behavior_result["generated_details"]["raw_score"]
    generated_behavior_quantile = (
        behavior_result["generated_behavior"]
        if target_behavior_key == "behavior_quantile_score_selected_agents"
        else behavior_score_from_raw_tensor(generated_behavior_raw, behavior_normalizer, "behavior_quantile_score_selected_agents")
    )
    generated_behavior_aggressiveness = behavior_score_from_raw_tensor(
        generated_behavior_raw, behavior_normalizer, "behavior_aggressiveness_score_selected_agents"
    )
    total = (
        float(config["losses"].get("interaction_field_weight", 0.0)) * interaction_result["loss"]
        + float(config["losses"].get("anchor_weight", 1.0)) * anchor_result["loss"]
        + float(config["losses"].get("supervised_control_weight", 1.0)) * supervised_control_term
        + float(config["losses"].get("supervised_rollout_weight", 1.0)) * supervised_rollout_term
        + float(config["losses"].get("residual_diffusion_weight", 1.0)) * residual_diffusion_result["loss"]
        + float(config["losses"].get("anchor_residual_diffusion_weight", 1.0)) * anchor_residual_result["loss"]
        + float(config["losses"]["behavior_control_weight"]) * behavior_result["loss"]
        + float(config["losses"].get("control_delta_weight", 0.0)) * control_delta_result["loss"]
        + realism_result["weighted"]
    )
    target_behavior = target_behavior_quantile.detach().float()
    target_behavior_low = (target_behavior < 0.333).float().sum().detach()
    target_behavior_mid = ((target_behavior >= 0.333) & (target_behavior < 0.667)).float().sum().detach()
    target_behavior_high = (target_behavior >= 0.667).float().sum().detach()
    behavior_label_mae = torch.as_tensor(
        0.0 if behavior_label_debug is None else behavior_label_debug["behavior_label_mae"],
        dtype=torch.float32,
        device=batch["target_behavior"].device,
    )
    debug_payload = {
        "main_training_losses": active_training_losses(config),
        "generator_condition_mode": condition_mode,
        "behavior_target_key": target_behavior_key,
        "generator_condition_behavior_key": generator_condition_behavior_key,
        "behavior_label_debug": behavior_label_debug,
        "realism_disabled_reason": realism_result["disabled_reason"],
    }
    generated_vs_retrieved_behavior_mae = torch.abs(generated_behavior_quantile - target_behavior_quantile).mean().detach()
    generated_vs_gt_behavior_mae = generated_vs_retrieved_behavior_mae
    result = {
        "total": total,
        "interaction_field": interaction_result["loss"].detach(),
        "interaction_field_min_distance_mae": interaction_result["min_distance_mae"],
        "interaction_field_conflict_mae": interaction_result["conflict_mae"],
        "interaction_field_distance_change_mae": interaction_result["distance_change_mae"],
        "interaction_field_final_distance_mae": interaction_result["final_distance_mae"],
        "interaction_field_pair_latent_std": interaction_result["pair_latent_std"],
        "anchor": anchor_result["loss"].detach(),
        "anchor_mid_mae": anchor_result["mid_mae"],
        "anchor_final_mae": anchor_result["final_mae"],
        "anchor_heading_mae": anchor_result["heading_mae"],
        "supervised_control": supervised_control_term.detach(),
        "supervised_rollout": supervised_rollout_term.detach(),
        "oracle_knot_rollout": oracle_knot_rollout_term.detach(),
        "mu_knot_mae": mu_knot_mae.detach(),
        "mu_knot_abs_p95": mu_knot_abs_p95.detach(),
        "mu_knot_abs_p99": mu_knot_abs_p99.detach(),
        "mu_rollout_gap_to_oracle": mu_rollout_gap_to_oracle.detach(),
        "residual_diffusion": residual_diffusion_result["loss"].detach(),
        "residual_diffusion_unweighted": residual_diffusion_result["unweighted"].detach(),
        "residual_target_std": residual_diffusion_result["target_std"].detach(),
        "anchor_residual_diffusion": anchor_residual_result["loss"].detach(),
        "anchor_residual_diffusion_unweighted": anchor_residual_result["unweighted"].detach(),
        "anchor_residual_target_std": anchor_residual_result["target_std"].detach(),
        "residual_knot_std": residual_knot_std.detach(),
        "sampled_residual_std": sampled_residual_std.detach(),
        "final_knot_std": final_knot_std.detach(),
        "gt_knot_std": gt_knot_std.detach(),
        "residual_clip_fraction": residual_clip_fraction.detach(),
        "residual_energy_ratio": residual_energy_ratio.detach(),
        "anchor_residual_std": anchor_residual_std.detach(),
        "sampled_anchor_residual_std": sampled_anchor_residual_std.detach(),
        "anchor_mu_std": anchor_mu_std.detach(),
        "anchor_target_std": anchor_target_std.detach(),
        "final_anchor_std": final_anchor_std.detach(),
        "control_delta": control_delta_result["loss"].detach(),
        "realism_enabled": realism_result["enabled"].detach(),
        "realism_unweighted": realism_result["loss"].detach(),
        "realism_weight": realism_result["weight"].detach(),
        "realism_weighted": realism_result["weighted"].detach(),
        "realism_feature_dim": realism_result["feature_dim"].detach(),
        "realism_valid_count": realism_result["valid_count"].detach(),
        "realism_pred_feature_std": realism_result["pred_feature_std"],
        "realism_gt_feature_std": realism_result["gt_feature_std"],
        "sample_path_behavior_mae": zero,
        "sample_path_behavior_bias": zero,
        "sample_path_motion_guardrail": zero,
        "sample_path_active_fraction": zero,
        "generated_behavior_quantile_mean": generated_behavior_quantile.mean().detach(),
        "generated_behavior_quantile_std": generated_behavior_quantile.std(unbiased=False).detach(),
        "generated_behavior_quantile_min": generated_behavior_quantile.min().detach(),
        "generated_behavior_quantile_max": generated_behavior_quantile.max().detach(),
        "generated_behavior_aggressiveness_mean": generated_behavior_aggressiveness.mean().detach(),
        "generated_behavior_aggressiveness_std": generated_behavior_aggressiveness.std(unbiased=False).detach(),
        "generated_behavior_raw_score_mean": generated_behavior_raw.mean().detach(),
        "generated_behavior_raw_score_std": generated_behavior_raw.std(unbiased=False).detach(),
        "retrieved_behavior_quantile_mean": target_behavior_quantile.mean().detach(),
        "retrieved_behavior_quantile_std": target_behavior_quantile.std(unbiased=False).detach(),
        "gt_behavior_quantile_mean": target_behavior_quantile.mean().detach(),
        "gt_behavior_quantile_std": target_behavior_quantile.std(unbiased=False).detach(),
        "target_behavior_quantile_mean": target_behavior_quantile.mean().detach(),
        "target_behavior_quantile_std": target_behavior_quantile.std(unbiased=False).detach(),
        "target_behavior_quantile_min": target_behavior_quantile.min().detach(),
        "target_behavior_quantile_p05": _tensor_percentile(target_behavior_quantile, 0.05).to(target_behavior.device),
        "target_behavior_quantile_p25": _tensor_percentile(target_behavior_quantile, 0.25).to(target_behavior.device),
        "target_behavior_quantile_p50": _tensor_percentile(target_behavior_quantile, 0.50).to(target_behavior.device),
        "target_behavior_quantile_p75": _tensor_percentile(target_behavior_quantile, 0.75).to(target_behavior.device),
        "target_behavior_quantile_p95": _tensor_percentile(target_behavior_quantile, 0.95).to(target_behavior.device),
        "target_behavior_quantile_max": target_behavior_quantile.max().detach(),
        "target_behavior_aggressiveness_mean": target_behavior_aggressiveness.mean().detach(),
        "target_behavior_aggressiveness_std": target_behavior_aggressiveness.std(unbiased=False).detach(),
        "target_behavior_raw_score_mean": target_behavior_raw.mean().detach(),
        "target_behavior_raw_score_std": target_behavior_raw.std(unbiased=False).detach(),
        "target_behavior_mean": target_behavior_quantile.mean().detach(),
        "target_behavior_std": target_behavior_quantile.std(unbiased=False).detach(),
        "target_behavior_min": target_behavior.min().detach(),
        "target_behavior_p05": _tensor_percentile(target_behavior, 0.05).to(target_behavior.device),
        "target_behavior_p25": _tensor_percentile(target_behavior, 0.25).to(target_behavior.device),
        "target_behavior_p50": _tensor_percentile(target_behavior, 0.50).to(target_behavior.device),
        "target_behavior_p75": _tensor_percentile(target_behavior, 0.75).to(target_behavior.device),
        "target_behavior_p95": _tensor_percentile(target_behavior, 0.95).to(target_behavior.device),
        "target_behavior_max": target_behavior.max().detach(),
        "target_behavior_low_count": target_behavior_low,
        "target_behavior_mid_count": target_behavior_mid,
        "target_behavior_high_count": target_behavior_high,
        "generated_vs_retrieved_behavior_mae": generated_vs_retrieved_behavior_mae,
        "generated_vs_gt_behavior_mae": generated_vs_gt_behavior_mae,
        "behavior_label_mae": behavior_label_mae,
        "supervised_control_weight": torch.as_tensor(
            float(config["losses"].get("supervised_control_weight", 1.0)),
            dtype=torch.float32,
            device=batch["target_behavior"].device,
        ),
        "anchor_weight": torch.as_tensor(
            float(config["losses"].get("anchor_weight", 1.0)),
            dtype=torch.float32,
            device=batch["target_behavior"].device,
        ),
        "supervised_rollout_weight": torch.as_tensor(
            float(config["losses"].get("supervised_rollout_weight", 1.0)),
            dtype=torch.float32,
            device=batch["target_behavior"].device,
        ),
        "residual_diffusion_weight": torch.as_tensor(
            float(config["losses"].get("residual_diffusion_weight", 1.0)),
            dtype=torch.float32,
            device=batch["target_behavior"].device,
        ),
        "anchor_residual_diffusion_weight": torch.as_tensor(
            float(config["losses"].get("anchor_residual_diffusion_weight", 1.0)),
            dtype=torch.float32,
            device=batch["target_behavior"].device,
        ),
        "behavior_control_weight": torch.as_tensor(
            float(config["losses"]["behavior_control_weight"]),
            dtype=torch.float32,
            device=batch["target_behavior"].device,
        ),
        "behavior_control_loss_in_total": torch.as_tensor(
            1.0 if behavior_loss_enabled else 0.0,
            dtype=torch.float32,
            device=batch["target_behavior"].device,
        ),
        "control_delta_weight": torch.as_tensor(
            float(config["losses"].get("control_delta_weight", 0.0)),
            dtype=torch.float32,
            device=batch["target_behavior"].device,
        ),
        "sample_path_loss_weight": zero,
        "diffusion_weight": torch.as_tensor(
            0.0,
            dtype=torch.float32,
            device=batch["target_behavior"].device,
        ),
        "reconstruction_weight": torch.as_tensor(
            0.0,
            dtype=torch.float32,
            device=batch["target_behavior"].device,
        ),
        "reconstruction_low_noise_only": torch.as_tensor(
            0.0,
            dtype=torch.float32,
            device=batch["target_behavior"].device,
        ),
        "reconstruction_max_timestep_ratio": torch.as_tensor(
            0.0,
            dtype=torch.float32,
            device=batch["target_behavior"].device,
        ),
        "debug": debug_payload,
    }
    if generator_consumes_behavior(config):
        result["generator_condition_behavior_mean"] = generator_condition_quantile.mean().detach()
        result["generator_condition_behavior_std"] = generator_condition_quantile.std(unbiased=False).detach()
    if behavior_loss_enabled:
        result["behavior_control"] = behavior_result["loss"].detach()
        result["behavior_control_mae_quantile"] = torch.abs(
            behavior_result["generated_behavior"] - batch["target_behavior"]
        ).mean().detach()
        result["behavior_control_mae"] = result["behavior_control_mae_quantile"]
        result["behavior_loss_active_batch_fraction"] = behavior_result["active_fraction"].detach()
    return result
