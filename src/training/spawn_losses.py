from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from isgen.semantics.interaction_features import extract_current_pair_features
from isgen.semantics.spawn_ordering import canonical_sort_scene, safe_norm
from isgen.semantics.spawn_plan import SUMMARY_FEATURE_NAMES, extract_spawn_plan_targets


def _wrap_angle(delta: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(delta), torch.cos(delta))


def _masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < pred.ndim:
        mask = mask.unsqueeze(-1)
    if mask.shape != pred.shape:
        mask = mask.expand_as(pred)
    if int(mask.sum().item()) <= 0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    diff = F.smooth_l1_loss(pred, target, reduction="none")
    diff = torch.where(mask, diff, torch.zeros_like(diff))
    denom = torch.clamp(mask.float().sum(), min=1.0)
    return diff.sum() / denom


def _weighted_summary_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    feature_weights: torch.Tensor,
) -> torch.Tensor:
    if pred.numel() <= 0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    weight = feature_weights.to(device=pred.device, dtype=pred.dtype).view(1, -1)
    diff = F.smooth_l1_loss(pred, target, reduction="none") * weight
    denom = torch.clamp(weight.sum() * float(pred.shape[0]), min=1e-6)
    return diff.sum() / denom


def _prototype_pair_structure_losses(
    generated: torch.Tensor,
    generated_mask: torch.Tensor,
    prototype_states: torch.Tensor | None,
    prototype_mask: torch.Tensor | None,
    map_radius_m: float,
    margin_m: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prototype_states is None or prototype_mask is None:
        zero = torch.zeros((), device=generated.device, dtype=generated.dtype)
        return zero, zero
    prototype_states = prototype_states.to(device=generated.device, dtype=generated.dtype)
    prototype_mask = prototype_mask.to(device=generated.device, dtype=torch.bool)
    generated_mask = generated_mask.to(device=generated.device, dtype=torch.bool)
    valid = generated_mask & prototype_mask
    pair_valid = valid.unsqueeze(-1) & valid.unsqueeze(-2)
    eye = torch.eye(generated.shape[1], device=generated.device, dtype=torch.bool).unsqueeze(0)
    pair_valid = pair_valid & ~eye
    if int(pair_valid.sum().item()) <= 0:
        zero = torch.zeros((), device=generated.device, dtype=generated.dtype)
        return zero, zero
    gen_dist = torch.cdist(generated[..., 0:2], generated[..., 0:2])
    proto_dist = torch.cdist(prototype_states[..., 0:2], prototype_states[..., 0:2])
    scale = max(float(map_radius_m), 1e-6)
    preservation = _masked_smooth_l1(gen_dist / scale, proto_dist / scale, pair_valid)
    compression = F.relu((proto_dist - gen_dist - float(margin_m)) / scale)
    compression = torch.where(pair_valid, compression, torch.zeros_like(compression))
    compression_loss = compression.sum() / torch.clamp(pair_valid.float().sum(), min=1.0)
    return preservation, compression_loss


def _pair_distance_floor_loss(
    pred_distance: torch.Tensor,
    pair_valid: torch.Tensor,
    map_radius_m: float,
    floor_m: float,
) -> torch.Tensor:
    if int(pair_valid.sum().item()) <= 0:
        return torch.zeros((), device=pred_distance.device, dtype=pred_distance.dtype)
    floor = float(floor_m)
    scale = max(float(map_radius_m), 1e-6)
    penalty = F.relu((floor - pred_distance) / scale)
    penalty = torch.where(pair_valid, penalty, torch.zeros_like(penalty))
    return penalty.sum() / torch.clamp(pair_valid.float().sum(), min=1.0)


def _nearest_pair_distance_loss(
    pred_distance: torch.Tensor,
    gt_distance: torch.Tensor,
    pair_valid: torch.Tensor,
    map_radius_m: float,
) -> torch.Tensor:
    if int(pair_valid.sum().item()) <= 0:
        return torch.zeros((), device=pred_distance.device, dtype=pred_distance.dtype)
    sentinel = torch.full_like(pred_distance, 1e6)
    pred_nearest = torch.where(pair_valid, pred_distance, sentinel).amin(dim=-1)
    gt_nearest = torch.where(pair_valid, gt_distance, sentinel).amin(dim=-1)
    valid_agent = pred_nearest < 1e5
    return _masked_smooth_l1(
        pred_nearest / max(float(map_radius_m), 1e-6),
        gt_nearest / max(float(map_radius_m), 1e-6),
        valid_agent,
    )


def _excess_conflict_loss(
    pred_conflict: torch.Tensor,
    gt_conflict: torch.Tensor,
    pair_valid: torch.Tensor,
) -> torch.Tensor:
    if int(pair_valid.sum().item()) <= 0:
        return torch.zeros((), device=pred_conflict.device, dtype=pred_conflict.dtype)
    excess = F.relu(pred_conflict - gt_conflict)
    excess = torch.where(pair_valid, excess, torch.zeros_like(excess))
    return excess.sum() / torch.clamp(pair_valid.float().sum(), min=1.0)


def _map_alignment_loss(
    generated: torch.Tensor,
    generated_mask: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    map_radius_m: float,
    margin_m: float,
) -> torch.Tensor:
    map_polylines = batch.get("map_polylines")
    map_point_mask = batch.get("map_point_mask")
    map_polyline_mask = batch.get("map_polyline_mask")
    if map_polylines is None or map_point_mask is None or map_polyline_mask is None:
        return torch.zeros((), device=generated.device, dtype=generated.dtype)
    generated_mask = generated_mask.to(device=generated.device, dtype=torch.bool)
    if int(generated_mask.sum().item()) <= 0:
        return torch.zeros((), device=generated.device, dtype=generated.dtype)

    map_polylines = map_polylines.to(device=generated.device)
    map_point_mask = map_point_mask.to(device=generated.device, dtype=torch.bool)
    map_polyline_mask = map_polyline_mask.to(device=generated.device, dtype=torch.bool)
    map_valid = map_point_mask & map_polyline_mask.unsqueeze(-1)
    map_valid_flat = map_valid.reshape(map_valid.shape[0], -1)
    has_map = map_valid_flat.any(dim=-1)
    if int(has_map.sum().item()) <= 0:
        return torch.zeros((), device=generated.device, dtype=generated.dtype)

    agent_xy = torch.nan_to_num(generated[..., 0:2].float(), nan=0.0, posinf=0.0, neginf=0.0)
    map_xy = torch.nan_to_num(map_polylines[..., 0:2].float(), nan=0.0, posinf=0.0, neginf=0.0)
    map_xy = map_xy.reshape(map_xy.shape[0], -1, 2)
    distances = torch.cdist(agent_xy, map_xy)
    distances = torch.where(
        map_valid_flat.unsqueeze(1),
        distances,
        torch.full_like(distances, 1e6),
    )
    nearest = distances.amin(dim=-1)
    valid = generated_mask & has_map.unsqueeze(-1)
    if int(valid.sum().item()) <= 0:
        return torch.zeros((), device=generated.device, dtype=generated.dtype)

    scale = max(float(map_radius_m), 1e-6)
    excess = F.relu((nearest - float(margin_m)) / scale)
    excess = torch.where(valid, excess, torch.zeros_like(excess))
    loss = (excess * excess).sum() / torch.clamp(valid.float().sum(), min=1.0)
    return loss.to(dtype=generated.dtype)


def compute_spawn_loss(
    model_output: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    config: Dict,
    prototype_class_weight: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    loss_cfg = config.get("spawn_losses", {})
    spawn_cfg = config.get("spawn", {})
    data_cfg = config.get("data", {})
    map_radius_m = max(float(data_cfg.get("map_radius_m", 80.0)), 1e-6)
    max_speed_mps = max(float(spawn_cfg.get("max_speed_mps", 20.0)), 1e-6)
    generated = model_output["generated_current_states"]
    gt = batch["current_states"]
    gt_mask = batch["agent_mask"]
    presence_logits = model_output["generated_presence_logits"]
    presence_probs = model_output["generated_presence_probs"]
    count_logits = model_output["generated_count_logits"]
    count_probs = model_output["generated_count_probs"]
    predicted_count = model_output.get("predicted_count_head", model_output["generated_count"])
    expected_count = model_output["generated_expected_count"]
    predicted_plan = model_output["predicted_scene_plan"]
    prototype_prior_logits = model_output.get("prototype_prior_logits")
    prototype_target_ids = batch.get("prototype_target_ids")
    selected_prototype_count = model_output.get("selected_prototype_count")
    selected_prototype_plan = model_output.get("selected_prototype_plan_features")
    selected_prototype_states = model_output.get("selected_prototype_states")
    selected_prototype_mask = model_output.get("selected_prototype_mask")
    hybrid_train_proxy = bool(model_output.get("hybrid_train_proxy", False))
    prototype_cfg = config.get("spawn_prototypes", {})

    plan_targets = extract_spawn_plan_targets(
        current_states=gt,
        agent_mask=gt_mask,
        map_radius_m=float(data_cfg.get("map_radius_m", 80.0)),
        max_speed_mps=float(spawn_cfg.get("max_speed_mps", 20.0)),
        conflict_distance_m=float(loss_cfg.get("conflict_distance_m", 5.0)),
        radial_bins=int(spawn_cfg.get("plan_radial_bins", 4)),
        angular_bins=int(spawn_cfg.get("plan_angular_bins", 8)),
    )
    summary_dim = len(SUMMARY_FEATURE_NAMES)

    gt_count = plan_targets["count_targets"]
    gt_sorted, gt_sorted_mask, _ = canonical_sort_scene(gt, gt_mask)
    generated_mask = model_output["generated_agent_mask"].to(device=generated.device, dtype=torch.bool)
    matched_generated, matched_mask, matched_presence_logits = canonical_sort_scene(
        generated,
        generated_mask,
        presence_logits,
    )
    if matched_presence_logits is None:
        matched_presence_logits = presence_logits
    comparison_mask = matched_mask & gt_sorted_mask

    if hybrid_train_proxy:
        zero = torch.zeros((), device=generated.device, dtype=generated.dtype)
        position_loss = zero
        velocity_loss = zero
        heading_loss = zero
        presence_loss = zero
    else:
        position_loss = _masked_smooth_l1(
            matched_generated[..., 0:2] / map_radius_m,
            gt_sorted[..., 0:2] / map_radius_m,
            comparison_mask,
        )
        velocity_loss = _masked_smooth_l1(
            matched_generated[..., 2:4] / max_speed_mps,
            gt_sorted[..., 2:4] / max_speed_mps,
            comparison_mask,
        )
        heading_delta = _wrap_angle(matched_generated[..., 4] - gt_sorted[..., 4])
        heading_loss = _masked_smooth_l1(heading_delta, torch.zeros_like(heading_delta), comparison_mask)
        presence_loss = F.binary_cross_entropy_with_logits(matched_presence_logits, gt_sorted_mask.float())
    if selected_prototype_count is not None and str(config.get("spawn", {}).get("architecture", "")) == "support_guided_prototype":
        max_delta = int(prototype_cfg.get("count_residual_max_delta", 6))
        prototype_count_long = selected_prototype_count.round().to(device=generated.device, dtype=torch.long)
        residual_target = (gt_count - prototype_count_long).clamp(min=-max_delta, max=max_delta) + max_delta
        count_loss = F.cross_entropy(count_logits, residual_target)
    else:
        count_loss = F.cross_entropy(count_logits, gt_count.clamp(min=0, max=count_logits.shape[-1] - 1))
    count_consistency_loss = F.smooth_l1_loss(expected_count, gt_count.float())

    plan_summary_loss = F.smooth_l1_loss(predicted_plan[..., :summary_dim], plan_targets["summary_features"])
    plan_occupancy_loss = F.smooth_l1_loss(predicted_plan[..., summary_dim:], plan_targets["occupancy_features"])

    if hybrid_train_proxy:
        pair_distance_loss = zero
        soft_conflict_loss = zero
        nearest_pair_distance_loss = zero
        excess_conflict_loss = zero
        min_pair_distance_floor_loss = zero
        scene_summary_loss = zero
        pred_scene_plan = {
            "summary_features": predicted_plan[..., :summary_dim],
            "plan_features": predicted_plan,
        }
    else:
        pred_pair_features, pred_pair_valid = extract_current_pair_features(matched_generated, matched_mask)
        gt_pair_features, gt_pair_valid = extract_current_pair_features(gt_sorted, gt_sorted_mask)
        pair_valid = pred_pair_valid & gt_pair_valid
        pred_distance = pred_pair_features[..., 2]
        gt_distance = gt_pair_features[..., 2]
        pair_distance_loss = _masked_smooth_l1(pred_distance / map_radius_m, gt_distance / map_radius_m, pair_valid)

        conflict_distance = float(loss_cfg.get("conflict_distance_m", 5.0))
        conflict_temperature = float(loss_cfg.get("conflict_temperature_m", 1.0))
        pred_conflict = torch.sigmoid((conflict_distance - pred_distance) / max(conflict_temperature, 1e-3))
        gt_conflict = torch.sigmoid((conflict_distance - gt_distance) / max(conflict_temperature, 1e-3))
        soft_conflict_loss = _masked_smooth_l1(pred_conflict, gt_conflict, pair_valid)
        nearest_pair_distance_loss = _nearest_pair_distance_loss(
            pred_distance,
            gt_distance,
            pair_valid,
            map_radius_m,
        )
        excess_conflict_loss = _excess_conflict_loss(pred_conflict, gt_conflict, pair_valid)
        min_pair_distance_floor_loss = _pair_distance_floor_loss(
            pred_distance,
            pair_valid,
            map_radius_m,
            floor_m=float(loss_cfg.get("min_pair_distance_floor_m", 3.0)),
        )

        pred_scene_plan = extract_spawn_plan_targets(
            current_states=matched_generated,
            agent_mask=matched_mask,
            map_radius_m=float(data_cfg.get("map_radius_m", 80.0)),
            max_speed_mps=float(spawn_cfg.get("max_speed_mps", 20.0)),
            conflict_distance_m=float(loss_cfg.get("conflict_distance_m", 5.0)),
            radial_bins=int(spawn_cfg.get("plan_radial_bins", 4)),
            angular_bins=int(spawn_cfg.get("plan_angular_bins", 8)),
        )
        scene_summary_loss = F.smooth_l1_loss(pred_scene_plan["plan_features"], plan_targets["plan_features"])
    prototype_pair_preservation_loss, prototype_pair_compression_loss = _prototype_pair_structure_losses(
        generated=generated,
        generated_mask=generated_mask,
        prototype_states=selected_prototype_states,
        prototype_mask=selected_prototype_mask,
        map_radius_m=map_radius_m,
        margin_m=float(loss_cfg.get("prototype_pair_compression_margin_m", 0.75)),
    )
    map_alignment_weight = float(loss_cfg.get("map_alignment_weight", 0.0))
    if map_alignment_weight > 0.0 and not hybrid_train_proxy:
        map_alignment_loss = _map_alignment_loss(
            generated=generated,
            generated_mask=generated_mask,
            batch=batch,
            map_radius_m=map_radius_m,
            margin_m=float(loss_cfg.get("map_alignment_margin_m", 4.0)),
        )
    else:
        map_alignment_loss = torch.zeros((), device=generated.device, dtype=generated.dtype)
    relative_summary_weights_cfg = loss_cfg.get("prototype_relative_summary_feature_weights", {})
    relative_summary_weights = torch.tensor(
        [
            float(relative_summary_weights_cfg.get("mean_speed_norm", 2.0)),
            float(relative_summary_weights_cfg.get("mean_radius_norm", 1.5)),
            float(relative_summary_weights_cfg.get("mean_pair_distance_norm", 1.5)),
            float(relative_summary_weights_cfg.get("min_pairwise_distance_norm", 1.5)),
            float(relative_summary_weights_cfg.get("conflict_ratio", 2.0)),
            float(relative_summary_weights_cfg.get("heading_std", 0.5)),
        ],
        device=generated.device,
        dtype=generated.dtype,
    )
    if selected_prototype_plan is not None:
        prototype_summary = selected_prototype_plan[..., :summary_dim].to(device=generated.device, dtype=generated.dtype)
        target_relative_summary = plan_targets["summary_features"] - prototype_summary
        pred_relative_summary = pred_scene_plan["summary_features"] - prototype_summary
        relative_summary_loss = _weighted_summary_smooth_l1(
            pred_relative_summary,
            target_relative_summary,
            relative_summary_weights,
        )
    else:
        relative_summary_loss = torch.zeros((), device=generated.device, dtype=generated.dtype)
    if prototype_prior_logits is not None and prototype_target_ids is not None:
        prototype_target_ids = prototype_target_ids.to(device=generated.device, dtype=torch.long)
        class_weight = (
            prototype_class_weight.to(device=generated.device, dtype=generated.dtype)
            if prototype_class_weight is not None
            else None
        )
        prototype_loss = F.cross_entropy(prototype_prior_logits, prototype_target_ids, weight=class_weight)
        prototype_acc = (
            prototype_prior_logits.argmax(dim=-1).eq(prototype_target_ids).float().mean().detach()
        )
    else:
        prototype_loss = torch.zeros((), device=generated.device, dtype=generated.dtype)
        prototype_acc = torch.zeros((), device=generated.device, dtype=generated.dtype)

    if hybrid_train_proxy:
        total = (
            float(loss_cfg.get("count_weight", 1.0)) * count_loss
            + float(loss_cfg.get("prototype_relative_summary_weight", 0.6)) * relative_summary_loss
            + float(loss_cfg.get("count_consistency_weight", 0.1)) * count_consistency_loss
            + float(loss_cfg.get("plan_summary_weight", 0.7)) * plan_summary_loss
            + float(loss_cfg.get("plan_occupancy_weight", 1.0)) * plan_occupancy_loss
            + float(loss_cfg.get("prototype_weight", 1.0)) * prototype_loss
        )
    else:
        total = (
            float(loss_cfg.get("presence_weight", 1.0)) * presence_loss
            + float(loss_cfg.get("count_weight", 1.0)) * count_loss
            + float(loss_cfg.get("position_weight", 1.0)) * position_loss
            + float(loss_cfg.get("velocity_weight", 0.3)) * velocity_loss
            + float(loss_cfg.get("heading_weight", 0.1)) * heading_loss
            + float(loss_cfg.get("pair_distance_weight", 0.2)) * pair_distance_loss
            + float(loss_cfg.get("soft_conflict_weight", 0.2)) * soft_conflict_loss
            + float(loss_cfg.get("nearest_pair_distance_weight", 0.0)) * nearest_pair_distance_loss
            + float(loss_cfg.get("excess_conflict_weight", 0.0)) * excess_conflict_loss
            + float(loss_cfg.get("min_pair_distance_floor_weight", 0.0)) * min_pair_distance_floor_loss
            + float(loss_cfg.get("prototype_pair_preservation_weight", 0.0)) * prototype_pair_preservation_loss
            + float(loss_cfg.get("prototype_pair_compression_weight", 0.0)) * prototype_pair_compression_loss
            + map_alignment_weight * map_alignment_loss
            + float(loss_cfg.get("scene_summary_weight", 0.3)) * scene_summary_loss
            + float(loss_cfg.get("prototype_relative_summary_weight", 0.6)) * relative_summary_loss
            + float(loss_cfg.get("count_consistency_weight", 0.1)) * count_consistency_loss
            + float(loss_cfg.get("plan_summary_weight", 0.7)) * plan_summary_loss
            + float(loss_cfg.get("plan_occupancy_weight", 1.0)) * plan_occupancy_loss
            + float(loss_cfg.get("prototype_weight", 1.0)) * prototype_loss
        )

    count_mae = torch.abs(expected_count - gt_count.float()).mean().detach()
    hard_count_mae = torch.abs(predicted_count - gt_count.float()).mean().detach()
    if hybrid_train_proxy:
        min_pairwise_mae = torch.zeros((), device=generated.device, dtype=generated.dtype)
    else:
        min_pairwise_pred = torch.where(
            pair_valid,
            pred_distance,
            torch.full_like(pred_distance, 1e3),
        ).amin(dim=(-1, -2))
        min_pairwise_gt = torch.where(
            pair_valid,
            gt_distance,
            torch.full_like(gt_distance, 1e3),
        ).amin(dim=(-1, -2))
        min_pairwise_mae = torch.abs(min_pairwise_pred - min_pairwise_gt).mean().detach()

    return {
        "total": total,
        "spawn_presence": presence_loss.detach(),
        "spawn_count": count_loss.detach(),
        "spawn_count_consistency": count_consistency_loss.detach(),
        "spawn_plan_summary": plan_summary_loss.detach(),
        "spawn_plan_occupancy": plan_occupancy_loss.detach(),
        "spawn_position": position_loss.detach(),
        "spawn_velocity": velocity_loss.detach(),
        "spawn_heading": heading_loss.detach(),
        "spawn_pair_distance": pair_distance_loss.detach(),
        "spawn_soft_conflict": soft_conflict_loss.detach(),
        "spawn_nearest_pair_distance": nearest_pair_distance_loss.detach(),
        "spawn_excess_conflict": excess_conflict_loss.detach(),
        "spawn_min_pair_distance_floor": min_pair_distance_floor_loss.detach(),
        "spawn_prototype_pair_preservation": prototype_pair_preservation_loss.detach(),
        "spawn_prototype_pair_compression": prototype_pair_compression_loss.detach(),
        "spawn_map_alignment": map_alignment_loss.detach(),
        "spawn_scene_summary": scene_summary_loss.detach(),
        "spawn_relative_summary": relative_summary_loss.detach(),
        "spawn_prototype": prototype_loss.detach(),
        "count_mae": count_mae,
        "hard_count_mae": hard_count_mae,
        "min_pairwise_distance_mae": min_pairwise_mae,
        "prototype_id_acc": prototype_acc,
    }
