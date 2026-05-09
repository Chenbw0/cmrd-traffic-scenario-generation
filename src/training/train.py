from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from isgen import load_json, resolve_output_path, save_json, resolve_device, seed_everything
from isgen.data.dataset import ScenarioSliceDataset, resolve_dataset_history_settings
from isgen.models.scenario_generator import DifficultyConditionedScenarioDiffusion
from isgen.semantics.behavior import (
    BehaviorNormalizer,
    behavior_target_key_flags,
    load_behavior_normalizer,
    resolve_generated_score_space,
    resolve_generator_condition_score_key,
    resolve_behavior_target_key,
)
from isgen.semantics.control_normalization import ControlNormalizer, load_control_normalizer, summarize_control_tensor
from isgen.semantics.residual_normalization import ResidualNormalizer, load_residual_normalizer, save_residual_stats
from isgen.semantics.generator_condition import (
    behavior_control_loss_in_total,
    generator_consumes_behavior,
    generator_consumes_difficulty,
    resolve_generator_condition_mode,
    resolve_main_training_objective,
)
from isgen.training.checkpointing import load_checkpoint, save_checkpoint, save_history
from isgen.training.logging_utils import summarize_metrics
from isgen.training.losses import active_training_losses, compute_total_loss
from isgen.training.runtime_audit import capture_runtime_sources, inspect_runtime_state

LOGGER = logging.getLogger(__name__)

DIAGNOSTIC_ONLY_METRICS = {
    "oracle_knot_rollout",
    "mu_knot_mae",
    "mu_knot_abs_p95",
    "mu_knot_abs_p99",
    "mu_rollout_gap_to_oracle",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_hash(config: Dict) -> str:
    return hashlib.sha1(json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    moved = {}
    non_blocking = device.type == "cuda"
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=non_blocking)
        else:
            moved[key] = value
    return moved


def build_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    cache_dir = config["data"]["cache_dir"]
    overfit_one_batch = bool(config["training"].get("overfit_one_batch", False))
    num_workers = int(config["training"]["num_workers"])
    target_score_key = resolve_behavior_target_key(config)
    if os.name == "nt" or overfit_one_batch:
        num_workers = 0
    pin_memory = str(config["training"].get("device", "auto")).lower() != "cpu"
    persistent_workers = num_workers > 0
    prefetch_factor = int(config["training"].get("dataloader_prefetch_factor", 2))
    pin_memory_device = str(config["training"].get("dataloader_pin_memory_device", "auto")).lower()
    history_mode, recent_history_frames, history_dropout_prob, use_history = resolve_dataset_history_settings(config)
    train_dataset = ScenarioSliceDataset(
        cache_dir=cache_dir,
        split="train",
        history_mode=history_mode,
        recent_history_frames=recent_history_frames,
        history_dropout_prob=history_dropout_prob,
        training=True,
        target_score_key=target_score_key,
        use_history=use_history,
    )
    val_dataset = ScenarioSliceDataset(
        cache_dir=cache_dir,
        split="val",
        history_mode=history_mode,
        recent_history_frames=recent_history_frames,
        history_dropout_prob=0.0,
        training=False,
        target_score_key=target_score_key,
        use_history=use_history,
    )
    loader_kwargs: Dict[str, Any] = {
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = max(prefetch_factor, 2)
    if pin_memory and pin_memory_device not in {"", "auto"}:
        loader_kwargs["pin_memory_device"] = pin_memory_device
    train_loader = DataLoader(
        train_dataset,
        shuffle=not overfit_one_batch,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader


def resolve_scheduler_config(config: Dict) -> Dict[str, Any]:
    scheduler = dict(config.get("training", {}).get("scheduler", {}) or {})
    scheduler_type = str(scheduler.get("type", "none")).lower()
    scheduler["type"] = scheduler_type
    scheduler["min_lr_ratio"] = float(scheduler.get("min_lr_ratio", 0.1))
    if scheduler["min_lr_ratio"] <= 0.0 or scheduler["min_lr_ratio"] > 1.0:
        raise ValueError(f"training.scheduler.min_lr_ratio must be in (0, 1], got {scheduler['min_lr_ratio']}")
    return scheduler


def apply_epoch_lr_schedule(optimizer: AdamW, config: Dict, epoch: int) -> float:
    base_lr = float(config["training"]["lr"])
    scheduler = resolve_scheduler_config(config)
    schedule_type = str(scheduler.get("type", "none")).lower()
    total_epochs = max(1, int(config["training"]["num_epochs"]))
    if schedule_type == "none":
        lr = base_lr
    elif schedule_type == "cosine":
        progress = 0.0 if total_epochs <= 1 else min(max(epoch, 0), total_epochs - 1) / float(total_epochs - 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_lr_ratio = float(scheduler.get("min_lr_ratio", 0.1))
        lr = base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)
    else:
        raise ValueError(f"Unsupported training.scheduler.type: {schedule_type}")
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def resolve_amp_config(config: Dict, device: torch.device) -> Dict[str, Any]:
    training_cfg = config.get("training", {})
    amp_cfg = dict(training_cfg.get("amp", {}) or {})
    enabled = bool(amp_cfg.get("enabled", False)) and device.type == "cuda"
    dtype_name = str(amp_cfg.get("dtype", "bf16")).lower()
    if dtype_name not in {"bf16", "fp16"}:
        raise ValueError(f"Unsupported training.amp.dtype: {dtype_name}")
    autocast_dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16
    use_grad_scaler = enabled and dtype_name == "fp16"
    return {
        "enabled": enabled,
        "dtype_name": dtype_name,
        "autocast_dtype": autocast_dtype,
        "use_grad_scaler": use_grad_scaler,
    }


def configure_runtime_performance(config: Dict, device: torch.device) -> Dict[str, Any]:
    training_cfg = config.get("training", {})
    allow_tf32 = bool(training_cfg.get("allow_tf32", True)) and device.type == "cuda"
    cudnn_benchmark = bool(training_cfg.get("cudnn_benchmark", True)) and device.type == "cuda"
    matmul_precision = str(training_cfg.get("float32_matmul_precision", "high")).lower()
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cudnn.benchmark = cudnn_benchmark
    if hasattr(torch, "set_float32_matmul_precision") and matmul_precision in {"high", "medium", "highest"}:
        torch.set_float32_matmul_precision(matmul_precision)
    return {
        "allow_tf32": allow_tf32,
        "cudnn_benchmark": cudnn_benchmark,
        "float32_matmul_precision": matmul_precision,
    }


def _augment_targets(real_difficulty: torch.Tensor) -> torch.Tensor:
    noise = torch.empty_like(real_difficulty).uniform_(-0.15, 0.15)
    return torch.clamp(real_difficulty + noise, 0.0, 1.0)


def _prepare_training_batch(batch: Dict, training: bool, config: Dict) -> Dict:
    target_behavior_key = resolve_behavior_target_key(config)
    generator_condition_key = resolve_generator_condition_score_key(config)
    condition_mode = resolve_generator_condition_mode(config)
    if "target_behavior" not in batch:
        raise KeyError(
            f"Training batch is missing target_behavior derived from '{target_behavior_key}'. "
            "Behavior target fallback is disabled."
        )
    if generator_consumes_behavior(config) and generator_condition_key not in batch:
        raise KeyError(
            f"Training batch is missing generator condition field '{generator_condition_key}'. "
            "Generator condition fallback is disabled."
        )
    base_behavior = batch["target_behavior"]
    generator_condition_behavior = batch.get(generator_condition_key, base_behavior)
    target_behavior = base_behavior
    batch["target_difficulty"] = batch["difficulty_score_selected_agents"]
    batch["target_behavior"] = target_behavior
    batch["behavior_target_key"] = target_behavior_key
    batch["generator_condition_mode"] = condition_mode
    batch["generator_consumes_behavior"] = generator_consumes_behavior(config)
    batch["generator_consumes_difficulty"] = generator_consumes_difficulty(config)
    batch["generator_condition_behavior"] = generator_condition_behavior
    batch["generator_condition_behavior_key"] = generator_condition_key
    return batch


def _reduce_epoch_metrics(metrics: Iterable[Dict[str, float]]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    count = 0
    for item in metrics:
        count += 1
        for key, value in item.items():
            totals[key] = totals.get(key, 0.0) + float(value)
    return {key: value / max(count, 1) for key, value in totals.items()}


def _tensor_summary(value: torch.Tensor) -> Dict[str, float]:
    value = value.detach().float()
    if value.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(value.mean().item()),
        "std": float(value.std(unbiased=False).item()),
        "min": float(value.min().item()),
        "max": float(value.max().item()),
    }


def _collect_non_finite_gradients(model: torch.nn.Module) -> list[str]:
    names: list[str] = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            names.append(name)
    return names


def _collect_non_finite_parameters(model: torch.nn.Module) -> list[str]:
    names: list[str] = []
    for name, param in model.named_parameters():
        if not torch.isfinite(param).all():
            names.append(name)
    return names


def _write_step_history_record(handle: Any, record: Dict[str, Any], flush: bool) -> None:
    handle.write(json.dumps(_json_safe(record), sort_keys=True) + "\n")
    if flush:
        handle.flush()


def run_epoch(
    model: DifficultyConditionedScenarioDiffusion,
    loader: DataLoader,
    optimizer: AdamW | None,
    device: torch.device,
    behavior_normalizer: BehaviorNormalizer,
    config: Dict,
    amp_config: Dict[str, Any],
    grad_scaler: torch.cuda.amp.GradScaler | None,
    progress_desc: str | None = None,
    epoch: int | None = None,
    lr: float | None = None,
    step_history_path: Path | None = None,
) -> tuple[Dict[str, float], Dict[str, Any] | None]:
    training = optimizer is not None
    model.train(training)
    metrics = []
    first_batch_debug: Dict[str, Any] | None = None
    iterator = tqdm(loader, desc=progress_desc or ("train" if training else "val"), leave=False)
    max_batches = int(config["training"].get("debug_max_batches", 0))
    debug_first_batch_only = bool(config["training"].get("debug_first_batch_only", True))
    primary_metric_name = (
        "behavior_control_mae_quantile"
        if behavior_control_loss_in_total(config)
        else "supervised_rollout"
    )
    record_step_history = step_history_path is not None and bool(config["training"].get("record_step_history", True))
    step_history_handle = None
    if record_step_history:
        assert step_history_path is not None
        step_history_path.parent.mkdir(parents=True, exist_ok=True)
        step_history_handle = step_history_path.open("a", encoding="utf-8")
    try:
        for batch_idx, batch in enumerate(iterator):
            batch = move_batch_to_device(batch, device)
            batch = _prepare_training_batch(batch, training=training, config=config)
            autocast_context = (
                torch.autocast(device_type=device.type, dtype=amp_config["autocast_dtype"])
                if amp_config["enabled"]
                else nullcontext()
            )
            with autocast_context:
                model_output = model(
                    history_states=batch["history_states"],
                    history_mask=batch["history_mask"],
                    current_states=batch["current_states"],
                    future_states=batch["future_states"],
                    future_mask=batch["future_mask"],
                    agent_mask=batch["agent_mask"],
                    map_polylines=batch["map_polylines"],
                    map_point_mask=batch["map_point_mask"],
                    map_polyline_mask=batch["map_polyline_mask"],
                    target_difficulty=batch["target_difficulty"] if generator_consumes_difficulty(config) else None,
                    target_behavior=batch["generator_condition_behavior"] if generator_consumes_behavior(config) else None,
                )
                losses = compute_total_loss(
                    model_output,
                    batch,
                    behavior_normalizer,
                    config,
                    model=model,
                    training=training,
                    batch_idx=batch_idx,
                    compute_debug_metrics=(not debug_first_batch_only) or (first_batch_debug is None),
                )
            non_finite_loss_keys = []
            non_finite_diagnostic_keys = []
            for key, value in losses.items():
                if key == "debug" or not isinstance(value, torch.Tensor):
                    continue
                if not torch.isfinite(value).all():
                    if key.startswith("diagnostic_") or key in DIAGNOSTIC_ONLY_METRICS:
                        non_finite_diagnostic_keys.append(key)
                    else:
                        non_finite_loss_keys.append(key)
            if non_finite_diagnostic_keys:
                LOGGER.warning(
                    "Non-finite diagnostic-only metrics detected during %s batch %d: %s",
                    "train" if training else "val",
                    batch_idx,
                    ", ".join(sorted(non_finite_diagnostic_keys)),
                )
            if non_finite_loss_keys:
                raise RuntimeError(
                    f"Non-finite loss terms detected during {'train' if training else 'val'} batch {batch_idx}: "
                    + ", ".join(sorted(non_finite_loss_keys))
                )
            if first_batch_debug is None:
                first_batch_debug = dict(losses.get("debug", {}))
                first_batch_debug["model_runtime"] = dict(model_output.get("model_debug", {}))
                valid_mask = batch["future_mask"] & batch["agent_mask"].unsqueeze(-1)
                first_batch_debug["batch_validity"] = {
                    "future_valid_fraction": float(valid_mask.float().mean().item()),
                    "future_valid_count_mean": float(valid_mask.float().sum(dim=(1, 2)).mean().item()),
                    "future_valid_count_min": float(valid_mask.float().sum(dim=(1, 2)).min().item()),
                    "future_valid_count_max": float(valid_mask.float().sum(dim=(1, 2)).max().item()),
                    "agent_count_mean": float(batch["agent_mask"].float().sum(dim=1).mean().item()),
                    "agent_count_min": float(batch["agent_mask"].float().sum(dim=1).min().item()),
                    "agent_count_max": float(batch["agent_mask"].float().sum(dim=1).max().item()),
                }
                first_batch_debug["forward_tensor_stats"] = {
                    "target_encoded_future": _tensor_summary(model_output["target_encoded_future"]),
                    "noise": _tensor_summary(model_output["noise"]),
                    "predicted_noise": _tensor_summary(model_output["predicted_noise"]),
                    "pred_x0": _tensor_summary(model_output["pred_x0"]),
                    "decoded_future": _tensor_summary(model_output["decoded_future"]),
                    "gt_future": _tensor_summary(batch["future_states"]),
                    "decoded_future_mae": float(
                        torch.abs(model_output["decoded_future"] - batch["future_states"])[valid_mask].mean().item()
                    )
                    if valid_mask.any()
                    else 0.0,
                }
                if "target_raw_controls" in model_output:
                    first_batch_debug["gt_raw_control_stats"] = summarize_control_tensor(
                        model_output["target_raw_controls"],
                        valid_mask,
                    )
                    first_batch_debug["pred_x0_raw_control_stats"] = summarize_control_tensor(
                        model_output.get("pred_x0_raw_controls_clamped", model_output["target_raw_controls"]),
                        valid_mask,
                    )
                LOGGER.info(
                    "First batch %s validity future_valid_fraction=%.4f future_valid_count_mean=%.2f agent_count_mean=%.2f decoded_future_mae=%.4f",
                    "train" if training else "val",
                    first_batch_debug["batch_validity"]["future_valid_fraction"],
                    first_batch_debug["batch_validity"]["future_valid_count_mean"],
                    first_batch_debug["batch_validity"]["agent_count_mean"],
                    first_batch_debug["forward_tensor_stats"]["decoded_future_mae"],
                )
                LOGGER.info(
                    "First batch %s tensors encoded_target(mean=%.4f,std=%.4f) noise(mean=%.4f,std=%.4f) predicted_noise(mean=%.4f,std=%.4f) pred_x0(mean=%.4f,std=%.4f)",
                    "train" if training else "val",
                    first_batch_debug["forward_tensor_stats"]["target_encoded_future"]["mean"],
                    first_batch_debug["forward_tensor_stats"]["target_encoded_future"]["std"],
                    first_batch_debug["forward_tensor_stats"]["noise"]["mean"],
                    first_batch_debug["forward_tensor_stats"]["noise"]["std"],
                    first_batch_debug["forward_tensor_stats"]["predicted_noise"]["mean"],
                    first_batch_debug["forward_tensor_stats"]["predicted_noise"]["std"],
                    first_batch_debug["forward_tensor_stats"]["pred_x0"]["mean"],
                    first_batch_debug["forward_tensor_stats"]["pred_x0"]["std"],
                )
            if training:
                optimizer.zero_grad(set_to_none=True)
                if grad_scaler is not None:
                    grad_scaler.scale(losses["total"]).backward()
                    grad_scaler.unscale_(optimizer)
                else:
                    losses["total"].backward()
                non_finite_grad_names = _collect_non_finite_gradients(model)
                if non_finite_grad_names:
                    raise RuntimeError(
                        "Non-finite gradients detected after backward on parameters: "
                        + ", ".join(non_finite_grad_names[:10])
                        + (" ..." if len(non_finite_grad_names) > 10 else "")
                    )
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(config["training"]["grad_clip_norm"]))
                if grad_scaler is not None:
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    optimizer.step()
                non_finite_param_names = _collect_non_finite_parameters(model)
                if non_finite_param_names:
                    raise RuntimeError(
                        "Non-finite model parameters detected after optimizer.step() on parameters: "
                        + ", ".join(non_finite_param_names[:10])
                        + (" ..." if len(non_finite_param_names) > 10 else "")
                    )
            metric_record = {}
            for key, value in losses.items():
                if key == "debug":
                    continue
                metric_record[key] = float(value.item()) if isinstance(value, torch.Tensor) else float(value)
            metrics.append(metric_record)
            postfix = {"total": f"{metrics[-1]['total']:.4f}"}
            if primary_metric_name in metrics[-1]:
                postfix[primary_metric_name] = f"{metrics[-1][primary_metric_name]:.4f}"
            iterator.set_postfix(postfix)
            if step_history_handle is not None:
                num_batches = max(len(loader), 1)
                epoch_idx = int(epoch) if epoch is not None else 0
                step_record = {
                    "split": "train" if training else "val",
                    "epoch": epoch_idx,
                    "batch_idx": int(batch_idx),
                    "batch_count": int(num_batches),
                    "step_in_run": int(epoch_idx * num_batches + batch_idx + 1),
                    "epoch_fraction": float(epoch_idx + (batch_idx + 1) / num_batches),
                    "lr": float(lr) if lr is not None else None,
                    "batch_size": int(batch["current_states"].shape[0]) if "current_states" in batch else None,
                    **metric_record,
                }
                _write_step_history_record(
                    step_history_handle,
                    step_record,
                    flush=((batch_idx + 1) % int(config["training"].get("step_history_flush_every", 50)) == 0),
                )
            if training and bool(config["training"]["overfit_one_batch"]):
                break
            if max_batches > 0 and (batch_idx + 1) >= max_batches:
                break
    finally:
        if step_history_handle is not None:
            step_history_handle.close()
    return _reduce_epoch_metrics(metrics), first_batch_debug


def train_model(config: Dict, project_root: str | Path) -> Dict[str, float]:
    seed_everything(int(config["training"]["seed"]))
    device = resolve_device(config["training"]["device"])
    runtime_perf = configure_runtime_performance(config, device)
    amp_config = resolve_amp_config(config, device)
    cache_dir = Path(project_root) / config["data"]["cache_dir"]
    checkpoints_dir = resolve_output_path(config, project_root, "checkpoints_dir", Path("outputs") / "checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    latest_path = checkpoints_dir / "latest.pt"
    resume_requested = bool(config["training"].get("resume", True))
    behavior_target_key = resolve_behavior_target_key(config)
    generator_condition_score_key = resolve_generator_condition_score_key(config)
    generator_condition_mode = resolve_generator_condition_mode(config)
    behavior_target_flags = behavior_target_key_flags(behavior_target_key)
    train_loader, val_loader = build_dataloaders(config)
    behavior_normalizer = load_behavior_normalizer(cache_dir / "behavior_stats_selected_agents.json")
    control_stats_path = cache_dir / "control_stats.json"
    residual_stats_path = checkpoints_dir / "residual_stats.json"
    anchor_residual_stats_path = checkpoints_dir / "anchor_residual_stats.json"
    control_normalizer: ControlNormalizer | None = None
    if bool(config["model"].get("use_control_normalizer", True)):
        if not control_stats_path.exists():
            raise FileNotFoundError(f"Control normalizer is enabled but missing: {control_stats_path}")
        control_normalizer = load_control_normalizer(control_stats_path)
    model = DifficultyConditionedScenarioDiffusion(config).to(device)
    model.set_control_normalizer(control_normalizer)
    if residual_stats_path.exists():
        model.set_residual_normalizer(load_residual_normalizer(residual_stats_path))
    if anchor_residual_stats_path.exists():
        model.set_anchor_residual_normalizer(load_residual_normalizer(anchor_residual_stats_path))
    scenario_generator_path = Path(model.get_runtime_metadata()["runtime_source_file"])
    runtime_audit = inspect_runtime_state(
        config,
        checkpoint_path=latest_path if (resume_requested and latest_path.exists()) else None,
        snapshot_dir=checkpoints_dir / "runtime_source_snapshot",
        strict=True,
    )
    optimizer = AdamW(model.parameters(), lr=float(config["training"]["lr"]), weight_decay=float(config["training"]["weight_decay"]))
    grad_scaler = torch.cuda.amp.GradScaler(enabled=bool(amp_config["use_grad_scaler"]))
    scheduler_config = resolve_scheduler_config(config)
    start_epoch = 0
    best_val = float("inf")
    checkpoint_decoder_type = runtime_audit.get("checkpoint_decoder_type")
    if not resume_requested and latest_path.exists():
        LOGGER.info("resume=false; ignoring existing checkpoint at %s for this training run.", latest_path)
    if resume_requested and latest_path.exists():
        payload = load_checkpoint(latest_path)
        checkpoint_fingerprint = payload.get("run_metadata", {}).get("code_fingerprint")
        current_fingerprint = _file_sha256(scenario_generator_path)
        checkpoint_generator_condition_mode = payload.get("run_metadata", {}).get("generator_condition_mode")
        checkpoint_behavior_target_key = payload.get("run_metadata", {}).get("behavior_target_key")
        checkpoint_generator_condition_key = payload.get("run_metadata", {}).get("generator_condition_score_key")
        if checkpoint_fingerprint is not None and checkpoint_fingerprint != current_fingerprint:
            raise RuntimeError(
                "Checkpoint runtime source fingerprint does not match current scenario_generator.py. "
                f"checkpoint={checkpoint_fingerprint} current={current_fingerprint}"
            )
        if checkpoint_behavior_target_key is not None and checkpoint_behavior_target_key != behavior_target_key:
            raise RuntimeError(
                "Checkpoint behavior_target_key does not match current config. "
                f"checkpoint={checkpoint_behavior_target_key} current={behavior_target_key}"
            )
        if checkpoint_generator_condition_mode is not None and checkpoint_generator_condition_mode != generator_condition_mode:
            raise RuntimeError(
                "Checkpoint generator_condition_mode does not match current config. "
                f"checkpoint={checkpoint_generator_condition_mode} current={generator_condition_mode}"
            )
        if checkpoint_generator_condition_key is not None and checkpoint_generator_condition_key != generator_condition_score_key:
            raise RuntimeError(
                "Checkpoint generator_condition_score_key does not match current config. "
                f"checkpoint={checkpoint_generator_condition_key} current={generator_condition_score_key}"
            )
        try:
            model.load_state_dict(payload["model"])
            optimizer.load_state_dict(payload["optimizer"])
            checkpoint_residual_stats = payload.get("run_metadata", {}).get("residual_stats")
            if checkpoint_residual_stats is not None:
                model.set_residual_normalizer(
                    ResidualNormalizer(
                        channel_names=list(checkpoint_residual_stats.keys()),
                        channel_stats={key: dict(value) for key, value in checkpoint_residual_stats.items()},
                        enabled=True,
                    )
                )
            elif residual_stats_path.exists():
                model.set_residual_normalizer(load_residual_normalizer(residual_stats_path))
            checkpoint_anchor_residual_stats = payload.get("run_metadata", {}).get("anchor_residual_stats")
            if checkpoint_anchor_residual_stats is not None:
                model.set_anchor_residual_normalizer(
                    ResidualNormalizer(
                        channel_names=list(checkpoint_anchor_residual_stats.keys()),
                        channel_stats={key: dict(value) for key, value in checkpoint_anchor_residual_stats.items()},
                        enabled=True,
                    )
                )
            elif anchor_residual_stats_path.exists():
                model.set_anchor_residual_normalizer(load_residual_normalizer(anchor_residual_stats_path))
            start_epoch = int(payload["epoch"]) + 1
            best_val = float(payload.get("best_val", best_val))
            LOGGER.info("Resumed training from epoch %d.", start_epoch)
        except RuntimeError as error:
            LOGGER.warning("Skipping incompatible checkpoint resume from %s: %s", latest_path, error)
    run_metadata = {
        **runtime_audit,
        "decoder_type": model.decoder_type,
        "control_representation": model.control_representation,
        "num_control_knots": model.num_control_knots,
        "control_interpolation": model.control_interpolation,
        "use_anchor_latent": model.use_anchor_latent,
        "anchor_dim": model.anchor_dim,
        "anchor_channel_names": list(model.anchor_channel_names),
        "interaction_oracle_mode": model.interaction_oracle_mode,
        "interaction_oracle_feature_dim": model.interaction_oracle_feature_dim,
        "interaction_oracle_num_trace_points": model.interaction_oracle_num_trace_points,
        "interaction_oracle_conflict_distance_m": model.interaction_oracle_conflict_distance_m,
        "interaction_oracle_conflict_temperature_m": model.interaction_oracle_conflict_temperature_m,
        "learned_interaction_field_enabled": model.learned_interaction_field_enabled,
        "learned_interaction_field_target": model.learned_interaction_field_target,
        "learned_interaction_field_feature_dim": model.learned_interaction_feature_dim,
        "learned_interaction_field_latent_dim": model.learned_interaction_field_latent_dim,
        "learned_interaction_num_message_passing_steps": model.learned_interaction_num_message_passing_steps,
        "learned_interaction_field_condition_residual": model.learned_interaction_field_condition_residual,
        "residual_diffusion_enabled": model.residual_diffusion_enabled,
        "anchor_residual_diffusion_enabled": model.anchor_residual_diffusion_enabled,
        "normalize_residual": model.normalize_residual,
        "normalize_anchor_residual": model.normalize_anchor_residual,
        "residual_clip_std": model.residual_clip_std,
        "anchor_residual_clip_std": model.anchor_residual_clip_std,
        "residual_sample_scale": model.residual_sample_scale,
        "anchor_residual_sample_scale": model.anchor_residual_sample_scale,
        "residual_sample_steps": model.residual_sample_steps,
        "anchor_residual_sample_steps": model.anchor_residual_sample_steps,
        "encoder_type": model.encoder_type,
        "generator_input_use_history": model.use_history,
        "generator_input_history_mode": model.generator_input_history_mode,
        "require_history_at_sample": model.require_history_at_sample,
        "control_dim": model.control_dim,
        "local_future_dim": model.local_future_dim,
        "control_channel_names": list(model.control_channel_names),
        "control_normalizer_enabled": bool(model.use_control_normalizer),
        "control_normalizer_loaded": control_normalizer is not None,
        "control_stats_path": str(control_stats_path),
        "control_stats": control_normalizer.summary() if control_normalizer is not None else None,
        "residual_normalizer_loaded": model.residual_normalizer is not None,
        "residual_stats_path": str(residual_stats_path),
        "residual_stats": model.residual_stats_summary(),
        "anchor_residual_normalizer_loaded": model.anchor_residual_normalizer is not None,
        "anchor_residual_stats_path": str(anchor_residual_stats_path),
        "anchor_residual_stats": model.anchor_residual_stats_summary(),
        "sample_steps": int(config["sampling"]["sample_steps"]),
        "guidance_scale": float(config["sampling"]["guidance_scale"]),
        "sampler": str(config["sampling"].get("sampler", "ddpm")),
        "ddim_eta": float(config["sampling"].get("ddim_eta", 0.0)),
        "use_residual_diffusion": bool(config["sampling"].get("use_residual_diffusion", True)),
        "use_anchor_residual_diffusion": bool(config["sampling"].get("use_anchor_residual_diffusion", False)),
        "checkpoint_decoder_type": checkpoint_decoder_type,
        "checkpoint_exists": latest_path.exists(),
        "encoded_target_shape": None,
        "encoded_target_channel_names": list(model.control_channel_names) if model.decoder_type == "kinematic_controls" else ["dx", "dy", "dvx", "dvy", "dheading"],
        "guidance_semantics": "standard_cfg: prediction = uncond + guidance_scale * (cond - uncond); guidance_scale=1.0 equals the plain conditional prediction, guidance_scale=0.0 equals the unconditional prediction.",
        "runtime_source_file": str(scenario_generator_path),
        "source_sha256": _file_sha256(scenario_generator_path),
        "code_fingerprint": _file_sha256(scenario_generator_path),
        "config_hash": _config_hash(config),
        "checkpoint_path": str(latest_path),
        "active_training_losses": active_training_losses(config),
        "anchor_weight": float(config["losses"].get("anchor_weight", 1.0)),
        "interaction_field_weight": float(config["losses"].get("interaction_field_weight", 0.0)),
        "interaction_field_feature_weights": list(config["losses"].get("interaction_field_feature_weights", [2.0, 0.5, 0.5, 0.5, 0.5, 1.5, 2.0, 2.0])),
        "supervised_control_weight": float(config["losses"].get("supervised_control_weight", 1.0)),
        "supervised_rollout_weight": float(config["losses"].get("supervised_rollout_weight", 1.0)),
        "residual_diffusion_weight": float(config["losses"].get("residual_diffusion_weight", 1.0)),
        "anchor_residual_diffusion_weight": float(config["losses"].get("anchor_residual_diffusion_weight", 1.0)),
        "diffusion_target_type": str(config.get("residual_diffusion", {}).get("target_type", config["diffusion"].get("target_type", "epsilon"))),
        "anchor_diffusion_target_type": str(config.get("anchor_residual_diffusion", {}).get("target_type", "x0")),
        "timestep_sampling": str(config["diffusion"].get("timestep_sampling", "uniform")),
        "min_snr_gamma": float(config["diffusion"].get("min_snr_gamma", 0.0)),
        "behavior_control_weight": float(config["losses"].get("behavior_control_weight", 1.0)),
        "behavior_control_loss_in_total": behavior_control_loss_in_total(config),
        "control_delta_weight": float(config["losses"].get("control_delta_weight", 0.0)),
        "control_delta_jerk_weight": float(config["losses"].get("control_delta_jerk_weight", 1.0)),
        "control_delta_yaw_accel_weight": float(config["losses"].get("control_delta_yaw_accel_weight", 0.5)),
        "lr": float(config["training"]["lr"]),
        "amp_enabled": bool(amp_config["enabled"]),
        "amp_dtype": str(amp_config["dtype_name"]),
        "allow_tf32": bool(runtime_perf["allow_tf32"]),
        "cudnn_benchmark": bool(runtime_perf["cudnn_benchmark"]),
        "float32_matmul_precision": str(runtime_perf["float32_matmul_precision"]),
        "dataloader_prefetch_factor": int(config["training"].get("dataloader_prefetch_factor", 2)),
        "dataloader_pin_memory_device": str(config["training"].get("dataloader_pin_memory_device", "auto")),
        "scheduler_type": str(scheduler_config.get("type", "none")),
        "scheduler_min_lr_ratio": float(scheduler_config.get("min_lr_ratio", 0.1)),
        "realism_mmd_weight": float(config["losses"].get("realism_mmd_weight", 0.0)),
        "realism_disabled_reason": "weight_zero" if float(config["losses"].get("realism_mmd_weight", 0.0)) <= 0.0 else None,
        "sample_path_loss_enabled": False,
        "sample_path_loss_weight": 0.0,
        "sample_path_steps": 0,
        "sample_path_every_n_batches": 0,
        "sample_path_batch_fraction": 0.0,
        "main_training_objective": resolve_main_training_objective(config),
        "generator_condition_mode": generator_condition_mode,
        "generator_consumes_behavior": generator_consumes_behavior(config),
        "generator_consumes_difficulty": generator_consumes_difficulty(config),
        "behavior_target_key": behavior_target_key,
        "behavior_score_key_from_config": behavior_target_key,
        "batch_target_behavior_source": behavior_target_key,
        "generator_condition_score_key": generator_condition_score_key if generator_consumes_behavior(config) else None,
        "generated_behavior_score_space": resolve_generated_score_space(config),
        **behavior_target_flags,
    }
    behavior_distribution_report_path = cache_dir / "behavior_distribution_report.json"
    if behavior_distribution_report_path.exists():
        run_metadata["behavior_distribution_report"] = load_json(behavior_distribution_report_path)
        split_report = run_metadata["behavior_distribution_report"].get("splits", {})
        for split_name in ("train", "val", "test"):
            summary = split_report.get(split_name, {}).get(behavior_target_key, {})
            if summary:
                LOGGER.info(
                    "Behavior target distribution %s key=%s count=%s mean=%.4f std=%.4f min=%.4f p05=%.4f p50=%.4f p95=%.4f max=%.4f low=%s mid=%s high=%s",
                    split_name,
                    behavior_target_key,
                    summary.get("count", 0),
                    float(summary.get("mean", 0.0)),
                    float(summary.get("std", 0.0)),
                    float(summary.get("min", 0.0)),
                    float(summary.get("p05", 0.0)),
                    float(summary.get("p50", 0.0)),
                    float(summary.get("p95", 0.0)),
                    float(summary.get("max", 0.0)),
                    summary.get("low_count", 0),
                    summary.get("mid_count", 0),
                    summary.get("high_count", 0),
                )
                if (
                    behavior_target_key == "behavior_quantile_score_selected_agents"
                    and split_name == "train"
                    and float(summary.get("std", 0.0)) < 0.05
                ):
                    raise RuntimeError(
                        "Configured behavior.target_score_key is quantile, but the train target distribution is still very narrow. "
                        f"key={behavior_target_key} std={float(summary.get('std', 0.0)):.4f}"
                    )
    run_metadata["runtime_source_manifest"] = capture_runtime_sources(
        checkpoints_dir / "runtime_source_snapshot",
        phase="train",
        config_hash=run_metadata["config_hash"],
        checkpoint_hash=None,
    )
    run_metadata["runtime_code_fingerprints"] = run_metadata["runtime_source_manifest"]["files"]
    run_metadata["runtime_code_fingerprints"] = {
        key: value["sha256"] for key, value in run_metadata["runtime_code_fingerprints"].items()
    }
    save_json(run_metadata, checkpoints_dir / "run_metadata.json")
    LOGGER.info(
        "Runtime decoder=%s control_representation=%s knots=%d anchor_latent=%s interaction_oracle=%s learned_interaction=%s forward_path=%s sample_decode_path=%s channels=%s control_dim=%d control_norm=%s sample_steps=%d guidance=%.3f use_residual_diffusion=%s use_anchor_residual_diffusion=%s checkpoint_decoder=%s generator_condition_mode=%s consumes_behavior=%s consumes_difficulty=%s behavior_target_key=%s generator_condition_key=%s",
        model.decoder_type,
        model.control_representation,
        int(model.num_control_knots),
        bool(model.use_anchor_latent),
        str(model.interaction_oracle_mode),
        str(model.learned_interaction_field_enabled),
        run_metadata["forward_path"],
        run_metadata["sample_decode_path"],
        run_metadata["encoded_target_channel_names"],
        model.control_dim,
        "on" if control_normalizer is not None else "off",
        int(config["sampling"]["sample_steps"]),
        float(config["sampling"]["guidance_scale"]),
        bool(run_metadata["use_residual_diffusion"]),
        bool(run_metadata["use_anchor_residual_diffusion"]),
        checkpoint_decoder_type,
        generator_condition_mode,
        generator_consumes_behavior(config),
        generator_consumes_difficulty(config),
        behavior_target_key,
        run_metadata["generator_condition_score_key"],
    )
    LOGGER.info(
        "Performance amp=%s(dtype=%s) tf32=%s cudnn_benchmark=%s matmul_precision=%s prefetch_factor=%s pin_memory_device=%s",
        bool(run_metadata["amp_enabled"]),
        str(run_metadata["amp_dtype"]),
        bool(run_metadata["allow_tf32"]),
        bool(run_metadata["cudnn_benchmark"]),
        str(run_metadata["float32_matmul_precision"]),
        int(run_metadata["dataloader_prefetch_factor"]),
        str(run_metadata["dataloader_pin_memory_device"]),
    )
    LOGGER.info(
        "Loss weights interaction_field=%.4f anchor=%.4f supervised_control=%.4f supervised_rollout=%.4f residual_diffusion=%.4f(target=%s) anchor_residual_diffusion=%.4f(target=%s timestep_sampling=%s min_snr_gamma=%.2f) behavior_control=%.4f(in_total=%s) control_delta=%.4f realism=%.4f",
        float(run_metadata["interaction_field_weight"]),
        float(run_metadata["anchor_weight"]),
        float(run_metadata["supervised_control_weight"]),
        float(run_metadata["supervised_rollout_weight"]),
        float(run_metadata["residual_diffusion_weight"]),
        str(run_metadata["diffusion_target_type"]),
        float(run_metadata["anchor_residual_diffusion_weight"]),
        str(run_metadata["anchor_diffusion_target_type"]),
        str(run_metadata["timestep_sampling"]),
        float(run_metadata["min_snr_gamma"]),
        float(run_metadata["behavior_control_weight"]),
        bool(run_metadata["behavior_control_loss_in_total"]),
        float(run_metadata["control_delta_weight"]),
        float(run_metadata["realism_mmd_weight"]),
    )
    if float(run_metadata["realism_mmd_weight"]) <= 0.0:
        LOGGER.warning("Realism loss disabled_reason=weight_zero")
    history = []
    step_history_path = checkpoints_dir / "step_history.jsonl"
    if bool(config["training"].get("record_step_history", True)):
        run_metadata["step_history_path"] = str(step_history_path)
        run_metadata["step_history_frequency"] = "every_train_and_val_batch"
        run_metadata["step_history_flush_every"] = int(config["training"].get("step_history_flush_every", 50))
        if start_epoch == 0:
            step_history_path.write_text("", encoding="utf-8")
    if start_epoch >= int(config["training"]["num_epochs"]):
        LOGGER.warning(
            "Start epoch %d is already beyond configured num_epochs=%d; no training steps will run.",
            start_epoch,
            int(config["training"]["num_epochs"]),
        )
    for epoch in range(start_epoch, int(config["training"]["num_epochs"])):
        current_lr = apply_epoch_lr_schedule(optimizer, config, epoch)
        epoch_display = f"{epoch + 1}/{int(config['training']['num_epochs'])}"
        train_metrics, train_debug = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            behavior_normalizer,
            config,
            amp_config,
            grad_scaler,
            progress_desc=f"train {epoch_display} lr={current_lr:.2e}",
            epoch=epoch,
            lr=current_lr,
            step_history_path=step_history_path,
        )
        with torch.no_grad():
            val_metrics, val_debug = run_epoch(
                model,
                val_loader,
                None,
                device,
                behavior_normalizer,
                config,
                amp_config,
                None,
                progress_desc=f"val {epoch_display} lr={current_lr:.2e}",
                epoch=epoch,
                lr=current_lr,
                step_history_path=step_history_path,
            )
        record = {
            "epoch": epoch,
            "lr": current_lr,
            "train": train_metrics,
            "val": val_metrics,
            "train_debug_first_batch": train_debug,
            "val_debug_first_batch": val_debug,
            "run_metadata": run_metadata,
            "anchor_weight": run_metadata["anchor_weight"],
            "supervised_control_weight": run_metadata["supervised_control_weight"],
            "supervised_rollout_weight": run_metadata["supervised_rollout_weight"],
            "residual_diffusion_weight": run_metadata["residual_diffusion_weight"],
            "anchor_residual_diffusion_weight": run_metadata["anchor_residual_diffusion_weight"],
            "behavior_control_weight": run_metadata["behavior_control_weight"],
            "behavior_control_loss_in_total": run_metadata["behavior_control_loss_in_total"],
            "control_delta_weight": run_metadata["control_delta_weight"],
            "realism_mmd_weight": run_metadata["realism_mmd_weight"],
            "sample_path_loss_enabled": run_metadata["sample_path_loss_enabled"],
            "sample_path_loss_weight": run_metadata["sample_path_loss_weight"],
            "sample_path_steps": run_metadata["sample_path_steps"],
            "sample_path_every_n_batches": run_metadata["sample_path_every_n_batches"],
            "sample_path_batch_fraction": run_metadata["sample_path_batch_fraction"],
            "main_training_objective": run_metadata["main_training_objective"],
            "generator_condition_mode": run_metadata["generator_condition_mode"],
        }
        if isinstance(train_debug, dict):
            model_runtime = dict(train_debug.get("model_runtime", {}))
            if model_runtime:
                run_metadata["encoded_target_shape"] = model_runtime.get("encoded_target_shape")
                run_metadata["encoded_target_channel_names"] = model_runtime.get(
                    "encoded_target_channel_names",
                    run_metadata["encoded_target_channel_names"],
                )
                run_metadata["decoder_type"] = model_runtime.get("decoder_type", run_metadata["decoder_type"])
                run_metadata["control_dim"] = model_runtime.get("control_dim", run_metadata["control_dim"])
                run_metadata["local_future_dim"] = model_runtime.get("local_future_dim", run_metadata["local_future_dim"])
        run_metadata["residual_normalizer_loaded"] = model.residual_normalizer is not None
        run_metadata["residual_stats"] = model.residual_stats_summary()
        run_metadata["anchor_residual_normalizer_loaded"] = model.anchor_residual_normalizer is not None
        run_metadata["anchor_residual_stats"] = model.anchor_residual_stats_summary()
        run_metadata["current_lr"] = current_lr
        if model.residual_normalizer is not None:
            save_residual_stats(model.residual_normalizer, residual_stats_path)
        if model.anchor_residual_normalizer is not None:
            save_residual_stats(model.anchor_residual_normalizer, anchor_residual_stats_path)
        save_json(run_metadata, checkpoints_dir / "run_metadata.json")
        record_json = _json_safe(record)
        history.append(record_json)
        LOGGER.info("Epoch %s | lr %.6g", epoch_display, current_lr)
        LOGGER.info("Epoch %s | train %s", epoch_display, summarize_metrics(train_metrics))
        LOGGER.info("Epoch %s | val   %s", epoch_display, summarize_metrics(val_metrics))
        for split_name, split_metrics in (("train", train_metrics), ("val", val_metrics)):
            if (
                bool(run_metadata["behavior_control_loss_in_total"])
                and split_metrics.get("target_behavior_std", 1.0) < 0.05
            ):
                LOGGER.warning(
                    "Target behavior distribution is very narrow. behavior_control_mae is not sufficient evidence of controllability. "
                    "split=%s epoch=%d std=%.4f",
                    split_name,
                    epoch,
                    split_metrics.get("target_behavior_std", 0.0),
                )
            if split_metrics.get("realism_weight", 0.0) <= 0.0:
                LOGGER.info(
                    "Epoch %s | %s realism disabled_reason=weight_zero",
                    epoch_display,
                    split_name,
                )
            elif (
                abs(split_metrics.get("realism_unweighted", 0.0)) < 1e-8
                and abs(split_metrics.get("realism_pred_feature_std", 0.0) - split_metrics.get("realism_gt_feature_std", 0.0)) > 1e-4
            ):
                LOGGER.warning(
                    "Epoch %s | %s realism_unweighted is ~0 despite feature std mismatch pred=%.6f gt=%.6f",
                    epoch_display,
                    split_name,
                    split_metrics.get("realism_pred_feature_std", 0.0),
                    split_metrics.get("realism_gt_feature_std", 0.0),
                )
        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "run_metadata": run_metadata,
            "residual_stats": model.residual_stats_summary(),
            "anchor_residual_stats": model.anchor_residual_stats_summary(),
            "best_val": min(best_val, val_metrics["total"]),
        }
        save_checkpoint(checkpoints_dir / "latest.pt", payload)
        latest_checkpoint_hash = _file_sha256(checkpoints_dir / "latest.pt")
        run_metadata["checkpoint_file_sha256"] = latest_checkpoint_hash
        run_metadata["checkpoint_exists"] = True
        run_metadata["runtime_source_manifest"] = capture_runtime_sources(
            checkpoints_dir / "runtime_source_snapshot",
            phase="train",
            config_hash=run_metadata["config_hash"],
            checkpoint_hash=latest_checkpoint_hash,
        )
        run_metadata["runtime_code_fingerprints"] = {
            key: value["sha256"] for key, value in run_metadata["runtime_source_manifest"]["files"].items()
        }
        save_json(run_metadata, checkpoints_dir / "run_metadata.json")
        payload["run_metadata"] = run_metadata
        save_checkpoint(checkpoints_dir / "latest.pt", payload)
        save_history(checkpoints_dir / "history.json", history)
        save_json(record_json, checkpoints_dir / "latest_epoch.json")
        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            save_checkpoint(checkpoints_dir / "best.pt", payload)
            if model.residual_normalizer is not None:
                save_residual_stats(model.residual_normalizer, checkpoints_dir / "best_residual_stats.json")
            if model.anchor_residual_normalizer is not None:
                save_residual_stats(model.anchor_residual_normalizer, checkpoints_dir / "best_anchor_residual_stats.json")
    return {"best_val": best_val}
