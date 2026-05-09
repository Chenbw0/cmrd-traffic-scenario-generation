from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from isgen import resolve_output_path, resolve_device
from isgen.data.cache import load_cache_metadata, load_slices_from_cache
from isgen.data.slice_builder import materialize_cached_map_fields
from isgen.models.scenario_generator import DifficultyConditionedScenarioDiffusion
from isgen.retrieval.retriever import DifficultyRetriever, RetrievalCandidate
from isgen.retrieval.slice_index import SliceIndex
from isgen.retrieval.support import KNNSupportEstimator, SupportEstimator
from isgen.sampling.calibration import BehaviorCalibrator, load_behavior_calibrator
from isgen.semantics.behavior import (
    BehaviorNormalizer,
    behavior_score_bundle_from_features,
    behavior_score_bundle_from_raw,
    compute_behavior_aggressiveness_features,
    load_behavior_normalizer,
    resolve_behavior_target_key,
    resolve_generated_score_space,
    resolve_generator_condition_score_key,
)
from isgen.semantics.control_normalization import compute_raw_controls_numpy, load_control_normalizer
from isgen.semantics.residual_normalization import ResidualNormalizer, load_residual_normalizer
from isgen.semantics.difficulty import DifficultyNormalizer, compute_difficulty_features, load_difficulty_normalizer
from isgen.semantics.generator_condition import (
    generator_consumes_behavior,
    generator_consumes_difficulty,
    resolve_generator_condition_mode,
)
from isgen.training.checkpointing import load_checkpoint
from isgen.training.runtime_audit import RUNTIME_SOURCE_FILES, file_sha256


def _required_behavior_value(slice_item: Dict, key: str) -> float:
    if key not in slice_item:
        raise KeyError(
            f"Sampling requires cached behavior field '{key}', but it is missing for slice_id={slice_item.get('slice_id', '<unknown>')}."
        )
    return float(slice_item[key])


def _default_checkpoint_info(checkpoint_used: str | None = None, checkpoint_path: str | None = None) -> Dict[str, Any]:
    return {
        "checkpoint_used": checkpoint_used,
        "checkpoint_exists": False,
        "checkpoint_sha256": None,
        "checkpoint_epoch": None,
        "checkpoint_selection_metric": None,
        "checkpoint_path": checkpoint_path,
    }


def _generation_history_flags(slice_item: Dict, config: Dict) -> Dict[str, bool]:
    generator_input = config.get("generator_input", {})
    use_history = bool(generator_input.get("use_history", True))
    history_available = bool(np.asarray(slice_item.get("history_mask", []), dtype=bool).any())
    return {
        "generator_used_history": use_history,
        "history_available_in_retrieved_slice": history_available,
        "history_used_for_generation": use_history,
        "current_state_used": True,
    }


def _stack_batch(items: Sequence[Dict]) -> Dict[str, torch.Tensor]:
    tensor_keys = [
        "agent_ids",
        "agent_types",
        "agent_sizes",
        "history_states",
        "history_mask",
        "current_states",
        "future_states",
        "future_mask",
        "map_polylines",
        "map_point_mask",
        "map_polyline_mask",
        "agent_mask",
        "world_origin",
        "world_heading",
    ]
    batch = {}
    for key in tensor_keys:
        batch[key] = torch.stack([torch.as_tensor(item[key]) for item in items], dim=0)
    batch["slice_id"] = [item["slice_id"] for item in items]
    batch["location_id"] = [item["location_id"] for item in items]
    batch["scenario_id"] = [item["scenario_id"] for item in items]
    batch["difficulty_score_selected_agents"] = torch.as_tensor(
        [item.get("difficulty_score_selected_agents", item.get("difficulty_score", 0.0)) for item in items],
        dtype=torch.float32,
    )
    batch["difficulty_score_full_scene"] = torch.as_tensor(
        [item.get("difficulty_score_full_scene", item.get("difficulty_score", 0.0)) for item in items],
        dtype=torch.float32,
    )
    batch["behavior_aggressiveness_score_selected_agents"] = torch.as_tensor(
        [item.get("behavior_aggressiveness_score_selected_agents", item.get("difficulty_score_selected_agents", 0.0)) for item in items],
        dtype=torch.float32,
    )
    batch["behavior_quantile_score_selected_agents"] = torch.as_tensor(
        [item.get("behavior_quantile_score_selected_agents", item.get("behavior_aggressiveness_score_selected_agents", 0.0)) for item in items],
        dtype=torch.float32,
    )
    batch["behavior_raw_score_selected_agents"] = torch.as_tensor(
        [item.get("behavior_raw_score_selected_agents", 0.0) for item in items],
        dtype=torch.float32,
    )
    batch["difficulty_score"] = batch["difficulty_score_selected_agents"]
    batch["retrieval_embedding"] = torch.stack([torch.as_tensor(item.get("retrieval_embedding")) for item in items], dim=0).float()
    return batch


def _move_batch(batch: Dict, device: torch.device) -> Dict:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _resolve_checkpoint_path(
    config: Dict,
    project_root: str | Path,
    checkpoint: str | Path | None = None,
) -> tuple[Path, str, str]:
    checkpoints_dir = resolve_output_path(config, project_root, "checkpoints_dir", Path("outputs") / "checkpoints")
    request = str(checkpoint or "best.pt")
    lowered = request.lower()
    if lowered in {"best", "best.pt"}:
        return checkpoints_dir / "best.pt", "best.pt", "best_val_total"
    if lowered in {"latest", "latest.pt"}:
        return checkpoints_dir / "latest.pt", "latest.pt", "latest_epoch"
    candidate = Path(request)
    if not candidate.is_absolute():
        candidate = (Path(project_root) / candidate).resolve()
    return candidate, request, "custom_path"


def load_model_for_sampling(
    config: Dict,
    project_root: str | Path,
    checkpoint: str | Path | None = None,
    required_runtime_files: Sequence[str] | None = None,
) -> tuple[DifficultyConditionedScenarioDiffusion, Dict, Dict[str, Any]]:
    device = resolve_device(config["training"]["device"])
    model = DifficultyConditionedScenarioDiffusion(config).to(device)
    checkpoint_path, checkpoint_used, selection_metric = _resolve_checkpoint_path(config, project_root, checkpoint)
    checkpoint_exists = checkpoint_path.exists()
    if not checkpoint_exists:
        raise FileNotFoundError(f"Sampling checkpoint not found: {checkpoint_path}")
    checkpoint_payload = load_checkpoint(checkpoint_path)
    checkpoint_decoder_type = checkpoint_payload.get("config", {}).get("model", {}).get("decoder_type")
    config_decoder_type = config["model"].get("decoder_type")
    checkpoint_generator_condition_mode = checkpoint_payload.get("run_metadata", {}).get("generator_condition_mode")
    config_generator_condition_mode = resolve_generator_condition_mode(config)
    checkpoint_behavior_target_key = checkpoint_payload.get("run_metadata", {}).get("behavior_target_key")
    config_behavior_target_key = resolve_behavior_target_key(config)
    checkpoint_generator_condition_key = checkpoint_payload.get("run_metadata", {}).get("generator_condition_score_key")
    config_generator_condition_key = resolve_generator_condition_score_key(config)
    checkpoint_generated_score_space = checkpoint_payload.get("run_metadata", {}).get("generated_behavior_score_space")
    config_generated_score_space = resolve_generated_score_space(config)
    checkpoint_control_representation = checkpoint_payload.get("run_metadata", {}).get("control_representation")
    checkpoint_num_control_knots = checkpoint_payload.get("run_metadata", {}).get("num_control_knots")
    checkpoint_control_interpolation = checkpoint_payload.get("run_metadata", {}).get("control_interpolation")
    checkpoint_use_anchor_latent = checkpoint_payload.get("run_metadata", {}).get("use_anchor_latent")
    checkpoint_anchor_dim = checkpoint_payload.get("run_metadata", {}).get("anchor_dim")
    checkpoint_residual_diffusion_enabled = checkpoint_payload.get("run_metadata", {}).get("residual_diffusion_enabled")
    checkpoint_anchor_residual_diffusion_enabled = checkpoint_payload.get("run_metadata", {}).get("anchor_residual_diffusion_enabled")
    checkpoint_diffusion_target_type = checkpoint_payload.get("run_metadata", {}).get("diffusion_target_type")
    checkpoint_anchor_diffusion_target_type = checkpoint_payload.get("run_metadata", {}).get("anchor_diffusion_target_type")
    if checkpoint_decoder_type is not None and config_decoder_type != checkpoint_decoder_type:
        raise ValueError(
            f"Checkpoint decoder_type ({checkpoint_decoder_type}) does not match config decoder_type ({config_decoder_type})."
        )
    if checkpoint_generator_condition_mode is not None and checkpoint_generator_condition_mode != config_generator_condition_mode:
        raise ValueError(
            f"Checkpoint generator_condition_mode ({checkpoint_generator_condition_mode}) does not match config ({config_generator_condition_mode})."
        )
    if checkpoint_behavior_target_key is not None and checkpoint_behavior_target_key != config_behavior_target_key:
        raise ValueError(
            f"Checkpoint behavior_target_key ({checkpoint_behavior_target_key}) does not match config ({config_behavior_target_key})."
        )
    if checkpoint_generator_condition_key is not None and checkpoint_generator_condition_key != config_generator_condition_key:
        raise ValueError(
            f"Checkpoint generator_condition_score_key ({checkpoint_generator_condition_key}) does not match config ({config_generator_condition_key})."
        )
    if checkpoint_generated_score_space is not None and checkpoint_generated_score_space != config_generated_score_space:
        raise ValueError(
            f"Checkpoint generated_behavior_score_space ({checkpoint_generated_score_space}) does not match config ({config_generated_score_space})."
        )
    if checkpoint_control_representation is not None and checkpoint_control_representation != config["model"].get("control_representation"):
        raise ValueError(
            f"Checkpoint control_representation ({checkpoint_control_representation}) does not match config ({config['model'].get('control_representation')})."
        )
    if checkpoint_num_control_knots is not None and int(checkpoint_num_control_knots) != int(config["model"].get("num_control_knots", 0)):
        raise ValueError(
            f"Checkpoint num_control_knots ({checkpoint_num_control_knots}) does not match config ({config['model'].get('num_control_knots', 0)})."
        )
    if checkpoint_control_interpolation is not None and checkpoint_control_interpolation != config["model"].get("control_interpolation"):
        raise ValueError(
            f"Checkpoint control_interpolation ({checkpoint_control_interpolation}) does not match config ({config['model'].get('control_interpolation')})."
        )
    if checkpoint_use_anchor_latent is not None and bool(checkpoint_use_anchor_latent) != bool(config["model"].get("use_anchor_latent", False)):
        raise ValueError(
            f"Checkpoint use_anchor_latent ({checkpoint_use_anchor_latent}) does not match config ({config['model'].get('use_anchor_latent', False)})."
        )
    if checkpoint_anchor_dim is not None and int(checkpoint_anchor_dim) != int(config["model"].get("anchor_dim", 0)):
        raise ValueError(
            f"Checkpoint anchor_dim ({checkpoint_anchor_dim}) does not match config ({config['model'].get('anchor_dim', 0)})."
        )
    if checkpoint_residual_diffusion_enabled is not None and bool(checkpoint_residual_diffusion_enabled) != bool(config["model"].get("residual_diffusion_enabled", True)):
        raise ValueError(
            f"Checkpoint residual_diffusion_enabled ({checkpoint_residual_diffusion_enabled}) does not match config ({config['model'].get('residual_diffusion_enabled', True)})."
        )
    if checkpoint_anchor_residual_diffusion_enabled is not None and bool(checkpoint_anchor_residual_diffusion_enabled) != bool(config.get("anchor_residual_diffusion", {}).get("enabled", False)):
        raise ValueError(
            f"Checkpoint anchor_residual_diffusion_enabled ({checkpoint_anchor_residual_diffusion_enabled}) does not match config ({config.get('anchor_residual_diffusion', {}).get('enabled', False)})."
        )
    expected_diffusion_target_type = config.get("residual_diffusion", {}).get("target_type", config["diffusion"].get("target_type", "epsilon"))
    if checkpoint_diffusion_target_type is not None and checkpoint_diffusion_target_type != expected_diffusion_target_type:
        raise ValueError(
            f"Checkpoint diffusion_target_type ({checkpoint_diffusion_target_type}) does not match config ({expected_diffusion_target_type})."
        )
    expected_anchor_diffusion_target_type = config.get("anchor_residual_diffusion", {}).get("target_type", "x0")
    if checkpoint_anchor_diffusion_target_type is not None and checkpoint_anchor_diffusion_target_type != expected_anchor_diffusion_target_type:
        raise ValueError(
            f"Checkpoint anchor_diffusion_target_type ({checkpoint_anchor_diffusion_target_type}) does not match config ({expected_anchor_diffusion_target_type})."
        )
    checkpoint_fingerprint = checkpoint_payload.get("run_metadata", {}).get("code_fingerprint")
    runtime_source_file = Path(model.get_runtime_metadata()["runtime_source_file"])
    runtime_fingerprint = file_sha256(runtime_source_file) if runtime_source_file.exists() else None
    if checkpoint_fingerprint is not None and runtime_fingerprint is not None and checkpoint_fingerprint != runtime_fingerprint:
        raise RuntimeError(
            "Checkpoint runtime source fingerprint does not match current scenario_generator.py. "
            f"checkpoint={checkpoint_fingerprint} current={runtime_fingerprint}"
        )
    checkpoint_runtime_fingerprints = checkpoint_payload.get("run_metadata", {}).get("runtime_code_fingerprints", {})
    runtime_scope = tuple(required_runtime_files or RUNTIME_SOURCE_FILES.keys())
    for target_name in runtime_scope:
        source_path = RUNTIME_SOURCE_FILES[target_name]
        current_sha = file_sha256(source_path)
        expected_sha = checkpoint_runtime_fingerprints.get(target_name)
        if expected_sha is not None and expected_sha != current_sha:
            raise RuntimeError(
                f"Checkpoint/runtime source mismatch for {target_name}: checkpoint={expected_sha} current={current_sha}"
            )
    model.load_state_dict(checkpoint_payload["model"])
    control_stats_path = Path(project_root) / config["data"]["cache_dir"] / "control_stats.json"
    if bool(config["model"].get("use_control_normalizer", True)):
        if not control_stats_path.exists():
            raise FileNotFoundError(f"Control normalizer is enabled but missing: {control_stats_path}")
        model.set_control_normalizer(load_control_normalizer(control_stats_path))
    residual_cfg = config.get("residual_diffusion", {})
    if bool(residual_cfg.get("normalize_residual", True)):
        checkpoint_residual_stats = checkpoint_payload.get("residual_stats") or checkpoint_payload.get("run_metadata", {}).get("residual_stats")
        residual_stats_path = checkpoint_path.parent / "best_residual_stats.json"
        if not residual_stats_path.exists():
            residual_stats_path = checkpoint_path.parent / "residual_stats.json"
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
    anchor_residual_cfg = config.get("anchor_residual_diffusion", {})
    if bool(anchor_residual_cfg.get("normalize_residual", True)):
        checkpoint_anchor_residual_stats = checkpoint_payload.get("anchor_residual_stats") or checkpoint_payload.get("run_metadata", {}).get("anchor_residual_stats")
        anchor_residual_stats_path = checkpoint_path.parent / "best_anchor_residual_stats.json"
        if not anchor_residual_stats_path.exists():
            anchor_residual_stats_path = checkpoint_path.parent / "anchor_residual_stats.json"
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
    model.eval()
    checkpoint_info = {
        "checkpoint_used": checkpoint_used,
        "checkpoint_exists": True,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_epoch": checkpoint_payload.get("epoch"),
        "checkpoint_selection_metric": selection_metric,
        "checkpoint_path": str(checkpoint_path.resolve()),
    }
    return model, checkpoint_payload, checkpoint_info


def build_retriever(config: Dict, project_root: str | Path, split: str = "train") -> DifficultyRetriever:
    cache_dir = Path(project_root) / config["data"]["cache_dir"]
    slices = load_slices_from_cache(cache_dir, split, materialize_maps=False)
    slice_index = SliceIndex.load(cache_dir / "slice_index.pt")
    support_full_scene = SupportEstimator(
        knn_estimator=KNNSupportEstimator(
            index=slice_index,
            bandwidth=float(config["retrieval"]["support_bandwidth"]),
            neighbors=int(config["retrieval"]["knn_neighbors"]),
            difficulty_space="full_scene",
        )
    )
    support_selected_agents = SupportEstimator(
        knn_estimator=KNNSupportEstimator(
            index=slice_index,
            bandwidth=float(config["retrieval"]["support_bandwidth"]),
            neighbors=int(config["retrieval"]["knn_neighbors"]),
            difficulty_space="selected_agents",
        )
    )
    return DifficultyRetriever(
        slices=slices,
        slice_index=slice_index,
        support_estimator=support_full_scene,
        selected_support_estimator=support_selected_agents,
        config=config,
    )


def _materialize_sampling_slice_items(
    slice_items: Sequence[Dict],
    cache_dir: Path,
) -> List[Dict]:
    metadata = load_cache_metadata(cache_dir)
    data_root = metadata.get("data_root")
    max_polyline_points = int(metadata.get("map_materialization", {}).get("max_polyline_points", 20))
    if not data_root:
        return [dict(item) for item in slice_items]
    return [
        materialize_cached_map_fields(item, data_root, max_polyline_points)
        if "map_polylines" not in item and "map_polyline_indices" in item
        else item
        for item in slice_items
    ]


def _safe_corr(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    array_a = np.asarray(values_a, dtype=np.float32)
    array_b = np.asarray(values_b, dtype=np.float32)
    if array_a.size <= 1 or array_b.size <= 1:
        return 0.0
    if float(array_a.std()) < 1e-6 or float(array_b.std()) < 1e-6:
        return 0.0
    return float(np.corrcoef(array_a, array_b)[0, 1])


def _summary_stats(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=np.float32)
    if array.size == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _generated_full_scene_proxy(
    slice_item: Dict,
    generated_future: torch.Tensor,
    normalizer_full_scene: DifficultyNormalizer,
    config: Dict,
) -> float:
    selected_mask = np.asarray(slice_item["agent_mask"], dtype=bool)
    selected_current = np.asarray(slice_item["current_states"], dtype=np.float32)[selected_mask]
    selected_future = generated_future.cpu().numpy().astype(np.float32)[selected_mask]
    selected_future_mask = np.asarray(slice_item["future_mask"], dtype=bool)[selected_mask]
    background_current = np.asarray(slice_item.get("background_current_states", np.zeros((0, 5), dtype=np.float32)), dtype=np.float32)
    background_future = np.asarray(
        slice_item.get("background_future_states", np.zeros((0, selected_future.shape[1] if selected_future.ndim == 3 else 0, 5), dtype=np.float32)),
        dtype=np.float32,
    )
    background_future_mask = np.asarray(
        slice_item.get("background_future_mask", np.zeros(background_future.shape[:2], dtype=bool)),
        dtype=bool,
    )
    background_agent_mask = np.asarray(
        slice_item.get("background_agent_mask", np.ones(background_current.shape[0], dtype=bool)),
        dtype=bool,
    )
    combined_current = np.concatenate([selected_current, background_current], axis=0)
    combined_future = np.concatenate([selected_future, background_future], axis=0)
    combined_future_mask = np.concatenate([selected_future_mask, background_future_mask], axis=0)
    combined_agent_mask = np.concatenate([np.ones(len(selected_current), dtype=bool), background_agent_mask], axis=0)
    proxy_slice = {
        "slice_id": slice_item["slice_id"],
        "current_states": combined_current,
        "future_states": combined_future,
        "future_mask": combined_future_mask,
        "agent_mask": combined_agent_mask,
        "map_polylines": np.asarray(slice_item["map_polylines"], dtype=np.float32),
        "map_point_mask": np.asarray(slice_item["map_point_mask"], dtype=bool),
        "map_polyline_mask": np.asarray(slice_item["map_polyline_mask"], dtype=bool),
    }
    proxy_features = compute_difficulty_features(proxy_slice, config)
    proxy_score, _ = normalizer_full_scene.transform_difficulty(proxy_features)
    return float(proxy_score)


def _control_stats_from_scene(slice_item: Dict, future_states: np.ndarray, config: Dict) -> Dict[str, Dict[str, float]]:
    raw_controls = compute_raw_controls_numpy(
        current_states=np.asarray(slice_item["current_states"], dtype=np.float32),
        future_states=np.asarray(future_states, dtype=np.float32),
        future_mask=np.asarray(slice_item["future_mask"], dtype=bool),
        agent_mask=np.asarray(slice_item["agent_mask"], dtype=bool),
        dt=float(config["data"]["timestep_sec"]),
        accel_scale_mps2=float(config["model"]["accel_scale_mps2"]),
        yaw_rate_scale_radps=float(config["model"]["yaw_rate_scale_radps"]),
    )
    valid = np.asarray(slice_item["future_mask"], dtype=bool) & np.asarray(slice_item["agent_mask"], dtype=bool)[:, None]
    payload: Dict[str, Dict[str, float]] = {}
    for channel_idx, channel_name in enumerate(["accel", "yaw_rate"]):
        values = raw_controls[..., channel_idx][valid].astype(np.float32)
        payload[channel_name] = _summary_stats(values)
    return payload


def _compute_behavior_score_bundle(
    slice_item: Dict,
    future_states: np.ndarray,
    behavior_normalizer: BehaviorNormalizer,
    config: Dict,
) -> Dict[str, float]:
    features = compute_behavior_aggressiveness_features(
        future_states=future_states,
        future_mask=np.asarray(slice_item["future_mask"], dtype=bool),
        current_states=np.asarray(slice_item["current_states"], dtype=np.float32),
        agent_mask=np.asarray(slice_item["agent_mask"], dtype=bool),
        config=config,
    )
    return behavior_score_bundle_from_features(features, behavior_normalizer)


def _apply_behavior_calibration(
    target_behavior: float,
    calibrator: BehaviorCalibrator | None,
) -> float:
    if calibrator is None:
        return float(target_behavior)
    conditioned = calibrator.invert_target(np.asarray([target_behavior], dtype=np.float32))
    return float(np.clip(conditioned[0], 0.0, 1.0))


def _behavior_bundle_from_condition_value(
    value: float,
    key: str,
    normalizer: BehaviorNormalizer,
) -> Dict[str, float]:
    if key == "behavior_quantile_score_selected_agents":
        raw_tensor = normalizer.inverse_quantile_score_tensor(torch.tensor([value], dtype=torch.float32))
        raw_value = float(raw_tensor[0].item())
        return behavior_score_bundle_from_raw(raw_value, normalizer)
    if key == "behavior_aggressiveness_score_selected_agents":
        raw_tensor = normalizer.inverse_score_tensor(torch.tensor([value], dtype=torch.float32))
        raw_value = float(raw_tensor[0].item())
        return behavior_score_bundle_from_raw(raw_value, normalizer)
    if key == "behavior_raw_score_selected_agents":
        return behavior_score_bundle_from_raw(float(value), normalizer)
    raise ValueError(f"Unsupported behavior condition key: {key}")


def _conditioning_trace_records(
    model: DifficultyConditionedScenarioDiffusion,
    slice_items: Sequence[Dict],
    requested_target_difficulty: float | Sequence[float] | np.ndarray | torch.Tensor,
    requested_target_behavior: float | Sequence[float] | np.ndarray | torch.Tensor,
    model_input_target_difficulty: torch.Tensor,
    model_input_target_behavior: torch.Tensor,
    support_full_scene: Sequence[float],
    support_selected_agents: Sequence[float],
    seed: int,
) -> List[Dict[str, object]]:
    condition_embedding = model.compute_condition_embedding(
        target_difficulty=model_input_target_difficulty,
        target_behavior=model_input_target_behavior,
        force_drop=False,
    ).detach().cpu()
    if isinstance(requested_target_difficulty, torch.Tensor):
        requested_difficulty_values = requested_target_difficulty.detach().cpu().numpy().astype(np.float32)
    else:
        requested_difficulty_values = np.asarray(requested_target_difficulty, dtype=np.float32)
    if requested_difficulty_values.ndim == 0:
        requested_difficulty_values = np.full(len(slice_items), float(requested_difficulty_values), dtype=np.float32)
    if isinstance(requested_target_behavior, torch.Tensor):
        requested_behavior_values = requested_target_behavior.detach().cpu().numpy().astype(np.float32)
    else:
        requested_behavior_values = np.asarray(requested_target_behavior, dtype=np.float32)
    if requested_behavior_values.ndim == 0:
        requested_behavior_values = np.full(len(slice_items), float(requested_behavior_values), dtype=np.float32)
    traces: List[Dict[str, object]] = []
    condition_score_key = resolve_generator_condition_score_key(model.config)
    for idx, slice_item in enumerate(slice_items):
        retrieved_behavior = _required_behavior_value(slice_item, condition_score_key)
        traces.append(
            {
                "requested_target_difficulty": float(requested_difficulty_values[idx]),
                "requested_target_behavior": float(requested_behavior_values[idx]),
                "model_input_target_difficulty": float(model_input_target_difficulty[idx].item()),
                "model_input_target_behavior": float(model_input_target_behavior[idx].item()),
                "generator_condition_score_key": condition_score_key,
                "retrieved_slice_behavior": float(retrieved_behavior),
                "retrieved_behavior_quantile": _required_behavior_value(slice_item, "behavior_quantile_score_selected_agents"),
                "retrieved_behavior_aggressiveness": _required_behavior_value(slice_item, "behavior_aggressiveness_score_selected_agents"),
                "retrieved_behavior_raw_score": _required_behavior_value(slice_item, "behavior_raw_score_selected_agents"),
                "retrieved_slice_stress_difficulty_selected": float(slice_item.get("difficulty_score_selected_agents", slice_item.get("difficulty_score", 0.0))),
                "retrieved_slice_stress_difficulty_full": float(slice_item.get("difficulty_score_full_scene", slice_item.get("difficulty_score", 0.0))),
                "support_selected_agents": float(support_selected_agents[idx]),
                "support_full_scene": float(support_full_scene[idx]),
                "condition_embedding_norm": float(condition_embedding[idx].norm().item()),
                "condition_embedding_first_5_values": [float(value) for value in condition_embedding[idx, :5].tolist()],
                "seed": int(seed),
                "slice_id": str(slice_item["slice_id"]),
            }
        )
    return traces


def _load_optional_behavior_calibrator(config: Dict, project_root: str | Path) -> BehaviorCalibrator | None:
    if not bool(config["sampling"].get("behavior_calibration_enabled", False)):
        return None
    path = Path(project_root) / config["sampling"]["behavior_calibration_file"]
    if not path.exists():
        raise FileNotFoundError(f"Behavior calibration is enabled but missing: {path}")
    return load_behavior_calibrator(path)


def _make_record(
    sample_id: str,
    slice_item: Dict,
    target_difficulty: float,
    target_behavior: float,
    conditioning_behavior_input: float,
    generated_future: torch.Tensor,
    support_full_scene: float,
    support_selected_agents: float,
    retrieval_score: float,
    behavior_normalizer: BehaviorNormalizer,
    normalizer_selected: DifficultyNormalizer,
    normalizer_full_scene: DifficultyNormalizer,
    config: Dict,
    checkpoint_info: Dict[str, Any] | None = None,
    generation_mode: str = "retrieval_behavior_conditioned_generation",
    is_same_scene_diagnostic: bool = False,
    generator_behavior_low_support: bool = False,
    sample_debug: Dict[str, Any] | None = None,
) -> Dict:
    generated_np = generated_future.cpu().numpy()
    target_behavior_key = resolve_behavior_target_key(config)
    generator_condition_key = resolve_generator_condition_score_key(config)
    condition_mode = resolve_generator_condition_mode(config)
    consumes_behavior = generator_consumes_behavior(config)
    consumes_difficulty = generator_consumes_difficulty(config)
    generated_score_space = resolve_generated_score_space(config)
    generated_slice = dict(slice_item)
    generated_slice["future_states"] = generated_np.astype(np.asarray(slice_item["future_states"]).dtype)
    generated_features = compute_difficulty_features(generated_slice, config)
    generated_difficulty, _ = normalizer_selected.transform_difficulty(generated_features)
    generated_behavior_bundle = _compute_behavior_score_bundle(slice_item, generated_np, behavior_normalizer, config)
    generated_full_scene_proxy = _generated_full_scene_proxy(
        slice_item=slice_item,
        generated_future=generated_future,
        normalizer_full_scene=normalizer_full_scene,
        config=config,
    )
    coverage = slice_item.get("selection_coverage", {})
    generated_control_stats = _control_stats_from_scene(slice_item, generated_np, config)
    gt_control_stats = _control_stats_from_scene(slice_item, np.asarray(slice_item["future_states"], dtype=np.float32), config)
    retrieved_behavior_quantile = _required_behavior_value(slice_item, "behavior_quantile_score_selected_agents")
    retrieved_behavior_aggressiveness = _required_behavior_value(slice_item, "behavior_aggressiveness_score_selected_agents")
    retrieved_behavior_raw_score = _required_behavior_value(slice_item, "behavior_raw_score_selected_agents")
    conditioning_behavior_bundle = _behavior_bundle_from_condition_value(
        float(conditioning_behavior_input),
        generator_condition_key,
        behavior_normalizer,
    )
    checkpoint_info = dict(_default_checkpoint_info(), **(checkpoint_info or {}))
    sample_debug = dict(sample_debug or {})
    history_flags = _generation_history_flags(slice_item, config)
    return {
        "sample_id": sample_id,
        "mode": str(generation_mode),
        "generation_mode": str(generation_mode),
        "target_difficulty_requested": float(target_difficulty),
        "retrieved_slice_id": slice_item["slice_id"],
        "retrieved_location_id": slice_item["location_id"],
        "retrieved_scenario_id": slice_item["scenario_id"],
        "slice_id": slice_item["slice_id"],
        "location_id": slice_item["location_id"],
        "scenario_id": slice_item["scenario_id"],
        "target_difficulty": float(target_difficulty),
        "target_behavior": float(target_behavior),
        "target_behavior_score_key": target_behavior_key,
        "generator_condition_behavior_score_key": generator_condition_key if consumes_behavior else None,
        "generator_condition_mode": condition_mode,
        "generator_consumes_behavior": consumes_behavior,
        "generator_consumes_difficulty": consumes_difficulty,
        "generated_behavior_score_space": generated_score_space,
        "control_representation": str(config["model"].get("control_representation", "per_step")),
        "num_control_knots": int(config["model"].get("num_control_knots", 0)),
        "control_interpolation": str(config["model"].get("control_interpolation", "linear")),
        "anchor_enabled": bool(config["model"].get("use_anchor_latent", False)),
        "anchor_dim": int(config["model"].get("anchor_dim", 0)),
        "interaction_oracle_mode": str(config.get("interaction_oracle", {}).get("mode", "none")),
        "residual_diffusion_enabled": bool(config["model"].get("residual_diffusion_enabled", False)),
        "anchor_residual_diffusion_enabled": bool(config.get("anchor_residual_diffusion", {}).get("enabled", False)),
        "sampling_use_residual_diffusion": bool(config["sampling"].get("use_residual_diffusion", True)),
        "sampling_use_anchor_residual_diffusion": bool(config["sampling"].get("use_anchor_residual_diffusion", False)),
        "diffusion_target_type": str(config.get("residual_diffusion", {}).get("target_type", config["diffusion"].get("target_type", "epsilon"))),
        "anchor_diffusion_target_type": str(config.get("anchor_residual_diffusion", {}).get("target_type", "x0")),
        "rollout_sampling_mode": sample_debug.get("rollout_sampling_mode"),
        "interaction_oracle_mode_used": sample_debug.get("interaction_oracle_mode_used"),
        "residual_scale": sample_debug.get("residual_scale"),
        "anchor_residual_scale": sample_debug.get("anchor_residual_scale"),
        "generator_condition_behavior": float(conditioning_behavior_input) if consumes_behavior else None,
        "conditioning_behavior_input": float(conditioning_behavior_input) if consumes_behavior else None,
        "slice_difficulty": float(slice_item.get("difficulty_score_selected_agents", slice_item.get("difficulty_score", 0.0))),
        "slice_difficulty_selected_agents": float(slice_item.get("difficulty_score_selected_agents", slice_item.get("difficulty_score", 0.0))),
        "slice_difficulty_full_scene": float(slice_item.get("difficulty_score_full_scene", slice_item.get("difficulty_score", 0.0))),
        "slice_stress_difficulty_selected_agents": float(slice_item.get("difficulty_score_selected_agents", slice_item.get("difficulty_score", 0.0))),
        "slice_stress_difficulty_full_scene": float(slice_item.get("difficulty_score_full_scene", slice_item.get("difficulty_score", 0.0))),
        "slice_behavior_aggressiveness_selected_agents": retrieved_behavior_aggressiveness,
        "slice_behavior_quantile_selected_agents": retrieved_behavior_quantile,
        "slice_behavior_raw_score_selected_agents": retrieved_behavior_raw_score,
        "retrieved_full_scene_difficulty": float(slice_item.get("difficulty_score_full_scene", slice_item.get("difficulty_score", 0.0))),
        "retrieved_selected_agents_difficulty": float(slice_item.get("difficulty_score_selected_agents", slice_item.get("difficulty_score", 0.0))),
        "retrieved_behavior_quantile": retrieved_behavior_quantile,
        "retrieved_behavior_aggressiveness": retrieved_behavior_aggressiveness,
        "retrieved_behavior_raw_score": retrieved_behavior_raw_score,
        "generator_condition_behavior_quantile": float(conditioning_behavior_bundle["quantile"]) if consumes_behavior else None,
        "generator_condition_behavior_aggressiveness": float(conditioning_behavior_bundle["aggressiveness"]) if consumes_behavior else None,
        "generator_condition_behavior_raw_score": float(conditioning_behavior_bundle["raw"]) if consumes_behavior else None,
        "generated_difficulty": float(generated_difficulty),
        "generated_difficulty_selected_agents": float(generated_difficulty),
        "generated_difficulty_full_scene_proxy": float(generated_full_scene_proxy),
        "generated_stress_difficulty_selected_agents": float(generated_difficulty),
        "generated_full_scene_stress_proxy": float(generated_full_scene_proxy),
        "generated_selected_stress_difficulty": float(generated_difficulty),
        "generated_full_scene_proxy": float(generated_full_scene_proxy),
        "generated_behavior_quantile": float(generated_behavior_bundle["quantile"]),
        "generated_behavior_aggressiveness": float(generated_behavior_bundle["aggressiveness"]),
        "generated_behavior_raw_score": float(generated_behavior_bundle["raw"]),
        "gt_behavior_quantile": retrieved_behavior_quantile,
        "gt_behavior_aggressiveness": retrieved_behavior_aggressiveness,
        "gt_behavior_raw_score": retrieved_behavior_raw_score,
        "gt_selected_stress_difficulty": float(slice_item.get("difficulty_score_selected_agents", slice_item.get("difficulty_score", 0.0))),
        "checkpoint_used": checkpoint_info.get("checkpoint_used"),
        "checkpoint_exists": bool(checkpoint_info.get("checkpoint_exists", False)),
        "checkpoint_sha256": checkpoint_info.get("checkpoint_sha256"),
        "checkpoint_epoch": checkpoint_info.get("checkpoint_epoch"),
        "checkpoint_selection_metric": checkpoint_info.get("checkpoint_selection_metric"),
        "checkpoint_path": checkpoint_info.get("checkpoint_path"),
        "support_score": float(support_selected_agents),
        "support_selected_agents": float(support_selected_agents),
        "support_full_scene": float(support_full_scene),
        "retrieval_score": float(retrieval_score),
        "coverage": float(
            coverage.get(
                "selected_agents_combined_coverage",
                coverage.get("selected_agents_difficulty_coverage", coverage.get("selected_agents_raw_coverage", 1.0)),
            )
        ),
        "selected_agents_coverage_score": float(
            coverage.get(
                "selected_agents_combined_coverage",
                coverage.get("selected_agents_difficulty_coverage", coverage.get("selected_agents_raw_coverage", 1.0)),
            )
        ),
        "selected_agents_difficulty_coverage": float(
            coverage.get(
                "selected_agents_combined_coverage",
                coverage.get("selected_agents_difficulty_coverage", coverage.get("selected_agents_raw_coverage", 1.0)),
            )
        ),
        **history_flags,
        "is_same_scene_diagnostic": bool(is_same_scene_diagnostic),
        "generator_behavior_low_support": bool(generator_behavior_low_support),
        "novelty_score": None,
        "sampled_residual_std": sample_debug.get("sampled_residual_std"),
        "final_knot_std": sample_debug.get("final_knot_std"),
        "mu_knot_std": sample_debug.get("mu_knot_std"),
        "residual_clip_fraction": sample_debug.get("residual_clip_fraction"),
        "sampled_anchor_residual_std": sample_debug.get("sampled_anchor_residual_std"),
        "anchor_mu_std": sample_debug.get("anchor_mu_std"),
        "final_anchor_std": sample_debug.get("final_anchor_std"),
        "anchor_residual_clip_fraction": sample_debug.get("anchor_residual_clip_fraction"),
        "generated_control_stats": generated_control_stats,
        "gt_control_stats": gt_control_stats,
        "generated_control_accel_mean": float(generated_control_stats["accel"]["mean"]),
        "generated_control_accel_std": float(generated_control_stats["accel"]["std"]),
        "generated_control_accel_p95": float(generated_control_stats["accel"]["p95"]),
        "generated_control_accel_p99": float(generated_control_stats["accel"]["p99"]),
        "generated_control_yaw_rate_mean": float(generated_control_stats["yaw_rate"]["mean"]),
        "generated_control_yaw_rate_std": float(generated_control_stats["yaw_rate"]["std"]),
        "generated_control_yaw_rate_p95": float(generated_control_stats["yaw_rate"]["p95"]),
        "generated_control_yaw_rate_p99": float(generated_control_stats["yaw_rate"]["p99"]),
        "gt_control_accel_mean": float(gt_control_stats["accel"]["mean"]),
        "gt_control_accel_std": float(gt_control_stats["accel"]["std"]),
        "gt_control_accel_p95": float(gt_control_stats["accel"]["p95"]),
        "gt_control_accel_p99": float(gt_control_stats["accel"]["p99"]),
        "gt_control_yaw_rate_mean": float(gt_control_stats["yaw_rate"]["mean"]),
        "gt_control_yaw_rate_std": float(gt_control_stats["yaw_rate"]["std"]),
        "gt_control_yaw_rate_p95": float(gt_control_stats["yaw_rate"]["p95"]),
        "gt_control_yaw_rate_p99": float(gt_control_stats["yaw_rate"]["p99"]),
        "generated_future": generated_future.cpu(),
        "gt_future": torch.as_tensor(slice_item["future_states"]).float(),
        "current_states": torch.as_tensor(slice_item["current_states"]).float(),
        "history_states": torch.as_tensor(slice_item["history_states"]).float(),
        "map_polylines": torch.as_tensor(slice_item["map_polylines"]).float(),
        "future_mask": torch.as_tensor(slice_item["future_mask"]).bool(),
        "agent_mask": torch.as_tensor(slice_item["agent_mask"]).bool(),
        "map_point_mask": torch.as_tensor(slice_item["map_point_mask"]).bool(),
        "map_polyline_mask": torch.as_tensor(slice_item["map_polyline_mask"]).bool(),
        "metadata": slice_item.get("metadata", {}),
    }


def _target_bin_diagnostics(records: List[Dict]) -> Dict[str, Dict[str, object]]:
    grouped: Dict[str, List[Dict]] = {}
    for record in records:
        label = f"{float(record.get('target_behavior', record['target_difficulty'])):.3f}"
        grouped.setdefault(label, []).append(record)
    output: Dict[str, Dict[str, object]] = {}
    for label, items in grouped.items():
        target_behavior = np.asarray([float(item.get("target_behavior", item["target_difficulty"])) for item in items], dtype=np.float32)
        sampled_behavior = np.asarray([float(item.get("generated_behavior_quantile", 0.0)) for item in items], dtype=np.float32)
        generated_accel = np.asarray([float(item.get("generated_control_accel_mean", 0.0)) for item in items], dtype=np.float32)
        generated_yaw = np.asarray([float(item.get("generated_control_yaw_rate_mean", 0.0)) for item in items], dtype=np.float32)
        gt_accel = np.asarray([float(item.get("gt_control_accel_mean", 0.0)) for item in items], dtype=np.float32)
        gt_yaw = np.asarray([float(item.get("gt_control_yaw_rate_mean", 0.0)) for item in items], dtype=np.float32)
        output[label] = {
            "count": len(items),
            "target_behavior_mean": float(target_behavior.mean()) if target_behavior.size else 0.0,
            "sampled_behavior_mean": float(sampled_behavior.mean()) if sampled_behavior.size else 0.0,
            "sampled_behavior_mae": float(np.mean(np.abs(sampled_behavior - target_behavior))) if sampled_behavior.size else 0.0,
            "sampled_behavior_corr": _safe_corr(target_behavior, sampled_behavior),
            "sampled_behavior_bias": float((sampled_behavior - target_behavior).mean()) if sampled_behavior.size else 0.0,
            "sampled_control_accel": _summary_stats(generated_accel),
            "sampled_control_yaw_rate": _summary_stats(generated_yaw),
            "gt_control_accel": _summary_stats(gt_accel),
            "gt_control_yaw_rate": _summary_stats(gt_yaw),
            "location_distribution": {str(key): int(value) for key, value in _count_by_key(items, "location_id").items()},
            "agent_count_distribution": _summary_stats([int(torch.as_tensor(item["agent_mask"]).sum().item()) for item in items]),
        }
    return output


def _count_by_key(items: List[Dict], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        value = str(item.get(key, ""))
        counts[value] = counts.get(value, 0) + 1
    return counts


def _compute_batch_behavior(
    slice_items: Sequence[Dict],
    future_batch: torch.Tensor,
    behavior_normalizer: BehaviorNormalizer,
    config: Dict,
) -> np.ndarray:
    score_space = resolve_generated_score_space(config)
    values = []
    futures = future_batch.detach().cpu().numpy()
    for slice_item, future in zip(slice_items, futures):
        bundle = _compute_behavior_score_bundle(slice_item, future, behavior_normalizer, config)
        values.append(float(bundle[score_space]))
    return np.asarray(values, dtype=np.float32)


def _deterministic_denoise_diagnostics(
    model: DifficultyConditionedScenarioDiffusion,
    batch: Dict,
    slice_items: Sequence[Dict],
    target_behavior: torch.Tensor,
    behavior_normalizer: BehaviorNormalizer,
    config: Dict,
) -> Dict[str, Dict[str, float]]:
    if model.decoder_type != "kinematic_controls":
        return {}
    encoded_target = model.normalize_controls(
        model.encode_future_controls(
            current_states=batch["current_states"],
            future_states=batch["future_states"],
            future_mask=batch["future_mask"],
            agent_mask=batch["agent_mask"],
            dt=float(config["data"]["timestep_sec"]),
        )
    )
    num_steps = int(config["diffusion"]["num_steps"])
    bucket_ratios = {
        "low_noise": 0.10,
        "mid_noise": 0.50,
        "high_noise": 0.90,
    }
    diagnostics: Dict[str, Dict[str, float]] = {}
    for name, ratio in bucket_ratios.items():
        timestep_value = int(round(ratio * max(num_steps - 1, 1)))
        timesteps = torch.full((encoded_target.shape[0],), timestep_value, device=encoded_target.device, dtype=torch.long)
        noise = torch.randn_like(encoded_target)
        noisy_future = model.scheduler.q_sample(encoded_target, timesteps, noise)
        predicted_noise = model.predict_noise(
            noisy_future=noisy_future,
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
            target_behavior=target_behavior,
        )
        pred_x0 = model.scheduler.predict_start_from_noise(noisy_future, timesteps, predicted_noise)
        raw_controls = model.denormalize_controls(pred_x0)
        raw_controls = model.clamp_raw_controls_to_train_quantiles(raw_controls)
        decoded_future, _ = model.decode_controls_to_future(
            current_states=batch["current_states"],
            controls=raw_controls,
            future_mask=batch["future_mask"],
            agent_mask=batch["agent_mask"],
            dt=float(config["data"]["timestep_sec"]),
        )
        decoded_behavior = _compute_batch_behavior(slice_items, decoded_future, behavior_normalizer, config)
        target_np = target_behavior.detach().cpu().numpy()
        diagnostics[name] = {
            "count": int(len(decoded_behavior)),
            "generated_behavior_mean": float(decoded_behavior.mean()) if decoded_behavior.size else 0.0,
            "behavior_mae": float(np.mean(np.abs(decoded_behavior - target_np))) if decoded_behavior.size else 0.0,
            "behavior_bias": float((decoded_behavior - target_np).mean()) if decoded_behavior.size else 0.0,
        }
    return diagnostics


def _pred_x0_vs_ddpm_diagnostics(
    model: DifficultyConditionedScenarioDiffusion,
    batch: Dict,
    slice_items: Sequence[Dict],
    sampled_future: torch.Tensor,
    target_behavior: torch.Tensor,
    behavior_normalizer: BehaviorNormalizer,
    config: Dict,
) -> Dict[str, object]:
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
        target_difficulty=batch.get("target_difficulty"),
        target_behavior=target_behavior,
    )
    pred_x0_behavior = _compute_batch_behavior(slice_items, model_output["decoded_future"], behavior_normalizer, config)
    ddpm_behavior = _compute_batch_behavior(slice_items, sampled_future, behavior_normalizer, config)
    target_np = target_behavior.detach().cpu().numpy()
    return {
        "train_pred_x0_behavior_mean": float(pred_x0_behavior.mean()) if pred_x0_behavior.size else 0.0,
        "train_pred_x0_behavior_mae": float(np.mean(np.abs(pred_x0_behavior - target_np))) if pred_x0_behavior.size else 0.0,
        "ddpm_sample_behavior_mean": float(ddpm_behavior.mean()) if ddpm_behavior.size else 0.0,
        "ddpm_sample_behavior_mae": float(np.mean(np.abs(ddpm_behavior - target_np))) if ddpm_behavior.size else 0.0,
        "ddpm_sample_behavior_corr": _safe_corr(target_np, ddpm_behavior),
        "pred_x0_behavior_corr": _safe_corr(target_np, pred_x0_behavior),
        "deterministic_denoise": _deterministic_denoise_diagnostics(
            model=model,
            batch=batch,
            slice_items=slice_items,
            target_behavior=target_behavior,
            behavior_normalizer=behavior_normalizer,
            config=config,
        ),
    }


def _collect_group_diagnostics(
    model: DifficultyConditionedScenarioDiffusion,
    batch: Dict,
    slice_items: Sequence[Dict],
    sampled_future: torch.Tensor,
    target_behavior: torch.Tensor,
    behavior_normalizer: BehaviorNormalizer,
    config: Dict,
) -> Dict[str, object]:
    valid_mask = batch["future_mask"] & batch["agent_mask"].unsqueeze(-1)
    gt_raw_controls = model.encode_future_controls(
        current_states=batch["current_states"],
        future_states=batch["future_states"],
        future_mask=batch["future_mask"],
        agent_mask=batch["agent_mask"],
        dt=float(config["data"]["timestep_sec"]),
    )
    generated_raw_controls = torch.zeros_like(gt_raw_controls)
    if model.decoder_type == "kinematic_controls":
        generated_encoded = model.normalize_controls(gt_raw_controls)
        _ = generated_encoded  # kept for explicit path verification in runtime metadata
        sample_debug = dict(model.last_sample_debug)
        generated_control_stats = sample_debug.get("sampled_control_stats_raw", {})
    else:
        generated_control_stats = {}
    return {
        "model_runtime": dict(model.get_runtime_metadata()),
        "sample_runtime": dict(model.last_sample_debug),
        "gt_raw_control_stats": {
            "accel": _summary_stats(gt_raw_controls[..., 0][valid_mask].detach().cpu().numpy()),
            "yaw_rate": _summary_stats(gt_raw_controls[..., 1][valid_mask].detach().cpu().numpy()),
        },
        "generated_raw_control_stats": generated_control_stats,
        "pred_x0_vs_ddpm": _pred_x0_vs_ddpm_diagnostics(
            model=model,
            batch=batch,
            slice_items=slice_items,
            sampled_future=sampled_future,
            target_behavior=target_behavior,
            behavior_normalizer=behavior_normalizer,
            config=config,
        ),
    }


def _final_sampling_diagnostics(
    records: List[Dict],
    per_target_diagnostics: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    target_behavior = np.asarray([float(record.get("target_behavior", record["target_difficulty"])) for record in records], dtype=np.float32)
    sampled_behavior = np.asarray([float(record.get("generated_behavior_quantile", 0.0)) for record in records], dtype=np.float32)
    summary = {
        "overall": {
            "count": len(records),
            "target_behavior_mean": float(target_behavior.mean()) if target_behavior.size else 0.0,
            "sampled_behavior_mean": float(sampled_behavior.mean()) if sampled_behavior.size else 0.0,
            "sampled_behavior_mae": float(np.mean(np.abs(sampled_behavior - target_behavior))) if sampled_behavior.size else 0.0,
            "sampled_behavior_corr": _safe_corr(target_behavior, sampled_behavior),
            "sampled_behavior_bias": float((sampled_behavior - target_behavior).mean()) if sampled_behavior.size else 0.0,
        },
        "by_target_behavior": _target_bin_diagnostics(records),
        "per_target_model_diagnostics": per_target_diagnostics,
    }
    return summary


def _make_initial_noise(shape: Tuple[int, ...], device: torch.device, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    generator.manual_seed(int(seed))
    return torch.randn(shape, generator=generator, device=device)


def _sample_noise_shape(
    model: DifficultyConditionedScenarioDiffusion,
    future_mask: torch.Tensor,
    agent_mask: torch.Tensor,
    use_residual_diffusion: bool,
    use_anchor_residual_diffusion: bool = False,
) -> Tuple[int, ...]:
    if bool(model.use_anchor_latent) and bool(use_anchor_residual_diffusion):
        return (
            int(agent_mask.shape[0]),
            int(agent_mask.shape[1]),
            int(model.anchor_dim),
        )
    if (
        model.decoder_type == "kinematic_controls"
        and model.control_representation == "knots"
        and bool(use_residual_diffusion)
    ):
        return (
            int(agent_mask.shape[0]),
            int(agent_mask.shape[1]),
            int(model.num_control_knots),
            int(model.control_dim),
        )
    return (*future_mask.shape, int(model.model_future_dim))


def _filter_candidates_by_behavior_range(
    candidates: Sequence[RetrievalCandidate],
    retriever: DifficultyRetriever,
    behavior_range: Tuple[float, float] | None,
) -> List[RetrievalCandidate]:
    if behavior_range is None:
        return list(candidates)
    low, high = behavior_range
    condition_key = resolve_generator_condition_score_key(retriever.config)
    filtered = [
        candidate
        for candidate in candidates
        if low <= _required_behavior_value(retriever.slice_by_id[candidate.slice_id], condition_key) <= high
    ]
    if not filtered:
        raise ValueError(
            f"No retrieved slices satisfy behavior_range={behavior_range}. "
            "Relax the requested behavior range or increase top_k."
        )
    return filtered


def _candidates_from_filtered_indices(
    retriever: DifficultyRetriever,
    candidate_indices: Sequence[int],
    target_difficulty: float,
    *,
    strategy: str,
    diversity_seed: int | None = None,
    top_k: int | None = None,
) -> List[RetrievalCandidate]:
    if not candidate_indices:
        raise ValueError("Retriever found no slices that satisfy the requested constraints.")
    top_k = int(top_k or retriever.config["retrieval"]["top_k"])
    embeddings = retriever.slice_index.embeddings[list(candidate_indices)]
    target = torch.full((len(candidate_indices),), float(target_difficulty), dtype=torch.float32)
    support_full_scene, support_selected_agents = retriever.score_supports(embeddings, target)
    full_scene_difficulties = retriever.slice_index.difficulty_scores_full_scene[list(candidate_indices)]
    selected_difficulties = retriever.slice_index.difficulty_scores_selected_agents[list(candidate_indices)]
    difficulty_gap_full_scene = torch.abs(full_scene_difficulties - target)
    difficulty_gap_selected_agents = torch.abs(selected_difficulties - target)
    coverage_score = torch.as_tensor(
        [
            float(retriever.slice_index.metadata[index].get("selected_agents_difficulty_coverage", 1.0))
            for index in candidate_indices
        ],
        dtype=torch.float32,
    )
    retrieval_cfg = retriever.config["retrieval"]
    if strategy == "random":
        rng = np.random.default_rng(diversity_seed)
        score = torch.as_tensor(rng.normal(size=len(candidate_indices)), dtype=torch.float32)
    elif strategy == "no_support":
        score = (
            -float(retrieval_cfg.get("full_scene_difficulty_weight", retrieval_cfg.get("difficulty_weight", 1.0)))
            * difficulty_gap_full_scene
            - float(retrieval_cfg.get("selected_agents_gap_weight", 0.3)) * difficulty_gap_selected_agents
            + float(retrieval_cfg.get("coverage_weight", 0.4)) * coverage_score
        )
    else:
        raise ValueError(f"Unsupported retrieval candidate strategy: {strategy}")
    best_score, order = torch.topk(score, k=min(top_k, len(candidate_indices)))
    candidates: List[RetrievalCandidate] = []
    for rank, order_idx in enumerate(order.tolist()):
        original_index = int(candidate_indices[order_idx])
        candidates.append(
            RetrievalCandidate(
                slice_id=retriever.slice_index.slice_ids[original_index],
                retrieval_score=float(best_score[rank].item()),
                support_score=float(support_selected_agents[order_idx].item()),
                difficulty_gap=float(difficulty_gap_selected_agents[order_idx].item()),
                support_full_scene=float(support_full_scene[order_idx].item()),
                support_selected_agents=float(support_selected_agents[order_idx].item()),
                difficulty_gap_full_scene=float(difficulty_gap_full_scene[order_idx].item()),
                difficulty_gap_selected_agents=float(difficulty_gap_selected_agents[order_idx].item()),
                coverage_score=float(coverage_score[order_idx].item()),
                index=original_index,
            )
        )
    return candidates


def _retrieve_candidates(
    retriever: DifficultyRetriever,
    *,
    target_difficulty: float,
    location_id: str | None,
    map_id: str | None,
    behavior_range: Tuple[float, float] | None,
    seed: int,
    strategy: str = "default",
) -> List[RetrievalCandidate]:
    if strategy == "default":
        candidates = retriever.retrieve(
            target_difficulty=float(target_difficulty),
            location_id=location_id,
            map_id=map_id,
            top_k=int(retriever.config["retrieval"]["top_k"]),
        )
    else:
        candidate_indices = retriever._filter_indices(location_id=location_id, map_id=map_id, agent_count_range=None)
        candidates = _candidates_from_filtered_indices(
            retriever,
            candidate_indices,
            target_difficulty=float(target_difficulty),
            strategy=strategy,
            diversity_seed=seed,
            top_k=int(retriever.config["retrieval"]["top_k"]),
        )
    return _filter_candidates_by_behavior_range(candidates, retriever, behavior_range)


def _resolve_generator_behavior_inputs(
    slice_items: Sequence[Dict],
    explicit_generator_target_behavior: float | None,
    local_behavior_support_radius: float,
    config: Dict,
) -> tuple[np.ndarray, np.ndarray]:
    generator_condition_key = resolve_generator_condition_score_key(config)
    retrieved_behavior = np.asarray(
        [_required_behavior_value(item, generator_condition_key) for item in slice_items],
        dtype=np.float32,
    )
    if not generator_consumes_behavior(config):
        return retrieved_behavior, np.zeros(len(slice_items), dtype=bool)
    if explicit_generator_target_behavior is None:
        return retrieved_behavior, np.zeros(len(slice_items), dtype=bool)
    requested = np.full(len(slice_items), float(explicit_generator_target_behavior), dtype=np.float32)
    low_support = np.abs(requested - retrieved_behavior) > float(local_behavior_support_radius)
    return requested, low_support


def sample_retrieval_augmented_generation(
    config: Dict,
    project_root: str | Path,
    target_difficulties: Sequence[float],
    location_id: str | None = None,
    map_id: str | None = None,
    behavior_range: Tuple[float, float] | None = None,
    num_slices: int | None = None,
    num_rollouts_per_slice: int = 1,
    generator_target_behavior: float | None = None,
    checkpoint: str | Path | None = None,
    generation_mode: str = "retrieval_augmented_generation",
    retrieval_strategy: str = "default",
    seed: int = 0,
) -> Tuple[List[Dict], Dict[str, object], List[Dict[str, object]]]:
    device = resolve_device(config["training"]["device"])
    model, _, checkpoint_info = load_model_for_sampling(config, project_root, checkpoint=checkpoint)
    retriever = build_retriever(config, project_root, split="train")
    cache_dir = Path(project_root) / config["data"]["cache_dir"]
    normalizer_selected = load_difficulty_normalizer(
        Path(project_root) / config["data"]["cache_dir"] / "difficulty_stats.json",
        label_space="selected_agents",
    )
    behavior_normalizer = load_behavior_normalizer(
        Path(project_root) / config["data"]["cache_dir"] / "behavior_stats_selected_agents.json"
    )
    normalizer_full_scene = load_difficulty_normalizer(
        Path(project_root) / config["data"]["cache_dir"] / "difficulty_stats.json",
        label_space="full_scene",
    )
    calibrator = _load_optional_behavior_calibrator(config, project_root)
    records: List[Dict] = []
    per_target_diagnostics: Dict[str, Dict[str, object]] = {}
    conditioning_trace: List[Dict[str, object]] = []
    num_slices = int(num_slices or config["sampling"]["num_samples_per_difficulty"])
    num_rollouts_per_slice = max(int(num_rollouts_per_slice), 1)
    local_behavior_support_radius = float(config["sampling"].get("local_behavior_support_radius", 0.10))
    condition_mode = resolve_generator_condition_mode(config)
    for target in tqdm(target_difficulties, desc="Sampling targets", unit="target"):
        candidates = _retrieve_candidates(
            retriever,
            target_difficulty=float(target),
            location_id=location_id,
            map_id=map_id,
            behavior_range=behavior_range,
            seed=int(seed + round(float(target) * 1000)),
            strategy=retrieval_strategy,
        )
        selected = retriever.sample_diverse(
            candidates,
            num_samples=min(num_slices, len(candidates)),
            diversity_seed=int(seed + round(float(target) * 1000)),
        )
        selected_items = _materialize_sampling_slice_items([retriever.slice_by_id[item.slice_id] for item in selected], cache_dir)
        batch = _move_batch(_stack_batch(selected_items), device)
        generator_behavior_inputs, low_support_flags = _resolve_generator_behavior_inputs(
            selected_items,
            explicit_generator_target_behavior=generator_target_behavior,
            local_behavior_support_radius=local_behavior_support_radius,
            config=config,
        )
        generator_behavior_inputs = np.asarray(
            [_apply_behavior_calibration(float(value), calibrator) for value in generator_behavior_inputs],
            dtype=np.float32,
        )
        batch["target_difficulty"] = torch.full((len(selected_items),), float(target), device=device)
        batch["target_behavior"] = torch.as_tensor(generator_behavior_inputs, dtype=torch.float32, device=device)
        all_generated: List[torch.Tensor] = []
        for rollout_idx in range(num_rollouts_per_slice):
            rollout_seed = int(seed + round(float(target) * 1000) + rollout_idx)
            guidance_scale = (
                0.0
                if generation_mode == "retrieval_unconditional_generation"
                else float(config["sampling"]["guidance_scale"])
            )
            generated = model.sample(
                history_states=batch["history_states"],
                history_mask=batch["history_mask"],
                current_states=batch["current_states"],
                future_mask=batch["future_mask"],
                agent_mask=batch["agent_mask"],
                map_polylines=batch["map_polylines"],
                map_point_mask=batch["map_point_mask"],
                map_polyline_mask=batch["map_polyline_mask"],
                target_difficulty=batch["target_difficulty"] if generator_consumes_difficulty(config) else None,
                target_behavior=batch["target_behavior"] if generator_consumes_behavior(config) else None,
                sample_steps=int(config["sampling"]["sample_steps"]),
                guidance_scale=guidance_scale,
                sampler=str(config["sampling"].get("sampler", "ddpm")),
                ddim_eta=float(config["sampling"].get("ddim_eta", 0.0)),
                use_residual_diffusion=bool(config["sampling"].get("use_residual_diffusion", True)),
                use_anchor_residual_diffusion=bool(config["sampling"].get("use_anchor_residual_diffusion", False)),
                initial_noise=_make_initial_noise(
                    _sample_noise_shape(
                        model,
                        batch["future_mask"],
                        batch["agent_mask"],
                        bool(config["sampling"].get("use_residual_diffusion", True)),
                        bool(config["sampling"].get("use_anchor_residual_diffusion", False)),
                    ),
                    device=device,
                    seed=rollout_seed,
                ),
            )
            if generator_consumes_behavior(config) or generator_consumes_difficulty(config):
                conditioning_trace.extend(
                    _conditioning_trace_records(
                        model=model,
                        slice_items=selected_items,
                        requested_target_difficulty=float(target),
                        requested_target_behavior=generator_behavior_inputs,
                        model_input_target_difficulty=batch["target_difficulty"],
                        model_input_target_behavior=batch["target_behavior"],
                        support_full_scene=[candidate.support_full_scene for candidate in selected],
                        support_selected_agents=[candidate.support_selected_agents for candidate in selected],
                        seed=rollout_seed,
                    )
                )
            if rollout_idx == 0:
                per_target_diagnostics[f"{float(target):.3f}"] = _collect_group_diagnostics(
                    model=model,
                    batch=batch,
                    slice_items=selected_items,
                    sampled_future=generated,
                    target_behavior=batch["target_behavior"],
                    behavior_normalizer=behavior_normalizer,
                    config=config,
                )
            all_generated.append(generated)
        for rollout_idx, generated in enumerate(all_generated):
            for idx, candidate in enumerate(tqdm(selected, desc=f"Exporting target={target:.2f}", unit="sample", leave=False)):
                records.append(
                    _make_record(
                        sample_id=f"{generation_mode}:{target:.2f}:{idx}:rollout{rollout_idx}",
                        slice_item=selected_items[idx],
                        target_difficulty=float(target),
                        target_behavior=float(generator_behavior_inputs[idx]),
                        conditioning_behavior_input=float(generator_behavior_inputs[idx]),
                        generated_future=generated[idx].cpu(),
                        support_full_scene=candidate.support_full_scene,
                        support_selected_agents=candidate.support_selected_agents,
                        retrieval_score=candidate.retrieval_score,
                        behavior_normalizer=behavior_normalizer,
                        normalizer_selected=normalizer_selected,
                        normalizer_full_scene=normalizer_full_scene,
                        config=config,
                        checkpoint_info=checkpoint_info,
                        generation_mode=generation_mode,
                        is_same_scene_diagnostic=False,
                        generator_behavior_low_support=bool(low_support_flags[idx]),
                        sample_debug=model.last_sample_debug,
                    )
                )
    diagnostics = _final_sampling_diagnostics(records, per_target_diagnostics)
    diagnostics["generator_condition_mode"] = condition_mode
    diagnostics["generator_consumes_behavior"] = generator_consumes_behavior(config)
    diagnostics["generator_consumes_difficulty"] = generator_consumes_difficulty(config)
    return records, diagnostics, conditioning_trace


def sample_retrieval_level(
    config: Dict,
    project_root: str | Path,
    target_difficulties: Sequence[float],
    seed: int = 0,
) -> Tuple[List[Dict], Dict[str, object], List[Dict[str, object]]]:
    return sample_retrieval_augmented_generation(
        config,
        project_root,
        target_difficulties=target_difficulties,
        checkpoint=None,
        generation_mode="retrieval_level",
        seed=seed,
    )


def sample_same_scene_diagnostic(
    config: Dict,
    project_root: str | Path,
    slice_id: str,
    target_difficulties: Sequence[float],
    checkpoint: str | Path | None = None,
    seed: int = 0,
) -> Tuple[List[Dict], Dict[str, object], List[Dict[str, object]]]:
    device = resolve_device(config["training"]["device"])
    model, _, checkpoint_info = load_model_for_sampling(config, project_root, checkpoint=checkpoint)
    retriever = build_retriever(config, project_root, split="train")
    cache_dir = Path(project_root) / config["data"]["cache_dir"]
    normalizer_selected = load_difficulty_normalizer(
        Path(project_root) / config["data"]["cache_dir"] / "difficulty_stats.json",
        label_space="selected_agents",
    )
    behavior_normalizer = load_behavior_normalizer(
        Path(project_root) / config["data"]["cache_dir"] / "behavior_stats_selected_agents.json"
    )
    normalizer_full_scene = load_difficulty_normalizer(
        Path(project_root) / config["data"]["cache_dir"] / "difficulty_stats.json",
        label_space="full_scene",
    )
    calibrator = _load_optional_behavior_calibrator(config, project_root)
    slice_item = _materialize_sampling_slice_items([retriever.slice_by_id[slice_id]], cache_dir)[0]
    conditioning_values = [_apply_behavior_calibration(float(target), calibrator) for target in target_difficulties]
    batch = _move_batch(_stack_batch([slice_item] * len(target_difficulties)), device)
    target_tensor = torch.as_tensor(target_difficulties, dtype=torch.float32, device=device)
    conditioning_tensor = torch.as_tensor(conditioning_values, dtype=torch.float32, device=device)
    batch["target_difficulty"] = target_tensor
    batch["target_behavior"] = conditioning_tensor
    support_full_scene, support_selected_agents = retriever.score_supports(batch["retrieval_embedding"], target_tensor)
    generated = model.sample(
        history_states=batch["history_states"],
        history_mask=batch["history_mask"],
        current_states=batch["current_states"],
        future_mask=batch["future_mask"],
        agent_mask=batch["agent_mask"],
        map_polylines=batch["map_polylines"],
        map_point_mask=batch["map_point_mask"],
        map_polyline_mask=batch["map_polyline_mask"],
        target_difficulty=target_tensor,
        target_behavior=conditioning_tensor,
        sample_steps=int(config["sampling"]["sample_steps"]),
        guidance_scale=float(config["sampling"]["guidance_scale"]),
        sampler=str(config["sampling"].get("sampler", "ddpm")),
        ddim_eta=float(config["sampling"].get("ddim_eta", 0.0)),
        use_residual_diffusion=bool(config["sampling"].get("use_residual_diffusion", True)),
        use_anchor_residual_diffusion=bool(config["sampling"].get("use_anchor_residual_diffusion", False)),
    )
    records = []
    condition_embedding = model.compute_condition_embedding(
        target_difficulty=target_tensor,
        target_behavior=conditioning_tensor,
        force_drop=False,
    ).detach().cpu()
    conditioning_trace = [
        {
            "requested_target_difficulty": float(target_difficulties[idx]),
            "requested_target_behavior": float(target_difficulties[idx]),
            "model_input_target_difficulty": float(target_tensor[idx].item()),
            "model_input_target_behavior": float(conditioning_tensor[idx].item()),
            "generator_condition_score_key": resolve_generator_condition_score_key(config),
            "retrieved_slice_behavior": _required_behavior_value(slice_item, resolve_generator_condition_score_key(config)),
            "retrieved_behavior_quantile": _required_behavior_value(slice_item, "behavior_quantile_score_selected_agents"),
            "retrieved_behavior_aggressiveness": _required_behavior_value(slice_item, "behavior_aggressiveness_score_selected_agents"),
            "retrieved_behavior_raw_score": _required_behavior_value(slice_item, "behavior_raw_score_selected_agents"),
            "retrieved_slice_stress_difficulty_selected": float(slice_item.get("difficulty_score_selected_agents", slice_item.get("difficulty_score", 0.0))),
            "retrieved_slice_stress_difficulty_full": float(slice_item.get("difficulty_score_full_scene", slice_item.get("difficulty_score", 0.0))),
            "support_selected_agents": float(support_selected_agents[idx].item()),
            "support_full_scene": float(support_full_scene[idx].item()),
            "condition_embedding_norm": float(condition_embedding[idx].norm().item()),
            "condition_embedding_first_5_values": [float(value) for value in condition_embedding[idx, :5].tolist()],
            "seed": int(seed),
            "slice_id": str(slice_item["slice_id"]),
        }
        for idx in range(len(target_difficulties))
    ]
    for idx, target in enumerate(tqdm(target_difficulties, desc="Same-scene diagnostic", unit="target")):
        support_score = float(support_selected_agents[idx].item())
        record = _make_record(
            sample_id=f"diagnostic:{slice_id}:{target:.2f}",
            slice_item=slice_item,
            target_difficulty=float(target),
            target_behavior=float(conditioning_values[idx]),
            conditioning_behavior_input=float(conditioning_values[idx]),
            generated_future=generated[idx].cpu(),
            support_full_scene=float(support_full_scene[idx].item()),
            support_selected_agents=support_score,
            retrieval_score=1.0,
            behavior_normalizer=behavior_normalizer,
            normalizer_selected=normalizer_selected,
            normalizer_full_scene=normalizer_full_scene,
            config=config,
            checkpoint_info=checkpoint_info,
            generation_mode="same_scene_diagnostic",
            is_same_scene_diagnostic=True,
        )
        record["low_support"] = support_score < float(config["retrieval"]["min_support_for_main_eval"])
        records.append(record)
    diagnostics = _final_sampling_diagnostics(
        records,
        {
            "same_scene": _collect_group_diagnostics(
                model=model,
                batch=batch,
                slice_items=[slice_item] * len(target_difficulties),
                sampled_future=generated,
                target_behavior=conditioning_tensor,
                behavior_normalizer=behavior_normalizer,
                config=config,
            )
        },
    )
    return records, diagnostics, conditioning_trace


def retrieval_replay_baseline(
    config: Dict,
    project_root: str | Path,
    target_difficulties: Sequence[float],
    location_id: str | None = None,
    map_id: str | None = None,
    behavior_range: Tuple[float, float] | None = None,
    num_slices: int | None = None,
    num_rollouts_per_slice: int = 1,
    checkpoint_info: Dict[str, Any] | None = None,
    generation_mode: str = "retrieval_replay",
    retrieval_strategy: str = "default",
    seed: int = 0,
) -> Tuple[List[Dict], Dict[str, object], List[Dict[str, object]]]:
    retriever = build_retriever(config, project_root, split="train")
    normalizer_selected = load_difficulty_normalizer(
        Path(project_root) / config["data"]["cache_dir"] / "difficulty_stats.json",
        label_space="selected_agents",
    )
    behavior_normalizer = load_behavior_normalizer(
        Path(project_root) / config["data"]["cache_dir"] / "behavior_stats_selected_agents.json"
    )
    normalizer_full_scene = load_difficulty_normalizer(
        Path(project_root) / config["data"]["cache_dir"] / "difficulty_stats.json",
        label_space="full_scene",
    )
    records = []
    conditioning_trace: List[Dict[str, object]] = []
    num_slices = int(num_slices or config["sampling"]["num_samples_per_difficulty"])
    num_rollouts_per_slice = max(int(num_rollouts_per_slice), 1)
    cache_dir = Path(project_root) / config["data"]["cache_dir"]
    generator_condition_key = resolve_generator_condition_score_key(config)
    for target in tqdm(target_difficulties, desc="Preparing replay baseline", unit="target"):
        candidates = _retrieve_candidates(
            retriever,
            target_difficulty=float(target),
            location_id=location_id,
            map_id=map_id,
            behavior_range=behavior_range,
            seed=int(seed + round(float(target) * 1000)),
            strategy=retrieval_strategy,
        )
        selected = retriever.sample_diverse(
            candidates,
            num_samples=min(num_slices, len(candidates)),
            diversity_seed=int(seed + round(float(target) * 1000)),
        )
        selected_items = _materialize_sampling_slice_items([retriever.slice_by_id[item.slice_id] for item in selected], cache_dir)
        for rollout_idx in range(num_rollouts_per_slice):
            for idx, candidate in enumerate(tqdm(selected, desc=f"Replay target={target:.2f}", unit="sample", leave=False)):
                slice_item = selected_items[idx]
                retrieved_behavior = _required_behavior_value(slice_item, generator_condition_key)
                record = _make_record(
                    sample_id=f"{generation_mode}:{target:.2f}:{idx}:rollout{rollout_idx}",
                    slice_item=slice_item,
                    target_difficulty=float(target),
                    target_behavior=retrieved_behavior,
                    conditioning_behavior_input=retrieved_behavior,
                    generated_future=torch.as_tensor(slice_item["future_states"]).float(),
                    support_full_scene=candidate.support_full_scene,
                    support_selected_agents=candidate.support_selected_agents,
                    retrieval_score=candidate.retrieval_score,
                    behavior_normalizer=behavior_normalizer,
                    normalizer_selected=normalizer_selected,
                    normalizer_full_scene=normalizer_full_scene,
                    config=config,
                    checkpoint_info=checkpoint_info,
                    generation_mode=generation_mode,
                    is_same_scene_diagnostic=False,
                )
                records.append(record)
                conditioning_trace.append(
                    {
                        "requested_target_difficulty": float(target),
                        "requested_target_behavior": float(retrieved_behavior),
                        "model_input_target_difficulty": float(target),
                        "model_input_target_behavior": float(retrieved_behavior),
                        "generator_condition_score_key": generator_condition_key,
                        "retrieved_slice_behavior": float(retrieved_behavior),
                        "retrieved_behavior_quantile": _required_behavior_value(slice_item, "behavior_quantile_score_selected_agents"),
                        "retrieved_behavior_aggressiveness": _required_behavior_value(slice_item, "behavior_aggressiveness_score_selected_agents"),
                        "retrieved_behavior_raw_score": _required_behavior_value(slice_item, "behavior_raw_score_selected_agents"),
                        "retrieved_slice_stress_difficulty_selected": float(record.get("retrieved_selected_agents_difficulty", 0.0)),
                        "retrieved_slice_stress_difficulty_full": float(record.get("retrieved_full_scene_difficulty", 0.0)),
                        "support_selected_agents": float(candidate.support_selected_agents),
                        "support_full_scene": float(candidate.support_full_scene),
                        "condition_embedding_norm": 0.0,
                        "condition_embedding_first_5_values": [],
                        "seed": int(seed + rollout_idx),
                        "slice_id": str(slice_item["slice_id"]),
                    }
                )
    return records, _final_sampling_diagnostics(records, {}), conditioning_trace


def dataset_replay_baseline(
    config: Dict,
    project_root: str | Path,
    target_difficulties: Sequence[float] | None = None,
) -> Tuple[List[Dict], Dict[str, object], List[Dict[str, object]]]:
    targets = list(target_difficulties or config["sampling"]["target_difficulties"])
    return retrieval_replay_baseline(
        config,
        project_root,
        target_difficulties=targets,
        num_slices=min(int(config["analysis"]["num_visualizations"]), int(config["sampling"]["num_samples_per_difficulty"])),
        num_rollouts_per_slice=1,
        generation_mode="dataset_replay_baseline",
        seed=0,
    )


def random_retrieval_replay(
    config: Dict,
    project_root: str | Path,
    target_difficulties: Sequence[float],
    location_id: str | None = None,
    map_id: str | None = None,
    behavior_range: Tuple[float, float] | None = None,
    num_slices: int | None = None,
    num_rollouts_per_slice: int = 1,
    seed: int = 0,
) -> Tuple[List[Dict], Dict[str, object], List[Dict[str, object]]]:
    return retrieval_replay_baseline(
        config,
        project_root,
        target_difficulties=target_difficulties,
        location_id=location_id,
        map_id=map_id,
        behavior_range=behavior_range,
        num_slices=num_slices,
        num_rollouts_per_slice=num_rollouts_per_slice,
        retrieval_strategy="random",
        generation_mode="random_retrieval_replay",
        seed=seed,
    )


def retrieval_unconditional_generation(
    config: Dict,
    project_root: str | Path,
    target_difficulties: Sequence[float],
    location_id: str | None = None,
    map_id: str | None = None,
    behavior_range: Tuple[float, float] | None = None,
    num_slices: int | None = None,
    num_rollouts_per_slice: int = 1,
    checkpoint: str | Path | None = None,
    seed: int = 0,
) -> Tuple[List[Dict], Dict[str, object], List[Dict[str, object]]]:
    return sample_retrieval_augmented_generation(
        config,
        project_root,
        target_difficulties=target_difficulties,
        location_id=location_id,
        map_id=map_id,
        behavior_range=behavior_range,
        num_slices=num_slices,
        num_rollouts_per_slice=num_rollouts_per_slice,
        generator_target_behavior=0.0,
        checkpoint=checkpoint,
        generation_mode="retrieval_unconditional_generation",
        seed=seed,
    )


def retrieval_behavior_conditioned_generation(
    config: Dict,
    project_root: str | Path,
    target_difficulties: Sequence[float],
    location_id: str | None = None,
    map_id: str | None = None,
    behavior_range: Tuple[float, float] | None = None,
    num_slices: int | None = None,
    num_rollouts_per_slice: int = 1,
    generator_target_behavior: float | None = None,
    checkpoint: str | Path | None = None,
    seed: int = 0,
) -> Tuple[List[Dict], Dict[str, object], List[Dict[str, object]]]:
    return sample_retrieval_augmented_generation(
        config,
        project_root,
        target_difficulties=target_difficulties,
        location_id=location_id,
        map_id=map_id,
        behavior_range=behavior_range,
        num_slices=num_slices,
        num_rollouts_per_slice=num_rollouts_per_slice,
        generator_target_behavior=generator_target_behavior,
        checkpoint=checkpoint,
        generation_mode="retrieval_behavior_conditioned_generation",
        seed=seed,
    )


def no_support_retrieval_generation(
    config: Dict,
    project_root: str | Path,
    target_difficulties: Sequence[float],
    location_id: str | None = None,
    map_id: str | None = None,
    behavior_range: Tuple[float, float] | None = None,
    num_slices: int | None = None,
    num_rollouts_per_slice: int = 1,
    generator_target_behavior: float | None = None,
    checkpoint: str | Path | None = None,
    seed: int = 0,
) -> Tuple[List[Dict], Dict[str, object], List[Dict[str, object]]]:
    return sample_retrieval_augmented_generation(
        config,
        project_root,
        target_difficulties=target_difficulties,
        location_id=location_id,
        map_id=map_id,
        behavior_range=behavior_range,
        num_slices=num_slices,
        num_rollouts_per_slice=num_rollouts_per_slice,
        generator_target_behavior=generator_target_behavior,
        checkpoint=checkpoint,
        generation_mode="no_support_retrieval_generation",
        retrieval_strategy="no_support",
        seed=seed,
    )
