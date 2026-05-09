from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from isgen.data.cache import load_cache_metadata, load_slices_from_cache
from isgen.data.slice_builder import materialize_cached_map_fields
from isgen.semantics.behavior import resolve_behavior_target_key


def resolve_dataset_history_settings(config_or_metadata: Dict) -> tuple[str, int, float, bool]:
    generator_input = config_or_metadata.get("generator_input", {})
    data_cfg = config_or_metadata.get("data", {})
    use_history = bool(generator_input.get("use_history", True))
    requested_mode = str(generator_input.get("history_mode", data_cfg.get("history_mode", "full_history")))
    if not use_history:
        history_mode = "current_only"
    elif requested_mode in {"current_only", "recent_k", "full_history", "history_dropout"}:
        history_mode = requested_mode
    else:
        history_mode = str(data_cfg.get("history_mode", "full_history"))
    return (
        history_mode,
        int(data_cfg.get("recent_history_frames", 1)),
        float(data_cfg.get("history_dropout_prob", 0.0)),
        use_history,
    )


def apply_history_mode(
    history_states: np.ndarray,
    history_mask: np.ndarray,
    mode: str,
    recent_k: int,
    dropout_prob: float,
    training: bool,
) -> tuple[np.ndarray, np.ndarray]:
    states = history_states.copy()
    mask = history_mask.copy()
    if mode == "full_history":
        return states, mask
    if mode == "recent_k":
        keep = np.zeros_like(mask)
        keep[:, -recent_k:] = mask[:, -recent_k:]
        states[~keep] = 0.0
        return states, keep
    if mode == "current_only":
        keep = np.zeros_like(mask)
        keep[:, -1] = mask[:, -1]
        states[~keep] = 0.0
        return states, keep
    if mode == "history_dropout":
        if training:
            random_drop = np.random.rand(*mask.shape) < dropout_prob
            random_drop[:, -1] = False
            keep = mask & (~random_drop)
            states[~keep] = 0.0
            return states, keep
        return states, mask
    raise ValueError(f"Unsupported history_mode: {mode}")


class ScenarioSliceDataset(Dataset):
    def __init__(
        self,
        cache_dir: str,
        split: str,
        history_mode: str,
        recent_history_frames: int,
        history_dropout_prob: float,
        training: bool = False,
        slices: Optional[List[Dict]] = None,
        target_score_key: Optional[str] = None,
        use_history: bool = True,
    ) -> None:
        self.cache_dir = cache_dir
        self.split = split
        self.training = training
        self.history_mode = history_mode
        self.recent_history_frames = recent_history_frames
        self.history_dropout_prob = history_dropout_prob
        self.use_history = bool(use_history)
        self.slices = slices if slices is not None else load_slices_from_cache(cache_dir, split, materialize_maps=False)
        metadata = load_cache_metadata(cache_dir)
        self.data_root = metadata.get("data_root")
        self.max_polyline_points = int(metadata.get("map_materialization", {}).get("max_polyline_points", 20))
        synthetic_cfg = {"behavior": {"target_score_key": target_score_key or metadata.get("behavior_target_key", "behavior_quantile_score_selected_agents")}}
        self.target_score_key = resolve_behavior_target_key(synthetic_cfg)
        cached_target_key = metadata.get("behavior_target_key")
        if cached_target_key is not None and cached_target_key != self.target_score_key:
            raise RuntimeError(
                f"Cache behavior_target_key mismatch for split '{split}': "
                f"cache={cached_target_key} current={self.target_score_key}. "
                "Rerun prepare_data for this experiment cache before training."
            )
        missing_slice_id = next(
            (
                slice_item.get("slice_id", str(idx))
                for idx, slice_item in enumerate(self.slices)
                if self.target_score_key not in slice_item
            ),
            None,
        )
        if missing_slice_id is not None:
            raise KeyError(
                f"Cache split '{split}' is missing behavior target field '{self.target_score_key}' "
                f"(first missing slice: {missing_slice_id}). "
                "Rerun prepare_data for this experiment cache before training."
            )

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str | float]:
        item = self.slices[index]
        if self.data_root and "map_polylines" not in item and "map_polyline_indices" in item:
            item = materialize_cached_map_fields(item, self.data_root, self.max_polyline_points)
            self.slices[index] = item
        history_states, history_mask = apply_history_mode(
            history_states=item["history_states"],
            history_mask=item["history_mask"],
            mode=self.history_mode,
            recent_k=self.recent_history_frames,
            dropout_prob=self.history_dropout_prob,
            training=self.training,
        )
        output: Dict[str, torch.Tensor | str | float] = {
            "slice_id": item["slice_id"],
            "split": item["split"],
            "location_id": item["location_id"],
            "scenario_id": item["scenario_id"],
            "case_id": item["case_id"],
            "center_timestamp": float(item["center_timestamp"]),
            "focal_agent_id": float(item["focal_agent_id"]),
            "agent_ids": torch.as_tensor(item["agent_ids"], dtype=torch.long),
            "agent_types": torch.as_tensor(item["agent_types"], dtype=torch.long),
            "agent_sizes": torch.as_tensor(item["agent_sizes"], dtype=torch.float32),
            "history_states": torch.as_tensor(history_states, dtype=torch.float32),
            "history_mask": torch.as_tensor(history_mask, dtype=torch.bool),
            "current_states": torch.as_tensor(item["current_states"], dtype=torch.float32),
            "future_states": torch.as_tensor(item["future_states"], dtype=torch.float32),
            "future_mask": torch.as_tensor(item["future_mask"], dtype=torch.bool),
            "map_polylines": torch.as_tensor(item["map_polylines"], dtype=torch.float32),
            "map_point_mask": torch.as_tensor(item["map_point_mask"], dtype=torch.bool),
            "map_polyline_mask": torch.as_tensor(item["map_polyline_mask"], dtype=torch.bool),
            "agent_mask": torch.as_tensor(item["agent_mask"], dtype=torch.bool),
            "world_origin": torch.as_tensor(item["world_origin"], dtype=torch.float32),
            "world_heading": torch.as_tensor(item["world_heading"], dtype=torch.float32),
            "difficulty_score_selected_agents": torch.as_tensor(item.get("difficulty_score_selected_agents", item.get("difficulty_score", 0.0)), dtype=torch.float32),
            "difficulty_score_full_scene": torch.as_tensor(item.get("difficulty_score_full_scene", item.get("difficulty_score", 0.0)), dtype=torch.float32),
            "difficulty_score": torch.as_tensor(item.get("difficulty_score_selected_agents", item.get("difficulty_score", 0.0)), dtype=torch.float32),
            "behavior_aggressiveness_score_selected_agents": torch.as_tensor(
                item.get("behavior_aggressiveness_score_selected_agents", 0.0), dtype=torch.float32
            ),
            "behavior_quantile_score_selected_agents": torch.as_tensor(
                item.get("behavior_quantile_score_selected_agents", 0.0), dtype=torch.float32
            ),
            "behavior_raw_score_selected_agents": torch.as_tensor(
                item.get("behavior_raw_score_selected_agents", 0.0), dtype=torch.float32
            ),
            "retrieval_embedding": torch.as_tensor(
                item.get("retrieval_embedding", np.zeros(8, dtype=np.float32)), dtype=torch.float32
            ),
        }
        output["target_behavior"] = torch.as_tensor(item[self.target_score_key], dtype=torch.float32)
        output["behavior_target_source_key"] = self.target_score_key
        output["difficulty_features_selected_agents"] = item.get("difficulty_features_selected_agents", item.get("difficulty_features", {}))
        output["difficulty_features_full_scene"] = item.get("difficulty_features_full_scene", {})
        output["difficulty_features"] = item.get("difficulty_features_selected_agents", item.get("difficulty_features", {}))
        output["difficulty_level_selected_agents"] = item.get("difficulty_level_selected_agents", item.get("difficulty_level", "unknown"))
        output["difficulty_level_full_scene"] = item.get("difficulty_level_full_scene", "unknown")
        output["difficulty_level"] = item.get("difficulty_level_selected_agents", item.get("difficulty_level", "unknown"))
        output["behavior_features_selected_agents"] = item.get("behavior_features_selected_agents", {})
        output["behavior_aggressiveness_level_selected_agents"] = item.get("behavior_aggressiveness_level_selected_agents", "unknown")
        output["generator_used_history"] = self.use_history
        output["history_available_in_retrieved_slice"] = bool(np.asarray(item["history_mask"], dtype=bool).any())
        output["history_used_for_generation"] = self.use_history
        output["current_state_used"] = True
        output["metadata"] = item.get("metadata", {})
        return output
