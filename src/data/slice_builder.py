from __future__ import annotations

import logging
import os
import hashlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from isgen import ensure_dir
from isgen.data import MapData
from isgen.data.map_reader import read_interaction_map
from isgen.data.transforms import rotation_matrix, wrap_angle
from isgen.semantics.behavior import BEHAVIOR_CODE_VERSION, compute_behavior_aggressiveness_features
from isgen.semantics.difficulty import compute_difficulty_features
from isgen.semantics.difficulty import DIFFICULTY_CODE_VERSION
from isgen.data.interaction_reader import INTERACTION_READER_VERSION

LOGGER = logging.getLogger(__name__)
SLICE_BUILDER_VERSION = "2026-04-20-scene-centric-shards-v7"

STATE_DIM = 5
AGENT_TYPE_TO_ID = {
    "car": 0,
    "pedestrian/bicycle": 1,
    "pedestrian": 2,
    "bicycle": 3,
    "truck_bus": 4,
    "motorcycle": 5,
    "unknown": 6,
}


@dataclass
class SliceBuildResult:
    slices_by_split: Dict[str, List[Dict]]
    warnings: List[str]
    split_metadata: Dict[str, Dict[str, int]]
    build_profile_rows: List[Dict[str, Any]]
    build_performance: Dict[str, Any]


@dataclass
class DenseCaseData:
    case_id: str
    timestamps: np.ndarray
    track_ids: np.ndarray
    states: np.ndarray
    presence: np.ndarray
    agent_types: np.ndarray
    agent_sizes: np.ndarray
    present_tracks_by_time: List[np.ndarray]


@dataclass
class PreparedMap:
    stacked_polylines: np.ndarray
    point_masks: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    polyline_centers: np.ndarray
    point_xy: np.ndarray
    map_dim: int
    available: bool


@dataclass
class CenterInteractionCache:
    current_dist: np.ndarray
    current_rel_speed: np.ndarray
    current_ttc: np.ndarray
    current_density_score: np.ndarray
    future_min_dist: np.ndarray
    future_conflict_steps: np.ndarray
    agent_conflict_involvement: np.ndarray


@dataclass
class BuildFileProfile:
    file: str
    split: str
    location_id: str
    num_cases: int
    num_timestamps: int
    num_slices: int
    time_read_sec: float
    time_dense_sec: float
    time_interaction_matrix_sec: float
    time_agent_select_sec: float
    time_map_crop_sec: float
    time_feature_compute_sec: float
    time_write_sec: float
    total_time_sec: float
    cache_hit: bool
    map_crop_cache_hits: int
    map_crop_cache_queries: int
    shard_path: str = ""
    shard_signature: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "file": self.file,
            "split": self.split,
            "location_id": self.location_id,
            "num_cases": int(self.num_cases),
            "num_timestamps": int(self.num_timestamps),
            "num_slices": int(self.num_slices),
            "time_read_sec": float(self.time_read_sec),
            "time_dense_sec": float(self.time_dense_sec),
            "time_interaction_matrix_sec": float(self.time_interaction_matrix_sec),
            "time_agent_select_sec": float(self.time_agent_select_sec),
            "time_map_crop_sec": float(self.time_map_crop_sec),
            "time_feature_compute_sec": float(self.time_feature_compute_sec),
            "time_write_sec": float(self.time_write_sec),
            "total_time_sec": float(self.total_time_sec),
            "cache_hit": bool(self.cache_hit),
            "map_crop_cache_hits": int(self.map_crop_cache_hits),
            "map_crop_cache_queries": int(self.map_crop_cache_queries),
            "shard_path": str(self.shard_path),
        }
        if self.shard_signature is not None:
            payload["shard_signature"] = self.shard_signature
        return payload


def _resolve_preprocess_workers(value: object | None) -> int:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "auto":
            return max(1, min(8, (os.cpu_count() or 1)))
        try:
            value = int(lowered)
        except ValueError:
            return 1
    if value is None:
        return 1
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1


def _resolve_slice_builder_config(config: Dict) -> Dict[str, Any]:
    data_cfg = config.get("data", {})
    raw_cfg = dict(config.get("slice_builder", {}))
    resolved = {
        "mode": str(raw_cfg.get("mode", "scene_centric")),
        "slice_stride_frames": int(raw_cfg.get("slice_stride_frames", data_cfg.get("slice_stride", 1))),
        "max_focals_per_timestamp": int(raw_cfg.get("max_focals_per_timestamp", max(1, int(data_cfg.get("max_focal_agents_per_frame", 1) or 1)))),
        "focal_selection": str(raw_cfg.get("focal_selection", "most_interactive")),
        "max_slices_per_file": raw_cfg.get("max_slices_per_file"),
        "min_future_valid_ratio": float(raw_cfg.get("min_future_valid_ratio", 0.8)),
        "min_history_valid_ratio": float(raw_cfg.get("min_history_valid_ratio", 0.3)),
    }
    if resolved["mode"] not in {"scene_centric", "limited_focal", "all_focal"}:
        raise ValueError(f"Unsupported slice_builder.mode: {resolved['mode']}")
    if resolved["slice_stride_frames"] <= 0:
        raise ValueError("slice_builder.slice_stride_frames must be >= 1")
    if resolved["max_slices_per_file"] is not None:
        resolved["max_slices_per_file"] = int(resolved["max_slices_per_file"])
    return resolved


def _resolve_map_processing_config(config: Dict) -> Dict[str, Any]:
    raw_cfg = dict(config.get("map", {}))
    return {
        "crop_cache_grid_m": float(raw_cfg.get("crop_cache_grid_m", 5.0)),
        "save_map_polyline_indices_only": bool(raw_cfg.get("save_map_polyline_indices_only", True)),
    }


def _resolve_build_processing_config(config: Dict) -> Dict[str, Any]:
    processing_cfg = dict(config.get("processing", {}))
    return {
        "num_build_workers": _resolve_preprocess_workers(processing_cfg.get("num_build_workers", config.get("data", {}).get("preprocess_num_workers", "auto"))),
        "build_worker_chunksize": max(1, int(processing_cfg.get("build_worker_chunksize", 1))),
        "reuse_slice_shards": bool(processing_cfg.get("reuse_slice_shards", True)),
        "shard_cache_dir": str(processing_cfg.get("shard_cache_dir", Path(config.get("data", {}).get("cache_dir", "outputs/cache")) / "slice_shards")),
        "include_background_future_for_proxy": bool(processing_cfg.get("include_background_future_for_proxy", True)),
    }


def _scene_summary_from_arrays(current_states: np.ndarray, agent_mask: np.ndarray, interaction_radius_m: float) -> Dict[str, float]:
    valid_states = current_states[agent_mask]
    if len(valid_states) == 0:
        return {"num_agents": 0.0, "mean_speed": 0.0, "density": 0.0}
    speed = np.linalg.norm(valid_states[:, 2:4], axis=-1)
    density = float(np.sum(np.linalg.norm(valid_states[:, 0:2], axis=-1) <= interaction_radius_m))
    return {"num_agents": float(len(valid_states)), "mean_speed": float(speed.mean()), "density": density}


def _select_background_proxy_positions(
    dense_case: DenseCaseData,
    center_idx: int,
    focal_track_index: int,
    present_tracks: np.ndarray,
    full_ordered_tracks: np.ndarray,
    background_positions: np.ndarray,
    interaction_cache: CenterInteractionCache,
    max_background_agents: int,
) -> np.ndarray:
    if max_background_agents <= 0 or background_positions.size <= max_background_agents:
        return background_positions
    focal_xy = dense_case.states[focal_track_index, center_idx, 0:2]
    background_track_indices = full_ordered_tracks[background_positions]
    background_xy = dense_case.states[background_track_indices, center_idx, 0:2]
    current_distance = np.linalg.norm(background_xy - focal_xy[None, :], axis=-1)
    present_lookup = {int(track): idx for idx, track in enumerate(present_tracks.tolist())}
    conflict_involvement = np.asarray(
        [
            float(interaction_cache.agent_conflict_involvement[present_lookup.get(int(track_idx), 0)])
            if int(track_idx) in present_lookup
            else 0.0
            for track_idx in background_track_indices
        ],
        dtype=np.float32,
    )
    # Prefer conflict-involved and nearby background agents for the auxiliary full-scene proxy.
    score = 4.0 * conflict_involvement - 0.05 * current_distance
    keep_order = np.argsort(-score, kind="stable")[:max_background_agents]
    return background_positions[np.sort(keep_order)]


def _agent_type_id(agent_type: str) -> int:
    return AGENT_TYPE_TO_ID.get(agent_type.lower(), AGENT_TYPE_TO_ID["unknown"])


def _prepare_dense_case(case_df: pd.DataFrame) -> DenseCaseData:
    case_df = case_df.sort_values(["timestamp_ms", "track_id"]).copy()
    timestamps = np.sort(case_df["timestamp_ms"].unique()).astype(np.int64)
    track_codes, track_ids = pd.factorize(case_df["track_id"], sort=True)
    track_ids = np.asarray(track_ids, dtype=np.int64)
    time_index = np.searchsorted(timestamps, case_df["timestamp_ms"].to_numpy(dtype=np.int64))
    states = np.zeros((len(track_ids), len(timestamps), STATE_DIM), dtype=np.float32)
    presence = np.zeros((len(track_ids), len(timestamps)), dtype=bool)
    state_values = case_df[["x", "y", "vx", "vy", "heading"]].to_numpy(dtype=np.float32)
    states[track_codes, time_index] = state_values
    presence[track_codes, time_index] = True
    meta_df = case_df.sort_values(["track_id", "timestamp_ms"]).drop_duplicates("track_id", keep="first")
    meta_track_index = np.searchsorted(track_ids, meta_df["track_id"].to_numpy(dtype=np.int64))
    agent_types = np.full(len(track_ids), AGENT_TYPE_TO_ID["unknown"], dtype=np.int64)
    agent_types[meta_track_index] = np.asarray([_agent_type_id(str(value)) for value in meta_df["agent_type"].tolist()], dtype=np.int64)
    agent_sizes = np.zeros((len(track_ids), 2), dtype=np.float32)
    agent_sizes[meta_track_index] = meta_df[["length", "width"]].to_numpy(dtype=np.float32)
    present_tracks_by_time = [np.flatnonzero(presence[:, time_idx]) for time_idx in range(len(timestamps))]
    return DenseCaseData(
        case_id=str(case_df["case_id"].iloc[0]),
        timestamps=timestamps,
        track_ids=track_ids,
        states=states,
        presence=presence,
        agent_types=agent_types,
        agent_sizes=agent_sizes,
        present_tracks_by_time=present_tracks_by_time,
    )


def _prepare_map(map_data: MapData, max_points: int) -> PreparedMap:
    if not map_data.polyline_list:
        return PreparedMap(
            stacked_polylines=np.zeros((0, max_points, 4), dtype=np.float32),
            point_masks=np.zeros((0, max_points), dtype=bool),
            bbox_min=np.zeros((0, 2), dtype=np.float32),
            bbox_max=np.zeros((0, 2), dtype=np.float32),
            polyline_centers=np.zeros((0, 2), dtype=np.float32),
            point_xy=np.zeros((0, max_points, 2), dtype=np.float32),
            map_dim=4,
            available=False,
        )
    max_dim = max(polyline.shape[-1] for polyline in map_data.polyline_list)
    num_polylines = len(map_data.polyline_list)
    stacked = np.zeros((num_polylines, max_points, max_dim), dtype=np.float32)
    point_masks = np.zeros((num_polylines, max_points), dtype=bool)
    bbox_min = np.zeros((num_polylines, 2), dtype=np.float32)
    bbox_max = np.zeros((num_polylines, 2), dtype=np.float32)
    centers = np.zeros((num_polylines, 2), dtype=np.float32)
    for idx, polyline in enumerate(map_data.polyline_list):
        limit = min(len(polyline), max_points)
        stacked[idx, :limit, : polyline.shape[-1]] = polyline[:limit]
        point_masks[idx, :limit] = True
        xy = stacked[idx, point_masks[idx], 0:2]
        bbox_min[idx] = xy.min(axis=0)
        bbox_max[idx] = xy.max(axis=0)
        centers[idx] = xy.mean(axis=0)
    return PreparedMap(
        stacked_polylines=stacked,
        point_masks=point_masks,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        polyline_centers=centers,
        point_xy=stacked[..., 0:2],
        map_dim=max_dim,
        available=map_data.map_available,
    )


def _normalize_positive(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=np.float32)
    clipped = values.copy()
    clipped[~finite] = 0.0
    max_value = float(clipped.max())
    if max_value <= 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return (clipped / max_value).astype(np.float32)


def _compute_center_interaction_cache(
    dense_case: DenseCaseData,
    center_idx: int,
    present_tracks: np.ndarray,
    future_frames: int,
    conflict_distance_m: float,
    ttc_max_sec: float,
    interaction_radius_m: float,
) -> CenterInteractionCache:
    num_present = len(present_tracks)
    current_states = dense_case.states[present_tracks, center_idx]
    rel_xy = current_states[:, None, 0:2] - current_states[None, :, 0:2]
    rel_vel = current_states[:, None, 2:4] - current_states[None, :, 2:4]
    current_dist = np.linalg.norm(rel_xy, axis=-1).astype(np.float32)
    current_rel_speed = np.linalg.norm(rel_vel, axis=-1).astype(np.float32)
    current_density_score = np.sum((current_dist < interaction_radius_m).astype(np.float32), axis=-1).astype(np.float32)
    closing_rate = -(rel_xy * rel_vel).sum(axis=-1)
    rel_speed_sq = np.clip(np.sum(rel_vel**2, axis=-1), 1e-3, None)
    current_ttc = np.where(
        closing_rate > 0.0,
        np.clip(closing_rate / rel_speed_sq, 0.0, ttc_max_sec),
        ttc_max_sec,
    ).astype(np.float32)
    np.fill_diagonal(current_ttc, ttc_max_sec)
    future_min_dist = np.full((num_present, num_present), np.inf, dtype=np.float32)
    future_conflict_steps = np.zeros((num_present, num_present), dtype=np.float32)
    valid_future_count = max(0, min(future_frames, len(dense_case.timestamps) - center_idx - 1))
    if valid_future_count > 0 and num_present > 1:
        future_indices = np.arange(center_idx + 1, center_idx + 1 + valid_future_count, dtype=np.int64)
        future_xy = dense_case.states[present_tracks][:, future_indices, 0:2]
        future_valid = dense_case.presence[present_tracks][:, future_indices]
        for future_offset in range(valid_future_count):
            present_mask = future_valid[:, future_offset]
            local_indices = np.flatnonzero(present_mask)
            if len(local_indices) <= 1:
                continue
            xy = future_xy[local_indices, future_offset]
            pair_dist = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1).astype(np.float32)
            np.fill_diagonal(pair_dist, np.inf)
            grid = np.ix_(local_indices, local_indices)
            future_min_dist[grid] = np.minimum(future_min_dist[grid], pair_dist)
            future_conflict_steps[grid] += (pair_dist < conflict_distance_m).astype(np.float32)
    np.fill_diagonal(current_dist, np.inf)
    np.fill_diagonal(future_min_dist, np.inf)
    np.fill_diagonal(future_conflict_steps, 0.0)
    agent_conflict_involvement = (future_conflict_steps > 0.0).sum(axis=-1).astype(np.float32)
    return CenterInteractionCache(
        current_dist=current_dist,
        current_rel_speed=current_rel_speed,
        current_ttc=current_ttc,
        current_density_score=current_density_score,
        future_min_dist=future_min_dist,
        future_conflict_steps=future_conflict_steps,
        agent_conflict_involvement=agent_conflict_involvement,
    )


def _transform_states_local(states: np.ndarray, origin_xy: np.ndarray, origin_heading: float, mask: np.ndarray | None = None) -> np.ndarray:
    local = states.copy()
    if mask is None:
        mask = np.ones(states.shape[:-1], dtype=bool)
    rotation = rotation_matrix(-origin_heading)
    flat_mask = mask.reshape(-1)
    flat_xy = states[..., 0:2].reshape(-1, 2)
    flat_vel = states[..., 2:4].reshape(-1, 2)
    flat_heading = states[..., 4].reshape(-1)
    transformed_xy = np.zeros_like(flat_xy)
    transformed_vel = np.zeros_like(flat_vel)
    transformed_heading = np.zeros_like(flat_heading)
    transformed_xy[flat_mask] = (flat_xy[flat_mask] - origin_xy[None, :]) @ rotation.T
    transformed_vel[flat_mask] = flat_vel[flat_mask] @ rotation.T
    transformed_heading[flat_mask] = wrap_angle(flat_heading[flat_mask] - origin_heading)
    local[..., 0:2] = transformed_xy.reshape(states[..., 0:2].shape)
    local[..., 2:4] = transformed_vel.reshape(states[..., 2:4].shape)
    local[..., 4] = transformed_heading.reshape(states[..., 4].shape)
    return local


def _materialize_map_from_indices(
    prepared_map: PreparedMap,
    chosen_idx: np.ndarray,
    focal_xy: np.ndarray,
    origin_heading: float,
    max_polylines: int,
    to_local: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not prepared_map.available or prepared_map.stacked_polylines.shape[0] == 0:
        return (
            np.zeros((max_polylines, prepared_map.point_masks.shape[1], prepared_map.map_dim), dtype=np.float32),
            np.zeros((max_polylines, prepared_map.point_masks.shape[1]), dtype=bool),
            np.zeros(max_polylines, dtype=bool),
        )
    point_count = prepared_map.point_masks.shape[1]
    selected_polylines = np.zeros((max_polylines, point_count, prepared_map.map_dim), dtype=np.float32)
    selected_masks = np.zeros((max_polylines, point_count), dtype=bool)
    polyline_mask = np.zeros(max_polylines, dtype=bool)
    if chosen_idx.size == 0:
        return selected_polylines, selected_masks, polyline_mask
    chosen_idx = np.asarray(chosen_idx, dtype=np.int64)[:max_polylines]
    k = int(len(chosen_idx))
    selected_polylines[:k] = prepared_map.stacked_polylines[chosen_idx]
    selected_masks[:k] = prepared_map.point_masks[chosen_idx]
    polyline_mask[:k] = True
    if to_local and k > 0:
        rotation = rotation_matrix(-origin_heading)
        xy = selected_polylines[:k, :, 0:2].reshape(-1, 2)
        xy = (xy - focal_xy[None, :]) @ rotation.T
        selected_polylines[:k, :, 0:2] = xy.reshape(k, point_count, 2)
        if prepared_map.map_dim > 2:
            selected_polylines[:k, :, 2] = wrap_angle(selected_polylines[:k, :, 2] - origin_heading)
    return selected_polylines, selected_masks, polyline_mask


def _crop_map_indices(
    prepared_map: PreparedMap,
    focal_xy: np.ndarray,
    radius_m: float,
    max_polylines: int,
) -> np.ndarray:
    if not prepared_map.available or prepared_map.stacked_polylines.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    dx = np.maximum(np.maximum(prepared_map.bbox_min[:, 0] - focal_xy[0], 0.0), focal_xy[0] - prepared_map.bbox_max[:, 0])
    dy = np.maximum(np.maximum(prepared_map.bbox_min[:, 1] - focal_xy[1], 0.0), focal_xy[1] - prepared_map.bbox_max[:, 1])
    bbox_distance = np.sqrt(dx * dx + dy * dy)
    candidate_idx = np.flatnonzero(bbox_distance <= radius_m)
    if len(candidate_idx) == 0:
        return np.zeros((0,), dtype=np.int64)
    point_diff = prepared_map.point_xy[candidate_idx] - focal_xy[None, None, :]
    point_distance = np.linalg.norm(point_diff, axis=-1)
    point_distance = np.where(prepared_map.point_masks[candidate_idx], point_distance, np.inf)
    polyline_distance = point_distance.min(axis=1)
    k = min(max_polylines, len(candidate_idx))
    nearest_local = np.argpartition(polyline_distance, kth=k - 1)[:k]
    nearest_local = nearest_local[np.argsort(polyline_distance[nearest_local], kind="stable")]
    return candidate_idx[nearest_local].astype(np.int64)


def _crop_map_indices_cached(
    prepared_map: PreparedMap,
    focal_xy: np.ndarray,
    radius_m: float,
    max_polylines: int,
    grid_size_m: float,
    cache: Dict[tuple[int, int], np.ndarray],
) -> tuple[np.ndarray, bool]:
    if grid_size_m <= 0.0:
        return _crop_map_indices(prepared_map, focal_xy, radius_m, max_polylines), False
    grid_key = (int(np.floor(focal_xy[0] / grid_size_m)), int(np.floor(focal_xy[1] / grid_size_m)))
    if grid_key in cache:
        return cache[grid_key], True
    chosen_idx = _crop_map_indices(prepared_map, focal_xy, radius_m, max_polylines)
    cache[grid_key] = chosen_idx
    return chosen_idx, False


def materialize_cached_map_fields(
    slice_item: Dict[str, Any],
    data_root: str | Path,
    max_polyline_points: int,
) -> Dict[str, Any]:
    if "map_polylines" in slice_item and "map_point_mask" in slice_item and "map_polyline_mask" in slice_item:
        return slice_item
    if "map_polyline_indices" not in slice_item:
        return slice_item
    prepared_map = _prepare_map(
        read_interaction_map(str(slice_item["location_id"]), data_root, target_points=max_polyline_points),
        max_points=max_polyline_points,
    )
    focal_xy = np.asarray(slice_item.get("map_center", slice_item.get("world_origin", np.zeros(2, dtype=np.float32))), dtype=np.float32)
    origin_heading = float(slice_item.get("map_heading", slice_item.get("world_heading", 0.0)))
    max_polylines = int(slice_item.get("map_max_polylines", slice_item.get("map_polyline_indices", np.zeros(0, dtype=np.int64)).shape[0] or 0))
    polylines, point_mask, polyline_mask = _materialize_map_from_indices(
        prepared_map=prepared_map,
        chosen_idx=np.asarray(slice_item.get("map_polyline_indices", np.zeros(0, dtype=np.int64)), dtype=np.int64),
        focal_xy=focal_xy,
        origin_heading=origin_heading,
        max_polylines=max_polylines,
        to_local=bool(slice_item.get("map_local_coordinates", slice_item.get("local_coordinates", True))),
    )
    enriched = dict(slice_item)
    enriched["map_polylines"] = polylines
    enriched["map_point_mask"] = point_mask
    enriched["map_polyline_mask"] = polyline_mask
    return enriched


def _crop_map(
    prepared_map: PreparedMap,
    focal_xy: np.ndarray,
    radius_m: float,
    max_polylines: int,
    origin_heading: float,
    to_local: bool,
    grid_size_m: float = 0.0,
    cache: Dict[tuple[int, int], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    chosen_idx, cache_hit = _crop_map_indices_cached(
        prepared_map=prepared_map,
        focal_xy=focal_xy,
        radius_m=radius_m,
        max_polylines=max_polylines,
        grid_size_m=grid_size_m,
        cache=cache if cache is not None else {},
    )
    polylines, point_mask, polyline_mask = _materialize_map_from_indices(
        prepared_map=prepared_map,
        chosen_idx=chosen_idx,
        focal_xy=focal_xy,
        origin_heading=origin_heading,
        max_polylines=max_polylines,
        to_local=to_local,
    )
    return polylines, point_mask, polyline_mask, chosen_idx, cache_hit


def _assemble_slice(
    dense_case: DenseCaseData,
    center_idx: int,
    focal_track_index: int,
    ordered_track_indices: np.ndarray,
    history_frames: int,
    future_frames: int,
    max_agents: int,
    local_coordinates: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    history_indices = np.arange(center_idx - history_frames + 1, center_idx + 1, dtype=np.int64)
    A = max_agents
    H = history_frames
    T = future_frames
    track_count = len(ordered_track_indices)
    agent_ids = np.full(A, -1, dtype=np.int64)
    agent_types = np.full(A, AGENT_TYPE_TO_ID["unknown"], dtype=np.int64)
    agent_sizes = np.zeros((A, 2), dtype=np.float32)
    history_states = np.zeros((A, H, STATE_DIM), dtype=np.float32)
    history_mask = np.zeros((A, H), dtype=bool)
    current_states = np.zeros((A, STATE_DIM), dtype=np.float32)
    future_states = np.zeros((A, T, STATE_DIM), dtype=np.float32)
    future_mask = np.zeros((A, T), dtype=bool)
    agent_mask = np.zeros(A, dtype=bool)
    world_origin = dense_case.states[focal_track_index, center_idx, 0:2].copy()
    world_heading = float(dense_case.states[focal_track_index, center_idx, 4])
    selected = ordered_track_indices[:track_count]
    agent_ids[:track_count] = dense_case.track_ids[selected]
    agent_types[:track_count] = dense_case.agent_types[selected]
    agent_sizes[:track_count] = dense_case.agent_sizes[selected]
    history_states[:track_count] = dense_case.states[selected][:, history_indices]
    history_mask[:track_count] = dense_case.presence[selected][:, history_indices]
    current_states[:track_count] = dense_case.states[selected, center_idx]
    if T > 0:
        valid_future_count = max(0, min(T, len(dense_case.timestamps) - center_idx - 1))
        if valid_future_count > 0:
            future_indices = np.arange(center_idx + 1, center_idx + 1 + valid_future_count, dtype=np.int64)
            future_states[:track_count, :valid_future_count] = dense_case.states[selected][:, future_indices]
            future_mask[:track_count, :valid_future_count] = dense_case.presence[selected][:, future_indices]
    agent_mask[:track_count] = dense_case.presence[selected, center_idx]
    if local_coordinates:
        history_states = _transform_states_local(history_states, world_origin, world_heading, mask=history_mask)
        current_states = _transform_states_local(current_states, world_origin, world_heading, mask=agent_mask)
        future_states = _transform_states_local(future_states, world_origin, world_heading, mask=future_mask)
    return (
        agent_ids,
        agent_types,
        agent_sizes,
        history_states,
        history_mask,
        current_states,
        future_states,
        future_mask,
        agent_mask,
        world_origin,
        world_heading,
    )


def _subselect_assembled_slice(
    agent_ids_full: np.ndarray,
    agent_types_full: np.ndarray,
    agent_sizes_full: np.ndarray,
    history_states_full: np.ndarray,
    history_mask_full: np.ndarray,
    current_states_full: np.ndarray,
    future_states_full: np.ndarray,
    future_mask_full: np.ndarray,
    agent_mask_full: np.ndarray,
    selected_positions: np.ndarray,
    max_agents: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    history_frames = history_states_full.shape[1]
    future_frames = future_states_full.shape[1]
    if max_agents <= 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, history_frames, STATE_DIM), dtype=np.float32),
            np.zeros((0, history_frames), dtype=bool),
            np.zeros((0, STATE_DIM), dtype=np.float32),
            np.zeros((0, future_frames, STATE_DIM), dtype=np.float32),
            np.zeros((0, future_frames), dtype=bool),
            np.zeros((0,), dtype=bool),
        )
    A = int(max_agents)
    chosen = np.asarray(selected_positions, dtype=np.int64)[:A]
    count = len(chosen)
    agent_ids = np.full(A, -1, dtype=np.int64)
    agent_types = np.full(A, AGENT_TYPE_TO_ID["unknown"], dtype=np.int64)
    agent_sizes = np.zeros((A, 2), dtype=np.float32)
    history_states = np.zeros((A, history_frames, STATE_DIM), dtype=np.float32)
    history_mask = np.zeros((A, history_frames), dtype=bool)
    current_states = np.zeros((A, STATE_DIM), dtype=np.float32)
    future_states = np.zeros((A, future_frames, STATE_DIM), dtype=np.float32)
    future_mask = np.zeros((A, future_frames), dtype=bool)
    agent_mask = np.zeros(A, dtype=bool)
    if count > 0:
        agent_ids[:count] = agent_ids_full[chosen]
        agent_types[:count] = agent_types_full[chosen]
        agent_sizes[:count] = agent_sizes_full[chosen]
        history_states[:count] = history_states_full[chosen]
        history_mask[:count] = history_mask_full[chosen]
        current_states[:count] = current_states_full[chosen]
        future_states[:count] = future_states_full[chosen]
        future_mask[:count] = future_mask_full[chosen]
        agent_mask[:count] = agent_mask_full[chosen]
    return (
        agent_ids,
        agent_types,
        agent_sizes,
        history_states,
        history_mask,
        current_states,
        future_states,
        future_mask,
        agent_mask,
    )


def _valid_ratio(
    dense_case: DenseCaseData,
    track_indices: np.ndarray,
    time_indices: np.ndarray,
) -> np.ndarray:
    if len(track_indices) == 0:
        return np.zeros((0,), dtype=np.float32)
    if time_indices.size == 0:
        return np.ones((len(track_indices),), dtype=np.float32)
    valid = dense_case.presence[track_indices][:, time_indices]
    return valid.mean(axis=1).astype(np.float32)


def _scene_interaction_scores(
    interaction_cache: CenterInteractionCache,
    config: Dict,
) -> np.ndarray:
    weights = dict(
        config["data"].get(
            "selection_weights",
            {
                "distance": 0.35,
                "relative_speed": 0.15,
                "ttc": 0.2,
                "future_min_distance": 0.2,
                "focal_conflict": 0.25,
                "global_conflict_involvement": 0.15,
            },
        )
    )
    nearest_distance = np.min(interaction_cache.current_dist, axis=1)
    nearest_ttc = np.min(interaction_cache.current_ttc, axis=1)
    nearest_future_distance = np.min(interaction_cache.future_min_dist, axis=1)
    relative_speed_max = np.max(interaction_cache.current_rel_speed, axis=1)
    score = (
        float(weights.get("distance", 0.35)) * _normalize_positive(1.0 / np.clip(nearest_distance, 1.0, None))
        + float(weights.get("relative_speed", 0.15)) * _normalize_positive(relative_speed_max)
        + float(weights.get("ttc", 0.2))
        * _normalize_positive(
            np.where(
                nearest_ttc < float(config["difficulty"]["ttc_max_sec"]),
                1.0 / np.clip(nearest_ttc + 0.1, 0.1, None),
                0.0,
            )
        )
        + float(weights.get("future_min_distance", 0.2)) * _normalize_positive(1.0 / np.clip(nearest_future_distance, 1.0, None))
        + float(weights.get("focal_conflict", 0.25)) * _normalize_positive(interaction_cache.agent_conflict_involvement)
        + float(weights.get("global_conflict_involvement", 0.15)) * _normalize_positive(interaction_cache.current_density_score)
    )
    return score.astype(np.float32)


def _select_scene_anchor_and_focals(
    dense_case: DenseCaseData,
    center_idx: int,
    present_tracks: np.ndarray,
    interaction_cache: CenterInteractionCache,
    config: Dict,
    split: str,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    builder_cfg = _resolve_slice_builder_config(config)
    history_frames = int(config["data"]["history_frames"])
    future_frames = int(config["data"]["future_frames"])
    history_indices = np.arange(center_idx - history_frames + 1, center_idx + 1, dtype=np.int64)
    history_valid_ratio = _valid_ratio(dense_case, present_tracks, history_indices)
    valid_future_count = max(0, min(future_frames, len(dense_case.timestamps) - center_idx - 1))
    if valid_future_count > 0:
        future_indices = np.arange(center_idx + 1, center_idx + 1 + valid_future_count, dtype=np.int64)
        future_valid_ratio = _valid_ratio(dense_case, present_tracks, future_indices)
    else:
        future_valid_ratio = np.ones((len(present_tracks),), dtype=np.float32) if split == "test" else np.zeros((len(present_tracks),), dtype=np.float32)
    eligible = history_valid_ratio >= float(builder_cfg["min_history_valid_ratio"])
    if split != "test":
        eligible &= future_valid_ratio >= float(builder_cfg["min_future_valid_ratio"])
    if not np.any(eligible):
        eligible = np.ones((len(present_tracks),), dtype=bool)
    interaction_scores = _scene_interaction_scores(interaction_cache, config)
    eligible_positions = np.flatnonzero(eligible)
    anchor_pos = int(eligible_positions[np.argmax(interaction_scores[eligible_positions])]) if eligible_positions.size else 0
    mode = str(builder_cfg["mode"])
    if mode == "scene_centric":
        focal_positions = np.asarray([anchor_pos], dtype=np.int64)
    elif mode == "limited_focal":
        k = max(1, int(builder_cfg["max_focals_per_timestamp"]))
        selection = str(builder_cfg["focal_selection"])
        candidate_positions = eligible_positions
        if selection == "highest_speed":
            speeds = np.linalg.norm(dense_case.states[present_tracks, center_idx, 2:4], axis=-1)
            focal_positions = candidate_positions[np.argsort(-speeds[candidate_positions], kind="stable")[:k]]
        elif selection == "highest_conflict":
            focal_positions = candidate_positions[np.argsort(-interaction_cache.agent_conflict_involvement[candidate_positions], kind="stable")[:k]]
        elif selection == "random_valid":
            rng = np.random.default_rng(int(center_idx + len(present_tracks) * 17))
            permuted = candidate_positions.copy()
            rng.shuffle(permuted)
            focal_positions = permuted[:k]
        else:
            focal_positions = candidate_positions[np.argsort(-interaction_scores[candidate_positions], kind="stable")[:k]]
        if anchor_pos not in focal_positions:
            focal_positions = np.concatenate([np.asarray([anchor_pos], dtype=np.int64), focal_positions])[:k]
    else:
        focal_tracks = _select_focal_track_indices(dense_case, center_idx, config)
        focal_positions = np.asarray(
            [int(np.where(present_tracks == track_idx)[0][0]) for track_idx in focal_tracks if np.any(present_tracks == track_idx)],
            dtype=np.int64,
        )
        if focal_positions.size == 0:
            focal_positions = np.asarray([anchor_pos], dtype=np.int64)
    focal_positions = np.unique(focal_positions.astype(np.int64))
    return anchor_pos, focal_positions, interaction_scores, history_valid_ratio, future_valid_ratio


def _select_scene_context_tracks(
    dense_case: DenseCaseData,
    center_idx: int,
    present_tracks: np.ndarray,
    anchor_position: int,
    interaction_scores: np.ndarray,
    history_valid_ratio: np.ndarray,
    future_valid_ratio: np.ndarray,
    config: Dict,
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    max_agents = int(config["data"]["max_agents"])
    anchor_track = int(present_tracks[anchor_position])
    anchor_xy = dense_case.states[anchor_track, center_idx, 0:2]
    current_xy = dense_case.states[present_tracks, center_idx, 0:2]
    distance_to_anchor = np.linalg.norm(current_xy - anchor_xy[None, :], axis=-1).astype(np.float32)
    sort_order = np.lexsort(
        (
            distance_to_anchor,
            -future_valid_ratio,
            -history_valid_ratio,
            -interaction_scores,
        )
    )
    chosen_positions: List[int] = [anchor_position]
    reasons: List[str] = ["scene_anchor"]
    for position in sort_order:
        position = int(position)
        if position == anchor_position or position in chosen_positions:
            continue
        if len(chosen_positions) >= max_agents:
            break
        chosen_positions.append(position)
        if future_valid_ratio[position] >= history_valid_ratio[position] and future_valid_ratio[position] > 0.0:
            reasons.append("top_interaction_future_valid")
        elif interaction_scores[position] > 0.0:
            reasons.append("top_interaction")
        else:
            reasons.append("distance_fill")
    selected_positions = np.asarray(chosen_positions, dtype=np.int64)
    return present_tracks[selected_positions], interaction_scores[selected_positions], reasons


def _select_context_tracks(
    dense_case: DenseCaseData,
    center_idx: int,
    focal_track_index: int,
    present_tracks: np.ndarray,
    interaction_cache: CenterInteractionCache,
    config: Dict,
) -> np.ndarray:
    if len(present_tracks) <= 1:
        return present_tracks
    max_agents = int(config["data"]["max_agents"])
    weights = dict(
        config["data"].get(
            "selection_weights",
            {
                "distance": 0.35,
                "relative_speed": 0.15,
                "ttc": 0.2,
                "future_min_distance": 0.2,
                "focal_conflict": 0.25,
                "global_conflict_involvement": 0.15,
            },
        )
    )
    focal_position = int(np.where(present_tracks == focal_track_index)[0][0])
    current_dist = interaction_cache.current_dist[focal_position].copy()
    rel_speed = interaction_cache.current_rel_speed[focal_position].copy()
    ttc = interaction_cache.current_ttc[focal_position].copy()
    ttc_risk = np.where(
        ttc < float(config["difficulty"]["ttc_max_sec"]),
        1.0 / np.clip(ttc + 0.1, 0.1, None),
        0.0,
    ).astype(np.float32)
    future_min_dist = interaction_cache.future_min_dist[focal_position].copy()
    focal_conflict = interaction_cache.future_conflict_steps[focal_position].copy()
    global_conflict = interaction_cache.agent_conflict_involvement.copy()
    risk_score = (
        float(weights.get("distance", 0.35)) * _normalize_positive(1.0 / np.clip(current_dist, 1.0, None))
        + float(weights.get("relative_speed", 0.15)) * _normalize_positive(rel_speed)
        + float(weights.get("ttc", 0.2)) * _normalize_positive(ttc_risk)
        + float(weights.get("future_min_distance", 0.2)) * _normalize_positive(1.0 / np.clip(future_min_dist, 1.0, None))
        + float(weights.get("focal_conflict", 0.25)) * _normalize_positive(focal_conflict)
        + float(weights.get("global_conflict_involvement", 0.15)) * _normalize_positive(global_conflict)
    )
    risk_score[focal_position] = np.inf
    tie_break_distance = np.where(np.isfinite(current_dist), current_dist, 1e6)
    candidate_positions = np.arange(len(present_tracks))
    non_focal = candidate_positions[candidate_positions != focal_position]
    order = np.lexsort((tie_break_distance[non_focal], -risk_score[non_focal]))
    selected_positions = np.concatenate(
        [
            np.asarray([focal_position], dtype=np.int64),
            non_focal[order],
        ]
    )[:max_agents]
    return present_tracks[selected_positions]


def _selection_coverage_metrics(
    selected_features: Dict[str, float],
    full_features: Dict[str, float],
    selected_tracks: np.ndarray,
    present_tracks: np.ndarray,
    interaction_cache: CenterInteractionCache,
) -> Dict[str, float]:
    selected_set = {int(track) for track in selected_tracks.tolist()}
    conflict_agents = {
        int(present_tracks[idx])
        for idx, involvement in enumerate(interaction_cache.agent_conflict_involvement.tolist())
        if involvement > 0.0
    }
    if conflict_agents:
        key_agent_coverage = len(conflict_agents & selected_set) / float(len(conflict_agents))
    else:
        key_agent_coverage = 1.0

    def safe_ratio(selected_value: float, full_value: float, harder_when_smaller: bool = False) -> float:
        selected_value = float(selected_value)
        full_value = float(full_value)
        if harder_when_smaller:
            if full_value <= 1e-6:
                return 1.0
            return float(np.clip(full_value / max(selected_value, 1e-6), 0.0, 1.0))
        if full_value <= 1e-6:
            return 1.0
        return float(np.clip(selected_value / full_value, 0.0, 1.0))

    conflict_coverage = safe_ratio(selected_features.get("conflict_count", 0.0), full_features.get("conflict_count", 0.0))
    density_coverage = safe_ratio(selected_features.get("interaction_density", 0.0), full_features.get("interaction_density", 0.0))
    required_decel_coverage = safe_ratio(
        selected_features.get("required_deceleration_proxy", 0.0),
        full_features.get("required_deceleration_proxy", 0.0),
    )
    min_distance_coverage = safe_ratio(
        selected_features.get("min_pairwise_distance_future", 100.0),
        full_features.get("min_pairwise_distance_future", 100.0),
        harder_when_smaller=True,
    )
    ttc_coverage = safe_ratio(
        selected_features.get("ttc_proxy", 8.0),
        full_features.get("ttc_proxy", 8.0),
        harder_when_smaller=True,
    )
    raw_coverage = float(
        np.mean(
            [
                conflict_coverage,
                density_coverage,
                required_decel_coverage,
                min_distance_coverage,
                ttc_coverage,
                key_agent_coverage,
            ]
        )
    )
    return {
        "selected_conflict_count_coverage": conflict_coverage,
        "selected_interaction_density_coverage": density_coverage,
        "selected_required_deceleration_coverage": required_decel_coverage,
        "selected_min_pairwise_distance_coverage": min_distance_coverage,
        "selected_ttc_coverage": ttc_coverage,
        "selected_key_agent_coverage": float(key_agent_coverage),
        "selected_agents_raw_coverage": raw_coverage,
    }


def _select_focal_track_indices(
    dense_case: DenseCaseData,
    center_idx: int,
    config: Dict,
) -> np.ndarray:
    present_tracks = dense_case.present_tracks_by_time[center_idx]
    if len(present_tracks) == 0:
        return present_tracks
    data_cfg = config["data"]
    min_speed = float(data_cfg.get("min_focal_speed_mps", 0.0))
    focal_stride = int(data_cfg.get("focal_agent_stride", 1))
    max_focal_agents = int(data_cfg.get("max_focal_agents_per_frame", 0))
    focal_type_names = data_cfg.get("focal_agent_types")
    if focal_type_names:
        allowed_types = {_agent_type_id(name) for name in focal_type_names}
    else:
        allowed_types = None
    current_states = dense_case.states[present_tracks, center_idx]
    current_speeds = np.linalg.norm(current_states[:, 2:4], axis=-1)
    keep_mask = current_speeds >= min_speed
    if allowed_types is not None:
        keep_mask &= np.isin(dense_case.agent_types[present_tracks], list(allowed_types))
    focal_tracks = present_tracks[keep_mask]
    if len(focal_tracks) == 0:
        focal_tracks = present_tracks
    if focal_stride > 1:
        focal_tracks = focal_tracks[::focal_stride]
    if max_focal_agents > 0 and len(focal_tracks) > max_focal_agents:
        focal_speeds = np.linalg.norm(dense_case.states[focal_tracks, center_idx, 2:4], axis=-1)
        order = np.argsort(-focal_speeds, kind="stable")[:max_focal_agents]
        focal_tracks = focal_tracks[np.sort(order)]
    return focal_tracks


def _build_slices_from_track_file_all_focal(
    track_df: pd.DataFrame,
    split: str,
    location_id: str,
    data_root: str | Path,
    config: Dict,
    show_case_progress: bool = True,
) -> tuple[List[Dict], Dict[str, Any]]:
    history_frames = int(config["data"]["history_frames"])
    future_frames = int(config["data"]["future_frames"])
    stride = int(_resolve_slice_builder_config(config)["slice_stride_frames"])
    max_agents = int(config["data"]["max_agents"])
    processing_cfg = dict(config.get("processing", {}))
    include_background_future = bool(processing_cfg.get("include_background_future_for_proxy", True))
    diagnostic_profile = bool(processing_cfg.get("include_same_scene_diagnostic_cache", False)) or str(
        processing_cfg.get("profile", "main_retrieval_generation")
    ) in {"diagnostic_same_scene", "full_debug"}
    max_background_agents = int(config["data"].get("max_background_agents_for_proxy", 0) or 0) if include_background_future else 0
    local_coordinates = bool(config["data"]["local_coordinates"])
    map_radius = float(config["data"]["map_radius_m"])
    max_polylines = int(config["data"]["max_map_polylines"])
    max_points = int(config["data"]["max_polyline_points"])
    map_data = read_interaction_map(location_id, data_root, target_points=max_points)
    prepared_map = _prepare_map(map_data, max_points=max_points)
    map_cfg = _resolve_map_processing_config(config)
    map_crop_cache: Dict[tuple[int, int], np.ndarray] = {}
    slices: List[Dict] = []
    profile = BuildFileProfile(
        file=str(location_id),
        split=split,
        location_id=location_id,
        num_cases=0,
        num_timestamps=0,
        num_slices=0,
        time_read_sec=0.0,
        time_dense_sec=0.0,
        time_interaction_matrix_sec=0.0,
        time_agent_select_sec=0.0,
        time_map_crop_sec=0.0,
        time_feature_compute_sec=0.0,
        time_write_sec=0.0,
        total_time_sec=0.0,
        cache_hit=False,
        map_crop_cache_hits=0,
        map_crop_cache_queries=0,
    )
    case_groups = list(track_df.groupby("case_id", sort=False))
    for case_id, case_df in tqdm(
        case_groups,
        desc=f"Slicing {split}:{location_id}",
        unit="case",
        leave=False,
        disable=not show_case_progress,
    ):
        case_dense_start = perf_counter()
        dense_case = _prepare_dense_case(case_df)
        profile.time_dense_sec += perf_counter() - case_dense_start
        profile.num_cases += 1
        num_timestamps = len(dense_case.timestamps)
        profile.num_timestamps += num_timestamps
        if num_timestamps < history_frames:
            continue
        min_center = history_frames - 1
        if future_frames > 0 and num_timestamps >= history_frames + future_frames:
            max_center = num_timestamps - future_frames - 1
        elif split == "test":
            max_center = num_timestamps - 1
        else:
            continue
        for center_idx in range(min_center, max_center + 1, stride):
            select_start = perf_counter()
            focal_track_indices = _select_focal_track_indices(dense_case, center_idx, config)
            if len(focal_track_indices) == 0:
                profile.time_agent_select_sec += perf_counter() - select_start
                continue
            profile.time_agent_select_sec += perf_counter() - select_start
            present_tracks = dense_case.present_tracks_by_time[center_idx]
            interaction_start = perf_counter()
            interaction_cache = _compute_center_interaction_cache(
                dense_case=dense_case,
                center_idx=center_idx,
                present_tracks=present_tracks,
                future_frames=future_frames,
                conflict_distance_m=float(config["difficulty"]["conflict_distance_m"]),
                ttc_max_sec=float(config["difficulty"]["ttc_max_sec"]),
                interaction_radius_m=float(config["difficulty"]["interaction_radius_m"]),
            )
            profile.time_interaction_matrix_sec += perf_counter() - interaction_start
            for focal_local_idx, focal_track_index in enumerate(focal_track_indices):
                ordered_tracks = _select_context_tracks(
                    dense_case=dense_case,
                    center_idx=center_idx,
                    focal_track_index=focal_track_index,
                    present_tracks=present_tracks,
                    interaction_cache=interaction_cache,
                    config=config,
                )
                full_ordered_tracks = np.concatenate(
                    [np.asarray([focal_track_index], dtype=np.int64), present_tracks[present_tracks != focal_track_index]]
                )
                (
                    full_agent_ids,
                    full_agent_types,
                    full_agent_sizes,
                    full_history_states,
                    full_history_mask,
                    full_current_states,
                    full_future_states,
                    full_future_mask,
                    full_agent_mask,
                    world_origin,
                    world_heading,
                ) = _assemble_slice(
                    dense_case=dense_case,
                    center_idx=center_idx,
                    focal_track_index=focal_track_index,
                    ordered_track_indices=full_ordered_tracks,
                    history_frames=history_frames,
                    future_frames=future_frames,
                    max_agents=len(full_ordered_tracks),
                    local_coordinates=local_coordinates,
                )
                track_to_full_position = {int(track): idx for idx, track in enumerate(full_ordered_tracks.tolist())}
                selected_positions = np.asarray(
                    [track_to_full_position[int(track)] for track in ordered_tracks if int(track) in track_to_full_position],
                    dtype=np.int64,
                )
                (
                    agent_ids,
                    agent_types,
                    agent_sizes,
                    history_states,
                    history_mask,
                    current_states,
                    future_states,
                    future_mask,
                    agent_mask,
                ) = _subselect_assembled_slice(
                    agent_ids_full=full_agent_ids,
                    agent_types_full=full_agent_types,
                    agent_sizes_full=full_agent_sizes,
                    history_states_full=full_history_states,
                    history_mask_full=full_history_mask,
                    current_states_full=full_current_states,
                    future_states_full=full_future_states,
                    future_mask_full=full_future_mask,
                    agent_mask_full=full_agent_mask,
                    selected_positions=selected_positions,
                    max_agents=max_agents,
                )
                selected_position_mask = np.zeros(len(full_ordered_tracks), dtype=bool)
                selected_position_mask[selected_positions] = True
                background_positions = np.flatnonzero(~selected_position_mask)
                background_positions = _select_background_proxy_positions(
                    dense_case=dense_case,
                    center_idx=center_idx,
                    focal_track_index=focal_track_index,
                    present_tracks=present_tracks,
                    full_ordered_tracks=full_ordered_tracks,
                    background_positions=background_positions,
                    interaction_cache=interaction_cache,
                    max_background_agents=max_background_agents,
                )
                if include_background_future and background_positions.size > 0:
                    (
                        background_agent_ids,
                        _background_agent_types,
                        _background_agent_sizes,
                        _background_history_states,
                        _background_history_mask,
                        background_current_states,
                        background_future_states,
                        background_future_mask,
                        background_agent_mask,
                    ) = _subselect_assembled_slice(
                        agent_ids_full=full_agent_ids,
                        agent_types_full=full_agent_types,
                        agent_sizes_full=full_agent_sizes,
                        history_states_full=full_history_states,
                        history_mask_full=full_history_mask,
                        current_states_full=full_current_states,
                        future_states_full=full_future_states,
                        future_mask_full=full_future_mask,
                        agent_mask_full=full_agent_mask,
                        selected_positions=background_positions,
                        max_agents=len(background_positions),
                    )
                else:
                    background_agent_ids = np.zeros((0,), dtype=np.int64)
                    background_current_states = np.zeros((0, STATE_DIM), dtype=np.float32)
                    background_future_states = np.zeros((0, future_frames, STATE_DIM), dtype=np.float32)
                    background_future_mask = np.zeros((0, future_frames), dtype=bool)
                    background_agent_mask = np.zeros((0,), dtype=bool)
                map_crop_start = perf_counter()
                map_polylines, map_point_mask, map_polyline_mask, map_polyline_indices, cache_hit = _crop_map(
                    prepared_map=prepared_map,
                    focal_xy=world_origin,
                    radius_m=map_radius,
                    max_polylines=max_polylines,
                    origin_heading=world_heading,
                    to_local=local_coordinates,
                    grid_size_m=float(map_cfg["crop_cache_grid_m"]),
                    cache=map_crop_cache,
                )
                profile.time_map_crop_sec += perf_counter() - map_crop_start
                profile.map_crop_cache_queries += 1
                profile.map_crop_cache_hits += int(cache_hit)
                slice_dict = {
                    "slice_id": f"{split}:{location_id}:{dense_case.case_id}:{int(dense_case.timestamps[center_idx])}:{int(dense_case.track_ids[focal_track_index])}",
                    "split": split,
                    "location_id": location_id,
                    "scenario_id": f"{location_id}:{dense_case.case_id}",
                    "case_id": dense_case.case_id,
                    "center_timestamp": int(dense_case.timestamps[center_idx]),
                    "focal_agent_id": int(dense_case.track_ids[focal_track_index]),
                    "agent_ids": agent_ids,
                    "agent_types": agent_types,
                    "agent_sizes": agent_sizes,
                    "history_states": history_states,
                    "history_mask": history_mask,
                    "current_states": current_states,
                    "future_states": future_states,
                    "future_mask": future_mask,
                    "map_polylines": map_polylines,
                    "map_point_mask": map_point_mask,
                    "map_polyline_mask": map_polyline_mask,
                    "agent_mask": agent_mask,
                    "world_origin": world_origin.astype(np.float32),
                    "world_heading": float(world_heading),
                    "local_coordinates": local_coordinates,
                    "metadata": {"map_available": prepared_map.available},
                    "selected_agent_ids": agent_ids[agent_mask].copy(),
                    "full_scene_num_agents": int(full_agent_mask.sum()),
                }
                if diagnostic_profile:
                    slice_dict["same_scene_group_key"] = f"{location_id}:{dense_case.case_id}:{int(dense_case.track_ids[focal_track_index])}"
                    slice_dict["same_scene_rank"] = -1
                feature_start = perf_counter()
                selected_features = compute_difficulty_features(slice_dict, config)
                full_scene_slice = {
                    "slice_id": slice_dict["slice_id"],
                    "current_states": full_current_states,
                    "future_states": full_future_states,
                    "future_mask": full_future_mask,
                    "agent_mask": full_agent_mask,
                    "map_polylines": map_polylines,
                    "map_point_mask": map_point_mask,
                    "map_polyline_mask": map_polyline_mask,
                }
                full_features = compute_difficulty_features(full_scene_slice, config)
                coverage = _selection_coverage_metrics(
                    selected_features=selected_features,
                    full_features=full_features,
                    selected_tracks=ordered_tracks,
                    present_tracks=present_tracks,
                    interaction_cache=interaction_cache,
                )
                slice_dict["difficulty_features_selected_agents"] = selected_features
                slice_dict["difficulty_features_full_scene"] = full_features
                behavior_features = compute_behavior_aggressiveness_features(
                    future_states=future_states,
                    future_mask=future_mask,
                    current_states=current_states,
                    agent_mask=agent_mask,
                    config=config,
                )
                slice_dict["behavior_features_selected_agents"] = behavior_features
                slice_dict["scene_summary_selected_agents"] = _scene_summary_from_arrays(
                    current_states=current_states,
                    agent_mask=agent_mask,
                    interaction_radius_m=float(config["difficulty"]["interaction_radius_m"]),
                )
                slice_dict["scene_summary_full_scene"] = _scene_summary_from_arrays(
                    current_states=full_current_states,
                    agent_mask=full_agent_mask,
                    interaction_radius_m=float(config["difficulty"]["interaction_radius_m"]),
                )
                slice_dict["selection_coverage"] = coverage
                if include_background_future:
                    slice_dict["background_agent_ids"] = background_agent_ids
                    slice_dict["background_current_states"] = background_current_states
                    slice_dict["background_future_states"] = background_future_states
                    slice_dict["background_future_mask"] = background_future_mask
                    slice_dict["background_agent_mask"] = background_agent_mask
                slice_dict["metadata"]["full_scene_agent_count"] = int(full_agent_mask.sum())
                slice_dict["metadata"]["selected_agent_count"] = int(agent_mask.sum())
                slice_dict["metadata"]["background_proxy_agent_count"] = int(background_agent_mask.sum()) if include_background_future else 0
                slice_dict["metadata"].update(coverage)
                profile.time_feature_compute_sec += perf_counter() - feature_start
                if bool(map_cfg["save_map_polyline_indices_only"]):
                    slice_dict["map_polyline_indices"] = map_polyline_indices.astype(np.int64)
                    slice_dict["map_center"] = world_origin.astype(np.float32)
                    slice_dict["map_heading"] = float(world_heading)
                    slice_dict["map_local_coordinates"] = bool(local_coordinates)
                    slice_dict["map_max_polylines"] = int(max_polylines)
                slices.append(slice_dict)
    profile.num_slices = len(slices)
    return slices, profile.to_dict()


def build_slices_from_track_file(
    track_df: pd.DataFrame,
    split: str,
    location_id: str,
    data_root: str | Path,
    config: Dict,
    show_case_progress: bool = True,
) -> tuple[List[Dict], Dict[str, Any]]:
    builder_cfg = _resolve_slice_builder_config(config)
    if str(builder_cfg["mode"]) == "all_focal":
        return _build_slices_from_track_file_all_focal(
            track_df=track_df,
            split=split,
            location_id=location_id,
            data_root=data_root,
            config=config,
            show_case_progress=show_case_progress,
        )
    history_frames = int(config["data"]["history_frames"])
    future_frames = int(config["data"]["future_frames"])
    stride = int(builder_cfg["slice_stride_frames"])
    max_agents = int(config["data"]["max_agents"])
    processing_cfg = dict(config.get("processing", {}))
    include_background_future = bool(processing_cfg.get("include_background_future_for_proxy", True))
    max_background_agents = int(config["data"].get("max_background_agents_for_proxy", 0) or 0) if include_background_future else 0
    diagnostic_profile = bool(processing_cfg.get("include_same_scene_diagnostic_cache", False)) or str(
        processing_cfg.get("profile", "main_retrieval_generation")
    ) in {"diagnostic_same_scene", "full_debug"}
    local_coordinates = bool(config["data"]["local_coordinates"])
    map_radius = float(config["data"]["map_radius_m"])
    max_polylines = int(config["data"]["max_map_polylines"])
    max_points = int(config["data"]["max_polyline_points"])
    map_cfg = _resolve_map_processing_config(config)
    map_data = read_interaction_map(location_id, data_root, target_points=max_points)
    prepared_map = _prepare_map(map_data, max_points=max_points)
    map_crop_cache: Dict[tuple[int, int], np.ndarray] = {}
    slices: List[Dict] = []
    profile = BuildFileProfile(
        file=str(location_id),
        split=split,
        location_id=location_id,
        num_cases=0,
        num_timestamps=0,
        num_slices=0,
        time_read_sec=0.0,
        time_dense_sec=0.0,
        time_interaction_matrix_sec=0.0,
        time_agent_select_sec=0.0,
        time_map_crop_sec=0.0,
        time_feature_compute_sec=0.0,
        time_write_sec=0.0,
        total_time_sec=0.0,
        cache_hit=False,
        map_crop_cache_hits=0,
        map_crop_cache_queries=0,
    )
    case_groups = list(track_df.groupby("case_id", sort=False))
    max_slices_per_file = builder_cfg["max_slices_per_file"]
    for _, case_df in tqdm(
        case_groups,
        desc=f"Slicing {split}:{location_id}",
        unit="case",
        leave=False,
        disable=not show_case_progress,
    ):
        dense_start = perf_counter()
        dense_case = _prepare_dense_case(case_df)
        profile.time_dense_sec += perf_counter() - dense_start
        profile.num_cases += 1
        num_timestamps = len(dense_case.timestamps)
        profile.num_timestamps += num_timestamps
        if num_timestamps < history_frames:
            continue
        min_center = history_frames - 1
        if future_frames > 0 and num_timestamps >= history_frames + future_frames:
            max_center = num_timestamps - future_frames - 1
        elif split == "test":
            max_center = num_timestamps - 1
        else:
            continue
        for center_idx in range(min_center, max_center + 1, stride):
            if max_slices_per_file is not None and len(slices) >= int(max_slices_per_file):
                break
            present_tracks = dense_case.present_tracks_by_time[center_idx]
            if len(present_tracks) == 0:
                continue
            interaction_start = perf_counter()
            interaction_cache = _compute_center_interaction_cache(
                dense_case=dense_case,
                center_idx=center_idx,
                present_tracks=present_tracks,
                future_frames=future_frames,
                conflict_distance_m=float(config["difficulty"]["conflict_distance_m"]),
                ttc_max_sec=float(config["difficulty"]["ttc_max_sec"]),
                interaction_radius_m=float(config["difficulty"]["interaction_radius_m"]),
            )
            profile.time_interaction_matrix_sec += perf_counter() - interaction_start
            select_start = perf_counter()
            anchor_pos, focal_positions, interaction_scores, history_valid_ratio, future_valid_ratio = _select_scene_anchor_and_focals(
                dense_case=dense_case,
                center_idx=center_idx,
                present_tracks=present_tracks,
                interaction_cache=interaction_cache,
                config=config,
                split=split,
            )
            anchor_track_index = int(present_tracks[anchor_pos])
            scene_selected_tracks, scene_selected_scores, scene_selection_reason = _select_scene_context_tracks(
                dense_case=dense_case,
                center_idx=center_idx,
                present_tracks=present_tracks,
                anchor_position=anchor_pos,
                interaction_scores=interaction_scores,
                history_valid_ratio=history_valid_ratio,
                future_valid_ratio=future_valid_ratio,
                config=config,
            )
            profile.time_agent_select_sec += perf_counter() - select_start
            scene_full_tracks = np.concatenate(
                [np.asarray([anchor_track_index], dtype=np.int64), present_tracks[present_tracks != anchor_track_index]]
            )
            (
                full_agent_ids,
                full_agent_types,
                full_agent_sizes,
                full_history_states,
                full_history_mask,
                full_current_states,
                full_future_states,
                full_future_mask,
                full_agent_mask,
                scene_world_origin,
                scene_world_heading,
            ) = _assemble_slice(
                dense_case=dense_case,
                center_idx=center_idx,
                focal_track_index=anchor_track_index,
                ordered_track_indices=scene_full_tracks,
                history_frames=history_frames,
                future_frames=future_frames,
                max_agents=len(scene_full_tracks),
                local_coordinates=local_coordinates,
            )
            map_crop_start = perf_counter()
            full_map_polylines, full_map_point_mask, full_map_polyline_mask, map_polyline_indices, cache_hit = _crop_map(
                prepared_map=prepared_map,
                focal_xy=scene_world_origin,
                radius_m=map_radius,
                max_polylines=max_polylines,
                origin_heading=scene_world_heading,
                to_local=local_coordinates,
                grid_size_m=float(map_cfg["crop_cache_grid_m"]),
                cache=map_crop_cache,
            )
            profile.time_map_crop_sec += perf_counter() - map_crop_start
            profile.map_crop_cache_queries += 1
            profile.map_crop_cache_hits += int(cache_hit)
            full_slice_for_features = {
                "slice_id": f"{split}:{location_id}:{dense_case.case_id}:{int(dense_case.timestamps[center_idx])}:{int(dense_case.track_ids[anchor_track_index])}:scene",
                "current_states": full_current_states,
                "future_states": full_future_states,
                "future_mask": full_future_mask,
                "agent_mask": full_agent_mask,
                "map_polylines": full_map_polylines,
                "map_point_mask": full_map_point_mask,
                "map_polyline_mask": full_map_polyline_mask,
            }
            feature_start = perf_counter()
            full_features = compute_difficulty_features(full_slice_for_features, config)
            full_summary = _scene_summary_from_arrays(
                current_states=full_current_states,
                agent_mask=full_agent_mask,
                interaction_radius_m=float(config["difficulty"]["interaction_radius_m"]),
            )
            profile.time_feature_compute_sec += perf_counter() - feature_start
            for focal_position in focal_positions:
                if max_slices_per_file is not None and len(slices) >= int(max_slices_per_file):
                    break
                focal_track_index = int(present_tracks[int(focal_position)])
                if builder_cfg["mode"] == "scene_centric":
                    track_to_full_position = {int(track): idx for idx, track in enumerate(scene_full_tracks.tolist())}
                    selected_positions = np.asarray(
                        [track_to_full_position[int(track)] for track in scene_selected_tracks if int(track) in track_to_full_position],
                        dtype=np.int64,
                    )
                    (
                        agent_ids,
                        agent_types,
                        agent_sizes,
                        history_states,
                        history_mask,
                        current_states,
                        future_states,
                        future_mask,
                        agent_mask,
                    ) = _subselect_assembled_slice(
                        agent_ids_full=full_agent_ids,
                        agent_types_full=full_agent_types,
                        agent_sizes_full=full_agent_sizes,
                        history_states_full=full_history_states,
                        history_mask_full=full_history_mask,
                        current_states_full=full_current_states,
                        future_states_full=full_future_states,
                        future_mask_full=full_future_mask,
                        agent_mask_full=full_agent_mask,
                        selected_positions=selected_positions,
                        max_agents=max_agents,
                    )
                    world_origin = scene_world_origin
                    world_heading = scene_world_heading
                    selected_scores = scene_selected_scores
                    selection_reasons = list(scene_selection_reason)
                    background_track_positions = np.flatnonzero(~np.isin(scene_full_tracks, scene_selected_tracks))
                    background_track_positions = _select_background_proxy_positions(
                        dense_case=dense_case,
                        center_idx=center_idx,
                        focal_track_index=focal_track_index,
                        present_tracks=present_tracks,
                        full_ordered_tracks=scene_full_tracks,
                        background_positions=background_track_positions,
                        interaction_cache=interaction_cache,
                        max_background_agents=max_background_agents,
                    )
                    if include_background_future and background_track_positions.size > 0:
                        (
                            background_agent_ids,
                            _background_agent_types,
                            _background_agent_sizes,
                            _background_history_states,
                            _background_history_mask,
                            background_current_states,
                            background_future_states,
                            background_future_mask,
                            background_agent_mask,
                        ) = _subselect_assembled_slice(
                            agent_ids_full=full_agent_ids,
                            agent_types_full=full_agent_types,
                            agent_sizes_full=full_agent_sizes,
                            history_states_full=full_history_states,
                            history_mask_full=full_history_mask,
                            current_states_full=full_current_states,
                            future_states_full=full_future_states,
                            future_mask_full=full_future_mask,
                            agent_mask_full=full_agent_mask,
                            selected_positions=background_track_positions,
                            max_agents=len(background_track_positions),
                        )
                    else:
                        background_agent_ids = np.zeros((0,), dtype=np.int64)
                        background_current_states = np.zeros((0, STATE_DIM), dtype=np.float32)
                        background_future_states = np.zeros((0, future_frames, STATE_DIM), dtype=np.float32)
                        background_future_mask = np.zeros((0, future_frames), dtype=bool)
                        background_agent_mask = np.zeros((0,), dtype=bool)
                    map_polylines = full_map_polylines
                    map_point_mask = full_map_point_mask
                    map_polyline_mask = full_map_polyline_mask
                else:
                    select_start = perf_counter()
                    ordered_tracks = _select_context_tracks(
                        dense_case=dense_case,
                        center_idx=center_idx,
                        focal_track_index=focal_track_index,
                        present_tracks=present_tracks,
                        interaction_cache=interaction_cache,
                        config=config,
                    )
                    profile.time_agent_select_sec += perf_counter() - select_start
                    (
                        agent_ids,
                        agent_types,
                        agent_sizes,
                        history_states,
                        history_mask,
                        current_states,
                        future_states,
                        future_mask,
                        agent_mask,
                        world_origin,
                        world_heading,
                    ) = _assemble_slice(
                        dense_case=dense_case,
                        center_idx=center_idx,
                        focal_track_index=focal_track_index,
                        ordered_track_indices=ordered_tracks,
                        history_frames=history_frames,
                        future_frames=future_frames,
                        max_agents=max_agents,
                        local_coordinates=local_coordinates,
                    )
                    relative_positions = np.asarray(
                        [int(np.where(present_tracks == track)[0][0]) for track in ordered_tracks if np.any(present_tracks == track)],
                        dtype=np.int64,
                    )
                    selected_scores = interaction_scores[relative_positions][: max_agents]
                    selection_reasons = ["limited_focal_context"] * int(min(len(ordered_tracks), max_agents))
                    if include_background_future:
                        background_tracks = present_tracks[~np.isin(present_tracks, ordered_tracks)]
                        (
                            background_agent_ids,
                            _background_agent_types,
                            _background_agent_sizes,
                            _background_history_states,
                            _background_history_mask,
                            background_current_states,
                            background_future_states,
                            background_future_mask,
                            background_agent_mask,
                            _,
                            _,
                        ) = _assemble_slice(
                            dense_case=dense_case,
                            center_idx=center_idx,
                            focal_track_index=focal_track_index,
                            ordered_track_indices=background_tracks[:max_background_agents],
                            history_frames=history_frames,
                            future_frames=future_frames,
                            max_agents=min(len(background_tracks), max_background_agents),
                            local_coordinates=local_coordinates,
                        )
                    else:
                        background_agent_ids = np.zeros((0,), dtype=np.int64)
                        background_current_states = np.zeros((0, STATE_DIM), dtype=np.float32)
                        background_future_states = np.zeros((0, future_frames, STATE_DIM), dtype=np.float32)
                        background_future_mask = np.zeros((0, future_frames), dtype=bool)
                        background_agent_mask = np.zeros((0,), dtype=bool)
                    map_polylines, map_point_mask, map_polyline_mask = _materialize_map_from_indices(
                        prepared_map=prepared_map,
                        chosen_idx=map_polyline_indices,
                        focal_xy=world_origin,
                        origin_heading=world_heading,
                        max_polylines=max_polylines,
                        to_local=local_coordinates,
                    )
                slice_dict = {
                    "slice_id": f"{split}:{location_id}:{dense_case.case_id}:{int(dense_case.timestamps[center_idx])}:{int(dense_case.track_ids[focal_track_index])}",
                    "split": split,
                    "location_id": location_id,
                    "scenario_id": f"{location_id}:{dense_case.case_id}",
                    "case_id": dense_case.case_id,
                    "center_timestamp": int(dense_case.timestamps[center_idx]),
                    "focal_agent_id": int(dense_case.track_ids[focal_track_index]),
                    "agent_ids": agent_ids,
                    "agent_types": agent_types,
                    "agent_sizes": agent_sizes,
                    "history_states": history_states,
                    "history_mask": history_mask,
                    "current_states": current_states,
                    "future_states": future_states,
                    "future_mask": future_mask,
                    "map_polylines": map_polylines,
                    "map_point_mask": map_point_mask,
                    "map_polyline_mask": map_polyline_mask,
                    "agent_mask": agent_mask,
                    "world_origin": world_origin.astype(np.float32),
                    "world_heading": float(world_heading),
                    "local_coordinates": local_coordinates,
                    "metadata": {
                        "map_available": prepared_map.available,
                        "slice_builder_mode": str(builder_cfg["mode"]),
                        "full_scene_num_agents": int(full_agent_mask.sum()),
                    },
                    "selected_agent_ids": agent_ids[agent_mask].copy(),
                    "selected_agent_scores": np.asarray(selected_scores, dtype=np.float32),
                    "selected_agent_selection_reason": list(selection_reasons),
                    "full_scene_num_agents": int(full_agent_mask.sum()),
                }
                if diagnostic_profile:
                    slice_dict["same_scene_group_key"] = f"{location_id}:{dense_case.case_id}:{int(dense_case.track_ids[focal_track_index])}"
                    slice_dict["same_scene_rank"] = -1
                feature_start = perf_counter()
                selected_features = compute_difficulty_features(slice_dict, config)
                behavior_features = compute_behavior_aggressiveness_features(
                    future_states=future_states,
                    future_mask=future_mask,
                    current_states=current_states,
                    agent_mask=agent_mask,
                    config=config,
                )
                slice_dict["difficulty_features_selected_agents"] = selected_features
                slice_dict["difficulty_features_full_scene"] = full_features
                slice_dict["behavior_features_selected_agents"] = behavior_features
                slice_dict["scene_summary_selected_agents"] = _scene_summary_from_arrays(
                    current_states=current_states,
                    agent_mask=agent_mask,
                    interaction_radius_m=float(config["difficulty"]["interaction_radius_m"]),
                )
                slice_dict["scene_summary_full_scene"] = full_summary
                coverage = _selection_coverage_metrics(
                    selected_features=selected_features,
                    full_features=full_features,
                    selected_tracks=np.asarray(slice_dict["selected_agent_ids"], dtype=np.int64),
                    present_tracks=present_tracks,
                    interaction_cache=interaction_cache,
                )
                slice_dict["selection_coverage"] = coverage
                slice_dict["full_scene_num_agents"] = int(full_agent_mask.sum())
                if include_background_future:
                    slice_dict["background_agent_ids"] = background_agent_ids
                    slice_dict["background_current_states"] = background_current_states
                    slice_dict["background_future_states"] = background_future_states
                    slice_dict["background_future_mask"] = background_future_mask
                    slice_dict["background_agent_mask"] = background_agent_mask
                slice_dict["metadata"]["selected_agent_count"] = int(agent_mask.sum())
                slice_dict["metadata"]["background_proxy_agent_count"] = int(background_agent_mask.sum()) if include_background_future else 0
                slice_dict["metadata"].update(coverage)
                profile.time_feature_compute_sec += perf_counter() - feature_start
                if bool(map_cfg["save_map_polyline_indices_only"]):
                    slice_dict["map_polyline_indices"] = map_polyline_indices.astype(np.int64)
                    slice_dict["map_center"] = world_origin.astype(np.float32)
                    slice_dict["map_heading"] = float(world_heading)
                    slice_dict["map_local_coordinates"] = bool(local_coordinates)
                    slice_dict["map_max_polylines"] = int(max_polylines)
                slices.append(slice_dict)
        if max_slices_per_file is not None and len(slices) >= int(max_slices_per_file):
            break
    profile.num_slices = len(slices)
    return slices, profile.to_dict()


def _map_source_signature(location_id: str, data_root: str | Path) -> Dict[str, Any]:
    maps_root = Path(data_root) / "maps"
    xy_path = maps_root / f"{location_id}.osm_xy"
    osm_path = maps_root / f"{location_id}.osm"
    candidate = xy_path if xy_path.exists() else osm_path
    if not candidate.exists():
        return {"path": "", "size": 0, "mtime_ns": 0}
    stat = candidate.stat()
    return {
        "path": str(candidate),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _slice_shard_signature(interaction_file, config: Dict, data_root: str | Path) -> Dict[str, Any]:
    slice_builder_cfg = _resolve_slice_builder_config(config)
    map_cfg = _resolve_map_processing_config(config)
    processing_cfg = _resolve_build_processing_config(config)
    source_path = Path(interaction_file.file_path)
    source_stat = source_path.stat()
    return {
        "source_file": str(source_path),
        "source_size": int(source_stat.st_size),
        "source_mtime_ns": int(source_stat.st_mtime_ns),
        "map_signature": _map_source_signature(interaction_file.location_id, data_root),
        "slice_builder_version": SLICE_BUILDER_VERSION,
        "interaction_reader_version": INTERACTION_READER_VERSION,
        "difficulty_code_version": DIFFICULTY_CODE_VERSION,
        "behavior_code_version": BEHAVIOR_CODE_VERSION,
        "processing_profile": str(config.get("processing", {}).get("profile", "main_retrieval_generation")),
        "slice_builder_mode": str(slice_builder_cfg["mode"]),
        "slice_stride_frames": int(slice_builder_cfg["slice_stride_frames"]),
        "max_focals_per_timestamp": int(slice_builder_cfg["max_focals_per_timestamp"]),
        "max_agents": int(config["data"]["max_agents"]),
        "history_frames": int(config["data"]["history_frames"]),
        "future_frames": int(config["data"]["future_frames"]),
        "map_radius_m": float(config["data"]["map_radius_m"]),
        "save_map_polyline_indices_only": bool(map_cfg["save_map_polyline_indices_only"]),
        "include_background_future_for_proxy": bool(processing_cfg["include_background_future_for_proxy"]),
    }


def _slice_shard_path(interaction_file, shard_cache_dir: Path) -> Path:
    source_hash = hashlib.sha1(str(Path(interaction_file.file_path).resolve()).encode("utf-8")).hexdigest()[:10]
    safe_stem = f"{Path(interaction_file.file_path).stem}_{source_hash}"
    return shard_cache_dir / f"{safe_stem}.pt"


def _load_slice_shard(shard_path: Path, expected_signature: Dict[str, Any]) -> tuple[List[Dict], Dict[str, Any]] | None:
    if not shard_path.exists():
        return None
    payload = torch.load(shard_path, map_location="cpu", weights_only=False)
    if payload.get("shard_signature") != expected_signature:
        return None
    return list(payload.get("slices", [])), dict(payload.get("profile", {}))


def _save_slice_shard(shard_path: Path, shard_signature: Dict[str, Any], slices: List[Dict], profile: Dict[str, Any]) -> None:
    ensure_dir(shard_path.parent)
    torch.save(
        {
            "shard_signature": shard_signature,
            "slices": slices,
            "profile": profile,
        },
        shard_path,
    )


def _build_slices_for_interaction_file(
    interaction_file,
    config: Dict,
    data_root: str | Path,
    show_case_progress: bool = True,
    shard_cache_dir: Path | None = None,
    reuse_slice_shards: bool = False,
) -> tuple[str, List[Dict], Dict[str, Any]]:
    start_time = perf_counter()
    shard_signature = _slice_shard_signature(interaction_file, config, data_root)
    shard_path = _slice_shard_path(interaction_file, shard_cache_dir) if shard_cache_dir is not None else None
    if reuse_slice_shards and shard_path is not None:
        loaded = _load_slice_shard(shard_path, shard_signature)
        if loaded is not None:
            file_slices, profile = loaded
            profile["cache_hit"] = True
            profile["time_read_sec"] = float(getattr(interaction_file, "read_time_sec", 0.0))
            profile["file"] = str(interaction_file.file_path)
            profile["split"] = interaction_file.split
            profile["location_id"] = interaction_file.location_id
            profile["total_time_sec"] = float(perf_counter() - start_time)
            profile["shard_path"] = str(shard_path)
            profile["shard_signature"] = shard_signature
            return interaction_file.split, file_slices, profile
    file_slices, profile = build_slices_from_track_file(
        track_df=interaction_file.dataframe,
        split=interaction_file.split,
        location_id=interaction_file.location_id,
        data_root=data_root,
        config=config,
        show_case_progress=show_case_progress,
    )
    if interaction_file.split == "test" and not interaction_file.has_future:
        for item in file_slices:
            item["future_states"] = np.zeros_like(item["future_states"])
            item["future_mask"] = np.zeros_like(item["future_mask"], dtype=bool)
            item["metadata"]["observation_only_test"] = True
            difficulty_item = materialize_cached_map_fields(item, data_root, int(config["data"]["max_polyline_points"]))
            zero_future_features = compute_difficulty_features(difficulty_item, config)
            item["difficulty_features_selected_agents"] = zero_future_features
            item["difficulty_features_full_scene"] = zero_future_features
            item["difficulty_features"] = zero_future_features
    write_start = perf_counter()
    if shard_path is not None:
        _save_slice_shard(shard_path, shard_signature, file_slices, profile)
    profile["time_write_sec"] = float(profile.get("time_write_sec", 0.0) + (perf_counter() - write_start))
    profile["cache_hit"] = False
    profile["time_read_sec"] = float(getattr(interaction_file, "read_time_sec", 0.0))
    profile["file"] = str(interaction_file.file_path)
    profile["split"] = interaction_file.split
    profile["location_id"] = interaction_file.location_id
    profile["total_time_sec"] = float(perf_counter() - start_time)
    if shard_path is not None:
        profile["shard_path"] = str(shard_path)
        profile["shard_signature"] = shard_signature
    return interaction_file.split, file_slices, profile


def build_slices(scan_result, config: Dict, data_root: str | Path, cache_dir: str | Path | None = None) -> SliceBuildResult:
    total_start = perf_counter()
    warnings: List[str] = list(scan_result.warnings)
    slices_by_split: Dict[str, List[Dict]] = {"train": [], "val": [], "test": []}
    split_metadata: Dict[str, Dict[str, int]] = {}
    build_profile_rows: List[Dict[str, Any]] = []
    valid_files = [interaction_file for interaction_file in scan_result.files if interaction_file.split in slices_by_split]
    build_cfg = _resolve_build_processing_config(config)
    num_workers = int(build_cfg["num_build_workers"])
    shard_cache_dir = None
    if cache_dir is not None:
        shard_root = Path(build_cfg["shard_cache_dir"])
        shard_cache_dir = shard_root if shard_root.is_absolute() else Path(cache_dir) / "slice_shards"
        ensure_dir(shard_cache_dir)
    if num_workers <= 1 or len(valid_files) <= 1:
        results = []
        for interaction_file in tqdm(valid_files, desc="Building slices", unit="file"):
            results.append(
                (
                    interaction_file,
                    _build_slices_for_interaction_file(
                        interaction_file,
                        config,
                        data_root,
                        show_case_progress=True,
                        shard_cache_dir=shard_cache_dir,
                        reuse_slice_shards=bool(build_cfg["reuse_slice_shards"]),
                    ),
                )
            )
    else:
        completed: list[tuple[int, object, tuple[str, List[Dict], Dict[str, Any]]]] = []
        executor_cls = ProcessPoolExecutor
        try:
            executor_context = executor_cls(max_workers=num_workers)
        except (PermissionError, OSError) as exc:
            LOGGER.warning("Falling back to ThreadPoolExecutor for slice building because process workers could not start: %s", exc)
            executor_cls = ThreadPoolExecutor
            executor_context = executor_cls(max_workers=num_workers)
        with executor_context as executor:
            future_to_item = {
                executor.submit(
                    _build_slices_for_interaction_file,
                    interaction_file,
                    config,
                    data_root,
                    False,
                    shard_cache_dir,
                    bool(build_cfg["reuse_slice_shards"]),
                ): (idx, interaction_file)
                for idx, interaction_file in enumerate(valid_files)
            }
            progress = tqdm(total=len(valid_files), desc="Building slices", unit="file")
            for future in as_completed(future_to_item):
                idx, interaction_file = future_to_item[future]
                split_name, file_slices, profile = future.result()
                completed.append((idx, interaction_file, (split_name, file_slices, profile)))
                progress.update(1)
                progress.set_postfix_str(
                    f"last={interaction_file.location_id} slices={len(file_slices)} hit={bool(profile.get('cache_hit', False))}",
                    refresh=False,
                )
            progress.close()
        completed.sort(key=lambda item: item[0])
        results = [(interaction_file, payload) for _, interaction_file, payload in completed]
    for interaction_file, (split_name, file_slices, profile_row) in results:
        slices_by_split[split_name].extend(file_slices)
        build_profile_rows.append(profile_row)
        split_metadata.setdefault(split_name, {"files": 0, "locations": 0, "slices": 0})
        split_metadata[split_name]["files"] += 1
        split_metadata[split_name]["slices"] += len(file_slices)
    for split, split_slices in slices_by_split.items():
        split_metadata.setdefault(split, {"files": 0, "locations": 0, "slices": 0})
        split_metadata[split]["locations"] = len({item["location_id"] for item in split_slices})
        LOGGER.info("Built %d slices for split=%s.", len(split_slices), split)
    total_slices = int(sum(len(items) for items in slices_by_split.values()))
    total_build_time_sec = float(perf_counter() - total_start)
    map_crop_queries = int(sum(int(row.get("map_crop_cache_queries", 0)) for row in build_profile_rows))
    map_crop_hits = int(sum(int(row.get("map_crop_cache_hits", 0)) for row in build_profile_rows))
    shard_cache_hits = int(sum(1 for row in build_profile_rows if bool(row.get("cache_hit", False))))
    slice_builder_cfg = _resolve_slice_builder_config(config)
    build_performance = {
        "total_raw_files": len(valid_files),
        "num_workers": int(num_workers),
        "total_build_time_sec": total_build_time_sec,
        "total_slices": total_slices,
        "slices_per_sec": float(total_slices / max(total_build_time_sec, 1e-6)),
        "avg_slices_per_file": float(total_slices / max(len(valid_files), 1)),
        "avg_time_per_file": float(total_build_time_sec / max(len(valid_files), 1)),
        "shard_cache_hit_rate": float(shard_cache_hits / max(len(build_profile_rows), 1)),
        "map_crop_cache_hit_rate": float(map_crop_hits / max(map_crop_queries, 1)),
        "mode": str(slice_builder_cfg["mode"]),
        "slice_stride_frames": int(slice_builder_cfg["slice_stride_frames"]),
        "max_focals_per_timestamp": int(slice_builder_cfg["max_focals_per_timestamp"]),
        "skipped_all_focal_expansion": bool(str(slice_builder_cfg["mode"]) != "all_focal"),
    }
    if str(slice_builder_cfg["mode"]) == "all_focal":
        warnings.append("all_focal mode is expensive and not needed for retrieval-augmented main experiments.")
    return SliceBuildResult(
        slices_by_split=slices_by_split,
        warnings=warnings,
        split_metadata=split_metadata,
        build_profile_rows=build_profile_rows,
        build_performance=build_performance,
    )
