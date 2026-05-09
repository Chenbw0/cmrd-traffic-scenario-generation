from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

from isgen import ensure_dir, save_json
from isgen.data.cache import load_slices_from_cache
from isgen.semantics.spawn_ordering import canonical_sort_scene
from isgen.semantics.spawn_plan import extract_spawn_plan_targets

MAX_AGENTS_FOR_PROTOTYPE_EMBEDDING = 12
SUPPORT_PROTOTYPE_BANK_VERSION = 3
SCENE_METRIC_NAMES = (
    "count_norm",
    "mean_speed_norm",
    "std_speed_norm",
    "mean_radius_norm",
    "std_radius_norm",
    "mean_pair_distance_norm",
    "min_pairwise_distance_norm",
    "conflict_count_norm",
    "conflict_ratio",
    "heading_std",
)


def _safe_norm_np(value: np.ndarray, axis: int = -1, eps: float = 1e-6) -> np.ndarray:
    return np.sqrt(np.sum(np.square(value), axis=axis) + eps)


def _difficulty_bucket(value: float) -> str:
    if value < (1.0 / 3.0):
        return "low"
    if value < (2.0 / 3.0):
        return "mid"
    return "high"


def _scene_feature_vector(states: np.ndarray, mask: np.ndarray, conflict_distance_m: float = 5.0) -> Dict[str, float]:
    states = np.asarray(states, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    valid = states[mask]
    if valid.size == 0:
        return {
            "count": 0.0,
            "mean_speed": 0.0,
            "std_speed": 0.0,
            "mean_radius": 0.0,
            "std_radius": 0.0,
            "mean_pair_distance": 0.0,
            "min_pairwise_distance": 100.0,
            "conflict_count": 0.0,
            "conflict_ratio": 0.0,
            "heading_std": 0.0,
        }
    speed = _safe_norm_np(valid[:, 2:4], axis=-1)
    radius = _safe_norm_np(valid[:, 0:2], axis=-1)
    pairwise = _safe_norm_np(valid[:, None, 0:2] - valid[None, :, 0:2], axis=-1)
    if len(valid) > 1:
        pairwise = pairwise + np.eye(len(valid), dtype=np.float32) * 1e6
        mean_pair_distance = float(pairwise[pairwise < 1e5].mean())
        min_pairwise_distance = float(pairwise.min())
        conflict_count = float((pairwise < conflict_distance_m).sum() / 2.0)
        max_pairs = float(len(valid) * (len(valid) - 1) / 2.0)
        conflict_ratio = float(conflict_count / max(max_pairs, 1.0))
    else:
        mean_pair_distance = 100.0
        min_pairwise_distance = 100.0
        conflict_count = 0.0
        conflict_ratio = 0.0
    heading_complex = np.exp(1j * valid[:, 4].astype(np.float64))
    heading_std = float(np.clip(1.0 - np.abs(heading_complex.mean()), 0.0, 1.0))
    return {
        "count": float(len(valid)),
        "mean_speed": float(speed.mean()),
        "std_speed": float(speed.std()),
        "mean_radius": float(radius.mean()),
        "std_radius": float(radius.std()),
        "mean_pair_distance": mean_pair_distance,
        "min_pairwise_distance": min_pairwise_distance,
        "conflict_count": conflict_count,
        "conflict_ratio": conflict_ratio,
        "heading_std": heading_std,
    }


def _scene_metric_vector(
    feature_metrics: Dict[str, float],
    map_radius_m: float,
    max_speed_mps: float,
    max_agents: int,
) -> np.ndarray:
    distance_scale = max(map_radius_m * 2.0, 1e-6)
    max_pairs = max(float(max_agents * (max_agents - 1) / 2.0), 1.0)
    return np.asarray(
        [
            float(feature_metrics["count"]) / max(float(max_agents), 1.0),
            float(feature_metrics["mean_speed"]) / max(max_speed_mps, 1e-6),
            float(feature_metrics["std_speed"]) / max(max_speed_mps, 1e-6),
            float(feature_metrics["mean_radius"]) / max(map_radius_m, 1e-6),
            float(feature_metrics["std_radius"]) / max(map_radius_m, 1e-6),
            float(feature_metrics["mean_pair_distance"]) / distance_scale,
            float(feature_metrics["min_pairwise_distance"]) / distance_scale,
            float(feature_metrics["conflict_count"]) / max_pairs,
            float(feature_metrics["conflict_ratio"]),
            float(feature_metrics["heading_std"]),
        ],
        dtype=np.float32,
    )


def _prototype_bank_settings_signature(config: Dict) -> Dict[str, object]:
    proto_cfg = config.get("spawn_prototypes", {})
    spawn_cfg = config.get("spawn", {})
    data_cfg = config.get("data", {})
    return {
        "version": SUPPORT_PROTOTYPE_BANK_VERSION,
        "plan_radial_bins": int(spawn_cfg.get("plan_radial_bins", 4)),
        "plan_angular_bins": int(spawn_cfg.get("plan_angular_bins", 8)),
        "map_radius_m": float(data_cfg.get("map_radius_m", 80.0)),
        "max_speed_mps": float(spawn_cfg.get("max_speed_mps", 20.0)),
        "max_agents": int(data_cfg.get("max_agents", 24)),
        "max_agents_for_embedding": int(proto_cfg.get("max_agents_for_embedding", MAX_AGENTS_FOR_PROTOTYPE_EMBEDDING)),
        "max_prototypes_per_location": int(proto_cfg.get("max_prototypes_per_location", 16)),
        "records_per_prototype": int(proto_cfg.get("records_per_prototype", 20)),
        "embedding_state_weight": float(proto_cfg.get("embedding_state_weight", 0.35)),
        "embedding_plan_weight": float(proto_cfg.get("embedding_plan_weight", 1.0)),
        "embedding_metric_weight": float(proto_cfg.get("embedding_metric_weight", 1.75)),
        "embedding_difficulty_weight": float(proto_cfg.get("embedding_difficulty_weight", 0.5)),
        "difficulty_bucketed": bool(proto_cfg.get("difficulty_bucketed", True)),
    }


def _allocate_bucket_clusters(bucket_sizes: Dict[str, int], max_total: int) -> Dict[str, int]:
    nonempty = {bucket: size for bucket, size in bucket_sizes.items() if size > 0}
    allocation = {bucket: 0 for bucket in bucket_sizes}
    if not nonempty or max_total <= 0:
        return allocation
    if len(nonempty) >= max_total:
        ranked = sorted(nonempty.items(), key=lambda item: item[1], reverse=True)
        for bucket, _ in ranked[:max_total]:
            allocation[bucket] = 1
        return allocation
    remaining = max_total
    for bucket in nonempty:
        allocation[bucket] = 1
        remaining -= 1
    if remaining <= 0:
        return allocation
    remaining_budget = remaining
    total_size = float(sum(nonempty.values()))
    fractional: List[Tuple[float, str]] = []
    for bucket, size in nonempty.items():
        desired = remaining_budget * (float(size) / max(total_size, 1.0))
        extra = min(int(math.floor(desired)), max(size - allocation[bucket], 0))
        allocation[bucket] += extra
        remaining -= extra
        fractional.append((desired - math.floor(desired), bucket))
    if remaining > 0:
        for _, bucket in sorted(fractional, reverse=True):
            available = max(nonempty[bucket] - allocation[bucket], 0)
            if available <= 0:
                continue
            allocation[bucket] += 1
            remaining -= 1
            if remaining <= 0:
                break
    return allocation


def _build_scene_embedding(
    current_states: np.ndarray,
    agent_mask: np.ndarray,
    difficulty: float,
    radial_bins: int,
    angular_bins: int,
    map_radius_m: float,
    max_speed_mps: float,
    conflict_distance_m: float,
    max_agents_for_embedding: int,
    max_agents: int,
    embedding_state_weight: float,
    embedding_plan_weight: float,
    embedding_metric_weight: float,
    embedding_difficulty_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    states_t = torch.from_numpy(np.asarray(current_states, dtype=np.float32)).unsqueeze(0)
    mask_t = torch.from_numpy(np.asarray(agent_mask, dtype=bool)).unsqueeze(0)
    sorted_states_t, sorted_mask_t, _ = canonical_sort_scene(states_t, mask_t)
    sorted_states = sorted_states_t[0].cpu().numpy().astype(np.float32)
    sorted_mask = sorted_mask_t[0].cpu().numpy().astype(bool)

    top_states = np.zeros((max_agents_for_embedding, 6), dtype=np.float32)
    top_mask = np.zeros((max_agents_for_embedding,), dtype=np.float32)
    keep = min(max_agents_for_embedding, sorted_states.shape[0])
    top_states[:keep, 0:2] = sorted_states[:keep, 0:2] / max(map_radius_m, 1e-6)
    top_states[:keep, 2:4] = sorted_states[:keep, 2:4] / max(max_speed_mps, 1e-6)
    top_states[:keep, 4] = np.sin(sorted_states[:keep, 4])
    top_states[:keep, 5] = np.cos(sorted_states[:keep, 4])
    top_mask[:keep] = sorted_mask[:keep].astype(np.float32)

    plan = extract_spawn_plan_targets(
        current_states=states_t,
        agent_mask=mask_t,
        map_radius_m=map_radius_m,
        max_speed_mps=max_speed_mps,
        conflict_distance_m=conflict_distance_m,
        radial_bins=radial_bins,
        angular_bins=angular_bins,
    )
    plan_features = plan["plan_features"][0].cpu().numpy().astype(np.float32)
    feature_metrics = _scene_feature_vector(current_states, agent_mask, conflict_distance_m=conflict_distance_m)
    metric_vector = _scene_metric_vector(
        feature_metrics,
        map_radius_m=map_radius_m,
        max_speed_mps=max_speed_mps,
        max_agents=max_agents,
    )
    embedding = np.concatenate(
        [
            embedding_state_weight * top_states.reshape(-1),
            embedding_state_weight * top_mask,
            embedding_plan_weight * plan_features,
            embedding_metric_weight * metric_vector,
            np.asarray([embedding_difficulty_weight * float(difficulty)], dtype=np.float32),
        ],
        axis=0,
    )
    return embedding, sorted_states, sorted_mask.astype(np.bool_), plan_features, feature_metrics


def _resolve_prototype_dir(cache_dir: str, config: Dict) -> Path:
    spawn_proto_cfg = config.get("spawn_prototypes", {})
    bank_dir = spawn_proto_cfg.get("bank_dir")
    if bank_dir is None:
        return ensure_dir(Path(cache_dir) / "support_prototypes")
    bank_path = Path(bank_dir)
    if not bank_path.is_absolute():
        bank_path = Path(cache_dir) / bank_path
    return ensure_dir(bank_path)


def build_support_prototype_bank(cache_dir: str, config: Dict) -> Dict[str, Path]:
    proto_cfg = config.get("spawn_prototypes", {})
    radial_bins = int(config.get("spawn", {}).get("plan_radial_bins", 4))
    angular_bins = int(config.get("spawn", {}).get("plan_angular_bins", 8))
    map_radius_m = float(config.get("data", {}).get("map_radius_m", 80.0))
    max_speed_mps = float(config.get("spawn", {}).get("max_speed_mps", 20.0))
    max_agents = int(config.get("data", {}).get("max_agents", 24))
    conflict_distance_m = float(config.get("spawn_losses", {}).get("conflict_distance_m", 5.0))
    max_agents_for_embedding = int(proto_cfg.get("max_agents_for_embedding", MAX_AGENTS_FOR_PROTOTYPE_EMBEDDING))
    max_prototypes_per_location = int(proto_cfg.get("max_prototypes_per_location", 16))
    records_per_prototype = max(int(proto_cfg.get("records_per_prototype", 20)), 1)
    embedding_state_weight = float(proto_cfg.get("embedding_state_weight", 0.35))
    embedding_plan_weight = float(proto_cfg.get("embedding_plan_weight", 1.0))
    embedding_metric_weight = float(proto_cfg.get("embedding_metric_weight", 1.75))
    embedding_difficulty_weight = float(proto_cfg.get("embedding_difficulty_weight", 0.5))
    difficulty_bucketed = bool(proto_cfg.get("difficulty_bucketed", True))
    seed = int(config.get("training", {}).get("seed", 7))
    prototype_dir = _resolve_prototype_dir(cache_dir, config)
    bank_path = prototype_dir / "prototype_bank.pt"
    assignment_path = prototype_dir / "prototype_assignments.json"
    metadata_path = prototype_dir / "prototype_metadata.json"

    train_slices = load_slices_from_cache(cache_dir, "train", materialize_maps=False)
    split_records: Dict[str, List[Dict[str, object]]] = {"train": [], "val": [], "test": []}
    for split in split_records:
        slices = train_slices if split == "train" else load_slices_from_cache(cache_dir, split, materialize_maps=False)
        for item in slices:
            difficulty = float(item.get("difficulty_score_selected_agents", item.get("difficulty_score", 0.0)))
            embedding, sorted_states, sorted_mask, plan_features, feature_metrics = _build_scene_embedding(
                current_states=np.asarray(item["current_states"], dtype=np.float32),
                agent_mask=np.asarray(item["agent_mask"], dtype=bool),
                difficulty=difficulty,
                radial_bins=radial_bins,
                angular_bins=angular_bins,
                map_radius_m=map_radius_m,
                max_speed_mps=max_speed_mps,
                conflict_distance_m=conflict_distance_m,
                max_agents_for_embedding=max_agents_for_embedding,
                max_agents=max_agents,
                embedding_state_weight=embedding_state_weight,
                embedding_plan_weight=embedding_plan_weight,
                embedding_metric_weight=embedding_metric_weight,
                embedding_difficulty_weight=embedding_difficulty_weight,
            )
            split_records[split].append(
                {
                    "slice_id": str(item["slice_id"]),
                    "location_id": str(item["location_id"]),
                    "difficulty": difficulty,
                    "difficulty_bucket": _difficulty_bucket(difficulty),
                    "embedding": embedding,
                    "sorted_states": sorted_states,
                    "sorted_mask": sorted_mask,
                    "plan_features": plan_features,
                    "feature_metrics": feature_metrics,
                }
            )

    train_by_location: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for record in split_records["train"]:
        train_by_location[str(record["location_id"])].append(record)

    prototype_states: List[np.ndarray] = []
    prototype_masks: List[np.ndarray] = []
    prototype_plan_features: List[np.ndarray] = []
    prototype_feature_metrics: List[np.ndarray] = []
    prototype_difficulty_mean: List[float] = []
    prototype_location_ids: List[str] = []
    prototype_slice_ids: List[str] = []
    prototype_difficulty_buckets: List[str] = []
    cluster_metadata: List[Dict[str, object]] = []
    centroid_cache: Dict[Tuple[str, str], Tuple[StandardScaler, np.ndarray, List[int]]] = {}
    global_bucket_records: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    global_prototype_offset = 0

    for location_id, records in train_by_location.items():
        bucket_records: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        if difficulty_bucketed:
            for record in records:
                bucket_records[str(record["difficulty_bucket"])].append(record)
                global_bucket_records[str(record["difficulty_bucket"])].append(record)
        else:
            bucket_records["all"] = list(records)
            global_bucket_records["all"].extend(records)
        bucket_sizes = {bucket: len(items) for bucket, items in bucket_records.items()}
        cluster_allocation = _allocate_bucket_clusters(bucket_sizes, max_prototypes_per_location)
        for bucket, bucket_items in bucket_records.items():
            if not bucket_items:
                continue
            num_clusters = max(1, min(cluster_allocation.get(bucket, 0), len(bucket_items)))
            if num_clusters <= 0:
                continue
            embeddings = np.stack([record["embedding"] for record in bucket_items], axis=0)
            scaler = StandardScaler()
            scaled = scaler.fit_transform(embeddings)
            if len(bucket_items) <= num_clusters:
                labels = np.arange(len(bucket_items), dtype=np.int64)
                centroids = scaled
                exemplar_indices = list(range(len(bucket_items)))
            else:
                kmeans = MiniBatchKMeans(
                    n_clusters=num_clusters,
                    random_state=seed,
                    batch_size=min(2048, len(bucket_items)),
                    n_init=10,
                )
                labels = kmeans.fit_predict(scaled)
                centroids = kmeans.cluster_centers_
                distances = ((scaled[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1)
                exemplar_indices = distances.argmin(axis=0).tolist()
            cluster_sizes = np.bincount(labels, minlength=len(exemplar_indices))
            global_ids_for_bucket: List[int] = []
            for local_cluster_idx, exemplar_idx in enumerate(exemplar_indices):
                exemplar = bucket_items[int(exemplar_idx)]
                prototype_states.append(np.asarray(exemplar["sorted_states"], dtype=np.float32))
                prototype_masks.append(np.asarray(exemplar["sorted_mask"], dtype=np.bool_))
                prototype_plan_features.append(np.asarray(exemplar["plan_features"], dtype=np.float32))
                prototype_feature_metrics.append(
                    _scene_metric_vector(
                        exemplar["feature_metrics"],
                        map_radius_m=map_radius_m,
                        max_speed_mps=max_speed_mps,
                        max_agents=max_agents,
                    )
                )
                member_difficulties = [
                    float(bucket_items[idx]["difficulty"])
                    for idx, lbl in enumerate(labels)
                    if lbl == local_cluster_idx
                ]
                prototype_difficulty_mean.append(
                    float(np.mean(member_difficulties)) if member_difficulties else float(exemplar["difficulty"])
                )
                prototype_location_ids.append(location_id)
                prototype_slice_ids.append(str(exemplar["slice_id"]))
                prototype_difficulty_buckets.append(bucket)
                global_ids_for_bucket.append(global_prototype_offset + local_cluster_idx)
            centroid_cache[(location_id, bucket)] = (scaler, centroids, global_ids_for_bucket)
            global_prototype_offset += len(exemplar_indices)
            cluster_metadata.append(
                {
                    "location_id": location_id,
                    "difficulty_bucket": bucket,
                    "num_train_records": len(bucket_items),
                    "num_prototypes": len(exemplar_indices),
                    "mean_cluster_size": float(cluster_sizes.mean()) if len(cluster_sizes) else 0.0,
                    "median_cluster_size": float(np.median(cluster_sizes)) if len(cluster_sizes) else 0.0,
                    "singleton_fraction": float((cluster_sizes <= 1).mean()) if len(cluster_sizes) else 0.0,
                }
            )

    for bucket, bucket_items in global_bucket_records.items():
        if not bucket_items:
            continue
        embeddings = np.stack([record["embedding"] for record in bucket_items], axis=0)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(embeddings)
        bucket_proto_ids = [
            idx
            for idx, proto_bucket in enumerate(prototype_difficulty_buckets)
            if proto_bucket == bucket
        ]
        if not bucket_proto_ids:
            continue
        prototype_embedding_rows = []
        for proto_id in bucket_proto_ids:
            slice_id = prototype_slice_ids[proto_id]
            exemplar = next(record for record in bucket_items if str(record["slice_id"]) == slice_id)
            prototype_embedding_rows.append(np.asarray(exemplar["embedding"], dtype=np.float32))
        prototype_embedding_rows_np = np.stack(prototype_embedding_rows, axis=0)
        prototype_centroids = scaler.transform(prototype_embedding_rows_np)
        centroid_cache[("__global__", bucket)] = (scaler, prototype_centroids, bucket_proto_ids)

    assignments: Dict[str, Dict[str, int]] = {split: {} for split in split_records}
    for split, records in split_records.items():
        for record in records:
            location_id = str(record["location_id"])
            bucket = str(record["difficulty_bucket"]) if difficulty_bucketed else "all"
            cache_key = (location_id, bucket)
            if cache_key not in centroid_cache:
                cache_key = ("__global__", bucket)
            if cache_key not in centroid_cache:
                continue
            scaler, centroids, global_ids = centroid_cache[cache_key]
            scaled = scaler.transform(np.expand_dims(np.asarray(record["embedding"], dtype=np.float32), axis=0))
            cluster_idx = int(((scaled - centroids) ** 2).sum(axis=-1).argmin())
            assignments[split][str(record["slice_id"])] = int(global_ids[cluster_idx])

    bank = {
        "prototype_states": torch.as_tensor(np.stack(prototype_states, axis=0), dtype=torch.float32),
        "prototype_masks": torch.as_tensor(np.stack(prototype_masks, axis=0), dtype=torch.bool),
        "prototype_plan_features": torch.as_tensor(np.stack(prototype_plan_features, axis=0), dtype=torch.float32),
        "prototype_feature_metrics": torch.as_tensor(np.stack(prototype_feature_metrics, axis=0), dtype=torch.float32),
        "prototype_feature_metric_names": list(SCENE_METRIC_NAMES),
        "prototype_difficulty_mean": torch.as_tensor(np.asarray(prototype_difficulty_mean, dtype=np.float32)),
        "prototype_location_ids": prototype_location_ids,
        "prototype_difficulty_buckets": prototype_difficulty_buckets,
        "prototype_slice_ids": prototype_slice_ids,
        "radial_bins": radial_bins,
        "angular_bins": angular_bins,
    }
    torch.save(bank, bank_path)
    assignment_path.write_text(json.dumps(assignments), encoding="utf-8")
    save_json(
        {
            "bank_version": SUPPORT_PROTOTYPE_BANK_VERSION,
            "settings_signature": _prototype_bank_settings_signature(config),
            "num_prototypes": len(prototype_states),
            "avg_prototypes_per_location": float(np.mean([row["num_prototypes"] for row in cluster_metadata])) if cluster_metadata else 0.0,
            "locations": cluster_metadata,
        },
        metadata_path,
    )
    return {
        "bank_path": bank_path,
        "assignment_path": assignment_path,
        "metadata_path": metadata_path,
    }


def ensure_support_prototype_bank(cache_dir: str, config: Dict) -> Dict[str, Path]:
    prototype_dir = _resolve_prototype_dir(cache_dir, config)
    bank_path = prototype_dir / "prototype_bank.pt"
    assignment_path = prototype_dir / "prototype_assignments.json"
    metadata_path = prototype_dir / "prototype_metadata.json"
    if bank_path.exists() and assignment_path.exists() and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                int(metadata.get("bank_version", 0)) == SUPPORT_PROTOTYPE_BANK_VERSION
                and metadata.get("settings_signature") == _prototype_bank_settings_signature(config)
            ):
                return {"bank_path": bank_path, "assignment_path": assignment_path, "metadata_path": metadata_path}
        except Exception:
            pass
    return build_support_prototype_bank(cache_dir, config)


def load_support_prototype_bank(path: str | Path) -> Dict:
    return torch.load(Path(path), map_location="cpu")


def load_support_prototype_assignments(path: str | Path) -> Dict[str, Dict[str, int]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
