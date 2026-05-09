from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


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
            "heading_std": 0.0,
        }
    speed = np.linalg.norm(valid[:, 2:4], axis=-1)
    radius = np.linalg.norm(valid[:, 0:2], axis=-1)
    pairwise = np.linalg.norm(valid[:, None, 0:2] - valid[None, :, 0:2], axis=-1)
    if len(valid) > 1:
        pairwise = pairwise + np.eye(len(valid), dtype=np.float32) * 1e6
        mean_pair_distance = float(pairwise[pairwise < 1e5].mean())
        min_pairwise_distance = float(pairwise.min())
        conflict_count = float((pairwise < conflict_distance_m).sum() / 2.0)
    else:
        mean_pair_distance = 100.0
        min_pairwise_distance = 100.0
        conflict_count = 0.0
    heading = valid[:, 4]
    heading_complex = np.exp(1j * heading.astype(np.float64))
    heading_std = float(np.sqrt(max(0.0, 1.0 - np.abs(heading_complex.mean()))))
    return {
        "count": float(len(valid)),
        "mean_speed": float(speed.mean()),
        "std_speed": float(speed.std()),
        "mean_radius": float(radius.mean()),
        "std_radius": float(radius.std()),
        "mean_pair_distance": mean_pair_distance,
        "min_pairwise_distance": min_pairwise_distance,
        "conflict_count": conflict_count,
        "heading_std": heading_std,
    }


def _stack_feature_rows(rows: List[Dict[str, float]]) -> tuple[np.ndarray, List[str]]:
    if not rows:
        return np.zeros((0, 0), dtype=np.float32), []
    keys = list(rows[0].keys())
    matrix = np.asarray([[float(row[key]) for key in keys] for row in rows], dtype=np.float32)
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0), keys


def _swd(real: np.ndarray, fake: np.ndarray, num_projections: int = 32) -> float:
    if len(real) == 0 or len(fake) == 0:
        return 0.0
    rng = np.random.default_rng(0)
    values = []
    for _ in range(num_projections):
        projection = rng.normal(size=(real.shape[1],))
        projection = projection / max(np.linalg.norm(projection), 1e-6)
        values.append(wasserstein_distance(real @ projection, fake @ projection))
    return float(np.mean(values))


def _c2st_auc(real: np.ndarray, fake: np.ndarray) -> float:
    if len(real) < 4 or len(fake) < 4:
        return 0.5
    x = np.concatenate([real, fake], axis=0)
    y = np.concatenate([np.ones(len(real)), np.zeros(len(fake))], axis=0)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0, stratify=y)
    clf = LogisticRegression(max_iter=500)
    clf.fit(x_train, y_train)
    probs = clf.predict_proba(x_test)[:, 1]
    return float(roc_auc_score(y_test, probs))


def _flatten_state(states: np.ndarray, mask: np.ndarray, target_dim: int = 128) -> np.ndarray:
    valid = np.asarray(states, dtype=np.float32)[np.asarray(mask, dtype=bool)]
    flat = valid.reshape(-1)
    if len(flat) >= target_dim:
        return flat[:target_dim]
    padded = np.zeros((target_dim,), dtype=np.float32)
    padded[: len(flat)] = flat
    return padded


def _diversity_score(matrix: np.ndarray, block_size: int = 256) -> float:
    matrix = np.asarray(matrix, dtype=np.float32)
    num_rows = int(matrix.shape[0])
    if num_rows <= 1:
        return 0.0
    total = 0.0
    for start_i in range(0, num_rows, block_size):
        end_i = min(start_i + block_size, num_rows)
        block_i = matrix[start_i:end_i]
        for start_j in range(start_i, num_rows, block_size):
            end_j = min(start_j + block_size, num_rows)
            block_j = matrix[start_j:end_j]
            distances = np.linalg.norm(block_i[:, None, :] - block_j[None, :, :], axis=-1)
            if start_i == start_j:
                tri_i, tri_j = np.triu_indices(end_i - start_i, k=1)
                total += float(distances[tri_i, tri_j].sum()) * 2.0
            else:
                total += float(distances.sum()) * 2.0
    return total / float(num_rows * num_rows)


def _offroad_distance_summary(states: np.ndarray, mask: np.ndarray, record: Dict) -> Dict[str, float]:
    if "map_polylines" not in record:
        return {"mean": 0.0, "max": 0.0, "over_3m_rate": 0.0, "over_5m_rate": 0.0, "available": 0.0}
    map_polylines = np.asarray(record["map_polylines"], dtype=np.float32)
    map_point_mask = np.asarray(record["map_point_mask"], dtype=bool)
    map_polyline_mask = np.asarray(
        record.get("map_polyline_mask", np.ones(map_polylines.shape[0], dtype=bool)),
        dtype=bool,
    )
    valid_points = map_polylines[map_point_mask & map_polyline_mask[:, None]][:, 0:2]
    valid_states = np.asarray(states, dtype=np.float32)[np.asarray(mask, dtype=bool)]
    if len(valid_points) == 0 or len(valid_states) == 0:
        return {"mean": 0.0, "max": 0.0, "over_3m_rate": 0.0, "over_5m_rate": 0.0, "available": 0.0}
    distances = np.linalg.norm(valid_states[:, None, 0:2] - valid_points[None, :, :], axis=-1).min(axis=-1)
    return {
        "mean": float(distances.mean()),
        "max": float(distances.max()),
        "over_3m_rate": float((distances > 3.0).mean()),
        "over_5m_rate": float((distances > 5.0).mean()),
        "available": 1.0,
    }


def _masked_agent_state_distance(
    left_states: np.ndarray,
    left_mask: np.ndarray,
    right_states: np.ndarray,
    right_mask: np.ndarray,
) -> float:
    left_states = np.asarray(left_states, dtype=np.float32)
    right_states = np.asarray(right_states, dtype=np.float32)
    left_mask = np.asarray(left_mask, dtype=bool)
    right_mask = np.asarray(right_mask, dtype=bool)
    valid = left_mask & right_mask
    if not np.any(valid):
        return 0.0
    left = left_states[valid]
    right = right_states[valid]
    pos = np.linalg.norm(left[:, 0:2] - right[:, 0:2], axis=-1)
    speed = np.abs(np.linalg.norm(left[:, 2:4], axis=-1) - np.linalg.norm(right[:, 2:4], axis=-1))
    heading = np.abs(np.arctan2(np.sin(left[:, 4] - right[:, 4]), np.cos(left[:, 4] - right[:, 4])))
    return float(np.mean(pos + speed + heading))


def _prototype_adaptation_rows(records: List[Dict]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for record in records:
        prototype_states = record.get("selected_prototype_states")
        prototype_mask = record.get("selected_prototype_mask")
        if prototype_states is None or prototype_mask is None:
            continue
        proto_states = np.asarray(prototype_states, dtype=np.float32)
        proto_mask = np.asarray(prototype_mask, dtype=bool)
        gen_states = np.asarray(record["generated_current_states"], dtype=np.float32)
        gen_mask = np.asarray(record["generated_agent_mask"], dtype=bool)
        gt_states = np.asarray(record["gt_current_states"], dtype=np.float32)
        gt_mask = np.asarray(record["gt_agent_mask"], dtype=bool)
        gen_features = _scene_feature_vector(gen_states, gen_mask)
        proto_features = _scene_feature_vector(proto_states, proto_mask)
        gt_features = _scene_feature_vector(gt_states, gt_mask)
        shared = proto_mask & gen_mask
        if np.any(shared):
            shift = np.linalg.norm(gen_states[shared, 0:2] - proto_states[shared, 0:2], axis=-1)
            speed_delta = (
                np.linalg.norm(gen_states[shared, 2:4], axis=-1)
                - np.linalg.norm(proto_states[shared, 2:4], axis=-1)
            )
        else:
            shift = np.asarray([0.0], dtype=np.float32)
            speed_delta = np.asarray([0.0], dtype=np.float32)
        rows.append(
            {
                "prototype_id": float(record.get("selected_prototype_id", -1) or -1),
                "target_difficulty": float(record.get("target_difficulty", 0.0)),
                "prototype_count": float(proto_mask.sum()),
                "generated_count": float(gen_mask.sum()),
                "gt_count": float(gt_mask.sum()),
                "keep_count": float(np.logical_and(proto_mask, gen_mask).sum()),
                "drop_count": float(np.logical_and(proto_mask, ~gen_mask).sum()),
                "add_count": float(np.logical_and(~proto_mask, gen_mask).sum()),
                "keep_rate": float(np.logical_and(proto_mask, gen_mask).sum() / max(float(proto_mask.sum()), 1.0)),
                "mean_position_shift": float(np.mean(shift)),
                "max_position_shift": float(np.max(shift)),
                "mean_speed_delta": float(np.mean(speed_delta)),
                "mean_abs_speed_delta": float(np.mean(np.abs(speed_delta))),
                "generated_to_prototype_state_distance": _masked_agent_state_distance(
                    gen_states,
                    gen_mask,
                    proto_states,
                    proto_mask,
                ),
                "gt_to_prototype_state_distance": _masked_agent_state_distance(
                    gt_states,
                    gt_mask,
                    proto_states,
                    proto_mask,
                ),
                "generated_to_gt_state_distance": _masked_agent_state_distance(
                    gen_states,
                    gen_mask,
                    gt_states,
                    gt_mask,
                ),
                "count_delta_generated_minus_proto": gen_features["count"] - proto_features["count"],
                "radius_delta_generated_minus_proto": gen_features["mean_radius"] - proto_features["mean_radius"],
                "pair_distance_delta_generated_minus_proto": gen_features["mean_pair_distance"] - proto_features["mean_pair_distance"],
                "min_pair_distance_delta_generated_minus_proto": gen_features["min_pairwise_distance"] - proto_features["min_pairwise_distance"],
                "conflict_delta_generated_minus_proto": gen_features["conflict_count"] - proto_features["conflict_count"],
                "speed_delta_generated_minus_proto": gen_features["mean_speed"] - proto_features["mean_speed"],
                "radius_error_generated": abs(gen_features["mean_radius"] - gt_features["mean_radius"]),
                "radius_error_prototype": abs(proto_features["mean_radius"] - gt_features["mean_radius"]),
                "pair_distance_error_generated": abs(gen_features["mean_pair_distance"] - gt_features["mean_pair_distance"]),
                "pair_distance_error_prototype": abs(proto_features["mean_pair_distance"] - gt_features["mean_pair_distance"]),
                "conflict_error_generated": abs(gen_features["conflict_count"] - gt_features["conflict_count"]),
                "conflict_error_prototype": abs(proto_features["conflict_count"] - gt_features["conflict_count"]),
            }
        )
    return rows


def analyze_spawn_records(
    records: List[Dict],
    train_states: Iterable[Dict] | None = None,
) -> tuple[Dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    generated_rows = [
        _scene_feature_vector(record["generated_current_states"], record["generated_agent_mask"])
        for record in records
    ]
    real_rows = [
        _scene_feature_vector(record["gt_current_states"], record["gt_agent_mask"])
        for record in records
    ]
    fake, feature_names = _stack_feature_rows(generated_rows)
    real, _ = _stack_feature_rows(real_rows)
    realism_row = {
        f"{feature_name}_w1": float(wasserstein_distance(real[:, idx], fake[:, idx])) if len(real) and len(fake) else 0.0
        for idx, feature_name in enumerate(feature_names)
    }
    realism_row["sliced_wasserstein"] = _swd(real, fake)
    realism_row["c2st_auc"] = _c2st_auc(real, fake)

    count_mae = float(np.mean(np.abs(fake[:, feature_names.index("count")] - real[:, feature_names.index("count")]))) if len(real) else 0.0
    speed_mae = float(np.mean(np.abs(fake[:, feature_names.index("mean_speed")] - real[:, feature_names.index("mean_speed")]))) if len(real) else 0.0
    min_pairwise_mae = float(
        np.mean(np.abs(fake[:, feature_names.index("min_pairwise_distance")] - real[:, feature_names.index("min_pairwise_distance")]))
    ) if len(real) else 0.0
    metrics = {
        "count_mae": count_mae,
        "mean_speed_mae": speed_mae,
        "min_pairwise_distance_mae": min_pairwise_mae,
        "spawn_sliced_wasserstein": realism_row["sliced_wasserstein"],
        "spawn_c2st_auc": realism_row["c2st_auc"],
        "spawn_count_w1": realism_row.get("count_w1", 0.0),
        "spawn_conflict_count_w1": realism_row.get("conflict_count_w1", 0.0),
        "spawn_min_pairwise_distance_w1": realism_row.get("min_pairwise_distance_w1", 0.0),
    }
    selected_prototype_counts = [record.get("selected_prototype_count") for record in records if record.get("selected_prototype_count") is not None]
    if selected_prototype_counts:
        metrics["selected_prototype_count_mean"] = float(np.mean(selected_prototype_counts))
        metrics["selected_prototype_count_mae"] = float(
            np.mean(np.abs(np.asarray(selected_prototype_counts, dtype=np.float32) - real[:, feature_names.index("count")]))
        )
    generated_offroad = [
        _offroad_distance_summary(record["generated_current_states"], record["generated_agent_mask"], record)
        for record in records
    ]
    gt_offroad = [
        _offroad_distance_summary(record["gt_current_states"], record["gt_agent_mask"], record)
        for record in records
    ]
    prototype_offroad = [
        _offroad_distance_summary(record["selected_prototype_states"], record["selected_prototype_mask"], record)
        for record in records
        if record.get("selected_prototype_states") is not None and record.get("selected_prototype_mask") is not None
    ]
    metrics["map_constraint_available_rate"] = (
        float(np.mean([item["available"] for item in generated_offroad])) if generated_offroad else 0.0
    )
    generated_offroad_available = [item for item in generated_offroad if item["available"] > 0.5]
    gt_offroad_available = [item for item in gt_offroad if item["available"] > 0.5]
    prototype_offroad_available = [item for item in prototype_offroad if item["available"] > 0.5]
    if generated_offroad_available:
        metrics["generated_offroad_mean"] = float(np.mean([item["mean"] for item in generated_offroad_available]))
        metrics["generated_offroad_max_mean"] = float(np.mean([item["max"] for item in generated_offroad_available]))
        metrics["generated_offroad_max_p95"] = float(np.percentile([item["max"] for item in generated_offroad_available], 95))
        metrics["generated_offroad_over_3m_rate"] = float(np.mean([item["over_3m_rate"] for item in generated_offroad_available]))
        metrics["generated_offroad_over_5m_rate"] = float(np.mean([item["over_5m_rate"] for item in generated_offroad_available]))
    if gt_offroad_available:
        metrics["gt_offroad_mean"] = float(np.mean([item["mean"] for item in gt_offroad_available]))
        metrics["gt_offroad_max_mean"] = float(np.mean([item["max"] for item in gt_offroad_available]))
        metrics["gt_offroad_over_3m_rate"] = float(np.mean([item["over_3m_rate"] for item in gt_offroad_available]))
        metrics["gt_offroad_over_5m_rate"] = float(np.mean([item["over_5m_rate"] for item in gt_offroad_available]))
    if prototype_offroad_available:
        metrics["prototype_offroad_mean"] = float(np.mean([item["mean"] for item in prototype_offroad_available]))
        metrics["prototype_offroad_max_mean"] = float(np.mean([item["max"] for item in prototype_offroad_available]))
        metrics["prototype_offroad_over_3m_rate"] = float(np.mean([item["over_3m_rate"] for item in prototype_offroad_available]))
        metrics["prototype_offroad_over_5m_rate"] = float(np.mean([item["over_5m_rate"] for item in prototype_offroad_available]))

    train_feature_vectors: List[np.ndarray] = []
    if train_states is not None:
        for item in train_states:
            train_feature_vectors.append(_flatten_state(item["current_states"], item["agent_mask"]))
    generated_vectors = [_flatten_state(record["generated_current_states"], record["generated_agent_mask"]) for record in records]
    novelty_scores: List[float] = []
    if train_feature_vectors:
        train_matrix = np.stack(train_feature_vectors, axis=0)
        for vector, record in zip(generated_vectors, records):
            distances = np.linalg.norm(train_matrix - vector[None, :], axis=-1)
            gt_vector = _flatten_state(record["gt_current_states"], record["gt_agent_mask"])
            gt_copy_distance = float(np.linalg.norm(vector - gt_vector))
            nn_distance = float(distances.min())
            novelty_scores.append(nn_distance / (nn_distance + gt_copy_distance + 1e-6))
    metrics["novelty_score"] = float(np.mean(novelty_scores)) if novelty_scores else 0.0
    if len(generated_vectors) > 1:
        matrix = np.stack(generated_vectors, axis=0)
        metrics["diversity_score"] = _diversity_score(matrix)
    else:
        metrics["diversity_score"] = 0.0

    realism_df = pd.DataFrame([realism_row])
    feature_df = pd.DataFrame(
        [
            {
                "feature": feature_name,
                "generated_mean": float(fake[:, idx].mean()) if len(fake) else 0.0,
                "real_mean": float(real[:, idx].mean()) if len(real) else 0.0,
                "w1": float(wasserstein_distance(real[:, idx], fake[:, idx])) if len(real) and len(fake) else 0.0,
            }
            for idx, feature_name in enumerate(feature_names)
        ]
    )
    difficulty_rows: List[Dict[str, float | str | int]] = []
    if records:
        for bucket in ("low", "mid", "high"):
            bucket_indices = [
                idx
                for idx, record in enumerate(records)
                if _difficulty_bucket(float(record.get("target_difficulty", 0.0))) == bucket
            ]
            if not bucket_indices:
                continue
            bucket_fake = fake[bucket_indices]
            bucket_real = real[bucket_indices]
            bucket_difficulties = np.asarray(
                [float(records[idx].get("target_difficulty", 0.0)) for idx in bucket_indices],
                dtype=np.float32,
            )
            bucket_proto_counts = [
                float(records[idx]["selected_prototype_count"])
                for idx in bucket_indices
                if records[idx].get("selected_prototype_count") is not None
            ]
            bucket_proto_difficulties = [
                float(records[idx]["selected_prototype_difficulty_mean"])
                for idx in bucket_indices
                if records[idx].get("selected_prototype_difficulty_mean") is not None
            ]
            difficulty_rows.append(
                {
                    "bucket": bucket,
                    "num_scenes": int(len(bucket_indices)),
                    "target_difficulty_mean": float(bucket_difficulties.mean()),
                    "generated_count_mean": float(bucket_fake[:, feature_names.index("count")].mean()),
                    "real_count_mean": float(bucket_real[:, feature_names.index("count")].mean()),
                    "generated_mean_speed_mean": float(bucket_fake[:, feature_names.index("mean_speed")].mean()),
                    "real_mean_speed_mean": float(bucket_real[:, feature_names.index("mean_speed")].mean()),
                    "generated_mean_radius_mean": float(bucket_fake[:, feature_names.index("mean_radius")].mean()),
                    "real_mean_radius_mean": float(bucket_real[:, feature_names.index("mean_radius")].mean()),
                    "generated_mean_pair_distance_mean": float(bucket_fake[:, feature_names.index("mean_pair_distance")].mean()),
                    "real_mean_pair_distance_mean": float(bucket_real[:, feature_names.index("mean_pair_distance")].mean()),
                    "generated_min_pairwise_distance_mean": float(bucket_fake[:, feature_names.index("min_pairwise_distance")].mean()),
                    "real_min_pairwise_distance_mean": float(bucket_real[:, feature_names.index("min_pairwise_distance")].mean()),
                    "generated_conflict_count_mean": float(bucket_fake[:, feature_names.index("conflict_count")].mean()),
                    "real_conflict_count_mean": float(bucket_real[:, feature_names.index("conflict_count")].mean()),
                    "selected_prototype_count_mean": float(np.mean(bucket_proto_counts)) if bucket_proto_counts else 0.0,
                    "selected_prototype_difficulty_mean": float(np.mean(bucket_proto_difficulties)) if bucket_proto_difficulties else 0.0,
                }
            )
    difficulty_df = pd.DataFrame(difficulty_rows)
    prototype_adaptation_rows = _prototype_adaptation_rows(records)
    prototype_adaptation_df = pd.DataFrame(prototype_adaptation_rows)
    if not prototype_adaptation_df.empty:
        for key in [
            "keep_rate",
            "mean_position_shift",
            "max_position_shift",
            "mean_abs_speed_delta",
            "generated_to_prototype_state_distance",
            "gt_to_prototype_state_distance",
            "generated_to_gt_state_distance",
            "radius_error_generated",
            "radius_error_prototype",
            "pair_distance_error_generated",
            "pair_distance_error_prototype",
            "conflict_error_generated",
            "conflict_error_prototype",
        ]:
            metrics[f"prototype_adaptation_{key}_mean"] = float(prototype_adaptation_df[key].mean())
    return metrics, realism_df, feature_df, difficulty_df, prototype_adaptation_df


def write_spawn_report(
    root: Path,
    config: Dict,
    metrics: Dict[str, float],
    difficulty_df: pd.DataFrame | None = None,
) -> None:
    lines = [
        "# Spawn Analysis Report",
        "",
        "## Runtime Summary",
        "",
        f"- experiment: `{config.get('experiment', {}).get('name', root.name)}`",
        f"- spawn_architecture: `{config.get('spawn', {}).get('architecture', 'interaction_autoregressive')}`",
        f"- spawn_hidden_dim: `{config.get('spawn', {}).get('hidden_dim', 128)}`",
        f"- spawn_noise_dim: `{config.get('spawn', {}).get('noise_dim', 64)}`",
        f"- spawn_ar_layers: `{config.get('spawn', {}).get('num_ar_layers', config.get('spawn', {}).get('num_slot_layers', 2))}`",
        f"- spawn_ar_heads: `{config.get('spawn', {}).get('num_ar_heads', config.get('spawn', {}).get('num_slot_heads', 4))}`",
        f"- spawn_plan_radial_bins: `{config.get('spawn', {}).get('plan_radial_bins', 4)}`",
        f"- spawn_plan_angular_bins: `{config.get('spawn', {}).get('plan_angular_bins', 8)}`",
        f"- support_guided_prototypes: `{bool(config.get('spawn_prototypes'))}`",
        "- spawn_sampling_policy: `count_distribution_topk`",
        f"- spawn_eval_split: `{config.get('spawn_eval', {}).get('split', 'test')}`",
        f"- batch_size: `{config.get('training', {}).get('batch_size')}`",
        f"- num_epochs: `{config.get('training', {}).get('num_epochs')}`",
        "",
        "## Current Scene Metrics",
        "",
        f"- count_mae: `{metrics.get('count_mae', 0.0):.6f}`",
        f"- mean_speed_mae: `{metrics.get('mean_speed_mae', 0.0):.6f}`",
        f"- min_pairwise_distance_mae: `{metrics.get('min_pairwise_distance_mae', 0.0):.6f}`",
        f"- spawn_sliced_wasserstein: `{metrics.get('spawn_sliced_wasserstein', 0.0):.6f}`",
        f"- spawn_c2st_auc: `{metrics.get('spawn_c2st_auc', 0.0):.6f}`",
        f"- selected_prototype_count_mean: `{metrics.get('selected_prototype_count_mean', 0.0):.6f}`",
        f"- selected_prototype_count_mae: `{metrics.get('selected_prototype_count_mae', 0.0):.6f}`",
        f"- novelty_score: `{metrics.get('novelty_score', 0.0):.6f}`",
        f"- diversity_score: `{metrics.get('diversity_score', 0.0):.6f}`",
        "",
        "## Prototype Adaptation",
        "",
        f"- keep_rate_mean: `{metrics.get('prototype_adaptation_keep_rate_mean', 0.0):.6f}`",
        f"- mean_position_shift: `{metrics.get('prototype_adaptation_mean_position_shift_mean', 0.0):.6f}`",
        f"- max_position_shift: `{metrics.get('prototype_adaptation_max_position_shift_mean', 0.0):.6f}`",
        f"- mean_abs_speed_delta: `{metrics.get('prototype_adaptation_mean_abs_speed_delta_mean', 0.0):.6f}`",
        f"- generated_to_prototype_state_distance: `{metrics.get('prototype_adaptation_generated_to_prototype_state_distance_mean', 0.0):.6f}`",
        f"- gt_to_prototype_state_distance: `{metrics.get('prototype_adaptation_gt_to_prototype_state_distance_mean', 0.0):.6f}`",
        f"- radius_error_generated/prototype: `{metrics.get('prototype_adaptation_radius_error_generated_mean', 0.0):.6f}` / `{metrics.get('prototype_adaptation_radius_error_prototype_mean', 0.0):.6f}`",
        f"- pair_distance_error_generated/prototype: `{metrics.get('prototype_adaptation_pair_distance_error_generated_mean', 0.0):.6f}` / `{metrics.get('prototype_adaptation_pair_distance_error_prototype_mean', 0.0):.6f}`",
        f"- conflict_error_generated/prototype: `{metrics.get('prototype_adaptation_conflict_error_generated_mean', 0.0):.6f}` / `{metrics.get('prototype_adaptation_conflict_error_prototype_mean', 0.0):.6f}`",
        "",
        "## Map Constraint Diagnostics",
        "",
        f"- map_constraint_available_rate: `{metrics.get('map_constraint_available_rate', 0.0):.6f}`",
        f"- generated_offroad_mean: `{metrics.get('generated_offroad_mean', 0.0):.6f}`",
        f"- generated_offroad_max_mean: `{metrics.get('generated_offroad_max_mean', 0.0):.6f}`",
        f"- generated_offroad_max_p95: `{metrics.get('generated_offroad_max_p95', 0.0):.6f}`",
        f"- generated_offroad_over_3m_rate: `{metrics.get('generated_offroad_over_3m_rate', 0.0):.6f}`",
        f"- generated_offroad_over_5m_rate: `{metrics.get('generated_offroad_over_5m_rate', 0.0):.6f}`",
        f"- gt_offroad_mean: `{metrics.get('gt_offroad_mean', 0.0):.6f}`",
        f"- gt_offroad_over_3m_rate: `{metrics.get('gt_offroad_over_3m_rate', 0.0):.6f}`",
        f"- gt_offroad_over_5m_rate: `{metrics.get('gt_offroad_over_5m_rate', 0.0):.6f}`",
        f"- prototype_offroad_mean: `{metrics.get('prototype_offroad_mean', 0.0):.6f}`",
        "",
        "## Difficulty Diagnostics",
        "",
    ]
    if difficulty_df is not None and not difficulty_df.empty:
        for row in difficulty_df.to_dict(orient="records"):
            lines.append(
                "- `{bucket}` n=`{num_scenes}` difficulty=`{target_difficulty_mean:.3f}` "
                "proto_count=`{selected_prototype_count_mean:.2f}` gen_count=`{generated_count_mean:.2f}` "
                "gen_speed=`{generated_mean_speed_mean:.2f}` gen_conflict=`{generated_conflict_count_mean:.2f}` "
                "gen_min_pair=`{generated_min_pairwise_distance_mean:.2f}`".format(**row)
            )
    else:
        lines.append("- no difficulty-bucket diagnostics available")
    lines.extend(
        [
            "",
        "## Notes",
        "",
        "- This is spawn-only current scene evaluation; it does not yet chain into rollout.",
        "- Difficulty conditions the spawn model; retrieval is not used at inference for these generated current scenes.",
        ]
    )
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
