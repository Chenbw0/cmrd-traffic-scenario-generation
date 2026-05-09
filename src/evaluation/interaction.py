from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


def _wrap_angle(delta: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(delta), np.cos(delta))


def _pair_metrics_for_future(
    current_states: np.ndarray,
    future_states: np.ndarray,
    future_mask: np.ndarray,
    conflict_distance_m: float = 5.0,
) -> Dict[str, object]:
    num_agents = int(future_states.shape[0])
    pair_min_distances: List[float] = []
    priority_signs: List[int] = []
    following_pairs = 0
    crossing_pairs = 0
    for src_idx in range(num_agents):
        for dst_idx in range(src_idx + 1, num_agents):
            overlap = future_mask[src_idx] & future_mask[dst_idx]
            if not bool(np.any(overlap)):
                continue
            src_xy = future_states[src_idx, overlap, 0:2]
            dst_xy = future_states[dst_idx, overlap, 0:2]
            min_distance = float(np.linalg.norm(src_xy - dst_xy, axis=-1).min())
            pair_min_distances.append(min_distance)
            src_final_xy = future_states[src_idx, overlap][-1, 0:2]
            dst_final_xy = future_states[dst_idx, overlap][-1, 0:2]
            src_disp = float(np.linalg.norm(src_final_xy - current_states[src_idx, 0:2]))
            dst_disp = float(np.linalg.norm(dst_final_xy - current_states[dst_idx, 0:2]))
            disp_gap = src_disp - dst_disp
            if abs(disp_gap) > 0.5:
                priority_signs.append(1 if disp_gap > 0.0 else -1)
            heading_gap = abs(float(_wrap_angle(np.asarray(current_states[src_idx, 4] - current_states[dst_idx, 4], dtype=np.float32))))
            if heading_gap < 0.35:
                following_pairs += 1
            elif heading_gap > 0.70:
                crossing_pairs += 1
    pair_min_distances_np = np.asarray(pair_min_distances, dtype=np.float32)
    return {
        "pair_min_distances": pair_min_distances_np,
        "conflict_pair_count": float((pair_min_distances_np < conflict_distance_m).sum()) if pair_min_distances_np.size else 0.0,
        "priority_signs": priority_signs,
        "following_pair_count": float(following_pairs),
        "crossing_pair_count": float(crossing_pairs),
    }


def _arrival_step(displacement: np.ndarray, threshold_ratio: float = 0.9) -> int:
    if displacement.size == 0:
        return 0
    final_value = float(displacement[-1])
    if final_value <= 1e-6:
        return int(displacement.size - 1)
    threshold = threshold_ratio * final_value
    reached = np.flatnonzero(displacement >= threshold)
    return int(reached[0]) if reached.size else int(displacement.size - 1)


def _pair_relation_record(
    current_states: np.ndarray,
    future_states: np.ndarray,
    future_mask: np.ndarray,
    conflict_distance_m: float = 5.0,
) -> Dict[str, object]:
    num_agents = int(future_states.shape[0])
    pair_min_distances: List[float] = []
    time_to_min_distance: List[float] = []
    priority_labels: List[str] = []
    following_pairs = 0
    crossing_pairs = 0
    pair_count = 0
    for src_idx in range(num_agents):
        for dst_idx in range(src_idx + 1, num_agents):
            overlap = future_mask[src_idx] & future_mask[dst_idx]
            if not bool(np.any(overlap)):
                continue
            overlap_idx = np.flatnonzero(overlap)
            src_xy = future_states[src_idx, overlap, 0:2]
            dst_xy = future_states[dst_idx, overlap, 0:2]
            distances = np.linalg.norm(src_xy - dst_xy, axis=-1)
            min_local_idx = int(np.argmin(distances))
            pair_min_distances.append(float(distances[min_local_idx]))
            horizon_denom = max(int(overlap_idx[-1]), 1)
            time_to_min_distance.append(float(overlap_idx[min_local_idx]) / float(horizon_denom))

            src_disp = np.linalg.norm(src_xy - current_states[src_idx, 0:2], axis=-1)
            dst_disp = np.linalg.norm(dst_xy - current_states[dst_idx, 0:2], axis=-1)
            src_arrival = _arrival_step(src_disp)
            dst_arrival = _arrival_step(dst_disp)
            if abs(src_arrival - dst_arrival) <= 1:
                priority_labels.append("none")
            elif src_arrival < dst_arrival:
                priority_labels.append("i_before_j")
            else:
                priority_labels.append("j_before_i")

            heading_gap = abs(
                float(
                    _wrap_angle(
                        np.asarray(current_states[src_idx, 4] - current_states[dst_idx, 4], dtype=np.float32)
                    )
                )
            )
            if heading_gap < 0.35:
                following_pairs += 1
            elif heading_gap > 0.70:
                crossing_pairs += 1
            pair_count += 1
    pair_min_distances_np = np.asarray(pair_min_distances, dtype=np.float32)
    time_to_min_distance_np = np.asarray(time_to_min_distance, dtype=np.float32)
    if pair_count <= 0:
        return {
            "pairwise_min_distance": pair_min_distances_np,
            "time_to_min_distance": time_to_min_distance_np,
            "conflict_pair_count": 0.0,
            "following_pair_rate": 0.0,
            "crossing_pair_rate": 0.0,
            "priority_i_before_j_rate": 0.0,
            "priority_j_before_i_rate": 0.0,
            "priority_none_rate": 0.0,
        }
    priority_i_before = float(sum(label == "i_before_j" for label in priority_labels)) / float(pair_count)
    priority_j_before = float(sum(label == "j_before_i" for label in priority_labels)) / float(pair_count)
    priority_none = float(sum(label == "none" for label in priority_labels)) / float(pair_count)
    return {
        "pairwise_min_distance": pair_min_distances_np,
        "time_to_min_distance": time_to_min_distance_np,
        "conflict_pair_count": float((pair_min_distances_np < conflict_distance_m).sum()) if pair_min_distances_np.size else 0.0,
        "following_pair_rate": float(following_pairs) / float(pair_count),
        "crossing_pair_rate": float(crossing_pairs) / float(pair_count),
        "priority_i_before_j_rate": priority_i_before,
        "priority_j_before_i_rate": priority_j_before,
        "priority_none_rate": priority_none,
    }


def compute_interaction_diagnostics(records: List[Dict], conflict_distance_m: float = 5.0) -> pd.DataFrame:
    gt_pair_distances: List[float] = []
    generated_pair_distances: List[float] = []
    gt_conflict_counts: List[float] = []
    generated_conflict_counts: List[float] = []
    priority_matches: List[float] = []
    following_counts: List[float] = []
    crossing_counts: List[float] = []
    for record in records:
        current_states = np.asarray(record.get("current_states"), dtype=np.float32)
        future_mask = np.asarray(record.get("future_mask"), dtype=bool)
        generated_future = np.asarray(record.get("generated_future"), dtype=np.float32)
        gt_future = np.asarray(record.get("gt_future"), dtype=np.float32)
        gt_metrics = _pair_metrics_for_future(current_states, gt_future, future_mask, conflict_distance_m=conflict_distance_m)
        generated_metrics = _pair_metrics_for_future(current_states, generated_future, future_mask, conflict_distance_m=conflict_distance_m)
        gt_pair_distances.extend(gt_metrics["pair_min_distances"].tolist())
        generated_pair_distances.extend(generated_metrics["pair_min_distances"].tolist())
        gt_conflict_counts.append(float(gt_metrics["conflict_pair_count"]))
        generated_conflict_counts.append(float(generated_metrics["conflict_pair_count"]))
        gt_priority = list(gt_metrics["priority_signs"])
        generated_priority = list(generated_metrics["priority_signs"])
        match_count = 0
        compare_count = min(len(gt_priority), len(generated_priority))
        for idx in range(compare_count):
            if gt_priority[idx] == generated_priority[idx]:
                match_count += 1
        priority_matches.append(float(match_count) / float(max(compare_count, 1)))
        following_counts.append(float(gt_metrics["following_pair_count"]))
        crossing_counts.append(float(gt_metrics["crossing_pair_count"]))
    gt_pair_np = np.asarray(gt_pair_distances, dtype=np.float32)
    generated_pair_np = np.asarray(generated_pair_distances, dtype=np.float32)
    row = {
        "count": int(len(records)),
        "generated_conflict_pair_count_mean": float(np.mean(generated_conflict_counts)) if generated_conflict_counts else 0.0,
        "gt_conflict_pair_count_mean": float(np.mean(gt_conflict_counts)) if gt_conflict_counts else 0.0,
        "conflict_pair_count_gap": float(np.mean(generated_conflict_counts) - np.mean(gt_conflict_counts)) if generated_conflict_counts else 0.0,
        "pairwise_min_distance_w1": float(wasserstein_distance(gt_pair_np, generated_pair_np)) if gt_pair_np.size and generated_pair_np.size else 0.0,
        "pair_priority_match_rate": float(np.mean(priority_matches)) if priority_matches else 0.0,
        "following_pair_count_mean": float(np.mean(following_counts)) if following_counts else 0.0,
        "crossing_pair_count_mean": float(np.mean(crossing_counts)) if crossing_counts else 0.0,
    }
    return pd.DataFrame([row])


def compute_pair_relation_diagnostics(
    records: List[Dict],
    conflict_distance_m: float = 5.0,
    difficulty_bin_width: float = 0.1,
) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(
            [
                {
                    "feature": "pairwise_min_distance",
                    "real_mean": 0.0,
                    "generated_mean": 0.0,
                    "W1_or_gap": 0.0,
                    "per_location_gap": "{}",
                    "per_target_difficulty_gap": "{}",
                    "top_gap_rank": 0,
                    "relation_driven": False,
                }
            ]
        )
    record_rows: List[Dict[str, object]] = []
    for record in records:
        current_states = np.asarray(record.get("current_states"), dtype=np.float32)
        future_mask = np.asarray(record.get("future_mask"), dtype=bool)
        generated_future = np.asarray(record.get("generated_future"), dtype=np.float32)
        gt_future = np.asarray(record.get("gt_future"), dtype=np.float32)
        gt_metrics = _pair_relation_record(current_states, gt_future, future_mask, conflict_distance_m=conflict_distance_m)
        generated_metrics = _pair_relation_record(current_states, generated_future, future_mask, conflict_distance_m=conflict_distance_m)
        record_rows.append(
            {
                "location_id": str(record.get("retrieved_location_id", record.get("location_id", ""))),
                "target_difficulty_bin": f"{round(float(record.get('target_difficulty_requested', record.get('target_difficulty', 0.0))) / difficulty_bin_width) * difficulty_bin_width:.1f}",
                "gt": gt_metrics,
                "generated": generated_metrics,
            }
        )

    distribution_features = {
        "pairwise_min_distance": "pairwise_min_distance",
        "time_to_min_distance": "time_to_min_distance",
    }
    scalar_features = {
        "conflict_pair_count": "conflict_pair_count",
        "following_pair_rate": "following_pair_rate",
        "crossing_pair_rate": "crossing_pair_rate",
        "priority_i_before_j_rate": "priority_i_before_j_rate",
        "priority_j_before_i_rate": "priority_j_before_i_rate",
        "priority_none_rate": "priority_none_rate",
    }

    def _distribution_gap(rows: List[Dict[str, object]], feature_key: str) -> tuple[float, float, float]:
        real = np.concatenate(
            [np.asarray(item["gt"][feature_key], dtype=np.float32) for item in rows if np.asarray(item["gt"][feature_key]).size],
            axis=0,
        ) if rows else np.zeros((0,), dtype=np.float32)
        generated = np.concatenate(
            [np.asarray(item["generated"][feature_key], dtype=np.float32) for item in rows if np.asarray(item["generated"][feature_key]).size],
            axis=0,
        ) if rows else np.zeros((0,), dtype=np.float32)
        real_mean = float(real.mean()) if real.size else 0.0
        generated_mean = float(generated.mean()) if generated.size else 0.0
        gap = float(wasserstein_distance(real, generated)) if real.size and generated.size else 0.0
        return real_mean, generated_mean, gap

    def _scalar_gap(rows: List[Dict[str, object]], feature_key: str) -> tuple[float, float, float]:
        real = np.asarray([float(item["gt"][feature_key]) for item in rows], dtype=np.float32)
        generated = np.asarray([float(item["generated"][feature_key]) for item in rows], dtype=np.float32)
        real_mean = float(real.mean()) if real.size else 0.0
        generated_mean = float(generated.mean()) if generated.size else 0.0
        gap = float(abs(generated_mean - real_mean))
        return real_mean, generated_mean, gap

    feature_rows: List[Dict[str, object]] = []
    all_features = list(distribution_features.items()) + list(scalar_features.items())
    for feature_name, feature_key in all_features:
        if feature_name in distribution_features:
            real_mean, generated_mean, gap = _distribution_gap(record_rows, feature_key)
            gap_fn = _distribution_gap
        else:
            real_mean, generated_mean, gap = _scalar_gap(record_rows, feature_key)
            gap_fn = _scalar_gap
        per_location: Dict[str, float] = {}
        for location_id in sorted({str(item["location_id"]) for item in record_rows}):
            subset = [item for item in record_rows if str(item["location_id"]) == location_id]
            _, _, subset_gap = gap_fn(subset, feature_key)
            per_location[location_id] = float(subset_gap)
        per_target: Dict[str, float] = {}
        for bucket in sorted({str(item["target_difficulty_bin"]) for item in record_rows}):
            subset = [item for item in record_rows if str(item["target_difficulty_bin"]) == bucket]
            _, _, subset_gap = gap_fn(subset, feature_key)
            per_target[bucket] = float(subset_gap)
        feature_rows.append(
            {
                "feature": feature_name,
                "real_mean": real_mean,
                "generated_mean": generated_mean,
                "W1_or_gap": gap,
                "per_location_gap": json.dumps(per_location, sort_keys=True),
                "per_target_difficulty_gap": json.dumps(per_target, sort_keys=True),
                "top_gap_rank": 0,
                "relation_driven": True,
            }
        )

    feature_rows.sort(key=lambda item: float(item["W1_or_gap"]), reverse=True)
    for rank, row in enumerate(feature_rows, start=1):
        row["top_gap_rank"] = int(rank)
    return pd.DataFrame(feature_rows)


def compute_interaction_oracle_diagnostics(
    records: List[Dict],
    conflict_distance_m: float = 5.0,
    conflict_temperature_m: float = 1.0,
) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(
            [
                {
                    "real_min_pairwise_distance_mean": 0.0,
                    "real_min_pairwise_distance_std": 0.0,
                    "generated_min_pairwise_distance_mean": 0.0,
                    "generated_min_pairwise_distance_std": 0.0,
                    "min_pairwise_distance_w1": 0.0,
                    "real_conflict_count_mean": 0.0,
                    "real_conflict_count_std": 0.0,
                    "generated_conflict_count_mean": 0.0,
                    "generated_conflict_count_std": 0.0,
                    "conflict_count_w1": 0.0,
                    "soft_conflict_count_mae": 0.0,
                    "pair_distance_quantile_w1": 0.0,
                }
            ]
        )
    gt_pair_distances: List[np.ndarray] = []
    generated_pair_distances: List[np.ndarray] = []
    gt_conflict_counts: List[float] = []
    generated_conflict_counts: List[float] = []
    gt_soft_conflicts: List[float] = []
    generated_soft_conflicts: List[float] = []
    temperature = max(float(conflict_temperature_m), 1e-3)
    for record in records:
        current_states = np.asarray(record.get("current_states"), dtype=np.float32)
        future_mask = np.asarray(record.get("future_mask"), dtype=bool)
        generated_future = np.asarray(record.get("generated_future"), dtype=np.float32)
        gt_future = np.asarray(record.get("gt_future"), dtype=np.float32)
        gt_metrics = _pair_relation_record(current_states, gt_future, future_mask, conflict_distance_m=conflict_distance_m)
        generated_metrics = _pair_relation_record(current_states, generated_future, future_mask, conflict_distance_m=conflict_distance_m)
        gt_distance = np.asarray(gt_metrics["pairwise_min_distance"], dtype=np.float32)
        generated_distance = np.asarray(generated_metrics["pairwise_min_distance"], dtype=np.float32)
        gt_pair_distances.append(gt_distance)
        generated_pair_distances.append(generated_distance)
        gt_conflict_counts.append(float(gt_metrics["conflict_pair_count"]))
        generated_conflict_counts.append(float(generated_metrics["conflict_pair_count"]))
        gt_soft_conflicts.append(float((1.0 / (1.0 + np.exp((gt_distance - conflict_distance_m) / temperature))).sum()) if gt_distance.size else 0.0)
        generated_soft_conflicts.append(float((1.0 / (1.0 + np.exp((generated_distance - conflict_distance_m) / temperature))).sum()) if generated_distance.size else 0.0)

    gt_pair_concat = np.concatenate([item for item in gt_pair_distances if item.size], axis=0) if any(item.size for item in gt_pair_distances) else np.zeros((0,), dtype=np.float32)
    generated_pair_concat = np.concatenate([item for item in generated_pair_distances if item.size], axis=0) if any(item.size for item in generated_pair_distances) else np.zeros((0,), dtype=np.float32)
    quantiles = np.asarray([0.1, 0.25, 0.5, 0.75, 0.9], dtype=np.float32)
    if gt_pair_concat.size and generated_pair_concat.size:
        gt_quantiles = np.quantile(gt_pair_concat, quantiles)
        generated_quantiles = np.quantile(generated_pair_concat, quantiles)
        pair_distance_quantile_w1 = float(np.mean(np.abs(gt_quantiles - generated_quantiles)))
        min_distance_w1 = float(wasserstein_distance(gt_pair_concat, generated_pair_concat))
    else:
        pair_distance_quantile_w1 = 0.0
        min_distance_w1 = 0.0
    conflict_w1 = float(wasserstein_distance(np.asarray(gt_conflict_counts, dtype=np.float32), np.asarray(generated_conflict_counts, dtype=np.float32))) if gt_conflict_counts and generated_conflict_counts else 0.0
    return pd.DataFrame(
        [
            {
                "real_min_pairwise_distance_mean": float(gt_pair_concat.mean()) if gt_pair_concat.size else 0.0,
                "real_min_pairwise_distance_std": float(gt_pair_concat.std()) if gt_pair_concat.size else 0.0,
                "generated_min_pairwise_distance_mean": float(generated_pair_concat.mean()) if generated_pair_concat.size else 0.0,
                "generated_min_pairwise_distance_std": float(generated_pair_concat.std()) if generated_pair_concat.size else 0.0,
                "min_pairwise_distance_w1": min_distance_w1,
                "real_conflict_count_mean": float(np.mean(gt_conflict_counts)) if gt_conflict_counts else 0.0,
                "real_conflict_count_std": float(np.std(gt_conflict_counts)) if gt_conflict_counts else 0.0,
                "generated_conflict_count_mean": float(np.mean(generated_conflict_counts)) if generated_conflict_counts else 0.0,
                "generated_conflict_count_std": float(np.std(generated_conflict_counts)) if generated_conflict_counts else 0.0,
                "conflict_count_w1": conflict_w1,
                "soft_conflict_count_mae": float(np.mean(np.abs(np.asarray(generated_soft_conflicts) - np.asarray(gt_soft_conflicts)))) if gt_soft_conflicts else 0.0,
                "pair_distance_quantile_w1": pair_distance_quantile_w1,
            }
        ]
    )
