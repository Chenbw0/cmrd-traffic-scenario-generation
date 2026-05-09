from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from isgen.data.cache import load_slices_from_cache


def _trajectory_flat(record: Dict) -> np.ndarray:
    future = np.asarray(record["generated_future"], dtype=np.float32)
    mask = np.asarray(record["future_mask"], dtype=bool)
    valid = future[mask]
    return valid.reshape(-1) if valid.size else np.zeros(1, dtype=np.float32)


def _ade(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return float(np.mean(np.abs(a[:n] - b[:n])))


def compute_diversity_metrics(records: List[Dict], project_root: str | Path, cache_dir: str | Path) -> tuple[pd.DataFrame, List[Dict[str, float]]]:
    groups = {}
    for record in tqdm(records, desc="Grouping records for diversity", unit="sample", leave=False):
        bucket = (
            str(record.get("retrieved_slice_id", record.get("slice_id"))),
            round(record.get("target_difficulty_requested", record.get("target_difficulty", 0.0)) / 0.1) * 0.1,
        )
        groups.setdefault(bucket, []).append(record)
    metrics = {
        "ade_diversity": 0.0,
        "fde_diversity": 0.0,
        "feature_diversity": 0.0,
        "location_coverage": float(len({record["location_id"] for record in records})),
        "retrieved_slice_coverage": float(len({record.get("retrieved_slice_id", record.get("slice_id")) for record in records})),
    }
    pair_ades = []
    pair_fdes = []
    for items in groups.values():
        if len(items) < 2:
            continue
        for idx in range(len(items) - 1):
            a = np.asarray(items[idx]["generated_future"], dtype=np.float32)
            b = np.asarray(items[idx + 1]["generated_future"], dtype=np.float32)
            mask = np.asarray(items[idx]["future_mask"], dtype=bool) & np.asarray(items[idx + 1]["future_mask"], dtype=bool)
            if not mask.any():
                continue
            pair_ades.append(float(np.mean(np.linalg.norm(a[..., 0:2][mask] - b[..., 0:2][mask], axis=-1))))
            pair_fdes.append(float(np.mean(np.linalg.norm(a[:, -1, 0:2] - b[:, -1, 0:2], axis=-1))))
    metrics["ade_diversity"] = float(np.mean(pair_ades)) if pair_ades else 0.0
    metrics["fde_diversity"] = float(np.mean(pair_fdes)) if pair_fdes else 0.0
    feature_vectors = np.stack([_trajectory_flat(record)[:64] for record in records], axis=0) if records else np.zeros((0, 64), dtype=np.float32)
    if len(feature_vectors) > 1:
        metrics["feature_diversity"] = float(np.mean(np.linalg.norm(feature_vectors[:, None] - feature_vectors[None, :], axis=-1)))
    metrics["gt_copy_ade"] = float(np.mean([_ade(_trajectory_flat(record), np.asarray(record["gt_future"], dtype=np.float32).reshape(-1)) for record in records])) if records else 0.0
    train_slices = load_slices_from_cache(Path(project_root) / cache_dir, "train")
    train_futures = [np.asarray(item["future_states"], dtype=np.float32).reshape(-1) for item in train_slices if np.asarray(item["future_mask"]).any()]
    per_sample = []
    novelty_scores = []
    if train_futures:
        target_dim = min(max(len(future) for future in train_futures), 256)
        train_matrix = np.stack(
            [np.pad(future[:target_dim], (0, max(0, target_dim - len(future[:target_dim])))) for future in train_futures],
            axis=0,
        )
        for record in tqdm(records, desc="Computing novelty vs train set", unit="sample", leave=False):
            flat = _trajectory_flat(record)[: train_matrix.shape[1]]
            if len(flat) < train_matrix.shape[1]:
                flat = np.pad(flat, (0, train_matrix.shape[1] - len(flat)))
            distances = np.linalg.norm(train_matrix - flat[None, :], axis=-1)
            nn_distance = float(distances.min())
            gt_distance = float(_ade(flat, np.asarray(record["gt_future"], dtype=np.float32).reshape(-1)))
            novelty = nn_distance / (nn_distance + gt_distance + 1e-6)
            novelty_scores.append(novelty)
            per_sample.append({"sample_id": record["sample_id"], "nn_train_distance": nn_distance, "novelty_score": novelty, "gt_copy_ade": gt_distance})
    metrics["nn_train_distance"] = float(np.mean([item["nn_train_distance"] for item in per_sample])) if per_sample else 0.0
    metrics["novelty_score"] = float(np.mean(novelty_scores)) if novelty_scores else 0.0
    metrics["rollout_diversity_ADE"] = metrics["ade_diversity"]
    metrics["rollout_diversity_FDE"] = metrics["fde_diversity"]
    metrics["rollout_diversity_feature"] = metrics["feature_diversity"]
    metrics["nearest_train_distance"] = metrics["nn_train_distance"]
    metrics["diversity_score"] = metrics["feature_diversity"]
    return pd.DataFrame([metrics]), per_sample
