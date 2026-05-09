from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


def _safe_scalar(value, default: float = 0.0) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(scalar):
        return float(default)
    return scalar


def _safe_array(value, dtype=np.float32) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.size == 0:
        return array
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def _finite_feature_row(row: Dict[str, float]) -> Dict[str, float]:
    return {key: _safe_scalar(value, 0.0) for key, value in row.items()}


def _finite_joint_arrays(real: np.ndarray, fake: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if real.size == 0 or fake.size == 0:
        return real, fake
    real = np.nan_to_num(real.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    fake = np.nan_to_num(fake.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    real_keep = np.isfinite(real).all(axis=1)
    fake_keep = np.isfinite(fake).all(axis=1)
    return real[real_keep], fake[fake_keep]


def _sample_feature_vector(record: Dict) -> Dict[str, float]:
    generated = _safe_array(record["generated_future"], dtype=np.float32)
    gt = _safe_array(record["gt_future"], dtype=np.float32)
    future_mask = np.asarray(record["future_mask"], dtype=bool)
    agent_mask = np.asarray(record["agent_mask"], dtype=bool)
    valid_gen = generated[agent_mask][:, :, :]
    valid_gt = gt[agent_mask][:, :, :]
    valid_mask = future_mask[agent_mask]
    if valid_gen.size == 0 or not valid_mask.any():
        return {
            key: 0.0
            for key in [
                "speed",
                "acceleration",
                "jerk",
                "yaw_rate",
                "control_accel_mean",
                "control_accel_std",
                "control_yaw_rate_mean",
                "control_yaw_rate_std",
                "final_displacement",
                "min_pairwise_distance",
                "ttc_proxy",
                "conflict_count",
                "behavior_quantile",
                "behavior_aggressiveness",
                "stress_selected_agents",
            ]
        }
    velocities = valid_gen[..., 2:4]
    speed_full = np.linalg.norm(velocities, axis=-1)
    acceleration_full = np.linalg.norm(np.diff(velocities, axis=1, prepend=velocities[:, :1]), axis=-1)
    jerk_full = np.abs(np.diff(acceleration_full, axis=1, prepend=acceleration_full[:, :1]))
    yaw_rate_full = np.abs(np.diff(valid_gen[..., 4], axis=1, prepend=valid_gen[:, :1, 4]))
    speed = speed_full[valid_mask]
    acceleration = acceleration_full[valid_mask]
    jerk = jerk_full[valid_mask]
    yaw_rate = yaw_rate_full[valid_mask]
    final_disp = np.linalg.norm(valid_gen[:, -1, 0:2] - valid_gen[:, 0, 0:2], axis=-1)
    pairwise = np.linalg.norm(valid_gen[:, None, -1, 0:2] - valid_gen[None, :, -1, 0:2], axis=-1)
    pairwise[pairwise == 0.0] = 1e6
    min_pairwise = pairwise.min() if len(pairwise) > 1 else 100.0
    rel_pos = valid_gt[:, 0, 0:2][:, None, :] - valid_gt[:, 0, 0:2][None, :, :]
    rel_vel = valid_gt[:, 0, 2:4][:, None, :] - valid_gt[:, 0, 2:4][None, :, :]
    rel_speed_sq = np.maximum(np.sum(rel_vel**2, axis=-1), 1e-3)
    ttc = np.clip(-(rel_pos * rel_vel).sum(axis=-1) / rel_speed_sq, 0.0, 8.0)
    ttc[ttc == 0.0] = 8.0
    return _finite_feature_row({
        "speed": float(speed.mean()) if speed.size else 0.0,
        "acceleration": float(acceleration.mean()) if acceleration.size else 0.0,
        "jerk": float(jerk.mean()) if jerk.size else 0.0,
        "yaw_rate": float(yaw_rate.mean()) if yaw_rate.size else 0.0,
        "control_accel_mean": _safe_scalar(record.get("generated_control_accel_mean", 0.0)),
        "control_accel_std": _safe_scalar(record.get("generated_control_accel_std", 0.0)),
        "control_yaw_rate_mean": _safe_scalar(record.get("generated_control_yaw_rate_mean", 0.0)),
        "control_yaw_rate_std": _safe_scalar(record.get("generated_control_yaw_rate_std", 0.0)),
        "final_displacement": float(final_disp.mean()) if final_disp.size else 0.0,
        "min_pairwise_distance": float(min_pairwise),
        "ttc_proxy": float(ttc.min()) if ttc.size else 8.0,
        "conflict_count": float((pairwise < 5.0).sum() / 2.0) if len(pairwise) > 1 else 0.0,
        "behavior_quantile": _safe_scalar(record.get("generated_behavior_quantile", 0.0)),
        "behavior_aggressiveness": _safe_scalar(record.get("generated_behavior_aggressiveness", 0.0)),
        "stress_selected_agents": _safe_scalar(record.get("generated_stress_difficulty_selected_agents", record.get("generated_difficulty", 0.0))),
    })


def _gt_feature_vector(record: Dict) -> Dict[str, float]:
    clone = dict(record)
    clone["generated_future"] = record["gt_future"]
    clone["generated_behavior_quantile"] = record.get("gt_behavior_quantile", 0.0)
    clone["generated_behavior_aggressiveness"] = record.get("slice_behavior_aggressiveness_selected_agents", 0.0)
    clone["generated_stress_difficulty_selected_agents"] = record.get("slice_stress_difficulty_selected_agents", record.get("slice_difficulty", 0.0))
    clone["generated_control_accel_mean"] = record.get("gt_control_accel_mean", 0.0)
    clone["generated_control_accel_std"] = record.get("gt_control_accel_std", 0.0)
    clone["generated_control_yaw_rate_mean"] = record.get("gt_control_yaw_rate_mean", 0.0)
    clone["generated_control_yaw_rate_std"] = record.get("gt_control_yaw_rate_std", 0.0)
    return _sample_feature_vector(clone)


def _stack_joint_features(feature_rows: List[Dict[str, float]]) -> np.ndarray:
    keys = list(feature_rows[0].keys()) if feature_rows else []
    stacked = np.asarray([[row[key] for key in keys] for row in feature_rows], dtype=np.float32)
    return np.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0)


def _sliced_wasserstein(real: np.ndarray, fake: np.ndarray, num_projections: int = 32) -> float:
    if len(real) == 0 or len(fake) == 0:
        return 0.0
    rng = np.random.default_rng(0)
    distances = []
    for _ in range(num_projections):
        projection = rng.normal(size=(real.shape[1],))
        projection = projection / np.linalg.norm(projection)
        real_proj = real @ projection
        fake_proj = fake @ projection
        distances.append(wasserstein_distance(real_proj, fake_proj))
    return float(np.mean(distances))


def _mmd(real: np.ndarray, fake: np.ndarray, gamma: float = 0.5) -> float:
    if len(real) == 0 or len(fake) == 0:
        return 0.0
    xx = np.exp(-gamma * np.square(np.linalg.norm(real[:, None] - real[None, :], axis=-1))).mean()
    yy = np.exp(-gamma * np.square(np.linalg.norm(fake[:, None] - fake[None, :], axis=-1))).mean()
    xy = np.exp(-gamma * np.square(np.linalg.norm(real[:, None] - fake[None, :], axis=-1))).mean()
    return float(xx + yy - 2.0 * xy)


def _c2st_auc(real: np.ndarray, fake: np.ndarray) -> float:
    real, fake = _finite_joint_arrays(real, fake)
    if len(real) < 4 or len(fake) < 4:
        return 0.5
    x = np.concatenate([real, fake], axis=0)
    y = np.concatenate([np.ones(len(real)), np.zeros(len(fake))], axis=0)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0, stratify=y)
    clf = LogisticRegression(max_iter=500)
    clf.fit(x_train, y_train)
    probs = clf.predict_proba(x_test)[:, 1]
    return float(roc_auc_score(y_test, probs))


def _prdc(real: np.ndarray, fake: np.ndarray, k: int = 5) -> Dict[str, float]:
    real, fake = _finite_joint_arrays(real, fake)
    if len(real) <= k or len(fake) <= k:
        return {"precision": 0.0, "recall": 0.0, "density": 0.0, "coverage": 0.0}
    real_dist = np.linalg.norm(real[:, None] - real[None, :], axis=-1)
    np.fill_diagonal(real_dist, np.inf)
    real_radius = np.partition(real_dist, k, axis=1)[:, k]
    fake_to_real = np.linalg.norm(fake[:, None] - real[None, :], axis=-1)
    real_to_fake = fake_to_real.T
    precision = np.mean(np.any(fake_to_real <= real_radius[None, :], axis=1))
    recall = np.mean(np.any(real_to_fake <= np.partition(fake_to_real, min(k, len(fake) - 1), axis=0)[min(k, len(fake) - 1)], axis=1))
    density = np.mean(np.sum(fake_to_real <= real_radius[None, :], axis=1) / k)
    coverage = np.mean(np.min(real_to_fake, axis=1) <= real_radius)
    return {"precision": float(precision), "recall": float(recall), "density": float(density), "coverage": float(coverage)}


def compute_c2st_feature_diagnostics(records: List[Dict]) -> pd.DataFrame:
    generated_rows = [_sample_feature_vector(record) for record in records]
    real_rows = [_gt_feature_vector(record) for record in records]
    if not generated_rows or not real_rows:
        return pd.DataFrame(
            [
                {
                    "feature": "none",
                    "generated_mean": 0.0,
                    "generated_std": 0.0,
                    "real_mean": 0.0,
                    "real_std": 0.0,
                    "w1": 0.0,
                    "standardized_mean_gap": 0.0,
                    "linear_coef": 0.0,
                }
            ]
        )
    feature_names = list(generated_rows[0].keys())
    fake = _stack_joint_features(generated_rows)
    real = _stack_joint_features(real_rows)
    real, fake = _finite_joint_arrays(real, fake)
    if len(real) == 0 or len(fake) == 0:
        return pd.DataFrame(
            [
                {
                    "feature": "none",
                    "generated_mean": 0.0,
                    "generated_std": 0.0,
                    "real_mean": 0.0,
                    "real_std": 0.0,
                    "w1": 0.0,
                    "standardized_mean_gap": 0.0,
                    "linear_coef": 0.0,
                }
            ]
        )
    x = np.concatenate([real, fake], axis=0)
    y = np.concatenate([np.ones(len(real)), np.zeros(len(fake))], axis=0)
    clf = LogisticRegression(max_iter=500)
    clf.fit(x, y)
    rows: List[Dict[str, float | str]] = []
    for feature_idx, feature_name in enumerate(feature_names):
        real_values = real[:, feature_idx]
        fake_values = fake[:, feature_idx]
        pooled = float(np.sqrt(max(real_values.var() + fake_values.var(), 1e-8) / 2.0))
        rows.append(
            {
                "feature": feature_name,
                "generated_mean": float(fake_values.mean()),
                "generated_std": float(fake_values.std()),
                "real_mean": float(real_values.mean()),
                "real_std": float(real_values.std()),
                "w1": float(wasserstein_distance(real_values, fake_values)),
                "standardized_mean_gap": float((fake_values.mean() - real_values.mean()) / max(pooled, 1e-6)),
                "linear_coef": float(clf.coef_[0, feature_idx]),
            }
        )
    return pd.DataFrame(rows)


def compute_realism_metrics(records: List[Dict], enable_c2st: bool = True, enable_prdc: bool = True) -> tuple[pd.DataFrame, List[Dict[str, float]]]:
    feature_rows = [_sample_feature_vector(record) for record in tqdm(records, desc="Realism features (generated)", unit="sample", leave=False)]
    gt_rows = [_gt_feature_vector(record) for record in tqdm(records, desc="Realism features (ground truth)", unit="sample", leave=False)]
    metrics = {}
    for key in feature_rows[0].keys() if feature_rows else []:
        metrics[f"w1_{key}"] = wasserstein_distance([row[key] for row in gt_rows], [row[key] for row in feature_rows])
    joint_real = _stack_joint_features(gt_rows)
    joint_fake = _stack_joint_features(feature_rows)
    joint_real, joint_fake = _finite_joint_arrays(joint_real, joint_fake)
    metrics["sliced_wasserstein"] = _sliced_wasserstein(joint_real, joint_fake)
    metrics["mmd"] = _mmd(joint_real, joint_fake)
    metrics["c2st_auc"] = _c2st_auc(joint_real, joint_fake) if enable_c2st else 0.5
    if enable_prdc:
        metrics.update(_prdc(joint_real, joint_fake))
    alias_map = {
        "w1_speed": "speed_w1",
        "w1_acceleration": "accel_w1",
        "w1_jerk": "jerk_w1",
        "w1_yaw_rate": "yaw_rate_w1",
        "w1_final_displacement": "final_displacement_w1",
        "w1_min_pairwise_distance": "min_distance_w1",
        "w1_ttc_proxy": "ttc_proxy_w1",
        "w1_conflict_count": "conflict_count_w1",
        "w1_behavior_aggressiveness": "behavior_w1",
        "w1_stress_selected_agents": "selected_stress_w1",
    }
    for source_key, alias_key in alias_map.items():
        if source_key in metrics:
            metrics[alias_key] = metrics[source_key]
    metrics["C2ST_AUC"] = metrics["c2st_auc"]
    metrics["collision_rate"] = float(
        np.mean([float(record.get("collision_rate", record.get("collision", 0.0))) for record in records])
    ) if records else 0.0
    metrics["offroad_rate"] = float(
        np.mean([float(record.get("offroad_rate", record.get("offroad", 0.0))) for record in records])
    ) if records else 0.0
    metrics["validity_rate"] = float(
        np.mean([1.0 - float(record.get("invalid_rate", 0.0)) for record in records])
    ) if records else 0.0
    metrics["realism_valid_row_count"] = int(min(len(joint_real), len(joint_fake)))
    metrics["realism_total_record_count"] = int(len(records))
    if "precision" in metrics:
        metrics["prdc_precision"] = metrics["precision"]
        metrics["prdc_recall"] = metrics["recall"]
        metrics["prdc_density"] = metrics["density"]
        metrics["prdc_coverage"] = metrics["coverage"]
    frame = pd.DataFrame([metrics])
    return frame, feature_rows
