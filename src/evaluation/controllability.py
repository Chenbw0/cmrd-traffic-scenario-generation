from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import torch

from isgen.semantics.anchors import extract_anchor_targets


def _safe_corr(values_a: np.ndarray, values_b: np.ndarray, kind: str) -> float:
    if values_a.size <= 1 or values_b.size <= 1:
        return 0.0
    if float(values_a.std()) < 1e-8 or float(values_b.std()) < 1e-8:
        return 0.0
    if kind == "pearson":
        return float(pearsonr(values_a, values_b).statistic)
    return float(spearmanr(values_a, values_b).statistic)


def _safe_scalar(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def compute_retrieval_controllability_metrics(records: List[Dict]) -> pd.DataFrame:
    target_difficulty = np.asarray(
        [float(record.get("target_difficulty_requested", record.get("target_difficulty", 0.0))) for record in records],
        dtype=np.float32,
    )
    retrieved_full = np.asarray([float(record.get("retrieved_full_scene_difficulty", 0.0)) for record in records], dtype=np.float32)
    retrieved_selected = np.asarray(
        [float(record.get("retrieved_selected_agents_difficulty", record.get("slice_difficulty_selected_agents", 0.0))) for record in records],
        dtype=np.float32,
    )
    retrieved_behavior = np.asarray(
        [float(record.get("retrieved_behavior_quantile", record.get("slice_behavior_quantile_selected_agents", 0.0))) for record in records],
        dtype=np.float32,
    )
    support_full = np.asarray([float(record.get("support_full_scene", 0.0)) for record in records], dtype=np.float32)
    support_selected = np.asarray([float(record.get("support_selected_agents", 0.0)) for record in records], dtype=np.float32)
    coverage = np.asarray(
        [float(record.get("coverage", record.get("selected_agents_coverage_score", 1.0))) for record in records],
        dtype=np.float32,
    )
    metrics = {
        "count": int(len(records)),
        "target_difficulty_mean": float(target_difficulty.mean()) if len(records) else 0.0,
        "retrieved_full_scene_difficulty_mean": float(retrieved_full.mean()) if len(records) else 0.0,
        "retrieved_selected_agents_difficulty_mean": float(retrieved_selected.mean()) if len(records) else 0.0,
        "retrieved_behavior_mean": float(retrieved_behavior.mean()) if len(records) else 0.0,
        "retrieval_full_scene_mae": float(np.mean(np.abs(retrieved_full - target_difficulty))) if len(records) else 0.0,
        "retrieval_selected_agents_mae": float(np.mean(np.abs(retrieved_selected - target_difficulty))) if len(records) else 0.0,
        "retrieval_behavior_mae": float(np.mean(np.abs(retrieved_behavior - target_difficulty))) if len(records) else 0.0,
        "retrieval_full_scene_pearson": _safe_corr(target_difficulty, retrieved_full, "pearson"),
        "retrieval_full_scene_spearman": _safe_corr(target_difficulty, retrieved_full, "spearman"),
        "retrieval_selected_agents_pearson": _safe_corr(target_difficulty, retrieved_selected, "pearson"),
        "retrieval_selected_agents_spearman": _safe_corr(target_difficulty, retrieved_selected, "spearman"),
        "retrieval_behavior_pearson": _safe_corr(target_difficulty, retrieved_behavior, "pearson"),
        "retrieval_behavior_spearman": _safe_corr(target_difficulty, retrieved_behavior, "spearman"),
        "support_full_scene_mean": float(support_full.mean()) if len(records) else 0.0,
        "support_selected_agents_mean": float(support_selected.mean()) if len(records) else 0.0,
        "coverage_mean": float(coverage.mean()) if len(records) else 0.0,
        "coverage_p05": float(np.quantile(coverage, 0.05)) if len(records) else 0.0,
        "coverage_p50": float(np.quantile(coverage, 0.50)) if len(records) else 0.0,
        "coverage_p95": float(np.quantile(coverage, 0.95)) if len(records) else 0.0,
        "unique_slice_count": int(len({record.get("retrieved_slice_id", record.get("slice_id")) for record in records})),
        "unique_location_count": int(len({record.get("retrieved_location_id", record.get("location_id")) for record in records})),
        "unique_map_count": int(
            len({record.get("map_id", record.get("retrieved_location_id", record.get("location_id"))) for record in records})
        ),
    }
    metrics["target_vs_retrieved_full_scene_difficulty_mae"] = metrics["retrieval_full_scene_mae"]
    metrics["target_vs_retrieved_selected_agents_difficulty_mae"] = metrics["retrieval_selected_agents_mae"]
    metrics["target_vs_retrieved_behavior_mae"] = metrics["retrieval_behavior_mae"]
    metrics["target_vs_retrieved_full_scene_difficulty_pearson"] = metrics["retrieval_full_scene_pearson"]
    metrics["target_vs_retrieved_full_scene_difficulty_spearman"] = metrics["retrieval_full_scene_spearman"]
    metrics["target_vs_retrieved_selected_agents_difficulty_pearson"] = metrics["retrieval_selected_agents_pearson"]
    metrics["target_vs_retrieved_selected_agents_difficulty_spearman"] = metrics["retrieval_selected_agents_spearman"]
    metrics["target_vs_retrieved_behavior_pearson"] = metrics["retrieval_behavior_pearson"]
    metrics["target_vs_retrieved_behavior_spearman"] = metrics["retrieval_behavior_spearman"]
    return pd.DataFrame([metrics])


def compute_generation_preservation_metrics(records: List[Dict]) -> pd.DataFrame:
    generated_behavior_quantile = np.asarray([float(record.get("generated_behavior_quantile", 0.0)) for record in records], dtype=np.float32)
    retrieved_behavior_quantile = np.asarray(
        [float(record.get("retrieved_behavior_quantile", 0.0)) for record in records],
        dtype=np.float32,
    )
    generated_behavior_aggressiveness = np.asarray([float(record.get("generated_behavior_aggressiveness", 0.0)) for record in records], dtype=np.float32)
    retrieved_behavior_aggressiveness = np.asarray(
        [float(record.get("retrieved_behavior_aggressiveness", record.get("slice_behavior_aggressiveness_selected_agents", 0.0))) for record in records],
        dtype=np.float32,
    )
    generated_behavior_raw = np.asarray([float(record.get("generated_behavior_raw_score", 0.0)) for record in records], dtype=np.float32)
    retrieved_behavior_raw = np.asarray([float(record.get("retrieved_behavior_raw_score", 0.0)) for record in records], dtype=np.float32)
    generated_selected_stress = np.asarray(
        [float(record.get("generated_selected_stress_difficulty", record.get("generated_stress_difficulty_selected_agents", 0.0))) for record in records],
        dtype=np.float32,
    )
    retrieved_selected_stress = np.asarray(
        [float(record.get("retrieved_selected_agents_difficulty", record.get("slice_difficulty_selected_agents", 0.0))) for record in records],
        dtype=np.float32,
    )
    generated_full_proxy = np.asarray(
        [float(record.get("generated_full_scene_proxy", record.get("generated_full_scene_stress_proxy", 0.0))) for record in records],
        dtype=np.float32,
    )
    retrieved_full = np.asarray([float(record.get("retrieved_full_scene_difficulty", 0.0)) for record in records], dtype=np.float32)
    ade_values = []
    fde_values = []
    for record in records:
        generated_future = np.asarray(record["generated_future"], dtype=np.float32)
        gt_future = np.asarray(record["gt_future"], dtype=np.float32)
        mask = np.asarray(record["future_mask"], dtype=bool)
        if mask.any():
            ade_values.append(float(np.mean(np.linalg.norm(generated_future[..., 0:2][mask] - gt_future[..., 0:2][mask], axis=-1))))
            final_mask = mask[..., -1]
            if final_mask.any():
                ade_axes = np.linalg.norm(generated_future[:, -1, 0:2][final_mask] - gt_future[:, -1, 0:2][final_mask], axis=-1)
                fde_values.append(float(np.mean(ade_axes)))
    metrics = {
        "generated_vs_retrieved_behavior_mae": float(np.mean(np.abs(generated_behavior_quantile - retrieved_behavior_quantile))) if len(records) else 0.0,
        "generated_vs_retrieved_behavior_bias": float(np.mean(generated_behavior_quantile - retrieved_behavior_quantile)) if len(records) else 0.0,
        "generated_vs_retrieved_behavior_pearson": _safe_corr(generated_behavior_quantile, retrieved_behavior_quantile, "pearson"),
        "generated_vs_retrieved_behavior_spearman": _safe_corr(generated_behavior_quantile, retrieved_behavior_quantile, "spearman"),
        "generated_behavior_quantile_vs_retrieved_behavior_quantile_mae": float(np.mean(np.abs(generated_behavior_quantile - retrieved_behavior_quantile))) if len(records) else 0.0,
        "generated_behavior_quantile_vs_retrieved_behavior_quantile_corr": _safe_corr(generated_behavior_quantile, retrieved_behavior_quantile, "pearson"),
        "generated_behavior_aggressiveness_vs_retrieved_behavior_aggressiveness_mae": float(np.mean(np.abs(generated_behavior_aggressiveness - retrieved_behavior_aggressiveness))) if len(records) else 0.0,
        "generated_behavior_raw_score_vs_retrieved_behavior_raw_score_mae": float(np.mean(np.abs(generated_behavior_raw - retrieved_behavior_raw))) if len(records) else 0.0,
        "generated_vs_retrieved_selected_stress_mae": float(np.mean(np.abs(generated_selected_stress - retrieved_selected_stress))) if len(records) else 0.0,
        "generated_vs_retrieved_selected_stress_bias": float(np.mean(generated_selected_stress - retrieved_selected_stress)) if len(records) else 0.0,
        "generated_full_scene_proxy_vs_retrieved_full_scene_mae": float(np.mean(np.abs(generated_full_proxy - retrieved_full))) if len(records) else 0.0,
        "generated_full_scene_proxy_vs_retrieved_full_scene_bias": float(np.mean(generated_full_proxy - retrieved_full)) if len(records) else 0.0,
        "generated_vs_gt_ade": float(np.mean(ade_values)) if ade_values else 0.0,
        "generated_vs_gt_fde": float(np.mean(fde_values)) if fde_values else 0.0,
        "generator_condition_behavior_mean": float(
            np.mean([_safe_scalar(record.get("generator_condition_behavior", record.get("target_behavior", 0.0))) for record in records])
        )
        if len(records)
        else 0.0,
    }
    metrics["generated_vs_gt_ADE"] = metrics["generated_vs_gt_ade"]
    metrics["generated_vs_gt_FDE"] = metrics["generated_vs_gt_fde"]
    metrics["generated_behavior_correlation_with_retrieved_behavior"] = metrics["generated_vs_retrieved_behavior_pearson"]
    metrics["generated_behavior_vs_retrieved_behavior_mae"] = metrics["generated_vs_retrieved_behavior_mae"]
    metrics["generated_full_scene_proxy_vs_retrieved_full_scene_difficulty_mae"] = metrics[
        "generated_full_scene_proxy_vs_retrieved_full_scene_mae"
    ]
    metrics["generated_selected_stress_vs_retrieved_selected_difficulty_mae"] = metrics[
        "generated_vs_retrieved_selected_stress_mae"
    ]
    return pd.DataFrame([metrics])


def compute_same_slice_diagnostic_metrics(records: List[Dict]) -> pd.DataFrame:
    diagnostic_records = [record for record in records if bool(record.get("is_same_scene_diagnostic", False))]
    if not diagnostic_records:
        return pd.DataFrame(
            [
                {
                    "count": 0,
                    "note": "Same-scene target sweep is diagnostic only and was not run for this sample set.",
                }
            ]
        )
    target_behavior = np.asarray([float(record.get("target_behavior", 0.0)) for record in diagnostic_records], dtype=np.float32)
    generated_behavior = np.asarray(
        [float(record.get("generated_behavior_quantile", 0.0)) for record in diagnostic_records],
        dtype=np.float32,
    )
    low_support = np.asarray(
        [
            bool(record.get("low_support", False) or record.get("generator_behavior_low_support", False))
            for record in diagnostic_records
        ],
        dtype=bool,
    )
    metrics = {
        "count": int(len(diagnostic_records)),
        "same_slice_behavior_mae": float(np.mean(np.abs(generated_behavior - target_behavior))) if len(diagnostic_records) else 0.0,
        "same_slice_behavior_spearman": _safe_corr(target_behavior, generated_behavior, "spearman"),
        "low_support_fraction": float(np.mean(low_support)) if len(diagnostic_records) else 0.0,
        "note": "Diagnostic only. Same-scene target sweep is not used for the primary project conclusions.",
    }
    return pd.DataFrame([metrics])


def compute_anchor_metrics(records: List[Dict]) -> pd.DataFrame:
    mid_errors: list[float] = []
    final_errors: list[float] = []
    heading_errors: list[float] = []
    for record in records:
        current_states = torch.as_tensor(record.get("current_states")).float().unsqueeze(0)
        future_mask = torch.as_tensor(record.get("future_mask")).bool().unsqueeze(0)
        agent_mask = torch.as_tensor(record.get("agent_mask")).bool().unsqueeze(0)
        gt_future = torch.as_tensor(record.get("gt_future")).float().unsqueeze(0)
        generated_future = torch.as_tensor(record.get("generated_future")).float().unsqueeze(0)
        gt_anchor, anchor_valid_mask = extract_anchor_targets(
            current_states=current_states,
            future_states=gt_future,
            future_mask=future_mask,
            agent_mask=agent_mask,
        )
        generated_anchor, _ = extract_anchor_targets(
            current_states=current_states,
            future_states=generated_future,
            future_mask=future_mask,
            agent_mask=agent_mask,
        )
        valid = anchor_valid_mask[0]
        if not bool(valid.any()):
            continue
        diff = torch.abs(generated_anchor[0] - gt_anchor[0])
        mid_errors.extend(diff[valid, 0:2].reshape(-1).tolist())
        final_errors.extend(diff[valid, 2:5].reshape(-1).tolist())
        heading_delta = torch.atan2(
            torch.sin(generated_anchor[0, :, 5:6] - gt_anchor[0, :, 5:6]),
            torch.cos(generated_anchor[0, :, 5:6] - gt_anchor[0, :, 5:6]),
        )
        heading_errors.extend(torch.abs(heading_delta[valid]).reshape(-1).tolist())
    return pd.DataFrame(
        [
            {
                "anchor_mid_mae": float(np.mean(mid_errors)) if mid_errors else 0.0,
                "anchor_final_mae": float(np.mean(final_errors)) if final_errors else 0.0,
                "anchor_heading_mae": float(np.mean(heading_errors)) if heading_errors else 0.0,
            }
        ]
    )


def compute_controllability_metrics(records: List[Dict], support_bins: List[float] | None = None) -> pd.DataFrame:
    _ = support_bins
    return compute_retrieval_controllability_metrics(records)
