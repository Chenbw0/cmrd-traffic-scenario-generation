from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from isgen import ensure_dir


def _table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    return df.to_string(index=False)


def _json_block(payload: Dict | List | None) -> List[str]:
    return ["```json", json.dumps(payload or {}, indent=2), "```", ""]


def _select_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    keep = [column for column in columns if column in df.columns]
    if not keep:
        return df
    return df[keep]


def write_analysis_report(
    output_dir: str | Path,
    data_stats: Dict,
    retrieval_df: pd.DataFrame,
    realism_df: pd.DataFrame,
    preservation_df: pd.DataFrame,
    diversity_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    diagnostic_df: pd.DataFrame,
    per_sample_metrics: List[Dict],
    warnings: List[str],
    sample_metadata: Dict | None = None,
    calibration_recommendation: Dict | None = None,
    conditioning_summary: Dict | None = None,
    run_metadata: Dict | None = None,
) -> None:
    root = ensure_dir(output_dir)
    retrieval_df.to_csv(root / "retrieval_metrics.csv", index=False)
    realism_df.to_csv(root / "realism_metrics.csv", index=False)
    preservation_df.to_csv(root / "preservation_metrics.csv", index=False)
    diversity_df.to_csv(root / "diversity_metrics.csv", index=False)
    diversity_df.to_csv(root / "novelty_metrics.csv", index=False)
    diagnostic_df.to_csv(root / "diagnostic_same_slice_metrics.csv", index=False)
    baseline_df.to_csv(root / "baseline_comparison.csv", index=False)
    with (root / "per_sample_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_sample_metrics:
            handle.write(json.dumps(row) + "\n")
    if calibration_recommendation is not None:
        (root / "behavior_calibration_recommendation.json").write_text(
            json.dumps(calibration_recommendation, indent=2),
            encoding="utf-8",
        )

    split_metadata = data_stats.get("split_metadata", {})
    diagnostics = data_stats.get("preprocessing_diagnostics", {})
    behavior_target_key = data_stats.get("behavior_target_key", "behavior_quantile_score_selected_agents")
    behavior_distribution_report = data_stats.get("behavior_distribution_report", {})
    behavior_train = behavior_distribution_report.get("splits", {}).get("train", {}).get(behavior_target_key, {})
    difficulty_train = diagnostics.get("difficulty_score_full_scene", {}).get("train", {})
    cache_signature = data_stats.get("cache_signature", {})
    sample_runtime = (sample_metadata or {}).get("sample_runtime", {})
    runtime = dict(run_metadata or {})
    runtime.setdefault("decoder_type", sample_runtime.get("actual_model_decoder_type"))
    runtime.setdefault("encoder_type", sample_runtime.get("encoder_type"))
    runtime.setdefault("control_representation", sample_runtime.get("control_representation"))
    runtime.setdefault("num_control_knots", sample_runtime.get("num_control_knots"))
    runtime.setdefault("control_interpolation", sample_runtime.get("control_interpolation"))
    runtime.setdefault("use_anchor_latent", sample_runtime.get("use_anchor_latent"))
    runtime.setdefault("anchor_dim", sample_runtime.get("anchor_dim"))
    runtime.setdefault("residual_diffusion_enabled", sample_runtime.get("residual_diffusion_enabled"))
    runtime.setdefault("anchor_residual_diffusion_enabled", sample_runtime.get("anchor_residual_diffusion_enabled"))
    runtime.setdefault("diffusion_target_type", sample_runtime.get("diffusion_target_type"))
    runtime.setdefault("anchor_diffusion_target_type", sample_runtime.get("anchor_diffusion_target_type"))
    runtime.setdefault("use_residual_diffusion", sample_runtime.get("use_residual_diffusion"))
    runtime.setdefault("use_anchor_residual_diffusion", sample_runtime.get("use_anchor_residual_diffusion"))
    runtime.setdefault("source_sha256", sample_runtime.get("source_sha256"))
    runtime.setdefault("runtime_source_file", sample_runtime.get("runtime_source_file"))
    runtime.setdefault("generator_condition_mode", sample_runtime.get("generator_condition_mode"))
    runtime.setdefault("generator_consumes_behavior", sample_runtime.get("generator_consumes_behavior"))
    runtime.setdefault("generator_consumes_difficulty", sample_runtime.get("generator_consumes_difficulty"))
    if sample_runtime.get("checkpoint_exists") is not None:
        runtime["checkpoint_exists"] = sample_runtime.get("checkpoint_exists")
    if sample_runtime.get("checkpoint_sha256") is not None:
        runtime["checkpoint_sha256"] = sample_runtime.get("checkpoint_sha256")
    if sample_runtime.get("checkpoint_epoch") is not None:
        runtime["checkpoint_epoch"] = sample_runtime.get("checkpoint_epoch")
    if sample_runtime.get("checkpoint_used") is not None:
        runtime["checkpoint_used"] = sample_runtime.get("checkpoint_used")
    if sample_runtime.get("checkpoint_path") is not None:
        runtime["checkpoint_path"] = sample_runtime.get("checkpoint_path")

    retrieval_report_df = _select_columns(
        retrieval_df,
        [
            "count",
            "retrieval_full_scene_mae",
            "retrieval_selected_agents_mae",
            "target_vs_retrieved_full_scene_difficulty_mae",
            "target_vs_retrieved_selected_agents_difficulty_mae",
            "coverage_mean",
            "unique_slice_count",
            "unique_location_count",
        ],
    )
    preservation_report_df = _select_columns(
        preservation_df,
        [
            "generated_vs_retrieved_behavior_mae",
            "generated_vs_retrieved_behavior_bias",
            "generated_vs_retrieved_behavior_pearson",
            "generated_vs_gt_ade",
            "generated_vs_gt_fde",
            "generated_full_scene_proxy_vs_retrieved_full_scene_mae",
            "generated_vs_retrieved_selected_stress_mae",
        ],
    )
    realism_report_df = _select_columns(
        realism_df,
        [
            "sliced_wasserstein",
            "c2st_auc",
            "speed_w1",
            "accel_w1",
            "jerk_w1",
            "yaw_rate_w1",
            "min_distance_w1",
            "conflict_count_w1",
            "collision_rate",
            "offroad_rate",
        ],
    )
    diversity_report_df = _select_columns(
        diversity_df,
        [
            "novelty_score",
            "diversity_score",
            "rollout_diversity_ADE",
            "rollout_diversity_FDE",
            "nn_train_distance",
            "gt_copy_ade",
        ],
    )
    baseline_report_df = _select_columns(
        baseline_df,
        [
            "mode",
            "ADE",
            "FDE",
            "SWD",
            "C2ST_AUC",
            "generated_behavior_preservation_mae_quantile",
            "generated_behavior_bias_quantile",
            "generated_behavior_corr_with_retrieved",
            "novelty_score",
            "diversity_score",
        ],
    )

    report_lines = [
        "# Analysis Report",
        "",
        "## Runtime Audit Summary",
        "",
        f"- decoder_type: `{runtime.get('decoder_type', 'unknown')}`",
        f"- forward_path: `{runtime.get('forward_path', 'unknown')}`",
        f"- sample_decode_path: `{runtime.get('sample_decode_path', 'unknown')}`",
        f"- active_training_losses: `{runtime.get('active_training_losses', ['unknown'])}`",
        f"- main_training_objective: `{runtime.get('main_training_objective', 'unknown')}`",
        f"- source sha256: `{runtime.get('source_sha256', runtime.get('code_fingerprint', 'unknown'))}`",
        f"- generator_condition_mode: `{runtime.get('generator_condition_mode', 'unknown')}`",
        f"- generator_consumes_behavior: `{runtime.get('generator_consumes_behavior', 'unknown')}`",
        f"- generator_consumes_difficulty: `{runtime.get('generator_consumes_difficulty', 'unknown')}`",
        f"- behavior_target_key: `{runtime.get('behavior_target_key', behavior_target_key)}`",
        f"- generator_condition_score_key: `{runtime.get('generator_condition_score_key', 'unknown')}`",
        f"- generated_behavior_score_space: `{runtime.get('generated_behavior_score_space', 'unknown')}`",
        f"- encoder_type: `{runtime.get('encoder_type', 'unknown')}`",
        f"- generator_used_history: `{runtime.get('generator_input_use_history', sample_runtime.get('generator_used_history', 'unknown'))}`",
        f"- require_history_at_sample: `{runtime.get('require_history_at_sample', sample_runtime.get('require_history_at_sample', 'unknown'))}`",
        f"- sampler: `{runtime.get('sampler', sample_runtime.get('sampler', 'unknown'))}`",
        f"- control_representation: `{runtime.get('control_representation', 'unknown')}`",
        f"- num_control_knots: `{runtime.get('num_control_knots', 'unknown')}`",
        f"- control_interpolation: `{runtime.get('control_interpolation', 'unknown')}`",
        f"- use_anchor_latent: `{runtime.get('use_anchor_latent', 'unknown')}`",
        f"- anchor_dim: `{runtime.get('anchor_dim', 'unknown')}`",
        f"- learned_interaction_field_enabled: `{runtime.get('learned_interaction_field_enabled', 'unknown')}`",
        f"- learned_interaction_field_target: `{runtime.get('learned_interaction_field_target', 'unknown')}`",
        f"- learned_interaction_field_feature_dim: `{runtime.get('learned_interaction_field_feature_dim', 'unknown')}`",
        f"- learned_interaction_field_latent_dim: `{runtime.get('learned_interaction_field_latent_dim', 'unknown')}`",
        f"- learned_interaction_num_message_passing_steps: `{runtime.get('learned_interaction_num_message_passing_steps', 'unknown')}`",
        f"- learned_interaction_field_condition_residual: `{runtime.get('learned_interaction_field_condition_residual', 'unknown')}`",
        f"- residual_diffusion_enabled: `{runtime.get('residual_diffusion_enabled', 'unknown')}`",
        f"- anchor_residual_diffusion_enabled: `{runtime.get('anchor_residual_diffusion_enabled', 'unknown')}`",
        f"- use_residual_diffusion: `{runtime.get('use_residual_diffusion', sample_runtime.get('use_residual_diffusion', 'unknown'))}`",
        f"- use_anchor_residual_diffusion: `{runtime.get('use_anchor_residual_diffusion', sample_runtime.get('use_anchor_residual_diffusion', 'unknown'))}`",
        f"- normalize_residual: `{runtime.get('normalize_residual', 'unknown')}`",
        f"- residual_scale: `{runtime.get('residual_sample_scale', sample_runtime.get('residual_scale', 'unknown'))}`",
        f"- anchor_residual_scale: `{runtime.get('anchor_residual_sample_scale', sample_runtime.get('anchor_residual_scale', 'unknown'))}`",
        f"- residual_clip_std: `{runtime.get('residual_clip_std', 'unknown')}`",
        f"- anchor_residual_clip_std: `{runtime.get('anchor_residual_clip_std', 'unknown')}`",
        f"- residual_sample_steps: `{runtime.get('residual_sample_steps', 'unknown')}`",
        f"- anchor_residual_sample_steps: `{runtime.get('anchor_residual_sample_steps', 'unknown')}`",
        f"- optimizer_lr: `{runtime.get('lr', 'unknown')}`",
        f"- scheduler_type: `{runtime.get('scheduler_type', 'none')}`",
        f"- scheduler_min_lr_ratio: `{runtime.get('scheduler_min_lr_ratio', 'unknown')}`",
        f"- current_lr: `{runtime.get('current_lr', 'unknown')}`",
        f"- supervised_control_weight: `{runtime.get('supervised_control_weight', 'unknown')}`",
        f"- supervised_rollout_weight: `{runtime.get('supervised_rollout_weight', 'unknown')}`",
        f"- residual_diffusion_weight: `{runtime.get('residual_diffusion_weight', 'unknown')}`",
        f"- interaction_field_weight: `{runtime.get('interaction_field_weight', 'unknown')}`",
        f"- interaction_field_feature_weights: `{runtime.get('interaction_field_feature_weights', 'unknown')}`",
        f"- anchor_weight: `{runtime.get('anchor_weight', 'unknown')}`",
        f"- anchor_residual_diffusion_weight: `{runtime.get('anchor_residual_diffusion_weight', 'unknown')}`",
        f"- diffusion_target_type: `{runtime.get('diffusion_target_type', 'unknown')}`",
        f"- anchor_diffusion_target_type: `{runtime.get('anchor_diffusion_target_type', 'unknown')}`",
        f"- timestep_sampling: `{runtime.get('timestep_sampling', 'unknown')}`",
        f"- min_snr_gamma: `{runtime.get('min_snr_gamma', 'unknown')}`",
        f"- behavior_control_weight: `{runtime.get('behavior_control_weight', 'unknown')}`",
        f"- behavior_control_loss_in_total: `{runtime.get('behavior_control_loss_in_total', 'unknown')}`",
        f"- control_delta_weight: `{runtime.get('control_delta_weight', 'unknown')}`",
        f"- realism_mmd_weight: `{runtime.get('realism_mmd_weight', 'unknown')}`",
        f"- realism_disabled_reason: `{runtime.get('realism_disabled_reason', 'none')}`",
        f"- sample_path_loss_enabled: `{runtime.get('sample_path_loss_enabled', 'unknown')}`",
        f"- sample_path_loss_weight: `{runtime.get('sample_path_loss_weight', 'unknown')}`",
        f"- sample_path_steps: `{runtime.get('sample_path_steps', 'unknown')}`",
        f"- sample_path_every_n_batches: `{runtime.get('sample_path_every_n_batches', 'unknown')}`",
        f"- sample_path_batch_fraction: `{runtime.get('sample_path_batch_fraction', 'unknown')}`",
        f"- checkpoint_exists: `{runtime.get('checkpoint_exists', 'unknown')}`",
        f"- checkpoint_used: `{runtime.get('checkpoint_used', 'unknown')}`",
        f"- checkpoint_epoch: `{runtime.get('checkpoint_epoch', 'unknown')}`",
        f"- checkpoint_sha256: `{runtime.get('checkpoint_sha256', 'unknown')}`",
        "",
        "We do not assume paired counterfactual data.",
        "We do not require a fixed scene to realize arbitrary difficulty levels.",
        "We do not require user-provided history at generation time.",
        "We retrieve current multi-agent scene snapshots from a naturalistic scenario library.",
        "Difficulty controls retrieval.",
        "The generator conditions on the retrieved current scene snapshot and map.",
        "The generator is unconditional with respect to behavior and difficulty unless a conditioning ablation is explicitly enabled.",
        "Behavior preservation is an evaluation metric, not a training objective.",
        "Same-scene arbitrary behavior control is not claimed.",
        "The generator rolls out future trajectories from current states.",
        "History, if stored, is used only for analysis or visualization unless explicitly enabled.",
        "Difficulty control is achieved through support-aware retrieval.",
        "The generator produces human-like futures from retrieved supported slices.",
        "Same-scene target sweep is diagnostic only.",
        "",
        "## Dataset and Cache Summary",
        "",
        f"- split strategy: `{data_stats.get('split_strategy', split_metadata.get('split_by', 'grouped_split'))}`",
        f"- train/val/test slices: `{data_stats.get('train_slice_count', 0)}` / `{data_stats.get('val_slice_count', 0)}` / `{data_stats.get('test_slice_count', 0)}`",
        f"- location count: `{len(split_metadata.get('train_groups', [])) if isinstance(split_metadata.get('train_groups'), list) else split_metadata.get('train_group_count', 'n/a')}`",
        f"- data_mode: `{data_stats.get('data_mode', 'unknown')}`",
        f"- debug_profile: `{data_stats.get('debug_profile', 'unknown')}`",
        f"- debug_max_cases: `{data_stats.get('debug_max_cases', 'null')}`",
        f"- synthetic: `{data_stats.get('synthetic', False)}`",
        f"- decoder_type: `{runtime.get('decoder_type', 'unknown')}`",
        f"- source sha256: `{runtime.get('source_sha256', runtime.get('code_fingerprint', 'unknown'))}`",
        "",
        "### Warning List",
        "",
    ]
    if warnings:
        report_lines.extend([f"- {warning}" for warning in warnings])
    else:
        report_lines.append("- None")
    report_lines.extend(
        [
            "",
            "## Retrieval Controllability Summary",
            "",
            _table(retrieval_report_df),
            "",
            "## Generation Preservation Summary",
            "",
            _table(preservation_report_df),
            "",
            "## Generation Realism Summary",
            "",
            _table(realism_report_df),
            "",
            "## Novelty and Diversity Summary",
            "",
            _table(diversity_report_df),
            "",
            "## Baseline Comparison",
            "",
            _table(baseline_report_df),
            "",
            "## Notes",
            "",
            "- Detailed retrieval, preservation, realism, novelty, and baseline files are saved alongside this report as CSV/JSON.",
            "- Calibration output, runtime metadata, and per-sample diagnostics are intentionally omitted from the report body to keep it concise.",
            "- Same-scene target sweep remains diagnostic only and was not used as primary evidence.",
        ]
    )

    (root / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
