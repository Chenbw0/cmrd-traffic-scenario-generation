from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from tqdm.auto import tqdm

from isgen import resolve_output_path
from isgen.data.cache import load_cache_metadata
from isgen.evaluation.controllability import (
    compute_anchor_metrics,
    compute_generation_preservation_metrics,
    compute_retrieval_controllability_metrics,
    compute_same_slice_diagnostic_metrics,
)
from isgen.evaluation.diversity import compute_diversity_metrics
from isgen.evaluation.interaction import compute_interaction_diagnostics
from isgen.evaluation.realism import compute_c2st_feature_diagnostics, compute_realism_metrics
from isgen.evaluation.report import write_analysis_report
from isgen.sampling.calibration import calibration_error_summary, fit_behavior_calibrator
from isgen.training.runtime_audit import capture_runtime_sources


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _baseline_comparison(records: List[Dict], project_root: Path, cache_dir: str | Path, config: Dict) -> pd.DataFrame:
    rows: List[Dict[str, float | str | int]] = []
    grouped: Dict[str, List[Dict]] = {}
    for record in records:
        mode = str(record.get("mode", record.get("generation_mode", "unknown")))
        grouped.setdefault(mode, []).append(record)
    for mode, mode_records in grouped.items():
        realism_df, _ = compute_realism_metrics(
            mode_records,
            enable_c2st=bool(config["analysis"]["c2st_enabled"]),
            enable_prdc=bool(config["analysis"]["prdc_enabled"]),
        )
        retrieval_df = compute_retrieval_controllability_metrics(mode_records)
        preservation_df = compute_generation_preservation_metrics(mode_records)
        diversity_df, _ = compute_diversity_metrics(mode_records, project_root=project_root, cache_dir=cache_dir)
        rows.append(
            {
                "mode": mode,
                "count": int(len(mode_records)),
                "retrieval_full_mae": float(retrieval_df.iloc[0]["retrieval_full_scene_mae"]),
                "retrieval_selected_mae": float(retrieval_df.iloc[0]["retrieval_selected_agents_mae"]),
                "retrieved_behavior_mae_to_target": float(retrieval_df.iloc[0]["retrieval_behavior_mae"]),
                "generated_behavior_preservation_mae_quantile": float(preservation_df.iloc[0]["generated_vs_retrieved_behavior_mae"]),
                "generated_behavior_bias_quantile": float(preservation_df.iloc[0]["generated_vs_retrieved_behavior_bias"]),
                "generated_behavior_corr_with_retrieved": float(preservation_df.iloc[0]["generated_vs_retrieved_behavior_pearson"]),
                "generated_full_scene_proxy_mae": float(preservation_df.iloc[0]["generated_full_scene_proxy_vs_retrieved_full_scene_mae"]),
                "ADE": float(preservation_df.iloc[0]["generated_vs_gt_ADE"]),
                "FDE": float(preservation_df.iloc[0]["generated_vs_gt_FDE"]),
                "SWD": float(realism_df.iloc[0]["sliced_wasserstein"]),
                "C2ST_AUC": float(realism_df.iloc[0]["C2ST_AUC"]),
                "speed_w1": float(realism_df.iloc[0].get("speed_w1", 0.0)),
                "accel_w1": float(realism_df.iloc[0].get("accel_w1", 0.0)),
                "jerk_w1": float(realism_df.iloc[0].get("jerk_w1", 0.0)),
                "yaw_rate_w1": float(realism_df.iloc[0].get("yaw_rate_w1", 0.0)),
                "min_distance_w1": float(realism_df.iloc[0].get("min_distance_w1", 0.0)),
                "ttc_w1": float(realism_df.iloc[0].get("ttc_proxy_w1", 0.0)),
                "novelty_score": float(diversity_df.iloc[0]["novelty_score"]),
                "nearest_train_distance": float(diversity_df.iloc[0]["nearest_train_distance"]),
                "diversity_score": float(diversity_df.iloc[0]["rollout_diversity_feature"]),
                "collision_rate": float(realism_df.iloc[0].get("collision_rate", 0.0)),
                "offroad_rate": float(realism_df.iloc[0].get("offroad_rate", 0.0)),
                "unique_slice_count": float(retrieval_df.iloc[0].get("unique_slice_count", 0.0)),
                "unique_location_count": float(retrieval_df.iloc[0].get("unique_location_count", 0.0)),
            }
        )
    if not rows:
        rows.append(
            {
                "mode": "none",
                "count": 0,
                "retrieval_full_mae": 0.0,
                "generated_behavior_preservation_mae": 0.0,
                "ADE": 0.0,
                "FDE": 0.0,
                "SWD": 0.0,
                "C2ST_AUC": 0.5,
                "novelty_score": 0.0,
                "diversity_score": 0.0,
                "collision_rate": 0.0,
                "offroad_rate": 0.0,
            }
        )
    return pd.DataFrame(rows)


def _infer_data_mode(metadata: Dict) -> str:
    synthetic = bool(metadata.get("synthetic", False))
    debug_max_cases = metadata.get("debug_max_cases")
    debug_profile = str(metadata.get("debug_profile", "full") or "full")
    if synthetic:
        return "synthetic_smoke"
    if debug_max_cases is None and debug_profile == "full":
        return "real_interaction_main"
    return "real_interaction_debug"


def _filtered_warnings(metadata: Dict, behavior_distribution_report: Dict) -> list[str]:
    data_mode = _infer_data_mode(metadata)
    warnings = list(metadata.get("warnings", []))
    if data_mode == "real_interaction_main":
        warnings = [item for item in warnings if "debug subset not suitable" not in str(item).lower()]
        test_quantile = (
            behavior_distribution_report.get("splits", {})
            .get("test", {})
            .get("behavior_quantile_score_selected_agents", {})
        )
        low_count = int(test_quantile.get("low_count", 0) or 0)
        mid_count = int(test_quantile.get("mid_count", 0) or 0)
        high_count = int(test_quantile.get("high_count", 0) or 0)
        if min(low_count, mid_count, high_count) == 0:
            warning = "Test behavior support is imbalanced across quantile bins. Interpret absolute behavior calibration with caution."
            if warning not in warnings:
                warnings.append(warning)
    return warnings


def _behavior_preservation_by_bin(records: List[Dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(
            [{"bin": "[0.0,0.2)", "count": 0, "retrieved_behavior_mean": 0.0, "generated_behavior_mean": 0.0, "bias": 0.0, "MAE": 0.0, "correlation": 0.0, "location_distribution": "{}", "agent_count_mean": 0.0, "support_mean": 0.0, "coverage_mean": 0.0}]
        )
    rows = []
    bin_edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.000001]
    bin_labels = ["[0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1.0]"]
    for lower, upper, label in zip(bin_edges[:-1], bin_edges[1:], bin_labels):
        bucket = [
            record
            for record in records
            if lower <= float(record.get("retrieved_behavior_quantile", 0.0)) < upper
        ]
        retrieved = pd.Series([float(record.get("retrieved_behavior_quantile", 0.0)) for record in bucket], dtype=float)
        generated = pd.Series([float(record.get("generated_behavior_quantile", 0.0)) for record in bucket], dtype=float)
        locations: Dict[str, int] = {}
        for record in bucket:
            key = str(record.get("retrieved_location_id", record.get("location_id", "")))
            locations[key] = locations.get(key, 0) + 1
        rows.append(
            {
                "bin": label,
                "count": int(len(bucket)),
                "retrieved_behavior_mean": float(retrieved.mean()) if not retrieved.empty else 0.0,
                "generated_behavior_mean": float(generated.mean()) if not generated.empty else 0.0,
                "bias": float((generated - retrieved).mean()) if not retrieved.empty else 0.0,
                "MAE": float((generated - retrieved).abs().mean()) if not retrieved.empty else 0.0,
                "correlation": float(retrieved.corr(generated)) if len(bucket) > 1 and float(retrieved.std()) > 1e-8 and float(generated.std()) > 1e-8 else 0.0,
                "location_distribution": json.dumps(locations, ensure_ascii=True, sort_keys=True),
                "agent_count_mean": float(pd.Series([float(torch.as_tensor(record.get("agent_mask", [])).bool().sum().item()) for record in bucket], dtype=float).mean()) if bucket else 0.0,
                "support_mean": float(pd.Series([float(record.get("support_selected_agents", 0.0)) for record in bucket], dtype=float).mean()) if bucket else 0.0,
                "coverage_mean": float(pd.Series([float(record.get("coverage", 0.0)) for record in bucket], dtype=float).mean()) if bucket else 0.0,
            }
        )
    return pd.DataFrame(rows)


def analyze_samples(config: Dict, project_root: str | Path, samples_path: str | Path) -> Dict[str, float]:
    project_root = Path(project_root)
    records: List[Dict] = torch.load(samples_path, map_location="cpu")
    cache_dir = project_root / config["data"]["cache_dir"]
    metadata = load_cache_metadata(cache_dir)
    checkpoints_dir = resolve_output_path(config, project_root, "checkpoints_dir", Path("outputs") / "checkpoints")
    analysis_dir = resolve_output_path(config, project_root, "analysis_dir", Path("outputs") / "analysis")
    run_metadata_path = checkpoints_dir / "run_metadata.json"
    run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8")) if run_metadata_path.exists() else {}
    analyze_runtime_snapshot = capture_runtime_sources(
        analysis_dir / "runtime_source_snapshot",
        phase="analyze",
        config_hash=run_metadata.get("config_hash"),
        checkpoint_hash=run_metadata.get("checkpoint_file_sha256"),
    )
    expected_runtime_fingerprints = run_metadata.get("runtime_code_fingerprints", {})
    for file_name, file_payload in analyze_runtime_snapshot.get("files", {}).items():
        expected_sha = expected_runtime_fingerprints.get(file_name)
        current_sha = file_payload.get("sha256")
        if expected_sha is not None and current_sha is not None and expected_sha != current_sha:
            raise RuntimeError(
                f"Analyze runtime source mismatch for {file_name}: checkpoint={expected_sha} current={current_sha}"
            )
    sample_metadata_path = Path(samples_path).with_name("metadata.json")
    sample_metadata = {}
    if sample_metadata_path.exists():
        sample_metadata = json.loads(sample_metadata_path.read_text(encoding="utf-8"))
    sample_runtime = dict((sample_metadata or {}).get("sample_runtime", {}))
    sample_mode = str((sample_metadata or {}).get("mode", ""))
    requires_checkpoint = sample_mode not in {"retrieval_replay", "random_retrieval_replay", "dataset_replay_baseline"}
    if requires_checkpoint and not bool(sample_runtime.get("checkpoint_exists", False)):
        raise RuntimeError("Formal generation analysis requires checkpoint provenance, but checkpoint_exists=false in sample metadata.")
    conditioning_trace_path = Path(samples_path).with_name("conditioning_trace.jsonl")
    conditioning_rows = []
    if conditioning_trace_path.exists():
        with conditioning_trace_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    conditioning_rows.append(json.loads(line))
    with tqdm(total=4, desc="Analysis phases", unit="phase") as phase_bar:
        realism_df, _ = compute_realism_metrics(
            records,
            enable_c2st=bool(config["analysis"]["c2st_enabled"]),
            enable_prdc=bool(config["analysis"]["prdc_enabled"]),
        )
        phase_bar.update(1)
        retrieval_df = compute_retrieval_controllability_metrics(records)
        preservation_df = compute_generation_preservation_metrics(records)
        anchor_df = compute_anchor_metrics(records)
        anchor_df.to_csv(analysis_dir / "anchor_metrics.csv", index=False)
        diagnostic_df = compute_same_slice_diagnostic_metrics(records)
        interaction_df = compute_interaction_diagnostics(records)
        interaction_df.to_csv(analysis_dir / "interaction_diagnostics.csv", index=False)
        c2st_feature_df = compute_c2st_feature_diagnostics(records)
        c2st_feature_df.to_csv(analysis_dir / "c2st_feature_diagnostics.csv", index=False)
        behavior_bin_df = _behavior_preservation_by_bin(records)
        behavior_bin_df.to_csv(analysis_dir / "behavior_preservation_by_bin.csv", index=False)
        phase_bar.update(1)
        diversity_df, per_sample_rows = compute_diversity_metrics(records, project_root=project_root, cache_dir=config["data"]["cache_dir"])
        per_sample_rows = [
            {
                **row,
                "target_difficulty_requested": _safe_float(record.get("target_difficulty_requested", record.get("target_difficulty", 0.0))),
                "target_behavior": _safe_float(record.get("target_behavior", record.get("target_difficulty", 0.0))),
                "generator_condition_behavior": _safe_float(record.get("generator_condition_behavior", record.get("conditioning_behavior_input", record.get("target_behavior", 0.0)))),
                "conditioning_behavior_input": _safe_float(record.get("conditioning_behavior_input", record.get("target_behavior", record.get("target_difficulty", 0.0)))),
                "retrieved_slice_id": str(record.get("retrieved_slice_id", record.get("slice_id", ""))),
                "retrieved_location_id": str(record.get("retrieved_location_id", record.get("location_id", ""))),
                "retrieved_full_scene_difficulty": float(record.get("retrieved_full_scene_difficulty", record.get("slice_difficulty_full_scene", 0.0))),
                "retrieved_selected_agents_difficulty": float(record.get("retrieved_selected_agents_difficulty", record.get("slice_difficulty_selected_agents", 0.0))),
                "retrieved_behavior_aggressiveness": float(record.get("retrieved_behavior_aggressiveness", record.get("slice_behavior_aggressiveness_selected_agents", 0.0))),
                "retrieved_behavior_quantile": float(record.get("retrieved_behavior_quantile", 0.0)),
                "retrieved_behavior_raw_score": float(record.get("retrieved_behavior_raw_score", 0.0)),
                "slice_difficulty_selected_agents": float(record.get("slice_difficulty_selected_agents", record.get("slice_difficulty", 0.0))),
                "slice_difficulty_full_scene": float(record.get("slice_difficulty_full_scene", record.get("slice_difficulty", 0.0))),
                "slice_behavior_aggressiveness_selected_agents": float(record.get("slice_behavior_aggressiveness_selected_agents", 0.0)),
                "slice_behavior_quantile_selected_agents": float(record.get("slice_behavior_quantile_selected_agents", 0.0)),
                "slice_behavior_raw_score_selected_agents": float(record.get("slice_behavior_raw_score_selected_agents", 0.0)),
                "generated_difficulty_selected_agents": float(record.get("generated_difficulty_selected_agents", record.get("generated_difficulty", 0.0))),
                "generated_difficulty_full_scene_proxy": float(record.get("generated_difficulty_full_scene_proxy", 0.0)),
                "generated_behavior_quantile": float(record.get("generated_behavior_quantile", 0.0)),
                "generated_behavior_aggressiveness": float(record.get("generated_behavior_aggressiveness", 0.0)),
                "generated_behavior_raw_score": float(record.get("generated_behavior_raw_score", 0.0)),
                "generated_stress_difficulty_selected_agents": float(record.get("generated_stress_difficulty_selected_agents", 0.0)),
                "generated_full_scene_stress_proxy": float(record.get("generated_full_scene_stress_proxy", 0.0)),
                "gt_behavior_quantile": float(record.get("gt_behavior_quantile", 0.0)),
                "gt_behavior_raw_score": float(record.get("gt_behavior_raw_score", 0.0)),
                "target_difficulty": float(record.get("target_difficulty", 0.0)),
                "support_score": float(record.get("support_score", 0.0)),
                "support_selected_agents": float(record.get("support_selected_agents", record.get("support_score", 0.0))),
                "support_full_scene": float(record.get("support_full_scene", record.get("support_score", 0.0))),
                "coverage": float(record.get("coverage", record.get("selected_agents_coverage_score", 1.0))),
                "selected_agents_coverage_score": float(record.get("selected_agents_coverage_score", record.get("selected_agents_difficulty_coverage", 1.0))),
                "selected_agents_difficulty_coverage": float(record.get("selected_agents_difficulty_coverage", 1.0)),
                "generation_mode": str(record.get("generation_mode", "")),
                "mode": str(record.get("mode", record.get("generation_mode", ""))),
                "is_same_scene_diagnostic": bool(record.get("is_same_scene_diagnostic", False)),
                "generator_condition_behavior": _safe_float(record.get("generator_condition_behavior", 0.0)),
            }
            for row, record in zip(per_sample_rows, records)
        ]
        phase_bar.update(1)
        warnings = _filtered_warnings(metadata, metadata.get("behavior_distribution_report", {}))
        conditioning_summary = {}
        if conditioning_rows:
            requested = pd.Series([float(row["requested_target_behavior"]) for row in conditioning_rows], dtype=float)
            actual = pd.Series([float(row["model_input_target_behavior"]) for row in conditioning_rows], dtype=float)
            conditioning_summary = {
                "count": int(len(conditioning_rows)),
                "requested_target_behavior_mean": float(requested.mean()),
                "model_input_target_behavior_mean": float(actual.mean()),
                "target_minus_input_mae": float((requested - actual).abs().mean()),
                "target_minus_input_max_abs": float((requested - actual).abs().max()),
            }
        targets = [float(record.get("target_behavior", record.get("target_difficulty", 0.0))) for record in records]
        sampled = [float(record.get("generated_behavior_quantile", 0.0)) for record in records]
        calibration_recommendation = None
        debug_subset_unsuitable = any("not suitable for absolute behavior calibration" in warning.lower() for warning in warnings)
        if debug_subset_unsuitable:
            calibration_recommendation = {"disabled_reason": "debug subset lacks sufficient behavior support; calibration conclusions withheld."}
        elif len(records) >= 8:
            method = str(config["sampling"].get("behavior_calibration_method", "linear"))
            calibrator = fit_behavior_calibrator(targets, sampled, method=method)
            calibration_recommendation = {
                "method": method,
                "parameters": calibrator.to_dict(),
                **calibration_error_summary(targets, sampled, calibrator),
            }
        baseline_df = _baseline_comparison(records, project_root=project_root, cache_dir=config["data"]["cache_dir"], config=config)
        write_analysis_report(
            output_dir=analysis_dir,
            data_stats=metadata,
            retrieval_df=retrieval_df,
            realism_df=realism_df,
            preservation_df=preservation_df,
            diversity_df=diversity_df,
            baseline_df=baseline_df,
            diagnostic_df=diagnostic_df,
            per_sample_metrics=per_sample_rows,
            warnings=warnings,
            sample_metadata=sample_metadata,
            calibration_recommendation=calibration_recommendation,
            conditioning_summary=conditioning_summary,
            run_metadata=run_metadata,
        )
        (analysis_dir / "analyze_runtime_snapshot.json").write_text(
            json.dumps(analyze_runtime_snapshot, indent=2),
            encoding="utf-8",
        )
        phase_bar.update(1)
    return {
        "retrieval_full_scene_mae": float(retrieval_df.iloc[0]["retrieval_full_scene_mae"]),
        "behavior_preservation_mae": float(preservation_df.iloc[0]["generated_vs_retrieved_behavior_mae"]),
        "sliced_wasserstein": float(realism_df.iloc[0]["sliced_wasserstein"]),
        "novelty_score": float(diversity_df.iloc[0]["novelty_score"]),
        "analyze_runtime_snapshot_files": int(len(analyze_runtime_snapshot.get("files", {}))),
    }
