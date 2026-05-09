from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict


def configure_logging(output_dir: str | Path) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def summarize_metrics(metrics: Dict[str, float]) -> str:
    behavior_in_total = float(metrics.get("behavior_control_loss_in_total", 1.0)) > 0.5
    preferred_order = [
        "total",
        "anchor",
        "anchor_mid_mae",
        "anchor_final_mae",
        "anchor_heading_mae",
        "supervised_control",
        "supervised_rollout",
        "residual_diffusion",
        "anchor_residual_diffusion",
        "control_delta",
        "oracle_knot_rollout",
        "mu_knot_mae",
        "mu_rollout_gap_to_oracle",
        "anchor_residual_std",
        "sampled_anchor_residual_std",
        "anchor_mu_std",
        "final_anchor_std",
        "generated_vs_retrieved_behavior_mae",
        "generated_vs_gt_behavior_mae",
        "residual_knot_std",
        "sampled_residual_std",
        "final_knot_std",
        "gt_knot_std",
        "residual_clip_fraction",
        "residual_energy_ratio",
        "behavior_control",
        "behavior_control_mae_quantile",
        "generated_behavior_quantile_mean",
        "generated_behavior_quantile_std",
        "retrieved_behavior_quantile_mean",
        "retrieved_behavior_quantile_std",
        "gt_behavior_quantile_mean",
        "gt_behavior_quantile_std",
    ]
    hidden_keys = set()
    if not behavior_in_total:
        hidden_keys.update(
            {
                "behavior_control",
                "behavior_control_mae_quantile",
            }
        )
    parts = []
    for key in preferred_order:
        if key in metrics and key not in hidden_keys and abs(float(metrics[key])) > 1e-12:
            parts.append(f"{key}={metrics[key]:.4f}")
    return " | ".join(parts)
