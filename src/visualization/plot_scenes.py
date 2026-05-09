from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from isgen import ensure_dir


def _draw_agent_box(ax, state: np.ndarray, size: np.ndarray, color: str, linewidth: float = 1.5) -> None:
    x, y, _, _, heading = state
    length, width = size
    corners = np.asarray(
        [
            [length / 2.0, width / 2.0],
            [length / 2.0, -width / 2.0],
            [-length / 2.0, -width / 2.0],
            [-length / 2.0, width / 2.0],
            [length / 2.0, width / 2.0],
        ],
        dtype=np.float32,
    )
    c, s = np.cos(heading), np.sin(heading)
    rotation = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    rotated = corners @ rotation.T + np.asarray([x, y], dtype=np.float32)
    ax.plot(rotated[:, 0], rotated[:, 1], color=color, linewidth=linewidth)


def plot_sample_scene(record: Dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(7, 7))
    map_polylines = np.asarray(record["map_polylines"])
    map_point_mask = np.asarray(record["map_point_mask"])
    for polyline, mask in zip(map_polylines, map_point_mask):
        if mask.any():
            ax.plot(polyline[mask, 0], polyline[mask, 1], color="lightgray", linewidth=1.0)
    history = np.asarray(record["history_states"])
    current = np.asarray(record["current_states"])
    generated = np.asarray(record["generated_future"])
    gt = np.asarray(record["gt_future"])
    future_mask = np.asarray(record["future_mask"])
    agent_mask = np.asarray(record["agent_mask"])
    sizes = np.full((len(current), 2), [4.8, 2.0], dtype=np.float32)
    focal_index = 0
    for idx, valid in enumerate(agent_mask):
        if not valid:
            continue
        color = "tab:red" if idx == focal_index else "tab:blue"
        hist_mask = np.linalg.norm(history[idx], axis=-1) > 0
        ax.plot(history[idx, hist_mask, 0], history[idx, hist_mask, 1], color=color, alpha=0.5, linewidth=1.0)
        _draw_agent_box(ax, current[idx], sizes[idx], color=color, linewidth=2.0 if idx == focal_index else 1.0)
        if future_mask[idx].any():
            ax.plot(gt[idx, future_mask[idx], 0], gt[idx, future_mask[idx], 1], color="tab:green", linestyle="--", linewidth=1.5)
            ax.plot(generated[idx, future_mask[idx], 0], generated[idx, future_mask[idx], 1], color=color, linewidth=2.0)
    ax.set_aspect("equal")
    ax.set_title(
        " | ".join(
            [
                f"target={record['target_difficulty']:.2f}",
                f"target_b={record.get('target_behavior', record['target_difficulty']):.2f}",
                f"gen_b={record.get('generated_behavior_aggressiveness', 0.0):.2f}",
                f"stress_sel={record.get('generated_stress_difficulty_selected_agents', record.get('generated_difficulty_selected_agents', record['generated_difficulty'])):.2f}",
                f"stress_full={record.get('generated_full_scene_stress_proxy', record.get('generated_difficulty_full_scene_proxy', 0.0)):.2f}",
                f"s_sel={record.get('support_selected_agents', record['support_score']):.2f}",
                f"s_full={record.get('support_full_scene', record['support_score']):.2f}",
            ]
        )
    )
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_difficulty_grid(records: List[Dict], output_path: str | Path, target_values: List[float]) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(len(target_values), 1, figsize=(8, 3 * len(target_values)))
    axes = np.atleast_1d(axes)
    for ax, target in zip(axes, target_values):
        subset = [record for record in records if abs(record["target_difficulty"] - target) < 1e-6][:5]
        for idx, record in enumerate(subset):
            gen = np.asarray(record["generated_future"])[0]
            mask = np.asarray(record["future_mask"])[0]
            ax.plot(
                gen[mask, 0],
                gen[mask, 1],
                label=f"s{idx}:b={record.get('generated_behavior_aggressiveness', 0.0):.2f}",
            )
        ax.set_title(f"target difficulty = {target:.2f}")
        if subset:
            ax.legend(fontsize=7)
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_same_scene_diagnostic(records: List[Dict], output_path: str | Path) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(1, len(records), figsize=(4 * len(records), 4))
    axes = np.atleast_1d(axes)
    for ax, record in zip(axes, records):
        generated = np.asarray(record["generated_future"])
        future_mask = np.asarray(record["future_mask"])
        for idx in range(min(len(generated), 4)):
            if future_mask[idx].any():
                ax.plot(generated[idx, future_mask[idx], 0], generated[idx, future_mask[idx], 1], linewidth=1.5)
        title = (
            f"t={record['target_difficulty']:.2f}\n"
            f"sel={record.get('support_selected_agents', record['support_score']):.2f}\n"
            f"full={record.get('support_full_scene', record['support_score']):.2f}"
        )
        if record.get("low_support", False):
            title += "\nLOW SUPPORT"
            ax.set_facecolor("#ffe6e6")
        ax.set_title(title, color="red" if record.get("low_support", False) else "black")
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
