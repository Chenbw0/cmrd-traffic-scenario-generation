"""Compatibility package for the flattened src/ layout.

This project originally used ``src/isgen/...``. If the repository is later
flattened to ``src/<module_group>/...``, existing imports such as
``from isgen.data.cache import ...`` should keep working.
"""

from __future__ import annotations

import json
import random
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    base_ref = payload.pop("_base_", None)
    if base_ref is None:
        return payload
    base_path = Path(base_ref)
    if not base_path.is_absolute():
        base_path = (config_path.parent / base_path).resolve()
    base_config = load_config(base_path)
    return _deep_merge_dict(base_config, payload)


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(root: str | Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(root) / path


def resolve_output_path(config: Dict[str, Any], root: str | Path, key: str, default: str | Path) -> Path:
    outputs_cfg = config.get("outputs", {})
    target = outputs_cfg.get(key, default)
    return resolve_path(root, target)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


_PROJECT_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_SRC_ROOT))


def _alias_module(alias: str, target: str) -> None:
    sys.modules[alias] = import_module(target)


for _package in (
    "data",
    "evaluation",
    "models",
    "retrieval",
    "sampling",
    "semantics",
    "training",
    "visualization",
):
    _alias_module(f"{__name__}.{_package}", _package)

for _module in (
    "data.cache",
    "data.dataset",
    "data.interaction_reader",
    "data.map_reader",
    "data.slice_builder",
    "data.synthetic",
    "data.transforms",
    "evaluation.analyze",
    "evaluation.controllability",
    "evaluation.diversity",
    "evaluation.realism",
    "evaluation.report",
    "models.blocks",
    "models.diffusion",
    "models.encoders",
    "models.spawn_generator",
    "models.scenario_generator",
    "models.support_estimator",
    "retrieval.retriever",
    "retrieval.slice_index",
    "retrieval.support",
    "sampling.export",
    "sampling.calibration",
    "sampling.sample",
    "semantics.behavior",
    "semantics.control_normalization",
    "semantics.residual_normalization",
    "semantics.difficulty",
    "semantics.kinematics",
    "semantics.normalization",
    "semantics.scene_features",
    "semantics.spawn_plan",
    "semantics.spawn_ordering",
    "semantics.support_prototypes",
    "semantics.validity",
    "training.checkpointing",
    "training.logging_utils",
    "training.losses",
    "training.spawn_losses",
    "training.train",
    "evaluation.spawn_analyze",
    "visualization.plot_scenes",
):
    _alias_module(f"{__name__}.{_module}", _module)
