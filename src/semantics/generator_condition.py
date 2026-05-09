from __future__ import annotations

from typing import Any, Dict


GENERATOR_CONDITION_MODES = {
    "none",
    "behavior_only",
    "difficulty_only",
    "behavior_plus_difficulty",
}


def resolve_generator_condition_mode(config: Dict[str, Any]) -> str:
    generator_condition = config.get("generator_condition", {})
    mode = str(generator_condition.get("mode", "none")).lower()
    if mode not in GENERATOR_CONDITION_MODES:
        raise ValueError(
            f"Unsupported generator_condition.mode: {mode}. "
            f"Expected one of {sorted(GENERATOR_CONDITION_MODES)}."
        )
    return mode


def generator_consumes_behavior(config: Dict[str, Any]) -> bool:
    return resolve_generator_condition_mode(config) in {"behavior_only", "behavior_plus_difficulty"}


def generator_consumes_difficulty(config: Dict[str, Any]) -> bool:
    return resolve_generator_condition_mode(config) in {"difficulty_only", "behavior_plus_difficulty"}


def behavior_control_loss_in_total(config: Dict[str, Any]) -> bool:
    return generator_consumes_behavior(config) and float(config.get("losses", {}).get("behavior_control_weight", 0.0)) > 0.0


def resolve_main_training_objective(config: Dict[str, Any]) -> str:
    return (
        "retrieval_augmented_rollout_only"
        if resolve_generator_condition_mode(config) == "none"
        else "retrieval_augmented_behavior_rollout"
    )
