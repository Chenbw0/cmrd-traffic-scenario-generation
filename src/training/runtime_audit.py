from __future__ import annotations

import hashlib
import inspect
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import isgen

from isgen import ensure_dir, save_json
from isgen.models.scenario_generator import DifficultyConditionedScenarioDiffusion
from isgen.training.checkpointing import load_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_SOURCE_FILES = {
    "scenario_generator_runtime.py": PROJECT_ROOT / "src" / "models" / "scenario_generator.py",
    "losses_runtime.py": PROJECT_ROOT / "src" / "training" / "losses.py",
    "behavior_runtime.py": PROJECT_ROOT / "src" / "semantics" / "behavior.py",
    "difficulty_runtime.py": PROJECT_ROOT / "src" / "semantics" / "difficulty.py",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_contains(path: Path, pattern: str) -> bool:
    return pattern in path.read_text(encoding="utf-8")


def _source_method_contains(obj: Any, method_name: str, pattern: str) -> bool:
    method = getattr(obj, method_name)
    return pattern in inspect.getsource(method)


def capture_runtime_sources(snapshot_dir: str | Path, phase: str, config_hash: str | None = None, checkpoint_hash: str | None = None) -> Dict[str, Any]:
    snapshot_root = ensure_dir(snapshot_dir)
    phase_root = ensure_dir(Path(snapshot_root) / phase)
    manifest: Dict[str, Any] = {
        "phase": phase,
        "config_hash": config_hash,
        "checkpoint_hash": checkpoint_hash,
        "files": {},
    }
    for target_name, source_path in RUNTIME_SOURCE_FILES.items():
        source_path = source_path.resolve()
        target_path = phase_root / target_name
        shutil.copy2(source_path, target_path)
        manifest["files"][target_name] = {
            "source_path": str(source_path),
            "snapshot_path": str(target_path),
            "sha256": file_sha256(source_path),
        }
    save_json(manifest, phase_root / "runtime_sources.json")
    return manifest


def inspect_runtime_state(
    config: Dict,
    checkpoint_path: str | Path | None = None,
    snapshot_dir: str | Path | None = None,
    strict: bool = True,
) -> Dict[str, Any]:
    checkpoint_payload: Dict[str, Any] | None = None
    checkpoint_decoder_type = None
    checkpoint_fingerprint = None
    checkpoint_path_resolved: Path | None = None
    if checkpoint_path is not None:
        checkpoint_path_resolved = Path(checkpoint_path)
        checkpoint_payload = load_checkpoint(checkpoint_path_resolved)
        checkpoint_decoder_type = checkpoint_payload.get("config", {}).get("model", {}).get("decoder_type")
        checkpoint_fingerprint = checkpoint_payload.get("run_metadata", {}).get("code_fingerprint")

    config_decoder_type = config["model"].get("decoder_type")
    model = DifficultyConditionedScenarioDiffusion(config)
    runtime_metadata = model.get_runtime_metadata()
    runtime_source_file = Path(runtime_metadata["runtime_source_file"]).resolve()
    runtime_source_sha256 = file_sha256(runtime_source_file)

    snapshot_root = ensure_dir(snapshot_dir or (Path.cwd() / "runtime_source_snapshot"))
    snapshot_manifest = capture_runtime_sources(snapshot_root, phase="inspect")
    snapshot_path = Path(snapshot_manifest["files"]["scenario_generator_runtime.py"]["snapshot_path"])
    snapshot_sha256 = snapshot_manifest["files"]["scenario_generator_runtime.py"]["sha256"]
    repo_source_file = (PROJECT_ROOT / "src" / "models" / "scenario_generator.py").resolve()
    repo_source_sha256 = file_sha256(repo_source_file)

    has_encode_future_controls = _source_contains(runtime_source_file, "def encode_future_controls")
    has_encode_future_controls_to_knots = _source_contains(runtime_source_file, "def encode_future_controls_to_knots")
    has_decode_controls_to_future = _source_contains(runtime_source_file, "def decode_controls_to_future")
    has_decode_control_knots_to_future = _source_contains(runtime_source_file, "def decode_control_knots_to_future")
    forward_contains_controls = _source_method_contains(model, "forward", "self.encode_future_controls(")
    forward_contains_control_knots = _source_method_contains(model, "forward", "self.encode_future_controls_to_knots(")
    forward_contains_local = _source_method_contains(model, "forward", "self.encode_future_local(")
    sample_contains_controls = _source_method_contains(model, "sample", "self.decode_controls_to_future(")
    sample_contains_control_knots = _source_method_contains(model, "sample", "self.decode_control_knots_to_future(")
    sample_contains_local = _source_method_contains(model, "sample", "self.decode_future_local(")

    payload = {
        "python_executable": sys.executable,
        "sys_path_head": sys.path[:8],
        "isgen_package_path": str(Path(inspect.getsourcefile(isgen) or "").resolve()),
        "scenario_generator_module_path": str(Path(inspect.getsourcefile(sys.modules[DifficultyConditionedScenarioDiffusion.__module__]) or "").resolve()),
        "runtime_source_file": str(runtime_source_file),
        "runtime_source_sha256": runtime_source_sha256,
        "repo_source_file": str(repo_source_file),
        "repo_source_sha256": repo_source_sha256,
        "runtime_source_snapshot": str(snapshot_path),
        "runtime_source_snapshot_sha256": snapshot_sha256,
        "runtime_code_fingerprints": {
            key: value["sha256"] for key, value in snapshot_manifest["files"].items()
        },
        "source_contains_encode_future_controls": has_encode_future_controls,
        "source_contains_encode_future_controls_to_knots": has_encode_future_controls_to_knots,
        "source_contains_decode_controls_to_future": has_decode_controls_to_future,
        "source_contains_decode_control_knots_to_future": has_decode_control_knots_to_future,
        "source_contains_encode_future_local": _source_contains(runtime_source_file, "def encode_future_local"),
        "source_contains_decode_future_local": _source_contains(runtime_source_file, "def decode_future_local"),
        "forward_contains_encode_future_controls": forward_contains_controls,
        "forward_contains_encode_future_controls_to_knots": forward_contains_control_knots,
        "forward_contains_encode_future_local": forward_contains_local,
        "sample_contains_decode_controls_to_future": sample_contains_controls,
        "sample_contains_decode_control_knots_to_future": sample_contains_control_knots,
        "sample_contains_decode_future_local": sample_contains_local,
        "decoder_type": model.decoder_type,
        "control_representation": getattr(model, "control_representation", "per_step"),
        "num_control_knots": getattr(model, "num_control_knots", 0),
        "control_interpolation": getattr(model, "control_interpolation", "linear"),
        "residual_diffusion_enabled": getattr(model, "residual_diffusion_enabled", False),
        "diffusion_target_type": getattr(model, "diffusion_target_type", "epsilon"),
        "local_future_dim": model.local_future_dim,
        "control_dim": model.control_dim,
        "encoded_target_channel_names": list(model.control_channel_names if model.decoder_type == "kinematic_controls" else ["dx", "dy", "dvx", "dvy", "dheading"]),
        "forward_path": runtime_metadata["forward_path_name"],
        "sample_decode_path": runtime_metadata["sample_decode_path_name"],
        "uses_encode_future_controls": bool(runtime_metadata["uses_encode_future_controls"]),
        "uses_decode_controls_to_future": bool(runtime_metadata["uses_decode_controls_to_future"]),
        "checkpoint_decoder_type": checkpoint_decoder_type,
        "checkpoint_source_sha256": checkpoint_fingerprint,
        "config_decoder_type": config_decoder_type,
        "checkpoint_path": None if checkpoint_path_resolved is None else str(checkpoint_path_resolved.resolve()),
    }

    errors: list[str] = []
    if config_decoder_type != model.decoder_type:
        errors.append(f"config decoder_type ({config_decoder_type}) != runtime model decoder_type ({model.decoder_type})")
    if checkpoint_decoder_type is not None and checkpoint_decoder_type != config_decoder_type:
        errors.append(f"checkpoint decoder_type ({checkpoint_decoder_type}) != config decoder_type ({config_decoder_type})")
    if runtime_source_sha256 != repo_source_sha256:
        errors.append("runtime source sha256 does not match repository scenario_generator.py")
    if snapshot_sha256 != runtime_source_sha256:
        errors.append("runtime source snapshot sha256 does not match runtime source")
    if config_decoder_type == "kinematic_controls":
        if payload["control_representation"] == "knots":
            if not has_encode_future_controls_to_knots or not has_decode_control_knots_to_future:
                errors.append("kinematic_controls+knots requested but runtime source is missing knot encode/decode control methods")
            if payload["forward_path"] != "encode_future_controls_to_knots" or not forward_contains_control_knots:
                errors.append("kinematic_controls+knots requested but forward path is not encode_future_controls_to_knots")
            if payload["sample_decode_path"] != "decode_control_knots_to_future" or not sample_contains_control_knots:
                errors.append("kinematic_controls+knots requested but sample decode path is not decode_control_knots_to_future")
        else:
            if not has_encode_future_controls or not has_decode_controls_to_future:
                errors.append("kinematic_controls requested but runtime source is missing encode/decode control methods")
            if payload["forward_path"] != "encode_future_controls" or not forward_contains_controls:
                errors.append("kinematic_controls requested but forward path is not encode_future_controls")
            if payload["sample_decode_path"] != "decode_controls_to_future" or not sample_contains_controls:
                errors.append("kinematic_controls requested but sample decode path is not decode_controls_to_future")
    checkpoint_runtime_fingerprints = {}
    if checkpoint_payload is not None:
        checkpoint_runtime_fingerprints = checkpoint_payload.get("run_metadata", {}).get("runtime_code_fingerprints", {})
    for file_name, sha256 in payload["runtime_code_fingerprints"].items():
        expected = checkpoint_runtime_fingerprints.get(file_name)
        if expected is not None and expected != sha256:
            errors.append(f"checkpoint runtime fingerprint mismatch for {file_name}: checkpoint={expected} current={sha256}")
    if checkpoint_fingerprint is not None and checkpoint_fingerprint != runtime_source_sha256:
        errors.append(
            f"checkpoint source sha256 ({checkpoint_fingerprint}) != runtime source sha256 ({runtime_source_sha256})"
        )
    payload["errors"] = errors
    if strict and errors:
        raise RuntimeError("Runtime inspection failed: " + " | ".join(errors))
    return payload
