# INTERACTION Scenario Generation

Clean PyTorch research code for support-aware retrieval-augmented scenario generation on the INTERACTION dataset.

## Positioning

- We do not use paired counterfactual data.
- We do not claim strict causal counterfactual recovery.
- We do not require a fixed scene to realize arbitrary difficulty levels.
- We do not require user-provided history at generation time.
- Difficulty control is achieved primarily through support-aware retrieval.
- We retrieve current scene snapshots from a naturalistic scenario library.
- The generator conditions on the retrieved current scene snapshot and map, not on a requested behavior target.
- Behavior preservation is evaluated after generation and is not the main training objective.
- The generator produces human-like future rollouts from current states.
- Same-scene arbitrary behavior control is not claimed.
- Same-scene target sweep is diagnostic only.

## Core Task

Given a target stress difficulty and optional map/location constraints:

1. Retrieve real INTERACTION scene slices whose support matches the requested difficulty.
2. Keep the real map, real current multi-agent snapshot, and real interaction context.
3. Generate multi-agent future rollouts from the retrieved current-state snapshot.
4. Evaluate:
   - retrieval controllability
   - generation realism
   - preservation of retrieved slice behavior/stress
   - novelty and diversity

This project does not generate maps from scratch and does not generate agent spawn states from scratch.

## Main Terms

- `scene slice`: a real map crop plus a temporal window with current agent states and future trajectories; history may be stored for analysis or visualization but is not required by the main generation interface
- `target difficulty`: the requested retrieval-level stress target in `[0, 1]`
- `retrieval support`: whether a slice plausibly supports the requested target
- `retrieval_augmented_generation`: retrieve supported real slices first, then generate futures on those slices

## Modes

- `retrieval_augmented_generation`
  Main mode. Retrieve slices by target difficulty, then generate futures from the retrieved current scene snapshot and map.

- `retrieval_behavior_conditioned_generation`
  Explicit behavior-conditioned baseline on retrieved slices.

- `retrieval_unconditional_generation`
  Retrieve slices, but sample futures without behavior conditioning.

- `retrieval_replay`
  Retrieve slices and replay the real ground-truth future.

- `same_scene_diagnostic`
  Diagnostic only. Not part of the main claim.

## Data

The project reads local data from `data/` and never downloads data automatically.

Supported trajectory formats:

- CSV
- Parquet

Supported map formats:

- `.osm`
- `.osm_xy`
- lanelet/xml/json fallbacks when available

If required fields are missing:

- heading: estimated from velocity direction
- velocity: estimated causally from past position differences
- length/width: default to `4.8m x 2.0m`
- map: trajectory-only fallback is allowed with warnings

## Project Layout

```text
project_root/
  configs/
  data/
  outputs/
  scripts/
  src/
  tests/
  README.md
  requirements.txt
```

## Main Outputs

- `outputs/cache/`
  preprocessed slices, difficulty stats, behavior stats, control stats, slice index, preprocessing report

- `outputs/checkpoints/`
  model checkpoints, training history, runtime metadata

- `outputs/samples/`
  generated samples, conditioning trace, figures, sampling metadata

- `outputs/analysis/`
  markdown report plus retrieval/realism/preservation/diversity metrics

- `outputs/experiments/`
  isolated experiment roots for reproducible paper runs

## Commands

Install:

```bash
pip install -r requirements.txt
```

Prepare data:

```bash
python scripts/prepare_data.py --config configs/default.yaml
python scripts/prepare_data.py --config configs/default.yaml --force_rebuild --debug_profile behavior_balanced
```

Train:

```bash
python scripts/train.py --config configs/default.yaml
python scripts/train.py --config configs/default.yaml --resume false
```

Main sampling mode:

```bash
python scripts/sample.py --config configs/default.yaml --mode retrieval_augmented_generation --target_difficulties 0.2 0.4 0.6 0.8 --num_rollouts_per_slice 3
```

Baselines:

```bash
python scripts/sample.py --config configs/default.yaml --mode retrieval_replay --target_difficulties 0.2 0.4 0.6 0.8
python scripts/sample.py --config configs/default.yaml --mode retrieval_unconditional_generation --target_difficulties 0.2 0.4 0.6 0.8
python scripts/sample.py --config configs/default.yaml --mode retrieval_behavior_conditioned_generation --target_difficulties 0.2 0.4 0.6 0.8
```

Diagnostic only:

```bash
python scripts/sample.py --config configs/default.yaml --mode same_scene_diagnostic --slice_id <slice_id>
```

Analyze:

```bash
python scripts/analyze.py --config configs/default.yaml --samples outputs/samples/samples.pt
```

Runtime/source audit:

```bash
python scripts/inspect_runtime.py --config configs/default.yaml --checkpoint outputs/checkpoints/latest.pt
```

Run one isolated experiment:

```bash
python scripts/run_experiment.py --config configs/experiments/debug_behavior_balanced.yaml
python scripts/run_experiment.py --config configs/experiments/main_location_split.yaml
```

Run the protocol wrapper:

```bash
python scripts/run_main_protocol.py --profile debug_sanity
python scripts/run_main_protocol.py --profile main_experiment
python scripts/run_main_protocol.py --profile baseline_experiment
```

Smoke test:

```bash
python scripts/run_smoke_test.py --config configs/default.yaml
```

## Report Structure

The main report is organized as:

1. Dataset and Cache Summary
2. Retrieval Controllability Summary
3. Generation Preservation Summary
4. Generation Realism Summary
5. Novelty and Diversity Summary
6. Baseline Comparison
7. Diagnostic Same-Slice Summary

Only the last section is diagnostic and should not be used as the primary conclusion.

## Experiment Configs

- `configs/experiments/debug_behavior_balanced.yaml`
  Fast sanity profile. No paper conclusion.

- `configs/experiments/main_location_split.yaml`
  Main paper-facing experiment profile.

- `configs/experiments/ablation_no_behavior_condition.yaml`
  Retrieval baseline without behavior conditioning.

- `configs/experiments/ablation_unconditional.yaml`
  Alias config for unconditional generation ablation.

- `configs/experiments/ablation_replay.yaml`
  Retrieval replay baseline.

- `configs/experiments/ablation_current_only_generation.yaml`
  Main generation setting: map plus current multi-agent snapshot, without using history for generation.

- `configs/experiments/ablation_recent_k_history_generation.yaml`
  History-enabled ablation using only a recent short history window.

- `configs/experiments/ablation_full_history_generation.yaml`
  History-enabled ablation using the full cached history window.
