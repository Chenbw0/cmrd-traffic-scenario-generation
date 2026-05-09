# CMRD: Difficulty-Controllable Traffic Scenario Generation

This repository provides the public code and reproducibility materials for the manuscript:

**Difficulty-Controllable Traffic Scenario Generation for Automated-Driving Evaluation via Context-Matched Residual Diffusion**

CMRD is a context-matched residual diffusion framework for generating traffic scenarios at requested interaction-difficulty levels. The framework first retrieves a real current traffic snapshot compatible with the requested difficulty level and then synthesizes new future trajectories in acceleration-yaw-rate control space.

## Overview

Testing automated-driving planners requires traffic scenarios whose interaction difficulty can be varied without producing physically invalid or unrealistic behavior. CMRD follows a feasibility-first generation principle:

1. Difficulty calibration maps a requested difficulty level to an empirical traffic-difficulty percentile.
2. Context matching retrieves a real current traffic snapshot whose map topology, density, spacing and interaction geometry can support the requested difficulty level.
3. Reference-control residual diffusion synthesizes future trajectories by predicting a reference acceleration-yaw-rate control sequence and sampling residual control corrections.
4. Bounded target guidance moves generated candidates toward the requested difficulty.
5. Must-pass validity checks remove candidates with collision, map, dynamics or feature-envelope violations before target-error minimization.
6. Traffic-semantic evaluation checks whether generated scenarios remain controllable and realistic.

## Current public release

The current public release contains reproducibility materials associated with the submitted manuscript, including:

- configuration files for reported experiments;
- data-preparation interfaces;
- evaluation-related scripts and metric utilities;
- example commands for available preprocessing and evaluation components;
- documentation for the experimental protocol.

Additional training and scenario-generation modules are being cleaned and will be uploaded as they are ready. The README will be updated when new modules are released.

## Repository structure

```text
configs/      Configuration files for experiments
src/          Public source-code modules
evaluation/   Evaluation scripts and metric utilities
scripts/      Command-line scripts and utilities
examples/     Minimal examples and usage templates
docs/         Public documentation, if available
```

The structure may be updated as additional modules are released.

## Installation

A typical Python environment can be created with:

```bash
conda create -n cmrd python=3.10
conda activate cmrd
pip install -r requirements.txt
```

If an `environment.yml` file is provided, the environment can also be created with:

```bash
conda env create -f environment.yml
conda activate cmrd
```

## Dataset

This repository does **not** redistribute the raw INTERACTION dataset. Users should obtain the dataset from its official provider and follow the preparation instructions required by the dataset license.

After downloading the dataset, place or link the data according to the paths specified in the configuration files. Raw data, processed data, generated outputs and model checkpoints should not be committed to this repository.

Example directory layout:

```text
/path/to/datasets/INTERACTION/
```

Update the dataset path in the configuration file before running the scripts.

## Basic usage

The available scripts depend on the current release. A typical workflow is:

```bash
python scripts/prepare_data.py --config configs/default.yaml
python evaluation/run_metrics.py --config configs/default.yaml
```

If a script is not included in the current public release, the corresponding module is still being cleaned and will be uploaded in a later update.

## Reproducibility note

This repository is associated with the submitted manuscript and is intended to support reproducibility of the reported experimental protocol. The current release focuses on documentation, configuration files, data-preparation interfaces and evaluation-related scripts. Additional cleaned modules for training, generation and full table reproduction will be uploaded as they are ready.

The repository does not include:

- raw INTERACTION dataset files;
- private notes or submission materials;
- patent-related files;
- manuscript source files;
- unpublished large checkpoints or generated data dumps unless explicitly released.

## Citation

If this repository is useful for your research, please cite the corresponding manuscript:

```text
Chen, B., Guo, F., Liu, Y., Zhong, H., Deng, C., Chen, B., and Chen, Z.
Difficulty-Controllable Traffic Scenario Generation for Automated-Driving Evaluation via Context-Matched Residual Diffusion.
Transportation Research Part C: Emerging Technologies, under review.
```

The citation entry will be updated after publication.

## License

Please check the license file in this repository before using the code.

If no license file is provided, all rights are reserved until a license is added.

## Contact

For questions about the repository or manuscript, please contact:

```text
Zheng Chen
Faculty of Transportation Engineering
Kunming University of Science and Technology
Email: [add email address]
```