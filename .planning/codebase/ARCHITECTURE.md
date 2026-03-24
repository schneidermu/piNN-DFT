# Architecture

## Overview

- The repository is organized by workflow rather than by a reusable application package.
- Core mathematical definitions live in shared modules, while training and evaluation are driven by standalone scripts.
- The main architectural seam is: learned local constants -> functional energy evaluation -> chemistry workflow integration.

## Main Data Flow

1. Raw per-system `.h5` files are downloaded into `train_models/data/`.
2. `train_models/prepare_data.py` converts raw data into grouped train/test/predopt pickle artifacts.
3. `train_models/predopt_train.py` loads those pickles, builds `pcPBELMLOptimizerV2`, and trains with distributed PyTorch.
4. Best checkpoints are written to `train_models/best_models/`.
5. `test_models/DFT/functional.py` loads checkpoint weights and exposes an `NN_FUNCTIONAL.eval_xc` hook compatible with PySCF.
6. Benchmark scripts in `test_models/` run SCF or reaction-energy calculations and write result files for later analysis.

## Core Layers

### Functional Math Layer

- `dft_functionals/PBE.py` implements the baseline exchange-correlation math and accepts either true constants or NN-modified constants.
- `dft_functionals/constants.py` centralizes descriptor indices, epsilon values, spin-scaling multipliers, and canonical constant tensors.
- This layer is pure numerical logic and is reused by both training and inference code.

### Model Layer

- `train_models/NN_models.py` defines the neural architectures that output modified PBE constants.
- The key model family is `pcPBELMLOptimizer*`, referenced by `train_models/predopt_train.py` and validated by `train_models/test.py`.
- The model output is not a direct energy; it is a structured constant vector that preserves analytical constraints by construction.

### Dataset and Preprocessing Layer

- `train_models/dataset.py` indexes `.h5` files, expands grid augmentations, constructs reaction samples, and computes local PBE energies.
- `train_models/prepare_data.py` controls train/test splitting and serializes processed datasets.
- Utility batching logic is in `train_models/utils.py`, especially `stack_reactions(...)`.

### Training Orchestration Layer

- `train_models/predopt_train.py` is the main orchestrator for pre-optimization, main training, validation, checkpointing, plotting, and MLflow logging.
- Distributed training is handled with `DistributedSampler`, DDP, and `torchrun`.
- Hyperparameter sweep orchestration is in `train_models/calculations.py`, while newer tuning and analysis scripts live in `train_models/optuna_joint.py`, `train_models/optuna_vxc.py`, and related helpers.

### Inference and Benchmark Layer

- `test_models/DFT/functional.py` adapts trained models for PySCF’s `eval_xc` contract.
- `test_models/script.py`, `test_models/calculate_system_energies.py`, `test_models/run_molden.py`, and `test_models/plot_exc.py` are executable analysis entry points.
- `test_models/DFT/numint.py` extends PySCF numerical integration for Laplacian-capable models.

## Execution Style

- Most workflows are script-first and depend on the current working directory.
- There is little separation between library code and executable code; modules often assume local relative paths and side-effectful execution.
- Job generation and submission are part of the codebase, not external tooling.

## Shared Abstractions

- The most important shared abstraction is the constant tensor schema defined in `dft_functionals/constants.py`.
- Reaction dictionaries are the other cross-cutting data structure, with keys like `Grid`, `Weights`, `Densities`, `Gradients`, and `Database`.
- Model input tensors are derived from those reaction or grid structures by helpers like `train_models/utils.py:_grid_to_model_input`.

## Boundaries and Coupling

- Training and inference share the mathematical core, but they duplicate some feature-building logic.
- `test_models/DFT/functional.py` contains inference-specific feature extraction and gradient computation instead of importing a smaller shared adapter.
- Several modules use `sys.path` manipulation to cross package boundaries instead of formal packaging.
