# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**piNN-DFT** trains neural networks to locally modify parameters of the Perdew-Burke-Ernzerhof (PBE) DFT functional, producing **NN-PBE-D3(BJ)** — a physics-informed functional that reduces thermochemical error by ~30% over PBE. The key idea: the NN outputs modified PBE constants (not a raw energy), so exact physical constraints are preserved by design.

## Common Commands

### Training Pipeline

```bash
# Step 1: Preprocess .h5 dataset files → pickled train/test splits (run from train_models/)
python train_models/prepare_data.py

# Step 2: Submit SLURM hyperparameter sweep jobs (trains all omega variants)
python train_models/calculations.py
```
Checkpoints save to `train_models/best_models/`, logs to `train_models/logs/`.
Training requires 2× V100 GPUs, ~10h per functional.

### Benchmarking (Diet-GMTKN55)

```bash
# From test_models/ — requires DietGMTKN55 repo files (InterfaceG16.py, GIF/, GoodSamples/)

# Generate geometry/job files
python InterfaceG16.py --Mode GE
python calculate_system_energies.py --Mode GE

# Submit energy calculation jobs
python calculate_system_energies.py --Mode CE --Functional NN_PBE_067
python calculate_system_energies.py --Mode CE --Functional PBE

# Analyze results and generate CSV
python InterfaceG16.py --Functional NN_PBE_067 > Results/NN_PBE.txt
cd Results/ && python txt_to_csv.py
```

### Other Analysis

```bash
# Enhancement factor plots (Fig. 4) — generates Results/exc.npy
python test_models/plot_exc.py

# Molecular density accuracy (avRANE) — run from test_models/
python run_molden.py --Functional NN_PBE_067
# Then from den_mol_or/:
python calcden.py && python dniad

# Atomic density accuracy (MaxNE) — from denrho/dtestin/NN_PBE_067/:
./swfn
# Then from denrho/:
./krms NN_PBE_067 CCSD
```

### Dependencies

```bash
pip install -r test_models/requirements.txt   # for inference/benchmarking
pip install -r train_models/requirements.txt  # additionally for training
```

## Architecture

### Module Layout

```
dft_functionals/   — PyTorch implementations of PBE and SVWN3 (LDA); shared constants
train_models/      — data prep, NN architectures, training loop, SLURM launcher
test_models/       — pre-trained model loading, PySCF integration, benchmarking scripts
den_mol_or/        — molecular electron density accuracy (avRANE)
denrho/            — atomic electron density accuracy (MaxNE)
MN_dataset/        — dataset download instructions and format docs
```

### Data Flow

```
Raw .h5 files (MN dataset)
  → prepare_data.py              (stratified 80/20 split + grid augmentation)
  → train_models/checkpoints/data_{train,test}.pickle
  → predopt_train.py + DDP       (core training loop)
  → best_models/state_dict_{omega}.pth
  → test_models/DFT/functional.py (NN_FUNCTIONAL loads checkpoint)
  → script.py / PySCF integration → benchmarking
```

### Key Files

**`train_models/NN_models.py`** — All NN architectures:
- `ResBlock` — residual block with LayerNorm + GELU + dropout
- `MLOptimizer` — base class; computes 7-dim MGGA density descriptors (ρ^(1/3), reduced gradients sα/sβ/s_total, normalized τα/τβ); uses dm21-like sigmoid activation outputting [0, 2]
- `pcPBEMLOptimizer` — primary model (6 layers, 32 hidden); enforces PBE constraints (all-sigma-zero, all-sigma-inf, all-rho-inf via hard-coded boundary conditions); outputs 21 modified PBE constants
- `pcPBELMLOptimizer` / `V2` — extended variants with Laplacian (9-dim) or zeta descriptors
- `pcPBEstar` / `pcPBEdoublestar` — ablation variants (no constraints / Nagai et al. approach)

**`train_models/predopt_train.py`** — Training loop: DDP multi-GPU, LinearLR warmup → CosineAnnealingLR, log-scaled density/gradient/tau for numerical stability, D3(BJ) dispersion integration.

**`train_models/dataset.py`** — Stratified splitting with hardcoded molecule overrides; `group_and_augment_reactions()` creates multiple augmentations per reaction using different grid types.

**`dft_functionals/PBE.py`** — Pure PyTorch PBE; functions `rs_z_calc`, `xs_xt_calc`, `f_zeta`, `g_aux` are reused during training to evaluate the energy given NN-modified constants.

**`dft_functionals/constants.py`** — Defines `true_constants_PBE` (27 params), `true_constants_SVWN3` (21 params), descriptor index mappings, and spin-scaling multipliers.

**`test_models/DFT/functional.py`** — `NN_FUNCTIONAL` class: loads `.pth` checkpoint, wraps `eval_xc()` for PySCF DFT interface, supports all omega variants (0, 0.067, 0.18, 0.33, 0.50, 0.67, 0.82, 0.93, 0.99).

**`test_models/DFT/numint.py`** — `RKS_with_Laplacian` / `UKS_with_Laplacian`: extended PySCF numerical integration for Laplacian-based models.

### H5 Data Format

Each `.h5` file contains one molecule with two datasets:
- `ener` (3 floats): kinetic+potential energy, HF exchange, total PBE0 energy
- `grid` (N×12): columns are [x, y, z, weight, ρα, ρβ, σαα, σαβ, σββ, τα, τβ, local HF exchange]

### Pre-trained Checkpoints

Located in `test_models/DFT/checkpoints/NN_PBE/`:
- `state_dict_0.067.pth` — primary NN-PBE model
- `state_dict_star_0.067.pth` — ablation (no constraints)
- `state_dict_star_star_0.18.pth` — ablation (Nagai et al. approach)

The omega (Ω) parameter controls the balance between exchange and correlation fitting; Chebyshev polynomial roots are used as the sweep grid in `calculations.py`.

<!-- GSD:project-start source:PROJECT.md -->
## Project

**piNN-DFT Evaluation Automation**

This project reshapes the existing `test_models/` evaluation workflow in `piNN-DFT` into a single-entry experiment pipeline for trained checkpoints. It is for maintainers of this repository who currently have to rename checkpoints, place them in special locations, and manually run multiple scripts to evaluate thermochemical and density-accuracy metrics.

The intended result is one command that accepts a checkpoint path and experiment name, stages an experiment folder, submits the required SLURM work for WTMAD-2 and avRANE, waits for completion, and writes a single summary with metrics, failures or skips, logs, and the tested model artifact.

**Core Value:** A newly trained checkpoint can be evaluated reproducibly with one command and one experiment folder, without manual file shuffling or piecing results together by hand.

### Constraints

- **Execution backend**: Keep SLURM-based execution � existing evaluation scripts and cluster workflows already depend on it
- **Brownfield compatibility**: Build on top of the current repository and scripts � this is an internal workflow improvement, not a fresh greenfield evaluator
- **Scope**: v1 covers WTMAD-2 and avRANE only � atomic density metrics are deferred
- **Artifacts**: Every run must live under one experiment folder � results need to be easy to inspect, compare, and archive
- **Failure handling**: Partial summaries are required � a failed branch must not erase successful metrics from the same experiment
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Overview
- Primary language: Python 3.9+ per `README.md`.
- Main numerical and ML stack: PyTorch, NumPy, SciPy, pandas, h5py.
- Chemistry stack for inference and analysis: PySCF, dftd3, pylibxc-style workflows, Multiwfn-based post-processing.
- Project shape: research repository with training code, benchmark/inference scripts, reference datasets, and generated experiment artifacts stored together.
## Runtime Environments
- Training environment is defined in `train_models/requirements.txt`.
- Inference and benchmarking environment is defined in `test_models/requirements.txt`.
- A local `venv/` directory is present in the repository root, which suggests environment artifacts may live alongside source.
- Several workflows assume Linux/HPC execution even though the repository is portable enough to inspect on Windows.
## Core Dependencies
### Training
- `torch==2.4.1` in `train_models/requirements.txt`.
- `mlflow>=2.10.0` in `train_models/requirements.txt` for experiment tracking.
- `python-dotenv>=0.19.0` in `train_models/requirements.txt` for optional MLflow configuration.
- `scikit-learn==1.5.0` in `train_models/requirements.txt` for stratified splitting.
- `matplotlib==3.9.0`, `sympy==1.12`, `tqdm==4.65.0`.
### Inference and Benchmarking
- `torch==2.3.0` in `test_models/requirements.txt`.
- `pyscf==2.7.0` and `dftd3==1.2.1` in `test_models/requirements.txt`.
- `matplotlib==3.10.1`, `seaborn==0.13.2`, `h5py==3.10.0`, `pandas==2.2.3`.
- `PyYAML<5.1` and `unicodeit==0.7.5`.
## First-Party Packages
- `dft_functionals/`: shared functional math and constants, especially `dft_functionals/PBE.py` and `dft_functionals/constants.py`.
- `train_models/`: dataset preparation, neural architectures, distributed training, Optuna-style tuning, diagnostics, and generated outputs.
- `test_models/`: checkpoint loading, PySCF integration, benchmark runners, plotting, and convergence checks.
- `den_mol_or/`: molecular density accuracy workflow.
- `denrho/`: atomic density accuracy workflow.
- `MN_dataset/`: CSV metadata and dataset download instructions.
## Data and Artifact Formats
- Raw scientific input data is expected as `.h5` files under `train_models/data/`.
- Processed training artifacts are stored as `.pickle` files under `train_models/checkpoints/`.
- Model checkpoints are `.pth` files in `train_models/best_models/` and `test_models/DFT/checkpoints/`.
- Hyperparameter studies and analysis outputs are stored as `.db`, `.json`, `.md`, `.svg`, `.html`, and `.png` files under `train_models/optuna_*`.
- Benchmark outputs are plain-text energy lists and CSV summaries under `test_models/Results/`.
## Platform and Orchestration Assumptions
- Distributed training uses `torchrun` and NCCL in `train_models/predopt_train.py`.
- Cluster submission is baked into `train_models/calculations.py` and `test_models/calculate_system_energies.py` via SLURM scripts and `sbatch`.
- External tools are assumed rather than provisioned by code: DietGMTKN55 assets, Multiwfn, and reference density files.
## Configuration Style
- There is no central package manager or project-wide config file at the root.
- Dependency management is split by sub-workflow rather than unified.
- Environment behavior is controlled mostly by script arguments and a few environment variables such as `ENABLE_MLFLOW` and `MLFLOW_TRACKING_URI` in `train_models/predopt_train.py`.
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## General Style
- The codebase is mostly plain Python scripts and modules with selective use of docstrings.
- Scientific variables use domain naming such as `rho`, `sigma`, `tau`, `vrho`, `omega`, and `zeta`.
- Most code is written in snake_case for functions and variables.
- Class names follow CapWords, for example `NN_FUNCTIONAL`, `EarlyStopper`, and `EpochSampledAugmentedDataset`.
## Import Patterns
- Imports are often grouped loosely rather than strictly formatted.
- Several modules modify `sys.path` directly to import sibling or parent packages, notably `train_models/dataset.py` and `test_models/DFT/functional.py`.
- Relative imports are used inside package-like directories, but not consistently across the repo.
## Script-First Conventions
- Many files are intended to be run directly and include `if __name__ == "__main__":`.
- CLI parsing is split between `argparse` in newer training and Optparse `OptionParser` in older benchmarking scripts.
- Working-directory assumptions are common; many scripts expect execution from within their own folder.
## Data Structures
- Reaction samples are dictionaries populated with tensors and metadata fields such as `Database`, `Components`, `Weights`, and `PBE_local_energies`.
- Descriptor ordering is standardized through index constants in `dft_functionals/constants.py`.
- Constraint-aware models output a fixed-width tensor of constants whose indices have semantic meaning.
## Numerical and ML Conventions
- Torch tensors are used throughout the learning path, with NumPy used for CSV/HDF5 preprocessing and post-analysis.
- Explicit epsilons and threshold guards are common in numerical routines, for example `EPS_RHO`, `EPS_SIGMA`, and local `eps_add` values.
- Deterministic training is intentionally enabled in `train_models/utils.py:set_random_seed`.
- Training code uses DDP, `DistributedSampler`, custom collate functions, and explicit seeding utilities.
## Logging and Diagnostics
- Training scripts use `print(...)` extensively for progress and diagnostic output.
- Optional experiment tracking uses MLflow in `train_models/predopt_train.py`.
- Debug helpers like `catch_nan(...)` and `save_tensors(...)` write tensors into local `log/` directories.
## Testing Conventions
- The most formal tests are in `train_models/test.py` and are written with `pytest`.
- Other files with `test` in the name, such as `train_models/test_new.py` and `test_models/test_functionals_convergence.py`, are diagnostic or exploratory scripts rather than automated test modules.
- Assertions are also used in utility code for internal invariants, such as optimizer parameter grouping in `train_models/utils.py`.
## Documentation Conventions
- README files exist at the root and in major workflow directories.
- Comments are strongest where numerical formulas or physics constraints need explanation.
- There is no generated API documentation or package reference site.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## Overview
- The repository is organized by workflow rather than by a reusable application package.
- Core mathematical definitions live in shared modules, while training and evaluation are driven by standalone scripts.
- The main architectural seam is: learned local constants -> functional energy evaluation -> chemistry workflow integration.
## Main Data Flow
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
- `test_models/DFT/functional.py` adapts trained models for PySCF�s `eval_xc` contract.
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
<!-- GSD:architecture-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd:quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd:debug` for investigation and bug fixing
- `/gsd:execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd:profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
