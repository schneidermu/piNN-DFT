# CLAUDE.md — piNN-DFT / lap_full_vxc

> CURRENT AGENT INSTRUCTIONS (2026-10-09). Scientific validity, reproducibility, and preservation of existing research outrank development speed. These instructions describe the four-task Laplacian research branch, NOT the historical NN-PBE training pipeline. The GSD-generated reference material below is legacy unless independently confirmed by current code.

## Mandatory agent behavior

1. **Inspect first**: check `git status`, current HEAD/branch, requested source files, actual CLI entry points, referenced checkpoint/configuration, frozen manifests, and relevant tests. Do not assume a historical command is current.
2. **Stay inside authorization**: do only the explicitly requested investigation or change. Do NOT autonomously run a training update, evaluation, GPU-intensive test, Slurm job, SCF, future-test benchmark, architecture change, parameter sweep, or experiment continuation.
3. **Fail closed**: STOP and report a mismatch if any required file/checkpoint, digest, optimizer state, scientific convention, dataset split, source revision, manifest, or result is missing or ambiguous. Never invent it or silently substitute a similar artifact.
4. **Protect prior work**: do not overwrite historical checkpoints, datasets, manifests, reports, receipts, plots, issue tickets, managed agent files, or another agent's uncommitted changes. No force pushes. For concurrent edits use an isolated branch/PR; recheck HEAD before write.
5. **Minimal diff**: reuse qualified code and numerical paths; do not "simplify" the operator, redesign the trainer, change physical math, change precision, or refactor unrelated files. Do not execute commands copied from legacy docs without validating them.
6. **Explicit experiment contract**: before executing an approved experiment, record precise scientific question, source commit, fixed samples, training/evaluation manifests, initial checkpoint tensor/file SHA, optimizer state, coefficients, update limit, evaluation checkpoints, resource budget, outputs, and stop gates.
7. **Check then report**: verify finite parameters, objective definitions, exact evaluation populations, provenance integrity, checkpoint/resume state, and appropriate focused tests. Report units, measured runtime/memory, per-task values, known unknowns, and what was not run. Never claim a test passed if it was not executed.

## Active code and data — verify, do not guess

| Concern | Canonical starting points |
| --- | --- |
| Tau-free Lap model | `train_models/NN_models_lap.py` (`pcPBELMLOptimizerV2Lap`) |
| Chemistry and mRKS objectives | `train_models/lap_training.py`; `train_models/lap_moo_training.py` |
| Full variational XC AO operator | `train_models/lap_operator.py`; `train_models/lap_operator_data.py` |
| Lap XC energy and derivatives | `train_models/lap_vxc.py`; `dft_functionals/PBE.py` |
| Native four-task AdamW | `train_models/lap_fixed_adamw.py`; `train_lap_microbatch.py` and experiment-specific wrappers |
| Chemistry variant policy | `train_models/lap_chemistry_sampling.py`; `relchem_joint_epoch_report.md` |
| Current evidence | `relchem_joint_epoch_report.md`; `relchem_joint_clean28_report.md`; `iid_adamw_t59_t90_report.md`; `iid_adamw_lr_stabilization_report.md` |

- Raw Lap model input uses nine columns; tau columns are **not** used by the tau-free model. Correlation and tied spin-exchange descriptors, PBE physical anchors and spin symmetry are scientifically required.
- Logical publication dataset SHA256: `61c221a19b9987717e69cac182ad545241f8807db4126c0949a99992e4c210ef`. Validate the actual immutable dataset manifest before use; this SHA is not permission to reconstruct missing datasets.
- `train_models/predopt_train.py`, `train_models/optuna_joint.py`, `train_models/replay_trial_19_bridge.py` and the older `test_models` scripts contain historical/legacy workflows or shared utilities. Do NOT treat their optimizer, epoch schedule, data layout or old H5 description as the current Lap protocol.

## Four independently required scientific objectives

| Objective | Frozen scientific evaluation | Interpretation |
| --- | --- | --- |
| `relchem` | 251 reaction identities | Minnesota relative chemical energies, AE17 excluded |
| `ae17` | 17 identities | Atomicization, separate from relchem |
| `exc` | all 90 mRKS systems | Integrated E_xc |
| `op` | all 90 mRKS systems | Complete weak-form variational AO XC operator |

- All four are independent scientific requirements. Never quietly merge relchem with AE17, remove one from a claimed joint experiment, or replace the full AO operator with local `v_rho`, a stencil potential, energy fitting, or a subsampled operator.
- **PERMANENT CHEMISTRY EVALUATION POLICY**: exactly **ONE fixed quadrature variant per chemical identity**, using the independent frozen evaluation manifest. **NEVER evaluate or average all eight variants** for any purpose, including final qualification. Available variants may be used only for explicitly approved stochastic training augmentation. Identity count does not multiply by number of grids.
- Preserve inherited chemical database/frequency factors from the existing qualified `batch_fchem`-based singleton loss. Do not double-weight or remove database weights. For full chemistry endpoints reuse the same selected reaction variant at every checkpoint.
- Full90 mRKS objectives use the same complete physical systems and same immutable targets. Preserve gauge, spin and overlap conventions. AO-operator matrices use AO values and derivatives in the complete weak-form operator; never approximate the operator with spatial finite differences.
- **Precision boundary**: preserve the qualified F64 learned descriptor/model-local derivative/partials and F64 AO assembly path; keep the qualified F32 PBE arithmetic and native F32 AdamW parameter state. Compute/aggregate task gradients in F64 and cast ONCE at the optimizer boundary. Do not "fix" numerical behavior by globally changing tensor dtypes or reverting local derivatives to F32.
- **Scientific eligibility**: every exact objective must be finite and **each of the four ratios to the same frozen reference checkpoint must be strictly < 1**. A good external validation metric alone NEVER establishes eligibility. Report all four objective values/ratios explicitly.
- Historical S5 nine-database `train_fchem` INCLUDES AE17 and aggregates online epoch errors. It is not interchangeable with current fixed-model relchem on eight databases. Compare historical runs only with explicitly matched metric definitions, datasets and optimizer-update counts.

## Clean28 external validation and holdout protection

- **Clean28** is the leakage-clean Diet-GMTKN55-30-derived **mean of 28 Diet-weighted absolute reaction errors**, kcal/mol, on frozen PBE0 densities using the qualified PBE0-D3(BJ) correction. It is **NOT canonical Full30 WTMAD-2**, and it is not an SCF result.
- Use the already qualified evaluator and exact same 28 identities, references, weights, density files, dispersion, and leakage exclusions. No Full30 selection, future holdout, Diet100, new SCF calculation, or density regeneration without a separate explicit instruction.
- Checkpoint selection should consider Clean28 AND all four scientific eligibility requirements. Save signed/absolute errors and per-reaction score contributions. Do not claim independent validation from multiple checkpoints of the same 28 reactions.
- Evidence only, **not optimization targets or defaults**: corrected P536 initial `Clean28=9.553190636`, IID t70 `8.629660230`, LR=3e-5/t80 `8.619694172`; the latter has relchem/t0 `1.034869578` and therefore fails scientific eligibility. Do not automatically resume any of these.

## Training, resuming, distributed work

- Native AdamW diagnostic reference used `LR=1e-4`, betas `(0.9,0.999)`, epsilon `1e-8`, weight decay `0.01`, and fixed task coefficients `relchem=0.017015480965588553`; `ae17=5.141254618347414e-05`; `exc=1.5094644512009712e-05`; `op=0.33597561607048215`. These are **experiment-specific recorded settings**, not automatic defaults for new research.
- The training variant sampler and fixed evaluation manifest are different protocols. Do not evaluate all variants, repeat a sampling draw, change the number of examples per update, introduce phases/curriculum, or reset AdamW moments when continuing a fixed experiment, unless expressly authorized.
- Resume requires byte-verified starting checkpoint plus model/buffers, optimizer moments and step counters, RNG, cursor, manifest and scientific hashes. Checkpoint filename alone is not evidence of identity.
- Distinguish updates, microbatches, identities, and epochs. In two-rank runs, a logger may sum step counts over ranks. Distributed operators/gradients require verified correct global task normalization. Two V100 GPUs may be more useful for independent replicas; do not assume DDP speedup.
- Stop on nonfinite gradients/parameters/state, dataset or hash mismatch, OOM, or the fixed step/time budget. No silent retries with changed precision, coefficient, grid, batch size or source code. Do not submit Slurm or launch 2xV100 without explicit user authorization.
- Use small appropriate focused tests, Ruff and compileall when relevant; distinguish CPU tests from actual GPU parity tests.

## Skills, issue tracker and GSD

- Preserve the Matt Pocock `## Agent skills` configuration and `docs/agents/*.md` once installed. Do not overwrite installer outputs or change its tracker's conventions. Existing tickets use `.scratch/<feature>/issues/NN.md` and must not be migrated without approval.
- When the user **explicitly invokes** Matt Pocock engineering skills, run their workflow directly; they are an exception to the general "run GSD first" requirement below. Other file-changing tasks follow the existing GSD instructions unless the user explicitly requests a bypass.
- Do not modify GSD-managed sections below by hand. If a GSD-generated STACK/PROJECT/ARCHITECTURE section conflicts with the current verified Lap code or the rules above, consider that snapshot historical and report the conflict rather than changing science to match it.

---

## Preserved GSD-managed historical snapshot

All managed material below is preserved **verbatim** for workflow compatibility. It describes older NN-PBE/evaluation-automation work and must not override current four-task `lap_full_vxc` scientific rules.

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
