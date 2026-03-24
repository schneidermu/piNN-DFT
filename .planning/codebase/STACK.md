# Stack

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
