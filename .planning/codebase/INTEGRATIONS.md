# Integrations

## Overview

- This repository is mostly self-contained mathematically, but operationally it depends on several external scientific tools, data sources, and cluster services.
- Most integrations are file-system or process based rather than API based.

## External Data Sources

- The MN training dataset is downloaded manually from the OneDrive link documented in `MN_dataset/README.md`.
- Reaction metadata and reference energies are read from `MN_dataset/total_dataframe_sorted_final.csv` and `MN_dataset/Reference_data.csv`.
- Additional benchmark inputs come from the external DietGMTKN55 repository, which must be copied into `test_models/` per `test_models/README.md`.
- Reference CCSD densities are required for density-accuracy workflows under `den_mol_or/` and `denrho/`.

## HPC and Process Integrations

- `train_models/calculations.py` generates SLURM scripts and calls `sbatch` through `subprocess.check_output(...)`.
- `test_models/calculate_system_energies.py` creates and submits per-system SLURM scripts via `os.system("sbatch ...")`.
- `train_models/predopt_train.py` expects a distributed launch environment with `LOCAL_RANK`, NCCL, and multiple GPUs.

## Chemistry and Numerical Libraries

- `test_models/DFT/functional.py` integrates custom neural functionals into PySCF through `eval_xc`.
- `test_models/requirements.txt` includes `pyscf` and `dftd3`, which are needed for SCF and dispersion-corrected evaluation.
- `train_models/test_new.py` uses `pylibxc` for reference `vrho` comparisons.
- `dft_functionals/PBE.py` and `dft_functionals/SVWN3.py` act as local implementations that other modules call directly.

## External Executables and Manual Tooling

- Multiwfn is an expected dependency for density post-processing in `test_models/get_molden.py`, `den_mol_or/`, and `denrho/content/swfn`.
- `den_mol_or/dniad` and `denrho/krms` are repository-local executable scripts or binaries used in downstream analysis.
- The workflow described in `CLAUDE.md` assumes shell execution in a Unix-like environment for `./swfn`, `./krms`, and `torchrun`.

## Experiment Tracking

- MLflow is optional but integrated in `train_models/predopt_train.py`.
- Tracking URI and enablement are configured through environment variables loaded with `dotenv`.
- Artifacts logged include metrics, plots, and checkpoint-related outputs.

## Path-Based Coupling

- `train_models/dataset.py` and related scripts depend on relative paths like `../MN_dataset/Reference_data.csv`.
- `test_models/DFT/functional.py` mutates `sys.path` to import from the repository root.
- `train_models/dataset.py` also inserts the parent directory into `sys.path` so local code can import `dft_functionals`.
- These integrations are simple but fragile if scripts are run from unexpected working directories.

## Missing or Implicit Integrations

- No package installation metadata exists for importing this repo as a library.
- No CI service, container setup, or reproducible environment lockfile is present.
- No secret-managed service integrations are visible; the main operational dependencies are scientific software, datasets, and scheduler access.
