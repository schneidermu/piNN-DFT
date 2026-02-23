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
