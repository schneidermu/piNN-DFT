# Structure

## Top-Level Layout

- `README.md`: project overview, setup, and workflow summary.
- `CLAUDE.md`: repository-specific operational notes for coding agents.
- `dft_functionals/`: shared functional implementations and constants.
- `train_models/`: training pipeline, Optuna studies, diagnostics, and generated outputs.
- `test_models/`: benchmark and inference scripts using pretrained checkpoints.
- `MN_dataset/`: dataset metadata CSV files and download instructions.
- `den_mol_or/`: molecular density analysis scripts and templates.
- `denrho/`: atomic density analysis scripts and helper content.
- `venv/`: local virtual environment currently present in the repository.

## Source-Oriented Directories

### `dft_functionals/`

- `dft_functionals/PBE.py`: main PBE implementation reused during training and inference.
- `dft_functionals/SVWN3.py`: X-alpha and SVWN3-related functional logic.
- `dft_functionals/constants.py`: shared constants and descriptor index definitions.
- `dft_functionals/__init__.py`: convenience exports.

### `train_models/`

- `train_models/NN_models.py`: primary neural architectures.
- `train_models/predopt_train.py`: main training entry point.
- `train_models/dataset.py`: reaction assembly and augmentation.
- `train_models/prepare_data.py`: preprocessing and pickle generation.
- `train_models/reaction_energy_calculation.py`: computes reaction/local energies.
- `train_models/utils.py`: batching, seeding, optimizer setup, sigma fixes.
- `train_models/calculations.py`: generates and submits SLURM training jobs.
- `train_models/test.py`: pytest-based model constraint suite.

## Analysis and Artifact Subtrees

- `train_models/best_models/`: saved checkpoints.
- `train_models/dispersions/`: dispersion data used during training.
- `train_models/h5_vrho/`: Vxc diagnostic datasets.
- `train_models/optuna_joint_runs/`: per-run trial JSON artifacts and checkpoints.
- `train_models/optuna_joint_analysis/`: generated reports and SVG/HTML visuals.

### `test_models/`

- `test_models/DFT/`: checkpoint loaders, functional wrappers, numerical integration helpers.
- `test_models/Results/`: benchmark outputs and conversion scripts.
- `test_models/molden/`: molecule geometry inputs for density workflows.
- `test_models/calculate_system_energies.py`: benchmark job script generation and submission.
- `test_models/script.py`: SCF and energy calculation entry point.
- `test_models/test_functionals_convergence.py`: manual convergence check script.

### Density Accuracy Directories

- `den_mol_or/calcden.py` and `den_mol_or/dniad`: molecular density processing.
- `den_mol_or/geoms/`: geometry inputs.
- `den_mol_or/tmpls/`: calculation templates.
- `denrho/krms`: atomic density comparison executable/script.
- `denrho/content/`: Multiwfn helper templates and scripts.

## Naming Conventions

- Script names are task-oriented: `prepare_data.py`, `plot_exc.py`, `calculate_system_energies.py`.
- Neural checkpoints encode hyperparameters and metrics directly in filenames under `train_models/best_models/`.
- Optuna trial data is stored in numbered JSON files under `train_models/optuna_joint_runs/.../trials/`.
- Functional identifiers are embedded in filenames and CLI flags, for example `NN_PBE_067` or `NN_PBE_star_star`.

## Structural Characteristics

- Source files, datasets, outputs, checkpoints, and visualizations coexist in the same directories.
- There is no `src/` package boundary or installable package manifest.
- Root-level organization is clear by scientific workflow, but cleanliness within subtrees is mixed because generated artifacts are checked in.
