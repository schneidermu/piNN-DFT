# Concerns

## High-Impact Risks

- Reproducibility depends on manual data downloads and manual placement of external assets such as DietGMTKN55 files and reference densities.
- Many workflows assume SLURM, Multiwfn, and Linux-style execution, which makes local onboarding and cross-platform execution fragile.
- The repository currently contains generated artifacts and a checked-in `venv/`, which increases noise and makes source-of-truth boundaries less clear.

## Codebase Fragility

- Several scripts depend on relative paths like `../MN_dataset/...` and on being run from a specific directory.
- `sys.path` mutation in files such as `train_models/dataset.py` and `test_models/DFT/functional.py` is convenient but brittle.
- Benchmark submission code in `test_models/calculate_system_energies.py` uses `os.system(...)` and broad `try/except` blocks, which can hide failures.
- Training orchestration in `train_models/calculations.py` embeds scheduler settings directly in string templates, including email and cluster-specific constraints.

## Environment Drift

- `train_models/requirements.txt` and `test_models/requirements.txt` pin different Torch and HDF5 versions, so reproducing both workflows in one environment may be awkward.
- `pytest` is used by `train_models/test.py` but is not declared in the training requirements file.
- A local `venv/` directory in the repo root suggests environment state may leak into version control or tooling behavior.

## Data and Artifact Hygiene

- Large experiment outputs, databases, images, HTML, and checkpoint files live under `train_models/` and appear alongside source files.
- `git status` already shows many untracked experiment outputs under `train_models/`, which can complicate review and change isolation.
- The repo mixes curated pretrained checkpoints with ad hoc generated outputs, making it harder to distinguish canonical artifacts from work-in-progress results.

## Testing Gaps

- The only clear automated suite is the constraint-focused pytest module in `train_models/test.py`.
- There is no CI, no smoke test for preprocessing, and no automated benchmark/inference regression harness.
- HPC submission paths, PySCF integration, and density-analysis pipelines are not validated in an automated way.

## Maintainability Issues

- The largest training script, `train_models/predopt_train.py`, is monolithic and combines CLI parsing, DDP setup, data loading, training, validation, plotting, checkpointing, and MLflow integration.
- Older scripts still use `optparse`, while newer ones use `argparse`, which reflects incremental evolution rather than a unified interface layer.
- Source and experiment history are not clearly separated, so future cleanup or packaging work will require careful triage.

## Operational Notes

- No obvious API secrets were found in the source inspected, but `train_models/calculations.py` includes a hardcoded email address in the SLURM template.
- Because the workflow depends on external binaries and copied-in benchmark assets, new contributors may hit setup failures before they reach any scientific code.
- This is a strong research codebase, but it is not yet arranged like a reproducible productized package.
