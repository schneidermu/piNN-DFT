# Research: Evaluation Automation Stack

## Current Stack

- Primary implementation language is Python.
- Existing evaluation logic already depends on `torch`, `pyscf`, `dftd3`, `h5py`, `numpy`, and `pandas` through `test_models/requirements.txt`.
- WTMAD-2 benchmarking also depends on copied-in DietGMTKN55 assets described in `test_models/README.md`.
- avRANE depends on PySCF job execution, Multiwfn-based post-processing, and scripts in `den_mol_or/`.
- Job execution is SLURM-based today through scripts such as `test_models/calculate_system_energies.py` and `test_models/run_molden.py`.

## Fit For The Planned Work

- Python is already the right orchestration layer because the current benchmark and density code is Python-first.
- SLURM remains the correct execution backend for v1 because it is already how long-running evaluation jobs are launched.
- The new runner should prefer structured JSON or text status files in the experiment directory over implicit success-by-filename checks.
- Shared experiment metadata should be stored once and consumed by each evaluation branch so checkpoint naming stops being the coordination mechanism.

## Recommended Technical Direction

- Build a reusable evaluation package or module under `test_models/` rather than another one-off shell wrapper.
- Separate reusable library code from CLI entrypoints so WTMAD-2 and avRANE can be run individually or together.
- Standardize experiment manifests, job records, summary files, and artifact paths in one location.
- Use one experiment folder per run containing copied checkpoint, rendered SLURM scripts, logs, intermediate status, and final summaries.

## Compatibility Notes

- Keep compatibility with current DietGMTKN55 and Multiwfn-dependent workflows instead of replacing them immediately.
- Do not assume internet access or dynamically downloaded dependencies at runtime.
- Treat external dependencies as optional branches that can fail independently and still be summarized.
