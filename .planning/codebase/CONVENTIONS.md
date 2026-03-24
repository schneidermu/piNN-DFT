# Conventions

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
