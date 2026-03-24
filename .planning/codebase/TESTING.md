# Testing

## Overview

- Automated testing exists, but it is narrow and centered on model-constraint behavior rather than end-to-end workflow validation.
- Most benchmark and density-analysis verification is manual or script-driven.

## Formal Test Surface

- `train_models/test.py` is the main automated test module and is designed for `pytest`.
- It checks output shape, exact-constraint preservation, pass-through constants, disabled-feature baselines, and spin symmetry for `pcPBELMLOptimizerV2`.
- The tests use deterministic synthetic inputs instead of external datasets or SCF jobs, which keeps them lightweight and focused.

## Informal or Diagnostic Checks

- `train_models/test_new.py` is a standalone diagnostic script comparing `vrho` behavior against reference and libxc-based baselines.
- `test_models/test_functionals_convergence.py` is a command-line convergence probe for functionals in a PySCF workflow.
- `train_models/reaction_energy_calculation.py` includes helper functions with names like `test_energy_PBE`, but those are not a pytest suite.

## What Is Not Covered

- No automated test covers the full preprocessing -> training -> checkpoint -> PySCF inference loop.
- No test suite validates SLURM script generation or cluster submission behavior.
- No automated checks verify the external DietGMTKN55 integration, Multiwfn workflow, or density-analysis binaries.
- No CI config or test runner automation is present at the repository root.

## Test Dependencies

- `pytest` is imported directly in `train_models/test.py`, but it is not listed in `train_models/requirements.txt`.
- Running chemistry-side scripts additionally requires heavy scientific dependencies like PySCF and possibly external executables.
- Some tests assume imports from neighboring modules rather than an installed package.

## Quality Signals

- Constraint tests are a strong signal that the most important scientific invariants are treated seriously.
- The presence of exploratory diagnostics such as `train_models/test_new.py` suggests ongoing scientific iteration outside a hardened QA pipeline.
- Because generated artifacts are checked in, regression validation likely relies partly on human comparison of plots, metrics, and saved outputs.

## Suggested Reading for Test Context

- `train_models/test.py` for the most reliable current expectations.
- `train_models/predopt_train.py` for what would need integration coverage.
- `test_models/README.md` for the manual benchmarking workflow that currently stands in for automated end-to-end testing.
