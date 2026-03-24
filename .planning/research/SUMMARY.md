# Research Summary: piNN-DFT Evaluation Automation

## Domain

Internal evaluation automation for ML-DFT checkpoints in an HPC-first scientific Python repository.

## Key Findings

- The existing repository already has the scientific pieces needed for WTMAD-2 and avRANE, but they are split across scripts that assume manual checkpoint placement, manual execution, and manual result collation.
- SLURM should remain the execution backend for v1 because the current workflows already depend on it and replacing it would add scope without solving the core pain.
- The most important architectural shift is to make the experiment directory, not the checkpoint filename, the source of truth for run state and outputs.
- The most important product behavior is a final summary that survives partial branch failure and still preserves successful metrics.

## Recommended v1 Shape

- A single Python CLI entrypoint under `test_models/`
- Shared experiment manifest and artifact layout
- Reusable WTMAD-2 branch module
- Reusable avRANE branch module
- Unified status, logging, and summary writers

## Watch Outs

- Legacy scripts may still depend on hardcoded paths after the new runner exists unless they are reshaped underneath.
- External tools and copied benchmark assets should be treated as explicit dependencies with branch-level failure reporting.
- Partial success must be a first-class outcome, not an afterthought.
