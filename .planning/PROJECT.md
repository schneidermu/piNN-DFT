# piNN-DFT Evaluation Automation

## What This Is

This project reshapes the existing `test_models/` evaluation workflow in `piNN-DFT` into a single-entry experiment pipeline for trained checkpoints. It is for maintainers of this repository who currently have to rename checkpoints, place them in special locations, and manually run multiple scripts to evaluate thermochemical and density-accuracy metrics.

The intended result is one command that accepts a checkpoint path and experiment name, stages an experiment folder, submits the required SLURM work for WTMAD-2 and avRANE, waits for completion, and writes a single summary with metrics, failures or skips, logs, and the tested model artifact.

## Core Value

A newly trained checkpoint can be evaluated reproducibly with one command and one experiment folder, without manual file shuffling or piecing results together by hand.

## Requirements

### Validated

- ? The codebase can train and store neural density-functional checkpoints via the `train_models/` workflow — existing
- ? The codebase can run WTMAD-2 style benchmark calculations through `test_models/` scripts plus DietGMTKN55 assets — existing
- ? The codebase can compute molecular density accuracy through the current avRANE workflow using PySCF, Multiwfn, and downstream processing scripts — existing
- ? The codebase already relies on SLURM job submission for benchmark and density workflows — existing

### Active

- [ ] A single CLI entrypoint evaluates one checkpoint end-to-end for WTMAD-2 and avRANE
- [ ] The evaluation pipeline stages all outputs under a named experiment directory
- [ ] The pipeline submits required SLURM jobs automatically and waits for completion
- [ ] The pipeline produces a final summary even if some evaluation branches fail or are skipped
- [ ] The experiment directory contains logs, copied model artifact, metric summaries, and downstream result references or copies

### Out of Scope

- Atomic density evaluation such as MaxNE or max RMSD in v1 — intentionally deferred so the first automation target stays focused on WTMAD-2 and avRANE
- Replacing SLURM with a different execution backend — current workflows already depend on cluster submission and that is not the immediate bottleneck
- Full reliability engineering for every failure mode — the first version should optimize for the normal successful path while still reporting partial results

## Context

The current brownfield codebase already contains working but fragmented benchmark and density-analysis workflows in `test_models/`, `den_mol_or/`, and `denrho/`. Evaluation today depends on manual checkpoint placement and naming, manually running several scripts, and collecting outputs from multiple directories. That makes iteration on new checkpoints slower than it should be.

The work is specifically centered on improving `test_models/` and its neighboring evaluation scripts, not redesigning the training pipeline. The existing repository structure, codebase map, and current benchmark scripts should be treated as the starting point and progressively reshaped into a cleaner orchestration layer.

Research and requirements for this project should focus on experiment orchestration, artifact layout, SLURM job lifecycle handling, and reducing filename- and path-dependent behavior in the current evaluation flow.

## Constraints

- **Execution backend**: Keep SLURM-based execution — existing evaluation scripts and cluster workflows already depend on it
- **Brownfield compatibility**: Build on top of the current repository and scripts — this is an internal workflow improvement, not a fresh greenfield evaluator
- **Scope**: v1 covers WTMAD-2 and avRANE only — atomic density metrics are deferred
- **Artifacts**: Every run must live under one experiment folder — results need to be easy to inspect, compare, and archive
- **Failure handling**: Partial summaries are required — a failed branch must not erase successful metrics from the same experiment

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Focus v1 on WTMAD-2 and avRANE only | These are the metrics the user wants automated now, and narrowing scope reduces setup churn | — Pending |
| Keep SLURM as the orchestration backend | Current evaluation already depends on cluster jobs, so replacing the scheduler would add unnecessary scope | — Pending |
| Reshape the underlying `test_models/` scripts instead of only wrapping them | The current pain is partly caused by filename-dependent internals, so a shallow wrapper would leave too much fragility underneath | — Pending |
| Produce a partial final summary on failure | Evaluation branches are long-running and expensive, so successful outputs should still be retained when one branch fails | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `$gsd-transition`):
1. Requirements invalidated? > Move to Out of Scope with reason
2. Requirements validated? > Move to Validated with phase reference
3. New requirements emerged? > Add to Active
4. Decisions to log? > Add to Key Decisions
5. "What This Is" still accurate? > Update if drifted

**After each milestone** (via `$gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-03-24 after initialization*
