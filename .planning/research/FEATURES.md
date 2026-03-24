# Research: Evaluation Automation Features

## Table Stakes For This Internal Workflow

- One CLI command starts the full evaluation flow for a checkpoint.
- Checkpoint path is passed explicitly instead of relying on hardcoded filename placement.
- Experiment name selects a dedicated output folder.
- WTMAD-2 branch can be launched, tracked, and summarized from the same entrypoint.
- avRANE branch can be launched, tracked, and summarized from the same entrypoint.
- Final reports include metrics, failures, skips, and artifact locations.

## Strongly Recommended Workflow Features

- Copy the evaluated checkpoint into the experiment directory for provenance.
- Persist a run manifest with CLI args, resolved paths, and timestamps.
- Write one machine-readable summary plus one human-readable report.
- Store SLURM submission metadata and job IDs so post-run debugging is straightforward.
- Keep per-branch logs and status files separate but linked from the top-level summary.

## Nice-To-Have Follow-Ups

- Resume or rehydrate an interrupted experiment.
- Retry failed branches without re-running successful ones.
- Optional support for MaxNE or other density metrics later.
- Comparison utilities across multiple experiment folders.

## Feature Priorities For v1

### Must Have

- Explicit checkpoint intake
- Single experiment directory
- WTMAD-2 execution and summary
- avRANE execution and summary
- Partial final summary on branch failure

### Should Have

- Stable folder layout
- Reusable internals instead of one-off wrappers
- Clear error summaries with pointers to branch logs

### Defer

- Atomic density metrics
- Scheduler abstraction beyond SLURM
- Full resumability and retries
