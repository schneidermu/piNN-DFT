# Research: Evaluation Automation Architecture

## Proposed Component Boundaries

- `test_models` should gain a top-level experiment runner entrypoint that parses CLI args, creates the experiment folder, and coordinates branch execution.
- WTMAD-2 should be refactored into a reusable module or command layer that can stage inputs, submit jobs, wait for completion, and extract metrics without depending on manual checkpoint renaming.
- avRANE should be refactored the same way, with explicit staging, submission, completion checks, and result extraction.
- A shared experiment model should own manifest writing, log paths, copied artifacts, branch status, and final reporting.

## Data Flow

1. User passes checkpoint path and experiment name.
2. Runner creates experiment directory and copies the checkpoint there.
3. Runner materializes branch-specific configs and generated SLURM scripts under the experiment folder.
4. Each branch submits jobs and records job IDs and status files.
5. Runner waits or polls for branch completion.
6. Branch result collectors parse existing downstream outputs into normalized metrics.
7. Summary writer merges branch outcomes into final human-readable and machine-readable reports.

## Suggested Build Order

- First define experiment folder schema and branch interfaces.
- Next refactor WTMAD-2 into reusable internals because its current workflow is the most obviously script-driven.
- Then refactor avRANE using the same orchestration patterns.
- After both branches exist, add the unified runner, wait logic, and final reporting.
- Finish with documentation and operator checks.

## Architectural Principles

- Replace filename conventions with explicit metadata and paths.
- Keep branch-specific scientific logic inside branch modules; keep orchestration generic.
- Make the experiment folder the single source of truth for run state.
- Prefer append-only logs and status snapshots so partial failures stay inspectable.
