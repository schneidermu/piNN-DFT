# Research: Evaluation Automation Pitfalls

## Current Fragility To Avoid

- Hardcoded checkpoint names and locations create silent coupling between scripts.
- Working-directory assumptions make orchestration brittle when called from a different folder.
- Script boundaries are currently shaped around manual operator behavior rather than explicit interfaces.
- External assets such as DietGMTKN55 and Multiwfn can fail independently and should not collapse the whole experiment summary.

## Likely Failure Modes

- SLURM submission succeeds for one branch and fails for another.
- Downstream result files exist but are incomplete or stale from previous runs.
- avRANE post-processing may fail after SCF jobs complete, leaving partial artifacts.
- Benchmark scripts may still assume legacy checkpoint naming unless refactored underneath.

## Prevention Strategies

- Record all resolved input and output paths in the experiment manifest.
- Namespace every run under its own experiment folder to avoid stale file ambiguity.
- Separate branch status values such as `pending`, `running`, `complete`, `failed`, and `skipped`.
- Normalize final summary generation so failure in one branch still preserves successful metrics from the other.
- Refactor existing scripts to accept explicit checkpoint and output locations instead of implicit repo-global paths.

## Phase Mapping

- Path and artifact contract issues should be solved first in Phase 1.
- WTMAD-2 and avRANE internal reshaping belong in Phases 2 and 3.
- Unified status handling and partial-failure reporting belong in Phases 4 and 5.
