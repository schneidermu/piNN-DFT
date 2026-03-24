# Phase 1: Experiment Contract Foundation - Context

**Gathered:** 2026-03-24
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase defines the shared contract for evaluation runs: CLI inputs, experiment folder structure, staged model identity, and branch execution/reporting expectations. It does not yet fully implement the WTMAD-2 or avRANE branches themselves; it establishes the run model those branches must follow.

</domain>

<decisions>
## Implementation Decisions

### CLI contract
- **D-01:** The top-level evaluator CLI should take only the checkpoint path and experiment name as primary inputs.
- **D-02:** The CLI should also support a smoke-test mode for reduced-scope validation runs.

### Smoke mode
- **D-03:** Smoke mode should still submit real work, but only for a minimal subset rather than the full evaluation workload.
- **D-04:** For avRANE smoke runs, only `H2` and `N2` should be evaluated.
- **D-05:** For WTMAD-2 smoke runs, only `BH76-5` and `SIE4X4` should be evaluated.

### Experiment workspace
- **D-06:** Each experiment should use nested subfolders rather than a flat directory.
- **D-07:** The preferred top-level structure is `input/`, `jobs/`, `outputs/`, `reports/`, and `logs/`.

### Model identity
- **D-08:** Downstream evaluation should use a generated functional name instead of relying on the raw checkpoint filename directly.
- **D-09:** The generated functional name should be derived from both the experiment name and the checkpoint filename.

### Branch lifecycle
- **D-10:** WTMAD-2 and avRANE branches should run in parallel.
- **D-11:** A branch should be marked failed when it raises an error.
- **D-12:** If one branch fails and another succeeds, the top-level experiment status should be `failed`, while preserving all successful branch outputs.

### Reporting
- **D-13:** Final reporting should include full metrics tables rather than only condensed summaries.

### the agent's Discretion
- Exact CLI flag names beyond the required inputs and smoke mode
- Exact generated functional-name formatting rules and sanitization details
- Exact file names used within `input/`, `jobs/`, `outputs/`, `reports/`, and `logs/`
- Exact polling cadence and status-file format for branch waiting

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project scope and requirements
- `.planning/PROJECT.md` — Project goal, constraints, and high-level decisions for the evaluation automation effort
- `.planning/REQUIREMENTS.md` — v1 requirements that Phase 1 must support, especially `EVAL-01`, `EVAL-02`, `EVAL-03`, and `EXPR-01`
- `.planning/ROADMAP.md` — Phase 1 goal, success criteria, and plan breakdown anchor

### Existing evaluation code
- `test_models/calculate_system_energies.py` — Current WTMAD-2 SLURM job generation/submission flow and legacy functional-name assumptions
- `test_models/script.py` — Current per-system benchmark execution and `Results/` output behavior
- `test_models/run_molden.py` — Current avRANE SLURM submission pattern for molecule and atom density jobs
- `test_models/get_molden.py` — Current avRANE density generation flow, path assumptions, and downstream artifact creation
- `test_models/README.md` — Existing benchmark and avRANE operator workflow, external dependencies, and manual steps

### Codebase guidance
- `.planning/codebase/ARCHITECTURE.md` — Brownfield architecture and where evaluation code sits in the repo
- `.planning/codebase/STACK.md` — Runtime stack and external tool assumptions relevant to evaluator orchestration
- `.planning/codebase/CONCERNS.md` — Existing fragility around hardcoded names, paths, and HPC assumptions

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `test_models/calculate_system_energies.py`: already generates per-system SLURM scripts for the WTMAD-2 workflow
- `test_models/script.py`: already performs individual benchmark energy calculations and writes result lines
- `test_models/run_molden.py`: already submits avRANE-related SLURM jobs for molecule and atom density generation
- `test_models/get_molden.py`: already handles PySCF setup, checkpoint-backed functional loading, and density artifact emission

### Established Patterns
- Evaluation code is script-first and heavily depends on working-directory-relative paths
- SLURM submission is currently done inline through generated shell scripts and `sbatch`
- Functional selection is currently encoded through string names and legacy checkpoint naming conventions
- Output artifacts are currently spread across shared repo directories like `Results/`, `logs/`, `den_mol_or/`, and `denrho/`

### Integration Points
- Phase 1 will need to sit at the boundary between new experiment orchestration and legacy scripts under `test_models/`
- The new experiment contract must be consumable by both the future WTMAD-2 refactor and the future avRANE refactor
- The generated functional-name layer will likely connect the new runner to `test_models/DFT/functional.py` and any refactored checkpoint-loading logic

</code_context>

<specifics>
## Specific Ideas

- Smoke mode should execute real branch logic on tiny representative subsets, not merely validate staging or submission.
- The experiment workspace should feel like an operator-owned run bundle with clearly separated inputs, jobs, outputs, reports, and logs.
- Successful branch outputs must remain inspectable even when the overall experiment is marked failed.

</specifics>

<deferred>
## Deferred Ideas

None - discussion stayed within phase scope.

</deferred>

---

*Phase: 01-experiment-contract-foundation*
*Context gathered: 2026-03-24*
