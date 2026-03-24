# Phase 1: Experiment Contract Foundation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md - this log preserves the alternatives considered.

**Date:** 2026-03-24
**Phase:** 1-Experiment Contract Foundation
**Areas discussed:** CLI contract, Experiment workspace, Model identity, Branch lifecycle, Reporting

---

## CLI contract

| Option | Description | Selected |
|--------|-------------|----------|
| Checkpoint + experiment name only | Keep the top-level interface minimal | ? |
| Add broader control flags | Expose more branch/runtime controls at the top level | |

**User's choice:** Keep the top-level interface to checkpoint path and experiment name, plus smoke mode.
**Notes:** Smoke mode should remain a real execution path rather than a no-op validation mode.

---

## Smoke mode

| Option | Description | Selected |
|--------|-------------|----------|
| Minimal real subset | Run tiny representative subsets for both branches | ? |
| Submission-only validation | Validate staging and job submission without collecting metrics | |
| Reduced full-run heuristic | Use generic reduced workloads rather than named subsets | |

**User's choice:** Minimal real subset.
**Notes:** avRANE subset should be `H2` and `N2`. WTMAD-2 subset should be `BH76-5` and `SIE4X4`.

---

## Experiment workspace

| Option | Description | Selected |
|--------|-------------|----------|
| Nested subfolders | Separate run assets by role for easier inspection | ? |
| Flat folder | Keep all outputs in one directory | |

**User's choice:** Nested subfolders.
**Notes:** Preferred structure is `input/`, `jobs/`, `outputs/`, `reports/`, and `logs/`.

---

## Model identity

| Option | Description | Selected |
|--------|-------------|----------|
| Experiment name only | Generated functional name depends only on the run name | |
| Checkpoint filename only | Generated functional name follows the model file stem | |
| Combined identity | Generated functional name uses both experiment name and checkpoint filename | ? |

**User's choice:** Combined identity.
**Notes:** This should replace the current dependency on special checkpoint naming conventions.

---

## Branch lifecycle

| Option | Description | Selected |
|--------|-------------|----------|
| Parallel branches | WTMAD-2 and avRANE run simultaneously | ? |
| Sequential branches | Run one branch after the other | |

**User's choice:** Parallel branches.
**Notes:** A branch should be marked failed if it raises an error.

---

## Top-level failure state

| Option | Description | Selected |
|--------|-------------|----------|
| `partial_failure` | Separate top-level partial status from branch failures | |
| `failed` with preserved outputs | Mark run failed overall but keep successful branch results | ? |

**User's choice:** `failed` with successful branch outputs preserved.
**Notes:** Reporting still needs to surface successful metrics and artifacts from unaffected branches.

---

## Reporting

| Option | Description | Selected |
|--------|-------------|----------|
| Full metrics tables | Preserve complete metric detail in final reports | ? |
| Concise summary only | Optimize for short operator readouts | |

**User's choice:** Full metrics tables.
**Notes:** Human-readable reporting should be rich enough that operators do not need to hunt through raw branch outputs first.

---

## the agent's Discretion

- Exact flag names and manifest schema
- Exact functional-name formatting and sanitization
- Exact branch polling implementation details

## Deferred Ideas

None.
