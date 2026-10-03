# Precision-audit tickets

**Current state: all three tickets complete; independent review accepted B primary and A secondary.**

## 1. Freeze and review the audit protocol — complete

Write and review `spec.md`: pinned cursor-2 inputs, exact five-step set, float32 update semantics, controlled widened-source float64 policy, runtime flags, numerical floors, thresholds, and stop/classification rules. Root accepted the protocol and source-dtype policy before execution. The classification clarification is incorporated: arithmetic sensitivity alone is not category C, and finite one-sided residuals alone do not prove C or D.

**Done when:** root accepts the predeclared protocol and confirms the source dtype policy. No training, panel, or parameter trial is part of this ticket.

## 2. Run the bounded frozen precision capture — execution complete; review pending

The bounded capture ran from the unchanged cursor-2 state using the existing factories/helpers and independent native FS solves. It includes exactly the five specified `t` values and matched-state controls only at `2^-15` and `2^-20`. Canonical scripts, arrays, results, and receipts with hashes are recorded in `spec.md`. Float32 and float64 snapshots were restored; input provenance, runtime flags, zero-step, dtype-round-trip, simplex/KKT, and rollback checks are recorded in the receipts. The three canonical drivers passed Ruff and `py_compile`.

**Done when:** all requested metrics are captured for exactly five step values, receipts record the bounded gates and rollback, and artifacts are hashed. Met; independent review accepted the canonical receipts.

## 3. Independent review and classification — complete

Review the executed drivers and hashes, five-point table, dtype/input labels, response-floor handling, and rollback receipts. Treat all thresholds as diagnostic labels, not causal proofs. Arithmetic sensitivity alone is not a defect or category C; finite one-sided residuals alone do not distinguish C from D. No category is forced before review.

**Done when:** the reviewer accepts a supported category or records E/inconclusive with the exact unresolved measurement. Met: primary B and secondary A; no centered follow-up needed. The single next diagnostic is a fixed-candidate float32/double chemistry component trace at t=2^-15.
