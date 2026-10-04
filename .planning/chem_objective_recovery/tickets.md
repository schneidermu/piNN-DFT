# Chemistry objective recovery tickets

## 1. Normalize scalar database labels at the chemistry boundary

**Blocked by:** None
**Delivers:** Single-reaction records use their real Minnesota database identity; neither chemistry reducer silently consumes a database name as characters.

- [x] reaction_loss adapts scalar labels to a one-element sequence and preserves existing sequences.
- [x] Both batch_fchem helpers reject scalar strings with TypeError.
- [x] Existing list-based factors, grouping, reductions, losses, and gradients stay unchanged.

## 2. Lock scalar, list, gradient, and predopt behavior

**Blocked by:** 1
**Delivers:** Regression coverage proves scalar and explicit-list routes agree for NCCE31 and another database, and valid predopt behavior remains stable.

- [x] Cover scalar NCCE31, scalar AE17, singleton lists, and a heterogeneous list.
- [x] Assert direct scalar calls to both helpers fail clearly.
- [x] Compare finite parameter gradients for scalar-adapted and equivalent list inputs.
- [x] Verify valid predopt list inputs and canonical constants-MSE predopt remain unchanged.

## 3. Replay the frozen chemistry case with corrected weights

**Blocked by:** 1, 2
**Delivers:** The stored state pair is evaluated with the named database factor in F32 and matched-F64 arithmetic, with gradients and input/state identities preserved.

- [x] Reproduce corrected F32 and matched-F64 chemistry values and gradients from the frozen parameters and inputs.
- [x] Record corrected sign, magnitude, resolution, and source identities; do not rewrite the historical capture.

## 4. Establish clean chemistry geometry from verified predopt

**Blocked by:** 3
**Delivers:** A hash-verified predopt initialization and corrected double-precision chemistry gradients establish whether the first-order direction is valid before any update.

- [x] Verify the checkpoint and immutable input hashes.
- [x] Check finite loss/gradient, scalar-versus-list parity, and the declared clean-geometry gate.
- [x] Stop before updating if any identity or geometry gate fails.

## 5. Run the gated short Armijo restart

**Blocked by:** 4 and the independent root gate
**Delivers:** If the fixed-state recovery gates pass, a fresh run tests no more than five Armijo updates using the corrected chemistry objective.

- [x] Start from the verified predopt checkpoint and corrected source.
- [x] Record each trial and require the declared componentwise acceptance rule.
- [x] Run zero updates when the gate fails; otherwise stop at the root-authorized two accepted updates or earlier on failure.

## 6. Independently review and close the recovery record

**Blocked by:** applicable completion of 1–5
**Delivers:** A reviewer confirms source behavior, test evidence, replay provenance, gated-run outcome, and historical-impact classifications.

- [x] Verify current source and artifact hashes and required test results.
- [x] Resolve every historical row as affected, unaffected, or needing replay; preserve prior reports and checkpoints.
- [x] Report any gate failure and stop reason without claiming unrun outcomes.
