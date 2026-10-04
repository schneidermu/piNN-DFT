# Chemistry database-label recovery

**Status:** implementation contract; static audit only  
**Audited checkout:** lap_full_vxc, commit 1b740325d8bc120d7da9ef1da117338e6ba98fcc

## Problem

Minnesota records store Database as one string. The scalar reaction_loss route forwards it to optuna_joint.batch_fchem, whose zip treats a string as an iterable of characters. For a one-row NCCE31 record, only key N pairs with the prediction; the remaining characters are truncated and the fallback weight is used. This changes the chemistry loss and gradient weight.

## Source fix contract

- At the reaction_loss boundary, convert a scalar database label to a one-element label sequence. Preserve an existing sequence exactly.
- Both optuna_joint.batch_fchem and the separate predopt_train.batch_fchem reject a scalar string with TypeError; neither helper may silently iterate characters.
- Valid list callers retain their current per-database grouping, factors, reduction, loss value, and gradient.
- Do not change canonical predopt loss, reaction-energy calculation, factor tables, optimizer, or historical artifacts.

## Runtime evidence and scope

Canonical records and MinnesotaGroupStore.load_variant retain scalar strings. dataset.collate_fn/stack_reactions turn batched records into a list of database labels. The confirmed scalar reaction_loss callers are lap_moo_training.make_reaction_objective over store records and train_lap_s5.collect_objective_calibration over canonical records. train_lap, diagnostics, the real pilot, and the S5 training-loader route use collated lists. predopt_train.extend_bases already passes lists to its helper, including when its input record is scalar. The existing reaction-loss test starts with scalar EA13 but collates it before calling reaction_loss, then checks only finiteness; it does not exercise the raw scalar route or validate the factor.

## Acceptance

1. Scalar NCCE31 and another weighted database (for example AE17) through reaction_loss equal the corresponding one-label-list result and use the named database's factor.
2. Existing singleton and heterogeneous label lists preserve current loss and grouping behavior.
3. Direct scalar-string input to either batch_fchem helper raises TypeError.
4. Scalar-adapted and explicit-list chemistry losses produce matching finite parameter gradients.
5. Valid predopt list inputs retain their prior loss and gradient; canonical run_predopt remains on its constants-MSE path.

## Recovery gates after the source fix

- Replay the frozen chemistry case with corrected weights in F32 and matched-F64 arithmetic, including gradients, against the exact stored states and inputs.
- From a hash-verified predopt checkpoint, establish the corrected double-precision chemistry-gradient contract and a clean first-order geometry result before attempting a step.
- Run at most five fresh Armijo updates only after the independent root gate passes. Stop without updates if frozen replay, geometry, or the gate fails.
- Independently review the code, receipts, tests, and historical-impact classifications before calling recovery complete.

## Out of scope

Changing precision strategy, energy assembly, PCD or line-search policy, broad retraining, and rewriting historical checkpoints or reports. The gated diagnostics above assess the corrected objective; they do not prescribe a new training method.

