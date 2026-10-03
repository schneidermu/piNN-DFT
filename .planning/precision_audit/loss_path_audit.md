# Frozen FS loss and gradient path audit

**Conclusion: no gradient/Armijo objective mismatch found for chemistry, integrated `E_xc`, or AO operator.** The exact same three objective factory callables produce the scalars differentiated by `compute_isolated_task_gradients` and reevaluated by `_evaluate_trial_losses`. This is a static code-path finding; it does not establish finite-precision directional-derivative convergence.

## Identity and provenance

- Audited repository HEAD: `79f89dda4c9b1ce7bcfdec609774bb005b3c3080` (`lap_full_vxc`). The pinned frozen run names source commit `5b913c02545e4528562ffef69abda24cc999acb0`; the relevant objective/model files listed below have no diff between those commits. The HEAD change is the report-only commit `docs: record frozen PCD and FS direction diagnostics`.
- Pinned external driver: `C:/Dev/readWFN_share_ms/lap_direction_recovery_runs_20261003/native_fs_batch_v2.py`, SHA-256 `b24102c52e7c36cbb10b8f3a671b0f53fbc5c20eecc2b2c1c9a186c714dc1be3`.
- Driver binds one reaction and one mRKS system, builds `objectives` once, and passes that mapping to `train_moo_update` (`native_fs_batch_v2.py:177-189,247-260`). Frozen task order is `chem`, `exc`, `op`; `world_size=1`.
- `train_moo_update` calls `compute_isolated_task_gradients(model, objective_factories)` and rank-averages its scalar values (`lap_moo_training.py:377-380`). The direct Armijo branch restores the same base model state before each trial and calls `_evaluate_trial_losses(model, objective_factories)` (`lap_moo_training.py:438-462`). Both code paths invoke the identical mapping entries, in `TASK_NAMES` order.
- The gradient pass unwraps an optional tuple and reshapes its scalar (`_scalar_loss`, lines 97-103); trial evaluation unwraps the same optional tuple and converts the scalar to a Python float after detach (`_evaluate_trial_losses`, lines 253-277). The three factories return scalar tensors, so neither path selects a different tuple member or reduction.

## Per-task identity

| Task | Gradient scalar | Armijo scalar | Identity result |
|---|---|---|---|
| Chemistry | `make_reaction_objective` calls `reaction_loss(model, reaction, reaction["Energy"], device, dtype, dispersions)` (`lap_moo_training.py:571-588`; `lap_training.py:36-62`). | Same `objectives["chem"]` closure passed to `_evaluate_trial_losses`. | **Same scalar.** Same frozen `NCCE31/reaction 0/level2_mura` record, target, database, grid, components and coefficients. |
| Integrated `E_xc` | `energy_objective` runs `LapEnergy` through `integrated_energy`, adds the named mRKS dispersion once, then calls `batch_exc` (`lap_moo_training.py:608-624`; `lap_vxc.py:327-344`; `optuna_joint.py:418-436,804-821`). | Same `objectives["exc"]` closure. | **Same scalar.** Same `BH` feature/weight arrays, `Exc` target, dispersion mapping, 256-point chunking, kcal/mol factor `627.5095`, and one-system RMSE reduction. |
| AO operator | `operator_objective` assembles `V` from the fixed AO chunks and calls `operator_loss` (`lap_moo_training.py:626-652`; `lap_operator.py:287-309,369-384`). | Same `objectives["op"]` closure. | **Same scalar.** Same AO chunks, reference, overlap, `S^-1/2` convention, squared Hilbert–Schmidt reduction, and division by `nAO`. |

### Chemistry details

`reaction_loss` casts the floating reaction record and the passed reference energy to the requested device/dtype, checkpoints the same model forward, and calls `calculate_reaction_energy` before `batch_fchem` (`lap_training.py:36-62`). `calculate_reaction_energy` integrates each component, adds HF energies and configured component dispersions, combines components by reaction coefficients, and converts to kcal/mol with `627.5095` (`reaction_energy_calculation.py:86-126,129-151`). `batch_fchem` applies the same database/frequency weight and RMSE reduction in both passes because both pass through `reaction_loss` (`optuna_joint.py:61-89,779-801`). The reaction target passed into `reaction_loss` is the same `reaction["Energy"]` value captured by the factory.

### `E_xc` details

`integrated_energy` uses the same `LapEnergy`, full central grid, weights, and fixed 256-point chunks on each invocation; it accumulates the local terms in float64 (`lap_vxc.py:327-344`). The factory applies `_add_mrks_dispersion_once(..., system.name, ..., True)` exactly once, casts the stored target to the prediction dtype, and calls `batch_exc([system.name], ...)` (`lap_moo_training.py:608-624`). `batch_exc` computes the per-system RMSE and multiplies by `627.5095` (`optuna_joint.py:804-821`). The captured `BH` system identity and tensors are unchanged between calls.

### AO operator details

Each call requires contiguous AO chunks covering the entire grid, applies the same inner 256-point assembly chunks, casts each chunk operator to the cached reference dtype, and sums in the same order (`lap_moo_training.py:626-652`). `operator_loss` forms `S^-1/2 (Vpred−Vref) S^-1/2`, sums its squared entries, and divides by `nAO` (`lap_operator.py:341-384`). `local_partials` detaches descriptor inputs only to define the local density derivatives; `create_graph=True` keeps the mixed model-parameter path through those partials (`lap_vxc.py:149-171`; `lap_operator.py:268-283`). References and overlap are fixed in both passes.

## Shared mode, dtype and reduction behavior

The driver sets the model to training mode once (`native_fs_batch_v2.py:161-163`) and never switches mode for trials. The Lap model requires dropout `0.0` (`NN_models_lap.py:16-27`); its inherited stochastic modules therefore do not create a train/eval formula split. Gradient and trial calls use the same float32 model, same loaded cached inputs, same objective closures, and same chunk boundaries. `CentralAOCache` returns features, quadrature weights, and AO factors in the configured model dtype; it loads reference operator, overlap, and `Exc` as float64 (`train_lap_moo.py:288-333,359-447`).

Chemistry and integrated-energy objectives use `torch.utils.checkpoint`; backward may recompute their forward segments. The Lap architecture has dropout disabled, and both the trial forward and gradient forward call those same checkpointed factories. The operator path builds local partials with `create_graph=True` in both calls. No task path has a separate Armijo formula, altered normalization, changed batch identity, omitted dispersion, reference detach difference, or mode switch.

For the frozen one-rank case, averaging does not change the scalar: the gradient path and Armijo path both pass their task values through `_average_task_losses` (`lap_moo_training.py:232-241,274-277,377-380`). The only difference is downstream use: autograd consumes the graph in the gradient pass; Armijo detaches the already-computed scalar for reporting/comparison.

## Finding and limit

**Finding:** for each of the three tasks, the gradient is autograd of the exact mathematical scalar that Armijo reevaluates. No objective/gradient consistency defect is visible in the audited code paths.

**Limit:** code identity does not rule out float32 cancellation, rounded parameter displacement, or a finite-step response whose signal is below evaluation precision. No loss or gradient evaluation was executed for this audit. The preceding FS report's zero-step helper comparison is corroboration only; it reports exact agreement for all three base losses but does not test directional derivatives.

**Blocker:** none for static loss-path identity. Numerical classification still requires the separately authorized frozen-state precision/directional-response diagnostic.

## Inspected files and functions

- External pinned `native_fs_batch_v2.py`: objective construction, frozen sample binding, `model.train()`, task order, and call to `train_moo_update`.
- `train_models/lap_moo_training.py`: `_scalar_loss`, `compute_isolated_task_gradients`, `_evaluate_trial_losses`, `train_moo_update`, `make_reaction_objective`, `make_mrks_objective_factories`, `make_three_objective_factories`, `_average_task_losses`.
- `train_models/lap_training.py`: `tensor_record`, `reaction_loss`.
- `train_models/reaction_energy_calculation.py`: `integration`, `get_energy_reaction`, `calculate_reaction_energy`.
- `train_models/optuna_joint.py`: dispersion helper, `batch_fchem`, `batch_exc`, task weighting constants.
- `train_models/lap_vxc.py`: `LapEnergy`, `local_partials`, `integrated_energy`.
- `train_models/lap_operator.py`: `assemble_rks_operator`, overlap orthonormalization, `operator_loss`.
- `train_models/train_lap_moo.py`: `CentralAOCache`.
- `train_models/NN_models_lap.py` and `train_models/NN_models.py`: Lap dropout requirement and inherited dropout implementation.
