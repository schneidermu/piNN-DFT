# Chemistry objective recovery

Branch: `lap_full_vxc`. Audited base: `1b740325d8bc120d7da9ef1da117338e6ba98fcc`. Source/test hashes and all external receipts are recorded in `lap_chem_objective_recovery_results.json` and the protocol.

The database-label bug is fixed. Matched-F64 chemistry scalars and gradients are validated. Two clean Armijo updates passed independent review and decreased all three sampled losses. The broader panel chemistry median still worsens; the overall three-objective goal is not achieved.

## Bug, scope, and minimal repair

`reaction_loss` passed scalar `Database="NCCE31"` into `batch_fchem`, which zipped characters against singleton predictions. It consumed `N` and applied fallback factor **8.979622680758972**, instead of NCCE31's **2.896652477664184**. Corrected/legacy factor is 10/31. AE17's intended factor is **0.5282130988681748**.

The caller now wraps scalar labels in a singleton list. Both `batch_fchem` definitions reject direct scalar strings. Valid batched list behavior and formulas remain unchanged. Three production files changed: `lap_training.py`, `optuna_joint.py`, `predopt_train.py`; 8 source LOC added, 1 removed. No production source file was added. `test_lap_protocol.py` adds 109 test LOC and removes 1.

| Path | Runtime labels | Effect |
|---|---|---|
| Store-backed one-stage MOO, PCD/FS diagnostics | scalar string | Affected |
| Raw panel absolute chemistry evaluation | scalar string | Affected |
| S5 `collect_objective_calibration` | scalar string | Affected |
| Historical S5 and other collated training | label lists | Unaffected |
| Current `predopt_train` chemistry callers | extended label lists | Unaffected |
| Canonical PBE `run_predopt` | constants MSE, no fchem | Unaffected |

Full call-site evidence is in `.planning/chem_objective_recovery/scope.json`. Canonical predopt SHA-256: **ed4ba8231d93c376ce5aa8fc81e0670f6a5a44c95c2a552651b12d6a1d4b63f8**. Its report confirms constants-only PBE predopt, without reaction/mRKS objectives.

## Corrected frozen replay

The original cursor-2 checkpoint was used only diagnostically. The exact previously stored rounded `t=2^-15` candidate was preserved.

| Arithmetic | Legacy base / candidate | Corrected base / candidate | Corrected delta |
|---|---:|---:|---:|
| F32 | 4.171715260 / 4.214704514 | 1.345714569 / 1.359582067 | +0.013867497444152832 |
| Matched F64 | 4.189337819 / 4.189335516 | 1.351399296383204 / 1.3513985534652042 | -7.429179997853197e-7 |

The sign reversal survives the semantic fix. Matched F64 means double arithmetic on the same loaded F32 source values and widened checkpoint/candidate values, with original stored corrections preserved; native source precision was not restored.

| Arithmetic | Legacy gradient norm | Corrected norm | Norm ratio | Cosine | Relative residual after expected scaling |
|---|---:|---:|---:|---:|---:|
| F32 | 144.977431568 | 46.739464987 | 0.322391316229 | 0.999999148353 | 0.00143149 |
| Matched F64 | 144.951194747 | 46.758449918 | 0.322580645161 | 1.000000000000 | 7.3689e-12 |

Forward prediction/target/pre-factor RMSE are identical between weighting controls within each dtype. F64 scaling agrees to backward-reduction roundoff. The preserved driver label `failed_predeclared_gradient_scaling_gate` refers to driver-added cutoffs absent from the actual predeclared protocol. Do not reinterpret it as a scientific gate or curvature evidence.

## Precision contract and clean geometry

Use a persistent F64 deepcopy with independent F64 parameter leaves for chemistry. Synchronize main-model stored F32 parameters, buffers, and mode into the shadow by exact name/shape/value at each base/trial. Chemistry autograd differentiates the same corrected F64 scalar reevaluated for acceptance. E/operator forwards and gradients stay on the existing F32 model; widen their computed gradients only at the aggregation boundary.

This is the smallest tested faithful route. A master-weight framework adds checkpoint/update machinery. Converting every task to double adds unproven cost. `functional_call` through `p32.to(double)` returns gradients to F32 leaves; actual double leaves are required. Do not assign double gradients to F32 `.grad` buffers or the F32 optimizer API. Direct rounded parameter copies are explicitly tested. Production MOO still defaults to its existing precision path; mixed-precision optimization is demonstrated by an external diagnostic, not a second production trainer.

At verified clean predopt, the predeclared NCCE31/BH sample reproduced all losses and gradient vectors exactly twice:

| Task | Loss | Gradient norm |
|---|---:|---:|
| Chemistry, matched F64 | 1.1126000851297508 | 46.83005514461214 |
| E_xc, existing F32 path | 31.27274717005688 | 4401.032297613772 |
| Operator, existing F32 path | 0.04194787968835473 | 0.4094464789677722 |

Fresh PCD tau=0.02 is feasible with negative-step dots -1554.09827784 / -5661.82646864 / -1.7148476224. Native FS also gives strict common descent, with chemistry/E/operator weights 0.00543054253 / 0 / 0.99456945747. This geometry sample is separate from the restart's first two manifest entries.

## Two clean updates and panel screen

Root completed driver wiring and static/synthetic checks, then ran exactly two updates from canonical predopt. Independent post-run review passed before commit. PCD tau=0.02, beta=0.999, eps=1e-8, qp_tolerance=1e-9; direct updates, no RAdamW/momentum/weight decay/OMEGA/S5 phases. Armijo c=1e-4, rho=0.5, maximum 20 backtracks, alpha0=6.632573669086685e-7. Every acceptance uses the realized stored-F32 displacement, strict decrease, and sufficient decrease in all three tasks.

| Original manifest sample | Chemistry before / after | E_xc before / after | Operator before / after | Backtracks |
|---|---:|---:|---:|---:|
| pTC13 reaction 12 / BeH2 / level3_mura | 7.093891075 / 7.090445343 | 24.721812865 / 24.673538264 | 0.033298859 / 0.033296612 | 0 |
| AE17 reaction 16 / H2 / level2_gauss_chebyshev | 56.848639695 / 39.625324765 | 5.240688612 / 1.555236374 | 0.050651290 / 0.049108250 | 1 |

Accepted alphas: 6.632573669086685e-7 and 3.3162868345433423e-7. Realized step norms: 7.198201864e-5 and 0.005658057837. EMA advances t=1 then t=2 only on acceptance. All rejected-trial and final original-model/shadow/RNG rollback checks pass. The new checkpoint is external and SHA-256 bound; no contaminated checkpoint was continued.

One matched 27-row baseline/cursor-2 screen used corrected matched-F64 chemistry and unchanged E/operator paths:

| Objective | Median ratio | p90 | Max | Paired wins / rows |
|---|---:|---:|---:|---:|
| Chemistry | 1.014138201 | 2.317356324 | 32.197713549 | 10 / 27 |
| E_xc | 0.499176940 | 0.648377043 | 0.660308290 | 27 / 27 |
| Operator | 0.957638579 | 0.962422519 | 0.969534668 | 27 / 27 |

All three medians are not below one. This is a training-panel sanity screen, not generalization or model selection. No further updates ran.

Total two-step plus panel runtime: 617.86 seconds. Torch peak allocated/reserved: 21,013,132,288 / 27,703,377,920 bytes. These exceed physical ~16GB VRAM, so Windows shared/oversubscribed GPU memory is likely involved. Resident VRAM peak was not measured; 2xV100 memory fit/performance is unverified.

Main sources and initial pins have start/end hashes. External helpers are additionally bound to earlier independently reviewed hashes and during/post-run checks. Selected restart files were hash-verified at load against pinned manifests, then independently rehashed during/post run. These supplements are not fabricated start snapshots. The final verifier enforces all rollback flags, EMA state, actual-step Armijo inequalities, checkpoint/hash identities, and matched panel ratios.

## Historical validity and acceptance

Still valid: fixed-candidate arithmetic sign reversal, audited gradient/objective formula identity, AO operator/architecture validation, and algebraic same-row panel factor cancellation. Requires rebaselining: absolute chemistry losses/norms, fixed calibration, raw FS geometry, finite-state PCD geometry/trajectories, and optimizer/checkpoint comparisons trained through the defective route. Historical reports remain untouched; label them legacy chemistry-weighting semantics. Algebraic panel factor cancellation does not make defective training checkpoints canonical.

Tests: Windows **218 passed, 3 skipped**; WSL/PySCF **139 passed, 2 skipped**. Focused **45 passed** is included in the broader validation, not added to its counts. Compileall and diff checks pass. Ruff has **112 pre-existing findings**, unchanged from baseline; **zero introduced**, not a globally clean Ruff result. Synthetic driver checks cover overshoot recovery, no-op rejection, all 21 candidate positions, and EMA progression. Independent review passed source, tests, frozen replay, clean geometry, actual two-step traces, provenance, and panel calculations.

## Eleven recovery answers

1. **Bug confirmed?** Yes: scalar NCCE31 was consumed as character N.
2. **Affected paths?** Raw store-backed MOO and its PCD/FS/panel routes, plus S5 raw calibration; collated historical S5 training is unaffected.
3. **Actual/intended factors?** NCCE31 8.979622680758972 / 2.896652477664184; AE17 intended 0.5282130988681748.
4. **Predopt unaffected?** Yes: verified constants-MSE checkpoint, no batch_fchem objective.
5. **Prior conclusions still valid?** Fixed-state precision pathology, operator validation, audited formula identity, and algebraic panel factor cancellation, with the stated provenance limits.
6. **What must be rebaselined?** Absolute chemistry/norms, scale-sensitive geometry/calibration, and trajectories/checkpoint comparisons from defective training.
7. **Corrected exact candidate still disagrees in sign?** Yes: F32 +0.013867497444152832 versus matched F64 -7.429179997853197e-7.
8. **Minimal tested consistent double scalar/gradient?** Synced persistent F64 chemistry shadow with actual F64 leaves; E/operator stay F32 and their gradient maps widen only for aggregation.
9. **Usable clean direction?** Yes: PCD/FS geometry and two PCD accepted updates demonstrate same-sample common descent. This does not guarantee panel-wide chemistry improvement.
10. **Tiny restart justified?** Yes; two updates completed and independently passed. The panel does not justify a 25/100-update continuation.
11. **Single next experiment?** A read-only corrected-double gradient/direction audit at the new clean cursor 2, before any further update.

No Diet30/Diet100, Slurm, V100 jobs, full-90 cache generation, or production training was used.

Chemistry objective semantics and precision are corrected; a tiny clean restart from canonical predopt is justified.
