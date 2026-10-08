# Relchem sequential accumulation screen: K1/K2/K4/K6

Decision: retain K1. No larger K improves the observed chemistry probe mean versus control. K2/K4 chemistry differences are unfavorable with paired intervals above zero; K6 is inconclusive and much slower. Other-task gains at K2/K4 are real on the limited panel, so this is a trade-off, not a claim that batching has no benefits. No full-corpus or external-validation qualification is claimed.

## Implementation and mathematical objective

The original qualified trainer, task functions, physical constraints and AdamW implementation are unchanged. A scoped diagnostic measurement wrapper computes the original four gradients once, sequentially evaluates K−1 extra relchem singletons at the same parameters, releases each reaction graph/tensors, and accumulates detached F64 parameter gradients. It divides by K before applying the unchanged relchem coefficient. AE17/Exc/operator are evaluated exactly once per update. The native optimizer receives one F64 scalarized gradient followed by the existing single F32 cast. There is one optimizer step per K relchem reactions—not K optimizer steps.

For uniformly selected canonical identity r and variant v, Pr(r,v)=1/(251×8). With the qualified singleton loss ℓ including its existing database-dependent factors, E[g_K]=(1/K)Σ_j E[∇ℓ(r_j,v_j)]=∇[(1/(251×8))Σ_(r,v)ℓ(r,v)] wherever differentiable. Existing subgradient semantics at loss kinks are preserved. No nonlinear multi-reaction RMSE was substituted. Duplicate draws retain their multiplicity. AE17 keeps its independent singleton estimator.

Sampling preserves every original primary relchem draw and all AE17/mRKS draws. Five independent replacement domains seed941011..941015 provide nested prefixes. The same frozen90-entry manifest is used by K2/K4/K6, but only entries0..19 execute. The copied calibration manifest-hash metadata changes to reference this extended manifest; its four coefficients and all calibration statistics remain unchanged.

Fixed λ: relchem0.017015480965588553; AE17 5.141254618347414e-05; Exc1.5094644512009712e-05; operator0.33597561607048215. Constant AdamW LR1e-4, betas(.9,.999), eps1e-8, weight_decay.01. Exc chunk4096, operator256; chemistry16384 above131072 points. Original corrected P536 tensor SHA `3c2e45d6a86cccf80d7b4dc458e67f9b406935ed60d4e57433bb441a6df88da6`; immutable dataset logical SHA `61c221a19b9987717e69cac182ad545241f8807db4126c0949a99992e4c210ef`. No S5 initialization, SVRG, surgery, scheduler or line search.

## Numerical qualification / control reuse

K1 raw gradients for all four tasks are bitwise equal to the historical initial update. New checkpoint0 model/buffers equal control exactly; other three initial gradients equal control bitwise; primary/AE17/mRKS sequences are identical. The unchanged trainer/physics and fixed optimizer establish control equivalence, so historical K1 t20 is reused.

F64 sequential means versus independent NumPy reference: maximum relative error 1e-16 (frozen gate1e-12). Large DBH76 reaction33, level3_delley:263320 points. Loss is exactly equal for qualified16384 versus32768 chunks; gradient relative disagreement 1.06e-12 (gate1e-10). Model remains unchanged during qualification.

30 focused/relevant tests pass, including exact1/K normalization, detached means, nested sampling, one other-task evaluation per update, actual native AdamW state/resume equivalence, and paired-bootstrap arithmetic. Model, optimizer moments, RNG, cursor, manifest and source hashes remain resumable. Ruff, compileall and diff-check PASS. All60 new updates completed; no nonfinite state, OOM, retry or recovery occurred. Historical cursor59 SHA remains unchanged.

## Frozen training-only probe

Ratios are t20/t0, not exact full-corpus objectives. Frozen panel:16 relchem identities,8 AE17 identities,8 mRKS systems shared for Exc/operator, population-stratum weighted. Paired5000 within-stratum bootstrap resamples seed931002. Coverage and training-seed uncertainty are not captured by these intervals.

| K | Relchem | AE17 | Exc | Operator |
|---|---|---|---|---|
| 1 | 0.979260 [0.914638, 1.037269] | 0.560224 [0.519426, 0.582552] | 0.569967 [0.545721, 0.589416] | 0.936553 [0.922172, 0.950245] |
| 2 | 1.008607 [0.948627, 1.069810] | 0.070147 [0.037908, 0.146304] | 0.077368 [0.044024, 0.114528] | 0.908267 [0.891348, 0.923277] |
| 4 | 0.993975 [0.931643, 1.053782] | 0.173622 [0.131048, 0.222004] | 0.211981 [0.191239, 0.228356] | 0.916398 [0.901726, 0.929053] |
| 6 | 0.988556 [0.945067, 1.027749] | 0.483021 [0.458992, 0.523151] | 0.498166 [0.479088, 0.514201] | 0.943743 [0.937019, 0.950226] |

| K/task | Ratio difference vs K1 | Paired95% interval |
|---|---:|---|
| 2/relchem | +0.029347 | [+0.019629, +0.045691] |
| 2/ae17 | -0.490076 | [-0.538132, -0.384185] |
| 2/exc | -0.492599 | [-0.541622, -0.432085] |
| 2/op | -0.028286 | [-0.030608, -0.026393] |
| 4/relchem | +0.014715 | [+0.009846, +0.023078] |
| 4/ae17 | -0.386601 | [-0.412085, -0.331661] |
| 4/exc | -0.357986 | [-0.375612, -0.337658] |
| 4/op | -0.020155 | [-0.022784, -0.018076] |
| 6/relchem | +0.009296 | [-0.013189, +0.034835] |
| 6/ae17 | -0.077203 | [-0.101190, -0.021712] |
| 6/exc | -0.071801 | [-0.082274, -0.057164] |
| 6/op | +0.007190 | [-0.002045, +0.015715] |

## Initial gradient variance diagnostic

Four independent nested batches per K and a separate12-singleton reference were frozen before calculation. All gradients are at θ0. The reference is an estimate, never gold/full251. Statistics below are descriptive: K1 happens to draw small gradients (norm range2.9–28.2), whereas the independent reference contains a643.6-norm gradient. This makes the four-batch variance estimate particularly unreliable. Population IID covariance decreases as1/K, but this finite Monte Carlo panel does not demonstrate that reduction and must not be used to reject the mathematical estimator.

| K | Norm CV | Mean pairwise squared-gradient variance | Mean pairwise direction cosine | Reference cosine min / median / max |
|---|---:|---:|---:|---|
| 1 | 0.615784 | 164.064353 | 0.755684 | -0.888554 / -0.578312 / -0.181890 |
| 2 | 0.754167 | 5319.486423 | 0.103275 | -0.532279 / 0.125033 / 0.815867 |
| 4 | 0.930237 | 2879.723765 | 0.258202 | -0.306965 / 0.531619 / 0.970144 |
| 6 | 0.854773 | 1204.431019 | 0.105524 | -0.409206 / 0.302144 / 0.961694 |

Mean pairwise variance is the average ||g_i−g_j||²/2 over six distinct batch pairs; direction dispersion is1 minus the mean pairwise unit-vector cosine. Interaction with the same fixed independent AE17/Exc/operator gradients is stored in JSON as each task’s combined-gradient directional progress. No gradient surgery or parameter step is used in this diagnostic. The reference identity list, all diagnostic identities and the external NPZ SHA are frozen and retained.

## Cost and memory

| K | Timed stages s/update | Actual wall s/update | Live allocated GiB | Peak reserved GiB | Chemistry gain percentage points / training minute |
|---|---:|---:|---:|---:|---:|
| 1 | 9.344 | 9.358 | 12.232 | 14.256 | +0.664890 |
| 2 | 15.500 | 15.519 | 12.232 | 25.477 | -0.166391 |
| 4 | 24.778 | 24.797 | 12.232 | 25.904 | +0.072897 |
| 6 | 30.884 | 30.905 | 12.232 | 25.938 | +0.111089 |

Timed stages include task loading/gradients, aggregation/step and saved gradient diagnostics. Actual wall uses checkpoint0-to-checkpoint20 file timestamps and includes bookkeeping/checkpoint I/O; these arms ran without an intervening pause. No claim of physical VRAM residency is made from reserved allocator bytes. Although live allocation stays12.232GiB, reserved memory rises above25GiB. No OOM occurred, but memory pressure/offload cannot be excluded; no memory-policy tuning was performed.

The three new20-update arms took 23.740 training minutes in total; qualification and probes are additional. New workload:60 optimizer updates,240 relchem and60 AE17 reactions,60 Exc and60 operator gradients. Other task evaluations do not multiply with K.

| K | Observed variance / K1 | Extra wall seconds/update | Observed variance reduction / extra second |
|---|---:|---:|---:|
| 2 | 32.423170 | 6.160569 | -836.841854 |
| 4 | 17.552404 | 15.439018 | -175.895864 |
| 6 | 7.341211 | 21.546700 | -48.284269 |

These negative finite-panel variance reductions mean no reduction was demonstrated on this tiny panel; they do not contradict the IID population result. The cost-normalized chemistry gains favor K1. They are descriptive endpoint efficiencies, not a matched-time trajectory comparison.

Duplicates within each20-update training arm (excess identity occurrences / exact identity+variant repeats): K1: 0 / 0, K2: 0 / 0, K4: 1 / 0, K6: 1 / 0. No draws were deduplicated.

## Task conflicts and decision

| K | Relchem sampled ascent count | AE17 | Exc | Operator |
|---|---:|---:|---:|---:|
| 1 | 2/20 | 9/20 | 9/20 | 9/20 |
| 2 | 4/20 | 8/20 | 6/20 | 8/20 |
| 4 | 4/20 | 7/20 | 7/20 | 8/20 |
| 6 | 5/20 | 8/20 | 8/20 | 12/20 |

These are signed first-order predictions from the actual rounded AdamW displacement, not observed full-objective changes. The sampled relchem gradient represents a different K-batch at each arm, so counts do not independently quantify full-population conflict. Larger K did not remove the observed task conflicts.

Retain K1. K2/K4 give stronger AE17/Exc/operator progress while chemistry convergence worsens versus K1. K6 has no credible chemistry advantage, a slightly worse operator mean, and about3.30× the wall cost. All chemistry-versus-t0 bootstrap intervals include1. This screen does not establish that chemistry noise is unimportant or prove a single cause of conflict; it does not support larger chemistry batches as the practical correction.

Exact next experiment, only after separate approval: finish the existing K1 constant-LR1e-4 control from its preserved cursor59 to t90, then evaluate exact four-objective endpoints and clean28 once. Do not rerun completed updates. No larger-K continuation or500-update run was launched.

Artifacts/checkpoints remain outside Git at `C:/Dev/readWFN_share_ms/lap_relchem_accumulation_20261008`; every checkpoint20 and sampling-manifest hash is in metrics. No historical run, dataset, scientific target, architecture, physical convention, precision boundary, objective coefficient or optimizer behavior was modified. No clean28, SCF, Diet100 or full-corpus gradient/endpoint calculation was performed.
