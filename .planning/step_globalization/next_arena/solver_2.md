# Next globalization falsification: native common-descent direction

## Proposed first mechanism

Test the exact Fliege–Svaiter multiobjective steepest-descent direction on the frozen cursor-2 batch, then reuse componentwise vector Armijo. This is a direction-generation test, not another line search on canonical PCD: PCD’s proposal has an adverse chemistry slope at cursor 2, so backtracking along that fixed ray cannot repair it. The separate raw-gradient convex-hull receipt shows strict common descent exists at that same state. Its minimum-norm convex-hull point `c` gives `v=-c`, with directional derivatives `(-0.0919399, -879.0716, -0.0919399)` for chemistry, E_xc, and AO operator; this is the KKT/dual form of the native steepest direction, not a custom projection hybrid.

Chemistry remains the primary outcome: require strict finite-step improvement in chemistry, E_xc, and AO operator on the training batch, then compare chemistry first on the matched cursor-2 panel; auxiliary gains cannot compensate for a chemistry regression. Fliege–Svaiter is symmetric and assumes no task-ordering weights, so treat it only as a one-step diagnostic, not a chemistry-priority replacement for canonical PCD. Do not add preference weights or alter the objectives in this test.

## Minimal protocol

Starting from the immutable seed-41 cursor-2 checkpoint, reuse the hash-bound raw gradients in the read-only receipt (or recompute and require exact identity) on the frozen NCCE31/BH batch and solve

`v = argmin_d max_i(g_i · d) + 0.5 ||d||²`.

Verify every raw slope is strictly negative, compare `v` with the already logged convex-hull witness, and test `t = 1, 1/2, 1/4, …` using the same batch for slopes and trial losses. Predeclare `beta_A=1e-4`, `rho=0.5`, `t0=1`; accept the largest tested `t` satisfying the three componentwise Armijo inequalities. Reuse the current 20-halving guard for runtime; on cap, reject without state mutation. Evaluate the accepted parameter displacement read-only on the matched cursor-2 panel. Do not advance optimizer, EMA, scheduler, sampler, or RNG state, and do not perform a multi-update training run in this falsification.

Cost: reuse the three stored gradients, solve only a three-variable simplex QP, evaluate at most 21 frozen-batch trial points (including `t0`), then make one matched-panel evaluation. No optimizer update or GPU training run is needed.

Expected signature: all three same-batch losses satisfy vector Armijo, with chemistry decrease reported as the primary result. If the batch step passes but chemistry fails to improve on the matched panel (or panel E/operator guardrails regress), treat that as sample-to-panel mismatch and trigger the next strategy: larger/DB-balanced per-objective batches with canonical PCD fixed. If the panel also improves all three, this one-point result supports chosen-direction failure; it does not establish training or generalization. A missing negative slope means receipt/replay identity was not reproduced. Strict slopes but rejection through the existing cap is inconclusive about the hypothesis and calls for a finite-precision/objective-replay audit before a new optimizer choice.

The alternatives are less diagnostic first: [Liu and Vicente’s stochastic multi-gradient method](https://doi.org/10.1007/s10479-021-04033-z) treats noisy objective gradients and explicitly analyzes bias even when each task estimator is unbiased, so larger/DB-balanced batches or variance reduction fit if the panel test exposes transfer noise. [Chen, Tang, and Yang’s BBDMO](https://doi.org/10.1016/j.ejor.2023.04.022) dynamically rescales directions to address imbalance-driven small Armijo steps; the present stop has zero trials because chemistry’s PCD slope is positive, and only one five-backtrack rescue is documented. [Carrizo, Lotito, and Maciel’s vector trust-region method](https://doi.org/10.1007/s10107-015-0962-6) uses vector predicted-versus-actual decrease and radius updates; reserve it for repeated model-prediction mismatch, which this one-step witness has not shown. The frozen-batch direction-plus-panel probe is smaller and changes no objective or architecture.

## Primary literature

Fliege and Svaiter, “Steepest Descent Methods for Multicriteria Optimization,” *Mathematical Methods of Operations Research* 51(3), 479–494 (2000), [DOI](https://doi.org/10.1007/s001860000043): §3, problem (2) gives the native direction; Lemma 1 proves strict common descent away from Pareto-criticality; §4 and Algorithm 1 specify componentwise Armijo backtracking. Mita, Fukuda, and Yamashita, “Nonmonotone line searches for unconstrained multiobjective optimization problems,” *Journal of Global Optimization* 75(1), 63–90 (2019), [DOI](https://doi.org/10.1007/s10898-019-00802-0), provide a later monotone multiobjective line-search comparison and report `δ=10^-4`, `μ=1`, `ρ=0.5` in §7 (author-preprint pp. 4, 22). These constants are paper-reported numerical choices, not Fliege–Svaiter defaults.
