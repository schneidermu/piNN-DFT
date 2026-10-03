# Conditional complete-cycle gradient contract

**Status: literature audit complete; use only if same-batch FS Armijo passes and the paired cursor-2 panel gate fails.** No cycle experiment is authorized or run by this review.

## What the cited stochastic-MGDA paper supports

Mercier, Poirion & Désidéri (2018) study stochastic multiobjective problems whose objectives are expectations of random functions. Their SMGDA combines the deterministic common-descent construction with stochastic-gradient iterations and states mean-square and almost-sure convergence under classical, restrictive stochastic-approximation assumptions. The paper distinguishes that approach from sample-average approximation (SAA), which estimates the expected objectives from scenarios.

The conditional experiment here is **not** a reproduction of SMGDA and inherits none of its convergence results. It is a deterministic, fixed-checkpoint SAA-style diagnostic over one immutable finite training-manifest cycle. The literature motivates testing the gradient estimate before changing the direction rule; it does not prove that the current cursor-2 failure is variance-driven or that this particular manifest estimates a population expectation.

For a finite empirical objective \(F_i(\theta)=\sum_{r=1}^{27}w_r f_{ir}(\theta)\), the task gradient is the same weighted sum of per-entry gradients when the existing reductions and weights define that objective. Since PCD is nonlinear in its inputs, applying it to one entry’s gradients and then averaging directions is generally different from applying unchanged PCD once to the aggregated task gradients. This is the reason to accumulate each objective’s raw gradients first, then call canonical PCD once. This equivalence/ordering statement is a mathematical inference for the declared finite empirical objective, not a theorem claimed by Mercier et al.

## Frozen diagnostic contract

Only after the stated batch-pass/panel-fail gate:

1. Load the exact immutable cursor-2 checkpoint and pre-update PCD state. Traverse every one of the 27 immutable **training** entries once at the same parameter state; never form gradients from panel rows.
2. For each objective, compute and accumulate its raw gradient using the current per-entry task reduction and the manifest’s existing weighting. Preserve objective order and reductions. Do not introduce task weights, objective normalization, architecture changes, or an aggregation hybrid.
3. Average each objective’s raw gradients over the declared complete-cycle weights. Record manifest identity, entry order, per-objective reduction/weight, and aggregate-gradient hashes.
4. Run the unchanged canonical PCD once on those three aggregated gradients, using a scratch copy of the frozen incoming EMA and current canonical hyperparameters. Do not update or commit EMA, parameters, RNG, cursor, optimizer, or scheduler.
5. Check the all-three first-order geometry. If it passes, test vector Armijo on the **same complete cycle objective** used to build those gradients, with all three unchanged reductions, objective losses, and line-search constants. The evaluation panel stays gradient-free and is used only for the paired transfer decision.
6. Compare one passing candidate with the cursor-2 base on the frozen panel. Report per-task median ratio candidate/base, p90, max, and row wins; require all three medians below one, with chemistry reported first.

The 27 entries are a fixed diagnostic manifest, not 27 independent systems or a population sample. A changed direction would show sensitivity to sample composition at this frozen state; one cycle cannot estimate variance or establish generalization. Keep this as the single conditional diagnostic, not a training run.

Primary reference: [Mercier, Poirion & Désidéri (2018), *A stochastic multiple gradient descent algorithm*, ScienceDirect article record](https://doi.org/10.1016/j.ejor.2018.05.064).
