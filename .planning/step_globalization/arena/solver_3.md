# Solver 3 — next globalization experiment after `b20d153`

## Recommendation

Test a **monotone vector Armijo backtracking rule on the existing PCD direction** first. Keep the chemistry-first PCD QP and all three objectives fixed; choose only a scalar multiplier on the already computed update. This is the smallest test that directly probes the demonstrated failure: on the second repeat of the same AE17/H2 sample, plain SGD retained near-perfect direction alignment, all three local dots predicted descent, but chemistry rose 56.64× and E_xc 21.78×. That observation makes a finite-step/curvature overshoot more directly supported than optimizer rotation or minibatch variance as the first cause.

For a parameter displacement `p = -ηd`, use the componentwise vector Armijo test

`f_i(θ + αp) ≤ f_i(θ) + c α ∇f_i(θ)·p`, for `i ∈ {chem, E_xc, AO-op}`.

With `∇f_i·p < 0`, this requires sufficient decrease in every task. This is the established multiobjective steepest-descent globalization pattern: Fliege and Svaiter’s multicriteria method uses Armijo backtracking, and later analyses state the componentwise vector condition and its simultaneous-decrease consequence (Fliege & Svaiter, 2000; Cocchi et al., 2020). Use it as a scalar globalization of the validated PCD vector; do not replace PCD with the literature’s different direction QP.

## Why it fits better than the alternatives

- **Multiobjective trust region:** Carrizo, Lotito, and Maciel (2016) extend predicted-versus-actual reduction and radius updates to nonconvex multiobjective optimization. This is a principled second choice if the PCD ray has no usable common-decrease step or if the accepted α is repeatedly negligible. It needs a model/radius policy and can alter the trial direction, so it costs more and confounds the simplest “same direction, smaller finite step” test.
- **BBDMO / BB spectral descent:** Chen, Tang, and Yang (2023) use BB secant scales inside the direction-finding subproblem to address objective imbalance; their method still uses line search. That is relevant if Armijo repeatedly returns a tiny α because one task’s curvature/scale throttles the common step. It changes the direction and adds multi-objective secant state, while this report already shows favorable directional derivatives followed by a catastrophic finite step. Test it after the scalar-step hypothesis.
- **Variance reduction / larger balanced batches:** Mercier, Poirion, and Désidéri (2018) give a stochastic multiple-gradient method for expected objectives; Johnson and Zhang (2013) introduce SVRG for finite-sum gradient variance. Both address noisy gradient estimates. The observed failure recurred on the same sample under plain SGD, so gradient variance cannot explain that event by itself. Batch balancing becomes the next test if a step accepted on the repeated training sample fails on a fixed balanced evaluation panel.

## Smallest falsification protocol

1. Use the exact `b20d153` five-update plain-SGD control and restore its saved state immediately before the failing second AE17/H2 update. Preserve the fresh predopt, seed and sample identity, `τ=.02`, EMA `β=.999`, `ε=1e-8`, task order `chem, exc, op`, dtype, and calibrated `η=6.632573669086685e-7`. Keep momentum, weight decay, and scheduler absent as in that control.
2. Compute the existing PCD direction once. Log all three raw `g_i·d` values and confirm they are positive (the equivalent update dots `g_i·p` must be negative); if not, this is not the overshoot replay. Record baseline values for the three existing scalar losses.
3. Try `α = 1, 1/2, 1/4, …, 2^-20`, restoring the same parameter snapshot for every trial. Evaluate all three unchanged losses on the exact same AE17/H2 sample and reduction at each candidate; do not recompute gradients, advance EMA/RNG, or update optimizer state between trials. Use `c=0.1`, contraction `ρ=0.5`, and accept the first (largest) α that satisfies the vector Armijo inequality for chemistry, E_xc, and AO operator. Chemistry is the first gate; both secondary tasks remain hard decrease gates. α=1 must reproduce the reported overshoot before interpreting a smaller accepted trial.
4. At the accepted candidate only, evaluate the paired before/after losses on the report’s existing fixed balanced panel (27 chemistry reactions and the available 15 AO systems), with exactly the existing task reductions. This is a diagnostic confirmation, not a new tuning set. A full success is strict finite decrease in chemistry, E_xc, and AO operator on that panel; report the chemistry change first, then secondary changes, accepted α, and parameter-step norm. Keep the candidate only for this isolated replay; no objective or checkpoint redesign is needed.

One direction computation plus at most 21 trial evaluations directly answers whether shrinking the exact failing step restores common decrease. The fixed panel distinguishes a same-sample correction from useful cross-sample chemistry progress.

## Decision signatures and next trigger

- **Overshoot supported:** α=1 reproduces the chemistry/E_xc explosion; some α<1 passes vector Armijo, and the paired fixed-panel values all fall. Next, replay the same five-update SGD stream with per-update backtracking and compare the three finite-step trajectories to the fixed-α control.
- **Sample-specific step only:** same-sample Armijo passes, but the fixed balanced panel does not improve all tasks. Next test a larger balanced batch / stochastic variance reduction with direction and line-search logic held fixed; do not claim chemistry improvement from the training batch alone.
- **No usable scalar step:** all three local dots are descent, but no trial through `2^-20` passes, or the evaluator disagrees with the saved baseline. First audit exact loss reductions, determinism, dtype/rounding, and snapshot restoration. If those are sound, move to a multiobjective trust-region trial because the evidence then points beyond simple ray-length control.
- **Very small accepted α:** this still confirms local overshoot but may make training uneconomic. Record α and per-task curvature limits. If the limiting constraint reflects objective-scale imbalance, BBDMO is the next targeted comparison; if the common ray itself is inadequate, use trust-region globalization.

## Primary literature

- J. Fliege and B. F. Svaiter (2000), “Steepest descent methods for multicriteria optimization,” *Mathematical Methods of Operations Research* 51, 479–494. [https://doi.org/10.1007/s001860000043](https://doi.org/10.1007/s001860000043)
- G. Cocchi, G. Liuzzi, S. Lucidi, and M. Sciandrone (2020), “On the convergence of steepest descent methods for multiobjective optimization,” *Computational Optimization and Applications* 77, 1–27. [https://doi.org/10.1007/s10589-020-00192-0](https://doi.org/10.1007/s10589-020-00192-0)
- G. A. Carrizo, P. A. Lotito, and M. C. Maciel (2016), “Trust region globalization strategy for the nonconvex unconstrained multiobjective optimization problem,” *Mathematical Programming* 159, 339–369. [https://doi.org/10.1007/s10107-015-0962-6](https://doi.org/10.1007/s10107-015-0962-6)
- J. Chen, L. Tang, and X. Yang (2023), “A Barzilai–Borwein descent method for multiobjective optimization problems,” *European Journal of Operational Research* 311(1), 196–209. [https://doi.org/10.1016/j.ejor.2023.04.022](https://doi.org/10.1016/j.ejor.2023.04.022)
- Q. Mercier, F. Poirion, and J.-A. Désidéri (2018), “A stochastic multiple gradient descent algorithm,” *European Journal of Operational Research* 271(3), 808–817. [https://doi.org/10.1016/j.ejor.2018.05.064](https://doi.org/10.1016/j.ejor.2018.05.064)
- R. Johnson and T. Zhang (2013), “Accelerating stochastic gradient descent using predictive variance reduction,” *Advances in Neural Information Processing Systems 26*. [https://papers.nips.cc/paper/4937-accelerating-stochastic-gradient-descent-using-predictive-variance-reduction](https://papers.nips.cc/paper/4937-accelerating-stochastic-gradient-descent-using-predictive-variance-reduction)
