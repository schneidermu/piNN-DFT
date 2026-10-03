# Independent native FS / Armijo contract review

**Status: native FS geometry and the bounded batch Armijo implementation independently reviewed; batch Armijo rejected.** No candidate exists, so no panel-transfer or full-cycle gate ran. This is a guard-limited diagnostic, not evidence that no useful representable step exists outside the tested interval.

## Native direction and KKT gate

For raw objective gradients \(g_i=\nabla f_i(\theta)\), Fliege–Svaiter’s unconstrained steepest common-descent direction solves

\[
v^* = \arg\min_v\left(\max_i g_i^T v + \tfrac12\|v\|^2\right).
\]

The equivalent dual is

\[
\lambda^*\in\arg\min_{\lambda\ge0,\;\mathbf 1^T\lambda=1}
\tfrac12\left\|\sum_i\lambda_i g_i\right\|^2,
\qquad
\bar g=\sum_i\lambda_i g_i,
\qquad
v^*=-\bar g.
\]

For this simplex QP, KKT gives \(g_i^T\bar g=\|\bar g\|^2\) on positive-weight tasks and \(g_i^T\bar g\ge\|\bar g\|^2\) on zero-weight tasks. Thus every \(g_i^Tv^*\le-\|\bar g\|^2<0\) when the minimum-norm point is nonzero. A zero minimum norm means \(0\in\operatorname{conv}\{g_i\}\), the first-order Pareto-critical case. Recompute this QP independently from the frozen raw gradients; the previously stored witness is a comparison value, not a solver result.

Report primal simplex/nonnegativity residuals, active/inactive KKT residuals, \(\|\bar g\|\), and all three raw \(g_i^Tv^*\) values. Use raw objective gradients and state their units. Independent positive rescaling of each objective preserves common-descent existence and Pareto-criticality, but generally changes the minimum-norm convex combination and selected direction. A common scale changes the direction magnitude and therefore the finite \(t_0=1\) trial. Do not silently normalize or rescale task gradients.

The sign convention needs explicit names: the native step vector is \(v^*=-\bar g\), the trial is \(\theta+t v^*\), and descent means \(g_i^Tv^*<0\). If an interface instead calls \(d=\bar g\) the descent gradient and writes \(\theta-td\), then its gate is \(g_i^Td>0\). Do not mix these conventions in the solver receipt.

## Existing injectable Armijo path

train_models/lap_moo_training.py exposes an aggregator injection on train_moo_update; in direct mode it still computes all three task gradients from the objective factories, all-reduces them, then passes the raw task gradients to the injected aggregator. It requires method="pcd", no optimizer or scheduler, and the direct vector-Armijo step rule.

The adapter treats aggregator output as a gradient: proposal = -alpha0 * joint (around line 418). To send the native Fliege–Svaiter trial vector through it, return joint = bar_g = -v* and set initial_step_size = 1.0. The resulting proposal is exactly \(v^*\), and backtracks are \(t_j=0.5^j\), beginning at \(t_0=1\). Do not reuse PCD’s calibrated alpha0.

The adapter computes slopes \(s_i=g_i^T\,proposal\), requires every slope to be negative, evaluates the same objective factories on each restored trial, and accepts only if every component satisfies \(L_i(\theta+t proposal)\le L_i(\theta)+c t s_i\) and every actual reduction is strictly positive (around lines 440–520). This is the correct componentwise Armijo sign for a plus-step vector. The existing finite guard with max_backtracks=20 tries 21 values, \(j=0\ldots20\); the original paper’s geometric search has no such finite cap. If the guard expires, report a guard-limited diagnostic, not a failure of the exact line-search theorem.

The direct adapter restores state between trial evaluations, but commits accepted parameters in memory on success. The frozen diagnostic must treat the model as scratch, retain the one accepted candidate only for paired panel evaluation, and then restore or discard it. Verify that the checkpoint, incoming EMA, RNG, cursor, optimizer/scheduler state, and source artifacts remain unchanged. Compare the panel candidate against the cursor-2 base on the same 27-row/15-system panel; the old predopt ratios are context, not this transfer gate.

## Literature boundary

Fliege & Svaiter (2000) formulate an Armijo-like componentwise condition \(F(\theta+t v)\le F(\theta)+\beta t JF(\theta)v\), start at \(t=1\), and halve until it passes. Their convergence result assumes smooth deterministic objectives and the specified exact or controlled inexact direction; it does not prioritize chemistry and does not transfer wholesale to a finite-precision PCD trainer or a capped diagnostic.

Primary references: [Fliege & Svaiter (2000), Springer record](https://doi.org/10.1007/s001860000043); [full author-uploaded paper](https://www.researchgate.net/publication/225743009_Steepest_descent_methods_for_multicriteria_optimization); [existing step-globalization report and protocol](../../lap_step_globalization_report.md).

## Static review of the pinned FS driver

Reviewed native_fs_batch.py at SHA-256 8d1b825d305014d6770e2888d71cc34391d54854fbfd8f211853561e0f9a9d3b. It pins the cursor-2 checkpoint, raw-gradient NPZ, sampling manifest and protocol; checks the sample identity; recomputes objective gradients through train_moo_update; and compares replayed gradients with the frozen NPZ. It enumerates all seven simplex faces for the three-objective minimum-norm problem, applies no per-task normalization, verifies positive dots against \(\bar g\), and maps the native step \(v=-\bar g\) through the adapter with alpha0=1. The comparison with the stored witness occurs only after the independent solve.

**KKT receipt caveat:** the code divides a maximum containing dimensionless simplex residuals and gradient-squared KKT residuals by the largest Gram entry. This is dimensionally mixed and can weaken the simplex check at large gradient scale. Review the raw simplex sum/nonnegativity and active/inactive KKT residual fields individually; the single scaled boolean is insufficient evidence by itself.

The normal success path snapshots an accepted candidate and then restores model/buffers and RNG, while EMA/cursor remain external inputs and are checked. Candidate saving and witness comparisons occur before that restore, outside a finally block; an exception on that path would leave the scratch model at the accepted point and may leave a partial artifact. Inspect the completion receipt for the normal-path rollback flags and file hashes. No FS numerical result, Armijo decision, panel transfer, or final acceptance is available yet.

## Independent review of hardened v2 run

Reviewed `native_fs_batch_v2.py` (SHA-256 `b24102c52e7c36cbb10b8f3a671b0f53fbc5c20eecc2b2c1c9a186c714dc1be3`), result `fs_direction_armijo_v2.json` (SHA-256 `05ea74cfea1f3f171783a4238f17b5f39925778fe2e97eebebb086fa5c9448cf`), and receipt (SHA-256 `9e0c61dda08fcfff288433b259e171af834f334e69dd1e20072a736b6de173be`). The independent solve recomputes the raw-gradient Gram matrix and seven simplex faces; its minimizer has weights `(0.0014191159015, 0, 0.9985808840985)`, support `{chem, op}`, and `||bar_g||=0.303215948844`. The active stationarity absolute residual is `3.98e-15`; inactive reduced-gradient violation is zero; simplex sum error and negativity are zero. Emitted float32 common-gradient dots are positive (`0.0919396711, 879.071572, 0.0919399104`), so all native step dots are strictly negative. This independently accepts the all-objective first-order geometry and adapter sign mapping.

The frozen run tried exactly 21 steps `t=1,...,2^-20` and rejected every one. At `2^-20`, chemistry's measured reduction was `-0.042989254` and operator's was `-2.18659e-6`; strict componentwise decrease and Armijo therefore fail, despite the favorable first-order slopes. There is no candidate snapshot; panel comparison and the conditional full-cycle gate are not applicable. The guard result does **not** establish that all practical steps fail.

The hardened result records same-helper zero-step losses exactly equal to gradient replay (all three deltas zero), and the receipt verifies rollback for model/buffers, RNG, EMA, and cursor. Independent simplex and gradient-KKT residuals are now separate, correcting the dimensional-rescaling caveat in the earlier v1 static review. The v2 normal and exception paths restore state in `finally`.

**One recommended next diagnostic, not run here:** perform a single frozen-state, local-response precision check before considering any globalization change. Reconstruct the same native FS vector from the pinned checkpoint/gradients; for a small predeclared set of already-tested `t` values, capture the *realized* parameter displacement after dtype rounding, its cosine with `t v`, and each raw-gradient dot with that realized displacement. Compare same-objective directional response with a float64 scratch evaluation from the identical base state (never commit parameters, EMA, or RNG). If the float32 realized direction loses common descent while the double-precision local response preserves it, precision/quantization is implicated; if the realized direction preserves all three negative dots yet the same deterministic objective rises, investigate local curvature or gradient/evaluator consistency. This is one discriminating diagnostic, not a new line-search sweep or a trust-region justification. The v2 trial field `parameter_delta_norm` is the intended `t||v||`, not the norm of the rounded realized displacement, so existing output cannot settle this distinction.

**Decision:** accept native FS KKT/strict common-descent geometry, same-batch Armijo rejection through the stated finite guard, and rollback/provenance. Do not claim a transfer result, full-cycle result, or absence of all practical steps. Keep existing task reductions and current direction methods unchanged; no hybrid or task reweighting is indicated by this result.
