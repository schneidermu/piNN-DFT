# Round-2 recommendation: isolate the direction failure

**Locked-task SHA-256:** `43e7ae9d4a54922f830fc7a90a87dbaab3707d732eb23543167d11b80027f9de`.

## Proposed mechanism

Run one frozen-state **Fliege–Svaiter multiobjective steepest-descent direction** test at cursor 2, retaining the existing componentwise vector Armijo rule. For the three already-computed raw gradients (g_C,g_E,g_{AO}), compute

\[
d_{FS}=\arg\min_d\max_{i\in\{C,E,AO\}}g_i^\top d+\tfrac12\|d\|^2
=-\sum_i\lambda_i g_i,\qquad
\lambda\in\arg\min_{\lambda\in\Delta_3}\|\sum_i\lambda_i g_i\|^2.
\]

This is the standard steepest common-descent construction of Fliege and Svaiter; when its minimum-norm convex-hull point is nonzero, its direction has strictly negative first-order slopes for all three objectives. The stored cursor-2 witness already establishes that a strict common-descent direction exists on this batch, while the current PCD direction has chemistry slope (+0.0093064), so the configured Armijo eligibility check correctly performs no trial. This changes only the candidate direction in a diagnostic replay; it does not alter PCD, any objective, or training policy. [Fliege & Svaiter (2000)](https://doi.org/10.1007/s001860000043); [Désidéri (2012), MGDA](https://doi.org/10.1016/j.crma.2012.03.014).

Fliege–Svaiter explicitly assumes no objective ordering or weights. Do not imply that its direction itself encodes chemistry priority. Preserve the established preference at the experiment boundary: chemistry improvement is the primary go/no-go endpoint, and a candidate is provisionally eligible only when the existing componentwise Armijo test also strictly decreases both (E_{xc}) and the AO-operator objective. Do not add a chemistry-weighted projection or combine this direction with PCD.

## Minimal protocol and signatures

1. Load the immutable cursor-2 NCCE31/BH checkpoint/EMA and the exact stored batch and raw task gradients. Verify receipt and source identities; do not advance cursor/RNG/EMA or touch optimizer state.
2. Solve the three-variable simplex QP above from the recorded (3\times3) Gram matrix. Record λ, direction norm, and all three raw predicted slopes. This costs only a tiny simplex problem once the gradients exist.
3. On that same batch and base state, try the existing line-search candidates (\alpha_j=\alpha_0 2^{-j}), (j=0,\ldots,20), with the same (c_1=10^{-4}), ρ=0.5, finite checks, and componentwise Armijo tests. Record predicted slopes and actual loss changes for every trial. No sample, EMA, model, PCD, or RNG commit is permitted.
4. If a common Armijo candidate exists, evaluate only that candidate and the cursor-2 base checkpoint on the same fixed 27-member chemistry/E/AO panels. Compare candidate/base panel ratios and win counts; no training continuation is part of this experiment. The locked cursor-2 ratios to predopt were chemistry median 1.01452 (11/27 wins; p90 2.4604), (E_{xc}) median 0.64867, and AO median 0.95511 (both 27/27 wins), so the paired candidate/base comparison must report chemistry first.

**Direction-failure signature:** the computed direction has all three negative slopes and at least one finite Armijo trial passes for all three tasks, unlike the PCD direction. That confirms the same-batch stop was caused by direction choice; it does not establish population improvement. **Chemistry remains decisive:** report the paired chemistry-panel change first; E/AO panel values remain companion safeguards, with no scalarized score. If the batch curve passes but the chemistry panel does not improve, the witness is sample-local and the next experiment should be larger/DB-balanced per-objective batches or a stochastic variance-reduction method while retaining PCD. [Mercier, Poirion & Désidéri (2018)](https://doi.org/10.1016/j.ejor.2018.05.064) study stochastic multiobjective objectives as expectations, supporting that as a later sample-estimation test.

## Why this first; deferred alternatives

The existing vector Armijo already cured the original overshoot along its PCD ray, and the first two adaptive updates decreased all three same-sample objectives. At cursor 2 the issue is different: the selected PCD direction is not common descent, despite a strict feasible direction on the exact batch. Reusing Armijo with a standard common-descent generator changes one cause at a time and is the smallest causal test. The direction QP is only three-dimensional; once gradients are available it uses their (3\times3) Gram matrix, stores no extra model-sized history, and requires at most 21 same-batch objective trial evaluations plus one paired-panel evaluation.

- **Larger or balanced batches / variance reduction:** best if the batch-only rescue fails the fixed chemistry panel, but it changes the estimator and direction together if tested first. Keep chemistry the first panel endpoint. The stochastic MGDA literature addresses expected objectives, not this demonstrated same-batch direction mismatch.
- **BBDMO/spectral:** Chen, Tang & Yang’s BBDMO uses Barzilai–Borwein scaling to address imbalanced objectives and very small Armijo steps. Here the observed event is a no-trial common-descent rejection, not a chronic tiny-step sequence; BB history and gradient rescaling would add confounds. [Chen et al. (2023)](https://doi.org/10.1016/j.ejor.2023.04.022).
- **Multiobjective trust region:** established for nonconvex multiobjective problems, with model/trust-radius updates based on predicted versus actual reduction; it adds a trust-region subproblem and radius/model policy that is unnecessary for this first one-batch direction diagnosis. [Carrizo, Lotito & Maciel (2016)](https://doi.org/10.1007/s10107-015-0962-6).

**Single follow-up trigger:** if this direction passes the frozen-batch curve but fails the paired chemistry panel, move next to a larger/DB-balanced or variance-reduced batch test with canonical PCD; that separates sample-to-panel variance from the now-demonstrated batch-local direction problem. Do not claim success from the two-update history or extrapolate this one batch to general training behavior.
