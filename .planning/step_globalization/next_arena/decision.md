# Round 2 Arena decision

**Locked-task SHA-256:** `43e7ae9d4a54922f830fc7a90a87dbaab3707d732eb23543167d11b80027f9de`
**Decision:** run one frozen-state Fliege–Svaiter common-descent direction diagnostic at cursor 2, retaining the validated componentwise vector Armijo test. This is a proposal only; it has not been run.

| Rank | Candidate | Causal fit, cost, and disposition |
|---:|---|---|
| 1 | Native Fliege–Svaiter direction + existing vector Armijo | Best isolation: the raw-gradient receipt proves this exact batch has a strict common-descent direction while the chemistry-primary PCD ray has positive chemistry slope. It changes only the diagnostic direction, reuses the existing gate, and requires one 3-variable simplex solve, at most 21 same-batch trials, and one paired-panel evaluation. Keep it diagnostic: the symmetric direction encodes no chemistry preference. |
| 2 | One full 27-entry training-manifest cycle of per-objective gradient accumulation, canonical PCD unchanged | Preserves PCD's chemistry-primary rule and tests sample-composition sensitivity, but changes the gradient estimate and costs one full-cycle accumulation. Best follow-up if a same-batch direction rescue does not transfer to chemistry on the paired panel. Preserve existing task reductions and weights; use training examples, not panel rows, to form gradients. [Mercier, Poirion & Désidéri (2018)](https://doi.org/10.1016/j.ejor.2018.05.064). |
| 3 | Multiobjective trust region | Literature-supported for predicted/actual vector decrease, but adds a model, subproblem, and radius policy; the observed cursor-2 event is an adverse direction with zero line-search trials, not a radius failure. [Carrizo, Lotito & Maciel (2016)](https://doi.org/10.1007/s10107-015-0962-6). |
| 4 | BBDMO/spectral scaling | Addresses objective imbalance and very small line-search steps, while this record has one five-halving rescue and a later positive chemistry slope. Less directly matched and changes direction geometry. [Chen, Tang & Yang (2023)](https://doi.org/10.1016/j.ejor.2023.04.022). |

## Protocol and decision gates

Restore the immutable seed-41 cursor-2 checkpoint, exact NCCE31/BH batch, and pre-update EMA. Reproduce the recorded raw gradients and PCD slopes first. From the same three raw gradients, solve

\[
\lambda^*\in\arg\min_{\lambda\ge0,\;\mathbf1^T\lambda=1}\left\|\sum_i\lambda_i g_i\right\|^2,
\qquad v=-\sum_i\lambda_i^*g_i.
\]

This is the native steepest common-descent construction: Fliege and Svaiter prove strict common descent away from Pareto-criticality, but assume neither objective ordering nor weights ([2000, *Mathematical Methods of Operations Research* 51, 479–494](https://doi.org/10.1007/s001860000043)). Verify all three raw slopes are negative; disagreement with the stored witness is a replay/identity failure, not a method result.

On that unchanged state and batch, follow the native Fliege–Svaiter schedule \(t_j=2^{-j}\), \(j=0,\ldots,20\), starting at \(t_0=1\), with the current \(c=10^{-4}\), \(\rho=0.5\), and strict actual decrease plus componentwise Armijo for chemistry, E_xc, and operator. The trial is \(\theta+t_jv\); do not reuse PCD's calibrated \(\alpha_0\), because \(v=-\sum_i\lambda_i^*g_i\) is already the native direction. Accept the largest passing trial. The numeric Armijo values are supported by Mita, Fukuda & Yamashita (2019, [DOI](https://doi.org/10.1007/s10898-019-00802-0)); they are not Fliege–Svaiter defaults. Reuse the existing implementation's 20-halving finite-precision/runtime guard only to bound the diagnostic; it is not a paper parameter. If exhausted, reject without state change and classify as guard-limited. Make no model, EMA, RNG, cursor, optimizer, or scheduler commit.

If a trial passes, compare only that candidate with the cursor-2 base on the fixed diagnostic panel (27 reaction rows, 15 cached systems). Report paired chemistry change first; all three panel objectives must strictly improve for a three-task success. If any panel objective fails to strictly improve, transfer of this one-step direction rescue is not established. This does not imply general PCD failure or establish training success.

**One conditional fallback:** only after a same-batch all-three Armijo pass fails the paired panel's strict all-three gate, accumulate one full 27-entry cycle from the immutable training manifest, then recompute canonical PCD once with unchanged EMA/hyperparameters and existing task reductions/weights. All three panel objectives must strictly improve; chemistry is reported first among successful three-objective solutions. Keep panel rows evaluation-only and retain the same three-loss acceptance gates. If the diagnostic has negative slopes but no representable Armijo point within the guard, stop for numerical replay review; do not reinterpret that as evidence for the fallback.

## Evidence boundary

At cursor 2, PCD's proposal slopes are chemistry `+0.0093064049`, E_xc `−0.4226231444`, operator `−0.0000329078`; no trial or update was made. The independent convex-hull witness has weights `(0.001419116, 0, 0.998580884)` and gives the direction `−c` strict dots `(−0.0919399, −879.071583, −0.0919399)`. Thus the batch is not Pareto-critical, but feasibility alone does not show a finite Armijo step or panel benefit. The cursor-2 chemistry panel median ratio is `1.01452039` (11/27 row wins), versus E_xc `0.64866504` and operator `0.95511256` (27/27 each). These are 27 reaction-row summaries over 15 cached systems, not 27 independent molecules, and only two adaptive updates exist. No 5/10/25/100 gate, training-success, SCF, or Slurm claim follows.
