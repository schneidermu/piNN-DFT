# Arena decision: PCD finite-step globalization

All four proposals choose same-ray, componentwise Armijo. I select that as the sole primary experiment: it directly tests the plain-SGD second-step overshoot while preserving PCD direction, task order, and objectives. Ranks assess protocol quality, not different method families.

| Rank | Proposal | Causal / literature / falsifiability | Cost / trainer / interpretability / diff | Judgment |
|---:|---|---|---|---|
| 1 | Solver 2 | H / H / H | H / H / H / M | Clearest fixed-state test and slope gate; add strict finite-decrease and float32 guards. |
| 2 | Solver 4 | H / H / H | H / H / H / M | Nearly tied; 2^-12 cap may stop before useful falsification. |
| 3 | Solver 1 | H / H / H | M / H / H / M | Paired panel audit is valuable follow-up work, extra cost for the first causal probe. |
| 4 | Solver 3 | H / H / H | M / H / H / M | c=.1 is valid, but panel improvement as an initial pass gate mixes mechanism and transfer. |

## T2 protocol and gates

Reconstruct the fixed-LR five-update SGD reference from the fresh predopt and seed-41 stream. Replay update 0 **unchanged** (pTC13 reaction 12 / BeH2), then probe the exact pre-update state for update 1 (AE17 reaction 16 / H2). The receipt lacks pre-update weights/EMA, so reconstruct them. Preserve tau=.02, task order `chem, exc, op`, EMA beta=.999/epsilon=1e-8, dtype, `eta0=6.632573669086685e-7`, and plain SGD (no momentum, decay, or scheduler).

From this snapshot, compute the three task gradients and canonical PCD direction once. Set `Delta0=-eta0*d`; log raw-loss slopes `g_i·Delta0` and proceed only when all are finite and strictly negative. PCD need not be common descent at every state. Freeze batch/stochastic state, parameters, buffers, and pre-update EMA. Score temporary candidates `theta+t*Delta0`, restoring the same snapshot each time, for `t=1,1/2,…,2^-20`; stop if float32 rounding makes the entire candidate vector identical to base. The j=20 limit/no-op stop are numerical guards, not source-algorithm defaults.

For each trial log predicted reduction `-t*g_i·Delta0`, actual change, ratio, finiteness, norm, and `t`. Accept the largest trial satisfying both `F_i(theta+t*Delta0) <= F_i(theta)+c*t*g_i·Delta0` and strict actual decrease, for every unchanged raw loss `i∈{chem,exc,op}`. Predeclare `c=1e-4`, `rho=1/2`. Fliege–Svaiter specify the componentwise rule, strict common descent, start `t=1`, halve, and `0<c<1`; they give no numeric default or cap. Mita, Fukuda & Yamashita (2019, Algorithm 1, §7) independently use `delta=1e-4`, `rho=.5`; these are not Fliege–Svaiter defaults. The fixed-batch test does not inherit the paper's global convergence theorem: PCD solves a different direction problem. [Fliege–Svaiter (2000)](https://doi.org/10.1007/s001860000043); [Mita et al. (2019), author PDF](https://optimization-online.org/wp-content/uploads/2018/09/6804.pdf).

**T2 pass:** `t=1` reproduces the chemistry/E_xc spike and operator decrease; slopes are all negative; some `t<1` is finite and passes Armijo plus strict all-three decrease. Report the rescue, backtracks, every predicted/actual triplet, and step norm. A batch pass is not a panel/generalization claim. If no representable fraction passes or `t=1` does not reproduce, mark failed/inconclusive, restore the control snapshot, and do not commit. Never call `train_moo_update` in trials or mutate optimizer, EMA, scheduler, RNG, cursor, or checkpoint before the gate.

Keep two distinct first-five traces. For the historical reference, restore the update-1 base after probing, apply its original `t=1` step, and replay stream updates 2–4 unchanged. The T2 curve is an added counterfactual at that same update-1 base. Only after T2 passes, start a separately labeled safeguarded branch by replaying update 0 unchanged and committing the accepted update-1 candidate once; for updates 2–4, recompute PCD, recheck all three slopes, and use the same bounded rule. Stop that branch on any adverse slope or rejected search rather than silently advancing EMA/cursor. After the first rescue, score the next two smaller representable fractions as non-selecting curve points (within the cap). Use the balanced panel only as a later transfer check, never to select `t`.

## Single fallback

If all slopes are negative but the bounded search finds no accepted representable step, or later fixed-stream fractions repeatedly collapse to the precision floor, test one fallback: **multiobjective trust region** with vector actual/predicted reduction (Carrizo, Lotito & Maciel, 2016, [DOI](https://doi.org/10.1007/s10107-015-0962-6)). First audit deterministic evaluation, reductions, and float32 movement. An adverse slope does not trigger this fallback; it means this ray is not a common descent direction. Defer BB until tiny steps implicate task-scale imbalance, and batch enlargement until balanced batches show sampling instability.
