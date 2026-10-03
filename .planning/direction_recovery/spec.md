# Frozen cursor-2 PCD tau geometry and step gate

## Inputs and replay gate

Use the exact seed-41 cursor-2 checkpoint, NCCE31 reaction 0 / BH / `level2_mura`, pre-update PCD EMA, model buffers, RNG, and original raw objective definitions. The recorded checkpoint SHA-256 is `a53f46807b1ce492c8fe02657e615881ff310294bd79d273eb4bb0c694e3670c`; the raw task-gradient receipt SHA-256 is `fd5eb469339d68ac3c7fc21728ce0138cec878ea8780c7c05277d87cf353f6c0`. Verify identities from the source protocol before use. First reproduce tau `0.02` and its three recorded slopes. Any identity or replay mismatch stops the run without method conclusions.

Compute the would-be EMA normalization once from the frozen pre-update EMA and the frozen gradients. Use an isolated scratch copy for each QP; parameters, persistent EMA, RNG, cursor, optimizer, scheduler, batch, and losses remain unchanged across the map. Recompute canonical K=3 PCD at exactly `tau = 0, 0.001, 0.0025, 0.005, 0.01, 0.02`; do not interpolate or add tau values.

## Geometry receipt

For every tau, record normalized coefficients/multipliers, active constraints, feasibility and KKT residual, emitted raw-direction norm, raw-gradient dots, normalized cosines against each task gradient, and angle from the raw chemistry gradient. Use PCD direction convention: `theta_trial = theta - alpha*d`; positive `g_i·d` predicts descent, equivalent to negative `g_i·(theta_trial-theta)`. Compute dots/cosines from the exact emitted direction with float64 reductions. Compare the raw-gradient dots with the stored Fliege–Svaiter witness.

At tau zero, independently project the frozen normalized chemistry gradient onto the intersection of the two secondary non-ascent halfspaces. Compare that projection with the canonical QP direction; report feasibility/KKT residual and check `g̃_chem·d̃ ≈ ||d̃||²`. Do not require strict secondary progress at tau zero.

The geometry gate is `g_i·d/(||g_i||·||d||) >= 1e-6` for each nonzero raw task gradient. This is a predeclared numerical-margin guard, not a PCD parameter; it is about 81 times the `1.23e-8` float32 cosine-residual magnitude recorded in the prior PCD study. Select the smallest positive tested tau passing all three margins with a feasible QP. Selection uses frozen geometry only. If no positive tau passes, follow the locked Fliege–Svaiter fallback.

## Finite step and transfer

On the selected frozen PCD direction, reuse the validated componentwise vector Armijo implementation, its existing `alpha0`, `c=1e-4`, `rho=0.5`, rollback, and finite-precision guard. Search shared fractions `1, 1/2, …, 2^-20`; accept the largest passing fraction that gives strict actual decrease and Armijo for all three original batch losses. Do not select tau or alpha from panel values. Restore the exact snapshot after every trial and make no optimizer/EMA/RNG/cursor commit.

Only after a same-batch pass, evaluate that single accepted candidate against the base on the frozen 27-row/15-system panel. Report chemistry first, then E_xc and operator: median candidate/base ratio, p90, max, and row wins. Transfer passes only when every objective median is strictly below 1. A batch pass with panel failure triggers exactly one conditional diagnostic: accumulate each objective's gradients over one complete immutable 27-entry training-manifest cycle, using training samples only; recompute canonical PCD with the unchanged selected tau, EMA rule, task reductions, and weights; then test direction geometry and the existing componentwise vector Armijo on that same training cycle. Require strict all-three directional descent and the ordinary Armijo gate. Keep the evaluation panel out of gradient formation and line-search selection. Do not begin long training.

If no positive PCD geometry passes, the next diagnostic is native Fliege–Svaiter: independently solve the raw-gradient simplex minimum-norm problem, verify residuals and strict dots, then use its native `theta_trial = theta + t*v`, `t=1,1/2,…`; do not reuse PCD `alpha0`. No state commit. Do not start long training.
