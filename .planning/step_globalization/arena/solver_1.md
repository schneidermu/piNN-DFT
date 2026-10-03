# First optimization test for PCD finite-step overshoot

## Recommendation

Test a **componentwise multiobjective Armijo backtracking line search along the already-emitted PCD direction first**. Keep tau=.02, chemistry-primary PCD, its EMA/normalization, the task losses, and model/operator fixed. Select the largest trial step that gives sufficient actual decrease in each of chemistry, integrated E_xc, and AO-operator loss on the current fixed batch. Chemistry remains preferred through the unchanged PCD direction; do not add weights or alter constraints.

This is the most direct causal test. In the reported five-step plain-SGD control, the direction and applied update were almost exactly antiparallel (median cosine -0.99999984), yet the second same-sample AE17/H2 step increased chemistry 56.64-fold and E_xc 21.78-fold despite all three local gradient–update dots predicting decrease. Stochastic sample variation is not required for that failure, and optimizer rotation is not required either. The observation points to a step leaving the region where the local directional derivative predicts the nonlinear loss change. Shrinking only the scalar step tests that claim while preserving the direction and all objectives.

## Candidate comparison

| Candidate | Fit to this evidence | Why it is not first |
|---|---|---|
| Vector Armijo backtracking | **Best first test.** Directly checks actual vector-loss changes along the exact PCD direction and shortens the finite displacement when local slopes stop predicting outcomes. Fliege and Svaiter establish descent methods for multicriteria optimization without scalarizing objectives or assuming weights; componentwise Armijo is a natural globalization for an existing common descent direction. | Each trial costs objective evaluation(s), but the one-step diagnostic requires no extra gradients or QPs. |
| Multiobjective trust region | Strong second option if backtracking repeatedly collapses the step or objective evaluations are costly. Vector trust-region methods compare actual and predicted reductions and adapt a radius, a good nonlinear-globalization mechanism (Carrizo, Lotito & Maciel, 2016). | It requires a model/radius and ratio-update policy. That is more machinery than needed to test whether the already observed full step is simply too large. |
| BB/spectral multiobjective descent (BBDMO) | Established method for tuning objective gradient magnitudes and addressing slow/small steps associated with objective imbalance (Chen, Tang & Yang, 2023). Consider only if measured Armijo steps become persistently tiny and imbalance is the bottleneck. | It changes the direction-finding problem and objective gradient scaling. That confounds this diagnosis and could enlarge steps in the presence of known overshoot. It targets small-step/imbalance behavior, not this demonstrated large-step failure. |
| Larger balanced batches / variance reduction | Could reduce noisy gradient variation; SVRG is explicitly designed to reduce stochastic-gradient variance (Johnson & Zhang, 2013). Useful if independent batches reveal direction noise after the step-size issue is controlled. | The damaging step used the same sample, so sampling variance is not necessary for the failure. A bigger batch changes gradients and cost but still gives no finite-step decrease check. |

## Smallest falsification protocol

Run a deterministic replay of the existing plain-SGD control, not a fresh sample. Use the same fresh predopt, seed-41 first stream entries and exact AE17/H2 batch, tau=.02, EMA beta=.999, epsilon=1e-8, and common initial alpha `6.632573669086685e-7`. Retain the control's no-momentum, no-decay, no-scheduler SGD. Reproduce update one unchanged, then snapshot parameters and all state immediately before the known problematic second update. Freeze that batch for every trial and compute the PCD direction and the three current raw task gradients once.

For candidate `alpha_j = alpha0 * 2^-j`, `j=0,...,20`, evaluate a temporary parameter vector `theta_trial = theta - alpha_j * d` without committing optimizer or model state. With `f_chem`, `f_exc`, `f_op` the exact existing component losses and `g_i` their current-batch gradients, record each directional slope and accept the largest alpha satisfying all three:

`g_i dot d > 0` and `f_i(theta_trial) <= f_i(theta) - 1e-4 * alpha_j * (g_i dot d)`, for every `i` in {chem, exc, op}.

Use one deterministic forward evaluation per candidate that returns all available component losses; do not redraw or resample between trials. If no representable nonzero trial passes, reject the update (leave theta unchanged) and record the failure. Keep raw values and ratios; do not normalize/reweight the acceptance tests. First confirm the original `j=0` trial reproduces the reported chemistry/E_xc blow-up while operator loss falls. Then report the accepted alpha, backtrack count, predicted versus actual reduction in each component, and parameter-delta norm.

As a non-tuning audit, compare the pre-step state and accepted candidate on the already declared 27-reaction / 15-system panel using its existing chemistry, E_xc and AO-operator evaluators. Record paired row-level changes and medians. Do not use this panel to select alpha. These are objective/loss checks, not held-out WTMAD claims. This audit distinguishes a same-batch safeguard from a step that also helps the locked panel.

## Decision signatures, cost, and next trigger

The overshoot hypothesis is supported if the unchanged full step reproduces the spike and a smaller Armijo step, with the same direction and sample, yields actual decrease in all three batch losses. Panel improvement in all three medians is the stronger positive result. It is falsified for this replay if the full-step spike reproduces but no representable trial satisfies componentwise sufficient decrease despite positive slopes; first check evaluator determinism and gradient/loss consistency. If the spike does not reproduce, the original result is not reproduced and no method comparison is interpretable.

The diagnostic adds at most 21 trial forwards, usually fewer, and no backpropagation/QP per trial. It is easy with plain SGD because candidates are temporary `theta-alpha*d` vectors. RAdamW should be a separate subsequent test: its reported median direction-to-actual-update angle is about 93 degrees, so scaling the PCD direction alone would not diagnose its transformed update.

If this replay supports overshoot and the panel improves, run a paired 25-update pilot with line search as the only change; report accepted-alpha distribution, rejected-step count, wall time, and paired chemistry/E_xc/operator medians against the fixed-alpha control. If accepted steps repeatedly require very heavy backtracking, test a vector trust-region globalization next. If batch losses pass but locked-panel metrics do not, investigate batch variance with larger balanced batches next. Test BBDMO only if persistent tiny accepted steps point to objective imbalance and a changed direction is acceptable.

## Primary literature

- Fliege, J. & Svaiter, B. F. “Steepest descent methods for multicriteria optimization.” *Mathematical Methods of Operations Research* 51, 479–494 (2000). [Publisher / DOI](https://doi.org/10.1007/s001860000043).
- Carrizo, G. A., Lotito, P. A. & Maciel, M. C. “Trust region globalization strategy for the nonconvex unconstrained multiobjective optimization problem.” *Mathematical Programming* 159, 339–369 (2016). [Publisher / DOI](https://doi.org/10.1007/s10107-015-0962-6).
- Chen, J., Tang, L. & Yang, X. “A Barzilai-Borwein descent method for multiobjective optimization problems.” *European Journal of Operational Research* 311(1), 196–209 (2023). [Publisher / DOI](https://doi.org/10.1016/j.ejor.2023.04.022); [author manuscript](https://arxiv.org/abs/2204.08616).
- Johnson, R. & Zhang, T. “Accelerating Stochastic Gradient Descent using Predictive Variance Reduction.” *NeurIPS* 26 (2013). [Proceedings paper](https://proceedings.neurips.cc/paper_files/paper/2013/hash/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Paper.pdf).
