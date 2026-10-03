# Literature contract: multiobjective Armijo on PCD updates

## Exact Fliege–Svaiter rule

The original [Fliege and Svaiter (2000) paper](https://doi.org/10.1007/s001860000043) considers a continuously differentiable objective vector `F: R^n -> R^m`, with vector inequalities interpreted componentwise. For a trial direction `v`, require strict common descent, `JF(x) v < 0` in every component, then accept the largest `t` from `{1, 1/2, 1/4, ...}` satisfying

\[
F(x+tv)\le F(x)+\beta_A t JF(x)v
\]

componentwise. The paper specifies `0 < beta_A < 1`, starts at `t=1`, and halves until acceptance (a contraction factor of `1/2` if called `rho`). It gives no numeric default for `beta_A`. Its `sigma in (0,1]` is a direction-subproblem accuracy tolerance, not the Armijo parameter. So `beta_A=1e-4` may be a reasonable experiment choice, but it is not the paper's default; `0.5` is also admissible but likewise not a stated default.

The paper has no `alpha_min`, backtracking cap, or rejection-on-cap rule. Finiteness follows analytically: if `F` is differentiable and `JF(x)v < 0`, every sufficiently small positive `t` satisfies the condition, so geometric halving eventually accepts. A numerical floor or finite cap is an implementation guard and departs from that exact algorithm. If reached, log rejection rather than taking an unaccepted step or claiming the paper's guarantee applies.

## Numerics-backed first-trial constants

Predeclare `beta_A = 1e-4`, `rho = 0.5`, and initial `t = 1` for the falsification run. These are experiment choices, not Fliege–Svaiter defaults. A direct multiobjective precedent is Mita, Fukuda, and Yamashita, “Nonmonotone line searches for unconstrained multiobjective optimization problems,” *Journal of Global Optimization* 75(1), 63–90 (2019), [DOI](https://doi.org/10.1007/s10898-019-00802-0), [author preprint](https://optimization-online.org/wp-content/uploads/2018/09/6804.pdf): Algorithm 1 states the monotone vector Armijo condition (preprint p. 4), and §7 includes the usual monotone line search among its numerical methods and reports `δ = 10^-4`, `μ = 1`, `ρ = 0.5` (preprint p. 22). Here `μ=1` is the initial trial `t=1`; the proposed experiment uses only the monotone all-components condition, matching the requirement that chemistry, `E_xc`, and AO operator each improve on the accepted finite step.

For PCD's update `theta_new = theta - eta0*d`, define the line-search direction as the full initial proposal `v = Delta0 = -eta0*d`, then start at `t=1`. This retains the calibrated learning-rate proposal while matching the paper's starting step. Check `grad(F_i)(theta) dot Delta0 < 0` for chemistry, E_xc, and operator, and test the original unnormalized losses componentwise. Equivalently, for raw PCD `d`, test `F_i(theta - eta0*t*d) <= F_i(theta) - beta_A*t*eta0*(grad(F_i)(theta) dot d)`. Use the same objective data and direction for slopes and trial losses. If any slope is zero or adverse, shrinking a positive scalar along that proposal cannot guarantee strict common descent for all tasks.

## Theorem scope and stochastic limit

The local Armijo-existence result needs only differentiability and a strict common-descent direction. It therefore supports a fixed-sample test on the deterministic batch objective `F_B`, if the same sample defines each slope and trial loss. The paper's global convergence theorem is narrower: directions must solve its native steepest-direction problem exactly or meet its stated `sigma` approximation condition; `F` is continuously differentiable; and accepted objective vectors decrease componentwise. It concludes every accumulation point is Pareto critical; bounded componentwise initial level sets additionally give a bounded sequence with an accumulation point. PCD solves a different chemistry-prioritized, normalized-gradient QP. Favorable PCD slopes make a local line search valid, but do not make its direction satisfy the native QP condition or inherit the global convergence theorem. PCD `tau=.02` is distinct from Armijo `beta_A`, native direction tolerance `sigma`, and the paper's optional stopping tolerance.

The original algorithm stops at a Pareto-critical point; its practical suggestion is a separate threshold on the native steepest-subproblem value. Accepted-step success is not itself a convergence stop. Deterministic theory also does not transfer to trial losses or slopes measured on independently redrawn minibatches. Freezing a batch makes this a deterministic line search for that batch only; it is no population or held-out guarantee. [Paquette and Scheinberg (2020)](https://doi.org/10.1137/18M1216250) adapt Armijo to a scalar stochastic objective using dynamically adjusted accuracy with probabilistic guarantees; that is useful context, not a direct theorem for three-objective PCD.

## Relevant alternatives

- [Mita, Fukuda, and Yamashita (2019)](https://doi.org/10.1007/s10898-019-00802-0) give max/average and multiobjective hybrid nonmonotone line searches. These permit some objective increases, so they conflict with an experiment requiring every accepted finite step to improve chemistry, E_xc, and operator. They are a later convergence-speed option, not the clean first test.
- [Chen, Tang, and Yang (2023)](https://doi.org/10.1016/j.ejor.2023.04.022) propose BBDMO, using BB scaling in direction construction to address objective imbalance; they report standard multiobjective Armijo can produce very small steps under imbalance. BBDMO changes the direction subproblem, so it is relevant if measured backtracks are repeatedly bottlenecked by one task's scale, not as an explanation of this fixed-ray overshoot.
- [Carrizo, Lotito, and Maciel (2016)](https://doi.org/10.1007/s10107-015-0962-6) formulate nonconvex multiobjective trust-region globalization using vector decrease/predicted reduction and radius updates. Consider this if line-search acceptance is unstable or actual changes repeatedly disagree with local predictions; it costs more model machinery.

## Synchronous DDP contract

Use one frozen global proposal and one shared step. At each trial, all-reduce each objective's loss sum and sample count, then test the count-weighted global means (a mean of rank means is equivalent only for equal counts). Use the globally averaged task gradients for the common slopes. Every rank must compare the same global loss vector and use the same accept/reject result and multiplier; synchronize the chosen halving count. Rank-local acceptance is invalid because ranks can diverge in parameters.

Cache the parameter/optimizer/EMA snapshot, direction, and proposal once. Trial forwards must not advance optimizer moments, EMA, scheduler, training step, or sample stream. Commit an accepted update and advance state exactly once; on capped rejection, commit no parameter or optimizer update and record the rejection. Checkpoint search settings and the accepted multiplier/backtrack count. For RAdamW, line-search its actual proposed parameter displacement and first verify all three global raw-loss slopes are negative; PCD's raw-direction slope conditions do not automatically survive optimizer preconditioning.
