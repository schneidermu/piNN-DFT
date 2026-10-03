# Direction recovery: frozen cursor-2 diagnostic

**Status: PCD geometry is CASE B. Independent review accepted the native FS direction, its strict common-descent geometry, and the v2 bounded Armijo rejection through `t=2^-20`. No candidate exists, so panel transfer and the conditional 27-entry cycle are not applicable. The single next diagnostic is a frozen-state realized-direction/precision audit; the current guard-limited result does not justify a trust-region change.**

The frozen source is repository commit `5b913c02545e4528562ffef69abda24cc999acb0`, cursor 2, sample `NCCE31/reaction 0 / BH / level2_mura`. The checkpoint and raw task gradients replayed exactly; the six tau cases reused the same raw gradients and scratch EMA normalization. Parameters, model buffers, incoming EMA, RNG, sampling cursor, checkpoint, and gradient artifact remained unchanged. No repository source or test files were changed.

## Answers to the nine questions

1. **Was tau=0.02 specifically responsible for losing chemistry descent at cursor 2?** The evidence implicates positive secondary pressure at this cursor, but does not isolate `0.02`: chemistry descent is positive at `tau=0`, while every tested positive tau from `0.001` through `0.02` has negative chemistry slope. The scan does not locate the transition below `0.001`.
2. **Does any smaller fixed canonical PCD tau yield strict first-order descent in chemistry, E_xc, and operator simultaneously?** No. Every tested positive tau has an adverse chemistry slope; `tau=0` has a numerically flat, slightly negative E_xc slope.
3. **If yes, does vector Armijo turn that direction into actual all-three decrease?** Not applicable: no positive tau passed the geometry gate, so no PCD direction was selected and no PCD Armijo trial ran.
4. **Does the one-step improvement transfer to the frozen panel?** There is no accepted PCD or FS candidate. The panel was not evaluated, so transfer is not established or failed.
5. **If PCD fails, does native Fliege–Svaiter produce a reproducible strict common-descent direction?** Yes. The independently recomputed raw-gradient simplex solution in v2 returned weights `(0.0014191159015, 0, 0.9985808840985)`, zero simplex residuals, active stationarity residual `3.98e-15`, zero inactive violation, and strict common-descent dots. Independent review accepted the sign mapping and KKT geometry.
6. **Does FS + vector Armijo produce actual all-three batch decrease?** No. Both v1 and hardened v2 rejected all 21 tested steps `t=1,...,2^-20`; independent review accepted this as a guard-limited rejection of the tested interval. It does not establish that every useful representable step fails.
7. **Does that FS step improve all three panel medians?** Not applicable: no candidate was accepted, so the panel was not run and this is not a panel-transfer failure. The 27-entry gradient-cycle gate was not triggered.
8. **Is the remaining bottleneck PCD secondary pressure, symmetric direction selection, finite-step curvature, or stochastic gradient estimation?** Positive secondary pressure is the verified cause of the PCD direction failure at this cursor. Native FS restores strict common-descent geometry, but its tested steps all fail the finite-step gate. The reviewer could not distinguish realized-direction precision from local objective response or derivative consistency, so curvature is not established. Stochastic estimation was not triggered.
9. **What is the single next experiment?** Run one frozen-state realized-direction/precision audit on a small predeclared subset of already-tested `t` values. Measure the rounded parameter displacement, its cosine with requested `t*v`, and each raw-gradient dot with the realized displacement; compare the same-objective response in a float64 scratch evaluation from identical inputs and reductions. Do not commit state or sweep new step sizes.

## Arena decision

The Arena skill was unavailable, so four independent Luna MAX solver memos were manually adjudicated against the locked task, literature, and repository evidence at the pinned HEAD. The judge ranking was:

| Rank | Candidate | Causal fit | Chemistry priority | All-objective descent | Literature fit | Minimal compute | Minimal code change | Interpretation |
|---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | Smaller fixed canonical PCD tau + vector Armijo | High | High | High | High | High | High | High |
| 2 | Native FS/MGDA + Armijo | High | Low | High | High | Medium | Medium | High |
| 3 | One full training-manifest gradient cycle + unchanged PCD | Medium | High | Medium | Medium | Low | High | Medium |
| 4 | PMGDA/EPO or trust region | Low | Medium | Medium | High | Low | Low | Medium |

The frozen geometry experiment was therefore run first. Its predeclared positive-tau geometry gate failed, authorizing the native FS/MGDA diagnostic. A symmetric FS direction is a bounded diagnostic and does not become the production rule automatically because it has no chemistry-priority guarantee.

## Frozen PCD geometry

The sign convention is `theta_trial = theta - alpha*d`; positive raw `g_i dot d` predicts descent. All QPs were feasible, with E_xc as the only active secondary constraint. The normalized direction coefficients below are ordered `(chemistry, E_xc, operator)`; normalized multipliers are ordered `(E_xc, operator)`. The raw direction norm is the norm of the emitted parameter-space direction.

| tau | Normalized coefficients | Multipliers | Active | Feasible / max KKT residual | Raw direction norm |
|---:|---|---|---|---|---:|
| 0 | (5.972754e-6, 1.349410e-7, 0) | (4.480599e-4, 0) | E_xc | yes / 0 | 144.977434 |
| 0.001 | (5.972754e-6, 4.361082e-7, 0) | (1.448060e-3, 0) | E_xc | yes / 6.51e-19 | 144.977427 |
| 0.0025 | (5.972754e-6, 8.878591e-7, 0) | (2.948060e-3, 0) | E_xc | yes / 0 | 144.977430 |
| 0.005 | (5.972754e-6, 1.640777e-6, 0) | (5.448060e-3, 0) | E_xc | yes / 0 | 144.977433 |
| 0.01 | (5.972754e-6, 3.146613e-6, 0) | (1.044806e-2, 0) | E_xc | yes / 0 | 144.977430 |
| 0.02 | (5.972754e-6, 6.158286e-6, 0) | (2.044806e-2, 0) | E_xc | yes / 0 | 144.977435 |

| tau | Raw dots `(chemistry, E_xc, operator)` | Cosines `(chemistry, E_xc, operator)` | Angle from raw chemistry gradient |
|---:|---|---|---:|
| 0 | (15310.600, -0.000919, 5.671532) | (0.728436, -1.44e-9, 0.106713) | 43.245° |
| 0.001 | (-6415.389, 575414.721, 47.122250) | (-0.305226, 0.902790, 0.886633) | 107.772° |
| 0.0025 | (-11279.530, 626107.774, 49.681170) | (-0.536649, 0.982324, 0.934780) | 122.456° |
| 0.005 | (-12882.922, 634500.533, 49.809190) | (-0.612934, 0.995492, 0.937189) | 127.802° |
| 0.01 | (-13655.163, 636651.978, 49.708200) | (-0.649675, 0.998867, 0.935289) | 130.517° |
| 0.02 | (-14031.363, 637193.290, 49.615409) | (-0.667573, 0.999716, 0.933543) | 131.880° |

## Native FS direction and v2 Armijo result

The separate native FS solver minimized the raw-gradient convex-hull norm without PCD EMA normalization or objective rescaling. V1 and v2 independently returned weights `(0.0014191159014695647, 0, 0.9985808840985304)`. In v2, the simplex sum error and nonnegativity violation were zero; active stationarity residual was `3.98e-15`, inactive reduced-gradient violation was zero, and the separately scaled gradient KKT residual was `2.06e-22`. The minimum-norm point had norm `0.303215948844`. Under `bar_g = Σ λ_i g_i`, the raw dots were `(0.0919396711, 879.0715722, 0.0919399104)`; for native step `v=-bar_g`, every `g_i dot v` is strictly negative. The independently recomputed weights match the stored witness, which was used for comparison only. The independent review accepted the geometry and adapter sign convention.

The existing componentwise Armijo path tested `t=1, 1/2, ..., 2^-20` (21 values; `c=1e-4`, `rho=0.5`) in v1 and v2 and accepted none. At `t=2^-20`, v2 reported reductions `(chemistry, E_xc, operator) = (-0.042989254, +0.000653889, -2.18659e-6)` while predicted chemistry reduction was `+8.77e-8`. The same-helper zero-step losses matched the gradient-pass losses exactly; all three zero-step deltas were zero. V2 saved no candidate and its receipt verifies rollback of model/buffers, RNG, EMA, and cursor. Its `parameter_delta_norm` field is requested `t*||v||`, not the norm of the rounded realized parameter displacement; measuring the latter is the purpose of the next diagnostic.

Independent review accepts this as a guard-limited rejection on the tested interval. It does not prove that all useful representable steps fail or isolate local curvature from realized-direction precision/response. The recommended next diagnostic measures the rounded realized parameter displacement and its dot products at a small predeclared subset of already-tested `t` values, then compares the same local response in a float64 scratch evaluation from the identical state. This is not a new step-size sweep and does not justify a trust-region change.

## Decision basis and literature scope

The smallest chemistry-priority test was the frozen-state PCD tau geometry map. Its result is CASE B: no tested small positive tau provides robust strict descent for all three objectives. Native FS independently reproduced strict common-descent geometry, while both bounded Armijo runs rejected all 21 tested steps. The independent review accepted the guard-limited result and recommends one realized-direction/precision diagnostic before classifying the step response. Fliege–Svaiter supplies a symmetric common-descent construction and componentwise Armijo search, but no chemistry-priority guarantee.

PCD v2 Proposition 4.7 provides a two-objective (`K=2`) directional expression; it does not give a three-objective strict-descent guarantee. The K=3 result must be read from the measured canonical QP geometry. The PCD claims apply to its normalized direction and idealized first-order step, not a general convergence guarantee for the deployed stochastic trainer. PMGDA's predict/correct method can allow objective deterioration in its correction phase and needs an explicit preference specification, so it is not a drop-in strict common-descent replacement. The bounded Armijo rejection does not justify a trust-region change because the realized displacement has not been checked; first run the reviewer’s precision/response audit. A complete 27-entry stochastic-gradient diagnostic is conditional only on same-batch FS Armijo passing and the paired panel gate failing.

Primary sources: [PCD v2, Proposition 4.7](https://arxiv.org/abs/2606.29521); [Fliege & Svaiter (2000)](https://doi.org/10.1007/s001860000043); [PMGDA](https://arxiv.org/abs/2402.09492); [Carrizo et al., multiobjective trust regions](https://doi.org/10.1007/s10107-015-0962-6); [Mercier et al., stochastic MGDA](https://doi.org/10.1016/j.ejor.2018.05.064).

## Provenance and validation

Frozen artifacts are outside the Git repository. The FS v2 result JSON SHA-256 is `05ea74cfea1f3f171783a4238f17b5f39925778fe2e97eebebb086fa5c9448cf`; v2 receipt JSON is `9e0c61dda08fcfff288433b259e171af834f334e69dd1e20072a736b6de173be`; executed v2 driver is `b24102c52e7c36cbb10b8f3a671b0f53fbc5c20eecc2b2c1c9a186c714dc1be3`. The FS v1 result JSON SHA-256 is `4a14bdd794428008c3d3778d4a6169ff7b878145a073b5fa0e45ecc695ad8454`; FS v1 receipt JSON is `ba5b44ed5fbb694f0b7e4173d943942781da06781968a9b322eaaea063d77675`; FS v1 executed driver is `8d1b825d305014d6770e2888d71cc34391d54854fbfd8f211853561e0f9a9d3b`. The geometry JSON SHA-256 is `34d48cad8c26718a59e0c31feeb79536569a0ac2bbbaed3ba595408b72a58071`; replay receipt JSON is `30f8ade82bc4f09ccd85790d2dac557b1081420e1f45a8bc26e2107a2116197c`; executed frozen-geometry driver is `151e8d4a9c36671632fcc0a2e1ead26db7a3d3011876fd5408a7f6772d9f7882`. The source replay receipt is `4ab29379dfafd5c03dffa41cbf415591a3e505e7c63d9ad82a62bdc937562475`; the stored common-descent witness is `8ebe9435089a57b47bdc737b1608c7642f1b7509234a93104c8dd2b959d80aab`. The cursor-2 checkpoint SHA-256 is `a53f46807b1ce492c8fe02657e615881ff310294bd79d273eb4bb0c694e3670c`; the raw task-gradient NPZ SHA-256 is `fd5eb469339d68ac3c7fc21728ce0138cec878ea8780c7c05277d87cf353f6c0`.


The historical FS v2 invocation used the repository as its working directory and the `libxc` conda environment. **Do not rerun this command to inspect the existing result:** it writes to the frozen external result and receipt paths above.

```powershell
Set-Location 'C:\Dev\readWFN_share_ms\lap_full_vxc'
$env:OMP_NUM_THREADS = '1'
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
$env:PYTHONPATH = 'train_models'
$env:PYTHONIOENCODING = 'utf-8'
& 'C:\Users\schne\miniconda3\envs\libxc\python.exe' 'C:\Dev\readWFN_share_ms\lap_direction_recovery_runs_20261003\native_fs_batch_v2.py'
```

Focused Windows validation: 58 passed, 2 skipped. WSL/PySCF operator and unit validation: 14 passed with one existing Torch deprecation warning. `compileall` for `train_models` and `git diff --check` passed. The frozen external PCD driver has a Ruff F401 (`TASK_NAMES` unused); it is an immutable, out-of-repository diagnostic artifact, not production source. Do not report Ruff as fully passing. `python -m ruff check --no-cache` passed for the external FS v2 driver. No production source or test files were changed.
Independent FS review: `.planning/direction_recovery/independent_fs_review.md` accepted the v2 geometry, guard-limited Armijo rejection, zero-step helper equality, and rollback. It recommends the precision/realized-direction audit as the single next diagnostic.
