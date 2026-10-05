# Cheap dyadic q-window localization

**CASE B: h2 is the largest validated usable window in the frozen ladder.** Worst h2/h3 disagreement is 4.1282291058404454%, below 5%; no slope sign changes occur in this pair. Parameter controllability remains unresolved: no parameter Jacobians were computed.

## Independent review and scope

Independent GPT-6 Luna MAX review of the failed probe: PASS before new evaluation. It verified q conversion, fixed-rho raw-Laplacian perturbation, sigma convention, contrast normalization, row ordering, exact 12 states, probe shape/dtype, 34 immutable hashes and the stopped disposition. Models are widened before local arithmetic; no F32 output is cast afterward. Independent review of the new window result also passes.

Starting commit: `8f69b40f3e22f72d774bddf3158ec5e0469e0023`. Production changes: zero. Seeds11/23/41, P67/P536/PBE_TANGENT/PBE_HEAD. The exact 64-point probe SHA is `bcc3d01872d0181b197800e2a0295178224fe1d81e9ebfb2b8b8e2f64437b81a`. All 64 points were evaluated; the failed receipt used eight fixed validation points. The metric formula and maximum-over-state/spin rule are unchanged. The original eight-row errors are also retained, so the historical 100.208% result is not confused with the full64 value53.893% for h_f/h1.

## Frozen ladder and numerical path

Widths are dimensionless unbounded q half-widths, alpha/beta respectively:

- h_f = `[0.09287594376391345, 0.17011650277242574]`; exact hex serialization: `['0x1.7c6b7c50d17bcp-4', '0x1.5c660a7f56f64p-3']`.
- h1 = `[0.046437971881956724, 0.08505825138621287]`.
- h2 = `[0.023218985940978362, 0.042529125693106434]`.
- h3 = `[0.011609492970489181, 0.021264562846553217]`.

The external protocol was saved before evaluation. Only these widths were evaluated; h_f is the repeated baseline. Existing `probe.model_for`, `qden`, `energy`, model descriptors and F_PBE are reused unchanged. Source F32 data/checkpoint numerical values are widened to F64; model inputs, parameters, forward and epsilon remain F64. No AO or objective evaluation is involved. Smaller h requires no implementation change to the evaluator.

Here C is the established epsilon_xc observable, and S(h)=(epsilon(q+h)-epsilon(q-h))/(2h), exactly the prior normalized contrast. It is not a finite difference of an already finite-differenced observable. Each spin is displaced independently, with all other inputs fixed. Two deterministic evaluations are performed for every state/window.

## Linearity and sign consistency

For each state/spin block, disagreement is norm(S_large-S_small)/max(norm(S_small),1e-12). The authoritative aggregate is the maximum over all24 blocks, not a pooled ratio that can mask one state.

| Pair | Maximum disagreement | Median state | p90 state | Worst state | Sign changes | Signal resolved |
|---|---:|---:|---:|---|---:|---|
| h_f/h1 | 53.892914% | 12.859177% | 40.218165% | 11 P536 | 10 (6 states) | True |
| h1/h2 | 15.844966% | 4.019354% | 11.816197% | 11 P536 | 0 (0 states) | True |
| h2/h3 | 4.128229% | 1.074796% | 3.088996% | 11 P536 | 0 (0 states) | True |

Every non-head state/spin error decreases monotonically along the three adjacent comparisons. h1/h2 fails, so h1 cannot be selected. h2/h3 passes. h3 lacks a smaller reference and is not selectable. Pairwise state medians/p90/worst identities, exact sign counts and sign-change energy fractions are in the JSON.

| State | h_f/h1 alpha/beta % | h1/h2 alpha/beta % | h2/h3 alpha/beta % |
|---|---:|---:|---:|
| 11 P67 | 17.2483701/41.2976585 | 4.88632814/12.1099455 | 1.26641216/3.15277031 |
| 11 P536 | 22.9074018/53.8929136 | 6.60162927/15.8449661 | 1.72006305/4.12822911 |
| 11 PBE_TANGENT | 13.5719864/22.9749709 | 4.04909056/9.05998098 | 1.0573851/2.51502662 |
| 11 PBE_HEAD | 0/0 | 0/0 | 0/0 |
| 23 P67 | 3.7302889/9.85500186 | 0.963023609/2.74060384 | 0.242821125/0.705166738 |
| 23 P536 | 13.349807/30.502726 | 3.60480264/8.87768121 | 0.91943291/2.31138988 |
| 23 PBE_TANGENT | 4.06572782/11.7457305 | 1.15142514/4.23475432 | 0.297567847/1.1772512 |
| 23 PBE_HEAD | 0/0 | 0/0 | 0/0 |
| 41 P67 | 7.3929425/13.9726235 | 1.97551861/3.80395381 | 0.502558452/0.972340188 |
| 41 P536 | 16.3477366/27.4793042 | 4.69705924/9.17246328 | 1.21850741/2.46821801 |
| 41 PBE_TANGENT | 3.91900724/9.12535563 | 1.05471624/2.59377363 | 0.269058691/0.671822651 |
| 41 PBE_HEAD | 0/0 | 0/0 | 0/0 |

## Signed slope convergence

Equal-weight pooled state/point/spin descriptive statistics; zero-head controls remain included. A zero median reflects signed cancellation and the three structural zero-head controls, not an absence of non-head response.

| Window | Median S | p10 S | p90 S |
|---|---:|---:|---:|
| h_f | 0 | -0.000281340064 | 0.000282497602 |
| h1 | 0 | -0.0002891367 | 0.000276504451 |
| h2 | 0 | -0.000281091868 | 0.000275780436 |
| h3 | 0 | -0.000277271418 | 0.000274534031 |

Per-state slope statistics are retained in JSON. The comparison shows stable block-norm convergence over the tested ladder; it is not a proof of asymptotic convergence at every point.

## Signal floor and representable contrast

Contrast means the absolute numerator |epsilon(q+h)-epsilon(q-h)|, not S. All repeated plus/minus outputs and slopes are bitwise identical: maximum and median repeat difference are zero at every width. Thus signal/floor ratios are undefined, not an invented infinite-confidence or epsilon-based PASS.

| Window | Median contrast all12 | p10 all12 | Median non-head9 | p10 non-head9 | Repeat floor |
|---|---:|---:|---:|---:|---:|
| h_f | 5.39254486e-07 | 0 | 8.6827679e-06 | 4.16264245e-14 | 0.0 |
| h1 | 2.56397875e-07 | 0 | 4.38557986e-06 | 2.07167616e-14 | 0.0 |
| h2 | 1.33813699e-07 | 0 | 2.17911725e-06 | 1.03361764e-14 | 0.0 |
| h3 | 6.40055624e-08 | 0 | 1.07705438e-06 | 5.16808818e-15 | 0.0 |

All12-state inequality checks include the structural HEAD zeros. Those zeros are reported explicitly and are not evidence of resolved signal. Each of the nine non-head state/window blocks additionally has positive p10 signal. At h2, pooled non-head p10 is388.6 output ULPs, median4.70e10 ULPs; at h3 p10 is194.3 ULPs. The smallest observed nonzero numerator is6.938893903907228e-18. Some individual probes round to exact zero; no claim is made that every row is resolved. Median/p10 evidence and norm-level agreement do not arise from wholesale contrast collapse. These are diagnostic resolution observations, not new production acceptance thresholds.

## Central asymmetry

A=|epsilon_plus+epsilon_minus-2epsilon_center|/max(|epsilon_plus-epsilon_minus|,repeat_floor). Zero/zero is undefined and omitted from quantiles; counts are recorded. No epsilon floor is manufactured. This is descriptive, not a selection gate.

| Window | Median A | p90 A | Max A |
|---|---:|---:|---:|
| h_f | 0.106673854 | 0.778711289 | 75.4150916 |
| h1 | 0.053371965 | 0.37718664 | 40.3307023 |
| h2 | 0.0267267582 | 0.198216836 | 14.8237032 |
| h3 | 0.0133226137 | 0.105835565 | 7.05989845 |

Median/p90 asymmetry decreases with h. Large maximum ratios at tiny/zero-signaling rows prevent interpreting every point as uniformly linear. They do not replace the predeclared block-norm criterion.

## Decision and answers

1. A numerically resolved <=5% window exists under the unchanged state/spin norm gate: h2.
2. Largest acceptable tested global window: h2, alpha0.023218985940978362/beta0.042529125693106434.
3. All nine non-head states have monotone adjacent-error reduction in both spins. This supports convergence over this ladder only.
4. No wholesale signal collapse explains the passing pair. Exact repeats give zero floor; positive non-head p10 and ULP comparisons support resolution, with individual zero rows disclosed.
5. Selected h2/h3 signs are stable across all12 states: zero changes. h_f/h1 has10 pointwise flips across the reported states.
6. The probe passes the bounded finite-contrast prerequisite. A parameter-Jacobian study still requires its own frozen protocol/review; no capacity conclusion is made here.
7. Single next experiment: the bounded parameter-Jacobian q-controllability audit using frozen h2 and the same states/probe. It was not launched.

## Validation and safety

34 original immutable hashes verified before and after; zero mismatches; model state hashes unchanged, parameter .grad storage empty. Six diagnostic tests pass; external py_compile passes. Large48-block local arrays are external: `C:\Dev\readWFN_share_ms\lap_q_controllability_runs_20261005\window_arrays.pt`, SHA256`447d747e26b5f0021afb245129414cafeee0a5d6e07064d7d2ecea36f95fc264`. Protocol/driver/state identities and repeat data are bound in JSON. No trained checkpoint is created. Repository diff check PASS; independent final review PASS. A non-blocking review caveat notes that the driver combines energy and slope repeat differences into one floor: they are all exactly zero here, so this result is unchanged; future nonzero floors must retain separate units.

No training, optimizer change, cursor10 advancement, parameter Jacobian, expensive objective-gradient recomputation, full90, SCF, Diet, Slurm or100-update pilot was performed.
