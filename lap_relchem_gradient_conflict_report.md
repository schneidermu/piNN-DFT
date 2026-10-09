# Frozen gradient conflict at P536 and IID t70

This diagnostic evaluates qualified singleton and four-system gradients at two frozen checkpoints. It does not update parameters. `C_d` is a database contribution to the 251-identity relchem gradient. `G_d` is the mean gradient inside that database. The Exc and operator vectors are four-system diagnostic gradients, not full90 gradients. `C_focus` omits the other 226 relchem identities.

## A. Provenance

Source commit `5b9e2e7c5ac3ee661bf4521c075930d895a9ef33`. Dataset logical SHA256 `61c221a19b9987717e69cac182ad545241f8807db4126c0949a99992e4c210ef`. Evaluation manifest SHA256 `132bd550161be6bc6761f8df1de8d3755c0b0debab53663208c3a9bbf56cb805`.
S0 file `C:\Dev\readWFN_share_ms\lap_init_landscape_runs_20261005\states\seed11_P536.pt` `0ca0f77367214c171e6edf0576bdb043d64479245ad1a0e539fa11fb0cee542d`; tensor `3c2e45d6a86cccf80d7b4dc458e67f9b406935ed60d4e57433bb441a6df88da6`.
S70 file `C:\Dev\readWFN_share_ms\lap_iid_adamw_t59_t90_20261009\ordinary_sgd_adamw\checkpoint_70.pt` `04b8e549c17375988be576e2878881d13f55b217b42ec16a7e6101fdddd05443`; tensor `59ab4b98550805b13852e13745281fb1d072efcbed5c2c8622bf3a6840c3ff51`.
A byte-identical t70 copy is `04b8e549c17375988be576e2878881d13f55b217b42ec16a7e6101fdddd05443`. It was hashed and not loaded.
mRKS metadata SHA256 `b0e76f00e5f4f744647c71ce31afe9e495d29dd6a5943da6d5e67dbaf65502ea`. Selected systems, sorted by `(n_grid, id)` at indexes 11, 33, 56 and 78:

| Index | System | Source | n_grid | n_ao |
|---:|---|---|---:|---:|
| 11 | `mrks_6d0f8432e3e2305a83fbca72` | N2 | 71820 | 168 |
| 33 | `mrks_b04483c49464df988fa87dbc` | FHO | 93936 | 198 |
| 56 | `mrks_49efb0f22b50db3616858795` | AlHMg | 96738 | 248 |
| 78 | `mrks_80306958d0118adca3be93af` | H2N2 | 115964 | 228 |

New isolated gradient evaluations: 100. New GPU seconds: 396.178. Wall seconds: 399.356. Peak live CUDA bytes: 12994189312. Peak reserved bytes: 18343788544.
Checkpoint file hashes were unchanged at the end of the run. No optimizer was constructed. Every chemistry singleton loss matched its frozen receipt to within 1e-6 before its gradient was stored.

## B. Protected chemistry gradients

### s0

| Object | Contribution norm | Mean-gradient norm | Within-database cancellation |
|---|---:|---:|---:|
| ABDE4 | 4.792637e+00 | 3.007380e+02 | 3.325610e-01 |
| pTC13 | 8.109230e+00 | 1.565705e+02 | 1.990803e-01 |
| PA8 | 2.171493e+00 | 6.813060e+01 | 7.740272e-01 |
| Focus25 | 1.445285e+01 |  |  |

Reaction-gradient norm dispersion, from the qualified singleton gradients:

- ABDE4: min 2.599010e+02, median 4.494351e+02, max 6.435687e+02
- pTC13: min 2.086718e+01, median 2.099074e+02, max 2.944291e+02
- PA8: min 2.170735e+02, median 3.127462e+02, max 3.971784e+02

Vector-cancellation ratios between contribution gradients, `1 - ||a+b||/(||a||+||b||)`:

- ABDE4:pTC13: 4.382526e-02
- ABDE4:PA8: 2.909595e-02
- pTC13:PA8: 6.024124e-03

### s70

| Object | Contribution norm | Mean-gradient norm | Within-database cancellation |
|---|---:|---:|---:|
| ABDE4 | 6.719021e+00 | 4.216186e+02 | 6.587386e-02 |
| pTC13 | 9.690953e+00 | 1.871099e+02 | 4.213897e-02 |
| PA8 | 6.347913e+00 | 1.991658e+02 | 3.380516e-01 |
| Focus25 | 2.144437e+01 |  |  |

Reaction-gradient norm dispersion, from the qualified singleton gradients:

- ABDE4: min 2.590539e+02, median 4.503722e+02, max 6.456047e+02
- pTC13: min 2.109737e+01, median 2.096488e+02, max 2.941075e+02
- PA8: min 2.176661e+02, median 3.113422e+02, max 3.955660e+02

Vector-cancellation ratios between contribution gradients, `1 - ||a+b||/(||a||+||b||)`:

- ABDE4:pTC13: 6.854478e-02
- ABDE4:PA8: 6.685295e-02
- pTC13:PA8: 6.659487e-04

## C. Conflict with other objectives

### s0

Four-system panel norms: Exc 1.843738e+04, operator 1.041435e+00. Exact AE17 norm 5.545236e+03. Weighted diagnostic `G_other` norm 7.964009e-01.

| Gradient | ABDE4 | pTC13 | PA8 | Focus25 | AE17 | Exc_panel | Op_panel | G_other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ABDE4 | 1.0000e+00 | 8.1641e-01 | 8.6638e-01 | 9.1985e-01 | -1.8847e-03 | -4.0989e-02 | -7.1590e-01 | -3.2953e-01 |
| pTC13 | 8.1641e-01 | 1.0000e+00 | 9.6395e-01 | 9.7664e-01 | -4.9144e-01 | -5.2858e-01 | -9.5858e-01 | -7.8179e-01 |
| PA8 | 8.6638e-01 | 9.6395e-01 | 1.0000e+00 | 9.7840e-01 | -2.6715e-01 | -3.0841e-01 | -9.3870e-01 | -6.1582e-01 |
| Focus25 | 9.1985e-01 | 9.7664e-01 | 9.7840e-01 | 1.0000e+00 | -3.1650e-01 | -3.5651e-01 | -9.1627e-01 | -6.4045e-01 |
| AE17 | -1.8847e-03 | -4.9144e-01 | -2.6715e-01 | -3.1650e-01 | 1.0000e+00 | 9.9899e-01 | 4.7527e-01 | 9.1589e-01 |
| Exc_panel | -4.0989e-02 | -5.2858e-01 | -3.0841e-01 | -3.5651e-01 | 9.9899e-01 | 1.0000e+00 | 5.1217e-01 | 9.3209e-01 |
| Op_panel | -7.1590e-01 | -9.5858e-01 | -9.3870e-01 | -9.1627e-01 | 4.7527e-01 | 5.1217e-01 | 1.0000e+00 | 7.8846e-01 |
| G_other | -3.2953e-01 | -7.8179e-01 | -6.1582e-01 | -6.4045e-01 | 9.1589e-01 | 9.3209e-01 | 7.8846e-01 | 1.0000e+00 |

| Contribution | dot AE17 | dot Exc panel | dot operator panel | dot G_other | cos G_other | predicted delta under -G_other |
|---|---:|---:|---:|---:|---:|---:|
| ABDE4 | -5.008902e+01 | -3.621926e+03 | -3.573218e+00 | -1.257761e+00 | -3.2953e-01 | 1.257761e+00 |
| pTC13 | -2.209898e+04 | -7.903018e+04 | -8.095430e+00 | -5.048964e+00 | -7.8179e-01 | 5.048964e+00 |
| PA8 | -3.216890e+03 | -1.234767e+04 | -2.122838e+00 | -1.064994e+00 | -6.1582e-01 | 1.064994e+00 |
| Focus25 | -2.536596e+04 | -9.499978e+04 | -1.379149e+01 | -7.371719e+00 | -6.4045e-01 | 7.371719e+00 |

Weighted pieces of `dot(C, G_other)`:

- ABDE4: AE17 -2.575204e-03, Exc panel -5.467168e-02, operator panel -1.200514e+00
- pTC13: AE17 -1.136165e+00, Exc panel -1.192932e+00, operator panel -2.719867e+00
- PA8: AE17 -1.653885e-01, Exc panel -1.863837e-01, operator panel -7.132218e-01
- Focus25: AE17 -1.304128e+00, Exc panel -1.433988e+00, operator panel -4.633603e+00

### s70

Four-system panel norms: Exc 1.847539e+04, operator 7.343615e-01. Exact AE17 norm 4.738214e+03. Weighted diagnostic `G_other` norm 4.984989e-01.

| Gradient | ABDE4 | pTC13 | PA8 | Focus25 | AE17 | Exc_panel | Op_panel | G_other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ABDE4 | 1.0000e+00 | 7.2624e-01 | 7.4132e-01 | 8.6096e-01 | -5.8484e-02 | 6.2318e-03 | -2.8563e-01 | -1.6646e-01 |
| pTC13 | 7.2624e-01 | 1.0000e+00 | 9.9722e-01 | 9.7465e-01 | 4.7723e-01 | 5.4446e-01 | -7.3823e-01 | 1.7242e-01 |
| PA8 | 7.4132e-01 | 9.9722e-01 | 1.0000e+00 | 9.7894e-01 | 4.7681e-01 | 5.4305e-01 | -7.0105e-01 | 1.8983e-01 |
| Focus25 | 8.6096e-01 | 9.7465e-01 | 9.7894e-01 | 1.0000e+00 | 3.3849e-01 | 4.0875e-01 | -6.3063e-01 | 8.1957e-02 |
| AE17 | -5.8484e-02 | 4.7723e-01 | 4.7681e-01 | 3.3849e-01 | 1.0000e+00 | 9.9678e-01 | -3.0015e-01 | 8.9776e-01 |
| Exc_panel | 6.2318e-03 | 5.4446e-01 | 5.4305e-01 | 4.0875e-01 | 9.9678e-01 | 1.0000e+00 | -3.5494e-01 | 8.7087e-01 |
| Op_panel | -2.8563e-01 | -7.3823e-01 | -7.0105e-01 | -6.3063e-01 | -3.0015e-01 | -3.5494e-01 | 1.0000e+00 | 1.4970e-01 |
| G_other | -1.6646e-01 | 1.7242e-01 | 1.8983e-01 | 8.1957e-02 | 8.9776e-01 | 8.7087e-01 | 1.4970e-01 | 1.0000e+00 |

| Contribution | dot AE17 | dot Exc panel | dot operator panel | dot G_other | cos G_other | predicted delta under -G_other |
|---|---:|---:|---:|---:|---:|---:|
| ABDE4 | -1.861920e+03 | 7.735913e+02 | -1.409343e+00 | -5.575538e-01 | -1.6646e-01 | 5.575538e-01 |
| pTC13 | 2.191345e+04 | 9.748162e+04 | -5.253711e+00 | 8.329577e-01 | 1.7242e-01 | -8.329577e-01 |
| PA8 | 1.434144e+04 | 6.368919e+04 | -3.268042e+00 | 6.007134e-01 | 1.8983e-01 | -6.007134e-01 |
| Focus25 | 3.439297e+04 | 1.619444e+05 | -9.931096e+00 | 8.761173e-01 | 8.1957e-02 | -8.761173e-01 |

Weighted pieces of `dot(C, G_other)`:

- ABDE4: AE17 -9.572603e-02, Exc panel 1.167709e-02, operator panel -4.735049e-01
- pTC13: AE17 1.126626e+00, Exc panel 1.471450e+00, operator panel -1.765119e+00
- PA8: AE17 7.373302e-01, Exc panel 9.613657e-01, operator panel -1.097982e+00
- Focus25: AE17 1.768230e+00, Exc panel 2.444493e+00, operator panel -3.336606e+00

## D. Actual versus predicted change

Parameter displacement norm 1.406912e-01; relative to S0 6.641980e-03; unchanged F32 coordinates 0 / 9446 (0.000000).

The dots below multiply one endpoint contribution gradient by the full 70-update displacement. They are not a line integral and are not an AdamW step.

| Database | Actual contribution change | C(S0)·delta | C(S70)·delta | cos(C(S0), delta) | cos(C(S70), delta) |
|---|---:|---:|---:|---:|---:|
| ABDE4 | 5.921836e-02 | 4.222903e-02 | 7.090132e-02 | 6.2628e-02 | 7.5003e-02 |
| pTC13 | 5.859292e-02 | 4.896795e-02 | 5.974972e-02 | 4.2921e-02 | 4.3823e-02 |
| PA8 | 2.954491e-02 | 9.086053e-03 | 4.133130e-02 | 2.9741e-02 | 4.6279e-02 |
| Focus25 total | 1.473562e-01 | 1.002830e-01 | 1.719823e-01 | 4.9318e-02 | 5.7004e-02 |
| AE17 | -1.847075e+01 | -3.018221e+01 | 2.537313e+01 | -3.8687e-02 | 3.8062e-02 |

AE17 uses its exact 17-identity mean gradient, so its actual change and dots share a definition. The S70 AE17 projection has the opposite sign from the measured AE17 decrease. That reversal is an AE17 path observation.

Four-system diagnostic projections onto the same displacement, with no full90 loss comparison:

- Exc panel: C(S0)·delta -9.955986e+01, C(S70)·delta 1.002540e+02.
- Operator panel: C(S0)·delta -3.720961e-03, C(S70)·delta 4.443861e-04.

## E. Hypothesis adjudication

- H1 inter-task conflict: `INCONCLUSIVE`.
- H2 chemistry-internal conflict: `INCONCLUSIVE`. Mutual three-database clause: `WEAKENED`. Conflict with the other 226 identities: `NOT VERIFIED`.
- H3 finite-step/path effects: `WEAKENED`.

H1 asks whether the weighted direction `G_other` locally opposes the three contribution gradients at both frozen states. At S0 every protected cosine with `G_other` is negative and every predicted change under `-G_other` is positive: ABDE4 -3.2953e-01 / 1.257761e+00, pTC13 -7.8179e-01 / 5.048964e+00, PA8 -6.1582e-01 / 1.064994e+00. At S70 only ABDE4 keeps that pattern (-1.6646e-01, predicted 5.575538e-01). pTC13 and PA8 have positive cosines 1.7242e-01 and 1.8983e-01, so `-G_other` locally decreases those two contributions. The four-system operator panel stays negatively aligned with all three databases at both states: S0 cosines -7.1590e-01, -9.5858e-01, -9.3870e-01; S70 cosines -2.8563e-01, -7.3823e-01, -7.0105e-01. At S70 the positive AE17 and Exc-panel pieces outweigh that operator piece for pTC13 and PA8. ABDE4 is nearly orthogonal to AE17 at both states. These statements describe local directional geometry of the exact AE17 mean and a four-system diagnostic panel. They leave the historical AdamW cause unidentified, and the panel gradients are four-system diagnostics.

H2 asks whether the three databases oppose one another. All six pairwise contribution cosines are positive, from 7.2624e-01 to 9.9722e-01. Mutual opposition among ABDE4, pTC13, and PA8 is therefore weakened. No SHA-matched full 251-identity parameter gradient for S0 or S70 was reused, so conflict with the other 226 identities remains unverified.

H3 asks whether the endpoint gradients fail to predict the sign of the measured contribution changes along `theta_70 - theta_0`. For ABDE4, pTC13, and PA8 the measured change and both endpoint dots are positive. The cosines of those gradients with the displacement are small, 0.030 to 0.075, so most of the displacement lies in other directions, while the projected component has the observed sign. PA8 at S0 projects to 9.086e-03 against a measured change of 2.954e-02; ABDE4 and pTC13 stay within the same order at both ends. One endpoint dot is a local linearization, not the nonlinear change along the 70-update path. The S0-versus-S70 reversal of the AE17 and panel projections is recorded and is kept separate from the three chemistry signs.

## F. Historical context

The cursor10 chemistry-gradient audit used another frozen state and a different sampling/PCD protocol. Its database cosines are not S0 or S70 gradients. In that older audit, ABDE4 aligned with the secondary tasks while pTC13 opposed them. That pattern is context only.

J251, file SHA256 `11cee17f61c017e26902c905b6d9ace0b576ab7eaacf5d5f93f5172a49e3257a`, remains a historically eligible joint checkpoint on the same fixed panel: relchem/t0 0.979708, AE17/t0 0.081012, Exc/t0 0.101105, operator/t0 0.938121. Its existing singleton-mean ratios still rise for ABDE4 (1.097354), PA8 (1.065882) and pTC13 (1.107175). No new J251 gradient was computed.

Skala-1.1 uses hierarchical then relative-excess dataset sampling. That training fact does not identify which optimizer intervention would protect these three databases.

## G. Recommended next experiment

No further experiment is recommended. The weighted competing direction opposes all three databases at S0 and only ABDE4 at S70, so a chemistry-protection arm is not selected. The measured contribution increases have the same sign as both endpoint projections onto the historical displacement, so an optimizer-path replay is not selected to explain those signs. Another IID replica is not selected because this audit did not measure stream sensitivity. Expected additional GPU cost: none.

## Source hashes

- `train_lap_microbatch.py` `2e837c3a88ca3d4737c3c3397dcfbb98adeb7f6917adc8c693cecf205dc37e9e`
- `train_models/lap_moo_training.py` `e3dafbe76279e13b73f2a6dd5f2547294902387d21fc1b78b4ce917430495113`
- `train_models/lap_fixed_adamw.py` `9effc074bd18236ad7af4c4e05fa04be1754d35e228b9a0567163c4b0bd5eb0a`
- `train_models/lap_training.py` `ca789703e536f9311178ada07fd41d92ba7f79afd35b25ad240647747b3e9b77`
- `train_models/lap_operator.py` `7fae09c0857187875377fbf5c1c7c8d19b1afc869942310f75f5462f7bd2e864`
- `train_models/lap_operator_data.py` `8d411952a579728dac395e38d1255e5e5bb412ea77d4e2fb8c9f6116e55f93c9`
- `train_models/lap_vxc.py` `ca22ac54ae9ffe4d2593f1e7072b11210277b0a321701d5240f564803711c287`
- `train_models/lap_chemistry_sampling.py` `313f90bcd0fe438f3e82ca00e7625c7bee502797b772532fef16adb2adad9fad`
- `train_models/publication_data/loader.py` `27415bd43f68e64467eec21b1ddf03ac1fc540d93c077483b1585f547579bb6f`
