# Skala-inspired chemistry loss audit

**Decision: NO-GO.** Both normalized losses place under 10% of residual-derivative mass on ABDE4, pTC13, and PA8 combined and over 70% on one other database.

This is a frozen-model loss and gradient audit. A lower training loss does not establish a lower Clean28. Clean28 was not evaluated, and it was not used to choose the denominator, DELTA, the diagnostic identities, or a task coefficient.

## 1. Decision

NO-GO. Both normalized losses place under 10% of residual-derivative mass on ABDE4, pTC13, and PA8 combined and over 70% on one other database. Qualification remains the original fixed-panel relchem mean, AE17, full90 Exc, and the full90 AO operator. The normalized losses were examined as training surrogates only.

The population mean of Loss A is 1.2354710978866068 at S0 and 1.2905829668177973 at S70. Those are the published fixed-panel relchem objectives, and every identity matched its frozen receipt with maximum absolute discrepancy 0. At S0, DBH76 holds 0.811 of the Loss B residual-derivative mass and 0.845 of Loss B. ABDE4, pTC13, and PA8 together hold 0.046. At S70 those figures are 0.811 and 0.052. The stubborn reactions have reference energies near 90 to 220 kcal/mol, so the reference-energy denominator suppresses them. Eight reactions with |Eref| below 1 kcal/mol hold 0.241 of |dL_B/de| at S0 while their mean absolute error is 1.337 kcal/mol. The 144 reactions with |Eref| at or above 50 kcal/mol have mean absolute error 13.015 kcal/mol and hold 0.053. Huber reduces the S0 top-five |dL/de| share from 0.371 to 0.191 and leaves 32 reactions in its linear region, but DBH76 still holds 0.717 and the focus trio holds 0.055. The normalized losses move training emphasis toward small-reference DBH76 reactions, which already improve under the current loss, and away from the large-reference reactions that currently worsen. That is the opposite of the failure this audit was asked to address. A lower value of either surrogate on these 251 reactions would not establish a lower Clean28.

## 2. Skala equation and implementation

Supplement B.1, equation (31), arXiv:2506.14665v6: E[ |DeltaE - DeltaE^ref|^2 / (1e-4 Eh + |DeltaE^ref|) ]. The denominator uses the reference reaction energy.

Section 2.1 states that training uses a reaction-energy regression loss. The explicit formula is equation (31) in Supplement B.1. The v6 HTML rendering of the denominator is `1e-4 Eh + |DeltaE^ref|`. It uses the reference energy. Loss B implements one reaction as `e_H^2 / (1e-4 + abs(E_ref_H))` with `K = 627.5095` kcal/mol per Eh. It does not multiply by the FCHEM factor. The expectation over Skala hierarchical sampling is not implemented. Supplement B.2 and B.4 describe that sampling and the Muon/Adam optimizer; neither is used here.

Loss A is `a_db * sqrt(e_kcal^2 + 1e-20)`, with `a_db` taken from the qualified `batch_fchem` constants, including AE17 inside `MEAN_WEIGHT`. Loss C uses `z = e_H / sqrt(D_H)` and `DELTA = 0.1`: `z^2` inside the threshold and `2*DELTA*|z| - DELTA^2` outside. `z^2` equals Loss B in the quadratic region. Numerically `z` is `e_H/sqrt(D_H)` with both energies in Hartree, so Loss B and Loss C are in Hartree in that region. They are not in the same units as Loss A.

## 3. Three-loss comparison

### s0

| loss | unit | mean | median | p90 | p95 | p99 | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | qualified singleton | 1.23547 | 0.635748 | 3.26488 | 5.34088 | 8.59398 | 11.4994 |
| B | Hartree | 0.00616544 | 0.000696112 | 0.0121371 | 0.0233181 | 0.100652 | 0.18176 |
| C | Hartree | 0.00461211 | 0.000696112 | 0.0120337 | 0.020529 | 0.0534514 | 0.0752667 |

Absolute amplification versus Loss A, median/p99/max: B 0.00464454 / 0.18103 / 0.337807; C 0.00464454 / 0.0591612 / 0.106477. Huber regions: 219 quadratic, 32 linear.

### s70

| loss | unit | mean | median | p90 | p95 | p99 | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | qualified singleton | 1.29058 | 0.503646 | 3.36725 | 6.0395 | 8.5108 | 16.9032 |
| B | Hartree | 0.00498556 | 0.000515492 | 0.0102888 | 0.0211923 | 0.0713979 | 0.148287 |
| C | Hartree | 0.00389553 | 0.000515492 | 0.0102868 | 0.0191012 | 0.0434408 | 0.0670161 |

Absolute amplification versus Loss A, median/p99/max: B 0.0039667 / 0.165917 / 0.284511; C 0.00388258 / 0.0531945 / 0.106477. Huber regions: 223 quadratic, 28 linear.

Loss A has a saturated residual derivative `a_db * sign(e)` once the residual is outside the `1e-20` smoothing. Loss B grows linearly with the residual and is divided by the reference-energy denominator. Loss C follows Loss B and then saturates. Raw Loss A numbers are not compared with raw Loss B numbers as a common score. Every identity residual derivative, ratio, Huber region, reference magnitude, database, and FCHEM factor is stored for both checkpoints in `lap_skala_loss_audit_metrics.json` under `states.<checkpoint>.derivatives`.

## 4. Database loss and gradient distributions

### s0

| database | n | mean |e| | mean |Eref| | a_db | share LA | share LB | share LC | share |dA| | share |dB| | share |dC| |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ABDE4 | 4 | 2.499 | 93.42 | 2.245 | 0.072 | 0.000 | 0.001 | 0.067 | 0.001 | 0.001 |
| DBH76 | 70 | 8.56 | 18.84 | 0.1182 | 0.228 | 0.845 | 0.798 | 0.061 | 0.811 | 0.717 |
| EA13 | 11 | 2.482 | 38.68 | 0.6907 | 0.061 | 0.003 | 0.005 | 0.056 | 0.008 | 0.013 |
| IP13 | 13 | 3.418 | 253.8 | 0.6907 | 0.099 | 0.001 | 0.001 | 0.067 | 0.001 | 0.002 |
| MGAE109 | 104 | 16.21 | 504.4 | 0.0174 | 0.095 | 0.098 | 0.131 | 0.013 | 0.041 | 0.064 |
| NCCE31 | 28 | 0.9037 | 3.537 | 2.897 | 0.236 | 0.015 | 0.020 | 0.602 | 0.093 | 0.149 |
| PA8 | 8 | 1.449 | 163.3 | 1.122 | 0.042 | 0.000 | 0.000 | 0.067 | 0.001 | 0.001 |
| pTC13 | 13 | 5.755 | 168.8 | 0.6907 | 0.167 | 0.036 | 0.045 | 0.067 | 0.045 | 0.052 |

Selected-panel sums of parameter-gradient norms. This panel contains every ABDE4, pTC13, and PA8 identity and three controls from each other database, so these sums are not a 251-reaction gradient.

| database | n in panel | sum ||gA|| | sum ||gB|| | sum ||gC|| |
| --- | --- | --- | --- | --- |
| ABDE4 | 4 | 1802 | 0.07502 | 0.07502 |
| DBH76 | 3 | 32.26 | 0.2256 | 0.2256 |
| EA13 | 3 | 621.6 | 0.2619 | 0.2619 |
| IP13 | 3 | 482 | 0.01558 | 0.01558 |
| MGAE109 | 3 | 20.28 | 0.0963 | 0.0963 |
| NCCE31 | 3 | 124.6 | 0.0837 | 0.0837 |
| PA8 | 8 | 2412 | 0.06584 | 0.06584 |
| pTC13 | 13 | 2541 | 0.7937 | 0.6455 |

### s70

| database | n | mean |e| | mean |Eref| | a_db | share LA | share LB | share LC | share |dA| | share |dB| | share |dC| |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ABDE4 | 4 | 4.154 | 93.42 | 2.245 | 0.115 | 0.001 | 0.002 | 0.067 | 0.002 | 0.003 |
| DBH76 | 70 | 7.88 | 18.84 | 0.1182 | 0.201 | 0.861 | 0.825 | 0.061 | 0.811 | 0.732 |
| EA13 | 11 | 2.708 | 38.68 | 0.6907 | 0.064 | 0.006 | 0.008 | 0.056 | 0.011 | 0.017 |
| IP13 | 13 | 3.736 | 253.8 | 0.6907 | 0.104 | 0.001 | 0.002 | 0.067 | 0.002 | 0.003 |
| MGAE109 | 104 | 11.94 | 504.4 | 0.0174 | 0.067 | 0.072 | 0.092 | 0.013 | 0.035 | 0.053 |
| NCCE31 | 28 | 0.7264 | 3.537 | 2.897 | 0.182 | 0.014 | 0.018 | 0.602 | 0.090 | 0.135 |
| PA8 | 8 | 2.274 | 163.3 | 1.122 | 0.063 | 0.001 | 0.001 | 0.067 | 0.001 | 0.002 |
| pTC13 | 13 | 7.393 | 168.8 | 0.6907 | 0.205 | 0.043 | 0.053 | 0.067 | 0.049 | 0.056 |

Selected-panel sums of parameter-gradient norms. This panel contains every ABDE4, pTC13, and PA8 identity and three controls from each other database, so these sums are not a 251-reaction gradient.

| database | n in panel | sum ||gA|| | sum ||gB|| | sum ||gC|| |
| --- | --- | --- | --- | --- |
| ABDE4 | 4 | 1805 | 0.1297 | 0.1297 |
| DBH76 | 3 | 32.38 | 0.2072 | 0.2072 |
| EA13 | 3 | 622.5 | 0.387 | 0.387 |
| IP13 | 3 | 482.6 | 0.02614 | 0.02614 |
| MGAE109 | 3 | 20.04 | 0.07166 | 0.07166 |
| NCCE31 | 3 | 123.8 | 0.07413 | 0.07413 |
| PA8 | 8 | 2407 | 0.103 | 0.103 |
| pTC13 | 13 | 2539 | 0.8977 | 0.7711 |

## 5. Reaction-level tail concentration

s0: largest 5/10/20 share of total loss A 0.146/0.255/0.410; B 0.367/0.524/0.672; C 0.245/0.391/0.564. Absolute-residual shares 0.094/0.173/0.292. Absolute residual-derivative shares B 0.371/0.503/0.636; C 0.191/0.294/0.453.

Selected-panel parameter-gradient norm mass at s0: top1/top5 old 0.080/0.292, B 0.183/0.523, C 0.136/0.475.

s70: largest 5/10/20 share of total loss A 0.161/0.272/0.445; B 0.357/0.520/0.679; C 0.251/0.405/0.589. Absolute-residual shares 0.089/0.163/0.277. Absolute residual-derivative shares B 0.361/0.494/0.634; C 0.202/0.310/0.473.

Selected-panel parameter-gradient norm mass at s70: top1/top5 old 0.080/0.292, B 0.189/0.539, C 0.202/0.506.

Largest Loss B reactions at S70:

| identity | database | |e| kcal/mol | |Eref| kcal/mol | LB Hartree | region | ratio B |
| --- | --- | --- | --- | --- | --- | --- |
| reaction_7b812edddfdc846c21bbd022 | DBH76 | 11.75 | 1.42 | 0.1483 | linear | 0.2137 |
| reaction_d791c93a7646841e88c1245a | DBH76 | 12.74 | 3 | 0.08449 | linear | 0.1122 |
| reaction_31d81288f4369a4ceb89bf33 | DBH76 | 4.248 | 0.34 | 0.0714 | linear | 0.2845 |
| reaction_96209f27ca6129ee07792d1b | DBH76 | 4.248 | 0.34 | 0.0714 | linear | 0.2845 |
| reaction_a2afa0cb288b7d118498b49a | DBH76 | 10.22 | 2.27 | 0.0713 | linear | 0.1181 |
| reaction_0d74cab283e7f6e83f164854 | DBH76 | 10.36 | 3.2 | 0.05245 | linear | 0.08568 |
| reaction_01cbd50d0fb61ab355eada26 | DBH76 | 7.192 | 1.7 | 0.04677 | linear | 0.1101 |
| reaction_2c3d80b0262d09af0f3fb477 | DBH76 | 16.53 | 9.57 | 0.04521 | linear | 0.04629 |
| reaction_90bd41f40cb1bb64fa9d8604 | DBH76 | 11.58 | 6.73 | 0.03148 | linear | 0.046 |
| reaction_b32ec4df9c2baf0bac65a2e3 | DBH76 | 9.387 | 4.9 | 0.02829 | linear | 0.05102 |

Previously stubborn reactions:

### s0

| identity | database | |e| | |Eref| | LA | LB | LC | ratio B | region |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reaction_499996e5b8084d7129c856a5 | ABDE4 | 5.122 | 91.51 | 11.5 | 0.0004566 | 0.0004566 | 7.942e-05 | quadratic |
| reaction_c258198ad576955c8267a32b | ABDE4 | 1.106 | 89.79 | 2.483 | 2.169e-05 | 2.169e-05 | 1.747e-05 | quadratic |
| reaction_18a4cfbaf87fde8157323b2d | ABDE4 | 3.047 | 95 | 6.841 | 0.0001557 | 0.0001557 | 4.551e-05 | quadratic |
| reaction_dce81e2069e365d6f6f375e6 | PA8 | 2.943 | 156.6 | 3.304 | 8.811e-05 | 8.811e-05 | 5.334e-05 | quadratic |
| reaction_2711e62a8820baac39b75e55 | pTC13 | 6.21 | 219.7 | 4.289 | 0.0002797 | 0.0002797 | 0.0001304 | quadratic |

### s70

| identity | database | |e| | |Eref| | LA | LB | LC | ratio B | region |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reaction_499996e5b8084d7129c856a5 | ABDE4 | 7.53 | 91.51 | 16.9 | 0.0009866 | 0.0009866 | 0.0001167 | quadratic |
| reaction_c258198ad576955c8267a32b | ABDE4 | 3.483 | 89.79 | 7.819 | 0.0002152 | 0.0002152 | 5.504e-05 | quadratic |
| reaction_18a4cfbaf87fde8157323b2d | ABDE4 | 4.647 | 95 | 10.43 | 0.000362 | 0.000362 | 6.94e-05 | quadratic |
| reaction_dce81e2069e365d6f6f375e6 | PA8 | 5.262 | 156.6 | 5.906 | 0.0002816 | 0.0002816 | 9.536e-05 | quadratic |
| reaction_2711e62a8820baac39b75e55 | pTC13 | 8.741 | 219.7 | 6.038 | 0.0005541 | 0.0005541 | 0.0001835 | quadratic |

## 6. Near-zero reference energies

s0: 2 reactions have |Eref| at or below the 0.06275095 kcal/mol floor equivalent, and 8 have |Eref| below 1 kcal/mol. Spearman(|Eref|, Loss B) = -0.282376; Spearman(|e|, Loss B) = 0.64033.

| bin |Eref| kcal/mol | count | mean |e| | share |dB| | share |dC| |
| --- | --- | --- | --- | --- |
| 0-1 | 8 | 1.337 | 0.241 | 0.149 |
| 1-10 | 47 | 5.254 | 0.547 | 0.534 |
| 10-50 | 52 | 6.658 | 0.159 | 0.236 |
| 50+ | 144 | 13.02 | 0.053 | 0.082 |

Denominator-floor sensitivity of Loss B, reported as a probe and not used to change the 1e-4 Eh floor: 1e-05 Eh mean 0.00637763, max |dL/de_kcal| 0.0464224, 1e-04 Eh mean 0.00616544, max |dL/de_kcal| 0.0399128, 1e-03 Eh mean 0.00507346, max |dL/de_kcal| 0.0202431.

s70: 2 reactions have |Eref| at or below the 0.06275095 kcal/mol floor equivalent, and 8 have |Eref| below 1 kcal/mol. Spearman(|Eref|, Loss B) = -0.325541; Spearman(|e|, Loss B) = 0.630251.

| bin |Eref| kcal/mol | count | mean |e| | share |dB| | share |dC| |
| --- | --- | --- | --- | --- |
| 0-1 | 8 | 1.122 | 0.227 | 0.151 |
| 1-10 | 47 | 4.807 | 0.557 | 0.539 |
| 10-50 | 52 | 6.116 | 0.165 | 0.234 |
| 50+ | 144 | 10.21 | 0.051 | 0.076 |

Denominator-floor sensitivity of Loss B, reported as a probe and not used to change the 1e-4 Eh floor: 1e-05 Eh mean 0.00514635, max |dL/de_kcal| 0.0390985, 1e-04 Eh mean 0.00498556, max |dL/de_kcal| 0.0336159, 1e-03 Eh mean 0.00413202, max |dL/de_kcal| 0.0182844.

## 7. S0 versus S70

Mean absolute residual moves from 9.87262 kcal/mol at S0 to 8.0602 kcal/mol at S70. Mean Loss A moves from 1.23547 to 1.29058. Mean Loss B moves from 0.00616544 to 0.00498556 Hartree. Mean Loss C moves from 0.00461211 to 0.00389553 Hartree. These are training-population summaries. They are not Clean28.

## 8. Parameter-gradient parity

| identity | loss | relative L2 | cosine | status |
| --- | --- | --- | --- | --- |
| reaction_c258198ad576955c8267a32b | B | 3.73e-13 | 1 | pass |
| reaction_c258198ad576955c8267a32b | C | 3.704e-13 | 1 | pass |
| reaction_2711e62a8820baac39b75e55 | B | 2.663e-13 | 1 | pass |
| reaction_2711e62a8820baac39b75e55 | C | 2.663e-13 | 1 | pass |
| reaction_a985a6452336f1d11401aceb | B | 1.763e-12 | 1 | pass |
| reaction_a985a6452336f1d11401aceb | C | 1.759e-12 | 1 | pass |

Minimum per-reaction cosine on the resolved 40-identity panel: S0 B/C 1/1, S70 B/C 1/1. A positive scalar factor makes the single-reaction parameter gradient parallel to the qualified gradient. The aggregate direction can still change because reactions receive different factors. Direct checks were run at S0 for three predeclared identities and both new losses, six backwards. S70 uses the same transformation after that parity passed.

## 9. Effective relchem emphasis

Loss A assigns each reaction a residual derivative whose magnitude is essentially its database factor once the residual leaves the smoothing scale. Loss B removes that factor and uses `2 e_H / D_H`. Loss C uses the same normalization and caps the derivative after `|z| > 0.1`. The 251 derivative shares are the population weighting. The 40-reaction parameter gradients are a separate, focus-enriched check and are not a verified full251 gradient.

An offline counterfactual multiplies the Loss B residual derivative by the existing FCHEM factor and renormalizes. It was not differentiated and is not a candidate. Retaining `a_db` moves mass from DBH76 onto NCCE31, because NCCE31 combines a large factor with small reference energies. ABDE4 remains near one percent. Database weights of this magnitude do not undo the reference-energy denominator.

| database | S0 literal B | S0 times a_db | S70 literal B | S70 times a_db |
| --- | --- | --- | --- | --- |
| ABDE4 | 0.001 | 0.005 | 0.002 | 0.010 |
| DBH76 | 0.811 | 0.236 | 0.811 | 0.237 |
| EA13 | 0.008 | 0.013 | 0.011 | 0.019 |
| IP13 | 0.001 | 0.002 | 0.002 | 0.003 |
| MGAE109 | 0.041 | 0.002 | 0.035 | 0.002 |
| NCCE31 | 0.093 | 0.664 | 0.090 | 0.644 |
| PA8 | 0.001 | 0.002 | 0.001 | 0.003 |
| pTC13 | 0.045 | 0.076 | 0.049 | 0.083 |

## 10. Task coefficients

The historical lambdas were calibrated for Loss A. They are not reused. The values below match the norm of the 40-identity mean gradient at the frozen state. They are offline magnitude-matching estimates, not qualified optimizer coefficients. AE17, Exc, and the operator losses were not changed and their gradients were not recomputed.

| state | loss | ||mean g old|| | ||mean g new|| | new/old | offline match | offline lambda | cosine of means |
| --- | --- | --- | --- | --- | --- | --- | --- |
| s0 | B | 88.0073 | 0.0165157 | 0.000187663 | 5328.69 | 90.6703 | 0.241377 |
| s0 | C | 88.0073 | 0.0131555 | 0.000149482 | 6689.77 | 113.83 | 0.326221 |
| s70 | B | 133.06 | 0.0222682 | 0.000167355 | 5975.32 | 101.673 | 0.637679 |
| s70 | C | 133.06 | 0.0204895 | 0.000153987 | 6494.06 | 110.5 | 0.720291 |

Selected-panel cancellation, `1 - ||sum g|| / sum ||g||`: S0 old/B/C 0.561959/0.591598/0.641874; S70 old/B/C 0.33746/0.530346/0.536953.

## 11. Proposed training experiment

No training arm is authorized by this audit. A later experiment would still need an independently frozen training-only coefficient calibration and a matched old-loss control. One arm that changes both the loss and the coefficient cannot identify which change moved Clean28.

## 12. Runtime and gradient counts

New GPU-bound wall time 424.031 s. Peak allocated CUDA 4549206528 bytes. Peak reserved 7585398784 bytes. Qualified singleton backwards 80. Direct new-loss backwards 6. Forward residual evaluations 504. Optimizer steps 0.

## 13. Tests and numerical checks

15 CPU pytest tests in tests/test_lap_skala_loss_audit.py passed. Ruff and compileall passed on the audit module and its tests. git diff --check passed on the new files. No broad GPU or SCF suite was run.

Receipt agreement stayed within 1e-06. Formula agreement with `batch_fchem` stayed within 1e-08. The first identity at each state also matched `ChemistryBatchObjective` on the F64 shadow. Amplification is undefined when `|dL_A/de_kcal| < 1e-08`; unresolved panel gradients: S0 0, S70 0.

## 14. Commit and push

Recorded after commit.

## 15. Limitations

- The 40-reaction parameter gradients over-represent ABDE4, pTC13, and PA8. They are not a full251 relchem gradient.
- Single-reaction cosines are determined by the sign of one scalar factor. Aggregate geometry is the informative quantity.
- Loss B and Loss C both drop the qualified database factors. That is a second change bundled with the normalized residual.
- The offline lambda matches the 40-identity mean-gradient norm at one frozen state. It is not a calibrated four-task coefficient.
- Prior four-system Exc and operator gradients from the gradient-conflict audit are a different population and were not recomputed.
- J251 was not differentiated. No new Clean28, Diet100, future-test, SCF, or optimizer update was run.
- `z = e_H / sqrt(D_H)` matches Loss B near zero. With Hartree inputs it is not a dimensionally empty number; DELTA = 0.1 is the specified numeric threshold and was not tuned.
- A reduction in Loss B or Loss C on these 251 reactions would not by itself show a Clean28 improvement.

**0 optimizer updates. 0 production loss changes. 0 new checkpoints. 0 SCF runs. 0 Diet100 evaluations. 0 future-test accesses. 0 Slurm jobs.**

