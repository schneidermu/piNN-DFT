# Training-metric gradient alignment audit

Diagnostic only. Optimizer updates: 0. J251-MAE and J251-WTMAD-style were measurement targets and did not enter a training loss.

J251 is the Minnesota/M06 training collection, not GMTKN55. J251-WTMAD-style is `(1/251) Σ |E_pred - E_ref| / A_d`, where `A_d` is the mean absolute J251 reference energy inside database `d`. It is not literal WTMAD-2. No GMTKN55 subset weight or global calibration constant is used. J251-MAE is the equal-weight mean absolute reaction-energy error in kcal/mol.

The existing relchem objective remains the mean of 251 singleton `batch_fchem` values. A singleton is `factor_d * sqrt(1e-20 + residual^2)`, with the original database and frequency factors. Absolute value uses subgradient 0 at a zero residual.

Publication logical SHA256 `61c221a19b9987717e69cac182ad545241f8807db4126c0949a99992e4c210ef`. Evaluation manifest SHA256 `132bd550161be6bc6761f8df1de8d3755c0b0debab53663208c3a9bbf56cb805`. Production coefficients, applied once: relchem `0.017015480965588553`, ae17 `5.141254618347414e-05`, exc `1.5094644512009712e-05`, op `0.33597561607048215`.

S5 was not loaded. Saved Clean28 and production-task gradients were reused only after their checkpoint hash, parameter order, norms, and committed cosines matched. They were not recalibrated.

## Checkpoints and scalar parity

- `b8_t128` protocol `b8-adamw-j251-v1`, AdamW step `128`, relchem `1.195939284232` (receipt `1.195939284232`). J251-MAE `8.536568` kcal/mol. J251-WTMAD-style `0.156415`. Checkpoint SHA256 `1f38f7f8ea74334ff8f5124d2a6538619323d1f453296fa9ca6b3cd7958c437b`. Model SHA256 `c1c6ecee465decf386f6b6b33523c414bca7c1c5a4f93c1df15a9290a4f0760a`. Coordinates 9446.
- `continuation_raw_t251` protocol `b8-fixed-lr-swa-v1`, AdamW step `379`, relchem `1.170649184133` (receipt `1.170649184133`). J251-MAE `8.046518` kcal/mol. J251-WTMAD-style `0.150596`. Checkpoint SHA256 `63d72b83dbc2ee49b466d366454ded341086547ddf1594893e15547c6410adf5`. Model SHA256 `491b6b1e02682c6fdc7ca7b4f3cb9808dd210395de29b253d98806a494b7c87a`. Coordinates 9446.

## Reference scales

| Database | Count | A_d (kcal/mol) |
| --- | ---: | ---: |
| ABDE4 | 4 | 93.422501 |
| DBH76 | 70 | 18.836000 |
| EA13 | 11 | 38.680000 |
| IP13 | 13 | 253.828461 |
| MGAE109 | 104 | 504.447019 |
| NCCE31 | 28 | 3.536786 |
| PA8 | 8 | 163.325002 |
| pTC13 | 13 | 168.787692 |

## Primary alignment

### b8_t128

| Metric | Task | Cosine | Dot | Weighted dot | Predicted change | Weighted predicted change | Unit predicted change |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mae | relchem | 0.271102 | 672.067131 | 11.435545 | -672.067131 | -11.435545 | -44.573679 |
| mae | ae17 | -0.617405 | -568235.626612 | -29.214440 | 568235.626612 | 29.214440 | 101.511597 |
| mae | exc | -0.614972 | -2034407.979510 | -30.708665 | 2034407.979510 | 30.708665 | 101.111522 |
| mae | op | -0.166421 | -16.782842 | -5.638626 | 16.782842 | 5.638626 | 27.362364 |
| mae | total | -0.610595 | -54.126186 | -54.126186 | 54.126186 | 54.126186 | 100.391784 |
| mae | wtmad | 0.590030 | 11610.572465 | undefined | -11610.572465 | undefined | -97.010665 |
| jw | relchem | -0.662494 | -15.098891 | -0.256915 | 15.098891 | 0.256915 | 1.001408 |
| jw | ae17 | 0.283546 | 2399.192852 | 0.123349 | -2399.192852 | -0.123349 | -0.428600 |
| jw | exc | 0.289205 | 8795.724898 | 0.132768 | -8795.724898 | -0.132768 | -0.437154 |
| jw | op | 0.752422 | 0.697593 | 0.234374 | -0.697593 | -0.234374 | -1.137340 |
| jw | total | 0.286609 | 0.233576 | 0.233576 | -0.233576 | -0.233576 | -0.433231 |
| jw | wtmad | 0.898334 | 162.517878 | undefined | -162.517878 | undefined | -1.357898 |

Norms: MAE 164.416449, J251-WTMAD-style 1.511572, relchem 15.077668, operator 0.613355, total 0.539150, Clean28 119.683465.
Combined-task cancellation ratio 0.488536.
mae directional derivative along the total descent splits as relchem -11.435545, ae17 29.214440, exc 30.708665, op 5.638626; sum 54.126186.
jw directional derivative along the total descent splits as relchem 0.256915, ae17 -0.123349, exc -0.132768, op -0.234374; sum -0.233576.

### continuation_raw_t251

| Metric | Task | Cosine | Dot | Weighted dot | Predicted change | Weighted predicted change | Unit predicted change |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mae | relchem | 0.224573 | 541.439566 | 9.212855 | -541.439566 | -9.212855 | -36.023339 |
| mae | ae17 | 0.744526 | 56815.910647 | 2.921051 | -56815.910647 | -2.921051 | -119.427846 |
| mae | exc | -0.594050 | -1899808.348546 | -28.676932 | 1899808.348546 | 28.676932 | 95.290261 |
| mae | op | -0.116251 | -11.552158 | -3.881243 | 11.552158 | 3.881243 | 18.647611 |
| mae | total | -0.551837 | -20.424270 | -20.424270 | 20.424270 | 20.424270 | 88.518968 |
| mae | wtmad | 0.560785 | 7576.675393 | undefined | -7576.675393 | undefined | -89.954280 |
| jw | relchem | -0.673026 | -15.456857 | -0.263006 | 15.456857 | 0.263006 | 1.028384 |
| jw | ae17 | 0.036042 | 26.199776 | 0.001347 | -26.199776 | -0.001347 | -0.055072 |
| jw | exc | 0.300123 | 9142.883822 | 0.138009 | -9142.883822 | -0.138009 | -0.458587 |
| jw | op | 0.765154 | 0.724288 | 0.243343 | -0.724288 | -0.243343 | -1.169154 |
| jw | total | 0.339497 | 0.119693 | 0.119693 | -0.119693 | -0.119693 | -0.518751 |
| jw | wtmad | 0.872650 | 112.310336 | undefined | -112.310336 | undefined | -1.333407 |

Norms: MAE 160.407890, J251-WTMAD-style 1.527999, relchem 15.030244, operator 0.619498, total 0.230733, Clean28 84.228070.
Combined-task cancellation ratio 0.707668.
mae directional derivative along the total descent splits as relchem -9.212855, ae17 -2.921051, exc 28.676932, op 3.881243; sum 20.424270.
jw directional derivative along the total descent splits as relchem 0.263006, ae17 -0.001347, exc -0.138009, op -0.243343; sum -0.119693.

Predicted change is `-dot(g_metric, direction)`. Positive means descent along that direction is predicted to increase the diagnostic metric. Weighted uses each frozen production coefficient once. These are not AdamW steps.

## Training subsets

### b8_t128

| Database | Count | A_d | Mean abs error | JW scalar | Gradient norm | Cosine JW | Cosine relchem | Cosine operator | Dot relchem | Dot operator | Leave-one-out cosine relchem |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DBH76 | 70 | 18.836000 | 7.857969 | 0.116345 | 1.042494 | 0.942473 | -0.837531 | 0.890864 | -13.164628 | 0.569636 | -0.202501 |
| NCCE31 | 28 | 3.536786 | 0.717074 | 0.022617 | 0.552610 | 0.974918 | -0.632007 | 0.737603 | -5.265930 | 0.250008 | -0.665079 |
| EA13 | 11 | 38.680000 | 2.560592 | 0.002901 | 0.061843 | -0.609294 | 0.145683 | -0.252885 | 0.135842 | -0.009592 | -0.651870 |
| ABDE4 | 4 | 93.422501 | 3.000499 | 0.000512 | 0.022845 | -0.920085 | 0.538779 | -0.634239 | 0.185584 | -0.008887 | -0.661428 |
| IP13 | 13 | 253.828461 | 3.505517 | 0.000715 | 0.028363 | 0.132878 | 0.461491 | -0.358373 | 0.197354 | -0.006234 | -0.672714 |
| PA8 | 8 | 163.325002 | 1.714795 | 0.000335 | 0.034809 | -0.894357 | 0.912686 | -0.952156 | 0.479009 | -0.020329 | -0.669684 |
| pTC13 | 13 | 168.787692 | 6.639701 | 0.002037 | 0.083483 | -0.878178 | 0.922071 | -0.963134 | 1.160634 | -0.049317 | -0.680202 |
| MGAE109 | 104 | 504.447019 | 13.334307 | 0.010953 | 0.316926 | 0.472271 | 0.245525 | -0.142453 | 1.173244 | -0.027691 | -0.776276 |

The most negative relchem dot is DBH76 (-13.164628, fraction of the total dot 0.871894). Leave-one-out keeps the original 1/251 denominator and the frozen A_d.

### continuation_raw_t251

| Database | Count | A_d | Mean abs error | JW scalar | Gradient norm | Cosine JW | Cosine relchem | Cosine operator | Dot relchem | Dot operator | Leave-one-out cosine relchem |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DBH76 | 70 | 18.836000 | 7.646430 | 0.113212 | 1.050050 | 0.945441 | -0.839645 | 0.897493 | -13.251707 | 0.583823 | -0.230964 |
| NCCE31 | 28 | 3.536786 | 0.656835 | 0.020717 | 0.554418 | 0.975125 | -0.633924 | 0.737882 | -5.282517 | 0.253434 | -0.680333 |
| EA13 | 11 | 38.680000 | 2.574011 | 0.002916 | 0.061538 | -0.606123 | 0.142529 | -0.242294 | 0.131830 | -0.009237 | -0.662269 |
| ABDE4 | 4 | 93.422501 | 3.167337 | 0.000540 | 0.022954 | -0.919492 | 0.540906 | -0.636833 | 0.186615 | -0.009056 | -0.671860 |
| IP13 | 13 | 253.828461 | 3.426262 | 0.000699 | 0.028446 | 0.121348 | 0.468957 | -0.374054 | 0.200502 | -0.006592 | -0.683183 |
| PA8 | 8 | 163.325002 | 1.767348 | 0.000345 | 0.034982 | -0.899675 | 0.913674 | -0.954784 | 0.480395 | -0.020691 | -0.679907 |
| MGAE109 | 104 | 504.447019 | 12.297020 | 0.010101 | 0.309308 | 0.501532 | 0.196799 | -0.090404 | 0.914916 | -0.017323 | -0.778759 |
| pTC13 | 13 | 168.787692 | 6.729296 | 0.002065 | 0.083832 | -0.883811 | 0.923097 | -0.964106 | 1.163110 | -0.050069 | -0.689997 |

The most negative relchem dot is DBH76 (-13.251707, fraction of the total dot 0.857335). Leave-one-out keeps the original 1/251 denominator and the frozen A_d.

## Reactions

### b8_t128

Most negative contributions to `dot(g_JW, g_relchem)`:

| Database | Reaction | Signed residual | JW scalar | Gradient norm | Dot relchem | Dot operator |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| NCCE31 | 5 | 0.751609 | 0.000847 | 0.121861 | -0.965301 | 0.047025 |
| DBH76 | 39 | -28.087133 | 0.005941 | 0.057702 | -0.644134 | 0.028911 |
| DBH76 | 71 | -1.150002 | 0.000243 | 0.042959 | -0.596921 | 0.025123 |
| DBH76 | 27 | -13.482958 | 0.002852 | 0.047981 | -0.576572 | 0.025383 |
| DBH76 | 10 | -13.278347 | 0.002809 | 0.045632 | -0.523838 | 0.023426 |
| NCCE31 | 4 | 0.201151 | 0.000227 | 0.071051 | -0.507876 | 0.025404 |
| DBH76 | 73 | -1.973685 | 0.000417 | 0.032686 | -0.465579 | 0.019481 |
| DBH76 | 14 | -10.866101 | 0.002298 | 0.039677 | -0.450532 | 0.020170 |

Largest J251-WTMAD-style scalar contributions:

| Database | Reaction | Absolute residual | JW scalar | Gradient norm | Dot relchem |
| --- | --- | ---: | ---: | ---: | ---: |
| DBH76 | 39 | 28.087133 | 0.005941 | 0.057702 | -0.644134 |
| DBH76 | 47 | 23.194514 | 0.004906 | 0.041878 | -0.405112 |
| DBH76 | 26 | 16.824441 | 0.003559 | 0.032073 | -0.371304 |
| DBH76 | 49 | 16.518217 | 0.003494 | 0.017653 | -0.201966 |
| DBH76 | 45 | 14.710149 | 0.003111 | 0.036915 | -0.355625 |

No reaction is in the top 15 scalar contributions and the bottom half of gradient norms.

### continuation_raw_t251

Most negative contributions to `dot(g_JW, g_relchem)`:

| Database | Reaction | Signed residual | JW scalar | Gradient norm | Dot relchem | Dot operator |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| NCCE31 | 5 | 0.538657 | 0.000607 | 0.122071 | -0.963015 | 0.047329 |
| DBH76 | 39 | -27.900654 | 0.005901 | 0.057856 | -0.644156 | 0.029277 |
| DBH76 | 71 | -1.096852 | 0.000232 | 0.043326 | -0.600796 | 0.025751 |
| DBH76 | 27 | -13.368870 | 0.002828 | 0.048084 | -0.576642 | 0.025649 |
| DBH76 | 10 | -13.072245 | 0.002765 | 0.045698 | -0.523479 | 0.023669 |
| NCCE31 | 4 | 0.027423 | 0.000031 | 0.071409 | -0.510718 | 0.025863 |
| DBH76 | 73 | -1.835058 | 0.000388 | 0.032857 | -0.467282 | 0.019858 |
| DBH76 | 14 | -10.683237 | 0.002260 | 0.039762 | -0.450550 | 0.020419 |

Largest J251-WTMAD-style scalar contributions:

| Database | Reaction | Absolute residual | JW scalar | Gradient norm | Dot relchem |
| --- | --- | ---: | ---: | ---: | ---: |
| DBH76 | 39 | 27.900654 | 0.005901 | 0.057856 | -0.644156 |
| DBH76 | 47 | 22.868974 | 0.004837 | 0.042299 | -0.414050 |
| DBH76 | 26 | 16.662956 | 0.003524 | 0.032125 | -0.371681 |
| DBH76 | 49 | 16.212610 | 0.003429 | 0.017726 | -0.201943 |
| DBH76 | 45 | 14.399435 | 0.003046 | 0.037285 | -0.364387 |

No reaction is in the top 15 scalar contributions and the bottom half of gradient norms.

## Parameter blocks

### b8_t128

| Metric | Task | Block | Metric share | Block cosine | Full cosine | Sign reversal |
| --- | --- | --- | ---: | ---: | ---: | --- |
| mae | relchem | exchange | 0.888082 | 0.249614 | 0.271102 | no |
| mae | relchem | correlation | 0.059899 | 0.618030 | 0.271102 | no |
| mae | relchem | normalization | 0.052019 | 0.775725 | 0.271102 | no |
| mae | op | exchange | 0.888082 | -0.147015 | -0.166421 | no |
| mae | op | correlation | 0.059899 | -0.727250 | -0.166421 | no |
| mae | op | normalization | 0.052019 | -0.758743 | -0.166421 | no |
| mae | total | exchange | 0.888082 | -0.611905 | -0.610595 | no |
| mae | total | correlation | 0.059899 | -0.007922 | -0.610595 | no |
| mae | total | normalization | 0.052019 | -0.894980 | -0.610595 | no |
| jw | relchem | exchange | 0.977647 | -0.685526 | -0.662494 | no |
| jw | relchem | correlation | 0.015340 | 0.883467 | -0.662494 | yes |
| jw | relchem | normalization | 0.007014 | -0.201076 | -0.662494 | no |
| jw | op | exchange | 0.977647 | 0.768553 | 0.752422 | no |
| jw | op | correlation | 0.015340 | -0.768486 | 0.752422 | yes |
| jw | op | normalization | 0.007014 | 0.249923 | 0.752422 | no |
| jw | total | exchange | 0.977647 | 0.303045 | 0.286609 | no |
| jw | total | correlation | 0.015340 | 0.389484 | 0.286609 | no |
| jw | total | normalization | 0.007014 | -0.087509 | 0.286609 | no |

### continuation_raw_t251

| Metric | Task | Block | Metric share | Block cosine | Full cosine | Sign reversal |
| --- | --- | --- | ---: | ---: | ---: | --- |
| mae | relchem | exchange | 0.900436 | 0.201322 | 0.224573 | no |
| mae | relchem | correlation | 0.048470 | 0.578769 | 0.224573 | no |
| mae | relchem | normalization | 0.051094 | 0.762412 | 0.224573 | no |
| mae | op | exchange | 0.900436 | -0.094367 | -0.116251 | no |
| mae | op | correlation | 0.048470 | -0.686534 | -0.116251 | no |
| mae | op | normalization | 0.051094 | -0.740712 | -0.116251 | no |
| mae | total | exchange | 0.900436 | -0.546913 | -0.551837 | no |
| mae | total | correlation | 0.048470 | 0.244071 | -0.551837 | yes |
| mae | total | normalization | 0.051094 | -0.891865 | -0.551837 | no |
| jw | relchem | exchange | 0.980801 | -0.693132 | -0.673026 | no |
| jw | relchem | correlation | 0.012217 | 0.874966 | -0.673026 | yes |
| jw | relchem | normalization | 0.006982 | -0.224398 | -0.673026 | no |
| jw | op | exchange | 0.980801 | 0.779420 | 0.765154 | no |
| jw | op | correlation | 0.012217 | -0.733262 | 0.765154 | yes |
| jw | op | normalization | 0.006982 | 0.279428 | 0.765154 | no |
| jw | total | exchange | 0.980801 | 0.355414 | 0.339497 | no |
| jw | total | correlation | 0.012217 | 0.670796 | 0.339497 | no |
| jw | total | normalization | 0.006982 | -0.068377 | 0.339497 | no |

## Comparison with the saved Clean28 audit

| Checkpoint | cos(JW, relchem) | cos(Clean28, relchem) | cos(JW, operator) | cos(Clean28, operator) | cos(JW, Clean28) | cos(MAE, relchem) | cos(MAE, operator) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| b8_t128 | -0.662494 | -0.559828 | 0.752422 | 0.643603 | 0.898334 | 0.271102 | -0.166421 |
| continuation_raw_t251 | -0.673026 | -0.593576 | 0.765154 | 0.665669 | 0.872650 | 0.224573 | -0.116251 |

## Numerical checks

`b8_t128` exact zero residuals: none.
- jw target 1e-02: relative gap 2.174e-02, epsilon 6.616e-03
- jw target 1e-03: relative gap 2.505e-04, epsilon 6.616e-04
- jw target 1e-04: relative gap 3.083e-05, epsilon 6.616e-05
- jw smallest relative gap 3.083e-05 at target 1e-04.
- mae target 1e-03: relative gap 5.797e-04, epsilon 6.082e-06
- mae smallest relative gap 5.797e-04 at target 1e-03.

`continuation_raw_t251` exact zero residuals: none.
- jw target 1e-02: relative gap 8.140e-02, epsilon 6.545e-03
- jw target 1e-03: relative gap 1.442e-02, epsilon 6.545e-04
- jw target 1e-04: relative gap 7.585e-06, epsilon 6.545e-05
- jw smallest relative gap 7.585e-06 at target 1e-04.
- mae target 1e-03: relative gap 3.823e-04, epsilon 6.234e-06
- mae smallest relative gap 3.823e-04 at target 1e-03.

Finite differences perturb float32 parameters and repeat the no-grad kcal/mol energy path. At both checkpoints the smallest J251-WTMAD-style step is the closest match. The larger steps show curvature.

## Conclusion

No verification gate failed. Both relchem receipts were reproduced, every gradient has 9,446 coordinates in the saved order, the styled-metric gradient matches the sum of its reaction contributions, and no residual is exactly zero.

The recorded outcome is C. Equal-weight J251-MAE and reference-normalized J251-WTMAD-style do not share an alignment with relchem, so this is not a single objective-mismatch or generalization-mismatch label.

At `b8_t128`, cos(MAE, relchem) is 0.271102 and cos(JW, relchem) is -0.662494. cos(JW, Clean28) is 0.898334; cos(Clean28, relchem) is -0.559828. cos(JW, operator) is 0.752422, cos(Clean28, operator) is 0.643603, and cos(MAE, operator) is -0.166421. Descent on the combined training gradient changes JW by -0.233576 and MAE by 54.126186.

DBH76, with 70 reactions and A_d 18.836000 kcal/mol, contributes -13.164628 to dot(g_JW, g_relchem). Removing it without renormalizing leaves cosine -0.202501. NCCE31 has the smallest reference scale, 3.536786 kcal/mol, and contributes -5.265930; removing only NCCE31 leaves cosine -0.665079. The exchange block holds 0.977647 of the JW squared norm and has block cosine -0.685526 with relchem.

At `continuation_raw_t251`, cos(MAE, relchem) is 0.224573 and cos(JW, relchem) is -0.673026. cos(JW, Clean28) is 0.872650; cos(Clean28, relchem) is -0.593576. cos(JW, operator) is 0.765154, cos(Clean28, operator) is 0.665669, and cos(MAE, operator) is -0.116251. Descent on the combined training gradient changes JW by -0.119693 and MAE by 20.424270.

DBH76, with 70 reactions and A_d 18.836000 kcal/mol, contributes -13.251707 to dot(g_JW, g_relchem). Removing it without renormalizing leaves cosine -0.230964. NCCE31 has the smallest reference scale, 3.536786 kcal/mol, and contributes -5.282517; removing only NCCE31 leaves cosine -0.680333. The exchange block holds 0.980801 of the JW squared norm and has block cosine -0.693132 with relchem.

What is established at both frozen vectors: relchem is weakly aligned with equal-weight training absolute error and opposed to the reference-normalized training metric. That normalized gradient is also aligned with the saved Clean28 gradient. The Clean28 conflict with relchem is therefore already present on J251 once reactions are divided by the frozen A_d. It is not present for the unnormalized mean absolute error. The operator gradient aligns with J251-WTMAD-style and with Clean28; it does not align with J251-MAE. About 98% of the normalized-metric gradient lies in the exchange network, and that is where the opposition to relchem sits. The correlation-network sign reversal is a block of about 1–2% of the squared norm.

The subset decomposition supports a concentrated conflict. DBH76 is most of the net dot with relchem. After it is removed, the remaining gradient is still weakly opposed to relchem. NCCE31 adds more opposition, including reactions whose absolute errors are small and whose gradient dots are large, but removing NCCE31 alone does not remove the negative cosine. MGAE109 has the most reactions and the largest reference scale, so the 1/A_d factor gives it a modest JW contribution despite a large mean absolute error.

What was not isolated: the original relchem database and frequency factors were not ablated, so this audit does not show that one of those factors is the cause. The comparison that was performed is equal-weight absolute error against the same absolute errors divided by frozen A_d. Residual transformation, reference normalization, and dataset composition all remain visible; only the normalization step was switched on and off.

A negative cosine is not a reason to weaken or remove a physical loss. No training arm was run, and the model-selection gates are unchanged. The cross-dataset cosine was not used to fit any parameter, normalization, or optimizer setting.

## Git

Provenance is the commit that contains this report. The diagnostic started from `6342d46`.
