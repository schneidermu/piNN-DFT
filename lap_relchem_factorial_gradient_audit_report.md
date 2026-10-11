# Relchem loss factorial gradient audit

Diagnostic only. Optimizer updates: 0. The eight factorial objectives were not trained.

## Production formula

The Full251 relchem objective differentiated in the preceding audits is the arithmetic mean of 251 singleton calls. Each call is one J251 reaction, one quadrature variant, and weight 1 inside `ChemistryBatchObjective`.

`r_i = E_pred_i - E_ref_i` in kcal/mol, on the fixed PBE0-density production path.

`phi_orig(r) = sqrt(1e-20 + r^2)`.

`factor_d = FCHEM_DB_WEIGHTS[d] * FREQ_WEIGHTS[d] / MEAN_WEIGHT`.

`MEAN_WEIGHT` is the mean of `FCHEM_DB_WEIGHTS * FREQ_WEIGHTS` over the nine coded databases, including AE17, which is not one of the 251 relchem reactions. AE17 therefore changes only the global scale.

`L_relchem = (1/251) sum_i factor_d(i) * phi_orig(r_i)`.

This is `C * sum_i w_i * phi_i` with `C = 1/251`, `w_i = factor_d(i)`, and `phi_i = phi_orig`. The frequency weights are the legacy inverse database sizes (`1/76` for DBH76, `1/109` for MGAE109, `1/31` for NCCE31), not the J251 subset counts. The eight B8 presentations are not factors in this scalar. `phi_orig` has no reference-energy denominator. The `1e-20` term is a smoothing epsilon, not a reference scale.

F000 is equal-weight mean absolute error. F010 is the previous J251-WTMAD-style diagnostic. F101 uses the same penalty and legacy factors as production, with those factors rescaled to mean 1 over the 251 reactions. It is proportional to `L_relchem` by that mean factor. The proportionality was checked on the scalar and on the gradient.

Publication logical SHA256 `61c221a19b9987717e69cac182ad545241f8807db4126c0949a99992e4c210ef`. Evaluation manifest SHA256 `132bd550161be6bc6761f8df1de8d3755c0b0debab53663208c3a9bbf56cb805`. S5 was not loaded. Clean28 was not recomputed.

## Checkpoints

- `b8_t128` relchem `1.195939284232` against receipt `1.195939284232`. Chain-rule gradient relative L2 `2.738e-12`. Checkpoint SHA256 `1f38f7f8ea74334ff8f5124d2a6538619323d1f453296fa9ca6b3cd7958c437b`. Model SHA256 `c1c6ecee465decf386f6b6b33523c414bca7c1c5a4f93c1df15a9290a4f0760a`. Coordinates 9446.
- `continuation_raw_t251` relchem `1.170649184133` against receipt `1.170649184133`. Chain-rule gradient relative L2 `3.142e-12`. Checkpoint SHA256 `63d72b83dbc2ee49b466d366454ded341086547ddf1594893e15547c6410adf5`. Model SHA256 `491b6b1e02682c6fdc7ca7b4f3cb9808dd210395de29b253d98806a494b7c87a`. Coordinates 9446.

## Factorial gradients

### b8_t128

| Variant | Penalty | 1/A_d | Legacy | Scalar | Norm | cos F010 | cos relchem | cos operator | cos total | Predicted F010 change | Unit change |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F000 | L1 | no | no | 8.536568 | 164.416449 | 0.462192 | 0.271102 | -0.166421 | -0.610595 | -114.867371 | -0.698637 |
| F001 | L1 | no | yes | 2.228456 | 28.095009 | -0.662494 | 1.000000 | -0.984409 | -0.748305 | 28.134555 | 1.001408 |
| F010 | L1 | yes | no | 0.156415 | 1.511572 | 1.000000 | -0.662494 | 0.752422 | 0.286609 | -2.284851 | -1.511572 |
| F011 | L1 | yes | yes | 0.158164 | 2.894723 | 0.980410 | -0.631957 | 0.738042 | 0.176797 | -4.289864 | -1.481960 |
| F100 | original | no | no | 8.536568 | 164.416449 | 0.462192 | 0.271102 | -0.166421 | -0.610595 | -114.867371 | -0.698637 |
| F101 | original | no | yes | 2.228456 | 28.095009 | -0.662494 | 1.000000 | -0.984409 | -0.748305 | 28.134555 | 1.001408 |
| F110 | original | yes | no | 0.156415 | 1.511572 | 1.000000 | -0.662494 | 0.752422 | 0.286609 | -2.284851 | -1.511572 |
| F111 | original | yes | yes | 0.158164 | 2.894723 | 0.980410 | -0.631957 | 0.738042 | 0.176797 | -4.289864 | -1.481960 |

| | F000 | F001 | F010 | F011 | F100 | F101 | F110 | F111 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F000 | 1.000000 | 0.271102 | 0.462192 | 0.491310 | 1.000000 | 0.271102 | 0.462192 | 0.491310 |
| F001 | 0.271102 | 1.000000 | -0.662494 | -0.631957 | 0.271102 | 1.000000 | -0.662494 | -0.631957 |
| F010 | 0.462192 | -0.662494 | 1.000000 | 0.980410 | 0.462192 | -0.662494 | 1.000000 | 0.980410 |
| F011 | 0.491310 | -0.631957 | 0.980410 | 1.000000 | 0.491310 | -0.631957 | 0.980410 | 1.000000 |
| F100 | 1.000000 | 0.271102 | 0.462192 | 0.491310 | 1.000000 | 0.271102 | 0.462192 | 0.491310 |
| F101 | 0.271102 | 1.000000 | -0.662494 | -0.631957 | 0.271102 | 1.000000 | -0.662494 | -0.631957 |
| F110 | 0.462192 | -0.662494 | 1.000000 | 0.980410 | 0.462192 | -0.662494 | 1.000000 | 0.980410 |
| F111 | 0.491310 | -0.631957 | 0.980410 | 1.000000 | 0.491310 | -0.631957 | 0.980410 | 1.000000 |

### continuation_raw_t251

| Variant | Penalty | 1/A_d | Legacy | Scalar | Norm | cos F010 | cos relchem | cos operator | cos total | Predicted F010 change | Unit change |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F000 | L1 | no | no | 8.046518 | 160.407890 | 0.490508 | 0.224573 | -0.116251 | -0.551837 | -120.224914 | -0.749495 |
| F001 | L1 | no | yes | 2.181332 | 28.006640 | -0.673026 | 1.000000 | -0.986854 | -0.749348 | 28.801571 | 1.028384 |
| F010 | L1 | yes | no | 0.150596 | 1.527999 | 1.000000 | -0.673026 | 0.765154 | 0.339497 | -2.334780 | -1.527999 |
| F011 | L1 | yes | yes | 0.147366 | 2.905152 | 0.980528 | -0.634144 | 0.738943 | 0.224172 | -4.352633 | -1.498246 |
| F100 | original | no | no | 8.046518 | 160.407890 | 0.490508 | 0.224573 | -0.116251 | -0.551837 | -120.224914 | -0.749495 |
| F101 | original | no | yes | 2.181332 | 28.006640 | -0.673026 | 1.000000 | -0.986854 | -0.749348 | 28.801571 | 1.028384 |
| F110 | original | yes | no | 0.150596 | 1.527999 | 1.000000 | -0.673026 | 0.765154 | 0.339497 | -2.334780 | -1.527999 |
| F111 | original | yes | yes | 0.147366 | 2.905152 | 0.980528 | -0.634144 | 0.738943 | 0.224172 | -4.352633 | -1.498246 |

| | F000 | F001 | F010 | F011 | F100 | F101 | F110 | F111 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F000 | 1.000000 | 0.224573 | 0.490508 | 0.530188 | 1.000000 | 0.224573 | 0.490508 | 0.530188 |
| F001 | 0.224573 | 1.000000 | -0.673026 | -0.634144 | 0.224573 | 1.000000 | -0.673026 | -0.634144 |
| F010 | 0.490508 | -0.673026 | 1.000000 | 0.980528 | 0.490508 | -0.673026 | 1.000000 | 0.980528 |
| F011 | 0.530188 | -0.634144 | 0.980528 | 1.000000 | 0.530188 | -0.634144 | 0.980528 | 1.000000 |
| F100 | 1.000000 | 0.224573 | 0.490508 | 0.530188 | 1.000000 | 0.224573 | 0.490508 | 0.530188 |
| F101 | 0.224573 | 1.000000 | -0.673026 | -0.634144 | 0.224573 | 1.000000 | -0.673026 | -0.634144 |
| F110 | 0.490508 | -0.673026 | 1.000000 | 0.980528 | 0.490508 | -0.673026 | 1.000000 | 0.980528 |
| F111 | 0.530188 | -0.634144 | 0.980528 | 1.000000 | 0.530188 | -0.634144 | 0.980528 | 1.000000 |

## Paired factor changes

A cosine difference is an observable paired change, not an additive share of causation.

### b8_t128

| Edge | Factor | Pair cosine | cos(source, F010) | cos(target, F010) | cos(source, relchem) | cos(target, relchem) | Relchem sign flip | F010 sign flip |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| F000->F100 | penalty | 1.000000 | 0.462192 | 0.462192 | 0.271102 | 0.271102 | no | no |
| F010->F110 | penalty | 1.000000 | 1.000000 | 1.000000 | -0.662494 | -0.662494 | no | no |
| F001->F101 | penalty | 1.000000 | -0.662494 | -0.662494 | 1.000000 | 1.000000 | no | no |
| F011->F111 | penalty | 1.000000 | 0.980410 | 0.980410 | -0.631957 | -0.631957 | no | no |
| F000->F010 | reference_normalization | 0.462192 | 0.462192 | 1.000000 | 0.271102 | -0.662494 | yes | no |
| F100->F110 | reference_normalization | 0.462192 | 0.462192 | 1.000000 | 0.271102 | -0.662494 | yes | no |
| F001->F011 | reference_normalization | -0.631957 | -0.662494 | 0.980410 | 1.000000 | -0.631957 | yes | yes |
| F101->F111 | reference_normalization | -0.631957 | -0.662494 | 0.980410 | 1.000000 | -0.631957 | yes | yes |
| F000->F001 | legacy_weight | 0.271102 | 0.462192 | -0.662494 | 0.271102 | 1.000000 | no | yes |
| F010->F011 | legacy_weight | 0.980410 | 1.000000 | 0.980410 | -0.662494 | -0.631957 | no | no |
| F100->F101 | legacy_weight | 0.271102 | 0.462192 | -0.662494 | 0.271102 | 1.000000 | no | yes |
| F110->F111 | legacy_weight | 0.980410 | 1.000000 | 0.980410 | -0.662494 | -0.631957 | no | no |

### continuation_raw_t251

| Edge | Factor | Pair cosine | cos(source, F010) | cos(target, F010) | cos(source, relchem) | cos(target, relchem) | Relchem sign flip | F010 sign flip |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| F000->F100 | penalty | 1.000000 | 0.490508 | 0.490508 | 0.224573 | 0.224573 | no | no |
| F010->F110 | penalty | 1.000000 | 1.000000 | 1.000000 | -0.673026 | -0.673026 | no | no |
| F001->F101 | penalty | 1.000000 | -0.673026 | -0.673026 | 1.000000 | 1.000000 | no | no |
| F011->F111 | penalty | 1.000000 | 0.980528 | 0.980528 | -0.634144 | -0.634144 | no | no |
| F000->F010 | reference_normalization | 0.490508 | 0.490508 | 1.000000 | 0.224573 | -0.673026 | yes | no |
| F100->F110 | reference_normalization | 0.490508 | 0.490508 | 1.000000 | 0.224573 | -0.673026 | yes | no |
| F001->F011 | reference_normalization | -0.634144 | -0.673026 | 0.980528 | 1.000000 | -0.634144 | yes | yes |
| F101->F111 | reference_normalization | -0.634144 | -0.673026 | 0.980528 | 1.000000 | -0.634144 | yes | yes |
| F000->F001 | legacy_weight | 0.224573 | 0.490508 | -0.673026 | 0.224573 | 1.000000 | no | yes |
| F010->F011 | legacy_weight | 0.980528 | 1.000000 | 0.980528 | -0.673026 | -0.634144 | no | no |
| F100->F101 | legacy_weight | 0.224573 | 0.490508 | -0.673026 | 0.224573 | 1.000000 | no | yes |
| F110->F111 | legacy_weight | 0.980528 | 1.000000 | 0.980528 | -0.673026 | -0.634144 | no | no |

## Databases

### b8_t128

| Variant | Database | Count | A_d | Objective | Gradient norm | Dot F010 | Dot relchem | Dot operator | Leave-one-out cos F010 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F000 | ABDE4 | 4 | 93.422501 | 0.047817 | 2.134255 | -2.968267 | 17.337684 | -0.830254 | 0.470191 |
| F000 | DBH76 | 70 | 18.836000 | 2.191465 | 19.636425 | 27.974379 | -247.968943 | 10.729657 | 0.353442 |
| F000 | EA13 | 11 | 38.680000 | 0.112217 | 2.392092 | -2.203096 | 5.254381 | -0.371033 | 0.465958 |
| F000 | IP13 | 13 | 253.828461 | 0.181561 | 7.199288 | 1.446013 | 50.094157 | -1.582472 | 0.472942 |
| F000 | MGAE109 | 104 | 504.447019 | 5.524972 | 159.872401 | 114.128385 | 591.839513 | -13.968714 | 0.058587 |
| F000 | NCCE31 | 28 | 3.536786 | 0.079992 | 1.954464 | 2.880214 | -18.624465 | 0.884224 | 0.453264 |
| F000 | PA8 | 8 | 163.325002 | 0.054655 | 5.685133 | -7.685647 | 78.234144 | -3.320173 | 0.490958 |
| F000 | pTC13 | 13 | 168.787692 | 0.343889 | 14.090855 | -18.704609 | 195.900659 | -8.324077 | 0.531512 |
| F001 | ABDE4 | 4 | 93.422501 | 0.200020 | 8.927695 | -12.416411 | 72.524403 | -3.472996 | -0.424959 |
| F001 | DBH76 | 70 | 18.836000 | 0.482474 | 4.323166 | 6.158855 | -54.592977 | 2.362247 | -0.713354 |
| F001 | EA13 | 11 | 38.680000 | 0.144434 | 3.078843 | -2.835589 | 6.762874 | -0.477553 | -0.601747 |
| F001 | IP13 | 13 | 253.828461 | 0.233685 | 9.266149 | 1.861152 | 64.475809 | -2.036788 | -0.787543 |
| F001 | MGAE109 | 104 | 504.447019 | 0.179157 | 5.184142 | 3.700812 | 19.191430 | -0.452960 | -0.771782 |
| F001 | NCCE31 | 28 | 3.536786 | 0.431758 | 10.549188 | 15.545911 | -100.525254 | 4.772585 | -0.809211 |
| F001 | PA8 | 8 | 163.325002 | 0.114312 | 11.890599 | -16.074724 | 163.628680 | -6.944225 | -0.445363 |
| F001 | pTC13 | 13 | 168.787692 | 0.442617 | 18.136233 | -24.074560 | 252.142249 | -10.713856 | -0.200985 |
| F010 | ABDE4 | 4 | 93.422501 | 0.000512 | 0.022845 | -0.031773 | 0.185584 | -0.008887 | 0.999983 |
| F010 | DBH76 | 70 | 18.836000 | 0.116345 | 1.042494 | 1.485155 | -13.164628 | 0.569636 | 0.835107 |
| F010 | EA13 | 11 | 38.680000 | 0.002901 | 0.061843 | -0.056957 | 0.135842 | -0.009592 | 0.999499 |
| F010 | IP13 | 13 | 253.828461 | 0.000715 | 0.028363 | 0.005697 | 0.197354 | -0.006234 | 0.999826 |
| F010 | MGAE109 | 104 | 504.447019 | 0.010953 | 0.316926 | 0.226245 | 1.173244 | -0.027691 | 0.979604 |
| F010 | NCCE31 | 28 | 3.536786 | 0.022617 | 0.552610 | 0.814359 | -5.265930 | 0.250008 | 0.992103 |
| F010 | PA8 | 8 | 163.325002 | 0.000335 | 0.034809 | -0.047057 | 0.479009 | -0.020329 | 0.999949 |
| F010 | pTC13 | 13 | 168.787692 | 0.002037 | 0.083483 | -0.110817 | 1.160634 | -0.049317 | 0.999683 |
| F011 | ABDE4 | 4 | 93.422501 | 0.002141 | 0.095563 | -0.132906 | 0.776306 | -0.037175 | 0.979967 |
| F011 | DBH76 | 70 | 18.836000 | 0.025614 | 0.229516 | 0.326973 | -2.898332 | 0.125411 | 0.974076 |
| F011 | EA13 | 11 | 38.680000 | 0.003734 | 0.079598 | -0.073309 | 0.174842 | -0.012346 | 0.978077 |
| F011 | IP13 | 13 | 253.828461 | 0.000921 | 0.036506 | 0.007332 | 0.254013 | -0.008024 | 0.981678 |
| F011 | MGAE109 | 104 | 504.447019 | 0.000355 | 0.010277 | 0.007336 | 0.038044 | -0.000898 | 0.980483 |
| F011 | NCCE31 | 28 | 3.536786 | 0.122076 | 2.982705 | 4.395491 | -28.422772 | 1.349413 | -0.497189 |
| F011 | PA8 | 8 | 163.325002 | 0.000700 | 0.072803 | -0.098422 | 1.001859 | -0.042518 | 0.981046 |
| F011 | pTC13 | 13 | 168.787692 | 0.002622 | 0.107450 | -0.142632 | 1.493843 | -0.063475 | 0.980975 |

### continuation_raw_t251

| Variant | Database | Count | A_d | Objective | Gradient norm | Dot F010 | Dot relchem | Dot operator | Leave-one-out cos F010 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F000 | ABDE4 | 4 | 93.422501 | 0.050475 | 2.144417 | -3.012868 | 17.433995 | -0.846009 | 0.498259 |
| F000 | DBH76 | 70 | 18.836000 | 2.132470 | 19.778750 | 28.573039 | -249.609156 | 10.996885 | 0.379994 |
| F000 | EA13 | 11 | 38.680000 | 0.112805 | 2.380302 | -2.204531 | 5.099190 | -0.357285 | 0.493789 |
| F000 | IP13 | 13 | 253.828461 | 0.177456 | 7.220376 | 1.338801 | 50.893080 | -1.673147 | 0.503156 |
| F000 | MGAE109 | 104 | 504.447019 | 5.095180 | 156.029663 | 119.571702 | 461.526427 | -8.738453 | 0.051010 |
| F000 | NCCE31 | 28 | 3.536786 | 0.073272 | 1.960858 | 2.921660 | -18.683130 | 0.896340 | 0.481729 |
| F000 | PA8 | 8 | 163.325002 | 0.056330 | 5.713383 | -7.854202 | 78.460433 | -3.379390 | 0.519339 |
| F000 | pTC13 | 13 | 168.787692 | 0.348529 | 14.149740 | -19.108688 | 196.318727 | -8.451099 | 0.559695 |
| F001 | ABDE4 | 4 | 93.422501 | 0.211142 | 8.970204 | -12.602978 | 72.927278 | -3.538897 | -0.435317 |
| F001 | DBH76 | 70 | 18.836000 | 0.469486 | 4.354500 | 6.290656 | -54.954087 | 2.421080 | -0.723318 |
| F001 | EA13 | 11 | 38.680000 | 0.145191 | 3.063668 | -2.837435 | 6.563128 | -0.459859 | -0.612637 |
| F001 | IP13 | 13 | 253.828461 | 0.228402 | 9.293292 | 1.723161 | 65.504096 | -2.153494 | -0.798042 |
| F001 | MGAE109 | 104 | 504.447019 | 0.165220 | 5.059534 | 3.877321 | 14.965800 | -0.283360 | -0.778757 |
| F001 | NCCE31 | 28 | 3.536786 | 0.395487 | 10.583702 | 15.769612 | -100.841900 | 4.837981 | -0.817814 |
| F001 | PA8 | 8 | 163.325002 | 0.117815 | 11.949684 | -16.427262 | 164.101969 | -7.068080 | -0.455853 |
| F001 | pTC13 | 13 | 168.787692 | 0.448589 | 18.212023 | -24.594647 | 252.680341 | -10.877346 | -0.208491 |
| F010 | ABDE4 | 4 | 93.422501 | 0.000540 | 0.022954 | -0.032250 | 0.186615 | -0.009056 | 0.999983 |
| F010 | DBH76 | 70 | 18.836000 | 0.113212 | 1.050050 | 1.516938 | -13.251707 | 0.583823 | 0.842596 |
| F010 | EA13 | 11 | 38.680000 | 0.002916 | 0.061538 | -0.056994 | 0.131830 | -0.009237 | 0.999511 |
| F010 | IP13 | 13 | 253.828461 | 0.000699 | 0.028446 | 0.005274 | 0.200502 | -0.006592 | 0.999829 |
| F010 | MGAE109 | 104 | 504.447019 | 0.010101 | 0.309308 | 0.237035 | 0.914916 | -0.017323 | 0.981529 |
| F010 | NCCE31 | 28 | 3.536786 | 0.020717 | 0.554418 | 0.826078 | -5.282517 | 0.253434 | 0.992344 |
| F010 | PA8 | 8 | 163.325002 | 0.000345 | 0.034982 | -0.048089 | 0.480395 | -0.020691 | 0.999952 |
| F010 | pTC13 | 13 | 168.787692 | 0.002065 | 0.083832 | -0.113211 | 1.163110 | -0.050069 | 0.999700 |
| F011 | ABDE4 | 4 | 93.422501 | 0.002260 | 0.096018 | -0.134903 | 0.780618 | -0.037881 | 0.980036 |
| F011 | DBH76 | 70 | 18.836000 | 0.024925 | 0.231180 | 0.333970 | -2.917503 | 0.128535 | 0.973899 |
| F011 | EA13 | 11 | 38.680000 | 0.003754 | 0.079205 | -0.073357 | 0.169678 | -0.011889 | 0.978169 |
| F011 | IP13 | 13 | 253.828461 | 0.000900 | 0.036612 | 0.006789 | 0.258064 | -0.008484 | 0.981866 |
| F011 | MGAE109 | 104 | 504.447019 | 0.000328 | 0.010030 | 0.007686 | 0.029668 | -0.000562 | 0.980634 |
| F011 | NCCE31 | 28 | 3.536786 | 0.111821 | 2.992464 | 4.458741 | -28.512302 | 1.367903 | -0.496386 |
| F011 | PA8 | 8 | 163.325002 | 0.000721 | 0.073165 | -0.100580 | 1.004757 | -0.043276 | 0.981293 |
| F011 | pTC13 | 13 | 168.787692 | 0.002658 | 0.107899 | -0.145714 | 1.497031 | -0.064444 | 0.981287 |

## Reactions

### b8_t128

Largest negative contributions of the production-proportional variant F101 to `dot(g_F010, g_F101)`:

| Database | Reaction | Residual | A_d | Legacy weight | Jacobian norm | Dot F010 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ABDE4 | 2 | -1.963745 | 93.422501 | 4.183050 | 286.363645 | -6.575574 |
| ABDE4 | 3 | -6.159603 | 93.422501 | 4.183050 | 274.555035 | -6.347410 |
| PA8 | 2 | 4.018700 | 163.325002 | 2.091525 | 319.190415 | -3.663479 |
| PA8 | 0 | 0.438357 | 163.325002 | 2.091525 | 297.196463 | -3.298998 |
| PA8 | 5 | 2.098059 | 163.325002 | 2.091525 | 294.423588 | -3.178555 |
| pTC13 | 6 | 7.423240 | 168.787692 | 1.287092 | 427.138301 | -2.955327 |
| pTC13 | 5 | 5.923316 | 168.787692 | 1.287092 | 425.090110 | -2.952464 |
| pTC13 | 7 | 9.482906 | 168.787692 | 1.287092 | 427.027982 | -2.944920 |

### continuation_raw_t251

Largest negative contributions of the production-proportional variant F101 to `dot(g_F010, g_F101)`:

| Database | Reaction | Residual | A_d | Legacy weight | Jacobian norm | Dot F010 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ABDE4 | 2 | -2.220765 | 93.422501 | 4.183050 | 287.572858 | -6.673620 |
| ABDE4 | 3 | -6.494008 | 93.422501 | 4.183050 | 275.782633 | -6.441308 |
| PA8 | 2 | 4.127889 | 163.325002 | 2.091525 | 321.527223 | -3.748400 |
| PA8 | 0 | 0.546041 | 163.325002 | 2.091525 | 298.672334 | -3.371554 |
| PA8 | 5 | 2.175982 | 163.325002 | 2.091525 | 294.406890 | -3.238648 |
| pTC13 | 6 | 7.515527 | 168.787692 | 1.287092 | 428.921108 | -3.018066 |
| pTC13 | 5 | 6.020065 | 168.787692 | 1.287092 | 426.888639 | -3.014832 |
| pTC13 | 7 | 9.574234 | 168.787692 | 1.287092 | 428.720774 | -3.007313 |

## Parameter blocks

### b8_t128

| Variant | Exchange share | Exchange cos F010 | Exchange dot share | Correlation share | Correlation cos F010 | Correlation dot share | Normalization share | Remaining share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F000 | 0.888082 | 0.459444 | 0.926248 | 0.059899 | 0.870358 | 0.057081 | 0.052019 | 0.000000 |
| F001 | 0.966475 | -0.685526 | 1.005839 | 0.003828 | 0.883467 | -0.010219 | 0.029697 | 0.000000 |
| F010 | 0.977647 | 1.000000 | 0.977647 | 0.015340 | 1.000000 | 0.015340 | 0.007014 | 0.000000 |
| F011 | 0.978668 | 0.981855 | 0.979599 | 0.013600 | 0.914140 | 0.013468 | 0.007732 | 0.000000 |
| F100 | 0.888082 | 0.459444 | 0.926248 | 0.059899 | 0.870358 | 0.057081 | 0.052019 | 0.000000 |
| F101 | 0.966475 | -0.685526 | 1.005839 | 0.003828 | 0.883467 | -0.010219 | 0.029697 | 0.000000 |
| F110 | 0.977647 | 1.000000 | 0.977647 | 0.015340 | 1.000000 | 0.015340 | 0.007014 | 0.000000 |
| F111 | 0.978668 | 0.981855 | 0.979599 | 0.013600 | 0.914140 | 0.013468 | 0.007732 | 0.000000 |

### continuation_raw_t251

| Variant | Exchange share | Exchange cos F010 | Exchange dot share | Correlation share | Correlation cos F010 | Correlation dot share | Normalization share | Remaining share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F000 | 0.900436 | 0.491884 | 0.942397 | 0.048470 | 0.850983 | 0.042217 | 0.051094 | 0.000000 |
| F001 | 0.967398 | -0.693132 | 1.003175 | 0.003071 | 0.874966 | -0.007963 | 0.029531 | 0.000000 |
| F010 | 0.980801 | 1.000000 | 0.980801 | 0.012217 | 1.000000 | 0.012217 | 0.006982 | 0.000000 |
| F011 | 0.981190 | 0.981939 | 0.982407 | 0.011091 | 0.903916 | 0.010731 | 0.007719 | 0.000000 |
| F100 | 0.900436 | 0.491884 | 0.942397 | 0.048470 | 0.850983 | 0.042217 | 0.051094 | 0.000000 |
| F101 | 0.967398 | -0.693132 | 1.003175 | 0.003071 | 0.874966 | -0.007963 | 0.029531 | 0.000000 |
| F110 | 0.980801 | 1.000000 | 0.980801 | 0.012217 | 1.000000 | 0.012217 | 0.006982 | 0.000000 |
| F111 | 0.981190 | 0.981939 | 0.982407 | 0.011091 | 0.903916 | 0.010731 | 0.007719 | 0.000000 |

## Numerical checks

- `b8_t128` F100 symmetric difference at target 1e-03: relative gap 5.797e-04, epsilon 6.082e-06.
- Exact zero residuals: none.
- `continuation_raw_t251` F100 symmetric difference at target 1e-03: relative gap 3.823e-04, epsilon 6.234e-06.
- Exact zero residuals: none.

## Conclusion

No Clean28 evaluation was repeated. The Full251 relchem scalar and the chain-rule gradient matched the saved production autodiff at both checkpoints. F000 and F010 reproduced the previous MAE and JW scalars and their cosines with relchem and the operator.

The production Full251 objective is the mean of 251 singleton penalties. For one reaction, `batch_fchem` returns `factor_d * sqrt(1e-20 + r^2)`, and `factor_d = database_weight_d * frequency_weight_d / mean_database_factor`. The factor multiplies the penalty. It is not inside the square root, and the penalty does not divide by `A_d` or by `|E_ref|`. The eight cells are therefore a real factorization of that singleton-mean objective. `batch_fchem` also contains an unused grouped-RMSE branch that would apply if several reactions of one database were passed to one call. The Full251 gradient does not use that branch: each call receives one reaction.

At `b8_t128`, every penalty edge has pair cosine 1.000000 and the derivative gap is 0.000000. Adding external `1/A_d` changes the cosine with relchem on F000->F010 from 0.271102 to -0.662494, F100->F110 from 0.271102 to -0.662494, F001->F011 from 1.000000 to -0.631957, F101->F111 from 1.000000 to -0.631957. Adding legacy weights changes the cosine with F010 on F000->F001 from 0.462192 to -0.662494, F010->F011 from 1.000000 to 0.980410, F100->F101 from 0.462192 to -0.662494, F110->F111 from 1.000000 to 0.980410.
Under the production-proportional gradient F101, database contributions to `dot(g_F010, g_F101)` run from pTC13 -24.074560, PA8 -16.074724, ABDE4 -12.416411, EA13 -2.835589, IP13 1.861152, MGAE109 3.700812, DBH76 6.158855, NCCE31 15.545911. DBH76 after `1/A_d`, inside F010, contributes -13.164628 to `dot(g_F010, g_relchem)`.

At `continuation_raw_t251`, every penalty edge has pair cosine 1.000000 and the derivative gap is 0.000000. Adding external `1/A_d` changes the cosine with relchem on F000->F010 from 0.224573 to -0.673026, F100->F110 from 0.224573 to -0.673026, F001->F011 from 1.000000 to -0.634144, F101->F111 from 1.000000 to -0.634144. Adding legacy weights changes the cosine with F010 on F000->F001 from 0.490508 to -0.673026, F010->F011 from 1.000000 to 0.980528, F100->F101 from 0.490508 to -0.673026, F110->F111 from 1.000000 to 0.980528.
Under the production-proportional gradient F101, database contributions to `dot(g_F010, g_F101)` run from pTC13 -24.594647, PA8 -16.427262, ABDE4 -12.602978, EA13 -2.837435, IP13 1.723161, MGAE109 3.877321, DBH76 6.290656, NCCE31 15.769612. DBH76 after `1/A_d`, inside F010, contributes -13.251707 to `dot(g_F010, g_relchem)`.

Q1. The original penalty is not responsible. With normalization and legacy weights held fixed, L1 and `sqrt(1e-20+r^2)` have pair cosine 1 at both checkpoints, and their derivatives agree at every retained residual. The production penalty contains no `A_d` and no `|E_ref|` denominator, so this comparison is a real loss-shape comparison. The shape does not move the gradient.

Q2. External `1/A_d` reverses the cosine with the production relchem gradient under both penalties and both weighting schemes. On equal weights the cosine with relchem goes from positive to about -0.66. On legacy weights it goes from +1 to about -0.63, and the cosine with F010 goes from about -0.66 to about +0.98. The unnormalized and normalized equal-weight gradients remain partly aligned with each other (pair cosine about 0.46 to 0.49). `1/A_d` changes the sign against relchem. It does not make the two gradients point opposite each other.

Q3. Legacy weights do reverse the gradient relative to F010 when external normalization is absent. F000 to F001 changes the cosine with F010 from +0.462 to -0.662 at t128 and from +0.491 to -0.673 at t251, and F001 is exactly the production relchem direction. When `1/A_d` is already present, the same weights do not reverse that sign: F010 and F011 stay aligned at cosine 0.980. Legacy weights create the production conflict. They do not keep it once reference normalization is added. The two factors interact, and `1/A_d` dominates the joint cell.

Q4. DBH76 does not generate the negative projection of the production gradient onto F010. That projection is positive for DBH76 under every factorial cell, including F101. At both checkpoints the negative F101 pieces are pTC13, PA8, and ABDE4, in that order. The equal-weight gradient is aligned with F010 mainly through MGAE109; the legacy frequency and database factors suppress that positive piece and leave the negative pieces in front. A second, different decomposition remains true: inside F010, the DBH76 slice is most of the negative dot with the full relchem gradient, and removing DBH76 from F010 without renormalizing weakens that cosine from about -0.66 to about -0.20. Removing pTC13 from F101 weakens its cosine with F010 from about -0.66 to about -0.20. The conflict is the interaction of those two weightings. It is not the original penalty, and it is not a large DBH76 legacy multiplier. The DBH76 frequency factor is `1/76`.

Q5. Locally, at both frozen vectors, the production relchem gradient is the legacy-weighted absolute-error gradient and is opposed to F010. That is a local fact about these two checkpoints. It is not a statement about the rest of the path or about AdamW.

Q6. Next experiments, ranked from this J251 geometry only. None is run here. Each keeps the operator, Exc, and AE17 objectives.

1. Keep the original per-reaction penalty and multiply it by the frozen external `1/A_d`. The cube puts that gradient on the F010 side whether or not the legacy factors remain (cosine 1 or 0.980). Judge the arm on the training-set normalized absolute error.

2. Do not spend an arm replacing `sqrt(1e-20+r^2)` with `|r|`. Those gradients are identical here.

3. If an `1/A_d` arm is not the one being run, the smaller weight experiment is to change the pTC13, PA8, ABDE4, and MGAE109 factors inside the current penalty. A DBH76-only frequency change is not the correction this cube supports: DBH76's own production-weighted contribution points with F010.

Both checkpoints agree on the penalty identity, on which edges flip, and on which databases carry the negative production projection. A negative cosine is not a reason to remove a physical loss.

## Git

Provenance is the commit that contains this report. The diagnostic started from `acf6a49`.
