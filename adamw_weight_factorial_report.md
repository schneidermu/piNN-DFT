# AdamW chemistry/operator 2×2 factorial screen

All 60 new updates completed (B/C/D:20 each); A reused its verified historical20 updates. No continuation was run. The LR1e-4 historical cursor59 checkpoint is unchanged. These are training-only limited-probe results, not exact full-corpus objectives, scientific qualification or validation scores.

## Control evolution

| Task | t0 loss | t20 loss | t59 loss | t20/t0 | t59/t0 |
|---|---:|---:|---:|---:|---:|
| relchem | 1.64235072 | 1.60828791 | 1.63528906 | 0.979260 | 0.995700 |
| ae17 | 23.5486691 | 13.1925189 | 5.94122293 | 0.560224 | 0.252295 |
| exc | 94.2354755 | 53.7110725 | 26.2641517 | 0.569967 | 0.278708 |
| op | 0.0302990028 | 0.0283766279 | 0.029041099 | 0.936553 | 0.958484 |

Only8/16 relchem probe reactions improved at t59 relative to t0. The control does not show strong consistent chemistry improvement; operator also worsens from t20 to t59 while remaining below t0. During updates21–59, actual rounded AdamW displacements predicted sampled-task ascent in relchem6/39, AE17 16/39, Exc14/39 and operator17/39. All stored movements are finite; these conflicts are permitted, not update-rejection gates.

## Frozen coefficients and control reuse

| Arm | relchem λ | AE17 λ | Exc λ | Operator λ |
|---|---:|---:|---:|---:|
| A | 0.017015480965588553 | 5.1412546183474142e-05 | 1.5094644512009712e-05 | 0.33597561607048215 |
| B | 0.034030961931177106 | 5.1412546183474142e-05 | 1.5094644512009712e-05 | 0.33597561607048215 |
| C | 0.017015480965588553 | 5.1412546183474142e-05 | 1.5094644512009712e-05 | 0.67195123214096431 |
| D | 0.034030961931177106 | 5.1412546183474142e-05 | 1.5094644512009712e-05 | 0.67195123214096431 |

Multipliers were applied directly without renormalization or recalibration. A=(1,1), B=(2,1), C=(1,2), D=(2,2). All arms use original seed11P536 tensor SHA `3c2e45d6a86cccf80d7b4dc458e67f9b406935ed60d4e57433bb441a6df88da6`; dataset logical SHA `61c221a19b9987717e69cac182ad545241f8807db4126c0949a99992e4c210ef`; sampling-manifest SHA `e84e237d88449edfae5c68f7caecd85791b57f60a4b6239a37e1ae1350a71089`. Constant LR1e-4, native AdamW betas(.9,.999), eps1e-8, weight_decay.01, unchanged Exc4096/operator256 and scientific precision. No SVRG.

Control reuse verifies trainer SHA, four scientific-source SHAs, constant-LR protocol, starting-model equality, manifest equality and bitwise-identical four initial raw F64 gradients for B/C/D versus A. Checkpoint/source/probe hashes are retained in JSON. Cursor59 SHA is `8938753bee6cfcb55ccdac9c02a15cbb3e3b2093290c9122e5e00ad400a966aa`. A read-only byte-identical copy under t59 was used to reuse the generic checkpoint20-named probe entrypoint; its actual checkpoint cursor is59.

## Four-arm probe ratios and paired uncertainty

Each cell: t20/t0 [paired bootstrap95% interval]. The frozen probe contains16 relchem,8 AE17 and8 mRKS systems shared for Exc/operator, population-stratum-weighted. Bootstrap:5000 paired within-stratum resamples, seed931002. It measures limited-panel sensitivity, not training-seed uncertainty or exact corpus accuracy. No validation is used.

| Arm | Relchem | AE17 | Exc | Operator |
|---|---|---|---|---|
| A | 0.979260 [0.914638, 1.037269] | 0.560224 [0.519426, 0.582552] | 0.569967 [0.545721, 0.589416] | 0.936553 [0.922172, 0.950245] |
| B | 0.961322 [0.902170, 1.008693] | 1.123040 [1.040436, 1.160007] | 1.124880 [1.105957, 1.138419] | 0.972375 [0.961128, 0.985079] |
| C | 0.996642 [0.936205, 1.053953] | 0.365822 [0.324563, 0.393820] | 0.377023 [0.344815, 0.403114] | 0.923563 [0.906371, 0.939774] |
| D | 0.976407 [0.911192, 1.035111] | 0.617143 [0.575561, 0.639328] | 0.626535 [0.603080, 0.644219] | 0.938998 [0.924456, 0.953105] |

| Arm/task | Ratio difference vs A | Paired95% interval |
|---|---:|---|
| B/relchem | -0.017938 | [-0.037995, -0.003789] |
| B/ae17 | +0.562817 | [+0.495779, +0.597204] |
| B/exc | +0.554914 | [+0.536148, +0.576250] |
| B/op | +0.035822 | [+0.033147, +0.039011] |
| C/relchem | +0.017383 | [+0.009888, +0.029081] |
| C/ae17 | -0.194401 | [-0.206168, -0.175187] |
| C/exc | -0.192944 | [-0.201143, -0.186248] |
| C/op | -0.012991 | [-0.015801, -0.010471] |
| D/relchem | -0.002852 | [-0.005466, -0.001240] |
| D/ae17 | +0.056919 | [+0.050477, +0.060721] |
| D/exc | +0.056568 | [+0.054537, +0.058748] |
| D/op | +0.002444 | [+0.002015, +0.002939] |

## Factorial effects

Effects are differences in normalized probe ratios; negative is improvement. Relchem main effect=(B+D−A−C)/2; operator main=(C+D−A−B)/2; interaction=D−B−C+A.

| Task | Effect | Estimate | Paired95% interval |
|---|---|---:|---|
| relchem | relchem_main | -0.019087 | [-0.033445, -0.009868] |
| relchem | operator_main | +0.016234 | [+0.008181, +0.028266] |
| relchem | interaction | -0.002297 | [-0.018278, +0.013686] |
| ae17 | relchem_main | +0.407069 | [+0.360501, +0.432045] |
| ae17 | operator_main | -0.350149 | [-0.371244, -0.310314] |
| ae17 | interaction | -0.311496 | [-0.331147, -0.270635] |
| exc | relchem_main | +0.402213 | [+0.388487, +0.417351] |
| exc | operator_main | -0.345644 | [-0.358698, -0.333950] |
| exc | interaction | -0.305402 | [-0.317839, -0.294308] |
| op | relchem_main | +0.025629 | [+0.023332, +0.028247] |
| op | operator_main | -0.023184 | [-0.025922, -0.020491] |
| op | interaction | -0.020387 | [-0.022496, -0.018796] |

Doubling relchem has a detectable favorable chemistry effect on this panel, but sacrifices the other tasks. Doubling operator improves operator and AE17/Exc while sacrificing chemistry. Chemistry interaction is unresolved; substantial negative interactions in AE17/Exc/operator mitigate the damage of B when both weights are doubled. This is evidence of coefficient-sensitive trade-offs, not proof that inadequate weights alone explain chemistry convergence. Every arm’s chemistry-versus-t0 interval still includes1.

## Independent direction/actual AdamW diagnostic

Three independent training-only draws were frozen before training (seed931003). All four raw gradients are measured at the same initial parameters. Actual first-step displacement includes native F32 AdamW normalization and weight decay; predicted ΔL=g·(theta_after−theta_before), so negative predicts descent. Raw gradient-space dots use the opposite descent-direction convention: positive g·d predicts descent. No finite trial loss or line search was used.

| Draw/arm | Step L2 | Predicted Δrelchem | ΔAE17 | ΔExc | Δoperator |
|---|---:|---:|---:|---:|---:|
| 0/A | 0.00967448 | +0.0059735248 | -1.748751 | -20.593798 | -0.0010938793 |
| 0/B | 0.00968700 | +0.0054656703 | -1.7586064 | -20.729016 | -0.0010470291 |
| 0/C | 0.00967436 | +0.006276349 | -1.6152666 | -18.846163 | -0.001176343 |
| 0/D | 0.00968619 | +0.0060327882 | -1.6091111 | -18.753655 | -0.0011701422 |
| 1/A | 0.00967550 | +0.004404054 | -18.802681 | -1.1992506 | -0.0005583632 |
| 1/B | 0.00968708 | +0.0018430216 | -17.825194 | -1.0654181 | -0.00052293965 |
| 1/C | 0.00967110 | +0.0049542848 | -18.881697 | -1.227592 | -0.00056855374 |
| 1/D | 0.00968622 | +0.0025576926 | -18.091567 | -1.116239 | -0.00054121738 |
| 2/A | 0.00965219 | -0.017182623 | -7.5425702 | -10.621568 | -0.0010334747 |
| 2/B | 0.00967759 | -0.023492281 | -5.6564334 | -8.2701005 | -0.00092574968 |
| 2/C | 0.00964878 | -0.016068999 | -7.5789451 | -10.6879 | -0.0010656293 |
| 2/D | 0.00967734 | -0.020394663 | -6.5652388 | -9.4225831 | -0.0010138714 |

Chemistry opposes all three other task gradients in draws0/1 (cosines from−.647 to−.854), and aligns in draw2. B improves predicted chemistry change in all3 draws, yet still predicts ascent in the first2. C improves operator in all3 while worsening chemistry. D’s chemistry prediction improves in2/3 and operator in2/3; no uniformly better direction exists on this small panel. Step norms remain approximately.00965–.00969 across all arms, demonstrating that coefficient doubling does not double AdamW displacement. All weighted-gradient pairwise cosines, task dots, normalized progresses and full task cosine matrices are in JSON.

## Chemistry reaction-level control distribution

| Database / identity | Variant | t0 loss | t20 loss | t59 loss |
|---|---|---:|---:|---:|
| ABDE4 / reaction_99331671b618716acb0d2398 | level3 | 1.6808778 | 0.34680862 | 1.5903927 |
| ABDE4 / reaction_18a4cfbaf87fde8157323b2d | level2_mura | 6.8572228 | 7.8452 | 9.898514 |
| DBH76 / reaction_71b6c9fe875ac94cb12eefa9 | level2_gauss_chebyshev | 1.0728452 | 0.93955526 | 1.0099973 |
| DBH76 / reaction_ff107a34e174c2d5873e9ef4 | level2_mura | 3.402851 | 3.1999753 | 3.2896512 |
| EA13 / reaction_1f5392cf2354f38876e28e0b | level3 | 0.51683232 | 1.6297597 | 1.1177886 |
| EA13 / reaction_fae3e20e06d2071806ca53ed | level2_delley | 1.9736495 | 2.7829375 | 2.6433215 |
| IP13 / reaction_c951e12704d4e9f0d83d7faa | level2_gauss_chebyshev | 2.5139818 | 2.535749 | 2.9003308 |
| IP13 / reaction_62fe6712dbf4bdeda18637d6 | level3 | 0.73324736 | 0.99333524 | 1.3156977 |
| MGAE109 / reaction_095a6d7e688cd5a2c0db804a | level3 | 0.71605737 | 0.54317597 | 0.55652456 |
| MGAE109 / reaction_d1bb588cb787979685e999cc | level2_delley | 0.71090328 | 0.49773338 | 0.54735035 |
| NCCE31 / reaction_39ea7fe5a023668c06e780b2 | level3_gauss_chebyshev | 4.6685709 | 4.5311664 | 4.4005075 |
| NCCE31 / reaction_a932b7106b3ffa0001c2236d | level3_gauss_chebyshev | 1.0675235 | 0.49437884 | 0.60548854 |
| PA8 / reaction_2141fbc9d785ade28191a592 | level3_gauss_chebyshev | 2.463234 | 0.16445189 | 1.4179273 |
| PA8 / reaction_a9a0d336c0d974685834be56 | level2_gauss_chebyshev | 1.1520924 | 2.7015807 | 1.8789251 |
| pTC13 / reaction_976cf34c643b73837e9db4ea | level3_gauss_chebyshev | 3.246149 | 5.1395723 | 4.2711285 |
| pTC13 / reaction_98e0120f8190c1b6ea1d880f | level3_delley | 2.0971193 | 3.9765425 | 3.1487028 |

## Runtime and integrity

| Arm | Mean update seconds | 20-update seconds | Peak live GiB |
|---|---:|---:|---:|
| A | 9.344 | 186.877 | 12.232 |
| B | 9.657 | 193.141 | 12.219 |
| C | 9.350 | 186.998 | 12.232 |
| D | 9.301 | 186.015 | 12.232 |

New training wall time summed over60 updates: 9.436 minutes; independent direction/probe/I/O overhead excluded. All arms saved optimizer moments, RNG, sampling cursor and parameter order externally at `C:\Dev\readWFN_share_ms\lap_adamw_weight_factorial_20261008`. Historical control remains at `C:\Dev\readWFN_share_ms\lap_adamw_lr_sweep_20261008/1e-4`. All hashes and initial gradients matched; every final parameter is finite. No historical training was rerun, and no physics/optimizer/dataset source was changed.

26 focused/relevant tests passed, compileall and diff-check PASS. Ruff PASS with a narrowly declared F401 exemption for the unused NumPy import retained in the frozen driver. Checkpoint/resume behavior reuses the qualified trainer and its tests.

## Decision

B is not a four-task improvement: AE17/Exc regress above t0 with panel intervals above1. C is an operator-oriented trade-off, not a chemistry correction. D keeps all four probe means below t0, but its chemistry benefit over A is small (ratio difference−.002852), and A is better on the other3 tasks. Do not automatically promote D; no scientifically superior weighting is established.

Recommended next experiment: a separately approved matched90-update A-versus-D continuation at constant LR1e-4. D resumes its t20 state; A resumes the preserved t59 state on the identical original trajectory, without rerunning completed updates. Preserve all moments and samples. This tests whether the small chemistry gain survives longer convergence and warrants the observed sacrifices. No continuation was launched.

No clean28, full251/full90 endpoint, SCF or Diet100 evaluation was performed. No architecture, loss definition, physical constraint, operator, dataset, sampler, precision boundary or optimizer algorithm was changed.
