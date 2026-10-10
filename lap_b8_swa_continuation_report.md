# B8 fixed-LR continuation with weight averaging

Classification: **NO-GO**.

One continuation from B8 t128. The learning rate was fixed at one quarter of the B8 rate. SWA was a readout and did not enter training.

## Initialization

- Parent checkpoint: `C:\Dev\readWFN_share_ms\lap_b8_adamw_20261010\ordinary_sgd_adamw\checkpoint_128.pt`
- Parent file SHA256: `1f38f7f8ea74334ff8f5124d2a6538619323d1f453296fa9ca6b3cd7958c437b`
- Optimizer moment SHA256: `4659e23cce1cf2c66f701b0b445b897d58838739ca15cffc9300fa0fdc1a40d0`
- Loaded model SHA256: `c1c6ecee465decf386f6b6b33523c414bca7c1c5a4f93c1df15a9290a4f0760a`
- Manifest SHA256: `b249964d410f0db4a59867de53bb35c6442ec23aa81eefb0b96b50242d94f6bc`
- Parent AdamW step: 128
- Trainable coordinates: 9446 float32
- Continuation learning rate: `2.5e-05` = 0.25 * `0.0001`
- Betas (0.9, 0.999), epsilon 1e-8, weight decay 0.01, foreach false, no clipping, no scheduler

## SWA

Post-update trainable parameters at local updates 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 251. Count 29. Accumulation is float64. Evaluation restores float32. Buffers, frozen tensors, and optimizer moments are not averaged.

## Clean28

| Candidate | Kind | Local update | Clean28 | Versus B8 t128 | Versus historical best |
| --- | --- | ---: | ---: | ---: | ---: |
| raw_64 | adamw | 64 | 9.115542884 | +0.018843 | +0.495849 |
| raw_128 | adamw | 128 | 9.089046807 | -0.007653 | +0.469353 |
| raw_192 | adamw | 192 | 9.098889822 | +0.002190 | +0.479196 |
| raw_251 | adamw | 251 | 9.087061816 | -0.009638 | +0.467368 |
| swa_251 | swa | 251 | 9.090915194 | -0.005785 | +0.471221 |

B8 t128 reference: 9.096700112. Historical best: 8.619694172. Target remains substantially below 7. Full30 was discarded.

Selected after all five evaluations: `raw_251` at Clean28 9.087061816.
No candidate is below the historical best, so the four scientific ratios were not evaluated.

## Full251 diagnostic

`raw_251` relchem objective 1.170649184, ratio 0.947532634 versus the corrected P536 reference 1.235471098. No gradients. This diagnostic is not scientific qualification.

| Database | Count | Mean | Contribution |
| --- | ---: | ---: | ---: |
| DBH76 | 70 | 0.903448091 | 0.251957635 |
| pTC13 | 13 | 4.648195327 | 0.240743184 |
| NCCE31 | 28 | 1.902623689 | 0.212244874 |
| IP13 | 13 | 2.366656926 | 0.122575857 |
| ABDE4 | 4 | 7.110373890 | 0.113312731 |
| MGAE109 | 104 | 0.213997282 | 0.088668197 |
| EA13 | 11 | 1.777973097 | 0.077919140 |
| PA8 | 8 | 1.983764914 | 0.063227567 |

## Clean28 reactions versus B8 t128

| Reaction | B8 t128 | Selected | Delta |
| --- | ---: | ---: | ---: |
| ACONF-10 | 13.302181598 | 15.671131034 | +2.368949 |
| Amino20x4-28 | 16.289362843 | 17.289740849 | +1.000378 |
| Amino20x4-54 | 0.507264556 | 0.530302097 | +0.023038 |
| BHPERI-11 | 19.079453849 | 17.888786955 | -1.190667 |
| BHROT27-16 | 15.098467006 | 15.006620534 | -0.091846 |
| BHROT27-26 | 2.178258935 | 2.088175508 | -0.090083 |
| BSR36-31 | 19.193255211 | 20.757539775 | +1.564285 |
| BUT14DIOL-13 | 1.947469185 | 0.943831403 | -1.003638 |
| CDIE20-9 | 8.740121341 | 8.882976517 | +0.142855 |
| DC13-1 | 2.530135221 | 2.548911357 | +0.018776 |
| DIPCS10-7 | 0.031848197 | 0.014076381 | -0.017772 |
| FH51-24 | 7.233258179 | 7.765910063 | +0.532652 |
| FH51-30 | 2.233310240 | 2.202256554 | -0.031054 |
| G21EA-14 | 0.154262712 | 0.002156719 | -0.152106 |
| HAL59-40 | 2.340318388 | 0.907273521 | -1.433045 |
| HAL59-57 | 1.007265592 | 2.163540287 | +1.156275 |
| HEAVY28-16 | 24.510507559 | 22.671539752 | -1.838968 |
| MB16-43-10 | 1.032896247 | 0.881546151 | -0.151350 |
| MCONF-1 | 1.503469324 | 2.230536049 | +0.727067 |
| PNICO23-16 | 2.072151224 | 1.358147536 | -0.714004 |
| PX13-9 | 15.986794029 | 15.727169167 | -0.259625 |
| S66-6 | 8.473876818 | 7.545002607 | -0.928874 |
| S66-50 | 1.008670700 | 1.659851947 | +0.651181 |
| SIE4x4-15 | 56.003635515 | 55.876619184 | -0.127016 |
| W4-11-30 | 0.306952672 | 0.355371036 | +0.048418 |
| W4-11-57 | 0.600730003 | 0.539107817 | -0.061622 |
| W4-11-132 | 5.897706214 | 5.805554490 | -0.092152 |
| WCPT18-15 | 25.443979769 | 25.124055548 | -0.319924 |

17 of 28 reactions have a lower weighted absolute error than B8 t128. Named comparisons include SIE4x4-15, WCPT18-15, HEAVY28-16, BSR36-31, BHPERI-11.

## Raw trajectory and SWA

Best raw Clean28 is 9.087061816 at `raw_251`. SWA Clean28 is 9.090915194. The average is not lower than the best raw checkpoint on this seed.
That comparison is an observed Clean28 difference. It is not evidence about loss-landscape flatness.

## Recommendation

The best Clean28 is 0.010 below B8 t128 and remains 0.467 above the historical best. The raw checkpoints do not form a descending sequence, and the SWA readout is 0.004 higher than the best raw checkpoint. Another 251 updates at 2.5e-5, or a different window of this average, is not supported by that gap. The next experiment should start again from B8 t128, keep the same loss, and change the update direction. The Full251 diagnostic moved farther than Clean28. That chemistry-objective change is not a reason to extend this arm.

## Resources

- Completed local updates: 251
- Cumulative optimizer step: 379
- Sum of recorded update seconds: 6542.834
- Peak allocated bytes: 14088923136
- Peak reserved bytes: 14206107648
- Final displacement from B8 t128: 1.090767e-01
- Norm-ratio min / median / max: 0.176086 / 0.650610 / 0.958734
- The first process stopped while recording displacement after an unsaved local update. Resume used the saved local-0 checkpoint, whose AdamW step was still 128, and then completed 251 new updates.

## Preflight

- Manifest index 0: chemistry relative L2 0.000e+00, scalar relative 0.000e+00
- Manifest index 125: chemistry relative L2 0.000e+00, scalar relative 0.000e+00
- Manifest index 250: chemistry relative L2 0.000e+00, scalar relative 0.000e+00
- Optimizer step during preflight: False
- SWA probe: True

## Git

Base commit before this report: `fb43755bea47e88258b96cca43c215b398dc70af`. The experiment commit is the child of that commit.
