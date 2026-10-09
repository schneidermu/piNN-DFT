# Residual-aware quadratic chemistry loss

**Decision: NO-GO**

One new 251-update four-task AdamW arm, matched to historical J251. The only intended change is the singleton relchem gradient multiplier `f = 0.5 * (1 + L_A / (a * s))`. J251 was not retrained.

## Frozen scale and parity

Training-only residual scale `s = 6.260755624276499`.
Calibration receipt SHA256: `50a1a62787805b9451241db514e19d72789f030c0f369415da7cdcb46da81cbb`.
This median uses the 251 J251 training-manifest variants at corrected P536. Clean28 did not enter the scale, the learning rate, or the four task coefficients.

`c = 0.5` anchors the derivative ratio near 1 at the initial median residual. It does not make the joint-gradient norm or the AdamW step identical to J251.

| Manifest index | Relative L2 | Qualified-path relative L2 | Factor |
|---:|---:|---:|---:|
| 0 | 2.186e-13 | 0.000e+00 | 1.900438 |
| 125 | 1.462e-12 | 0.000e+00 | 0.579156 |
| 250 | 1.080e-12 | 0.000e+00 | 3.636895 |

Joint preflight norm `0.5292902021797636`, factor applications `1`, optimizer step `False`.

P536 calibration multipliers: min 0.506002, median 1.000000, max 4.833752, above 1: 125, below 1: 125.
Training-update multipliers: min 0.502094, median 0.991781, max 4.216884, above 1: 123, below 1: 128.

## Evidence that only relchem changed

Other-task gradients left identical: `True`.
Relchem factor applications per update: `[1]`.
AE17, Exc, and operator gradients were reused from `measure`. Historical lambdas were applied once by the existing scalarization. The single F32 cast remains inside `adamw_step`.

## Runtime

Completed optimizer updates: `251`.
Cumulative new GPU time: `3581.207` seconds.
Peak allocated CUDA memory: `14090340864` bytes (13.123 GiB).
Peak reserved CUDA memory: `19040043008` bytes (17.732 GiB).
Updates 2–10 reserved 17.73 GiB because the CUDA cache kept the previous system's blocks. Later updates release that cache between steps. The loss, the gradients, and the AdamW update are unchanged.

## Comparison

| Model | Updates | Clean28 | Relchem ratio | AE17 ratio | Exc ratio | Op ratio | Eligible |
|---|---:|---:|---:|---:|---:|---:|---|
| P536 | 0 | 9.553190636 | 1 | 1 | 1 | 1 | No, strict threshold |
| J251 control | 251 | 9.346760659 | 0.979707732 | 0.081012145 | 0.101104757 | 0.938121207 | Yes |
| Historical best | 80 | 8.619694172 | 1.034869578 | 0.080727952 | 0.059058103 | 0.944069705 | No |
| New t90 | 90 | 8.869935223 | NOT EVALUATED — ACCURACY GATE | NOT EVALUATED — ACCURACY GATE | NOT EVALUATED — ACCURACY GATE | NOT EVALUATED — ACCURACY GATE | NOT EVALUATED — ACCURACY GATE |
| New t175 | 175 | 8.833974360 | NOT EVALUATED — ACCURACY GATE | NOT EVALUATED — ACCURACY GATE | NOT EVALUATED — ACCURACY GATE | NOT EVALUATED — ACCURACY GATE | NOT EVALUATED — ACCURACY GATE |
| New t251 | 251 | 8.974477726 | NOT EVALUATED — ACCURACY GATE | NOT EVALUATED — ACCURACY GATE | NOT EVALUATED — ACCURACY GATE | NOT EVALUATED — ACCURACY GATE | NOT EVALUATED — ACCURACY GATE |

## Clean28 reactions

### t90

Improved versus J251: 20. Worse versus J251: 8.

| Subset | Count | Weighted sum |
|---|---:|---:|
| ACONF | 1 | 11.792668 |
| Amino20x4 | 2 | 13.895712 |
| BHPERI | 1 | 19.658858 |
| BHROT27 | 2 | 16.146971 |
| BSR36 | 1 | 18.389292 |
| BUT14DIOL | 1 | 2.042297 |
| CDIE20 | 1 | 8.589466 |
| DC13 | 1 | 2.596946 |
| DIPCS10 | 1 | 0.071335 |
| FH51 | 2 | 8.110143 |
| G21EA | 1 | 0.389302 |
| HAL59 | 2 | 4.386753 |
| HEAVY28 | 1 | 26.052826 |
| MB16-43 | 1 | 0.965680 |
| MCONF | 1 | 0.923495 |
| PNICO23 | 1 | 2.813043 |
| PX13 | 1 | 15.166456 |
| S66 | 2 | 9.301518 |
| SIE4x4 | 1 | 55.645567 |
| W4-11 | 3 | 6.256123 |
| WCPT18 | 1 | 25.163737 |

Largest five weighted contributions:

- `SIE4x4-15` (SIE4x4): signed 32.926371, weighted 55.645567, delta weighted vs P536 -1.191555, delta weighted vs J251 -0.603350.
- `HEAVY28-16` (HEAVY28): signed 0.568963, weighted 26.052826, delta weighted vs P536 -3.614829, delta weighted vs J251 -0.118281.
- `WCPT18-15` (WCPT18): signed -15.533171, weighted 25.163737, delta weighted vs P536 -1.628078, delta weighted vs J251 -0.829555.
- `BHPERI-11` (BHPERI): signed -7.227521, weighted 19.658858, delta weighted vs P536 -2.876438, delta weighted vs J251 -0.548426.
- `BSR36-31` (BSR36): signed -5.239114, weighted 18.389292, delta weighted vs P536 2.886505, delta weighted vs J251 0.304554.

### t175

Improved versus J251: 19. Worse versus J251: 9.

| Subset | Count | Weighted sum |
|---|---:|---:|
| ACONF | 1 | 12.684273 |
| Amino20x4 | 2 | 15.445027 |
| BHPERI | 1 | 18.754426 |
| BHROT27 | 2 | 16.715526 |
| BSR36 | 1 | 19.359578 |
| BUT14DIOL | 1 | 1.305939 |
| CDIE20 | 1 | 8.878407 |
| DC13 | 1 | 2.583618 |
| DIPCS10 | 1 | 0.100706 |
| FH51 | 2 | 8.465358 |
| G21EA | 1 | 0.547607 |
| HAL59 | 2 | 3.405425 |
| HEAVY28 | 1 | 24.904787 |
| MB16-43 | 1 | 0.585541 |
| MCONF | 1 | 1.154980 |
| PNICO23 | 1 | 2.367363 |
| PX13 | 1 | 15.080967 |
| S66 | 2 | 9.069579 |
| SIE4x4 | 1 | 55.438916 |
| W4-11 | 3 | 6.091593 |
| WCPT18 | 1 | 24.411668 |

Largest five weighted contributions:

- `SIE4x4-15` (SIE4x4): signed 32.804092, weighted 55.438916, delta weighted vs P536 -1.398207, delta weighted vs J251 -0.810001.
- `HEAVY28-16` (HEAVY28): signed 0.543891, weighted 24.904787, delta weighted vs P536 -4.762868, delta weighted vs J251 -1.266320.
- `WCPT18-15` (WCPT18): signed -15.068931, weighted 24.411668, delta weighted vs P536 -2.380147, delta weighted vs J251 -1.581624.
- `BSR36-31` (BSR36): signed -5.515549, weighted 19.359578, delta weighted vs P536 3.856792, delta weighted vs J251 1.274840.
- `BHPERI-11` (BHPERI): signed -6.895010, weighted 18.754426, delta weighted vs P536 -3.780870, delta weighted vs J251 -1.452858.

### t251

Improved versus J251: 19. Worse versus J251: 9.

| Subset | Count | Weighted sum |
|---|---:|---:|
| ACONF | 1 | 12.828581 |
| Amino20x4 | 2 | 17.425789 |
| BHPERI | 1 | 18.658964 |
| BHROT27 | 2 | 16.689696 |
| BSR36 | 1 | 19.388234 |
| BUT14DIOL | 1 | 1.833831 |
| CDIE20 | 1 | 8.943983 |
| DC13 | 1 | 2.607187 |
| DIPCS10 | 1 | 0.015639 |
| FH51 | 2 | 9.381065 |
| G21EA | 1 | 0.379553 |
| HAL59 | 2 | 3.266144 |
| HEAVY28 | 1 | 23.716371 |
| MB16-43 | 1 | 0.856924 |
| MCONF | 1 | 1.353381 |
| PNICO23 | 1 | 1.977896 |
| PX13 | 1 | 15.589245 |
| S66 | 2 | 9.236568 |
| SIE4x4 | 1 | 55.701550 |
| W4-11 | 3 | 6.253685 |
| WCPT18 | 1 | 25.181092 |

Largest five weighted contributions:

- `SIE4x4-15` (SIE4x4): signed 32.959497, weighted 55.701550, delta weighted vs P536 -1.135573, delta weighted vs J251 -0.547367.
- `WCPT18-15` (WCPT18): signed -15.543884, weighted 25.181092, delta weighted vs P536 -1.610723, delta weighted vs J251 -0.812200.
- `HEAVY28-16` (HEAVY28): signed 0.517938, weighted 23.716371, delta weighted vs P536 -5.951284, delta weighted vs J251 -2.454736.
- `BSR36-31` (BSR36): signed -5.523713, weighted 19.388234, delta weighted vs P536 3.885448, delta weighted vs J251 1.303496.
- `BHPERI-11` (BHPERI): signed -6.859913, weighted 18.658964, delta weighted vs P536 -3.876332, delta weighted vs J251 -1.548320.

## Scientific databases

NOT EVALUATED — ACCURACY GATE

## Did the quadratic term help?

Matched t251 Clean28 versus frozen J251: `yes` (8.974477726 versus 9.346760659).
The lowest of the three milestones is t175 at 8.833974360. That is 0.214280188 above the exact historical best, 8.619694172. t90 is 0.250241052 above it, and t251 is 0.354783554 above it. All three fail the accuracy gate.
A lower training surrogate is not evidence of a better functional. The accuracy gate is Clean28 strictly below the exact B80 receipt `8.619694171538775`, followed by all four scientific ratios strictly below 1.

## Provenance

Initial file SHA256 `0ca0f77367214c171e6edf0576bdb043d64479245ad1a0e539fa11fb0cee542d`.
Initial tensor SHA256 `3c2e45d6a86cccf80d7b4dc458e67f9b406935ed60d4e57433bb441a6df88da6`.
Training manifest SHA256 `72b827655ce4421f2fc933c082cda1c2dc3e5d9903ec29e0c5e3d62e82ac37f6`.
Evaluation manifest SHA256 `132bd550161be6bc6761f8df1de8d3755c0b0debab53663208c3a9bbf56cb805`.
J251 calibration SHA256 `4d97df1ec78aa01a2c9380a86a323b2f0b46828dfc5975f5e3a63c75f4440096`.
Production physics files were hashed against the J251 protocol before training.

## Next recommendation

Do not sweep `c`, `s`, or another seed of this residual-squared multiplier. The matched arm did not put an eligible functional below the historical Clean28 best. The useful next study is a read-only attribution of the stubborn ABDE4, pTC13, and PA8 training reactions, with no further optimization.

This recommendation was not executed.
