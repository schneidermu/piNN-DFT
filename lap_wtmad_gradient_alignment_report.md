# Literal Clean28 WTMAD-2 gradient alignment

Diagnostic only. Optimizer updates: 0. No parameter was saved from a perturbed or averaged model.

The differentiated scalar is the historical Clean28 evaluator: the mean, over the 28 `diet30_clean_validation` reactions, of `abs(error_kcal) * diet_weight`. `error_kcal = (stoichiometry · species_energy) * 627.5095 - reference`. Each species energy is the frozen non-XC term plus the model XC integral on the frozen PBE0 density plus the frozen D3(BJ) term. `diet_weight` is the stored GMTKN55 subset weight. Those weights were not recomputed from these 28 reactions. The subgradient of `abs` at a zero residual is 0. This is not canonical full-GMTKN55 WTMAD-2 and not a squared loss.

Publication logical SHA256 `61c221a19b9987717e69cac182ad545241f8807db4126c0949a99992e4c210ef`. Evaluation manifest SHA256 `132bd550161be6bc6761f8df1de8d3755c0b0debab53663208c3a9bbf56cb805` (`one-variant-per-identity-v1`). Chemistry is the mean of 251 singleton gradients. AE17 is the mean of 17 singletons. Exc and the weak-form operator are means of the same 90 mRKS systems. Coefficients are applied once: relchem `0.017015480965588553`, ae17 `5.141254618347414e-05`, exc `1.5094644512009712e-05`, op `0.33597561607048215`.

S5 was not evaluated. Its historical run uses a different architecture and a nine-database RMSE that includes AE17, so the same coordinates, task definitions, and exact evaluator were not established. No S5 weights were loaded.

## Scalar parity

- `b8_t128` historical evaluator `9.096700111658418` equals receipt `9.096700111658418`. The differentiated path returned `9.096700111609751` (absolute difference `4.867e-11`). Checkpoint SHA256 `1f38f7f8ea74334ff8f5124d2a6538619323d1f453296fa9ca6b3cd7958c437b`. Loaded-model SHA256 `c1c6ecee465decf386f6b6b33523c414bca7c1c5a4f93c1df15a9290a4f0760a`. Coordinates 9446.
- `continuation_raw_t251` historical evaluator `9.087061815657618` equals receipt `9.087061815657618`. The differentiated path returned `9.087061815607186` (absolute difference `5.043e-11`). Checkpoint SHA256 `63d72b83dbc2ee49b466d366454ded341086547ddf1594893e15547c6410adf5`. Loaded-model SHA256 `491b6b1e02682c6fdc7ca7b4f3cb9808dd210395de29b253d98806a494b7c87a`. Coordinates 9446.

## Cosine similarity

### b8_t128

| Gradient | wtmad | relchem | ae17 | exc | op | combined |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wtmad | 1.000000 | -0.559828 | -0.008929 | -0.001648 | 0.643603 | -0.026092 |
| relchem | -0.559828 | 1.000000 | -0.770286 | -0.775273 | -0.984409 | -0.748305 |
| ae17 | -0.008929 | -0.770286 | 1.000000 | 0.999966 | 0.699376 | 0.997862 |
| exc | -0.001648 | -0.775273 | 0.999966 | 1.000000 | 0.704798 | 0.997562 |
| op | 0.643603 | -0.984409 | 0.699376 | 0.704798 | 1.000000 | 0.684132 |
| combined | -0.026092 | -0.748305 | 0.997862 | 0.997562 | 0.684132 | 1.000000 |

### continuation_raw_t251

| Gradient | wtmad | relchem | ae17 | exc | op | combined |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wtmad | 1.000000 | -0.593576 | 0.292906 | 0.068152 | 0.665669 | 0.062490 |
| relchem | -0.593576 | 1.000000 | 0.478236 | -0.780690 | -0.986854 | -0.749348 |
| ae17 | 0.292906 | 0.478236 | 1.000000 | -0.918479 | -0.393015 | -0.916402 |
| exc | 0.068152 | -0.780690 | -0.918479 | 1.000000 | 0.719369 | 0.990519 |
| op | 0.665669 | -0.986854 | -0.393015 | 0.719369 | 1.000000 | 0.704830 |
| combined | 0.062490 | -0.749348 | -0.916402 | 0.990519 | 0.704830 | 1.000000 |

## Norms, dots, and first-order changes

### b8_t128

WTMAD norm 119.683465. Combined norm 0.539150. Cosine with the combined training gradient -0.026092. Cancellation ratio 0.488536.

| Task | Raw norm | Weighted norm | Dot | Weighted dot | Raw predicted change | Weighted predicted change | Unit predicted change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| relchem | 15.077668 | 0.256554 | -1010.236841 | -17.189666 | 1010.236841 | 17.189666 | 67.002194 |
| ae17 | 5597.740972 | 0.287794 | -5981.980742 | -0.307549 | 5981.980742 | 0.307549 | 1.068642 |
| exc | 20120.436769 | 0.303711 | -3969.670279 | -0.059921 | 3969.670279 | 0.059921 | 0.197295 |
| op | 0.613355 | 0.206072 | 47.245869 | 15.873460 | -47.245869 | -15.873460 | -77.028591 |
| combined | 0.539150 | 0.539150 | -1.683675 | -1.683675 | 1.683675 | 1.683675 | 3.122836 |

Predicted change is `-dot(g_wtmad, direction)`. A positive value means descent along that direction is predicted to increase Clean28 at this frozen vector. Raw uses the task gradient, weighted uses each frozen coefficient once, and unit uses the task gradient normalized to length 1. Cosine is not an AdamW update prediction. The large raw AE17 and Exc changes come from unnormalized loss scales; the weighted column is their contribution to the training objective.

Production objective scalars at this vector: relchem 1.195939, ae17 3.275465, exc 14.679045, op 0.030933.

### continuation_raw_t251

WTMAD norm 84.228070. Combined norm 0.230733. Cosine with the combined training gradient 0.062490. Cancellation ratio 0.707668.

| Task | Raw norm | Weighted norm | Dot | Weighted dot | Raw predicted change | Weighted predicted change | Unit predicted change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| relchem | 15.030244 | 0.255747 | -751.448632 | -12.786260 | 751.448632 | 12.786260 | 49.995772 |
| ae17 | 475.734198 | 0.024459 | 11736.795752 | 0.603419 | -11736.795752 | -0.603419 | -24.670910 |
| exc | 19937.067304 | 0.300943 | 114444.783465 | 1.727503 | -114444.783465 | -1.727503 | -5.740302 |
| op | 0.619498 | 0.208136 | 34.734024 | 11.669785 | -34.734024 | -11.669785 | -56.068015 |
| combined | 0.230733 | 0.230733 | 1.214447 | 1.214447 | -1.214447 | -1.214447 | -5.263424 |

Predicted change is `-dot(g_wtmad, direction)`. A positive value means descent along that direction is predicted to increase Clean28 at this frozen vector. Raw uses the task gradient, weighted uses each frozen coefficient once, and unit uses the task gradient normalized to length 1. Cosine is not an AdamW update prediction. The large raw AE17 and Exc changes come from unnormalized loss scales; the weighted column is their contribution to the training objective.

Production objective scalars at this vector: relchem 1.170649, ae17 2.077345, exc 9.883612, op 0.030792.

## Parameter blocks

### b8_t128

| Gradient | Exchange | Correlation | Normalization | Remaining |
| --- | ---: | ---: | ---: | ---: |
| wtmad | 0.976551 | 0.002191 | 0.021258 | 0.000000 |
| relchem | 0.966475 | 0.003828 | 0.029697 | 0.000000 |
| ae17 | 0.924572 | 0.000273 | 0.075155 | 0.000000 |
| exc | 0.925008 | 0.000333 | 0.074659 | 0.000000 |
| op | 0.976036 | 0.000267 | 0.023698 | 0.000000 |
| combined | 0.922599 | 0.000300 | 0.077101 | 0.000000 |

| Gradient | Exchange cosine | Correlation cosine | Normalization cosine |
| --- | ---: | ---: | ---: |
| wtmad | 1.000000 | 1.000000 | 1.000000 |
| relchem | -0.588823 | 0.717556 | 0.403371 |
| ae17 | 0.020960 | -0.587608 | -0.710285 |
| exc | 0.028497 | -0.603259 | -0.708319 |
| op | 0.668100 | -0.557355 | -0.366822 |
| combined | 0.003149 | 0.361579 | -0.725571 |

Blocks holding at least 1% of the WTMAD squared norm whose cosine sign differs from the full-vector cosine:
- relchem / normalization: block cosine 0.403371, full cosine -0.559828.
- ae17 / exchange: block cosine 0.020960, full cosine -0.008929. Both values are near zero, so this sign difference is not strong opposition.
- exc / exchange: block cosine 0.028497, full cosine -0.001648. Both values are near zero, so this sign difference is not strong opposition.
- op / normalization: block cosine -0.366822, full cosine 0.643603.

### continuation_raw_t251

| Gradient | Exchange | Correlation | Normalization | Remaining |
| --- | ---: | ---: | ---: | ---: |
| wtmad | 0.974255 | 0.001284 | 0.024460 | 0.000000 |
| relchem | 0.967398 | 0.003071 | 0.029531 | 0.000000 |
| ae17 | 0.925267 | 0.000121 | 0.074612 | 0.000000 |
| exc | 0.925643 | 0.000287 | 0.074070 | 0.000000 |
| op | 0.976702 | 0.000237 | 0.023060 | 0.000000 |
| combined | 0.922726 | 0.001590 | 0.075684 | 0.000000 |

| Gradient | Exchange cosine | Correlation cosine | Normalization cosine |
| --- | ---: | ---: | ---: |
| wtmad | 1.000000 | 1.000000 | 1.000000 |
| relchem | -0.618744 | 0.256145 | 0.245733 |
| ae17 | 0.277384 | 0.250896 | 0.689272 |
| exc | 0.097752 | -0.242920 | -0.576289 |
| op | 0.687765 | -0.156619 | -0.216604 |
| combined | 0.093213 | 0.212721 | -0.608758 |

Blocks holding at least 1% of the WTMAD squared norm whose cosine sign differs from the full-vector cosine:
- relchem / normalization: block cosine 0.245733, full cosine -0.593576.
- exc / normalization: block cosine -0.576289, full cosine 0.068152.
- op / normalization: block cosine -0.216604, full cosine 0.665669.

## Reactions and SIE4x4-15 removed

### b8_t128

| Reaction | Signed residual | Scalar contribution | Gradient norm | Cosine with full WTMAD |
| --- | ---: | ---: | ---: | ---: |
| SIE4x4-15 | 33.138246 | 2.000130 | 6.386662 | 0.823263 |
| WCPT18-15 | -15.706160 | 0.908714 | 15.071808 | 0.911411 |
| HEAVY28-16 | 0.535281 | 0.875375 | 13.799854 | 0.781440 |
| BSR36-31 | -5.468164 | 0.685473 | 15.120221 | -0.747477 |
| BHPERI-11 | -7.014505 | 0.681409 | 12.819647 | 0.689627 |

Removing SIE4x4-15 without changing the 1/28 denominator leaves a remainder whose cosine with the full gradient is 0.999498.
Its cosines with the physical gradients are relchem -0.536741, ae17 -0.039173, exc -0.031884, op 0.621755, and -0.056633 with the combined gradient.
Scalar contribution and gradient contribution are different quantities. The full reaction table is in the reactions CSV.

### continuation_raw_t251

| Reaction | Signed residual | Scalar contribution | Gradient norm | Cosine with full WTMAD |
| --- | ---: | ---: | ---: | ---: |
| SIE4x4-15 | 33.063088 | 1.995594 | 6.439194 | 0.825887 |
| WCPT18-15 | -15.508676 | 0.897288 | 15.062723 | 0.888471 |
| HEAVY28-16 | 0.495120 | 0.809698 | 13.835896 | 0.719410 |
| BSR36-31 | -5.913829 | 0.741341 | 15.188076 | -0.636698 |
| BHPERI-11 | -6.576760 | 0.638885 | 12.949884 | 0.692121 |

Removing SIE4x4-15 without changing the 1/28 denominator leaves a remainder whose cosine with the full gradient is 0.998943.
Its cosines with the physical gradients are relchem -0.561835, ae17 0.328718, exc 0.028225, op 0.634602, and 0.021367 with the combined gradient.
Scalar contribution and gradient contribution are different quantities. The full reaction table is in the reactions CSV.

## Local alignment

At `b8_t128`:
- relchem: its coefficient-weighted contribution locally opposes Clean28 reduction (weighted `17.189666`, raw `1010.236841`, cosine `-0.559828`).
- ae17: its coefficient-weighted contribution locally opposes Clean28 reduction (weighted `0.307549`, raw `5981.980742`, cosine `-0.008929`).
- exc: its coefficient-weighted contribution locally opposes Clean28 reduction (weighted `0.059921`, raw `3969.670279`, cosine `-0.001648`).
- op: its coefficient-weighted contribution locally agrees with Clean28 reduction (weighted `-15.873460`, raw `-47.245869`, cosine `0.643603`).
- combined training gradient: weighted predicted change `1.683675`, cosine `-0.026092`.
- SIE4x4-15 scalar contribution `2.000130` versus gradient norm `6.386662` (full WTMAD norm `119.683465`).

At `continuation_raw_t251`:
- relchem: its coefficient-weighted contribution locally opposes Clean28 reduction (weighted `12.786260`, raw `751.448632`, cosine `-0.593576`).
- ae17: its coefficient-weighted contribution locally agrees with Clean28 reduction (weighted `-0.603419`, raw `-11736.795752`, cosine `0.292906`).
- exc: its coefficient-weighted contribution locally agrees with Clean28 reduction (weighted `-1.727503`, raw `-114444.783465`, cosine `0.068152`).
- op: its coefficient-weighted contribution locally agrees with Clean28 reduction (weighted `-11.669785`, raw `-34.734024`, cosine `0.665669`).
- combined training gradient: weighted predicted change `-1.214447`, cosine `0.062490`.
- SIE4x4-15 scalar contribution `1.995594` versus gradient norm `6.439194` (full WTMAD norm `84.228070`).

Across both frozen vectors the weak-form operator gradient is aligned with the Clean28 gradient, and coefficient-weighted descent on it is predicted to decrease Clean28. The relchem gradient is opposed at both vectors. AE17 and Exc change sides: nearly orthogonal with a very small weighted increase at B8 t128, and weakly aligned with a small weighted decrease at the continuation checkpoint. The combined training gradient stays nearly orthogonal to Clean28. Its first-order predicted change is a small residual, positive at B8 t128 and negative at the continuation checkpoint, after the larger operator and relchem contributions cancel. That is consistent with the training objective moving while Clean28 stays near 9, and it does not identify operator, Exc, or AE17 as a constraint to remove.

A negative cosine or a positive predicted change is a local observation at these two frozen vectors. It does not show that a physical task is globally harmful, and it is not a reason to weaken or remove that constraint.

## Numerical checks

`b8_t128` symmetric finite differences along the WTMAD gradient:
- target change 1.0e-02: relative gap 3.084e-05, epsilon 8.355e-05, finite difference 119.679774, autodiff norm 119.683465
- target change 1.0e-03: relative gap 6.364e-04, epsilon 8.355e-06, finite difference 119.607303, autodiff norm 119.683465
- target change 1.0e-04: relative gap 1.801e-02, epsilon 8.355e-07, finite difference 117.528335, autodiff norm 119.683465
- Smallest relative gap is 3.084e-05 at target change 1e-02.
- Exact zero residuals: none.
- Disconnected task parameters: {'relchem': [], 'ae17': [], 'exc': [], 'op': []}.

`continuation_raw_t251` symmetric finite differences along the WTMAD gradient:
- target change 1.0e-02: relative gap 5.941e-02, epsilon 1.187e-04, finite difference 89.231951, autodiff norm 84.228070
- target change 1.0e-03: relative gap 6.932e-04, epsilon 1.187e-05, finite difference 84.169685, autodiff norm 84.228070
- target change 1.0e-04: relative gap 6.737e-03, epsilon 1.187e-06, finite difference 83.660584, autodiff norm 84.228070
- Smallest relative gap is 6.932e-04 at target change 1e-03.
- Exact zero residuals: none.
- Disconnected task parameters: {'relchem': [], 'ae17': [], 'exc': [], 'op': []}.

WTMAD derivatives use the float64 evaluator copy. Task derivatives use the production F64 chemistry shadow and F64 mRKS gradients on the same 9,446 float32 coordinates. Finite differences perturb that float32 storage and then call the historical evaluator. At B8 t128 the largest step is the closest match and the smallest step shows float32 rounding. At the continuation checkpoint the middle step is the closest match; the largest step shows curvature and the smallest step again shows rounding.

The two checkpoints are the same loss family. Agreement between them is not independence from the Diet28 sample, the fixed PBE0 densities, or the one-variant chemistry manifest.

## Frozen AdamW moments

- `b8_t128` saved momentum cosine with WTMAD 0.154218. Hypothetical full-objective AdamW displacement cosine -0.038540. The displacement was not applied.
- `continuation_raw_t251` saved momentum cosine with WTMAD -0.054868. Hypothetical full-objective AdamW displacement cosine 0.070283. The displacement was not applied.

## Git

Provenance is the commit that contains this report. The diagnostic started from `13c1070`.
