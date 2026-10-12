# NN-PBE-Lap correlation architecture audit

## Executive scientific conclusion

Outcome **C**. On the frozen 90-system mRKS grids, `pcPBELMLOptimizerV2Lap` does not produce positive local correlation energy at any of the five audited checkpoints. `G_c > 1` is common, but those values stay below the pointwise sign-change threshold `G_c^crit = -ε_c^PW92 / H_PBE`, so the excess does not flip `ε_c`. The correlation block is not inactive: it carries about 76–78% of the squared norm of the gradient of `E_c`. Training-objective gradients remain exchange-dominated because those objectives are exchange-heavy, not because the `G_c` derivative has died. The production shifted-ELU slope is 1 on essentially all of the electron density.

The frozen smooth bound with `k = 4` removes `G_c > 1` by construction and makes `E_c` slightly more negative. It also drops the electron-weighted mean of `|dG_c/dz|` from 1 to 0.015–0.030. That is a large suppression of the `G_c` channel near the PBE anchor. The correlation-parameter gradient of `E_c` falls only about 4%, because `β` and `γ` still carry it. There is no demonstrated sign violation for that suppression to repair.

This audit did not evaluate Clean28 and does not explain the Clean28 trajectory.

**KEEP CURRENT CORRELATION ARCHITECTURE**

## Provenance

- Repository branch `lap_full_vxc`. Parent commit `18987c548080dc6ce91266a4ab8b24107f5764d6`.
- Model class `pcPBELMLOptimizerV2Lap`, `lap_architecture_version` 1, 9,446 trainable parameters. Dropout rate 0.
- Production `G_c` is the inherited V2 map `shifted_elu(z - 1)`. For `z >= 1`, `G_c = z` and the slope is 1. For `z < 1`, `G_c = exp(z - 1) > 0` and the slope equals `G_c`. The current activation cannot produce `G_c < 0`.
- Historical S5 `pcPBELMLOptimizerV2GcSveluMirror` uses `2 - shifted_elu(|z - 1|)`, which equals `1 - |z - 1|` and therefore enforces `G_c <= 1` with a cusp at `z = 1`. That class is not the Lap production path.
- The counterfactual uses the existing `pcPBELMLOptimizerV2GcSoftplusMirror` formula at sharpness `k = 4`:

  `G_c = 1 - (2/k) log cosh(k(z-1)/2)`.

  It satisfies `G_c(1) = 1`, `G_c <= 1`, and zero derivative at `z = 1`. `k` was not tuned. The substitution was applied only on temporary copies, using the same weights and the same preactivation path.

- Correlation energy in `dft_functionals/PBE.py`: `ε_c = f_pw + G_c * fH`. Learned `β` and `γ` enter `H` only. PW92 reads the fixed pass-through constants. `LapEnergy` multiplies the per-particle `F_PBE` by `ρ_total`, and the audit integrates with the stored quadrature weights in Hartree.
- Checkpoints, all loaded as the 9,446-parameter Lap model, file SHA256 verified before and after:

| Checkpoint | SHA256 |
|---|---|
| B8 t128 | `1f38f7f8ea74334ff8f5124d2a6538619323d1f453296fa9ca6b3cd7958c437b` |
| F010 t128 | `bd5ecc45fa44793e95d82492e2666d1c40c6fa6068b4f2e266824d3774fbc822` |
| F010 t251 | `1dba8af1a11d5f98fdc5b40d8be6b137d9773af3a6f2bc9eb1482750b3a5feb3` |
| F011 t64 | `31aad5509924d7a61a641a85b4127cb52070dbb5761d64a5c2f817db134abc7a` |
| F011 t251 | `8be1fd6de97455f76575a1b353383dd276e7abda3f67e5a08e7eece48a753486` |

- Density population: the canonical 90-system mRKS corpus, full original grids, 8,271,091 points per checkpoint. Each grid has 14,756 nonpositive quadrature weights. Those weights stay in `E_c` and `E_xc`. Electron-weighted fractions use `max(w, 0) * max(ρ, 0)`.
- Operator panel: the 15 historical systems with `historical_loss_gradient_parity = PASS`: AlBeH, BH, BeH2, C2H2_iso2, CH2O, CH4, CO, ClH, ClHS, H2, H2O, H4Si, HLi, HPSi_iso2, N2.
- Stride-400 samples (20,720 points per checkpoint) were taken by grid index before any use of the values. They are point samples, not exact integrals.
- Precision: checkpoints remain float32. Grid diagnostics widen those stored values to float64. Widening does not recover pre-quantization precision. The operator scalar uses the float32 production operator path. Integrals and statistics accumulate in float64. No optimizer state was loaded or stepped.

## Mathematical constraints

PW92 is nonpositive on the physical densities seen here, and `H_PBE > 0` on the conditioned sample (`H > 1e-8` and `ε_PW92 < 0`, which covered the entire stride-400 sample). At `G_c = 1` the case-B electron fraction is 0 on every full grid, so `ε_PW92 + H <= 0` at the `1e-8` Ha/particle threshold. Under that measured bound, `G_c <= 1` would be sufficient for `ε_c <= 0`. It is not necessary on these densities: learned `G_c` exceeds 1 and `ε_c` stays negative because `G_c < G_c^crit`.

This is a numerical verification on the frozen densities, not a proof over the whole adaptive `(β, γ)` domain. `β` is architecturally limited to `(0.75, 1.25) β_PBE`. `γ` has a floor of `0.1 γ_PBE` and no upper bound. Observed factors on these grids stay inside a narrower window: `β/β_PBE` about 0.891–1.181 and `γ/γ_PBE` about 0.980–1.106. The gamma floor is not active. Production regularizers (`1e-7` in `rs, ζ`, `1e-10` in the reduced-gradient denominators, and the `f2 <= -1` logarithm clamp) were left as implemented. No nonfinite PW92, `H`, or `G_c` values occurred.

The decomposition `ε_c = ε_PW92 + G_c H` matched `PBE_C` with maximum absolute residual 0 in float64 on every system.

## Full-grid `G_c` and energy

Equal-system means are the headline. Pooled electron fractions, which give more weight to systems with more electrons, are in parentheses.

| Checkpoint | Points with `G_c>1` | Electron fraction `G_c>1` | Electron `G_c>1.01` | Electron `G_c>1.1` | `G_c>1.5` or `G_c<0` |
|---|---:|---:|---:|---:|---:|
| B8 t128 | 91.27% | 99.80% (pooled 99.87%) | 31.40% | 9.9e-8 | 0 |
| F010 t128 | 91.59% | 99.98% (99.99%) | 40.80% | 7.9e-8 | 0 |
| F010 t251 | 91.88% | 99.98% (99.99%) | 46.32% | 1.0e-5 | 0 |
| F011 t64 | 91.35% | 99.97% (99.98%) | 35.10% | 7.9e-8 | 0 |
| F011 t251 | 91.25% | 99.86% (99.90%) | 37.45% | 4.0e-7 | 0 |

Electron-weighted histogram quantiles, bin width 0.001, for B8 t128: 0.1% = 0.999, median = 1.003, 95% = 1.021, 99.9% = 1.037. Exact per-system maxima reach 1.129 (B8), 1.124 (F010 t128), 1.123 (F010 t251), 1.124 (F011 t64), and 1.119 (F011 t251). `|G_c - 1| <= 0.001` covers 17–24% of the electrons.

Positive local correlation, threshold `ε_c > 1e-8` Ha/particle:

- Positive points: 0 / 8,271,091 at every checkpoint.
- `E_c^+ = 0` Hartree for every system.
- Sum of `E_c` over 90 systems: B8 −54.231 Ha, F010 t128 −53.529 Ha, F010 t251 −52.981 Ha, F011 t64 −53.949 Ha, F011 t251 −53.698 Ha.
- Stride-400 maximum `ε_c`: −5.21e-11 (B8), −4.94e-11 (F010 t128), −4.77e-11 (F010 t251), −5.05e-11 (F011 t64), −4.89e-11 (F011 t251).
- On the 15-system panel the full-grid maximum is −5.3e-11 or lower.

No system has a positive contribution. The systems where capping `G_c` changes `E_c` the most are ClHS, ClHSi, and HPS. At F010 t251 those shifts are −0.00998, −0.00926, and −0.00912 Ha. The equal-system mean shift is −0.00277 Ha at B8 and −0.00604 Ha at F010 t251. The same shift appears in `E_xc` to within 1e-14 Ha, which is the expected identity because the activation does not enter exchange.

## Mechanism

On the stride-400 sample, every conditioned point has `G_c < G_c^crit`. The median critical value is 1.26–1.28. For `ρ > 1e-3` bohr^-3 the smallest gap `G_c^crit - G_c` is 0.00147–0.00162, and the least negative `ε_c` on those points is about −4e-5 Ha/particle. Case B (`ε_PW92 + H > 1e-8`) has electron fraction 0. Nonfinite counts are 0. The classification is therefore not A, B, C, or D: `G_c > 1` amplifies a positive `H`, but not past the zero of `ε_c`.

## Analytical constraints on the trained weights

Float32 and float64 manufactured-grid tests, using `test_lap.raw_grid` and the Lap anchors, gave maximum absolute residual 0 for every checkpoint:

- Repeat forward, tau-column perturbation, and full spin permutation of the 29 outputs.
- UEG: `β` factor = 1, `G_c = 1`, `G_NN = 0`.
- High-density: `γ` factor = 1 and `G_c = 1`.
- Rapidly varying: `G_c = 1`.
- Pass-through columns 2:22 equal `PBE_CONSTANTS`, and the corresponding factors equal 1.
- `β`, `γ`, and `G_c` finite.
- Shifted-ELU slopes at `z = 0.5, 1, 1.5` equal `exp(-0.5), 1, 1`.

These anchor identities are algebraic properties of the Lagrange correction, confirmed numerically on the trained weights. The architecture does not by itself guarantee `ε_c <= 0` for arbitrary densities once `G_c` may exceed 1. That property holds on the audited grids for the reason above.

## Sensitivity

Trainable split: exchange 4,259, correlation 4,547, normalization 640, remainder 0.

Squared-norm fraction in the correlation block, from the saved full-objective gradients (JW, original relchem, 90-system mean Exc and operator):

| Checkpoint | JW | relchem | Exc | operator |
|---|---:|---:|---:|---:|
| B8 t128 | 0.0153 | 0.0038 | 0.0003 | 0.0003 |
| F010 t128 | 0.0220 | 0.0013 | 0.0004 | 0.0009 |
| F010 t251 | 0.0178 | 0.0012 | 0.0002 | 0.0008 |
| F011 t64 | 0.0173 | 0.0010 | 0.0003 | 0.0008 |
| F011 t251 | 0.0178 | 0.0007 | 0.0003 | 0.0006 |

The exchange block holds 0.92–0.99 of those squared norms. Correlation norms are not zero. For B8 JW the correlation-block norm is 0.187 against a full norm of 1.512.

The gradient of integrated `E_c` on the 15-system panel has the opposite split. Correlation squared-norm fractions are 0.764 (B8), 0.773 (F010 t128), 0.782 (F010 t251), 0.768 (F011 t64), and 0.770 (F011 t251). The correlation network can change the correlation energy. The small JW share is the geometry of an exchange-dominated objective.

Electron-weighted `dG_c/dz` on the 15-system panel is 1.000 for the production activation. The network is in the linear region of shifted ELU, not on a saturated tail. About 74–83% of those electrons have `z > 1.001`, and 17–26% lie within 0.001 of the anchor.

Operator losses on the same 15 systems stay near 0.033–0.035. Correlation-block norms of the saved 90-system operator gradients are about 0.010–0.011.

A central difference of `c_output_layer.weight` along its own `E_c` gradient, step `1e-5`, on the first 4,096 points of AlBeH, matched autodiff with relative error `1.17e-9`. The weights were restored. Central differences of `dG_c/dz` at step `1e-6` had relative errors `2.6e-11` at `z = 0.5`, `2.5e-7` at `z = 1`, and `8.2e-11` at `z = 1.5` and `z = 1.02`.

## Frozen-weight smooth counterfactual

Not a trained comparison. Same weights, same densities, same preactivation, different `G_c` map.

- `G_c^smooth(1) = 1` and the value stays at most 1. `E_c^+` remains 0.
- Equal-system mean `ΔE_c`: −0.00277 Ha (B8), −0.00440 (F010 t128), −0.00604 (F010 t251), −0.00329 (F011 t64), −0.00398 (F011 t251).
- Electron-weighted mean `|dG_c^smooth/dz|` on the 15-system panel: 0.0150, 0.0221, 0.0297, 0.0165, 0.0185. The production slope on the same electrons is 1, so the `G_c` channel is suppressed by a factor of about 34–67.
- The correlation-block norm of `dE_c/dθ` falls by about 4% (B8: 0.457 to 0.440). Cosine between the original and smooth 15-system mean `E_c` gradients is 0.965–0.978.
- Mean operator-loss change on the 15-system panel: −8.4e-5 (B8), −1.6e-4 (F010 t128), −2.3e-4 (F010 t251), −1.1e-4 (F011 t64), −1.4e-4 (F011 t251). The operator was differentiated from the counterfactual energy, not assembled by inserting new `G_c` into old derivatives.

A smooth upper bound would therefore improve a sign constraint that these checkpoints do not violate, while flattening `G_c` updates in the density region the model actually occupies.

## Limitations

- The sign bound is grid evidence, not a theorem for every admissible density and every allowed `(β, γ)`.
- Histogram quantiles have resolution 0.001. Exact minima and maxima are the per-system values in the CSV.
- Stride-400 plots are point samples. Energy totals and `G_c` fractions use the full grids.
- Saved JW, relchem, Exc, and operator gradients are the previously stored 90-system or J251 vectors. The new operator numbers are the 15-system scalar losses.
- No new SCF densities, no Clean28, no chemistry benchmark under the counterfactual.
- F010 t128 has a much smaller saved Exc-gradient norm (214) than F010 t251 (1.22e4). That matches `λ_Exc * ||g||` in the F010 geometry receipt. It is reported as stored.

## Recommendation

Do not change the correlation activation for the next training experiment on the basis of this audit. Keep the production shifted ELU. A smooth `G_c <= 1` bound is a defensible constraint only if a later, separate study shows sign violations on densities these 90 systems do not represent, and that study must also account for the measured collapse of `dG_c/dz` near `z = 1`.

## Reproducibility

From `lap_full_vxc`, with `KMP_DUPLICATE_LIB_OK=TRUE`, TF32 disabled, and one CPU thread:

```text
python -c "import lap_correlation_architecture_audit as audit; audit.execute()"
python -c "import lap_correlation_architecture_followup as follow; follow.execute()"
python -c "import lap_correlation_architecture_render as render; render.execute()"
```

Scratch cube: `lap_correlation_architecture_audit_20261012` (not committed). Optimizer updates: 0.

Artifact SHA256:

| File | SHA256 |
|---|---|
| `lap_correlation_gc_distribution.csv` | `e5027880bf0a0246ae6f718f5ec402f5a89810441442809a933d263f68b18e4a` |
| `lap_correlation_positive_energy.csv` | `cea30fc47536e6ae36c4130daad2c90ee591d6dc9f3f151fc430f3fa7f69ca61` |
| `lap_correlation_energy_decomposition.csv` | `762cd42e0b613180232365675e0dc03065c33a18cb9dce962e1c4889815bc914` |
| `lap_correlation_physical_constraints.csv` | `4dbed9a8c7e437332afe80631ae4a2ee880b5cd41830dd7df81390160b4767be` |
| `lap_correlation_parameter_sensitivity.csv` | `0893410160b151a991d200cf5b2252c89e9b22fe39cf11ca6a70d6c492ed0736` |
| `lap_correlation_activation_saturation.csv` | `1de6c2596340dc58e8998be53511de97a1f3cb37d6735d88063f7aa1bb7ee650` |
| `lap_correlation_operator_sensitivity.csv` | `c96a2a2cae7c570d299807523cc5bd2c4ddab7c34d2ca0e2078bb97c536111ad` |
| `lap_correlation_smooth_gc_counterfactual.csv` | `fcfa3e0295d4b00db73ee0e08998733dff76caaac401230668545aec90bd5553` |
| `lap_correlation_gc_distribution.png` | `91daf013b4f1ad72fa453ed2af18e79f51b5088772f0adaedc392bd3161e2f8f` |
| `lap_correlation_local_energy.png` | `237e32ebe9e2e1735ad107832d7040982db81928aeb983f175f91edd02472443` |
| `lap_correlation_positive_negative.png` | `030c354ff506dce80871b89b41fd507a6704785d65549633764feeb1f1f316ee` |
| `lap_correlation_gc_versus_critical.png` | `2511928e1a4f5464e334291a5d6a01a12ecdf0a93ec5ee8122520a24b2eec3e3` |
| `lap_correlation_activation_comparison.png` | `6ef73eeff6e06561090ea182b11d06819cac409c20edf1cf2d3db443e98d7965` |
| `lap_correlation_exchange_response.png` | `e57a20cf7f82cc16e24bb91acf291c738d82d8333dab47b30533752273757a8e` |
| `lap_correlation_activation_derivatives.png` | `151def2970883b0c40fcaaf09b0ad39085c02c43ee5880c35e692693769d0409` |
