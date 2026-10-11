# F011 reference-normalized chemistry training

Classification: **NO-GO — Clean28 gate not passed**.

One arm from B8 t128. The chemistry term is `c * factor_d / A_d * sqrt(r^2 + 1e-20)`, averaged as in the B8 minibatch. Operator, Exc, AE17, lambdas, and the learning rate were not changed.

The restored AdamW moments were accumulated under the original chemistry objective. That is a matched continuation control, not a claim that those moments are optimal for F011.

## Calibration

- `c` = `9.705596345235`
- `||g_old||` = `15.077668`
- `||g_F011||` = `1.553502`
- `||c g_F011||` = `15.077668`
- Relative norm error `0.000e+00`
- cos(old, c g_F011) = `-0.631957`
- cos(JW, c g_F011) = `0.980410`
- cos(JW, new total) = `0.660116`
- First-order predicted JW change under the full new gradient: `-0.870694`
- A_d SHA256 `3233a7b3194e68af447bb63edda0d154325da7d051e0df6fe60a7a788cebc73d`

## Training-set metrics

| Checkpoint | JW | MAE | Original relchem | F011 | AE17 | Exc | Operator | Composite |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| initial | 0.156415 | 8.536568 | 1.195939 | 0.823824 | 3.275465 | 14.679045 | 0.030933 | 0.024801 |
| t64 | 0.148275 | 8.273987 | 1.265879 | 0.740539 | 3.849199 | 8.360124 | 0.029226 | 0.022744 |
| t128 | 0.145491 | 8.155332 | 1.292944 | 0.721445 | 1.234652 | 6.980487 | 0.029371 | 0.022313 |
| t192 | 0.144079 | 8.140767 | 1.293710 | 0.710459 | 2.231732 | 11.120048 | 0.029379 | 0.022242 |
| t251 | 0.143531 | 8.271702 | 1.271155 | 0.698142 | 1.378884 | 7.837101 | 0.029235 | 0.021891 |

JW moved from `0.156415` to `0.143531` (`-0.012883`).
The quarter-rate continuation ended at JW `0.150596` and original relchem `1.170649`.
The first-order prediction is the directional derivative at t128. It is not the predicted result of 251 AdamW updates.

DBH76 and NCCE31, the small-reference databases emphasized by `1/A_d`, improved on JW. pTC13, PA8, and ABDE4, the databases whose legacy-weighted gradients opposed JW, became worse. MGAE109 changed little. The new objective did not improve every database that had carried a negative projection under the old relchem gradient.

## Focus databases

| Database | Initial JW | Final JW | Initial MAE | Final MAE |
| --- | ---: | ---: | ---: | ---: |
| DBH76 | 0.417178 | 0.383656 | 7.857969 | 7.226538 |
| pTC13 | 0.039338 | 0.049804 | 6.639701 | 8.406233 |
| PA8 | 0.010499 | 0.020400 | 1.714795 | 3.331872 |
| ABDE4 | 0.032118 | 0.040071 | 3.000499 | 3.743506 |
| MGAE109 | 0.026434 | 0.025352 | 13.334307 | 12.788812 |
| NCCE31 | 0.202747 | 0.161954 | 0.717074 | 0.572796 |

## Clean28

| Checkpoint | Clean28 | Versus continuation t251 | Versus historical best |
| --- | ---: | ---: | ---: |
| t64 | 8.782020261263 | -0.305042 | +0.162326 |
| t128 | 8.800214350716 | -0.286847 | +0.180520 |
| t192 | 8.857739596910 | -0.229322 | +0.238045 |
| t251 | 8.938697016838 | -0.148365 | +0.319003 |

B8 t128 reference `9.096700111658`. Continuation t251 reference `9.087061815658`.
Clean28 was not used to train, to set `c`, or to choose a checkpoint. It was used in earlier gradient audits, so this score is not an untouched holdout.

## Runtime

- Updates: 251
- Cumulative AdamW step: 379
- Wall seconds inside optimizer updates: `7528.864`
- Peak allocated bytes: `14091253248`
- GPU: NVIDIA GeForce RTX 5070 Ti
- PyTorch: `2.11.0+cu128`
- TF32 disabled, one CPU thread
- Chemistry shadow float64, stored parameters float32

## Reproducibility

- Start commit: `4bc8392087c8f40964976614f5c98bc9efcd2a8e`
- Parent checkpoint SHA256: `1f38f7f8ea74334ff8f5124d2a6538619323d1f453296fa9ca6b3cd7958c437b`
- Initial AdamW moment SHA256: `4659e23cce1cf2c66f701b0b445b897d58838739ca15cffc9300fa0fdc1a40d0`
- Final checkpoint SHA256: `8be1fd6de97455f76575a1b353383dd276e7abda3f67e5a08e7eece48a753486`
- Final AdamW moment SHA256: `1f1ab35b860177a0fc2488a9e47c7d4189c31265bb2045ea56843b9f6bcc55a9`
- t64 SHA256: `31aad5509924d7a61a641a85b4127cb52070dbb5761d64a5c2f817db134abc7a`
- t128 SHA256: `acab3005f56d178aadeac3e7db54707b41cb32710e826f2d559f4cc78e98531c`
- t192 SHA256: `c20bed8b1339d3747fd74e997f68d1d7919a712256e6bb3a523814fab3744548`
- B8 manifest SHA256: `b249964d410f0db4a59867de53bb35c6442ec23aa81eefb0b96b50242d94f6bc`
- Learning rate `2.5e-5`, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `0.01`, foreach false
- Checkpoints are stored outside the repository under `lap_f011_normalized_chemistry_20261011`

## Git

Provenance is the commit that contains this report.
