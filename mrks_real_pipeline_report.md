# Real mRKS Lap/full-Vxc pilot report

## Scope and code identity

This report records the local prototype run on branch `lap_full_vxc`. The source implementation and the scientific runs used commit `f5be5834613c5ffe329a895ca9806b5405d4ed5b` (parent `b9262ce1a23d73c18eca869e2cf90f4f0b41bc7d`). The report is committed separately after that implementation. The work stayed in this checkout and did not modify the `vxc_training` branch or checkout.

The prototype reached a real forward/backward/optimizer/save/SCF path using legacy mRKS targets, AO density evaluation from `dm_ks`, Cartesian stencils, the tau-free Lap energy functional, and Minnesota PBE pre-optimization. It is a one-system, 20-step pipeline test. It does not validate production finite-difference settings or model quality.

## Inputs and immutable identities

| Input | Path | SHA256 |
|---|---|---|
| Source archive | `C:\Users\schne\Downloads\MRKS.zip` | `15a49f4517c35f2159f1e66d02a817945dbf1e1e66deb46f6e612d553890ad16` |
| Legacy mRKS target pickle | `C:\Users\schne\Downloads\data_vxc_train.pickle` | `0ccc0cfb09814cadcc9e0537953cc324fe6ce5551654fe3254f310a25cc6ee49` |
| NPZ source directory | `C:\Users\schne\Downloads\mrks_90_ccsd_pt\mrks_90_ccsd_pt` | Per-file digests are in the audit JSON |
| Extrapolation CSV | `C:\Users\schne\Downloads\extrapolation_e.csv` | `4040c5ddd21cb73edcf6f54356114ff2cf216fe47346e227c5ca83180fde7e4e` |
| Minnesota grouped pickle | `C:\Dev\ML-DFT\piNN-DFT\train_models\data_train_grouped.pickle` | `a888609de0807356dd6eadee288fcaf0caa0532426df2e1c2f8f5b1c25fef444` |

The machine-readable 90-system source and center audit is [`train_models/mrks_source_audit_v2.json`](train_models/mrks_source_audit_v2.json), SHA256 `300d09095cca9c8e8809f9e45ba78d6a694c30542cf35a314cf6d74db7a7e131`. It indexes normalized system names and records per-system paths, hashes, basis and electron metadata, point counts, target statistics, and center reconstruction errors. The ZIP was inspected without changing the source files.

## Source audit and center provenance

The audit found exactly 90 NPZ systems, 90 legacy systems, and 90 CSV system rows, with 90/90 normalized-name matches. The NPZ files contain 9,448,296 source-grid points; the legacy target contains 8,271,091 selected training points. The legacy `Grid` layout was confirmed from repository code as:

`x, y, z, weight, rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb, tau_a, tau_b, lapl_a, lapl_b`.

All 8,271,091 legacy central points were audited in float64. The density matrix was `dm_ks`, split equally between the two RKS spin channels. `dm1_ao_wf` was not used. The old coordinates are stored rounded to float32; the audit mapped each stored coordinate tuple back to its unique exactly matching float32 representation in the source coordinates and evaluated at that source row's float64 coordinates. This is an exact identity lookup, not spatial interpolation; ambiguous or missing coordinate matches fail closed. This preserves the original quadrature-point identity and avoids inventing coordinate precision by widening rounded values.

The largest normalized error score over systems was 0.01187 for rho, 0.01189 for sigma, and 0.01189 for the Laplacian, where 1.0 is the declared float32-level tolerance. Across all center values, the maximum absolute errors were:

| Quantity | Maximum absolute error | Maximum relative error | Largest per-system RMS |
|---|---:|---:|---:|
| `rho_a`, `rho_b` | `6.1033e-5` | `5.9567e-8` | `4.3963e-6` |
| sigma channels | `127.8705` | `5.9575e-8` | `5.2505` |
| `lapl_a`, `lapl_b` | `15.9903` | `5.9585e-8` | `0.61485` |

The large absolute sigma and Laplacian maxima occur at correspondingly large field magnitudes; their relative errors and normalized tolerance scores are within float32 storage precision. `Tr(dm_ks S)` agrees with the NPZ electron-count metadata to within `1e-6` for all systems. The largest difference between legacy quadrature-integrated electron count and the `dm_ks` count is `2.4224e-4` electrons.

## Vxc and energy-target provenance

The legacy target is one common RKS channel with shape `(N,)`; the new schema stores it as two identical channels `(N, 2)`. Coordinate identity matching (no nearest-neighbor lookup) matched all 8,271,091 points to the NPZ `vxc_grid`, with zero unmatched points. The maximum absolute Vxc difference was `4.76834e-7 Ha`; the worst per-system RMS was `5.14726e-8 Ha`; the median of per-system median absolute differences was `7.69486e-9 Ha`. No potential gauge projection or constant removal was applied.

For every generated record, `E_xc` is copied from the legacy training pickle and compared after dtype conversion. H2, BeH2, and CO mini-corpus targets were all exactly equal to their legacy values after the record conversion; regression coverage also checks this preservation. **E_xc source = preserved legacy mRKS training target. E_xc original formula = unresolved. NPZ `exc_wf` was not substituted.** The NPZ `exc_wf` differs from the old target by as much as `0.88914 Ha` (median absolute difference `0.43928 Ha`). The CSV holds total/CBS/correlation energies and was used for identity/audit only; it was not used to derive a replacement E_xc.

## Real stencil mini-corpus

The explicit pilot step was `h = 0.005 Bohr`. For every legacy selected center, the builder evaluated `dm_ks` at the center and the six Cartesian displacements, storing rho, xyz gradients, and Laplacians for alpha and beta. Legacy coordinates, weights, density cutoff, selected-point population, Vxc, and E_xc remained canonical. All center points in each pilot system received a full center-consistency check before writing.

| System | Legacy centers | Stencil evaluations (`7N`) | Generation time | H5 size | Center max abs error: rho / sigma / laplacian |
|---|---:|---:|---:|---:|---:|
| H2 | 42,508 | 297,556 | 0.327 s | 34,018,696 B | `7.44e-9 / 7.42e-9 / 9.41e-7` |
| BeH2 | 77,240 | 540,680 | 2.961 s | 61,804,296 B | `8.03e-7 / 8.84e-4 / 3.84e-3` |
| CO | 72,110 | 504,770 | 3.426 s | 57,700,296 B | `7.63e-6 / 0.25 / 0.125` |

The three files total 153,523,288 B. Their hashes and matching NPZ hashes are recorded in the audit JSON. Observed AO-evaluation time was about 5.0 seconds per million stencil points in aggregate for these systems (individual measurements varied about 1.1–6.8 seconds per million). The all-90 workload is approximately 57.90 million stencil evaluations and roughly 6.6 GB of H5 data at the measured pilot record size. A rough AO-evaluation-only estimate is 5–7 minutes; source indexing, hashing, full center checks, serialization, and storage contention add time. This is an estimate, not a production throughput benchmark.

The external pilot files are under `C:\Dev\readWFN_share_ms\lap_full_vxc_timed_20261001`; no generated H5 or raw corpus was committed.

## Finite-difference h study

Pure canonical PBE was evaluated through the full Euler machinery on 128 deterministic real quadrature centers for each system H2, BeH2, CO, N2, and ClHS. Candidate steps were 0.04, 0.02, 0.01, and 0.005 Bohr. Values below are RMS differences between results at `h` and the next smaller step; `max |ΔVxc|` is the maximum pointwise difference. Float32 and float64 were compared; for H2 at 0.01→0.005 their Vxc RMS differences agreed to about `2e-8`. For this pure PBE functional, `C` is independent of h (so its difference is zero) and `lap B` is zero; the measured h dependence is in `-div A`.

| System | h → h/2 (Bohr) | ΔC RMS | Δ(-div A) RMS | Δ(+lap B) RMS | ΔVxc RMS | max |ΔVxc| |
|---|---|---:|---:|---:|---:|---:|
| H2 | .04 → .02 | 0 | .018048 | 0 | .018048 | .098445 |
| H2 | .02 → .01 | 0 | .007024 | 0 | .007024 | .039184 |
| H2 | .01 → .005 | 0 | .001976 | 0 | .001976 | .011101 |
| BeH2 | .04 → .02 | 0 | .127161 | 0 | .127161 | 1.005595 |
| BeH2 | .02 → .01 | 0 | .274735 | 0 | .274735 | 2.311129 |
| BeH2 | .01 → .005 | 0 | .308296 | 0 | .308296 | 2.766160 |
| CO | .04 → .02 | 0 | .202331 | 0 | .202331 | 1.126412 |
| CO | .02 → .01 | 0 | .374710 | 0 | .374710 | 2.294020 |
| CO | .01 → .005 | 0 | .788170 | 0 | .788170 | 5.166215 |
| N2 | .04 → .02 | 0 | .204936 | 0 | .204936 | 1.111128 |
| N2 | .02 → .01 | 0 | .374680 | 0 | .374680 | 2.133718 |
| N2 | .01 → .005 | 0 | .854247 | 0 | .854247 | 5.257064 |
| ClHS | .04 → .02 | 0 | .211239 | 0 | .211239 | 1.233251 |
| ClHS | .02 → .01 | 0 | .382513 | 0 | .382513 | 2.462702 |
| ClHS | .01 → .005 | 0 | .624062 | 0 | .624062 | 4.881681 |

For a local H2-only smoke, `0.005 Bohr` was explicitly selected because it was the finest tested H2 interval and reduced the H2 pairwise difference. This is a **pilot h only**. The heavy-system differences increase at the finer pairs, including near-nuclear points, so the study does not establish a general or production h.

## Minnesota audit and PBE pre-optimization

The supplied grouped Minnesota pickle was streamed and verified against the repository's Diet-clean protocol: 268 base reaction groups, 2,144 stored variants (8 per group), 268 unique reaction identities, protocol `diet-clean-mn-all-mrks-external-scf-v1`, and the same 16 excluded reactions defined in repository code. No Minnesota data was reconstructed from external databases. The source file resides in the separate `vxc_training` checkout and was read-only during this work.

A deterministic 268-reaction canonical predopt view was derived from that grouped file without a separate `data_predopt.pickle`: select `default` when present, otherwise use the existing `canonical_variant(...)` stable rule. All 268 groups selected `level2`. The view contains 21,073,642 density points. Its source hash, output hash, and counts are in the external manifest `C:\Dev\readWFN_share_ms\lap_full_vxc_pilotdata\mn_canonical_268_manifest.json`.

A fresh tau-free NN-PBE-Lap model was pre-optimized on all canonical points for the requested 2 epochs at `lr=1e-2`. Whole-grid point-weighted adaptive-parameter metrics changed from MSE `0.030079666`, MAE `0.11999791` to MSE `4.74341e-7`, MAE `0.000294265`. The epoch-level reaction-averaged metrics were epoch 1 MSE `9.59480e-4`, MAE `0.00490175`; epoch 2 MSE `6.13288e-7`, MAE `0.000346079`. The Lap model's adaptive outputs retained beta, gamma, kappa/mu spin channels, `G_NN` spin channels, and `G_c`, with PBE targets and zero tau dependence.

## Joint real loss, backward, and GPU pilot

The pilot used H2 mRKS plus Minnesota group 0 / MGAE109 ReactionID 0 with its `level2` reaction variant, explicit `h=0.005 Bohr`, float32, point chunk 128, Adam learning rate `1e-5`, and 20 steps. Pilot-only objective weights were reaction `1e-4`, E_xc `1`, full Vxc `1`. These weights are not proposed for production. The reaction input was from the canonical grouped artifact; the production code retains support for all 268 base groups and their epoch-wise variant sampling.

The machine was an NVIDIA GeForce RTX 5070 Ti with 17,094,475,776 bytes total CUDA VRAM (16,303 MiB reported by the driver) and 15,767,437,312 bytes free at pilot start. The run stayed at chunk 128 without OOM. Peak allocated memory was 1,111,750,144 bytes (~1.04 GiB); peak reserved was 1,287,651,328 bytes (~1.20 GiB). There were no nonfinite objective values or gradients. Mean step time was 36.50 s (median 36.55 s; range 35.21–37.76 s).

Objective values below compare the same real reaction group and H2 mRKS system after PBE pre-optimization and after 20 joint optimizer steps. The E_xc objective is reported in the trainer's kcal/mol-scaled units; Vxc is the density-weighted pointwise MSE in Ha².

| Objective | After predopt | After 20 steps | Change |
|---|---:|---:|---:|
| Minnesota reaction | 0.00894216 | 0.01444094 | increased |
| Legacy E_xc | 5.144181 | 2.335787 | decreased |
| Full Vxc | 0.05546987 | 0.05430851 | decreased |

The reaction term rose slightly; no loss-weight tuning was attempted. Component gradient norms after predopt → after 20 steps were reaction `1.54114 → 1.53437`, E_xc `893.6483 → 890.5760`, and full Vxc `0.388641 → 0.375641`. The final combined gradient norm was `891.0605`. A one-step diagnostic on a loaded copy of the saved checkpoint observed gradient norm `890.9082` and parameter update norm `9.6972e-4`; it did not save or modify that checkpoint. The diagnostic records its source checkpoint SHA256 `c7659255dbc0eeb38fc24ecdbf03e4e4ca8fca5abb094c81c1ef6a440e40b70b`.

Whole-H2-grid diagnostics after predopt → after training:

| Diagnostic | After predopt | After 20 steps |
|---|---:|---:|
| E_xc prediction (target `-0.7023566365 Ha`) | `-0.6941588626 Ha` | `-0.6986343237 Ha` |
| Weighted Vxc RMSE | `0.2355204 Ha` | `0.2330419 Ha` |
| Weighted Vxc MAE | `0.2270349 Ha` | `0.2240993 Ha` |
| Weighted Vxc bias | `0.2266149 Ha` | `0.2236036 Ha` |
| Maximum absolute Vxc error | `3.73150 Ha` | `4.18739 Ha` |
| Weighted RMS `C` | `0.4459522 Ha` | `0.4490389 Ha` |
| Weighted RMS `-div A` | `0.0421790 Ha` | `0.0425108 Ha` |
| Weighted RMS `+lap B` | `0.0554873 Ha` | `0.0565439 Ha` |

The E_xc prediction, weighted Vxc errors, and Vxc MSE improved, while the maximum Vxc error and component RMS values did not. The checkpoint is `C:\Dev\readWFN_share_ms\lap_full_vxc_pilot_gpu_20261001\lap_real_pilot.pt`, SHA256 `c7659255dbc0eeb38fc24ecdbf03e4e4ca8fca5abb094c81c1ef6a440e40b70b`.

A separate float64 CPU reference used 16 actual H2 centers, h=0.005, and chunk 4. E_xc and full-Vxc forward/backward were finite, with gradient norms `3.3853e-5` and `0.0076720`; nonfinite fraction was zero. This tiny subset established a numerical reference path, not a whole-system accuracy result.

## RKS SCF smoke

The saved pilot checkpoint was passed through `LapFunctional` and `RKS_with_Laplacian` for H2 using its NPZ geometry, PySCF 2.14, CPU float64, grid level 1, `max_cycle=40`, and `conv_tol=1e-8`. SCF converged in 5 recorded cycles to `-1.1712295872374765 Ha`. The recorded energy trace was `-1.1697582295`, `-1.1710475006`, `-1.1712295743`, `-1.17122958722`, `-1.171229587237 Ha`. `vtau` was exactly zero, tau independence was verified, and all cycle energies and outputs were finite.

## Hardening and verification

The implementation also aligns `optuna_joint.py`'s `lap` parser/factory registration, restores the predopt default to 2 epochs at `1e-2`, corrects final partial gradient-accumulation normalization, binds source and generated-stencil hashes, checks numeric center rows, and separates build-time raw-source verification from training-time corpus verification. Full-center audit counts are required for production corpus assembly; the immutable corpus keeps source identities/hashes without requiring the external raw paths to remain mounted.

Relevant Windows tests: **83 passed, 2 skipped**, with two existing PyTorch scheduler warnings. WSL SCF tests: **3 passed**, with one PyTorch deprecation warning. `compileall` passed for `train_models` and `test_models`; `git diff --check` passed before the source commit. Ruff passed on the changed/new files and import sorting for `optuna_joint.py`. A full Ruff run on legacy `optuna_joint.py` still reports 120 existing findings (97 UP006, 15 UP045, 3 UP035, 2 F401, and one each BLE001, F841, PLC0414); those unrelated modernization findings were not mass-edited.

## Remaining production blockers and next commands

Only H2, BeH2, and CO have full real stencil H5s. The 90-system stencil corpus has not been generated, no immutable full Minnesota training corpus directory with the `prepare_training_corpus` manifest was available in this checkout, and no production training was run. The supplied grouped Minnesota artifact is verified and its canonical predopt view is available, but the production assembler additionally requires the repository's prepared Diet-clean Minnesota corpus. The h study does not settle a production h, especially for heavier systems. E_xc's original functional formula remains unresolved by design. The short pilot weights are not production weights.

To build/check all 90 real stencil records later, the following is the exact builder command. `0.005` is explicitly the H2 pilot setting for reproducing this prototype; do not treat it as production validated until a broader stable h study is done.

```powershell
wsl.exe -d Ubuntu-22.04 -- /home/schneidermu/.cache/pinn-lap-tests/bin/python /mnt/c/Dev/readWFN_share_ms/lap_full_vxc/train_models/build_mrks_stencils.py --legacy-pickle /mnt/c/Users/schne/Downloads/data_vxc_train.pickle --npz-root /mnt/c/Users/schne/Downloads/mrks_90_ccsd_pt/mrks_90_ccsd_pt --csv /mnt/c/Users/schne/Downloads/extrapolation_e.csv --source-zip /mnt/c/Users/schne/Downloads/MRKS.zip --output-dir /mnt/c/Dev/readWFN_share_ms/lap_full_vxc_stencils_all90_h005 --h-bohr 0.005 --verify-full-centers --chunk-size 512
```

The deterministic canonical Minnesota view can be regenerated with:

```powershell
wsl.exe -d Ubuntu-22.04 -- /home/schneidermu/.cache/pinn-lap-tests/bin/python /mnt/c/Dev/readWFN_share_ms/lap_full_vxc/train_models/canonical_minnesota_view.py /mnt/c/Dev/ML-DFT/piNN-DFT/train_models/data_train_grouped.pickle /mnt/c/Dev/readWFN_share_ms/lap_full_vxc_pilotdata/mn_canonical_268.pickle --manifest /mnt/c/Dev/readWFN_share_ms/lap_full_vxc_pilotdata/mn_canonical_268_manifest.json --pilot-group 0 --pilot-output /mnt/c/Dev/readWFN_share_ms/lap_full_vxc_pilotdata/mn_pilot_group_0.pickle
```

After preparing the standard immutable Diet-clean Minnesota corpus and verifying its manifest, assemble (do not train yet) the combined full corpus with:

```powershell
wsl.exe -d Ubuntu-22.04 -- /home/schneidermu/.cache/pinn-lap-tests/bin/python /mnt/c/Dev/readWFN_share_ms/lap_full_vxc/train_models/lap_data.py --mn-corpus /mnt/c/Dev/readWFN_share_ms/lap_full_vxc/checkpoints_dietclean_noval_v1 --stencil-dir /mnt/c/Dev/readWFN_share_ms/lap_full_vxc_stencils_all90_h005 --output-dir /mnt/c/Dev/readWFN_share_ms/lap_full_vxc_corpus_h005
```

That last command will reject a missing/unverified Minnesota corpus or any stencil set without 90 full-center-verified records. Select and document the production h and objective weights before starting a production training run.

No production training jobs or Slurm jobs were submitted.
