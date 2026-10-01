# Local S5-equivalent Lap/full-Vxc implementation report

## Result and code identity

The tau-free Lap model now runs through the shared S5 training engine for the three historical objectives: Minnesota reaction energy, legacy mRKS `E_xc`, and full Euler–Lagrange `Vxc`. The local run exercised two optimizer updates in each of the five S5 phases, saved and reloaded a provenance-bearing checkpoint, and converged an H2 RKS smoke calculation using that checkpoint.

Branch: `lap_full_vxc`. The implementation is commit `8e72748d5de048220661cd13bc4abd14457ee1b0`, based on `0ff1494dafea0065d4689f832c792a8956bc5478`. This report is committed separately. The historical reference was read directly from `vxc_training` at `fc483a1f50d4bd6cace7b93a3860cf039b8edfea`; that branch and the separate dirty `vxc_training` checkout were not modified.

The implementation refactors the common `optuna_joint.py` S5 engine, adds explicit Lap-S5 schedule and provenance validation, connects full-stencil loss adapters to the shared objective-gradient machinery, and adds a `train_lap_s5.py` entry point for local phase smoke and strictly verified full-corpus mode. A final audit also fixed the runner's model-factory handoff so both modes instantiate `PBE-Lap-LGxGc_6_32` with `model_type="lap"` through the same registered factory.

Changed files: `train_models/optuna_joint.py`, `train_models/lap_data.py`, `train_models/lap_s5_protocol.py`, `train_models/lap_s5_provenance.py`, `train_models/train_lap_s5.py`, `train_models/test_lap_protocol.py`, `train_models/test_training_protocol.py`, `train_models/test_lap_s5_protocol.py`, `train_models/test_lap_s5_provenance.py`, `train_models/test_lap_s5_training.py`, and `train_models/test_lap_s5_runner.py`.

Only two scientific changes are intended relative to historical S5: the architecture is `pcPBELMLOptimizerV2Lap`, and the potential prediction is the full Euler expression `C - div(A) + lap(B)`. `Fchem`, `E_xc`, loss units and normalization, Minnesota weighting, dispersion, optimizer/scheduler, gradient scaling/clipping/merging, augmentation, preoptimization, and logging retain S5 semantics. No tau input or output is used.

## Historical S5 protocol retained

The reference preset is `E3_SCHEDULE_PRESETS["simple4_two_step_40_10"]` in `train_models/replay_trial_19_bridge.py`, with the two-GPU launcher `train_models/trial19_simple4_sota_sweep/s5_two_step_40_10.slurm` and shared arguments in `run_simple4_job.sh`. The preset is checked against fixed phase boundaries and parameters in code and tests. `batch_fchem(...)`, `calculate_reaction_energy(...)`, and `compute_fchem_from_errors(...)` remain the reaction objective and reporting functions. `FCHEM_DB_WEIGHTS`, `FREQ_WEIGHTS`, and `MEAN_WEIGHT` are unchanged. `batch_exc(...)` remains the `E_xc` loss in kcal/mol; the Lap adapter integrates NN energy density and adds the historical mRKS dispersion by system `Name` exactly once. The target remains the legacy `E_xc`, with regression coverage for exact target preservation after record dtype conversion. Full `Vxc` uses the same NN energy and real displaced stencils with the historical density-weighted pointwise MSE. For equal-spin RKS and identical target channels, tests establish equivalence to `sum(rho_total * w * (Vpred - Vref)^2) / sum(rho_total * w)`. No gauge centering or constant projection is applied.

The trainer retains `OMEGA = 0.5` as separate metadata and uses the historical objective-wise gradient preparation, clipping, scaling, merge, accumulation, and optimizer cadence. The full training settings remain 500 epochs, batch size 1, Vxc batch size 1, dropout 0, weight decay `1e-2`, RAdamW through `configure_optimizers`, and the existing warmup/cosine scheduler at the historical initial training LR `0.0003588259475602772`. Checkpoints are saved every 10 epochs with no internal validation. Predopt remains two epochs at learning rate `1e-2`.

| S5 phase | Epochs | Accumulation / reaction merge | Reaction scale / clip | `Vxc` scale / clip | `E_xc` scale / clip / merge | `OMEGA × Vxc scale` |
|---|---:|---|---|---|---|---:|
| Anchor | 1–72 | 3 / clip-then-sum | 0.6 / none | 75 / 5 | 1 / none / sum | 37.5 |
| Representation 40 | 73–176 | 2 / sum | 1 / none | 40 / 2 | 1 / none / sum | 20 |
| Representation 10 | 177–280 | 2 / sum | 1 / none | 10 / 2 | 1 / none / sum | 5 |
| Chemical repair | 281–440 | 2 / clip-then-sum | 0.75 / none | 15 / 2 | 3 / 2 / clip-then-sum | 7.5 |
| Consolidation | 441–500 | 2 / sum | 1 / none | 7 / 2 | 1 / none / sum | 3.5 |

## Minnesota data and PBE predopt

The supplied `data_train_grouped.pickle` was read from `C:\Dev\ML-DFT\piNN-DFT\train_models\data_train_grouped.pickle`; its SHA256 is `a888609de0807356dd6eadee288fcaf0caa0532426df2e1c2f8f5b1c25fef444` (13,462,405,299 bytes). It contains the verified Diet-clean Minnesota protocol with 268 base reaction groups and 2,144 stored augmentation variants. No Diet30/Diet100 data or external database reconstruction was used.

The deterministic predopt view selects `default` when present and otherwise uses the stable canonical variant rule. All 268 selected `level2`; the resulting view has 21,073,642 grid points and SHA256 `70d23f32286e62c05bf2794e20426f6a03cf516bb1e0dc44e9d3a7d73266bce3`. Its manifest SHA256 is `d33b1ddaac5add399381067f385630f01e9ccd6a64e8a10ba15dbb610d89e6cf`. The joint smoke used reproducible Minnesota group 0, with all eight variants available; the one-group pilot view SHA256 is `991443a6d79e5b3e8df7bb97217d31c2d6967a32ae2a2c234446a79900803114`. The raw-gradient diagnostic selected `C_mgae109__level2`.

Fresh NN-PBE-Lap predopt used Adam, two epochs, learning rate `1e-2`, and no Vxc predopt steps. On all 21,073,642 density points, adaptive-parameter MSE/MAE changed from `0.0823227691 / 0.1921368972` before predopt to `4.8787537e-7 / 0.0002829271` after predopt. The per-epoch MSE/MAE values were `0.0012131648 / 0.0050999668` and `7.3313698e-7 / 0.0003231157`.

## Real mRKS provenance and finite-difference evidence

The earlier all-system source audit is recorded in [`mrks_real_pipeline_report.md`](mrks_real_pipeline_report.md) and [`train_models/mrks_source_audit_v2.json`](train_models/mrks_source_audit_v2.json); the audit JSON SHA256 is `300d09095cca9c8e8809f9e45ba78d6a694c30542cf35a314cf6d74db7a7e131`. It matched all 90 NPZ systems, all 90 legacy systems, and all 90 CSV names. It reconstructed all 8,271,091 legacy central points from `dm_ks` in float64, not `dm1_ao_wf`; the maximum normalized center-error scores were 0.01187 for rho, 0.01189 for sigma, and 0.01189 for Laplacian, where 1.0 is the declared float32 storage tolerance. Exact coordinate identity matching (no interpolation) covered every legacy Vxc point; maximum absolute difference was `4.76834e-7 Ha`, worst per-system RMS was `5.14726e-8 Ha`, and median per-system median absolute difference was `7.69486e-9 Ha`.

The old target pickle SHA256 is `0ccc0cfb09814cadcc9e0537953cc324fe6ce5551654fe3254f310a25cc6ee49`; `MRKS.zip` is `15a49f4517c35f2159f1e66d02a817945dbf1e1e66deb46f6e612d553890ad16`; `extrapolation_e.csv` is `4040c5ddd21cb73edcf6f54356114ff2cf216fe47346e227c5ca83180fde7e4e`. The H2 NPZ SHA256 is `4d99d3ec1f94ccbc3c9caa12d22428e03438dcbaeb07a765d3f30459d7e6df95`. The H2 stencil SHA256 is `79ee143eee0de1ccbf00001f050c6c7771e526dde81f0eaf126365888a65bf38`. The production mRKS dispersion map used by `batch_exc` is `train_models/dispersions/dispersions_mrks.pickle`, SHA256 `a2d5b556a5007fa31505c5e38a2be537d4f1755f961c31ef2c49d07605087440`; the Minnesota reaction dispersion map is `train_models/dispersions/dispersions.pickle`, SHA256 `f7bd56d6b8ad133b7729dbb9f47013bf2c9d0d14fcb41296ed7a4df100855927`.

The 90-system audit confirms that `E_xc` is copied from the legacy training target; its original formula remains unresolved. NPZ `exc_wf` was not substituted, and no CSV energy was used to derive a replacement. For H2, legacy `E_xc = -0.7023566365242004 Ha`; `exc_wf = -0.7346086554014271 Ha` is audit-only. Real full-center stencils exist for H2, BeH2, and CO; the S5 training smoke used H2 at explicit `h = 0.005 Bohr`.

Pure PBE full Euler Vxc was compared at 128 deterministic real centers per system for five representative systems, in float64 and float32. The table gives float64 RMS change from `h` to the next smaller step; for this pure PBE test, `C` and `lap(B)` do not change with h, so the observed difference is in `-div(A)`. Float32 and float64 H2 results for `.01 → .005` agreed in Vxc RMS to about `2e-8`.

| System | `h → h/2` (Bohr) | ΔC RMS | Δ(-div A) RMS | Δ(lap B) RMS | ΔVxc RMS | max |ΔVxc| |
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

`0.005 Bohr` was chosen only for this H2 smoke because it was the finest tested H2 interval and reduced H2 pairwise change. BeH2, CO, N2, and ClHS become less stable at finer pairs, particularly near nuclei. No general or production h is validated.

## Objectives, gradient calibration, and phase smoke

After predopt, the deterministic Minnesota reaction 0 had historical `batch_fchem` loss `5.8823957` and raw parameter-gradient norm `668.9852`; reactions 1 and 2 had losses `35.4798965 / 39.1325264` and norms `2189.3215 / 1519.1728`. The one-system raw mRKS objective and gradient diagnostics were:

| System | `batch_exc` loss (kcal/mol) | E gradient norm | Full Vxc loss (Ha²) | Vxc gradient norm |
|---|---:|---:|---:|---:|
| H2 | 5.246210 | 663.4212 | 0.05378489 | 0.3583057 |
| BeH2 | 24.721813 | 3647.1006 | 0.11369839 | 20.59990 |
| CO | 55.326755 | 15013.6676 | 0.28901777 | 38.59053 |

For the fixed reaction-0/H2 calibration pair, pairwise raw gradient cosines were reaction/Vxc `-0.689517`, reaction/E `-0.638637`, and Vxc/E `0.928807`. A scale sweep is diagnostic only; it did not change the S5 weights used by the phase smoke and it does not select production weights.

| Pilot scale candidate in representation-40 phase | Vxc loss scale (effective coefficient after Ω) | E scale | Effective reaction / Vxc / E gradient norms | Combined norm | Cosines (R/V, R/E, V/E) | One-step update norm | Finite |
|---|---:|---:|---|---:|---|---:|---|
| Reference S5 | 40 (20) | 1 | 668.985 / 7.166 / 663.421 | 568.386 | -.68952 / -.63864 / .92881 | 0.00020395 | yes |
| Raw reaction-matched diagnostic | 3734.159 (1867.079) | 1.0084 | 668.985 / 668.985 / 668.985 | 992.560 | -.68952 / -.63864 / .92881 | 0.00035615 | yes |
| Half matched diagnostic | 1867.079 (933.540) | 1 | 668.985 / 334.493 / 663.421 | 732.332 | -.68952 / -.63864 / .92881 | 0.00026278 | yes |

Applying the unmodified reference settings to the fixed diagnostic gradients gave these effective norms after historical S5 scaling and clipping:

| S5 phase | Reaction | Full Vxc | E_xc | Combined |
|---|---:|---:|---:|---:|
| Anchor | 401.391 | 5.000 | 663.421 | 514.329 |
| Representation 40 | 668.985 | 7.166 | 663.421 | 568.386 |
| Representation 10 | 668.985 | 1.792 | 663.421 | 566.876 |
| Chemical repair | 501.739 | 2.000 | 2.000 | 499.091 |
| Consolidation | 668.985 | 1.254 | 663.421 | 566.728 |

The real local phase smoke used the unmodified S5 phase settings, one reproducibly selected Minnesota reaction group, and H2 full-Vxc data. It ran two updates in each phase, ten total. The entries below are the first and second update in each phase; `Fchem` is the historical reported metric from `compute_fchem_from_errors`, `E_xc` is the unchanged `batch_exc` loss, and Vxc is the full Euler loss.

| Phase | Steps | Reported Fchem | E_xc loss (kcal/mol) | Full Vxc loss (Ha²) | Gradient norm | Result |
|---|---:|---:|---:|---:|---:|---|
| Anchor | 1–2 | .136862 → .138380 | 5.246210 → 5.087334 | .05378489 → .05370519 | 667.563 → 667.545 | finite, updated |
| Representation 40 | 3–4 | .623305 → .887765 | 26.681009 → 43.054867 | .04472391 → .04525411 | 659.733 → 654.073 | finite, updated |
| Representation 10 | 5–6 | .742642 → .168213 | 33.687569 → 2.942981 | .04396952 → .05295750 | 637.795 → 616.047 | finite, updated |
| Chemical repair | 7–8 | .166696 → .166696 | 2.943430 → 2.943815 | .05296075 → .05296324 | 3.29148 → 3.29150 | finite, updated |
| Consolidation | 9–10 | .167707 → .165684 | 2.944110 → 2.925224 | .05296107 → .05295830 | 615.037 → 614.852 | finite, updated |

All ten updates had finite losses, gradients, and nonzero parameter updates. From the first to the tenth update, mRKS `E_xc` and Vxc losses decreased; reported reaction Fchem rose slightly. This ten-step phase-mechanics check is not a compressed S5 run, a model-quality result, or weight selection. No optional 50–200-step pilot was run. Checkpoint save/load round-trip was exact. The committed-code run summary and checkpoint are outside the repository at `C:\Dev\readWFN_share_ms\lap_s5_local_run_20261001_factory\lap_s5_pilot_summary.json` and `C:\Dev\readWFN_share_ms\lap_s5_local_run_20261001_factory\lap_s5_phase_smoke.pt`; checkpoint SHA256 is `d9571c9259baad1374cf4369ab7f6f69679691777e1a438294b6b49b21a54152`. No stencil, corpus, or checkpoint data was committed.

## RTX 5070 Ti throughput and SCF

Training used CUDA float32 on the NVIDIA GeForce RTX 5070 Ti, which reported 17,094,475,776 total VRAM bytes and 15,549,333,504 free bytes at the chunk benchmark start. Candidate chunks 128, 256, 512, 1024, and 2048 completed without OOM. Chunk 256 was selected as the fastest candidate meeting the run's gradient-equivalence tolerance: relative Vxc loss difference from chunk 128 was `1.16e-15`, relative L2 parameter-gradient difference was `2.00e-5`, and the measured Vxc pass time fell from 28.19 s at chunk 128 to 14.33 s at chunk 256. Chunks 512 and above were faster but exceeded the configured gradient tolerance. The ten phase steps averaged 19.72 s each (median 19.74 s, maximum 20.13 s); peak training memory was about 1,085 MiB allocated and 1,210 MiB reserved. This is a local RTX result, not a V100 memory or throughput guarantee. A one-GPU trajectory is not numerically equivalent to the historical two-GPU S5 run.

The saved checkpoint was passed to `LapFunctional` and `RKS_with_Laplacian` for H2 through the WSL CPU/PySCF runtime. SCF converged in five cycles at grid level 1 and `conv_tol=1e-8` to `-1.170291110299433 Ha`. `vtau` was exactly zero, tau independence was true, and all cycle energies and outputs were finite. This verifies SCF plumbing, not energetic accuracy.

## Verification, readiness, and next command

The final focused suite passed: **115 passed, 2 skipped** across the Lap model, SCF, distributed regression, full Euler loss, S5 protocol/provenance/training/runner, accumulation, Minnesota, and training-state tests. The two warnings are existing PyTorch scheduler-order warnings in `test_training_state.py`. `compileall` passed for `train_models` and `test_models`; `git diff --check` passed. Ruff passed on all other changed/new files. Ruff reports 105 legacy findings across `optuna_joint.py`, with **zero findings on changed lines**.

The code is **not yet ready for a 2×V100 full-corpus pilot**. The DDP-capable engine and CPU two-rank tests are in place, but there is no immutable, full-center-verified 90-system stencil corpus at a generally stable h, and the real h study shows worsening fine-step behavior for heavier systems. The repository contains only H2, BeH2, and CO pilot stencils at `h=0.005`; that h and the diagnostic scale sweep are not production validated. A 2×V100 chunk-equivalence/memory check also remains outstanding. No 90-system corpus or production training was run.

After choosing and validating a stable h, building the complete 90-system corpus, and checking the two-GPU chunk/memory behavior, the production runner's exact 500-epoch DDP template is:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=2 train_models/train_lap_s5.py \
  --mode production \
  --lap-corpus /shared/path/lap_full_vxc_corpus_h_validated \
  --output-dir /shared/path/lap_s5_trial19 \
  --device cuda --dtype float32 --point-chunk-size 128 \
  --predopt-chunk-size 4096 --predopt-epochs 2 --predopt-lr 1e-2 \
  --n-train 500 --training-state-every 10 \
  --mrks-dispersions-pickle train_models/dispersions/dispersions_mrks.pickle \
  --reaction-dispersions-pickle train_models/dispersions/dispersions.pickle
```

The chunk size 128 is only a conservative starting value for future V100 validation; the RTX-selected value 256 is not transferred as a V100 recommendation. The command above was documented but not run.

No production training jobs or Slurm jobs were submitted.
