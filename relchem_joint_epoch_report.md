# Relchem-only versus joint AdamW: one chemistry epoch

Both fresh arms completed exactly 251 optimizer updates with no numerical failure. **Case D:** removing the competing tasks increases relchem improvement, but joint training also improves relchem and all three other full training objectives. This is a matched single-seed diagnostic, not a final functional or an external-validation result.

## Permanent chemistry policy

Exactly one selected variant per identity is mandatory for every chemical objective evaluation. The fixed panel contains 251 relchem and 17 AE17 identities; all 90 mRKS systems are evaluated as equal-system means. No exhaustive eight-variant objective was computed. The policy supersedes historical exhaustive endpoint plans without rewriting their reports. It is linked from both READMEs, enforced by the selected-variant manifest validator and endpoint evaluator, and covered by fail-closed tests. Storage-only array comparisons may still inspect augmentation arrays.

## Frozen experiment and numerical qualification

Initial model: corrected seed11 P536, 536 PBE predopt updates, no additional displacement and never S5. Tensor SHA256: `3c2e45d6a86cccf80d7b4dc458e67f9b406935ed60d4e57433bb441a6df88da6`. Dataset logical SHA256: `61c221a19b9987717e69cac182ad545241f8807db4126c0949a99992e4c210ef`.

Native F32 AdamW: constant LR 1e-4, betas (0.9, 0.999), epsilon 1e-8, weight decay 0.01, foreach=false, fresh moments. R uses only relchem with lambda 0.017015480965588553. Uniform positive gradient scaling is approximately cancelled by AdamW moments, apart from epsilon and rounding. J uses the unchanged coefficients:

| Task | Lambda |
|---|---:|
| relchem | 0.017015480965588553 |
| ae17 | 5.1412546183474142e-05 |
| exc | 1.5094644512009712e-05 |
| op | 0.33597561607048215 |

The relchem epoch is a deterministic shuffled permutation of all 251 identities without replacement, with one independently selected variant per identity. R/J consume byte-identical relchem manifests. J additionally draws an independent AE17 identity/variant per update and uses one shared Exc/operator system from balanced 90-system cycles. All 90 mRKS systems are sampled (two complete cycles plus 71 systems). Training seed 202610092 and independent evaluation seed 202610091 were frozen before training; accidental train/evaluation variant coincidences are allowed and do not change the independence of the streams.

The executed singleton loss is ell=a_db*sqrt((Epred-Eref)^2+1e-20), with a_db=FCHEM_DB_WEIGHTS*FREQ_WEIGHTS/MEAN_WEIGHT. It is not a nonlinear multi-reaction RMSE. At fixed parameters, the expected shuffled-epoch mean equals the uniform-identity mean of the variant expectation. Parameters evolve during training: random reshuffling is not an IID conditional-unbiasedness claim at each update. Evaluations are exact for the declared fixed-variant population, not exhaustive augmentation averages.

Chemistry losses and full F64 gradients match the existing implementation bitwise at preflight. J reuses the qualified four-task measurement and scalarization; R invokes none of the competing gradient factories. The actual R loop was tested to reject any joint-gradient call and to reproduce uninterrupted tensors and native AdamW moments after a runtime pause/resume. F64 derivatives are combined before the single F32 optimizer-boundary cast. Chemistry chunk threshold/chunk remain 131072/16384; Exc/operator chunks remain 4096/256. Corrected operator derivatives, F32 PBE arithmetic, F64 AO assembly, overlap convention and physical gauge are unchanged.

Frozen SHA256s:

- Training manifest: `72b827655ce4421f2fc933c082cda1c2dc3e5d9903ec29e0c5e3d62e82ac37f6`.
- Evaluation manifest: `132bd550161be6bc6761f8df1de8d3755c0b0debab53663208c3a9bbf56cb805`.
- Protocol: `827f337e70077e60fea714eb0535c42bd7d0a9f112ce0ec7dcb8320046f0123d`.
- Numerical preflight receipt: `3a6ec628285f170aa9c43578cc0496f272273430aa9c408841159f03b80baa3f`.

## Exact fixed-panel endpoints

| Objective | Common t0 | R t251 | R ratio | J t251 | J ratio |
|---|---:|---:|---:|---:|---:|
| relchem | 1.235471098 | 1.127941952 | 0.912965 | 1.210400587 | 0.979708 |
| ae17 | 24.9010471 | 75.65475895 | 3.038216 | 2.017287242 | 0.081012 |
| exc | 92.17480502 | 279.0209322 | 3.027085 | 9.31931122 | 0.101105 |
| op | 0.03314821682 | 0.04056216823 | 1.223661 | 0.03109704518 | 0.938121 |

R relchem falls 8.703%; J falls 2.029%. The joint gap is 6.674 percentage points: 23.32% of R's objective reduction survives joint training. R improves 183/251 identities (72.91%); J improves 202/251 (80.48%). J improves more identities, but R's larger magnitude gains yield the greater mean reduction.

R worsens AE17 and Exc by about 3x and operator loss by 22.37%; it is diagnostic only. J improves all four declared training objectives, but no external validation was run and no final functional is promoted. Full90 t0 values were reused only after verifying the old receipt/checkpoint hashes, exact initial tensor identity, dataset, chunks, physics hashes and unchanged model/objective sources; both t251 full90 panels were newly evaluated. Every evaluation preserved model hashes and performed no parameter backward or optimizer step. Local input derivatives required for the operator remain exact.

| Database | n | t0 singleton mean | R ratio | J ratio | R improved | J improved |
|---|---:|---:|---:|---:|---:|---:|
| ABDE4 | 4 | 5.610455 | 1.354702 | 1.097354 | 0 | 0 |
| DBH76 | 70 | 1.011435 | 0.981949 | 0.940931 | 44 | 69 |
| EA13 | 11 | 1.714299 | 1.132119 | 0.992694 | 4 | 5 |
| IP13 | 13 | 2.361088 | 0.799510 | 1.095345 | 7 | 3 |
| MGAE109 | 104 | 0.2820114 | 0.620480 | 0.904489 | 87 | 94 |
| NCCE31 | 28 | 2.617567 | 0.859377 | 0.854298 | 26 | 28 |
| PA8 | 8 | 1.62591 | 1.037329 | 1.065882 | 4 | 3 |
| pTC13 | 13 | 3.975477 | 0.824742 | 1.107175 | 11 | 0 |

R has especially larger gains in MGAE109, IP13 and pTC13. J is better in DBH76, EA13, NCCE31 and ABDE4, although ABDE4 still worsens. Both worsen PA8. The reduction is not uniformly distributed, and an improved-identity count alone misses magnitude/tail differences. Complete paired identity losses and per-database RMSE values are in the JSON.

## The S5 comparison is a different metric

Historical optuna_joint.compute_fchem_from_errors sums FCHEM_DB_WEIGHTS[db]*RMSE_db, including AE17 as the ninth chemistry database. S5 had 268 identities including AE17, batch1 per GPU, world_size2, 134 microbatches per GPU: 45 optimizer steps at accumulation3 (nominal global batch6; partial last window4), then 67 at accumulation2 (global batch4). One variant was selected per identity/epoch. Its shorter mRKS loader was cycled with chemistry microbatches. This experiment uses 251 single-GPU optimizer steps, with AE17 separate in J. RAdamW/warmup, legacy architecture/potential supervision, precision and reporting time policy also differ.

We reconstruct the exact same database-RMSE aggregation formula from the squared singleton errors, removing the 1e-20 smoothing term before RMSE. No second model forward or extra variant is needed. Reaction energies/errors are kcal/mol. These fixed-checkpoint scores are not numerically identical historical baselines: S5 epoch training metrics accumulated predictions from changing model states.

| S5-style score | Common t0 | R t251 | R ratio | J t251 | J ratio |
|---|---:|---:|---:|---:|---:|
| Eight non-AE databases | 46.125046 | 42.948464 | 0.931131 | 45.146732 | 0.978790 |
| Nine databases including AE17 | 102.658303 | 218.076104 | 2.124291 | 51.182314 | 0.498570 |

The nine-database score already falls 50.14% in J after this epoch; 98.10% of that decrease comes from AE17. Therefore a historical ~50% 'chemistry' reduction cannot be interpreted as ~50% relchem improvement without separating AE17 and verifying the reporting scalar. This metric distinction is a major part of the apparent discrepancy.

## Runtime and memory

RTX 5070 Ti, single GPU, synchronized update wall time including loading/transfers/gradient computation/AdamW; checkpoint and logging overhead reported separately through checkpoint timestamps. Reserved allocator memory is not a measurement of physical resident VRAM.

| Arm | Steps/examples | Update-work min | Checkpoint-interval min | Mean s/update | Peak allocated GiB | Peak reserved GiB | Relchem reduction percentage points/GPU-minute |
|---|---:|---:|---:|---:|---:|---:|
| R | 251/251 | 8.079 | 8.160 | 1.931 | 5.254 | 7.406 | 1.07724 |
| J | 251/251 | 57.216 | 57.308 | 13.677 | 13.103 | 19.309 | 0.03547 |

Normalized relchem reduction per step/example: R 0.00034675, J 0.00008085. First/second-half sampled training means: R 1.442196/1.026708; J 1.539630/1.068888. These halves contain different identities and changing parameters; they are not population convergence estimates.

## Interpretation and exactly one next experiment

Case D: the matched objective contrast establishes that joint objectives suppress some chemistry improvement under this optimizer/seed/sampler, while improving the other tasks dramatically. It does not identify which individual competing objective causes the gap or prove incompatibility of final targets. Existing three-draw factorial diagnostics show chemistry opposing the other tasks in two draws (cosines -0.647 to -0.854), but they are supporting local evidence only. No new gradient audit was performed.

Neither relchem objective reproduces a 50% relchem reduction after one epoch; the S5-compatible combined score does. Both arms share the new sampler, so this design does not independently establish sampler causality relative to historical IID training.

**Recommend one additional matched epoch only:** resume both preserved t251 checkpoints to t502 with unchanged constant LR, coefficients and AdamW moments, a newly frozen shared epoch1 relchem permutation/variants, and the identical fixed evaluation manifest. This tests sustained chemistry learning and the evolution of the trade-off over a two-epoch horizon before changing coefficients/optimizer or removing tasks. A second epoch is useful to resolve trajectory persistence; it is not needed to establish the current matched gap. A third epoch is not justified automatically. This continuation was not launched.

## Tests, provenance and preservation

65 tests passed; 3 Linux/WSL Gloo tests skipped on Windows. Ruff, compileall and git diff --check passed. Tests cover unique/no-replacement epoch coverage, independent deterministic selected variants, exhaustive/missing evaluation rejection, exact F64 scalarization/native boundary, R exclusion of joint gradients and paused-loop resume tensors/moments. Real preflight chemistry losses/gradients were bitwise equal. All 9446 coordinates are represented once, optimizer step counters are251, all moments/parameters are finite, log hash chains match final checkpoints, frozen source/manifest hashes match, and all endpoint model-restoration checks pass.

Historical LR1e-4 cursor59 checkpoint remains unchanged: `8938753bee6cfcb55ccdac9c02a15cbb3e3b2093290c9122e5e00ad400a966aa`. Dataset access was read-only through the qualified publication API with its hash checks. Large checkpoints/logs/endpoint receipts remain outside Git at `C:/Dev/readWFN_share_ms/lap_relchem_joint_epoch_20261009`. Small metrics contain checkpoint and receipt hashes. Final checkpoint file SHA256s:

- R: `6601b4e0949862fb428429a6b1090d21fb23df0eefdce76ba482c66cfd8524c7`.
- J: `11cee17f61c017e26902c905b6d9ace0b576ab7eaacf5d5f93f5172a49e3257a`.

Ponytail review: reused the qualified objectives, joint gradient path and native AdamW; added only epoch/selected-variant policy and the relchem-only branch. Pocock review: identical initial tensor hashes and matched relchem manifests; fixed evaluation identities/variants; retained DB factors and precision; no conditional-IID overclaim; optimizer/checkpoint integrity checked; metric comparability and single-seed limits explicit. No production optimizer, architecture, physics or target changed.

Exactly two fresh arms, 251 updates each. No exhaustive variant objective, historical rerun, SVRG, gradient surgery, extra optimizer/seed, second epoch, external validation, SCF or Diet100 evaluation was run.
