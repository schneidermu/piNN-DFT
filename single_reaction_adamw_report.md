# Single-reaction four-objective AdamW qualification

## Outcome and scope

The separate ordinary-stochastic AdamW protocol is numerically ready for a short convergence pilot. It has **no SVRG**, no full251 reference computation or reuse, no per-step Pareto/Armijo condition, and no retained training updates in this qualification. The previous AdamW experiment remains at cursor 7; its checkpoint SHA256 is unchanged: `9cf49177f71d5eb3c4f11bc95d7098a1fd74ead14302e03a2977c324839e83fa`. Historical SVRG code and artifacts were not edited.

Dataset logical SHA256: `61c221a19b9987717e69cac182ad545241f8807db4126c0949a99992e4c210ef`. Corrected seed11 P536 initial model SHA256: `3c2e45d6a86cccf80d7b4dc458e67f9b406935ed60d4e57433bb441a6df88da6`. Initialization is the verified corrected PBE-predoptimized state, never S5. No dataset content, model architecture, target, gauge, dispersion convention or scientific precision boundary changed.

Entry point: `train_lap_microbatch.py`. Default invocation performs bounded qualification; `--train-updates` explicitly opts into a separate resumable run. No such convergence run was launched. The previous experiment is neither resumed nor overwritten. The new fixed protocol uses one relchem reaction, one independent AE17 reaction and one shared Exc/operator mRKS system per update. Chemistry graphs are evaluated and released sequentially; only detached parameter gradients are retained for combination.

## Mathematical objective and estimator

Let `a_d = FCHEM_DB_WEIGHTS[d] * FREQ_WEIGHTS[d] / MEAN_WEIGHT`, exactly as executed by `optuna_joint.batch_fchem`. For one reaction and variant,

`ell(r,v;theta) = a_database(r) * sqrt((E_pred(r,v;theta)-E_ref(r))^2 + 1e-20)`.

Reaction integration, source-F32 rounding before matched-F64, native dispersion scalar semantics, and the F64 shadow derivative are unchanged. The qualified `ChemistryBatchObjective` averages **separately evaluated singleton losses**. It does not call `batch_fchem` on a whole DB and thereby introduce a nonlinear DB RMSE. A mean of singleton smoothed absolute errors is not the derivative of multi-reaction RMSE; neither is claimed interchangeable here.

The newly declared augmentation-averaged chemistry objectives are

`F_rel = (1/251) sum_r (1/8) sum_v ell(r,v;theta)`;

`F_ae = (1/17) sum_r (1/8) sum_v ell(r,v;theta)`.

For independent uniform identity and uniform variant draws, at the current parameters, `E[grad ell(R,V;theta)] = grad F_task(theta)`. There is no additional DB factor outside `ell`: grouping this identity expectation by DB already yields `sum_d (n_d/251) mean_d(grad ell)`. Adding another `n_d/251` to uniform draws would change the objective. Eight variants do not create eight independently weighted reaction identities.

The latest user instruction is implemented as ordinary seeded uniform draws **with replacement**, rather than a no-replacement reaction cycle whose later conditional distribution depends on already consumed identities. Relchem and AE17 streams have independent seed domains. All 251 and 17 identities remain in their respective populations. mRKS keeps a deterministic balanced 90-system cycle; Exc/operator share the same system. A reporting epoch of 90 updates means one mRKS cycle, not complete chemistry coverage or an S5 epoch.

Historical exact full251 diagnostics used one frozen variant per identity. They are not the eight-variant mean and cannot be relabeled as its exact value. This task did not compute a new full-corpus baseline or claim scientific convergence. Future endpoint selection must distinguish the declared augmentation-averaged training objective from the established fixed-variant diagnostic and compare like-for-like against the corrected initial model.

The earlier suspicion of an SVRG reference/variant mismatch was disproved: all 200 checked historical sampled relchem rows used their identity's exact frozen reference variant. No historical SVRG defect is inferred. Its removal is the explicitly requested new protocol.

## Fixed coefficients and optimizer

Three sample draws (frozen manifest positions 1, 5 and 6) were calibrated at the corrected initial state, before the final timed updates. No validation or loss-based LR selection was used.

| Task | Median gradient norm | Frozen lambda |
|---|---:|---:|
| relchem | 7.997660353595703 | 0.03125914191737355 |
| ae17 | 880.4312969983481 | 0.000283951741438911 |
| exc | 39886.66558676216 | 0.000006267758819202265 |
| op | 0.6460599527776881 | 0.3869609916620633 |

`lambda_i = 1/max(s_i,1e-12)/4`, frozen for the run. This is a new three-batch calibration, not reuse of the old 8+17 estimator's weights. It is deliberately small and may be noisy; no claim of globally optimal scaling is made. Task ordering and all 9,446 parameter coordinates are explicit. Qualified gradients are combined in F64, with one cast at the native F32 AdamW gradient-storage boundary, as in the preceding AdamW implementation.

AdamW: LR `1e-6`, betas `(0.9,0.999)`, epsilon `1e-8`, weight decay `0.01`, `foreach=False`. The opt-in short run uses cosine LR to `1e-7` over its predeclared update count. No learning-rate sweep, adaptive task weighting or optimizer change was added. Individual tasks may rise during an update; nonfinite values stop execution. Checkpoint selection still requires full-objective evidence and is not inferred from these sampled profiles.

## Historical S5 reconstruction

`EpochSampledAugmentedDataset.resample` selects one variant per canonical reaction each epoch. `DistributedSampler(drop_last=False)`, batch size 1 per GPU and two ranks give `ceil(N_chem/2)` chemistry microbatches per GPU. mRKS batch size is also 1, with `ceil(N_mrks/2)` batches per local pass. `train_one_epoch` uses `n_steps = max(len(train_loader),len(vxc_train_loader))` and cycles the shorter loader. It calls `optimizer.step()` at each accumulation boundary and at the final partial window: `ceil(n_steps/accum_iter)` calls per epoch, not one per microbatch.

For the documented Diet-clean replay population (268 chemistry identities, 90 mRKS systems):

| S5 phase | Epochs | Accumulation | Chemistry microbatches/GPU/epoch | Actual optimizer calls/epoch | Global chemistry examples/full step |
|---|---|---:|---:|---:|---:|
| Initial anchor | 1–72 | 3 | 134 | 45 | 6 |
| Subsequent phases | 73–500 | 2 | 134 | 67 | 4 |

The initial phase's final partial window has two microbatches per rank (four global examples). A mRKS system is processed at every local microbatch, jointly for Exc and pointwise-vrho, so there are 134 system observations per GPU/epoch: two passes through its 45-system shard plus 44 repeated observations. These are source-derived counts for the stated corpus, not a throughput claim for an unidentified historical run. Older replay logs with different populations/presets must not be substituted into this calculation. The reported 100+ updates/minute was not independently measured here.

The new single-GPU step contains two distinct chemistry task examples and one mRKS system. It preserves corrected F64 chemistry, full density/gradient/Laplacian dependence and the weak-form AO operator, unlike S5's F32 legacy pointwise-vrho supervision. It does not copy the scientifically flawed S5 architecture. No dual-V100 speedup is inferred.

Two-GPU execution currently fails closed. To keep a future global batch at one example per task, partition that global batch and reduce each task's gradient sum using its **global task sample count**, not a blind average of independently doubled per-GPU batches. No new distributed framework was implemented.

## Synchronized timing and memory

The old three measurements are reused, not rerun. They used cursor-7 parameters and AlH3/CS/CO. New measurements use the corrected initial parameters and frozen AlHS/SSi/HNaO draws. Thus these are representative workload comparisons, not matched-system speedup measurements; varying systems and warm/cold allocator behavior confound a causal overall speedup estimate.

Seconds, CUDA synchronized at boundaries. New data time combines chemistry and mRKS loading/transfers. Measured total includes the small norm/finite-check bookkeeping residual. Final timed updates actually computed all four gradients and applied a temporary AdamW step after calibration, then restored the initial model and RNG exactly.

| Configuration / draw | Data | Relchem | AE17 | Exc | Operator | AdamW | Total | Peak live GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Old 8+17 / 1 | 3.099 | 18.803 | 4.452 | 14.855 | 28.404 | 0.119 | 69.730 | 10.37 |
| Old 8+17 / 5 | 1.559 | 29.536 | 4.184 | 10.217 | 18.103 | 0.025 | 63.624 | 20.27 |
| Old 8+17 / 6 | 0.673 | 17.109 | 5.161 | 11.238 | 14.878 | 0.028 | 49.087 | 12.18 |
| New 1+1 / 1 | 1.621 | 5.366 | 0.381 | 17.988 | 6.607 | 0.015 | 31.994 | 10.44 |
| New 1+1 / 5 | 0.809 | 1.124 | 0.198 | 13.132 | 4.635 | 0.013 | 19.913 | 7.86 |
| New 1+1 / 6 | 1.224 | 1.432 | 0.134 | 18.319 | 6.178 | 0.014 | 27.304 | 9.79 |

Old chemistry processed 1,044,618 / 1,282,488 / 1,121,725 grid points and 25 current reaction backwards per update, plus checkpoint recomputation. New chemistry processes 281,180 / 78,064 / 86,256 points and two current reaction backwards. New timing uses **zero reference gradients**. Reserved memory in the final new profiles was 10.49 / 11.80 / 13.17 GiB, below the GPU's physical 15.92 GiB. The three final totals average **26.404 seconds**, versus **60.814 seconds** for the old representative set. Do not attribute that entire difference solely to sampling.

An earlier unchunked preliminary new set took 44.19–52.16 seconds. It is retained in external `metrics.json`, not discarded. The final set uses bounded chemistry and a fresh allocator process; paging/allocator effects were not separately measured. Both sets show that AdamW itself is negligible.

## Large-reaction memory gate

Single-reaction processing alone did not resolve pTC13-7, variant `level3_mura`, 500,568 points. A narrowly scoped optional `model_point_chunk_size` field now chunks only the checkpointed pointwise NN call; reaction integration and all scientific arithmetic remain unchanged. Existing callers without that field take their original path.

| NN model chunk | Loss | Gradient time | Peak live GiB |
|---|---:|---:|---:|
| Whole grid | 5.7088005255572165 | 13.868 s | 20.001 |
| 16,384 points | 5.7088005255572165 | 10.519 s | 1.161 |

Loss equality is exact. Gradient relative L2 difference is `3.0706352181870266e-12`; maximum coordinate error is `2.545759159033878e-10`. Predeclared tolerances were loss absolute `1e-10`, gradient relative L2 `1e-10`, and gradient maximum absolute `1e-9`; all passed. This is numerical gradient parity, not bitwise gradient equality. Parameters were unchanged.

New training enables 16,384-point model chunks above 131,072 grid points. Checkpointing is retained: simply removing it would retain the large NN activations rather than bound their live working set. Small-reaction checkpoint removal was not benchmarked or implemented. No AO/operator approximation or operator implementation change was made.

## Validation, recommendation and limitations

Individual singleton reaction loss and gradient replay: exact loss and bitwise gradients for the checked relchem and AE17 examples. Fixed aggregation algebra: exact. Native AdamW displacement semantics and opposing-task updates: tested. A real disk checkpoint restored model, optimizer and RNG and reproduced the next step exactly; a focused scheduled-checkpoint test also restored the scheduler and cursor. Manifest/calibration mismatches fail closed. Dataset references are loaded through the immutable publication API.

Tests: **40 passed, 2 skipped** (existing opt-in live distributed checks); Ruff, compileall and `git diff --check` passed. Full90 gradients, full-corpus endpoints, external validation and future-test evaluation were not run. No all-four-objective scientific improvement is claimed from timing tests.

Measured projection: **about 39.6 minutes for 90 updates**, with the observed per-update range projecting 29.9–48.0 minutes. This excludes full-corpus/validation evaluations and is based on three systems, not a full 90-system timing census or V100 benchmark.

The smallest remaining measured bottleneck is **Exc integration (13.1–18.3 s)** with the unchanged 256-point chunk, followed by the corrected operator (4.6–6.6 s in the final set). No further optimizer research or performance changes were made. A larger existing Exc chunk would require a separate loss/gradient parity test before retention; it is not part of this protocol.

Recommendation: proceed, on explicit authorization, with **10 updates of this ordinary single-reaction AdamW protocol**, frozen coefficients above, LR `1e-6` with declared cosine schedule, bounded large-reaction chunks, one independent AE17 example and shared Exc/operator system. Expect roughly 4–6 minutes of update compute, with separate bounded endpoint evaluation later. This is a numerical readiness recommendation, not scientific promotion; convergence and clean-validation improvement remain untested.

External artifacts: `C:/Dev/readWFN_share_ms/lap_single_reaction_adamw_20261008`. Small hash-bound results and receipts are in `single_reaction_adamw_metrics.json`. Historical cursor-7 work and the paused full90 qualification remain preserved. The frozen publication dataset was not changed.
