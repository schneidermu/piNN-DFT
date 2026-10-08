# Exc throughput and 90-update AdamW qualification

One corrected seed11 P536 trajectory completed all 90 updates. Numerical qualification PASS. Not all four exact objectives improved. Clean28 worsened; the competitive validation target was not met.

## Exc checkpoint chunks

Systems selected by sorted (grid length, canonical ID): positions 0,45,89 of 90. Operator stays at 256. Frozen tolerances: energy absolute 5e-7 Ha; loss relative 2e-6; full-gradient relativeL2 and max-coordinate/max-reference-coordinate5e-6. All six candidate-vs-256 comparisons passed; all nine evaluations were finite; 4096 was fastest and safely below the frozen 80% live-allocation limit. A non-checkpointed variant was unnecessary once Exc ceased to be the primary bottleneck.

| System | Grid | Chunk | Exc forward/backward(s) | Allocated(GiB) | Reserved(GiB) | Gradient relL2 | Chunks/recomputations |
|---|---:|---:|---:|---:|---:|---:|---:|
| H2 | 42508 | 256 | 7.9770 | 0.074 | 0.088 | 0 | 167/167 |
| H2 | 42508 | 1024 | 1.6551 | 0.090 | 0.104 | 1.07e-06 | 42/42 |
| H2 | 42508 | 4096 | 0.4365 | 0.153 | 0.166 | 8.35e-07 | 11/11 |
| ClHMg | 95661 | 256 | 16.4109 | 0.529 | 0.537 | 0 | 374/374 |
| ClHMg | 95661 | 1024 | 4.0207 | 0.502 | 0.520 | 2.07e-06 | 94/94 |
| ClHMg | 95661 | 4096 | 0.9499 | 0.564 | 0.582 | 1.4e-06 | 24/24 |
| H4Si | 125620 | 256 | 21.7093 | 1.021 | 1.109 | 0 | 491/491 |
| H4Si | 125620 | 1024 | 5.1122 | 0.584 | 0.691 | 1.34e-06 | 123/123 |
| H4Si | 125620 | 4096 | 1.1948 | 0.647 | 0.754 | 1.56e-06 | 31/31 |

Largest energy discrepancy 6.91e-09 Ha; largest loss relative discrepancy 7.11e-08; largest gradient relativeL2 2.07e-06.

## Fixed calibration

12 frozen draws cover all 8 chemistry DBs and mRKS grid-size quantiles. epsilon=1e-12, C=4; lambda=1/(4*median_norm). Old three-draw coefficients are retained in the metrics. New coefficients were frozen before update 1.

| Task | Median norm | IQR | p90 | Max | Old lambda | New lambda | Median weighted-norm share |
|---|---:|---:|---:|---:|---:|---:|---:|
| relchem | 162.38 | 11.2935–227.258 | 289.104 | 397.178 | 0.0312591419 | 0.00153959468 | 24.68% |
| ae17 | 4052.78 | 2123.68–9257.54 | 10951.8 | 13367.9 | 0.000283951741 | 6.16859973e-05 | 26.32% |
| exc | 20811.9 | 15510.8–27231.6 | 28496.3 | 39886.7 | 6.26775882e-06 | 1.2012379e-05 | 23.87% |
| op | 0.502743 | 0.376143–0.634959 | 0.737314 | 0.906515 | 0.386960992 | 0.497272084 | 23.08% |

With old coefficients, median relchem weighted-norm share on this panel was 76.69%; revised median task shares are 23–26%. These norm shares are scale diagnostics, not additive attribution of squared combined-gradient norm; gradients can cancel. Maximum revised panel share was 57.70% (AE17). Outlier identities are SHA-bound in calibration_manifest.json and per-draw records. The panel deliberately covers each DB, so its median is not a population-frequency-matched estimate for uniform identity training. The completed run exposes that limitation; coefficients were not changed during training.

| Task | Actual 90-update median raw norm | Actual median weighted-norm share |
|---|---:|---:|
| relchem | 13.7443 | 2.26% |
| ae17 | 4868.73 | 32.63% |
| exc | 18008.8 | 22.29% |
| op | 0.683066 | 33.47% |

Relchem actual median raw norm 13.744 vs calibration 162.380: its typical weighted norm is suppressed by about 11.8× relative to the intended calibration scale. Actual median task shares are 2.26%,32.63%,22.29%,33.47% (relchem,AE17,Exc,operator). This supports a calibration-distribution mismatch; recorded training norms are at evolving parameters, so the comparison alone is not causal proof of validation deterioration.

## Frozen training and resume

Initial tensor SHA256: `3c2e45d6a86cccf80d7b4dc458e67f9b406935ed60d4e57433bb441a6df88da6`. File SHA256: `0ca0f77367214c171e6edf0576bdb043d64479245ad1a0e539fa11fb0cee542d`. This is the independently replayed seed11 P536 snapshot after 536 predopt Adam steps; states.py captures it directly, with no additional displacement. It is not the different-seed canonical seed41 artifact and never S5.

LR 1e-6→1e-7 cosine over 90 updates; AdamW betas=(0.9,0.999), eps=1e-8, weight_decay=0.01, foreach=False. One independent uniform relchem identity/variant, one AE17 identity/variant, one shared mRKS system per update; no SVRG, Armijo or Pareto constraints. Chemistry matched-F64 shadow on historical rounded inputs; Exc native F32 functional/F64 quadrature; operator learned F64/PBEF32/AOassemblyF64 unchanged. t10 checkpoint restored exact model,RNG,cursor and 90-step scheduler (LR 9.7286167935e-7); continuation retained optimizer state. 42 tests include optimizer/scheduler/RNG resume replay. Snapshots 0/10/45/90 are external and SHA-bound.

The inherited outer metadata contains historical LR-calibration fields that are not executed. microbatch_90update_protocol.json records the actual pre-update executable protocol, manifest/calibration hashes and fixed settings; no LR scan was performed.

90 updates: 18.105 min synchronized work; 18.232 min across both invocation logs including startup/checkpointing; mean 12.070 s, median 11.669 s. Old three-update microbatch profile averaged 26.40 s; this is a descriptive comparison across different sampled systems. Exc same-system benchmark supplies the controlled speedup evidence.

| Exclusive stage | Mean seconds/update |
|---|---:|
| chemistry_loading_transfers | 0.0941 |
| relchem | 1.8994 |
| ae17 | 0.2706 |
| mrks_loading_transfers | 0.8832 |
| exc | 1.0439 |
| op | 7.8659 |
| weighted_aggregation_adamw | 0.0127 |

Peak allocated 13.100 GiB; peak reserved 29.672 GiB on 15.92 GiB RTX 5070 Ti. Reserved allocator memory is not live allocation; Windows reservations/paging must not be mistaken for extra physical VRAM. No nonfinite/OOM events occurred. Operator dominates measured compute. Coverage:90/90 unique mRKS;78/251 relchem identities;17/17 AE17 identities. Sampled loss trajectories are retained in the structured metrics and are not full-corpus estimates.

## Exact endpoints

Chemistry: all 251×8 relchem and 17×8 AE17 singleton losses, uniform identity/variant means. Historical fixed-variant full251 diagnostics are a different objective and were not substituted or rerun. Exc/operator: equal-system average over all 90, no parameter backward. Operator still requires local density derivatives. Every endpoint stage verified unchanged model tensors.

Relchem/AE17 losses retain database factors and kcal/mol units; Exc is kcal/mol; operator is the qualified overlap-orthogonalized squared loss. **Rmax=1.007042283; t90 is not scientifically eligible because relchem increased.** This does not invalidate numerical qualification.

| Objective | t0 | t90 | R=t90/t0 |
|---|---:|---:|---:|
| relchem | 1.23675509377 | 1.24546467299 | 1.007042283 |
| ae17 | 24.8923555012 | 19.7401812644 | 0.793021828 |
| exc | 92.17480502 | 73.3262539229 | 0.795512981 |
| op | 0.0331482168161 | 0.0326912765242 | 0.986215238 |

| Validation(kcal/mol) | t0 | t90 | Change |
|---|---:|---:|---:|
| clean28 | 9.553191 | 9.571633 | +0.018442 |
| full30 | 9.865317 | 9.889270 | +0.023953 |

Exact frozen PBE0 densities/grids and PBE0-D3(BJ); full30 selection_allowed=false. Clean28 target 6.281513 (B3LYP), r2SCAN 6.320373, S5 historical 6.971800, published NN-PBE 7.453939, PBE 9.479572. No baseline was recomputed. t90 does not reach the competitive target.

## Decision and integrity

**One next experiment:** Reconsider fixed coefficients in one separately declared bounded calibration experiment: match the uniform identity/variant training distribution for calibration medians, retain rare-DB coverage as a separate diagnostic, and keep the same inverse-median/C=4 rule. Do not change this run or extend it automatically.

Dataset logical SHA256 remains `61c221a19b9987717e69cac182ad545241f8807db4126c0949a99992e4c210ef`; immutable dataset opened read-only. Initial source and paused cursor7 checkpoint hashes matched; source physics hashes match frozen run. Full-gradient benchmark arrays, checkpoints and full endpoint receipts stay outside Git at`C:\Dev\readWFN_share_ms\lap_adamw_90update_20261008`. Numerical qualification PASS; external generalization did not improve. No Diet100, SCF, optimizer surgery, architecture/precision changes, SVRG or extra trajectory.

Focused suite: 42 passed / 2 expected skips; Ruff, compileall, git diff --check PASS. Metrics include every checkpoint SHA, external receipt SHA, sample/loss trajectories and fixed protocol evidence.
