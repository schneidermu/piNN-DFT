# Frozen global chemistry objective audit

## Scope and identity

**STOP before any checkpoint forward evaluation.** The mandatory sampling-provenance gate fails. Production source is unchanged. No model parameters, RNG, EMA, cursor, optimizer, or scheduler were loaded or advanced. No training, Diet30/Diet100, Slurm, V100 jobs, or AO generation occurred.

Repository `schneidermu/piNN-DFT`, branch `lap_full_vxc`, inspected/pushed starting HEAD `3dd2a244fa4135328d07071a0b82ad0ed153556f`. This report's commit is the commit introducing these three artifacts; no recursive self-commit hash is fabricated. Exact file/source hashes are in the protocol.

- Predopt SHA-256: `ed4ba8231d93c376ce5aa8fc81e0670f6a5a44c95c2a552651b12d6a1d4b63f8`.
- Reviewed clean cursor-2 SHA-256: `14c1c5d50cf1e3f17e55d66d0be353d7d8871291718dbc10d3176746b2bbaf69`.
- Both checkpoint file hashes were verified, without loading their tensors.
- Intended precision remains matched-F64 arithmetic on exact F32-loaded values widened to double, not restored native source precision.
- Installed skills used: `caveman`, `ponytail`, `to-spec`, `to-tickets`. The user-requested concise local protocol/tickets replace external tracker publication. No new production abstraction.

## Exact reducer derivation

Inspected `optuna_joint.batch_fchem`, `compute_fchem_from_errors`, weighting constants; `lap_training.reaction_loss`; `lap_moo_protocol` catalog/permutation/builders; and the external group-store manifest.

Let error e_r be predicted minus reference reaction energy in kcal/mol; w_d be FCHEM_DB_WEIGHTS, f_d=FREQ_WEIGHTS=1/N_d, and m=MEAN_WEIGHT=0.11136325384169465. For a corrected singleton database label:

`ell_r = (w_d f_d / m) sqrt(e_r^2 + 1e-20)`.

For exactly one selected variant of each cleaned reaction, M=268:

`J_cycle = (1/(268 m)) sum_d w_d f_d sum_(r in d) sqrt(e_r^2 + 1e-20)`

`        = (1/(268 m)) sum_d w_d (n_d/N_d) smoothed_MAE_d`.

This is a cleaned-count-adjusted weighted database smoothed-MAE functional. It is **not** proportional to the ordinary `sum_d w_d MAE_d` across these cleaned databases, because n_d/N_d varies. It is not weighted RMSE. The epsilon is retained exactly; ignoring it is an approximation.

The actual reporting reducer instead computes:

`J_RMSE = sum_d w_d sqrt(mean_(r in d) e_r^2)`.

It does not divide by 9 or MEAN_WEIGHT and does not use FREQ_WEIGHTS. Were all 268 rows passed to batch_fchem in one batch, it would compute a third quantity: `(1/(9 m)) sum_d w_d f_d sqrt(1e-20 + mean_d e_r^2)`. No full-batch checkpoint evaluation was run.

## Database inventory

| Database | Cleaned n_d | Historical N_d | w_d | n_d/N_d |
|---|---:|---:|---:|---:|
| ABDE4 | 4 | 4 | 1 | 1 |
| AE17 | 17 | 17 | 1 | 1 |
| DBH76 | 70 | 76 | 1 | 0.921052631579 |
| EA13 | 11 | 13 | 1 | 0.846153846154 |
| IP13 | 13 | 13 | 1 | 1 |
| MGAE109 | 104 | 109 | 0.211240310078 | 0.954128440367 |
| NCCE31 | 28 | 31 | 10 | 0.903225806452 |
| PA8 | 8 | 8 | 1 | 1 |
| pTC13 | 13 | 13 | 1 | 1 |

Counts total 268 and cover exactly the nine expected databases. Historical frequency denominators sum to 284, rather than the cleaned 268. The existing panel contains three reactions per database; it is not a uniform-probability sample of the cleaned 268 population. That structural difference alone does not establish its numerical representativeness.

## Sampling gate and execution identity

The clean restart manifest has **27 catalog identities and reaction_cycle_length=27**. Increasing its update count to 268 repeats those identities; it does not create a 268-unique cycle. This was detected before loading checkpoints.

Building from the actual full 268-group inventory using the unchanged existing builder, seed 41, world_size 1, same source hashes, same 15 mRKS names, and available augmentation suffixes gives:

| Position | Required clean restart | Actual full-catalog cycle |
|---|---|---|
| 0 | pTC13/12, level3_mura, BeH2 | PA8/3, level2_mura, BeH2 |
| 1 | AE17/16, level2_gauss_chebyshev, H2 | pTC13/2, level2_delley, H2 |

The builder sorts the complete catalog then shuffles it. Changing the catalog membership changes the permutation, even with the same seed. Reversing input catalog order reproduces the full manifest exactly, excluding accidental store ordering as the cause.

Rejected full-manifest canonical identity: `f20225fc6d15e26d0729eb5ad33300841ffc15355608df6ee4dcae58add44691`. External manifest and driver paths/file hashes are in results. It is diagnostic evidence, **not an accepted sampling manifest**. The rejected manifest was not used to evaluate either checkpoint. No per-reaction model arrays exist for this task.

## Global metrics and panel comparison

| Metric | Predopt | Cursor 2 | Ratio/delta | Interpretation |
|---|---|---|---|---|
| Mean singleton training loss | Not run | Not run | Unavailable | Provenance gate failed |
| Derived weighted-smoothed-MAE objective | Not run | Not run | Unavailable | Formula verified synthetically only |
| Weighted database RMSE | Not run | Not run | Unavailable | Formula verified synthetically only |
| Overall unweighted MAE | Not run | Not run | Unavailable | No checkpoint evaluation |
| Overall unweighted RMSE | Not run | Not run | Unavailable | No checkpoint evaluation |
| Median row error ratio | Not run | Not run | Unavailable | Baseline floor predeclared 1e-6 kcal/mol |
| Fraction improved | Not run | Not run | Unavailable | No numerical panel representativeness claim |

Per-database measured errors, tail statistics, and improvement counts are unavailable. The earlier 27-panel results remain historical evidence; they were not re-evaluated or silently generalized to all 268.

## Deterministic acceptance checks

Passed: pinned checkpoint/store/old-manifest byte hashes; 268 unique identities once each; nine database inventory; catalog-order invariance; singleton factor identity; mean-versus-derived identity on synthetic errors; actual weighted-RMSE reducer identity on those same synthetic errors. Synthetic values are explicitly separate in JSON and are not scientific checkpoint outcomes.

Failed: first two full-cycle entries reproduce the clean restart. Therefore later finite-forward checks, actual 268-mean equality, panel transfer evaluation, runtime/GPU profiling, and any success/failure classification of chemistry improvements were not run. Both checkpoint files remained unchanged. Production source changes: zero. Existing full suites were not rerun because source is unchanged.

## Classification and single next experiment

**Inconclusive due to incompatible requested sampling identities**, not precision, optimizer, or objective-transfer evidence. No Case 1вЂ“4 numerical improvement classification is justified.

The missing prerequisite is one explicitly defined full-268 diagnostic manifest rule compatible with the frozen restart's 27-catalog provenance. Then run the chemistry-only two-checkpoint audit. Do not silently splice the first two samples, change seed, invent augmentation selection, or continue optimization to evade the gate.

## Independent review

Passed one independent GPT-6 Luna MAX review of actual source, derivation, gate receipt, and non-execution claims. Reviewer confirmed the STOP and unavailable numerical outcomes; no rerun or reimplementation occurred. Review: `C:/Dev/readWFN_share_ms/lap_chem_global_audit_runs_20261004/review.md`, SHA-256 `805ae11ee6e6cff361347faaecb913fbeb021aa48b848da223e6250ffbb91ae0`. External diagnostic syntax, artifact-consistency, and staged whitespace checks passed. Ruff is not applicable to the three committed Markdown/JSON artifacts; no Python source/tooling is committed.

## Eleven answers

1. **Exact singleton expectation?** The formula above: `(268 m)^-1 sum_d w_d f_d sum_r sqrt(e_r^2+1e-20)` for the specified full cycle.
2. **Weighted MAE or RMSE?** Cleaned-count-adjusted weighted smoothed database MAE; neither ordinary canonical weighted MAE nor weighted RMSE.
3. **Explicit actual 268 mean agrees?** Not evaluated; synthetic arithmetic agrees to rtol 1e-14. Actual acceptance blocked.
4. **Singleton expectation improved?** Unknown.
5. **Weighted DB-RMSE improved?** Unknown.
6. **Which database MAEs improved/worsened?** Unknown.
7. **Which database RMSEs improved/worsened?** Unknown.
8. **How many reactions improved?** Unknown.
9. **Is panel deterioration representative?** Not established; panel has equal DB counts, whereas full-cycle counts differ.
10. **Category?** Inconclusive provenance gate failure before numerical evaluation.
11. **Single next experiment?** Establish the compatible full-cycle diagnostic manifest rule, then execute the frozen chemistry-only audit.

The global chemistry audit is numerically inconclusive; the single missing diagnostic is a provenance-compatible full-268 frozen chemistry evaluation.
