# Frozen full-268 chemistry evaluation

## Scope and provenance

Starting branch `lap_full_vxc`, HEAD `e4d9b00f2b474c044f99b05f7e0ce37252467201`. Production source unchanged. No gradients, optimizer, update 3, PCD/FS computation, E/operator evaluation, AO cache, Diet30/Diet100, Slurm, V100, or training occurred. Only the two pinned checkpoints were loaded, widened exactly to double, and evaluated under `torch.no_grad()` using the corrected production chemistry factory. The temporary reducer wrapper observes predictions/targets and calls the actual `batch_fchem`; it does not replace objective arithmetic.

Theta0 is canonical PBE predopt; SHA `ed4ba8231d93c376ce5aa8fc81e0670f6a5a44c95c2a552651b12d6a1d4b63f8`. Theta2 SHA `14c1c5d50cf1e3f17e55d66d0be353d7d8871291718dbc10d3176746b2bbaf69`. Theta2 was trained for two bounded diagnostic updates on **27 reaction identities**, not the full 268-catalog. This audit tests transfer of those two steps.

Precision: **matched-F64 arithmetic on F32-loaded source/checkpoint values widened to double**, with existing HF/dispersion values preserved; not reconstructed native-F64 source precision. All hash pins and source hashes are in protocol/results. Detailed rows stay external at `C:/Dev/readWFN_share_ms/lap_chem_full268_eval_runs_20261004/rows.jsonl`, SHA `fb35e08a1188ec724455e2dd821318217eccffdb930a31bec75754911617e597`. Driver, accepted manifest, overlap table and raw results are hash-linked in results.

Installed skills used directly: caveman, ponytail, to-spec, to-tickets. Four concise dependency-aware tickets are in protocol: provenance gate; forward audit; exact aggregation; review/report. No new production abstraction or source edit.

## Evaluation-only manifest rule

The old `e4d9b00` rejection remains unchanged: its positional gate was correctly enforced. This new manifest is a **frozen full-268 evaluation-only chemistry manifest**, not a continuation or restart stream.

Existing builder, seed41, rank0, world_size1, 268 entries, full cleaned inventory, original source hashes reproduce canonical identity `f20225fc6d15e26d0729eb5ad33300841ffc15355608df6ee4dcae58add44691` and file SHA `b47a7271b9cbbe696e735f6f332345d59812fb4dbe87bebca32556104e82c070`. Counts and all 268 unique identities match the inventory. Selected variant RNG seed depends on `variant:rank:reaction_cycle:database:reaction_id`, not permutation position. Every one of the 27 shared identities has an identical selected cycle-0 variant. The complete old-position/new-position/suffix mapping is in results and external overlap.json. For example pTC13/12 moves from position0 to189; AE17/16 from1 to125; both keep their original variants. Prefix equality and mRKS identity compatibility are irrelevant to chemistry-only frozen aggregates.

## Exact objective semantics

Current source was inspected again. Let e_r be signed kcal/mol error, w_d=FCHEM_DB_WEIGHTS, f_d=FREQ_WEIGHTS=1/N_historical,d, and m=0.11136325384169465. Actual corrected singleton loss is `(w_d f_d/m) sqrt(e_r^2+1e-20)`.

For this one frozen variant realization:

`J_cycle = (268 m)^-1 sum_d w_d f_d sum_(r in cleaned d) sqrt(e_r^2+1e-20)`

`        = (268 m)^-1 sum_d w_d (n_clean,d/N_historical,d) smoothed_MAE_d`.

It is a cleaned-count-adjusted weighted smoothed database-MAE functional, not ordinary weighted database MAE, and not weighted RMSE. Historical counts total284; cleaned counts268. Actual `compute_fchem_from_errors` computes `J_RMSE=sum_d w_d sqrt(mean_d(e^2))` without frequency factors, MEAN_WEIGHT, epsilon, or division by nine. Actual mean and analytic reconstruction agree at rtol/atol1e-12 for both checkpoints; independently aggregated raw rows and direct RMSE algebra agree as well. No full-batch third reducer diagnostic was needed.

## Global metrics

| Metric | Predopt | Clean cursor 2 | Ratio/delta |
|---|---:|---:|---:|
| Mean singleton cycle loss | 2.74657100975 | 2.2329055273 | ratio 0.812979354758; delta -0.513665482446 |
| Analytic cycle objective | 2.74657100975 | 2.2329055273 | ratio 0.812979354758; delta -0.513665482446 |
| Weighted DB-RMSE | 103.111944566 | 88.0807722865 | ratio 0.854224723016; delta -15.0311722795 |
| Overall MAE | 12.1327384915 | 12.4248316626 | ratio 1.02407479328; delta 0.292093171089 |
| Overall RMSE | 19.6807525829 | 18.6858183971 | ratio 0.949446334351; delta -0.99493418579 |
| Median row abs-error ratio | 1 | 1.06021591398 | descriptive paired ratio |
| p90 row ratio | 1 | 1.51377883285 | descriptive paired ratio |
| Reactions improved | - | 79/268 | 189 worsened; 0 unchanged |

Row-ratio floor is baseline absolute error >=1e-6 kcal/mol: 268 included, 0 excluded; excluded positions are in JSON. Maximum ratio 120.823060227. Wins use absolute error decrease >1e-10 kcal/mol. All absolute-error median/p90/max values and counts are in results. Ratios and wins are descriptive quantities, not the training objective.

## Per-database metrics

| DB | n | w_d | hist N_d | MAE0 | MAE2 | MAE ratio | RMSE0 | RMSE2 | RMSE ratio | wins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ABDE4 | 4 | 1 | 4 | 2.7195552 | 2.1544317 | 0.792200029 | 3.38793796 | 2.68539373 | 0.792633681 | 3 |
| AE17* | 17 | 1 | 17 | 47.3366994 | 29.2599306 | 0.61812359 | 56.79165 | 38.7499998 | 0.682318613 | 16 |
| DBH76 | 70 | 1 | 76 | 8.62676413 | 8.61945685 | 0.999152953 | 10.0028888 | 10.0381261 | 1.00352271 | 36 |
| EA13 | 11 | 1 | 13 | 2.47212828 | 2.7731909 | 1.12178276 | 3.10553842 | 3.67034809 | 1.18187174 | 5 |
| IP13 | 13 | 1 | 13 | 3.43613768 | 5.05581246 | 1.47136492 | 4.62090605 | 5.92512992 | 1.28224419 | 2 |
| MGAE109 | 104 | 0.21124031 | 109 | 15.8482429 | 19.3112981 | 1.21851351 | 19.8239679 | 23.9486568 | 1.20806576 | 10 |
| NCCE31 | 28 | 10 | 31 | 0.881692809 | 0.950926843 | 1.07852399 | 1.24825495 | 1.32349455 | 1.06027583 | 1 |
| PA8 | 8 | 1 | 8 | 1.47383017 | 1.40789754 | 0.955264432 | 1.72149687 | 1.69378549 | 0.983902741 | 4 |
| pTC13* | 13 | 1 | 13 | 5.81075507 | 5.99704342 | 1.03205923 | 6.81135582 | 7.02412187 | 1.03123696 | 2 |

*AE17 and pTC13 contain the two actual training reactions; no special weighting was applied. All values are kcal/mol before stated loss factors. Per-DB worsened/unchanged counts and smoothed MAEs are in JSON; smoothing is not materially different at the displayed scale.

## Shared 27-row subset versus full realization

The same 27 identities/variants were extracted from already evaluated rows, without another model forward. Shared-subset median ratio 1.014138201, p90 2.31735632363, max 32.1977135492, wins 10/27 reproduce the historical median1.014138201 and wins10/27 within declared1e-8 tolerance. Full268 median 1.06021591398, wins 79/268. Subset singleton-cycle ratio 0.861306362657; subset weighted-RMSE ratio 0.853081475574. The subset has3 rows/database; full counts differ. That fact alone is not a claim of bias. The panel is directionally representative of **row-level** deterioration: both medians exceed1 and most reactions worsen (17/27 versus189/268). It is not representative as a failure gate for the weighted global aggregates, which both decrease.

Measured aggregate evidence selects **Case 1**. A median row ratio cannot substitute for either mean singleton loss or weighted-RMSE metric. This is one selected variant per reaction, not the full augmentation-distribution expectation and not held-out/generalization evidence.

## Runtime and checks

Elapsed 158.345s; device NVIDIA GeForce RTX 5070 Ti; Torch peak allocated 1780240896 bytes; reserved 4253024256 bytes. Two small double models are kept, reactions evaluated sequentially, one group cached. Peaks describe the Torch allocator, not an independently measured resident-VRAM peak.

Passed: 268 unique once-each identities; exact nine DB counts; canonical manifest; all27 overlap variants; byte pins before/after; finite536 chemistry outputs; literal singleton factors; analytic cycle identity; direct/repository RMSE identity; model tensor hashes before/after; no parameter gradients; evaluation RNG restored. Checkpoint RNG/EMA/optimizer/scheduler state was not applied or rewritten. Scratch model construction precedes the evaluation RNG snapshot. Production source untouched. External driver syntax passes; no full Windows/WSL suites rerun, no committed Python tooling requiring Ruff. Final artifact/staged diff checks follow review.

## Classification and next experiment

**Case 1**. AE17 weighted-RMSE contribution changes by -18.0416501, larger in magnitude than the total weighted-RMSE reduction; improvements are concentrated, not universal. Per-DB aggregate-change contributions are recorded in JSON. Singleton ratio 0.812979354758, weighted-RMSE ratio 0.854224723016; classification tie band was predeclared1e-8 relative, and these changes are assessed against it. Source preservation and identity-wise comparison exclude permutation order as a confound; this does not prove a universal stochastic-variance mechanism or performance across all augmentation variants. The single next experiment is **short clean continuation with predefined full-268 chemistry checkpoints; not executed here**. It was not run.

## Independent review

Passed one independent GPT-6 Luna MAX curated review. No rerun, competing implementation, or additional agent review occurred. Reviewer validated provenance, equations, aggregates, Case 1, and restrained interpretation; no blockers. Review `C:/Dev/readWFN_share_ms/lap_chem_full268_eval_runs_20261004/review.md`, SHA `a2794f070ce12c845416b824d41a3f60ac4de3ce56ae9674ea3b8ac71ade9882`. Before any future continuation, predeclare its update budget, evaluation checkpoint schedule, and decision gate. Diagnostic syntax/self-checks, source-hash/artifact consistency, and staged diff checks passed.

## Twelve answers

1. **Manifest valid despite prefix difference?** Yes: separate frozen evaluation role, exact pinned identity/source hashes, and identity-wise variant equality.
2. **All27 variants preserved?** Yes, complete mapping verified.
3. **Exact functional?** The cleaned-count-adjusted weighted smoothed-MAE J_cycle above, for one cycle-0 variant realization.
4. **Explicit mean equals analytic?** Yes, rtol/atol1e-12 for both states and independent raw-row aggregation.
5. **Singleton-cycle objective improved?** Yes, ratio 0.812979354758.
6. **Weighted DB-RMSE improved?** Yes, ratio 0.854224723016.
7. **DB MAEs?** Improved: ABDE4, AE17, DBH76, PA8. Worsened: EA13, IP13, MGAE109, NCCE31, pTC13.
8. **DB RMSEs?** Improved: ABDE4, AE17, PA8. Worsened: DBH76, EA13, IP13, MGAE109, NCCE31, pTC13.
9. **Reactions improved?** 79/268.
10. **Old27 deterioration representative?** Yes for row-level deterioration: both medians exceed1 and both sets have a majority of worsened rows. No as an aggregate failure proxy: both weighted global aggregates improve.
11. **Classification?** Case 1.
12. **Single next experiment?** short clean continuation with predefined full-268 chemistry checkpoints; not executed here.

Both global chemistry aggregates improve; the 27-row panel was not a valid two-update failure gate, and the next experiment is a short clean continuation with predefined full-268 chemistry checkpoints.
