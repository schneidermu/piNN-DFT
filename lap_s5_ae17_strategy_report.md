# S5 baseline and AE17 optimization-design audit

## Identity, scope and plan

Branch `lap_full_vxc`, starting HEAD `7811f6755a2ca36903828b622560c26fc75911db`. Production source changes: **zero**. No optimization updates, training, S5 reproduction, PCD change, SCF, Diet-GMTKN, Slurm or V100 job occurred. Existing evaluators/store/model builder/named-gradient and dot/norm helpers were reused in external diagnostics. Caveman, Ponytail, to-spec and to-tickets applied directly; five dependency-aware tickets/spec/acceptance conditions are in protocol. Prior reports remain unchanged.

This is one frozen cycle-0 variant realization: 268 unique cleaned identities, including17 AE17 and251 non-AE17. Theta2 is a two-update diagnostic from a27-reaction catalog, not a full-Minnesota trajectory; it is only evaluated/reused here. Any future restart begins at clean predopt.

## S5 provenance

Selected checkpoint `C:\Dev\readWFN_share_ms\replay_trial_19_simple4_s5_two_step_40_10_500_snapshots_from280_gc_svelu_mirror\checkpoints\trial_19_selected.pt`, SHA `946d93f12656415288d4cb98b92183a203ad78fa13b57c8acaeef7ac5d1776de`. It is final epoch500 `PBE-LGxGc_6_32`, `pcPBELMLOptimizerV2GcSveluMirror`, model type `gc_svelu_mirror`, dropout0, 6layers/32width, Gx/Gc enabled. `load_state_dict(strict=True)` succeeds without dropped keys. Associated training state identifies the architecture/500epochs and has exactly the same weights. Its complete epoch schedule equals current `simple4_two_step_40_10`:75/40/10/15/7 Vxc plateaus across1-72/73-176/177-280/281-440/441-500, including matching chemistry/Exc settings.

The non-snapshot S5 run has **identical checkpoint bytes**; these are copies, not competing candidates. An unlabelled `C:/Dev/ML-DFT/piNN-DFT/train_models/trial_19_selected.pt` has different hash `b9da73dfa243db71b0f3883ba6f32e3d3e0e9a0eea27951bc4c971479c87dd4e` and lacks evidence tying it to the specified S5 schedule, so it was not selected. No performance-based checkpoint selection occurred. Provenance metadata hashes/candidate inventory are linked in results.

Historical source commit and training-corpus hashes are not embedded and remain unavailable; no equivalence to the current cleaned corpus is asserted. Seed41 is documented in the runner, not proven by the bare selected checkpoint. The current source hashes define the evaluated forward. S5 is a **training-distribution orientation baseline**, not validation data, oracle, or final scientific benchmark.

## Common chemistry evaluator and precision

Accepted evaluation-only manifest `f20225fc6d15e26d0729eb5ad33300841ffc15355608df6ee4dcae58add44691`, file SHA `b47a7271b9cbbe696e735f6f332345d59812fb4dbe87bebca32556104e82c070`. Same pinned group store, identities, suffixes, targets, HF and D3 corrections as7811f67. Old Lap rows were reused and byte-hash verified, not re-evaluated for baseline tables. S5 original N x9 tensors, including **tau and Laplacian columns**, were preserved. Legacy S5 descriptors use both tau and q; Lap architecture is tau-free. Both produce adaptive PBE constants through the same reaction integration/corrections and corrected singleton reducer. S5 eval mode with dropout0 introduces no dropout discrepancy.

Numerics are **matched-F64 arithmetic on stored checkpoint/source values widened to double**, not recovered native-double data or a historical bitwise replay. S5 predictions and targets were verified finite and identical in row/target provenance to reused Lap rows. No S5 operator score was constructed.

## Exact split semantics

Each existing singleton loss is `ell_r=(w_d f_d/m) sqrt(e_r^2+1e-20)`. Existing weighting remains intact. Define `J_all=sum_all ell/268`, `J_rel=sum_nonAE ell/251`, `J_AE=sum_AE ell/17`. All three model rows satisfy:

`J_all = (251/268) J_rel + (17/268) J_AE`

at rtol/atol1e-12. Eight-DB weighted RMSE simply excludes AE17 from `sum_d w_d RMSE_d`, with **no renormalization**. Scalar residuals, per-DB wins against predopt and S5, and non-AE row-ratio distributions/floor1e-6 are in JSON.

## Chemistry baselines

| Metric | Predopt | Lap theta2 | S5 | theta2/S5 |
|---|---:|---:|---:|---:|
| J_all | 2.74657101 | 2.232905527 | 1.357655859 | 1.64467712 |
| all-nine weighted DB-RMSE | 103.1119446 | 88.08077229 | 52.08311978 | 1.69115776 |
| J_rel (251) | 1.239104904 | 1.33735277 | 0.8975751135 | 1.48996195 |
| non-AE17 weighted DB-RMSE | 46.32029459 | 49.33077246 | 33.25572357 | 1.48337691 |
| non-AE17 overall MAE | 9.748406477 | 11.28460584 | 6.115935562 | 1.84511523 |
| non-AE17 overall RMSE | 13.96852272 | 16.46537676 | 8.888824169 | 1.85236837 |
| J_AE | 25.00386469 | 15.4554786 | 8.150612751 | 1.89623518 |
| AE17 MAE | 47.33669941 | 29.25993057 | 15.43053887 | 1.89623518 |
| AE17 RMSE | 56.79164998 | 38.74999983 | 18.82739621 | 2.05817094 |

Both predopt/S5 and theta2/S5 required ratios are in JSON. Values at/below1 mean reached that S5 **training-distribution** metric only. No final hard model-quality threshold is inferred.

## Per-database errors

| DB | n | Predopt MAE | Lap2 MAE | S5 MAE | Predopt RMSE | Lap2 RMSE | S5 RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| ABDE4 | 4 | 2.7195552 | 2.1544317 | 0.728567575 | 3.38793796 | 2.68539373 | 0.961736837 |
| AE17 (absolute atomic) | 17 | 47.3366994 | 29.2599306 | 15.4305389 | 56.79165 | 38.7499998 | 18.8273962 |
| DBH76 | 70 | 8.62676413 | 8.61945685 | 3.98108465 | 10.0028888 | 10.0381261 | 5.16685254 |
| EA13 | 11 | 2.47212828 | 2.7731909 | 3.04115575 | 3.10553842 | 3.67034809 | 3.73132106 |
| IP13 | 13 | 3.43613768 | 5.05581246 | 3.48032581 | 4.62090605 | 5.92512992 | 4.06162464 |
| MGAE109 | 104 | 15.8482429 | 19.3112981 | 10.2551161 | 19.8239679 | 23.9486568 | 12.7444571 |
| NCCE31 | 28 | 0.881692809 | 0.950926843 | 0.529314983 | 1.24825495 | 1.32349455 | 0.742871198 |
| PA8 | 8 | 1.47383017 | 1.40789754 | 1.56822663 | 1.72149687 | 1.69378549 | 2.15709807 |
| pTC13 | 13 | 5.81075507 | 5.99704342 | 6.22415189 | 6.81135582 | 7.02412187 | 7.05623536 |

Reaction errors are kcal/mol. All17 AE17 records were checked: one named atomic component each, coefficient+1. This confirms absolute atomic-energy supervision in the actual source, distinguished from the other relative-energy databases; no AE17 target/weight was altered. Structure receipt is externally hash-bound. Rows/wins are descriptive, not a separate hidden objective.

## Full-cycle predopt gradient decomposition

Only clean canonical Lap predopt was differentiated. Each exact F64 scalar was differentiated once with `autograd.grad`; graphs were released per reaction. Three independent running sums use1/268,1/251,1/17 coefficients. This computes the exact average-objective gradients for this frozen realization without storing268 individual parameter gradients. Per-DB means were accumulated from the same temporary row gradients at negligible additional forward cost. This is a linearity/decomposition check, not three independent full reruns.

| Quantity | Value |
|---|---:|
| norm g_all | 434.350230416 |
| norm g_rel | 11.0438338964 |
| norm g_AE | 6992.42952006 |
| norm (251/268) g_rel | 10.343292194 |
| norm (17/268) g_AE | 443.549633735 |
| weighted AE/rel norm ratio | 42.8828293174 |
| cos(g_rel,g_AE) | -0.891843997804 |
| cos(g_all,g_rel) | -0.886919723773 |
| cos(g_all,g_AE) | 0.999941982899 |
| cos(h_rel,h_AE) | -0.891843997804 |
| decomposition relative residual | 4.44253757388e-16 |
| AE signed projection share | 1.02112044432 |

Relative residual `norm(g_all-(251/268)g_rel-(17/268)g_AE)/max(norm(g_all),tiny)` passes the **predeclared1e-10** gate. Diagnostic norms/cosines and accumulation are F64. Gradient-enabled forwards agree with reused no-grad predopt scalars within1e-10. Weights/buffers unchanged, no `.grad` tensors attached, evaluation-process RNG restored. Aggregate gradient tensors remain external and hashed.

AE weighted contribution over mixed-gradient norm is 1.0211796902; its share of the sum of contribution norms is 0.977212043627. **Vector norms are not additive**: these are not literal percentages of a uniquely partitioned norm. The signed projection shares on g_all sum to1 and can exceed1 or become negative through cancellation. Error magnitude alone does not establish gradient dominance: the measured sensitivities, weights and cancellation geometry do. The residual derivative is e/sqrt(e^2+1e-20), approximately a sign away from zero; larger errors do not alone multiply singleton gradients.

## Strategy decision

Recommend **four tasks: relchem primary; AE17, Exc and operator secondary**. Theta2/predopt relative singleton ratio 1.07928938463 and non-AE weighted-RMSE ratio 1.06499263212, while AE MAE/RMSE ratios 0.618123589776/0.682318612807. Thus the prior mixed aggregate improvements conceal relative-chemistry degradation. Measured contribution ratio 42.8828293174 and cos(rel,AE) -0.891843997804 provide direct gradient evidence rather than an argument from large atomic errors. AE17 is **dominant and conflicting** at predopt: weighted ratio42.8828, cosine-0.891844. Because cos(g_all,g_rel)=-0.886920, the ideal negative mixed-chemistry gradient has positive relative-objective directional product 4254.45794807; this is a first-order diagnostic, not an executed update or a claim about every PCD direction. This evidence is local to predopt and one realization; alignment may evolve.

The S5 gaps differ between relative and absolute supervision and provide separate monitoring lines. AE17 remains an explicit secondary objective, not discarded or arbitrarily downweighted. No custom coefficient, clipping, or loss was introduced.

Current `_solve_pcd_qp` requires a3x3 Gram and is **unchanged**. Next implementation must use the published general construction for order `relchem,ae17,exc,op`; three secondary constraints yield the empty active set plus at most7 nonempty subsets. Preserve canonical EMA, primary coefficient1, raw-primary norm rescale, infeasibility policy, task order and global-before-PCD DDP semantics; validate official parity/KKT, resume, and realized tested Armijo steps before any tiny clean restart. No four-task method was implemented here. Task separation clarifies priority but does not prove a feasible four-task common-descent step; the next validation must fail closed when direction or realized Armijo progress is unsuitable.

Optional15-system E/operator gradient geometry and S5 integrated Exc were skipped to keep the primary audit focused; no AO cache was loaded.

## Proposed transition-to-SCF spending gate

Before a future run, fix its tiny update budget/evaluation schedule and a new immutable task-sampling manifest. Separate251-relative and17-atomic marginals must estimate the declared J_rel/J_AE means with unchanged database factors, not silently reuse the old three-task sampler. At **two consecutive predefined checkpoints**, require: (1) relative singleton and eight-DB weighted-RMSE ratios to S5 <=1; (2) AE17 MAE and RMSE no worse than clean predopt, corresponding to S5-relative ceilings 3.067728213/3.016436758; (3) integrated Exc and operator **mean and median loss ratios <=1** against canonical predopt on the same immutable15-system diagnostic panel, all finite. Report all four S5-relative chemistry ratios throughout; do not stop on mixed J_all or median row ratio alone.

This is a proposed operational **compute-spending gate**, not proof of generalization or a production success claim. AE must additionally show progress as a training objective; the spending gate itself allows a no-worse baseline guardrail. Before applying this future gate, predeclare an operational AE17 progress threshold against the starting checkpoint; no threshold was selected from these frozen results. S5 supplies no operator benchmark. Crossing this gate would justify the first external SCF validation, not its outcome. No such validation ran now.

## Runtime, evidence and review

S5 forward 98.717s; gradient stage 820.452s; total 919.215s. Device NVIDIA GeForce RTX 5070 Ti; Torch peaks allocated 21472044032 and reserved 31935430656 bytes. Allocator peaks are not independent resident-VRAM measurements. Both exceed physical RTX memory (~16GB), indicating likely Windows shared/oversubscribed behavior; V100 memory fit is unverified. Bounded-memory here means one reaction graph, not a small peak. Memory-safe chemistry chunking must be verified separately before any cluster use. Drivers/raw S5 rows/aggregate gradients/protocol/receipts are external and hash-bound in results. Production LOC added0; committed Python tooling0, so Ruff/full production suites not required. Syntax/self-checks and final artifact/diff checks apply. One independent Luna MAX review passed with no blockers; review receipt and SHA are bound in both JSON artifacts. Its nonblocking request to predeclare an AE17 progress threshold is recorded above. No rerun or competing implementation occurred.

## Thirteen answers

1. **Which S5 and why?** Exact path/hash and metadata above; final500 state, exact S5 schedule, strict6x32GcMirror load and duplicate bytes establish identity.
2. **S5 relative baseline?** J_rel 0.897575113547; eight-DB weighted RMSE 33.255723573.
3. **S5 AE17?** MAE 15.430538865; RMSE 18.8273962061.
4. **Lap2 relative gap?** Singleton ratio 1.48996195343; weighted-RMSE ratio 1.48337690946 versusS5.
5. **Lap2 AE gap?** MAE/RMSE ratios 1.89623517518/2.05817094444 versusS5.
6. **AE contribution?** Weighted AE/rel norm ratio 42.8828293174; signed mixed-direction projection share 1.02112044432, with nonadditive-norm caveat.
7. **Cosine rel/AE?** -0.891843997804.
8. **Dominates/conflicts?** Yes: dominant weighted contribution42.8828x and strongly conflicting cosine-0.891844; mixed g_all aligns with AE but opposes relative chemistry.
9. **Separate?** Yes; mixing obscures the relative target and the gradient evidence quantifies the coupling.
10. **Primary?** Relative chemistry, aligned with intended external relative-energy validation; AE remains protected secondary.
11. **Task count?** Four, design only.
12. **SCF trigger?** The conjunctive two-checkpoint S5-relative/AE/Exc/operator spending gate above; not yet achieved or tested.
13. **Single next experiment?** Generalize canonical PCD to four tasks (relchem primary; ae17, exc, op secondary), validate published KKT/parity/EMA/DDP/resume and exact realized-step Armijo semantics, then a tiny clean full-268 restart from canonical predopt; not implemented or run here.

S5 is established as the frozen chemistry baseline; AE17 should be a separate objective, so the next experiment is a validated four-task PCD implementation with relchem primary.
