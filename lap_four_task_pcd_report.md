# Four-task PCD qualification

Branch `lap_full_vxc`, starting commit `10fc2a3026e67abef53a15d1b039e82254032206`. Final artifact commit is the commit containing this report. Upstream `e9afdbc9f4cb09343934eadb31dc72ea5ebac9b0` (MIT) was locally verified clean. Large drivers, gradients, logs and checkpoint remain outside Git; exact paths/SHA-256 are in results.

## Canonical contract and implementation

For normalized gradients u, stationarity of the convex PCD Lagrangian gives d=u_0+sum_j mu_j u_j. For an independent active subset S, solve G_SS mu_S=tau diag(G_SS)-G_S0; retain nonnegative multipliers, active complementarity and every inactive primal inequality. Enumerate empty and then increasing-size subsets in deterministic order. This is the pinned author implementation for arbitrary K, not convex-combination MGDA. Preserve EMA/bias correction/epsilon placement, primary-zero halt, zero-secondary omission, rank-deficient-subset skipping, infeasible primary fallback and final raw-primary-norm rescale. K=4 has seven nonempty subsets.

Existing aggregator/task/Armijo/protocol/checkpoint helpers were extended. Explicit task order is serialized; legacy three-task defaults remain. Protocol v5 and sampling v2 separate relchem/ae17 batches; old checkpoints are not reinterpreted. All-reduce each raw task gradient before PCD; trial scalars use unweighted rank mean. A bounded-memory chemistry batch reuses existing reaction loss, synchronizes the F64 shadow from stored main values and returns derivatives by exact names. F32 E/op derivatives are widened after autograd. No architecture, target, fchem factor, optimizer or S5 edit.

Production files added0, modified3; source lines added275, removed77; test lines added325. No new production module/framework or dependency. The single batch callable is required to avoid retaining8/17 reaction graphs and F32 leaf-gradient quantization.

## PCD validation

| Check | Result |
|---|---|
| K=3 parity | Exact100 synthetic sequential cases and2 cached real frozen cases; entire diagnostics/state/direction equality |
| K=4 independent-reference parity | 150 QPs and80 full EMA updates versus pinned author code |
| empty active set | pass |
| one active | pass |
| two active | pass |
| three active | pass |
| rank deficient | pass; dependent working subset skipped, author parity |
| infeasible fallback | pass; canonical primary fallback |
| primary zero | pass; EMA then halt |
| zero secondary | pass |
| DDP world_size2 | pass; global raw gradients/losses, state/direction/parameters and resume parity |
| save/resume | pass; order/count/hyperparameter/manifest fail closed |
| four-task Armijo | pass; finite, requested/realized descent, sufficient and strict actual decrease |

One read-only replay exposed missing mixed-gradient promotion in the shared Armijo path. It failed before any accepted update. The minimal fix widens existing lower-precision derivatives when a chemistry shadow is present; a regression exercises the actual mixed-precision Armijo route. The failed driver/protocol/receipt remain hash-bound. The replay then passed.

Windows 233 passed/4 skipped; WSL/PySCF/Gloo 235 passed/2 skipped. Ruff, compileall and diff checks passed. No unrelated lint fixes.

## Estimator qualification

Unbiasedness: E[sum_d(n_d/251) ell_Rd]=sum_d(n_d/251)(1/n_d)sum_r ell_r=J_rel. Database factors remain inside each existing ell; no duplicated n_d. All251/17 individual gradients were evaluated once, with scalar diagnostics retained;32 separately seeded stratified sums reused those derivatives without storing251 model-sized gradients. Full aggregates reproduce both pinned gold gradients exactly. This covers one frozen variant realization, not augmentation expectation.

| Estimator | Cost seconds | Median cos(full) | p10 cos | Fraction positive dot | Median norm ratio |
|---|---:|---:|---:|---:|---:|
| R1 | 2.395017 | 0.732086422 | -0.773576088 | 0.784860558 | 1.316879665 |
| R2 | 31.208188 | 0.846344244 | -0.439829045 | 0.812500000 | 1.679174484 |

R2 cost is a median standalone estimate from summed measured constituent gradient times, not32 independently timed batch runs. Dataset I/O and shared diagnostic overhead are separate. Selection was fixed before geometry/updates: R2 improves median/p10 alignment and positive-dot frequency, and narrows extreme norm ratios (R1 max54.97 versus R2 max3.81). Its p10 is still negative;6/32 draws oppose the full gradient, and median vector error is higher (1.343 vs1.216). This is a provisional per-sample-gated estimator, not a population-descent guarantee. First draw is fixed index0, full-gradient cosine0.812847; not cherry-picked.

| AE17 estimator | Cost seconds | Alignment with full g_AE | Practical? |
|---|---:|---:|---|
| singleton distribution | median 0.325899 | median 0.999852209, p10 0.996141183 | stochastic norm range0.0244-2.4400 |
| full17 | 5.366276 | 1; exact gold match | yes locally; peak allocated 896488448 bytes |

Select full17:17 identities are cheap enough and eliminate protected AE identity variance. Peak is a Torch allocation measure, not independently measured resident VRAM. No invented atom minibatch.

## Frozen predopt geometry and Armijo

Actual selected R2/full17 plus predeclared first immutable mRKS stream entry BeH2. Chemistry uses matched-F64 arithmetic on F32 source/checkpoint values widened, not restored native-double precision. E/operator keep their existing precision.

| Task | Raw norm | Normalized norm | g_i dot d |
|---|---:|---:|---:|
| relchem | 12.253324232 | 0.999999999967 | 115.845963997 |
| ae17 | 6992.42952006 | 1 | 12365.2207767 |
| exc | 3647.10056779 | 1 | 6002.41350487 |
| op | 0.27636846096 | 0.999999934537 | 0.085980303603 |

| cosine | relchem | ae17 | exc | op |
|---|---:|---:|---:|---:|
| relchem | 1.000000000 | -0.493806224 | -0.513829408 | -0.616354476 |
| ae17 | -0.493806224 | 1.000000000 | 0.995934513 | 0.954638731 |
| exc | -0.513829408 | 0.995934513 | 1.000000000 | 0.973721959 |
| op | -0.616354476 | 0.954638731 | 0.973721959 | 1.000000000 |

Active constraint: ['op']; multipliers {'ae17': 0.0, 'exc': 0.0, 'op': 0.6363545166411023}. All raw dots are strictly positive. This is stochastic-instance geometry; full g_rel versus full g_AE cosine remains the established -0.891844, not the table's sampled cosine.

Frozen replay accepted t=1.0 without backtracking, alpha0=6.632573669086685e-7, c=1e-4, rho=.5, cap20. Requested/realized norms 8.12710756598e-06/8.13047706154e-06, cosine 0.999033804256; unchanged fraction 0.598454372. All realized dots negative and all four actual losses decrease. Operator actual/predicted reduction ratio is 25.707886; this discrepancy is reported, not used to claim a precise linear model. Frozen replay was rolled back; no tentative EMA/model state initialized the pilot.

## Five-update clean qualification pilot

Fresh canonical predopt only; no theta2 or S5 weights loaded. Entire five-update manifest and task estimators were frozen before update0. Systems: BeH2,H2,BH,CH4,HPSi_iso2; same frozen cycle0 chemistry variants. No optimizer, scheduler, momentum, decay, OMEGA or phase curriculum. Cursor advances only on acceptance. All5 updates accepted t=1 with0 backtracks, requested/realized descent and strict componentwise Armijo. No accepted violation or nonfinite value.

Predopt metrics reuse pinned full268 values and unique15-system E/operator baseline values from the prior predopt panel. Repeated-system equality was checked; current frozen BeH2 matches exactly. Checkpoints2/5 recomputed all268 and every unique15 system. Old chemistry semantics are not used for optimization.

| Checkpoint | J_rel/S5 | nonAE RMSE/S5 | AE MAE/predopt | AE RMSE/predopt | E mean/predopt | Op mean/predopt |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 1.380502740650 | 1.392851804534 | 1.000000000000 | 1.000000000000 | 1.000000000000 | 1.000000000000 |
| 2 | 1.380417675815 | 1.392761312910 | 0.999542375337 | 0.999517397737 | 0.999588380088 | 0.999991146005 |
| 5 | 1.380385178071 | 1.392752868497 | 0.998166802459 | 0.998040833193 | 0.998295341700 | 1.000075831166 |

| Checkpoint | E median/p90/max ratios; improved | Op median/p90/max ratios; improved |
|---|---|---|
| 0 | 1.000000000000/1.000000000000/1.000000000000; 0/15 | 1.000000000000/1.000000000000/1.000000000000; 0/15 |
| 2 | 0.999620688172/0.999770363715/0.999815396987; 15/15 | 0.999999296021/1.000213685323/1.000297215644; 8/15 |
| 5 | 0.998458978796/0.999274066009/0.999517791797; 15/15 | 1.000046167799/1.000263945076/1.000367485678; 7/15 |

| Checkpoint | AE MAE/S5 | AE RMSE/S5 | Relative wins/predopt | AE wins/predopt |
|---|---:|---:|---:|---:|
| 0 | 3.067728212520 | 3.016436758144 | 0/251 | 0/17 |
| 2 | 3.066324344429 | 3.014981018938 | 199/251 | 17/17 |
| 5 | 3.062104460703 | 3.010527055372 | 162/251 | 17/17 |

**Pilot fails the operator guard.** At5, operator mean ratio1.000075831166 and median1.000046167799;7/15 improve. Relative J_rel 1.238999382946, nonAE DB-RMSE 46.317004400171, AE MAE/RMSE and E aggregate improve. The operator increases are small (+0.007583% mean,+0.004617% median), but no post-hoc tolerance, rounding or redefinition converts them into a pass. Their causal attribution/reproducibility is not established. All requested per-DB metrics/win counts, S5 reference values and checkpoint identities are in JSON. No held-out/generalization claim.

Pilot elapsed 1128.626s including evaluations; step times [61.725, 63.495, 52.746, 88.153, 77.235]. RTX5070Ti; F32 main with F64 chemistry; point chunk256/AO4096. Peak allocated/reserved 21997332480/31876710400 bytes exceed physical16GB, indicating likely Windows shared/oversubscribed behavior. Resident VRAM was not measured; no V100-fit/performance claim. Caller model/RNG restored after the bounded run; final checkpoint remains external and hash-linked.

## Next gate

Read-only operator-estimator sampling-transfer audit on the same frozen15-system cache: compare sampled-system operator gradients/directional products with the unique15-system aggregate at the pinned predopt and final pilot checkpoints; establish numerical reproducibility of the tiny aggregate changes. No further training or new MOO method.

No further training is justified by this pilot. Generic PCD/Armijo infrastructure is validated; the global operator guard is not. Before any production/full-data run, qualify the intended complete operator stream (currently15 of90 cached), augmentation, DDP/V100 memory and a longer checkpoint schedule. The future SCF spending gate requires both relative metrics reach S5, AE MAE/RMSE no worse than predopt with measurable progress reported, E/operator mean<=predopt and median<=1, at two consecutive future predeclared checkpoints. It is not reached; no Diet, SCF benchmark, Slurm or production job ran.

## Sixteen answers

1. Yes: general KKT construction and pinned author parity for K2-5.
2. Yes: exact sequential diagnostics/state/direction parity, including2 cached real cases.
3. Yes: author reference, all active-set cases and primal/dual/complementarity/stationarity tests.
4. R2 eight-DB stratification: better observed directional stability/tails; still imperfect, not a population guarantee.
5. Full17: exact protected mean/gradient, practical measured streamed cost.
6. R2 median cosine0.846344, p10-0.439829,26/32 positive; full17 cosine1. First R2 draw cosine0.812847.
7. The complete sampled four-by-four matrix appears above; it is not chemistry population geometry.
8. Only op at frozen tau.02; mu_op0.636354516641.
9. Yes on the frozen instance, and all5 accepted pilot instances.
10. Yes: frozen t1 and all5 direct pilot steps; realized descent and all-four sufficient/strict decrease.
11. Yes: both full251 J_rel and nonAE DB-RMSE improve.
12. Yes: full17 AE MAE/RMSE decrease.
13. E yes; operator no at5: mean and median guards fail.
14. No longer-run qualification: operator sampling-transfer guard must be resolved first.
15. Complete intended operator stream, augmentation, DDP/V100 memory/performance and predeclared longer checkpoints, plus the failed operator guard.
16. Read-only operator-estimator sampling-transfer audit on the same frozen15-system cache: compare sampled-system operator gradients/directional products with the unique15-system aggregate at the pinned predopt and final pilot checkpoints; establish numerical reproducibility of the tiny aggregate changes. No further training or new MOO method.

Independent review: PASS WITH SCOPE LIMIT. Review SHA-256 `f8a7485853f1af2f1e845bf46cb78c6021049e1872a5f2d100908d0caa5108d7`. Independent review confirms the mathematics and external helper-API pilot, with a repository-native CLI integration limitation: `train_lap_moo.py` still constructs three objectives and does not consume four-task `task_samples`. This task qualifies the shared four-task helpers and frozen diagnostic driver only. The normal CLI is not four-task capable; wiring and qualifying it remains required before any repository-native four-task run. No production-ready claim is made.

The four-task pilot ran but failed the global operator mean/median guard; the next experiment is a stochastic operator-estimator sampling-transfer audit.
