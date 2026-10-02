# Clean one-stage Lap MOO local study

**Status: Gate 3 is complete for four 100-update cosine runs, their matched 27-identity training panel, and three-system CPU SCFs.** The diagnostic shortlist is Nash-MTL primary for the optimizer comparison, IMTL-G as an operator-priority boundary case, and fixed scalarization as reference; no overall winner is claimed. The earlier F1/F2 runs are invalid for cosine selection because their scheduler was not passed into the update function. The corrected main runs used RAdamW and stopped at cursor 100 on a 150-update horizon; those historical v1 hashes remain unchanged. Prospective v2 metadata binds the AO-cache chunk size and preserves the exact frozen sampling-manifest identity. Two-step gates, all three matched cursor-25 optimizer panels, and the selected checkpoints' SCF checks are complete. No overall method or optimizer winner is claimed.

## Code and source identity

- Repository: `lap_full_vxc`, base HEAD `61837e4346ecaf46247572df24e3f2b757a7b415`.
- New path: `train_models/moo_aggregators.py`, `train_models/lap_moo_protocol.py`,
  `train_models/lap_moo_training.py`, and `train_models/train_lap_moo.py`.
- The isolated independent Gate 2 test file is
  `train_models/test_moo_aggregators_independent.py`.
- Historical S5 remains unchanged. The new protocol rejects keys containing
  `omega`, `s5_`, or `phase_schedule`; each update constructs all three tasks.

## Literature definitions and fixed defaults

| Method | Update rule used by the new path | Fixed solver/default |
|---|---|---|
| Fixed scalarization | `d = (1/K) Σ λᵢ gᵢ`; calibration weights have unit geometric mean and remain frozen | `λᵢ ∝ 1/Gᵢ`, where `Gᵢ` is the initial raw-gradient median; calibrated weights and report hash are recorded in the main-training protocol |
| IMTL-G | `d = Gα`, `Σαᵢ=1`, with equal `d·uᵢ` for normalized task directions `uᵢ=gᵢ/||gᵢ||`; negative coefficients are allowed | Float64 SVD, minimum-norm solution for consistent rank deficiency, no ridge; zero task norm fails closed; exact cancellation returns zero with an explicit degenerate status |
| CAGrad | `ḡ=(1/K)Σgᵢ`; `w* = argmin_{w∈simplex} (Gw)·ḡ + c||ḡ||||Gw||`; `d=ḡ+c||ḡ||Gw*/||Gw*||` | `c=0.4`, paper-unscaled update; SLSQP simplex solve with KKT check; undefined zero `Gw*` fails closed |
| Nash-MTL | `d=Gα`, where `GᵀGα=1/α`, the stationary condition of the Nash bargaining product | Strictly convex potential `0.5 αᵀKα−Σlogαᵢ`; damped Newton, update every step, tolerance `1e-10`, no Gram ridge; warm-started from prior α; infeasible zero/opposite utilities fail closed |

The IMTL-G definition follows [Towards Impartial Multi-task Learning](https://openreview.net/pdf?id=IMPnRXEWpvr)
and its reference implementation; OpenReview required browser verification, so
the exact equation was cross-checked against the [official Nash-MTL reference
implementation's IMTL-G code](https://github.com/AvivNavon/nash-mtl/blob/main/methods/weight_methods.py).
CAGrad follows [Conflict-Averse Gradient Descent for Multi-task Learning,
NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/9d27fdf2477ffbff837d73ef7ae23db9-Paper.pdf).
The code's paper-unscaled convention deliberately omits the extra `/ (1+c)`
scaling in the authors' illustrative two-task toy script. Nash-MTL follows
[Multi-Task Learning as a Bargaining Game, ICML 2022](https://proceedings.mlr.press/v162/navon22a/navon22a.pdf)
and the [authors' reference implementation](https://github.com/AvivNavon/nash-mtl/blob/main/methods/weight_methods.py).

Fixed weights are deliberately scale-sensitive after their one-time
calibration. IMTL-G equalizes projections onto unit task gradients and allows
negative coefficients; independent positive rescaling preserves only a signed
ray, which can reverse orientation. CAGrad solves a simplex problem around the
mean gradient and remains sensitive to task rescaling. Nash uses positive
bargaining weights and a strict Newton solve; zero/opposite utilities can be
infeasible, and this study adds neither Gram regularization nor fallback.

## Objective and update protocol

Every update computes raw `g_chem`, `g_exc`, and `g_op` from one Minnesota
reaction and one full mRKS system. The same mRKS system supplies the energy and
operator objectives. Chemistry retains the established Minnesota reaction,
stoichiometry, database weighting, dispersion, and kcal/mol conventions.
`L_E` retains the legacy mRKS `E_xc` target and one-addition dispersion
treatment. `L_V` is the validated h-free variational AO XC operator loss,
orthogonalized by the overlap and divided by `n_AO`.

For DDP, each raw task gradient is all-reduced and averaged first; the
nonlinear aggregator then runs on the same global task gradients at every rank.
This avoids changing the MOO definition by aggregating locally and averaging
afterward. The new main path contains no `OMEGA`, S5 phases, epoch-dependent
task weights, or task-specific clipping.

## Implemented main-training contract

The executable path is [train_lap_moo.py](train_models/train_lap_moo.py). It
loads the verified PBE predopt checkpoint with architecture
`pcPBELMLOptimizerV2Lap-v1` (`num_layers=6`, `h_dim=32`, zero dropout,
`use_g_x=true`, `use_g_c=true`; 9,446 trainable parameters) in float32. Its
one-stage run protocol is built by `make_protocol_metadata()` and rejected by
`validate_protocol_metadata()` if the method, task order, source hashes,
operator version, fixed weights, or objective metadata disagree. Four real
LR-range trial metadata files—one per method—passed that validator and carry
`one_stage_main_training=true`; each trial is a partial two-update run against
a 150-update target, not a completed or selected main run. Their paths and
hashes are recorded in [lap_moo_protocol.json](lap_moo_protocol.json).

The fixed method uses the survey calibration weights
`[0.1246674180, 0.0149441680, 536.7540039]`, with unit geometric mean. IMTL-G
uses its raw equal-unit-projection solve. CAGrad is fixed to `c=0.4` and the
paper-unscaled update. Nash-MTL uses the positive Newton-potential solver with
100 iterations maximum, `1e-10` tolerance, and coefficients recomputed every
update. The intended path uses RAdamW (`betas=(0.9, 0.999)`, `eps=1e-8`, decoupled
weight decay `0.01` on linear weights only; bias and norm weights are excluded)
and one cosine schedule down to 10% of the base rate. Runtime audit found the
pre-fix CLI did not pass its scheduler to `train_moo_update`, so F1/F2 actually
used a constant rate even though run metadata declared cosine. The corrected
source now passes the scheduler and focused regression tests pass. Corrected
cosine screening completed the 100-update horizon segment for all four
methods; the matched 27-pair panel and H2/BeH2/CO SCF smokes also completed.
Every method stayed below the 100× panel-stability maximum. These runs used
historical v1 metadata and retain their original hashes. The corrected Nash
solver passes the captured CUDA regression under unchanged strict tolerance.
The diagnostic shortlist and limitations are recorded in the Gate 3 section.
The pre-fix training-source hashes used by F1/F2 were `train_lap_moo.py`
`427ec1e655c593ad31052f2c5ccdb0fe27ae3774e7b5db3bf251e8e93eb1cefa` and
`lap_moo_training.py`
`cef616238faaaedf56c2d9b5e3d6260c2a853fd055530061f30b13c6bb26bcdc`. The
corrected-source hashes are `train_lap_moo.py`
`2c519563eaf74ffedef6237ded156bee3a16afcecc9a388ac68da645d109c25e` and
`lap_moo_training.py`
`2e70d39c7bc1284236213c6700d84bfc13fd50f5eef14fcbc168b1290244f77c`.
The cursor-100 fixed run's execution provenance binds its training script to
the later hash `1204e2c255848ae0d97eeb2a42ec29d5553e866fb512181a07fc8002d2d0fb0c`;
the scheduler fix remains present in this version.
Validated trial chunk sizes are 256 model
points and 4096 AO-cache rows. Prospective v2 run metadata binds both chunk
sizes. AO cache and model inputs are float32; reference operators and overlaps
remain float64.

The available data sources are the all-268-group, 2,144-variant Minnesota
store, the hash-verified 90-system central operator corpus, and a current
hash-bound AO cache for 15 systems. The trainer's full-catalog mode covers all
268 reaction groups and every system present in the supplied AO cache; the
existing cache does not provide AO factors for all 90 systems. The 27-reaction,
15-system stream hash
`550f22488df27b9e1326905bd3569fa7a84671e90b428ae3be190f1e409bf928` is the
calibration/LR-pilot stream. The final main-run stream used the same 27-reaction/15-system manifest hash
`550f22488df27b9e1326905bd3569fa7a84671e90b428ae3be190f1e409bf928`; the
sample identities are retained in the external hash-bound manifest. The full
source, cache, checkpoint, dispersion, fixed-calibration, and run protocol
hashes are recorded in [lap_moo_protocol.json](lap_moo_protocol.json).

## Data panel and immutable source identities

| Artifact | Identity / content | SHA-256 or size |
|---|---|---|
| Minnesota source | `C:\Dev\ML-DFT\piNN-DFT\train_models\data_train_grouped.pickle`; all 268 reaction groups | `a888609de0807356dd6eadee288fcaf0caa0532426df2e1c2f8f5b1c25fef444`; 13,462,405,299 bytes |
| Indexed Minnesota store | External `C:\Dev\readWFN_share_ms\lap_moo_runs_20261001\mn_group_store_268`; 268 groups, all 2,144 variants | Manifest `ec254952f51d854d8b23c01ff3d316d3b4fd756b58287c637d3f07ed8c385d2a`; 13,462,704,378 total group-file bytes |
| Minnesota calibration panel | 27 reactions, three from each of nine databases, all eight suffix variants retained | Panel definition `f98c112344dc52df401fb5eff50fd8ce609cc9c8f69085223cc9c0dee9c66131`; 27×8 artifact `75c0753d3b173bbdc66ae2195291e61e506a42b35b084de966b8707f0820cb1f` |
| Central operator corpus | External `C:\Dev\readWFN_share_ms\lap_operator_runs_20261001\all90`; h-free `lap-operator-central-ao-noh-v1` | Manifest `7005cd869ea8be9636b03f385e7069f4e9023c9582fe7defc5437a7d6609b887`; 90 systems, 8,271,091 centers |
| mRKS AO cache | External `C:\Dev\readWFN_share_ms\lap_moo_runs_20261001\mrks_15system_ao_cache`; 15 fixed systems | Manifest `60631c23d1683dcf7855ef5897464994addb990c1c8cf33ea2572adbf532a00a`; 3,175,900,181 bytes |
| Panel definition | External `C:\Dev\readWFN_share_ms\lap_moo_runs_20261001\panel_definition.json`; 27 Minnesota reactions across all nine DBs, 15 mRKS systems | SHA-256 `f98c112344dc52df401fb5eff50fd8ce609cc9c8f69085223cc9c0dee9c66131` |
| Combined preparation manifest | External `C:\Dev\readWFN_share_ms\lap_moo_runs_20261001\panel_manifest.json`; binds the Minnesota store, selected panel, central corpus, and AO cache | SHA-256 `2b989ba3ce09b5707561e50fc26b227a83417ff32d33de6fa38b2bc111b839c4`; 4,288 bytes |
| Initial checkpoint | Canonical two-epoch PBE predopt, seed 41, learning rate 0.01 | SHA-256 `ed4ba8231d93c376ce5aa8fc81e0670f6a5a44c95c2a552651b12d6a1d4b63f8` |

The Minnesota panel spans ABDE4, AE17, DBH76, EA13, IP13, MGAE109, NCCE31,
PA8, and pTC13. The fixed mRKS panel is H2, HLi, BH, BeH2, H2O, CH4, N2, CO,
CH2O, C2H2_iso2, H4Si, AlBeH, ClH, ClHS, and HPSi_iso2 (60–248 AOs; maximum
atomic number 17). The full 268-group store is available for training; the
27-reaction subset is for calibration/diagnostics only.

## Gate 2 synthetic algorithm review

The independent CPU suite checked the four formulas against analytic vector
references, including fixed weights, raw IMTL-G coefficients and projection
equalization, identical and orthogonal tasks, a conflicting CAGrad active-face
case, positive task rescaling with both signs of the IMTL-G scale factor,
singular/near-singular Gram matrices, zero and opposite gradients, Nash
stationarity, and deterministic structured gradients with unused `None`
entries. A three-dimensional Gram realization of a measured survey row
reproduces the IMTL-G direction reversal under a positive 1000× operator-task
rescaling.

**Result: PASS.** The independent aggregator review passed 27 tests on Windows and WSL. After
the scheduler and Nash fixes, the focused training suite passed 12 tests with
one Linux-only skip on Windows and 11 tests with two CUDA-only skips on WSL;
CPU two-rank Gloo passed. Later full pre-stream-split checks were Windows new
MOO 75 passed/1 skipped, WSL MOO/operator/SCF 88 passed/2 CUDA-only skips,
and historical Windows 110/1. After the frozen-stream version split,
`test_lap_moo_training.py` passed 14 tests/1 CUDA-only skip on Windows. Final
end-to-end and cross-platform v2 verification is complete: the combined Windows
suite passed 185 tests with 4 skips, the visible-CUDA RNG test passed, and the
WSL suite passed 93 tests with 2 CUDA-only skips. Commands and environments are
recorded in `lap_moo_protocol.json`.

## Paired raw-gradient survey

The pre-training survey paired 27 distinct Minnesota reactions (three from
each of the nine databases) with a deterministic cycle through all 15 selected
mRKS systems. It used the same frozen checkpoint for all objectives, one
seed-selected augmentation per reaction, float32 model gradients on CUDA, and
zero optimizer updates. All four aggregators returned finite diagnostics on
all 27 samples. Runtime was 504.56 seconds. The raw report SHA-256 is
`edb7a786ad013d406307cfc326f90c2a1cb23dbf4ed70f6fa41b5a7490d0c981`; the raw
float64-vector snapshot SHA-256 is
`cc7c2f0de49044dc2bed98cbd279ea871fd41aca69dc42b14cab978cdb97a2b`. Both
remain outside the repository. Full update identities and per-database and
per-system quantiles are retained in [lap_moo_results.json](lap_moo_results.json)
and the external hash-bound summary.

| Task | Median raw gradient norm | Ratio to operator median |
|---|---:|---:|
| Chemistry | 1,762.87 | 4,305.5× |
| mRKS energy | 14,706.21 | 35,917.3× |
| mRKS operator | 0.40945 | 1× |

The frozen unit-geometric-mean fixed weights are `chem=0.1246674180`,
`exc=0.0149441680`, and `op=536.7540039`. Thus calibration equalizes median
gradient-norm contributions by construction. It does not ensure equal task
progress at every paired sample.

The raw gradient cosines show the principal conflict: chemistry/energy had
median `−0.159` (negative in 17/27 samples), chemistry/operator median `−0.339`
(negative in 17/27), while energy/operator had median `+0.966` and was positive
in all 27. The sample size is three reactions per database, so these are panel
descriptions rather than population estimates.

| Method | Median effective coefficients on raw gradients (chem / exc / op) | Negative `d·gᵢ` counts (chem / exc / op) | Median cosine of `d` with each task gradient (chem / exc / op) |
|---|---|---|---|
| Fixed | `0.04156 / 0.004981 / 178.918` | `12 / 2 / 2` | `0.027 / 0.947 / 0.933` |
| IMTL-G | `0.0000977 / −0.0000207 / 0.999956` | `10 / 10 / 10` | `0.355 / 0.355 / 0.355` |
| CAGrad | `0.6993 / 0.3333 / 2103.46` | `10 / 1 / 0` | `0.001 / 0.974 / 0.936` |
| Nash-MTL | `0.001041 / 0.0000900 / 2.16292` | `0 / 0 / 0` | `0.450 / 0.735 / 0.637` |

`d·gᵢ` is the unnormalized inner product between the combined direction and a
raw task gradient. A negative value predicts an adverse first-order change
under a direct negative-gradient SGD step; Adam's transformed parameter step
can differ. IMTL-G's three unit-gradient projections are equal by definition,
and the common projection was negative for all three tasks in 10/27 samples.
An independent implementation of the official IMTL-G equations reproduced
these common-ascent cases (maximum absolute direction difference `1.65e-8`),
so this is canonical algorithm behavior, not an implementation defect. A
measured case (`NCCE31`, reaction 0, `level2_mura`, paired with `BH`) has
coefficients `[-0.00465214, -0.000241609, 1.00489375]` and common projection
`−0.200846`. IMTL-G
is not positively direction-invariant under arbitrary independent positive
task rescaling: if `gᵢ′=kᵢgᵢ`, then `d′=d/S`, where
`S=Σᵢ αᵢ/kᵢ`; this proves collinearity only, because `S` may have either
sign, and orientation is preserved only for `S>0`. For survey update 2
(`NCCE31`, reaction 0, `level2_mura`, `BH`), multiplying only the operator
gradient by 1000 gives `S=−0.00388886` and reverses the direction to about
`−257.145 d`. The CAGrad coefficients above are effective coefficients on
unscaled task gradients, not its simplex solver weights. All methods' solver
status was successful on all 27 rows. These findings expose substantial
task-scale disparity and chemistry conflict but do not rank optimization
outcomes.

The full run identities, loss and gradient-norm quantiles, raw task cosines,
method coefficients, `d·gᵢ` values, normalized projections, joint/task
cosines, and solver diagnostics by database and system are in
[lap_moo_results.json](lap_moo_results.json). Protocol and external source
hashes are in [lap_moo_protocol.json](lap_moo_protocol.json). The external
raw JSON and NPZ are authoritative for per-sample values.

## F1/F2 execution audit: constant-rate exploratory runs

The F1 screen and F2 extension are preserved outside Git under
`C:\Dev\readWFN_share_ms\lap_moo_runs_20261001\f1_20261002` and
`...\f2_20261002`. F1 used 25 updates per candidate; F2 resumed selected
checkpoints from update 25 through update 100 on the same manifest. The logged
learning rate stayed constant for every row, contradicting the cosine schedule
in the metadata. The cause was a missing scheduler argument in the pre-fix CLI
call to `train_moo_update`. The corrected source now passes the scheduler;
F1/F2 are explicitly excluded from method/LR selection.

F1 lineage and forward-only 27-pair loss ratios (candidate checkpoint loss /
predopt loss) are retained for audit. These constant-rate ratios are not
comparative evidence for the intended protocol.

| Candidate | Constant rate | Median chem / E / V loss ratio | Maximum chem ratio | Outcome |
|---|---:|---:|---:|---|
| Fixed | `1e-6` | `1.432 / 3.066 / 0.899` | `89.16` | 25-update checkpoint; full-panel diagnostic completed |
| IMTL-G | `1e-2` | `1.791 / 3.195 / 0.716` | `93.11` | 25-update checkpoint; full-panel diagnostic completed |
| Nash-MTL | `1e-3` | `1.006 / 0.386 / 0.905` | `33.96` | 25-update checkpoint; later strict solver failure |
| CAGrad retry | `1e-7` | `1.450 / 3.396 / 0.891` | `97.17` | 25-update checkpoint; full-panel diagnostic completed |
| CAGrad initial | `1e-6` | `6.763 / 39.962 / 4.598` | `916.21` | Failed full-panel stability gate; retained as failed attempt |

The exact F1 checkpoint SHA-256 lineage is: fixed `493123a69930a0f3c2e9415b245dc3929d56f29e7d761f20abf3e088f2b18aa1`; IMTL-G `6997cf7403027cac88f7e598c487e2cdf3f46d59988304be4bba4df854dff093`; CAGrad `1e-6` failed-panel checkpoint `fb7aafb7064e583eea9fb1a0f83f5d70cd723e684a738b59ffab8c614f0249ca`; CAGrad `1e-7` retry `00b042033b69a8fd5f281a5b8a1613d11753a55deda0e7660ff48727f735b8d5`; Nash `1e-2` failed-solver checkpoint `d5939a88bb54f39a705b58ab63b735a337182f740bbc53ef465e530efbd54c20`; and Nash `1e-3` retry `e80025e91c314920caab526dce1970e00adda88b1fae55afa877f31652108be7`.

Nash-MTL at `1e-2` failed closed at F1 update 8: the 100-iteration potential
solver residual was `9.621e-10`, above the unchanged `1e-10` threshold. The
The old constant-rate `1e-3` retry completed its F1 screen, then failed at attempted update 27 with
residual `6.004e-8`; the saved checkpoint stayed at the F1 retry checkpoint
`e80025e9...`. No fallback or tolerance relaxation was used.
That old failed update paired PA8 reaction 0 (`level3_gauss_chebyshev`) with
AlBeH. Its gradients and Gram matrix were not saved, so that failure's cause is
unclassified. The F2 Nash `latest.pt` and run summary are stale and
match the F1 cursor-25 checkpoint even though the log reaches update indices
0–26; neither represents a post-failure checkpoint.

F2 completed 100 logged updates for fixed (`1e-6`), CAGrad (`1e-7`), and
IMTL-G (`1e-2`), resuming their F1 checkpoints. Their checkpoint SHA-256
values are fixed `856836f75018155d426da60c9347c5488988435f925559249af55a344636384c`,
CAGrad `09dfe0fe67efc7fc6fc9d0b17ddce5c05561d2c617fd6722d9f02b8738c8d301`,
and IMTL-G `837e8277de385b70b745a8347ebbb5c659ba473671e15bdbf8084063873ba048`.
Nash did not advance past its F1 checkpoint `e80025e91c314920caab526dce1970e00adda88b1fae55afa877f31652108be7`. The F2 run index SHA-256 is
`972e6c7516a7a4662a4d575b467804154588f5cd8b0dc0278f5857005b3f2e3e`.
The F1 index SHA-256 is
`4dc08bed6674bce7c8940663f228b06285fa25b27a88de4daa9661185ab34602`; its
full-panel diagnostic SHA-256 is
`2166b9a28300acda801e1a807d98f3b2203ff1e2561dfb4bc06e16a0556b6d5f`, and
the CAGrad retry diagnostic SHA-256 is
`a66452548e2ba2b87cf52648668894bcf543078140bf538d629fe147a55ceb74`.

The F2 logs record the dot product of each raw task gradient with the
optimizer-applied parameter delta. Negative values are local first-order
decreases; positive values are local increases. Among the 100 logged updates,
the negative/positive counts (chem / E / V) were fixed `67/33, 81/19, 53/47`,
CAGrad `73/27, 89/11, 73/27`, and IMTL-G `48/52, 66/34, 60/40`. These are
constant-rate diagnostics only: they do not imply a common descent direction,
Pareto progress, or comparative quality.

The runner recorded median step times of 20.47 s (fixed), 16.19 s (CAGrad),
and 15.42 s (IMTL-G) across the 100 logged rows. Its peak-reserved memory
telemetry exceeds the same rows' reported device total, so it is not accepted
as capacity evidence. Runtime and VRAM need a clean measurement in the corrected
run.

## Corrected-cosine screening and Gate 3

The valid main-training comparison is four matched one-stage runs from the same
PBE predopt checkpoint (`ed4ba823…b63f8`), seed 41, and sampling manifest
(`550f2248…928`). The runner computed the same chemistry, integrated-energy,
and AO-operator objectives at every update. It used RAdamW, float32 training,
and a cosine schedule to 10% across a 150-update horizon; all four stored
checkpoints stop at cursor 100 (`run_complete=false`). Each run contains 100
updates, not 150. The first 25-update screens were carried forward into the
same-schedule continuations; the 27-row panel evaluates the same training
identities for each method. No external held-out set was used.

| Method | Base LR | Cursor-100 checkpoint SHA-256 | Update log SHA-256 |
|---|---:|---|---|
| Fixed scalarization | `1e-6` | `6b7bcf06635c7dd6be2d79978488748f13c2354d4f4f10ac165eaa494d114b1a` | `1d758271b6be07e857bbb85098d5a4318db9bfc37130c1e7eb7881b608ac5e2c` |
| CAGrad | `1e-7` | `e06e1c765b2280c93861e0661a8fe096181e5daebd2742044ab87c36db3b2f7d` | `34abb6e227e8e1bb824f1b179f7190f7a0ad46b525250c9f270a90a14c2c228f` |
| IMTL-G | `1e-2` | `3790b6bb3c2b2698349597cda3307b7c5a27fe3598e1d19fd3d44a49e295f00f` | `a0363ddff773e9c3abbd57ff68b2e76c581cc6474f28915ecb75ef575036ec35` |
| Nash-MTL | `1e-2` | `954fdcc149f16f45bf4381d51a6e309face78c07e7500bd962c798af135a8b28` | `1313115daa3bfbc24449359f76a969e4b03a56f2f11a675828d24c44894945e6` |

The final paired evaluation is
`C:\Dev\readWFN_share_ms\lap_moo_runs_20261001\f2_cosine_20261002\final_panel_cosine_cursor100.json`, SHA-256
`352a322c79fc80d3399994a5fb77d606b7a43ba78e7c0d5bde47e93ea0ab659a`.
It took 895.83 s on the RTX 5070 Ti. Ratios below compare each trained model
with predopt on the same sampled identity; they are diagnostic training errors,
not validation generalization.

| Method | Chemistry median / p90 / max | Integrated E median / p90 / max | Operator median / p90 / max |
|---|---:|---:|---:|
| Fixed | 1.432 / 4.972 / 88.981 | 3.057 / 3.461 / 3.495 | 0.900 / 0.930 / 0.941 |
| CAGrad | 1.450 / 5.418 / 97.114 | 3.394 / 3.829 / 3.874 | 0.891 / 0.926 / 0.937 |
| IMTL-G | 3.493 / 13.319 / 97.850 | 5.001 / 6.169 / 6.365 | 0.506 / 0.735 / 0.832 |
| Nash-MTL | 2.537 / 9.902 / 49.827 | 0.880 / 2.385 / 2.447 | 0.654 / 0.868 / 0.913 |

All candidates remained below the prespecified 100× maximum stability screen.
That is a stability check, not a quality threshold. On the paired comparisons
against fixed, Nash has lower E error on 27/27 rows and lower operator error
on 25/27, while chemistry is lower on only 8/27; Nash dominates fixed on 7
rows and has mixed outcomes on 20. IMTL has lower operator error on 27/27 but
higher E on 27/27 and higher chemistry error on 25/27. CAGrad's median
chemistry/E/operator ratios relative to fixed are `1.022 / 1.112 / 0.991`;
its small operator gain comes with worse E on all 27 rows and no pointwise
dominance. Each method's median three-objective vector is nondominated. No
scalar score was constructed and no method is a demonstrated overall winner.

The diagnostic shortlist is Nash-MTL as the primary candidate for the next
optimizer diagnostic, IMTL-G as a secondary boundary case only if operator
error receives priority, and fixed scalarization as the required reference
control. CAGrad is retained in the record but not shortlisted for the primary
comparison. This is a screening choice, not a selection claim: the evidence is
one seed and one training panel. Exact by-database, by-system, paired, and
worst-row values are preserved in `lap_moo_results.json` and the external raw
panel.

### Update geometry

The cursor-100 logs contain per-update raw gradients, aggregate directions,
optimizer parameter deltas, coefficients, and solver diagnostics. The median
raw gradient cosines (chem/E, chem/operator, E/operator) were fixed
`+0.745, −0.273, −0.484`; CAGrad `+0.745, −0.251, −0.470`; IMTL-G
`+0.760, −0.598, −0.951`; Nash-MTL `+0.655, −0.734, −0.950`. Thus, in this
sample the initial negative chem/E median and positive E/operator median did
not persist. The pattern is descriptive and has per-system exceptions.

The fraction of RAdamW parameter deltas with a negative first-order dot
against each task's raw gradient (chem/E/operator) was fixed `68/81/53%`,
CAGrad `72/89/74%`, IMTL-G `51/69/57%`, and Nash `71/63/65%`. The aggregate
gradient's dot with each task gradient was positive in `88/85/49%`,
`88/100/44%`, `96/96/96%`, and `100/100/100%`, respectively. Median
absolute coefficient-times-gradient-norm shares (chem/E/operator) were fixed
`0.368/0.260/0.286`, CAGrad `0.128/0.617/0.283`, IMTL-G
`0.126/0.480/0.433`, and Nash `0.209/0.359/0.480`. These shares are not
parameter-update attribution. The median cosine between the aggregate
unpreconditioned gradient and actual RAdamW delta was `−0.080/−0.108/−0.101/−0.088`;
this rotation is compatible with adaptive descent. None of these local
quantities proves task-wise improvement or starvation.

Summed measured update time by method was fixed 2,223.85 s, CAGrad 2,092.72 s,
IMTL-G 2,189.99 s, and Nash 1,984.40 s; per-update medians were
16.97/16.89/18.81/17.17 s. This is RTX 5070 Ti timing. The measured four-method
100-update seed plus its panel took about 2.61 h; extrapolating two additional
seeds gives approximately 5.21 h of additional work, excluding queue/startup,
SCF, and reruns. This is a cost estimate, not a V100 prediction or a gate.

### Historical metadata and prospective v2

The four cursor-100 Gate-3 runs are immutable historical **v1** records. An
independent review found that their protocol metadata did not include the
AO-cache chunk size. Their checkpoint, source, and protocol hashes remain
exactly as recorded; the missing field is not reconstructed or retrofitted.

The prospective **v2** contract requires `ao_cache_chunk_size` as an exact
positive integer, in addition to the point/grid chunk size. It is passed from
the CLI into AO-cache iteration and is part of canonical metadata, so a resume
with a different chunk size fails before state restore. V1 metadata is accepted
only by explicit read-only validation; the ordinary validator and checkpoint
load path reject it, and a v1 object cannot be retrofitted with the v2 field.

The first v2 launch attempt revealed that sampling seeds accidentally shared
the training metadata version constant. The fix now keeps sampling manifest
seed derivation frozen at v1 while checkpoint metadata is v2. A regression
rebuilds the compact production catalog and matches the existing full stream
manifest SHA-256 `550f22488df27b9e1326905bd3569fa7a84671e90b428ae3be190f1e409bf928`.
The four optimizer calibration gates below subsequently launched with v2
metadata, AO-cache chunk size 4096, point chunk size 256, and the same stream
hash. Historical v1 records remain unchanged.

## SCF route checks

Each cursor-100 checkpoint passed a CPU PySCF 2.14.0 RKS smoke on H2, BeH2,
and CO. All three systems converged with finite energies and XC outputs;
`Vτ` was exactly zero and the evaluated energy was exactly tau-independent.
These tests establish that the checkpoints can execute through the real SCF
route on the three small systems. They do not establish chemical accuracy.

| Model | H2 (Ha; cycles) | BeH2 (Ha; cycles) | CO (Ha; cycles) |
|---|---:|---:|---:|
| Fixed | −1.18248777 (5) | −15.95420712 (6) | −113.60826994 (8) |
| CAGrad | −1.18398214 (5) | −15.96202673 (6) | −113.63908605 (8) |
| IMTL-G | −1.19179709 (5) | −15.97906036 (7) | −113.76940016 (9) |
| Nash-MTL | −1.16990750 (5) | −15.87104441 (9) | −113.37902778 (9) |

The raw SCF artifacts and hashes are listed in `lap_moo_results.json` and
`lap_moo_protocol.json`.

## Optimizer diagnostic and pilot scope

The predeclared calibration rule selected the highest stable candidates from
AdamW `{3e-4, 1e-3, 1e-2}` and Muon `{0.01, 0.02}` with AdamW fallback
`3e-4`. The four v2 two-step gates are complete. AdamW `1e-3` and Muon
`0.02` passed; AdamW `1e-2` failed the 1% per-step parameter-delta cap.
The full table, gate JSONs, and hashes are recorded in
[lap_moo_results.json](lap_moo_results.json). The frozen run contract uses
`lap-moo-one-stage-v2`, seed 41, PBE checkpoint SHA
`ed4ba8231d93c376ce5aa8fc81e0670f6a5a44c95c2a552651b12d6a1d4b63f8`,
sampling stream SHA
`550f22488df27b9e1326905bd3569fa7a84671e90b428ae3be190f1e409bf928`,
point chunk 256, and AO-cache chunk 4096.

Matched fresh cursor-25 runs under Nash-MTL compared the existing RAdamW
control, AdamW `1e-3`, and Muon `0.02` with AdamW fallback `3e-4` on the same
27 panel identities. All were finite and nonzero at cursor/scheduler 25 on a
150-update cosine horizon. Their median ratios to the untrained predopt
checkpoint (chemistry/E/operator) were `1.230 / 2.105 / 0.807` for RAdamW,
`1.569 / 3.189 / 0.764` for AdamW, and `1.323 / 0.493 / 0.744` for Muon.

Paired row ratios against RAdamW were:

| Candidate | Median ratio, chem / E / operator | Per-task wins vs RAdamW, chem / E / operator | Row Pareto result |
|---|---:|---:|---|
| AdamW `1e-3` | `1.127 / 1.604 / 0.938` | `8 / 0 / 27` of 27 (1 chemistry tie) | 27 tradeoffs |
| Muon `0.02` + fallback `3e-4` | `1.149 / 0.248 / 0.929` | `13 / 27 / 22` of 27 | 11 Muon dominances, 16 tradeoffs |

AdamW lowers operator loss on all 27 rows, with higher E on all 27 and higher
chemistry on 18. Muon lowers E on all rows and operator on 22, while chemistry
is lower on 13; 14 chemistry rows are higher, and its overall median tradeoff
against RAdamW is not a domination. The lower median operator ratio for AdamW
is modest, while Muon's E gain is large. These are paired descriptive
training-panel observations, not held-out validation or tests of statistical
significance. The RAdamW control remains the conservative first 2xV100
optimizer recommendation because it has the complete 100-update method-screen
history and avoids native Muon's BF16 path, which the factory rejects on V100. Muon remains a strong
E/operator research candidate, not a declared winner.

The cursor-25 trajectories show why coefficient weights alone do not
characterize an update. Median raw-gradient cosine `(chem,E; chem,operator;
E,operator)` was `(+0.488, -0.687, -0.823)` for RAdamW,
`(+0.162, -0.626, -0.815)` for AdamW, and `(+0.816, -0.801, -0.943)` for
Muon. Median absolute coefficient-times-gradient-norm shares `(chem,E,
operator)` were `(0.257, 0.334, 0.448)`, `(0.281, 0.337, 0.390)`, and
`(0.252, 0.341, 0.476)`, respectively. The actual parameter delta had a
negative dot product with each raw task gradient on `92/68/76%` of RAdamW
updates, `76/56/76%` of AdamW updates, and `92/76/84%` of Muon updates. The
median cosine between the aggregated direction and actual optimizer delta was
`-0.138`, `-0.109`, and `-0.068`; a negative sign is expected for a descent
step. These projections are local first-order diagnostics, not task-specific
loss attribution. Median update norms were `0.02382`, `0.03641`, and `0.25484`;
Muon's maximum was `0.26987`. Relative to initial trainable norm `21.37525`,
that maximum is `1.263%`, beyond the two-step gate's 1% movement screen during
the later 25-step trajectory. All Muon updates remained finite and nonzero;
record this as a movement/stability caveat rather than a silent run rejection.
Median RTX 5070 Ti step times were `18.79`, `14.98`, and `14.84` seconds.

The optimizer comparisons stopped at cursor 25; they do not establish full
150-update optimizer behavior. For the next scoped pilot, use the full
150-update cosine horizon with minimum LR 10% of base and record the actual
update count. The completed local evidence is 100 updates for the four-method
screen and 25 updates for optimizer comparisons; neither is described as a
completed 150-update run. A 150-step pilot is schedule-consistent, whereas
100 would stop before the declared cosine endpoint.

The verified central operator corpus contains all 90 systems; the available
hash-bound AO-factor cache covers 15. The supported pilot scope is explicitly
limited to those cached 15 systems. The Minnesota store preserves all 268
reaction groups and 2,144 augmentation variants. CPU two-rank Gloo passed,
but no CUDA multi-GPU training run was made. No 2xV100 throughput, VRAM, or
Muon BF16 compatibility measurement exists; measure target-hardware behavior
during the pilot. One seed and training identities are descriptive evidence,
not statistical superiority evidence.

### Matched optimizer artifacts and route checks

All three cursor-25 comparisons used the same PBE checkpoint, seed 41, frozen
v2 sampling manifest, 27 panel identities, point chunk 256, and AO chunk
4096. The RAdamW control panel SHA is
`7996b706dbab7303a31e761f7892666506e9fe259824d80fb039a429b2695253`,
AdamW `1e-3` is
`815c8163bd79d8a88367182ad9eba7ebe0ebd978cb7fe8b9bdbb9b9ecbdeeb03`,
and Muon `.02` with AdamW fallback `.0003` is
`640394b9b95c208e40cbb172e9d7f0dc0f295828f102ccedd148d95a0960312f`.
The run checkpoints are RAdamW
`0104d679ed050c11163344fb37bf6b0b5da85981d808e06265eb9aaabcb76f5e`,
AdamW `7fb20e7d1a676e5f4bf5f6222a462c85199cc95f0aefec335d20983c26a88483`,
and Muon `51081e149d2de9d7ff87459d01653a920831480a4878080a6f84347f5f7588c0`.
Each panel has 27 paired rows. The two v2 candidates stopped at 25 updates on
a 150-update horizon. RAdamW is a preserved historical v1 control; its
metadata stays v1 and is not relabeled as v2.

RAdamW and Muon cursor-25 models both passed CPU SCF smokes for H2, BeH2,
and CO. RAdamW total energies were `−1.181792775`, `−15.936697375`, and
`−113.512962191` Ha (5/7/10 cycles); Muon gave `−1.174942185`,
`−15.886587609`, and `−113.350592993` Ha (5/10/9 cycles). All outputs and
cycle energies were finite, `vtau` was exactly zero, and energy was exactly
tau-independent. These route checks confirm evaluator operability; they do
not rank methods by total energy. Raw panel and SCF artifacts with hashes are
recorded in [lap_moo_results.json](lap_moo_results.json) and
[lap_moo_protocol.json](lap_moo_protocol.json).

## Ten requested final decisions

| # | Question | Current evidence-based answer |
|---:|---|---|
| 1 | Which MOO method should be assessed in the pilot? | Use Nash-MTL as the primary diagnostic candidate, IMTL-G as an operator-priority boundary case, and fixed scalarization as reference. Cursor-100 method outcomes are non-dominated tradeoffs; no overall method winner or superiority claim is supported. |
| 2 | Which optimizer should be used? | Use RAdamW `1e-2` for the first scoped V100 diagnostic: it has the complete 100-update method-screen history and avoids Muon's native BF16 implementation blocked by the V100 capability guard. Muon `0.02` has the strongest median E and operator errors but worse median chemistry, an observed later update of 1.263% of initial trainable norm, and an explicit V100 capability rejection. AdamW `1e-3` improves operator loss but worsens E on all 27 paired rows. This is a conservative pilot choice, not an optimizer-quality winner. |
| 3 | What schedule and update budget should be used? | Use cosine decay to 10% over 150 updates for the pilot and record actual cursor. Local method screens stopped at 100/150 and optimizer screens at 25/150; no complete 150-update run is claimed. |
| 4 | What method-specific hyperparameters remain? | Fixed weights `[0.1246674180, 0.0149441680, 536.7540039]`; CAGrad `c=0.4`, paper-unscaled; canonical IMTL-G; Nash Newton potential, `max_iter=100`, `update_every=1`, strict `tol=1e-10`, no fallback. The pilot uses Nash-MTL with RAdamW; other method settings remain frozen as recorded. |
| 5 | Is fixed scalarization competitive enough to retain? | Retain it as the reference control. It has the best median chemistry among the cursor-100 trained methods but trades energy/operator performance against the MOO methods. No scalar composite ranking is used. |
| 6 | Does chemistry remain systematically gradient-conflicting with E/V? | In cursor-100 raw gradients, median chem/E cosine is positive for all methods (`+0.655` to `+0.760`) while chem/operator is negative (`−0.251` to `−0.734`). Under Nash-MTL cursor-25, chem/E is positive and chem/operator negative for all three optimizers. These are stream-specific diagnostics. |
| 7 | Do E and operator remain strongly aligned? | No in the cursor-100 sample: median E/operator cosine is negative for all methods (`−0.470` to `−0.951`), unlike the initial `+0.966` median. The cursor-25 Nash optimizer sample also has negative E/operator medians for all three optimizers. |
| 8 | Does a task become starved? | No supported starvation conclusion. Coefficient-times-gradient-norm shares and actual optimizer-delta projections vary; neither is equivalent to task-specific loss attribution. |
| 9 | Is training genuinely one-stage? | Yes. Each update evaluates all three fixed objectives, without `OMEGA`, phase scheduling, or epoch-dependent task weights. Prospective v2 metadata binds AO-cache chunk size; historical v1 run records and hashes remain unchanged. |
| 10 | Is the protocol ready for the 2xV100 pilot? | Yes, for a diagnostic run restricted to the 15 cached systems, using Nash-MTL/RAdamW `1e-2` for 150 cosine updates. Validate timing/VRAM on the target devices and preserve the 27-reaction/15-system stream identity. This does not claim all-90 AO coverage or statistical superiority. |

## Verification and runtime

Source commit `408bdf4c991a64d6ef61185ce71dedfad6b67945` contains the
implementation and tests. The final combined Windows suite passed 185 tests
with 4 platform/CUDA skips; the CUDA RNG-restore test also passed when a
visible device was available. The final WSL MOO/operator/SCF suite passed 93
with 2 CUDA-only skips and included CPU two-rank Gloo. Ruff, compileall, and
staged diff checks passed. The combined-suite isolation fix keeps a historical
S5 test stub from leaking into MOO objective tests and skips CUDA-only checks
when Windows reports zero visible devices; these were test-environment issues,
not production math failures.

One RTX 5070 Ti telemetry snapshot during fixed at update 84 showed 3.69 GiB
peak allocated, 26.35 GiB reserved, and substantial Windows host/pagefile
pressure. It is not a clean capacity benchmark and does not imply 2xV100
runtime or VRAM. Cursor-25 optimizer panels took 356.25 seconds for RAdamW,
390.17 seconds for AdamW, and 354.22 seconds for Muon on the RTX 5070 Ti.
These are local timings, not V100 estimates.

The reproducible cluster command below uses a verified data-root variable.
Set `READWFN_ROOT` to the directory containing the hash-verified corpus/cache
on the target host and `RUN_OUTPUT_DIR` to a new directory outside the repo.
It launches a scoped 27-reaction/15-system diagnostic with the verified PBE
checkpoint and v2 chunk metadata:

```bash
torchrun --standalone --nproc_per_node=2 train_models/train_lap_moo.py \
  --predopt-checkpoint "$READWFN_ROOT/lap_operator_runs_20261001/predopt_fgpu_20261001T192623/lap_pbe_predopt.pt" \
  --minnesota-store-manifest "$READWFN_ROOT/lap_moo_runs_20261001/mn_group_store_268/manifest.json" \
  --central-data-dir "$READWFN_ROOT/lap_operator_runs_20261001/all90" \
  --ao-cache-dir "$READWFN_ROOT/lap_moo_runs_20261001/mrks_15system_ao_cache" \
  --panel-definition "$READWFN_ROOT/lap_moo_runs_20261001/panel_definition.json" \
  --sampling-manifest "$READWFN_ROOT/lap_moo_runs_20261001/raw_gradient_survey_20261001T/sampling_manifest.json" \
  --output-dir "$RUN_OUTPUT_DIR" --method nash_mtl \
  --optimizer-family radamw --learning-rate 0.01 --updates 150 --stop-after 150 \
  --seed 41 --dtype float32 --device auto --point-chunk-size 256 \
  --ao-cache-chunk-size 4096
```

For the local Windows checkout, the corresponding data root is
`C:\Dev\readWFN_share_ms`; keep the run output on an external path. This is
a proposed diagnostic command, not a run receipt.

The recommendation is limited to the verified 15-system/27-reaction diagnostic.
All-90 AO coverage, a completed local 150-update run, and model superiority are
not claimed. Native Muon is excluded from this V100 pilot: its BF16 path is
rejected by the capability guard on compute capability 7.0.

No production training jobs or Slurm jobs were submitted.

The clean one-stage MOO protocol is ready for a 2×V100 cluster pilot.
