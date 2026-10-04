# Precision-repaired five-update four-task pilot

Five-update qualification **PASS**. Fresh canonical predopt, exact original seed41 immutable sampling manifest, production precision repair1173a75 only. No continuation from a historical trained/diagnostic checkpoint.

## Provenance and frozen execution

Branch lap_full_vxc; start `1173a755dadb220066cb5f5a29c319413e978ef7`. Predopt SHA `ed4ba8231d93c376ce5aa8fc81e0670f6a5a44c95c2a552651b12d6a1d4b63f8`. Manifest canonical SHA `4e349a8d6a4f0004c196091adfa46fd4db32fb4ec8c9f108750d5bf0fae75141`; file SHA `36a250e904d00240b6e736ad3bc00f72b2b564f0dfecc2e22ee3336d955297fe`. Original manifest copied byte-for-byte, not regenerated. Exact source/checkpoint/data/runtime artifacts are SHA-bound in protocol/results.

relchem primary; task order relchem/ae17/exc/op. R2/full17, O1, tau=.02,beta=.999,eps=1e-8,QP tolerance1e-9. Direct PCD/componentwise Armijo, alpha0=6.632573669086685e-7,c=1e-4,rho=.5,cap20. Same cycle0 variants and BeH2,H2,BH,CH4,HPSi_iso2. Main parameters/checkpoints and cache stay F32; only chemistry uses its established matched-F64 shadow. Operator uses actual production F64 learned branch/F32 PBE/F64 AO path, not a shadow operator. No optimizer/scheduler/momentum/decay/OMEGA/curriculum.

Original driver reused with only fresh monitoring, byte-identical manifest reuse and logging changes. No production source edit. Source hash validation precedes evaluation. Checkpoint0 full268 chemistry and unique15 E/operator are recomputed before any update. Its operator values are the only denominators; historical operator values are information only. Evaluation asserts main state/RNG equality and no accumulated gradients. Canonical predopt metadata verifies predopt_only,2 epochs,lr=.01.

## Recomputed checkpoint0 operator baseline

Mean `0.036798155186999283`, median `0.03757516422082794`. Historical mean `0.036798618710660792`; informational difference `-4.6352366150914648e-07`. All15 individual baseline values and hashes are in results. Chemistry baseline predictions agree bitwise with historical canonical predopt; wins are recomputed against this run0.

## Accepted updates

| update | system | accepted | t | backtracks | alpha | active |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | BeH2 | True | 1.0 | 0 | 6.632573669086685e-07 | ['op'] |
| 1 | H2 | True | 1.0 | 0 | 6.632573669086685e-07 | ['ae17'] |
| 2 | BH | True | 1.0 | 0 | 6.632573669086685e-07 | ['exc', 'op'] |
| 3 | CH4 | True | 1.0 | 0 | 6.632573669086685e-07 | ['exc', 'op'] |
| 4 | HPSi_iso2 | True | 1.0 | 0 | 6.632573669086685e-07 | ['exc', 'op'] |

All raw/normalized norms, full4x4 cosines, multipliers/active sets, requested/realized products and step geometry, zero-coordinate fractions, before/after losses, actual/predicted ratios and Armijo margins remain in structured results. Accepted updates alone advance cursor/retained EMA. Proposed EMA is logged once per attempted update; rejected proposals are not persisted as accepted state.

## Full checkpoints

| checkpoint | J_rel | nonAE RMSE | AE MAE | AE RMSE | E mean ratio | E median ratio | Op mean ratio | Op median ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.2391049041903315 | 46.32029458968161 | 47.336699410627965 | 56.791649976318745 | 1.0 | 1.0 | 1.0 | 1.0 |
| 2 | 1.2390285343167058 | 46.31728458837722 | 47.31506800988994 | 56.76428112387434 | 0.999593233981917 | 0.9996253537438359 | 0.9999978821878048 | 1.0000018417009 |
| 5 | 1.238999385335012 | 46.317004556621946 | 47.249907071084095 | 56.68036689935841 | 0.9983018342333583 | 0.9984661049358013 | 0.9999765838840827 | 0.9999864118087319 |

Mean ratio in this table is the ratio of aggregate raw means, matching the guard. The separate arithmetic mean of individual ratios, median,p90,max,wins, raw means/medians and all15 losses/ratios are retained in results, along with per-DB chemistry, S5 ratios and baseline-paired chemistry/AE win counts. RMSE/AE errors use the unchanged kcal/mol definitions.

| checkpoint | op arithmetic mean individual ratio | op absolute mean delta | operator wins/15 | relative chemistry wins/251 | AE wins/17 |
| --- | --- | --- | --- | --- | --- |
| 0 | 1.0 | 0.0 | 0 | 0 | 0 |
| 2 | 0.9999891388793536 | -7.793158181279569e-08 | 7 | 199 | 17 |
| 5 | 0.9999649481118434 | -8.61669867402437e-07 | 9 | 162 | 17 |

## All15 repaired operator losses and ratios

| system | checkpoint0 loss | checkpoint2 loss | ratio2/0 | checkpoint5 loss | ratio5/0 |
| --- | --- | --- | --- | --- | --- |
| H2 | 0.05065144760149834 | 0.050650045037814805 | 0.9999723095044672 | 0.05064911597421242 | 0.9999539672132519 |
| HLi | 0.021227941618739317 | 0.02122807182821393 | 1.000006133871901 | 0.0212280920030225 | 1.0000070842612008 |
| BH | 0.04194661745040866 | 0.041946989412728904 | 1.0000088675164496 | 0.04194603869637718 | 0.9999862026054387 |
| BeH2 | 0.0332980676660685 | 0.0332980622200238 | 0.9999998364456234 | 0.033297943719713674 | 0.999996277671243 |
| H2O | 0.05311817567856048 | 0.05312027689888625 | 1.0000395574640681 | 0.05312008488146404 | 1.00003594255411 |
| CH4 | 0.03757516422082794 | 0.037575233423041705 | 1.0000018417009 | 0.03757465364230958 | 0.9999864118087319 |
| N2 | 0.0548397430769853 | 0.05484211730379707 | 1.0000432939083692 | 0.05484178174804309 | 1.0000371750658081 |
| CO | 0.04991018709685011 | 0.04991316244247751 | 1.0000596139946665 | 0.04991351074755897 | 1.0000665926317287 |
| CH2O | 0.03949757996330448 | 0.039499570226845464 | 1.0000503895059603 | 0.03950008645228915 | 1.0000634593052788 |
| C2H2_iso2 | 0.03369685883390478 | 0.033697985353384946 | 1.000033430993842 | 0.033697793601877635 | 1.000027740507727 |
| H4Si | 0.02745743766580724 | 0.027456021862200034 | 0.9999484364264271 | 0.027454738582166564 | 0.9999016993619897 |
| AlBeH | 0.01765758811192092 | 0.017657103340703174 | 0.9999725460116823 | 0.01765690539548098 | 0.9999613358044365 |
| ClH | 0.04077189079823243 | 0.04076819243621711 | 0.9999092913783759 | 0.040764378613256345 | 0.9998157508805942 |
| ClHS | 0.030487420761307275 | 0.030484083809694563 | 0.9998905466080966 | 0.030481407507561085 | 0.9998027627921277 |
| HPSi_iso2 | 0.019836207260573523 | 0.019834243235232783 | 0.9999009878594763 | 0.019832871191644882 | 0.999831819213985 |

## Exact final guards

| guard | PASS |
| --- | --- |
| five_accepted | True |
| J_rel | True |
| nonAE_RMSE | True |
| AE_MAE | True |
| AE_RMSE | True |
| exc_mean | True |
| exc_median_ratio | True |
| op_mean | True |
| op_median_ratio | True |

## Historical paired comparison

| update | same sample | active changed | old/new t | old/new backtracks | old/new op actual/predicted |
| --- | --- | --- | --- | --- | --- |
| 0 | True | False | [1.0, 1.0] | [0, 0] | [25.707886369740816, 0.9737911927408135] |
| 1 | True | False | [1.0, 1.0] | [0, 0] | [0.7406400326354896, 1.0000198343022055] |
| 2 | True | False | [1.0, 1.0] | [0, 0] | [4.258721457032871, 0.9950523089411387] |
| 3 | True | False | [1.0, 1.0] | [0, 0] | [8.408367825881795, 0.9970698889970935] |
| 4 | True | False | [1.0, 1.0] | [0, 0] | [9.217262189615449, 0.9986488289670602] |

### Historical versus repaired checkpoint5

| metric | historical | repaired |
| --- | --- | --- |
| J_rel | 1.238999382945787 | 1.238999385335012 |
| weighted_nonAE_RMSE | 46.317004400171044 | 46.317004556621946 |
| AE_MAE | 47.24992188965453 | 47.249907071084095 |
| AE_RMSE | 56.68038566077223 | 56.68036689935841 |
| exc mean ratio | 0.9982953417001039 | 0.9983018342333583 |
| op mean ratio | 1.0000758311657787 | 0.9999765838840827 |
| exc median ratio | 0.998458978795645 | 0.9984661049358013 |
| op median ratio | 1.0000461677988226 | 0.9999864118087319 |

Operator ratios use each run's own baseline arithmetic; this is a historical outcome comparison, not interchangeable operator denominators. All historical samples, active sets, t=1 and zero backtracks match. Chemistry/AE gains are preserved; E improves in both runs. Repaired operator wins rise from7 to9 of15. Operator actual/predicted reduction ratios become near1 on all five updates.


All task before/after deltas, gradient norm/cosine changes and historical panel values are retained; trajectories are not claimed bitwise comparable. The changed operator scalar and gradient intentionally influence the PCD direction from update0. The historical operator mean/median regression is comparison context only.

## Validation and decision

Windows97 tests passed (one skipped,two DDP deselected;40 operator/training plus57 four-task/Armijo/aggregator); WSL/PySCF15 passed. Ruff,compileall,diff checks pass. Data/manifests/source hashes checked; model/RNG evaluation invariance and caller restoration pass. Stored main parameters remain F32. No epsilon, rounding allowance or practical-equality exception is applied.

Runtime 1049.003s on RTX5070Ti/Torch2.11.0+cu128. Reported CUDA peak allocated 21987680256bytes, reserved 31878807552bytes. These allocator counters exceed the observed device physical-memory capacity17094475776bytes; they are not physical-VRAM residency measurements or V100 forecasts.

Independent review: PASS.

The minimal operator precision repair resolves the previous five-update qualification failure under the frozen seed-41 / R2 / full17 / 15-system pilot.

Single next gate: Repository-native four-task protocol/CLI qualification, including repaired precision identity and resume/DDP parity; no longer trajectory yet.

This qualifies the shared four-task helpers and frozen external driver, not a repository-native four-task CLI or production training scheme. No automatic longer run is authorized.

No >5-update run, no full90 run, no SCF benchmark, no Diet run, and no Slurm production job was launched.
