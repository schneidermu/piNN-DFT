# Four-task 25-update longitudinal qualification

Starting commit: `983ba9822927424492aee506be2f1440a9055fef`, branch `lap_full_vxc`. Final report identity is the Git commit containing this artifact. **Cursor10 gate PASS; continuation attempted; update10 rejected before any trial; cursor25 NOT REACHED.** Ten accepted updates remain committed. The 25-update qualification is not obtained. No production source/test changes were made.

## Frozen experiment and provenance

25-entry manifest canonical SHA256: `2e87f7e79a66b425628bdf59a821f785e05059f193a492d4a4f3130d7411e9f3`; file SHA256: `eb64fb2ba9a98eead1dc518a1b0853eeb90d66dcb92c117dbfba23964e6afcfc`. The byte-identical small manifest is committed as `lap_four_task_25_update_sampling_manifest.json`; runtime copy remains external.
Old five-entry canonical/file SHA256: `4e349a8d6a4f0004c196091adfa46fd4db32fb4ec8c9f108750d5bf0fae75141` / `36a250e904d00240b6e736ad3bc00f72b2b564f0dfecc2e22ee3336d955297fe`. Entries0-4 are exactly equal, including rank, task samples, variants, weights and system. All32 historical R2 draws were independently reproduced before training; the first25 were used. All25 mRKS systems match the original immutable diagnostic stream.

Generator recovery: original `estimator.py` builds database-position groups in frozen full268 cycle0 order, uses `random.Random(int.from_bytes(SHA256(b"41:relchem-estimator-audit-v1")[:8], "big"))`, and draws one reaction per non-AE database in sorted database order. Original `pilot.py` supplies weights n_database/251 and the same complete AE17 batch/order with weights1/17. Existing `build_sampling_manifest_from_catalog` supplies the independent system-cycle stream. No manual future sample selection or new sampler. Generator/source hashes and complete immutable entries are recorded.

Fresh start only: canonical predopt `ed4ba8231d93c376ce5aa8fc81e0670f6a5a44c95c2a552651b12d6a1d4b63f8`. Old five-update states were comparison references, never continuation inputs. Task order relchem,ae17,exc,op; relchem primary; R2/full17/singleton E/O1; tau.02, beta.999, eps1e-8, QP tolerance1e-9. Direct PCD/vector Armijo alpha0=6.632573669086685e-7, c1e-4, rho.5, cap20. F32 main, matched-F64 chemistry, repaired production operator, chunks256/4096, seed41/world1, no optimizer/scheduler/decay.

All training executes the unchanged `train_models/train_lap_moo.py`. External launcher only starts the CLI, checks the prefix, and archives atomic checkpoints. Stage1 uses `--updates 25 --stop-after 10 --four-task-pcd --checkpoint-every 5`. Stage2 uses the same output/protocol/25-entry manifest and `--resume .../run/latest.pt --stop-after 25`. Full commands are SHA-bound externally. No second trainer or custom update path.


A single `.gitattributes` rule marks only the frozen manifest `-text`, preserving its byte SHA under Windows `core.autocrlf=true`. Actual checkout-index byte-identity proof PASS is SHA-bound in results. This is reproducibility configuration, not a production numerical change.

## First-five reproduction and baseline

PASS exact: first5 samples, losses, full diagnostics, model tensors and EMA. All5 t1/zero backtracks; active sets op; ae17; exc+op; exc+op; exc+op. Fresh cursor5 monitoring is exactly equal to the previously qualified native report data. Fresh cursor0 was recomputed before training; all E/operator denominators and chemistry win counts use this new run only. Read-only monitoring reuses the qualified evaluator, changing only win-count baseline plumbing; model/RNG/gradient immutability checks pass.

## Monitoring

| Cursor | J_rel | nonAE RMSE | AE MAE | AE RMSE | E mean ratio | E median ratio | op mean ratio | op median ratio | E wins/15 | op wins/15 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.23910490419033 | 46.3202945896816 | 47.336699410628 | 56.7916499763187 | 1 | 1 | 1 | 1 | 0 | 0 |
| 5 | 1.23899938533501 | 46.3170045566219 | 47.2499070710841 | 56.6803668993584 | 0.998301834233358 | 0.998466104935801 | 0.999976583884083 | 0.999986411808732 | 15 | 9 |
| 10 | 1.2388180284701 | 46.3070827229865 | 47.0067984131841 | 56.3780354214853 | 0.993387908147245 | 0.993481974595918 | 0.99967790112342 | 0.999803018239577 | 15 | 15 |
| 25 | not reached | — | — | — | — | — | — | — | — | — |

| Cursor | J_rel/S5 | nonAE RMSE/S5 | AE MAE/0 | AE RMSE/0 | AE MAE/S5 | AE RMSE/S5 | relchem wins/251 | AE wins/17 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.38050274065 | 1.39285180453 | 1 | 1 | 3.06772821252 | 3.01643675814 | 0 | 0 |
| 5 | 1.38038518073 | 1.3927528732 | 0.998166489413 | 0.998040502838 | 3.06210350036 | 3.01052605888 | 162 | 17 |
| 10 | 1.38018312871 | 1.39245452355 | 0.99303075623 | 0.992716982954 | 3.04634846679 | 2.99446799782 | 194 | 17 |

| Cursor | Objective | Raw mean | Raw median | Absolute mean delta | p90 system ratio | max system ratio |
|---:|---|---:|---:|---:|---:|---:|
| 0 | exc | 70.388240340135553 | 53.869783395827675 | 0 | 1 | 1 |
| 0 | op | 0.036798155186999283 | 0.03757516422082794 | 0 | 1 | 1 |
| 5 | exc | 70.268709440015783 | 53.782875135786824 | -0.11953090011976997 | 0.999277766451 | 0.999521126763 |
| 5 | op | 0.036797293517131881 | 0.03757465364230958 | -8.6166986740243701e-07 | 1.00005294561 | 1.00006659263 |
| 10 | exc | 69.922826829652777 | 53.491797443235924 | -0.46541351048277591 | 0.996260249018 | 0.996494789347 |
| 10 | op | 0.036786302542553342 | 0.037566144643821485 | -1.1852644445940752e-05 | 0.999870827739 | 0.999892418449 |

| Cursor | Database | MAE | RMSE |
|---:|---|---:|---:|
| 0 | ABDE4 | 2.71955519672 | 3.38793795861 |
| 0 | AE17 | 47.3366994106 | 56.7916499763 |
| 0 | DBH76 | 8.6267641252 | 10.0028888472 |
| 0 | EA13 | 2.47212828281 | 3.10553842353 |
| 0 | IP13 | 3.43613768169 | 4.62090605285 |
| 0 | MGAE109 | 15.8482428779 | 19.8239678751 |
| 0 | NCCE31 | 0.881692808638 | 1.24825494972 |
| 0 | PA8 | 1.47383016994 | 1.72149687209 |
| 0 | pTC13 | 5.81075507087 | 6.81135581725 |
| 5 | ABDE4 | 2.71941554271 | 3.38809765673 |
| 5 | AE17 | 47.2499070711 | 56.6803668994 |
| 5 | DBH76 | 8.62590565182 | 10.0021351265 |
| 5 | EA13 | 2.47146239591 | 3.10534596861 |
| 5 | IP13 | 3.43736147048 | 4.62165470929 |
| 5 | MGAE109 | 15.8470271467 | 19.8228136588 |
| 5 | NCCE31 | 0.881527764892 | 1.24801442169 |
| 5 | PA8 | 1.47337495833 | 1.72128489418 |
| 5 | pTC13 | 5.81026655675 | 6.81096468051 |
| 10 | ABDE4 | 2.72413241793 | 3.39505508805 |
| 10 | AE17 | 47.0067984132 | 56.3780354215 |
| 10 | DBH76 | 8.6214130594 | 9.99752072947 |
| 10 | EA13 | 2.47141463005 | 3.1039208638 |
| 10 | IP13 | 3.43846165849 | 4.62159541228 |
| 10 | MGAE109 | 15.8311169315 | 19.8029699838 |
| 10 | NCCE31 | 0.880510962803 | 1.24675730714 |
| 10 | PA8 | 1.47406977234 | 1.7230963348 |
| 10 | pTC13 | 5.8152544206 | 6.81513570335 |

All15 individual E/operator losses and ratios at each reached cursor are in results; no systems excluded. Both aggregate mean and median gates were evaluated independently. From5 to10, Jrel/nonAE/AE and E/operator aggregate metrics improve further; this says nothing about unobserved cursor25.

## Cursor10 gate and true resume

Every predeclared cursor10 check passed, without epsilon or rounding allowances. The gate receipt was produced before Stage2. Native checkpoint identity was validated using existing `load_moo_checkpoint`; actual CLI resumed at the next unconsumed entry10, with retained EMA t10 and identical protocol/source/manifest identities. No new run or resampling.

## Rejected update10: read-only attribution

System ClHS; exact R2/full17 identities/variants/weights are preserved in the rejection receipt. QP feasible, active constraint exc, mu_exc=0.42481032728110985. The QP satisfies secondary constraints but its primary chemistry slope is ascent. This is a direction-gate failure, not an observed finite-step/Armijo or precision failure: no trial candidate was evaluated.

| Task | Raw gradient norm | g dot requested alpha0 step |
|---|---:|---:|
| relchem | 25.0684385608523 | 3.13346504335969e-05 |
| ae17 | 6992.40227426247 | -0.0299424113989479 |
| exc | 62100.8459977201 | -0.264796392561897 |
| op | 0.688329196827044 | -5.05671511597843e-06 |

Raw cosines: relchem/AE -0.982783209819936; relchem/E -0.9831013057984; relchem/op -0.913377216330677. The chosen direction conflicts with relchem on this sample. This does not prove that every possible common-descent direction is absent, nor establish Pareto criticality.

Proposed EMA t11 was discarded; retained EMA and cursor remain10. Full rejected checkpoint payload equals the pre-attempt cursor10 payload exactly: model/buffers/EMA/cursor/RNG/protocol and absent optimizer/scheduler state. No skipped rejection, altered tau, retry, or additional update. Cursor25 gate is not evaluable because25 updates were not accepted.

Rewritten checkpoint serialization has a different file SHA256, while every payload field and numeric tensor/RNG value is exact. Both before/after file hashes are preserved; no file-byte identity is claimed.

## Stream exposure and update trends

| Update | System | Active | t | Backtracks |
|---:|---|---|---:|---:|
| 0 | BeH2 | op | 1.0 | 0 |
| 1 | H2 | ae17 | 1.0 | 0 |
| 2 | BH | exc,op | 1.0 | 0 |
| 3 | CH4 | exc,op | 1.0 | 0 |
| 4 | HPSi_iso2 | exc,op | 1.0 | 0 |
| 5 | HLi | ae17 | 1.0 | 0 |
| 6 | N2 | none | 1.0 | 0 |
| 7 | CH2O | ae17 | 1.0 | 0 |
| 8 | C2H2_iso2 | ae17 | 1.0 | 0 |
| 9 | H4Si | exc | 1.0 | 0 |
| 10 | ClHS | exc | rejected/no trial | 0 |

Accepted10/attempted11 (0.909090909). All accepted t1/zero backtracks; minimum accepted t1, maximum accepted backtracks0. Cursor10 exposure: 10/15 distinct systems. Cursor25/all15 exposure was not reached. Rejected ClHS was attempted but not committed; it is not counted as accepted stream coverage.

| System | Accepted count through10 |
|---|---:|
| AlBeH | 0 |
| BH | 1 |
| BeH2 | 1 |
| C2H2_iso2 | 1 |
| CH2O | 1 |
| CH4 | 1 |
| CO | 0 |
| ClH | 0 |
| ClHS | 0 |
| H2 | 1 |
| H2O | 0 |
| H4Si | 1 |
| HLi | 1 |
| HPSi_iso2 | 1 |
| N2 | 1 |

The transition after5 is preserved: updates5,7,8 activate AE;6 has no active constraint;9 activates E;10 activates E but fails primary descent. Complete raw/normalized norms,4x4 cosines, multipliers, requested/realized geometry, losses, reduction ratios, Armijo margins and EMA/cursor receipts remain SHA-bound outside Git. Accepted-only distributions are separated into first5 and after5; the failed sample is reported separately rather than hidden in an average.

| Task | Accepted raw norm min/max | Accepted multiplier median/max | actual/predicted median/min/max |
|---|---|---|---|
| relchem | 7.8356075/42.130588 | 0/0 | 0.99960739/0.99734554/1.0024927 |
| ae17 | 6992.3499/6992.4295 | 0/1.7325609 | 1.0005106/0.99668988/1.0032806 |
| exc | 663.42097/50598.163 | 0/0.92084754 | 0.98779894/0.94368889/1.0945358 |
| op | 0.16933185/0.6456961 | 0/0.63629402 | 0.99857624/0.97379119/1.0000198 |

| Pair | first5 cosine median | after5 cosine median | all accepted min/max |
|---|---:|---:|---|
| ae17:exc | 0.996110879 | 0.999053519 | 0.984442322/0.999989838 |
| ae17:op | 0.954664036 | 0.802996363 | 0.701800449/0.980123749 |
| exc:op | 0.969456014 | 0.821602558 | 0.720978093/0.980346888 |
| relchem:ae17 | -0.896033835 | -0.934651028 | -0.984792571/0.949579724 |
| relchem:exc | -0.891565796 | -0.922528694 | -0.984969134/0.93569721 |
| relchem:op | -0.914078924 | -0.667018859 | -0.956718383/0.65533587 |

Attempted operator-gradient exposure is11 distinct systems, including rejected ClHS; accepted-update exposure remains10/15. Monitoring all15 does not imply all15 training systems received an accepted step.

## Stress checks, validation and limitations

Prefix drift: exact pre-training comparison and runtime first5 parity PASS. Baseline substitution: fresh own0 evaluator/denominators PASS. Survivorship: first rejected update terminates; retained payload exact PASS. Future manifest mutation and precision identity mismatch: rejected before model/RNG mutation using actual cursor10 checkpoint. Frozen25 plan prevents panel-driven selection. Mean/median gates remain separate. Scope is15 cached systems, never90.

Windows130 passed/5 deselected; WSL/PySCF/Gloo132 passed/2 skipped. Ruff, compileall, external tooling py_compile, diff check, source/data/manifest identity and repository-state checks PASS. No production-code fix was needed. Pre-existing unrelated untracked directories remain untouched. Independent review PASS: frozen-plan, gate, resume, rollback and reporting fidelity; receipt and SHA are recorded in results. This review does not grant the unfinished 25-update scientific qualification.

Runtime: accepted step median 79.142s, maximum 186.877s; peak logical Torch allocation 22003756544 bytes, physical RTX GPU 17094475776 bytes. These Windows/shared-memory diagnostics do not qualify V100 memory or throughput.

Ten-update local monitoring is successful, but the25-update longitudinal qualification fails to complete because canonical PCD produces primary ascent on the next fixed sample. The single next investigation is a read-only frozen cursor10/sample gradient-geometry audit distinguishing an unavoidable imposed-secondary-margin conflict from the particular chosen PCD projection. No new direction or protocol is proposed here. Full90/100-update/cluster qualification is not authorized; do not average this failure away with more systems.

No >25-update training, no 100-update run, no full90 run, no SCF benchmark, no Diet run, and no Slurm production job was launched.
