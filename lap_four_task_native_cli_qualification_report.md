# Native four-task CLI qualification

Starting commit: `abd2623e7fdeebc054a2299a09da0874ae9e0537`. Final implementation identity is the Git commit containing this report; numerical runs bind the exact working-source SHA256 values in the accompanying results. **PASS: all five updates and all nine unchanged endpoint gates.**

## Implementation

Production files modified: `train_models/train_lap_moo.py`, `train_models/lap_moo_protocol.py`, `train_models/lap_moo_training.py`. No source module or second trainer added. Production diff: 126 lines added, 22 removed. Test file added: `train_models/test_lap_native_four_task.py` (274 lines). Existing three-task branch retained; no objective, precision, optimizer, PCD, or sampling algorithm changed. Planning: two small files under `.planning/native_four_task/`; unrelated content untouched.

Explicit `--four-task-pcd` requires PCD/direct vector Armijo, F32 storage, and the immutable v2 manifest. It loads exact relchem/ae17 variants and weights into existing bounded-memory matched-F64 ChemistryBatchObjectives, then existing E/operator factories. Canonical task order is relchem, ae17, exc, op. Actual repaired production operator is used. No permanent whole-model-double conversion; F64 chemistry gradients avoid F32 leaf storage.

New metadata is `lap-moo-one-stage-v6`, retaining `lap-weakform-ao-v1`. Resume binds the numerical-source map plus lap_operator.py, lap_vxc.py, NN_models_lap.py, PBE.py, constants.py. Test/report hashes are excluded. v5 remains readable and cannot be retrofitted or resumed as v6. Changes to a bound precision source fail before state restoration.

## Frozen identities

Predopt SHA256: `ed4ba8231d93c376ce5aa8fc81e0670f6a5a44c95c2a552651b12d6a1d4b63f8`. Manifest canonical SHA256: `4e349a8d6a4f0004c196091adfa46fd4db32fb4ec8c9f108750d5bf0fae75141`; byte SHA256: `36a250e904d00240b6e736ad3bc00f72b2b564f0dfecc2e22ee3336d955297fe`. Existing manifest reused without regeneration. All source/data/cache/checkpoint identities and large external artifact hashes are in the results.

Seed41, world1, point chunk256, AO chunk4096. R2/full17/O1; tau.02, beta.999, eps1e-8, QP tolerance1e-9; alpha0=6.632573669086685e-7, c1e-4, rho.5, cap20. Direct vector stepping: no optimizer/scheduler state.

## Real-data parity and monitoring

Update0 parity passed before the five-update run. All five sample identities, losses, norms, full cosine matrices, multipliers, directional products, Armijo margins/ratios, requested/realized geometry, model tensors, EMA, and next cursor match the qualified external driver exactly. No numerical tolerance used.

| Update | System | Active constraints | t | Backtracks |
|---:|---|---|---:|---:|
| 0 | BeH2 | op | 1.0 | 0 |
| 1 | H2 | ae17 | 1.0 | 0 |
| 2 | BH | exc, op | 1.0 | 0 |
| 3 | CH4 | exc, op | 1.0 | 0 |
| 4 | HPSi_iso2 | exc, op | 1.0 | 0 |

Checkpoint0 full268 chemistry and unique15 E/operator were freshly evaluated with the current production path; historical pre-repair operator denominators were not used. Read-only monitoring reused the qualified evaluator verbatim and checks model/RNG/gradient immutability. Per-database chemistry and all15 per-system metrics/ratios/p90/max are retained in results.

| Cursor | J_rel | nonAE RMSE | AE MAE | AE RMSE | E mean / median ratio | Operator mean / median ratio |
|---:|---:|---:|---:|---:|---|---|
| 0 | 1.23910490419033 | 46.3202945896816 | 47.336699410628 | 56.7916499763187 | 1 / 1 | 1 / 1 |
| 2 | 1.23902853431671 | 46.3172845883772 | 47.3150680098899 | 56.7642811238743 | 0.999593233981917 / 0.999625353743836 | 0.999997882187805 / 1.0000018417009 |
| 5 | 1.23899938533501 | 46.3170045566219 | 47.2499070710841 | 56.6803668993584 | 0.998301834233358 / 0.998466104935801 | 0.999976583884083 / 0.999986411808732 |

Repaired checkpoint0 operator mean 0.036798155186999283; median loss 0.03757516422082794. Checkpoint5 operator wins9/15; absolute mean change -8.6166986740243701e-07. E wins15/15. Chemistry/per-system rows are exact against the external repaired run; no trajectory differences found. Tiny aggregate serialization differences are not acceptance allowances; all nine comparisons are direct strict/no-worse comparisons against this run's baseline.

## Resume, DDP, fail-closed validation

Fresh native update0 then save/resume/update1 matches uninterrupted native prefix2 bitwise: complete checkpoint payload, buffers, parameters, PCD EMA, cursor, Python/NumPy/Torch/CUDA RNG, protocol and scientific update1 diagnostics. Only wall-clock/cache timings and allocator instrumentation differ; these are not optimizer state or numerical diagnostics.

Two-rank CPU/Gloo test executes actual native main through v2 per-rank batches, existing ChemistryBatchObjective, raw-task global averaging before PCD, direct vector Armijo, rank0 checkpoint policy, and stop/resume. Distinct prescribed rank samples, identical global directions/EMA/model, exact resume and rejected-state rollback pass. External-data I/O and tiny objective formulas are fixture substitutions; this is wiring/DDP evidence, while real scientific formulas are qualified separately by GPU parity. World/schema/source/catalog/manifest/precision corruption and incompatible-mode tests fail closed.

Windows:130 passed,5 deselected. WSL/PySCF/Gloo:132 passed,2 skipped (platform-specific); Ruff, compileall, git diff --check pass. WSL uses explicit Git-directory mapping for the existing Windows worktree and installed existing test dependencies; no repository dependency or production portability workaround added.

## Independent review and next gate

Independent Luna Max review is recorded externally and SHA-bound in results. Review PASS: no substantive findings. Reviewed actual diff, tests, numerical parity, resume, DDP semantics and source-hash enforcement. No unnecessary second framework or numerical redesign is intended.

Next gate: a separately authorized longer local qualification using this single native entrypoint and the existing roadmap. This five-update wiring qualification does not establish production-scale or 2xV100 memory/performance readiness.

No >5-update training, no full90 run, no SCF benchmark, no Diet run, and no Slurm production job was launched.
