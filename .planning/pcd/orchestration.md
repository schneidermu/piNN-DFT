# PCD extension orchestration state

## Workspace and prior state

- Repository: `C:\Dev\readWFN_share_ms\lap_full_vxc`.
- Branch: `lap_full_vxc`, tracking `origin/lap_full_vxc`.
- Current base for this specification: `f824eec42ba0484b28030316b84558c5243afcde`.
- Prior implementation commit: `408bdf4c991a64d6ef61185ce71dedfad6b67945`; its documented feature base was `61837e4346ecaf46247572df24e3f2b757a7b415`.
- Existing MOO report says Gate 3 is complete for the corrected four-method cursor-100 screens, matched 27-row panels, and three-system CPU SCFs. It also records all three matched cursor-25 optimizer panels and scoped route checks.
- `lap_moo_protocol.json` still reports the earlier Nash/RAdamW pilot-ready state. Treat that as a historical prior-study status; do not rewrite or relabel its provenance, manifests, checkpoints, or run receipts. The report, protocol, and individual artifact versions are not interchangeable.
- The default sampling manifest in that study is world-size 1 with embedded canonical manifest identity `550f22488df27b9e1326905bd3569fa7a84671e90b428ae3be190f1e409bf928`; its file-bytes digest is separate. Neither identity was reused for T7. The new immutable world-size-2 manifest and both digests are recorded in the T7 execution receipt.
- The 268-group Minnesota store, 90-system central operator corpus, 15-system AO cache, panel definition, and external optimizer-study directory were verified against the inventory at `C:\Dev\readWFN_share_ms\lap_pcd_runs_20261002\input_inventory.json`. T6 screens and panels are complete with all six candidates rejected; T7 engineering-only two-rank and SCF receipts are accepted. No scientific tau candidate or cursor-100 run was selected.
- No `AGENTS.md` or `CONTEXT.md` was present at the project root. The repository and user-provided planning/reuse-first workflow govern. `.planning/quick/` was already untracked at audit start and is outside this task's ownership; preserve it.

## Current goal
T1–T5 are accepted and Gate 1 is closed. T6 completed with all six tau candidates rejected under the unchanged all-three-median improvement criterion; no candidate advanced to cursor 100. T7 passed as engineering-only two-rank and SCF validation at representative tau=0.02, with scientific tau selection remaining null. T8 final independent report review is pending.

## Integration findings

The reusable path is `train_models/lap_moo_training.py`: `materialize_task_zeros` replaces unused task gradients with exact zero tensors; `average_raw_task_gradients` all-reduces and averages each raw task gradient before nonlinear aggregation; `train_moo_update` invokes the aggregator once on the common global gradients and applies one optimizer step. Thus PCD's stateful EMA must update after this all-reduce and deterministically on every rank. Existing checkpoint save/load persists `aggregator_state`, protocol metadata, optimizer/scheduler state, update cursor, and per-rank RNG state.

The accepted minimal code surface is `moo_aggregators.py`, `lap_moo_protocol.py`, `train_lap_moo.py`, `lap_moo_training.py`, the method enumeration/configuration in `lap_moo_analysis.py` and `run_lap_moo_benchmark.py`, plus focused aggregator/training/protocol tests. Reuse current flatten-free Gram geometry and logging; do not add model-sized vector concatenation to training.

## Experiment and acceptance constraints

- Keep RAdamW conventions and the 150-update cosine schedule to 10% of the base rate, with point chunk 256 and AO-cache chunk 4096.
- PCD's canonical rescaling to the raw primary-gradient norm can have a very different magnitude from Nash's aggregate. Predeclare either one common PCD base learning rate or a minimal stability calibration before the six cursor-25 screens. Do not silently reuse `0.01` or hand-normalize the PCD direction.
- Compare six source-approved `tau` values: `0`, `0.005`, `0.01`, `0.02`, `0.05`, `0.10`. The unchanged primary success rule is median loss ratio below 1 for each of chemistry, mRKS energy, and operator on the matched panel; retain the prior 100x per-row stability ceiling. Chemistry is the primary objective when describing tradeoffs.
- Run six matched 25-update screens, then advance at most two or three eligible candidates to cursor 100 with their paired 27-identity panels and SCF route checks. No overall winner or statistical superiority claim follows from this single-seed training panel.
- The two-rank CPU Gloo check must use a new world-size-2 manifest and cover common-gradient ordering, same EMA/aggregator state on both ranks, and exact two-rank checkpoint/resume.
- Fresh independent final review and a report are required before calling the PCD integration complete.

## Planning/tool availability and ownership

The earlier instruction-file availability check and reuse-first planning workflow were recorded before implementation. T1–T7 are now closed at the statuses below. Final report review remains pending; `.planning/quick/` remains outside this task and untouched.

## Status

The orchestrator accepted the source contract, short spec, and eight tickets.

| Ticket | Status | Acceptance / blocker |
|---|---|---|
| T1 | Accepted | Pinned paper/code/license; 88 upstream tests passed; spec accepted. |
| T2 | Accepted | Source-faithful aggregator; fresh parity audit covered dtype edge behavior. |
| T3 | Accepted | After the RNG-cardinality fix at d450bd7: Windows 223 passed/3 skipped; WSL 114 passed/2 CUDA skipped, including two-rank PCD/resume. |
| T4 | Accepted | Independent upstream and integration review passed; Ruff, compileall and diff check passed. Gate 1 closed. |
| T5 | Accepted | All six tau values feasible on all 27 rows; float32/float64 active sets agree. Common bound-derived LR = 6.632573669086685e-7. |
| T6 | Completed — negative | All six candidates were rejected by the unchanged success criterion; none advanced to cursor 100. |
| T7 | Accepted — engineering only | New immutable WS=2 manifest and actual two-rank CPU Gloo CLI resume parity passed at representative tau=0.02; scientific selection remains null. PCD SCF smoke passed on the historical cursor-25 checkpoint. |
| T8 | Accepted — negative scientific outcome | Fresh independent artifact review verified source/input/run hashes and coherent completed-negative protocol; no supported tau or cluster recommendation. |

Gate 1 passed; T1–T5 are accepted. All six T6 candidates were rejected, and T7 is accepted as engineering-only validation with no selected scientific tau. Historical artifacts and
`.planning/quick/` remain untouched.
