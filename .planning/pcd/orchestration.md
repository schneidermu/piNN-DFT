# PCD extension orchestration state

## Workspace and prior state

- Repository: `C:\Dev\readWFN_share_ms\lap_full_vxc`.
- Branch: `lap_full_vxc`, tracking `origin/lap_full_vxc`.
- Current base for this specification: `f824eec42ba0484b28030316b84558c5243afcde`.
- Prior implementation commit: `408bdf4c991a64d6ef61185ce71dedfad6b67945`; its documented feature base was `61837e4346ecaf46247572df24e3f2b757a7b415`.
- Existing MOO report says Gate 3 is complete for the corrected four-method cursor-100 screens, matched 27-row panels, and three-system CPU SCFs. It also records all three matched cursor-25 optimizer panels and scoped route checks.
- `lap_moo_protocol.json` still reports the earlier Nash/RAdamW pilot-ready state. Treat that as a historical prior-study status; do not rewrite or relabel its provenance, manifests, checkpoints, or run receipts. The report, protocol, and individual artifact versions are not interchangeable.
- The default sampling manifest in that study is world-size 1 with embedded canonical manifest identity `550f22488df27b9e1326905bd3569fa7a84671e90b428ae3be190f1e409bf928`; its file-bytes digest is separate. Neither identity is valid for the new world-size-2 experiment. Generate and record a distinct world-size-2 manifest and its embedded and file-byte hashes.
- The 268-group Minnesota store, 90-system central operator corpus, 15-system AO cache, panel definition, and external optimizer-study directory were present when checked. The parent also recorded a verified seven-input inventory at `C:\Dev\readWFN_share_ms\lap_pcd_runs_20261002\input_inventory.json`. No PCD training run has started. The available local GPU is an RTX 5070 Ti; target V100 runtime and memory behavior remain unmeasured.
- No `AGENTS.md` or `CONTEXT.md` was present at the project root. The repository and user-provided planning/reuse-first workflow govern. `.planning/quick/` was already untracked at audit start and is outside this task's ownership; preserve it.

## Current goal

Integrate source-faithful PCD through the accepted minimal specification; pass
official parity, state/resume, and DDP gates; evaluate six tau values and at most
three cursor-100 candidates; generate the selected world-size-2 manifest; verify
exact two-rank execution; independently review and publish the three requested
reports. The authoritative source audit is complete and the all-three-median
improvement target remains unchanged.

Keep the existing objective definitions, model architecture, predopt initialization, optimizer family/convention, cosine horizon, data panel, logging, sampling, panel evaluation, and h-free operator path. Use task order `chem` (primary), `exc`, `op` (secondary). Do not modify Nash behavior, introduce fairness/Nash changes, add diet or Slurm workflows, or claim all-90 AO coverage.

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

The parent reports 94 installed instruction files were checked. Caveman, Ponytail, and Matt Pocock are unavailable. Do not fabricate substitutes for those skills; retain the direct-user planning and reuse-first workflow described in the active instructions. This agent owns only `.planning/pcd/spec.md`, `.planning/pcd/tickets.md`, and this file. No implementation or test run is authorized by this planning assignment. Wait for acceptance of the spec and ticket set before opening code work.

## Status

The orchestrator accepted the source contract, short spec, and eight tickets.

| Ticket | Status | Acceptance / blocker |
|---|---|---|
| T1 | Accepted | Pinned paper/code/license; 88 upstream tests passed; spec accepted. |
| T2 | Accepted | Source-faithful aggregator; fresh parity audit covered dtype edge behavior. |
| T3 | Accepted | Windows 219 passed/3 skipped; WSL 110 passed/2 CUDA skipped, including two-rank PCD/resume. |
| T4 | Accepted | Independent upstream and integration review passed; Ruff, compileall and diff check passed. Gate 1 closed. |
| T5 | Running | Six-tau real-gradient geometry and common learning-rate declaration; no model updates yet. |
| T6 | Pending | Requires T5 geometry and declared learning-rate rule. |
| T7 | Pending | Requires the locally selected T6 candidate and a new world-size-2 manifest. |
| T8 | Pending | Requires experiments, exact two-rank proof, and fresh final review. |

Gate 1 passed; real PCD training follows acceptance of T5 geometry and its common learning-rate rule. Historical artifacts and
`.planning/quick/` remain untouched.
