# PCD implementation tickets

Implementation is gated on acceptance of `spec.md` and this ticket set. Dependencies are listed explicitly; no code work starts before that gate.

## T1 — Freeze source contract

**Goal:** Keep the paper and upstream semantics auditable. **Files:** `.planning/pcd/source_contract.md`, `spec.md`. **Depends on:** none. **Acceptance:** authoritative source audit pins paper/PDF SHA, official commit, MIT license/attribution, formula, epsilon placement, K=3 tolerance, zero/infeasible cases, and upstream 88-test result; orchestrator accepts the contract before T2.

## T2 — Add PCD aggregation

**Goal:** Implement only canonical PCD behind the existing aggregator interface. **Files:** `train_models/moo_aggregators.py`, `test_moo_aggregators.py`, `test_moo_aggregators_independent.py`. **Depends on:** T1. **Acceptance:** Compare outputs, feasibility, active sets, multipliers, and failure/zero cases against independent analytic fixtures and pinned upstream; preserve task order and all Nash results.

## T3 — Bind protocol, CLI, state, and DDP resume

**Goal:** Route PCD through existing training/checkpoint infrastructure and bind all required metadata. **Files:** `lap_moo_protocol.py`, `train_lap_moo.py`, `lap_moo_training.py`, `lap_moo_analysis.py`, `run_lap_moo_benchmark.py`, focused tests. **Depends on:** T2. **Acceptance:** Validator rejects missing/mismatched PCD provenance, hyperparameters, world size, chunks, and hashes; two-rank Gloo shows common gradients and identical EMA/QP state; exact checkpoint resume restores EMA `v` and count `t`, optimizer, scheduler, cursor, and rank RNG.

## T4 — Fresh algorithm parity review

**Goal:** Independently review the implemented method and integration. **Files:** review note under `.planning/pcd/`. **Depends on:** T2–T3. **Acceptance:** Fresh reviewer checks the pinned source, every active-set/zero/infeasible case, EMA update order, raw-primary rescaling, finite-epsilon caveat, metadata, and unchanged Nash path; resolve all blocking findings before any training screen.

## T5 — Real-gradient six-tau geometry and rate declaration

**Goal:** Apply all six `tau` values to the hash-verified real raw-gradient snapshot and choose the common PCD base-rate rule before model updates. **Files:** external analysis/results under `C:\Dev\readWFN_share_ms\lap_pcd_runs_20261002\`; planning record. **Depends on:** T4. **Acceptance:** Bind the snapshot SHA and code hashes; report active sets, feasibility, direction norms/cosines, and primary-normalized rescaling versus raw and Nash magnitudes. Predeclare one common LR or a minimal stability calibration for all six screens.

## T6 — Matched screens, panels, and SCF

**Goal:** Train six fresh 25-update PCD candidates and advance at most two or three to cursor 100. **Files:** new external run directories, matched-panel/SCF receipts, final report. **Depends on:** T5. **Acceptance:** Same frozen 27-reaction/15-system panel, seed, 150-step RAdamW/cosine protocol, chunks 256/4096, and declared LR rule; report all three median loss ratios and 100× row ceiling. Run matched panels and route SCFs for advanced candidates; keep chemistry primary and claims descriptive.

## T7 — New world-size-2 manifest and exact CPU resume proof

**Goal:** Validate a separate 150-update world-size-2 stream and two-rank resume for the candidate chosen after local screens. **Files:** new external manifest/receipts; focused distributed tests only if a regression is needed. **Depends on:** T3–T6. **Acceptance:** Manifest has independently generated actual rank streams and distinct canonical/file-byte hashes; never reuse the world-size-1 `550f…` manifest. A tiny CPU `torchrun`/Gloo run uses those exact rank identities and proves globally averaged tasks, EMA/QP/direction, model/optimizer state, and checkpoint-resume parity. Prefer the real model/objectives if practical; otherwise label synthetic fixtures clearly and make no objective-proof claim. Never silently subsample or alter objectives; reuse the existing runtime, Gloo, and per-rank RNG helpers.

## T8 — Final independent review and report

**Goal:** Close the experiment with reproducible evidence and limitations. **Files:** `.planning/pcd/` final review; repository-root `lap_pcd_local_report.md`, `lap_pcd_results.json`, `lap_pcd_protocol.json`; external raw-artifact index. **Depends on:** T5–T7. **Acceptance:** Verify every provenance/hash and cursor, distinguish real-objective from synthetic distributed evidence, answer all 20 user questions, include changed-code LOC and an actual-diff audit, state exact ending cursor(s), selected or unselected `tau`/LR and pass/fail, record SCFs and limitations, and claim no unsupported winner or all-90 coverage. End the report with the exact user-required ready/not-ready PCD sentence. Commit/push only after required tests pass; keep raw logs/checkpoints/cache outside Git.
