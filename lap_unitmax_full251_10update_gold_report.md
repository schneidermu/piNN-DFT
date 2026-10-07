# Exact full251 UNIT_MAXMIN gold control — partial qualification

**PARTIAL PASS / causal validation.** The latest user instruction ended gold continuation at the next safe checkpoint: seed11 P536 cursor6. Seed11 P67 remains scientifically passed at cursor10, and both seed23 gold arms remain unstarted. This is not full four-arm scientific qualification. The user now authorizes only static SVRG K1 qualification.

Starting commit: `789d04d0638aa9ded44f0e9012cb5b4fc584a235` on `lap_full_vxc`.

Only the experimental chemistry factory was replaced by the existing qualified full251 evaluator. Its scalar is the arithmetic mean of251 corrected singleton `batch_fchem` losses with matched-F64 chemistry. The same evaluator supplies the gradient and real chemistry Armijo scalar. UNIT_MAXMIN, direct vector Armijo, AE17, Exc/operator, source precision, initialization and manifest remain unchanged. No production source changed.

Frozen protocol SHA: `284074b3919fdd262bc75927e6622bdd472a123be88ffd2ad6a9853cd341898c`. Manifest canonical SHA: `2e87f7e79a66b425628bdf59a821f785e05059f193a492d4a4f3130d7411e9f3`; file SHA: `eb64fb2ba9a98eead1dc518a1b0853eeb90d66dcb92c117dbfba23964e6afcfc`. Entries0–9 only; no entry10 stress probe.

## Completed monitoring

| Start | t | Full251 | AE17 objective | Exc mean15 | Operator mean15 | Rmax |
|---|---:|---:|---:|---:|---:|---:|
| 11_P67 | 0 | 1.237027054764855 | 29.08069447031722 | 80.9058999863264 | 0.04142061014895267 | 1 |
| 11_P67 | 5 | 1.236929755908782 | 29.03145337689741 | 80.77181849480051 | 0.04141132948425939 | 0.9999213446013988 |
| 11_P67 | 10 | 1.236832675408679 | 28.98223190275152 | 80.63500184299656 | 0.04140027517367099 | 0.9998428657195269 |
| 11_P536 | 0 | 1.237333214793654 | 24.87457454408775 | 69.98481862305971 | 0.03686135457451291 | 1 |
| 11_P536 | 5 | 1.237281109633465 | 24.84044190877502 | 69.89461031515985 | 0.03686038085217681 | 0.9999735841954986 |

## Endpoint ratios and status

| Start | Accepted | Status | Chemistry | AE17 | Exc | Operator |
|---|---:|---|---:|---:|---:|---:|
| 11_P67 | 10 | SCIENTIFIC-PASS | 0.9998428657195269 | 0.9966141603782469 | 0.9966516886484715 | 0.9995090614259773 |
| 11_P536 | 6 | USER_SAFE_STOP_CURSOR6 | not reached | not reached | not reached | not reached |
| 23_P67 | 0 | NOT_STARTED | not reached | not reached | not reached | not reached |
| 23_P536 | 0 | NOT_STARTED | not reached | not reached | not reached | not reached |

All16 committed updates accepted t=1 with zero backtracks and strict four-task decrease/Armijo. No step collapse or solver failure was observed. Both update0 full251 gradients matched their prior stored gold arrays bit for bit. Cursor5/10 resume checks were exact for the completed checkpoints.

After the authorized resume, seed11 P536 update5 completed at full displacement with zero backtracks. The latest user instruction stopped gold at its saved cursor6 checkpoint, before any trial at update6. The process exited and is no longer running. No claim is made about unsaved in-memory RNG after process termination.

## Frozen displacement and gamma

| Start | eta0 | Gamma min / median / max |
|---|---:|---|
| 11_P67 | 7.2568949448274828e-06 | 0.243492130147 / 0.244790126535 / 0.245376823941 |
| 11_P536 | 6.3835419098504467e-06 | 0.138560529912 / 0.147717168728 / 0.245486648165 |
| 23_P67 | 8.5392826698708043e-06 | not started |
| 23_P536 | 1.0204187208883716e-05 | not started |

## Validation and review

All61 frozen hash checks matched before/after; zero mismatches. Artifact-only F64 vector reconstruction exactly matches every logged progress, and replay of all15 F32 accepted steps matches saved model hashes bit for bit. Logged trial restorations are exact with maximum parameter difference0. Large checkpoints/gradients remain external and SHA-bound in the metrics.

The existing helpers save `latest.pt` directly rather than by atomic replacement. Resume provenance validation is checked before mutation; completed cursor0/5/10 copies are retained. This run does not claim crash-safe recovery from a partially written checkpoint.

Validation receipt: `{"tests": {"passed": 141, "skipped": 3, "scope": "existing MOO, PCD, Armijo, trajectory and diagnostic groups plus three full251 integration tests; Windows/Gloo platform skips"}, "ruff": "PASS", "compileall": "PASS", "hash_checks": 61, "hash_mismatches": 0, "exact_accepted_state_replay": "PASS all15", "resume": "exact cursor5/10 identity checks PASS", "process": "stopped at user request; no Python process remains", "git_diff_check": "PASS"}`.

Independent partial-scope review PASS for the earlier first15 updates; the safe cursor6 stop adds one accepted update outside that review scope. Overall scientific qualification remains incomplete. Receipt: `C:\Dev\readWFN_share_ms\lap_unitmax_full251_gold_runs_20261007\review.json`; SHA: `6d720726d46de7cdd79bfbe4ada02b6e5362a54bd524d5a660b66a1ae736a486`. The reviewer independently reconstructed the accepted states/progress and checked both available panels, exact cursor/resume identity and the unstarted seed23 arms.

Ponytail: only a narrow experimental full251 factory adapter; no new optimizer or production behavior. Pocock: fresh SHA-bound starts, own fresh denominators, unchanged unit direction/eta budgets, exact full251 gradient/scalar identity, exact accepted-state replay, no skipped/restarted failures. One arm passes; incomplete arms are retained explicitly rather than promoted.

## Decision

Gold qualification is **PARTIAL PASS / causal validation**, not full four-arm scientific qualification. The user stopped it safely at seed11 P536 cursor6. No remaining gold work is authorized. The next experiment is canonical static SVRG K1 under the separately frozen protocol.

Only the current qualified15-system operator panel was monitored; this is not full90 readiness. No SVRG, new optimizer, new initialization, adaptive sampler, new line search, or25/100-update extension was introduced. Two trajectories were started: one completed ten updates and one stopped safely after six at the user’s request. The two seed23 trajectories were not run.

Artifacts: `C:/Dev/readWFN_share_ms/lap_unitmax_full251_gold_runs_20261007`; see the machine-readable metrics for exact checkpoint, geometry, monitoring and starting-state SHAs.
