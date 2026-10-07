# UNIT_MAXMIN static SVRG K1 ten-update fixed-step qualification

Overall **PASS**; scientific passes **4/4**. Exact gate: each own-baseline maximum of four objective ratios is strictly below1 at t10. No tolerance.

The user prospectively replaced per-update full251 Armijo with the same frozen normalized displacement. The first four seed11 P67 steps were retained after bitwise direct-step replay and exact RNG/cursor/aggregator validation. Historical protocol/harness/checkpoints are preserved externally. No four-step rerun.

Static reference is original t0, never refreshed. Chemistry is full251 reference + current K1 - reference K1, with identical reaction IDs/variants and original n_d/251 weights. AE17, Exc, repaired operator, UNIT_MAXMIN and precision are unchanged. Main F32; chemistry shadow F64.

| Start | Accepted | Rmax5 | Rmax10 | Scientific PASS |
|---|---:|---:|---:|---|
| 11_P67 | 10 | 0.9999213433212162 | 0.9998428667412319 | True |
| 11_P536 | 10 | 0.9999735830828597 | 0.9999263673988517 | True |
| 23_P67 | 10 | 0.999927813257345 | 0.9998539463958817 | True |
| 23_P536 | 10 | 0.9998848678398473 | 0.999771117161251 | True |

| Start | chemistry t10 ratio | AE17 | Exc | operator |
|---|---:|---:|---:|---:|
| 11_P67 | 0.9998428667412319 | 0.9966141180846841 | 0.9966530016101383 | 0.9995090990042677 |
| 11_P536 | 0.9999186314203258 | 0.9975088762074871 | 0.9976435122053082 | 0.9999263673988517 |
| 23_P67 | 0.9998539463958817 | 0.9924111507571328 | 0.992919930823131 | 0.9996173923183114 |
| 23_P536 | 0.999771117161251 | 0.9917592317867232 | 0.9921340873801257 | 0.9995475571543823 |

Actual new chemistry backward calls: **584**, including interrupted safeguard work; reused t0 full251 reference calls: **1004**. Deployment count including one251 refresh per started arm: **1588**, versus10040 for40 exact-full251 updates. Backward-count ratio **0.158167**; this is not a measured wall-clock ratio.

Per-update full251 scalar safeguard removed after cursor4. No measured matched wall-clock speedup; gradients vary in cost. Removed 36 prospective full251 trial scalar passes; checkpoints remain. Count includes interrupted pre-switch work.

Validation:70 immutable source/data/reference/probe-transition hashes match; focused7 tests and relevant90 tests (2 skips) pass; Ruff/compileall pass. Existing checkpoint helper validates source/manifest/protocol before restoration; t5/t10 reload checks model, RNG, cursor and aggregator exactly. Large arrays/checkpoints remain outside Git and are SHA-bound in metrics. No additional audit campaign.

Next step: 25-update UNIT_MAXMIN + static SVRG K1 fixed-step qualification; do not launch

Gold remains PARTIAL PASS / causal validation and was stopped at seed11 P536 cursor6. No gold continuation. Per-update full251 Armijo evaluation was removed; no surrogate loss, new optimizer, adaptive sampler, refresh schedule, new initialization, or25/100-update run was introduced. No full90/SCF/Diet/Slurm work.

External artifacts: `C:\Dev\readWFN_share_ms\lap_unitmax_svrg_k1_runs_20261007`. Protocol SHA: `2e0bd16741b20062cd45ebb7e81179982ad353386e058f78fa85670f72a53643`. Manifest file SHA: `eb64fb2ba9a98eead1dc518a1b0853eeb90d66dcb92c117dbfba23964e6afcfc`.

Fixed-step continuation took approximately 99.2 minutes, measured from protocol freeze to final artifact generation. This includes required scientific checkpoints; it is not a matched timing comparison.
