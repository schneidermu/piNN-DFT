# Canonical chemistry-primary PCD local study

PCD passes implementation and engineering validation, but all six 25-update RAdamW candidates fail the training gate. No tau is selected and no cursor-100 continuation or cluster pilot is recommended. These are matched training-panel results, not held-out accuracy claims.

## Identity and implementation

Branch `lap_full_vxc`; base `f824eec42ba0484b28030316b84558c5243afcde`. PCD integration `0d422fab03ec3551d860c5aa73b038b7103bd2dd`; checkpoint guard `d450bd79cd2800c1bd587cd781554b0aa7307013`; SCF CLI registration `99fd9be9165ec8f5e50c50ee9cc2ea348aa609c6`. Training/DDP receipts retain d450; SCF uses 99fd. Report commit is the subsequent commit containing this artifact.

Authoritative paper: [OpenReview HT01yGHLEt](https://openreview.net/forum?id=HT01yGHLEt), [arXiv 2606.29521v2](https://arxiv.org/abs/2606.29521). Official [author repository](https://github.com/DaraVaram/priority-constrained-descent), pinned SHA `e9afdbc9f4cb09343934eadb31dc72ea5ebac9b0`, MIT; notice preserved in `train_models/PCD_LICENSE`. Full paper/appendices, README, normalize/qp/optim and upstream tests were audited. Upstream 88 tests passed; independent 480-update parity sequences and synthetic KKT/state/resume tests passed.

Installed skills were inventoried. Caveman, Ponytail and Matt Pocock skills were unavailable; no replacements were invented. Installed resume-project and PDF skills were used. Direct reuse-first and concise spec/ticket requirements were followed.

No Python modules were added. Seven production files were modified: moo_aggregators, lap_moo_protocol, lap_moo_training, lap_moo_analysis, run_lap_moo_benchmark, train_lap_moo, run_lap_moo_scf_smoke. Production Python adds 648/deletes 30 lines; tests add 984/delete 4. The license is additional. Independent review found no materially smaller reuse-based implementation preserving fidelity/testability. Existing objective, optimizer, sampling, checkpoint, DDP and AO utilities are reused; historical S5 and architecture remain unchanged.


Input file identities (absolute paths are in protocol/results JSON):

| Input | SHA-256 |
|---|---|
| predopt_fgpu_20261001T192623/lap_pbe_predopt.pt | ed4ba8231d93c376ce5aa8fc81e0670f6a5a44c95c2a552651b12d6a1d4b63f8 |
| mn_group_store_268/manifest.json | ec254952f51d854d8b23c01ff3d316d3b4fd756b58287c637d3f07ed8c385d2a |
| all90/manifest.json | 7005cd869ea8be9636b03f385e7069f4e9023c9582fe7defc5437a7d6609b887 |
| mrks_15system_ao_cache/manifest.json | 60631c23d1683dcf7855ef5897464994addb990c1c8cf33ea2572adbf532a00a |
| lap_moo_runs_20261001/panel_definition.json | f98c112344dc52df401fb5eff50fd8ce609cc9c8f69085223cc9c0dee9c66131 |
| raw_gradient_survey_20261001T/sampling_manifest.json | 666492800bd307efef1a3ea5f9012d3f32218c9fbcfe7ed7ab768635d28112df |
| raw_gradient_survey_20261001T/raw_task_gradients.npz | cc7c2f0de49044dc2bedd98cbd279ea871fd41aca69dc42b14cab978cdb97a2b |

## Canonical contract

Task order is exactly chem, exc, op; chemistry is primary. The same tau applies to E and operator. EMA uses beta=.999, epsilon=1e-8 inside sqrt and shared bias-correction count. Normalize raw gradients, minimize half squared distance to normalized chemistry subject to each normalized secondary dot direction >= tau times its squared norm. Released K=3 active-set/KKT code and 1e-9 tolerance are reproduced. Primary coefficient is one before rescaling; final direction is rescaled to raw chemistry-gradient norm. Zero primary/secondary, rank deficiency and infeasible primary-only fallback follow released behavior. No hidden weights, OMEGA or S5 phases enter PCD.

EMA is updated only after each raw task gradient is globally averaged. Configuration, order and EMA state are checkpointed and fail closed on mismatch before restoration. RNG rank cardinality is validated. Finite epsilon prevents claiming exact independent finite-step scale invariance; the documented asymptotic behavior and released arithmetic are tested. A float32/float64 stationary rounding edge follows official same-platform arithmetic.

Direction sign: positive g_i·d predicts descent under theta <- theta-lr*d; negative g_i·Delta_theta predicts descent for an actual parameter update. Ideal direction/update cosine is -1 (180 degrees). Direction guarantees do not transfer automatically through RAdamW.

## Geometry and local screen

The frozen initial panel is 27 reactions across nine databases and 15 cached mRKS systems. All six tau values were feasible on all 27 initial rows in float32/float64; active sets matched across dtypes. Worst solver slack was -5.56e-17, within declared tolerance. Detailed active-set, multiplier, alignment and margin distributions are in the machine-readable results and hash-bound T5 receipt.

Fresh canonical 268-group PBE predopt is shared by all runs (2 epochs, lr=.01, seed41). Six fresh RAdamW runs used the same one-rank stream, dtype, 150-update cosine horizon, chunks 256/4096 and common predeclared lr=6.632573669086685e-7. The LR calibration constrained update magnitude, not objective quality. All 150 F1 updates were finite and feasible with no fallback. No per-tau tuning was performed.

Ratios below are relative to the same predopt rows, at cursor25:

| tau | chemistry median | E median | operator median |
|---:|---:|---:|---:|
| 0.0 | 9.490181 | 57.291873 | 9.338703 |
| 0.005 | 9.491532 | 57.295101 | 9.331327 |
| 0.01 | 9.492205 | 57.298668 | 9.334956 |
| 0.02 | 9.492881 | 57.308917 | 9.329214 |
| 0.05 | 9.496258 | 57.336598 | 9.338094 |
| 0.1 | 9.501658 | 57.381099 | 9.346874 |

Every tau has 0/27 chemistry wins, 0/27 E wins and 1/27 operator win. Chemistry maxima are about 1298–1300, with two rows above 100. No candidate meets any median-improvement requirement. The unfiltered Pareto front among failed candidates is not a scientific shortlist. All candidates are rejected under the catastrophic-chemistry gate; cursor100 is skipped rather than redefining success.

Exact-budget fixed/Nash cursor25 controls have verified shared predopt rows, panel identities, manifest and chunks. Tau0 PCD/fixed median paired ratios are about 7.94/18.70/10.15; PCD/Nash are 9.85/27.43/11.91. Controls use their historical calibrated LRs, so this is observed protocol performance, not an isolated aggregator causal comparison. Historical cursor100 Nash (2.537/.880/.654) and fixed (1.432/3.057/.900) are context only; no PCD cursor100 comparison is fabricated. Full median/p90/max and paired wins/losses are in results JSON.

## Direction versus finite optimizer steps

All 150 QPs are feasible. Tau0 has 18 inactive and 7 operator-active updates; no E-active update. Tiny float32 emitted operator-dot violations (minimum normalized cosine -1.23e-8) are distinguished from solver margins. Final norms match raw chemistry within about 6.4e-8 relative error. All-tau counts, cosine distributions, multipliers and actual-step dot fractions are stored in results JSON.


Training active-set counts (25 updates per tau; all feasible):

| tau | neither | E only | V only | both | infeasible |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 18 | 0 | 7 | 0 | 0 |
| 0.005 | 14 | 1 | 7 | 3 | 0 |
| 0.01 | 12 | 1 | 10 | 2 | 0 |
| 0.02 | 8 | 3 | 11 | 3 | 0 |
| 0.05 | 3 | 5 | 14 | 3 | 0 |
| 0.1 | 2 | 5 | 15 | 3 | 0 |

RAdamW substantially transforms directions: median direction/actual-delta angle is about 93 degrees rather than ideal 180. Tau0 actual deltas have adverse task dots on 7/25 chemistry, 8/25 E and 8/25 operator updates. Stochastic identities and momentum confound causal attribution.

The optional five-update plain-SGD control uses tau=.02, identical fresh predopt/first stream entries/common LR, no momentum or decay, and no scheduler. Direction/update median cosine is -0.99999984; float32 rounding leaves at most .00447 relative delta residual. Yet the second same-sample AE17/H2 step moves .009 of initial parameter norm and increases chemistry 903.01->51143.72 (56.64x), E 5.174->112.68 (21.78x), while operator decreases. All three local gradient-delta dots were favorable. This directly demonstrates finite-step overshoot even with direction preservation; SGD is not a rescue. RAdamW rotation is a separate observed effect, not the sole explanation. No LR/model/loss redesign is made in this study.

## DDP, SCF and resources

The new WS2/150-update/seed41 manifest has canonical SHA `b1b4c71ae5c8895836c82164b1c2b2b1570e2e7ac072b61fd854f68d8417fd2b`, file SHA `8b059dccef9d46b3991347a0b2d2407e37b2e70781c32c6410b9464d64219f1e`. Each rank covers all 27 reactions and 15 systems; streams differ. The world1 identity is never reused for two ranks.

Engineering-only tau=.02 real CPU Gloo CLI ran two updates using the exact prospective manifest. Local task gradients differ; global averages enter PCD before EMA/QP; both ranks have identical EMA/directions/optimizer/parameters. Segmented/resumed and uninterrupted checkpoint objects and update histories match exactly, including rank RNG states. Serialized checkpoint file bytes differ; byte identity is not claimed. This is a two-update engineering proof, not a 150-update result.

Rejected tau=.02 cursor25 checkpoint SCF on CPU/PySCF2.14 converged for H2/BeH2/CO in 6/8/10 recorded cycles, energies -1.4057941825351212/-17.179448367481868/-118.64658893119076 Ha. All energies/XC outputs finite, vtau exactly zero, exact tau independence passed. These are stability checks, not accuracy rankings.

RTX5070Ti reported 17,094,475,776 total bytes (15.921 GiB). Chunks remain 256/4096; no OOM, timings and peak allocated/reserved bytes are in hash-bound raw logs/receipts. WDDM reserved memory can exceed physical capacity; no V100 capacity/performance inference is made. Central corpus has 90 systems/8,271,091 centers; AO factors cover 15 systems only. Full90 operator training remains unready.

F1 runtime aggregate: {"updates": 150, "median_step_seconds": 15.100740599998971, "total_step_seconds": 2623.3922941996134, "max_peak_allocated_bytes": 10572596736, "max_peak_reserved_bytes": 17278435328}. Peak reserved bytes are reported separately from physical memory.

## Acceptance and final scientific answers

Final source checks: Windows 224 passed/3 skipped; WSL/PySCF/Gloo 123 passed/2 CUDA skips; Ruff, compileall and git diff --check passed. SCF parser registration had a failing regression before its one-line fix, then nine tests passed. Source and report artifacts only are committed; large artifacts remain external and are referenced by SHA-256 in JSON.

1. PCD is numerically faithful to pinned paper/code within released arithmetic/tolerances.
2. Canonical local preference is reproduced, but observed training does not improve all objectives.
3. No tau value/range is supported by this screen.
4. Chemistry does not improve relative to predopt.
5. Integrated E does not improve.
6. Operator median does not improve.
7. No candidate has all three median ratios below one.
8. E-active frequencies are recorded per tau in training_geometry; tau0 is zero.
9. V-active frequencies are recorded per tau; tau0 is 7/25.
10. Both-active frequencies are recorded per tau; tau0 is zero.
11. Infeasibility is zero of 150 training updates and zero of initial panel cases.
12. Feasible solver constraints satisfy tolerance; tiny emitted float32 residuals are reported separately.
13. Initial/training alignment-versus-tau distributions are in results; tau does not rescue the screen.
14. RAdamW materially rotates the observed update geometry; theoretical PCD guarantees are not optimizer guarantees.
15. SGD preserves direction but demonstrates finite-step overshoot, so optimizer rotation alone cannot explain failure.
16. Nash performs better on the observed matched cursor25 comparison.
17. Fixed scalarization also outperforms PCD here; neither control is claimed to solve all objectives.
18. The world-size manifest mismatch is corrected with a distinct immutable WS2 artifact.
19. Exact real two-rank engineering execution/resume passes for two updates.
20. No scoped 2xV100 scientific pilot is recommended. A future separately specified step-size study is required before a new candidate; the existing engineering launch receipt retains reproducible commands but is not a selected cluster protocol.

No Diet30/Diet100 evaluation, full90 AO generation, held-out WTMAD claim, or production-scale training was performed. No production training jobs or Slurm jobs were submitted.

Canonical chemistry-primary PCD is not yet ready for a scoped 2×V100 diagnostic pilot.
