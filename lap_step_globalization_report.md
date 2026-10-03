# Step-globalization run report

**Status: Results finalized; the proposed next experiment is unrun.**

Both Arena rounds used four Luna MAX solvers and a Luna MAX judge. The first selected vector Armijo; the second selected the unrun common-descent diagnostic recorded in `.planning/step_globalization/next_arena/decision.md`. Arena, Caveman, Ponytail, and Matt Pocock skills were unavailable, so the Arena/specification/ticket/reuse workflow was followed manually.

## Result

The known-second-step T2 diagnostic passes its frozen-batch gate. On AE17 reaction 16 / H2, the unchanged full proposal reproduces the historical overshoot: chemistry and E_xc losses rise to 56.6367x and 21.7783x baseline while the operator loss falls to 0.447867x. At j=5, t=1/32 and alpha=2.072679271589589e-8, all three losses decrease and pass componentwise Armijo. The exact same PCD direction also passes at j=6 and j=7; the prescribed largest passing fraction remains j=5. Nonselecting probes j=8 and j=9 have all actual-to-predicted ratios within the predeclared 10% linear window. Trial state restoration is exact; the diagnostic makes no optimization commit.

| j | alpha | parameter step norm | actual/predicted: chem | E_xc | operator | all decrease + Armijo |
|---:|---:|---:|---:|---:|---:|:---|
| 0 | 6.632573669e-7 | 0.1923370650 | -0.900766 | -0.857786 | 0.526106 | no |
| 1 | 3.316286835e-7 | 0.0961685312 | -0.912624 | -0.815201 | 0.747242 | no |
| 2 | 1.658143417e-7 | 0.0480842634 | -0.862273 | -0.662802 | 0.871687 | no |
| 3 | 8.290717086e-8 | 0.0240421339 | -0.737682 | -0.336767 | 0.935787 | no |
| 4 | 4.145358543e-8 | 0.0120210650 | -0.480492 | 0.322214 | 0.968031 | no |
| 5 | 2.072679272e-8 | 0.0060105342 | 0.036868 | 0.999474 | 0.983953 | yes; selected |
| 6 | 1.036339636e-8 | 0.0030052670 | 0.999783 | 0.999760 | 0.992041 | yes |
| 7 | 5.181698179e-9 | 0.0015026336 | 1.000105 | 0.999893 | 0.996525 | yes |
| 8 | 2.590849089e-9 | 0.0007513171 | 1.000749 | 0.999926 | 1.000495 | yes; 10% window |
| 9 | 1.295424545e-9 | 0.0003756595 | 0.999461 | 1.000051 | 0.998669 | yes; 10% window |

The adaptive direct-step branch is a separate trajectory. It accepted two updates, then stopped before trying update 2. The first two accepted rows are:

| update | t | alpha | backtracks | losses before (chem, E_xc, op) | losses after (chem, E_xc, op) | actual/predicted (chem, E_xc, op) |
|---:|---:|---:|---:|:---|:---|:---|
| 0 | 1 | 6.632573669e-7 | 0 | (92.128960, 24.721813, 0.03329886) | (91.613091, 24.091498, 0.03328860) | (0.884743, 0.999061, 0.949531) |
| 1 | 1/32 | 2.072679272e-8 | 5 | (903.013306, 5.173923, 0.05064968) | (838.752869, 1.259455, 0.04901511) | (0.036868, 0.999474, 0.984024) |

At cursor 2, for NCCE31 reaction 0 / BH / level2_mura, the canonical PCD proposal slopes are chemistry +0.0093064049, E_xc -0.4226231444, and operator -0.0000329078. No line-search trial or update-2 commit occurred. A read-only replay matched the stop losses and left model parameters/buffers, RNG, and EMA unchanged. Raw gradient cosines are chem:exc -0.685114, chem:op -0.560289, and exc:op 0.931266. A separate raw-gradient simplex calculation finds a strict common-descent witness with weights (0.001419116, 0, 0.998580884), directional dots (-0.091940, -879.071583, -0.091940), and maximum KKT violation 2.8e-17. The canonical PCD ray fails the common-descent test at this cursor, but another common-descent direction exists. This feasibility result alone does not show that a finite Armijo step is representable or useful.

## Cursor-2 panel screen

Ratios are relative to predopt on the fixed training diagnostic panel: 27 reaction rows across 15 cached systems. The summaries have n=27 rows; they are not 27 independent molecules.

| objective | median | p90 | max | rows below 1 |
|:---|---:|---:|---:|---:|
| chemistry | 1.014520 | 2.460416 | 35.092074 | 11/27 |
| E_xc | 0.648665 | 0.813846 | 0.821546 | 27/27 |
| AO operator | 0.955113 | 0.963036 | 0.967696 | 27/27 |

Chemistry misses the all-objective panel gate. This is a cursor-2 screen after two accepted updates, not a 5/10/25/100 result or a population/held-out claim. The historical five-step fixed-SGD control remains distinct from this adaptive trajectory. Although the canonical 90-reaction central corpus contains 8,271,091 centers, only 15 AO systems are cached for this diagnostic; these results do not establish full-90 operator-training readiness. No 5/10/25/100 progression, completed training success, new SCF shortlist, Diet/cluster run, or Slurm job exists.

## Answers to the 16 closeout questions

1. **Primary overshoot hypothesis?** Supported for the known AE17/H2 replay: the unchanged full step reproduces the chemistry/E_xc spike while all objectives have negative base-ray slopes.
2. **Same-ray rescue?** Yes. The largest tested passing step is j=5 (t=1/32), with strict decrease and Armijo for all three losses.
3. **First-order alpha range?** alpha0=6.632573669e-7; the measured curve covers j=0 through 9, down to 1.295424545e-9.
4. **Armijo reliability?** Each accepted trial uses the same frozen sample and all three original raw losses. At the accepted j=5 point all three measured Armijo margins are positive; rollback was exact.
5. **Accepted-alpha distribution?** The adaptive branch has only two accepted steps: t={1, 1/32}, equivalently alpha={6.632573669e-7, 2.072679272e-8}. This n=2 record is not a population estimate.
6. **Backtracks?** The two direct updates used 0 and 5 halvings. The known-step curve first passes at j=5; j=6/7 also pass but are smaller.
7. **Did chemistry improve on the panel?** Not in aggregate: median ratio 1.014520; only 11/27 rows are below 1.
8. **Did E_xc improve?** Yes on this screen: median 0.648665 and 27/27 rows below 1.
9. **Did operator loss improve?** Yes on this screen: median 0.955113 and 27/27 rows below 1.
10. **Are all three medians below 1 by update 25 or 100?** Unknown; no such checkpoint was produced.
11. **What causes the remaining risk: curvature, variance, or conflict?** Evidence supports finite-step overshoot on one fixed sample and a conflicting canonical PCD ray at cursor 2. These runs do not isolate population variance or establish a general cause.
12. **Is PCD a useful direction?** It was common descent on accepted updates 0/1; at cursor 2 its selected ray was not, although a distinct strict common-descent witness exists.
13. **Is BBDMO justified now?** Not by these results: only two adaptive updates and one small-step rescue are available, with no evidence of persistent tiny-step behavior.
14. **Is a trust-region fallback justified?** No. The recorded proposal selects one frozen-state common-descent direction diagnostic with existing vector Armijo. Its only conditional follow-up is a full-cycle training-gradient batch with canonical PCD unchanged if the batch step passes but the paired panel fails the strict all-three gate. Every objective must strictly improve; chemistry is the primary outcome among successful three-objective solutions.
15. **Is a larger balanced batch justified now?** Not yet. The panel flags chemistry as unresolved but does not identify sampling variance as the cause.
16. **What is the single next experiment?** A frozen-state Fliege–Svaiter common-descent direction diagnostic at cursor 2, using the native trial sequence t=1, 1/2, … and the existing componentwise vector Armijo gate. Do not reuse PCD's alpha0 for this native direction. It is proposed and unrun.

## Protocol, runtime, and validation

The proposed native direction uses the Fliege–Svaiter trial sequence starting at t=1, halved geometrically; PCD's calibrated alpha0 is not reused. The componentwise Armijo values c=1e-4 and rho=0.5 are supported numerical choices from Mita, Fukuda & Yamashita (2019, [DOI](https://doi.org/10.1007/s10898-019-00802-0); [author preprint](https://optimization-online.org/wp-content/uploads/2018/09/6804.pdf)), not Fliege–Svaiter defaults. The existing implementation's max-20-halving finite-precision/runtime guard bounds the proposed diagnostic; it is not a paper parameter. The native direction comes from Fliege & Svaiter (2000, [DOI](https://doi.org/10.1007/s001860000043)). PCD remains chemistry-primary with tau=0.02, EMA beta=0.999, epsilon=1e-8, and QP tolerance 1e-9. A fixed-batch Armijo result does not establish population generalization or transfer the Fliege–Svaiter global convergence theorem to PCD's different direction subproblem.

Runtime: seed 41, float32, world size 1, CUDA 12.8 / PyTorch 2.11.0, NVIDIA RTX 5070 Ti, grid chunks 256 and AO-cache chunks 4096. Direct updates took 17.97 s and 22.60 s; the cursor-2 panel took 332.90 s. The recorded run peak was 10.57 GB allocated and 14.69 GB reserved. T2 curve and j=8/9 extension took 26.67 s and 11.62 s.

Validation: Windows tests 194 passed/3 skipped; WSL PySCF tests 90 passed/2 skipped; 16 independent Armijo tests passed, including two-rank CPU Gloo trial-mean/resume coverage; Ruff, compileall, and git diff checks passed. The validated implementation is commit e017768329a15f1389a6fdf94bc8fdb4c647f344 on base b20d153c175a9d8e2d762b5e3ccf2473d55509ac. The five production files are `train_models/evaluate_lap_moo_checkpoint_panel.py`, `train_models/lap_moo_protocol.py`, `train_models/lap_moo_training.py`, `train_models/run_lap_moo_scf_smoke.py`, and `train_models/train_lap_moo.py` (+521/-65). The two test files are `train_models/test_lap_moo_armijo.py` and `train_models/test_lap_moo_training.py` (+1004/-4), total seven files (+1525/-69). This report work adds no source or test edits. No new production module or trainer was added. The independent reuse review found no materially smaller implementation that preserves the required Armijo and state-fidelity coverage.

The direct run used a dirty tree at b20d153. Four direct-run source hashes match the validated final source. The direct run's train_lap_moo.py hash was 381675d2c7f5793479dd46dae31e7a9cafe57efa2c4759c12a9d3947b882cd20; the later file hash 521f6bfa9fdd28c07649c449e0f3b6ea582170df9b02dfef64c08283b13b9643 reflects subsequent stop-receipt logging changes only. Checkpoint hardening was included before the frozen run. Full source, immutable data identities, and receipt SHA-256 references are in the protocol artifact; measured trial details are in the results artifact.

The decision record is `.planning/step_globalization/next_arena/decision.md`. The proposed cursor-2 diagnostic, all panel gates beyond cursor 2, 5/10/25/100 progression, full-90 operator-training readiness, SCF shortlist, and Diet/cluster/Slurm execution remain unrun.

Finite-step globalization did not solve the current training failure; the evidence supports a frozen-state Fliege–Svaiter common-descent direction diagnostic with vector Armijo.
