# Matched LR-only AdamW branches from preserved t70

Exactly **40 new optimizer updates** completed: B at 3e-5 and C at 1e-5, 20 each. Both resumed the identical original t70 checkpoint with its AdamW moments, counters and RNG. Control A was reused read-only. No production trainer, objective, architecture, precision, dataset, coefficients or sampling changed.

Reduced LR substantially limits the control's validation deterioration, but neither branch stays below t70 at both measured checkpoints. Best new Clean28 is B/t80, **8.619694172 kcal/mol**, only 0.009966059 below t70. It is **not scientifically eligible**, because exact relchem/t0 remains 1.034869578. No checkpoint is promoted.

## Frozen comparison

Native F32 AdamW, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01, foreach=False; no scheduler. F64 scientific gradients and aggregation, single F32 optimizer-boundary cast; Exc chunk4096, operator chunk256, existing bounded chemistry chunks. Coefficients remained relchem=0.017015480965588553, AE17=5.141254618347414e-05, Exc=1.5094644512009712e-05, operator=0.33597561607048215.

The native trainer restores optimizer LR from the saved checkpoint, so the experimental wrapper overrides only that parameter-group field **after native restore**. Full optimizer-state equality is checked against a copy with only LR changed. No optimizer reset or scientific code patch. A scoped diagnostic hook archives the actual rounded parameter step; it does not alter the update.

Stage A completed before either branch trained. The original first70 logs remain identical, all20 new sample rows match the control, and first-update raw gradients match control **bitwise for all four tasks**. All checkpoint moment counters equal their cursor. A native AdamW resume test verifies preserved moments/RNG, the allowed LR change and proportional first displacement within F32 rounding tolerance.

## Clean28

Diet30-clean weighted error is the mean of 28 Diet-weighted absolute errors, **not canonical full30 WTMAD-2**. Same immutable PBE0 densities, PBE0-D3(BJ), split, references and weights. Full30 remains diagnostic, selection_allowed=false. All28 signed errors, predictions, absolute errors, weights and contributions are preserved in the JSON; their sum reconstructs every score within 1e-12.

| Arm | LR | t70 | t80 | t90 | t90 minus t70 | t90 minus control |
|---|---:|---:|---:|---:|---:|---:|
| A control, reused | 1e-4 | 8.629660230 | 8.649173724 | 9.273578777 | +0.643918547 | 0 |
| B | 3e-5 | 8.629660230 | 8.619694172 | 8.793220625 | +0.163560395 | -0.480358152 |
| C | 1e-5 | 8.629660230 | 8.623672178 | 8.657308152 | +0.027647922 | -0.616270625 |

B and C reduce the final control deterioration by 74.6% and 95.7%, respectively. Both improve at t80 and then worsen at t90; neither has sustained improvement below t70. C retains the initial validation region most closely. These are two correlated observations of one training seed, not evidence of convergence or reproducibility.

| Checkpoint | Improved / worse reactions vs t70 | Improved reactions vs matched control |
|---|---:|---:|
| A80 | 14 / 14 | — |
| A90 | 7 / 21 | — |
| B80 | 17 / 11 | 15 / 28 |
| B90 | 7 / 21 | 21 / 28 |
| C80 | 18 / 10 | 14 / 28 |
| C90 | 10 / 18 | 21 / 28 |

B80's largest improvements vs t70 are G21EA-14 (-0.007715919 score units), BHROT27-26 (-0.005017982), BSR36-31 (-0.004839676), HEAVY28-16 (-0.003782271). They are offset by PX13-9 (+0.006997481), BHPERI-11 (+0.004885297), FH51-24 (+0.003802910), SIE4x4-15 (+0.003094793). The small net gain spans multiple reactions but results from cancellation, not uniformly better chemistry. At t90 the reduction of control regression is also distributed: 21/28 reactions improve relative to control in each reduced-LR branch. No score is selected from full30 or a single reaction.

## Exact scientific audit

Independent evaluation manifest reused unchanged: **251 relchem + 17 AE17 identities, exactly one fixed variant each**; all90 mRKS Exc/operator systems, equal-system means. No exhaustive variant evaluation and no parameter backward. Original t0, t59 and t90 receipts reused. New audits: control t70/t80, and only best-new B80. No full90 evaluation of B90/C80/C90 was added; their scientific eligibility remains unknown.

| Checkpoint | relchem | AE17 | Exc | operator |
|---|---:|---:|---:|---:|
| t0 | 1.235471097887 | 24.901047104017 | 92.174805020006 | 0.033148216816 |
| A59, reused | 1.252015550937 | 6.475281568806 | 25.887655099411 | 0.032080681503 |
| A70, new | 1.290582966818 | 6.430294585569 | 18.130764080158 | 0.031215412268 |
| A80, new | 1.259116438808 | 8.498835307756 | 33.333456770388 | 0.031382493995 |
| A90, reused | 1.267924808154 | 15.354135667336 | 51.058492461595 | 0.031142655452 |
| B80, best new | 1.278551453814 | 2.010210533834 | 5.443669117210 | 0.031294227278 |

| Checkpoint | relchem/t0 | AE17/t0 | Exc/t0 | operator/t0 | Eligible, all ratios <1 |
|---|---:|---:|---:|---:|---|
| A59 | 1.013391210 | 0.260040533 | 0.280853918 | 0.967795091 | No |
| A70 | 1.044607979 | 0.258233903 | 0.196699782 | 0.941692051 | No |
| A80 | 1.019138724 | 0.341304334 | 0.361633060 | 0.946732495 | No |
| A90 | 1.026268288 | 0.616606025 | 0.553931114 | 0.939497157 | No |
| B80 | 1.034869578 | 0.080727952 | 0.059058103 | 0.944069705 | No |

B80 relative to t70: relchem=0.990677459, AE17=0.312615621, Exc=0.300244882, operator=1.002524875. Thus relchem improves slightly from t70 but remains above t0, while operator rises slightly from t70 but remains below t0. Compared with A80, B80 has better Clean28/AE17/Exc/operator and **worse relchem**. Validation and scientific-objective progress are distinct.

## Actual AdamW movement

Norms below use unique trainable parameters once, in the canonical 9446-coordinate order. Actual B/C step vectors are SHA-bound F64 representations of the rounded F32 parameter differences, not native gradients or hypothetical LR-scaled updates. Direct flattened task-dot checks agree with native instrumentation at rtol=1e-10, atol=1e-12; norm checks use rtol=atol=1e-12. Original control instantaneous vectors were not archived, so control was not replayed: its saved step norms/task directional progresses are reused and exact net vectors come from preserved t80/t90 checkpoints.

| Arm | Median step norm | Maximum step norm | Net displacement t80 from t70 | Net displacement t90 | t90 net cosine to control |
|---|---:|---:|---:|---:|---:|
| A | 0.002417040 | 0.003492253 | 0.016786538 | 0.027926559 | 1 |
| B | 0.000733271 | 0.001138780 | 0.005390314 | 0.008408157 | 0.973620 |
| C | 0.000250768 | 0.000320221 | 0.001821569 | 0.003397762 | 0.728801 |

Final displacement B/A=0.3011 and C/A=0.1217. Sum of actual step norms: A=0.049254503, B=0.015201296, C=0.004991438. Reducing LR chiefly reduces movement, but evolved gradients/moments also change directions. Median raw weighted-gradient cosine to control is 0.999620 (B) / 0.997832 (C), with minima -0.956749 / -0.944805. B/C actual-step cosine has median0.991178 and minimum0.483337. This is not an exact rescaling of the whole trajectory; later task-gradient sign changes are retained rather than hidden by averages.

| Optimizer update (one-based) | Relchem DB | A raw relchem norm | B norm | C norm | A step norm | B step norm | C step norm |
|---|---|---:|---:|---:|---:|---:|---:|
| 75 | PA8 | 217.603879 | 217.648546 | 217.661138 | 0.002309682 | 0.000693194 | 0.000231094 |
| 77 | PA8 | 331.036449 | 331.333484 | 331.416890 | 0.003114580 | 0.000933256 | 0.000310975 |
| 82 | EA13 | 208.010960 | 208.503622 | 208.643315 | 0.002934027 | 0.000773998 | 0.000271144 |
| 86 | EA13 | 129.590831 | 129.783508 | 129.909423 | 0.002779897 | 0.000911844 | 0.000248544 |
| 88 | EA13 | 188.982880 | 189.729135 | 189.948106 | 0.003492253 | 0.001138780 | 0.000320221 |

Raw relchem peaks persist under lower LR. A large raw norm does not by itself determine an AdamW step or identify a cause of validation regression. The matched comparison supports an LR/path-length effect, not attribution to PA8 or EA13 alone. Every update's four raw norms, weighted norm, step norm, displacement, directional progresses, model hashes and timings are in the metrics. Positive task progress predicts descent under the actual update; no Pareto gate was applied.

## Runtime and integrity

| Arm | New updates | Logged training-stage seconds | Seconds/update | Peak live CUDA GiB | Peak reserved GiB |
|---|---:|---:|---:|---:|---:|
| A, historical | 0 | 329.642 (reused 20-update segment) | 16.482 | 13.119 | 27.328 |
| B | 20 | 322.969 | 16.148 | 13.115 | 19.826 |
| C | 20 | 378.000 | 18.900 | 13.115 | 19.826 |

New training-stage total **700.969s (11.68min)**, excluding checkpoint/startup/evaluation overhead. Added native diagnostics took 0.427s in B and0.368s in C. Read-only scientific/Clean28 evaluation times are separately retained in endpoint receipts. Timing variation is not evidence that LR changes computation cost. Reserved memory is allocator accounting, not live device allocation.

No nonfinite gradients, parameters, moments, invalid states or OOM occurred. Checkpoint/model SHA chains, all sample rows, moment counters, RNG presence, finite values, source hashes and read-only endpoint state preservation passed. Original historical receipts/checkpoints were verified byte-for-byte before and after each stage.

| New checkpoint | SHA256 |
|---|---|
| B80 | 694460c08c5831359bffebeb14c6afb9171360b99c2f7ae4818ec52fda8d6b71 |
| B90 | dfe73074225f5e052ba47b561869fdbdb126e6e32bf292a17f4fb92a14326cbd |
| C80 | 178a2a2703b63be2f033b7e23aaed5cf4dcedce7f9e96910330abe8105abe7d6 |
| C90 | 2fa13b167e1fed17d4c3fb04555230ee053357eff452b20c88022927d7b80651 |

Shared source t70 SHA: `04b8e549c17375988be576e2878881d13f55b217b42ec16a7e6101fdddd05443`.
Common rows70–89 SHA: `f5b13371307dc26613254cf0c6a705ff152efb21762c6f4fd1a6988233a684e0`.
Evaluation manifest SHA: `132bd550161be6bc6761f8df1de8d3755c0b0debab53663208c3a9bbf56cb805`.
Dataset logical SHA: `61c221a19b9987717e69cac182ad545241f8807db4126c0949a99992e4c210ef`.

Full protocol, trainer/evaluator/physics source hashes, checkpoint/model hashes, all endpoint receipts and reaction contributions are in `iid_adamw_lr_stabilization_metrics.json`. Large checkpoints/raw gradients/actual steps and checkpoint manifest remain outside Git at `C:/Dev/readWFN_share_ms/lap_iid_adamw_lr_branches_20261009`.

## Questions and decision

1. **Q1:** Lower LR mitigates rather than completely prevents deterioration. C has only +0.027648 drift by t90 versus +0.643919 in control.
2. **Q2:** Both improve below t70 at t80: B by0.009966 and C by0.005988 kcal/mol. Neither remains below it at t90.
3. **Q3:** The best new point is isolated within the two measured post-start checkpoints. The broader favorable validation region is retained more closely by C, not proven converged.
4. **Q4:** Best-new B80 fails relchem eligibility; other three ratios pass. Control t70/t80 also fail relchem. C scientific endpoints were not evaluated and must not be called eligible.
5. **Q5:** Improvements affect multiple reactions (17/28 for B80, 18/28 for C80 vs t70); small aggregate gains contain compensating errors. Both reduced-LR t90 points beat control on21/28 reactions.
6. **Q6:** Smaller actual steps/path length explain much of the stabilization; later directions also diverge. No single high-norm sample is established as the cause.
7. **Q7 / one next experiment:** A **single independent sampling-stream confirmation of LR=1e-5 from the same preserved t70, bounded to20 updates**, with Clean28 at10/20 and one final fixed-variant/full90 scientific endpoint, is the only suggested follow-up. It tests whether C's stabilization persists beyond the matched stream. Keep all coefficients/scientific settings fixed and preserve optimizer moments. It is a diagnostic confirmation, not production promotion, and was not launched. A longer unchanged run or 2xV100 deployment is unsupported while relchem eligibility remains unresolved.

Ponytail review: qualified trainer/evaluator reused; only LR-resume wrapper and receipt analysis added, no new training framework, optimizer or expensive extra audit. Pocock review: only LR changes after exact restore; sample identity, first gradients, moments/counters, RNG, fixed variants, source SHA, native displacement and exact reaction-score reconstruction checked. MOO/numerical review: AdamW movement distinguished from gradient magnitude; task progress uses the actual rounded step; control vectors were not invented; no per-step common-descent requirement. Scientific review: Clean28 and scientific eligibility are separate; one seed/correlated checkpoints and unmeasured C endpoints remain explicit.

Validation: **20 focused tests PASS**, including native AdamW deterministic resume/moments/RNG and no-control-replay guards; Ruff (E402 ignored for existing executable-script import setup), compileall and git diff --check PASS. Receipt-only CPU BLAS uses sequential MKL to avoid duplicate OpenMP loading on Windows; it does not affect trained/evaluated scientific artifacts.

The experiment stopped after exactly40 new updates, with B/C at t90. No historical rerun, optimizer reset, dataset/architecture/physics/precision change, exhaustive chemistry-variant evaluation, SCF, future-test evaluation, Slurm submission, V100 launch or autonomous continuation occurred.
