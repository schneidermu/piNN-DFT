MECHANISM-ONLY — physical variance reduction confirmed without model promotion

One matched R0 continuation from B8 t128 with inherited AdamW moments. Scalar loss, LR, chemistry stream, parameter precision, and full weak-form AO operator are unchanged. Only physical gradients average four prefix presentations. The control is matched by optimizer updates and chemistry exposures, not compute.

| Task | Singleton vector MSE | PHYS4 vector MSE | Reduction |
|---|---:|---:|---:|
| ae17 | 19071091.66 | 6346725.628 | 66.7207% |
| exc | 148671737.8 | 40930577.75 | 72.4692% |
| op | 0.1375789508 | 0.03237555147 | 76.4677% |

Weighted physical MSE: 0.1220804598214209 → 0.04005805851091838 (67.1872% reduction). Composite MSE: 0.18464974513350546 → 0.10168189963100702 (44.9326% reduction).
The mean is preserved to <4e-16 L2. Averaging cannot increase physical MSE in exact arithmetic; that inequality alone is not evidence of better optimization. Chemistry/physical covariance trace changes from -0.005360956912 to -0.005833679008; both are included in composite MSE.

Top-one/top-five physical outliers account for 4.46%/16.35% of singleton MSE. Four groups contain three unique mRKS systems, 124 contain four. Duplicate-pair covariance contributes AE17 140718.955, Exc 2807181.188, operator 0.001465754; distinct-offset covariance also matters (see duplicate_covariance_attribution.csv). Reductions are not dominated by one outlier.

84 AE17 identity+variant records (17 identities) and 90 mRKS systems were computed once at frozen B8. Their empirical prefix means are distinct from uniform full-corpus gradients. Reference-population cosines are AE17 0.999998965, Exc 0.999996323, operator 0.999996021; relative vector differences are 3.319%, 0.886%, 0.642%. AE17 additionally differs in variant population.

Independent scalar-autograd parity passed its frozen native-precision bounds. AE17 relative L2 error 2.56e-16, Exc 2.50e-7, operator 1.18e-7. The latter two are native F32 leaf accumulation, not claimed bitwise F64 derivative parity. Chemistry live/cache relative L2 error 1.34e-12. Native state-restoration and source-position-zero parity passed.

| Cursor | Legacy relchem | AE17 | Full90 Exc | Full90 operator | JW |
|---|---:|---:|---:|---:|---:|
| 0 | 1.1959392842 | 3.2754649558 | 14.6790449900 | 0.0309331247 | 0.1564146381 |
| 64 | 1.2624269213 | 1.6191409517 | 5.9796453017 | 0.0295683176 | 0.1468002220 |
| 128 | 1.3093108886 | 2.5890618480 | 5.2020219246 | 0.0294180246 | 0.1422262498 |

| Objective | PHYS4 t128 | P536 ratio | Change from B8 | Change from F010 t128 |
|---|---:|---:|---:|---:|
| relchem | 1.3093108886 | 1.0597665059 | +0.1133716043 | +0.0029044645 |
| ae17 | 2.5890618480 | 0.1039740151 | -0.6864031078 | +0.3378611379 |
| exc | 5.2020219246 | 0.0564364842 | -9.4770230654 | +0.3221241213 |
| op | 0.0294180246 | 0.8874692961 | -0.0015151001 | -0.0000308283 |

Scientific eligibility: FAIL. Clean28: NOT EVALUATED — scientific eligibility failed. Historical best: NOT TESTED.

| Database | PHYS4 MAE | F010 t128 MAE | Difference | PHYS4 RMSE |
|---|---:|---:|---:|---:|
| ABDE4 | 4.450840 | 4.440383 | +0.010457 | 5.281044 |
| DBH76 | 7.170412 | 7.181423 | -0.011011 | 8.540359 |
| EA13 | 3.363128 | 3.352419 | +0.010709 | 3.951698 |
| IP13 | 3.081194 | 3.076312 | +0.004883 | 4.323818 |
| MGAE109 | 10.737501 | 10.719510 | +0.017991 | 13.724935 |
| NCCE31 | 0.593539 | 0.594431 | -0.000892 | 0.907150 |
| PA8 | 3.466553 | 3.430477 | +0.036076 | 3.855824 |
| pTC13 | 8.624308 | 8.589022 | +0.035285 | 9.237444 |

Chemistry MAE 8.536568 → 7.449999; F010 t128 7.441849. Macro database MAE 5.185934. JW PHYS4 0.1422262498, F010 0.1423698691.

AdamW displacement: PHYS4 0.1061839639, F010 0.1058906311; cosine 0.9999197634; endpoint separation 0.0013749137. Optimizer moment norms/hashes and per-step update/gradient norms are saved in optimizer_moments.csv and trajectory_comparison.csv. Averaging changes the realized trajectory, not merely a reporting statistic.

Training: 6353.9s (49.64s/update), peak allocated 13.122 GiB. Reserved allocator VRAM was not exported by the reused training harness; no invented measurement. Whole-workflow wall time including census, evaluations and verification so far: 31910.8s / 36000s. Initial projected remaining cost 11259.6s fit the budget.

The physical-gradient variance hypothesis is supported at frozen B8. Functional improvement is not established unless all four P536 gates and the development gate pass. This single-seed result cannot establish an R0 representation ceiling, the necessity of tau, or SCF stability.

One next experiment: Read-only matched gradient-conflict audit at B8 and PHYS4 t128 on eight preregistered existing chemistry/physical groups; preserve F010 loss and sources, no training, no new architecture or loss.

All 128 chemistry positions were reproduced exactly; physical positions remain 0..127, each used four times with original references/variants. 128 updates, 1024 chemistry, 512 AE17, 512 mRKS presentations. t0/t64/t128 retain complete optimizer, RNG, cursor and hashes. Constraints passed at both audited endpoints. No production architecture/operator changes, new data, clipping, curriculum, scheduler, optimizer reset or SCF. Exactly one fixed evaluation variant per chemistry identity; no exhaustive eight-variant sweep.


Explicit answers to the frozen scientific questions:

1–4. Chemistry prefix exact; physical positions restricted to 0..127; each appears four times; references and variants unchanged.
5–6. Averaging parity PASS; 84 distinct AE17 identity+variant records and 90 mRKS records.
7–9. Raw, weighted physical and composite vector MSEs/reductions are in the first table and joint_variance.csv.
10. Duplicate and offset-correlation contributions are retained explicitly in duplicate_covariance_attribution.csv and offset_correlations.csv.
11. Frozen-state smoothing is confirmed; a Jensen inequality alone does not establish useful training.
12–13. Resource projection PASS; exactly 128 actual AdamW steps completed, with inherited counters reaching 256.
14. PHYS4/control displacement cosine 0.999920; endpoint separation 0.001375.
15. MAE change versus F010 +0.008150 kcal/mol; JW change -0.0001436192.
16. MAE improved versus F010 in DBH76, NCCE31; deteriorated in ABDE4, EA13, IP13, MGAE109, PA8, pTC13.
17–18. Exact final objective values and all corrected-P536 ratios are tabulated above; no sampled estimates substitute for them.
19. Strict conjunctive scientific gate FAIL.
20. Analytical anchors, spin symmetry and tau independence PASS; variational operator source/metric/gauge unchanged.
21. Clean28 evaluations: 0; scientific gate controls access and no earlier checkpoint is selected.
22. Historical best not tested.
23. Whole-workflow wall time 31910.8s; actual training 6353.9s; unique frozen gradient calculations 797.6s. Whole-workflow wall includes evaluation, verification and intervening idle time; it is not presented as pure GPU-kernel time.
24. Read-only matched gradient-conflict audit at B8 and PHYS4 t128 on eight preregistered existing chemistry/physical groups; preserve F010 loss and sources, no training, no new architecture or loss.

Focused regression suite: 34 passed; Ruff and compileall PASS. Frozen producer/driver source hashes, all 174 singleton caches, checkpoints and complete log mapping verified. The final staged Git diff check and remote verification are performed before delivery.

Scientific classification: MECHANISM-ONLY
Architecture: R0
Scalar objective: unchanged F010
Physical prefix: 0..127
Physical offsets: 0,31,62,93
Physical modulo: 128
Preflight: PASS
Optimizer updates: 128/128
Chemistry presentations: 1024
AE17 presentations: 512
mRKS presentations: 512
ae17 singleton vector MSE: 19071091.66342308
ae17 PHYS4 vector MSE: 6346725.62817399
exc singleton vector MSE: 148671737.79867584
exc PHYS4 vector MSE: 40930577.74847078
op singleton vector MSE: 0.13757895080414972
op PHYS4 vector MSE: 0.032375551469315686
Weighted physical vector MSE reduction: 67.187166%
Composite vector MSE reduction: 44.932553%
Final relchem: 1.3093108885760607
P536 relchem ratio: 1.0597665059229342
Final ae17: 2.589061847997294
P536 ae17 ratio: 0.10397401511600063
Final exc: 5.202021924552712
P536 exc ratio: 0.05643648417182626
Final op: 0.029418024645710517
P536 op ratio: 0.8874692961300292
Physical eligibility: FAIL
Clean28: NOT EVALUATED — scientific eligibility failed
Historical best beaten: NOT TESTED
Training: YES
Architecture modified: NO
Loss modified: NO
Optimizer reset: NO
Curriculum: NO
Gradient clipping: NO
SCF executed: NO
New training data: NO
Total scientific wall time: 31910.8 seconds
Budget: 36000 seconds
Budget respected: YES
F021-PHYS4 commit: see Git commit containing this report (avoids self-referential SHA).
Git push / verified remote HEAD: returned with the final response after verification.
