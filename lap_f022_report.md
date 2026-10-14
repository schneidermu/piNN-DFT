C — FIXED MIXTURE NO-GO

The single prespecified chemistry mixture is feasible in chemistry-only geometry, but the unchanged physical contributions reverse legacy descent. The complete F022 gradient fails the primary gate. No optimizer was created or stepped; native counterfactual, finite-step evaluation, training, full90 re-evaluation and Clean28 were not run. No alternative alpha or coefficient was tested.

Population-S cosines: JW/legacy -0.660274205265; JW/operator 0.752852899563; legacy/operator -0.984063882318.
Strict alpha interval (0.660274205265, 0.765044742613); midpoint 0.712659473939.
Frozen a=13.2494272182828, b=0.957132282861253, K=20.0945611472081, D=0.752849380843209. Chemistry gradient norm 15.128177917992 equals c010 × ||gJW,S|| (relative tolerance 1e-10).

| Target (S chemistry/full physical) | F010 dot | F022 chemistry-only dot | F022 physics-only dot | F022 full dot |
|---|---:|---:|---:|---:|
| JW | 0.8839963878 | 0.2745545102 | 0.493594171 | 0.7681486812 |
| legacy | -12.4453598 | 0.2679912135 | -9.902374745 | -9.634383532 |
| MAE | -45.51883148 | 37.08113114 | -65.20593719 | -28.12480605 |
| ae17 | 4528.488114 | -509.5723083 | 4117.791377 | 3608.219069 |
| exc | 16328.9449 | -1816.938398 | 14823.41989 | 13006.48149 |
| op | 0.5000055193 | 0.01081103096 | 0.3811408277 | 0.3919518587 |

Positive dot predicts decrease along the negative gradient; negative dot predicts increase. Legacy chemistry-only margin 0.2679912135 is overwhelmed by physical contribution -9.9023747451. Full-gradient cancellation ratio is 0.5262326208. Operator has positive full-gradient progress; this is a linear diagnostic, not finite-step safety qualification.

| Population-S database | MAE derivative along −G010 | MAE derivative along −G022 |
|---|---:|---:|
| ABDE4 | 49.37000353 | 42.51502336 |
| DBH76 | -53.77028708 | -45.03995716 |
| EA13 | -7.34247807 | -5.81155395 |
| IP13 | 80.51511662 | 63.03066841 |
| MGAE109 | 102.07644408 | 63.28032787 |
| NCCE31 | -8.72966625 | -7.49110149 |
| PA8 | 132.36409239 | 106.13217359 |
| pTC13 | 199.63886598 | 158.93315270 |

Negative derivative means improvement. ABDE4, PA8 and pTC13 still have adverse derivatives, although smaller than F010; EA13 retains less favorable predicted improvement. DBH76/NCCE31 improvement also weakens. IP13/MGAE109 adverse slopes shrink. These are derivatives, not observed endpoint changes. Population-E derivatives are separately saved in database_directions.csv and full_gradient_geometry.csv.

| Estimator (first 64 historical batches) | Positive dots | p10 cosine | Median cosine | Max norm ratio | Relative gradient MSE |
|---|---:|---:|---:|---:|---:|
| F010 | 61/64 | 0.21025752 | 0.85302470 | 3.17646420 | 1.06256633 |
| F022 | 49/64 | -0.27218189 | 0.35573489 | 7.18303375 | 11.38435404 |

F022 stochastic fidelity deteriorates materially. ABDE4 contributes in 8/64 batches and its maximum component norm rises from 3.7850 to 81.1977; NCCE31 reaches 88.5181. PA8/IP13/pTC13 also have much larger components. Several contributions change and cancel; this does not establish single-database causality. Every database contribution, presentation frequency and combined penalty coefficient is retained in database_batch_contributions.csv; norms do not add because database gradients cancel. No rare-database sampler or post-hoc normalization was introduced.

Provenance: parent checkpoint, inherited moments, manifest, canonical A_d, F020 S Jacobian/mapping/source records, F013/F015 E Jacobian and full90 physical vectors were authenticated. Parameter ordering is sorted and unique (9446 coordinates). Legacy factors come directly from optuna_joint FCHEM_DB_WEIGHTS × FREQ_WEIGHTS / MEAN_WEIGHT. E scalar, gradient norm and JW cosine reproduce the qualified F010 receipts; tests independently verify production singleton scalar/gradient parity and single combined scalar equivalence. This does not substitute a multi-reaction database RMSE for the mean singleton objective.

S and E each use one existing variant per identity and are not conflated. Calibration uses S exclusively. All physical vectors are the same full17/full90 qualified gradients as F010, not PHYS4 prefix means or historical15 gradients. No live full-corpus gradient was recalculated.

B8 constraint/restoration receipts are reused because no model or moment state was perturbed. Relevant operator/AdamW/chemistry regression tests passed: 22 tests, Ruff and compileall PASS. No new GPU anchor calculation is claimed. No pilot runtime projection or endpoint reserve is fabricated after the early stopping gate.

Measured cached execution body: 1.984s. Frozen-workflow wall: 434.697s; counting this entire interval as active gives a conservative upper bound below 900/5400/10800s limits. Source inspection preceded the frozen numerical execution; no GPU computation was performed.

Answers to the frozen questions:

1–5. Authentication PASS; exact S cosines, strict interval, midpoint and fixed a/b are above.
6. NO: full G descends JW but increases legacy. Chemistry-only direction passes both.
7. Operator linear progress is positive; actual native/finite-step safety NOT RUN.
8. Inherited-moment native counterfactual NOT RUN because the earlier mandatory gate failed.
9. YES: stochastic alignment and relative MSE deteriorate (table above).
10–11. Training not authorized; 0/64 optimizer updates.
12–15. No final endpoint exists; no observed JW, legacy, MAE or database improvement claimed. Directional comparisons are above.
16–18. P536 eligibility NOT RUN; Clean28 NOT EVALUATED; historical best NOT TESTED.
19. Cached execution and conservative active/wall bounds are above; GPU evaluations zero.
20. Close this fixed mixture. One next experiment: Read-only cached affine-feasibility audit for simultaneous JW/legacy/operator descent with the original physical contribution fixed; determine necessary chemistry-contribution bounds, without choosing new coefficients or running training.

This failure applies to the frozen F022 mixture with unchanged physics coefficients. It does not establish an R0 representation ceiling or impossibility of all fixed scalarizations. No rescue experiment was launched.

Scientific classification: C
Architecture: R0
Scalar objective: fixed F022
Curriculum: NO
Gradient clipping: NO
Dynamic task weighting: NO
Alpha: 0.7126594739387087
Chemistry coefficient a: 13.249427218282788
Chemistry coefficient b: 0.957132282861253
Cached-gradient preflight: FAIL
Native-step preflight: NOT RUN
Training launched: NO
Optimizer updates: 0/64
Final JW: unavailable
Final legacy relchem: unavailable
Final Minnesota MAE: unavailable
P536 relchem ratio: unavailable
P536 AE17 ratio: unavailable
P536 Exc ratio: unavailable
P536 operator ratio: unavailable
Physical eligibility: NOT RUN
Clean28: NOT EVALUATED
Historical best beaten: NOT TESTED
Active scientific compute: <= 434.697 seconds (conservative bound)
Total frozen-workflow wall time: 434.697 seconds
Budget respected: YES
Optimizer reset: NO
Architecture modified: NO
New training data: NO
SCF executed: NO
F022 commit: commit containing this report; returned after Git verification
Git push / verified remote HEAD: returned after push, avoiding a self-referential commit hash
