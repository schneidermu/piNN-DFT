NO-GO — SIZE-AWARE PHYSICAL WEIGHTING INEFFECTIVE

One law N_e^-1/2 was qualified at B8 with task-wise initial norm preservation. It increased H leverage but did not improve the matched H/native-step outcome or the adverse legacy projection. The mechanism gate failed; no 128-update pilot, scientific endpoints or Clean28 evaluation was launched.

Electron counts: 17 AE17 identities (all 112 historical training identity/variant pairs authenticated; 84 in S128 evaluated) and 90 mRKS systems. Official Minnesota XYZ/G09 files establish nuclear identity and ionic charge, including H Z=1,q=0,N_e=1. mRKS uses saved Tr(dm_ks S), checked against nuclear charges minus molecular charge to 1e-6. AO count is not used as electron count.

Electron distributions: AE17 {'min': 1, 'max': 17, 'median': 9.0, 'counts': {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1}}; mRKS {'min': 2, 'max': 34, 'median': 18.0, 'counts': {2: 1, 4: 1, 6: 2, 8: 3, 10: 6, 12: 5, 14: 14, 16: 11, 18: 10, 20: 4, 22: 12, 24: 6, 26: 3, 28: 2, 30: 7, 32: 2, 34: 1}}.

| Task | Original S128 gradient norm | Size-weighted norm before calibration | Frozen scale | Calibrated norm | Old/new cosine |
|---|---:|---:|---:|---:|---:|
| ae17 | 5783.3480915436 | 1680.5577450124 | 3.441326612375 | 5783.3480915436 | 0.999921573560 |
| exc | 20290.1169854117 | 4479.6217816476 | 4.529426361069 | 20290.1169854117 | 0.999971497281 |

Physical old/new cosine 0.999969878487; full S251-chemistry/S128-physics cosine 0.999981045385. Scalarization coefficients, chemistry stream, operator metric/gauge/normalization and original precision boundaries are unchanged. No global task-strength reduction is hidden in the size weighting: AE17 and Exc each preserve their B8 S128 gradient norm.

| Atom | N_e | Calibrated multiplier | Original error B8 (kcal/mol) |
|---|---:|---:|---:|
| H_ae17 | 1 | 3.4413266124 | 1.3739066447 |
| He_ae17 | 2 | 2.4333853839 | 4.0031497291 |
| Li_ae17 | 3 | 1.9868508460 | 3.8516047939 |
| Cl_ae17 | 17 | 0.8346443009 | 6.7901045323 |

All 17 atoms and all S128 variants are in system_leverage.csv/hydrogen_diagnostics.csv. The latter includes historical B8/F010 t128/t251 fixed-evaluation errors, not a new model evaluation. H target remains native-F32 −313.7547607421875 kcal/mol; it was not replaced by analytic −0.5 Ha. The fixed-H functional decomposition gives exchange −0.3053895837025203 Ha and correlation −0.005830887702820334 Ha; their sum −0.3112204714053406 Ha matches direct integrated XC exactly. This is a predicted correlation component, not a new reference target or universal self-interaction claim. Hartree+XC remains unavailable because no separate authenticated Hartree field exists.

| Physical decomposition | Legacy dot | Physical norm | Full joint norm | JW dot joint | Full90 operator dot joint |
|---|---:|---:|---:|---:|---:|
| old | -10.0482907268 | 0.7630311167 | 0.9043969874 | 0.8896051872 | 0.5054131785 |
| AE17_only | -10.0821152349 | 0.7637323747 | 0.9058262558 | 0.8940730827 | 0.5069936179 |
| Exc_only | -10.0693467602 | 0.7634580368 | 0.9052699759 | 0.8923393802 | 0.5063955754 |
| combined | -10.1031712683 | 0.7641701636 | 0.9067073585 | 0.8968072757 | 0.5079760148 |

R_conflict old 2.6089918043, new 2.6232412808; maximum favorable same-norm chemistry dot 3.8514075476. Both ratios exceed one: no chemistry direction of that fixed norm can compensate the adverse physical projection locally. This is a first-order obstruction, not proof that every fixed scalar loss or R0 architecture fails. F022 used uniform physical populations (legacy dot −9.902374745); F023 correctly calibrates/compares S128 (−10.048290727 → −10.103171268). These populations are not interchanged.

| Native reversible step | Step norm | H absolute error after | Δ H error | Δ AE17 singleton | Δ Exc singleton | Δ operator singleton |
|---|---:|---:|---:|---:|---:|---:|
| old | 0.000649867030 | 1.361620811114 | -0.012285833576 | +0.2292913518 | +1.2977253845 | +0.0000065500 |
| new | 0.000649347893 | 1.361719177171 | -0.012187467519 | +0.2276478286 | +1.2907224421 | +0.0000064762 |

Both native steps improve H versus B8, but size weighting is worse than the matched original step by 0.000098366057 kcal/mol (frozen resolution 1e-6). Both raise the probed AE17/Exc/operator losses. Linear chemistry predictions and full-prefix physical predictions are separately retained in native_step_preflight.csv; they are not finite full-corpus changes. First-step displacement cosine 0.999998467601. All model/buffer/moment/RNG restoration checks passed; two temporary counterfactual steps, zero retained optimizer updates.

| Frozen estimator | Positive dots /128 | Worst cosine | p05 | p10 | Median | Relative vector MSE |
|---|---:|---:|---:|---:|---:|---:|
| old | 128 | 0.77339264 | 0.84136266 | 0.88498222 | 0.96098115 | 0.22680264 |
| new | 128 | 0.79992834 | 0.86410976 | 0.90103351 | 0.96532944 | 0.18380160 |

Complete vector MSE, outlier shares, norm ratios and chemistry/physics covariance are in stochastic_summary.csv. This is exact finite-prefix frozen-state algebra, not conditional unbiasedness during evolving training. All 128 original source positions and all F32-representability checks passed.

Original first AE17/Exc/operator live gradients matched cache exactly (relative gaps 0). Independent weighted-scalar autograd checks passed frozen bounds: AE17 relative gap 1.328e-14; Exc 1.085e-6 within the declared native-F32 leaf-accumulation bound. This is not claimed bitwise Exc gradient equality. All 174 signed forward receipts reproduce original losses within declared tolerance. The signed-Exc extraction caller initially used the wrong integration keyword/layout; a diagnostic-only adapter corrected it to the existing point_chunk_size and [point,1,10] API. Original native results and production source were unchanged; no failed mRKS forward result was retained.

No chemistry objective evaluated every grid variant. S128 uses its historical variants; signed forwards cover only the 84 cached AE17 variants. Electron metadata for additional historical variants is inspection only. The fixed E251/E17 scientific populations and unweighted P536 gates remain intact but were not evaluated for a new trained model because training did not occur.

Wall used: 2841.1s of 36000s. Native probe 24.220s, first physical live-gradient profile 9.360s. Full training/t64/t128 evaluation and conditional Clean28 projection were not run after the earlier mandatory mechanism NO-GO.

One next experiment: Read-only cached common-descent feasibility audit adding fixed-evaluation H error and legacy chemistry to JW/operator constraints at B8; no new loss, weights, architecture or training.

Answers: counts and distributions are authenticated; original losses reproduce; c_A/c_E and leverage are tabulated; rotations are small; legacy conflict and H matched-step outcome worsen; operator remains favorable along the plain joint gradient but regresses in the native finite probe. Stochastic changes are descriptive, not new thresholds. Preflight blocks training. No endpoint objective/H/full90/chemistry improvement or Clean28 superiority is claimed. The particular size-normalization hypothesis is not supported by this read-only screen; no other exponent or H-specific weight was tested.

Scientific classification: NO-GO
Architecture: R0
Electron-size exponent: 0.5
AE17 scale c_A: 3.441326612374757
Exc scale c_E: 4.529426361068562
Original chemistry loss: UNCHANGED
Operator loss: UNCHANGED
Hydrogen baseline error: 1.3739066446900097 kcal/mol
Final hydrogen error: unavailable
Hydrogen preservation: NOT RUN
Legacy conflict before: -10.04829072677243
Legacy conflict after: -10.103171268286035
Preflight: FAIL — mechanism gate; numerical PASS
Training performed: NO
Optimizer updates: 0/128
Final legacy relchem: unavailable
Final AE17: unavailable
Final Exc: unavailable
Final operator: unavailable
P536 ratios: NOT RUN
relchem: unavailable
AE17: unavailable
Exc: unavailable
operator: unavailable
Physical eligibility: NOT RUN
Clean28: NOT EVALUATED
Historical best beaten: NOT TESTED
Curriculum: NO
Gradient clipping: NO
Adaptive task weighting: NO
Architecture modified: NO
New training data: NO
SCF executed: NO
Scientific wall time: 2841.1 seconds
Budget respected: YES
F023 commit: commit containing this report; returned after verification
Git push / remote HEAD: returned after push, avoiding self-referential commit hash

Final verification: 23 tests passed (14.51 s); Ruff, compileall and git diff --check passed. E402 is excluded only for the existing script bootstrap import layout. Frozen source and native restoration receipts remain intact.
