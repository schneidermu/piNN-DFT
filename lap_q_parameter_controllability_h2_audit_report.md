# Frozen-h2 q parameter-controllability audit

**CASE D: parameter controllability remains unresolved.** The frozen parameter-direction finite difference disagrees with the autograd prediction by 99.950716%, exceeding the unchanged 5% gate. h2/h3 directional consistency and independent sketch repeatability pass. These passes do not rescue the failed parameter validation.

## Identity and frozen scope

Starting commit: `8946334d5f3d3d6c77839f13469fe7aeebeb30b7` on `lap_full_vxc`. Final commit is the Git commit containing this report; no self-referential commit hash is embedded.
The historical failed-window report is preserved. This report uses a new filename rather than rewriting the earlier experiment.
Twelve unchanged states: seeds 11, 23, 41, each with P67, P536, PBE_TANGENT, PBE_HEAD. The same 64 points are used throughout. No states, q widths, parameter epsilon, or sketch widths were selected after results.
Probe SHA256: `bcc3d01872d0181b197800e2a0295178224fe1d81e9ebfb2b8b8e2f64437b81a`.
h2 = `[0.023218985940978362, 0.042529125693106434]`; h3 = `[0.011609492970489181, 0.021264562846553217]`. Only h2 is sketched; h3 is used for the prescribed directional control.
The observable is the previously validated symmetric XC-energy q slope. Raw Laplacian perturbations use the existing production q denominator at fixed density and other ingredients. Diagnostic arithmetic is F64 on exact widened F32 source/state values; source precision is not reconstructed. Functional-call replay matched all 1,536 cached h2 slopes exactly (maximum difference 0).

## Parameters, rows and mathematical contract

34 unique production-trainable parameter tensors, 9,446 coordinates; sorted canonical names, shapes, flattened offsets and actual module groups are saved externally and hash-bound. Tied exchange parameters are counted once. No buffers or frozen PBE constants are differentiated.
Rows are ordered by seed, regime, alpha points 0–63, then beta points 0–63. FULL has 1,536 rows, NON_HEAD has 1,152 rows, PBE_HEAD_ONLY has 384 rows. PBE_HEAD current slopes are exactly zero, but its output-head parameter derivatives are nonzero. None of those rows is removed or relabeled as a structurally zero Jacobian. Hidden-block derivatives of the zero-head control are zero.
The primary Jacobian is an ensemble map under a **shared additive displacement in canonical raw parameter coordinates**: concatenate S(theta_state + delta) across nine NON_HEAD states. This is not a block-diagonal Jacobian with independent parameters for each state, an averaged functional, or an individual-state rank estimate. Hidden-unit correspondence across seeds is coordinate-dependent; ensemble subspaces must not be interpreted as universal physical parameter directions.
No dense M-by-P Jacobian was constructed. Both frozen Rademacher seeds (42 and 314159), width 40, use VJP J^T Omega, F64 QR, JVP JQ and small-matrix SVD. Directions are rotated by the small SVD to obtain leading parameter singular directions. The spectrum is width-limited. Frobenius values use the prescribed random-probe estimator; projected image energies are lower bounds, not exact per-state Jacobian norms.

## Spectrum and repeatability

| View / seed | numerical rank @1e-3 | r99 | entropy effective rank | sigma1 | Frobenius estimate |
| --- | --- | --- | --- | --- | --- |
| NON_HEAD_42 | 40 | 8 | 3.63956833159 | 267.134200118 | 353.917663684 |
| FULL_42 | 40 | 9 | 4.01232206212 | 267.13387972 | 381.258420807 |
| PBE_HEAD_ONLY_42 | 39 | 9 | 3.89565442353 | 43.2926716095 | 63.5577738731 |
| NON_HEAD_314159 | 40 | 8 | 3.6398194301 | 267.133834781 | 335.325917971 |
| FULL_314159 | 40 | 9 | 4.01317780523 | 267.13251245 | 313.846265274 |
| PBE_HEAD_ONLY_314159 | 40 | 9 | 3.8955959916 | 43.29277637 | 62.586024509 |

Rank 40 means all 40 sampled modes exceed the relative threshold, not that the full Jacobian has rank 40. Entropy effective rank and r99 answer different questions and are reported separately.
| NON_HEAD index | sigma seed42 | sigma seed314159 | normalized seed42 | cumulative energy |
| --- | --- | --- | --- | --- |
| 1 | 267.134200118 | 267.133834781 | 1 | 0.561450712758 |
| 2 | 166.684667635 | 166.683820271 | 0.623973521778 | 0.780047592844 |
| 3 | 128.091085911 | 128.087903475 | 0.479500887026 | 0.909136958687 |
| 4 | 74.188132886 | 74.1891521577 | 0.277718588085 | 0.952440312635 |
| 5 | 42.9342107042 | 42.9311120013 | 0.160721505091 | 0.966943371811 |
| 6 | 39.0573178433 | 39.058375375 | 0.146208601617 | 0.978945478537 |
| 7 | 32.915812185 | 32.9136624585 | 0.123218263219 | 0.987469838951 |
| 8 | 26.7525983243 | 26.7643280587 | 0.100146661538 | 0.9931008268 |
| 9 | 15.3121481829 | 15.3242290531 | 0.0573200592668 | 0.994945523195 |
| 10 | 12.1444433431 | 12.143206603 | 0.0454619563415 | 0.996105923618 |
| 11 | 10.2812067159 | 10.2735855264 | 0.0384870477512 | 0.996937574084 |
| 12 | 8.7302691057 | 8.74857743993 | 0.0326812107991 | 0.997537237996 |
| 13 | 7.86553199375 | 7.86733150551 | 0.0294441220565 | 0.998023991242 |
| 14 | 6.67361912768 | 6.68890477626 | 0.0249822715502 | 0.998374400432 |
| 15 | 5.94332742474 | 5.94483479978 | 0.0222484707017 | 0.998652315418 |
| 16 | 5.85543266885 | 5.90502328434 | 0.0219194422364 | 0.998922071121 |
| 17 | 5.07567538618 | 5.19754084147 | 0.0190004701155 | 0.999124764858 |
| 18 | 4.80845678219 | 4.79175932335 | 0.018000154155 | 0.999306678005 |
| 19 | 3.79407024511 | 3.79960660313 | 0.0142028622447 | 0.99941993457 |
| 20 | 3.73724360159 | 3.69279292631 | 0.0139901352951 | 0.999529823885 |
| 21 | 3.48988098206 | 3.39409854884 | 0.0130641489578 | 0.999625647795 |
| 22 | 2.95206570775 | 2.92948316559 | 0.0110508714588 | 0.999694213144 |
| 23 | 2.57009968292 | 2.64333217259 | 0.00962100577831 | 0.999746183128 |
| 24 | 2.34237977668 | 2.3338805865 | 0.00876855069715 | 0.99978935166 |
| 25 | 2.22163658771 | 2.26807458955 | 0.00831655619809 | 0.999828184458 |
| 26 | 2.06579833145 | 2.16492351655 | 0.00773318553198 | 0.999861760423 |
| 27 | 1.84438209379 | 1.78715125472 | 0.00690432783588 | 0.999888524634 |
| 28 | 1.59478768964 | 1.55995225385 | 0.00596998695389 | 0.999908535155 |
| 29 | 1.44429520242 | 1.49697443415 | 0.00540662783642 | 0.999924947272 |
| 30 | 1.38297667956 | 1.27981818257 | 0.00517708582037 | 0.999939995396 |
| 31 | 1.23525391416 | 1.25590452562 | 0.00462409498154 | 0.999952000478 |
| 32 | 1.17779333454 | 1.13293508559 | 0.00440899493222 | 0.999962914651 |
| 33 | 1.07830700855 | 1.01518768398 | 0.00403657415664 | 0.99997206289 |
| 34 | 1.01798109295 | 0.976216354 | 0.0038107479031 | 0.999980216165 |
| 35 | 0.883711836947 | 0.873233700548 | 0.0033081194267 | 0.999986360487 |
| 36 | 0.736699794029 | 0.746487999834 | 0.00275778913259 | 0.999990630545 |
| 37 | 0.643563007167 | 0.600683453209 | 0.00240913745557 | 0.999993889173 |
| 38 | 0.615638521018 | 0.520742881885 | 0.002304603906 | 0.999996871149 |
| 39 | 0.530138237959 | 0.472125848305 | 0.00198453899847 | 0.999999082364 |
| 40 | 0.341514267485 | 0.445922187025 | 0.00127843708269 | 1 |

NON_HEAD: repeat stability PASS; maximum top-12 singular-value relative difference 0.00209710995276.
| k | principal angles (degrees) |
| --- | --- |
| 4 | 0.000674786956727, 0.0037468515741, 0.00571888836166, 0.0188279254518 |
| 8 | 0.000467934946539, 0.00204378164397, 0.00361242778516, 0.0145983798463, 0.0237447676328, 0.036997459678, 0.0863952629628, 0.168543530523 |
| 12 | 0.000445175623643, 0.00136203283391, 0.00232538273549, 0.00706966545192, 0.0204699462394, 0.0233550871026, 0.0559175461553, 0.100549410088, 0.160499145089, 0.517200153962, 0.784017248751, 2.1073144638 |

FULL: repeat stability PASS; maximum top-12 singular-value relative difference 0.00201176287715.
| k | principal angles (degrees) |
| --- | --- |
| 4 | 0.00233623017502, 0.00402252615293, 0.0092339983834, 0.0301847045715 |
| 8 | 0.00172814331309, 0.00283258897891, 0.00470931033637, 0.0167253159622, 0.0415550340347, 0.0494815131013, 0.0826519077715, 0.286246401002 |
| 12 | 0.00115053118674, 0.00195412151583, 0.00321113019401, 0.0134909058355, 0.0226696099343, 0.0413670639237, 0.0648098238332, 0.162855903146, 0.201962758354, 0.354853550464, 0.876732068108, 1.25972067449 |

PBE_HEAD_ONLY: repeat stability PASS; maximum top-12 singular-value relative difference 0.000223318471447.
| k | principal angles (degrees) |
| --- | --- |
| 4 | 0.000374387288556, 0.000462477449959, 0.00175748014157, 0.00847617550284 |
| 8 | 0.00024382376256, 0.000250078447995, 0.00144051679193, 0.00354997439502, 0.00665558642573, 0.00931894565583, 0.010878501489, 0.0282719019969 |
| 12 | 0.000167748597802, 0.000207777705629, 0.000833178537406, 0.00257611141355, 0.00457147069774, 0.00619220383983, 0.0091478623171, 0.0171582402415, 0.0318789710609, 0.0455180317626, 0.0633140472067, 0.355219839713 |

NON_HEAD sigma1 / estimated ||J||F = 0.754791940412. Full spectra for every view/seed are retained in JSON and external CSVs.
NON_HEAD median nonzero sigma = 3.48988098206. Maximum QR orthogonality error across all six sketches = 6.32827124036e-15; frozen 1e-8 gate PASS.

## Group participation and state contributions

| Actual parameter module | sigma-squared weighted NON_HEAD direction fraction |
| --- | --- |
| c_input_layers | 2.62096879409e-07 |
| c_output_layer | 1.51174268775e-05 |
| c_post_symm_blocks | 4.9707808894e-08 |
| c_symmetrization_blocks | 1.55933011038e-07 |
| x_feature_extractor.0 | 0.0010075830851 |
| x_feature_extractor.1 | 0.910290105741 |
| x_feature_extractor.3 | 0.00561313583012 |
| x_feature_extractor.4 | 0.00289639922565 |
| x_output_layer | 0.0801771909538 |

The dominant group x_feature_extractor.1 is LayerNorm affine parameters, not the prediction head. Spectral direction participation and the gradient of response norm are distinct diagnostics.
| State | projected squared gain | fraction of NON_HEAD projected image energy |
| --- | --- | --- |
| 11_P67 | 2987.26137169 | 0.0235031393454 |
| 11_P536 | 2358.50326453 | 0.0185562038187 |
| 11_PBE_TANGENT | 29915.956587 | 0.235372405969 |
| 23_P67 | 569.160199765 | 0.00447803182262 |
| 23_P536 | 529.280565975 | 0.00416426731614 |
| 23_PBE_TANGENT | 70482.0806161 | 0.554538072152 |
| 41_P67 | 398.014174218 | 0.0031314911667 |
| 41_P536 | 419.505625081 | 0.00330058134715 |
| 41_PBE_TANGENT | 19440.7635209 | 0.152955807062 |

PBE_TANGENT states dominate ensemble projected gain. Positive ensemble gains do not establish substantial controllability or rank for any individual P536 state. No conclusion about P67-to-P536 capacity collapse is justified by this stacked sketch alone.
PBE_HEAD_ONLY output-head weighted participation = 1; exact HEAD row-VJP Frobenius norm = 64.5976093586. FULL projected image-energy fraction on HEAD rows = 0.0317103233002. HEAD remains a separate mechanistic control, not a production candidate.

## Current response and response-norm gradient

NON_HEAD ||s|| = 0.746638227058; ||J^T u_s|| = 58.1002140604; leading sketch output-space projection fraction = 0.985079245978.
||gradient R|| = 43.3798408178, for R = 0.5 ||s||^2. This is J^T s, not the gradient of ||s||. The latter is J^T u_s. Exact repeated VJPs differ by 0; no artificial denominator floor is used. These algebraic quantities are resolved, but failed finite-step validation prevents a trusted controllability conclusion.
| Actual module | fraction of squared gradient R norm |
| --- | --- |
| c_input_layers | 1.14304330041e-07 |
| c_output_layer | 1.50787358818e-05 |
| c_post_symm_blocks | 3.34868561959e-08 |
| c_symmetrization_blocks | 6.11232173604e-08 |
| x_feature_extractor.0 | 0.0130746818981 |
| x_feature_extractor.1 | 0.00330178353292 |
| x_feature_extractor.3 | 0.0898650330376 |
| x_feature_extractor.4 | 0.0633346393436 |
| x_output_layer | 0.830408574537 |

| State | ||s|| | ||gradient R|| | ||J^T u_s|| |
| --- | --- | --- | --- |
| 11_P67 | 0.691670293024 | 36.2400371605 | 52.394959746 |
| 11_P536 | 0.175407256698 | 7.8729266886 | 44.8837000066 |
| 11_PBE_TANGENT | 0.00181983804679 | 0.110973852357 | 60.9800704805 |
| 11_PBE_HEAD | 0 | 0 | undefined |
| 23_P67 | 0.174894215449 | 3.62775221763 | 20.7425511949 |
| 23_P536 | 0.017841222024 | 0.321487349071 | 18.0193570059 |
| 23_PBE_TANGENT | 0.00218947719137 | 0.177481042714 | 81.0609233169 |
| 23_PBE_HEAD | 0 | 0 | undefined |
| 41_P67 | 0.129862725897 | 2.28911693503 | 17.6272053372 |
| 41_P536 | 0.0133374718027 | 0.0968333068988 | 7.26024454491 |
| 41_PBE_TANGENT | 0.0183446176639 | 0.745380310672 | 40.6320984349 |
| 41_PBE_HEAD | 0 | 0 | undefined |

HEAD current-response normalization is undefined because its current slope vector is exactly zero; nonzero output-head Jacobian capacity is reported separately.

## Frozen validation gates

Parameter epsilon = 0.000211871515719; exact serialized value `0.0002118715157188727`. Frozen rule: 1e-5 times max(RMS of nine NON_HEAD base parameter norms, 1). A common unit displacement replicated over nine states has joint norm sqrt(9)*epsilon; this gives the requested relative joint-state scale without choosing epsilon after seeing results.
Parameter finite difference: relative L2 0.999507162553, absolute L2 267.002546381, cosine 0.171345419571: **FAIL**. Exact parameter restoration maximum difference = 0. No epsilon search or extra q window was run.
| NON_HEAD direction | h2/h3 relative L2 | cosine |
| --- | --- | --- |
| 1 | 0.00338319296179 | 0.999998713296 |
| 2 | 0.00381622242478 | 0.999997288476 |
| 3 | 0.00339139600158 | 0.999998653119 |
| 4 | 0.00351412526655 | 0.999998218477 |
| 5 | 0.00719429098102 | 0.999991295198 |
| 6 | 0.00494317072264 | 0.9999953255 |
| 7 | 0.0016083945627 | 0.999999013053 |
| 8 | 0.00439427543462 | 0.999991824225 |

h2/h3 median = 0.00366517384566, maximum = 0.00719429098102: PASS under frozen 5% / 10% gates.
| HEAD control direction | h2/h3 relative L2 | cosine |
| --- | --- | --- |
| 1 | 0.00396301293571 | 0.999997754392 |
| 2 | 0.00546781422455 | 0.999994275127 |

Independent review verified the FD direction mapping: J(Q V) = (JQ)V. A saved-sketch leading-direction adjoint identity agrees at relative 1.27e-11; it checks algebra inside the captured subspace, not the failed finite parameter response. No sign, shape or functional-call state mismatch was identified. The exact cause of the finite-parameter failure is not localized here; attributing it to curvature, LayerNorm amplification, lost capacity or the optimizer would exceed the evidence.

## Safety, tests and review

All 34 immutable hashes match before and after; state mutation NONE; maximum exact parameter restoration difference 0; source gradients remain unmodified. The frozen execution source, contract, manifests, rows and external arrays are SHA-bound in the machine-readable results.
15 focused tests PASS (nine new matrix-free diagnostic tests and six existing helper tests). Ruff, py_compile and git diff --check results are recorded in metrics. No production source changed. Optional rho/sigma sketches were skipped: no existing validated implementation warranted expanding this audit.
Independent Luna MAX artifact review: **PASS with scientific limitation CASE D**. Review found correct VJP/JVP direction mapping and challenged an unreachable generic CASE C branch; no executable A/B/C capacity conclusion is claimed for this failed-validation run. Review status and checks are recorded in the machine-readable artifact. The repository protocol adds a post-execution receipt exposing existing source/artifact SHA bindings; the original frozen external protocol is preserved byte-for-byte and separately hashed.

## Answers and next experiment

1. Useful first-order q parameter controllability is **unresolved**: nonzero derivatives exist, but the required real parameter finite difference fails.
2. Broad versus narrow useful capacity is unresolved. The ensemble sketch is concentrated (entropy effective rank about 3.64), but its interpretation failed validation.
3. Sketch gains are dominated by x_feature_extractor.1 LayerNorm affine parameters; HEAD controls are carried by prediction heads. Individual-state capacity is not inferred.
4. HEAD rows carry 3.17103% of FULL projected image energy. This is a captured-subspace fraction, not an exact full-Jacobian energy decomposition.
5. Yes: the PBE_HEAD_ONLY control has 100% output-head weighted participation and zero hidden-block derivatives; its current descriptor response remains zero.
6. The current response direction has a nonzero VJP and high sketched projection, but practical local controllability is unvalidated.
7. The gradient of R = 0.5||s||^2 is numerically repeatable and nonzero.
8. h2/h3 Jacobian-direction consistency passes for the prescribed eight NON_HEAD directions.
9. No. The single prescribed parameter finite difference has 99.950716% relative disagreement.
10. Dormant output-head capacity is established for PBE_HEAD. Neither retained useful NON_HEAD capacity nor lost NON_HEAD capacity is established because parameter validation failed. Observable contraction remains distinct from trainability.
11. CASE D: Jacobian interpretation is not trustworthy under the frozen validation gate.
12. ONE next experiment: a prospectively frozen, bounded localization of the failed leading parameter-direction linear response, retaining this q probe and comparing autograd versus finite parameter response by state/module. No training or new initialization. It was not launched.

No training, optimizer change, cursor10 advancement, expensive objective-gradient recomputation, full90, SCF, Diet, Slurm or 100-update pilot occurred. No model checkpoint was modified or saved. The unfinished expensive initialization matrix was not resumed.
