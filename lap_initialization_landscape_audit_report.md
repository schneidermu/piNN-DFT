# Initialization landscape audit — deadline-limited result

Starting commit: `d8682bddf6284ed5f3491ceec7c00e53673ff7ff`, branch `lap_full_vxc`. Production source changes: zero.

## Status and scope

The user imposed a 30-minute completion deadline after 14 state evaluations. The remaining matrix was stopped; this is an incomplete audit, not a completed five-seed study. All 30 initialization states were constructed and hash-verified. Fully evaluated: all six regimes for seeds 11 and 23, plus P0/P67 for seed 41. Seed-41 P134 was interrupted during the operator panel, after its complete chemistry cache was written. Its complete chemistry cache and partial operator-panel work are excluded from state-level inference. Seeds 73/101 have no objective/gradient evaluations. No regimes were dropped based on their values.

The original frozen protocol remains unchanged. This deadline receipt changes completion scope, not the scientific design or acceptance thresholds. Unequal available counts are reported explicitly. No missing values are imputed.

## Frozen experiment

Seeds: 11, 23, 41, 73, 101. Predopt strengths: 0, 67, 134, 268, 536 optimizer steps. Each seed has two independent deterministic 536-step replays; captured prefixes and Adam states match exactly. Seed-41 P536 model tensors match canonical predopt SHA `ed4ba8231d93c376ce5aa8fc81e0670f6a5a44c95c2a552651b12d6a1d4b63f8` exactly.

Canonical predopt is unchanged: F32, Adam lr=0.01, batch size 1, 268 canonical groups in fixed order, point chunk 4096. The additional PBE_HEAD zeroes output heads while retaining random hidden weights. Its finite sigmoid bias approximates the PBE kappa limit with a fixed one-ppm offset; it is near-PBE, not an exact analytic initialization. Initially hidden gradients are zero through the zero head, so its tangent-space comparison has that limitation.

Evaluation samples are independent of seed: full251 relative chemistry and full17 AE from the immutable full268 cycle-0 manifest; equal-system mean Exc/operator over H2, HLi, BH, BeH2, H2O, CH4, N2, CO, CH2O, C2H2_iso2, H4Si, AlBeH, ClH, ClHS, HPSi_iso2. Chemistry uses matched-F64 arithmetic on stored source values. Operator uses the repaired production path: stored F32 model, differentiable learned-local F64, PBE F32, AO assembly F64. No source precision is regenerated.

Function-space diagnostics cover 21,073,642 canonical grid points: nine adaptive PBE channels, normalized deviations, and local XC energy deviations. These point-weighted statistics are not independent-system statistics. Raw gradient norms and Gram matrices accompany unit-gradient geometry. No PCD EMA normalization is used.

## Available-state summary

Entries are median [min,max]. P0/P67 have n=3; remaining regimes n=2. These are descriptive available-subset summaries.

| Regime | n | gamma | chemistry–AE cosine | chemistry–Exc cosine | chemistry–operator cosine | full251 J_rel |
|---|---:|---:|---:|---:|---:|---:|
| P0 | 3 | 0.997786 [0.995194, 0.999724] | 0.996386 [0.994537, 0.999507] | 0.996401 [0.994559, 0.999511] | 0.991153 [0.980823, 0.998896] | 9.74972 [9.07425, 11.2845] |
| P67 | 3 | 0.178734 [0.13716, 0.24537] | -0.934108 [-0.956348, -0.87958] | -0.936093 [-0.957115, -0.87821] | -0.878349 [-0.903572, -0.401502] | 1.24432 [1.23703, 1.2458] |
| P134 | 2 | 0.166366 [0.148502, 0.18423] | -0.859516 [-0.92949, -0.789543] | -0.868019 [-0.931066, -0.804973] | -0.899784 [-0.907482, -0.892087] | 1.24052 [1.23814, 1.24289] |
| P268 | 2 | 0.162599 [0.145264, 0.179935] | -0.867663 [-0.931883, -0.803442] | -0.875579 [-0.933381, -0.817778] | -0.906649 [-0.91429, -0.899008] | 1.24175 [1.2343, 1.2492] |
| P536 | 2 | 0.164204 [0.145641, 0.182767] | -0.874369 [-0.932788, -0.815951] | -0.88122 [-0.933192, -0.829247] | -0.901039 [-0.918323, -0.883755] | 1.24407 [1.23733, 1.25081] |
| PBE_HEAD | 2 | 0.123092 [0.112015, 0.13417] | -0.963043 [-0.963784, -0.962301] | -0.963514 [-0.963997, -0.963031] | -0.95769 [-0.969721, -0.945658] | 1.24318 [1.24318, 1.24318] |

| Regime | adaptive PBE MSE | AE RMSE | Exc mean | operator mean |
|---|---:|---:|---:|---:|
| P0 | 0.0410599 [0.0377759, 0.0823228] | 3880.25 [3144.9, 4988.75] | 4410.57 [3532.56, 5674.35] | 0.268547 [0.238458, 0.725145] |
| P67 | 1.53375e-05 [1.43235e-05, 1.8375e-05] | 52.9474 [35.5806, 65.8052] | 65.1529 [47.149, 80.9059] | 0.0378029 [0.0374916, 0.0414206] |
| P134 | 6.35364e-06 [6.33361e-06, 6.37367e-06] | 63.1616 [61.0139, 65.3093] | 77.4784 [75.2999, 79.6568] | 0.0376244 [0.0371689, 0.0380799] |
| P268 | 3.2128e-06 [2.6751e-06, 3.75051e-06] | 59.1257 [58.8815, 59.3699] | 72.8409 [72.8136, 72.8682] | 0.037033 [0.0365553, 0.0375107] |
| P536 | 5.7953e-07 [4.29974e-07, 7.29086e-07] | 56.6437 [56.4661, 56.8214] | 70.0208 [69.9848, 70.0568] | 0.0365116 [0.0361618, 0.0368614] |
| PBE_HEAD | 1.33424e-13 [1.33424e-13, 1.33424e-13] | 57.2771 [57.2771, 57.2771] | 70.9126 [70.9126, 70.9126] | 0.0365017 [0.0365017, 0.0365017] |

All evaluated states passed manufactured F32/F64 physical anchors, exact tau independence, spin symmetry, finite input derivatives, deterministic forward replay, and no-dropout checks. Evaluated P0 seeds 11/23/41 pass; unevaluated P0 seeds 73/101 are not claimed to pass objective/derivative checks. Structural admissibility does not establish SCF stability or chemical accuracy.

## Geometry contract and validation

For unit gradients u_i, minimum-norm simplex combination m gives gamma=||m|| and d=m/gamma. Positive u_i dot d means descent for update -d. The unit-ball max-min optimum cannot be negative because d=0 is feasible. All completed states have positive margins; no updates are applied. Full simplex primal, active stationarity, inactive dual, and complementarity residuals are checked against the frozen tolerances. Raw norms, matrices, eigenvalues, rank, coefficients and residuals are retained in JSON.

The diagnostic solver passed orthogonal, opposing, and identical-gradient fixtures (gamma 0.5, 0, 1). Manufactured physical checks passed. Diagnostic tooling was restarted to strengthen finiteness, KKT and provenance guards without changing the objective or samples. The first seed-11 P0 receipt is preserved under its original SHA; post hoc guard certification is explicitly distinguished from original execution provenance.

## Paired interpretation

Available seed variation is not uniform across metrics. P67 gamma IQR is 0.05410 and AE RMSE CV is 0.29485 (n=3), despite J_rel CV 0.00378. P536 gamma IQR is 0.01856 and AE RMSE CV is 0.00444 (n=2). These unequal-count summaries cannot establish systematic variance reduction. Matched-seed raw values and all IQR/CV statistics are in JSON; cosine CV is omitted because its sign makes that statistic unsuitable.

1. Raw initialization sensitivity: P0 has gamma near one and positive task cosines, but J_rel 9.07–11.28 and AE RMSE 3145–4989. These aligned gradients accompany grossly poor initial function behavior. P0 is physically admissible in the manufactured tests, but is not a qualified starting point.

2. Predopt strength: in both complete paired seeds, P536 sharply improves PBE agreement and AE/Exc/operator starting values relative to P0, while chemistry-secondary cosines become negative and gamma drops. Intermediate trends are nonmonotonic. Seed-11 P67 operator cosine is -0.402, versus -0.904 for seed 23 and -0.878 for seed 41: weak-predopt tangent geometry is seed-sensitive.

3. Does full predopt cause conflict? The complete seeds support a transition from bad aligned raw functions to better PBE-like conflicting functions. They do not prove that full two-epoch predopt uniquely creates the conflict. PBE_HEAD has nearly identical PBE-like values across seeds 11/23 yet strongly negative chemistry-secondary cosines without predopt optimization. Function-space position and random tangent features both matter.

4. Intermediate strength: P67 is a promising weak control, particularly seed 11, but is not uniformly favorable. P134/P268 do not consistently dominate P536 in geometry or losses. A best intermediate regime is unresolved with two complete seeds.

5. No predopt: finite and structurally constrained for the three evaluated seeds, but its very poor losses/PBE deviations preclude recommending it solely from gamma. No SCF evaluation was performed.

6. Multi-seed need: observed P67 gradient variability supports multi-seed future comparisons. The audit does not establish that initialization variation dominates optimizer effects: historical optimizer geometry used different states/panels, so those magnitudes are not directly comparable.

H1 is partially supported for P0-to-P536 endpoints, not monotonic predopt depth. H2 has starting-quality support but not common-margin support. H3 is unresolved. H4 optimizer dominance is unresolved. Full predopt is mixed: beneficial as a PBE/starting-value prior, less favorable in normalized common-descent geometry than bad raw initializations. No production initialization is selected.

Provisional future controls, at most three: P67 (weak), P268 (intermediate), P536 (canonical). This is a control set for a future completed multi-seed comparison, not a performance ranking. P0 remains an admissibility/poor-quality diagnostic. Complete the frozen missing states before making a seed-general initialization recommendation.

## Provenance, state, tests and review

Frozen protocol SHA256: `41961dd45446b55778918e95187da6d1d892640c78fffb472ccc089f55e9785e`. Every source/input hash, all 30 state-file hashes, and all 14 completed array hashes were verified. Per-state model hashes are unchanged; no parameter gradient storage was retained. RNG restoration assertions cover evaluation after model construction; they do not claim that random model constructors consume no process RNG. No optimizer, sampling cursor, scheduler, or EMA was advanced by evaluation.

Large states, gradients, partial caches and scripts remain external at `C:/Dev/readWFN_share_ms/lap_init_landscape_runs_20261005`; JSON binds their hashes. Production files are unchanged. Frozen protocol plus compact per-state results/report are the only intended Git additions. Validation and final independent review receipts are recorded below.

## Decision

Repository artifacts use UTF-8/LF formatting. The recorded frozen-protocol byte SHA refers to the original external protocol; the repository protocol has identical parsed JSON content with normalized line endings. No scientific protocol field was changed.

Final validation: 25 focused Windows model/operator tests passed (with the repository's required train_models PYTHONPATH); three solver certificates and six manufactured state checks passed. External tooling compileall/py_compile and Ruff for the two new finalization scripts passed. Git diff --check passed; no tracked production file changed. The initial pytest collection failed because PYTHONPATH was omitted, then passed after that execution setup was corrected.

Independent GPT-6 Luna MAX review accepted the bounded scientific interpretation and provenance, with two documentation corrections: describe interrupted seed-41 P134 as a completed chemistry cache plus partial operator-panel work, and supply this review receipt. Both are corrected. The entire incomplete state remains excluded. No experiments were rerun by the reviewer.

The complete five-seed landscape question remains unresolved. The single next diagnostic is to finish the already-frozen 16 missing state evaluations, without altering seed/regime/sample design. No new optimizer is justified by this partial audit.

No main multi-objective training trajectory was run, no production optimizer was changed, no old cursor10 state was advanced, and no full90/SCF/Diet/Slurm/100-update run was launched.
