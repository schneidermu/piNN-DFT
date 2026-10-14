# SOURCE → CORPUS grid transfer and matched operator-loss impact

Mapped row selection dominates the reference-matrix change. Canonical PBE changes in almost the same matrix direction, but with only about 5–7% of the reference-change magnitude; compensation is partial. The matched squared operator loss decreases slightly on all three pilots. This is not evidence that the subset is physically superior or explains a trained-NN plateau.

Frozen HEAD `8752754c1bcc04efe0763b1ffc732ab7703bb709`; baseline gauge audit sources and three NPZ/corpus/array hashes verified. Versions: {'PySCF': '2.14.0', 'NumPy': '2.2.6', 'SciPy': '1.15.3', 'execution': 'existing WSL Python3'}. No old branch checkout. Primary formula: unchanged repository PBE with stored canonical constants widened; density/sigma regularization unchanged (PBE.py:50–85). No new Libxc oracle.

## Matched endpoints

|System|norm Δref|norm Δpred|norm ΔA|cos(Δref,Δpred)|L_SOURCE|L_CORPUS|ΔL|relative ΔL|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|H2|0.00114119644668|6.67112236579e-05|0.00107448587263|0.999990831808|0.050662338436|0.0506546766871|-7.66174893193e-06|-0.000151231647975|
|BeH2|0.0263383318134|0.0017340663671|0.024604278675|0.999992873575|0.0332284213972|0.0331468704883|-8.15509089744e-05|-0.00245425167809|
|CO|0.00102887913599|5.51278391208e-05|0.000974001988257|0.9956956462|0.0488242398555|0.0488209975364|-3.24231905149e-06|-6.64079781085e-05|

Matrix norms are orthonormal-AO Frobenius norms in Ha; loss is Ha² with the original n_AO denominator. ΔA=Δpred−Δref. Reference-only ||Δref||²/n_AO is not a lower bound on this loss. ΔL uses the same fixed overlap, gauge and basis.

|System|cross term 2⟨A_SOURCE,ΔA⟩/n_AO|quadratic term ||ΔA||²/n_AO|
|---|---:|---:|
|H2|-7.68099093011e-06|1.92419981747e-08|
|BeH2|-8.57548709822e-05|4.20396200774e-06|
|CO|-3.24796595549e-06|5.64690400671e-09|

The negative cross term dominates the positive quadratic term. The matrix residual changes measurably while the total loss shift remains −0.0151%, −0.2454%, −0.00664%. Those are continuous pilot results, not a universal R0 grid-sufficiency threshold.

## Frozen stage attribution

Exact stage definitions are in stage_definitions.json. R0 full SOURCE; R1 unique mapped indices; R2 original mapping/multiplicity; R3 corpus weights; R4 corpus vxc. R5 stored F64 descriptors; R6 production F32 descriptors widened; R7 cached phi; R8 cached AO derivatives; R9 actual production mixed local arithmetic. From R4 onward the original F64 RefAO is frozen. Prediction-only AO changes never alter RefAO.

|System/stage|reference increment norm|prediction increment norm|Δloss|
|---|---:|---:|---:|
|H2/R1|0.00114119603|6.67162617e-05|-7.66411827e-06|
|H2/R2|0|0|0|
|H2/R3|9.21952944e-09|5.9409009e-09|-5.62740338e-12|
|H2/R4|1.00700444e-08|0|3.19239505e-10|
|H2/R5|1.68217781e-13|1.71316322e-12|-1.32030965e-14|
|H2/R6|0|2.18778147e-09|-9.02658441e-12|
|H2/R7|0|2.09398328e-07|-9.25001434e-10|
|H2/R8|0|2.51146623e-09|-1.76793642e-12|
|H2/R9|0|1.03186237e-07|2.99153057e-09|
|BeH2/R1|0.0263383322|0.00173405467|-8.15531043e-05|
|BeH2/R2|0|0|0|
|BeH2/R3|5.09824e-08|4.59307037e-08|6.60393767e-11|
|BeH2/R4|2.46065199e-08|0|-7.60502997e-11|
|BeH2/R5|2.07327882e-12|2.05862877e-11|1.37946292e-14|
|BeH2/R6|0|2.26241325e-08|2.37274603e-10|
|BeH2/R7|0|1.33178265e-06|3.85973686e-09|
|BeH2/R8|0|4.20958824e-08|5.94871106e-11|
|BeH2/R9|0|4.48866763e-07|-1.95112765e-09|
|CO/R1|0.00102887946|5.49237146e-05|-3.2111518e-06|
|CO/R2|0|0|0|
|CO/R3|9.07568733e-08|8.30710957e-08|1.6850159e-10|
|CO/R4|8.6972577e-08|0|5.17171737e-10|
|CO/R5|1.03581027e-11|1.62541123e-11|-3.97745639e-14|
|CO/R6|0|2.54694007e-08|-2.22893916e-10|
|CO/R7|0|4.35693553e-06|-7.17291216e-09|
|CO/R8|0|7.79170281e-08|-4.63612955e-11|
|CO/R9|0|1.51738859e-06|-2.44107198e-08|

The saved attribution tables also give signed projections on the respective total matrix difference. Norms are not added as causal percentages. The sequential replacement order is prespecified and its attribution is order dependent.

## Mapping and historical reduction rule

|System|SOURCE / CORPUS|unique selected|unselected|excess mapped presentations|
|---|---:|---:|---:|---:|
|H2|47080 / 42508|42508|4572|0|
|BeH2|89200 / 77240|77235|11965|5|
|CO|85960 / 72110|72107|13853|3|

No distinct-coordinate F32 collisions were found. BeH2 and CO each have one group of identical SOURCE coordinates. The authenticated mapper chooses the first identical coordinate candidate; the extra mapped presentations have zero SOURCE and CORPUS weights, hence their matrix and energy effects are exactly zero. No records were deduplicated. Fixed-map deletion/excess bookkeeping closes independently. Historical ownership of coincident physical quadrature rows remains PARTIAL; no duplicate-data defect is inferred. Full per-source inventory with multiplicities and coincident-coordinate weight sums is outside Git, hash-bound by row_inventory.csv.

The counts were already reduced in the historical H5 sanity log. prepare_vxc merely copied the H5 grid and rounded data to F32. The corpus builder inherited those rows and restored original coordinates. A linked converter contains rho_total > rho_cutoff filtering, but its actual run/config linkage is missing; another helper only zeroes potential noise. Historical reduction rule: RULE_UNKNOWN, not a demonstrated low-density mask. No further broad search was performed. Checked paths, code ranges and hashes are saved in provenance.

## Numerical integrity and scope

SOURCE reference replays saved V_SOURCE. Cached CORPUS projection replays the prior gauge-audit V_CORPUS. R4 separately reproduces the true stored F64 RefAO: the previous cached-phi reference was a diagnostic path, not a replacement target. The final prediction independently replays assemble_rks_operator with actual F32 inputs and ConstantAdapter through LapEnergy precision repair; its scalar metric is the unchanged operator_loss definition. A fixed nontrainable zero sentinel selects that precision path; it is not a loaded NN checkpoint.

F64 arithmetic closure, independent selection/multiplicity sums, vxc-only prediction invariance, matrix/loss telescoping and cross-term identity pass. Chunk sensitivity is measured by reverse accumulation of the same R0 chunks. All inputs and production source hashes remain unchanged. Criteria were frozen before computation; no gauge shifts or overlap-mode removal.

matched_loss.csv also reports Q error, electron-number integrals, PBE energy, raw reference eigenpair residuals and occupied/virtual blocks with the original n_AO denominator. Saved MOs are complete and qualified; no virtual modes excluded.

|System|PBE XC energy change (Ha)|
|---|---:|
|H2|1.95348857535e-07|
|BeH2|3.22129154018e-07|
|CO|-2.77507782087e-07|

Tiny integrated-energy changes do not establish matrix accuracy. The large Q error and raw reference eigenpair differences are preserved, without fitted constants. This PBE GGA sensitivity test does not qualify the response of a trained Lap-functional. No chemistry, CBS, D3, all90 census or trained model was evaluated.

## Verdict and one next action

Transfer mechanism: selection, with minor weights/vxc precision and prediction descriptor/AO/local-arithmetic effects. No significant conversion defect is demonstrated. Matched PBE loss impact: partial cancellation and a small systematic downward shift on this pilot. Neither replacing only RefAO nor changing gauge is justified.

One next experiment: a separately frozen matched full-SOURCE-grid representation pilot that rebuilds both prediction inputs and reference together and evaluates the unchanged operator metric. It must not reuse a full-grid target with subset prediction. This report does not authorize its execution or any training.

Active numerical CPU time: 55.881s (includes read/hash/array operations); pilot profile [4.114, 21.18, 25.313]s. Scientific wall through evidence package: 772.262s / 5400s. No GPU compute. Tests and final checks are recorded in decision.json. Unknown historical rule and coincident-row lineage remain explicit limitations; the numerical three-system attribution is complete.

Trained checkpoints loaded: 0; optimizer states loaded: 0; training updates: 0; SCF: NO; gauge shifted: NO; production grids/targets/loss/architecture modified: NO; new systems/Clean28 access: NO.

R0–R4 use bounded source-index accumulation with exact fixed coefficient vectors for the specified selected indices and mapped multiplicities. This is algebraically the same R2 presentation list; computational summation order is reported rather than claimed bitwise identical. Direct CORPUS-order endpoint replay and reverse-chunk arithmetic checks constrain the observed reduction-order difference.
Validation receipt: {"Ruff": "PASS", "compileall": "PASS", "focused_tests": "6 PASS", "git_diff_check": "PASS", "historical_tracked_files_changed": 0, "pytest_seconds": 4.64}.