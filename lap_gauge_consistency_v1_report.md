# Stored mRKS gauge and eigenpair consistency

SOURCE potentials reproduce the saved orbitals and eigenvalues for H2, BeH2 and CO. The actual CORPUS grid path has a measurable nonconstant eigenpair discrepancy. One constant cannot remove it. The measured SOURCE result supports a common potential/orbital gauge; its EKT alignment is approximate at the reported metadata resolution, not an exact all90 identity.

Frozen HEAD: `f6635a5c7d487e62f73d89975c15a081ffb95dd0`. Python: `3.10.12 (main, Aug 31 2026, 10:18:17) [GCC 11.4.0]`; PySCF 2.14.0, NumPy 2.2.6, SciPy 1.15.3. All inputs are hash-bound in provenance. The baseline worktree was recorded without checkout/reset. Historical reports and raw NPZ/H5 remained byte-identical.

## Definitions and reporting resolution

`F=hcore+J(dm_ks)+Vxc`, with total RKS DM, Coulomb only, all-electron pilot basis, no SCF/exchange/D3. `R=S^(-1/2)(FC−SC diag(epsilon))`. Relative denominator is `||S^(-1/2)FC||F+||S^(-1/2)SC diag(epsilon)||F`; absolute norms below remain primary. All saved MOs are complete on this pilot. Occupations define HOMO and occupied/virtual masks. Signed metadata discrepancy is saved HOMO + positive raw `i_ekt` in Ha. Original unit annotation is absent; this uses the task/energy-bundle convention. Eigenpair maximum-orbital reporting resolution was frozen at 1e-4 Ha; it is not a claim about exporter convergence. F64, MO, DM, electron-number, RefAO and fitted-constant gates are frozen in protocol.

Implementation: [residual and shift arithmetic](tools/lap_gauge_consistency_v1.py#L41), [fixed-density molecular assembly](tools/lap_gauge_consistency_v1.py#L141). Molecule reconstruction follows the authenticated metadata contract in [original builder](train_models/build_mrks_stencils.py#L171), without importing its physics helpers.

## All90 metadata (not all90 potential qualification)

Coverage: 90/90, raw i_ekt available: 90, missing: 0. Median signed delta 0.000102205445176 Ha; median absolute 0.000144765318449 Ha; maximum absolute 0.000820202511447 Ha. Counts within 1e-6 / 1e-5 / 1e-4 / 1e-3 Ha: [0, 3, 32, 90]. These bands describe scale, not confidence intervals or training criteria. Previously published pilot values agree with the primary NPZ.

## Pilot raw eigenpairs and constant-component diagnostics

|System/path|max orbital R (Ha)|HOMO R (Ha)|occupied R Fro (Ha)|virtual R Fro (Ha)|full cQ to ADD (Ha)|max R after cQ (Ha)|occupied cQ (Ha)|
|---|---:|---:|---:|---:|---:|---:|---:|
|H2/SOURCE|5.93927305e-11|4.45008201e-11|4.45008201e-11|2.80878651e-10|2.32657514e-11|3.99917973e-11|2.67044032e-11|
|H2/CORPUS|0.000757286469|2.74347917e-05|2.74347917e-05|0.0011408492|-2.41455056e-05|0.000740611933|-1.9584564e-06|
|BeH2/SOURCE|1.88664624e-09|5.21846568e-10|1.15266841e-09|8.99448778e-09|-2.22107652e-10|1.89092712e-09|-2.16178019e-10|
|BeH2/CORPUS|0.0105239515|0.000165264284|0.000207551605|0.0263375215|-0.000317296504|0.010370379|-9.91069199e-07|
|CO/SOURCE|1.84952379e-08|3.26147365e-09|2.26675371e-08|9.36010043e-08|-2.92289221e-10|1.82159972e-08|-3.46667608e-10|
|CO/CORPUS|0.000390131972|8.70738976e-06|2.02811878e-05|0.0010286666|-1.08227503e-05|0.00038533376|-2.41317631e-07|

SOURCE: all three `NUMERICALLY_CONSISTENT_AT_REPORTED_RESOLUTION`. CORPUS: all three `NONCONSTANT_INCONSISTENCY` at the predefined reporting resolution. This identifies a downstream numerical grid path difference, not an erroneous primary potential or proof of a production implementation defect. SOURCE fitted constants are below 1e-6 Ha; there is no resolved relative SOURCE gauge offset. CORPUS full-space fits are negative, but occupied fits are far smaller; a single gauge shift is inadequate. All original raw residuals are preserved. The WF-Hartree counterfactual was not triggered.

MO orthonormality Frobenius errors are below 2.11e-12; reconstructed MO DM equals saved DM exactly. Electron counts are 2, 6, 14 within 4e-15. No MO correction, overlap-mode removal, virtual-space exclusion or diagonalization update was used. Per-orbital residuals, diagonal offsets, offdiagonal norms and residual outside MO span are saved in eigenpair_checks.csv. Q and S fits, full and occupied, use one fixed unchanged F; chunk-order fit variation and conservative F64 arithmetic bounds are in shift_decomposition.csv.

## SOURCE versus CORPUS

|System|SOURCE / CORPUS points|orthogonal delta V Fro (Ha)|CORPUS F32 AO vs RefAO (Ha)|F64 AO vs RefAO (Ha)|SOURCE / CORPUS ||XQX−I||2|
|---|---:|---:|---:|---:|---:|
|H2|47080 / 42508|0.00114117903|3.77858521e-07|6.72925765e-14|2.6083022e-09 / 0.00817740532|
|BeH2|89200 / 77240|0.026338339|1.54183769e-06|3.17623107e-13|3.16948679e-06 / 0.158448878|
|CO|85960 / 72110|0.00102886598|5.0096487e-06|2.457788e-13|3.75440336e-06 / 0.00585980067|

SOURCE coords, weights and potential are F64. CORPUS retains the actual subset, widened legacy weights/potential and cached F32 AO. All tested weights are positive. Exact source-row coordinate correspondence, molecule/AO labels, DM and overlap were verified. BeH2 contains 6 legacy rows mapped to one identical source coordinate; CO has 4. They were preserved without deduplication. The F64 CORPUS projection reproduces RefAO under its strict gate; the cached-F32 path passes the separate source-precision gate. Its AO rounding contribution is much smaller than the measured SOURCE→CORPUS matrix difference. The total difference includes subset and legacy weights/potential precision; this audit does not apportion those components further. Q−S is not a universal bound on integration error of a nonconstant potential.

## Gauge conclusion and limits

SOURCE HOMO+EKT is +5.8980085e-5 Ha (H2), −3.0885717e-6 Ha (BeH2), −1.0894259e-4 Ha (CO). SOURCE HOMO residuals are negligible on this scale, so their Rayleigh values can be interpreted as saved eigenvalues. H2/BeH2 pass the frozen 1e-4 Ha EKT band, CO lies just outside; all are within 1e-3 Ha. The data numerically support approximate common EKT-aligned gauge on the pilot, not exact alignment at 1e-4 Ha for every system. A small CORPUS Rayleigh/IP difference alone cannot qualify eigenpairs: BeH2 HOMO residual exceeds 1e-4 Ha. No historical exporter order or absolute alignment procedure was reconstructed. All90 matrix consistency remains untested.

One next step: separately audit the existing SOURCE→legacy/CORPUS grid conversion on this same pilot to attribute subset versus stored-value precision effects, preserving primary potentials and orbital energies. No gauge repair or training is justified by these fits.

## Integrity and resources

Focused fixtures: exact SPD eigenpairs, cS/cQ shifts with correct additive sign, nonconstant error, misleading EKT metadata, incomplete MO space, degenerate-space rotations and immutable inputs; no SCF or model dependency. Run focused tests, Ruff, compileall and git diff --check before committing. The initial diagnostic uniqueness assertion was overly restrictive for repeated grid coordinates; it was removed without changing any input or computed scientific value. Repeated rows are now explicitly recorded. Only these three molecules were reconstructed; all90 work was metadata/hash reading.

Scientific wall through final integrity: 945.338 seconds, budget 5400 seconds. Measured complete CPU execution passes: 15.693 + 11.117 + 11.920 seconds; interrupted validation execution ≤9.353 seconds, plus focused tests and report/hash verification. No GPU computation. Idle, coding and reporting time are included in wall time, not labelled GPU compute.

Trained checkpoints/optimizer states loaded: 0. Training updates: 0. SCF: NO. Targets, potentials, production code/operator loss: unchanged. New systems/messages/Clean28: NO.

Validation receipt: {"Ruff": "PASS", "compileall": "PASS", "focused_tests": "7 PASS", "git_diff_check": "PASS", "historical_tracked_files_changed": 0, "pytest_seconds": 0.41}.