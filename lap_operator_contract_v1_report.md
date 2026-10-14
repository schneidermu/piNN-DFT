# Independent XC operator and target-contract audit

Verdict: TARGET_PROVENANCE_UNRESOLVED + NUMERICAL_OR_METRIC_SENSITIVITY_IDENTIFIED. No production energy/operator/loss implementation bug demonstrated within tested scope. This is not a proof that the targets are physically correct.

Frozen HEAD: `5a49427d81dde3431e3805072e3d64e81718cbce`; branch `lap_full_vxc`. Versions: `{'python': '3.10.12', 'torch': '2.14.1+cpu', 'pyscf': '2.14.0', 'libxc': '7.0.0'}`. Initial worktree, hashes, tolerances and source/array receipts are saved in protocol/provenance. No trained checkpoints or optimizer states were loaded. Only H2, BeH2, CO received molecular numerical checks; all90 census was hashes/metadata/scalars/dispersion only.

## Target chain and limits

Primary inversion/export code was not found in the bounded checked locations. NPZ → legacy scalar potential row mapping → central corpus → current publication corpus is authenticated. The legacy pickle hash matches the builder; all90 NPZ and central-file hashes match, and current publication pilot file hashes match conversion receipts. Geometry/basis/AO ordering, total RKS dm_ks, restored NPZ coordinates, signed legacy weights and row selection are retained.

Current E_xc is the legacy F32 value widened to F64, in Hartree. The original mathematical definition remains TARGET_DEFINITION_UNRESOLVED. Builder [target preservation](train_models/build_lap_operator_data.py:229) explicitly says “original formula unresolved; NPZ exc_wf not substituted”. [prepare_vxc](train_models/prepare_vxc.py:49) copies an H5 E_xc, it does not establish its formula.

On the three pilots, the recipe E_CBS_1 − E_nuc − Tr(P_KS T) − Tr(P_KS V_ext) − 0.5 Tr(P_KS J_KS) reproduces legacy E_xc at F32 source resolution. This is numerical evidence for a total-energy-residual recipe, not authentication of its original exporter or of the CBS energy calculation. Nuclear energy is explicitly excluded before the electronic decomposition. The resulting scalar is not assumed to equal ∫n v_xc.

| System | Recipe − legacy, Ha | ∫|n_KS−n_WF|, electrons |
|---|---:|---:|
| H2 | -4.78e-09 | 0.000633653 |
| BeH2 | -9.08e-08 | 0.00588513 |
| CO | 9.35e-08 | 0.0112121 |

NPZ exc_wf equals 0.5∫n_WF v_xc_hole numerically on the pilots (zero stored-value discrepancy). This does not by itself authenticate the exporter definition; an interaction XC energy W_xc must not be substituted for full E_xc=W_xc+T_c. Stored t_wf and computed T_s use different WF/KS densities; density differences are nonzero. Their difference cannot be asserted as an exact same-density identity here.

Legacy Vrho matches NPZ vxc_grid at the original row identities, with only F32 preservation error. A discovered gen_h5_with_vrho exporter uses local Libxc vrho; it is NOT authenticated as the producer of these NPZ/legacy records and is not evidence that the current target is a partial derivative. Full common-RKS potential semantics and original gauge remain UNRESOLVED. Numeric HOMO/IP metadata evidence is recorded separately. No stationary-condition check was run because original F_ref Hartree/gauge conventions are not authenticated.

## Dispersion

The active path is [integrated_energy](train_models/lap_vxc.py:326) → [make_mrks_objective_factories](train_models/lap_moo_training.py:685) → [_add_mrks_dispersion_once](train_models/optuna_joint.py:418) → [batch_exc](train_models/optuna_joint.py:806). It adds one exact Name-key D scalar in Ha. All90 keys exist; missing keys would silently add zero; enabled with no dictionary raises. batch_exc converts singleton absolute/RMSE error to kcal/mol once. Effective pure-NN target is E_target−D; every scalar and loss shift is in dispersion_contract.csv.

The current dispersion JSON exactly matches the existing mrks_pbe0_d3bj JSON. Its [producer](train_models/compute_mrks_pbe0_d3bj.py:89) requests xc=PBE0, version=d3bj, Bohr geometry and no SCF. Original generating H5/geometry receipt and installed D3 parameter table were not available. Arithmetic/name/unit contract PASS; physical compatibility with unresolved E_xc and exact generating geometry/parameters UNRESOLVED. Geometry-only D has dD/dP=0; its absence from the AO operator is not a defect. No labels or corrections were changed.

## Independent PBE

Oracle: PySCF NumInt.nr_rks(PBE) at exact original dm_ks, coords and weights, no SCF/grid regeneration. nr_uks with dm_alpha=dm_beta=dm_total/2 agrees with RKS. Oracle never calls repository PBE formula/local-partial/AO-assembly helpers; AO evaluation library and source inputs are shared. The production canonical adapter explicitly emits PBE constants (Gx=0, Gc=1); zero NN weights are not used.

| System | Repo F64 energy error, Ha | Repo F64 orthogonal-matrix relative error | Frozen gate | Mixed gate | Ideal F64 relative error |
|---|---:|---:|---|---|---:|
| H2 | 2.25104e-06 | 1.27919e-05 | FAIL | PASS | 7.38e-14 |
| BeH2 | 5.11285e-06 | 2.67505e-05 | FAIL | PASS | 1.02e-13 |
| CO | 6.17139e-06 | 1.79527e-06 | PASS | PASS | 2.97e-13 |

Original independent repository-F64 gate remains FAIL for H2/BeH2; it was not widened. Ideal F64 unpolarized PBE/PW_MOD agrees to machine precision; it is a diagnostic different numerical functional. Production rs adds 1e-7 to total density; xs adds 1e-10 to density and its 8/3 power to sigma; constants are originally F32 and PW92 unpolarized A is rounded to 0.031091 rather than 0.0310907. Production z_thr is a no-op. The isolated source-copy comparison shows that precise constants alone barely reduce the matrix discrepancy; removing density/sigma additions as well reduces it to ~1e-13 relative. Thus conventions, predominantly regularization, explain this oracle discrepancy; widening F32 source values alone does not recover exact PBE. All counterfactuals leave production source untouched.

## Variational assembly and metric

Nonzero rho/sigma/Laplacian polynomial: independent symmetric-DM energy derivative versus production weak AO assembly, scalar operator-loss parameter gradient and full symmetric δE=Tr(VδP) contraction PASS. A fresh untrained R0 fixture passes density-chain checks; the separate full-9446-coordinate Gc-enabled regression compares every operator-loss parameter derivative. Initial default-constructor fixture had Gc disabled (9381 coordinates); no model quality was evaluated. Three deliberately wrong copies (extra spin factor, missing Laplacian AO terms, wrong sigma mapping) were detected. Three Laplacian product terms are separately regression-tested.

Original audit fixture FAIL is preserved in variational_checks.csv. It differentiated a graph assuming symmetric DM using unconstrained independent entries, producing an irrelevant antisymmetric derivative. Explicit symmetric leaf parameterization fixes the diagnostic graph; variational_corrected_checks.csv contains the corrected results. This was an audit defect, not a production bug. Original tests and tolerance limits were not weakened.

Independent real reference projection uses NumPy contraction ∑w v φφ, not the builder helper. RefAO relative gaps are ~4–6e-17; chunk137 agrees. Constant-field fixture is cQ, not assumed cS. operator_loss self-loss/gradient, known shift, independent 2 S^-1 ΔV S^-1/n gradient, and well-conditioned same-basis-span invariance PASS. No overlap modes were discarded.

| System | cond(S) | ||XQX−I||₂ | PBE−mRKS loss, Ha² | trace component fraction | discrete-Q component fraction |
|---|---:|---:|---:|---:|---:|
| H2 | 15662.9 | 0.00817744 | 0.050654678 | 99.2218% | 99.2310% |
| BeH2 | 27103.7 | 0.158449 | 0.033146867 | 89.9960% | 90.1520% |
| CO | 31563.8 | 0.00585986 | 0.048821005 | 70.8630% | 70.8648% |

All overlaps pass the existing numerical SPD gate. BeH2 quadrature is notably incomplete in some AO directions despite good electron-number checks. The exact cI and c_Q XQX decompositions close to ~1e-17. Trace dominance is a property of this PBE−mRKS residual, not proof of an arbitrary target gauge defect. Q≠S prevents automatic interpretation of cI as a constant grid-potential shift. No c or c_Q was applied to targets.

Occupied/virtual decomposition uses authenticated saved orbitals/occupations and verified CᵀSC≈I (largest Frobenius gap ~2.1e-12). Most loss is virtual–virtual on this pilot; this is disclosed, not removed or declared physically meaningless. Source-F64 versus preserved-F32 vxc projection errors are only ~1e-8–9e-8 orthogonal Frobenius on the pilot. This does not bound arbitrary basis expansions or explain trained-model plateaus; trained weights were not read.

## Decision and next step

No confirmed production implementation bug or authenticated target-contract mismatch. The energy formula and potential gauge/semantics are incompletely qualified; significant quadrature/metric structure and small regularization-driven PBE mismatch are measured. The single next justified task is to recover the original mRKS inversion/export source plus legacy E_xc construction and D3 geometry/parameter receipts, then authenticate the target energy/gauge contract. Do not change operator loss, gauge, energy penalty or training before that evidence is resolved.

No new molecular systems, training, SCF, trained model or optimizer reads, development/test access, production repairs or target/loss changes. No conclusion is made about the effectiveness of physics→chemistry training.

Measured scientific workflow elapsed since protocol freeze: 1094.69 s / 10800 s. Instrumented initial computation: 45.72 s; supplement: 15.07 s. Preliminary source inspection preceded protocol freeze and was not separately timed; no molecular computation occurred before freeze. Metadata/hash/report/test work is not claimed as GPU compute.

Validation: 30 focused tests PASS (7.41 s); Ruff, compileall and git diff --check PASS. One local S102 annotation documents execution of the hash-authenticated PBE source copy; no global style ignore or numerical threshold was added. The initial test of 9446 coordinates exposed a default-constructor Gc flag omission; the dedicated full-coordinate test explicitly enables Gc and retains the exact count assertion. Executed-supplement hash and current lint-clean source hashes are both recorded.
