# Dispersion contract

Status: author-intended physical convention UNRESOLVED; operational code/artifact lineage VERIFIED.

New direct historical evidence: utility created at 966ba34a8ceadafc12b5adb0a720c01bdec5d6dd. Optional mRKS-D3 support and the active artifact were added together at 4ed868e860d07b2d142d115008d25e2c7a8a3ca0. The artifact at that commit has SHA256 `a2d5b556a5007fa31505c5e38a2be537d4f1755f961c31ef2c49d07605087440`, exactly equal to the active 90-name pickle; this was a binary hash comparison, not optimizer/model deserialization. Current JSON and dictionary equality/arithmetic are already verified in the prior audit.

Producer compute_mrks_pbe0_d3bj.py:54–92 loads H5 atom/charge/spin, assumes Bohr, constructs a geometry carrier, and requests dftd3.pyscf.DFTD3Dispersion(xc='PBE0',version='d3bj'). It does not explicitly set numerical damping parameters, backend version or optional extra terms. The default original H5 directories under documented PROJECT_ROOT choices are absent; the sanity CSV and JSON survive. No actual run command, input H5 geometry hash, backend version/parameter table or settings receipt was found. We do not substitute current D3 defaults or perform a modern replay as evidence of an old version.

Current objective mathematically uses prediction=NN_XC_integral+D and target=stored_E_xc, so the integral is fitted to stored_E_xc−D. In an additional historical H5 consumer, finetune_scf_mrks.py:681–705 explicitly returns E_no_xc+NN_XC_integral+D for comparison with E_cbs_1; this is consumer code, not an authenticated generating run or primary author declaration.

A: whether E_target was intended as pure full ab initio XC is unresolved. B: the operative integral is residual under D addition, verified in code. C: optional inclusion is documented by the source commit/functions, but intended full-XC/residual semantics are not explicitly authenticated. D: current pipeline agrees with that operative equation; agreement with the author's original physical convention cannot be certified. No mismatch is asserted merely from adding D to an ab initio target. Geometry-only D has zero density-matrix derivative, so no D term in the AO operator is not an independent defect.

Reuse dispersion_contract.csv from operator-contract audit for all90 signs, units, name matching, effective targets and shifts. No D3 method sweep, target edit or recomputation was made.
