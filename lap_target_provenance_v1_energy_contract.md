# Legacy XC energy contract

Status: UNRESOLVED (primary semantic definition); verified downstream numerical preservation.

Current target is the F32 legacy E_xc (Hartree), widened to F64 by the central builder. Direct source evidence: prepare_vxc.py:49–52 at commit 6f91e72fb066cf894242beca60639d5a0f804873 imports the stored H5 E_xc as torch.float32. It does not calculate it. Builder:155–165,229–242 preserves it and explicitly records the unresolved original formula. Source hashes are in evidence.json.

New source: `C:\Dev\readWFN_share_ms\readWFN_share_ms\h5_vrho_from_mrks_sanity.csv` (SHA256 `4b39e7cf9b20d682ed6f6a90f60e89748086b5b710ffb4d462580700543b89e9`), 90 named H5 rows. Columns explicitly contain E_xc, E_cbs_1 and E_no_xc; the pilot identity E_xc=E_cbs_1−E_no_xc closes within fixed decimal-source precision, and casting logged E_xc to F32 reproduces legacy exactly. This surviving log supports a historical residual recipe but has no authenticated generating script/config; it does not independently prove how E_no_xc or the CBS reference were calculated.

Reuse the prior three-system matrix receipt, not new AO work: E_CBS_1−E_nuc−Tr(P_KS T)−Tr(P_KS V_ext)−0.5Tr(P_KS J_KS) agrees with legacy at source precision. Separate components were generated in the previous audit on stored dm_ks. The reference CSV headers mention SCF 5Z/QZ and correlation 5Z/QZ values; neither the method nor extrapolation law, reference basis, charge/geometry binding or empirical corrections are authenticated by headers alone.

A (actually executed exporter): unavailable. B (author physical label): downstream consumers call it exact E_xc, which is not primary proof. C (strictly established): the stored target and importer rounding plus numerical residual-recipe support. D (approximations): finite basis/inversion and nonidentical WF/KS densities measured previously; a CBS scalar cannot be declared exact E_xc[n_KS] without matching reference construction/density contract.

exc_wf numerically equals half the rho_WF*vxc_hole integral on the previous pilot; t_wf is a stored scalar label. Their producer formulas remain unresolved. An interaction XC quantity W_xc is not full E_xc=W_xc+T_c. T_WF−T_s across different densities is not an exact same-density T_c identity. No target is replaced; E_xc=integral(n*v_xc) is not assumed.
