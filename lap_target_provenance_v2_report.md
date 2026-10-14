# Primary CBS/XC evidence addendum

New evidence incorporated: PARTIAL. CSV original is authenticated; original private screenshots/formula image are unavailable. Definitions/formulas are incorporated from the user’s explicit transmission of supplier correspondence, not claimed as an image reading. No message dates, software versions or method labels are invented.

CSV: `C:\Users\schne\Downloads\extrapolation_e (1).csv`; SHA256 `4040c5ddd21cb73edcf6f54356114ff2cf216fe47346e227c5ca83180fde7e4e`. It is byte-identical to the historical CSV, including row order and numeric formatting. All90 exact System names, schema and finite values pass. A raw copy is retained outside Git; no private screenshots are committed.

| Topic | BEFORE | NEW EVIDENCE | AFTER |
|---|---|---|---|
| energy recipe | primary recipe unresolved | user text transmitting supplier recipe E_CBS_1−E_noXC[loaded DM], no SCF | DOCUMENTED_BY_PRIMARY_CORRESPONDENCE |
| CBS column choice | numeric support, author selection unavailable | explicit supplied recommendation E_CBS_1; CSV hash identical to historical source | E_CBS_1_DOCUMENTED |
| CBS arithmetic | extrapolation formulas unresolved | literal user-transmitted formulas all90 replay | PASS |
| actual target linkage | pilot residual/log rounding verified | all90 selected CBS scalar/log noXC/H5-log residual/F32/current corpus; independent noXC components on pilot | ALL90_VERIFIED |
| potential declared semantics | unresolved primary label | supplier via user describes final grid XC potential and converged KS dm_ks | FINAL_XC_POTENTIAL_DOCUMENTED |
| gauge | UNRESOLVED | no gauge rule supplied | UNRESOLVED |
| method/basis | UNRESOLVED | basis metadata exists, SCF/correlation column method and exact family not specified | PARTIAL |
| D3 intended convention | UNRESOLVED | new correspondence says nothing about pipeline-added D | UNRESOLVED |

CBS_1 max absolute replay gap: 0 Ha; CBS_2: 0; CBS_corr: 0. Binary64 operations follow the written formulas literally; predeclared bounds remain unchanged. e_cbs_none maps numerically to E_CBS_corr without renaming source columns. Delta columns reproduce the stated signed differences; original intent between signed/abs is not independently inferred when all differences are positive.

Alternative CBS_2: literal formula reproduces its CSV column, but differs from the two-point exponential-model solution on 90/90 rows; maximum difference 0.00568641 Ha. For E_X=E_inf+A exp(−1.63X), E_inf=E5+(E5−E4)/(exp(1.63)−1). This is a separate algebraic check, not a correction to the supplied formula. CBS_2 was not used in the verified scalar target chain; no labels or CSV values changed.

All90 linkage status: ALL90. Exact identity mapping joins CSV System to historical {System}.h5 rows and current corpus source_id. E_cbs_1 log values agree with CSV_1, logged E_noXC is consumed directly (not obtained from target), F64 subtraction then one F32 storage reproduces legacy, and current Exc equals that legacy value exactly. Original H5 bytes/export script are still absent; this is authenticated scalar/log linkage, not full reconstruction of all90 electronic calculations.

H2/BeH2/CO additionally reuse independently computed E_nuc+Tr(P T)+Tr(P V_ext)+0.5Tr(P J(P)) receipts and agree with logged noXC. Those stored components use the original fixed KS DM, geometry/basis/AO order and include nuclei; no exchange/XC/dispersion is added. This independent physical noXC coverage is PILOT, not all90. No SCF, inversion, new density/grid or matrix census was run. Separately rounding CBS and noXC to F32 is not the qualified storage order.

Target wording: “XC-energy reference, constructed as a CBS total-energy residual after subtraction of the no-XC energy evaluated on the supplied fixed KS density matrix.” This is not a claim of exact universal E_xc[n_KS]. It does not replace targets with exc_wf or invent T_c across differing densities.

Supplier-declared field semantics (via user text): mo_coeff AO-basis coefficients; mo_energy Ha; mo_occ 0/2; vxc_grid final grid XC potential; dm_ks converged KS DM; grid_coords/atom_coords Bohr; grid_weights quadrature; serialized basis. The previously authenticated all90 manifest is consistent with downstream use of these fields. Recommendation to replace old systems is documented, but the named mrks_90_ccsd_pt.tar.zst archive bytes were not provided/authenticated here; its name does not prove CCSD or CCSD(T). Gauge and full inversion implementation qualification remain separate.

No confirmed target mismatch warrants a repair. Keep the previous repository-F64 PBE FAIL, operator/RefAO PASS and metric findings unchanged. Remaining request is limited to gauge, electronic method/basis pedigree, and intended D3 residual convention; original image availability is listed separately. D3 questions are addressed to the source of its training-pipeline addition, not attributed automatically to the mRKS supplier.

Wall at computation/report generation: 715.34 s / 3600 s. Trained checkpoints/optimizer loads:0; training:0; SCF:NO; production/targets/loss changes:NO; messages:NO; Clean28/future-test access:NO.

Precision control: rounding the CBS and noXC operands separately to F32 would disagree with 79/90 legacy targets. Correct F64 subtraction followed by one F32 cast agrees with all90.

Validation: 8 focused parser/formula/identity/rounding/noncircular-linkage/receipt tests PASS (0.46 s); Ruff, compileall and git diff --check PASS. Historical artifacts and production source hashes unchanged. Finalization wall: 743.96 s from the first action, within3600 s.
