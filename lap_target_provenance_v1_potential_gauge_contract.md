# Potential and gauge contract

Status: potential semantics UNRESOLVED; GAUGE_UNRESOLVED.

Verified downstream: exact NPZ vxc_grid row identity -> legacy Vrho (F32) -> central common-scalar projection -> RefAO. All90 preservation and pilot independent projection receipts are reused. The consumer applies no new shift. This does not identify the original inversion functional/algorithm or determine whether a shift preceded NPZ export.

The authenticated MRKS.zip contains generic WFN/grid helpers, not a source creating inp_mrks.npz/vxc_grid. gen_h5_with_vrho.py:197–203 uses a Libxc local vrho, but its lineage to these current NPZ records is absent; it must not be substituted as their producer. convert_vxc_data_with_wfn.py:223–267 imports a separate scalar vxc H5 schema; no mapping to current90 is authenticated. The archive vxc_data_2 contains a sample HeH+ H5 with e_fci and vxc but no E_xc/config/gauge; only schema was inspected, not new molecular numerics.

NPZ input_file identifies remote ORCA paths /home/mmedvedev/lsvvt/chep/ready/{H2,BeH2,CO}/inp.orca.zst. No remote source access or author identity is inferred from that username. Original input/output, inversion code, export assignments, and gauge/convergence logs are missing.

Reused pilot HOMO+i_ekt residuals (Ha): H2 +5.898011167193573e-05; BeH2 −3.0889428246916317e-06; CO −1.0894252772397639e-04. These are approximate alignment evidence, not exact equality or proof of an applied vxc_grid shift. A rule tied to EKT/IP, asymptotic zero, or another convention remains unverified. Need the assignment modifying vxc_grid and whether orbital energies were shifted in the same export state.

Stored dm_ks is used as total closed-shell RKS density downstream; original WF/KS inversion semantics, filtering/masking/interpolation/tail treatment require primary source. The consumer's common-RKS use is verified; the upstream potential's physical meaning is not. No constant shift, target subtraction, cQ->cS substitution or metric change occurs.
