# XC target provenance recovery

Verdict: primary contracts remain UNRESOLVED; concrete external materials are enumerated. No implementation/target mismatch was confirmed. Search is closed, not deferred to an unspecified further search.

Start HEAD `ff8c86ce8b17e7381d3c9305a22188180afd03be`; branch lap_full_vxc; later/untracked work preserved. First-action stopwatch covers search, metadata, scalar replay and evidence preparation. Previous operator-contract audit classifications and calculations were reused unchanged; no PBE/AO/general operator audit repeated.

| Question | Confirmed producer/consumer boundary | Remaining blocker | Evidence scope |
|---|---|---|---|
| legacy E_xc | H5 import into F32 at 6f91e72; exact downstream preservation; surviving sanity log gives E_cbs_1−E_no_xc | original H5 component exporter and CBS method/config | all90 preservation reused; 3 pilot log rounding checks |
| vxc_grid/Vrho | NPZ values→legacy→RefAO | exact inversion/export implementation and gauge application | all90 preservation reused; 3 pilot alignment values reused |
| D3 | active binary artifact exactly matches 4ed868e; PBE0/d3bj utility; integral+D equation | input geometry/backend/params/run and author-intended energy convention | all90 artifact/consumer, historical run unknown |

The new sanity CSV corroborates a historical energy-residual recipe, but does not authenticate its generating source or make a CBS/WF reference an exact functional value at n_KS. exc_wf/t_wf source meanings remain unqualified. Approximate HOMO/i_ekt alignment cannot establish a gauge applied to saved vxc_grid. The code necessarily fits the NN integral to E_target−D; that operative fact does not prove the author's intended physical convention.

New scalar replay: all three named H5 rows satisfy the explicit logged subtraction within a predeclared decimal/F64 bound; their F32 values equal legacy exactly. No alternative CBS formulas were searched for numerical fit. No D3 evaluation was run because the original geometry/config/backend is missing. No new NN, AO, inversion, SCF or reference calculations were performed.

Sources, commits/hashes, code ranges and coverage/evidence levels: evidence.json. Checked locations, queries and stop reasons: search_log.csv. Three focused contracts: energy_contract.md, potential_gauge_contract.md, dispersion_contract.md. The only next action is obtaining the six narrowly specified materials in missing_materials.md from the original exporter. The neutral request is a draft; nothing was sent. Current evidence does not authorize a repair or new training.

Wall time at evidence generation: 927.71 s / 7200 s; search closed at 927.70 s / 3600 s. Final test/Git elapsed time will be separately recorded from the same first-action clock. No unmeasured preliminary search.

Validation: 8 focused parser/rounding/receipt tests PASS (0.30 s); Ruff, compileall and git diff --check PASS, without new style suppressions. No old scientific FAIL or tolerance changed. Initial sandbox temp-path failure was resolved with a fresh workspace temp; a descriptive archive-source count was corrected to the actual inventory of17. Finalization wall time: 1071.98 s from the first action; budget7200 s respected.
