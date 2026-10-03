# Independent frozen-geometry review

**Decision: accept the geometry/provenance gate as CASE B.** The immutable cursor-2 replay is valid, but no tested positive tau satisfies the predeclared robust all-objective cosine gate. Do not select a tau or run PCD Armijo from this result; proceed to the separately specified native Fliege–Svaiter diagnostic. Armijo has not run.

## Provenance and isolation

- Repository HEAD is the requested `5b913c02545e4528562ffef69abda24cc999acb0`.
- The replay binds to cursor 2 / sample NCCE31 reaction 0 BH `level2_mura`, protocol PCD tau 0.02. Checkpoint SHA-256 is `a53f46807b1ce492c8fe02657e615881ff310294bd79d273eb4bb0c694e3670c`; raw-gradient NPZ SHA-256 is `fd5eb469339d68ac3c7fc21728ce0138cec878ea8780c7c05277d87cf353f6c0`. Driver SHA-256 is `151e8d4a9c36671632fcc2a0e1ead26db7a3d3011876fd5408a7f6772d9f7882`; result JSON SHA-256 is `34d48cad8c26718a59e0c31feeb79536569a0ac2bbbaed3ba595408b72a58071`.
- Raw objective gradients reproduce the saved arrays exactly (max absolute difference 0; 9,446 coordinates per objective). All six runs start from the same deep-copied incoming EMA (`t=2`, same `v`) and advance only scratch normalization to `t=3`; normalization outputs agree across taus. The aggregator does not mutate its scratch EMA.
- The receipt and driver checks confirm model/buffer state, RNG, incoming EMA, cursor/sample, checkpoint, and raw-gradient artifact remain unchanged. Thus the six geometries are comparisons from one immutable incoming state, not sequential training steps.

## Frozen rule and result

The pre-run spec fixes the grid to `{0, 0.001, 0.0025, 0.005, 0.01, 0.02}` and the positive-tau feasibility rule to every objective cosine `>= 1e-6`; among passing positive taus, select the smallest. The spec timestamp precedes the driver run. Tau selection is geometry-only; Armijo and any panel pass/fail cannot choose tau.

At tau 0, raw `g·d` is chemistry `15310.5998`, E_xc `−0.0009186` (numerically flat), and operator `5.67153`. The independently checked tau-zero cone projection matches the canonical direction (relative error 0); KKT residual is 0 and projection identity error is `5.29e-23`. E_xc is the active face; its flat first-order slope is allowed for this projection check. Tau zero is not a positive-tau selection.

Every tested positive tau is QP-feasible and has E_xc as its only active face, but each has negative chemistry cosine: −0.305226 at 0.001, −0.536649 at 0.0025, −0.612934 at 0.005, −0.649675 at 0.01, and −0.667573 at 0.02. Therefore none meets the all-objective `1e-6` gate and the smallest-positive-feasible set is empty. This is the predeclared CASE B outcome, not a reason to change the margin or grid.

The tau-0.02 actual-displacement replay also reproduces the adverse chemistry slope (`+0.0093064`) alongside favorable E_xc (`−0.4226231`) and operator (`−3.29078e-5`) slopes. Sign is consistent with the driver's convention `slope = −alpha * g·d`.

## Scope of acceptance

Accepted: immutable replay, common incoming EMA, six-grid comparison, tau-zero projection/KKT check, and CASE B classification. Not accepted or run: any Armijo result, training-cycle result, or tau chosen from either. The FS fallback must remain explicitly labeled a native FS diagnostic if it uses the injectable aggregator path; it is not canonical PCD training and must leave EMA unchanged.
