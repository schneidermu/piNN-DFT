# Frozen operator globalization diagnostic
Compare the exact historical step4 requested displacement under production F32, matched-F64 continuous parameters, and matched-F64 on rounded F32 candidates. Reuse existing operator factory, immutable15 sources, point chunk256 and AO chunk4096. No solver, training, source changes, targets regenerated, or accepted updates.

Grid: zero and 2^-j for j=0..15. All diagnostic reductions are F64. Numerical floor: 64*eps64*max(1,abs(base loss)); meaningful relative derivatives require predicted change above this floor. F64 derivative consistency requires relative error below 1e-3 at a meaningful small step, supported by convergence across grid. Zero norm threshold1e-30. Gradient comparison reports actual discrepancies; no sign-only consistency claim.

Acceptance: pinned hashes; t1 candidate equals theta5 bitwise; realized delta equals recorded delta; genuine F64 leaves and exact widening of F32-loaded sources; full three-way curves; derivative and retrospective Armijo tables; restored model/RNG; one independent review. Fit quadratic only after derivative gate, using 1/32,1/64,1/128,1/256 above floor.
