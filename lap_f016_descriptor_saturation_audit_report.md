# F016 — bounded q resolution and constraint audit

**B — RESOLUTION IMPROVES, CHEMICAL RELEVANCE UNRESOLVED**

Zero training/optimizer updates. R0 production networks, Gc activation, AO implementation, dataset and checkpoints unchanged. Training-only F015 input view; no validation/test tensors or scores used.

## Provenance and physical contract

Parent F015 commit 8e2fe20b1874fbaded101ea8cbb2a949b3013a91; remote ancestry verified after fetch. All F015 source/checkpoint/cache pins reverified. One fixed variant per 251 identities;495 unique grids and23,231,999 stored points. No regeneration. Raw F32 sources widened to F64 do not regain source precision.

See lap_f016_anchor_contract.md and feature_layouts.json. R1 keeps 9446 parameters. R2 adds 96→9542. Tied exchange consumes(s,q,q4); both spin q4 features append to correlation. Gc metric deliberately remains original 8 coordinates, otherwise zero new weights DO NOT reproduce R0. This compatibility choice is part of the candidate, not a production change.

## Mathematical versus numerical information

All finite-real tanh transforms are injective. R1/R2 add no independent physical variable or tau information. Exact endpoints and finite-precision collisions are distinct. Collision tables count distinct raw-q values per same grid/path/spin output bin, not all-pairs multiplicity. Raw separation ≥1 is a declared dimensionless diagnostic, not a universal chemical threshold. Near-collision table uses 0/1/4 F32 ULPs on matched environment pairs.

R0 slope at zero 1; R1 .25; R2 joint √(1+1/16). R1 reduces local resolution near zero and increases it in moderate/large q regions. Stable sech² curves distinguish mathematical derivatives from rounded 1−tanh² and derivative zeroing. Full first/second derivative tables and both F32/F64collision populations are archived.

## Matched environments

F015 stride101 eligible linspace samples reused exactly, 50,000 per density floor. Original six non-q coordinates unchanged; R2 two q scales each weighted 1/√2. Distances are representation diagnostics, not interchangeable physical distances; comparisons retain exactly the same R0 nearest pair.

| Representation | rho≥1e-3 near pairs Δq≥1 | Resolved pairs | Electron-weighted resolved fraction |
|---|---:|---:|---:|
|R0|2353|0|0.000000|
|R1|2353|2201|0.974043|
|R2|2353|2146|0.961519|

Conditional tau ambiguity remains. Both representations are deterministic functions of q; nearest-neighbour changes cannot establish tau recovery. See all floors/distancebounds in tau_residual_ambiguity.csv.

## Actual trained R0 sensitivity and chemistry

All four frozen checkpoints evaluated at the same F015 physical samples. Learned beta/gamma/mu/kappa/GNN/Gc and separate exchange/correlation energy-density raw-q derivatives use exact autograd; tau derivatives exactly zero. Channels are outputs, not parameter blocks. R1 output shifts are not accuracy results; R2 zero-weight output is R0.

Complete eight-database signed/gross chemistry accounting reusesF015 integrals. Important species listed in chemical_species_contributors.csv. q99 fraction of gross B8–S5difference is modest, yet signed reaction terms cancel strongly; neither is an exact XC error. The comparison does not prove that numerical resolution causes S5 advantage. Database electron weights are statistical proxies, not substitutes for signed quadrature.

## Compatibility and variational checks

16 frozen model/representation/dtype combinations checked. R2 max F64 forward gap=0; F32gap=0. All anchor/spin symmetry/tau checks passed. F32 not claimed bitwise. Predeclared atol/rtol F64 forward 1e-12, input/operator1e-10;F32 forward/input 2e-6.
Existing weak-form operator dispatch reused including repaired F64 learned/F32 PBE path. 16 synthetic AO density-matrix directional FD checks passed; max gap=8.42e-06. This is variational formula/parity evidence, not full-corpus operator qualification. Synthetic AO test inputs are not new molecular training systems.

Near-zero-density stress at rho=0,1e-12,1e-6,1e-3,1 and q±1e12 isfinite. Very large physical Laplacian derivatives occur under regularized density denominators (up to~1e16 first/~1e35 second for learned constants). Bounded descriptors alone do not guarantee harmless potentials. R2 zero-new-weight inherits R0; nonzero new-weight tail stability remains unqualified. No trainedcandidate is claimed.

## Decision and limits

Numerical resolution/anchors/zero-weight embedding pass. Signed chemistry associations exist, but no controlled evidence ties resolved pairs to reference errors; new-weight tail stability is not qualified.

One bounded R0 versus zero-new-weight R2 matched training experiment, with original anchor metric, same initialization/data/optimizer and predeclared tail/operator safety checks. Do not run automatically.

No Clean28/WTMAD benefit inferred. No SCF, new data, fitted weights, optimizer, tau channels or exhaustive variant sweep. Existing F015 geometry/DMprovenance limitation retained. No independent molecular exact XC targets available. Zero-weight embedding does not prove safety after training. Constraint preservation holds algebraically for the declared candidate metric and numerically on tested inputs.

## Integrity and tests

32 focused model/operator/F016 tests passed. Ruff (E402 excepted only for script import bootstrap), compileall and git diff --check passed. An initial test run encountered a sandbox temporary-directory permission error; rerunning with an explicit workspace temporary directory passed. No scientific test was skipped or weakened.
Frozen protocol pins, external receipt hashes and source/checkpoint hashes are recorded in provenance. Large arrays remain outside Git.
