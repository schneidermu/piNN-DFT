NO-GO — F020 reveals a qualified scientific failure

Scientific classification B: the stream was reconstructed, but the original
stochastic-fidelity gate failed. Native finite-step qualification was therefore
not run. This is negative evidence for the DB-balanced intervention under the
frozen historical stream, not an architecture or physical-operator failure.

## Reconciliation and numerical qualification

Authenticated 251 identities, 2,008 presentations, exactly eight per identity,
one unchanged stream variant per identity; 39 matches and 212 mismatches.
39 existing Jacobian rows were reused after live residual/reference checks;
212 new rows were computed in matched F64. Independent production singleton
loss/gradient checks covered every database: maximum scalar gap 0, maximum relative
gradient gap 2.07127360989732e-11 (limit 1e-10). Each new row is finite,
9446-dimensional and source/checkpoint/parameter-order bound. No finite differences
were used. Source HDF5 access uses the existing SHA-validating training-only API.

JW cycle-gradient L2 residual: 1.00388308536075e-15.
DB cycle-gradient L2 residual: 5.00121572269445e-16.
DB reconstruction from eight database gradients: 4.2595517671347e-16.
Scalar-cycle gaps: JW 0, DB 0. Frozen F64 tolerance: 1e-10.
This identity holds at fixed parameters; it is not conditional unbiasedness
during optimization. Population E and S are kept separate.

## Population comparison

| Quantity | Population E | Population S |
|---|---:|---:|
| JW | 0.156414638138708 | 0.156445295628746 |
| DB-balanced | 0.101040446760582 | 0.1011655472208 |
| c019 | 27.5765197330366 | 27.0926926857078 |

S/E JW gradient cosine: 0.999988271796768.
S/E DB gradient cosine: 0.999631779672309.
JW/DB gradient cosine within S: 0.967954312748824.
Per-database scalar differences and gradient cosines are in the population CSV.

## Frozen stochastic gate

| Estimator | Positive /32 | Median cosine | p10 | Worst |
|---|---:|---:|---:|---:|
| F010 | 32 | 0.870801334 | 0.524762946 | 0.136657850 |
| F019 | 27 | 0.566207473 | -0.124753307 | -0.666769322 |

Negative-dot batch attribution (additive dot decomposition; not a causal ablation):

| Position | F019 cosine | Largest negative DB dot contribution |
|---|---:|---|
| 11 | -0.630513380 | EA13 |
| 73 | -0.666769322 | EA13 |
| 137 | -0.085782924 | PA8 |
| 138 | -0.456304301 | EA13 |
| 157 | -0.129083349 | pTC13 |

F019 passes the positive-dot count (27 ≥24), but fails the lower-tail gate:
p10 deterioration = 0.649516252 >0.10.
All 32 positions are exactly those frozen in F019. All batch gradients are
finite and representable in F32. Full versus stochastic norm/error statistics,
frozen-physical composite alignment and per-database contributions are retained.
Composite diagnostics reuse archived full physical gradients, not newly measured
per-position sampled physical gradients. No physical common descent is claimed.

## Resource discipline

GPU: NVIDIA GeForce RTX 5070 Ti. Cumulative scientific compute: 457.629 s
(including the initial profile and reused-row scalar checks), within the frozen
1800-second budget. The initial conservative projection included cold loading
and independent parity calls. Execution resumed only cached missing rows, with
cumulative budget checks before each row; the budget was not extended.
Runtime and allocator peaks are retained per computed row in the decision receipt.

## Required disposition

1–7: stream VERIFIED; 251 identities; 39 reused +212 computed; numerical checks
PASS; both JW and DB cycle reconstructions PASS.
8–11: population differences and c019 values/cosines reported above.
12: stochastic fidelity FAIL.
13: native AdamW finite-step NOT RUN (dependent gate blocked).
14: predicted database derivatives are saved, but no finite-step benefit is proven.
15: actual protected aggregate decrease NOT TESTED.
16: production physical constraints unchanged; no new physical finite-step safety
qualification is claimed. Relevant architecture regression tests passed.
17: a future 128-update arm is NOT justified by this preflight.
18: next experiment: a bounded read-only attribution of the failed lower-tail
batches using the now-qualified cache, to distinguish importance-weight amplification
from database-gradient cancellation. Do not rescue F019 by changing its sampler,
optimizer or objective, and do not launch training.

30 tests PASS; Ruff (existing E402 bootstrap exception), compileall, diff-check PASS.
Historical F019 artifacts verified byte-for-byte unchanged. No retained optimizer
updates, training, SCF, Clean28, Full30 or Diet100 access/evaluation occurred.
The original checkpoint including optimizer/RNG state remains byte-identical.

Scientific classification: B
Historical stream: VERIFIED
Unique reactions: 251
Presentations: 2008
Reused Jacobian rows: 39
New qualified Jacobian rows: 212
JW cycle-gradient parity: PASS
DB cycle-gradient parity: PASS
c019 evaluation-population: 27.576519733036633
c019 stream-population: 27.092692685707807
Stochastic fidelity: FAIL
Finite-step qualification: NOT RUN
Training authorized for next experiment: NO
Optimizer updates: 0
Training performed: NO
Clean28 accessed: NO
New training data: NO
Production architecture modified: NO
Historical stream modified: NO

Commit/push/remote HEAD receipts are supplied in the final task response.
