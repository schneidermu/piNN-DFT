# Exact single-state q parameter-Jacobian qualification

**PASS-LINEAR: seed11 P536 and seed11 P67.** Every frozen eta passes the unchanged <=5% relative-L2 and >=0.995 cosine gates for the exact leading singular direction and deterministic random control. The eta0 ensemble failure is not reproduced on either tested single state.

## Frozen scope and provenance

Starting commit `11b1f6716e24186f371ba65fa6095c21fea5c666`, branch `lap_full_vxc`; final commit is the Git commit containing this report.
Primary P536 was completed and qualified before P67 construction/evaluation. Only these two states were loaded for parameter diagnostics. No PBE_TANGENT, PBE_HEAD, other seeds, ensemble map or scientific objective gradients were evaluated. Historical reports remain unchanged.
Probe SHA256 `bcc3d01872d0181b197800e2a0295178224fe1d81e9ebfb2b8b8e2f64437b81a`; same64points, alpha64 then beta64. h2 `[0.023218985940978362, 0.042529125693106434]` unchanged.
Same existing epsilon_xc symmetric q slope, actual q denominator at fixed rho and other inputs, F64 diagnostic arithmetic on exact widened F32 checkpoint/source values. No source precision reconstruction or functional change. Both128-row replays match historical cached slopes bitwise. The cache file matches its historical SHA-bound manifest.
34 unique trainable tensors / 9,446 sorted coordinates. Frozen constants and buffers excluded. Flat-offset mapping round-trips bitwise. Exchange is the same reused module at each spin call, represented once; no separately registered duplicate parameter aliases were found. Existing production trainable-name helper is reused.
The new tooling reuses the previous loader/functional evaluator instantiated with exactly one state. No shared multi-state parameter displacement or randomized sketch is used.

| State | stored-file SHA256 | model-state SHA256 |
| --- | --- | --- |
| P536 | 0ca0f77367214c171e6edf0576bdb043d64479245ad1a0e539fa11fb0cee542d | 3c2e45d6a86cccf80d7b4dc458e67f9b406935ed60d4e57433bb441a6df88da6 |
| P67 | 264a8d189d53d49007d7de11d8cfb5382e8b8262e4b48068142c8771e293c834 | ee815caf918e8ded13e99896070a95a32293100499746fd0623444e2c0023523 |

Eta ladder was frozen before evaluation: 0.0002118715157188727, 5.2967878929718174e-05, 1.3241969732429543e-05, 3.310492433107386e-06, 8.276231082768465e-07. Directions: normalized exact-SVD v1, and Gaussian unit vector generated once with CPU seed42. Identity checks use seeds42,314159,271828; no directions selected by FD performance.

## Exact internal consistency

| State | shape | finite rows | max JVP rel L2 | max VJP rel L2 | replay / mapping | restoration |
| --- | --- | --- | --- | --- | --- | --- |
| P536 | [128, 9446] | PASS | 1.90234501744e-14 | 4.95286699814e-14 | bitwise PASS | 0.0 |
| P67 | [128, 9446] | PASS | 1.90168765331e-14 | 3.45052069773e-14 | bitwise PASS | 0.0 |

J is materialized from128 exact scalar VJPs. Dense Jv agrees with direct torch.func JVP, and J^T w agrees with VJP for three frozen random tests per state under the predeclared1e-8 relative gate. All stored parameters/buffers are hash-identical after each temporary functional perturbation; no in-place mutation or leaf-gradient accumulation occurs.

## Exact SVD — instrument characterization only

| State | sigma1 | stable rank | entropy effective rank | r99 | ranks at .01/.001/.0001/1e-6/1e-8 |
| --- | --- | --- | --- | --- | --- |
| P536 | 47.9478522062 | 1.02695807014 | 1.15944534876 | 2 | [6, 13, 24, 60, 79] |
| P67 | 53.9733594644 | 1.02627013413 | 1.15825752195 | 3 | [6, 13, 28, 61, 80] |

All128 singular values, normalized spectrum and cumulative energy are in JSON and external exact NPZs. SVD chooses a sensitive deterministic FD direction; these numbers are not interpreted as physical flexibility, useful capacity or a P67/P536 ranking.

## Central parameter finite differences

### P536

Stored parameter norm 21.1821139099. All directions have norm1; actual parameter values are baseF64 + eta*v, without F32 re-rounding. No eta or direction search occurred.
Direction v1.
| eta | relative L2 | cosine | absolute L2 | ||Jv|| | ||FD|| | eta/||theta|| | max coordinate delta | restore |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0002118715157188727 | 1.98438579801e-07 | 1 | 9.51470369631e-06 | 47.9478522062 | 47.9478431562 | 1.00023782622e-05 | 9.88467358299e-05 | 0.0 |
| 5.2967878929718174e-05 | 1.24034564662e-08 | 1 | 5.94719097488e-07 | 47.9478522062 | 47.9478516405 | 2.50059456554e-06 | 2.47116839575e-05 | 0.0 |
| 1.3241969732429543e-05 | 8.03896835095e-10 | 1 | 3.85451266382e-08 | 47.9478522062 | 47.9478521694 | 6.25148641385e-07 | 6.17792098937e-06 | 0.0 |
| 3.310492433107386e-06 | 9.40300465074e-11 | 1 | 4.50853877288e-09 | 47.9478522062 | 47.9478522054 | 1.56287160346e-07 | 1.54448024734e-06 | 0.0 |
| 8.276231082768465e-07 | 4.43028342861e-10 | 1 | 2.12422575067e-08 | 47.9478522062 | 47.9478522131 | 3.90717900866e-08 | 3.86120061836e-07 | 0.0 |

Direction random.
| eta | relative L2 | cosine | absolute L2 | ||Jv|| | ||FD|| | eta/||theta|| | max coordinate delta | restore |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0002118715157188727 | 7.25515695916e-10 | 1 | 1.25568221109e-10 | 0.173074437695 | 0.173074437783 | 1.00023782622e-05 | 7.88607700243e-06 | 0.0 |
| 5.2967878929718174e-05 | 1.54355051135e-09 | 1 | 2.67149136805e-10 | 0.173074437695 | 0.173074437683 | 2.50059456554e-06 | 1.97151925061e-06 | 0.0 |
| 1.3241969732429543e-05 | 9.74916004853e-09 | 1 | 1.6873303934e-09 | 0.173074437695 | 0.173074438345 | 6.25148641385e-07 | 4.92879812652e-07 | 0.0 |
| 3.310492433107386e-06 | 2.75080367365e-08 | 1 | 4.76093799025e-09 | 0.173074437695 | 0.173074439234 | 1.56287160346e-07 | 1.23219953163e-07 | 0.0 |
| 8.276231082768465e-07 | 1.70554039808e-07 | 1 | 2.95185445363e-08 | 0.173074437695 | 0.173074453368 | 3.90717900866e-08 | 3.08049882908e-08 | 0.0 |

Leading-direction successive larger/smaller error ratios: [15.998651693717447, 15.42916444586892, 8.549361240955164, 0.21224386209717877].
Parameter qualification: PASS-LINEAR; both directions pass all five etas. No module/LayerNorm/coordinate localization was triggered.
### P67

Stored parameter norm 21.188532106. All directions have norm1; actual parameter values are baseF64 + eta*v, without F32 re-rounding. No eta or direction search occurred.
Direction v1.
| eta | relative L2 | cosine | absolute L2 | ||Jv|| | ||FD|| | eta/||theta|| | max coordinate delta | restore |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0002118715157188727 | 2.04683838891e-07 | 1 | 1.1047474413e-05 | 53.9733594644 | 53.9733490558 | 9.99934845222e-06 | 0.000101198434118 | 0.0 |
| 5.2967878929718174e-05 | 1.27933418204e-08 | 1 | 6.90499636825e-07 | 53.9733594644 | 53.9733588138 | 2.49983711305e-06 | 2.52996085296e-05 | 0.0 |
| 1.3241969732429543e-05 | 7.95083579456e-10 | 1 | 4.29133318382e-08 | 53.9733594644 | 53.973359424 | 6.24959278263e-07 | 6.32490213239e-06 | 0.0 |
| 3.310492433107386e-06 | 2.17804438583e-10 | 1 | 1.17556372565e-08 | 53.9733594644 | 53.9733594561 | 1.56239819566e-07 | 1.5812255331e-06 | 0.0 |
| 8.276231082768465e-07 | 4.06468267763e-10 | 1 | 2.19384579268e-08 | 53.9733594644 | 53.9733594707 | 3.90599548915e-08 | 3.95306383274e-07 | 0.0 |

Direction random.
| eta | relative L2 | cosine | absolute L2 | ||Jv|| | ||FD|| | eta/||theta|| | max coordinate delta | restore |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0002118715157188727 | 8.74643780551e-10 | 1 | 2.12368406686e-10 | 0.242805598586 | 0.242805598659 | 9.99934845222e-06 | 7.88607700243e-06 | 0.0 |
| 5.2967878929718174e-05 | 1.00759445767e-09 | 1 | 2.44649575427e-10 | 0.242805598586 | 0.242805598551 | 2.49983711305e-06 | 1.97151925061e-06 | 0.0 |
| 1.3241969732429543e-05 | 1.06085186346e-08 | 1 | 2.57580771718e-09 | 0.242805598586 | 0.242805600347 | 6.24959278263e-07 | 4.92879812652e-07 | 0.0 |
| 3.310492433107386e-06 | 3.0732867218e-08 | 1 | 7.46211222113e-09 | 0.242805598586 | 0.242805598435 | 1.56239819566e-07 | 1.23219953163e-07 | 0.0 |
| 8.276231082768465e-07 | 8.03894930957e-08 | 1 | 1.95190189911e-08 | 0.242805598586 | 0.242805594822 | 3.90599548915e-08 | 3.08049882908e-08 | 0.0 |

Leading-direction successive larger/smaller error ratios: [15.99924724621836, 16.090562238995684, 3.650447092034194, 0.5358461062195717].
Parameter qualification: PASS-LINEAR; both directions pass all five etas. No module/LayerNorm/coordinate localization was triggered.
Leading-direction errors initially decrease close to16x for a4x reduction in eta, consistent with central O(eta^2) truncation. At the smallest steps errors reach roughly1e-10 and mildly rebound. Random-control errors grow from roughly1e-9 toward1e-7 as eta shrinks; they remain orders of magnitude inside both acceptance gates. This is consistent with a roundoff-sensitive weak-direction finite difference, but deterministic repeatability alone does not prove cancellation as a cause. Exact repeated evaluations differ by0 and every contrast numerator remains nonzero. No artificial denominator epsilon is used.
The frozen helper labels the tiny random-direction rebound NUMERICAL-CANCELLATION using a conservative2x/1e-8 flag. This is retained in raw metrics as a candidate diagnostic only. It does not satisfy the stricter improve-then-worsen definition in the user protocol and is not promoted to the scientific classification. Both states qualify PASS-LINEAR on the actual all-eta error/cosine evidence.
Independent review found a nonblocking general-helper edge case: the rebound branch can qualify before enforcing smaller-eta gate consistency. Every row in this experiment passes, so that branch cannot invalidate this result. The frozen source is retained as executed; the helper must be repaired before reuse as a general future gate. No future execution is authorized by this result.

## Interpretation and stop

The exact single-state autograd parameter Jacobian predicts the real central finite parameter response on seed11 P536 and P67, including the original eta0. No path inconsistency is detected on these states. It would be incorrect to explain the previous99.9507% ensemble discrepancy solely as eta0 exceeding these two states' local regime: eta0 passes here. That earlier failure involved a different ensemble direction and other states; its source remains unlocalized. No other states are evaluated to rescue or explain it in this task.
No module failure was observed, so no LayerNorm gamma/beta or one-coordinate checks were run. LayerNorm dominance from the previous ensemble remains uninterpreted. No inference about capacity retention/loss, objective alignment, optimizer performance or initialization preference is made.
ONE proposed next experiment, not launched: a separately frozen single-state P67/P536 q-controllability comparison using the now-qualified exact Jacobian bridge, without ensemble coordinates or new scientific-objective evaluations. Its scientific scope and metrics must be agreed before execution.

## Validation and safety

Four new synthetic instrument tests and15 related diagnostic/helper tests PASS. Ruff, py_compile and git diff --check PASS.34 immutable artifact/source hashes match before/after; zero mismatches; exact restoration difference0; state mutationNONE. Production-source changes0. Dense J/SVD arrays are external and SHA-bound; no model checkpoint was saved or changed.
Independent review: PASS. The review must check the identical AD/FD observable, canonical parameter mapping, tied weights, finite perturbations, convergence and restrained interpretation. Final review evidence is in machine-readable results.
No training trajectory, production optimizer change, old cursor10 advancement, unfinished initialization-matrix resumption, expensive scientific-objective gradient recomputation, full90, SCF, Diet, Slurm or100-update run occurred.
