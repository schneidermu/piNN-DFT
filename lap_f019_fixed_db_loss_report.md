PARTIAL — F019 stopped with documented limitations

The mandatory frozen-gradient full-cycle identity cannot be authenticated using
the specified full251 objective and historical training stream. The archived
full251 Jacobians use one fixed evaluation variant per identity. In the frozen
B8/F010 stream, 212 of 251 identities use a different variant: 1,696 of 2,008
presentations do not match the cached `(identity, variant)`.

Repeating every identity eight times proves identity coverage, but does not prove
gradient equality when its quadrature variant differs. Substituting the evaluation
variant or matching only identity would silently change the experiment. No such
substitution was made. This is a protocol/reconstruction limitation, not evidence
that the database-balanced objective performs poorly. Stage B fidelity, Stage C
native finite-step qualification, and training were not performed.

## Verified read-only results

The parent checkpoint, inherited moment digest, original manifest and archived
Jacobian hashes match the pinned receipts. The authoritative A_d hash is
`3233a7b3194e68af447bb63edda0d154325da7d051e0df6fe60a7a788cebc73d`.
The shortened/transcribed value in the request was not used; the explicitly
designated canonical F010/F013 receipt was used instead.

| State | JW | DB-balanced | Chemistry gradient cosine | Composite cosine |
|---|---:|---:|---:|---:|
| B8 t128 | 0.156414638139 | 0.101040446761 | 0.963757 | 0.997015 |
| F010 t128 | 0.142369869056 | 0.098599062671 | −0.211912 | −0.329275 |
| F010 t251 | 0.135822095425 | 0.098132204413 | −0.196965 | 0.469444 |
| F012 t128 | 0.142494767799 | 0.097742234648 | −0.281069 | 0.570707 |

These are the original fixed evaluation-variant population, not reconstructed
historical training batches. JW and DB-gradient reconstruction passed the frozen
absolute/relative tolerances of 1e-10. At B8, norms are 1.5115724830787063 and
0.5467574673657991. The resulting evaluation-population calibration is
**c019 = 27.576519733036633**; it is not qualified for launching training.

The negative unit composite-direction derivatives at B8 show smaller predicted
regressions in ABDE4, PA8 and pTC13, improved predicted EA13 descent, but slightly
worse IP13/MGAE109 progress and less descent in DBH76/NCCE31. ABDE4, PA8 and pTC13
still predict increases. The new composite is close to the old direction at B8;
this is not a common-descent certificate. Complete eight-database values at all
four states are in `lap_f019_database_gradient_geometry.csv`.

## Required questions and disposition

1. Full DB-balanced gradient reconstructed on the evaluation population: yes;
   historical-stream full-cycle reconstruction: blocked by variant mismatch.
2. c019: 27.576519733036633, evaluation-population calibration only.
3. B8 chemistry-gradient cosine with F010: 0.9637573454292919.
4. All four protected databases improve predicted progress relative to F010,
   but three still predict regression; no trajectory claim is possible.
5. IP13, MGAE109, DBH76 and NCCE31 lose predicted progress at B8.
6. Complete physical conflict audit: incomplete; composite cosine is reported.
7. Historical minibatch estimator adequacy: not qualified.
8. Native AdamW finite-step preflight: not run.
9–10. Training launched: no; optimizer updates: zero.
11–14. Final DB loss, macro MAE, protected-database and JW trajectory changes:
   unavailable; no new model exists.
15. Production architecture, constraints and precision were unchanged.
16. Physical scientific ratios: not run.
17–18. Clean28 not accessed/evaluated; historical best not tested.
19. Objective-mismatch hypothesis: unresolved.
20. Next step: reconcile the exact fixed-variant full251 population with the
   existing historical stream in a separately frozen protocol. Preserve the
   stream; explicitly distinguish evaluation-population gradients from
   stream-population gradients before attempting any exact-cycle gate. Do not
   launch an alternative sampler, optimizer, architecture or training arm.

## Validation and scope

25 focused/regression tests passed, including F010, F013, F016 and F018 tests.
Ruff, compileall and diff checks are recorded in the final receipt. No optimizer
was instantiated by F019, no gradients were newly backpropagated, no model state
was changed, and no chemistry reaction was reevaluated. Cached F64 Jacobians were
used only on their exact stored identity/variant population. No exhaustive
eight-variant evaluation, SCF, full90 evaluation, training or development/test
evaluation occurred. Historical artifacts remain untouched.

Scientific classification: PARTIAL
Architecture: R0
Training objective: fixed DB-balanced scalar (proposed, not trained)
Global chemistry scale: 27.576519733036633 (not training-qualified)
Preflight: BLOCKED
Training launched: NO
Optimizer updates: 0
Chemistry presentations: 0
Final DB-balanced objective: unavailable
Final JW: unavailable
Final Minnesota MAE: unavailable
Final macro database MAE: unavailable
AE17 physical gate: NOT RUN
Exc physical gate: NOT RUN
Operator physical gate: NOT RUN
Clean28: NOT EVALUATED
Historical best beaten: NOT TESTED
Architecture modified: NO
New training data: NO
Curriculum: NO
Adaptive task weighting: NO
Gradient clipping: NO

Commit and remote receipt are provided in the task response; embedding this
commit's own SHA in its contents would be self-referential.
