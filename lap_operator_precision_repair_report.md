# Operator precision-boundary audit

The factorial precision audit shows that assembly-only repair is insufficient; the single missing precision boundary is local model/PBE-partial evaluation upstream of AO contractions.

## Identity and numerical contract

Branch `lap_full_vxc`, exact source/start commit `df0b95a5aaed7ebce3bd84c7f46443240263c6f3`. Frozen theta4/theta5 and requested/realized historical step4 delta; immutable15 panel; point chunk256, AO chunk4096. All paths use the same weak-form equations, reference, overlap and /nAO loss. PCD/O1/tau/chemistry/E/Armijo/sampling are unchanged. No production source edits, optimizer, accepted updates, training, SCF, Diet, Slurm or full90 generation.

Matched-F64 is arithmetic on exact F32-loaded data/checkpoint/cache values widened; no native source precision is reconstructed. Original source/data hashes and all external arrays/tooling are bound in protocol/results. Existing reviewed df0b95a legacy/gold base gradients and both t1 responses are reused, not rerun; source, parameters, data and chunks remain hash-identical.

## Five-variant factorial

| Variant | Base loss | grad cos gold | grad relL2 | requested dot | continuous t1 ΔL | rounded-F32 t1 ΔL | response error | passes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P32_A32 | 0.0367989332571 | 0.999999978345 | 0.000239015455018 | -3.74320223583e-07 | not run | 2.47593574032e-06 | 2.8495751339e-06 | False |
| P32_A32_GEMM_A64ACC | 0.0367981301536 | 0.999999995653 | 0.000121287224242 | -3.74363125651e-07 | not run | -2.58650350959e-07 | 1.14989042616e-07 | False |
| P32_A64 | 0.0367976704658 | 1 | 1.27077742548e-06 | -3.74409739479e-07 | not run | -4.27207604587e-07 | 5.35682110112e-08 | False |
| P64_A32 | 0.0367962593235 | 0.999999960076 | 0.000373889856687 | -3.74839617459e-07 | 3.45627277762e-06 | 3.56041432954e-06 | 3.93405372312e-06 | False |
| P64_A64 | 0.0367976806843 | 1 | 0 | -3.74410167649e-07 | -3.7432235174e-07 | -3.73639393576e-07 | 0 | reference |

Continuous parameter candidates are only meaningful for F64 parameter leaves. P32 continuous entries are not applicable; their exact rounded candidate is the primary comparison. Raw per-system losses/gradients and realized-direction dots are retained. Aggregate responses use the F64 mean of per-system deltas; this differs from subtracting aggregate means by last-bit roundoff only.

Fixed response tolerance=`1.86819696788e-08` (max1e-10,5% of gold). P32_A64 error=`5.35682110112e-08` =`14.3368745192%`; accumulation-only error=`1.14989042616e-07` =`30.7754066069%`. All gradients meet the aggregate cosine threshold and directional descent sign; these do not waive response accuracy. All variants are finite. No non-gold boundary passes all phase1 criteria.

## Causal interpretation and STOP

F64 weak-form assembly substantially reduces the legacy error and restores descent in all15 systems. It does not reproduce the gold response within5%. F64 local partials with F32 assembly still give aggregate ascent, so upgrading local arithmetic alone cannot repair F32 assembly. Accumulation-only gives aggregate descent but is30.8% wrong and has individual sign failures. Both the contraction/assembly region and the upstream local-partial region contribute; this is not evidence for an isolated accumulation-only repair.

The P32_A64 gradient is extremely close to gold, while its finite response remains14.3% inaccurate. Static scalar agreement or gradient cosine cannot certify tiny parameter-induced changes. This audit localizes the residual upstream of F64 AO assembly, but does not separate adaptive NN output precision from descriptor/PBE derivative arithmetic. A full matched-F64 path is faithful; its being faithful does not prove it is the minimum production design.

Phase1 fails. Predopt qualification, performance selection, production patch, metadata migration, repaired dyadic regression and five-update pilot are blocked. No thresholds are relaxed and no additional mixed-precision variants are added.

## All15 coverage available at theta4

These are phase1 per-system diagnostics, not successful Phase2 qualification. Predopt was not evaluated because no candidate survived phase1.

| Metric / path | Predopt | theta4 P32_A64 | theta4 accumulation-only |
| --- | --- | --- | --- |
| min_gradient_cosine | not run | 0.999999999999 | 0.999999713015 |
| p10_gradient_cosine | not run | 0.999999999999 | 0.999999910955 |
| median_gradient_cosine | not run | 0.999999999999 | 0.999999993068 |
| max_scalar_abs_error | not run | 2.4486313123e-08 | 5.0061629522e-06 |
| median_response_abs_error | not run | 4.89987426674e-08 | 1.12904926702e-06 |
| max_response_abs_error | not run | 9.34081357401e-08 | 5.75216011834e-06 |
| meaningful_sign_disagreements | not run | 0 | 6 |

Per-system values for all five paths are in JSON. Every gold response is meaningful under64 eps64 max(1,|base loss|). Individual response sign agreement does not rescue the failed aggregate5% gate.

## Performance, historical repair and pilot

No candidate qualifies, so no selection benchmark is run. Phase1 driver timings are diagnostic execution receipts, not warmed median performance measurements. No RTX-to-V100 performance or memory claim. Cache remains storedF32; candidate promotions are external and per assembly call. The full F64 one-system shadow is a diagnostic gold/control, not a production cache or proposed master-parameter system.

| Path | Scalar time | Scalar+grad time | peak allocated | peak reserved |
| --- | --- | --- | --- | --- |
| legacy P32_A32 | not benchmarked | not benchmarked | not benchmarked | not benchmarked |
| repaired | no qualified repair | no qualified repair | no qualified repair | no qualified repair |

Historical repaired six-point scan: not run because no production boundary qualifies. Conditional pilot checkpoints0/2/5: not run. No old numerical baseline is reinterpreted or overwritten.

## Validation and independent review

External controls reuse the actual production function expression tree. Accumulation-only transforms each completed MatMul result to F64 and makes the accumulator F64, preserving F32 operands/products. The F64-assembly wrapper widens all seven inputs and calls the original helper. P64_A32 narrows all seven inputs before that same helper. Patch contexts restore the original helper.

Synthetic checks pass: differentiable F32 partials through F64 assembly to original F32 leaf; finite/nonzero gradient; manufactured linear-gradient parity; accumulation-control exact equality to the original when inputs are alreadyF64; finite accumulation-control autograd; no input tensor mutation. Existing Windows operator unit suite:6 passed in1.52s. Executed tooling py_compile and immutable source/input/tensor hashes pass. Scratch model and RNG restored; no persisted scientific state changes. Full production suites/DDP are not rerun because no production repair is permitted and source is unchanged.

One independent Luna MAX review: PASS. Review SHA 1ac32e394cbbe8d01fa70abf5cab4b8aa1346fa8e2a78a82f1445a2089605223 External factorial execution `230.6178301` seconds. All14 previous audit conclusions remain historical evidence; this task does not overwrite them.

## Required answers

1. F32 AO contraction/assembly materially contributes, and F32 local-partial evaluation leaves a further unresolved response error.
2. No. Accumulation-only fails the fixed response gate.
3. Higher-precision weak-form contractions are required among tested paths; F64 local partials with F32 assembly still fails.
4. Higher precision upstream of assembly is also needed to meet this gate. Full-F64 NN necessity is not yet established separately from local functional/descriptor differentiation.
5. No minimum production repair is qualified. Do not deploy assembly-only based on corrected sign.
6. P32_A64 frozen response `-4.27207604587e-07` versus gold `-3.73639393576e-07`; error14.3%, exceeds5%.
7. P32_A64 restores theta4 response signs across15 and gradients agree closely, but aggregate accuracy fails. Predopt qualification is therefore not run.
8. Yes for the external P32_A64 control: real F32 leaves receive finite autograd derivatives through connected F64 casts. This does not make its forward response accurate enough.
9. No qualified repair; runtime overhead was not benchmarked.
10. No qualified repair; production memory overhead was not benchmarked. No persistent F64 production cache is created.
11. Production numerical metadata remains unchanged because no repair is authorized. Prospective five-variant semantics and STOP gates are explicit in this diagnostic protocol.
12. Model initialization behavior is unchanged.
13. No new operator arithmetic is deployed; current resume behavior is unchanged. A future repair must fail closed on numerical-protocol mismatch.
14. No pilot ran; relative chemistry was not retrained.
15. No pilot ran; AE17 was untouched.
16. No pilot ran; E_xc was untouched.
17. No repaired pilot or operator-global-guard claim.
18. No. PCD/O1/tau and all other scientific settings remain unchanged.
19. No new repaired four-task path is qualified for longer training.
20. One frozen local-partial provenance trace, distinguishing adaptive NN outputs from descriptor/PBE differentiation and gradient-chain arithmetic before c/a/b are supplied to F64 assembly.

The factorial precision audit shows that assembly-only repair is insufficient; the single missing precision boundary is local model/PBE-partial evaluation upstream of AO contractions.
