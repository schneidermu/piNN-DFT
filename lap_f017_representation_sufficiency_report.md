# F017 — reference-conditioned representation sufficiency

**PARTIAL — F017 completed with documented limitations**

**Scientific classification: E — INCONCLUSIVE.** No architectural training is justified by this result. Production R0 is retained. An unresolved derivative gate is not evidence of redundancy or global insufficiency.

## Evidence categories

Mathematical identities: the complete functional dependence and block operator observation metric are defined in `lap_f017_representation_contract.md`. Symmetric flattening weights off-diagonals by sqrt2 and divides by sqrt(nAO), preserving the qualified Frobenius objective. No cross-molecule AO matrices are subtracted. No gauge changes.

Numerical certificates: energy-only stored-Jacobian SVD projections have explicit thresholds1e-6/1e-8/1e-10; limited-radius energy LS has independent stationarity checks. These are fifteen-system energy-only linear calculations, not full90 certification, finite-step validation or an operator projection. Thresholds below nativeF32 resolution require particular caution.

Empirical associations:90 systems yield270 predetermined nearest-system comparisons of weighted raw/R0/R2 descriptor moments. Pair rules were frozen before reference differences were read. Electron number, density prefactors and signed weights are retained; moment distances are dimension-scaled proxies, NOT complete weighted measures. No parameter-uniform response bound exists here. Certified reference contradictions:0. This does not prove sufficiency.

Unsupported hypotheses: q collisions imply identical energies; energy-only rank proves architecture adequacy; a nonzero new residual VJP proves an independent tangent direction; S5 is an exactXC teacher. None is asserted.

## Provenance and physical embedding

ParentF016: `3c0b9d61d4826e23c185e8e72601f8452774cb12`, verified remote ancestry. Existing F013–F016 hashes/source/checkpoint pins verified. Primary checkpointB8t128. Publication training-only adapter opens chemistry/mRKS indices only. No validation/test metadata, tensors or scores enter this computation.

All15/15 historical systems were evaluated on their full original grids and AO factors, with true reference energies/operators. R0 has9446 coordinates; embeddedR2 has9542, with64 new correlation and32 new tied-exchange weights. OriginalGc metric retained. Old parameters ordered by sorted unique names; exchange counted once.

Maximum physical embedding energy gap: 0Ha. Maximum normalized operator-component gap: 0. Source parameters/buffers were restored exactly with RNG checks. No optimizer constructed or state touched. Existing dispersion correction is added exactly once in native energy arithmetic before diagnostic widening; electronic reference equals storedExc minus that frozen correction.

## Failed operator tangent qualification — preserved, not relaxed

First system chosen by ascending grid size: H2,42508points,60AOs. One deterministic unit direction spanning all96 new coordinates was tested with the exact variational operator and native parameter boundary.

FD step=0.0001; observed relativeJVPgap=0.01677349; absolute gap=1.997472e-05. Frozen gate is absolute1e-6 plus relative0.005 times the analyticJVPnorm. **FAIL**. Possible F32 finite-difference noise is unproven; no tolerance relaxation or substitute energy derivative was used.

Measured full-system JVP=5.011s. A serial15×96 sweep projects to120.25minutes, exceeding the frozen30-minute diagnostic budget. This is a measured serial projection, not proof that a future validated batched/matrix-free implementation cannot be faster. No such implementation was developed to rescue this audit.

The full90 operator-Jacobian expansion, operator-space old/new projection and nonzero-weight safety/finite tests were stopped. Full90 reference metadata were used only for the predetermined pair analysis; full90 operators were not evaluated.

## Completed limited energy and operator evidence

Energy Jacobian shape=(15, 9542); old9446 plus new96. Tables provide rank, conditioning, projected new residual alignment and separate R0/R2 predictions at0.25r/r, r=0.0006498670299681904. These predictions were not implemented as parameter steps. Equal Euclidean radii in different parameterizations do not establish invariant superiority. No joint physical scale was invented.

At all three frozen thresholds, the old energy Jacobian has rank15/15 and additional projected rank0. This is the anticipated energy-only identifiability limitation, NOT proof of useful operator-space redundancy. At full radius the predicted squared-energy-residual reduction is74.2433307% for R0 versus74.2445981% for R2. The increment is about0.00127 percentage points. Finite actual reductions are unmeasured.

For all15 systems the actual operator residual was computed in its own overlap-orthonormalized block. The new-coordinate quantity J_N^T r_V was obtained exactly as −0.5 times the gradient of squared operator residual. It is an unprojected VJP. **It is not J_perp^T r and does not establish96 useful independent directions.** Full operator Jacobian columns produced:0; one unqualified JVP only. Residual vectors/VJPs are stored outsideGit and hash-bound.

## Answers to the scientific questions

1–3: No certified pair of globally indistinguishable systems with incompatible references was found. Zero complete raw-input hash matches among the selected different-system pairs; finite summary proximity is not full-representation equivalence. Individual tanh(q) collisions remain distinct from this claim.
4: Fifteen-system energy-only local linear corrections are quantified in local_attainability.csv; full90 and finite reductions are NOT measured.
5: R0 attainable operator-error correction is unknown: operator-space projection is blocked.
6–7: Energy-only extra rank/alignment is reported by threshold. Qualified independent operator directions and orthogonal operator-residual alignment are unknown.
8: Identical-radius energy-only predictions are reported. A useful joint/physical improvement is not established.
9–10: Nonzero finite-step and safety validation were not performed after the failed derivative prerequisite.
11–13: Additional chemistry sensitivities coverage0/251; all eight-database R2 benefit/trade-offs and physical no-worsening are unknown. F015 old R0 reaction caches and local correction evidence were reverified, not replaced by selected favourable reactions or S5 targets.
14–15: This audit does not justify R2 training and cannot choose representation versus optimization as a cause. F013/F015 optimization evidence remains historical evidence; it does not resolve the missing F017 operator comparison.
16: Minnesota exact molecularXC labels are absent. No pointwise exactXC labels are inferred. Finite systems and lack of parameter-uniform bounds block global representation claims.
17: One bounded training-only operator-JVP precision verification on H2 at the same zero-weight embedding: independently bound F32 numerical-integration FD noise before any tangent projection or architectural training. Retain production R0 meanwhile.

## Missing outputs and integrity

The protocol retains all intended artifact names. Finite-step validation, nonzero-weight safety and chemistry-transfer CSVs are intentionally absent, with exact reasons in decision.json; no empty fabricated results. Relevant architecture/physics/operator/restoration tests, Ruff, compileall, hashes and diff checks are recorded in provenance after final verification.

Preflight accounting: the new diagnostic symmetric flattening initially constructed sqrt2 in F32 before widening. A strict Frobenius-inner-product test caught this; it was corrected to construct sqrt2 directly in matrixF64 dtype, and the affected15-system observations were recomputed. The frozen pair summary's total positive electron mass was also explicitly included before final pair results. Neither correction changes the frozen scientific rules or production code. Preliminary receipts are archived as rejected and are not interpreted. Corrected full-system JVP still fails the same frozen gate.39 relevant tests passed; Ruff, compileall and diff checks passed.

Optimizer updates:0. Training:NO. Production architecture modified:NO. Clean28 accessed:NO. SCF:NO. No new systems, target calculations, tauchannels, trainedweights, or exhaustive augmentation evaluation. No follow-up experiment launched.
