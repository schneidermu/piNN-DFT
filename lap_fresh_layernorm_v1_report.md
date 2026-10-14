# Fresh R0 LayerNorm startup and AdamW memory audit

The startup spike is localized to the exchange UEG constraint path, not the ordinary grid calls. The dominant initial derivative is supported by independent H2 finite differences. First-gradient memory is locally adverse in update 2; its residual contribution in proposal 33 is much smaller and the total proposal predicts descent. No production bug is established. Full derivative status is **UNRESOLVED** because the secondary theta1 LayerNorm coordinate lacks two adjacent qualifying finite-difference steps under the frozen schedule.

## Frozen sources and precision

Baseline HEAD/ancestor: `7d0add17cf1aaa933650b3affc6a5b6e2bc747ce`. Original source hashes and all historical fresh-pilot small artifacts remain unchanged (protocol/provenance JSON). The constructor exactly reproduced initial model-state hash `359088b9713e532be0f811b4bd9f5f4f91b34d531a936db7548f46b2b410aedd`, seed 2026, six layers, width 32, Gx/Gc enabled, dropout zero, 9446 unique native F32 coordinates. Only authenticated FULL_SOURCE t0/t16/t32 pilot states and initial/endpoint gradients were read. No B8/P536/S5 model or optimizer was loaded.

FULL_SOURCE H2/BeH2/CO counts are 47080/89200/85960. Source AO/descriptors/quadrature remain F64. Learned shadow/partial accumulation is F64; the production PBE branch remains F32 (`train_models/lap_operator.py:271–302`). Native AdamW retains F32 storage, lr 2.5e-5, betas .9/.999, eps 1e-8, weight decay .01, foreach false. One mean-of-three F64 gradient is cast once per replay update. `tools/lap_fresh_grid_pilot_v1.py:289–312` supplies the unchanged two-pass matrix adjoint evaluation; the metric is `train_models/lap_operator.py:422–440`. No subset comparison was repeated.

## Exact replay and gradient ownership

| State | Mean loss | Gradient index | Gradient norm |
|---|---:|---|---:|
| theta0 | 1.268092669580413 | g1 | 33745589.04345739 |
| theta1 | 0.5587582287886662 | g2 | 694.8771866309861 |
| theta2 | 0.5637659321967876 | g3 | 430.5922970706028 |
| theta32 (authenticated cache) | 0.5199287840361003 | g33 | 201.43661503863174 |

Exactly TWO molecular native optimizer steps were executed, both exact replay, not new intervention training. theta16/theta32 were read only; t33 was not applied. Small synthetic optimizer unit fixtures are separate from molecular replay. Initial g1 agrees with its authenticated cache. Per-system and mean parameter gradients are in `lap_fresh_layernorm_v1_parameter_gradients.csv`; complete vectors are hash-receipted outside Git. Ownership uses actual LayerNorm module identity.

At theta0, `x_feature_extractor.1.bias` has norm 33744021.090396255; coordinate 6 has |gradient| 18267962.16256031. Its squared-norm fraction is 0.9999070742767314. Next are `x_feature_extractor.3.fc.1.bias` (325276.90109632746, max coordinate 16) and `.3.fc.4.bias` (3933.5875436277865). Correlation is not declared dead from its small operator-only share.

## Constraint-path attribution and hidden statistics

Semantic roles follow the actual calls in `train_models/NN_models_lap.py:85–150`: real exchange up/down, UEG exchange up/down, real/UEG/high-density/rapid-gradient correlation direct/swap and fused paths. Shared exchange hidden paths into correlation are retained, including their mixed density/parameter derivatives. Only LN affine parameter uses are expanded into equal replicas. These are diagnostic derivatives, not alternative training models. Energy and each density partial agree on a full H2 chunk, as do assembled matrices and full three-system losses. Gradient branch-sum relative closure is 2.322e-16, 2.401e-16, 3.233e-16 at theta0/1/2. Profile before full attribution: 0.854723 seconds per tested chunk.

At theta0 each UEG spin path contributes norm 16872794.880478688 and signed projection fraction 0.5000000106301892 onto the TOTAL gradient. Each ordinary exchange contribution has norm 5.329570659502747 and projection fraction -1.0631798756756775e-8. Fractions are signed dot projections, not independent percentages of norms. Full contributions/cancellation can be reconstructed from hash-bound branch arrays. The two UEG paths remain dominant after update 1 (each norm 347.41398334956574) and update 2 (215.26185253919408).

Every exchange UEG LN input is exactly zero at theta0, across 222240 repeated anchor calls per spin and module. These repeats are not independent physical observations. Actual eps = 1e-5 and inverse standard deviation = 316.227766. The first bias-free Linear still has zero anchor output at theta1/2. Its LN bias becomes nonzero (norm 0.00014142136 after update1). Downstream `.3.fc.1` variance rises from zero to 8.995244162e-11; `.3.fc.4` to 8.561932077e-7. Later `.4.fc.1` variance rises to 0.01164849535 and inverse standard deviation falls to 9.26144; `.4.fc.4` reaches variance 0.2438917697. Real-grid and artificial-constraint statistics are separately labelled in `lap_fresh_layernorm_v1_call_statistics.csv`. Downstream first-layer values are independently recorded in constraint_checks JSON.

The local LN input Jacobian at zero scales as diag(gamma)/sqrt(eps) times the centering projector. Its OWN bias derivative is identity, not 1/sqrt(eps). The dominant early bias receives downstream-chain amplification. Residual connections, GELU, weights and cancellations are included in the measured graph; no naive product of 316 factors is used. Positive eps makes this poorly conditioned startup rather than a mathematical singularity. Branch cancellation is quantified as norm(sum contributions)/sum(norm contributions): [0.9999996535281449, 0.9990906143299326, 0.9985267907875404] at theta0/1/2 (higher means less cancellation). These ratios are not causal percentages.

## Independent derivative checks and retained limits

Independent F64 LN formula uses biased feature variance and actual eps. Forward, first and mixed input/affine derivatives pass frozen atol 1e-10 / rtol 1e-9 on real observed minimum/maximum-variance inputs at theta0/1/2 (including zero anchors).

H2 directions, selected deterministically from mean g1 BEFORE finite differences: `.1.bias[6]`, `.3.fc.1.bias[16]`, `.3.fc.0.weight[887]`, all in x_feature_extractor. The corresponding H2 derivatives at theta0 are 6270290.387128715, -43644.8038976052, 3.89822656769507. All three have qualified adjacent-step FD agreement under the frozen 1% rule, with best relative errors approximately 2.23e-6, 1.57e-6 and 2.97e-7. Thus the enormous startup derivative is genuinely present in the tested scalar, not just a backward-accumulation norm artifact.

At theta1, the dominant bias and non-LN control qualify (best relative gaps 4.35e-5 and 2.65e-6). Secondary bias has AD 0.029319474349165; its smallest predetermined step gives FD 0.029128747796065 (0.6505% gap), but the next step has 2.626% gap and larger steps exhibit strong curvature. There is only one qualifying step, so its convergence status is UNRESOLVED. No step refinement or tolerance widening was performed. All 42 predeclared step sizes (7 per direction/state) and scalar values are retained.

The independent discrete-energy path differentiates energy with respect to a full symmetric-density-matrix perturbation, then to parameters. It calls neither the production AO assembler nor local-partial helper. It shares the physical PBE formula and model. Weighted F32 energy backprop rounds quadrature in a different place; matrix relative differences are 5.873612507e-9 / 3.634464309e-9, coordinate derivative relative gaps 1.28e-10 to 1.46e-8. Four comparisons exceed strict F64 rtol 1e-9 and remain explicit FAIL for that arithmetic comparison. They are not silently relabelled exact parity, nor treated as a demonstrated production bug. There is no promoted-F64 oracle or extra FD sweep. Perturbations are F64, not rounded back into F32 parameters. A fixed base matrix adjoint contraction has the original loss derivative at the base state; it is not a new training objective.

## AdamW memory in actual update directions

For native F32 g1, first-moment contribution is .1*.9^(t-1)*g1; first second-moment contribution is .001*.999^(t-1)*g1^2. Saved t16/t32 moments are authenticated. Intermediate gradient vectors were not saved: rest is accounting residual from saved moments, not an independent history replay. At t1, rest is solely native rounding, not another gradient. Negative v_rest entries are reported, not clipped; all lie within the explicitly recorded conservative F32 accumulation bound. Full per-coordinate ratios are outside Git with SHA receipts.

| t | ||m_first|| | ||m_rest|| | ||m_total|| | cos(m_first,current gradient) |
|---|---:|---:|---:|---:|
| 1 | 3374558.8798 | 0.11457 (rounding) | 3374558.9757 | -0.177259 |
| 2 | 3037102.9918 | 69.44674 | 3037090.7517 | -0.197617 |
| 16 | 694791.7481 | 213.61615 | 694777.1264 | unavailable (g17 not saved) |
| 32 | 128746.3136 | 194.62597 | 128733.3235 | -0.0965503 |

Components below use ONE actual full-proposal denominator, actual bias correction, native CUDA ordering, and separately measured weight decay/rounding. CPU arithmetic initially differed from CUDA in 103 coordinates by at most 3.7252903e-9; using the same CUDA path reproduces actual update2 parameters byte-for-byte. No optimizer reset or alternative trajectory was run.

| update | component | displacement norm | grad L dot displacement |
|---|---|---:|---:|
| 2 (actual replay) | first-gradient numerator | 0.00142811364 | +0.004817969742 |
| 2 | rest | 0.00085427823 | -0.000605736539 |
| 2 | decay | 5.16160e-6 | -1.35259e-7 |
| 2 | rounding | 6.79636e-7 | -1.06243e-8 |
| 2 | total actual | 0.00167724524 | +0.004212087320 |
| 33 (NOT applied) | first-gradient numerator | 3.12935e-5 | +6.70315e-7 |
| 33 | rest | 0.00192909440 | -0.002481288466 |
| 33 | total native proposal | 0.00193374085 | -0.002480760021 |

Actual replay update2 loss increases by 0.00500770340812, consistent in sign with its linear prediction. At update33, first-gradient displacement remains locally adverse but is small; the total proposal predicts descent. Raw large first-moment norm does not imply domination of the preconditioned update. Stored m32 first contribution has beta1^31; next m33 has beta1^32. The second-moment denominator is nonlinear and has no claimed additive causal percentages. Removing g1 might change the entire trajectory: this arithmetic accounting does not show that a reset or omission would improve training.

## Integrity, outcome and one next experiment

Tied-parameter UEG, spin, high-density, rapid-gradient and tau-independence constraints pass at theta0/1/2/16/32. Model/buffers/RNG restore checks pass; source/history hashes unchanged. No SCF, old functional checkpoints, chemistry/development data, production detach, architecture/loss change or optimizer change occurred. Gradient summation and bias correction were tested; see focused tests. Large arrays and diagnostic states are outside Git with SHA receipts. Seven focused tests passed before report generation; final check receipt records the complete final test/lint/compile run.

Statuses: DERIVATIVE **UNRESOLVED** (startup spike qualified, one secondary theta1 FD limit); STARTUP **CONSTRAINT_LAYER_NORM_SUPPORTED**; OPTIMIZER MEMORY **DIRECTIONALLY_ADVERSE_AT_TESTED_STATE**. H1 supported for initial dominant derivative, no production defect confirmed. H2 strongly supported by closure-qualified semantic attribution. H3 supported for actual second-step direction; not established as globally harmful at t32 or over a counterfactual trajectory.

One next controlled experiment: one matched fresh-seed FULL_SOURCE operator-only pair comparing original LN eps with ONE prospectively fixed larger eps, same weights/data/optimizer/horizon, after exact-constraint and derivative preflight. No eps value selected here, no sweep or new arm launched. This tests startup conditioning without conflating grid selection, LR or moment reset.

Scientific wall at report generation: 1086.182 seconds of 7200. Measured replay gradient-pass compute: 76.576 seconds. GPU-active time was not separately metered; workflow/coding/idle time is not labelled GPU computation. This is a three-system startup audit, not chemical generalization, full90 qualification, or proof of the effectiveness of any future intervention.
