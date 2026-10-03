# Frozen cursor-2 precision audit

**Status: complete; independent review accepted primary B (objective precision) and secondary A (parameter rounding).**

## Question and scope

Distinguish parameter rounding, objective-evaluation precision, gradient-identity error, and finite-step curvature for the already verified native Fliege–Svaiter direction. Read-only audits found the same objective factories/reductions on gradient and trial paths, but did not test numerical derivative agreement. This is one frozen-state diagnostic. It does not change objectives, reductions, direction methods, training state, or line-search policy. It does not run a panel, manifest cycle, centered-difference test, or training job.

## Frozen identity and replay gate

Restore cursor 2 from the seed-41 checkpoint, NCCE31 reaction 0 / BH / `level2_mura`, incoming PCD EMA at `t=2`, model buffers, and RNG. Expected identities: checkpoint SHA-256 `a53f46807b1ce492c8fe02657e615881ff310294bd79d273eb4bb0c694e3670c`; raw-gradient NPZ SHA-256 `fd5eb469339d68ac3c7fc21728ce0138cec878ea8780c7c05277d87cf353f6c0`; sampling-manifest SHA-256 `550f22488df27b9e1326905bd3569fa7a84671e90b428ae3be190f1e409bf928`; checkpoint protocol SHA-256 `26f710fe7c00423a389d5a4416560faf4333b6e9f6f9daac5688b14bfff7fb67`. Record current source HEAD `79f89dda4c9b1ce7bcfdec609774bb005b3c3080` and hashes of the objective, training, and loader sources used.

Recompute the three raw gradients from the existing objective factories. Compare the flattened gradients with the immutable NPZ using the established `rtol=1e-6, atol=1e-8` rule. Require task order `chem, exc, op`, 9,446 trainable coordinates, cursor/sample identity, replayed losses within the established `rtol=1e-6, atol=1e-8`, and the independent seven-face simplex/KKT gate before continuing. Compare the stored witness only after solving the new simplex problem. Any mismatch stops the audit.

## Predeclared trial set and arithmetic

Evaluate exactly `t = [1, 2^-5, 2^-10, 2^-15, 2^-20]` (`1`, `0.03125`, `0.0009765625`, `0.000030517578125`, `0.00000095367431640625`). These are five points already inside the completed dyadic Armijo guard. Do not add intermediate, smaller, or larger values.

For float32, use the existing direct-update arithmetic exactly: `proposal = -c` in the parameter dtype, then `theta_trial = theta_base + t * proposal` in float32. Capture `delta_realized = float64(theta_trial) - float64(theta_base)` before restoring the base. The requested displacement is `t * float64(proposal)`. Do not substitute a float64-rounded trial for the float32 trial.

Re-evaluate the same three loss factories at each trial. Define `deltaL_i(t) = L_i(theta_trial)-L_i(theta_base)`, `q_i(t)=deltaL_i(t)/t`, and `s_i=g_i^T v` for `v=-c`. Compare `q_i` with `s_i`; a negative value predicts decrease under the plus-step convention. Report actual deltas and absolute `q-s` for every finite result.

## Controlled float64 scratch path

The comparison path is **float64 arithmetic on the exact values loaded by the float32 replay**, not full native-float64 source data. Promote the checkpoint’s float32 values, Minnesota reaction tensors, mRKS features/weights, and cached AO `phi`, `grad_phi`, and `lap_phi` after loading them through the existing float32 path. Keep existing float64 `RefAO`, `Overlap`, and `Exc` tensors unchanged. Keep the exact objective helpers, NumPy dispersion constants, and default dtype unchanged. The chemistry correction constants are float64 NumPy values; the current float32 replay adds them into a float32 accumulator, while the double scratch accumulator remains float64. Record that deliberate arithmetic-path difference. Do not load native-double central features or regenerate AO factors; either changes the source values. Record each materialized input field’s source dtype, arithmetic dtype, and source hash. Verify promoted float32 tensors round-trip bitwise when cast back to float32.

Recompute all three objective gradients in this scratch path and solve a new raw-gradient native FS simplex problem in float64. Record this path as `float64-arithmetic-on-float32-source` with its mixed-provenance fields; never call it a fully float64 source run. Recompute the same five trial responses with float64 parameters and the same promoted input values. Also evaluate a matched-state control at exactly `t=2^-15` and `t=2^-20`: promote the exact float32-rounded trial parameter tensors to float64, use the same widened float32-loaded objective inputs, and compare with the float64 scratch base. This control holds the candidate parameter values fixed while changing arithmetic; it is required before attributing a float32/float64 loss-response difference to objective precision. No other matched-state t values are allowed.

Capture the original float32 runtime flags before any calculation and again in the receipt: `torch.get_float32_matmul_precision`, CUDA matmul and cuDNN TF32 flags, FP16/BF16 reduced-precision-reduction flags, autocast status, default dtype, and deterministic/cuDNN benchmark settings. The observed baseline is matmul precision `highest`, CUDA matmul TF32 off, cuDNN TF32 on (unused by these paths), reduced-precision flags on (unused), and autocast off. Do not toggle any flag for either path.

## Measurements and predeclared thresholds

Use float64 reductions on captured arrays for every norm, dot, cosine, ratio, and residual. Guard zero or nonfinite norm denominators and report the corresponding metric as undefined. Never serialize NaN or infinity.

For each float32 and scratch-float64 trial, report requested and realized displacement norms, realized/requested norm ratio, cosine, relative distortion, coordinate zero fraction, sign-mismatch fraction, and per-module-layer summaries. For every task, report `g_i·(t v)` and `g_i·delta_realized`, plus the corresponding cosine of each displacement with the descent direction. Use `>=1e-6` as the meaningful normalized common-descent margin, reusing the already declared cursor-2 geometry margin.

For each loss path and task, set the response floor to eight ULPs at the zero-step loss magnitude in that path’s scalar dtype: `floor_i = 8 * (nextafter(L_i(0), +inf)-L_i(0))`; use the smallest positive representable value if the loss is zero. This eight-ULP allowance is a conservative guard for scalar/reduction precision; the prior same-helper zero-step replay was exact. Mark the predicted response resolved only when `abs(t*s_i) > floor_i`. Always report `q_i` and `q_i-s_i`; report a relative response ratio only when resolved, with denominator at least `floor_i/t`. Otherwise set the ratio to null and label the point `below_loss_resolution`; do not infer curvature from an unresolved response.

For float32 versus scratch-float64 gradients, report per-task relative L2 error `||g32-g64||/max(||g32||,||g64||)` and cosine, plus the cosine and norm ratio between the independently recomputed FS directions. Mark their arithmetic geometry aligned only if all three relative L2 errors are `<=1e-3` and all three gradient and direction cosines are `>=0.99999`. The `1e-3` norm allowance is about 8,389 float32 epsilons and 30 times `sqrt(78,736)*eps32`; it is a conservative diagnostic tolerance for accumulated reductions, not a theorem. The cosine gate independently limits angular disagreement. If the maximum gradient norm is below `64*eps32*max(1, max(||g32||,||g64||))`, mark relative error and cosine undefined. Disagreement is arithmetic-sensitivity evidence, not proof of a defect.

Native-FS verification uses separate checks: simplex sum error `<=1e-12`, negative-weight violation `<=1e-12`, active stationarity relative to `max(abs(Gram))` `<=1e-8`, inactive reduced-gradient violation on the same scale `<=1e-8`, nonzero `||c||`, and strict positive `g_i·c` for every objective. The native step is `v=-c`, so strict descent requires negative `g_i·v`.

## Classification rule

All numeric thresholds above are diagnostic labels and screening gates, not causal proofs. A float32/float64 gradient or direction disagreement is evidence of arithmetic sensitivity; by itself it is not a gradient/objective defect and not category C. A finite one-sided derivative residual can reflect curvature and cannot establish category C by itself.

Use the following evidence labels:

- **A — parameter quantization supported:** requested float32 displacement passes the `1e-6` descent-cosine margin, but the realized float32 displacement loses that margin or reverses a raw-gradient dot; gradient/direction differences do not confound the comparison. Report matched-state loss values as context.
- **B — objective precision supported:** at a matched-state t (`2^-15` or `2^-20`), float32 loss response is below its ULP floor or has opposite sign to the resolved scratch-float64 response for the exact same float32-rounded parameter candidate and same loaded inputs; provenance and direction checks pass.
- **C — gradient identity discrepancy supported:** only an exact code-path mismatch or an independently verified derivative inconsistency after numerical precision and truncation effects have been excluded. The static audit found no code-path mismatch. The five one-sided points alone may leave C unresolved.
- **D — finite-step curvature supported:** only if parameter quantization, objective precision, and gradient identity explanations are excluded; both requested and realized directions preserve strict common descent; zero-step replay passes; and resolved scratch responses move toward their autodiff slopes as `t` decreases while larger steps depart from first-order response.
- **E — unresolved:** evidence is mixed, below a response floor, shows arithmetic sensitivity without a supported causal label, or cannot exclude curvature/gradient-evaluation ambiguity.

Report metrics and supported evidence without forcing one cause. If the one-sided results leave gradient identity versus curvature unresolved, request independent review and a separate root gate before any centered-difference follow-up. No centered-difference diagnostic is included here.

## State safety and outputs

Use the pinned Windows `libxc` Python environment with `OMP_NUM_THREADS=1`, `MKL_THREADING_LAYER=SEQUENTIAL`, `PYTHONPATH=train_models`, and UTF-8 output; record Python/PyTorch/CUDA versions and device. Keep point chunks at 256 and AO-cache chunks at 4096. Use a single external driver under `C:\Dev\readWFN_share_ms\lap_precision_audit_runs_20261003`. Snapshot and restore model parameters and buffers, RNG, incoming EMA, and cursor in `try/finally` around every trial and the whole audit. Verify exact restoration and frozen-input hashes. Do not save a checkpoint, update EMA, advance cursor, change optimizer/scheduler state, or commit candidate parameters. Keep any arrays and all result/receipt files outside Git; record their hashes. Do not edit production source.

Stop at any provenance mismatch, zero-step loss mismatch, nonfinite metric, runtime-flag change, input-promotion round-trip failure, or restoration failure. The only next step after the bounded measurements is independent review and classification; a centered diagnostic requires a new gate.

## Execution record

The frozen capture and controlled scratch runs completed from source HEAD `79f89dda4c9b1ce7bcfdec609774bb005b3c3080`. The predeclared five-point set and matched-state controls were unchanged. The canonical external artifacts are under `C:\Dev\readWFN_share_ms\lap_precision_audit_runs_20261003`:

- Float32 capture driver/result/arrays/receipt: `precision_capture32_v2.py` (SHA-256 `2d2f0de5598d84dc8d973244347fedf78ae68340bb9d4255e5caabf5a391649b`), `precision_capture32_v2.json` (`b753b5f7b58d05a0eba1633d510e481375260141ed4c84fa8bcf4a524496f3a1`), `precision_capture32_v2_trials.npz` (`87c11afde91e05336c1610c8aa633ba78c08c2a26ab9746eecf82c38d72cee4b`), and `precision_capture32_v2_receipt.json` (`29163561f50b372f7a5704a7b0da6fcf0798dbc15c6580073652c252d016ac51`).
- Matched-state float64 controls driver/result/arrays/receipt: `precision_matched64_v2.py` (SHA-256 `30bdaeff45178e48ab8a4694fb51db81e03a7d9f09e0be538772b8130654d04e`), `precision_matched64_v2.json` (`2b8c336a001cb91593cf71e77c63f39fc3a53f2ac6175857e33db34e865a8242`), `precision_matched64_v2_arrays.npz` (`2f7eec77c981473a3e055f2b67c01c0a66c8502105d0f76293a75e6204289c7e`), and `precision_matched64_v2_receipt.json` (`8a33397b9ce061259f8e716bf349deaf407a56c0fde1164b490cf3e72825f35e`).
- Canonical five-point float64 scratch driver/result/arrays/receipt: `precision_audit64_v3.py` (SHA-256 `31d3923e9e5b94aad29780bae1d774fce66cf2d6a9987a708a7329b5ef372880`), `precision_audit64_v3.json` (`09daccfe0a528c49b0e99d549bbfaa72b63c1473f2d41e4ca667d10263a650a4`), `precision_audit64_v3_arrays.npz` (`319d8a83db372e0a6a70f1aab37e1911b6dc98ba30f0fe79737fc051cfad5344`), and `precision_audit64_v3_receipt.json` (`b8e02e396796ee632b918a44bae90c791f24692fc43c7e88a76d235ba7d91070`).

All three canonical drivers passed Ruff and `py_compile`. Receipts record successful rollback and unchanged runtime flags; the float64 paths record the combined materialized-input hash `32f8dc30616f00f84fea675258e577083297dd5e101e1c6aa7efc3b7406f08ad` at start and end. The float32 replay independently reproduced the simplex witness and the float64 scratch path independently solved its own raw-gradient simplex problem. Matched-state responses show arithmetic sensitivity; the small-step float32 parameter realization also contains substantial zero/sign-mismatched coordinates. These observations are evidence for review, not causal classifications. The diagnostic thresholds and trial set above were not changed. No category is forced pending independent review.
