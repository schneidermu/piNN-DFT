# Frozen FS numerical precision audit

**Complete; independent numerical and provenance review passed.**

## Identity and scope

Branch `lap_full_vxc`; analyzed source HEAD `79f89dda4c9b1ce7bcfdec609774bb005b3c3080`. Fixed model: pcPBELMLOptimizerV2Lap. Cursor 2, NCCE31 reaction 0 / BH / level2_mura. CUDA replay uses point chunk 256 and AO-cache chunk 4096, world size 1, seed 41. No training, source changes, new step sizes, source regeneration, or committed optimization-state updates. Source LOC added: 0; test LOC added: 0.

| Input identity | SHA-256 |
| --- | --- |
| checkpoint_sha256 | a53f46807b1ce492c8fe02657e615881ff310294bd79d273eb4bb0c694e3670c |
| group_store_manifest_sha256 | ec254952f51d854d8b23c01ff3d316d3b4fd756b58287c637d3f07ed8c385d2a |
| selected_group_sha256 | c90140e87c652e251f1e788b240f0bb3a073bfb6b2cda6d7efc184ab903d2fe4 |
| central_BH_hdf5_sha256 | 50fcdab3e44e65297b94da9b77611bc85b9e36ef11a8370b38491b3110fb0688 |
| ao_cache_BH_sha256 | 5c24e7b2b636776e23764349fa8cd181340f9fe1c1d84aa550259c04e2545859 |

## Source dtype audit

The scratch path widens the exact float32-loaded model, Minnesota, central feature/weight, and AO values. Existing float64 Exc/reference/overlap values remain unchanged. This is not restored source precision. Dispersion constants remain float64; the original in-place float32 energy accumulation rounds them, whereas scratch accumulation remains double.

| Object | Storage | Replay arithmetic | Original precision | Native double recoverable | Double label |
| --- | --- | --- | --- | --- | --- |
| cursor-2 model parameters and float buffer | torch.float32 | torch.float32 forward/autograd | float32 checkpoint values | False | float64-arithmetic-on-float32-source |
| Minnesota NCCE31/0/level2_mura floating inputs and references | torch.float32 in frozen group pickle | torch.float32 after reaction_loss tensor_record conversion | float32 in frozen training artifact; upstream 13-GB source pickle not opened | False | float64-arithmetic-on-float32-source |
| Minnesota reaction dispersion constants | zero-dimensional NumPy float64 arrays in dispersion pickle | torch.tensor infers float64 on CPU/CUDA; in-place addition into float32 component-energy accumulator stores sum back as float32 | float64 stored constants | True | native-float64 correction; replay accumulation rounds component energy to float32 |
| BH NPZ grid coordinates and dm_ks | float64 in NPZ and central record | features are later cast to model dtype; dm_ks is provenance/reconstruction input | float64 source arrays | True | native-float64 |
| BH central mRKS features | float64 in central HDF5 | float32 after CentralAOCache cast | recomputed in double from float64 NPZ coordinates, dm_ks and PySCF AO; audited compatible with legacy float32 descriptors | True | native-float64 source, but use replay-loaded float32 values widened for first controlled comparison |
| BH mRKS weights, VxcLegacy, Exc target | float64 in central HDF5 | weights float32 in the system; Exc target float64; integrated energy/reduction float64 | float32 in hashed legacy source pickle; promoted to float64 for HDF5; Exc target retained from legacy, not NPZ exc_wf | False | float64-arithmetic-on-float32-source |
| BH reference operator RefAO | float64 in central HDF5 | float64 operator loss | double AO contraction using float32-origin legacy weights and Vrho | False | float64-arithmetic-on-float32-source |
| BH overlap | float64 in central HDF5 | float64 eigendecomposition/orthogonalization/loss | PySCF double integral | True | native-float64 |
| BH AO factor cache phi/grad_phi/lap_phi | float32 in HDF5 cache | float32 operator assembly; chunk cast to float64 before final loss | PySCF AO factors quantized to float32 on cache write | not from cached bytes; regenerable from pinned central molecule/coordinates with PySCF, not done | float64-arithmetic-on-float32-source |
| BH mRKS dispersion constant | Python float from dispersion pickle | float64 because integrated_energy returns float64 and correction uses prediction.dtype | float64 scalar | True | native-float64 |
| intermediate activations and reductions | not persisted as objective tensors | network/input path float32; integrated E reduction float64; operator reference algebra/loss float64 | mixed as above | True | mixed provenance; scalar loss dtype does not imply native-double forward inputs |

## Frozen-state reproduction and geometry

Float32 gradients and baseline losses reproduce the pinned state. The independent seven-face simplex solver reproduces weights (0.00141911590147, 0, 0.998580884099); KKT checks pass. Double-recomputed weights are (0.00142026647335, 0, 0.998579733527). Both ideal FS directions are strict common descent. The double direction norm is 0.303266280512; float32 emitted direction norm is 0.303215943758. Norm ratio is 1.00016601; cosine is 0.999998875. The chemistry gradient relative-L2 difference exceeds the predeclared 1e-3 diagnostic gate; do not call the cross-precision geometry identical or infer a gradient/objective defect.

| Task | Relative gradient L2 | Gradient cosine |
| --- | --- | --- |
| chem | 0.00117057206 | 0.999999331 |
| exc | 1.52603491e-06 | 1 |
| op | 0.000819206841 | 0.999999713 |

## Float32 requested versus realized displacement and response

Sign convention: theta_trial = theta + t*v; negative dot or loss change means descent. All diagnostic reductions use float64. Zero fractions count all trainable coordinates.

| t | Requested norm | Realized norm | Norm ratio | Cosine | Zero fraction | chem ideal | chem realized | chem actual | E ideal | E realized | E actual | op ideal | op realized | op actual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.303215944 | 0.303215949 | 1.00000002 | 1 | 0.00751640906 | -0.0919396711 | -0.0919397638 | 1.63362455 | -879.071572 | -879.07159 | 828.050494 | -0.0919399104 | -0.0919399119 | 0.0120533023 |
| 0.03125 | 0.00947549824 | 0.00947549726 | 0.999999896 | 0.999999998 | 0.0169383866 | -0.00287311472 | -0.00287261822 | 0.042989254 | -27.4709866 | -27.4709974 | 19.0736825 | -0.0028731222 | -0.0028731226 | -0.00276417555 |
| 0.0009765625 | 0.00029610932 | 0.000296112123 | 1.00000947 | 0.999998538 | 0.187910227 | -8.9784835e-05 | -8.99581632e-05 | 0 | -0.858468332 | -0.858471716 | -0.858335734 | -8.97850688e-05 | -8.97855422e-05 | -9.03540864e-05 |
| 3.05175781e-05 | 9.25341625e-06 | 9.25474393e-06 | 1.00014348 | 0.999481802 | 0.649163667 | -2.80577609e-06 | -2.20421657e-06 | 0.042989254 | -0.0268271354 | -0.0268331403 | -0.0267479324 | -2.8057834e-06 | -2.80558521e-06 | -2.15979933e-06 |
| 9.53674316e-07 | 2.89169258e-07 | 2.86352648e-07 | 0.990259649 | 0.979195659 | 0.964641118 | -8.76805029e-08 | 3.08747208e-07 | 0.042989254 | -0.000838347981 | -0.000827596684 | -0.000653888608 | -8.76807312e-08 | -8.55799137e-08 | 2.18658812e-06 |

## Double arithmetic directional response

Predicted change is t*(g dot v); directional quotient is actual change/t. The model gradients and FS direction were recomputed in double, not cast from old gradients.

| t | chem predicted | chem actual | chem quotient | E predicted | E actual | E quotient | op predicted | op actual | op quotient |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | -0.0919704369 | 1.6109935 | 1.6109935 | -879.81071 | 828.65316 | 828.65316 | -0.0919704369 | 0.0122776075 | 0.0122776075 |
| 0.03125 | -0.00287407615 | -0.00166010485 | -0.0531233553 | -27.4940847 | 19.0963898 | 611.084473 | -0.00287407615 | -0.00276448103 | -0.0884633929 |
| 0.0009765625 | -8.98148798e-05 | -8.865313e-05 | -0.0907808052 | -0.859190146 | -0.859197242 | -879.817976 | -8.98148798e-05 | -8.97080267e-05 | -0.0918610194 |
| 3.05175781e-05 | -2.80671499e-06 | -2.80551833e-06 | -0.0919312245 | -0.0268496921 | -0.0268496991 | -879.810939 | -2.80671499e-06 | -2.80661066e-06 | -0.091967018 |
| 9.53674316e-07 | -8.77098435e-08 | -8.76824533e-08 | -0.0919417161 | -0.000839052877 | -0.000839052885 | -879.810718 | -8.77098435e-08 | -8.77097456e-08 | -0.0919703342 |

## Matched-state control

At the exact rounded float32 candidate for t=2^-15, double arithmetic changes chemistry/E/operator by -2.303045799e-6 / -0.02683315547 / -2.805803841e-6. Float32 chemistry instead changes by +0.042989254. This comparison holds candidate parameters and source values fixed. At t=2^-20, double chemistry changes by +3.054872835e-7, confirming parameter rounding also destroys chemistry descent; double E and operator still decrease.

## Derivative consistency and exact loss paths

Double quotients approach their autograd dots as t decreases. At t=2^-20, relative errors are approximately 3.12e-4 / 9.31e-9 / 1.12e-6 for chemistry/E/operator. Absolute directional-quotient residuals and guarded response ratios are stored for every task in the result JSON. No centered follow-up is needed. Static audits verify the identical factory mapping and scalar definitions feed compute_isolated_task_gradients and _evaluate_trial_losses, including chemistry weights/units/dispersion, legacy Exc, and the AO operator normalization. See .planning/precision_audit/loss_path_audit.md for exact functions and lines.

## Classification and next experiment

Primary B: objective-evaluation precision. Secondary A: parameter-displacement quantization. The matched t=2^-15 state isolates arithmetic sign reversal where realized first-order products still predict common descent. The t=2^-20 state separately shows rounding-induced chemistry ascent. There is no identified gradient/objective formula defect. Genuine finite-step nonlinearity exists at larger steps, but it cannot be the primary explanation for the float32 rejection once the exact t=2^-15 candidate descends for all tasks under double arithmetic. The t=2^-20 candidate still ascends for chemistry under double arithmetic, as its realized direction predicts. Trust-region research is not justified by this audit. The single next experiment is a higher-precision line-search evaluation diagnostic: trace float32/double chemistry components and reductions at the exact frozen t=2^-15 candidate, with explicit realized-displacement checks. No training implementation is part of this task.

## Required answers

1. At 2^-20 the requested displacement is not represented exactly; only 3.54% of coordinates move.
2. 96.46% of coordinates are unchanged; relative displacement distortion is 20.32%, while the norm retains 99.03%.
3. Realized-direction cosine is 0.979196 at 2^-20 and 0.999482 at 2^-15.
4. No: chemistry becomes first-order ascent at 2^-20; E/operator dots remain negative, but operator loss still rises.
5. Yes: recomputed double FS has strict negative dots for all three tasks.
6. Yes: double quotients approach autograd directional derivatives within the tested range.
7. Yes: each task uses the identical scalar factory for gradients and reevaluation.
8. Primary objective precision; secondary parameter precision. No established mismatch.
9. No: precision failures are demonstrated and have not been corrected.
10. Higher-precision line-search evaluation diagnostic: the fixed t=2^-15 chemistry component/reduction trace.

## Verification and limitations

Windows: 58 passed, 2 skipped. WSL/PySCF: 14 passed, one existing deprecation warning. Compileall and git diff --check passed. All three canonical diagnostic drivers and the report renderer passed Ruff and py_compile. Independent numerical/provenance review passed. Earlier external versions are retained as superseded audit history; their known metric/provenance defects are corrected in the canonical float32-v2, matched-v2, and scratch-float64-v3 artifacts. Large arrays, raw drivers, and receipts remain outside Git and are hash-linked in the result JSON. This is one frozen training sample; no training or held-out accuracy claim follows. No training or Slurm jobs were run.

The failure is primarily objective-evaluation precision; the next experiment is a higher-precision line-search evaluation diagnostic.
