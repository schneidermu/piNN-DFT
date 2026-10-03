# Cursor-2 dtype and source provenance

Read-only audit for the frozen cursor-2 case `NCCE31/reaction 0 / BH / level2_mura`. No objective, training, or source-data generation was run.

## Identity and observed runtime

- Cursor-2 checkpoint: `C:\Dev\readWFN_share_ms\lap_step_globalization_runs_20261002\pcd_armijo_direct\run_seed41\latest.pt`, SHA-256 `a53f46807b1ce492c8fe02657e615881ff310294bd79d273eb4bb0c694e3670c`.
- The inspected FS-v2 driver selects CUDA 0 and `torch.float32`, loads cursor 2, calls `model.train()`, and makes `CentralAOCache(..., dtype=float32)`.
- Checkpoint state has 35 floating tensors, all `torch.float32`; this includes the float32 `scaling_array` buffer and model parameters. `lap_architecture_version` is `torch.int64`. Casting this checkpoint to double can only widen the stored float32 values.
- Runtime from the pinned `libxc` environment: PyTorch `2.11.0+cu128`; float32 matmul precision `highest`; CUDA matmul TF32 disabled; cuDNN TF32 enabled; FP16/BF16 reduced-precision-reduction flags enabled; CPU/CUDA autocast disabled; default float dtype float32. The FS-v2 source does not set these flags. FP16/BF16 reduction options and cuDNN TF32 do not apply to these float32 linear/matmul paths.

## Source dtype table

| Object | Stored dtype | Replay computation dtype | Original source precision | Native double available? / classification |
|---|---|---|---|---|
| Cursor-2 model parameters and floating buffer | float32 checkpoint tensors | float32 forward/autograd | float32 checkpoint values | No original double checkpoint. Cast-up only: `float64-arithmetic-on-float32-source`. |
| Model version buffer | int64 | integer metadata | integer | Not a floating input. |
| Minnesota reaction: `Grid`, `Densities`, `Gradients`, `Weights`, `Energy`, `HF_energies`, `Coefficients`, `backsplit_ind`; cached `PBE_local_energies` | selected group pickle stores floating tensors as `torch.float32`; `PBE_local_energies` is not read by `reaction_loss` | `reaction_loss` casts floating tensors to requested dtype; replay is float32 | The frozen group-store artifact is float32. Upstream 13-GB Minnesota source was not opened. | No double values in this frozen training artifact. Double scratch can widen exact values: `float64-arithmetic-on-float32-source`. |
| Minnesota reaction dispersion corrections | pickle values for both NCCE31 components are zero-dimensional NumPy float64 arrays | `torch.tensor(...)` infers float64 on CPU/CUDA; the current float32 component-energy accumulator uses in-place `+=`, which stores the sum back as float32 | NumPy float64 | The source correction is available in float64. Current replay rounds the accumulated component energy to float32; double scratch keeps the sum in float64. |
| BH mRKS coordinates and `dmks` | central HDF5 float64; NPZ `grid_coords` and `dm_ks` are float64 | central descriptors are cast to replay dtype | Original NPZ float64; central metadata restores original coordinates after exact float32 identity matching | Source doubles are present: `native-float64` before the training loader casts features. |
| BH mRKS central `DensityDescriptorsN10` | central HDF5 float64 | replay loader casts to float32 | Recomputed in the builder from float64 NPZ centers/`dm_ks` and PySCF AO values; stored central values are native-double descriptor calculations | Native double source is present, but reading it directly in the first scratch path changes the actual float32 inputs. Keep the replay-loaded float32 values and widen them for the controlled arithmetic comparison. |
| BH mRKS quadrature weights, `VxcLegacy`, `Exc` target | central HDF5 float64; current system has weights cast to float32, while target is held float64 | integrated-energy prediction/reduction and `Exc` target are float64; model input features/weights are float32 | Verified against the original all90 source pickle: `Weights`, `Vrho`, and scalar `E_xc` are `torch.float32`, then promoted into central HDF5. The source audit explicitly preserves this legacy `E_xc`; it does not substitute NPZ `exc_wf`. | No native-double legacy targets recoverable. Double arithmetic is on float32-origin target values: `float64-arithmetic-on-float32-source`. |
| BH reference AO operator `RefAO` | central HDF5 float64 | final operator loss is float64 | Built with float64 AO contraction, but input weights and `Vrho` originated as float32 legacy data | High-precision contraction with float32-source targets; not a native-double reference target. |
| BH overlap `Overlap` | central HDF5 float64 | float64 eigendecomposition, orthogonalization, and loss | PySCF `int1e_ovlp` computed/stored float64 | `native-float64`. |
| BH AO factors `phi`, `grad_phi`, `lap_phi` | AO HDF5 cache float32 | replay casts/loads float32; operator chunk is later cast to float64 for loss | Generated from the central molecular/coordinate record and explicitly quantized to float32 when cache was written | Cannot recover double AO factors from the cache. Regeneration from the stored central record is possible with PySCF, but was not performed. Upcast cache values are `float64-arithmetic-on-float32-source`. |
| BH mRKS dispersion | pickle Python `float`; loader normalizes with `float(...)` | cast to `prediction.dtype`; `integrated_energy` returns float64 even with a float32 model | Python float / float64 scalar | `native-float64` scalar in current E objective. |
| Intermediate precision | model activations are float32; `integrated_energy` accumulates in float64; operator prediction is assembled from the current inputs then cast to float64 before `operator_loss` | mixed float32 network/data and float64 reductions/reference algebra | As above | Objective scalar dtype alone does not establish source or forward precision. |

The exact BH source pickle at `C:\Users\schne\Downloads\data_vxc_train.pickle` is 496,371,047 bytes and hashes to the all90 manifest value `0ccc0cfb09814cadcc9e0537953cc324fe6ce5551654fe3254f310a25cc6ee49`. A read-only load inspected only the BH record: `Grid` `(56883,13)`, `Weights` `(56883,)`, `Vrho` `(56883,)`, and `E_xc` scalar are all `torch.float32`. No raw Minnesota source pickle was opened.

The frozen NCCE31 component corrections are `-0.00192936` and `-0.0005886` as float64 NumPy scalar arrays. In the pinned runtime, `torch.tensor(value)` is float64 on both CPU and CUDA despite default dtype float32. A direct `float32_energy += float64_correction` check leaves the accumulator float32 and matches rounding the sum back to float32. In double scratch, the same in-place addition remains float64.

## Recommended controlled float64 input policy

For the first double scratch comparison, load the exact frozen cursor-2 objective inputs through the existing float32 path, then widen those materialized float32 tensors and the checkpoint values to float64. Keep the already-float64 `RefAO`, `Overlap`, and `Exc` tensors unchanged; keep the exact objective helpers and reductions. This holds the realized source values fixed while testing double forward/autograd arithmetic. Label the path as mixed-provenance / `float64-arithmetic-on-float32-source`, not full float64. Do not swap in central HDF5 feature doubles or regenerate AO factors during this comparison; either changes source inputs and confounds the arithmetic comparison. A later source-upgrade test can regenerate AO factors from the pinned central record if needed.

## Code and data evidence

- Frozen identity/dtype selection: `C:\Dev\readWFN_share_ms\lap_direction_recovery_runs_20261003\native_fs_batch_v2.py:161-186`.
- Checkpoint conversion: `train_models/lap_checkpoint.py:117-130`.
- Minnesota group loading: `train_models/lap_moo_panel.py:821-930`; objective conversion and chemistry scalar: `train_models/lap_training.py:17-56`; bound factory: `train_models/lap_moo_training.py:571-589`.
- Original BH legacy precision and promotion: `train_models/build_lap_operator_data.py:102-163, 228-235`; BH source audit and input hash: `train_models/mrks_source_audit_v2.json` and `lap_operator_validation_report.md:113`.
- Native NPZ coordinates/density and feature reconstruction: `train_models/build_lap_operator_data.py:167-198`.
- HDF5 central normalization: `train_models/lap_operator_data.py:122-160, 210-230`.
- AO cache dtype enforcement/cast: `train_models/lap_moo_panel.py:1019-1112`; replay load/casts and float64 references: `train_models/train_lap_moo.py:288-438`.
- mRKS objective/reductions: `train_models/lap_moo_training.py:591-655`, `train_models/lap_vxc.py:327-341`, `train_models/lap_operator.py:369-390`.
- Dispersion conversion: `train_models/reaction_energy_calculation.py:89-107`, `train_models/optuna_joint.py:418-435`.

Inspected frozen-data hashes: Minnesota group-store manifest `ec254952f51d854d8b23c01ff3d316d3b4fd756b58287c637d3f07ed8c385d2a`; selected group shard `c90140e87c652e251f1e788b240f0bb3a073bfb6b2cda6d7efc184ab903d2fe4`; BH central record `50fcdab3e44e65297b94da9b77611bc85b9e36ef11a8370b38491b3110fb0688`; BH AO cache `5c24e7b2b636776e23764349fa8cd181340f9fe1c1d84aa550259c04e2545859`.

**Blocker:** none for a controlled float64-arithmetic scratch path. A fully native-double source path is impossible from the current model/Minnesota checkpoint and BH AO cache without new AO regeneration; the known legacy mRKS targets themselves are float32-origin.
