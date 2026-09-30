# S5 repair/finalization timing sweep

The sweep starts from `E3_SCHEDULE_PRESETS["simple4_two_step_40_10"]` and
changes only the repair start epoch `R` and finalization start epoch `F`.
Before `R`, each variant uses the original S5 parameters at the same absolute
epoch; repair and finalization use the original S5 objectives without rescaling
or stretching any phase.

| Epochs | S5 objective |
| --- | --- |
| 1-72 | Original Trial 19 `clip_drive` potential anchor |
| 73-176 | Vxc 40, Exc 1, reaction scale 1, `sum` |
| 177-280 | Vxc 10, Exc 1, reaction scale 1, `sum` |
| R-(F-1) | Vxc 15, Exc 3, reaction scale 0.75, `clip_then_sum` for Vxc and Exc |
| F-500 | Vxc 7, Exc 1, reaction scale 1, `sum` |

| Job | R | F | Preset |
| --- | ---: | ---: | --- |
| h01 | 241 | 441 | `s5_timing_r241_f441` |
| h02 | 261 | 441 | `s5_timing_r261_f441` |
| h03 | 221 | 441 | `s5_timing_r221_f441` |
| h04 | 281 | 441 | `s5_timing_r281_f441` |
| h05 | 241 | 421 | `s5_timing_r241_f421` |
| h06 | 261 | 421 | `s5_timing_r261_f421` |
| h07 | 281 | 421 | `s5_timing_r281_f421` |
| h08 | 221 | 421 | `s5_timing_r221_f421` |
| h09 | 241 | 461 | `s5_timing_r241_f461` |
| h10 | 261 | 461 | `s5_timing_r261_f461` |

`h04` (`R=281, F=441`) is epoch-by-epoch equivalent to the S5 baseline. The
standalone S5 reproduction is a separate job and output directory. All runners
write model snapshots every 10 epochs (10-500) and resumable training state
every 10 epochs. New runs refuse to start if their output directory already
exists.

All 11 jobs use the training-only protocol: 268 Minnesota base reactions
(284 minus the explicit 16 Diet overlaps), plus all valid mRKS E_xc/v_xc
systems. They have no internal validation or training-based checkpoint
selection. Each runner forces fresh predopt and refuses an existing output
directory; outputs have a `dietclean_noval_v1` suffix. No old preoptimization
checkpoint, snapshot, or training state is used to initialize these jobs.

After pulling the updated branch, run from the repository root:

```bash
bash submit_dietclean_s5_chain.sh
```

The wrapper submits exactly one CPU preprocessing job and the eleven GPU jobs
listed above. Every GPU job depends only on `afterok:<prep-id>`; they are parallel
siblings and may run concurrently after successful preprocessing. The wrapper
uses `sbatch --parsable`, accepts `JOBID;CLUSTER`, prints every role and job ID,
and exports the same absolute `CHECKPOINTS_DIR` to all training submissions.

Preprocessing uses `rocky`, one Type-D node, one task, four CPU cores and twelve
hours, with zero GPUs in `ML_param`. No explicit memory request is made: the
cluster reports `RealMemory=1` MB per node and `DefMemPerNode=UNLIMITED`, so
positive requests such as `--mem=128G` cannot match its scheduler metadata.
This uses the partition's default memory policy; it does not allocate all CPU
cores or request exclusive node access. The implementation has serial Python/HDF5
loops and native tensor kernels; four threads avoid reserving all 48 cores.
Resident grids and augmented arrays can consume substantial memory. Actual peak
memory and runtime are unmeasured because raw H5 sources are absent locally;
inspect resource usage after the first cluster run where accounting is available.
The script prints source directory sizes, host, commit and package versions.

The target `<repo>/train_models/checkpoints_dietclean_noval_v1` must not exist.
Missing Minnesota H5 or mRKS split inputs fail before submission; existing data is never removed.
Regeneration requires 284 source reactions, exactly 16 exclusions and 268 retained
reactions, with cleaned predopt and all valid mRKS systems. Final `--verify-only`
checks all artifact hashes. The manifest records counts, exclusions, provenance
and hashes and prints its SHA256. Both GPU runners independently verify it before
`torchrun` and require ancestry of `29b36ac2faa29d31a758e0067499f10b41c3ab39`.

Inspect the chain using:

```bash
squeue -u "$USER" -o "%i %j %T %r %E"
mj
sacct -j <prep-id> --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

Dependent GPU jobs remain pending until preprocessing succeeds. A failed
preprocessing job prevents them from launching (typically DependencyNeverSatisfied).
Check `<repo>/dietclean_noval_prep_<prep-id>.out` first, including the final manifest
and SHA256. Diagnose failed preprocessing before interpreting any training run.

Permanent epoch snapshots remain at 10, 20, ..., 500. Saved training states
are for fault recovery only and are not validation-selected checkpoints. The
final epoch-500 checkpoint is provided for convenience; a best checkpoint is
undefined until external evaluation.

After all jobs finish, evaluate each of the 50 snapshots per trajectory using
DietGMTKN30 fully self-consistent SCF. These results are the only validation
signal for schedule and checkpoint selection. Freeze the schedule, checkpoint
selection procedure and final checkpoint before evaluating the disjoint
98-reaction DietGMTKN100 test. Those test reactions must not guide debugging,
tuning, schedule selection, early stopping or checkpoint selection. External
SCF evaluation is a separate task; these launchers do not implement it.

The CPU job regenerates only Minnesota from raw H5. It concatenates the existing
`checkpoints/data_vxc_train.pickle` and `checkpoints/data_vxc_val.pickle` into
one training-only mRKS list, preserving reference `E_xc` and `Vrho` without
recalculation. Both source splits must be nonempty lists with named systems,
finite targets and consistent grid shapes; duplicate names fail explicitly.
Source paths, SHA256 hashes and split counts are recorded in the manifest and
protocol marker. Original split files remain unchanged. The source hashes record
which files were imported; they do not establish reference-data accuracy.
The raw-H5 mRKS CLI remains available for data that explicitly contains `E_xc`.

If an earlier preparation failed, preserve its partial output before resubmitting:

```bash
mv train_models/checkpoints_dietclean_noval_v1 \
   train_models/checkpoints_dietclean_noval_v1_failed_4364796
bash submit_dietclean_s5_chain.sh
```

Git is required only on the login/submission host for the chained launch. The
wrapper checks the minimum commit ancestry there and writes a per-submission
record under `train_models/launch_manifests/`, containing HEAD and SHA256 hashes
of tracked Python, shell, SLURM and CSV sources. It exports that record to every
job. CPU/GPU jobs verify checkout HEAD and those hashes with Python, without
calling Git. Changing the checkout or source files while jobs are queued fails
the checks; finish/cancel the chain before pulling more changes. This verifies
submission provenance without requiring Git in `ML_param` on compute nodes.

The submission wrapper finds Python 3.9+ using `python3`, `python`, or the
existing HSE Conda interpreter paths. No login-shell activation is required.
Set `PINN_SUBMIT_PYTHON=/absolute/path/to/python` to override discovery.
The provenance helper uses only the standard library on the submission host.
