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

After pulling the updated branch on the cluster, regenerate one shared corpus
from raw H5 sources before submitting any job. From the repository root:

```bash
cd train_models
python prepare_training_corpus.py --mn-dir data --mrks-dir h5_vrho_from_mrks --output-dir checkpoints_dietclean_noval_v1
cd ..
```

The command refuses an existing output directory. Missing source H5 files or
empty corpora fail explicitly; it never substitutes historical pickles.
`preprocessing_manifest.json` records the Git commit, UTC timestamp/version,
source/retained Minnesota counts, exact exclusions, augmentation counts,
mRKS count, and SHA256 hashes of the generated artifacts. Both runners default
to this same directory without fallback to `checkpoints/` or `../checkpoints/`.
They verify every artifact hash and print the manifest, its absolute path and
its SHA256 in each job log. If a different corpus path is required, export the
same absolute `CHECKPOINTS_DIR` for all 11 submissions.

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

Submit exactly the following 11 jobs from `/home/mmedvedev/schnm/piNN-DFT`
after regeneration and review:

```bash
sbatch train_models/trial19_simple4_sota_sweep/s5_two_step_40_10.slurm
sbatch train_models/trial19_occam3_timing_sweep/h01_r241_f441.slurm
sbatch train_models/trial19_occam3_timing_sweep/h02_r261_f441.slurm
sbatch train_models/trial19_occam3_timing_sweep/h03_r221_f441.slurm
sbatch train_models/trial19_occam3_timing_sweep/h04_r281_f441.slurm
sbatch train_models/trial19_occam3_timing_sweep/h05_r241_f421.slurm
sbatch train_models/trial19_occam3_timing_sweep/h06_r261_f421.slurm
sbatch train_models/trial19_occam3_timing_sweep/h07_r281_f421.slurm
sbatch train_models/trial19_occam3_timing_sweep/h08_r221_f421.slurm
sbatch train_models/trial19_occam3_timing_sweep/h09_r241_f461.slurm
sbatch train_models/trial19_occam3_timing_sweep/h10_r261_f461.slurm
```
