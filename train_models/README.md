# Training the Neural Network Functionals

Corrected `lap_full_vxc` experiments follow the permanent
[one-variant chemistry policy](../CHEMISTRY_EVALUATION_POLICY.md): fixed evaluation
populations are 251 relchem and 17 AE17 identities, not eight times those counts.
Historical experiment reports retain their original definitions.

The current `vxc_training` protocol trains on Diet-cleaned Minnesota reaction
energies and **all available valid mRKS E_xc and v_xc systems**. There is no
internal fixed-density validation dataset and no random train/validation split.

## Scientific split

| Role | Data | Evaluation |
| --- | --- | --- |
| Train | Minnesota minus the 16 Diet-overlap reactions, plus all valid mRKS E_xc/v_xc | Optimization and training diagnostics |
| Validation | DietGMTKN30: 30 reactions | Fully self-consistent SCF only |
| Test | DietGMTKN100 minus chemical overlaps with Diet30: 98 reactions | Fully self-consistent SCF only |

Diet30 and Diet100 share G21EA #25. Diet30 BH76 #5 and Diet100 BH76 #6
also represent the same symmetric H + HCl barrier. Remove both overlaps from
the test benchmark; equality of subset/ID alone does not establish disjointness.
No Diet SCF calculations are performed by data preprocessing or training.

Minnesota exclusions use `(Database, original ReactionID)` from
`MN_dataset/total_dataframe_sorted_final.csv`, never flattened dictionary keys:

```python
DIET_HELD_OUT_MN_REACTIONS = {
    "DBH76": {15, 35, 42, 43, 54, 55},
    "MGAE109": {18, 28, 34, 55, 74},
    "EA13": {4, 8},
    "NCCE31": {12, 21, 30},
}
```

The current CSV contains 284 base reactions: 268 remain before unavailable
molecular H5 files or grid combinations are considered. Symmetric DBH76
forward/reverse rows are intentionally excluded separately. Original ReactionID
provenance survives augmentation. Predopt uses this same cleaned source pool;
its canonical PBE targets are unchanged. Database Fchem weights are training
objective weights, not a validation set.

## Prepare the training corpora

Install `requirements.txt` and download Minnesota H5 data as described in
`MN_dataset/README.md`. From `train_models/`, place Minnesota H5 files in `data/`
and retain the existing mRKS splits in `checkpoints/data_vxc_train.pickle`
and `checkpoints/data_vxc_val.pickle`. For the planned S5/timing experiment,
submit preprocessing and training from the repository root:

```bash
bash submit_dietclean_s5_chain.sh
```

Run standalone preprocessing utilities only on a compute node under Slurm.

Artifacts are `data_predopt.pickle`, `data_train_grouped.pickle`, and
`data_vxc_train.pickle`. Every valid mRKS H5 containing grid, vrho, weights and
E_xc contributes to both training objectives. Invalid files are reported.
An empty input overwrites the corpus with an empty list; training rejects empty
objective loaders rather than retaining old data.

Preprocessing deletes `data_test_grouped.pickle` and `data_vxc_val.pickle` with
explicit messages. If they reappear, `load_chk()` warns and ignores them. It
returns `(data_predopt, data_train, data_vxc_train)` and requires the new
`minnesota_protocol.json` and `mrks_protocol.json` provenance markers. Old
unversioned random-split training pickles fail closed: regenerate both corpora.
Do not manually copy protocol markers onto historical pickles.

## Replay and external checkpoint selection

Use `replay_trial_19_bridge.py` with the current S5 or R/F timing presets.
Only reaction and mRKS training loaders are built. Epoch history contains
`train_*` diagnostics, objective/schedule metadata and learning rates. Those
metrics do not select or rank checkpoints.

The S5 and timing launchers retain:

```text
--snapshot-start-epoch 10
--snapshot-every 10
--training-state-every 10
```

Permanent model snapshots are stored under `checkpoints/epoch_snapshots/`,
every 10 epochs through epoch 500 (or the requested final epoch). The resumable
training state retains its 10-epoch cadence. A `trial_19_final.pt` checkpoint
is saved for convenience, with `final_checkpoint_path` in the history JSON.
There is no internally selected/best checkpoint. Historical target CLI flags
(`--train-fchem-target`, `--val-vxc-target`, `--val-fchem-soft-cap`) remain
accepted for launcher compatibility and are explicitly ignored. The historical
`--save-selected-checkpoints` flag is an alias for `--save-final-checkpoint`;
it does not select or rank epochs. Select snapshots later using
external Diet30 SCF validation, then evaluate the disjoint 98-reaction test set.

Old preoptimization metadata cannot be reused under the new protocol; predopt
is regenerated. Old training-state format 1 is rejected; format 2 records the
training protocol and contains no internal-validation selection state. Existing
trained checkpoints still require full retraining with the corrected correlation
alpha normalization and the cleaned training corpora.

Standalone `optuna_joint.py` and `optuna_joint_goal_bridge.py` searches are
disabled until external SCF validation can supply a model-selection objective.
The historical `predopt_train.py` executable is also disabled; its numerical
library routines remain available. Historical ARML replay is disabled because
its multiplier optimization requires held-out non-SCF gradients; substituting
training gradients would change that algorithm. Shared S5 training primitives
and the gradient-geometry replay remain available.

## Shared corpus for the 11 planned S5/timing jobs

From the repository root on the cluster, submit the complete chain:

```bash
bash submit_dietclean_s5_chain.sh
```

This submits one CPU Type-D preprocessing job and eleven GPU jobs, each with
`afterok` on that same preprocessing job. Substantial preprocessing runs on a
compute node. The wrapper refuses an existing corpus or missing Minnesota H5 files or mRKS split pickles.
The shared directory is `<repo>/train_models/checkpoints_dietclean_noval_v1`.
Its three pickles, protocol markers and manifest are verified after generation
and independently in each training job. The manifest and SHA256 appear in logs.
See `trial19_occam3_timing_sweep/README.md` for resources and status commands.

All jobs start from scratch with `--force-preopt`, architecture
`PBE-LGxGc_6_32`, `gc_svelu_mirror`, dropout 0.0, identical data and other
training settings. Only the S5/R/F objective timing differs. Outputs include
`dietclean_noval_v1` and refuse reuse. Snapshots at epochs 10..500 and saved
training states retain their ten-epoch cadence.

Use external Diet30 SCF on all 50 snapshots per trajectory to select the
schedule and checkpoint. Hold the 98-reaction test set until the schedule,
selection procedure and final checkpoint are frozen. Training diagnostics and
saved recovery states are not model-selection signals.

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
