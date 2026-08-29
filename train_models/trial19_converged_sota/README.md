# Trial 19 convergence retraining

These jobs retrain E3, S3, and S4 from the same preoptimization and reproduce
their original objectives and global learning-rate trajectory through epoch 500.
Epochs 501-800 keep each schedule's final objective fixed and use a fresh cosine
tail from `1e-5` to `1e-7`.

Each epoch atomically replaces `checkpoints/trial_19_training_state.pt`. The file
contains the model, RAdamW optimizer, scheduler, epoch history, and per-rank RNG
states. `trial_19_selected.pt` remains the final-epoch model used for evaluation.

Submit the initial runs from the repository root:

```bash
sbatch train_models/trial19_converged_sota/e3_converged.slurm
sbatch train_models/trial19_converged_sota/s3_converged.slurm
sbatch train_models/trial19_converged_sota/s4_converged.slurm
```

To continue a run beyond epoch 800 without losing optimizer state, keep the same
output directory and pass a larger total epoch count plus its state file:

```bash
N_TRAIN=900 \
RESUME_TRAINING_STATE=optuna_joint_runs/replay_trial_19_converged_s3_gc_svelu_mirror/checkpoints/trial_19_training_state.pt \
sbatch train_models/trial19_converged_sota/s3_converged.slurm
```

After epoch 800, continuation remains at `1e-7`; it does not restart the tail.
