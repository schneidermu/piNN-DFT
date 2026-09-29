# Trial 19 four-regime timing sweep

All ten runs retain O1's model, initialization, loss weights, gradient operators,
global LR schedule, seed, and 500-epoch budget. Only the start of chemical repair
(`R`) and the start of consolidation (`F`) change:

| Epochs | Objective |
| --- | --- |
| 1-140 | Vxc 40, Exc 1, reaction 1, sum |
| 141-(R-1) | Vxc 10, Exc 1, reaction 1, sum |
| R-(F-1) | Vxc 15, Exc 3, reaction 0.75, clip_then_sum |
| F-500 | Vxc 7, Exc 1, reaction 1, sum |

Priority order:

| Job | R | F |
| --- | ---: | ---: |
| h01 | 241 | 441 |
| h02 | 261 | 441 |
| h03 | 221 | 441 |
| h04 | 281 | 441 |
| h05 | 241 | 421 |
| h06 | 261 | 421 |
| h07 | 281 | 421 |
| h08 | 221 | 421 |
| h09 | 241 | 461 |
| h10 | 261 | 461 |

The O1 reference is R=281, F=401 (WTMAD-2 5.946, avRANE 0.4824).
No candidate has a validated 90% probability of beating WTMAD-2 5.5.

From the repository root on the GPU cluster, after pulling this change:

```bash
cd /home/mmedvedev/schnm/piNN-DFT
for job in train_models/trial19_occam3_timing_sweep/h*.slurm; do sbatch "$job"; done
```

Each job activates `/home/mmedvedev/anaconda3/envs/ML_param` itself and saves
training state every ten epochs plus model snapshots every twenty epochs from 200.
Outputs are under `train_models/optuna_joint_runs/replay_trial_19_occam3_*`.
