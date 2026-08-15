# Trial 19 Occam schedule sweep

All runs use the fixed 6x32 Gc-SVELU-mirror model, seed 41, mRKS dispersion,
500 epochs, and final-epoch checkpoint selection. The loss curriculum is always
300 epochs of representation learning, 100 epochs of energy repair, and 100
epochs of ordinary joint consolidation.

The initial Vxc scale is 64, the nearest power of two to the measured initial
Fchem/Vxc gradient-norm ratio of approximately 73. The representation stage
ends at one quarter of that scale, and consolidation uses one eighth. Reaction
weight moves from one half to one. A single repair order `m` multiplies E_xc
by `m` and sets the reaction scale to `m/(m+1)`.

The ten jobs form a 5x2 factorial:

- representation path: geometric, cosine, linear, quarter-step, or half-step;
- repair order: 2 (`double`) or 3 (`triple`).

Submit from the repository root:

```bash
for job in train_models/trial19_occam_schedule_sweep/o*.slurm; do
    sbatch "$job"
done
```

Submit from `train_models`:

```bash
for job in trial19_occam_schedule_sweep/o*.slurm; do
    sbatch "$job"
done
```
