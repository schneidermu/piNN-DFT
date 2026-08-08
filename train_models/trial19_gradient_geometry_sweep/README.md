# Trial 19 parameter-free gradient-geometry sweep

All ten jobs use one static 500-epoch training stage. They share the Trial 19
learning rate, architecture, data, seed, preoptimization, RAdamW optimizer, and
final-epoch checkpoint rule. There are no epoch transitions or objective loss
weights.

The primary experiment is `g1_primary_pcgrad.slurm`. G2-G10 identify whether
performance comes from chemistry anchoring, auxiliary voting strength, removal
of all primary-axis auxiliary motion, cosine gating, normalization, symmetric
projection, Pareto minimum-norm aggregation, or dimensionless log-loss
gradients.

Submit all jobs from the repository root or `train_models` directory:

```bash
for job in train_models/trial19_gradient_geometry_sweep/g*.slurm; do
    sbatch "$job"
done
```

When submitting from `train_models`, use:

```bash
for job in trial19_gradient_geometry_sweep/g*.slurm; do
    sbatch "$job"
done
```
