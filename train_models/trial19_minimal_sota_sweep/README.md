# Minimal three-stage Trial 19 sweep

All ten runs use the fixed 6x32 Gc-SVELU-mirror architecture, seed 41, mRKS
dispersion, 500 epochs, and final-epoch checkpoint selection. Every schedule
has exactly three constant-weight phases:

1. representation learning;
2. energy repair ending at epoch 440 (`Vxc=15`, `E_xc=3`, reaction `=3/4`);
3. joint consolidation on epochs 441-500 (`Vxc=7`, `E_xc=1`, reaction `=1`).

P1-P8 are a 2x2x2 factorial over the only representation controls:

- `Vxc=40`, the rounded time average of E3's first 280 epochs, or `Vxc=80`,
  the rounded initial gradient-balance scale;
- ordinary summation or clipped summation;
- reaction scale `1` or `3/4`.

P9 and P10 bracket E3's epoch-281 repair onset by 20 epochs while retaining
the central P1 representation parameters. These controls distinguish a
representation-state effect from elapsed repair time without adding a fourth
phase.

From the repository root:

```bash
for job in train_models/trial19_minimal_sota_sweep/p*.slurm; do
    sbatch "$job"
done
```

The target is WTMAD below 6 and avRANE below 0.45, but these external metrics
cannot be guaranteed from training losses. The factorial is designed to make
either success or failure mechanistically interpretable.
