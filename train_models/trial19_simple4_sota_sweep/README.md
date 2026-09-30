# Four-stage E3 compression sweep

These ten 500-epoch runs keep the fixed 6x32 Gc-SVELU-mirror architecture,
seed, data, optimizer, and mRKS dispersion treatment. Every run has four
conceptual stages:

1. potential anchoring, epochs 1-72;
2. joint representation learning, epochs 73-280;
3. clipped chemical repair, epochs 281-440 by default;
4. ordinary consolidation through epoch 500.

The repair uses the only balance supported by the previous schedule study:
`Vxc=15`, `E_xc=3`, and reaction scale `0.75`. Consolidation returns directly
to summed gradients with `Vxc=7`; no learning-rate restart is used.

S1-S7 compress E3's three joint-training plateaus into one interpretable
representation phase and compare smooth or constant Vxc paths. S8 tests a
ten-epoch-earlier repair exit. S9-S10 test whether reducing absolute `E_xc`
supervision during consolidation avoids the energy-generalization drift seen
in the 800-epoch runs. All jobs save model snapshots every ten epochs from
10 through 500 for later external DietGMTKN30 SCF selection.

From the repository root:

```bash
for job in train_models/trial19_simple4_sota_sweep/*.slurm; do
    sbatch "$job"
done
```

The targets are WTMAD-2 below 5.5 and avRANE below 0.47. They are experimental
targets, not outcomes inferable from the internal training losses.

## Planned S5 reproduction and timing sweep

The planned comparison uses only `s5_two_step_40_10.slurm` from this folder
plus the ten timing jobs in `../trial19_occam3_timing_sweep/`. Follow that
folder's README to regenerate the single `checkpoints_dietclean_noval_v1`
corpus and submit exactly those 11 jobs. The common runner requires its
verified preprocessing manifest, forces fresh predopt, and uses a fresh
`dietclean_noval_v1` output directory. It saves every tenth epoch from 10 to
500 and training state every ten epochs. It does no internal validation or
checkpoint selection. The R=281, F=441 control is schedule-equivalent to S5.
