# Occam-II schedule sweep

Ten 500-epoch runs use the fixed 6x32 Gc-SVELU-mirror model, seed, data,
optimizer, and mRKS dispersion treatment. The target is WTMAD-2 below 5.5;
avRANE remains a secondary physical-density diagnostic.

The design follows the completed schedule evidence:

- ordered high-to-low Vxc exposure is useful, while its interpolation curve is not;
- a finite clipped chemical-repair interval is better than a long repair;
- training must return to an ordinary summed objective before epoch 500;
- extending training beyond 500 epochs does not improve the functional.

X1-X5 retain those supported operators and vary one timing decision at a time.
O1-O5 are a nested Occam hierarchy. They remove the special anchor, merge the
two representation levels, remove repair, and finally remove all scheduling.
All phases use constant objectives; there are no per-epoch loss-weight curves.

| Run | Epoch schedule | Purpose |
|---|---|---|
| X1 | anchor 1-72; V40 73-140; V10 141-280; repair 281-440; V7 441-500 | earlier high-to-low switch |
| X2 | anchor 1-72; V40 73-210; V10 211-280; repair 281-440; V7 441-500 | later high-to-low switch |
| X3 | anchor 1-72; V40 73-176; V10 177-300; repair 301-400; V7 401-500 | short, delayed repair |
| X4 | anchor 1-72; V40 73-176; V10 177-280; repair 281-400; V7 401-500 | short repair and long consolidation |
| X5 | anchor 1-72; V40 73-176; V10 177-300; repair 301-420; V7 421-500 | delayed repair with balanced suffix |
| O1 | V40 1-140; V10 141-280; repair 281-400; V7 401-500 | four stages, no anchor |
| O2 | V40 1-280; repair 281-400; V7 401-500 | three stages, one representation level |
| O3 | V10 1-280; repair 281-400; V10 401-500 | one finite repair pulse |
| O4 | V40 1-200; V10 201-500 | two stages, no repair |
| O5 | V10 1-500 | one fixed objective; no schedule |

`repair` is the previously supported E3 operator: clipped gradient merge with
`Vxc=15`, `E_xc=3`, and reaction scale `3/4`. Ordinary phases use summed
gradients, `E_xc=1`, reaction scale `1`, and the stated Vxc scale.

Submit from the repository root:

```bash
for job in train_models/trial19_occam2_sota_sweep/*.slurm; do
    sbatch "$job"
done
```
