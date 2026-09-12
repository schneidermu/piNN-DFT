# Trial 19 no-repair ablation

These three runs remove only the chemical-repair phase from the best discrete
40-to-10 Vxc curriculum. The original potential anchor (epochs 1-72) and joint
consolidation (epochs 441-500, Vxc weight 7) are unchanged. The removed repair
interval becomes ordinary low-Vxc joint training.

| Run | High Vxc=40 | Low Vxc=10 | Consolidation Vxc=7 |
|---|---:|---:|---:|
| NR1 | 73-176 | 177-440 | 441-500 |
| NR2 | 73-140 | 141-440 | 441-500 |
| NR3 | 73-220 | 221-440 | 441-500 |
| NR4 | 73-176 | 177-440 | 441-500 |

NR1 is the exact single-factor repair ablation. NR2 and NR3 test whether a
shorter or longer high-Vxc exposure is preferable once repair is absent.
NR4 is identical to NR1 except that epochs 1-72 use ordinary `sum` instead of
`clip_then_sum`; it directly isolates the initial anchor merge strategy.

All runs save model snapshots every 10 epochs from epoch 50 onward. This allows
downstream evaluation at phase boundaries and intermediate stopping times
without repeating GPU training.

## Structural trajectories

| Run | Structural question |
|---|---|
| NR5 | Merge the special anchor into ordinary Vxc=40 summed training |
| NR6 | Test whether the anchor can transition directly to Vxc=10 |
| NR7 | Test whether the final Vxc=10 to Vxc=7 transition is necessary |
| NR8 | Retain repair weights but replace both clipped merges with `sum` |
| NR9 | Retain clipped repair while removing the special anchor |
| NR10 | Replace clipped repair by summed Vxc=10, E_xc=2 emphasis |

The frequent snapshots provide timing and stopping-time variants within each
trajectory. The six jobs therefore vary only structural operators that cannot
be recovered by selecting a different checkpoint from NR1-NR4.
