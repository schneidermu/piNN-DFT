# Trial 19 Hypothesis Sweep Analysis

Date: 2026-05-19

This note summarizes the first Trial 19 hypothesis sweep (`h1`-`h5`) against the original
`PBE-LGxGc_6_32` D3/mRKS baseline.

## Baseline

The reference baseline is the original Trial 19 bridge replay with D3/mRKS dispersion:

```text
WTMAD-2 = 6.5
avRANE  = 0.52
```

Training-history reference points:

```text
final train_fchem = 35.02
final val_fchem   = 40.47
final val_vxc     = 0.15492
final val_exc     = 1.057

best val_fchem epoch = 413
best val_fchem       = 39.41
best val_vxc         = 0.15569
best val_exc         = 1.079
```

## Sweep Summary

| run | model | training read | WTMAD-2 | avRANE | conclusion |
|---|---|---|---:|---:|---|
| H1 | `6x32` | slow VXC decay | 6.9 | 0.536 | worse than baseline |
| H2 | `6x32` | E_xc repair bridge | 6.75 | 0.523 | fchem promising, E_xc unstable |
| H3 | `6x32` | strong VXC | 7.03 | 0.517 | better density, worse thermochemistry |
| H4 | `6x48` | wider model | 6.5 | 0.506 | only Pareto improvement |
| H5 | `6x32` | fully clipped conflict handling | 11.618 | 1.140 | pathological |

## Training-Metric Highlights

| run | best train_fchem | best val_fchem | best val_vxc | best val_exc | notable behavior |
|---|---:|---:|---:|---:|---|
| baseline | 32.85 | 39.41 | 0.15126 | 0.982 | balanced reference |
| H1 | 34.17 | 42.20 | 0.15381 | 0.996 | no useful improvement |
| H2 | 28.93 | 33.10 | 0.14467 | 1.004 final/best late; huge mid-run spikes | best fchem, unstable E_xc |
| H3 | 36.54 | 42.23 | 0.14113 | 1.119 | VXC improves, fchem degrades |
| H4 | 35.77 | 40.26 | 0.14483 | 1.197 | density improves despite mediocre training metrics |
| H5 | 19.31 | 34.86 | 0.01266 | 1.676 | ultra-low VXC, bad E_xc and SCF density |

H2 and H5 are useful diagnostically:

- H2 proves much lower validation fchem is reachable.
- H5 proves very low train fchem and very low supervised VXC are reachable.
- Neither gives a usable balanced functional because exact energy and/or SCF density fail.

## avRANE Details

| run | mean RANE | rho | grad | lapl |
|---|---:|---:|---:|---:|
| baseline | 0.52 | n/a | n/a | n/a |
| H1 | 0.5365 | 0.5064 | 0.4524 | 0.6507 |
| H2 | 0.5235 | 0.4966 | 0.4381 | 0.6357 |
| H3 | 0.5171 | 0.4857 | 0.4368 | 0.6287 |
| H4 | 0.5065 | 0.4744 | 0.4409 | 0.6042 |
| H5 | 1.1395 | 0.7691 | 0.9828 | 1.6666 |

H4 is the best density model. Its improvement is broad and mostly comes from the Laplacian component.

H5 is not a single-system outlier. It is poor across all systems and especially bad in the Laplacian:

```text
F2   lapl NIAD = 0.938
HF   lapl NIAD = 0.909
CO   lapl NIAD = 0.893
LiF  lapl NIAD = 0.870
N2   lapl NIAD = 0.857
H2O  lapl NIAD = 0.820
```

## WTMAD-2 Details

| run | WTMAD-2 | delta vs baseline |
|---|---:|---:|
| baseline | 6.5 | 0.00 |
| H1 | 6.9 | +0.40 |
| H2 | 6.75 | +0.25 |
| H3 | 7.03 | +0.53 |
| H4 | 6.5 | 0.00 |
| H5 | 11.618 | +5.118 |

H4 preserves the baseline WTMAD-2 while improving avRANE.

## Correlations And Lessons

### 1. Supervised VXC Is Not Predictive Enough

H5 has extremely low supervised VXC:

```text
best val_vxc ~= 0.0127
final val_vxc ~= 0.0217
```

But externally:

```text
WTMAD-2 = 11.618
avRANE  = 1.140
```

So low VXC loss can be achieved in a nonphysical basin. It can match the supervised local target while producing poor SCF densities and poor thermochemistry.

### 2. Gradient Conflict Handling Is Powerful But Dangerous

H5 shows that long clipped-gradient training can protect and strongly optimize the VXC objective. However, it also decouples the local potential shape from integrated `E_xc` and SCF behavior. The resulting functional is not usable without stronger physical guardrails.

### 3. Width Helps Density

H4 is the only model that improves the Pareto front:

```text
baseline: WTMAD-2 = 6.5, avRANE ~= 0.52
H4:       WTMAD-2 = 6.5, avRANE  = 0.506
```

This suggests `6x48` has enough extra capacity to improve SCF density without paying a thermochemistry penalty, at least under the tested schedule.

### 4. Fchem And Density Are Weakly Coupled

H2 found much better validation fchem during training, but did not improve WTMAD-2 or avRANE enough. Training fchem improvements must be validated externally.

### 5. Laplacian Quality Dominates avRANE Differences

Across H1-H4, rho and gradient RANEs vary modestly. The Laplacian component is the main separator. H4 wins mainly by reducing the Laplacian RANE.

## Current Ranking

For a deployable candidate:

```text
1. H4
2. baseline
3. H3
4. H2
5. H1
6. H5
```

For diagnostic value:

```text
1. H5: proves low train fchem + low supervised VXC basin exists
2. H2: proves much lower validation fchem basin exists
3. H4: proves width improves external density without WTMAD penalty
```

## Recommended Next Direction

Do not continue H5 directly as a candidate functional. H5 is a useful diagnostic, but its SCF density and WTMAD-2 are pathological.

The next sweep should use:

```text
H4 architecture: 6x48
H5 lesson: mild objective-wise clipping
Avoid: extreme VXC collapse
Add: stronger E_xc guardrails
Goal: preserve WTMAD-2 while lowering avRANE further
```

Most promising next design:

```text
PBE-LGxGc_6_48
moderate VXC weight
mild clipped gradients
E_xc pressure throughout, not only late
avoid the H5 ultra-low-VXC basin
```

The new H6-H10 sweep should be interpreted with this in mind:

- H6-H9 test H5-derived repair strategies.
- H10 is the most conceptually aligned with the latest external evidence because it combines wider capacity with H5-style discipline.

If H10 improves avRANE while keeping WTMAD-2 near 6.5, it should become the new parent.

## H6-H9: What We Can Learn About Lowering Fchem

The second sweep (`trial19_hypothesis_sweep_2`) confirms that H5-style optimization contains real information about how to lower the chemistry objective, even though the same runs are not usable as final functionals.

Available completed histories:

| run | final train_fchem | best train_fchem | best val_fchem | val_vxc at best val_fchem | val_exc at best val_fchem |
|---|---:|---:|---:|---:|---:|
| H6 | 16.04 | 14.47 | 32.95 | 0.01471 | 11.73 |
| H7 | 15.10 | 13.90 | 31.09 | 0.01618 | 10.24 |
| H8 | 11.63 | 11.38 | 28.75 | 0.01492 | 7.90 |
| H9 | 16.06 | 14.70 | 31.81 | 0.01571 | 11.39 |

H8 is the strongest fchem result so far:

```text
epoch 845
train_fchem = 12.73
val_fchem   = 28.75
val_vxc     = 0.01492
val_exc     = 7.90
```

That is a real fchem basin, not just training noise. However, it is still exact-energy broken.

### Shared Fchem-Lowering Pattern

Across H6-H9, the useful fchem drop appears after roughly epoch 500, not during the early VXC collapse:

```text
H6 epoch 500 -> 800: train_fchem 25.71 -> 16.04
H7 epoch 500 -> 750: train_fchem 24.54 -> 15.10
H8 epoch 500 -> 900: train_fchem 25.82 -> 11.63
H9 epoch 500 -> 800: train_fchem 22.98 -> 16.06
```

The common ingredients are:

- objective-wise clipping during the high-conflict part of training
- low or moderate learning rate (`2.0e-4` to `2.4e-4` worked best here)
- long late training horizon, at least 750-900 epochs
- late reaction gradient scale restored to `1.0`
- `accum_iter = 2` in the late phase
- VXC weight reduced to about `8-15` late, not kept dominant
- E_xc kept present, but not strong enough to repair the basin

The fchem improvement is therefore probably coming from a late reaction-polish regime that only becomes stable after clipped multi-objective conditioning.

### What Did Not Explain The Fchem Drop

The lowest supervised VXC point is not the point with best chemistry:

```text
H8 best VXC epoch 176:
val_vxc   = 0.01030
val_fchem = 114.35
val_exc   = 98.76

H8 best fchem epoch 845:
val_vxc   = 0.01492
val_fchem = 28.75
val_exc   = 7.90
```

So the useful signal is not "make VXC as low as possible". The useful signal is "use clipped VXC/exc/reaction training to reach a parameter region where late fchem optimization can move aggressively without immediately losing supervised VXC".

### H8 vs H7/H9

H8 likely wins because it is more conservative early and longer overall:

```text
lr_train = 2.0e-4
accum_iter = 4 early, then 3, then 2
reaction_grad_scale = 0.25 early, then 0.6, then 1.0
n_train = 900
```

This delays aggressive chemistry optimization until the model is already conditioned. The late phase then has enough runway to keep reducing fchem:

```text
H8 epoch 750: train_fchem = 16.60, val_fchem = 30.30
H8 epoch 800: train_fchem = 14.99, val_fchem = 30.10
H8 epoch 850: train_fchem = 12.99, val_fchem = 30.55
H8 epoch 900: train_fchem = 11.63, val_fchem = 30.11
```

Val fchem stops improving much after epoch 750, but train fchem continues to improve. That suggests H8 crosses into overfitting or objective mismatch after the best validation point.

### Reusable Knowledge

The useful fchem recipe is:

```text
1. Start with clipped multi-objective conditioning.
2. Keep early reaction pressure weak.
3. Let VXC fall, but do not select based on minimum VXC.
4. Switch to late reaction-driven training with accum_iter=2.
5. Run long enough: 750-900 epochs.
6. Select by validation fchem plus E_xc guardrail, not by final epoch or VXC.
```

For future candidate sweeps, the target should not be H8 itself. The target should be H8's fchem schedule transplanted onto the H4/H10 wide architecture and constrained away from the E_xc-broken basin.

### Practical Next Hypothesis

Most likely next useful direction:

```text
architecture: PBE-LGxGc_6_48
lr_train: 1.4e-4 to 2.0e-4
early accum_iter: 4
late accum_iter: 2
early reaction_grad_scale: 0.25-0.35
late reaction_grad_scale: 1.0
gradient handling: clipped per objective through the conflict phases
late vxc_loss_scale: 10-15, not below 8 unless guarded
E_xc: persistent guardrail, stronger selection criterion rather than only more loss weight
selection: best val_fchem subject to val_exc cap, not final epoch
```

The key change from H6-H9 should be selection/guarding, not simply more E_xc weight. H6-H9 already tried stronger E_xc pressure (`2x-4x`) and still landed in high `val_exc`. That points to a basin problem, not just an underweighted term.

### Red Flags To Avoid

Avoid selecting checkpoints that look like this:

```text
val_vxc < 0.012
val_exc >> 2
val_fchem > 50
```

Those are the early collapsed-VXC points. They are not useful chemistry models.

Also avoid treating very low train fchem alone as success. H8 shows `train_fchem ~= 11`, but the exact-energy residual is still about one order of magnitude too high for a deployable functional.

## H11-H15 Design

The third sweep (`trial19_hypothesis_sweep_3`) is designed to extract the useful fchem-lowering behavior from H6-H9 while avoiding convergence into the collapsed-VXC basin.

H11-H15 still save the final epoch checkpoint. The point is to make the schedule converge to a better endpoint, not to rescue a transient mid-run checkpoint.

| run | architecture | main hypothesis |
|---|---|---|
| H11 | `6x48` | H8 fchem schedule on the H4-width model, with less extreme VXC collapse |
| H12 | `6x48` | same wide/H8 idea but with stronger persistent E_xc guard pressure |
| H13 | `6x48` | H4 external-density schedule plus a long clipped fchem tail |
| H14 | `6x48` | H9 low-decay fchem lesson moved to the wide model |
| H15 | `6x64` | wider capacity test with conservative H8-style guarded training |

Expected readout:

```text
H11: best chance to combine H8 fchem with H4 density behavior
H12: tests whether stronger E_xc pressure prevents the H8 energy blowup
H13: safest WTMAD/avRANE-preserving candidate
H14: tests whether low weight decay caused useful fchem movement rather than only pathology
H15: tests whether capacity, not just 6x48 width, is the external-density lever
```

Selection target for the JSON/checkpoint should be:

```text
val_fchem < 30 if possible
val_exc much lower than H8, ideally < 3 first and then < 1
val_vxc not used as the main success criterion
final epoch should satisfy the chemistry + E_xc filters
```
