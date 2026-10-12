# F013 — Multi-objective feasibility and AdamW geometry

## 1. Executive scientific conclusion

A strict common-descent direction for unscaled JW chemistry, AE17, integrated Exc, and the weak-form operator exists at all seven audited checkpoints. The normalized max-min margin is large at B8 t128 (0.801) and at F012 t251 (0.867), moderate at F010 t128 (0.110) and F012 t128 (0.077), and very small at F010 t64 (0.043), F010 t251 (0.0081), and F012 t64 (0.0037).

The fixed F010 and F012 weighted sums are inside that cone at B8 and at F012 t251. From t64 through t128, and at F010 t251, both weighted sums point outside it. At F012 t128 the F012 composite margin is −0.908: one historical-scale step along that training gradient raises JW and the operator. The common-descent step at the same length lowers all four objectives. Doubling the operator coefficient did not put the aggregate back inside the cone.

Exact AdamW is a separate mismatch at B8. There the full gradient is a useful common-descent direction, and the finite AdamW step still raises AE17, Exc, and the operator. The inherited first-moment state is the main rotation at that point. Later, the second-moment preconditioner follows the full gradient, and that gradient is already outside the cone.

Finite-step testing does not support the same conclusion at every checkpoint. At F010 t128 the common-descent direction lowers JW, AE17, and the operator, but Exc rises because curvature exceeds the small linear term. At F010 t251 and F012 t64 the certified margin itself is too small to be a useful AdamW-scale step. Minibatch composites match the full gradient at B8 and at four of the five predeclared F012 t128 batches. They flip task signs mainly where the full gradient is already outside the cone. The exchange block reproduces the full-space margin, so the conflict is not a missing correlation or normalization direction.

Primary classification: **A — FEASIBLE, WEIGHTED-SUM MISALIGNMENT**. AdamW mismatch and near-zero margins are real at other checkpoints and are ranked below.

## 2. Exact data and checkpoint provenance

Model `pcPBELMLOptimizerV2Lap`, 9,446 trainable parameters. No optimizer step was taken. Clean28 and Diet100 were not read.

| State | Checkpoint SHA256 | Role |
|---|---|---|
| B8 t128 | `1f38f7f8ea74334ff8f5124d2a6538619323d1f453296fa9ca6b3cd7958c437b` | parent |
| F010 t64 | `eeb9a3d5099c8db7b961b95ca11db08c4a037339c1037c3cf6ced900edc5e4dd` | supplementary |
| F010 t128 | `bd5ecc45fa44793e95d82492e2666d1c40c6fa6068b4f2e266824d3774fbc822` | priority |
| F010 t251 | `1dba8af1a11d5f98fdc5b40d8be6b137d9773af3a6f2bc9eb1482750b3a5feb3` | priority |
| F012 t64 | `8a1efed0c4feae8dc75290800d5c255534ae23b557cdf2e35563a7a0c8158838` | supplementary |
| F012 t128 | `100ce7f50ff6526c04e7747de2b2ba35466f6032ba3d28d789bca8c18f200996` | priority |
| F012 t251 | `a4d3a91e1f8e49cd3d10eb146ccc9a031a78fd7191a6db0eb5a9c37b63342d8a` | priority |

Every file hash matched before its gradients were used. Each checkpoint contains its own AdamW moments. B8 moments were not copied onto later states.

Training manifest SHA256 `b249964d410f0db4a59867de53bb35c6442ec23aa81eefb0b96b50242d94f6bc`. Immutable `A_d` SHA256 `3233a7b3194e68af447bb63edda0d154325da7d051e0df6fe60a7a788cebc73d`. Chemistry scale `c010 = 9.974823077811354`. F010 operator weight `0.33597561607048215`; F012 uses twice that value. The four primary gradients are the unscaled objectives. The weights appear only in the composite gradients.

Gradients were rebuilt from the existing 251-by-9,446 reaction Jacobians and the saved 90-system AE17, Exc, and operator gradients. Provenance hashes are in `lap_f013_gradient_receipts.json`.

## 3. Reproduction of previous scalar and gradient receipts

JW scalars match the published checkpoint metrics to the stored digits (absolute difference 0). Historical relchem, recomputed from the same residuals, matches the published relchem column, including 1.195939 at B8 and 1.301021 at F012 t128. It was not substituted for JW.

The identity `g_F012 = g_F010 + λ_op^F010 g_op` holds at every state. The largest residual norm is `5.4e-17`.

At F012 t128 the minimum cosine between a unit task gradient and the F012 composite is −0.908, carried by the operator. That reproduces the published `cos(operator, total) = −0.908276`. The JW–operator cosine is 0.394, matching the published 0.393768.

Database pieces rebuild the full JW gradient with residual below `1e-8`. SLSQP dual gaps are below `1e-16`, and the active-task margins agree with `γ*` to `1e-9`. A projected-simplex cross-check never found a direction with a higher minimum margin. Tolerances were fixed in `lap_f013_geometry_audit_protocol.json` before these comparisons.

The raw four-gradient Gram matrix is dominated by Exc and AE17 norms of order `10^2` to `10^4`. Its condition number is not used. The unit-gradient Gram matrix has numerical rank 3: one eigenvalue is zero because AE17 and Exc are nearly collinear (cosine +1.000 or −1.000 or −0.989).

## 4. Four-task feasibility

Descent direction `d` means the step `θ − η d`. A task descends when `u_i · d > 0`. `γ*` is the maximin of those normalized margins on the unit sphere.

| Checkpoint | γ* | Active tasks | F010 composite margin | F012 composite margin |
|---|---:|---|---:|---:|
| B8 t128 | 0.801107 | JW, AE17 | +0.6513 | +0.6794 |
| F010 t64 | 0.043141 | all four | −0.1948 | −0.5218 |
| F010 t128 | 0.109710 | JW, AE17, operator | −0.3220 | −0.6837 |
| F010 t251 | 0.008070 | all four | −0.9356 | −0.9773 |
| F012 t64 | 0.003686 | JW, AE17, Exc | −0.2806 | −0.7505 |
| F012 t128 | 0.076620 | JW, AE17, operator | −0.9587 | −0.9083 |
| F012 t251 | 0.867018 | JW, operator | +0.7243 | +0.6878 |

Inactive tasks have a larger margin than `γ*`. At B8, Exc and the operator are slack (margins 0.805 and 0.906). At F012 t251, AE17 and Exc are slack. A zero multiplier does not mean that task gets worse along `d*`.

The composite margin is the worst unit-task cosine with that weighted gradient. A negative value means the training direction is outside the common-descent cone. Cosines of `d*` with the F010 and F012 composites are 0.972 and 0.973 at B8, 0.531 and 0.509 at F010 t128, 0.175 and 0.297 at F012 t128, and 0.970 and 0.957 at F012 t251. Angle and margin are different statements: at F012 t128 the composite still has a positive cosine with `d*` while its worst-task margin is −0.908.

JW–operator cosines stay positive at every state (0.299 to 0.752). The sharp opposition is AE17 against Exc at F010 t64 (−0.989), F010 t251 (−1.000), and F012 t64 (−1.000). That opposition is what drives `γ*` below 0.05.

## 5. Database-level conflicts

Checkpoint chords, which span many AdamW steps, improve DBH76, IP13, MGAE109, and NCCE31 and worsen ABDE4, EA13, PA8, and pTC13 at every F010 and F012 interval that was compared. The sign is the same from B8 to t128 and from t128 to t251.

Adding pTC13, PA8, ABDE4, and EA13 as extra constraints, without removing them from JW, changes the margin as follows.

| Checkpoint | Four tasks | Plus four databases | Eight databases + AE17 + Exc + operator |
|---|---:|---:|---:|
| B8 t128 | 0.801 | 0.065 | 0.012 |
| F010 t64 | 0.043 | 0.043 | 0.031 |
| F010 t128 | 0.110 | 0.110 | 0.087 |
| F010 t251 | 0.0081 | 0.0033 | 0.0033 |
| F012 t64 | 0.0037 | 0.0037 | 0.0031 |
| F012 t128 | 0.077 | 0.077 | 0.056 |
| F012 t251 | 0.867 | 0.320 | 0.113 |

The four small databases are binding at B8 and at F012 t251, where the four-task margin was large. They are not binding at F010 t128, F012 t128, or F012 t64: the physical objectives already set the margin. The strongest trajectory conflict, read from the chord derivatives rather than from a cosine sign alone, is the set {ABDE4, EA13, PA8, pTC13}. At B8 the dual certificate for the extended problem puts the nonzero multipliers on JW, the operator, pTC13, and PA8.

Individual database gradients are normalized separately. They are constraints, not a second copy of JW in the sum.

## 6. Parameter-block feasibility

The verified split is exchange 4,259, correlation 4,547, normalization 640, and no remainder.

The exchange-only margin tracks the full-space margin to about 0.002 at every checkpoint (0.807 versus 0.801 at B8, 0.078 versus 0.077 at F012 t128, 0.868 versus 0.867 at F012 t251). Exchange plus normalization does the same. Correlation alone is feasible as well, and at the conflicted states its margin is larger than the full-space margin (0.051 versus 0.008 at F010 t251, 0.123 versus 0.077 at F012 t128). A small correlation-block JW norm is not incapacity: the correlation block still has a positive four-task margin. No task gradient is a projected zero in any block. Feasibility does not depend on a handful of normalization coordinates.

## 7. Exact AdamW reconstruction

The hypothetical step uses the checkpoint's own moments, bias correction, `ε = 1e-8`, decoupled weight decay 0.01, and learning rate `2.5e-5`. It is not an executed update. The displacement norm is about `6e-4` to `1e-3`.

A negative `g · Δθ` predicts improvement. At B8 the F010 full-gradient AdamW step improves JW (`−5.55e-5`) and worsens AE17 (`+0.237`), Exc (`+0.854`), and the operator (`+1.22e-5`). The F012-weighted gradient at the same B8 moments does the same. The cone margin of that AdamW direction is −0.071 for F010 and −0.059 for F012, against `γ* = 0.801`.

The same pattern holds at F010 t64, F010 t128, F010 t251, F012 t64, and F012 t128: the AdamW cone margin is negative. The only non-negative AdamW margin is F012 t251, and it is 0.027 against `γ* = 0.867`. The step is barely inside the cone and far from `d*`.

Three frozen directions, compared at the exact AdamW norm:

- Plain gradient descent is nearly orthogonal to AdamW at B8 (cosine −0.04) and remains below 0.17 later.
- Diagonal preconditioning with the updated second moment and no first-moment EMA has cosine 0.31 with AdamW at B8, then 0.94 to 0.99 once the run has moved (0.94 at F012 t128).
- At B8 the inherited first moment, trained under the historical relchem loss, supplies the rotation that plain gradient descent and diagonal preconditioning do not. Later, the second moment already points with AdamW, and the first moment is a smaller correction.

The 64-step chords from B8 still lower all four objectives. An unfavorable instantaneous AdamW cosine is not the multi-step displacement. From F012 t128 to t251 the chord lowers JW, AE17, and Exc and raises the operator (`g_op · Δθ = +2.82e-4`).

## 8. Full-gradient/minibatch comparison

Minibatch cursors 0, 63, 64, 127, and 128 were fixed before any minibatch gradient was inspected. Cursor 0 is the first continuation batch. The chemistry piece is the exact eight-reaction mean from the saved Jacobian. AE17, Exc, and the operator are the one-system gradients from the frozen manifest entry.

The cyclic schedule visits each of the 251 reactions eight times. The epoch average of the chemistry estimator is the population JW gradient. A single-batch difference is variance, not a sampling-probability bias.

JW-only cosines with the full JW gradient are 0.43 to 0.93. Composite cosines with the matching full composite are 0.82 to 0.99 at B8, with no task-sign change. At F012 t128 four of the five batches have composite cosine 0.85 to 0.95. Cursor 64 is the exception (cosine 0.012) and flips the F012 task signs. At F010 t128 and F010 t251 several batches have negative composite cosine and flip every task. Those states are exactly the ones whose full gradient is already outside the cone.

## 9. Finite-step validation

The step length is the executed `update_norm` of the next logged update: `6.499e-4` at B8, `9.124e-4` at F010 t128, and `9.341e-4` at F012 t128. Scales 0.25 and 1.0 were applied in memory and the parameter bytes, spin symmetry, and constant passthrough were checked after restoration. Spin and passthrough residuals stayed at 0 except for two F012 evaluations at `1.2e-7`, which is float32 noise. Every evaluation was restored. No checkpoint was written.

At B8, both the common-descent step and the negative F010 gradient lower all four objectives at both scales. At scale 1 the AE17 decrease is only 20% and 9% of the linear prediction, but the sign stays favorable. The exact AdamW step lowers JW and raises AE17, Exc, and the operator, in agreement with the linear prediction to about 0.1%. The database-constrained direction at scale 1 also lowers all four, with a smaller JW and operator change.

At F012 t128 the common-descent step lowers all four at both scales. At scale 1 the changes are JW `−6.05e-5`, AE17 `−0.372`, Exc `−1.594`, and operator `−1.62e-5`. The operator captures 58% of its linear prediction. The negative F012 gradient raises JW (`+1.41e-4`) and the operator (`+3.34e-4`) while lowering the energies. Exact AdamW lowers JW, AE17, and Exc and raises the operator (`+3.41e-5`).

At F010 t128 the common-descent step lowers JW, AE17, and the operator, but Exc rises (`+0.025` at scale 0.25 and `+0.403` at scale 1) against linear predictions of `−0.023` and `−0.092`. The negative F010 gradient and exact AdamW both raise Exc and the operator. Joint finite improvement was not observed for any tested direction at this state.

F010 t251 and F012 t64 were not finite-stepped. Their first-order JW changes at a `1e-3` step are about `6e-6` and `3e-6`.

## 10. Scientific limitations

The full gradients are the saved Jacobians, not a new backward. Scalar parity, the F012 identity, and the published F012 t128 operator cosine are the checks on those caches. One selected finite-difference of a live backward was not repeated in this audit.

Finite steps cover B8, F010 t128, and F012 t128. They do not cover t64 or t251. t251 has no next logged step; its AdamW norm in the counterfactual table is a hypothetical full-gradient step, not an executed update.

The B8 moments belong to the historical relchem run. That is a property of the checkpoint, not a correction applied here.

Chords are not single AdamW steps. Minibatch sign counts are five predeclared batches per state, not a variance estimate over the epoch.

A positive local margin is not a statement about global representability of NN-PBE-Lap. This audit does not repeat the correlation-architecture audit.

## 11. Decision for next experiment

Ranked mechanisms:

1. Weighted-sum misalignment. From t64 onward the F010 and F012 aggregates leave the common-descent cone. At F012 t128 a realistic step along the training gradient raises JW and the operator, while the common-descent step lowers all four. Doubling the operator weight does not repair that.
2. Near-zero margin and curvature. `γ* < 0.01` at F012 t64 and F010 t251. At F010 t128, `γ* = 0.110` is not large enough for Exc to survive one AdamW-scale step.
3. AdamW mismatch at B8. The full gradient is inside the cone and the exact AdamW step is not. Inherited first moments dominate that rotation. The following 64-step chord still lowers all four objectives, so this does not by itself explain the rest of the run.
4. Minibatch variance. It is large where the full gradient is already outside the cone. The sampling contract is balanced across the epoch.
5. Tangent-space limitation. Not supported. The exchange block carries the full-space margin, and the correlation block has its own positive margin.

The next experiment is one arm from B8 t128 whose direction is the normalized four-task common-descent direction defined in this audit, with AdamW moments reinitialized at the start. The reinitialization is part of that arm because the inherited moments were shown to rotate a valid B8 direction out of the cone. Do not search coefficients, do not change the architecture, and do not score directions on Clean28. Success is whether the four training objectives move together. This audit does not start that arm.

## 12. Reproducibility and artifact hashes

Scripts: `tools/lap_f013_geometry_audit.py`, `tools/lap_f013_local_probes.py`, `tools/lap_f013_render.py`. Tests: `tests/test_lap_f013_geometry_audit.py` (5 passed). Optimizer updates: 0.

| Artifact | SHA256 |
|---|---|
| `lap_f013_geometry_audit_protocol.json` | `23a233f19f8268d0b251088a1debd3be5b2d7ea7f03d4b89958a266d2873e7ff` |
| `lap_f013_gradient_receipts.json` | `672dc62546a9b15ac239e9d6fc10f28a8912e136efd149ff5f155409e87d1651` |
| `lap_f013_primary_cosines.csv` | `c0a68f00eafd71ee70c2c8057b7701509e3925f30aaea26039318df1192a1179` |
| `lap_f013_database_cosines.csv` | `41ab891f26ccc7bdb2b804ed7dcde4c4e7f44ab86f2f669cf3aa755188c27207` |
| `lap_f013_feasibility_results.csv` | `77dfc54fdd90c8a9a693c36588ced96a2b390a6b100cce4735d58cd2102d48ea` |
| `lap_f013_feasibility_certificates.json` | `540db58cff85838f2bdaff8c443ac46cf68a5abc279870e7e7be5d3ba51c4b3f` |
| `lap_f013_parameter_block_results.csv` | `8b3df084bf3f4214b49fe0711af045905e35c3677b4d0df39006fcc4b8b20054` |
| `lap_f013_adamw_counterfactuals.csv` | `a703b2cd9d481ab475f0a5fc91045492eab4745a57a2dbd10d7751abd078cc78` |
| `lap_f013_minibatch_full_gradient_comparison.csv` | `c865dc176ec9eed7e756c1aa58b1e2d444740af86bac38aa49a7fe95b3062d33` |
| `lap_f013_finite_step_validation.csv` | `efa0b0d80c13d75004502fefc50a047e37a8537fa5f4a02c3984ba83adb6568f` |
| `lap_f013_cosine_matrices.png` | `3dfed3fd08492db53e6b7707a8db31ede5d7e96a1a29776daf38cf1321083629` |
| `lap_f013_margins.png` | `7d947d1aeda5cde62adfcde9ca342512352921bd0b7c65c4b6560beac1c53de1` |
| `lap_f013_database_feasibility.png` | `9c910df9c118fe3da35935f83df6a3ff0aa80bd553e5e92a51caff4ff89d3a04` |
| `lap_f013_block_feasibility.png` | `4b007f0a406f681fb4ac53c7bb2f26ef31f533f43a473fe5afa6f54aa117f9f2` |
| `lap_f013_full_vs_adamw.png` | `d5d38c1cbc3e19ac78a96b1773f085615922a5aced8525717f7509f896486785` |
| `lap_f013_minibatch_angles.png` | `ab85999b2d3ce576329e7aec3f737884de32a20f37b77583993f321cb0b594e8` |
| `lap_f013_finite_step.png` | `0cd24744ce8572e00d2606d419972047e9008d29673b7f701df9f57cac3b9d88` |
