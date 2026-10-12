# F014 — Fixed nonlinear scalarization

## 1. Decision

**E — FIXED NONLINEAR SCALARIZATION NO-GO.** No training arm is justified.

FNL1 is a fixed scalar potential. At B8 t128 its gradient is the sum of the four unit task gradients, lies inside the common-descent cone (worst-task margin 0.687), and a finite step of historical AdamW length lowers JW, AE17, Exc, and the operator. That is not enough. At the five checkpoints where F013 found the linear sums in trouble, FNL1 is inside the cone only at F012 t251. At F010 t128 and F012 t128 the finite step moves at least two of the four objectives the wrong way. ABDE4, PA8, and pTC13 rise under the B8 FNL1 step. The naive minibatch product is a different vector from the declared full gradient.

## 2. Definition and provenance

\[
L_{\mathrm{FNL1}}(\theta)=\frac12\sum_i\frac{L_i(\theta)^2}{b_i s_i},\qquad
\nabla L_{\mathrm{FNL1}}=\sum_i\frac{L_i(\theta)}{b_i s_i}\nabla L_i(\theta).
\]

\(b_i\) and \(s_i\) are the B8 t128 losses and gradient norms from F013. They were checked against the saved Jacobians to \(1\mathrm{e}{-12}\). Contribution norms at B8 are 1.000 for all four tasks. A torch autodiff fixture matches the product rule. The B8 AdamW moment hash is `4659e23c…1a40d0` and the step counter is 128. No coefficient was changed after these measurements. Optimizer updates: 0.

## 3. Full-gradient geometry

Positive descent derivative means \(g_i^\top d_{\mathrm{FNL1}}>0\).

| Checkpoint | Worst margin | F013 \(\gamma^*\) | F010 margin | F012 margin | Cosine with \(d^*\) | Inside cone |
|---|---:|---:|---:|---:|---:|---|
| B8 t128 | 0.6869 | 0.8011 | 0.6513 | 0.6794 | 0.979 | yes |
| F010 t64 | −0.4079 | 0.0431 | −0.1948 | −0.5218 | 0.114 | no |
| F010 t128 | −0.6391 | 0.1097 | −0.3220 | −0.6837 | 0.392 | no |
| F010 t251 | −0.9507 | 0.0081 | −0.9356 | −0.9773 | 0.013 | no |
| F012 t64 | −0.6045 | 0.0037 | −0.2806 | −0.7505 | 0.086 | no |
| F012 t128 | −0.9687 | 0.0766 | −0.9587 | −0.9083 | 0.165 | no |
| F012 t251 | 0.7506 | 0.8670 | 0.7243 | 0.6878 | 0.979 | yes |

Equal contribution norms hold only at B8. Elsewhere the frozen scales reweight by the current loss. At F012 t128 the AE17 contribution norm is 1.533 and the Exc contribution norm is 1.019, against 0.512 for JW and 0.594 for the operator. Those two energy tasks dominate, and the worst-task margin is −0.969. FNL1 does not avoid the F010 t128, F010 t251, F012 t64, or F012 t128 conflicts. At F010 t128 and F012 t128 it is farther outside the cone than the F010 composite.

## 4. Chemistry databases

A negative \(g\cdot u\) predicts improvement for a unit step \(u\).

At B8, the negative FNL1 gradient improves DBH76, EA13, and NCCE31. It worsens ABDE4, IP13, MGAE109, PA8, and pTC13. The finite step agrees: at both 0.25× and 1.0× the reference norm \(6.499\mathrm{e}{-4}\), ABDE4, PA8, and pTC13 increase, and EA13 decreases. Historical-state AdamW at the same scales increases all four of ABDE4, EA13, PA8, and pTC13.

At F010 t128 only DBH76 and NCCE31 are predicted to improve. At F012 t128, PA8 and pTC13 are predicted to improve and ABDE4 and EA13 to worsen, while JW and the operator themselves worsen. FNL1 does not repair the small-database sacrifice. Three of eight databases improving at the start is not a balanced chemistry direction.

## 5. AdamW

At B8 the exact AdamW step that uses the saved moments and the full FNL1 gradient has cone margin +0.091. The finite step lowers all four objectives (JW \(-9.93\mathrm{e}{-5}\), AE17 \(-0.335\), Exc \(-1.205\), operator \(-3.64\mathrm{e}{-5}\) at scale 1). The original moments do not destroy the four-task direction.

The zero-moment counterfactual is also inside the cone at B8 (margin +0.067). Resetting moments is not what makes the B8 step succeed, and it is not authorized as a second arm. At F010 t128 and F012 t128 the historical FNL1 AdamW step stays outside the cone because the Euclidean gradient is already outside it. Exc at F010 t128 is the curvature case: the linear AdamW prediction is \(-0.0030\) and the observed change is \(+0.315\).

## 6. Stochastic estimator

The epoch average of the naive JW product \(\widehat L_J\nabla\widehat L_J\), taken over all 251 frozen batches, has cosine 0.998 with the declared JW term at B8 and relative \(\ell_2\) error 0.35. The direction of that one term is stable. Its scale is not. That is the covariance of a nonlinear product, computed exactly for chemistry. Per-system Exc and operator gradients are not cached, so the same expectation was not available for those tasks.

On the five predeclared batches the full naive FNL1 gradient has cosine 0.85–0.99 with the full gradient at B8, with relative \(\ell_2\) 0.54–0.91 and no task-sign flip. The operator coefficient swings from 0.58 to 3.05 against the full-gradient value 1.63. At F010 t128 three of five batches have negative cosine and flip every task sign. At F012 t128 the cosine stays near 0.98 while the relative \(\ell_2\) reaches 4.9. Five batches do not estimate a bias magnitude. They do show that substituting minibatch losses into FNL1 is not the declared gradient.

A full backward of 251 reactions and 90 mRKS systems is the only evaluation of the declared gradient in this audit. One such forward pass already costs minutes. That is not a practical per-update estimator under the existing training budget. No delayed-loss or EMA surrogate was treated as FNL1.

## 7. Finite steps

Reference norms are the executed next-step norms from F013. Parameters, the checkpoint file, spin symmetry, and PBE passthrough were restored after every perturbation. Spin and passthrough stayed at 0 except two readings of \(1.2\mathrm{e}{-7}\).

At B8, both the negative FNL1 gradient and the historical AdamW displacement lower all four objectives at 0.25× and 1.0×. At scale 1 the negative-gradient AE17 decrease is 0.382 against a linear prediction of 3.21. The sign holds. The three small databases above still rise.

At F010 t128 the negative FNL1 gradient lowers JW and the operator and raises AE17 by 2.83 and Exc by 9.10 at scale 1. The linear Exc prediction is already positive (+0.076); curvature amplifies it. This is the same Exc sensitivity F013 saw, now on a direction that is already outside the cone.

At F012 t128 the negative FNL1 gradient raises JW by \(2.63\mathrm{e}{-4}\) and the operator by \(3.57\mathrm{e}{-4}\) at scale 1, and lowers the two energies. The historical F012 composite did the same kind of split. FNL1 is not a materially better joint direction at that state.

## 8. What this does and does not say

The formula is fixed. The effective weights change only because the losses change. Equal contribution norms are a B8 identity, not a later guarantee. There is no common-descent theorem for this scalarization, and the measurements show none at five of the seven states. A favorable B8 gradient does not imply a favorable direction at a later historical checkpoint. Training losses and database signs are not Clean28 results. An unbiased task gradient does not make the product \(\widehat L\nabla\widehat L\) unbiased.

## 9. Recommendation

Do not train FNL1. Do not retune its squares, scales, or task set. The fixed quadratic penalty does not replace the linear weighted sum: it misses the common-descent cone on the conflicted checkpoints, continues to raise ABDE4, PA8, and pTC13 at B8, and cannot be formed from the current minibatch losses without changing the objective.

## 10. Artifact hashes

| Artifact | SHA256 |
|---|---|
| `lap_f014_fnl1_protocol.json` | `e44c44505d80d62b4da1a4c8644781cb99c56c7394fbab214cc77e4221cb8c9b` |
| `lap_f014_fnl1_reference_receipts.json` | `ffa81fa428bf62b83cc6caa525852c4ca7f52d364123d42331c0268448fcdb3d` |
| `lap_f014_fnl1_gradient_parity.json` | `5e78a18ca3251ae6e7ef9b85c467a8be2cadd66651aab85fcca45a5769ea4f6f` |
| `lap_f014_fnl1_checkpoint_geometry.csv` | `484ba3f4eb5b0b620b5a830649eb98d8e011bc030b502b879de97e3b107374a1` |
| `lap_f014_fnl1_database_directions.csv` | `b8f971e6b33f782a0dc004cdde475321d89b60a304a565b8c3711d4981f02154` |
| `lap_f014_fnl1_adamw_counterfactuals.csv` | `f1b44c600375221d9f6c55ad5bc8cb60adab06fb0c86b852091901fe890a7f73` |
| `lap_f014_fnl1_stochastic_estimator.csv` | `ab8d19fdfaf6d98fd4115df3bd9339683a8cfce418bc4beb721ebb1cee35f406` |
| `lap_f014_fnl1_finite_step.csv` | `c8c7cabe2b67f7fb8c978430afa4929512d8dcfc062631e9f51f5ec9e53934d4` |
| `lap_f014_fnl1_decision.json` | `10ef9fa0ecc0cfb842edd7c9f1c670e0e9239b24206e0be1c04c5dfbe62daad6` |
| `lap_f014_fnl1_contributions.png` | `5d7759e8d9a65d1bf573b9fbffcd118a2b62ca5013297a2567b9e96b1aa0838a` |
| `lap_f014_fnl1_margins.png` | `9bb02dcbb78c7e0e859dddb3ff7d73ca1d36745aab50960a86f7bafab58cc57e` |
| `lap_f014_fnl1_vs_mgda.png` | `67426052c899f4ecfe27cd6d644f07a351b6d6401b00b82d18a60498a409bcdb` |
| `lap_f014_fnl1_databases.png` | `86deaf7a9eec075672ca991965db54651cebfc03a762318426d592dcdda5a879` |
| `lap_f014_fnl1_adamw.png` | `592b66ece7cd16825f94c365ba1d99c5e5b9e439cc7b4f77209152960f86c2bc` |
| `lap_f014_fnl1_minibatch.png` | `b9ff8e3d141660a0d59aeca2373ce93aa864c7fc583e107ae4b62a06bf5be36b` |
| `lap_f014_fnl1_finite_step.png` | `da24a3115a23c7ea96a03611e8e039e19a2adec6f5b3ee38e8664bdb17e27832` |
