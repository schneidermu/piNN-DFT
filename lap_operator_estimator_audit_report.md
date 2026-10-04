# Operator estimator and aggregate-response audit

aggregate operator finite-step response/globalization failure, with secondary sampled-to-aggregate directional transfer errors.

## Identity and scope

Branch `lap_full_vxc`; starting/source commit `0dcba084c347a65c32c8872e910fccd49cecb2a6`. No production source or CLI changes, optimizer updates, new trajectory, Diet/SCF/Slurm work, or full90 cache generation. The historical `0dcba08` pilot remains **FAILED global operator guard**.

Predopt SHA-256 `ed4ba8231d93c376ce5aa8fc81e0670f6a5a44c95c2a552651b12d6a1d4b63f8`. Final pilot SHA-256 `c11aa31188b21e44711d82d364e2e8214eef4fce5b0e5ef1790ff312d35fb2bb`. All exact paths, central/cache/source hashes, per-record identities and external arrays are bound in protocol/results.

Panel order: H2, HLi, BH, BeH2, H2O, CH4, N2, CO, CH2O, C2H2_iso2, H4Si, AlBeH, ClH, ClHS, HPSi_iso2.

Unchanged local CUDA/F32 model and stored inputs; existing F64 AO/reference algebra. Gradients are existing F32 parameter derivatives widened after autograd, with F64 means/dots/norms. No source-precision upgrade. Same chunks: points256, AO4096; seed41. One-system graphs are released before the next system; no 15-system graph stack. Three fresh processes: five full-panel repeats in the first, one in each other, totaling seven repetitions per state. E was evaluated on the same systems as a matched control.

## Reproducibility

The reproducibility scale is the maximum pairwise range within/across processes at either state, with no invented floor. Zero measured range and nonzero drift give an infinite measured drift/noise ratio; this is not a proof that every possible execution is exact. Historical median means the median of individual system ratios; raw median losses are also retained in JSON.

| State | Metric | Observed drift | Repeat std | Repeat max range | Drift/noise |
|---|---|---:|---:|---:|---:|
| predopt | op mean | - | 0 | 0 | - |
| predopt | op median ratio | - | 0 | 0 | - |
| theta5 | op mean | 2.79048215587e-06 | 0 | 0 | infinity |
| theta5 | op median ratio | 4.61677988226e-05 | 0 | 0 | infinity |

The drift is resolved against measured repeatability. Summary variance is computed after centering on the first value, avoiding spurious rounding variance for identical numbers; the original unshifted summary and tooling are preserved externally. No model evaluations were rerun. This excludes a numerical tie from repeated evaluations; it does **not** exclude deterministic precision effects in parameter-induced responses.

## Individual-to-aggregate gradients

Exact predopt aggregate norms: operator `0.420425579405`, E_xc `18711.0099166`. All gradients are finite; exact F64 mean reconstruction is verified at all six states.

| Task | median cos | p10 cos | min cos | positive-dot fraction | median norm ratio |
|---|---:|---:|---:|---:|---:|---:|
| E_xc singleton vs full15 | 0.999519261856 | 0.996753589408 | 0.986004304501 | 1 | 0.785965462149 |
| operator singleton vs full15 | 0.950197169054 | 0.919545472268 | 0.897203817621 | 1 | 0.973885745838 |

E_xc has closer angular alignment. Operator singleton alignment is nevertheless positive for every system, with moderate norm tails. Static cosine fidelity alone does not guarantee population descent for a combined PCD direction near an objective constraint boundary.

## Exact historical replay

All five sampled scalar losses and realized displacement norms reproduce the pinned pilot. The final stored parameters and EMA match exactly. Directions and recorded t=1 displacements are reconstructed only in disposable scratch models; no optimizer, Armijo search, resampling or new trajectory is executed. Frozen checkpoint files and production source hashes remain unchanged.

Sign convention: negative gradient dot with realized Delta predicts descent. Actual deltas are after minus before, using the same immutable15 mean.

| step | system | sampled op dot | full15 op dot | actual full15 op delta | sampled E dot | full15 E dot | actual full15 E delta |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | BeH2 | -5.68390795209e-08 | -1.34419445867e-07 | -6.48121668292e-07 | -0.00397503614012 | -0.0214029710505 | -0.0207422862018 |
| 1 | H2 | -7.64287609249e-07 | 5.51446933142e-08 | 3.2230687784e-07 | -0.000406529597539 | -0.00852671675248 | -0.00823091509957 |
| 2 | BH | -4.13445543245e-07 | 3.58214219903e-09 | -1.81917645947e-08 | -0.00474511396809 | -0.0340053600986 | -0.0347156583386 |
| 3 | CH4 | -2.69176281512e-07 | -4.15071360493e-07 | 6.585529706e-07 | -0.00944390793704 | -0.0351040528462 | -0.0345651374903 |
| 4 | HPSi_iso2 | -2.85326091395e-07 | -3.73637054878e-07 | 2.47593574031e-06 | -0.0561405710144 | -0.0196663869525 | -0.0217339009806 |

Sampled-descent/aggregate-ascent: operator 2/5, E 0/5. Aggregate first-order/actual sign agreement: operator 2/5, E 5/5.

Operator net first-order prediction `-8.64401025724e-07` is descent, but actual net change `2.79048215587e-06` is ascent. Response remainder `3.65488318159e-06` dominates the net drift. Steps3/4 predict aggregate descent yet actually ascend; step4 is the largest positive actual contribution. Sampling-transfer errors therefore exist but do not explain the failed guard by themselves. Step2's positive aggregate dot is only3.58e-9: the strict-sign count is exact for these stored gradients, but its derivative-precision robustness is unvalidated. Scalar repeatability does not establish gradient precision. E has consistent descent signs throughout. No claim of a quantitatively accurate linear model is made.

## Operator estimator comparison

For uniform m-subset without replacement, P(s in batch)=m/15, so E[(1/m)sum_s_in_batch L_s]=(1/15)sum_s L_s; same for gradients. O1 enumerates all15 singletons. O3/O5 each use32 predeclared uniform distinct-system draws; literal SHA-derived seeds and identities are in protocol. No candidate was chosen to explain the old trajectory.

Selection rules were pinned before results: median cosine>0, p10 cosine>0, positive-dot fraction>=0.90, and no gross norm tail. Gross tail was prospectively operationalized as p90 or maximum norm ratio exceeding10 times the corresponding next-larger candidate. Choose the cheapest qualifying measured candidate.

| Estimator | Cost seconds | median cos | p10 cos | min cos | positive-dot fraction | median norm ratio | qualified |
|---|---:|---:|---:|---:|---:|---:|---|
| O1 | 1.68331999998 | 0.950197169054 | 0.919545472268 | 0.897203817621 | 1 | 0.973885745838 | yes |
| O3 | 5.03142879999 | 0.98805072937 | 0.955782208665 | 0.938610570538 | 1 | 1.0333634864 | yes |
| O5 | 12.4801899 | 0.994563167372 | 0.965357566423 | 0.954711925889 | 1 | 1.00544204408 | yes |
| O15 | 37.8971062 | 1 | 1 | 1 | 1 | 1 | yes |

Minimum qualified estimator: **O1**, retained. At theta5, median/p10/min cosines are 0.950319223184/0.919484667693/0.897216761463, positive-dot fraction 1; the same criteria pass. Larger batches improve geometric approximation but are not selected by the prospectively supplied rules.

Cost measurements are direct sequential operator-gradient evaluations, including load/hash time, with first pinned draw for O3/O5. Peak Torch allocation across benchmarks: 6895051776 bytes. Per-candidate allocated/reserved values are in JSON. Torch allocation is not resident VRAM; cache transition and warm-filesystem effects are included. One measurement per representative batch is not a performance ranking with uncertainty, and no V100 claim is made.

## Classification and single next experiment

Primary observed failure: aggregate operator finite-step response/globalization. Secondary effect: sampled-to-population directional transfer on steps1/2. This is not evidence that singleton gradients are uniformly poor, nor that O15 is required. O1 passes the prescribed statistical criteria. Those criteria do not guarantee a population-descent PCD direction, but raising thresholds after seeing this trajectory would be post-hoc.

Underlying arithmetic versus genuine nonlinear curvature remains unseparated. Fixed-state repeatability does not validate tiny directional changes. No trust-region, new MOO, tau change or batch change is justified here.

One read-only aggregate-operator globalization audit at historical theta4 along the recorded step4 direction: prospectively compare actual full15 responses with aggregate autograd predictions, checking arithmetic/realized-step consistency before attributing any mismatch to genuine curvature. No new training or estimator change.

## Validation and review

Production source changes:0. External tooling: py_compile; deterministic artifact checks validate input hashes, exact replay, all finite gradients, exact means, sampled gradient identity, endpoint scalar identity, sign tables and predeclared draws. No committed Python tooling, so Ruff is not applicable. Production suites need not be rerun. git diff --check passed for staged artifacts. Independent review **PASS; no blocker**, receipt SHA-256 `27e832387596d8b88fb36493d350f68faf1c733b3254944ad2df5fa52d6729ff`. Review retains the arithmetic-versus-curvature limitation and warns that step2's tiny dot is precision-sensitive. No experiments were rerun by the reviewer.

## Twelve answers

1. Yes; all seven evaluations per state across three processes are identical, while mean and median-ratio drift are nonzero.
2. 0.42042557940451358.
3. Median/p10/min cosines 0.950197169054/0.919545472268/0.897203817621; all15 positive.
4. E is closer: median/p10/min 0.999519261856/0.996753589408/0.986004304501; all15 positive.
5. 2/5.
6. 0/5.
7. Only 2/5; net prediction is descent but actual net response is ascent.
8. Aggregate finite-step response/globalization dominates the net failure; sampling-transfer errors also occur. Genuine curvature versus deterministic arithmetic is not identified.
9. O1 under the prospectively specified criteria; retain singleton.
10. Yes, at theta5 using the same15 enumerated singletons and criteria.
11. No changes to tau, PCD, task order or priority.
12. One read-only aggregate-operator globalization audit at theta4 along recorded step4; check arithmetic/realized-direction consistency before any curvature conclusion or new training.

Singleton operator gradients represent the aggregate objective, but finite-step aggregate behavior breaks the first-order prediction; the next experiment is an aggregate-operator globalization audit.
