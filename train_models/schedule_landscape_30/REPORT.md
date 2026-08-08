# Retrospective schedule analysis: 30 fixed-architecture runs

## Scope and validity

The primary dataset contains **30** 500-epoch replays: one constrained baseline, 20 earlier schedule variants, and nine completed points from the simple schedule sweep (S9 is absent). Architecture, data, seed, final-epoch checkpoint selection, and PBE-D3(BJ) evaluation protocol are fixed. Consequently, matched schedule contrasts are informative for this seed, but they are not estimates of seed variance.

The reported WTMAD value is the archived 30-reaction interface panel. It must not be described as full GMTKN55 WTMAD-2 until that has been independently verified. Failed or unconverged SCFs are retained as a reliability flag; a lower score accompanied by a failed case is not treated as clean evidence of improvement.

## External landscape

| Run | Family | WTMAD | avRANE | SCF |
|---|---|---:|---:|---:|
| e3 | h9 | 6.282 | 0.465224 | 83/84 |
| s4 | simple | 6.397 | 0.467974 | 83/84 |
| s3 | simple | 6.427 | 0.467841 | 84/84 |
| e1 | h9 | 6.584 | 0.473987 | 81/84 |
| s1 | simple | 6.658 | 0.468720 | 83/84 |
| h9 | h9 | 6.670 | 0.475769 | 84/84 |
| s6 | simple | 6.676 | 0.466666 | 82/84 |
| s2 | simple | 6.696 | 0.468549 | 83/84 |
| s5 | simple | 6.714 | 0.465649 | 82/84 |
| e5 | h9 | 6.798 | 0.487409 | 83/84 |
| e4 | h9 | 6.807 | 0.477192 | 83/84 |
| s8 | simple | 6.816 | 0.466390 | 84/84 |
| s7 | simple | 6.876 | 0.464989 | 84/84 |
| h6 | repair | 7.058 | 0.473063 | 84/84 |
| m2 | micro | 7.069 | 0.477476 | 82/84 |
| m3 | micro | 7.074 | 0.478796 | 83/84 |
| baseline | reference | 7.097 | 0.479944 | NA |
| m4 | micro | 7.109 | 0.476344 | 84/84 |
| m1 | micro | 7.139 | 0.475936 | 84/84 |
| h10 | repair | 7.204 | 0.480112 | 83/84 |
| s10 | simple | 7.209 | 0.468813 | 83/84 |
| x2 | h9 | 7.239 | 0.478459 | 83/84 |
| e2 | h9 | 7.244 | 0.478376 | 84/84 |
| m5 | micro | 7.289 | 0.476575 | 84/84 |
| h8 | repair | 7.300 | 0.464929 | 84/84 |
| x1 | h9 | 7.342 | 0.479355 | 83/84 |
| x3 | h9 | 7.371 | 0.479145 | 83/84 |
| x4 | h9 | 7.448 | 0.479636 | 84/84 |
| x5 | h9 | 7.532 | 0.479215 | 83/84 |
| h7 | repair | 7.586 | 0.467420 | 83/84 |

The observed Pareto set is: **e3** (6.282, 0.465224), **s7** (6.876, 0.464989), **h8** (7.300, 0.464929).

No completed schedule reaches either WTMAD < 6 or avRANE < 0.45. E3 remains the best energy point (6.282) and S7/H8 define the best density region (~0.465), so the remaining gaps are 0.282 WTMAD units and about 0.015 avRANE. The WTMAD target is a local extrapolation from the observed gain; the avRANE target is at the edge of what schedule-only variation has demonstrated.

## Clean simple-sweep evidence

| Run | Guard g | Repair end | WTMAD | avRANE | SCF |
|---|---:|---:|---:|---:|---:|
| s1 | 0.00 | 420 | 6.658 | 0.468720 | 83/84 |
| s2 | 0.00 | 440 | 6.696 | 0.468549 | 83/84 |
| s3 | 0.25 | 420 | 6.427 | 0.467841 | 84/84 |
| s4 | 0.25 | 440 | 6.397 | 0.467974 | 83/84 |
| s5 | 0.50 | 420 | 6.714 | 0.465649 | 82/84 |
| s6 | 0.50 | 440 | 6.676 | 0.466666 | 82/84 |
| s7 | 0.75 | 420 | 6.876 | 0.464989 | 84/84 |
| s8 | 0.75 | 440 | 6.816 | 0.466390 | 84/84 |
| s10 | 1.00 | 440 | 7.209 | 0.468813 | 83/84 |

The guard response is reproducibly U-shaped in WTMAD across both repair endpoints. Moving from g=0 to g=0.25 improves WTMAD by 0.231 (end 420) and 0.299 (end 440); moving from g=0.25 to g=0.50 loses 0.287 and 0.279. Thus a moderate early homotopy is supported, while stronger density protection over-regularizes energies.

Repair endpoint has only a small conditional effect: end 440 changes WTMAD by +0.038, -0.030, -0.038, and -0.060 at g=0, 0.25, 0.50, and 0.75. The corresponding avRANE changes are -0.000172, +0.000133, +0.001018, and +0.001401. These effects are much smaller than the guard curvature and are not monotonic. S3 is therefore the cleaner simple candidate: its 0.030 WTMAD disadvantage to S4 is coupled to 84/84 rather than 83/84 converged SCFs.

A descriptive quadratic fit with a repair-end indicator places the WTMAD minimum at g=0.270 and the avRANE minimum at g=0.580. These are not precise optima, but they quantify a real conflict: the energy-favorable pre-repair path is less density-protective than the avRANE-favorable path.

## Matched contrast ledger

| Contrast | Isolated or near-isolated change | dWTMAD | davRANE |
|---|---|---:|---:|
| baseline -> m1 | extend sum_repair by 15 epochs | 0.042 | -0.004008 |
| baseline -> m2 | extend fchem_polish by 20 epochs | -0.028 | -0.002468 |
| baseline -> m3 | terminal Vxc 5 -> 7 | -0.023 | -0.001148 |
| baseline -> m4 | drive Vxc 10 -> 12 and terminal 5 -> 7 | 0.012 | -0.003600 |
| baseline -> m5 | shorter early clipped anchor | 0.192 | -0.003369 |
| h9 -> e4 | repair E_xc 3 -> 4 | 0.137 | 0.001423 |
| h9 -> e5 | repair reaction scale 0.75 -> 1.0 | 0.128 | 0.011640 |
| h9 -> e1 | shorter repair / longer final relaxation | -0.086 | -0.001782 |
| h9 -> e2 | longer repair / shorter final relaxation | 0.574 | 0.002607 |
| h9 -> e3 | delay repair onset 40 and end 20 epochs later | -0.388 | -0.010545 |
| e3 -> s4 | replace E3 epochs 1-280 with smooth g=0.25 homotopy; epochs 281-500 identical | 0.115 | 0.002750 |
| s1 -> s2 | repair end 420 -> 440 at guard 0.00 | 0.038 | -0.000172 |
| s3 -> s4 | repair end 420 -> 440 at guard 0.25 | -0.030 | 0.000133 |
| s5 -> s6 | repair end 420 -> 440 at guard 0.50 | -0.038 | 0.001018 |
| s7 -> s8 | repair end 420 -> 440 at guard 0.75 | -0.060 | 0.001401 |

The strongest defensible conclusions are negative constraints on schedule design: repair E_xc=4 is worse than 3; repair reaction scale 1.0 is worse than 0.75; soft/clipped terminal exits are consistently poor; and extending a strong repair until epoch 460 is harmful.

The decisive new contrast is **E3 versus S4**. Both use exactly the same repair on epochs 281-440 and the same ordinary consolidation on 441-500. Replacing E3's clipped/stepped first 280 epochs with the smooth g=0.25 homotopy worsens WTMAD by 0.115 and avRANE by 0.002750. Therefore E3's remaining advantage over the best simple schedule is an early-path effect, not a repair-timing effect.

## Training trajectories

Terminal internal losses are not valid selection metrics. E2 and X1-X5 obtain train Fchem near 29-30 and val Vxc near 0.12, yet have WTMAD 7.24-7.53. E3 deliberately finishes at higher internal losses and gives the best external energy score. The schedule is selecting a basin and then relaxing objective bias, not minimizing any one logged loss to completion.

E3 versus S4 makes this especially clear. S4 has lower Vxc and E_xc losses through almost the entire trajectory, including the shared repair and consolidation, but its external metrics are worse. During the shared repair, E3 reaches about 1.0 lower mean train Fchem while retaining roughly 0.007 higher train Vxc and 0.84 higher train E_xc. In the final 60 epochs E3 still has lower train Fchem but worse validation Fchem/Vxc/E_xc. The early schedule therefore changes the representation/basin in a way that the held-out training objectives do not rank correctly.

The largest cross-run rank correlations are listed below as diagnostics, not causal effects, because schedule families are heavily confounded:

| Trajectory feature | target | n | Spearman rho |
|---|---|---:|---:|
| train_vxc_delta_e441_500 | wtmad | 30 | -0.725 |
| train_exc_mean_e161_240 | avrane | 30 | 0.691 |
| train_exc_mean_e73_160 | avrane | 30 | 0.690 |
| train_exc_delta_e161_240 | avrane | 30 | 0.687 |
| train_vxc_delta_e73_160 | avrane | 30 | -0.681 |
| val_vxc_delta_e73_160 | avrane | 30 | -0.678 |
| train_vxc_delta_e241_280 | avrane | 30 | -0.664 |
| val_vxc_delta_e241_280 | avrane | 30 | -0.659 |
| val_vxc_delta_e441_500 | wtmad | 30 | -0.648 |
| val_exc_mean_e73_160 | avrane | 30 | 0.636 |
| train_exc_mean_e241_280 | avrane | 30 | 0.616 |
| train_fchem_delta_e1_72 | avrane | 30 | -0.579 |

A family-held-out response model was intentionally not used to nominate an optimum: with only a few schedule families, continuous controls are aliases for family identity and such a model would extrapolate structure rather than estimate a stable causal surface. The clean factorial and matched contrasts carry more evidential weight.

## WTMAD decomposition

Raw reaction errors are not WTMAD contributions, but they show where schedules differ. The most schedule-sensitive reactions are:

| Reaction | Error range | rho(error, WTMAD) | E3 | S3 | S4 |
|---|---:|---:|---:|---:|---:|
| MB16-43-10 | 15.500 | 0.099 | 12.15 | 12.62 | 9.26 |
| BSR36-31 | 12.310 | 0.564 | 1.13 | 0.63 | 0.09 |
| W4-11-132 | 8.240 | -0.343 | 11.40 | 13.70 | 12.86 |
| SIE4x4-15 | 4.050 | 0.034 | 24.07 | 24.70 | 24.61 |
| DIPCS10-7 | 4.040 | 0.418 | 2.84 | 0.73 | 1.13 |
| FH51-24 | 4.030 | -0.130 | 3.17 | 3.92 | 3.82 |
| W4-11-57 | 3.730 | -0.143 | 4.90 | 5.66 | 4.86 |
| G21EA-25 | 3.160 | -0.629 | 3.05 | 3.48 | 4.18 |
| W4-11-30 | 2.690 | -0.120 | 3.55 | 3.77 | 3.48 |
| DC13-1 | 1.540 | -0.125 | 2.62 | 2.02 | 2.08 |
| BHPERI-11 | 1.440 | 0.462 | 3.28 | 3.08 | 3.00 |
| BHROT27-26 | 1.290 | 0.168 | 0.70 | 0.45 | 0.62 |

E3's advantage is distributed rather than attributable to one reaction. S3/S4 improve some large raw errors but lose on highly influential subsets, which explains why an unweighted mean absolute error can move in the opposite direction from official WTMAD.

Against S4 specifically, E3 is worse on MB16-43, DIPCS10, BSR36, G21EA-14, and DC13, but better on W4-11-132, G21EA-25, FH51-24, SIE4x4, and BH76. This is a chemically structured trade rather than uniform error shrinkage. Any next schedule that merely lowers aggregate Fchem is likely to move toward the wrong side of this trade.

## avRANE decomposition

| Component | Observed range | rho(component, WTMAD) |
|---|---:|---:|
| rho | 0.046821 | -0.023 |
| grad | 0.024185 | 0.325 |
| lapl | 0.052781 | 0.036 |

The systems with the largest schedule-induced mean-component ranges are:

| System | NIAD range | rho(system NIAD, total avRANE) | rho(system NIAD, WTMAD) |
|---|---:|---:|---:|
| F2 | 0.022958 | 0.472 | 0.008 |
| HF | 0.014529 | 0.463 | 0.023 |
| LiF | 0.010854 | 0.676 | 0.136 |
| N2 | 0.009809 | 0.343 | 0.025 |
| CO | 0.009359 | 0.582 | 0.078 |
| BH3 | 0.008760 | 0.794 | 0.522 |
| H2O | 0.007779 | 0.488 | 0.033 |
| Li2 | 0.006543 | 0.865 | 0.625 |
| LiH | 0.005887 | 0.911 | 0.468 |
| H2 | 0.004190 | 0.718 | 0.168 |

The density trade-off is component-specific. Moderate guard improves the energy/rho side, whereas stronger guard tends to improve gradient and Laplacian response while sacrificing WTMAD. This is why a scalar avRANE target alone is insufficient for schedule inference.

E3 versus S4 resolves the component origin of their avRANE difference: S4 is slightly better in rho (0.43005 versus 0.43159), but E3 is better in gradient (0.38987 versus 0.39425) and Laplacian (0.57421 versus 0.57962). E3's early stepped path therefore preserves derivative quality, not pointwise density alone.

## What the 30 runs actually support

1. **A late, finite energy-repair excursion is useful.** It must be followed by ordinary sum-gradient consolidation; leaving the model in a strongly repaired or softly clipped regime is consistently worse.
2. **Repair strength has a narrow useful range.** The best supported values remain Vxc=15, E_xc=3, and reaction scale=0.75. Increasing E_xc or reaction scale hurts both objectives in near-matched comparisons.
3. **The pre-repair path matters, but only moderately.** The simple factorial shows an optimum near guard g=0.25, not at either no guard or strong guard. This establishes curvature rather than a monotonic rule.
4. **Repair endpoint is secondary near the useful region.** At onset=281, changing endpoint 420 -> 440 moves WTMAD by only -0.030 at g=0.25. E3 and S4 prove that E3's advantage is instead created before epoch 281. Repair onset is not independently mapped, but it is no longer the main explanation for E3 versus the simple schedules.
5. **Lower internal losses are often anti-predictive beyond a point.** Driving Fchem/Vxc further selects externally worse models. The relevant signal is the response to transitions and recovery during consolidation, not the terminal minimum.
6. **One part of E3's extra complexity is buying quality.** The conceptually three-stage S3 reaches 6.427/0.46784 and beats most bespoke schedules, but the exactly matched E3-S4 suffix proves that E3's first 280 epochs improve both external metrics. The unresolved task is to compress that early path without deleting its useful mechanism.
7. **More epochs are not the mechanism.** Five secondary 550-epoch phase-extension runs score 7.178-7.323 WTMAD and 0.4788-0.4904 avRANE. They are excluded from the fixed-500 primary analysis, but uniformly show that simply lengthening any baseline phase does not recover E3-like quality.

## Consequence for the next design step

The next ten runs should not be chosen by a global black-box fit or by perturbing every E3 parameter. The primary block should decompose E3's first 280 epochs: early clipping, the 40/20/10 Vxc plateaus, and the transition into repair. All candidates should preserve the now-supported suffix (repair 281-440 at 15/3/0.75, then ordinary 7/1/1 consolidation). A smaller secondary block can test one repair-onset axis and the energy-density compromise near guard g=0.27-0.58. Exact runs should be selected only after ensuring each run removes one early-stage degree of freedom or resolves one remaining confound.

## Files

- `run_summary.csv`: external metrics, terminal metrics, schedule features, and robust trajectory features.
- `matched_contrasts.csv`: predeclared matched and near-matched comparisons.
- `simple_factorial.csv`: clean guard-by-endpoint sweep.
- `schedule_correlations.csv` and `trajectory_correlations.csv`: descriptive associations.
- `reaction_sensitivity.csv`: per-reaction raw-error variation.
- `avrane_components.csv` and `avrane_systems.csv`: density decomposition.
