# Frozen cursor10 chemistry gradient-geometry audit

Starting commit `8408b0ffebda952fefd247a0ad1d3b2419975ae9`, branch `lap_full_vxc`. Reproduction PASS. Classification: **mixed: structural full251 conflict with database-specific sampling amplification**. Final artifact commit is the commit containing this report.

## Identity and objective contract

Frozen checkpoint `C:\Dev\readWFN_share_ms\lap_25_runs_20261005\run\checkpoint_10.pt`; SHA256 `0dd853b5967a6e254da8218d3f5b89b63dcd10902364dc5ed3788eb7e27107a4`. Cursor10, retained EMA10, F32 main, no optimizer/scheduler. Manifest canonical `2e87f7e79a66b425628bdf59a821f785e05059f193a492d4a4f3130d7411e9f3`; file `eb64fb2ba9a98eead1dc518a1b0853eeb90d66dcb92c117dbfba23964e6afcfc`; protocol file `8bd4a039ad9403e2c532a731503afd03df5ff6f7be3460012201aed97f23fa67`. Production and precision-critical sources are unchanged and bound in protocol/results.

All gradients use the existing factories and production operator. Chemistry uses actual differentiable F64 shadow parameters synchronized from stored F32 model values and the same F32-loaded source values widened to F64; this does not restore native source precision. No full-model-double operator shortcut. J_rel is the mean of 251 existing database-weighted singleton chemistry losses, not the separate database-RMSE monitoring statistic. Per-database full gradients are means of the same singleton losses. Global J_rel is their n_database/251 combination. The original FCHEM factor and R2 sampling weight are recorded separately; removing a positive factor for reporting changes norms, not cosines.

No update, trial, optimizer, acceptance or checkpoint-save API is called. The PCD normalization update is computed on a copied retained state only. Scratch shadows change; frozen model/checkpoint/EMA/cursor do not. Caller RNG is restored on the success path using the existing helper; the primary receipt is code-path evidence rather than an independently captured post-restore equality check. A separate supplemental check verifies exact capture/restore equality. Frozen checkpoint RNG bytes are unchanged.

## Exact rejection reproduction

Existing isolated-gradient computation and PCD reproduce losses, raw norms/cosines, active set, multipliers, EMA state/scales, normalized norms/Gram and pre/post-rescale norms exactly. Slopes use the production per-tensor FP64 dot helper with a predeclared 1e-12 relative tolerance (1e-8 denominator floor). PCD QP feasible; active exc; proposed EMA11 is diagnostic only.

| Task | Loss | Raw norm | Requested alpha0 slope |
|---|---|---|---|
| relchem | 1.05627053654 | 25.0684385609 | 3.13346504336e-05 |
| ae17 | 24.8296066577 | 6992.40227426 | -0.0299424113989 |
| exc | 212.144294589 | 62100.8459977 | -0.264796392562 |
| op | 0.0304574095098 | 0.688329196827 | -5.05671511598e-06 |

## R2 components

The individual before-FCHEM column is the gradient of the unweighted singleton residual loss. Singleton gradients then include the fixed FCHEM factor; R2 adds n_database/251. Norms below separate both layers. Projection fractions are signed dot contributions to the final R2 squared norm, not independent variance shares.

| DB/reaction/variant | R2 weight | FCHEM factor | Pre-FCHEM norm | Weighted norm | cos R2 | cos AE | cos E | cos op | cos S1 | R2 projection fraction |
|---|---|---|---|---|---|---|---|---|---|---|
| ('ABDE4', 2, 'level2') | 0.0159362549801 | 2.24490567019 | 270.407843766 | 9.67394584437 | -0.846966515959 | 0.779836260717 | 0.784285340022 | 0.635081581221 | 0.740425323315 | -0.326845574665 |
| ('DBH76', 24, 'level3_gauss_chebyshev') | 0.278884462151 | 0.11815293001 | 20.445569279 | 0.673702287341 | -0.246890795874 | 0.258525214887 | 0.256452013974 | 0.131585708699 | 0.217943856872 | -0.0066350719651 |
| ('EA13', 0, 'level3') | 0.0438247011952 | 0.690740206212 | 349.930012534 | 10.5929004749 | 0.975824821597 | -0.934861490407 | -0.937494080214 | -0.836565719001 | -0.910939194379 | 0.412343799993 |
| ('IP13', 7, 'level2_delley') | 0.0517928286853 | 0.690740206212 | 565.057996172 | 20.2151697146 | 0.985201172335 | -0.956294251988 | -0.958724383718 | -0.873386250203 | -0.937468924592 | 0.794465473127 |
| ('MGAE109', 17, 'level2_gauss_chebyshev') | 0.414342629482 | 0.0174023695364 | 150.157203961 | 1.08271505895 | -0.758407844833 | 0.770266452721 | 0.769404081064 | 0.694141218949 | 0.750702872658 | -0.0327559130749 |
| ('NCCE31', 14, 'level3_delley') | 0.111553784861 | 2.89665247766 | 13.1976157528 | 4.26457919662 | 0.771896433149 | -0.688002668838 | -0.688567589425 | -0.529001203171 | -0.641395622105 | 0.131313063746 |
| ('PA8', 3, 'level2_mura') | 0.0318725099602 | 1.12245283509 | 87.9634735977 | 3.14692750038 | 0.10005279743 | -0.236029570862 | -0.22629188984 | -0.345657470771 | -0.269012017681 | 0.0125599725311 |
| ('pTC13', 2, 'level2_delley') | 0.0517928286853 | 0.690740206212 | 31.9978483417 | 1.14473547691 | 0.340620847422 | -0.242612591928 | -0.242530685858 | -0.0173463610046 | -0.170697661063 | 0.0155542503086 |

## Full-database reference

| DB | n | Full norm | sample/full cosine | cos AE | cos E | cos op |
|---|---|---|---|---|---|---|
| ABDE4 | 4 | 288.273742681 | 0.998477281663 | 0.764918144062 | 0.769067426373 | 0.616470690396 |
| DBH76 | 70 | 6.17892739936 | -0.111892821026 | 0.0381629670903 | 0.035876450423 | 0.24423410629 |
| EA13 | 11 | 46.6268870828 | -0.984669029137 | 0.937558537371 | 0.940240436055 | 0.854530891182 |
| IP13 | 13 | 143.568975629 | 0.988881383421 | -0.984800700725 | -0.985608209745 | -0.916800411493 |
| MGAE109 | 104 | 11.5721418721 | -0.914798990551 | -0.877579180982 | -0.878982248679 | -0.762332597939 |
| NCCE31 | 28 | 48.6586210375 | 0.979845830958 | -0.559382753762 | -0.561248503997 | -0.375786814016 |
| PA8 | 8 | 31.4822664447 | 0.794587765837 | 0.396686749458 | 0.405631515929 | 0.281920263718 |
| pTC13 | 13 | 62.654890405 | -0.296089624822 | -0.362437967185 | -0.354815790259 | -0.501711061714 |

## Full251 versus R2

| Quantity | Value |
|---|---|
| loss | 1.23881802847 |
| R2_loss | 1.05627053654 |
| norm | 11.0439585709 |
| R2_norm | 25.0684385609 |
| R2_full_cosine | 0.898364805936 |
| norm_ratio | 2.26987799708 |
| R2_projection_coefficient_on_full | 2.03917850635 |
| projected_norm | 22.5206029428 |
| residual_norm | 11.0113148611 |
| cos_ae17 | -0.891845720508 |
| cos_exc | -0.888997792907 |
| cos_op | -0.802098804928 |
| R2_vector_cancellation_ratio | 2.0262401039 |
| full_database_vector_cancellation_ratio | 2.74066958523 |
| R2_reconstruction_relative_error | 8.85825007553e-18 |

Full251 loss agrees with freshly monitored cursor10 J_rel within the predeclared absolute1e-13 arithmetic check. R2 reconstruction from row gradients agrees within predeclared relative1e-12. Each cached reaction/group is loaded through existing hash-checking store helpers. The supplemental verification independently binds the Minnesota store manifest, reaction dispersions, mRKS dispersions, central corpus manifest and AO-cache manifest to their exact native cursor10 protocol hashes; all match. Original checkpoint byte SHA is asserted unchanged. No source data were regenerated.

## Predefined 32-draw distribution

All32 original draws are reused and independently regenerated with the original SHA-derived seed and group ordering. No new or selected draw. Percentile is 100*count(alignment strictly below actual)/32; ascending rank includes the actual draw. This is the empirical frozen32 distribution, not a population confidence statement.

| Metric | min | p10 | p25 | median | p75 | p90 | max |
|---|---|---|---|---|---|---|---|
| norm | 7.22838794473 | 8.70675249107 | 12.4465124364 | 18.5446549654 | 25.8083080409 | 31.6193941258 | 42.1310894386 |
| cos_full251 | -0.762914565167 | -0.43982248161 | 0.538969267986 | 0.846347151007 | 0.938644442871 | 0.967768835912 | 0.986912515503 |
| cos_ae17 | -0.984792708475 | -0.942509665598 | -0.926086960608 | -0.723396446127 | -0.323765466903 | 0.690939753752 | 0.949580712978 |
| cos_exc | -0.984838456231 | -0.942772765153 | -0.922973135084 | -0.7201347786 | -0.325154168064 | 0.694208389296 | 0.953547156255 |
| cos_op | -0.949757190976 | -0.90155426183 | -0.838316368459 | -0.662183720533 | -0.346028441087 | 0.562519742782 | 0.918465206061 |
| cos_S1_equal | -0.974919505371 | -0.937426655818 | -0.904366963214 | -0.719011788356 | -0.333271012142 | 0.654703868647 | 0.946458588958 |

Actual draw10 full251 cosine 0.898364805936; ascending rank 19/32; strict empirical percentile 56.25%. Fractions full cosine >0 / >0.5 / >0.8: 0.8125 / 0.75 / 0.5625.

## Secondary consensus

| Definition | norm | cos R2 | cos full251 |
|---|---|---|---|
| S1_equal | 4.90365709714 | -0.967565966697 | -0.867787748595 |
| S2_secondary_MGDA | 1.00000126484 | -0.98278320982 | -0.891845720508 |
| S3_PCD_correction | 0.992451787679 | -0.983101305798 | -0.888997792907 |

S2 convex weights (AE,E,op): [1.0, 0.0, 0.0]; KKT residual 0. S1 sums actual EMA-normalized secondary gradients; S3 isolates current canonical PCD correction. Per-database consensus cosines are in JSON.

## Frozen direction Arena (no parameter steps)

d denotes a gradient-space combination; -d is the descent step. Positive raw g dot d means predicted decrease. PCD costs and projections are in normalized space, before the canonical positive magnitude rescale. Comparing B costs to A is valid because their anchor is identical. C/D use a different secondary anchor; their costs are not comparable preference scores. Zero primary product is non-ascent, not strict descent or an authorized production update.

| Candidate | Feasible/common | QP cost | relative B cost increase | R2 dot d | full251 dot d | AE dot d | E dot d | op dot d |
|---|---|---|---|---|---|---|---|---|
| A_canonical_PCD | True | 0.492480275434 | — | -0.34336391687 | -0.199622306086 | 328.107814072 | 2901.62887618 | 0.0554112933985 |
| B_primary_nonascent | True | 0.495279428824 | 0.00568378781855 | -1.74860126378e-14 | -0.0895281222033 | 328.714015074 | 2901.62887618 | 0.0657478795402 |
| B_primary_progress_normalized | True | 0.508666030828 | 0.0328657942284 | 0.482308529456 | 0.0651164438718 | 329.565519556 | 2901.62887618 | 0.0802672376225 |
| B_primary_progress_literal_raw_LHS | True | 0.495589325726 | 0.0063130453082 | 0.0185082484724 | -0.0835937464272 | 328.746690956 | 2901.62887618 | 0.0663050496308 |
| C_secondary_anchor_R2_safety | True | 11.2556701846 | — | -2.57571741713e-14 | 0.0779552665513 | 1558.80817151 | 13779.9385943 | 0.340375518654 |
| D_secondary_anchor_full251_safety | True | 9.05393182025 | — | -23.1073300679 | -6.43929354283e-15 | 7626.98070541 | 68518.8436262 | 0.973929053012 |
| E_four_task_MGDA_R2 | True | — | — | 0.215717566049 | 0.0369316892264 | 57.883095233 | 501.101413332 | 0.0177351118353 |
| E_four_task_MGDA_full251 | True | — | — | -1.05811860229 | 0.597226446385 | 378.129114946 | 3443.31806027 | 0.0564925416727 |

B normalized-progress imposes tilde_g_primary dot d >= tau||tilde_g_primary||². A separate literal raw-LHS variant is reported because the prompt displays a raw LHS with normalized RHS; these scale-dependent conditions are not identical. Existing secondary progress constraints are unchanged in all projection diagnostics. Full251 safety in D uses unit gradient for a scale-invariant zero margin. E uses consistently normalized gradients; the full251 version uses its unit gradient because no full251 EMA exists. All feasibility/margins/KKT residues are recorded; these are diagnostic QPs, not production changes.

## Interpretation and limits

Classification: **mixed: structural full251 conflict with database-specific sampling amplification**. Single next experiment: **Frozen cursor10 Armijo qualification of the explicit normalized-primary-progress constrained projection, before any training continuation.**.

The frozen failure is mixed, with a structural full-objective conflict rather than an unusually bad R2/full251 alignment: R2/full251 cosine0.898365 is above the draw median0.846347, at rank19/32 (56.25th strict percentile). Full251 itself is strongly anti-aligned with all three local/absolute secondary gradients. This is descriptive evidence at this frozen state, not a population inference.

Database structure matters. Full IP13, MGAE109, NCCE31 and pTC13 oppose the secondaries; ABDE4, EA13 and PA8 agree, while DBH76 is near-orthogonal to AE/E. Sampled EA13 reverses its full-database direction (cos−0.984669), amplifying anti-alignment; sampled MGAE109 reverses in the other direction (cos−0.914799). Weighted IP13 supplies0.794465 of the R2 squared-norm projection, EA13 supplies0.412344, NCCE31 supplies0.131313, while ABDE4 contributes−0.326846. These signed fractions sum to1; they are not variance proportions.

Canonical PCD ascends both sampled R2 and full251 at first order. Adding normalized primary progress remains feasible, costs3.28658% more than canonical nearest-primary projection, and gives positive raw products for R2, full251, AE17, E and operator. Thus the existing secondary margins do not force primary ascent: the canonical nearest-direction optimum chooses it. Both normalized MGDA diagnostics exhibit strict common descent for their specified objective sets, so this state is not shown Pareto-critical. Full251-based MGDA does not protect R2; safety for one chemistry gradient cannot be silently substituted for the other. Non-ascent-only B protects R2 numerically at its boundary but still ascends full251, whereas strict normalized-progress B protects both here. None of these gradient-level facts proves finite-step acceptance. The single next frozen Armijo qualification must use the existing raw-primary magnitude rescale before applying the unchanged step size and acceptance rule; no such trial was run in this task.

Positive objective-wise normalization cannot cause raw gradient anti-alignment. It changes canonical PCD progress margins and nearest-direction choice. AE17/E/operator agreement is not three independent votes: targets differ, derivatives share one learned functional and E/operator share the same sampled ClHS density. Primitive grid points and AO matrix elements are correlated contributions, not independent systems. This audit measures R2 diversity across32 frozen draws; it does not establish AE/E/operator estimator variance or universal lower variance. Reaction-energy signed combinations can cancel derivatives, but cancellation alone does not establish which gradient is preferable; full251 is the chemistry reference here.

## Validation and review

External diagnostic tooling only; production source/test changes zero. Five deterministic QP/simplex fixtures PASS; exact production reproduction PASS; py_compile, source/data/hash/state checks and git diff --check are recorded. Independent review covers Ponytail, Pocock, MOO signs/constraints/normalization, and DFT/statistical interpretation. Independent review PASS with no blocking findings; external receipt `C:\Dev\readWFN_share_ms\lap_cursor10_geometry_runs_20261005\review.md` SHA256 `850d81acd82cbecf40ed977f3e25892833efa88a0042247a109fb4d52c25892f`. Source/input hash gaps were closed by the supplemental verification. Primary RNG evidence is qualified explicitly rather than represented as an unperformed equality check. Large gradient arrays remain outside Git and SHA-bound.

No training update was applied, no cursor advanced, no model/EMA/RNG state was changed, and no full90/SCF/Diet/Slurm/100-update run was launched.
