# Historical step4 operator globalization audit

The operator guard failure is caused primarily by F32 operator forward/objective arithmetic; the next experiment is a minimal operator-precision-boundary repair.

## Identity and scope

Branch `lap_full_vxc`; exact source/start HEAD `839e03129c226006a37f6ec99dd39653a709f8d9`. Disposable external diagnostic: `C:\Dev\readWFN_share_ms\lap_operator_globalization_runs_20261004`. No production source changes, solver calls, accepted updates, optimizer/EMA/sampling progression, training, SCF, Diet, Slurm or full90 generation. Step3 skipped because step4 is decisive.

The 15-system panel and chunks remain fixed: points256, AO4096. All norms, dots, aggregate means and diagnostics use F64. Matched-F64 means **F64 arithmetic on exact F32-loaded source/checkpoint values widened**, not native-double source recovery. Exact data/source hashes and external per-system curves/gradients are in protocol/results.

theta4 file SHA `16b63fccd3dd70c11afdaf37f498486b9a6c641f9c8e75e3f7fb7b496e4c0c97`; theta5 `13c9d8ad8df1d8379a2e1e267f1026f0f0a20b0c3bf57ae1f2a21f362893cc9d`; step4 `1264b2c580d10aa39e1e47abcd0dc9af688af28d2a6feb3aeae63db7375ca645`; requested delta `62248f1646fec93b3aad39446a48505c1d10f2e0cb124e62f6d664535d3f56a0`.

Requested delta recovered by replaying recorded normalization coefficients and final rescale against the exact historical relchem gradient and stored E/operator gradients. The recorded AE17 coefficient is zero. No PCD QP or EMA recomputation. Requested norm matches the record; t=1 produces theta5 and recorded realized displacement bitwise. Production aggregate scalar and gradient reproduce historical theta4 exactly.

## Arithmetic contract

Inspection corrects an imprecise earlier description: within each AO chunk, `_assemble_from_local_partials` uses the dtype of `phi`; with the existing F32 cache/model this is F32. `make_mrks_objective_factories` converts the completed chunk matrix to F64 before accumulating it. Reference/overlap and final operator-loss algebra are F64. Thus F64 final accumulation cannot undo local or within-chunk F32 arithmetic errors.

Both shadows use the same `LapEnergy`, `local_partials`, `assemble_rks_operator`, `operator_loss`, reductions, source values and reference. The F64 shadow has differentiable F64 leaves and recomputed F64 autograd gradients. The rounded-candidate control widens the exact production F32 candidate parameters. No post-forward output casting substitutes for double computation.

## Gradient precision

| Quantity | Value |
| --- | --- |
| L32 | 0.0367989332571 |
| L64 | 0.0367976806843 |
| g32_norm | 0.420229056639 |
| g64_norm | 0.42027846889 |
| cosine | 0.999999978345 |
| relative_L2_g32_vs_g64 | 0.000239015455018 |
| g32_requested_dot | -3.74320223583e-07 |
| g64_requested_dot | -3.74410167649e-07 |
| g32_realized_dot | -3.73637054878e-07 |
| g64_realized_dot | -3.73727052937e-07 |
| historical_F32_gradient_bitwise | yes |
| historical_F32_scalar_exact | yes |

## Three-way line scan

| t | req norm | real norm | cos | zero frac | ΔL32 | pred32-real | ΔL64 continuous | pred64 | ΔL64 on F32 candidate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0 | — | 1 | 0 | 0 | 0 | 0 | 0 |
| 1 | 5.19702439734e-06 | 5.19648813662e-06 | 0.998398737311 | 0.612746135931 | 2.47593574031e-06 | -3.73637054878e-07 | -3.7432235174e-07 | -3.74410167649e-07 | -3.73639393578e-07 |
| 0.5 | 2.59851219867e-06 | 2.60304184381e-06 | 0.996073019355 | 0.730362058014 | -2.28000127932e-06 | -1.87008192176e-07 | -1.87183126665e-07 | -1.87205083824e-07 | -1.87031087659e-07 |
| 0.25 | 1.29925609933e-06 | 1.30463256926e-06 | 0.989834375362 | 0.82807537582 | -3.39598231677e-06 | -9.47010774569e-08 | -9.35970536214e-08 | -9.36025419122e-08 | -9.47180779226e-08 |
| 0.125 | 6.49628049667e-07 | 6.47663059374e-07 | 0.98186366523 | 0.897946220622 | -1.93406383329e-06 | -4.55703737337e-08 | -4.67998945153e-08 | -4.68012709561e-08 | -4.55804408592e-08 |
| 0.0625 | 3.24814024834e-07 | 3.19805402483e-07 | 0.969432254298 | 0.944103324158 | -5.32814612181e-07 | -2.34704840044e-08 | -2.34002876659e-08 | -2.34006354781e-08 | -2.34760558909e-08 |
| 0.03125 | 1.62407012417e-07 | 1.58302611646e-07 | 0.958444709295 | 0.970781283083 | -3.07571856456e-07 | -1.17177395848e-08 | -1.17002286262e-08 | -1.1700317739e-08 | -1.17204410413e-08 |
| 0.015625 | 8.12035062084e-08 | 7.75278877058e-08 | 0.95059749888 | 0.983061613381 | -2.1453959432e-06 | -5.04487889792e-09 | -5.85012534249e-09 | -5.85015886951e-09 | -5.04622999636e-09 |
| 0.0078125 | 4.06017531042e-08 | 3.79342228258e-08 | 0.942396065149 | 0.989625238196 | -7.55782720782e-07 | -2.93334694407e-09 | -2.92507581351e-09 | -2.92507943476e-09 | -2.93419046005e-09 |
| 0.00390625 | 2.03008765521e-08 | 2.14077843763e-08 | 0.930339867979 | 0.9941774296 | -1.49954173721e-06 | -1.82338375793e-09 | -1.46253854166e-09 | -1.46253971738e-09 | -1.82379993746e-09 |
| 0.001953125 | 1.01504382761e-08 | 9.3104344159e-09 | 0.869997748349 | 0.996400592844 | -2.47890607902e-07 | -8.00404922424e-11 | -7.3126375788e-10 | -7.31269858689e-10 | -7.99889113279e-11 |
| 0.0009765625 | 5.07521913803e-09 | 4.50419540827e-09 | 0.694802073405 | 0.998412026254 | -7.44869058593e-07 | -6.05515598734e-10 | -3.65629138077e-10 | -3.65634929345e-10 | -6.05619263072e-10 |
| 0.00048828125 | 2.53760956901e-09 | 1.40750299703e-09 | 0.505534034979 | 0.999153080669 | -4.35046508457e-07 | 1.19449523453e-10 | -1.82807546878e-10 | -1.82817464672e-10 | 1.1945987255e-10 |
| 0.000244140625 | 1.26880478451e-09 | 1.04434867699e-09 | 0.47283422253 | 0.999258945585 | -5.78149273525e-07 | 1.13984126844e-10 | -9.13983472239e-11 | -9.14087323362e-11 | 1.14004312934e-10 |
| 0.0001220703125 | 6.34402392253e-10 | 5.25936495062e-11 | 0.0660281861761 | 0.999682405251 | -1.01128954447e-08 | 5.1844949763e-13 | -4.57075974292e-11 | -4.57043661681e-11 | 5.27175525455e-13 |
| 6.103515625e-05 | 3.17201196127e-10 | 1.46651614128e-11 | 0.0629343239901 | 0.999788270167 | -2.84321169386e-09 | 1.7197859638e-13 | -2.28423391313e-11 | -2.2852183084e-11 | 1.75977288297e-13 |
| 3.0517578125e-05 | 1.58600598063e-10 | 1.45803092363e-11 | 0.0629341390244 | 0.999788270167 | -2.84306851672e-09 | 1.72741250067e-13 | -1.14147510888e-11 | -1.1426091542e-11 | 1.82388826264e-13 |

All predeclared rows retained. Zero means unchanged coordinates; a null cosine means a zero displacement. Changed counts, maximum coordinate discrepancy, relative distortion and requested predictions remain in JSON.

## Derivative consistency

| t | ΔL32/t | F32 error vs realized/t | ΔL64/t | F64 error | F64 relative error |
| --- | --- | --- | --- | --- | --- |
| 1 | 2.47593574031e-06 | 2.84957279519e-06 | -3.7432235174e-07 | 8.7815908961e-11 | 0.000234544669319 |
| 0.5 | -4.56000255865e-06 | -4.1859861743e-06 | -3.74366253331e-07 | 4.39143182649e-11 | 0.000117289331485 |
| 0.25 | -1.35839292671e-05 | -1.32051249573e-05 | -3.74388214486e-07 | 2.19531631482e-11 | 5.86339929978e-05 |
| 0.125 | -1.54725106663e-05 | -1.51079476764e-05 | -3.74399156122e-07 | 1.10115266625e-11 | 2.94103302046e-05 |
| 0.0625 | -8.52503379489e-06 | -8.14950605082e-06 | -3.74404602654e-07 | 5.5649945483e-12 | 1.4863363843e-05 |
| 0.03125 | -9.84229940659e-06 | -9.46733173988e-06 | -3.74407316039e-07 | 2.85160947612e-12 | 7.61627146513e-06 |
| 0.015625 | -0.000137305340365 | -0.000136982468115 | -3.74408021919e-07 | 2.14572967706e-12 | 5.73095995372e-06 |
| 0.0078125 | -9.67401882601e-05 | -9.63647198513e-05 | -3.74409704129e-07 | 4.63519750149e-13 | 1.23799990011e-06 |
| 0.00390625 | -0.000383882684725 | -0.000383415898483 | -3.74409866666e-07 | 3.00983099344e-13 | 8.03886019532e-07 |
| 0.001953125 | -0.000126919991246 | -0.000126879010514 | -3.74407044035e-07 | 3.12361411715e-12 | 8.34275985817e-06 |
| 0.0009765625 | -0.000762745915999 | -0.000762125868026 | -3.74404237391e-07 | 5.9302579234e-12 | 1.58389339708e-05 |
| 0.00048828125 | -0.00089097524932 | -0.000891219881944 | -3.74389856006e-07 | 2.03116428952e-11 | 5.42497096773e-05 |
| 0.000244140625 | -0.00236809942436 | -0.00236856630334 | -3.74367630229e-07 | 4.25374196698e-11 | 0.000113611817587 |
| 0.0001220703125 | -8.28448394827e-05 | -8.2849086621e-05 | -3.7443663814e-07 | -2.64704908273e-11 | 7.06991773046e-05 |
| 6.103515625e-05 | -4.65831803922e-05 | -4.65859980895e-05 | -3.74248884327e-07 | 1.6128332167e-10 | 0.000430766404349 |
| 3.0517578125e-05 | -9.3161669156e-05 | -9.31673295412e-05 | -3.74038563677e-07 | 3.71603971455e-10 | 0.000992505021401 |

Prospective numerical floor `1.42108547152e-14` =64 eps64 max(1,|L0|). Relative derivative tolerance1e-3; norms below1e-30 treated as zero. F64 qualifying t values: 1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125, 0.0009765625, 0.00048828125, 0.000244140625, 0.0001220703125, 6.103515625e-05, 3.0517578125e-05. F64 relative error improves from2.35e-4 at t=1 to8.04e-7 at t=1/256, then roundoff grows toward9.93e-4 at the smallest t. This is a resolved approach region, not monotonic convergence down to zero. F32 requested and realized derivative errors are both retained in JSON. Sign alone is not used as a consistency test.

## Attribution

At t=1: production ΔL `2.47593574031e-06`, matched-F64 continuous `-3.7432235174e-07`, matched-F64 on exact rounded candidate `-3.73639393578e-07`. Forward/arithmetic response difference is `2.84957513389e-06`; parameter-rounding response difference is `6.82958162246e-10`. These are controlled response differences, not a decomposition into independently linear error sources.

The historical full step descends in both double controls. Production ascent therefore does not establish finite-step overshoot. Near-zero F32 parameter rounding additionally distorts the requested direction; that secondary effect is explicit in the curves and does not explain t=1 ascent. Gradient agreement and double directional convergence exclude a dominant gradient-direction mismatch for this frozen step.

Attribution is for this reconstructed historical step and fixed15 panel; not established for other states or future updates.

## Quadratic diagnostic

Fixed a=`-3.74410167649e-07`, estimated b=`1.87923617777e-10` from valid t=1/32 and 1/64; t=1/128 and1/256 second-order remainders are below the predeclared floor and excluded. Formal t_min=`1992.35291486`, t_cross=`3984.70582972`. Per-point b, residuals and roundoff-floor flags are in JSON. These formal extrapolations are not validated crossings, step choices or production hyperparameters. Small second-order remainders can be near the declared roundoff floor; the observed full grid governs attribution.

## Retrospective aggregate operator Armijo

c=1e-4; production uses g32 dotted with the realized displacement, F64 uses g64 dotted with the continuous displacement. Strict actual decrease is required separately. No point is accepted or saved as training. At t=1/2048, production reports decrease and passes this scalar check while the double rounded-candidate response is positive; an uncorrected F32 guard can therefore also accept a precision artifact.

| t | F32 strict decrease | F32 Armijo | F64 strict decrease | F64 Armijo |
| --- | --- | --- | --- | --- |
| 0 | no | no | no | no |
| 1 | no | no | yes | yes |
| 0.5 | yes | yes | yes | yes |
| 0.25 | yes | yes | yes | yes |
| 0.125 | yes | yes | yes | yes |
| 0.0625 | yes | yes | yes | yes |
| 0.03125 | yes | yes | yes | yes |
| 0.015625 | yes | yes | yes | yes |
| 0.0078125 | yes | yes | yes | yes |
| 0.00390625 | yes | yes | yes | yes |
| 0.001953125 | yes | yes | yes | yes |
| 0.0009765625 | yes | yes | yes | yes |
| 0.00048828125 | yes | yes | yes | yes |
| 0.000244140625 | yes | yes | yes | yes |
| 0.0001220703125 | yes | yes | yes | yes |
| 6.103515625e-05 | yes | yes | yes | yes |
| 3.0517578125e-05 | yes | yes | yes | yes |

Largest passing t: production `0.5`, matched-F64 `1`. Production aggregate15 Armijo would reject historical t=1; a consistent F64 operator oracle would accept it. Adding the uncorrected F32 aggregate oracle is therefore not the next justified repair.

## Per-system t=1 response

| System | F32 ΔL | F64 ΔL | F64 rounded ΔL | g32·realized Δ | g64·requested Δ |
| --- | --- | --- | --- | --- | --- |
| H2 | 1.84274326354e-09 | -1.94258285013e-07 | -1.94007440307e-07 | -1.94020039577e-07 | -1.94271749165e-07 |
| HLi | -2.00893918959e-07 | -7.60424963622e-08 | -7.59150431129e-08 | -7.59285424527e-08 | -7.60550479622e-08 |
| BH | 3.08188392017e-06 | -3.92568868121e-07 | -3.91930907381e-07 | -3.91982927619e-07 | -3.92606282714e-07 |
| BeH2 | 5.2879694528e-07 | -1.46016949878e-07 | -1.457845481e-07 | -1.45820154359e-07 | -1.4603824247e-07 |
| H2O | 9.91100772867e-06 | -5.36244032671e-07 | -5.34941956176e-07 | -5.34599079441e-07 | -5.36367950905e-07 |
| CH4 | -3.03005682796e-06 | -3.05653865343e-07 | -3.05016052096e-07 | -3.05034625889e-07 | -3.05703964534e-07 |
| N2 | 5.40013076779e-06 | -6.77997623544e-07 | -6.76304189789e-07 | -6.76690566622e-07 | -6.78138099486e-07 |
| CO | -2.03918545495e-06 | -5.79831081364e-07 | -5.78182736667e-07 | -5.78051170391e-07 | -5.79976474896e-07 |
| CH2O | -6.09136598516e-07 | -3.90040964526e-07 | -3.88787817565e-07 | -3.88930251319e-07 | -3.90151981288e-07 |
| C2H2_iso2 | -1.44884094381e-06 | -3.76616439597e-07 | -3.75642057814e-07 | -3.75556840548e-07 | -3.7669123584e-07 |
| H4Si | 2.03594370383e-06 | -2.67526514046e-07 | -2.67338517095e-07 | -2.6680503798e-07 | -2.67582169743e-07 |
| AlBeH | 7.27602166016e-06 | -1.31585915539e-07 | -1.31392990665e-07 | -1.32018087862e-07 | -1.31633140017e-07 |
| ClH | 2.74808045672e-05 | -7.31299540639e-07 | -7.3077136465e-07 | -7.31279805772e-07 | -7.3148436765e-07 |
| ClHS | -8.61760768389e-06 | -5.24205530865e-07 | -5.23817469584e-07 | -5.22512601948e-07 | -5.24389255394e-07 |
| HPSi_iso2 | -2.6316745034e-06 | -2.84947168649e-07 | -2.84757812635e-07 | -2.85326091395e-07 | -2.85062552669e-07 |

## Required answers

1. F32 and matched-F64 aggregate gradients agree closely; exact norms/cosine/relative error are above.
2. Yes. Matched-F64 small-step responses converge to recomputed autograd directional derivatives within the declared tolerance.
3. No. Production small-step responses do not consistently track either requested or realized gradient predictions.
4. Rounding materially distorts sufficiently tiny steps; at historical t=1 it is small and preserves F64 descent.
5. Matched-F64 already shows actual descent at t=1; no reduction is needed within this grid.
6. No matched-F64 crossing into ascent is observed within the prescribed continuous grid.
7. No. Production signs fluctuate rather than reproducing a F64 curvature crossing.
8. `1`.
9. `0.5`.
10. Dominant historical t=1 mechanism: F32 operator forward/objective arithmetic. Small-t quantization is secondary.
11. Yes in production arithmetic; no with the consistent F64 scalar/gradient pair.
12. No. O1 stays unchanged.
13. No. PCD, task order and tau stay unchanged.
14. One frozen diagnostic isolating the minimum operator-forward precision boundary needed for consistent scalar and derivative response. No new pilot yet.

## Validation and review

Driver execution `1044.4685408` seconds. Exact historical F32 scalar/gradient and t=1 parameter/displacement identities pass. Caller model/RNG restoration recorded; immutable input/source hashes rechecked. External tools compiled; git diff --check passes. Full historical suites are not rerun because production source is unchanged. Independent review: PASS; SHA e9f78e393c5b8eb19e71dfbbc2d4da3120ed7b0d935c68ed9aae16598ea4bd14. No trained checkpoints or large arrays are committed.

The operator guard failure is caused primarily by F32 operator forward/objective arithmetic; the next experiment is a minimal operator-precision-boundary repair.
