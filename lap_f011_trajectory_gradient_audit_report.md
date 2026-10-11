# F011 trajectory and gradient geometry audit

Diagnostic only. Optimizer updates: 0. Clean28 was not recomputed. The five Clean28 numbers below are the frozen receipts.

`c = 9.705596345235`. F010 is defined as the same vector as JW, so their cosine is 1 by construction. F011 is the chemistry gradient after `c` and before `lambda_chem`.

## Scalars

| Checkpoint | JW | MAE | Relchem | F011 | AE17 | Exc | Operator | Composite | Clean28 receipt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| t0 | 0.156415 | 8.536568 | 1.195939 | 0.823824 | 3.275465 | 14.679045 | 0.030933 | 0.024801 | 9.096700111658 |
| t64 | 0.148275 | 8.273987 | 1.265879 | 0.740539 | 3.849199 | 8.360124 | 0.029226 | 0.022744 | 8.782020261263 |
| t128 | 0.145491 | 8.155332 | 1.292944 | 0.721445 | 1.234652 | 6.980487 | 0.029371 | 0.022313 | 8.800214350716 |
| t192 | 0.144079 | 8.140767 | 1.293710 | 0.710459 | 2.231732 | 11.120048 | 0.029379 | 0.022242 | 8.857739596910 |
| t251 | 0.143531 | 8.271702 | 1.271155 | 0.698142 | 1.378884 | 7.837101 | 0.029235 | 0.021891 | 8.938697016838 |

## Interval changes

| Interval | ΔJW | ΔMAE | Δrelchem | ΔF011 | ΔAE17 | ΔExc | Δoperator | ΔClean28 receipt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| t0->t64 | -0.008140 | -0.262581 | +0.069939 | -0.083285 | +0.573734 | -6.318921 | -0.001707 | -0.314680 |
| t64->t128 | -0.002784 | -0.118655 | +0.027065 | -0.019094 | -2.614547 | -1.379637 | +0.000145 | +0.018194 |
| t128->t192 | -0.001412 | -0.014565 | +0.000766 | -0.010985 | +0.997079 | +4.139561 | +0.000008 | +0.057525 |
| t192->t251 | -0.000548 | +0.130935 | -0.022555 | -0.012318 | -0.852848 | -3.282947 | -0.000144 | +0.080957 |

## Gradient alignment

| Checkpoint | JW/total | JW/operator | JW/F011 | JW/old relchem | JW/AE17 | JW/Exc | F010/F011 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| t0 | 0.660116 | 0.752422 | 0.980410 | -0.662494 | 0.283546 | 0.289205 | 0.980410 |
| t64 | -0.034853 | 0.387324 | 0.968977 | -0.811451 | -0.357878 | -0.390514 | 0.968977 |
| t128 | 0.504659 | 0.421085 | 0.504250 | -0.717418 | -0.520154 | 0.540249 | 0.504250 |
| t192 | 0.552486 | 0.477936 | 0.507890 | -0.690326 | 0.563385 | 0.556559 | 0.507890 |
| t251 | 0.543573 | 0.569405 | 0.508477 | -0.790333 | 0.526957 | 0.509546 | 0.508477 |

## Directional JW derivatives

Negative means descent on that coefficient-weighted task locally lowers JW.

| Checkpoint | F011 | AE17 | Exc | Operator | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| t0 | -0.380203 | -0.123349 | -0.132768 | -0.234374 | -0.870694 |
| t64 | -0.141765 | 0.092238 | 0.121877 | -0.058681 | 0.013669 |
| t128 | -0.034091 | 0.049015 | -0.087485 | -0.050126 | -0.122687 |
| t192 | -0.034797 | -0.121985 | -0.115432 | -0.058406 | -0.330620 |
| t251 | -0.036457 | -0.077668 | -0.112773 | -0.080637 | -0.307534 |

## Checkpoint displacements

A favorable JW descent has negative cosine between `g_JW` and the displacement. The linear estimate is `dot(g_JW, Δtheta)`, not an AdamW prediction. The trapezoid is an approximation, not a line integral. Momentum cosines in the displacement CSV are the raw first moment at the start of the interval, not the bias-corrected AdamW direction and not the next minibatch.

| Interval | ||Δθ|| | cos(JW, Δθ) | cos(total, Δθ) | Observed ΔJW | Linear | Remainder | Trapezoid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| t0->t64 | 0.046832 | -0.124137 | -0.079511 | -0.008140 | -0.008788 | +0.000648 | -0.007596 |
| t64->t128 | 0.031427 | -0.086045 | -0.065927 | -0.002784 | -0.003072 | +0.000289 | -0.002641 |
| t128->t192 | 0.025240 | -0.066825 | -0.024270 | -0.001412 | -0.001412 | +0.000000 | -0.001413 |
| t192->t251 | 0.019774 | -0.037984 | -0.028157 | -0.000548 | -0.000629 | +0.000081 | -0.000563 |

## Next logged minibatch

The displacement below is the analytical AdamW step for the historical next eight-reaction minibatch. It uses that minibatch gradient, not the full J251 gradient. It was not applied. A positive cosine with a gradient means the step points the same way as that gradient.

- `t0` manifest 0: cos(JW, step) `-0.050351`, cos(total, step) `0.053103`.
- `t64` manifest 64: cos(JW, step) `-0.091445`, cos(total, step) `-0.102967`.
- `t128` manifest 128: cos(JW, step) `-0.101676`, cos(total, step) `-0.087105`.
- `t192` manifest 192: cos(JW, step) `0.015768`, cos(total, step) `0.223665`.
- `t251`: no following update.

## Databases

JW contributions use the global `1/251` denominator. The full eight-database table is the database CSV.

### t0

| Database | JW contribution | Mean absolute residual | Dot with total | Dot with operator |
| --- | ---: | ---: | ---: | ---: |
| ABDE4 | 0.000512 | 3.000499 | -0.008732 | -0.008887 |
| DBH76 | 0.116345 | 7.857969 | 0.782086 | 0.569636 |
| MGAE109 | 0.010953 | 13.334307 | -0.082146 | -0.027691 |
| NCCE31 | 0.022617 | 0.717074 | 0.278412 | 0.250008 |
| PA8 | 0.000335 | 1.714795 | -0.025473 | -0.020329 |
| pTC13 | 0.002037 | 6.639701 | -0.061247 | -0.049317 |

### t64

| Database | JW contribution | Mean absolute residual | Dot with total | Dot with operator |
| --- | ---: | ---: | ---: | ---: |
| ABDE4 | 0.000626 | 3.668263 | -0.002506 | 0.000254 |
| DBH76 | 0.110785 | 7.482502 | -0.081671 | 0.235243 |
| MGAE109 | 0.010415 | 12.680332 | 0.074678 | -0.080352 |
| NCCE31 | 0.019544 | 0.619623 | 0.005427 | 0.038353 |
| PA8 | 0.000566 | 2.902719 | 0.003270 | -0.010126 |
| pTC13 | 0.002484 | 8.096072 | 0.005882 | -0.016669 |

### t128

| Database | JW contribution | Mean absolute residual | Dot with total | Dot with operator |
| --- | ---: | ---: | ---: | ---: |
| ABDE4 | 0.000664 | 3.894157 | 0.001183 | 0.002396 |
| DBH76 | 0.108389 | 7.320653 | 0.124186 | 0.150727 |
| MGAE109 | 0.010188 | 12.404100 | -0.057113 | -0.086565 |
| NCCE31 | 0.018787 | 0.595640 | 0.034455 | 0.044640 |
| PA8 | 0.000658 | 3.369322 | -0.004701 | -0.005566 |
| pTC13 | 0.002614 | 8.519426 | -0.007864 | -0.009414 |

### t192

| Database | JW contribution | Mean absolute residual | Dot with total | Dot with operator |
| --- | ---: | ---: | ---: | ---: |
| ABDE4 | 0.000666 | 3.907111 | 0.001008 | 0.001734 |
| DBH76 | 0.107218 | 7.241556 | 0.360354 | 0.177866 |
| MGAE109 | 0.010190 | 12.406043 | -0.139031 | -0.085642 |
| NCCE31 | 0.018407 | 0.583594 | 0.081834 | 0.046132 |
| PA8 | 0.000677 | 3.469702 | -0.016085 | -0.007148 |
| pTC13 | 0.002632 | 8.578165 | -0.026807 | -0.012037 |

### t251

| Database | JW contribution | Mean absolute residual | Dot with total | Dot with operator |
| --- | ---: | ---: | ---: | ---: |
| ABDE4 | 0.000639 | 3.743506 | -0.000103 | -0.000774 |
| DBH76 | 0.106996 | 7.226538 | 0.368711 | 0.291932 |
| MGAE109 | 0.010504 | 12.788812 | -0.113874 | -0.079726 |
| NCCE31 | 0.018067 | 0.572796 | 0.070726 | 0.049760 |
| PA8 | 0.000650 | 3.331872 | -0.015595 | -0.012765 |
| pTC13 | 0.002579 | 8.406233 | -0.025868 | -0.021147 |

## Reactions after t64

Largest JW-contribution increases:

- NCCE31 4: `+0.000192`
- NCCE31 2: `+0.000085`
- NCCE31 29: `+0.000075`
- NCCE31 1: `+0.000075`
- EA13 7: `+0.000069`
- NCCE31 27: `+0.000066`
- EA13 11: `+0.000062`
- DBH76 69: `+0.000062`

Largest JW-contribution decreases:

- NCCE31 8: `-0.000225`
- NCCE31 10: `-0.000195`
- NCCE31 9: `-0.000192`
- NCCE31 11: `-0.000189`
- NCCE31 18: `-0.000159`
- DBH76 47: `-0.000158`
- NCCE31 16: `-0.000147`
- NCCE31 5: `-0.000126`

## Blocks

| Checkpoint | JW exchange share | F011 exchange share | Operator exchange share | Exchange JW/total cosine |
| --- | ---: | ---: | ---: | ---: |
| t0 | 0.977647 | 0.978668 | 0.976036 | 0.679224 |
| t64 | 0.975788 | 0.977223 | 0.933006 | -0.037232 |
| t128 | 0.965367 | 0.933130 | 0.929000 | 0.509614 |
| t192 | 0.965407 | 0.933202 | 0.929384 | 0.559930 |
| t251 | 0.972309 | 0.935678 | 0.936750 | 0.553450 |

## Finite difference

t251 F011 symmetric difference at target 1e-3: relative gap `4.547e-05`.

## Patterns

Task cancellation ratios were t0 0.172207, t64 0.547975, t128 0.450994, t192 0.026389, t251 0.027934. Pattern A is the t64 snapshot: cos(JW, total) falls from 0.660116 to -0.034853, and the total JW directional derivative there is 0.013669. AE17 and Exc supply the positive parts (0.092238 and 0.121877); F011 and the operator stay negative. The cosine is positive again at t128, t192, and t251, and all four task derivatives are negative at t192 and t251. That is not a sign change that persists.

Pattern B is the scalar split after t64. JW decreases on every later interval while each frozen Clean28 receipt increases. MAE also rises on t192->t251 while JW falls. No Clean28 gradient was measured, so this is a discrepancy between recorded scalars, not a demonstrated reversal of a Clean28 gradient.

Pattern C is not supported for the four checkpoint chords. Each chord has negative cosine with g_JW, and the observed JW change has the same sign as `dot(g_JW, Δtheta)`. The t64 instantaneous total derivative is slightly positive, while the chord that leaves t64 still lowers JW. The raw first moment at t64 has cosine 0.888120 with g_JW, so the retained moment still points along JW there. Individual AdamW steps were not reconstructed. The t192 next-minibatch step, which was not applied, has cosine 0.015768 with g_JW.

Pattern D is supported. DBH76 and NCCE31 account for the JW decrease and keep improving after t64. pTC13, PA8, and ABDE4 worsen, mostly on t0->t64, and remain above their t64 JW contributions at t251. Their own JW-gradient pieces have negative cosine with the full g_JW at every checkpoint.

## Answers

Q1. JW changed by -0.008140, -0.002784, -0.001412, -0.000548 across the four intervals. It decreased on every interval, including after t64. The decrements become smaller. On t192->t251, MAE changes by +0.130935 while JW changes by -0.000548.
Q2. cos(JW, total) was 0.660116, -0.034853, 0.504659, 0.552486, 0.543573. It is negative only at t64, by a small amount, and returns to about 0.50–0.55 afterward. A positive cosine is not a claim that every AdamW update decreased JW.
Q3. cos(JW, operator) was 0.752422, 0.387324, 0.421085, 0.477936, 0.569405. It drops at t64 and then recovers, and it stays positive. The operator directional contribution was -0.234374, -0.058681, -0.050126, -0.058406, -0.080637, negative at every checkpoint.
Q4. Post-t64 JW contribution changes: DBH76 -0.003790, pTC13 +0.000095, PA8 +0.000084, ABDE4 +0.000013, MGAE109 +0.000089, NCCE31 -0.001477. DBH76 and NCCE31 continue down. pTC13, PA8, and ABDE4 stay above their t64 contributions. The same three already have negative cosine between their JW-gradient piece and the full g_JW at t0.
Q5. Checkpoint-chord cosines with JW were t0->t64 -0.124137, t64->t128 -0.086045, t128->t192 -0.066825, t192->t251 -0.037984. All four are negative, so each measured chord points toward JW descent. The linear estimates match the observed JW changes in sign. These chords are not the individual AdamW steps. Raw first-moment cosines with JW at the four interval starts were -0.026497, 0.888120, 0.259012, -0.098223.
Q6. JW kept falling while the frozen Clean28 receipt rose after t64: 9.096700111658, 8.782020261263, 8.800214350716, 8.857739596910, 8.938697016838. The full training gradient remained a separate object from that scalar. No Clean28 gradient was computed, so this audit does not identify a Clean28 cause.
Q7. cos(F010, F011) was 0.980410, 0.968977, 0.504250, 0.507890, 0.508477. F010 and F011 are nearly the same direction at t0 and t64. From t128 onward the cosine is about 0.51, so replacing F011 with F010 would materially change the chemistry gradient. The F011 directional contribution stays negative at every checkpoint (-0.380203, -0.141765, -0.034091, -0.034797, -0.036457), so the measured chemistry term does not itself turn against JW.
Q8. Two training-only hypotheses, neither executed and neither selected with Clean28. First, replace the legacy chemistry factors with the uniform JW gradient (F010) under the same B8 t128 controls, because cos(F010, F011) falls from 0.980410 to 0.504250 and stays near 0.508477. Second, do not spend the next arm on DBH76 alone or on another external 1/A_d factor: DBH76 remains aligned with g_JW and its JW contribution falls through t251, and 1/A_d is already inside F011. pTC13, PA8, ABDE4 keep a JW-gradient piece opposed to the full g_JW, but their combined t251 JW contribution is 0.003868. That is a smaller training question than the F010 substitution. The t64 cosine dip is not a reason to stop: the chords through and after it still lower JW.

## Runtime

- Seconds: `5374.306`
- Peak allocated bytes: `14091718144`

## Git

Provenance is the commit that contains this report. The diagnostic started from `f49df9fd10571902b841b0d1c1d885ffcd8fbe50`.
