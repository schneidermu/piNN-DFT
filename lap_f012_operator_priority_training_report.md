# F012 operator priority

**NO-GO — F012 primary Clean28 gate not passed**

The operator coefficient is `0.6719512321409643` = 2 × `0.33597561607048215`. Chemistry, AE17, Exc, AdamW, and the B8 manifest are the F010 settings. `c010` = `9.974823077811354`.

Clean28 is a development panel, not an untouched test set.

## Preflight

- JW dot F012 total: `1.1126650373704932`
- Operator dot F012 total: `0.6259362574788513`
- AE17 dot: `5331.755452589002`, Exc dot: `19238.56551341875`
- cos(JW, F012): `0.6793721779870134`
- Hypothetical AdamW dots: `{'jw': -6.0613889099548674e-05, 'op': 6.528621721645635e-06, 'ae17': 0.19496738132800784, 'exc': 0.7010671055191111}`

## Checkpoints

| Update | JW | F010 JW | Operator | F010 operator | AE17 | Exc | Clean28 | vs historical |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.156415 | 0.156415 | 0.030933 | 0.030933 | 3.275465 | 14.679045 |  |  |
| 64 | 0.146745 | 0.146819 | 0.029361 | 0.029476 | 2.078141 | 4.986270 | 8.723646689144 | +0.103953 |
| 128 | 0.142495 | 0.142370 | 0.028970 | 0.029449 | 5.407950 | 15.140465 | 8.712630886792 | +0.092937 |
| 192 | 0.139149 | 0.138887 | 0.029150 | 0.029534 | 3.090526 | 6.092808 | 8.777683726317 | +0.157990 |
| 251 | 0.136388 | 0.135822 | 0.029383 | 0.029510 | 1.208500 | 7.839061 | 8.845537715508 | +0.225844 |

Primary t251 Clean28 minus historical best `8.619694171538775`: `+0.225844`.

Pattern **B**. The 90-system operator loss is lower than F010 at every matched checkpoint, by `1.15e-4` to `4.79e-4`. Clean28 is higher than F010 at every matched checkpoint, by `0.0065` to `0.0308`. JW is lower than F010 only at t64 (`−0.000074`) and higher at t128, t192, and t251. AE17 is lower at t64 and t251 and higher at t128 and t192. Exc is lower only at t192.

The J251 database split is the same one seen in F010. From t0 to t251, within-database JW falls for DBH76 (`0.417` to `0.364`), MGAE109 (`0.0264` to `0.0193`), NCCE31 (`0.203` to `0.159`), and IP13 (`0.0138` to `0.0117`). It rises for pTC13, PA8, ABDE4, and EA13.

The first hypothetical AdamW step at B8 t128 descends JW (`g·Δθ = −6.06e-5`) and is slightly uphill for the raw operator (`+6.53e-6`). It does not oppose both. Later checkpoint chords are not individual AdamW steps. At t128 the instantaneous `cos(JW, total)` is `−0.157` and `cos(operator, total)` is `−0.908`, while the chord leaving that checkpoint still lowers both JW and the operator loss.

No checkpoint beat `8.619694171538775`, so the four scientific ratios were not run. The operator-priority hypothesis is not supported for this matched comparison. These results do not justify another operator-coefficient arm.

## Qualification

No scientific ratios were run.


Optimizer-update wall seconds: `6617.350`.
