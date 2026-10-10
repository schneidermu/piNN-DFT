# Balanced B8 chemistry AdamW

**Classification: NO-GO**

One new 251-update AdamW arm. The only intended change from J251 is the relchem
estimator: eight qualified singleton gradients, averaged with equal weight 1/8.
J251 was not retrained. Matched update counts are not matched data: J251 used 251
chemistry presentations and B8 uses 2008. A difference cannot be attributed only to
variance reduction. Successive batches are cyclic shifts of one permutation, not IID draws.
At frozen parameters, the mean over a uniform random update index equals the full
fixed-variant 251-reaction gradient. Parameters move during the run, so that equality
is not a conditional-unbiasedness claim at each optimizer update.

## Manifest

B8 manifest SHA256: `b249964d410f0db4a59867de53bb35c6442ec23aa81eefb0b96b50242d94f6bc`.
Offsets: `(0, 31, 62, 93, 124, 155, 186, 217)`.
Each of the 251 J251 identities is scheduled eight times, always with its original
J251 quadrature variant. The first reaction of update t is the J251 reaction at t.
AE17 and mRKS entries are the J251 entries. Database exposure is eight times the
J251 population. No validation or future-test identity is in the training manifest.

## Parity

Index 0: relative L2 `0.000e+00`, scalar relative `2.199e-16`, coordinates `9446`.
Index 125: relative L2 `0.000e+00`, scalar relative `0.000e+00`, coordinates `9446`.
Index 250: relative L2 `0.000e+00`, scalar relative `0.000e+00`, coordinates `9446`.
B1 singleton relative L2 `0.0`. Other-task relative L2 `{'ae17': 0.0, 'exc': 0.0, 'op': 0.0}`. Coefficient check `0.0`. Joint dtype `float64`, parameter dtype `float32`, optimizer step `False`.

## Other tasks

AE17, Exc, and the operator gradient are computed by the J251 measurement calls.
They are not averaged over eight systems. Scalarization applies the historical
coefficients once. The F32 cast remains inside `adamw_step`.

## Runtime

Completed optimizer updates: `251`.
Relchem presentations: `2008`.
Cumulative new GPU time: `7930.080` seconds.
Peak allocated CUDA memory: `14090057216` bytes.
Peak reserved CUDA memory: `14206107648` bytes.

Preflight median update estimate: `25.538` seconds. Projected total including the evaluation reserve: `8960.588`.

## Clean28

| Model | Updates | Clean28 | Relchem ratio | AE17 ratio | Exc ratio | Op ratio | Eligible |
|---|---:|---:|---:|---:|---:|---:|---|
| P536 | 0 | 9.553190636 | 1 | 1 | 1 | 1 | No, strict threshold |
| J251 control | 251 | 9.346760659 | 0.979707732 | 0.081012145 | 0.101104757 | 0.938121207 | Yes |
| Historical best | 80 | 8.619694172 | 1.034869578 | 0.080727952 | 0.059058103 | 0.944069705 | No |
| Best eligible interpolation |  | 9.168650862 | 0.991430353 | 0.122272970 | 0.054614319 | 0.929237644 | Yes |
| New t64 | 64 | 9.229562607 | NOT EVALUATED — ACCURACY GATE | NOT EVALUATED — ACCURACY GATE | NOT EVALUATED — ACCURACY GATE | NOT EVALUATED — ACCURACY GATE | NOT EVALUATED — ACCURACY GATE |
| New t128 | 128 | 9.096700112 | NOT EVALUATED — ACCURACY GATE | NOT EVALUATED — ACCURACY GATE | NOT EVALUATED — ACCURACY GATE | NOT EVALUATED — ACCURACY GATE | NOT EVALUATED — ACCURACY GATE |
| New t192 | 192 | 9.113418584 | NOT EVALUATED — ACCURACY GATE | NOT EVALUATED — ACCURACY GATE | NOT EVALUATED — ACCURACY GATE | NOT EVALUATED — ACCURACY GATE | NOT EVALUATED — ACCURACY GATE |
| New t251 | 251 | 9.161767874 | NOT EVALUATED — ACCURACY GATE | NOT EVALUATED — ACCURACY GATE | NOT EVALUATED — ACCURACY GATE | NOT EVALUATED — ACCURACY GATE | NOT EVALUATED — ACCURACY GATE |

Differences below use the exact receipts. t64, t128, and t192 are not matched to J251 t251.

t64: Clean28 `9.229562607`, versus P536 `-0.323628`, versus J251 `-0.117198`, versus historical best `+0.609868`.
Improved versus J251: 19. Worsened versus J251: 9.
- `SIE4x4-15` (SIE4x4): weighted `56.077017`, signed `33.181667`, delta J251 `-0.171900`.
- `HEAVY28-16` (HEAVY28): weighted `27.269294`, signed `0.595529`, delta J251 `+1.098188`.
- `WCPT18-15` (WCPT18): weighted `25.190921`, signed `-15.549951`, delta J251 `-0.802371`.
- `BHPERI-11` (BHPERI): weighted `20.440520`, signed `-7.514897`, delta J251 `+0.233236`.
- `BSR36-31` (BSR36): weighted `17.963745`, signed `-5.117876`, delta J251 `-0.120993`.
t128: Clean28 `9.096700112`, versus P536 `-0.456491`, versus J251 `-0.250061`, versus historical best `+0.477006`.
Improved versus J251: 20. Worsened versus J251: 8.
- `SIE4x4-15` (SIE4x4): weighted `56.003636`, signed `33.138246`, delta J251 `-0.245282`.
- `WCPT18-15` (WCPT18): weighted `25.443980`, signed `-15.706160`, delta J251 `-0.549312`.
- `HEAVY28-16` (HEAVY28): weighted `24.510508`, signed `0.535281`, delta J251 `-1.660599`.
- `BSR36-31` (BSR36): weighted `19.193255`, signed `-5.468164`, delta J251 `+1.108517`.
- `BHPERI-11` (BHPERI): weighted `19.079454`, signed `-7.014505`, delta J251 `-1.127831`.
t192: Clean28 `9.113418584`, versus P536 `-0.439772`, versus J251 `-0.233342`, versus historical best `+0.493724`.
Improved versus J251: 19. Worsened versus J251: 9.
- `SIE4x4-15` (SIE4x4): weighted `55.905610`, signed `33.080243`, delta J251 `-0.343307`.
- `WCPT18-15` (WCPT18): weighted `25.332829`, signed `-15.637549`, delta J251 `-0.660463`.
- `HEAVY28-16` (HEAVY28): weighted `22.700141`, signed `0.495745`, delta J251 `-3.470966`.
- `BSR36-31` (BSR36): weighted `20.578864`, signed `-5.862924`, delta J251 `+2.494126`.
- `BHPERI-11` (BHPERI): weighted `17.829194`, signed `-6.554851`, delta J251 `-2.378090`.
t251: Clean28 `9.161767874`, versus P536 `-0.391423`, versus J251 `-0.184993`, versus historical best `+0.542074`.
Improved versus J251: 18. Worsened versus J251: 10.
- `SIE4x4-15` (SIE4x4): weighted `55.868100`, signed `33.058047`, delta J251 `-0.380817`.
- `WCPT18-15` (WCPT18): weighted `25.090934`, signed `-15.488231`, delta J251 `-0.902358`.
- `BSR36-31` (BSR36): weighted `21.785977`, signed `-6.206831`, delta J251 `+3.701239`.
- `HEAVY28-16` (HEAVY28): weighted `21.176962`, signed `0.462480`, delta J251 `-4.994145`.
- `Amino20x4-28` (Amino20x4): weighted `18.250082`, signed `0.782929`, delta J251 `+1.833417`.

## Full251 diagnostic

Best Clean28 checkpoint t128 fixed-panel relchem ratio `0.968002640` (objective `1.195939284`). This is a no-gradient population diagnostic, not scientific qualification. J251 relchem ratio is `0.979707732`.

| Database | Count | Mean loss | Contribution |
|---|---:|---:|---:|
| ABDE4 | 4 | 6.735836178 | 0.107344003 |
| DBH76 | 70 | 0.928442061 | 0.258928065 |
| EA13 | 11 | 1.768704168 | 0.077512932 |
| IP13 | 13 | 2.421401714 | 0.125411244 |
| MGAE109 | 104 | 0.232048536 | 0.096147600 |
| NCCE31 | 28 | 2.077115335 | 0.231710077 |
| PA8 | 8 | 1.924776819 | 0.061347468 |
| pTC13 | 13 | 4.586308581 | 0.237537895 |

## Gradient diagnostics

`norm(mean(g_i)) / mean(norm(g_i))` over 251 updates: min `0.154295`, median `0.659499`, max `0.957832`.
The ratio compares those two norms. It is not a variance estimate.

## Provenance

P536 file `0ca0f77367214c171e6edf0576bdb043d64479245ad1a0e539fa11fb0cee542d`.
P536 tensor `3c2e45d6a86cccf80d7b4dc458e67f9b406935ed60d4e57433bb441a6df88da6`.
J251 manifest `72b827655ce4421f2fc933c082cda1c2dc3e5d9903ec29e0c5e3d62e82ac37f6`.
Evaluation manifest `132bd550161be6bc6761f8df1de8d3755c0b0debab53663208c3a9bbf56cb805`.
Calibration `4d97df1ec78aa01a2c9380a86a323b2f0b46828dfc5975f5e3a63c75f4440096`.
Production physics files were hashed against the J251 protocol before training.

## Next recommendation

Full251 relchem improved relative to both P536 and J251 while Clean28 stayed above the historical best. That is a generalization mismatch, not proof that the population estimator failed. Rank a longer fixed-loss, learning-rate-scheduled continuation of this same B8 loss ahead of SVRG. Do not search another batch size first. Do not launch it.

This recommendation was not executed.
