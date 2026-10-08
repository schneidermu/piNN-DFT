# Controlled AdamW LR sweep — stopped partial results

Stopped at user request. No further training, exact endpoint evaluation or external validation was performed. All six screens completed 20 real optimizer updates without nonfinite values or OOM. The sole continuation, LR=1e-4, is safely saved at cursor59; all other arms remain at cursor20. Model, optimizer moments, RNG, sampling manifest and fixed coefficients are preserved.

## Training-only probe (t20/t0)

These are limited independent training-probe ratios, not exact full-corpus ratios. No t90 or clean28 values exist for this sweep.

| LR | Relchem | AE17 | Exc | Operator | Saved updates | Mean screen seconds |
|---|---:|---:|---:|---:|---:|---:|
| 1e-6 | 1.001487 | 0.938617 | 0.940190 | 0.995664 | 20 | 9.41 |
| 1e-5 | 1.014864 | 0.391794 | 0.403809 | 0.958150 | 20 | 9.27 |
| 3e-5 | 1.019400 | 0.086687 | 0.105144 | 0.930604 | 20 | 9.30 |
| 1e-4 | 0.979260 | 0.560224 | 0.569967 | 0.936553 | 59 | 9.34 |
| 3e-4 | 1.021041 | 0.707924 | 0.657996 | 0.875104 | 20 | 9.28 |
| 1e-3 | 1.009751 | 0.441771 | 0.398213 | 0.938573 | 20 | 9.29 |

LR=1e-4 is the provisional training-probe lead: all four probe means improved. Its paired bootstrap chemistry ratio interval is [0.914638, 1.037269], so chemistry improvement is uncertain. The frozen training-only selection chose 1e-4, 1e-6 and 1e-3; the latter two continuations were not started. No scientifically optimal LR or long-horizon schedule can be selected from these partial results.

Peak screen live CUDA allocation: 12.22–12.23 GiB. Total completed optimizer updates: 159. All methods used identical initial parameters, dataset, sampling, fixed coefficients and constant-LR AdamW; only LR differed. No SVRG, MOO, precision or objective changes.

Checkpoints remain outside Git at `C:/Dev/readWFN_share_ms/lap_adamw_lr_sweep_20261008`. Latest cursor59 checkpoint SHA256: `8938753bee6cfcb55ccdac9c02a15cbb3e3b2093290c9122e5e00ad400a966aa`. Individual hashes and bootstrap intervals are in the metrics.

Validation: 25 focused/relevant tests passed; Ruff and compileall passed. Scientific gradient functions retained their reference implementation and initial raw gradients were bitwise equal across all six arms. Constant-LR checkpoint/resume was tested. No future-test evaluation or SCF occurred.
