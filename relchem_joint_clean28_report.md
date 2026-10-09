# Read-only Clean28 checkpoint comparison

| Checkpoint | Updates | Clean28 kcal/mol | Change from t0 | Full30 diagnostic |
|---|---:|---:|---:|---:|
| P536 t0 | 0 | 9.553190636 | +0.000000000 | 9.865316993 |
| R251 | 251 | 9.195173319 | -0.358017317 | 9.380093771 |
| J251 | 251 | 9.346760659 | -0.206429977 | 9.612715556 |
| IID LR=1e-4 t59 | 59 | 8.868473327 | -0.684717309 | 9.192675303 |

Same immutable dataset, fixed PBE0 densities and PBE0-D3(BJ). Full30 selection_allowed=false. No training, SCF, future-test access or retained model mutation.

P536 t0/R/J share exact initialization. IID t59 has fewer updates and different data presentation, so this is descriptive, not a matched causal comparison. R251 remains a diagnostic model with historical AE17/Exc/operator regressions.

Measured original optimizer-update time: R251 8.08 minutes; J251 57.22 minutes. Validation receipt times and all checkpoint hashes are in the JSON. No optional checkpoint expansion.

Ponytail: reused existing evaluator; no new evaluation framework. Pocock: verified checkpoint provenance, 28 clean and 30 diagnostic rows, fixed stoichiometry and dispersion, no model mutation; unequal update budgets explicitly retained.
