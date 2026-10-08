# Population-matched fixed calibration: controlled 90-update AdamW replay

Outcome: **MIXED / partial support**. Scientific eligibility (all exact ratios <1): **False**.

One trajectory; the only changed executable input was fixed task coefficients. The original model is the captured seed11 P536 state, with no additional displacement. Dataset, singleton objectives, original 90 draws, AdamW, precision, chunking and full-horizon schedule were unchanged. No SVRG or MOO solver was used.

## Calibration

48 IID draws, seed 930337; independently uniform chemistry identity and eight-variant choice, IID uniform shared mRKS system. Eight separate IID audit draws, seed 930338. 10,000 paired bootstrap resamples, seed 930339. Panel size, seeds, coefficient rule and replay gate were frozen before evaluation. No equal-database quota or adaptive sample additions.

| Task | Median norm | IQR | p90 | Median 95% bootstrap interval | Old lambda | New lambda | Ratio |
|---|---:|---|---:|---|---:|---:|---:|
| relchem | 14.692503 | 6.5600112–139.46325 | 231.0112 | 9.6116716–34.288455 | 0.001539594681 | 0.01701548097 | 11.0519 |
| ae17 | 4862.6263 | 1894.3891–7924.8371 | 12123.824 | 3243.4297–7191.6025 | 6.168599732e-05 | 5.141254618e-05 | 0.833456 |
| exc | 16562.165 | 12657.183–28346.319 | 39693.393 | 14320.117–23763.656 | 1.2012379e-05 | 1.509464451e-05 | 1.25659 |
| op | 0.74410162 | 0.52448507–1.1797964 | 1.4381356 | 0.60840781–0.9627381 | 0.497272084 | 0.3359756161 | 0.675637 |

The unchanged rule is lambda=1/(4*max(median norm,1e-12)). Uncertainty remains broad, especially for relchem; a median-scale correction is not proof of causality. Incidental calibration DB counts: {'MGAE109': 18, 'DBH76': 11, 'EA13': 3, 'IP13': 5, 'pTC13': 3, 'NCCE31': 4, 'PA8': 3, 'ABDE4': 1}. Rare-database norm/outlier diagnostics are retained in the hash-bound 48-record receipt; this small population panel is not a rare-DB performance qualification.

| Task | Old median weighted-norm share, calibration panel | New share | Old actual AdamW progress, independent audit | New progress |
|---|---:|---:|---:|---:|
| relchem | 0.026297 | 0.266195 | -0.065532 | 0.112374 |
| ae17 | 0.318507 | 0.146327 | 0.225591 | 0.186012 |
| exc | 0.197160 | 0.180491 | 0.226353 | 0.183377 |
| op | 0.351901 | 0.162558 | 0.167222 | 0.135799 |

Medians across draws need not sum to one. Weighted-norm shares are not directional effectiveness. Signed contributions, raw task dots, first-order loss changes, displacement norms and all audit vectors are retained in the compact metrics and external SHA-bound NPZ files. AdamW audit includes native coordinatewise normalization, F32 parameter rounding and weight decay.

Median weighted-vector rotation: 23.966700 degrees. Median actual fresh-AdamW displacement rotation: 62.135953 degrees. Chemistry displacement progress improved in 8/8 independent draws. Exact model restoration after each temporary step. The preregistered balance/rotation gate passed; no per-task Pareto gate.

## Controlled trajectory and exact endpoints

90 updates, synchronized update total 15.324 min; mean 10.2163 s/update (original 12.0697 s). Resumable checkpoints at 0,10,45,90 preserve native AdamW moments, RNG, cursor and the same 90-update cosine schedule. Large checkpoints remain outside Git.

| Task | Exact t0 | Exact new t90 | Original ratio | New ratio |
|---|---:|---:|---:|---:|
| relchem | 1.23675509377 | 1.24103243494 | 1.007042283 | 1.003458519 |
| ae17 | 24.8923555012 | 22.0351958226 | 0.793021828 | 0.885219393 |
| exc | 92.17480502 | 81.7485491562 | 0.795512981 | 0.886886055 |
| op | 0.0331482168161 | 0.0328833087916 | 0.986215238 | 0.992008378 |

Chemistry values are exact singleton means over 251×8 and 17×8; Exc/operator are exact equal-system means over 90. No parameter backwards in endpoint evaluation. Original t0 receipts were reused only after exact model equality, manifest equality and unchanged executable scientific sources; each originating receipt and checkpoint SHA is retained. No sampled or historical fixed-variant quantity substitutes for the objectives.

Clean28: initial 9.553190636, original endpoint 9.571633053, new endpoint **9.548535641 kcal/mol**. Improvement versus common t0: 0.004654994; versus original endpoint: 0.023097411. New Full30 9.863645539 kcal/mol, diagnostic only, selection_allowed=false. Frozen PBE0 density and PBE0-D3(BJ); no SCF.

Checkpoint t0-to-t90 timestamp interval (includes checkpoint/log overhead): 15.345827094713847 minutes.

Peak live CUDA allocation: 13.132059574127197 GiB; original 13.1000614 GiB. Peak reservation: 28.15234375 GiB (Windows reservation is not physical live allocation). Exclusive mean stage seconds: {'chemistry_loading_transfers': 0.07357722555552755, 'relchem': 2.010491776666337, 'ae17': 0.24981118888885653, 'mrks_loading_transfers': 0.871364595555335, 'exc': 1.0776094355555363, 'op': 5.921024046666652, 'weighted_aggregation_adamw': 0.01239101777763507}.

Relative to the original endpoint, the new AE17 loss is 11.626% higher and Exc loss is 11.486% higher. Both still improve from t0, but the trade-off is material: the coefficients cannot be declared generally superior. Relchem still regresses from t0, and the clean28 gain is small. This is partial support for calibration mismatch as one contributor, not evidence that it was the sole blocking cause.

## Recommendation

The favorable chemistry/validation trend supports one bounded longer test, not production promotion. Run one separately frozen 500-update AdamW trajectory from the same original corrected P536 checkpoint with these fixed coefficients, LR 1e-6 cosine to 1e-7 over the complete 500-update horizon, the same stochastic distributions and scientific precision; evaluate exact objectives and clean28 at declared milestones. Do not extend the completed 90-update scheduler.

This single paired replay tests coefficient calibration; it does not establish seed robustness or causal attribution from norm shares alone. Finite numerical training is reported separately from eventual all-four-loss eligibility and external competitiveness.

## Integrity and verification

45 relevant tests passed, 2 expected skips. Ruff, compileall and git diff --check passed. Trainer and scientific physics source hashes match the original run exactly. Sampling manifest SHA: e84e237d88449edfae5c68f7caecd85791b57f60a4b6239a37e1ae1350a71089. Initial tensor SHA: 3c2e45d6a86cccf80d7b4dc458e67f9b406935ed60d4e57433bb441a6df88da6. Dataset logical SHA: 61c221a19b9987717e69cac182ad545241f8807db4126c0949a99992e4c210ef.

External artifacts: C:/Dev/readWFN_share_ms/lap_population_adamw_90update_20261008. Checkpoint, calibration, protocol, audit-vector and endpoint receipt hashes are recorded in the metrics. Dataset and historical experiments were preserved. No architecture, objective, precision, sampler, optimizer rule or learning-rate changes; no SVRG, UNIT_MAXMIN rerun, SCF, Diet100 or next experiment launched.
