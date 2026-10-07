# Chemistry DB-impact and SVRG feasibility audit

Starting commit: `49af98facb2a055cea25fff3456ba6e7a9872e7f`. Final commit is the commit containing this report. Selection: **SVRG K1 static**; CASE A.

This is a read-only diagnostic on twelve exact pre-update states from the four prior UNIT_MAXMIN trajectories: seed11/23 × P67/P536, positions0/5/9. No scientific superiority of an initialization, method or long-horizon training behavior is claimed.

Frozen protocol SHA: `611a3db1856d30e2c23583997841b1fc1d4c1e299bddab163618ab5596a30525`. Byte-identical previous 16-replicate manifest SHA: `59913c53189401e92bcfbd40c07763e598519bd4021f3b0c50fa63669bca9e29`. Seeds41000–41015, canonical reaction variants and DB weights n_d/(251K) are unchanged. No draw is selected using any gradient value.

Exact objective: corrected batch_fchem singleton relative-chemistry scalar averaged equally over all251 non-AE canonical reaction groups. Eight DBs are weighted by n_d/251. Full251 and AE17/Exc/operator gradients are reused by SHA. Main model is F32, chemistry uses the existing matched-F64 shadow, and unique tied exchange parameters occur once in the9446-coordinate order.

The prior cache covers157–159 singleton reactions per state. Only missing singleton chemistry gradients were computed; existing rows were copied bitwise. Each completed251-row mean is checked against the previously stored full251 gradient with relativeL2<=1e−11. No full251 objective/gradient evaluation or secondary scientific objective evaluation was rerun.

DB variance is population mean ||g_j−mu_d||². The exact finite-population MSE for a K-sample contribution is (n_d/251)² (n_d−K)/(K(n_d−1)) times that dispersion. Empirical MSE, cosine dispersion and signed e_d·e_total/||e_total||² are reported per DB/state/K. Signed cross terms may be negative and are retained; squared marginal variances are not additive empirical attribution.

The primary DB ranking uses leave-one-DB-exact UNIT_MAXMIN progress, not variance: pooled probability uplift, then catastrophic-tail reduction, then median paired progress change, then alphabetical tie. All12 states and K1/K2/K4 have equal observation weight. This descriptive ranking is derived from the audit; its top pair is tested on the same frozen panel without claiming independent validation.

| Rank | DB | Success uplift | Base success | Exact-one success | Median delta p | p05 delta p | Tail count reduction | Extra backwards K1/K2/K4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | EA13 | 0.0572916667 | 0.333333333 | 0.390625 | 0.0190603535 | -0.208564021 | 33 | 10/9/7 |
| 2 | PA8 | 0.046875 | 0.333333333 | 0.380208333 | 0.0278324208 | -0.163945471 | 37 | 7/6/4 |
| 3 | MGAE109 | 0.0399305556 | 0.333333333 | 0.373263889 | 0.0205127555 | -0.111166678 | 31 | 103/102/100 |
| 4 | NCCE31 | 0.0364583333 | 0.333333333 | 0.369791667 | -0.014683668 | -0.212150999 | 14 | 27/26/24 |
| 5 | ABDE4 | 0.0329861111 | 0.333333333 | 0.366319444 | 0 | -0.0725895085 | 18 | 3/2/0 |
| 6 | IP13 | 0.00347222222 | 0.333333333 | 0.336805556 | -0.0123997953 | -0.266912296 | 4 | 12/11/9 |
| 7 | DBH76 | 0 | 0.333333333 | 0.333333333 | 0.00310979646 | -0.0858614051 | -1 | 69/68/66 |
| 8 | pTC13 | -0.00520833333 | 0.333333333 | 0.328125 | -0.00866483377 | -0.145497983 | -1 | 12/11/9 |

DB impact depends on minibatch size. The pooled top two are not universal harmful-DB labels: MGAE109 leads K1, PA8 leads K2, and NCCE31 leads K4. Some exactifications reduce success at a different K.

| DB | K1 success uplift | K2 success uplift | K4 success uplift | K4 catastrophic-tail reduction |
| --- | --- | --- | --- | --- |
| EA13 | 0.046875 | 0.109375 | 0.015625 | 0 |
| PA8 | -0.0104166667 | 0.161458333 | -0.0104166667 | 4 |
| MGAE109 | 0.078125 | 0.0625 | -0.0208333333 | 3 |
| NCCE31 | -0.0364583333 | 0.0416666667 | 0.104166667 | 12 |
| ABDE4 | 0.0677083333 | 0.03125 | 0 | 0 |
| IP13 | -0.0364583333 | 0.0260416667 | 0.0208333333 | -2 |
| DBH76 | -0.0208333333 | 0.015625 | 0.00520833333 | 2 |
| pTC13 | -0.015625 | -0.015625 | 0.015625 | -2 |

Best pair: EA13, PA8. No exhaustive pair search.

| K | Base success | Exact-pair success | Uplift | Delta p min/p05/median | Tail reduction | Extra backwards |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.203125 | 0.239583333 | 0.0364583333 | -0.587084896/-0.326565323/0.0414344265 | 5 | 17 |
| 2 | 0.223958333 | 0.442708333 | 0.21875 | -0.174770397/-0.101368189/0.11510127 | 43 | 15 |
| 4 | 0.572916667 | 0.677083333 | 0.104166667 | -0.263326824/-0.128158603/0.0417322756 | 16 | 11 |

SVRG formula: g_ref_full + g_sample(target) − g_sample(reference), with identical reaction IDs, variants and DB weights in the paired terms. References are exclusively earlier states. Target full gradients enter truth scoring and the fullchem-informed direction comparison, never the sampled estimator construction.

Unique cases per K: four u5/u0, four u9/u0, four u9/u5, giving192 rows. Static and refresh-5 candidates each use128 rows/eight cases; u5/u0 is shared between schedules. No update0/reference-equals-target observation is included in qualification. Shared draws/cases are paired evidence, not independent population replicates.

Frozen acceptance: >=95% strict p_full>0 overall; >=90% in every case; no p_full below−0.02; finite and qualified UNIT_MAXMIN KKT/common-descent solution. Every16-draw case needs at least15 successes and every128-row candidate needs at least122. No epsilon, relaxed tail gate or post-hoc tolerance changes selection.

| Candidate | Fullchem descent | Worst case | Cosine min/p05/median | p_full min/p05/median | Median gamma | Catastrophic tails | Solver qualified | Qualifies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SVRG K1 static | 128/128 | 1 | 0.999999918/0.999999954/0.999999995 | 0.143950766/0.143953905/0.182412958 | 0.182412759 | 0 | True | True |
| SVRG K1 refresh-5 | 128/128 | 1 | 0.999999974/0.999999983/0.999999997 | 0.143950766/0.143953905/0.182412334 | 0.18241204 | 0 | True | True |
| SVRG K2 static | 128/128 | 1 | 0.999999936/0.999999962/0.999999996 | 0.143951075/0.14395441/0.182413377 | 0.182412369 | 0 | True | True |
| SVRG K2 refresh-5 | 128/128 | 1 | 0.999999974/0.999999986/0.999999998 | 0.143951075/0.14395441/0.182412599 | 0.182412104 | 0 | True | True |

Per-case success:

| Candidate | Target/reference case | Fraction |
| --- | --- | --- |
| SVRG K1 static | 11_P67_u5 | 1 |
| SVRG K1 static | 11_P67_u9_static | 1 |
| SVRG K1 static | 11_P536_u5 | 1 |
| SVRG K1 static | 11_P536_u9_static | 1 |
| SVRG K1 static | 23_P67_u5 | 1 |
| SVRG K1 static | 23_P67_u9_static | 1 |
| SVRG K1 static | 23_P536_u5 | 1 |
| SVRG K1 static | 23_P536_u9_static | 1 |
| SVRG K1 refresh-5 | 11_P67_u5 | 1 |
| SVRG K1 refresh-5 | 11_P67_u9_refresh | 1 |
| SVRG K1 refresh-5 | 11_P536_u5 | 1 |
| SVRG K1 refresh-5 | 11_P536_u9_refresh | 1 |
| SVRG K1 refresh-5 | 23_P67_u5 | 1 |
| SVRG K1 refresh-5 | 23_P67_u9_refresh | 1 |
| SVRG K1 refresh-5 | 23_P536_u5 | 1 |
| SVRG K1 refresh-5 | 23_P536_u9_refresh | 1 |
| SVRG K2 static | 11_P67_u5 | 1 |
| SVRG K2 static | 11_P67_u9_static | 1 |
| SVRG K2 static | 11_P536_u5 | 1 |
| SVRG K2 static | 11_P536_u9_static | 1 |
| SVRG K2 static | 23_P67_u5 | 1 |
| SVRG K2 static | 23_P67_u9_static | 1 |
| SVRG K2 static | 23_P536_u5 | 1 |
| SVRG K2 static | 23_P536_u9_static | 1 |
| SVRG K2 refresh-5 | 11_P67_u5 | 1 |
| SVRG K2 refresh-5 | 11_P67_u9_refresh | 1 |
| SVRG K2 refresh-5 | 11_P536_u5 | 1 |
| SVRG K2 refresh-5 | 11_P536_u9_refresh | 1 |
| SVRG K2 refresh-5 | 23_P67_u5 | 1 |
| SVRG K2 refresh-5 | 23_P67_u9_refresh | 1 |
| SVRG K2 refresh-5 | 23_P536_u5 | 1 |
| SVRG K2 refresh-5 | 23_P536_u9_refresh | 1 |

Costs are deployable reaction backwards, not audit cache-construction work. Charge251 for each full reference refresh and16K for paired sampled terms per update. Static refreshes at0; refresh-5 at0 and5. Conservative totals charge paired sampling even at those identity updates. The final column discloses an optional reference-identity shortcut using the already charged full gradient: omit paired terms at update0 for static, and at updates0 and5 for refresh-5. Candidate ordering is unchanged. Costs exclude secondary tasks, memory/storage, solver work and runtime variation across reactions; they are not wall-clock benchmarks.

| Candidate | Reference backwards | Paired/update | Paired/10 | Total/10 | vs K1 | vs K4 | vs full251/update | Identity-shortcut total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SVRG K1 static | 251 | 16 | 160 | 411 | 5.1375 | 1.284375 | 0.16374502 | 395 |
| SVRG K1 refresh-5 | 502 | 16 | 160 | 662 | 8.275 | 2.06875 | 0.26374502 | 630 |
| SVRG K2 static | 251 | 32 | 320 | 571 | 7.1375 | 1.784375 | 0.22749004 | 539 |
| SVRG K2 refresh-5 | 502 | 32 | 320 | 822 | 10.275 | 2.56875 | 0.32749004 | 758 |

Comparators over ten updates: current K1=80, K4=320, full251 each update=2510 chemistry backwards. Cached reference/sample terms are never treated as free. A reference-refresh memory/performance implementation is outside this audit.

All raw DB counterfactuals, DB mean norms/variance summaries, CV norm ratios/angular errors, progresses, direction comparisons, coefficients and KKT receipts are externally SHA-bound at `C:\Dev\readWFN_share_ms\lap_chemistry_db_variance_svrg_runs_20261007\analysis_rows.json` (`39134787eee47485989585412e2e9cd90eec6088c222dda72fb64e1dfc2ab0eb`). Exact DB mean vectors are reproducible from the complete singleton caches. State identities and full-cache SHAs are in structured metrics; no large tensor is added to Git.

Tests/hash/state validation: PASS. Independent review: PASS. Structured metrics embed full receipts. Immutable sources/data/states are checked before and after; production source changes are zero.

A local CV result on these tiny ten-update trajectory displacements is a feasibility finding, not a deployment guarantee. No training gate has passed merely because this frozen-gradient screen passes. The next qualification must still measure deterministic full objectives and honest refresh cost, and bind the CV gradient to a consistent scalar estimator for the unchanged Armijo controller before execution.

Next experiment: A preregistered four-start × ten-update UNIT_MAXMIN qualification using SVRG K1 static chemistry only, keeping the shared manifest, initialization states and vector-Armijo controller unchanged. Not launched.

No training or retained model update occurred; UNIT_MAXMIN/controller/production behavior were unchanged. No new initialization, MOO method, adaptive sampler, K>2 SVRG, seed41, cursor advancement, 25/100-update run, full90, SCF, Diet or Slurm job was launched.
