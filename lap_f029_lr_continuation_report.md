# F029 — C

Exact F028 INTERVENTION t251 SHA256 `49ccd8da8194cb12a44829e524973c90a5fe11079a82c3e1938e57fb2a067d1e` restored once per process and copied into independent matched arms; parameters, buffers and saved moments identical. Only optimizer LR differs:2.5e-5 versus2.5e-6. No new initialization, moment reset, epsilon or objective change. Global counters251→283,32 retained updates per arm,64 total. No temporary molecular optimizer preflight update.

|Original metric|ORIGINAL t251|CONTROL t283|LOW_LR t283|
|---|---:|---:|---:|
|JW|0.110342125689|0.102648682396|0.107584430797|
|legacy|1.77451258576|1.69505661367|1.75098010503|
|MAE|7.17535837032|5.58392444356|6.57374750999|
|RMSE|10.3953841353|8.04191320123|9.46462504586|
|median_absolute_error|4.68071486778|3.7856180138|4.27869197824|
|macro_database_MAE|6.77444080855|6.16306553695|6.56740109585|
|ae17|18.8171827053|12.2922271454|4.11369468949|
|exc|70.7261752291|41.5525556936|16.5247135733|
|op|0.0376524268285|0.0349321721872|0.0363233697842|

JW/operator progress relative to original:True; relative to control:False; low-LR AE17/Exc below original P536 thresholds:True. Full signed/relative three-way differences are in metric_comparisons.csv. Scientific P536 eligibility is separate:CONTROL False, LOW_LR False; all ratios in p536_ratios.json, H signed/absolute errors and exact variant in hydrogen.json. Clean28 receipts:{'CONTROL': 'NOT EVALUATED', 'LOW_LR': 'NOT EVALUATED'}. Ineligible checkpoints were never passed to a development loader. No future/test access.

|Arm/global update|Total linear prediction|Actual same-presentation change|Overshoot|Chemistry direction conflict|Operator direction conflict|
|---|---:|---:|---|---|---|
|CONTROL/252|+0.000608763537|+0.000626922912|False|False|True|
|LOW_LR/252|+6.08643895e-05|+6.09060809e-05|False|False|True|
|CONTROL/253|-0.00271498899|-0.00270097301|False|False|False|
|LOW_LR/253|-0.000270165643|-0.000270038288|False|False|False|
|CONTROL/283|-0.00175555944|-0.000994689696|False|False|False|
|LOW_LR/283|-0.000180758752|-0.000180544717|False|False|False|

All task-wise before/after values, actual rounded displacement derivatives and classifications are in finite_steps.csv. A positive task derivative is a directional conflict; only negative total prediction with positive actual total change is called overshoot. Smaller LR does not itself reverse a fixed direction. No rejection/backtracking occurred.

Read-only original t128/t192/t251 characterization:historical_metrics.csv. No historical state selected a starting checkpoint; fixed t251 only. These sparse snapshots do not establish oscillations or overshoot. Next32 manifest is an explicitly defined next-cycle prefix, not a recovered historical post251 stream. Both arms used the same identities, variants, offsets and original paired physical presentations. F028 E251/E17/full90 evaluator reused unchanged, with only checkpoint loader/output routing injected; original task-specific precision, targets, gauge, overlap, dispersion and reductions retained. `relchem_loss`/chemistry task values denote c010-scaled F010 objective, never legacy metric.

Hard constraints pass at all retained updates and every restored milestone; native/shadow epsilon asserted after tensor loads. First complete raw task gradients pass deterministic parity before training. F64 weighted accumulation and one nativeF32 cast use unchanged F028 helpers. Both optimizers retain original moments/counters; independent storage checked. Scientific wall 4809.910s /7200s. Per-update execution wall is measured; GPU-kernel-active time not separately metered. Full per-reaction, eight-database, E17, Exc and90-system operator receipts are retained. Large tensors/checkpoints outside Git, hash-bound in provenance. Historical F028 and production files unchanged. Focused test/Ruff/compileall/diff receipts finalized separately.

This is a short stabilization/progress diagnostic, not convergence or evidence for long-horizon reduced-LR benefit. LR does not establish a new functional or cure directional conflicts by definition.

One next action:One separately authorized read-only attribution of the adverse update252 direction at the common theta251: separate fresh-gradient and saved-first-moment numerator contributions under its fixed actual AdamW denominator. Do not reset moments or apply any proposal.


## Final interpretation and complete scalar receipts

C — MIXED OR NO CLEAR LR BENEFIT. Both arms make resolved JW/operator progress relative to the shared original; LOW_LR is worse than CONTROL on both. Thus neither joint-progress gate A nor stabilization gate B passes. LOW_LR gives better AE17/Exc and a smaller H absolute error than CONTROL, while H absolute error worsens relative to original in both arms. This is a different compromise, not proof of convergence or universal LR ineffectiveness.

|Historical state|JW|Legacy|MAE / RMSE|AE17|Exc|Full90 operator|H signed kcal/mol|
|---|---:|---:|---:|---:|---:|---:|---:|
|t128|0.1202534083|1.78938021|8.2834171 / 11.907991|31.27399457|108.9281276|0.03569189643|-6.154720341|
|t192|0.1231323572|1.947609272|10.547353 / 15.981044|59.01359793|218.7229079|0.04224724489|-1.99759261|
|t251|0.1103421257|1.774512586|7.1753584 / 10.395384|18.81718271|70.72617523|0.03765242683|-3.901574285|

The three snapshots show deterioration t128→t192 and recovery t192→t251 in these aggregate objectives, with task-specific differences such as H. They do not establish systematic oscillations or old-step overshoot. Starting checkpoint remained fixed t251. The new t251 replay equals authenticated F028 headline values exactly at the reported precision.

|Central comparison|JW signed / relative change|Operator signed / relative change|
|---|---:|---:|
|LOW_LR-original|-0.00275769489201 / -2.49922219%|-0.00132905704429 / -3.52980447%|
|LOW_LR-CONTROL|+0.00493574840043 / +4.80838944%|+0.00139119759695 / +3.98256825%|
|CONTROL-original|-0.00769344329245 / -6.97235371%|-0.00272025464125 / -7.22464625%|

Prospective per-evaluation JW uncertainty 4.91779514923e-08; pairwise bound is its sum. Operator bound is the sum of max(1e-12,1.9073486328125e-6*abs(loss)) for the two compared values. All central differences exceed these bounds. These are numerical resolution bounds, not statistical confidence intervals. Full metric/H signed and absolute deltas:metric_comparisons.csv; all48 database MAE/RMSE deltas:database_comparisons.csv.

|Database (kcal/mol)|Original MAE / RMSE|CONTROL MAE / RMSE|LOW_LR MAE / RMSE|
|---|---:|---:|---:|
|ABDE4|13.11211 / 13.296889|11.704397 / 11.909364|12.712786 / 12.902623|
|DBH76|4.7816298 / 5.8135156|4.6375301 / 5.7073016|4.7500069 / 5.7848585|
|EA13|3.497565 / 4.0398583|2.7573251 / 3.3639809|3.1599034 / 3.7209745|
|IP13|3.1392374 / 3.9749593|3.5970989 / 4.4734476|3.3383079 / 4.1596594|
|MGAE109|10.658851 / 14.232048|6.9308323 / 9.9257212|9.222896 / 12.563748|
|NCCE31|0.60158702 / 0.85714276|0.54782138 / 0.76960354|0.57638261 / 0.81683791|
|PA8|6.0109792 / 6.1794741|6.3603668 / 6.5052713|6.1887764 / 6.3490677|
|pTC13|12.393568 / 13.227999|12.769153 / 13.585805|12.59015 / 13.420293|

|Original P536 ratio|CONTROL|LOW_LR|
|---|---:|---:|
|relchem|1.37199212234|1.41725703501|
|ae17|0.493642981921|0.165201674946|
|exc|0.450801666297|0.179275818047|
|op|1.05381753658|1.09578653916|

Both fail legacy and operator gates; both pass AE17 and Exc. Clean28 was NOT EVALUATED in either arm. H signed/absolute: original -3.9015742851/3.9015742851, CONTROL -5.36430348363/5.36430348363, LOW_LR -4.61068579023/4.61068579023 kcal/mol; original level3_mura reference unchanged.

Overshoot on the prospectively selected252/253/283 updates:CONTROL 0/3, LOW_LR 0/3. It is therefore not established that reduced LR prevented overshoot in these observations. At252 the total direction itself was adverse in both; smaller LR reduced ascent magnitude without reversing operator sign. At253 and283 total predictions and actual changes were negative in both. Across all32 already-computed updates, positive chemistry derivatives occurred CONTROL 12/32, LOW_LR 11/32; operator CONTROL 16/32, LOW_LR 14/32. Thus task-direction conflict remains. The two trajectories differ after update252; later direction counts are descriptive, not fixed-state causal percentages.

Full90 paired system counts:{'LOW_LR-original': {'improved': 90, 'regressed': 0, 'ties': 0}, 'LOW_LR-CONTROL': {'improved': 0, 'regressed': 90, 'ties': 0}, 'CONTROL-original': {'improved': 90, 'regressed': 0, 'ties': 0}}. All90 individual values and both reference-point deltas are in full90_operator.csv. Dataset/table `state` is the authoritative F029 branch/history label; the inherited evaluator `arm=INTERVENTION` alias denotes the common F028 origin, not F028 CONTROL weights. Each arm received256 chemistry,32 AE17 and32 paired mRKS/Exc/operator presentations. Original F028 CONTROL checkpoint was never loaded.

Integrity:14 focused/relevant tests PASS, zero skipped; Ruff/format/compileall/diff PASS. Ten restored milestone constraint receipts and64 retained-step constraint checks PASS; 102 consumed original files rehashed unchanged; historical checkpoints128/192/251 unchanged. The pretraining output-alias routing error and the corrected synthetic parameter-order fixture are explicitly retained in pretraining_development_receipt/auxiliary_sources, with zero molecular optimizer steps. Pretraining formatting correction preserved AST. No tolerance was relaxed; no production source or target was repaired.

Measured gradient/update wall:CONTROL 944.482s, LOW_LR 928.889s; training phase 1919.040s. Recorded phase wall sum 4114.953s; conservative scientific workflow wall 4959.210s /7200s. GPU-kernel-active time was not separately metered.

One next action:One separately authorized read-only attribution of the adverse update252 direction at the common theta251: separate fresh-gradient and saved-first-moment numerator contributions under its fixed actual AdamW denominator. Do not reset moments or apply any proposal. No follow-up is executed.
