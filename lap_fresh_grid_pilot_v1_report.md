# Fresh R0 matched full-grid/subset training pilot

Classification B. Both predeclared 32-update branches completed. Relative FULL_SOURCE benefit on the common SOURCE evaluation is B=-2.68250750062e-07; the 1% practical floor does not pass. This is one seed and three training systems, not chemical generalization or an architecture-capacity test.

## Initialization and fixed contract

Seed 2026 (Python/NumPy/Torch/CUDA), exact original constructor {'num_layers': 6, 'h_dim': 32, 'dropout': 0.0, 'use_g_x': True, 'use_g_c': True}, 9446 unique native-F32 trainable coordinates. One scientific constructor initialization, then deepcopy into independent SUBSET/FULL_SOURCE models and a synchronized F64 derivative carrier. Initial state hash `359088b9713e532be0f811b4bd9f5f4f91b34d531a936db7548f46b2b410aedd`; parameter/order/buffer receipts are in initialization.json. Old trained weights and optimizer states loaded: 0. New AdamW states empty at t0, identical single parameter-group policy; both native counters end at 32.

Frozen HEAD `a34574b03a3df615c3e221e3bebbcc2457f83db3`. GPU NVIDIA GeForce RTX 5070 Ti, Torch 2.11.0+cu128. Preparation uses the already authenticated PySCF 2.14.0 source/basis contract. Original production files and historical artifacts remain hash-identical.

## Representations

|System|FULL_SOURCE points|SUBSET points|AO dimension|SOURCE / SUBSET reference replay error, AO Fro|
|---|---:|---:|---:|---:|
|H2|47080|42508|60|[2.304889894971429e-15, 2.3007720438721984e-15]|
|BeH2|89200|77240|144|[5.7744843302929955e-15, 5.792850722774291e-15]|
|CO|85960|72110|168|[1.4059225195241373e-14, 1.4041318195992215e-14]|

Both representations use exact SOURCE F64 coordinates, weights, vxc, AO values/derivatives and descriptors. SUBSET uses the authenticated ordered index list, including repeated and zero-weight rows. No new cutoff or historical-mask reconstruction. Prediction and reference are assembled on the same grid in each case; legacy RefAO/rounded weights are not used. The reference replay is against prior source-precision R0/R2, not legacy R4.

The learned forward/descriptors and parameter derivatives use F64 shadow coordinates synchronized from live native weights for every batch. The existing _model_double_operator_partials keeps its native F32 PBE rho/sigma/constants and unchanged regularization. AO assembly and gradient sums are F64. The mean gradient is cast once to native F32 immediately before AdamW. This common source-precision experiment is not a byte-identical production-corpus retraining.

Loss = mean of the three fully assembled original weak-form matrix losses, ||S^(-1/2)(V−Vref)S^(-1/2)||F²/n_AO. No AE17, energy, relative chemistry, gauge shift, clipping, scheduler or coefficient calibration. Each gradient uses two bounded passes: full matrix first, then its one common matrix adjoint accumulated across chunks. The direct-autograd fixture and existing mixed-precision/operator regression tests cover this calculation; no chunk-square loss substitution.

## Initial gradient sensitivity

|Scope|norm SUBSET|norm FULL_SOURCE|absolute vector difference|relative difference|cosine|
|---|---:|---:|---:|---:|---:|
|full|33745430.91679171|33745589.04345739|159.35216990205524|4.722162938001834e-06|0.9999999999998294|
|correlation|0.37549815729703323|0.3754986983050779|1.929158693951009e-06|5.1375908962103595e-06|0.9999999999878402|
|normalization|33745430.91673738|33745589.04340307|159.35216990202858|4.722162938008645e-06|0.9999999999998294|
|exchange|60.576846486508906|60.576862263148534|9.269061182112872e-05|1.5301322709399615e-06|0.9999999999988631|
|system_H2|11533849.217318106|11533885.221185375|36.35968742373023|3.1524232057507353e-06|0.9999999999999033|
|system_BeH2|13747701.559851887|13748115.008786464|416.4998847879507|3.0295053868967803e-05|0.9999999999933015|
|system_CO|75962834.89014862|75962859.82028194|25.265605225970454|3.326047134842676e-07|0.9999999999999987|

Exchange/correlation/normalization ownership follows the authenticated F013 mapping: identify LayerNorm by module identity before assigning network prefixes, with no overlap or duplicate tied exchange parameters. Initial per-system vectors and the full gradient are saved outside Git with hashes. No arbitrary epsilon is used for zero norms; undefined metrics are blank in CSV.

Initial SOURCE mean loss 1.26809266958. Large raw gradient norms, dominated initially by normalization coordinates, were disclosed and were not used as a scientific rejection gate. Same-grid copy gradient gap was 0.0; full source/subset difference is a separate scientific measurement.

## Final t32 cross-evaluation

|Trained on|Evaluated on|mean loss|H2|BeH2|CO|improvement from same-grid theta0|displacement norm|
|---|---|---:|---:|---:|---:|---:|---:|
|SUBSET|SUBSET|0.519907746244|0.0385628155898|0.221641124466|1.29951929868|0.748160369654|0.0535532127336|
|SUBSET|FULL_SOURCE|0.519928644565|0.0385674965765|0.221697973184|1.29952046393|0.748164025016|0.0535532127336|
|FULL_SOURCE|SUBSET|0.51990788572|0.0385628063282|0.22164121755|1.29951963328|0.748160230177|0.0535531569617|
|FULL_SOURCE|FULL_SOURCE|0.519928784036|0.0385674873154|0.22169806625|1.29952079854|0.748163885544|0.0535531569617|

SOURCE is a shared numerical evaluation grid on the same three training systems, not independent validation/test data. t16 is diagnostic only; no endpoint selection or early stopping used. t0/t16/t32 system values and all updates are saved in cross_evaluation/training_log tables.

## Endpoint fixed-weight gradient comparisons

|Fixed endpoint|norm SUBSET|norm FULL_SOURCE|absolute vector difference|relative difference|cosine|
|---|---:|---:|---:|---:|---:|
|SUBSET_t32|201.43834989875677|201.43631246883416|0.0028373991293733384|1.408583732792634e-05|0.9999999999519474|
|FULL_SOURCE_t32|201.43865247554078|201.43661503863174|0.0028375457170596342|1.4086543881386443e-05|0.9999999999519372|

These gradients are recomputed at each live t32 weight vector on both grids; theta0 gradients are not substituted. Full block results and endpoint vectors are hash-bound.

Endpoint parameter distance ||theta_FULL−theta_SUBSET||=2.33415339948e-06; displacement-vector cosine=0.999999999051. Native moment comparison: `{"exp_avg": {"norm_SUBSET": 128732.71364433767, "norm_FULL_SOURCE": 128733.3234634472, "absolute_difference": 0.6158402683959519, "relative_difference": 4.783845020289675e-06, "cosine": 0.9999999999997773}, "exp_avg_sq": {"norm_SUBSET": 407137971212.1403, "norm_FULL_SOURCE": 407141850014.4776, "absolute_difference": 3883404.897784395, "relative_difference": 9.538211062425306e-06, "cosine": 0.9999999999998923}}`. No inherited moments, optimizer reset or alternative optimizer.

## Resource and integrity receipts

Complete-batch preflight times: {'SUBSET': 25.53055340005085, 'FULL_SOURCE': 24.0692287000129} seconds. Projected remaining cost 2735.192s, remaining budget 10246.658s, reserve 900s; GO before updates. Preflight GPU allocator peak 276895232 bytes (not a whole-run peak claim). SOURCE preparation 17.986s.

Measured training batch totals: {'SUBSET': 824.9591010998702, 'FULL_SOURCE': 801.4604069000925}; FULL/SUBSET end-to-end ratio 0.971515. Grid points per update {'SUBSET': 191858, 'FULL_SOURCE': 222240}; full grid count ratio 1.15836. Every branch has 96 optimization system presentations, with two computational passes per gradient; full grid evaluates more points. Disk/indexing/warmup affect observed wall cost, so a grid-count ratio is not asserted to equal compute-time overhead.

Run timer (CPU/GPU/transfer/checkpoint work together) 1839.390s; scientific wall through final integrity 3965.275s / 10800s. This is not isolated GPU kernel time. All source/hash, cache, counters, fresh-checkpoint receipts, buffers, constraints and finite checks passed. Safe restore of model/optimizer/RNG/cursors was tested on new hash-bound artifacts, with no old checkpoint load. Atomic resume snapshots represent completed paired updates; no scientific resume was needed.

Eight scientific checkpoint files: shared initialization, two t0, two t16, two t32 and one atomic resume. Focused optimizer fixtures are not scientific training arms or additional R0 optimization exposures. Validation receipt: {"focused_and_selected_regressions": "16 PASS, 15 unrelated tests deselected", "pytest_seconds": 2.66, "Ruff": "PASS", "compileall": "PASS", "git_diff_check": "PASS", "historical_tracked_files_changed": 0}.

## One next step and limits

If the measured classification is B, close this selection-only intervention for the tested seed/horizon; the small gradient/trajectory differences do not support an automatic full90 rebuild. A separate bounded derivative/leverage qualification of the large fresh normalization-block sensitivity is the next justified diagnostic. If A, the next step is an independent limited qualification of the matched full-grid path. Neither result authorizes extra seeds, longer training, chemistry stages or architectural changes.

Chemical generalization, Clean28, SCF stability, scientific promotion and a general explanation of historical NN plateau were not tested. No new systems, target labels, geometry, basis or density. Production corpus/loss/architecture unchanged.
