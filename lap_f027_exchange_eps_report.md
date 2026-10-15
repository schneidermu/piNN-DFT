# F027 — independent seed-2027 matched epsilon replication

Classification **A**; technical **VALID**. Matched B2027=0.187999247811, with the predeclared 1% floor True. Exactly 32 CONTROL and 32 INTERVENTION retained steps; no extra molecular preflight steps or seed-2026 weights/moments loaded.

## Frozen setup and initialization

One constructor initialization followed Python/NumPy/Torch CPU/CUDA seed2027, deterministic algorithms and disabled CUDA/cuDNN TF32. Exact R0 constructor 6 layers, width32, dropout0, Gx/Gc enabled, 9446 unique coordinates. Its newly saved initialization artifact is the sole template for all copies/restores; no historical initialization is loaded. Parameter/buffer equality, order and independent storage pass. Tensor hashes match while scientific hashes differ because all five genuine exchange LNs use 1e-5 versus 1e-3. Every correlation LN remains 1e-5. No topology/activation/anchor/PBE-regularization changes. Exchange hidden features still enter correlation.

Data: authenticated original H2/BeH2/CO FULL_SOURCE grids 47080/89200/85960, identical frozen densities, basis/AO order, quadrature, overlap, gauge and targets. Independent full-grid reference reprojection passes frozen 1e-10 criteria. No grids, densities, labels or systems generated. Source/manifest hashes are in protocol/provenance/source_verification.json; historical F026 and earlier files remain unchanged.

The complete operator matrix precedes the original overlap-normalized squared loss, with equal 1/3 system weights. Original bounded two-pass adjoint, F64 learned shadow/AO/gradient accumulation and original F32 PBE path are unchanged. One final gradient cast feeds native F32 AdamW with fresh independent moments, lr2.5e-5, betas.9/.999, eps1e-8, decay.01, foreach false. No energy/chemistry tasks, scheduler, clipping, gradient surgery or resets.

## Numerical qualification and checkpoint contract

F026 helper code is loaded in isolated per-arm namespaces. Only namespace/output paths, wall-budget callback and explicit epsilon default are specialized; the derivative mathematics is unchanged. The five modules and all correlation eps are asserted on native, F64 shadow, every load_state_dict and numerical copies. Every checkpoint stores architecture/order/shapes, per-module epsilon, separate parameter/buffer hashes, scientific hash, optimizer/config/counter, CPU RNG ByteTensors, data hashes and protocol hash. Missing metadata fails closed. Checkpoints t0/t1/t2/t16/t32 and atomic interleaved pair cursor live outside Git with SHA receipts.

t0 preflight includes two-pass/direct small-fixture parity, full mean-of-three gradient closure, full-batch cost profiles, independent LN first/mixed derivatives on actual zero/near-zero/ordinary hidden inputs, independent energy-first dE/dP path, and the fixed H2 coordinates x_feature_extractor.1.bias[6], .3.fc.1.bias[16], .3.fc.0.weight[887]. F026's prospectively fixed seven-step curvature-aware schedule and tolerances are unchanged. Both retained update1 states were qualified and restored before update2; no temporary molecular step was executed. Final checkpoints also reproduce full losses and gradients after restoration. Derivative statuses: [{'theta': 0, 'name': 'x_feature_extractor.1.bias', 'index': 6, 'status': 'QUALIFIED', 'arm': 'CONTROL'}, {'theta': 0, 'name': 'x_feature_extractor.3.fc.1.bias', 'index': 16, 'status': 'QUALIFIED', 'arm': 'CONTROL'}, {'theta': 0, 'name': 'x_feature_extractor.3.fc.0.weight', 'index': 887, 'status': 'QUALIFIED', 'arm': 'CONTROL'}, {'theta': 0, 'name': 'x_feature_extractor.1.bias', 'index': 6, 'status': 'QUALIFIED', 'arm': 'INTERVENTION'}, {'theta': 0, 'name': 'x_feature_extractor.3.fc.1.bias', 'index': 16, 'status': 'QUALIFIED', 'arm': 'INTERVENTION'}, {'theta': 0, 'name': 'x_feature_extractor.3.fc.0.weight', 'index': 887, 'status': 'QUALIFIED', 'arm': 'INTERVENTION'}, {'theta': 1, 'name': 'x_feature_extractor.1.bias', 'index': 6, 'status': 'QUALIFIED', 'arm': 'CONTROL'}, {'theta': 1, 'name': 'x_feature_extractor.3.fc.1.bias', 'index': 16, 'status': 'QUALIFIED', 'arm': 'CONTROL'}, {'theta': 1, 'name': 'x_feature_extractor.3.fc.0.weight', 'index': 887, 'status': 'QUALIFIED', 'arm': 'CONTROL'}, {'theta': 1, 'name': 'x_feature_extractor.1.bias', 'index': 6, 'status': 'QUALIFIED', 'arm': 'INTERVENTION'}, {'theta': 1, 'name': 'x_feature_extractor.3.fc.1.bias', 'index': 16, 'status': 'QUALIFIED', 'arm': 'INTERVENTION'}, {'theta': 1, 'name': 'x_feature_extractor.3.fc.0.weight', 'index': 887, 'status': 'QUALIFIED', 'arm': 'INTERVENTION'}]. Any unresolved necessary check forces classification D rather than claimed replication. Old F026/audit classifications are not rewritten.

## Matched results

Mean losses: CONTROL t0=0.906519217270033, t32=0.1967657466653052; INTERVENTION t0=0.9006528368630952, t32=0.1597739342972301. Different t0 losses with identical tensors reflect different mathematical functions, not initialization mismatch. Initial gradient norms 1907748.27651 versus 202.584290377; startup reduction factor 9417.05930388. Lower startup norm is not the endpoint success metric.

|System|C t0|I t0|C t32|I t32|I−C t32|relative benefit|
|---|---:|---:|---:|---:|---:|---:|
|H2|0.279099575224|0.277535608976|0.0658045261797|0.0294588876361|-0.03634563854|55.23273345%|
|BeH2|0.508991339341|0.505641916485|0.10420811527|0.061028268924|-0.04317984635|41.43616477%|
|CO|1.93146673725|1.91878098513|0.420284598546|0.388834646332|-0.03144995221|7.48301325%|

Maximum final system losses: CONTROL 0.420284598546, INTERVENTION 0.388834646332. Individual regressions, if present, remain explicit in system_comparison.csv; all systems improving is not an extra gate. The result is a uniform mean over three training systems, not a held-out generalization score.

|Arm|update|grad dot actual displacement|realized loss change|
|---|---:|---:|---:|
|CONTROL|1|-203.3834841|-0.1682839551|
|INTERVENTION|1|-0.03320279973|-0.03212621226|
|CONTROL|2|-0.01059667481|-0.0156658776|
|INTERVENTION|2|-0.03171480322|-0.03292167632|

Negative dot predicts local descent. Milestone metrics contain complete gradients, exchange-LN/outside-LN norms, moments, displacement, finite/configuration and actual hard-constraint checks at all five predetermined states. No earlier checkpoint selection. Optional read-only endpoint epsilon swaps: [{'weights': 'CONTROL_t32', 'epsilon': 0.001, 'mean_loss': 0.33651556446423686, 'system_losses': [0.13165926900394126, 0.19986329341037953, 0.6780241309783899], 'read_only': True}, {'weights': 'INTERVENTION_t32', 'epsilon': 1e-05, 'mean_loss': 0.3527239965356929, 'system_losses': [0.1322227714477409, 0.20856528649502262, 0.7173839316643152], 'read_only': True}]. They do not choose a model or imply causal startup-memory percentages.

## Cross-seed conclusion and limits

F026 seed2026 within-seed B=22.8140561503%; F027 seed2027 within-seed B=18.7999247811%. Different seeds' absolute losses are not treated as matched starts. Two seeds do not justify a confidence interval or universal efficacy claim. UEG recovery/Gx anchors, spin symmetry, high-density/rapid-gradient limits, tau independence and finite variational operators pass on both actual epsilon-configured functions and restored checkpoints.

Next action: One separately designed broader matched qualification covering full90 operator, original AE17/Exc/relative-chemistry metrics and H accuracy; no automatic full90 training or curriculum. Full90 operator, unweighted AE17 and Exc, original relative chemistry, and H accuracy remain mandatory before broad scientific qualification. Either fixed joint training or a separately qualified physics-first→chemistry strategy would require a new protocol. No automatic continuation, seed search, epsilon sweep, chemistry, full90 rebuild, SCF or Clean28 access occurred.

Scientific workflow wall at report generation 2846.912s, budget10800s, respected True. Recorded process timings are separate; GPU-active time was not metered. Tests/lint/compile/diff-check results are recorded in the final validation receipt.

## Final validation

14 selected tests passed (19 deselected); Ruff, compileall and git diff --check pass. All 12 predetermined t0/t1 coordinate checks qualified, all 20 original/restored constraint checks pass. Historical/production hashes remain unchanged. The Markdown-only CP1251 export failure was corrected with explicit UTF-8; no training or optional swaps were repeated. See validation.json and serialization_recovery.json.

Measured preflight plus training execution wall: 2307.5431976 s; retained training batches: 1603.994579299997 s. These are execution wall measurements, not GPU-active time. Workflow wall through final validation: 2938.440873861313 s, within 10800 s.
