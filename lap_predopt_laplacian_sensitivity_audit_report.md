# Predopt Laplacian sensitivity audit

Starting commit: `7a10dd6d3ce645d602946875a551503d4f266948`, branch `lap_full_vxc`. Production changes: zero. Existing states audited: 30. New diagnostic tangent states: three. Expensive four-objective evaluations: zero.

## Conclusion

PBE predopt reduces the magnitude of Laplacian response in every matched seed, including continued contraction after P67. It does not selectively erase Laplacian dependence: density/gradient controls also contract, and q-descriptor sensitivity rank remains 12 in every non-head state. The audit measures local function/descriptor derivatives, not the full parameter tangent space. It therefore does not establish that P536 removes useful optimization directions.

PBE_TANGENT with epsilon=0.003 is physically valid and roughly matched to P536 in aggregate PBE deviation, but has substantially less q response than P536. It fails the predeclared cheap flexibility gate. No full251/AE17/15-system objective or gradient was evaluated for it. Do not replace canonical initialization on this evidence.

## Frozen selection and mathematical conventions

The fixed subset contains 5824 source-F32 grid rows, selected before sensitivity evaluation from the canonical268 corpus. Candidate rows are evenly spaced within each reaction; stratification uses five log-density bins, four log(1+|q|) bins, and three absolute spin-polarization bins. Both spin densities must be >=1e-12 and total density >=1e-8. Exact source row identities and subset tensor are SHA-bound. Repeated molecule appearances are not independent statistical samples. All states use identical points and scales.

Raw input columns are rho_a,rho_b,sigma_aa,sigma_total,sigma_bb,tau placeholders,lapl_a,lapl_b. PBE consumes the production sigma_total_to_standard conversion. Tau inputs are never active. Reported q is the unbounded correlation convention q_sigma=lapl_sigma/[4(3pi^2)^(2/3)(rho_sigma+EPS_RHO)^(5/3)]. Production exchange spin scaling and tanh descriptor transforms are retained exactly by model.forward.

At fixed rho/sigma, d epsilon/d q_sigma equals d epsilon/d lapl_sigma times the denominator above. Raw rho controls hold lapl fixed, not q. Coordinate-scale-normalized summaries multiply each derivative by its frozen subset IQR. XC sensitivities retain Hartree units; adaptive-output derivatives additionally divide by NN_OUTPUT_SCALE_PBE and are dimensionless. All raw and normalized channel statistics (RMS, median absolute, p75/p90/p99/max, fraction <=1e-12) are in structured results. The zero cutoff is descriptive, not a production acceptance tolerance.

Stored F32 checkpoint/source values are widened to F64 before model/PBE arithmetic and autograd. These are F64 diagnostic derivatives of the mathematical functional, not a claim that native source precision was recovered or that the operator production evaluator was changed. Actual XC epsilon, exchange, correlation, and learned correction epsilon-minus-PBE are traced separately.

## Matched q trajectories

Normalized combined spin q derivative RMS of epsilon_xc:

| Seed | P0 | P67 | P134 | P268 | P536 | P536/P0 | P536/P67 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 11 | 0.464216 | 0.172492 | 0.114623 | 0.0828948 | 0.0512133 | 0.110322 | 0.296903 |
| 23 | 0.343682 | 0.0518573 | 0.0269235 | 0.0172056 | 0.00947234 | 0.0275614 | 0.182662 |
| 41 | 0.782372 | 0.0550033 | 0.0412866 | 0.0254019 | 0.0127977 | 0.0163575 | 0.232671 |
| 73 | 1.08631 | 0.043946 | 0.0337058 | 0.0222979 | 0.01101 | 0.0101352 | 0.250534 |
| 101 | 0.847459 | 0.0566351 | 0.0447748 | 0.0381939 | 0.0254812 | 0.0300678 | 0.449919 |

All five q trajectories decrease monotonically. Endpoint contraction is 9.1–98.7-fold; P67-to-P536 contraction is 2.2–5.5-fold. Seed variation remains substantial (P536 maximum/minimum q RMS about 5.4), but does not reverse the paired trend.

| Regime | q RMS median | q RMS range | adaptive-output q RMS median | q covariance rank | q participation median | subset PBE MSE median |
|---|---:|---:|---:|---:|---:|---:|
| P0 | 0.782372 | 0.343682–1.08631 | 3.85442 | 12–12 | 2.76893 | 0.0415562 |
| P67 | 0.0550033 | 0.043946–0.172492 | 0.316818 | 12–12 | 2.65547 | 1.17022e-05 |
| P134 | 0.0412866 | 0.0269235–0.114623 | 0.185081 | 12–12 | 3.6496 | 3.17561e-06 |
| P268 | 0.0254019 | 0.0172056–0.0828948 | 0.14374 | 12–12 | 3.52343 | 1.55757e-06 |
| P536 | 0.0127977 | 0.00947234–0.0512133 | 0.105891 | 12–12 | 3.79775 | 4.49462e-07 |
| PBE_HEAD | 0 | 0–0 | 0 | 0–0 | 0 | 1.43648e-13 |

PBE function-space MSE from the previous full-corpus audit is retained only where it was already evaluated (14 states); missing old full-corpus metrics remain null. The new fixed-subset F64 MSE is available for all30. New full-corpus forward-only calibration supplies P536 references for seeds11/23/41. These different arithmetic/subset summaries are labeled and are not substituted for one another.

## Controls and conditioning

| Seed | q RMS P536/P0 | learned-correction rho RMS P536/P0 | learned-correction sigma RMS P536/P0 |
|---|---:|---:|---:|
| 11 | 0.110322 | 0.0188341 | 0.0474097 |
| 23 | 0.0275614 | 0.000886012 | 0.000953928 |
| 41 | 0.0163575 | 0.000103907 | 4.45406e-05 |
| 73 | 0.0101352 | 0.00171463 | 0.00169461 |
| 101 | 0.0300678 | 0.00251096 | 0.0029344 |

Controls show broader response contraction rather than q-specific collapse. Raw sigma derivatives can be enormous at exact-zero source sigma because of square-root regularization. Their RMS is conditioning-dominated and must not be compared across coordinates as an intrinsic physical magnitude. Robust quantiles and exact-zero/all-positive source-sigma stratifications are reported; no threshold or gate is changed. A supplemental fixed-q density chain correction is calculated from stored derivatives and separately labeled.

## Descriptor-Jacobian richness and saturation

On the same256 frozen rows, the normalized adaptive-output descriptor Jacobian is flattened as 9x7 (all coordinates) or9x2(q). Centered covariance eigenvalues, numerical rank, participation ratio and leading variance share are recorded. The q rank stays12 throughout P0–P536. Participation ratio is nonmonotonic and increases from P0 to P536 in four of five seeds. Covariance magnitude falls, but the number of resolvable response patterns does not collapse.

This is descriptor-sensitivity covariance, not parameter-Jacobian rank and not a count of trainable descent directions. A small epsilon_q derivative need not imply a small derivative of that response with respect to model parameters.

Stored-output activation slopes show strong kappa saturation near the PBE ceiling. Beta and Gx activation slopes approach their unsaturated values near1 after predopt. Thus all-output activation saturation is not an adequate explanation; kappa-specific saturation is a real contributor. No hidden-layer conditioning diagnosis is claimed from these slopes alone.

## PBE_HEAD and PBE_TANGENT

All five PBE_HEAD states have exactly zero q and adaptive descriptor derivatives, nonzero aggregate output-head gradients, and exactly zero hidden gradients. Zero head weights block backward propagation into hidden layers; nonzero head gradients depend on their random hidden features. This construction-specific degeneracy is not evidence that those random parameter features are absent.

The trigger for a bounded tangent experiment passed. PBE_TANGENT keeps each raw-seed hidden tensor bitwise unchanged, uses the same finite near-PBE biases as PBE_HEAD, and sets each output-head weight to epsilon times its original raw F32 value. No representation training, new architecture, optimizer, or modules are introduced. Kappa is near-PBE with the inherited one-ppm sigmoid offset, not analytically exact.

Frozen epsilon grid: 1e-4,3e-4,1e-3,3e-3,1e-2. Selection uses only adaptive9 PBE MSE: shared epsilon minimizing median across seeds11/23/41 absolute log MSE ratio to P536, with smaller-epsilon tie break. Subset calibration was provisional; full canonical268 forward-only calibration confirms the sameepsilon0.003. No objective values, gamma or gradient conflict entered selection.

Full-corpus calibration caches only head-independent exchange hidden tensors and pre-output correlation features within each chunk. The ordinary forward_descriptors/activation equations and actual F32-constructed heads widened to F64 are used. All five epsilon candidates are bitwise equal to uncached ordinary forward on the first canonical chunk. Each seed covers21,073,642 rows. An initial4096-chunk pass was interrupted before a completed seed result; the final65536-chunk diagnostic improves throughput. This changes no production chunk or mathematical functional.

| Seed | full-corpus P536 MSE | tangent MSE | MSE ratio | q RMS tangent | q RMS P536 | q ratio |
|---|---:|---:|---:|---:|---:|---:|
| 11 | 7.29079e-07 | 2.57351e-07 | 0.352981 | 0.00166864 | 0.0512133 | 0.0325822 |
| 23 | 4.29959e-07 | 3.83121e-07 | 0.891066 | 0.001313 | 0.00947234 | 0.138614 |
| 41 | 4.87876e-07 | 8.61749e-07 | 1.76633 | 0.00517905 | 0.0127977 | 0.404687 |

Physical anchor, finite derivative, tau independence, spin symmetry, deterministic replay, nonzero hidden-gradient flow, near-PBE deviation and local-energy checks pass for the constructed candidates. The q-response gate fails for both required seeds11/23. No expensive full251/AE/Exc/operator values or geometry are available for tangent states; they are not inferred.

Aggregate adaptive MSE matching does not establish identical channel or local-energy behavior. It is a bounded parent-function proximity check. The tangent candidate is not numerically pathological; it fails the intended flexibility improvement. It should not be forced into a production or optimizer comparison on this evidence.

## Hypotheses and next gate

H1: q magnitude suppression is supported, but Laplacian-specific suppression with richness collapse is not. H2: broader response flattening is supported, with raw-coordinate conditioning limitations. H3: no amplitude suppression is contradicted, while preserved descriptor richness is supported. H4: seed variation is significant but is not the sole explanation because all paired depth trends contract.

Arena classification is mixed/outside a decisive A–D choice: this audit does not prove useful parameter-tangent suppression, and the tested PBE_TANGENT fails its cheap gate. No Arena ran and no production initialization is selected. Keep canonical/weak-predopt controls; do not replace P536 by the tested tangent construction.

The single next experiment is a bounded parameter Jacobian sketch of d epsilon_xc/d q with respect to theta, at matched P67/P536/PBE_TANGENT states on a small fixed descriptor subset. That measures whether departures from PBE are actually hard to learn, rather than conflating current descriptor response with trainable tangent capacity. No full-objective matrix or training is needed for that gate.

## Provenance, tests, review and scope

Stage1 protocol SHA: `95dc7b0b57edd74fc0f6298de2774f93a80201b388e197f2ac986f6f879df33c`. Subset SHA: `31217fc4529bc0ece591bb359de55465493f1119f22a5a27866a1ff1c777e9d2`. Results bind all30 stored state files, diagnostic arrays, source/data manifests and external scripts. Source/state/input hashes pass; existing state files remain unchanged. Numerical helpers use autograd, not finite differences.

Initial sensitivity receipts omitted an execution-time evaluator hash guard. Original receipts/arrays/source are preserved under pre_guard; a hash-bound read-only replay of all30 cheap local states checks exact equality of every numeric result. New receipts enforce the states-manifest SHA and evaluator binding. This did not resume the expensive old initialization matrix.

Canonical calibration imported the original audit helpers before the later provenance-only run() edit. The exact original helper source is preserved in pre_guard/audit.py and matches its bound dependency SHA. Numerical helper bodies are unchanged. Current sensitivity execution has the new guard binding; these source identities are distinguished, not falsely relabeled.

Five diagnostic tests and25 existing model/operator tests passed. All30 existing states additionally passed manufactured F32/F64 anchor, finite-derivative, tau-independence and deterministic replay checks. Tangent physical gates were rederived from saved receipts. External tooling compileall and git diff --check passed. All production-source, input, state-manifest, existing state-file and array hashes passed. There are zero production-source changes; the only repository metadata change preserves byte identity for the three new JSON artifacts across platforms.

Independent GPT-6 Luna MAX review: PASS, no substantive mathematical or scientific blocker. The reviewer checked chain rules, descriptor conventions, fixed scales, cached-forward parity, source/replay bindings, full-corpus epsilon calibration, and the correctly rejected expensive gate. Its report-completeness notes are resolved here. The exact canonical-calibration protocol is embedded in the tracked `lap_pbe_tangent_initialization_protocol.json`, alongside the original frozen plan, selected epsilon, state hashes and physical receipts.

No multi-objective training trajectory was run, no production optimizer was changed, no old cursor10 state was advanced, and the unfinished30-state expensive initialization matrix was not resumed. No full90, SCF, Diet, Slurm or100-update run was launched.
