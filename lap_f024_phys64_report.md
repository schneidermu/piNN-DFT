COMPLETED РІР‚вЂќ F024 physics-only pilot, classification C

One matched physical continuation from jointly trained B8, with inherited full-history AdamW moments. The only intervention was absence of the fresh relative-chemistry term. No model promotion, chemistry stage or Clean28 evaluation.

Both arms have64 AE17 and64 paired mRKS optimization presentations, counter128РІвЂ вЂ™192. F010 t64 additionally has512 fresh relative-reaction presentations; F024 has0. The comparison is matched by physical exposure and optimizer updates, not total compute or chemistry exposure.

| Checkpoint | JW E251 | Legacy E251 | AE17 E17 | Exc full90 | Operator full90 | MAE |
|---|---:|---:|---:|---:|---:|---:|
| B8 | 0.156414638139 | 1.195939284232 | 3.275464955840 | 14.679044990003 | 0.030933124718 | 8.536567892 |
| F010_t64 | 0.146818733084 | 1.264111903650 | 2.378211915468 | 4.704043037478 | 0.029475733365 | 7.884063652 |
| F024_t64 | 0.153697199022 | 1.237476007427 | 2.968825923493 | 4.747579159267 | 0.029540298932 | 8.690947144 |

Operator effect floor: False; threshold 0.02918097603147132. System statistics: {'mean': 0.029540298931596813, 'median': 0.02888336194129888, 'B8_median': 0.03041443053190144, 'control_median': 0.02887475056841811, 'delta_B8': -0.0013928257860030303, 'delta_control': 6.456556647426934e-05, 'relative_control_change': 0.002190465142104525, 'improved_B8': 86, 'improved_control': 28, 'regressed_B8': 4, 'regressed_control': 62, 'largest_improvements_control': [{'identity': 'mrks_012317d17db00bc547be5b46', 'loss': 0.03993728157751011, 'B8_loss': 0.039796738799509776, 'control_loss': 0.04027600842635225, 'delta_B8': 0.00014054277800033127, 'delta_control': -0.00033872684884213994}, {'identity': 'mrks_5ee423e3e8560207b3f8304b', 'loss': 0.037065921562094144, 'B8_loss': 0.037177855858715796, 'control_loss': 0.03735418304447905, 'delta_B8': -0.00011193429662165189, 'delta_control': -0.0002882614823849078}, {'identity': 'mrks_fba98748e406529285195a52', 'loss': 0.02892689285846412, 'B8_loss': 0.02889301186105567, 'control_loss': 0.029203354580197065, 'delta_B8': 3.388099740844752e-05, 'delta_control': -0.0002764617217329464}, {'identity': 'mrks_4d8fc59e0b3a27aa6b490309', 'loss': 0.03448900760668273, 'B8_loss': 0.0349347316425907, 'control_loss': 0.034762291753958986, 'delta_B8': -0.00044572403590797255, 'delta_control': -0.00027328414727625927}, {'identity': 'mrks_5dff3c42939bbbbba2033b16', 'loss': 0.04757583504636346, 'B8_loss': 0.0505131160547951, 'control_loss': 0.04777870297299512, 'delta_B8': -0.002937281008431644, 'delta_control': -0.0002028679266316627}], 'largest_regressions_control': [{'identity': 'mrks_43229092afb7f9a49b1d7dd1', 'loss': 0.02730569706805337, 'B8_loss': 0.030446985365151547, 'control_loss': 0.02681538393829522, 'delta_B8': -0.0031412882970981784, 'delta_control': 0.0004903131297581478}, {'identity': 'mrks_6a88b1aac58d1a15df8adaaf', 'loss': 0.028884569619567722, 'B8_loss': 0.03157903920775424, 'control_loss': 0.028487632674267516, 'delta_B8': -0.0026944695881865183, 'delta_control': 0.0003969369453002067}, {'identity': 'mrks_53f381bbc70ce684a0992992', 'loss': 0.030924449182986747, 'B8_loss': 0.0335638019277282, 'control_loss': 0.03059925456751514, 'delta_B8': -0.00263935274474145, 'delta_control': 0.0003251946154716086}, {'identity': 'mrks_38d4606ba7998ec2dea207d5', 'loss': 0.04175580041668579, 'B8_loss': 0.044956523944426764, 'control_loss': 0.04143498755036436, 'delta_B8': -0.0032007235277409715, 'delta_control': 0.0003208128663214349}, {'identity': 'mrks_3b87ab99603db60594a03181', 'loss': 0.03965695033980424, 'B8_loss': 0.04269954646846355, 'control_loss': 0.03933848189149224, 'delta_B8': -0.003042596128659314, 'delta_control': 0.000318468448311994}], 'top5_gross_improvement_share': 0.4118329119780487}.
The operator metric is the equal-system full90 mean of ||S^-1/2(V_predРІв‚¬вЂ™V_ref)S^-1/2||_FР’Р†/nAO (HaР’Р†), with the original physically fixed gauge and V=dE_xc/dP_total. No square root, extra electron weighting or changed overlap convention is used. Reaction and atomic errors are kcal/mol; historical task factors and the original Exc kcal/mol conversion remain unchanged.

H signed/absolute error: -1.777054030656/1.777054030656 kcal/mol. Preservation: False. Stored reference and fixed evaluation variant unchanged.
P536 ratios: {'relchem': 1.0016227895124377, 'ae17': 0.11922494307534826, 'exc': 0.051506256598395556, 'op': 0.8911580099623544}; eligibility False. Failures are handoff recovery debt, not broken architectural constraints.

| Database | MAE | RMSE | РћвЂќ MAE vs B8 | РћвЂќ MAE vs F010 t64 |
|---|---:|---:|---:|---:|
| ABDE4 | 3.236204574 | 3.912008786 | +0.235706037 | -0.641748672 |
| DBH76 | 7.710192903 | 9.108933929 | -0.147776096 | +0.278496104 |
| EA13 | 2.670128513 | 3.166540982 | +0.109536046 | -0.252065571 |
| IP13 | 3.468475916 | 4.631415684 | -0.037041343 | +0.264439034 |
| MGAE109 | 13.662495798 | 17.372036429 | +0.328188930 | +1.896445827 |
| NCCE31 | 0.675852966 | 1.032559753 | -0.041221434 | +0.059354873 |
| PA8 | 2.235961613 | 2.748494274 | +0.521166338 | -0.663382544 |
| pTC13 | 7.430513239 | 8.167845052 | +0.790812031 | -0.665400332 |

Population chemistry statistics: {'mae': 8.69094714376815, 'rmse': 12.401312419501444, 'median_abs': 5.908073169839417, 'p90_abs': 19.844831124373627, 'p95_abs': 26.859426973622906, 'signed_mean': 3.4410955968159245}; macro DB MAE 5.136228190. All251 reactions and17 atoms use exactly one frozen variant. Per-record changes are saved.

Physical gradient norms/cancellation on E17/full90: {'B8': {'P_norm': 0.7506759736759908, 'cancellation_ratio': 0.9411952945691879, 'ae17_component_norm': 0.2877941162405871, 'exc_component_norm': 0.30371084045950864, 'op_component_norm': 0.20607231340259502}, 'F024_t64': {'P_norm': 0.1721415608215993, 'cancellation_ratio': 0.37880573773179377, 'ae17_component_norm': 0.22668425507941647, 'exc_component_norm': 0.07219356760067727, 'op_component_norm': 0.15555447249142462}}. Individual raw norms, projections and the unchanged-F010 reintroduction geometry are in endpoint_geometry.csv. Lower sum norm can reflect cancellation, not convergence.
Matched-population old/new gradient cosines: {'JW': 0.9998325283072189, 'legacy': 0.9968603483282683, 'ae17': -0.9987437948370497, 'H': 0.9999266315698812, 'exc': -0.9997923249973263, 'op': 0.9202932553302375, 'physical': -0.8038048587365072, 'F010_joint': 0.16438116791712892}. Training exposed 17 atomic identities and 64 unique mRKS systems; full90 outcomes come from the complete endpoint evaluation, not sampled coverage.

| State/direction | Legacy negative-gradient derivative | H negative-gradient derivative | JW derivative | Operator derivative |
|---|---:|---:|---:|---:|
| B8/physical | +9.9512911645 | +224.0379997360 | -0.4904911827 | -0.3811408277 |
| B8/F010_reintroduction | +12.5139717817 | +260.3832518026 | -0.8782908085 | -0.4995407802 |
| F024_t64/physical | -1.1619061532 | -40.7854379752 | -0.0218427679 | +0.0527887863 |
| F024_t64/F010_reintroduction | +2.0312775957 | -5.1464660088 | -0.3947220864 | -0.0152474422 |

Inherited optimizer accounting at t32: {'local_update': '32', 'counter': '160', 'beta1_power': '0.03433683820292515', 'beta2_power': '0.9684910757595268', 'inherited_m_norm': '0.005253676375846482', 'new_m_norm': '0.15917351398630658', 'inherited_v_norm': '0.027299155105450074', 'new_v_norm': '0.002120129055740768', 'm_recurrence_relative_gap': '5.5197416140444165e-08', 'v_recurrence_relative_gap': '3.943282779895698e-07', 'history_update_norm': '2.4193899966815457e-05', 'fresh_update_norm': '0.0003153085443536341', 'decay_displacement_norm': '5.296159895703368e-06', 'update_decomposition_gap': '6.60145906655266e-07'}
At t64: {'local_update': '64', 'counter': '192', 'beta1_power': '0.0011790184577738603', 'beta2_power': '0.9379749638258454', 'inherited_m_norm': '0.00018039463568797088', 'new_m_norm': '0.10610821885562971', 'inherited_v_norm': '0.02643898809540352', 'new_v_norm': '0.004153318093189329', 'm_recurrence_relative_gap': '7.908524307130596e-08', 'v_recurrence_relative_gap': '9.206280068355233e-07', 'history_update_norm': '9.069513915902413e-07', 'fresh_update_norm': '0.0002609709176194685', 'decay_displacement_norm': '5.296133392897716e-06', 'update_decomposition_gap': '6.685559211707322e-07'}
Actual native-F32 decay/step accounting: t32 {'local_update': '32', 'actual_F32_decay_displacement_norm': '5.09920667602449e-06', 'actual_F32_trace_step_norm': '0.0003078375467869385', 'logged_step_norm_gap': '0.0'}; t64 {'local_update': '64', 'actual_F32_decay_displacement_norm': '5.099328639687052e-06', 'actual_F32_trace_step_norm': '0.00026095450614196746', 'logged_step_norm_gap': '0.0'}. A clone-only arithmetic trace from saved native inputs reproduced weights and both moments bitwise at t32/t64. No optimizer, objective, model or fresh gradient was evaluated by that trace. The decay norm in moment_history.csv is the ideal F64 expression; native_decay_accounting.csv records the actual rounded multiplicative displacement.
Beta powers are attenuation factors, not causal percentages. First-moment update terms share the actual bias correction/second-moment denominator; no additive causal second-moment decomposition is claimed. Native recurrence residuals and actual decay/update rounding gaps are recorded.

Training active 580.11s; endpoint active 1320.31s; missing control scalars active 269.88s; scientific wall 2676.89s /10800s. Reporting/idle time is not GPU compute. Peak allocated/reserved 13.122/13.234GiB.

Integrated Exc residual sign changes: 47/90. These absolute-value objectives can retain gradient norms after scalar losses decrease. S64-prefix endpoint gradients were omitted; primary geometry uses exactly E251/E17/full90, without mixing prefix gradients.
One clone-only native-F32 full-population AdamW proposal from saved t64 moments: displacement norm 0.000286971114, linear changes {'JW': -4.6585779825217003e-05, 'legacy': 0.0002364757361859582, 'ae17': -0.20757085647748916, 'H': -0.008993104106074578, 'exc': -0.21399479438901797, 'op': 5.46977780181373e-06}. This uses unchanged F010 coefficients and hypothetical counter193; no optimizer/model update is retained. It is not the next historical minibatch and does not prove finite-step safety.
The endpoint physical direction is locally more compatible with legacy/H than B8. However, reintroducing unchanged JW reverses the legacy-descent margin, and the saved-moment native proposal predicts both legacy and operator increases. This is useful geometry evidence, not a qualified handoff or a proof that physics-first training generally fails.
Interpretation is limited to this parent, inherited optimizer history, frozen stream and64-update horizon. No capacity-ceiling claim or automatic second stage is justified.
One next recommendation: Read-only matched operator/JW/legacy native-direction attribution from existing B8, F010 t64 and F024 t64 gradients and optimizer states. Do not launch a chemistry stage: the operator effect floor failed and the unchanged-F010 handoff still predicts legacy/operator regressions.

Architecture: R0
Training objective: fixed AE17 + Exc + operator
Fresh relative-chemistry gradient: ABSENT
Inherited optimizer moments: PRESERVED
Optimizer updates: 64/64
Relative-chemistry optimization presentations: 0
AE17 optimization presentations: 64
mRKS optimization presentations: 64
Primary endpoint: t64
Operator effect floor passed: NO
H preservation: FAIL
Original P536 eligibility: FAIL
Second training stage executed: NO
New chemistry coefficients selected: NO
Clean28 evaluated: NO
SCF executed: NO
New training data: NO
Architecture modified: NO
Scientific classification: C
Commit / push / verified remote HEAD: returned after commit verification

Final integrity: all frozen source, checkpoint, stream, evaluation and dataset hashes verified. 40 focused/regression tests passed in10.16s; Ruff, compileall and git diff --check passed. Final scientific wall 2676.89s; instrumented active GPU/CPU phases 2205.79s. Budget respected; no second stage, development data or SCF.
