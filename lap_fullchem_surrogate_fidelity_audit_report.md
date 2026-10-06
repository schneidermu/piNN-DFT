# Full251 chemistry surrogate-fidelity audit

Starting commit: `d40bbcb5e7797728cdaedf5bcc6ef4950a4c85f4`. Final commit: the Git commit containing this report.

**CASE A**. This is a local surrogate-fidelity diagnosis on twelve prospective saved states, not a scientific qualification or a training experiment.

The dominant bottleneck at the audited states is sampled chemistry direction fidelity: 6/12 accepted directions protect the sampled scalar while predicting and producing full251 ascent. Fullchem predicts every measured finite-step sign; 0/12 have fullchem-predicted descent overturned at the accepted displacement. Halving and quartering preserve the signs. Replacing only chemistry yields strict predicted four-task descent and actual full-eta chemistry descent in all twelve counterfactuals. No finite-step overshoot is needed to explain these selected failures.

Full chemistry scalar: Arithmetic mean of 251 corrected singleton batch_fchem losses; exact non-AE canonical variants from full268 manifest, F64 chemistry shadow on widened F32 stored state/source values. Scalar reductions match the Arena monitor (`sum/251`); F64 shadow gradients are accumulated with weight1/251. Non-AE RMSE is not substituted. The stochastic R2 gradients and other three task gradients are read from the exact saved geometry NPZ files; no AE/Exc/operator gradient is recomputed.

Protocol SHA: `a789813b6df1f2f40ffd5434b29c63afdf03cc3330068f88a8b86b8cd4a1af18`. Manifest byte SHA: `eb64fb2ba9a98eead1dc518a1b0853eeb90d66dcb92c117dbfba23964e6afcfc`; canonical SHA: `2e87f7e79a66b425628bdf59a821f785e05059f193a492d4a4f3130d7411e9f3`. Probe positions 0,5,9 were fixed before evaluation. All states are F32 and all chemistry arithmetic/leaves/gradients are F64 on original widened source values.

Provenance: 57 trajectory input files were checked directly against the hash inventory read from committed d40bbcb metrics; all twelve recovered before-state hashes match the anchored update receipts. All8 listed sampled reactions at positions0/5/9 use the same variants as full251, with database-count weights. No grid-variant change or gross weighting mismatch explains the observed directional errors. This does not by itself establish a statistical unbiasedness claim.

State identities:

| State | Tensor SHA256 |
| --- | --- |
| 11_P67_u0 | ee815caf918e8ded13e99896070a95a32293100499746fd0623444e2c0023523 |
| 11_P67_u5 | f43f0166d4483f0c08021ecf5e16984c774b569ce58920a7510806167ea494cf |
| 11_P67_u9 | 8bccb4bffd887ee676bf54365231370165d0218698be19340333baff4af430d4 |
| 11_P536_u0 | 3c2e45d6a86cccf80d7b4dc458e67f9b406935ed60d4e57433bb441a6df88da6 |
| 11_P536_u5 | 3fdd1714976299a06a8717aae1e14d6dd69770998e7cf626f105f9d80a1b7f6c |
| 11_P536_u9 | adda51706fe4c164fb7693d227cf1607c608cf822bcb958aeabe402145ebb61b |
| 23_P67_u0 | 69f6e7c90a5f4cf6f51e885b8f3c1b9679508c71a57a8363149de6783820e4d8 |
| 23_P67_u5 | 22e05f98bc8f34f61e8e9b7f60480d5e3da656ca51b2ae5c4f0dc7c0a173121e |
| 23_P67_u9 | bcfe3066bd1ab2342cb78c1829f28a2f9dc060e9217eb0e41c5ab65a1a149830 |
| 23_P536_u0 | 0b0c2aefc703c90be250c2fdb79e60be1df35273c7bc7d680ede50a8e7145074 |
| 23_P536_u5 | 8dbfd61b821408ee5cf055687a63e9d0a114b05bac86a45e11654344b5316a65 |
| 23_P536_u9 | 1efea2c4b5f312a93f30bcd05b95890b28a139fa2d1c3b7cce25560a5ebe387b |

Sample/full fidelity and actual-direction protection:

| State | Sample norm | Full norm | Norm ratio | Cosine | p_sample | p_full | Directional error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 11_P67_u0 | 20.3411161210458 | 10.9412956521699 | 1.8591140179099 | 0.414717022549111 | 0.330780948870034 | -0.136909396499223 | 0.467690345369256 |
| 11_P67_u5 | 26.3812189434657 | 10.9413574858663 | 2.41114678663448 | 0.981470657596131 | 0.223012876768994 | 0.228432020986104 | -0.00541914421711056 |
| 11_P67_u9 | 24.0249667066358 | 10.9415247191007 | 2.19576040117108 | 0.892998075090394 | 0.135926222975395 | 0.0436807190500966 | 0.092245503925298 |
| 11_P536_u0 | 18.2977926010344 | 9.62453223792005 | 1.90116175505572 | 0.781660289813684 | 0.312557148666124 | -0.253155226944756 | 0.56571237561088 |
| 11_P536_u5 | 25.7471843357891 | 9.62461343376168 | 2.67513957968139 | 0.771587493244841 | 0.225283978570627 | -0.0984776150592121 | 0.323761593629839 |
| 11_P536_u9 | 23.4886337601409 | 9.62468744438531 | 2.4404567832323 | 0.758736492561973 | 0.119922516355724 | -0.237420605815868 | 0.357343122171592 |
| 23_P67_u0 | 12.3108089442069 | 12.874764904114 | 0.956196795505998 | -0.104345205686537 | 0.78154161699175 | -0.670943296247327 | 1.45248491323908 |
| 23_P67_u5 | 44.6107573211784 | 12.8748055911573 | 3.46496550998937 | 0.954122867352435 | 0.0955090193576967 | 0.104774159494218 | -0.00926514013652088 |
| 23_P67_u9 | 46.829390284282 | 12.8748261846598 | 3.63728330096437 | 0.944256433527638 | 0.0745345635345831 | 0.0685459890167754 | 0.00598857451780767 |
| 23_P536_u0 | 13.6869549730319 | 15.3849587173735 | 0.889632219654637 | 0.822808696268752 | 0.420618467250228 | -0.105076800209955 | 0.525695267460183 |
| 23_P536_u5 | 58.9445820329378 | 15.3849337049689 | 3.83131855900689 | 0.979365995262345 | 0.153517986305697 | 0.151701204296934 | 0.00181678200876314 |
| 23_P536_u9 | 46.2206257500219 | 15.384995045582 | 3.00426653457356 | 0.96638863660947 | 0.0741870469173914 | 0.226686731483343 | -0.152499684565952 |

Counterfactual fullchem-informed UNIT_MAXMIN:

| State | Original gamma | Fullchem gamma | Rotation degrees | Four progress values | Active tasks | Strict common descent |
| --- | --- | --- | --- | --- | --- | --- |
| 11_P67_u0 | 0.330780948870035 | 0.245376823940614 | 39.9756744349411 | [0.2453768239406144, 0.24537682394061427, 0.26903503707913445, 0.2738409983748167] | ['relchem', 'ae17'] | True |
| 11_P67_u5 | 0.223012876768996 | 0.24537309010607 | 23.0863039886132 | [0.24537309010606825, 0.2453730901060707, 0.25826239176924487, 0.2836156056849022] | ['relchem', 'ae17'] | True |
| 11_P67_u9 | 0.135926222975396 | 0.243799748999322 | 68.3861486501543 | [0.24379974899932222, 0.24690088792384046, 0.24379974899932103, 0.35951568831595915] | ['relchem', 'exc'] | True |
| 11_P536_u0 | 0.312557148666124 | 0.147717168728416 | 53.0391714694115 | [0.14771716872841503, 0.14771716872841678, 0.16837036887837026, 0.14771716872841598] | ['relchem', 'ae17', 'op'] | True |
| 11_P536_u5 | 0.225283978570629 | 0.143955444537581 | 33.7358685173119 | [0.14395544453758047, 0.14395544453758233, 0.15798944675448587, 0.14395544453758016] | ['relchem', 'ae17', 'op'] | True |
| 11_P536_u9 | 0.119922516355725 | 0.249658912645159 | 91.5066643115233 | [0.24965891264515908, 0.24994942927685593, 0.24965891264515835, 0.24965891264515877] | ['relchem', 'exc', 'op'] | True |
| 23_P67_u0 | 0.78154161699175 | 0.160100687390975 | 62.8183942018855 | [0.160100687390976, 0.17786502542858323, 0.16010068739097502, 0.1601006873909748] | ['relchem', 'exc', 'op'] | True |
| 23_P67_u5 | 0.0955090193576916 | 0.157051651620348 | 35.801726400721 | [0.1570516516203516, 0.164559954814473, 0.1570516516203448, 0.15705165162034432] | ['relchem', 'exc', 'op'] | True |
| 23_P67_u9 | 0.0745345635345819 | 0.181506984610712 | 66.7071459096348 | [0.18150698461071155, 0.18150698461071268, 0.1816871204464977, 0.21117499735885906] | ['relchem', 'ae17'] | True |
| 23_P536_u0 | 0.420618467250228 | 0.182971263410633 | 21.7849253097049 | [0.1829712634106324, 0.18297126341063344, 0.19076957099198494, 0.18297126341063394] | ['relchem', 'ae17', 'op'] | True |
| 23_P536_u5 | 0.153517986305698 | 0.181349814457523 | 26.1625430358871 | [0.18134981445752105, 0.18134981445752482, 0.19446775533808483, 0.18134981445752252] | ['relchem', 'ae17', 'op'] | True |
| 23_P536_u9 | 0.0741870469173941 | 0.183317696603707 | 34.6986329160008 | [0.18331769660370748, 0.1833176966037073, 0.1833955584438171, 0.2211222828983144] | ['relchem', 'ae17'] | True |

Actual-direction finite chemistry response (temporary F32-rounded candidates):

| State | eta | Delta(eta) | Delta(eta/2) | Delta(eta/4) | Prediction at eta | Realized-displacement prediction at eta | Counterfactual Delta at eta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 11_P67_u0 | 7.25689494482748e-06 | 1.07927881698444e-05 | 5.48616213214004e-06 | 2.66480602628683e-06 | 1.08705832329682e-05 | 1.07926298868651e-05 | -1.95081219358517e-05 |
| 11_P67_u5 | 7.25689494482748e-06 | -1.81057973507315e-05 | -8.95857108540454e-06 | -4.40940346058127e-06 | -1.81375668450039e-05 | -1.81057830616478e-05 | -1.95063313754851e-05 |
| 11_P67_u9 | 7.25689494482748e-06 | -3.41207417986489e-06 | -1.85729759638598e-06 | -9.05268127349146e-07 | -3.46831441371852e-06 | -3.4121485449531e-06 | -1.93938035615293e-05 |
| 11_P536_u0 | 6.38354190985045e-06 | 1.54961446960922e-05 | 7.75766526728283e-06 | 3.82707378721037e-06 | 1.5553503967507e-05 | 1.54962595120384e-05 | -9.10011544918987e-06 |
| 11_P536_u5 | 6.38354190985045e-06 | 6.09768071568872e-06 | 3.08522442171721e-06 | 1.51657300473396e-06 | 6.05037832608658e-06 | 6.09766967827873e-06 | -8.84881985840913e-06 |
| 11_P536_u9 | 6.38354190985045e-06 | 1.45227605978793e-05 | 7.24767700255313e-06 | 3.57956748997523e-06 | 1.45870260251591e-05 | 1.45227906221282e-05 | -1.52806490860691e-05 |
| 23_P67_u0 | 8.5392826698708e-06 | 7.3774882261457e-05 | 3.68298931887612e-05 | 1.84502284219512e-05 | 7.37643492473112e-05 | 7.377469503294e-05 | -1.75655302567801e-05 |
| 23_P67_u5 | 8.5392826698708e-06 | -1.15363901664889e-05 | -5.71174435459731e-06 | -2.85049656723046e-06 | -1.1519039180052e-05 | -1.15364938757169e-05 | -1.72091753893877e-05 |
| 23_P67_u9 | 8.5392826698708e-06 | -7.56827088377321e-06 | -3.67237658815966e-06 | -1.88887247509761e-06 | -7.53606805233421e-06 | -7.56841649175103e-06 | -1.99084878262923e-05 |
| 23_P536_u0 | 1.02041872088837e-05 | 1.64607161636798e-05 | 8.36452564900725e-06 | 4.21270476502755e-06 | 1.64961118317484e-05 | 1.64607314436515e-05 | -2.88600826670038e-05 |
| 23_P536_u5 | 1.02041872088837e-05 | -2.38412734108184e-05 | -1.19174770929487e-05 | -5.91131003124801e-06 | -2.38156848860634e-05 | -2.38413459876364e-05 | -2.86044534663077e-05 |
| 23_P536_u9 | 1.02041872088837e-05 | -3.56648777262158e-05 | -1.77538724779325e-05 | -8.93657139666715e-06 | -3.55878604577017e-05 | -3.5664982282898e-05 | -2.8930779598646e-05 |

Summary: {"actual_full_eta_chemistry_ascent_count": 6, "actual_realized_prediction_sign_matches_finite_count": 12, "actual_requested_prediction_sign_matches_finite_count": 12, "cos_sample_full": {"max": 0.981470657596131, "median": 0.8579033856795731, "min": -0.10434520568653667}, "counterfactual_common_descent_count": 12, "counterfactual_full_eta_chemistry_descent_count": 12, "counterfactual_nondegenerate_count": 12, "fraction_cos_above_0_5": 0.8333333333333334, "fraction_cos_above_0_8": 0.5833333333333334, "fraction_cos_positive": 0.9166666666666666, "fraction_p_full_positive": 0.5, "full_requested_descent_finite_nondescent_count": 0, "gamma_change": {"max": 0.12973639628943434, "median": 0.02509602074444929, "min": -0.6214409296007749}, "maximum_absolute_realized_first_order_remainder": 1.8722851703257062e-10, "maximum_observed_repeatability_difference": 0.0, "maximum_relative_realized_first_order_remainder": 2.1794686836933667e-05, "median_p_full": -0.027398448004557735, "median_p_sample": 0.18826543153734532, "resolved_actual_full_eta_changes": 12, "resolved_finite_failure_with_requested_full_descent_count": 0, "rotation_degrees": {"max": 91.50666431152327, "median": 37.88870041783102, "min": 21.784925309704917}, "sample_descent_full_nondescent_count": 6, "sample_descent_full_nondescent_fraction": 0.5}.

Across all36 actual-direction trials, the maximum absolute remainder against the exact realized-displacement first-order prediction is 1.87228517032571e-10; the maximum relative remainder is 2.17946868369337e-05. Thus requested-versus-realized F32 displacement differences are recorded, but they do not reverse any of the fullchem-predicted signs here.

By prospective position: {"0": {"cosine": {"max": 0.8228086962687524, "median": 0.5981886561813972, "min": -0.10434520568653667}, "p_full": {"max": -0.10507680020995498, "median": -0.19503231172198915, "min": -0.6709432962473274}, "sample_full_nondescent_count": 4}, "5": {"cosine": {"max": 0.981470657596131, "median": 0.9667444313073899, "min": 0.7715874932448405}, "p_full": {"max": 0.22843202098610427, "median": 0.12823768189557566, "min": -0.0984776150592121}, "sample_full_nondescent_count": 1}, "9": {"cosine": {"max": 0.9663886366094703, "median": 0.918627254309016, "min": 0.7587364925619731}, "p_full": {"max": 0.22668673148334323, "median": 0.056113354033435994, "min": -0.23742060581586827}, "sample_full_nondescent_count": 1}}. Paired trajectory identities remain separate in the twelve-row tables; no initialization or seed is promoted.

Paired update0→5→9 progression: {"11_P536": {"cos_sample_full": [0.7816602898136835, 0.7715874932448405, 0.7587364925619731], "mismatch_count": 3, "p_full": [-0.2531552269447556, -0.0984776150592121, -0.23742060581586827], "positions": [0, 5, 9]}, "11_P67": {"cos_sample_full": [0.4147170225491107, 0.981470657596131, 0.8929980750903939], "mismatch_count": 1, "p_full": [-0.1369093964992227, 0.22843202098610427, 0.04368071905009663], "positions": [0, 5, 9]}, "23_P536": {"cos_sample_full": [0.8228086962687524, 0.9793659952623451, 0.9663886366094703], "mismatch_count": 1, "p_full": [-0.10507680020995498, 0.15170120429693376, 0.22668673148334323], "positions": [0, 5, 9]}, "23_P67": {"cos_sample_full": [-0.10434520568653667, 0.9541228673524347, 0.9442564335276382], "mismatch_count": 1, "p_full": [-0.6709432962473274, 0.10477415949421756, 0.06854598901677536], "positions": [0, 5, 9]}}. These are only three predeclared positions per arm: they establish local fidelity failures, not an exact decomposition of all ten accepted steps or the entire endpoint regression.

Position and stochastic sample change together, so this three-position comparison does not isolate causal trajectory drift. The mismatch already exists at update0 in all four starts; later failures are sample/state dependent rather than universally monotonic deterioration.

The norm ratio describes scale error; unit-gradient differences, directional protection and sign disagreements diagnose angular error. Do not attribute ascent to norm scale alone. Positive requested slope with a failed rounded finite step is a finite-step diagnostic, not automatically true curvature: the metrics separately preserve the full gradient dot exact realized displacement. Opposite sign under both requested and realized predictions supports estimator mismatch without requiring an overshoot explanation.

All full-eta actual candidates reproduce the original accepted after-state hashes exactly. Baseline and actual full-eta evaluations were repeated with the same numerical path; their observed repeat differences and signal-resolution labels are preserved per state. No invented epsilon floor is used.

Restoration is enforced by bitwise equality of every state tensor, the model digest and RNG state in a finally block after each perturbation. The stored zero restoration field denotes that successful exact-equality assertion, rather than a separately sampled maximum-error measurement. Counterfactual feasibility is first-order and only chemistry finite steps were evaluated; it does not certify finite-step decrease of all four objectives.

Validation: PASS. Relevant tests:128 passed,3 Windows-only Gloo skips; focused diagnostic tests:9 passed. Ruff,compileall and git diff --check passed. Immutable source/data/state checks:124 matched before/after,zero mismatches. Twelve accepted after-state hashes reproduce exactly; production-source changes:none. Pre-existing untracked planning files were preserved.

Independent review: PASS. Ponytail/Pocock/MOO/numerical/scientific review reconstructed all twelve QPs and all36 F32 actual trial displacements/hashes, including the two specified numerical cases. Full review receipt: [review.json](C:/Dev/readWFN_share_ms/lap_fullchem_surrogate_fidelity_runs_20261006/review.json). The receipt preserves the reviewed pre-embedding report/metrics SHA snapshot; final regeneration adds validation/review metadata and this editorial scope wording without changing results.

Next experiment: A preregistered chemistry-estimator variance-reduction/fidelity comparison on the same frozen states; keep UNIT_MAXMIN and the controller fixed. Not launched. Do not promote UNIT_MAXMIN to 25 updates on this audit.

No training trajectory was run, no model update was retained, no optimizer or aggregator behavior was changed, no new initialization or MOO method was introduced, no25/100-update continuation was launched, and every temporary finite-step perturbation was exactly restored. No full90, SCF, Diet, unrelated Slurm work, Nash expansion or per-database gradient localization was run.
