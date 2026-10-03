# Frozen chemistry precision trace

**Analysis repository:** branch `lap_full_vxc`, HEAD `fd65500da7d8a34a9106b2405ad14468a62228d8`
**Changed LOC:** production 0; source 0; tests 0.

**Classification:** Frozen-sample chemistry-path precision failure: F32 model/local-energy evaluation reverses the matched-F64 descent sign; F32 XC/HF/D3 component assembly produces the production sign flip.
**Review status:** review_approved
**Frozen protocol SHA-256:** `7293dd32bd5cde8c0fda8fbd1845c6c9f52e654dc0536fda9e504fc32a0e8968`

The first full capture is retained as provenance; its process ended nonzero in stale post-capture code. Acceptance uses the separate V2 capture and receipt.

## Frozen reaction and operation path

- Sample/checkpoint: `NCCE31` reaction `0`, `BH`, `level2_mura`, cursor `2`; checkpoint SHA-256 `a53f46807b1ce492c8fe02657e615881ff310294bd79d273eb4bb0c694e3670c`.
- Exact states only: `baseflat32` F32 `750ac52a10dc2a6ea319b9ef02b8c2fb1dd8eafd9481300be52de2d25fb36e08` and `candidate_t_2m15` F32 `cf65691418ce4cbe88f143ec2aa786db884a4d34515b271b571f0c3d328c51d6`, stored step `t=3.0517578125e-05`; matched F64 widens these same states and F32-loaded sources.
- Components/order/counts: `NH3-NH3_ncce31` (52488 points), then `NH3_ncce31` (26248 points); stored backsplit endpoints `[52488, 78736]`.
- Coefficients `[-1.0, 2.0]`; reaction indices stored=False, code fallback `[0, 2]`; HF Hartree `[-97.13409423828125, -48.567176818847656]`; D3 Hartree `{'NH3-NH3_ncce31': -0.00192936, 'NH3_ncce31': -0.0005886}` (source numpy.float64); target kcal/mol `[3.1500000953674316]`.
- Source grid tensors: `Grid` torch.float32 [78736, 9], `Densities` torch.float32 [78736, 2], `Gradients` torch.float32 [78736, 3], `Weights` torch.float32 [78736].
- Production order: requested-dtype reaction tensors в†’ adaptive NN constants в†’ local PBE в†’ stored component split в†’ `epsilon * (rho_a + rho_b) * weight` reduction в†’ HF add в†’ D3 add в†’ ordered stoichiometry в†’ 627.5095 kcal/mol per Hartree в†’ `batch_fchem`.
- Preserved weight path: scalar Database `NCCE31` is zipped unchanged, yielding key `N` and fallback factor `8.979622680758972`; listed NCCE31 factor would be `2.896652477664184`.

## Production trace

| Stage | F32 base | F32 candidate | F32 delta | F64-matched base | F64-matched candidate | F64 delta | ULP/conditioning note |
|---|---:|---:|---:|---:|---:|---:|---|
| NH3-NH3_ncce31: integrated XC (Ha) | -16.069904327392578 | -16.070062637329102 | -0.0001583099365234375 | -16.069903173698265 | -16.070062438891732 | -0.00015926519346720625 | F32 83 ULP downward; matched-F64 -44829166622 ULP; F32-term condition(base,cand)=(1.0106400377868665, 1.0106400635772279); directв€’F64-term sum Ha=(2.5855477048253306e-07, 6.3746722389623756e-07); ULP=(0.13555716350674629, 0.3342164158821106) |
| NH3-NH3_ncce31: total molecule energy (Ha) | -113.20592498779297 | -113.20608520507812 | -0.00016021728515625 | -113.20592677197952 | -113.20608603717299 | -0.00015926519347431167 | F32 21 ULP downward; matched-F64 -11207291656 ULP |
| NH3-NH3_ncce31: stoichiometric contribution (Ha) | 113.20592498779297 | 113.20608520507812 | 0.00016021728515625 | 113.20592677197952 | 113.20608603717299 | 0.00015926519347431167 | F32 21 ULP upward; matched-F64 +11207291656 ULP |
| NH3_ncce31: integrated XC (Ha) | -8.0323162078857422 | -8.0323963165283203 | -8.0108642578125e-05 | -8.0323163068753551 | -8.0323959396764568 | -7.9632801101681139e-05 | F32 84 ULP downward; matched-F64 -44829281671 ULP; F32-term condition(base,cand)=(1.0106389600876511, 1.0106389857868912); directв€’F64-term sum Ha=(7.8704104566895694e-07, 1.3552424960039389e-08); ULP=(0.82527235150337219, 0.014210747554898262) |
| NH3_ncce31: total molecule energy (Ha) | -56.600082397460938 | -56.60015869140625 | -7.62939453125e-05 | -56.600081725723015 | -56.600161358524112 | -7.9632801096352068e-05 | F32 20 ULP downward; matched-F64 -11207320417 ULP |
| NH3_ncce31: stoichiometric contribution (Ha) | -113.20016479492188 | -113.2003173828125 | -0.000152587890625 | -113.20016345144603 | -113.20032271704822 | -0.00015926560219270414 | F32 20 ULP downward; matched-F64 -11207320417 ULP |
| Reaction energy (Ha) | 0.00576019287109375 | 0.005767822265625 | 7.62939453125e-06 | 0.005763320533489491 | 0.0057633201247710986 | -4.0871839246392483e-10 | F32 16384 ULP upward; matched-F64 -471220224 ULP |
| Reaction energy (kcal/mol) | 3.6145758628845215 | 3.6193633079528809 | 0.004787445068359375 | 3.6165383863097236 | 3.6165381298350496 | -2.5647467394307455e-07 | F32 20080 ULP upward; matched-F64 -577529623 ULP |
| Residual (kcal/mol) | 0.46457576751708984 | 0.46936321258544922 | 0.004787445068359375 | 0.46653829094229193 | 0.46653803446761799 | -2.5647467394307455e-07 | F32 160640 ULP upward; matched-F64 -4620236984 ULP |
| Raw RMSE (kcal/mol) | 0.46457576751708984 | 0.46936321258544922 | 0.004787445068359375 | 0.46653829094229193 | 0.46653803446761799 | -2.5647467394307455e-07 | F32 160640 ULP upward; matched-F64 -4620236984 ULP |
| Weighted fchem loss | 4.171715259552002 | 4.2147045135498047 | 0.042989253997802734 | 4.1893378187879327 | 4.1893355157421333 | -2.3030457994011044e-06 | F32 90155 ULP upward; matched-F64 -2592999051 ULP |

## BASEв†’CANDIDATE saved point-term response

These rows reduce paired candidate-minus-base point-term differences in F64 from the frozen arrays. F32-source and matched-F64-source terms are kept separate. The captured production XC delta is shown beside the paired-term sum; it is the direct production endpoint subtraction, not a replacement for the pointwise response sum.

| Saved point-term source | Component | ОЈ(candidateв€’base) point terms (F64 accumulation, Ha) | ОЈ|candidateв€’base| (Ha) | Оє = ОЈ|О”| / |ОЈО”| | Captured production XC О” (Ha) | Production О” в€’ pointwise ОЈО” (Ha) |
|---|---|---:|---:|---:|---:|---:|
| F32 | NH3-NH3_ncce31 | -0.00015868884897542477 | 0.00016079332867400712 | 1.013261673470883 | -0.0001583099365234375 | 3.7891245198727391e-07 |
| F32 | NH3_ncce31 | -7.9335153957762295e-05 | 8.0385623974583723e-05 | 1.0132409148330472 | -8.0108642578125e-05 | -7.7348862036270469e-07 |
| F64 | NH3-NH3_ncce31 | -0.00015926519346482143 | 0.00016137811700889704 | 1.0132667000121549 | -0.00015926519346720625 | -2.3848110985991156e-15 |
| F64 | NH3_ncce31 | -7.9632801103779355e-05 | 8.0686217666286735e-05 | 1.0132284253210502 | -7.9632801101681139e-05 | 2.0982158068297285e-15 |

## True response versus F32 scalar resolution

The signed ratio is the matched-F64 BASEв†’CANDIDATE change divided by the direction-appropriate adjacent F32 spacing at that stageвЂ™s rounded F32 base. The separate endpoint column reports the actual integer ULP distance between the captured F32 endpoints, including when its direction differs from the matched-F64 response.

| Stage | Matched-F64 BASEв†’CANDIDATE О” (Ha) | F32 observed endpoint О” (Ha) | Directional F32 spacing at F32 base (Ha) | True F64 О” / F32 spacing (signed ULP) | Observed F32 endpoint displacement (signed integer ULP) | Note |
|---|---:|---:|---:|---:|---:|---|
| NH3-NH3_ncce31 integrated XC | -0.00015926519346720625 | -0.0001583099365234375 | 1.9073486328125e-06 | -83.500829752534628 | -83 | Endpoint displacement is rounded; signed continuous ratio shown separately. |
| NH3_ncce31 integrated XC | -7.9632801101681139e-05 | -8.0108642578125e-05 | 9.5367431640625e-07 | -83.501044047996402 | -84 | Endpoint displacement is rounded; signed continuous ratio shown separately. |
| NH3-NH3_ncce31 molecule total | -0.00015926519347431167 | -0.00016021728515625 | 7.62939453125e-06 | -20.87520743906498 | -21 | Endpoint displacement is rounded; signed continuous ratio shown separately. |
| NH3_ncce31 molecule total | -7.9632801096352068e-05 | -7.62939453125e-05 | 3.814697265625e-06 | -20.875261010602117 | -20 | Endpoint displacement is rounded; signed continuous ratio shown separately. |
| Net reaction Hartree partial sum | -4.0871839246392483e-10 | 7.62939453125e-06 | 4.6566128730773926e-10 | -0.877716064453125 | 16384 | True response is below one adjacent F32 step. Reaction О” is also -5.3571537137031555e-05 NH3-NH3_ncce31 total ULP; -0.00010714307427406311 NH3_ncce31 total ULP. |

## Reduction and component-assembly diagnostics (V3)

These checks use the exact captured point terms and component operands. The fixed-F32-total table compares the same saved totals under both reduction dtypes; same-operand assembly isolates rounding in the in-place XC в†’ HF в†’ D3 path.

### Same F32 point-term reduction and difference-of-sums diagnostics

| State | Component | F32 direct XC (Ha) | F64 sum of same F32 terms (Ha) | Directв€’F64 term sum (Ha) | Gap (F32 ULP) | F32 term condition | Separate F64 sums diff (Ha) | Sum of pointwise diffs (Ha) | Difference |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseflat32 | NH3-NH3_ncce31 | -16.069904327392578 | -16.069904585947349 | 2.5855477048253306e-07 | 0.13555716350674629 | 1.0106400377868665 | 1.4122490838985868e-06 | 1.4122490835446397e-06 | 3.5394711657031272e-16 |
| baseflat32 | NH3_ncce31 | -8.0323162078857422 | -8.0323169949267879 | 7.8704104566895694e-07 | 0.82527235150337219 | 1.0106389600876511 | 6.8805143271788438e-07 | 6.8805143503673374e-07 | -2.3188493607437526e-15 |
| candidate_t_2m15 | NH3-NH3_ncce31 | -16.070062637329102 | -16.070063274796325 | 6.3746722389623756e-07 | 0.3342164158821106 | 1.0106400635772279 | 8.3590459354354607e-07 | 8.3590459414799916e-07 | -6.0445308731427259e-16 |
| candidate_t_2m15 | NH3_ncce31 | -8.0323963165283203 | -8.0323963300807453 | 1.3552424960039389e-08 | 0.014210747554898262 | 1.0106389857868912 | 3.9040428845282804e-07 | 3.9040428901969404e-07 | -5.6686600027986301e-16 |

### Reconstructed production XC в†’ HF в†’ D3 component assembly

| State | Precision | Component | XC dtype | HF add dtype | D3 source в†’ tensor dtype | Accumulator dtype | Production total (Ha) | Reconstructs exactly |
|---|---|---|---|---|---|---|---:|---|
| baseflat32 | f32 | NH3-NH3_ncce31 | torch.float32 | torch.float32 | float64 в†’ torch.float64 | torch.float32 | -113.20592498779297 | true |
| baseflat32 | f32 | NH3_ncce31 | torch.float32 | torch.float32 | float64 в†’ torch.float64 | torch.float32 | -56.600082397460938 | true |
| baseflat32 | f64 | NH3-NH3_ncce31 | torch.float64 | torch.float64 | float64 в†’ torch.float64 | torch.float64 | -113.20592677197952 | true |
| baseflat32 | f64 | NH3_ncce31 | torch.float64 | torch.float64 | float64 в†’ torch.float64 | torch.float64 | -56.600081725723015 | true |
| candidate_t_2m15 | f32 | NH3-NH3_ncce31 | torch.float32 | torch.float32 | float64 в†’ torch.float64 | torch.float32 | -113.20608520507812 | true |
| candidate_t_2m15 | f32 | NH3_ncce31 | torch.float32 | torch.float32 | float64 в†’ torch.float64 | torch.float32 | -56.60015869140625 | true |
| candidate_t_2m15 | f64 | NH3-NH3_ncce31 | torch.float64 | torch.float64 | float64 в†’ torch.float64 | torch.float64 | -113.20608603717299 | true |
| candidate_t_2m15 | f64 | NH3_ncce31 | torch.float64 | torch.float64 | float64 в†’ torch.float64 | torch.float64 | -56.600161358524112 | true |

### F32 XC/HF/D3 component assembly from identical operands

| State | Component | F32 XC (Ha) | F32 production total (Ha) | F64 total with same F32 XC/HF/D3 operands (Ha) | F32в€’F64 assembly (Ha) | Coefficient-weighted error (Ha) |
|---|---|---:|---:|---:|---:|---:|
| baseflat32 | NH3-NH3_ncce31 | -16.069904327392578 | -113.20592498779297 | -113.20592792567383 | 2.9378808648061749e-06 | -2.9378808648061749e-06 |
| baseflat32 | NH3_ncce31 | -8.0323162078857422 | -56.600082397460938 | -56.600081626733399 | -7.7072753867923893e-07 | -1.5414550773584779e-06 |
| candidate_t_2m15 | NH3-NH3_ncce31 | -16.070062637329102 | -113.20608520507812 | -113.20608623561036 | 1.0305322319936749e-06 | -1.0305322319936749e-06 |
| candidate_t_2m15 | NH3_ncce31 | -8.0323963165283203 | -56.60015869140625 | -56.600161735375977 | 3.0439697269457611e-06 | 6.0879394538915221e-06 |

### Fixed F32 component totals reduced in F32 and F64

| State | Reduction dtype | Same F32 molecule-total Hartree sum | Reaction (kcal/mol) | fchem loss | Matches production F32 loss | Matches after-component shadow |
|---|---|---:|---:|---:|---|---|
| baseflat32 | torch.float32 | 0.00576019287109375 | 3.6145758628845215 | 4.171715259552002 | true | вЂ” |
| baseflat32 | torch.float64 | 0.00576019287109375 | 3.6145757484436034 | 4.1717140712912038 | вЂ” | true |
| candidate_t_2m15 | torch.float32 | 0.005767822265625 | 3.6193633079528809 | 4.2147045135498047 | true | вЂ” |
| candidate_t_2m15 | torch.float64 | 0.005767822265625 | 3.6193632659912112 | 4.2147041724462335 | вЂ” | true |

### Stoichiometry-weighted assembly error summary

| State | Stoichiometry-weighted assembly error (Ha) | (kcal/mol) |
|---|---:|---:|
| baseflat32 | -4.4793359421646528e-06 | -0.0028108258573997704 |
| candidate_t_2m15 | 5.0574072218978472e-06 | 0.0031735710771095072 |

## Scalar reproduction

- F32 replay: exact; base `4.171715259552002`, candidate `4.2147045135498047`.
- Matched-F64 candidate-minus-base delta `-2.3030457994011044e-06`; absolute error `0` against tolerance `1e-10`.

## Predeclared shadow boundaries

Relative error is `abs(shadow_delta - full_matched_F64_delta) / abs(full_matched_F64_delta)`. Sign-only uses the frozen negative-beyond-8-ULP-F64 floor; the table applies no closeness cutoff. Armijo uses the frozen `c`, `t`, and raw slope recorded with the approved review.

| Boundary | Base loss | Candidate loss | Delta | Sign-only (8 F64 ULP) | Relative error to full F64 delta | Frozen Armijo margin |
|---|---:|---:|---:|---|---:|---:|
| after-component-totals | 4.1717140712912038 | 4.2147041724462335 | 0.04299010115502977 | false | 18667.628846985644 | -0.042990101435607109 |
| after-float32-xc-integral | 4.1969542269119762 | 4.1862067016232167 | -0.010747525288759441 | true | 4665.6572117472788 | 0.010747525008182102 |
| before-xc-reduction | 4.189541495446905 | 4.1896459706561284 | 0.00010447520922340914 | false | 46.363930344145736 | -0.00010447548980074828 |
| before-integrand-multiplication | 4.1893816343924382 | 4.1898608765151204 | 0.00047924212268224409 | false | 209.09057414614534 | -0.00047924240325958323 |
| after-float32-nn-adaptive-outputs-before-local-pbe | 4.1894944424779608 | 4.1896759471662515 | 0.00018150468829070121 | false | 79.810715938823535 | -0.00018150496886804035 |
| full-matched-float64-reference | 4.1893378187879327 | 4.1893355157421333 | -2.3030457994011044e-06 | true | 0 | 2.3027652220619643e-06 |

## Ten answers

1. **Earliest operation / sign loss:** Earliest tested sign-loss boundary: F32 adaptive NN outputs fed into F64 local PBE still give +1.815046882907012e-4 loss delta. This localizes to the output-precision boundary only; no NN layer or matmul is isolated.
2. **Primary error stage:** The exact production sign flip is dominated by F32 component-local XC then in-place HF and D3 assembly. The candidate-minus-base assembly-rounding error is +9.5367431640625e-6 Ha, about +0.0537376264 weighted-loss units; upstream F32 local-energy error also remains.
3. **True change / ULP:** F32 delta is +0.042989253997802734 (+90,155 ULP at the base). Matched-F64 delta is -2.3030457994011044e-6 (-2,592,999,051 ULP). Matched F64 uses the same saved F32 states and F32-loaded sources widened.
4. **Origin of +0.042989254:** The after-F32-XC-integral shadow is negative (-0.0107475253) but 4,665.657 times the true delta. Subsequent F32 in-place HF/D3 component assembly shifts it upward and flips the production response; F32 PBE/integrand differences also contribute upstream.
5. **Are F32 local NN outputs sufficient with downstream F64?:** No for this frozen sample: F32 adaptive NN outputs followed by F64 local PBE still produce a positive delta, 79.8107 relative-error units from the matched-F64 delta, and a negative Armijo margin.
6. **Minimum repair boundary:** No tested intermediate shadow is quantitatively faithful. Full matched-F64 chemistry is the sole tested quantitative match; the minimum mixed-precision repair boundary is not identified.
7. **Close enough to matched F64 / Armijo margin?:** Only full matched F64 matches the reference (relative error 0; margin +2.3027652221e-6). The sign-only XC-integral shadow has a positive margin but is grossly too large; other mixed shadows have wrong signs and negative margins.
8. **Shadow-gradient comparison (say not run if absent):** No shadow-gradient comparison was run; no gradient or backward pass was added.
9. **Is NN F64 necessary?:** This sample shows F32 adaptive outputs are inadequate at the tested boundary, not that every NN operation or workload requires F64. Full F64 model/local-energy evaluation is the only tested faithful path.
10. **Single next experiment:** Run one double-precision chemistry-forward Armijo replay on the exact frozen state pair and same F32-loaded sources, using the already-frozen slope and no new optimization update.

## Evidence and run integrity

- V2 full trace: `47c159d78373f4323e6d96cdc30b1ab47a45276b560ae108489414e057c994c2`; arrays: `029d9838d7d94118eb53672b9822f69d5f05c2810e7f297839b27d0bfa2de5f5`; receipt: `0d50454d539f7a797e7b6d86a57272ed9e5b406415028068931e05c482d773b8`.
- Shadow artifact/receipt: `c89e7409665dbc8e0cd0087411222a7c5d8b82d29b047e2b599f9834b54dd463` / `10b3408ca7d8e5b1b2d954fa8f5353525f452e26eb6be856005ea58d430678d6`; approved review: `9f917f483d7003e6194d75fcb0e0862d9a97456e1660f0d31791ab60a6b89774`.
- V3 operation diagnostics/receipt: `e3e23f5143434edd0779f42641aacef9607e6c0b361523bc9fdca7dbf9cc9009` / `b31007d350b0eaf8b330093e417df693f004f288eb71ec7b18177c138fc87529`; they bind to the V2 full JSON and arrays and prove rollback.
- Paired BASEв†’CANDIDATE array diagnostic/receipt: `e12ef979572bfe68dc002caadb3e1344c93b62e579b7e609b2f24a37eea99d2a` / `fc22d70f8f4b3216257fa70742d46e76dfbf1772136d9c77dcbe14335e252871`; identity binds to the frozen protocol and V2 capture; caller runtime/RNG/device rollback verified.
- V2 rollback flags: `{"caller_rng_full_state": true, "caller_runtime_flags": true, "cuda_device": true, "cursor": true, "ema": true, "mode": true, "model32_and_buffers": true, "model64_and_buffers": true, "rng_full_state": true}`; caller runtime restored: `True`.
- Frozen Armijo inputs: `c=0.0001`, `t=3.0517578125e-05`, `gdotv=-0.091939671059904526`.
- Reviewer evidence: `[{"artifact": "predeclared_protocol.json", "finding": "Pins the sample, exact two F32 parameter vectors, same-source matched-F64 scope, six boundaries, no training, and no shadow gradients.", "sha256": "7293dd32bd5cde8c0fda8fbd1845c6c9f52e654dc0536fda9e504fc32a0e8968"}, {"artifact": "chem_precision_trace_full_v2.json", "finding": "Exact original-objective F32 scalar replay and matched-F64 negative delta; per-point local energies, reductions, assembly, stoichiometry, residual, RMSE, and fchem factor.", "sha256": "47c159d78373f4323e6d96cdc30b1ab47a45276b560ae108489414e057c994c2"}, {"artifact": "chem_precision_trace_full_v2_arrays.npz", "finding": "36 captured arrays for exactly two states at F32/F64, including adaptive NN outputs, local PBE epsilon, component weights, densities, and point integrands.", "sha256": "029d9838d7d94118eb53672b9822f69d5f05c2810e7f297839b27d0bfa2de5f5"}, {"artifact": "chem_precision_trace_full_v2_receipt.json", "finding": "Binds V2 driver/result/arrays; pinned source/input start-end hashes match; model/buffers, caller RNG/runtime flags, CUDA device, EMA, cursor, mode restored.", "sha256": "0d50454d539f7a797e7b6d86a57272ed9e5b406415028068931e05c482d773b8"}, {"artifact": "chem_precision_trace_shadows_v2.json", "finding": "Six frozen shadow boundaries with continuous relative errors and Armijo margins; F32-output/F64-PBE shadow remains wrong-sign; F32-XC sign-only shadow is grossly inaccurate.", "sha256": "c89e7409665dbc8e0cd0087411222a7c5d8b82d29b047e2b599f9834b54dd463"}, {"artifact": "chem_precision_trace_shadows_v2_receipt.json", "finding": "Caller RNG/runtime flags, CUDA device, and thread count are restored.", "sha256": "10b3408ca7d8e5b1b2d954fa8f5353525f452e26eb6be856005ea58d430678d6"}, {"artifact": "chem_precision_trace_operation_diagnostics_v3.json", "finding": "Exact same-F32-operand reduction comparisons; exact F32 component assembly reconstructions; candidate-minus-base weighted assembly error +9.5367431640625e-6 Ha; fixed F32 totals reduce to identical reaction sums in F32 and F64.", "sha256": "e3e23f5143434edd0779f42641aacef9607e6c0b361523bc9fdca7dbf9cc9009"}, {"artifact": "chem_precision_trace_operation_diagnostics_v3_receipt.json", "finding": "V3 diagnostics bind to the accepted capture and prove caller RNG/runtime/device rollback.", "sha256": "b31007d350b0eaf8b330093e417df693f004f288eb71ec7b18177c138fc87529"}]`.

The chemistry signal is already lost in float32 model/local-energy evaluation; the next experiment is a double-precision chemistry-forward Armijo replay.
