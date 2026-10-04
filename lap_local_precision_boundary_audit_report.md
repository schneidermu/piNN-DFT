# Local operator precision-boundary audit

Minimal tested missing boundary: learned-model raw descriptors, forward and derivative arithmetic in F64. Keep PBE local arithmetic F32; retain F64 AO contractions and accumulation. The frozen production case passes the unchanged 5% gate. No production pilot ran.

## Identity and reused baseline

Start `lap_full_vxc` at `f41acc47bd7eece55cdc2c01183e9e6b2b994342`. Reused df0b95a globalization audit and f41acc4 factorial artifacts after checking all recorded hashes. The exact theta4/rounded-theta5 pair, immutable15 panel, reference operators, overlap, weights and chunks256/4096 are unchanged. No O1/PCD/tau/Armijo/sampling/optimizer or mathematical functional changes.

theta4 SHA `16b63fccd3dd70c11afdaf37f498486b9a6c641f9c8e75e3f7fb7b496e4c0c97`; rounded theta5 SHA `13c9d8ad8df1d8379a2e1e267f1026f0f0a20b0c3bf57ae1f2a21f362893cc9d`. Baseline factorial SHA `90fb5f1053103f9fbd9095daec5bfccebb191df1e42200490efc408ffee411db`. All source/data/external array identities are bound in protocol/results and the prior immutable audit. No source data were regenerated.

Baseline P32_A64 was reused, not re-executed: ΔL=`-4.2720760458705986e-7`, descent; absolute error=`5.3568211011203114e-8`, relative discrepancy=`14.3368745192%`. Trusted matched-double ΔL=`-3.7363939357585675e-7`. Relative discrepancy is `abs(candidate_delta-gold_delta)/abs(gold_delta)` with F64 mean of per-system deltas, exactly the previous audit; absolute tolerance=max(1e-10,5%|gold|).

## Dtype and derivative contract

Matched-F64 means arithmetic on exact F32-loaded source/checkpoint values widened. It does not restore native source precision. Combined uses actual F64 model leaves and original LapEnergy/local_partials, including F64 sigma construction and chain factors. It reproduces all15 trusted base/candidate scalars within1e-14 and the aggregate gradient exactly. Its direct operators become the reference for isolation comparisons; historical direct operator matrices were not available, so this does not claim comparison to previously stored matrices.

Isolation separates two sigma leaves: each branch constructs sigma from the same gradient source values in its own dtype. The learned branch owns raw sigma-total descriptors, q/Laplacian descriptors, activations, adaptive outputs and their derivatives. The PBE branch owns its rho/sigma arithmetic, local exchange/correlation and direct sigma derivative. Both dependencies are differentiated; each sigma chain is evaluated in that branch dtype before their sum in F64. The all-double split-chain control agrees with original local_partials to1e-12. This prevents inadvertently giving an F32 branch already-double sigma arithmetic.

Combined dispatch trace: 2,885 floating operation outputs, allF64; no F64→F32 downcast. NN_OUTPUT_SCALE_PBE physical constants originate as F32 rounded values and are widened before F64 multiplication, exactly as in the trusted reference. No constants are recomputed. Python scalar constants follow the operand dtype. Tau is never read; there is no tau partial channel.

In the selected mixed path, narrowing F64 adaptive constants to F32 at PBE entry is explicit and intentional. PBE rho/sigma/energy and direct sigma chain stay F32. Its adjoints feed the F64 learned branch through differentiable casts; combined c/a/b remain F64 into AO contractions. No accidental round-trip is hidden.

## Staged scalar gates

| Condition | operator objective ΔL | discrepancy % | 5% gate |
| --- | --- | --- | --- |
| reused baseline | -4.2720760458705986e-07 | 14.3368745192 | FAIL |
| combined | -3.7363939357585675e-07 | 0.0 | PASS |
| M64 | -3.7333738637212087e-07 | 0.08082852315050802 | PASS |
| P64 | -4.2713958767944444e-07 | 14.318670628268753 | FAIL |
| production | -3.7333738637212087e-07 | 0.08082852315050802 | PASS |

Combined passes before isolation. M64-only passes; P64-only fails. The narrowed M64 boundary is selected rather than full-double PBE. No failed condition was rescued by expanding scope or relaxing thresholds.

## Direct operator and gradient evidence

| Condition | max response relL2 | min response cosine | max component difference | response norm ratio range |
| --- | --- | --- | --- | --- |
| M64 | 0.004482152349575734 | 0.9999903814760649 | 1.0002900996397557e-07 | [0.9988836217034012, 1.0010455083148215] |
| P64 | 0.0662595841524008 | 0.9979909021485955 | 1.4658826792413038e-06 | [1.0042785523258717, 1.042377599060832] |

M64 base/candidate operator max relL2: `9.75168184294628e-08` / `1.0427367812888986e-07`. Full matrix response comparisons cover all15, not only a scalar cancellation. Production matrices exactly match the selected isolation condition on every base/candidate.

Production aggregate gradient cosine to gold `0.9999999999998153`, relative L2 `1.2147881613916773e-06`, norm ratio `0.9999989484637971`. F64 learned arithmetic is differentiable back to stored F32 leaves; final leaf-gradient storage remains F32 and is recorded, not represented as an all-double parameter gradient.

Direct local channels e,C,A,B are sampled at the first256 centers of every system for base/candidate; full operator matrices and parameter gradients cover every center. Native shapes: e=(256,), C/B=(256,2), A=(256,2,3). Relative errors below summarize base samples, not an all-point local bound. An additive learned/PBE energy decomposition is not claimed because adaptive parameters couple nonlinearly to PBE.

| Condition | energy e relL2 max | rho C relL2 max | gradient A relL2 max | Laplacian B relL2 max |
| --- | --- | --- | --- | --- |
| M64 | 1.5749905849182912e-07 | 2.471313038177837e-07 | 3.617648685506083e-05 | 4.415026310460173e-07 |
| P64 | 7.097440358660638e-08 | 7.129149817016211e-08 | 1.003221805001324e-07 | 1.0 |

The P64 sampled B relative error of1 occurs in AlBeH where the gold norm is only3.4337261980739142e-15 and the F32 learned branch returns zero (max abs error1.7168630148661243e-15). This near-zero sample does not establish that B dominates the full operator residual. Full operator-response vectors, rather than this relative sample error, support the selection.

| Condition | energy e max abs | rho C max abs | gradient A max abs | Laplacian B max abs |
| --- | --- | --- | --- | --- |
| M64 | 9.853841189327384e-13 | 2.354914230284777e-08 | 4.473482295968417e-09 | 7.579995178300726e-22 |
| P64 | 6.883545264417231e-13 | 6.071912181382366e-09 | 4.799694218460271e-10 | 1.7168630148661243e-15 |

## Minimal production change

Only train_models/lap_operator.py and its unit tests change. F32 LapEnergy operators use a transient differentiable torch.func.functional_call with F64 parameter/buffer views. Stored parameter values/state are untouched. AO/features/weights widen within the existing bounded AO chunk, preserving cache values and ordering. The learned and PBE sigma chains reproduce the qualified isolation. Already-F64 models and non-Lap energy callables retain their existing path. E_xc, chemistry, architecture, constants and equations are unchanged. Numerical precision behavior changes only for F32 learned AO operators; existing checkpoints remain source values, not newly trained states.

Source: no files added, one modified (56 added lines before final formatting accounting); tests: one modified,36 added lines. No trainer, model architecture, objective stack, cache format or dependency added. Source commit binds the numerical repair; no pilot protocol migration or curriculum change occurred.

## Validation and independent review

Windows affected suite:40 passed,2 two-rank tests deselected. Includes the existing six operator tests plus the new dtype/state/autograd regression. WSL/PySCF:15 passed, including manufactured, gradient/density-matrix and actual PySCF matrix parity checks. Ruff on changed files passes; compileall and git diff --check pass. External dtype/branch-chain checks pass. Frozen production verification covers all15 and passes. After verification, only imports changed: Ruff formatting and reusing lap_vxc.PBE to preserve direct-script import support. The PBE module object is identical; source computation AST excluding imports is identical to the executed snapshot.

All historical baseline/source hashes checked before diagnostics. Production validation permits only the explicitly hash-bound lap_operator repair; all data and other original pinned source hashes must remain identical. Model/RNG restoration passed; no .grad accumulation, optimizer, EMA/cursor/sampling progression or permanent parameter update occurred. Unrelated pre-existing untracked directories remain untouched. Large arrays/tooling stay outside Git, SHA-bound in results.

Independent review: PASS; see SHA-bound external review.md.

## Decision

Gate PASS on the exact frozen reference case. F64 learned-model local evaluation is the minimum passing condition among the prescribed two branch isolations; this does not prove every individual learned-layer operation needs double. PBE-only is insufficient. The production boundary matches the selected diagnostic and yields0.0808285231505% discrepancy, far below5%. No multi-update production pilot, optimizer step, SCF, Diet, Slurm or full90 cache generation ran.

No production pilot was run.
