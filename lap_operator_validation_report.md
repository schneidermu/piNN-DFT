# Variational Laplacian XC-operator validation

## Current status

Gate 1, Gate 2, the three-system real-data pilot, post-pilot SCF checks, and the all-90 corpus gate have passed. No multi-objective trainer or production jobs have been started. The h-free variational XC-operator objective is validated and ready for multi-objective training design.

Starting repository identity: branch `lap_full_vxc`, HEAD `ccd78151f6562540d3a6ae52759c5a2dc565965a`. Source implementation commits: `4f7ce3191f22e8fe9185daca446b97cb02efbff8` (`Add validated h-free variational XC operator pipeline`) and `3d95d373815502d784c859b1c70734dbc0c036ba` (portable all-system provenance manifest, output guard, and tests). Relevant prior commits are `0ff1494dafea0065d4689f832c792a8956bc5478` (real Lap pilot), `8e72748d5de048220661cd13bc4abd14457ee1b0` (S5-equivalent Lap training), `ad2036799758d6f7dff6b7ef8c94fb020941461c` (S5 report), and `0a6c7abf7581e8eeff87188e267405a23621ab70` (stencil validation). The failed spatial-FD gate remains closed; this work uses central-grid AO derivatives and no spatial step `h`.

Focused agents used: `operator_theory` derived and cross-checked the spin/RKS variational matrix conventions and ran independent h-free validation; `operator_scf_audit` audited the existing SCF conversion path and implemented the Torch AO operator plus focused tests and a synthetic CUDA chunk diagnostic; `stencil_operator_tests` owns the independent validation test/runner work.

## AO operator and conventions

For spin densities and

\[
E_{xc}=\int f(\rho_\alpha,\rho_\beta,\sigma_{\alpha\alpha},\sigma_{\alpha\beta},\sigma_{\beta\beta},\nabla^2\rho_\alpha,\nabla^2\rho_\beta)\,d\mathbf r,
\]

the weak-form spin operator is

\[
V^s_{\mu\nu}=\sum_g w_g\left[C_s\phi_\mu\phi_\nu+A_s\cdot(\nabla\phi_\mu\,\phi_\nu+\phi_\mu\nabla\phi_\nu)+B_s\left(\phi_\mu\nabla^2\phi_\nu+\phi_\nu\nabla^2\phi_\mu+2\nabla\phi_\mu\cdot\nabla\phi_\nu\right)\right],
\]

where `C_s = ∂f/∂rho_s`, `B_s = ∂f/∂lapl_s`,

\[
A_\alpha=2f_{\sigma_{\alpha\alpha}}\nabla\rho_\alpha+f_{\sigma_{\alpha\beta}}\nabla\rho_\beta,\qquad
A_\beta=2f_{\sigma_{\beta\beta}}\nabla\rho_\beta+f_{\sigma_{\alpha\beta}}\nabla\rho_\alpha.
\]

For RKS, `P_alpha = P_beta = P_total/2`. Thus `V_RKS = (V_alpha + V_beta)/2`; equivalently, the common PySCF scalars already encode this average:

- `vrho = (C_alpha + C_beta)/2`
- `vsigma = (f_sigma_aa + f_sigma_ab + f_sigma_bb)/4`
- `vlapl = (B_alpha + B_beta)/2`
- `vtau = 0`

The final matrix passed through the common scalar RKS convention has no extra closed-shell half factor. It is symmetric and satisfies `δE_xc = Tr(V_RKS δP_total)` for a full symmetric AO density matrix. The mRKS reference matrix is the direct scalar projection `V_ref[mu,nu] = sum_g w_g phi[g,mu] vxc_ref[g] phi[g,nu]`, with no spin factor.

The recommended loss is

\[
L_V=\frac{1}{n_{AO}}\left\|S^{-1/2}(V_{pred}-V_{ref})S^{-1/2}\right\|_F^2,
\]

with symmetric positive-definite overlap `S`. The independent theory review agreed with the derivation and normalization.

## Existing SCF path and differentiable implementation

The SCF audit inspected `test_models/DFT/lap_functional.py` (`LapFunctional.eval_xc`), `test_models/DFT/numint.py` (`NumIntWithLaplacian.nr_rks`, `RKS_with_Laplacian`), `train_models/lap_vxc.py` (`LapEnergy`, `local_partials`), and `train_models/reaction_energy_calculation.py` (`integrate_xc_energy`). The specified WSL environment has PySCF 2.14.0. Existing RKS integration uses PySCF's GGA `eval_mat` for `vrho`/`vsigma` and zero tau, then explicitly adds the full AO product Laplacian, including `2 ∇phi_mu · ∇phi_nu`. Stock PySCF `eval_mat`'s non-`None` `vlapl` path alone only supplies the AO-Hessian pair, so it must not replace the custom full-product term.

The existing SCF adapter is not trainable through NN parameters: it converts densities to NumPy, uses `autograd.grad` without `create_graph=True`, detaches derivatives, and returns NumPy arrays; PySCF matrix assembly also uses NumPy. The Torch-native route reuses `LapEnergy` and `local_partials(create_graph=True)` and performs AO contractions in Torch. `reaction_energy_calculation.integrate_xc_energy` remains an energy-only Torch path.

The implementation is in `train_models/lap_operator.py`. It exposes spin/RKS density-feature construction from AO data and density matrices, integrated XC energy, spin and RKS AO operators, scalar reference projection, symmetric orthonormalization, `/nAO` operator loss, and frozen h-free protocol metadata. Grid chunks contract directly to AO matrices without a grid-by-AO-by-AO intermediate. Focused tests are in `train_models/test_lap_operator_unit.py`. The pilot corpus builder preserves the three requested central targets by default, supports a separately explicit `--all90` mode with system/point-count checks, and rejects output paths inside the repository.

## Confirmed implementation checks

Six focused tests passed on Windows and WSL. They cover the spin-resolved AO formula, the `sigma_ab` and complete Laplacian-product contributions, density-matrix autograd agreement, RKS half-density factors, scalar projection without a half factor, chunk equivalence, positive-definite overlap validation, basis-congruence invariance, `/nAO` normalization, frozen metadata rejection of `h`/`stencil`, and finite nonzero backpropagation to a fresh Lap-model's parameters. `py_compile`, Ruff, and `git diff --check` passed for the implementation and these tests.

An earlier combined Windows run reported 13 passed and one PySCF test skipped with `MKL_THREADING_LAYER=SEQUENTIAL` and `OMP_NUM_THREADS=1` set to avoid the native MKL abort. The later full focused regression is recorded below; no test was weakened.

A diagnostic-only CUDA chunk test ran on an NVIDIA GeForce RTX 5070 Ti with Torch 2.11.0+cu128, a fresh two-layer, width-eight Lap model, 4096 synthetic points, 40 AOs, and normalized weights. Compared with one full-grid chunk, chunk sizes 128, 256, 512, 1024, and 2048 had maximum relative operator error `1.73e-7`, parameter-gradient error `6.66e-8`, and relative raw-loss error `8.24e-8`; all outputs and gradients were finite. Measured forward times were 28.5–62.7 ms and backward times 35.1–105.7 ms for this small synthetic case. Forward peak was 63.74 MiB allocated / 66 MiB reserved; backward peak allocation was 62.07–62.68 MiB / 66 MiB reserved. These timings are plumbing diagnostics, not an all-system performance estimate. Full results are outside the repository at `C:\Users\schne\AppData\Local\Temp\lap_operator_cuda_diag_20261001_183417.json`.

## Gate 2 results and remaining work

Gate 2 **passes all six required conditions**. The real-data pilot passes on H2, BeH2, and CO, and all four post-pilot SCF runs converged with finite cycle energies. The all-90 corpus was independently verified against the required system and central-point counts, target identity, and descriptor compatibility. The required success statement appears at the end of this report; this work did not start multi-objective training.

1. **Manufactured LDA/GGA/Lap matrices: PASS.** The independent suite checks LDA on a 24³ Gauss-Hermite grid (`a={0.7,1.3}`, `rtol=atol=3e-12`), GGA on a 24³ grid (`rtol=4e-11`, `atol=4e-12`), and Laplacian-level energy on a 28³ grid (`rtol=8e-10`, `atol=8e-11`). These compare weak AO matrices with independently assembled exact strong projections; no spatial finite differences are used.
2. **Real PBE weak/strong projection: PASS with finite-grid quadrature residual reported.** Float64 calculations used the same repository canonical PBE energy in the variational weak operator and the independent analytic Euler-potential reference, at exact source coordinates/weights. The largest legacy-grid relative Frobenius residual is BeH2 (`7.559200e-6`, max element `4.482661e-5 Ha`, overlap-orthonormalized relative error `1.183229e-5`). It is a quadrature residual: complete NPZ and standard PySCF grids reproduce that scale, while unpruned refinement reduces BeH2 relative error to `7.629789e-7` at 100 radial × 434 angular points per atom and `1.443328e-7` at 120 × 590 points (52.37× below the legacy grid); the final maximum element error is `3.936132e-7 Ha`. No arbitrary tolerance loosening is used. The L5–L7 pruned grids vary between `7.35e-6` and `7.90e-6`, so they are not described as monotonic convergence evidence.
3. **Actual PySCF Lap `nr_rks` parity: PASS.** For H2/STO-3G with the same model instance and all 42,508 legacy points, Torch versus `RKS_with_Laplacian.NumIntWithLaplacian.nr_rks` gives relative Frobenius error `4.483280e-16`, maximum element error `5.551115e-16 Ha`, and orthonormalized relative error `2.541611e-13`.
4. **Density-matrix directional derivative: PASS.** Central-difference-in-DM energy slopes agree with the matrix contraction for LDA, PBE, and LapNN over `eps=(3e-4,1e-4,3e-5,1e-5)`. Absolute errors are LDA `[3.8755e-13,3.8752e-13,3.3481e-12,4.0534e-12]`, PBE `[1.17996e-11,4.39815e-12,3.00332e-12,1.78063e-11]`, and LapNN `[8.24874e-12,8.47267e-13,3.59363e-12,3.59363e-12]` Ha. The corresponding contractions are approximately `0.223196`, `-0.383622`, and `-0.325130` Ha. This finite difference perturbs the AO density matrix only; it is not a spatial derivative or stencil.
5. **Parameter backpropagation: PASS.** Operator loss produces finite, nonzero gradients for the fresh LapNN parameters; the one-parameter double-precision outer `gradcheck` passes.
6. **No hidden spatial h/FD dependence: PASS.** The operator and D4 projection use central AO values, first derivatives, Laplacians, and central-grid weights only. The analytic strong reference uses the analytic Euler chain rule; no displaced points or spatial finite differences are present.

The five source-grid D4 results are below. `n_AO`, points, and elapsed time refer to the legacy training subset; matrices and quadrature weights remain float64. Spin-potential alpha/beta maximum difference is at most `2.85e-14 Ha` across all five systems.

| System | AOs | Legacy points | Relative Frobenius | Max element (Ha) | Orthonormalized relative | Time (s) |
|---|---:|---:|---:|---:|---:|---:|
| H2 | 60 | 42,508 | 1.058904e-9 | 2.062023e-9 | 8.301457e-9 | 3.05 |
| BeH2 | 144 | 77,240 | 7.559200e-6 | 4.482661e-5 | 1.183229e-5 | 8.44 |
| CO | 168 | 72,110 | 9.426786e-7 | 6.025592e-6 | 1.933575e-6 | 10.12 |
| N2 | 168 | 71,820 | 9.977646e-7 | 6.057310e-6 | 2.021748e-6 | 13.21 |
| ClH | 139 | 58,114 | 1.963959e-7 | 1.413974e-6 | 3.427511e-7 | 7.73 |

BeH2 quadrature evidence: complete 89,200-point NPZ grid and PySCF level 5 both give relative Frobenius `7.559183e-6`; level 6 (130,976 points) gives `7.899399e-6`; level 7 (184,328) gives `7.346695e-6`. Unpruned 100×434 (130,200 points) gives relative/max/orth-relative errors `7.629789e-7 / 1.870351e-6 Ha / 1.590261e-6`; unpruned 120×590 (212,400 points) gives `1.443328e-7 / 3.936132e-7 Ha / 3.245862e-7`. Thus the source-grid residual is tracked as a discretization effect and independently reduces under increased, unpruned quadrature.

The reusable independent runner and tests are `train_models/run_lap_operator_validation.py` and `train_models/test_lap_operator_validation.py`. The runner records `spatial_finite_difference_used: false`, supports exact legacy-to-NPZ float32 coordinate identity, full NPZ comparisons, PySCF grid refinement, unpruned refinement, and actual `nr_rks` parity. The numerical JSON is outside Git at `C:\Dev\readWFN_share_ms\lap_operator_runs_20261001\d4_real_pbe_ao_validation.json` (SHA256 `05cd902af73ae338cafd18a613c1b48aa55c04144a91e4f4f97537dc7cdabe1c`). It contains the exact weak and strong matrices for audit. An auxiliary comparison of repository Torch PBE weak matrices against PySCF/libxc PBE on BeH2 has relative error `4.904575e-6`; it is diagnostic only and is not the D4 oracle because it compares distinct PBE evaluation implementations. D4 acceptance compares weak and analytic-strong derivatives of the same repository canonical energy.

The independent validation test file reports **8 passed** in 3.85 s in WSL (one PyTorch deprecation warning). The final orchestrator WSL focused run reports **41 passed** in 7.97 s (one PyTorch deprecation warning), covering operator unit/validation/data/pilot tests and historical scientific regressions. The final orchestrator Windows focused regression reports **177 passed, 5 skipped, 2 warnings** (one warning is historical); it used `MKL_THREADING_LAYER=SEQUENTIAL` and `OMP_NUM_THREADS=1` to avoid the native MKL runtime abort. The PySCF-only Windows skip is covered by WSL parity tests. Final Ruff passed on all 9 new Python files; `compileall` passed for `train_models` and `test_models`; `git diff --check` passed. No test was weakened.

## All-90 corpus status

The external all-90 central/operator corpus is at `C:\Dev\readWFN_share_ms\lap_operator_runs_20261001\all90`. It contains 90 systems and exactly 8,271,091 central points under `lap-operator-central-ao-noh-v1` / `lap-weakform-ao-v1`. The 90 compressed HDF5 records occupy 703,764,412 bytes; the external manifest is 27,736 bytes; total directory payload is 703,792,148 bytes. The external manifest SHA256 is `7005cd869ea8be9636b03f385e7069f4e9023c9582fe7defc5437a7d6609b887`. The builder passed strict verification before and after atomic publication, and the independent verifier passed: zero unmatched legacy Vxc centers, maximum absolute Vxc identity error `4.7683379e-7 Ha`, and all rho/sigma/Laplacian descriptors within legacy float32 compatibility (worst normalized error `0.0118932`). Generation plus both builder-side strict verifications took 406.12 s. The all-90 metadata-manifest writer/output-guard tests passed 8/8 on Windows in 1.82 s and 8/8 in WSL in 2.19 s; Ruff, compileall, and `git diff --check` passed after that change. The metadata-only repository manifest [lap_operator_corpus_manifest.json](lap_operator_corpus_manifest.json) is 2,191,045 bytes with SHA256 `65aab3a084eb1a5ac6eb78d3a2d04d9744524b4ea58616f5e29768f8731b1bd7`. The corpus data remain outside Git.
Reproduce the strict all-90 corpus integrity verification from the WSL environment:

```bash
wsl -d Ubuntu-22.04 --exec /home/schneidermu/.cache/pinn-lap-tests/bin/python -c "import json; from train_models.lap_operator_data import verify_operator_corpus; p='/mnt/c/Dev/readWFN_share_ms/lap_operator_runs_20261001/all90'; m=json.load(open(p+'/manifest.json')); print(verify_operator_corpus(p, expected_systems=m['built_systems'], require_all90=True))"
```

## Pilot data fixture, cache bridge, and current progress

The external H2/BeH2/CO fixture is built and hash-verified at `C:\Dev\readWFN_share_ms\lap_operator_runs_20261001\pilot_h2_beh2_co`. Its manifest SHA256 is `b55e569784ab0d3ebc1940849594be08a799e9c32b506668827a35cf5f4eb72b`; the records contain H2 (42,508 points, 60 AOs), BeH2 (77,240 points, 144 AOs), and CO (72,110 points, 168 AOs). The three compressed HDF5 records total 13,657,447 bytes (29,656,448 raw dataset bytes). Builder/fixture validation passed 13 focused WSL tests, compileall, and Ruff. The builder finished in 9.59 s; hash-verified loading took 0.432 s at 31.6 MB/s. H2 descriptor regeneration differed by at most `4.24e-22`, and the regenerated scalar reference AO projection differed by at most `1.11e-16 Ha`. Exact legacy row order, weights, common-RKS Vxc, and E_xc are retained; `exc_wf` is not substituted. The builder explicitly checks exact float32 coordinate identity, records original NPZ float64 centers, and rejects ambiguous distinct-float64 collisions.

The transient float32 AO cache at `C:\Dev\readWFN_share_ms\lap_operator_runs_20261001\ao_cache_fgpu_20261001T192623` bridges central records to GPU training. Its LZF/shuffle HDF5 files total 286,907,157 bytes on disk, versus 515,750,400 raw AO-factor bytes; generation took 10.7466 s with chunks of 2,048. Cache manifest SHA256 is `20ae74614a21ccb84eab1708e581c2c3e466860d7902615549d29c5e8aa37d15`. Cache files carry source record/file hashes and are rejected if the linked central manifest, protocol, shape, dtype, or bytes differ. The independent WSL rerun of `train_models/test_lap_operator_data.py` and `train_models/test_lap_operator_pilot.py` passed 15 tests in 4.88 s. Density-feature relative max errors for H2/BeH2/CO are `5.404e-8 / 1.126e-7 / 6.197e-8`; reference-projection relative Frobenius errors are `3.280e-9 / 8.127e-9 / 7.361e-9`.

The fresh two-epoch canonical PBE predopt completed on the RTX 5070 Ti in 186.95 s (seed 41, 268 reactions, 21,073,642 grid points). Its MSE/MAE moved from `0.0823227691 / 0.1921368972` before training to `4.8787537e-7 / 2.8292707e-4` after. Predopt checkpoint SHA256 is `ed4ba8231d93c376ce5aa8fc81e0670f6a5a44c95c2a552651b12d6a1d4b63f8`; report SHA256 is `7ffb34795ec10e2fde8d42c3eb432e6588f5a24f400e8242be14592abd537262`. The canonical source pickle and manifest hashes are recorded in `lap_operator_validation.json`.

The H2-only float32 branch completed 20 optimizer updates with 512-point chunks. On the saved post-update checkpoint, the E_xc error fell from `5.289421` to `3.829378 kcal/mol` and operator loss from `0.050650649` to `0.050052591 Ha²/AO`; the pre-training E/operator gradient norms were `7018.2243 / 0.2876159` with cosine `0.9801495`. All updates were finite. The post-update q check used 1,024 deterministic density-rank-stratified H2 points and found a nonzero Laplacian response: maximum energy change `2.926826e-3 Ha`, adaptive-constant change `1.666820e-2`, gradient L2 `7.124433e-3`, and maximum gradient `8.054397e-4`; tau perturbation remained exactly zero. This resolves the earlier 512-point sampling false negative. Peak allocated/reserved GPU memory was `3.424 / 3.704 GB`, with 5.57–7.81 s per step. Checkpoint SHA256 is `ee352f4ae1854f14fa1279c17a888837420e5ab55ea3f168a5797caeb153acad`; refreshed report SHA256 is `e8a99ef4c52b3ad83f8a76ee1fb4879794d990c4717368d5dd43bcdb2801bc0`. The combined H2+BeH2+CO branch also completed 20 updates. Its mean squared E_xc error fell from `1245.907227` to `354.505554 kcal²/mol²`, and mean operator loss from `0.044623729` to `0.043847021 Ha²/AO`; the E/operator gradient cosine was `0.9002036` initially and `0.8915371` after training. Final absolute E_xc errors were H2 `3.880919`, BeH2 `17.766476`, and CO `27.070417 kcal/mol`; the corresponding operator losses were `0.050108112`, `0.032716613`, and `0.048716336 Ha²/AO`. Density-stratified q diagnostics on 1,024 points per system gave nonzero Laplacian-energy changes of `0.002098799 / 0.693756 / 13.992676 Ha` for H2/BeH2/CO; maximum absolute Laplacian gradients were `6.823190e-4 / 1.105927e-3 / 1.011542e-3`. Tau perturbation effects were exactly zero throughout. These results, plus four converged SCF runs, pass the three-system pilot gate.

The external all-90 corpus, three-system fixture, AO cache, predopt files, and pilot checkpoints are not in Git. No production training or Slurm jobs were submitted, and no multi-objective trainer implementation was started. The next project is clean multi-objective training design; that work is out of scope for this validation.

The existing source audit matched 90 systems and reconstructed all 8,271,091 legacy central points from `dm_ks`; it matched all central reference Vxc values by exact coordinate identity (pooled RMS `3.4e-08 Ha`, maximum absolute difference `4.768e-07 Ha`). The legacy mRKS `E_xc` target remains unchanged and `exc_wf` was not substituted. Source audit file `train_models/mrks_source_audit_v2.json` has SHA256 `300d09095cca9c8e8809f9e45ba78d6a694c30542cf35a314cf6d74db7a7e131`; the legacy pickle SHA256 is `0ccc0cfb09814cadcc9e0537953cc324fe6ce5551654fe3254f310a25cc6ee49`, the CSV SHA256 is `4040c5ddd21cb73edcf6f54356114ff2cf216fe47346e227c5ca83180fde7e4e`, and `MRKS.zip` SHA256 is `15a49f4517c35f2159f1e66d02a817945dbf1e1e66deb46f6e612d553890ad16`.

The post-pilot SCF validation used grid level 1 and a 40-cycle limit. All four requested runs converged with finite cycle energies: H2-only checkpoint/H2 in 5 cycles (`-1.168859576696 Ha`), combined checkpoint/H2 in 5 (`-1.168777358858 Ha`), combined checkpoint/BeH2 in 6 (`-15.876881665409 Ha`), and combined checkpoint/CO in 8 (`-113.285754938281 Ha`). Every run reported `vtau=0` and exact tau independence. The SCF report is outside Git at `C:\Dev\readWFN_share_ms\lap_operator_runs_20261001\fgpu_operator_pilot_scf_wsl_20261001Tfinal\lap_operator_scf_report.json` (SHA256 `798862ce238fcc7c545e7e0944250d211599a19a2cdf42ae59e513ea38753abb`). The three-system central operator fixture and transient AO-factor cache remain outside Git. The all-90 build and independent verification passed after the pilot and SCF gates passed. No multi-objective trainer work, production training jobs, or Slurm jobs have started.

Current decision: **Gate 1, Gate 2, the H2/BeH2/CO real-data pilot, post-pilot SCF checks, and the all-90 corpus gate passed.** The h-free variational XC-operator objective is validated and ready for multi-objective training design. No multi-objective trainer or jobs were started.

No production training jobs or Slurm jobs were submitted.
