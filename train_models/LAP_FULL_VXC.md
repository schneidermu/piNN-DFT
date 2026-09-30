# Tau-free NN-PBE-Lap and full mRKS Vxc

Implementation branch: `lap_full_vxc`, based on the pulled `vxc_training`
HEAD `fc483a1`. The requested `1f9a6b87c1d84c01e0fb3b0050298f7c4c2141fb`
is in its ancestry. Existing local changes were preserved in the original
checkout; this branch was implemented in a separate worktree.

**Real-data preprocessing is blocked.** The existing center-only corpus cannot
provide independently evaluated displaced densities or spin gradient vectors.
No production corpus, production calibration results, final loss weights, or
production h have been fabricated.

## Implementation report

1. **Branch/commit:** `lap_full_vxc`; use `git rev-parse HEAD` for the final
   implementation commit. The accompanying final response gives its exact hash.
2. **Files:** `NN_models_lap.py`, `lap_vxc.py`, `lap_data.py`,
   `lap_checkpoint.py`, `lap_training.py`, `train_lap.py`, `lap_diagnostics.py`,
   `lap_analytic.py`, four `test_lap*.py` files, `run_lap_checks.py`, this report,
   and `test_models/DFT/lap_functional.py` are new. `optuna_joint.py` only adds
   explicit factory registration and a guard against using the old Vxc loss.
   Historical model implementations, checkpoints, S5/timing scripts and
   selection policies retain their behavior.
3. **Descriptors:** correlation is `[n_a,n_b,s_a,s_total,s_b,q_a,q_b,zeta]`
   (8 inputs); exchange uses tied `[s_sigma,q_sigma]` networks (2 inputs).
   Raw model input remains nine columns: rho_a,b; sigma_aa,total,bb;
   unused tau_a,b; lapl_a,b. Tau columns are never read into descriptors or
   energy. The output remains 29 constants with existing activations/scales,
   modified-PBE expressions, and Gx/Gc parameterization.
4. **Constraints:** remove alpha coordinates only in the new class. UEG sets
   all s and q to zero, keeping density and zeta; mu/reference=1,
   G_NN=0, beta/reference=1, Gc=1. The high-density anchor sets bounded
   n_a,n_b=1 and retains remaining descriptors; gamma/reference=1 and Gc=1.
   The rapid-gradient anchor sets the three bounded s=1 and retains q/zeta;
   Gc=1. Kappa retains the inherited sigmoid bound. Correlation is symmetric
   under spin exchange and exchange weights remain tied. The new s regularizer
   subtracts its zero value so raw zero sigma maps exactly to bounded s=0.
   Old constraint helpers are untouched. Zeroing all NN weights does not force
   canonical kappa: the inherited sigmoid remains; tests use explicit canonical
   PBE constants for the exact PBE limiting comparison.
5. **Full Vxc:** `LapEnergy` evaluates the same PBE energy per electron times
   total density. Predicted local constants remain attached. Autograd computes
   C=de/drho, standard esigma=de/d(aa,ab,bb), B=de/dlapl.
   A_a=2 esigma_aa grad_a+esigma_ab grad_b; the beta expression is analogous.
   Training uses create_graph=True. The prediction is C-div A+lap B, never C alone.
6. **Spatial derivatives/h:** central first differences for div A and the
   Cartesian seven-point Laplacian for B. Order: center,+x,-x,+y,-y,+z,-z.
   Coordinates, recorded h, and FD assembly use float64; h is positive and in
   Bohr. No production h default exists. Float64 assembly does not repair errors
   already incurred in float32 local derivatives. Training requires an explicit
   precision choice; diagnostic precision is independently selectable.
7. **Shifted reference density:** `evaluate_stencil` calls a supplied ORIGINAL
   reference evaluator at every actual Cartesian coordinate. The AO adapter
   uses the supplied reference spin density matrices in the exact molecule/AO
   basis and analytic AO derivatives through order two. It performs no SCF
   calculation and supplies no surrogate density. `write_stencil_h5` preserves
   validated reference targets and refuses overwrite. The original generator
   and reference matrices/wavefunctions are absent, so this adapter has only
   been tested on deliberately synthetic/reference-AO smoke fixtures.
8. **Protocol/schema:** `diet-clean-mn-all-mrks-lap-fullvxc-v1`, default separate
   directory `checkpoints_dietclean_lap_fullvxc_v1`. `StencilFeatures` is
   float64 `(N,7,10)`: rho_a,b, grad_a_xyz, grad_b_xyz, lapl_a,b. Center values
   are stencil index zero. Records also contain coordinates, stencil
   coordinates, weights, `(N,2)` Vxc, scalar E_xc, h, spin/target identity,
   explicit gauge availability and original source provenance. H5 dataset names
   are `coords,stencil_coords,stencil_features,weights,vxc,E_xc`. Required attrs
   are `protocol,stencil_version,h_bohr,source_spin,target_kind,vxc_layout,
   gauge_metadata,source_provenance`. JSON gauge metadata includes boolean
   `available`. Source provenance includes `generator,reference_density,
   molecule_basis_ao_order,full_vxc_target` identifiers/descriptions.
   Immutable manifest verification checks all 90 systems, 284/268 Minnesota
   counts, exact 16 exclusions, source/derived artifact hashes, stencil
   version/order/h/units, architecture, source H5 metadata and record metadata.
   The copied Minnesota artifacts must match their original verified manifest.
   Source files/manifests must remain accessible for verification.
9. **Potential loss:** sum(rho_sigma weight (Vpred_sigma-Vref_sigma)^2) divided
   by sum(rho_sigma weight), over points and both spins. Absolute pointwise MSE;
   no mean subtraction, offset fitting, gauge projection or integrated signed
   residual. Energy loss retains the existing `batch_exc` scaling (Hartree to
   kcal/mol); Minnesota retains `batch_fchem`, integration, D3 option, and
   seeded epoch augmentation. mRKS E_xc is the pure XC integral.
10. **Spin targets:** both channels are preserved. H5 axes are explicit
    `spin,point` or `point,spin`, including ambiguous 2x2 arrays. One-channel
    `point` targets require source_spin=0, target_kind=common-rks, and equal
    reference spin densities at all seven points. Arbitrary open-shell targets
    are never averaged. The current training CLI requires common-RKS records;
    the internal Euler math/schema support general two-spin data.
11. **Memory/distributed updates:** full-system electron normalization is fixed;
    each Vxc chunk is evaluated without retaining its training graph, then
    recomputed in backward, accumulated and released. Center-energy chunks use
    activation checkpointing. Reference tensors remain resident for the system;
    only the expensive derivative graph is bounded by the point chunk size.
    Dropout must be zero. torchrun uses one system/rank and explicit SUM/world
    rank averaging, matching the historical manual objective merger. The number
    of chunks never affects system sampling or optimizer-step cadence. Two-rank
    CPU Gloo equality is tested; V100/NCCL memory/timing remains unmeasured.
12. **SCF:** `LapFunctional.make_rks` uses the existing `RKS_with_Laplacian`.
    Local partial derivatives enter the variational AO matrix, including both
    lapl(AO)*AO terms and 2 grad(AO)*grad(AO); vtau is exactly zero. No spatial
    FD is inserted into SCF. RKS only; higher derivatives and new UKS evaluation
    are outside this implementation. The existing matrix code needed no change.
13. **Calibration:** `lap_diagnostics.py calibrate` reports a reproducible
    Minnesota sample plus one mRKS system at fresh initialization and after
    clean-pool predopt, with all three objective losses and parameter-gradient
    norms. This is a training diagnostic, not validation or checkpoint selection.
    System mode reports electron number, E_xc target/prediction, weighted
    RMSE/MAE/bias, max error, separate component RMS/relative RMS, nonfinite
    fraction, h, precision, and optional peak CUDA memory. No real-data
    calibration could be run without valid reference stencils.
14. **Validation:** 47 new tests pass on Linux, including analytic LDA/GGA/Lap
    RKS and unequal-spin checks, second-order convergence, NN derivative and
    parameter finite differences, exact tau independence/anchors, chunk loss
    and gradients, rank averaging/cadence, SCF smoke, variational matrix,
    canonical PBE limit, provenance corruption, checkpoint separation, and
    calibration/predopt smoke. All 135 historical training tests and 17
    scientific/finite-XC regression tests pass across Windows/Linux. Total:
    199 passing tests; 18 existing CUDA tests skipped on CPU. Historical test
    files need separate processes because they inject global import stubs.
    Windows shell-fixture tests were rerun successfully on Linux; legacy
    preprocessing tests needed GIT_DIR pointing to the Windows worktree's actual
    Git metadata when invoked through WSL. New-file Ruff passes; the existing
    optuna_joint.py has 121 pre-existing Ruff findings and zero newly introduced
    findings. Compileall, all existing
    training shell syntax checks, and git diff --check pass. The standalone
    `test_functionals_convergence.py` is an external dataset/SCF driver, not a
    unit-test suite; it was compiled but not run against Diet data.
15. **Unresolved:** obtain the exact original generator and reference evaluator
    inputs, generate all 90 verified stencil records, measure real-system h and
    precision convergence, run clean-pool calibration, and measure memory and
    timing on 2xV100 before selecting production weights/schedules. Diet30 stays
    external fully self-consistent validation; Diet100-minus-Diet30 (98
    reactions) remains untouched. No internal validation split was added.

No production training jobs or Slurm jobs were submitted.

## Source audit and required inputs

The current tree, Git history, and nearby workspace files were searched for
`gen_h5_with_vrho.py`, `genGRDh5`, mRKS/vrho/density-grid generators, AO matrices,
and original wavefunctions. Only a small Multiwfn command-input list
(`denrho/content/genGRD.txt`) was found; it is not a displaced-density evaluator.
The downloaded legacy pickle has 90 systems / 8,271,091 center points,
float32 center Grid/Weights/Vrho/E_xc data and no reference AO density matrices
or gradient directions. It is insufficient for this protocol.

Required from the data author:

- Original `gen_h5_with_vrho.py` / `genGRDh5` source and the mRKS export code,
  or an equivalent original reference-density evaluator.
- For every system, original spin AO density matrices or a wavefunction format
  with a reliable evaluator, plus exact molecular geometry, coordinate units,
  charge/spin, basis specification and AO ordering/normalization.
- Original central coordinates/quadrature weights, unchanged full mRKS Vxc
  channels and E_xc, spin-channel identity, and any available potential
  alignment/gauge metadata. Unavailable alignment must be recorded explicitly;
  it is not an instruction to project away a constant.

Nearest-neighbor interpolation, surrogate density fitting, recovering directions
from sigma norms, and fallback to partial Vrho are explicitly rejected.

## Numerical reference and measured precision

[NNLap, arXiv:2609.34194v1](https://arxiv.org/html/2609.34194v1) Eq.16 supplies
the methodological reference: autograd local derivatives and spatial finite
differences for the outer operators. The inspected paper/PDF did not specify
an authoritative FD spacing; no public implementation or supplement specifying
one was located. Its architecture and potential loss were not adopted.

For the analytic Lap toy, maximum absolute potential error is:

| h (Bohr) | float32 local derivatives | float64 local derivatives |
|---:|---:|---:|
| 0.2 | 6.2601e-3 | 6.2580e-3 |
| 0.1 | 1.5659e-3 | 1.5667e-3 |
| 0.05 | 4.3811e-4 | 3.9180e-4 |
| 0.02 | 3.2207e-4 | 6.2694e-5 |
| 0.01 | 9.7500e-4 | 1.5674e-5 |
| 0.001 | 1.7265e-1 | 1.5631e-7 |
| 0.0001 | 1.7588e1 | 4.0488e-8 |

Float64 shows the expected second-order truncation regime; float32 cancellation
eventually dominates. These values characterize this analytic density only,
not a validated molecular production spacing. AO-density PBE convergence is
also covered by a deterministic unit test.

## Commands and boundaries

Run commands from `train_models` with the project's working Torch environment:

```bash
python lap_diagnostics.py convergence
python run_lap_checks.py
python lap_data.py --mn-corpus checkpoints_dietclean_noval_v1 \
  --stencil-dir /path/to/original-reference-stencils
python lap_diagnostics.py system --corpus checkpoints_dietclean_lap_fullvxc_v1 \
  --system-index 0 --dtype float64 --point-chunk-size 4096
python lap_diagnostics.py calibrate --corpus checkpoints_dietclean_lap_fullvxc_v1 \
  --predopt-epochs 2 --dtype float64 --point-chunk-size 4096
```

Production training is not launched by preprocessing or diagnostics. The separate
`train_lap.py --help` entry point requires explicit epoch count, learning rate,
precision, output directory and all three objective weights; it has no inherited
S5 Vxc weight or production schedule. For the intended two-rank setup, use
torchrun with `--nproc_per_node=2` only after data, convergence and calibration
are ready. Every epoch snapshot records architecture/protocol/precision/h;
the run stores its arguments and full corpus manifest. No snapshot is selected
using training diagnostics.

Standard gradient variables are `(aa,ab,bb)` in local differentiation and PBE.
Only the NN input uses `(aa,total,bb)` where total=aa+2ab+bb. Stencil inputs store
actual vectors, so no historical center-grid sigma repair is applied to mRKS.
Minnesota boundary assertions reject disagreement between raw model sigma-total
and its standard PBE sigma tensors.

Checkpoint helpers require `pcPBELMLOptimizerV2Lap-v1` plus descriptor/protocol
metadata and a version marker in the state dict. Use
`LapFunctional.from_checkpoint(path)` for the new SCF adapter; old NN_FUNCTIONAL
keys retain the historical tau architecture.
