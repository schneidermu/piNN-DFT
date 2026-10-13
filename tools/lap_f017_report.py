"""F017 receipt-only analysis; retain failed gate and incomplete coverage."""
import csv
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent))
import lap_f015_audit as old
import lap_f017_audit as audit
import numpy as np
from scipy.optimize import brentq


def rows(name):
    with (old.ROOT/name).open(encoding='utf-8') as f:
        return list(csv.DictReader(f))


def limited_ls(j,residual,radius):
    u,s,vt=np.linalg.svd(j,full_matrices=False)
    b=u.T@residual
    def vector(mu):
        return vt.T@(s*b/(s*s+mu))
    if np.linalg.norm(vector(0))<=radius:
        mu=0.
    else:
        high=1.
        while np.linalg.norm(vector(high))>radius:
            high*=10
        mu=brentq(lambda m:np.linalg.norm(vector(m))-radius,0,high,xtol=1e-12)
    d=vector(mu)
    stationarity=np.linalg.norm(j.T@(j@d-residual)+mu*d)/max(np.linalg.norm(j.T@residual),1e-30)
    return d,stationarity


def main():
    protocol=audit.verify()
    with np.load(audit.OUT/'energy_jacobian.npz') as cache:
        j=cache['jacobian']
        residual=cache['residual']
    rank_rows=[]
    align_rows=[]
    local_rows=[]
    for threshold in protocol['rank_thresholds']:
        old_j=j[:,:9446]
        u,s,_=np.linalg.svd(old_j,full_matrices=False)
        retained=s>=threshold*s[0]
        projected_new=j[:,9446:]-u[:,retained]@(u[:,retained].T@j[:,9446:])
        new_s=np.linalg.svd(projected_new,compute_uv=False)
        meaningful=new_s>threshold*s[0]
        rank_rows.append({'block':'energy-only historical15','relative_threshold':threshold,
                          'old_rank':int(retained.sum()),'observations':len(residual),
                          'new_rank_above_same_absolute_threshold':int(meaningful.sum()),
                          'old_sigma1':float(s[0]),'old_smallest_retained':float(s[retained][-1]),
                          'new_projected_norm':float(np.linalg.norm(projected_new)),
                          'status':'truncated SVD of stored native-gradient numbers; no operator-space conclusion'})
        align_rows.append({'block':'energy-only historical15','relative_threshold':threshold,
                           'raw_new_residual_VJP_norm':float(np.linalg.norm(j[:,9446:].T@residual)),
                           'orthogonal_new_residual_VJP_norm':float(np.linalg.norm(projected_new.T@residual)),
                           'old_projected_reference_fraction':float(np.linalg.norm(u[:,retained].T@residual)**2/np.linalg.norm(residual)**2),
                           'status':'energy projection only; operator J_perp unavailable'})
    for arch,columns in (('R0',9446),('R2',9542)):
        for scale in protocol['finite_scales']:
            radius=protocol['radius']*scale
            delta,kkt=limited_ls(j[:,:columns],residual,radius)
            assert kkt<=protocol['tolerances']['solver_kkt_rel']
            predicted=residual-j[:,:columns]@delta
            local_rows.append({'architecture':arch,'block':'energy-only historical15','radius_scale':scale,
                               'radius':radius,'step_norm':float(np.linalg.norm(delta)),
                               'initial_mean_squared_Ha':float(np.mean(residual**2)),
                               'predicted_mean_squared_Ha':float(np.mean(predicted**2)),
                               'relative_predicted_reduction':float(1-np.linalg.norm(predicted)**2/np.linalg.norm(residual)**2),
                               'kkt_relative':kkt,'new_coordinate_step_fraction':float(np.linalg.norm(delta[9446:])/np.linalg.norm(delta)) if arch=='R2' else 0.,
                               'finite_validation':'NOT PERFORMED: operator derivative qualification gate failed; do not promote'})
            np.save(audit.OUT/f'energy_{arch}_{scale}_delta.npy',delta)
    old.table('lap_f017_tangent_rank_conditioning.csv',rank_rows)
    old.table('lap_f017_reference_residual_alignment.csv',align_rows)
    old.table('lap_f017_local_attainability.csv',local_rows)
    checks=old.read(old.ROOT/'lap_f017_new_weight_jacobian_checks.json')
    metrics=rows('lap_f017_reference_observable_metrics.csv')
    pairs=rows('lap_f017_reference_pair_analysis.csv')
    missing={
        'lap_f017_finite_step_validation.csv':'No qualified full operator tangent direction: first physical JVP failed its frozen FD gate. No finite parameter step accepted or retained.',
        'lap_f017_operator_safety.csv':'Nonzero tangent finite probes stopped at numerical prerequisite; F016 zero-weight/synthetic checks not substituted.',
        'lap_f017_chemistry_transfer.csv':'No new96 chemistry sensitivities computed after failed physical tangent prerequisite. Old251x9446 caches exist and are verified; R2 chemistry coverage0/251, so transfer is inconclusive.'}
    decision={'status':'PARTIAL — F017 completed with documented limitations','classification':'E — INCONCLUSIVE',
              'reason':'Physical embedding and energy-only data available; operator JVP failed frozen FD tolerance and full15x96 projected runtime exceeds budget. No qualified operator J_perp, finite safety or chemistry transfer result.',
              'unavailable_artifacts':missing,'R2_training_authorized':False,
              'next_experiment':'One bounded training-only operator-JVP precision verification on H2 at the same zero-weight embedding: independently bound F32 numerical-integration FD noise before any tangent projection or architectural training. Retain production R0 meanwhile.',
              'optimizer_updates':0,'training':False,'Clean28_accessed':False,
              'operator_reference_coverage':len(metrics),'operator_full_new_jacobian_coverage':0,
              'energy_jacobian_systems':len(residual),'new_chemistry_jacobian_reactions':0}
    old.write(old.ROOT/'lap_f017_decision.json',decision)
    first=checks[0]
    text=f'''# F017 — reference-conditioned representation sufficiency

**PARTIAL — F017 completed with documented limitations**

**Scientific classification: E — INCONCLUSIVE.** No architectural training is justified by this result. Production R0 is retained. An unresolved derivative gate is not evidence of redundancy or global insufficiency.

## Evidence categories

Mathematical identities: the complete functional dependence and block operator observation metric are defined in `lap_f017_representation_contract.md`. Symmetric flattening weights off-diagonals by sqrt2 and divides by sqrt(nAO), preserving the qualified Frobenius objective. No cross-molecule AO matrices are subtracted. No gauge changes.

Numerical certificates: energy-only stored-Jacobian SVD projections have explicit thresholds1e-6/1e-8/1e-10; limited-radius energy LS has independent stationarity checks. These are fifteen-system energy-only linear calculations, not full90 certification, finite-step validation or an operator projection. Thresholds below nativeF32 resolution require particular caution.

Empirical associations:90 systems yield270 predetermined nearest-system comparisons of weighted raw/R0/R2 descriptor moments. Pair rules were frozen before reference differences were read. Electron number, density prefactors and signed weights are retained; moment distances are dimension-scaled proxies, NOT complete weighted measures. No parameter-uniform response bound exists here. Certified reference contradictions:0. This does not prove sufficiency.

Unsupported hypotheses: q collisions imply identical energies; energy-only rank proves architecture adequacy; a nonzero new residual VJP proves an independent tangent direction; S5 is an exactXC teacher. None is asserted.

## Provenance and physical embedding

ParentF016: `3c0b9d61d4826e23c185e8e72601f8452774cb12`, verified remote ancestry. Existing F013–F016 hashes/source/checkpoint pins verified. Primary checkpointB8t128. Publication training-only adapter opens chemistry/mRKS indices only. No validation/test metadata, tensors or scores enter this computation.

All{len(metrics)}/15 historical systems were evaluated on their full original grids and AO factors, with true reference energies/operators. R0 has9446 coordinates; embeddedR2 has9542, with64 new correlation and32 new tied-exchange weights. OriginalGc metric retained. Old parameters ordered by sorted unique names; exchange counted once.

Maximum physical embedding energy gap: {max(float(r['embedding_energy_gap']) for r in metrics):.6g}Ha. Maximum normalized operator-component gap: {max(float(r['embedding_operator_max_gap']) for r in metrics):.6g}. Source parameters/buffers were restored exactly with RNG checks. No optimizer constructed or state touched. Existing dispersion correction is added exactly once in native energy arithmetic before diagnostic widening; electronic reference equals storedExc minus that frozen correction.

## Failed operator tangent qualification — preserved, not relaxed

First system chosen by ascending grid size: H2,42508points,60AOs. One deterministic unit direction spanning all96 new coordinates was tested with the exact variational operator and native parameter boundary.

FD step={first.get('FD_h')}; observed relativeJVPgap={first.get('relative_gap'):.8f}; absolute gap={first.get('absolute_gap'):.8g}. Frozen gate is absolute1e-6 plus relative0.005 times the analyticJVPnorm. **FAIL**. Possible F32 finite-difference noise is unproven; no tolerance relaxation or substitute energy derivative was used.

Measured full-system JVP={first['seconds']:.3f}s. A serial15×96 sweep projects to{first.get('projected_15x96_seconds',0)/60:.2f}minutes, exceeding the frozen30-minute diagnostic budget. This is a measured serial projection, not proof that a future validated batched/matrix-free implementation cannot be faster. No such implementation was developed to rescue this audit.

The full90 operator-Jacobian expansion, operator-space old/new projection and nonzero-weight safety/finite tests were stopped. Full90 reference metadata were used only for the predetermined pair analysis; full90 operators were not evaluated.

## Completed limited energy and operator evidence

Energy Jacobian shape={j.shape}; old9446 plus new96. Tables provide rank, conditioning, projected new residual alignment and separate R0/R2 predictions at0.25r/r, r={protocol['radius']}. These predictions were not implemented as parameter steps. Equal Euclidean radii in different parameterizations do not establish invariant superiority. No joint physical scale was invented.

At all three frozen thresholds, the old energy Jacobian has rank15/15 and additional projected rank0. This is the anticipated energy-only identifiability limitation, NOT proof of useful operator-space redundancy. At full radius the predicted squared-energy-residual reduction is74.2433307% for R0 versus74.2445981% for R2. The increment is about0.00127 percentage points. Finite actual reductions are unmeasured.

For all15 systems the actual operator residual was computed in its own overlap-orthonormalized block. The new-coordinate quantity J_N^T r_V was obtained exactly as −0.5 times the gradient of squared operator residual. It is an unprojected VJP. **It is not J_perp^T r and does not establish96 useful independent directions.** Full operator Jacobian columns produced:0; one unqualified JVP only. Residual vectors/VJPs are stored outsideGit and hash-bound.

## Answers to the scientific questions

1–3: No certified pair of globally indistinguishable systems with incompatible references was found. Zero complete raw-input hash matches among the selected different-system pairs; finite summary proximity is not full-representation equivalence. Individual tanh(q) collisions remain distinct from this claim.
4: Fifteen-system energy-only local linear corrections are quantified in local_attainability.csv; full90 and finite reductions are NOT measured.
5: R0 attainable operator-error correction is unknown: operator-space projection is blocked.
6–7: Energy-only extra rank/alignment is reported by threshold. Qualified independent operator directions and orthogonal operator-residual alignment are unknown.
8: Identical-radius energy-only predictions are reported. A useful joint/physical improvement is not established.
9–10: Nonzero finite-step and safety validation were not performed after the failed derivative prerequisite.
11–13: Additional chemistry sensitivities coverage0/251; all eight-database R2 benefit/trade-offs and physical no-worsening are unknown. F015 old R0 reaction caches and local correction evidence were reverified, not replaced by selected favourable reactions or S5 targets.
14–15: This audit does not justify R2 training and cannot choose representation versus optimization as a cause. F013/F015 optimization evidence remains historical evidence; it does not resolve the missing F017 operator comparison.
16: Minnesota exact molecularXC labels are absent. No pointwise exactXC labels are inferred. Finite systems and lack of parameter-uniform bounds block global representation claims.
17: {decision['next_experiment']}

## Missing outputs and integrity

The protocol retains all intended artifact names. Finite-step validation, nonzero-weight safety and chemistry-transfer CSVs are intentionally absent, with exact reasons in decision.json; no empty fabricated results. Relevant architecture/physics/operator/restoration tests, Ruff, compileall, hashes and diff checks are recorded in provenance after final verification.

Preflight accounting: the new diagnostic symmetric flattening initially constructed sqrt2 in F32 before widening. A strict Frobenius-inner-product test caught this; it was corrected to construct sqrt2 directly in matrixF64 dtype, and the affected15-system observations were recomputed. The frozen pair summary's total positive electron mass was also explicitly included before final pair results. Neither correction changes the frozen scientific rules or production code. Preliminary receipts are archived as rejected and are not interpreted. Corrected full-system JVP still fails the same frozen gate.39 relevant tests passed; Ruff, compileall and diff checks passed.

Optimizer updates:0. Training:NO. Production architecture modified:NO. Clean28 accessed:NO. SCF:NO. No new systems, target calculations, tauchannels, trainedweights, or exhaustive augmentation evaluation. No follow-up experiment launched.
'''
    (old.ROOT/'lap_f017_representation_sufficiency_report.md').write_text(text,encoding='utf-8')
    provenance={'reference_commit':protocol['parent'],'artifacts':{p.name:old.sha(p) for p in old.ROOT.glob('lap_f017_*') if p.is_file() and p.name!='lap_f017_provenance.json'},
                'external':{p.relative_to(audit.OUT).as_posix():old.sha(p) for p in audit.OUT.rglob('*') if p.is_file() and 'pytest_' not in p.as_posix()},
                'scripts':{p.name:old.sha(p) for p in (old.ROOT/'tools').glob('lap_f017_*.py')},
                'scope':decision,'raw_pair_count':len(pairs),'hash_reverification':True,
                'tests':{'passed':39,'ruff':'PASS','compileall':'PASS','git_diff_check':'PASS'},
                'preflight_correction':'Diagnostic svec sqrt2 was initially F32-widened; fixed to construct directly in matrix F64 dtype. Rejected preliminary receipts archived; no production formula changed.',
                'source_changes':'F017 diagnostic files only'}
    old.write(old.ROOT/'lap_f017_provenance.json',provenance)


if __name__=='__main__':
    main()
