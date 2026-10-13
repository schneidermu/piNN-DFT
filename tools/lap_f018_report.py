"""Receipt-only F018 conclusions; no model, grid or derivative execution."""
import csv
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent))
import lap_f015_audit as old
import lap_f018_audit as audit
import numpy as np


def rows(name):
    with (old.ROOT/name).open(encoding='utf-8') as f:
        return list(csv.DictReader(f))


def main():
    audit.verify()
    reproduction=old.read(old.ROOT/'lap_f018_original_reproduction.json')
    floor=old.read(old.ROOT/'lap_f018_precision_floor.json')
    mixed=old.read(old.ROOT/'lap_f018_native_mixed_explicit.json')
    promoted=old.read(old.ROOT/'lap_f018_promoted_adjoint.json')
    assert mixed['pass'] and all(c['pass'] for c in promoted['checks'])
    fd=rows('lap_f018_fd_convergence.csv')
    native=rows('lap_f018_adjoint_identity_checks.csv')
    h=np.array([float(r['h']) for r in fd])
    err=np.array([float(r['absolute_gap']) for r in fd])
    exponent=float(np.polyfit(np.log(h),np.log(err),1)[0])
    original=next(r for r in fd if r['original_F017_step']=='True')
    # Exact numerical triangle inequality for saved vectors, conditional on
    # using the stated higher-precision sensitivity counterfactual.
    with np.load(audit.OUT/'diagnostic_vectors.npz') as cache:
        j=cache['analytic_full'][1:]
        jhigh=cache['high_precision_jvp']
        native_fd=cache['fd_0.0001'][1:]
        high_fd=cache['high_fd_0.0001']
        conditional_bound=float(np.linalg.norm(native_fd-high_fd)+np.linalg.norm(high_fd-jhigh)+np.linalg.norm(jhigh-j))
        conditional_lower=float(max(0,np.linalg.norm(native_fd-high_fd)-np.linalg.norm(high_fd-jhigh)-np.linalg.norm(jhigh-j)))
        baseline_norm=float(np.linalg.norm(cache['baseline'][1:]))
    error_energy=float(original['energy_coordinate_gap'])
    full_gap=float(original['full_F017_absolute_gap'])
    components=[
        {'component':'F017 energy coordinate added to operator observation','measured_error_norm':error_energy,
         'status':'measured; accounts for squared-error fraction '+str(error_energy**2/full_gap**2),'interpretation':'original gate is joint E/V, not solely V; historical FAIL preserved'},
        {'component':'native F32 parameter-storage rounding','measured_error_norm':float(original['F32_parameter_JVP_correction_norm']),
         'status':'measured actual-direction analytical correction','interpretation':'far smaller than observed9.51e-6 operator error; midpoint zero'},
        {'component':'native vs F64 PBE/arithmetic/chain counterfactual','measured_error_norm':float(original['native_minus_counterfactual_discrepancy_norm']),
         'status':'measured jointly; not an individual primitive attribution','interpretation':'same source values, learnedF64 branch, AO/overlapF64; PBE and native gradient-chain arithmetic jointly differ'},
        {'component':'native operator repeated-forward variation','measured_error_norm':floor['repeat_max_coordinate_gap'],
         'status':'3repeat measurements','interpretation':'bitwise equal; deterministic roundoff is not stochastic forward noise'},
        {'component':'independent AO density-map/mixed route vs native weak assembly','measured_error_norm':mixed['high_energy_first_vs_native_operator_first']['absolute_gap'],
         'status':'measured complete-route agreement','interpretation':'does not isolate pure assembly error from all chain arithmetic'},
        {'component':'fixed PW92/PBE contribution','measured_error_norm':'unavailable',
         'status':'algebraic parameter independence; isolated matrix cancellation not measured before budget stop','interpretation':'passthrough2:22 independent of new96; cannot assign FD error to this component without measurement'},
        {'component':'final F64 subtraction/metric alone','measured_error_norm':'unavailable',
         'status':'scale estimate only','interpretation':'eps64*baseline/h≈'+str(np.finfo(np.float64).eps*baseline_norm/1e-4)+'; assumes perturbed norm of same order; not a rigorous bound'}]
    old.table('lap_f018_component_attribution.csv',components)
    discrepancy=[]
    for r in fd:
        discrepancy.append({'h':float(r['h']),'operator_error_norm':float(r['absolute_gap']),
                            'energy_error_norm':float(r['energy_coordinate_gap']),
                            'joint_error_norm':float(r['full_F017_absolute_gap']),
                            'error_numerator_norm_2h':2*float(r['h'])*float(r['absolute_gap']),
                            'F64_counterfactual_error_norm':float(r['F64_counterfactual_absolute_gap'])})
    old.table('lap_f018_derivative_discrepancy.csv',discrepancy)
    floor.update({'native_operator_error_at_original_h':float(original['absolute_gap']),
                  'joint_error_at_original_h':full_gap,'native_FD_error_power_fit':exponent,
                  'empirical_error_numerator_min':float(min(2*h*err)),
                  'empirical_error_numerator_max':float(max(2*h*err)),
                  'conditional_saved_vector_error_interval':[conditional_lower,conditional_bound],
                  'interval_assumption':'triangle inequality over native/F64 sensitivity vectors; local to this source state/direction/h, not a uniform physical bound',
                  'original_FD_uncertainty':'native FD at1e-4 unreliable for the frozen gate; measured local error9.51e-6',
                  'strict_native_adjoint_checks':'3FAIL retained; promoted primal/JVP unchanged and3PASS under same tolerance, actual pullback discrepancies covered by measured triangle bounds',
                  'second_order_region':'not conclusively established from stored-realized F32 perturbations; native is rounding-limited',
                  'resource_limit':{'allowed_seconds':1800,'observed_wall_seconds_at_stop':2083.7927817949094,
                                    'status':'wall budget exceeded before it was checked; scientific work stopped; no further model evaluations',
                                    'first_unperformed':'isolated fixed-PW92 matrix cancellation; further clean F64 continuous-parameter O(h²) check also omitted'}})
    old.write(old.ROOT/'lap_f018_precision_floor.json',floor)
    comparison=rows('lap_f018_analytic_jvp_comparison.csv')
    qualified=mixed['high_energy_first_vs_native_operator_first']
    comparison.append({'method':'explicit grad_theta(E) dot frozen v, then grad_DP; native PBE and parameter cast',
                       **qualified,'pass':mixed['pass'],'shared_primitives':'R2/PBE/casts/AO factors; independent direct density map, no weak-form assembly'})
    # Running report again should not duplicate the appended receipt.
    deduplicated={r['method']:r for r in comparison}
    old.table('lap_f018_analytic_jvp_comparison.csv',list(deduplicated.values()))
    decision={'status':'PARTIAL — F018 completed with documented limitations','classification':'A — ANALYTICAL JVP QUALIFIED; FD PRECISION LIMIT IDENTIFIED',
              'original_F017_gate':'FAIL','independent_analytical_qualification':'PASS',
              'scientific_scope':'H2 and original96-coordinate direction only; no R2 performance/tangent projection claim',
              'limitations':['late wall-budget check2083.8s versus1800s; work stopped','no separately isolated fixed-PW92 cancellation measurement','no clean certified O(h²) interval','conditional numerical uncertainty, not global arithmetic proof'],
              'next_experiment':'Freeze one bounded matrix-free JVP/VJP prototype using promoted tangent/cotangent arithmetic and qualified energy-first grad-dot checks, initially H2 only; estimate projection cost before any15-system expansion. No serial15x96 sweep or training.',
              'updates':0,'training':False,'systems':1,'parameter_directions':1,'native_adjoint_strict_flags_preserved':[r['pass'] for r in native]}
    old.write(old.ROOT/'lap_f018_decision.json',decision)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for field,label in (('absolute_gap','Absolute operator-JVP discrepancy'),('relative_gap','Relative operator-JVP discrepancy')):
        fig,ax=plt.subplots(figsize=(6,4))
        ax.loglog(h,[float(r[field]) for r in fd],'o-',label='native')
        highfield='F64_counterfactual_'+field
        ax.loglog(h,[float(r[highfield]) for r in fd],'o-',label='F64 sensitivity copy')
        ax.axvline(1e-4,color='black',linestyle=':',label='original F017 h')
        ax.set(xlabel='FD step',ylabel=label);ax.legend();fig.tight_layout()
        fig.savefig(old.ROOT/f'lap_f018_{field}.png',dpi=140);plt.close(fig)
    realized=rows('lap_f018_realized_f32_perturbations.csv')
    fig,ax=plt.subplots(figsize=(6,4))
    ax.loglog(h,[float(r['requested_norm']) for r in realized],'o-',label='requested')
    ax.loglog(h,[float(r['positive_realized_norm']) for r in realized],'x',label='realized F32')
    ax.axvline(1e-4,color='black',linestyle=':');ax.legend();ax.set(xlabel='h',ylabel='parameter displacement norm')
    fig.tight_layout();fig.savefig(old.ROOT/'lap_f018_realized_perturbation.png',dpi=140);plt.close(fig)
    with np.load(audit.OUT/'diagnostic_vectors.npz') as c:
        x=c['analytic_full'][1:];y=c['original_fd'][1:]
    old.table('lap_f018_agreement_plot.csv',[{'coordinate':i,'analytic':float(a),'FD_original_h':float(b)} for i,(a,b) in enumerate(zip(x,y,strict=True))])
    fig,ax=plt.subplots(figsize=(5,5));ax.scatter(x,y,s=3)
    ax.plot([x.min(),x.max()],[x.min(),x.max()],color='black');ax.set(xlabel='analytical JVP',ylabel='FD, original h=1e-4')
    fig.tight_layout();fig.savefig(old.ROOT/'lap_f018_analytical_fd_agreement.png',dpi=140);plt.close(fig)
    measured=[c for c in components if isinstance(c['measured_error_norm'],(int,float))]
    fig,ax=plt.subplots(figsize=(8,4));ax.barh([c['component'] for c in measured],[float(c['measured_error_norm']) for c in measured])
    ax.set_xscale('symlog',linthresh=1e-12);ax.set_xlabel('Measured discrepancy / comparison norm (not additive attribution; zero retained)')
    fig.tight_layout();fig.savefig(old.ROOT/'lap_f018_component_comparisons.png',dpi=140);plt.close(fig)
    table='\n'.join('|'+str(r['h'])+'|'+r['absolute_gap']+'|'+r['relative_gap']+'|'+r['energy_coordinate_gap']+'|' for r in fd)
    text=f'''PARTIAL — F018 completed with documented limitations

# H2 operator-JVP precision and mixed derivatives

**Scientific classification A: analytical JVP qualified; native FD precision limit identified.** Original F017 FAIL remains unchanged. No architectural/training superiority claim.

## Original failure and observation definition

Exact B8t128 checkpoint1f38f7f8ea74334ff8f5124d2a6538619323d1f453296fa9ca6b3cd7958c437b; H2,42508points,60AOs. Exact CUDA/F64 linspace direction and64+32 tied coordinate mapping recovered from pinned F017 source. Source DM/AO/cache/overlap identities verified; only H2 loaded for calculations. R0/R2 zero-weight energy/operator embedding passed; parameters/buffers/RNG restored. No source checkpoints or datasets changed.

F017 evaluated1831 coordinates: **one energy coordinate plus1830operator coordinates**. Its original error is reproduced exactly: absolute{full_gap:.12g}, relative{reproduction['full_F017_vector']['relative_gap']:.12g}. Pure operator error at1e-4 is{float(original['absolute_gap']):.12g}, relative{float(original['relative_gap']):.12g}. Energy-coordinate error{error_energy:.12g} contributes{error_energy**2/full_gap**2:.6%}of squared joint error. Mislabeling that joint error as operator-only was an audit-description defect; it does not erase the remaining operator FD FAIL.

## Independent analytical routes and failed diagnostic helper

Operator-first double-reverse JVP uses the existing complete weak form, learnedF64/PBEF32/AOF64 boundary. Independently, explicit grad_theta(E)·v followed by grad_P uses the direct anchored AO density-matrix map and no weak-form assembly. It preserves the source DM/base features and all rho/gradient/Laplacian chains. Native-PBE mixed route differs by{qualified['absolute_gap']:.12g}, relative{qualified['relative_gap']:.12g}: **PASS** under frozen1e-8+1e-5·norm mixed tolerance. A pureF64 arithmetic copy also agrees; it is a sensitivity check, not an exact native oracle.

The initial energy-first nested `autograd.functional.jvp(create_graph=True)` helper produced a77%mixed discrepancy, and a forward-AD helper produced90%. Those failures are preserved externally. Explicit parameter-gradient dot v produces the correct mixed derivative. The demonstrated divergence is in those higher-order diagnostic graph formulations, not in the physical operator; no general PyTorch bug or particular activation cause is claimed without a minimal proof. Same R2/PBE primitives remain shared; independent AO density differentiation validates the weak-form chain/contraction, not every possible primitive globally.

Qualification concerns the differentiable mathematical/AD functional at this source state. It does not assert that the discrete bitwise Float32 mapping has an ordinary infinitesimal derivative, nor that all parameter states are qualified. The full observation-space/operator identity is tested at the original physical point only.

Three native scalar-adjoint comparisons exceed the deliberately strict1e-12+1e-8·abs tolerance by only5–9e-12; their FAIL flags remain. With promoted F64 tangent/cotangent carriers, the primal operator and JVP are unchanged (JVP difference0), and all three SAME probes pass the SAME gate, errors≤4.53e-13. Measured native/promoted pullback triangle bounds cover native discrepancies. This isolates numerical cotangent reduction/cast effects, not a relaxed threshold.

## Frozen FD ladder and numerical floor

|h|Operator absolute error|Operator relative error|Energy coordinate error|
|---|---:|---:|---:|
{table}

Native discrepancy scales approximately h^{exponent:.3f}; error numerator2h·gap ranges{min(2*h*err):.3g}–{max(2*h*err):.3g}. This supports a rounding-limited regime, not a verified native O(h²) region. The F64 sensitivity-copy errors are much smaller, but realized F32 perturbations and remaining noise prevent a clean certified second-order interval. No best h is substituted into the historical gate.

All requested/realized perturbations are recorded. Midpoint shift is exactly0 because new weights start at0 and±casts are symmetric. At original h, realized-direction JVP correction is{float(original['F32_parameter_JVP_correction_norm']):.3g}, orders below9.51e-6. Native repeated forward operators are bitwise identical (3/3): stochastic forward noise0; deterministic arithmetic resolution is not0.

At original h the saved-vector, counterfactual-conditional triangle interval for native operator discrepancy is[{conditional_lower:.12g},{conditional_bound:.12g}]. It is a measured local comparison, not a parameter-uniform rigorous floating-point bound. Observed native FD uncertainty is9.51e-6. Final F64 subtraction scale eps64·baseline/h is about{np.finfo(np.float64).eps*baseline_norm/1e-4:.3g}, assuming perturbed norms of similar order; this alone cannot explain the measured discrepancy. Native PBE/chain arithmetic jointly explains the difference from the F64 copy; learned source values, AO/overlap arithmetic remain common. Pure AO error is not separately isolated.

Fixed PW92/pass-through contributions are parameter-independent algebraically. Their isolated matrix cancellation was NOT numerically measured before stopping. Parameter-dependent learned coefficients enter PBE arithmetic, so saying “PBE is independent oftheta” would be incorrect. No component is assigned an unsupported measured error.

## Resource limitation and answers

The main numerical batch took{floor['elapsed_seconds']:.2f}s, plus bounded single-system localization jobs. However a late audit-wall check found2083.79s versus the1800s limit: **the overall wall budget was exceeded before checking it**. Scientific work was stopped; this is a limitation, not a claim of budget compliance. First omitted diagnostic: isolated fixed-PW92 matrix cancellation; further continuous-parameter F64 O(h²) study also not run. Remaining work consists only of reporting, required tests/hash checks/Git.

1:Original failure reproduced.2:Explicit independent mixed derivative agrees; initial nested-JVP diagnostic does not.3:Native strict adjoint flags FAIL; identical-primal/JVP promoted checks PASS, with measured numerical bounds.4:FD error grows as h shrinks.5:No certified O(h²) interval.6:Repeated-forward variation0; h-specific FD floor9.51e-6 at1e-4.7:Storage rounding contribution≈1.93e-10.8:FinalF64 subtraction alone is insufficient; native arithmetic/cancellation jointly measured.9:Parameterized PBE arithmetic affects FD; fixed contribution separately unmeasured.10:Independent AO route agrees2.35e-11; no isolated pure-assembly bound.11:Joint-vector mislabel and unsupported nested diagnostic graph identified; no production operator patch.12:Joint1.677349% and operator0.801792% decomposed and consistent with measured rounding-limited native FD.13:Original frozen operator JVP independently qualified locally.14:Global error bounds, primitive-level attribution and clean second-order interval remain uncertain.15:Freeze a bounded H2-only matrix-free JVP/VJP feasibility protocol with promoted carriers and explicit energy-first verification; cost it before any15-system expansion. No serial15×96 sweep or training.

Scientific classification:A
Original F017 FD gate:FAIL
Independent analytical qualification:PASS
Numerical precision floor:observed native operator FD error9.509634e-6 at h1e-4; local conditional bound only
H2 systems evaluated:1
New parameter directions evaluated:1 (realized-rounding corrections belong to the same frozen direction)
Optimizer updates:0
Training performed:NO
Production architecture modified:NO
Clean28 accessed:NO
SCF executed:NO
F018 commit/Git push/remote HEAD:reported after commit and remote verification
'''
    (old.ROOT/'lap_f018_operator_jvp_precision_report.md').write_text(text,encoding='utf-8')
    old.write(old.ROOT/'lap_f018_provenance.json',{'parent':'072e98e6ef97f10328aa858aa068e4c938ca17ba',
              'artifacts':{p.name:old.sha(p) for p in old.ROOT.glob('lap_f018_*') if p.is_file() and p.name!='lap_f018_provenance.json'},
              'external':{p.name:old.sha(p) for p in audit.OUT.iterdir() if p.is_file()},
              'scripts':{p.name:old.sha(p) for p in (old.ROOT/'tools').glob('lap_f018_*.py')},
              'updates':0,'training':False,'production_sources_unchanged':True,
              'tests':'pending final required suite','restoration':'model/buffer/RNG context assertions PASS; checkpoint byte pins PASS'})


if __name__=='__main__':
    main()
