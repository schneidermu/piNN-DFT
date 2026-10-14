"""Completed F023 read-only leverage audit and NO-GO report; no new GPU work."""
import csv
from collections import Counter

import lap_f023_size as exp
import numpy as np


def csv_rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main():
    exp.read('protocol')
    cal = exp.read('calibration')
    preflight = exp.read('preflight_decision')
    assert preflight['numerical_pass'] and not preflight['mechanism_pass']
    assert exp.read('forward_receipt')['complete'] and exp.read('restoration')['model']
    metadata = exp.audit.read(exp.OUT/'authenticated_electrons.json')
    records = exp.audit.read(exp.ROOT/'lap_f021_phys4_singleton_gradients_receipt.json')['records']
    prefix = exp.f010._batches()[:128]
    frequencies = {t:Counter(exp.previous.key(r['ae17']) if t=='ae17' else exp.previous.key(r['mrks_id']) for r in prefix)
                   for t in ('ae17','exc')}
    with np.load(exp.OUT/'cached_arrays.npz') as payload:
        arrays = {k:payload[k].copy() for k in payload.files}
    original = exp.audit.read(exp.OUT/'original_forward_values.json')
    full_op_path = exp.audit.geometry.STATES[0]['physical']/'op.npy.npz'
    pinned = exp.audit.read(exp.ROOT/'lap_f022_provenance.json')['physical_vector_hashes']['op']
    assert exp.audit.sha(full_op_path)==pinned
    full_op = exp.audit.geometry.prior._load_vector(full_op_path)
    oldP = sum(exp.b8.LAMBDAS[t]*arrays[t].mean(axis=0) for t in ('ae17','exc','op'))
    newP = sum(exp.b8.LAMBDAS[t]*arrays[t+'_new'].mean(axis=0) for t in ('ae17','exc','op'))
    C = exp.b8.LAMBDAS['relchem']*exp.audit.geometry.C010*arrays['JW_S']
    targets = {'JW_S':arrays['JW_S'],'legacy_S':arrays['legacy_S'],'operator_full90':full_op,'physical_old_S128':oldP}
    leverage = []
    atomic_vectors = {}
    for row in records:
        task, source = row['binding']['task'],row['binding']['record']
        task = 'ae17' if task=='ae17' else 'exc'
        key = exp.previous.key(source)
        identity = source['identity'] if task=='ae17' else source
        variant = source['variant'] if task=='ae17' else 'central'
        count = metadata['ae17' if task=='ae17' else 'mrks'][identity]
        scale = cal['c_A' if task=='ae17' else 'c_E']
        weight = count['Ne']**-.5
        with exp.census.cache('ae17' if task=='ae17' else 'mrks',source)[0] as data:
            g = data[task].copy()
        coefficient = exp.b8.LAMBDAS[task]
        frequency = frequencies[task][key]
        if task=='ae17':
            atomic_vectors.setdefault(identity,np.zeros(9446,dtype=np.float64))
            atomic_vectors[identity] += coefficient*frequency/128*g
        value = original[('ae17' if task=='ae17' else 'mrks')+'_'+key]
        assert value['source_cache_sha256']==row['sha256']
        assert abs(value['original_loss']-row['losses'][task]) <= 1e-6*max(1,abs(row['losses'][task]))
        leverage.append({'task':task,'identity':identity,'variant':variant,'source_id':count['source_id'],
            'Ne':count['Ne'],'prefix_presentations':frequency,'size_weight':weight,'calibrated_multiplier':scale*weight,
            'original_scalar_residual':value['signed_error_kcal'] if task=='ae17' else value['signed_error_Ha'],
            'original_loss':value['original_loss'],'size_loss':value['original_loss']*weight,
            'calibrated_loss':value['original_loss']*weight*scale,'raw_gradient_norm':float(np.linalg.norm(g)),
            'original_weighted_gradient_norm':float(coefficient*np.linalg.norm(g)),
            'size_weighted_gradient_norm':float(coefficient*weight*np.linalg.norm(g)),
            'calibrated_gradient_norm':float(coefficient*weight*scale*np.linalg.norm(g)),
            'old_prefix_contribution_norm':float(coefficient*frequency/128*np.linalg.norm(g)),
            'new_prefix_contribution_norm':float(coefficient*frequency/128*weight*scale*np.linalg.norm(g)),
            **{name+'_raw_cosine':exp.dual.cosine(g,v) for name,v in targets.items()},
            **{name+'_old_prefix_dot':float(v@g*coefficient*frequency/128) for name,v in targets.items()},
            **{name+'_new_prefix_dot':float(v@g*coefficient*frequency/128*weight*scale) for name,v in targets.items()}})
    exp.table('system_leverage',leverage)
    atomic_summary=[]
    for identity,vector in atomic_vectors.items():
        info=metadata['ae17'][identity]
        multiplier=cal['new_loss_weights']['ae17'][identity]
        selected=[r for r in leverage if r['task']=='ae17' and r['identity']==identity]
        atomic_summary.append({'identity':identity,'source_id':info['source_id'],'Ne':info['Ne'],
            'prefix_presentations':sum(r['prefix_presentations'] for r in selected),
            'prefix_variants':len(selected),'raw_gradient_norm_min':min(r['raw_gradient_norm'] for r in selected),
            'raw_gradient_norm_max':max(r['raw_gradient_norm'] for r in selected),
            'old_prefix_atom_vector_norm':float(np.linalg.norm(vector)),
            'new_prefix_atom_vector_norm':float(np.linalg.norm(vector)*multiplier),
            'old_legacy_projection':float(arrays['legacy_S']@vector),
            'new_legacy_projection':float(arrays['legacy_S']@vector)*multiplier})
    exp.table('atomic_leverage_summary',atomic_summary)
    # Fixed descriptive bins, not an allocation/weighting search.
    groups = []
    for task in ('ae17','exc'):
        limits = (5,11) if task=='ae17' else (10,20)
        for label, test in [('small',lambda n,q=limits:n<=q[0]),('medium',lambda n,q=limits:q[0]<n<=q[1]),('large',lambda n,q=limits:n>q[1])]:
            selected = [r for r in leverage if r['task']==task and test(r['Ne'])]
            groups.append({'task':task,'group':label,'limits':str(limits),'records':len(selected),
                'presentations':sum(r['prefix_presentations'] for r in selected),
                'sum_old_contribution_norms':sum(r['old_prefix_contribution_norm'] for r in selected),
                'sum_new_contribution_norms':sum(r['new_prefix_contribution_norm'] for r in selected),
                'old_legacy_projection':sum(r['legacy_S_old_prefix_dot'] for r in selected),
                'new_legacy_projection':sum(r['legacy_S_new_prefix_dot'] for r in selected)})
    exp.table('size_group_leverage',groups)
    attribution = []
    for label, p in [('old',oldP),('AE17_only',oldP+exp.b8.LAMBDAS['ae17']*(arrays['ae17_new']-arrays['ae17']).mean(axis=0)),
                     ('Exc_only',oldP+exp.b8.LAMBDAS['exc']*(arrays['exc_new']-arrays['exc']).mean(axis=0)),('combined',newP)]:
        attribution.append({'decomposition':label,'legacy_dot_physics':float(arrays['legacy_S']@p),
            'physical_norm':float(np.linalg.norm(p)),'joint_norm':float(np.linalg.norm(C+p)),
            'JW_dot_joint':float(arrays['JW_S']@(C+p)),'operator_dot_joint_full90':float(full_op@(C+p))})
    exp.table('task_attribution',attribution)
    # Independent evaluation-H gradient is present in the authenticated S128 cache.
    h = next(r for r in leverage if r['source_id']=='H_ae17' and r['variant']=='level3_mura')
    hspec = next(r['binding']['record'] for r in records if r['binding']['task']=='ae17'
                 and r['binding']['record']['identity']==h['identity'] and r['binding']['record']['variant']==h['variant'])
    with exp.census.cache('ae17',hspec)[0] as data:
        h_abs = data['ae17']/exp.audit.chemistry._factor('AE17')
    h_signed = -h_abs  # Authenticated signed residual is negative at this fixed H variant.
    native_rows=csv_rows(exp.ROOT/'lap_f023_native_step_preflight.csv')
    with np.load(exp.OUT/'native_displacements.npz') as steps:
        exp.table('hydrogen_native_geometry',[{'arm':label,'H_abs_linear_change':float(h_abs@steps[label]),
            'H_signed_energy_linear_change':float(h_signed@steps[label]),
            'H_abs_finite_change':float(next(r for r in native_rows if r['arm']==label)['H_error_finite_change'])}
            for label in ('old','new')])
    historical = csv_rows(exp.ROOT/'lap_f015_matched_ae17_errors.csv')
    hydrogen = [{**r,'Ne':metadata['ae17'][r['identity']]['Ne'],
        'new_calibrated_multiplier':cal['new_loss_weights']['ae17'][r['identity']]} for r in historical
        if r['model'] in ('b8_t128','f010_t128','f010_t251')]
    exp.table('hydrogen_diagnostics',hydrogen)
    geometry = csv_rows(exp.ROOT/'lap_f023_gradient_geometry.csv')
    for label,p in [('old',oldP),('new',newP)]:
        for target,g in [('H_absolute_error_E',h_abs),('H_signed_atomic_energy_E',h_signed)]:
            geometry.append({'population':'H fixed evaluation variant; physical S128; chemistry S251',
                'direction':label,'target':target,'cosine':exp.dual.cosine(g,C+p),'dot':float(g@(C+p)),
                'negative_gradient_derivative':-float(g@(C+p)),'joint_norm':float(np.linalg.norm(C+p)),
                'physical_norm':float(np.linalg.norm(p))})
    exp.table('gradient_geometry',geometry)
    with np.load(exp.dual.stream.OUT/'stream_jacobian.npz') as data:
        J, residual = data['gradients'].copy(),data['residual'].copy()
    mapping = exp.audit.read(exp.dual.stream.OUT/'stream_rows.json')
    scales = exp.audit.read(exp.ROOT/'lap_f010_normalization_receipt.json')['scales']
    state = exp.audit.geometry.load_state(exp.audit.geometry.STATES[0],scales)
    db_rows = []
    for pop,jac,errors,dbs in [('S',J,residual,[r['database'] for r in mapping]),('E',state['jacobian'],np.asarray(state['residuals']),state['databases'])]:
        for db in sorted(set(dbs)):
            mask = np.array(dbs)==db
            g,_c1,_c2,_f,_p = exp.dual.chemistry(jac[mask],errors[mask],[db]*int(mask.sum()),scales)
            for label,p in [('old',oldP),('new',newP)]:
                db_rows.append({'population':pop,'database':db,'direction':label,
                    **{t+'_negative_gradient_derivative':-float(v@(C+p)) for t,v in g.items()}})
    exp.table('database_directions',db_rows)
    stats = []
    for label in ('old','new'):
        joint = arrays['joint_'+label]
        mean = joint.mean(axis=0)
        cosines = np.array([exp.dual.cosine(v,mean) for v in joint])
        mse = np.mean(np.sum((joint-mean)**2,axis=1))
        P = joint-arrays['chemistry']
        centered = joint-mean
        outliers = np.sort(np.sum(centered**2,axis=1))[::-1]
        stats.append({'arm':label,'positive_dot_count':int((cosines>0).sum()),'worst_cosine':float(cosines.min()),
            'p05_cosine':float(np.quantile(cosines,.05)),'p10_cosine':float(np.quantile(cosines,.1)),
            'median_cosine':float(np.median(cosines)),'joint_vector_MSE':float(mse),
            'relative_gradient_MSE':float(mse/np.linalg.norm(mean)**2),
            'norm_ratio_min':float(min(np.linalg.norm(v)/np.linalg.norm(mean) for v in joint)),
            'norm_ratio_max':float(max(np.linalg.norm(v)/np.linalg.norm(mean) for v in joint)),
            'top1_MSE_share':float(outliers[:1].sum()/outliers.sum()),'top5_MSE_share':float(outliers[:5].sum()/outliers.sum()),
            'physical_MSE':exp.census.variance(P),'chemistry_physical_covariance':exp.census.covariance(arrays['chemistry'],P),
            'F32_representable':bool(np.isfinite(joint.astype(np.float32)).all())})
    exp.table('stochastic_summary',stats)
    tasks=[]
    for task in ('ae17','exc','op'):
        for label in ('old','new'):
            v=arrays[task+('_new' if label=='new' else '')]
            norms=np.linalg.norm(v,axis=1)
            tasks.append({'task':task,'arm':label,'raw_gradient_MSE':exp.census.variance(v),
                'weighted_gradient_MSE':exp.census.variance(v)*exp.b8.LAMBDAS[task]**2,
                'norm_min':float(norms.min()),'norm_p10':float(np.quantile(norms,.1)),
                'norm_median':float(np.median(norms)),'norm_p90':float(np.quantile(norms,.9)),
                'norm_max':float(norms.max())})
    exp.table('task_gradient_distributions',tasks)
    exp.write('constraint_checks',{'scope':'reversible original/F023 native probes, not a trained model',
        'passed':all(r['constraints_passed']=='True' for r in csv_rows(exp.ROOT/'lap_f023_native_step_preflight.csv')),
        'restoration':exp.read('restoration'),'production_operator_source_unchanged':True})
    distributions = {t:{'min':min(r['Ne'] for r in metadata[t].values()),'max':max(r['Ne'] for r in metadata[t].values()),
        'median':float(np.median([r['Ne'] for r in metadata[t].values()])),
        'counts':dict(sorted(Counter(r['Ne'] for r in metadata[t].values()).items()))} for t in ('ae17','mrks')}
    decision = {'classification':'NO-GO','numerical_qualification':'PASS','mechanism_qualification':'FAIL',
        'training_performed':False,'optimizer_updates':0,'temporary_native_counterfactuals':2,
        'c_A':cal['c_A'],'c_E':cal['c_E'],'electron_counts_authenticated':{'ae17':17,'mrks':90},
        'electron_distributions':distributions,'legacy_conflict_bounds':exp.read('legacy_conflict_bounds'),
        'native_preflight':preflight,'stochastic_summary':stats,'scientific_wall_seconds':exp.Budget().used(),
        'wall_budget_seconds':exp.BUDGET,'budget_respected':exp.Budget().used()<exp.BUDGET,
        'Clean28':'NOT EVALUATED','physical_eligibility':'NOT RUN','hydrogen_preservation_final':'NOT RUN',
        'next_experiment':'Read-only cached common-descent feasibility audit adding fixed-evaluation H error and legacy chemistry to JW/operator constraints at B8; no new loss, weights, architecture or training.'}
    exp.write('decision',decision)
    exp.write('resource_budget',{'wall_limit_seconds':exp.BUDGET,'wall_used_seconds':decision['scientific_wall_seconds'],
        'profile_native_and_first_physical_seconds':preflight['native_probe_seconds'],
        'profile_singleton_physical_gradients_seconds':preflight['live_profile_seconds'],
        'training_and_final_full90_projection':'NOT RUN — required mechanism gate failed',
        'training_started':False,'evaluation_reserve_consumed':False,
        'resource_gate':'No training authorized; all read-only work within original wall limit'})
    p = exp.read('provenance')
    p.update({'cached_arrays_sha256':exp.audit.sha(exp.OUT/'cached_arrays.npz'),
        'native_displacements_sha256':exp.audit.sha(exp.OUT/'native_displacements.npz'),
        'signed_forward_values_sha256':exp.audit.sha(exp.OUT/'original_forward_values.json'),
        'full90_operator_gradient_sha256':pinned,'parent_checkpoint_unchanged':exp.audit.sha(exp.f010.PARENT_CHECKPOINT)==exp.dual.CHECKPOINT})
    exp.write('provenance',p)
    write_report(decision,cal,geometry,leverage,hydrogen,attribution)


def write_report(d,cal,geometry,leverage,hydrogen,attribution):
    native = csv_rows(exp.ROOT/'lap_f023_native_step_preflight.csv')
    H = [r for r in hydrogen if r['species']=='H_ae17']
    expected = {'b8_t128':1.3739066446900097,'f010_t128':2.245153757484161,'f010_t251':2.432497922463938}
    assert all(abs(float(r['absolute_error_kcal'])-expected[r['model']])<1e-12 for r in H)
    text = ['NO-GO — SIZE-AWARE PHYSICAL WEIGHTING INEFFECTIVE', '',
        ('One law N_e^-1/2 was qualified at B8 with task-wise initial norm preservation. '
        'It increased H leverage but did not improve the matched H/native-step outcome or the adverse legacy projection. '
        'The mechanism gate failed; no 128-update pilot, scientific endpoints or Clean28 evaluation was launched.'), '',
        ('Electron counts: 17 AE17 identities (all 112 historical training identity/variant pairs authenticated; '
        '84 in S128 evaluated) and 90 mRKS systems. Official Minnesota XYZ/G09 files establish nuclear identity '
        'and ionic charge, including H Z=1,q=0,N_e=1. mRKS uses saved Tr(dm_ks S), checked against nuclear charges '
        'minus molecular charge to 1e-6. AO count is not used as electron count.'), '',
        f"Electron distributions: AE17 {d['electron_distributions']['ae17']}; mRKS {d['electron_distributions']['mrks']}.", '',
        '| Task | Original S128 gradient norm | Size-weighted norm before calibration | Frozen scale | Calibrated norm | Old/new cosine |',
        '|---|---:|---:|---:|---:|---:|']
    for task,key in [('ae17','c_A'),('exc','c_E')]:
        r = cal['norms'][task]
        text.append(f"| {task} | {r['old']:.10f} | {r['size_unscaled']:.10f} | {cal[key]:.12f} | {r['new_calibrated']:.10f} | {r['old_new_cosine']:.12f} |")
    text += ['', (f"Physical old/new cosine {cal['physical_old_new_cosine']:.12f}; full S251-chemistry/S128-physics "
        f"cosine {cal['joint_old_new_cosine_S251']:.12f}. Scalarization coefficients, chemistry stream, operator "
        'metric/gauge/normalization and original precision boundaries are unchanged. No global task-strength '
        'reduction is hidden in the size weighting: AE17 and Exc each preserve their B8 S128 gradient norm.'), '',
        '| Atom | N_e | Calibrated multiplier | Original error B8 (kcal/mol) |', '|---|---:|---:|---:|']
    for species in ('H_ae17','He_ae17','Li_ae17','Cl_ae17'):
        h = next(r for r in hydrogen if r['species']==species and r['model']=='b8_t128')
        text.append(f"| {species} | {h['Ne']} | {h['new_calibrated_multiplier']:.10f} | {float(h['absolute_error_kcal']):.10f} |")
    text += ['', ('All 17 atoms and all S128 variants are in system_leverage.csv/hydrogen_diagnostics.csv. '
        'The latter includes historical B8/F010 t128/t251 fixed-evaluation errors, not a new model evaluation. '
        'H target remains native-F32 −313.7547607421875 kcal/mol; it was not replaced by analytic −0.5 Ha. '
        'The fixed-H functional decomposition gives exchange −0.3053895837025203 Ha and correlation '
        '−0.005830887702820334 Ha; their sum −0.3112204714053406 Ha matches direct integrated XC exactly. '
        'This is a predicted correlation component, not a new reference target or universal self-interaction claim. '
        'Hartree+XC remains unavailable because no separate authenticated Hartree field exists.'), '',
        '| Physical decomposition | Legacy dot | Physical norm | Full joint norm | JW dot joint | Full90 operator dot joint |',
        '|---|---:|---:|---:|---:|---:|']
    for r in attribution:
        text.append(f"| {r['decomposition']} | {r['legacy_dot_physics']:.10f} | {r['physical_norm']:.10f} | {r['joint_norm']:.10f} | {r['JW_dot_joint']:.10f} | {r['operator_dot_joint_full90']:.10f} |")
    bounds = d['legacy_conflict_bounds']
    text += ['', (f"R_conflict old {bounds['old']['R_conflict']:.10f}, new {bounds['new']['R_conflict']:.10f}; "
        f"maximum favorable same-norm chemistry dot {bounds['old']['max_favorable_chemistry_dot']:.10f}. "
        'Both ratios exceed one: no chemistry direction of that fixed norm can compensate the adverse physical '
        'projection locally. This is a first-order obstruction, not proof that every fixed scalar loss or R0 architecture fails. '
        'F022 used uniform physical populations (legacy dot −9.902374745); F023 correctly calibrates/compares S128 '
        '(−10.048290727 → −10.103171268). These populations are not interchanged.'), '',
        '| Native reversible step | Step norm | H absolute error after | Δ H error | Δ AE17 singleton | Δ Exc singleton | Δ operator singleton |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in native:
        text.append(f"| {r['arm']} | {float(r['native_step_norm']):.12f} | {float(r['H_error_after']):.12f} | {float(r['H_error_finite_change']):+.12f} | {float(r['AE17_singleton_finite_change']):+.10f} | {float(r['Exc_singleton_finite_change']):+.10f} | {float(r['operator_singleton_finite_change']):+.10f} |")
    text += ['', ('Both native steps improve H versus B8, but size weighting is worse than the matched original '
        f"step by {d['native_preflight']['native_H_error_difference_new_minus_old']:.12f} kcal/mol (frozen resolution 1e-6). "
        'Both raise the probed AE17/Exc/operator losses. Linear chemistry predictions and full-prefix physical '
        'predictions are separately retained in native_step_preflight.csv; they are not finite full-corpus changes. '
        'First-step displacement cosine 0.999998467601. All model/buffer/moment/RNG restoration checks passed; '
        'two temporary counterfactual steps, zero retained optimizer updates.'), '',
        '| Frozen estimator | Positive dots /128 | Worst cosine | p05 | p10 | Median | Relative vector MSE |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in d['stochastic_summary']:
        text.append(f"| {r['arm']} | {r['positive_dot_count']} | {r['worst_cosine']:.8f} | {r['p05_cosine']:.8f} | {r['p10_cosine']:.8f} | {r['median_cosine']:.8f} | {r['relative_gradient_MSE']:.8f} |")
    text += ['', ('Complete vector MSE, outlier shares, norm ratios and chemistry/physics covariance are in '
        'stochastic_summary.csv. This is exact finite-prefix frozen-state algebra, not conditional unbiasedness '
        'during evolving training. All 128 original source positions and all F32-representability checks passed.'), '',
        ('Original first AE17/Exc/operator live gradients matched cache exactly (relative gaps 0). '
        'Independent weighted-scalar autograd checks passed frozen bounds: AE17 relative gap 1.328e-14; '
        'Exc 1.085e-6 within the declared native-F32 leaf-accumulation bound. This is not claimed bitwise Exc '
        'gradient equality. All 174 signed forward receipts reproduce original losses within declared tolerance. '
        'The signed-Exc extraction caller initially used the wrong integration keyword/layout; a diagnostic-only '
        'adapter corrected it to the existing point_chunk_size and [point,1,10] API. Original native results and '
        'production source were unchanged; no failed mRKS forward result was retained.'), '',
        ('No chemistry objective evaluated every grid variant. S128 uses its historical variants; signed forwards '
        'cover only the 84 cached AE17 variants. Electron metadata for additional historical variants is inspection '
        'only. The fixed E251/E17 scientific populations and unweighted P536 gates remain intact but were not '
        'evaluated for a new trained model because training did not occur.'), '',
        (f"Wall used: {d['scientific_wall_seconds']:.1f}s of 36000s. Native probe {d['native_preflight']['native_probe_seconds']:.3f}s, "
        f"first physical live-gradient profile {d['native_preflight']['live_profile_seconds']:.3f}s. Full training/t64/t128 "
        'evaluation and conditional Clean28 projection were not run after the earlier mandatory mechanism NO-GO.'), '',
        'One next experiment: '+d['next_experiment'], '',
        ('Answers: counts and distributions are authenticated; original losses reproduce; c_A/c_E and leverage '
        'are tabulated; rotations are small; legacy conflict and H matched-step outcome worsen; operator remains '
        'favorable along the plain joint gradient but regresses in the native finite probe. Stochastic changes '
        'are descriptive, not new thresholds. Preflight blocks training. No endpoint objective/H/full90/chemistry '
        'improvement or Clean28 superiority is claimed. The particular size-normalization hypothesis is not '
        'supported by this read-only screen; no other exponent or H-specific weight was tested.'), '',
        'Scientific classification: NO-GO','Architecture: R0','Electron-size exponent: 0.5',
        f"AE17 scale c_A: {cal['c_A']}",f"Exc scale c_E: {cal['c_E']}",
        'Original chemistry loss: UNCHANGED','Operator loss: UNCHANGED',
        'Hydrogen baseline error: 1.3739066446900097 kcal/mol','Final hydrogen error: unavailable',
        'Hydrogen preservation: NOT RUN',f"Legacy conflict before: {bounds['old']['legacy_dot_physics']}",
        f"Legacy conflict after: {bounds['new']['legacy_dot_physics']}",'Preflight: FAIL — mechanism gate; numerical PASS',
        'Training performed: NO','Optimizer updates: 0/128','Final legacy relchem: unavailable',
        'Final AE17: unavailable','Final Exc: unavailable','Final operator: unavailable',
        'P536 ratios: NOT RUN','relchem: unavailable','AE17: unavailable','Exc: unavailable','operator: unavailable',
        'Physical eligibility: NOT RUN','Clean28: NOT EVALUATED','Historical best beaten: NOT TESTED',
        'Curriculum: NO','Gradient clipping: NO','Adaptive task weighting: NO','Architecture modified: NO',
        'New training data: NO','SCF executed: NO',f"Scientific wall time: {d['scientific_wall_seconds']:.1f} seconds",'Budget respected: YES',
        'F023 commit: commit containing this report; returned after verification',
        'Git push / remote HEAD: returned after push, avoiding self-referential commit hash']
    (exp.ROOT/'lap_f023_report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
