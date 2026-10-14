"""Render F024 only from authenticated numerical artifacts; no model selection."""
import csv

import lap_f024_phys64 as exp
import lap_f024_phys64_eval as evaluator
import numpy as np
import torch


def rows(path):
    with path.open(encoding='utf-8') as f:return list(csv.DictReader(f))


def main():
    folder=exp.OUT/'endpoint'
    receipt=exp.audit.read(folder/'receipt.json')
    for filename,sha in receipt['hashes'].items():assert exp.audit.sha(folder/filename)==sha
    chemistry,_jacobian=exp.audit.chemistry._load_reactions(folder/'reactions.npz')
    atoms=torch.load(folder/'atoms.pt',weights_only=False)
    physical=torch.load(folder/'physical.pt',weights_only=False)
    with np.load(folder/'vectors.npz') as data:vectors={k:data[k].copy() for k in data.files}
    scales=exp.audit.read(exp.ROOT/'lap_f010_normalization_receipt.json')['scales']
    gradient_receipts=exp.audit.read(exp.ROOT/'lap_f013_gradient_receipts.json')['states']
    baseline={};controls={}
    for state,label in zip(exp.audit.geometry.STATES[:2],('B8','F010_t64'),strict=True):
        bound=next(r for r in gradient_receipts if r['checkpoint']==state['name'])
        assert exp.audit.sha(state['checkpoint'])==bound['checkpoint_sha256']
        assert exp.audit.sha(state['reactions'])==bound['reactions_sha256']
        if state['physical'].suffix=='.npz':assert exp.audit.sha(state['physical'])==bound['physical_sha256']
        else:
            pins=exp.audit.read(exp.ROOT/'lap_f022_provenance.json')['physical_vector_hashes']
            for task in exp.TASKS:assert exp.audit.sha(state['physical']/(task+'.npy.npz'))==pins[task]
        loaded=exp.audit.geometry.load_state(state,scales)
        cr=loaded['rows']; cj=loaded['jacobian']
        v=evaluator.chemistry_vectors(cr,cj,scales)
        v.update(exp.audit.geometry._physical(state))
        if label=='B8':baseline={'rows':cr,'vectors':v}
        else:controls={'rows':cr,'vectors':v}
    H=next(r for r in atoms if r['species']=='H_ae17')
    cache_receipt=exp.audit.read(exp.ROOT/'lap_f021_phys4_singleton_gradients_receipt.json')
    hcache=next(r for r in cache_receipt['records'] if r['binding']['task']=='ae17'
                and r['binding']['record']['identity']==H['identity'] and r['binding']['record']['variant']==H['variant'])
    data,_receipt=exp.old.census.cache('ae17',hcache['binding']['record'])
    baseline['vectors']['H']=data['ae17'].copy()/exp.audit.chemistry._factor('AE17');data.close()
    controlscalar=exp.audit.read(exp.OUT/'control_scalars.json')
    official0=exp.audit.read(exp.f010.OUT/'metrics_initial.json')
    officialC=exp.audit.read(exp.f010.OUT/'metrics_t64.json')
    control_AE=float(np.mean([exp.audit.chemistry._factor('AE17')*np.sqrt(r['signed_error_kcal']**2+1e-20) for r in controlscalar['atoms']]))
    assert abs(control_AE-officialC['ae17'])<1e-8
    chemistry_errors=np.array([r['residual'] for r in chemistry])
    values={'relchem':float(np.mean([exp.audit.chemistry._factor(r['database'])*np.sqrt(r['residual']**2+1e-20) for r in chemistry])),
        'ae17':float(np.mean([r['loss'] for r in atoms])),
        'exc':float(np.mean([r['exc'] for r in physical.values()])),
        'op':float(np.mean([r['op'] for r in physical.values()]))}
    jw=float(np.mean([np.sqrt(r['residual']**2+1e-20)/scales[r['database']] for r in chemistry]))
    db=[];reactionrows=[]
    for i,r in enumerate(chemistry):
        a,c=baseline['rows'][i],controls['rows'][i]
        assert (r['identity'],r['variant'])==(a['identity'],a['variant'])==(c['identity'],c['variant'])
        reactionrows.append({**r,'B8_residual':a['residual'],'F010_t64_residual':c['residual'],
            'absolute_change_B8':abs(r['residual'])-abs(a['residual']),
            'absolute_change_control':abs(r['residual'])-abs(c['residual'])})
    for database in sorted({r['database'] for r in chemistry}):
        current=[r['residual'] for r in chemistry if r['database']==database]
        b=[r['residual'] for r in baseline['rows'] if r['database']==database]
        c=[r['residual'] for r in controls['rows'] if r['database']==database]
        db.append({'database':database,'count':len(current),**exp.audit.stats(current),
            'B8_MAE':exp.audit.stats(b)['mae'],'control_MAE':exp.audit.stats(c)['mae'],
            'delta_B8_MAE':exp.audit.stats(current)['mae']-exp.audit.stats(b)['mae'],
            'delta_control_MAE':exp.audit.stats(current)['mae']-exp.audit.stats(c)['mae']})
    exp.table('database_metrics',db);exp.table('reaction_metrics',reactionrows)
    oldatoms={r['species']:r for r in rows(exp.ROOT/'lap_f023_hydrogen_diagnostics.csv') if r['model']=='b8_t128'}
    controlatoms={r['species']:r for r in controlscalar['atoms']}
    atomrows=[{k:v for k,v in r.items() if k!='gradient'}|{
        'B8_absolute_error':float(oldatoms[r['species']]['absolute_error_kcal']),
        'control_absolute_error':controlatoms[r['species']]['absolute_error_kcal']} for r in atoms]
    exp.table('atomic_metrics',atomrows)
    original={}
    for r in cache_receipt['records']:
        if r['binding']['task']=='mrks':
            data,_receipt=exp.old.census.cache('mrks',r['binding']['record']);data.close()
            original[r['binding']['record']]=r['losses']
    oprows=[]
    for identity,r in physical.items():
        old=original[identity]['op'];c=controlscalar['physical'][identity]['op']
        oprows.append({'identity':identity,'loss':r['op'],'B8_loss':old,'control_loss':c,
                       'delta_B8':r['op']-old,'delta_control':r['op']-c})
    exp.table('operator_system_metrics',oprows)
    oldforwards=exp.audit.read(exp.old.OUT/'original_forward_values.json')
    exp.table('exc_system_metrics',[{'identity':identity,**{k:v for k,v in r.items() if 'gradient' not in k and k!='op'},
        'B8_signed_error_Ha':oldforwards['mrks_'+exp.old.previous.key(identity)]['signed_error_Ha'],
        'residual_sign_changed':bool(np.sign(r['signed_error_Ha'])!=np.sign(oldforwards['mrks_'+exp.old.previous.key(identity)]['signed_error_Ha']))} for identity,r in physical.items()])
    assert abs(np.mean([r['B8_loss'] for r in oprows])-official0['op'])<1e-8
    geometry=[];normstats={};physical_vectors={};joint_vectors={}
    for label,v in [('B8',baseline['vectors']),('F024_t64',vectors)]:
        pieces={t:exp.b8.LAMBDAS[t]*v[t] for t in exp.TASKS};P=sum(pieces.values())
        G=P+exp.b8.LAMBDAS['relchem']*exp.audit.geometry.C010*v['JW']
        physical_vectors[label]=P; joint_vectors[label]=G
        normstats[label]={'P_norm':float(np.linalg.norm(P)),
            'cancellation_ratio':float(np.linalg.norm(P)/sum(np.linalg.norm(g) for g in pieces.values())),
            **{t+'_component_norm':float(np.linalg.norm(g)) for t,g in pieces.items()}}
        for direction,d in [('physical',P),('F010_reintroduction',G)]:
            for task,g in v.items():
                geometry.append({'checkpoint':label,'population':'E251/E17/full90; fixed H variant',
                    'direction':direction,'target':task,'target_norm':float(np.linalg.norm(g)),
                    'direction_norm':float(np.linalg.norm(d)),'dot':float(g@d),
                    'negative_gradient_derivative':-float(g@d),
                    'cosine':float(g@d/(np.linalg.norm(g)*np.linalg.norm(d)))})
    exp.table('endpoint_geometry',geometry)
    training=exp.read('training_receipt'); logs=rows(exp.ROOT/(exp.PREFIX+'training_log.csv'))
    moments=rows(exp.ROOT/(exp.PREFIX+'moment_history.csv'))
    native_decay=rows(exp.ROOT/(exp.PREFIX+'native_decay_accounting.csv'))
    native_trace=exp.read('native_history_receipt')
    ratios={t:values[t]/exp.b8.BASELINE[t] for t in exp.b8.TASKS}
    effect=values['op']<=exp.read('protocol')['operator_effect_floor']
    Hpreserved=H['absolute_error_kcal']<=1.3739066446900097+1e-6
    eligible=all(x<1 for x in ratios.values())
    classification=('A' if Hpreserved and eligible else 'B') if effect else 'C'
    opsummary={'mean':values['op'],'median':float(np.median([r['loss'] for r in oprows])),
        'B8_median':float(np.median([r['B8_loss'] for r in oprows])),
        'control_median':float(np.median([r['control_loss'] for r in oprows])),
        'delta_B8':values['op']-official0['op'],'delta_control':values['op']-officialC['op'],
        'relative_control_change':values['op']/officialC['op']-1,
        'improved_B8':sum(r['delta_B8']<0 for r in oprows),'improved_control':sum(r['delta_control']<0 for r in oprows),
        'regressed_B8':sum(r['delta_B8']>0 for r in oprows),'regressed_control':sum(r['delta_control']>0 for r in oprows),
        'largest_improvements_control':sorted(oprows,key=lambda r:r['delta_control'])[:5],
        'largest_regressions_control':sorted(oprows,key=lambda r:r['delta_control'],reverse=True)[:5],
        'top5_gross_improvement_share':float(sum(max(0,-r['delta_control']) for r in sorted(oprows,key=lambda r:r['delta_control'])[:5])/
                                          max(1e-30,sum(max(0,-r['delta_control']) for r in oprows)))}
    rotations={t:float(baseline['vectors'][t]@vectors[t]/(np.linalg.norm(baseline['vectors'][t])*np.linalg.norm(vectors[t]))) for t in vectors}
    for name,states in [('physical',physical_vectors),('F010_joint',joint_vectors)]:
        rotations[name]=float(states['B8']@states['F024_t64']/(np.linalg.norm(states['B8'])*np.linalg.norm(states['F024_t64'])))
    # One affordable clone-only full-population proposal; no optimizer or model update.
    checkpoint=torch.load(exp.OUT/'checkpoint_64.pt',map_location='cpu',weights_only=False)
    names=exp.read('protocol')['parameter_names'];ids=checkpoint['optimizer']['param_groups'][0]['params']
    offset=0;delta=[]
    for n,i in zip(names,ids,strict=True):
        p=checkpoint['model'][n].cuda().clone();before=p.double().clone();count=p.numel()
        g=torch.as_tensor(joint_vectors['F024_t64'][offset:offset+count],device='cuda',dtype=torch.float32).reshape_as(p)
        m=checkpoint['optimizer']['state'][i]['exp_avg'].cuda().clone()
        v=checkpoint['optimizer']['state'][i]['exp_avg_sq'].cuda().clone()
        p.mul_(1-2.5e-5*.01);m.lerp_(g,1-.9);v.mul_(.999).addcmul_(g,g,value=1-.999)
        denominator=(v.sqrt()/((1-.999**193)**.5)).add_(1e-8)
        p.addcdiv_(m,denominator,value=-2.5e-5/(1-.9**193))
        delta.append((p.double()-before).cpu().numpy().ravel());offset+=count
    delta=np.concatenate(delta)
    proposal={t:float(g@delta) for t,g in vectors.items()}
    np.savez_compressed(exp.OUT/'handoff_proposal.npz',displacement=delta)
    exp.table('handoff_proposal',[{'target':t,'first_order_change':change,'displacement_norm':float(np.linalg.norm(delta)),
        'population':'E251/E17/full90; full-population counterfactual, not a minibatch',
        'retained_update':False} for t,change in proposal.items()])
    decision={'classification':classification,'updates':64,'counter':192,'values':values,'JW':jw,'ratios':ratios,
        'old_new_gradient_cosines':rotations,
        'full_population_handoff_proposal':proposal,'handoff_proposal_norm':float(np.linalg.norm(delta)),
        'unique_AE17_training_identities':len({r['ae17_identity'] for r in logs}),
        'unique_mrks_training_systems':len({r['mrks_id'] for r in logs}),
        'H_signed_error':H['signed_error_kcal'],'H_absolute_error':H['absolute_error_kcal'],'H_preserved':Hpreserved,
        'P536_eligible':eligible,'operator_floor_passed':effect,'operator_summary':opsummary,
        'gradient_norms':normstats,'chemistry_stats':exp.audit.stats(chemistry_errors),'macro_database_MAE':float(np.mean([r['mae'] for r in db])),
        'scientific_wall_seconds':exp.elapsed(),'training_active_seconds':training['training_active_seconds'],
        'endpoint_active_seconds':receipt['active_seconds'],'control_active_seconds':exp.read('control_receipt')['active_seconds'],
        'Exc_sign_changes':int(sum(np.sign(r['signed_error_Ha'])!=np.sign(oldforwards['mrks_'+exp.old.previous.key(i)]['signed_error_Ha']) for i,r in physical.items())),
        'peak_allocated_GiB':max(float(r['peak_allocated_GiB']) for r in logs),
        'peak_reserved_GiB':max(float(r['peak_reserved_GiB']) for r in logs),
        'moments_t32':moments[31],'moments_t64':moments[63],
        'actual_native_decay_t32':native_decay[31],'actual_native_decay_t64':native_decay[63],
        'recommendation':'Read-only matched operator/JW/legacy native-direction attribution from existing B8, F010 t64 and F024 t64 gradients and optimizer states. Do not launch a chemistry stage: the operator effect floor failed and the unchanged-F010 handoff still predicts legacy/operator regressions.',
        'native_trace':native_trace, 'Clean28':'NOT EVALUATED'}
    exp.write('decision',decision)
    metrics=[]
    for label,r in [('B8',official0),('F010_t64',officialC)]:
        metrics.append({'checkpoint':label,'JW':r['jw'],**{t:r[t] for t in exp.b8.TASKS},'MAE':r['mae']})
    metrics.append({'checkpoint':'F024_t64','JW':jw,**values,'MAE':decision['chemistry_stats']['mae']})
    exp.table('checkpoint_metrics',metrics)
    provenance={'protocol_sha256':exp.audit.sha(exp.ROOT/(exp.PREFIX+'protocol.json')),
        'checkpoint_receipts':training['checkpoints'],'control_sha256':exp.CONTROL_SHA,
        'endpoint_receipt':receipt,'preflight':exp.read('preflight'),'control_receipt':exp.read('control_receipt'),
        'source_hashes':{str(p.relative_to(exp.ROOT)).replace('\\','/'):exp.audit.sha(p) for p in exp.ROOT.glob('tools/lap_f024_phys64*.py')},
        'historical_gradient_receipts_sha256':exp.audit.sha(exp.ROOT/'lap_f013_gradient_receipts.json'),
        'evaluation_manifest_sha256':exp.audit.sha(exp.ROOT/'relchem_joint_epoch_evaluation_manifest.json'),
        'dataset_manifest_sha256':exp.audit.sha(exp.ROOT.parent/'publication_dataset_v1/dataset_manifest.json'),
        'dataset_logical_sha256':exp.audit.read(exp.ROOT.parent/'publication_dataset_v1/dataset_manifest.json')['logical_sha256'],
        'native_trace':native_trace,
        'handoff_proposal_sha256':exp.audit.sha(exp.OUT/'handoff_proposal.npz'),
        'parent_unchanged':exp.audit.sha(exp.f010.PARENT_CHECKPOINT)==exp.old.dual.CHECKPOINT,
        'architecture_modified':False,'Clean28_accessed':False,'SCF_executed':False,'second_stage':False}
    exp.write('provenance',provenance)
    render(decision,metrics,db,geometry)


def render(d,metrics,db,geometry):
    text=[f"COMPLETED — F024 physics-only pilot, classification {d['classification']}",'',
        ('One matched physical continuation from jointly trained B8, with inherited full-history AdamW moments. '
        'The only intervention was absence of the fresh relative-chemistry term. No model promotion, chemistry stage or Clean28 evaluation.'), '',
        'Both arms have64 AE17 and64 paired mRKS optimization presentations, counter128→192. F010 t64 additionally has512 fresh relative-reaction presentations; F024 has0. The comparison is matched by physical exposure and optimizer updates, not total compute or chemistry exposure.', '',
        '| Checkpoint | JW E251 | Legacy E251 | AE17 E17 | Exc full90 | Operator full90 | MAE |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in metrics:text.append(f"| {r['checkpoint']} | {r['JW']:.12f} | {r['relchem']:.12f} | {r['ae17']:.12f} | {r['exc']:.12f} | {r['op']:.12f} | {r['MAE']:.9f} |")
    text+=['',f"Operator effect floor: {d['operator_floor_passed']}; threshold 0.02918097603147132. System statistics: {d['operator_summary']}.",
        'The operator metric is the equal-system full90 mean of ||S^-1/2(V_pred−V_ref)S^-1/2||_F²/nAO (Ha²), with the original physically fixed gauge and V=dE_xc/dP_total. No square root, extra electron weighting or changed overlap convention is used. Reaction and atomic errors are kcal/mol; historical task factors and the original Exc kcal/mol conversion remain unchanged.',
        '',f"H signed/absolute error: {d['H_signed_error']:.12f}/{d['H_absolute_error']:.12f} kcal/mol. Preservation: {d['H_preserved']}. Stored reference and fixed evaluation variant unchanged.",
        f"P536 ratios: {d['ratios']}; eligibility {d['P536_eligible']}. Failures are handoff recovery debt, not broken architectural constraints.",'',
        '| Database | MAE | RMSE | Δ MAE vs B8 | Δ MAE vs F010 t64 |','|---|---:|---:|---:|---:|']
    for r in db:text.append(f"| {r['database']} | {r['mae']:.9f} | {r['rmse']:.9f} | {r['delta_B8_MAE']:+.9f} | {r['delta_control_MAE']:+.9f} |")
    text+=['',f"Population chemistry statistics: {d['chemistry_stats']}; macro DB MAE {d['macro_database_MAE']:.9f}. All251 reactions and17 atoms use exactly one frozen variant. Per-record changes are saved.",
        '',f"Physical gradient norms/cancellation on E17/full90: {d['gradient_norms']}. Individual raw norms, projections and the unchanged-F010 reintroduction geometry are in endpoint_geometry.csv. Lower sum norm can reflect cancellation, not convergence.",
        f"Matched-population old/new gradient cosines: {d['old_new_gradient_cosines']}. Training exposed {d['unique_AE17_training_identities']} atomic identities and {d['unique_mrks_training_systems']} unique mRKS systems; full90 outcomes come from the complete endpoint evaluation, not sampled coverage.",
        '', '| State/direction | Legacy negative-gradient derivative | H negative-gradient derivative | JW derivative | Operator derivative |', '|---|---:|---:|---:|---:|']
    for checkpoint in ('B8','F024_t64'):
        for direction in ('physical','F010_reintroduction'):
            vals={r['target']:r['negative_gradient_derivative'] for r in geometry if r['checkpoint']==checkpoint and r['direction']==direction}
            text.append(f"| {checkpoint}/{direction} | {vals['legacy']:+.10f} | {vals['H']:+.10f} | {vals['JW']:+.10f} | {vals['op']:+.10f} |")
    text+=['',f"Inherited optimizer accounting at t32: {d['moments_t32']}",f"At t64: {d['moments_t64']}",
        f"Actual native-F32 decay/step accounting: t32 {d['actual_native_decay_t32']}; t64 {d['actual_native_decay_t64']}. A clone-only arithmetic trace from saved native inputs reproduced weights and both moments bitwise at t32/t64. No optimizer, objective, model or fresh gradient was evaluated by that trace. The decay norm in moment_history.csv is the ideal F64 expression; native_decay_accounting.csv records the actual rounded multiplicative displacement.",
        'Beta powers are attenuation factors, not causal percentages. First-moment update terms share the actual bias correction/second-moment denominator; no additive causal second-moment decomposition is claimed. Native recurrence residuals and actual decay/update rounding gaps are recorded.',
        '',f"Training active {d['training_active_seconds']:.2f}s; endpoint active {d['endpoint_active_seconds']:.2f}s; missing control scalars active {d['control_active_seconds']:.2f}s; scientific wall {d['scientific_wall_seconds']:.2f}s /10800s. Reporting/idle time is not GPU compute. Peak allocated/reserved {d['peak_allocated_GiB']:.3f}/{d['peak_reserved_GiB']:.3f}GiB.",
        '', f"Integrated Exc residual sign changes: {d['Exc_sign_changes']}/90. These absolute-value objectives can retain gradient norms after scalar losses decrease. S64-prefix endpoint gradients were omitted; primary geometry uses exactly E251/E17/full90, without mixing prefix gradients.",
        f"One clone-only native-F32 full-population AdamW proposal from saved t64 moments: displacement norm {d['handoff_proposal_norm']:.12f}, linear changes {d['full_population_handoff_proposal']}. This uses unchanged F010 coefficients and hypothetical counter193; no optimizer/model update is retained. It is not the next historical minibatch and does not prove finite-step safety.",
        'The endpoint physical direction is locally more compatible with legacy/H than B8. However, reintroducing unchanged JW reverses the legacy-descent margin, and the saved-moment native proposal predicts both legacy and operator increases. This is useful geometry evidence, not a qualified handoff or a proof that physics-first training generally fails.',
        'Interpretation is limited to this parent, inherited optimizer history, frozen stream and64-update horizon. No capacity-ceiling claim or automatic second stage is justified.',
        'One next recommendation: '+d['recommendation'],
        '', 'Architecture: R0','Training objective: fixed AE17 + Exc + operator','Fresh relative-chemistry gradient: ABSENT',
        'Inherited optimizer moments: PRESERVED','Optimizer updates: 64/64','Relative-chemistry optimization presentations: 0',
        'AE17 optimization presentations: 64','mRKS optimization presentations: 64','Primary endpoint: t64',
        f"Operator effect floor passed: {'YES' if d['operator_floor_passed'] else 'NO'}",f"H preservation: {'PASS' if d['H_preserved'] else 'FAIL'}",
        f"Original P536 eligibility: {'PASS' if d['P536_eligible'] else 'FAIL'}",'Second training stage executed: NO',
        'New chemistry coefficients selected: NO','Clean28 evaluated: NO','SCF executed: NO','New training data: NO','Architecture modified: NO',
        f"Scientific classification: {d['classification']}",'Commit / push / verified remote HEAD: returned after commit verification']
    (exp.ROOT/(exp.PREFIX+'report.md')).write_text('\n'.join(text)+'\n',encoding='utf-8')


if __name__=='__main__':main()
