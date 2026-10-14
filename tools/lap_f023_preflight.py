"""F023 qualified forward receipts and reversible native first-step test."""
import copy
import json
import time

import lap_f023_size as exp
import numpy as np
import torch
from lap_f021_phys4_eval import constraints


def setup():
    exp.freeze()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    protocol_sha = exp.audit.sha(exp.ROOT/'lap_f023_protocol.json')
    cal_sha = exp.audit.sha(exp.ROOT/'lap_f023_calibration.json')
    execution = {'protocol_sha256': protocol_sha, 'calibration_sha256': cal_sha,
        'driver_sha256': exp.audit.sha(__file__)}
    path = exp.ROOT/'lap_f023_execution_contract.json'
    if path.exists():
        assert exp.read('execution_contract') == execution
    else:
        exp.write('execution_contract', execution)
    source = exp.b8.verify_sources()
    model, shadow, run = exp.audit.existing._load_eval_model(source, exp.f010.PARENT_CHECKPOINT)
    shadow.load_state_dict(model.state_dict(), strict=True)
    bundle = exp.audit.TrainingBundle(exp.ROOT.parent/'publication_dataset_v1')
    return model, shadow, run, bundle


def atomic_value(model, shadow, bundle, spec):
    shadow.load_state_dict(model.state_dict(), strict=True)
    reaction = exp.b8._load_reaction(bundle, 'ae17', spec)
    with torch.no_grad():
        predicted, target = exp.audit.chemistry._prediction(shadow, reaction, bundle.chemistry_dispersions())
    value = {'prediction_kcal': float(predicted), 'reference_kcal': float(target),
        'signed_error_kcal': float(predicted-target), 'absolute_error_kcal': abs(float(predicted-target))}
    del reaction
    return value


def originals(model, shadow, run, bundle, budget):
    """Recover signed residuals absent from the old gradient cache; no backward."""
    from train_models.lap_vxc import LapEnergy, integrated_energy
    from train_models.optuna_joint import _add_mrks_dispersion_once, batch_exc
    atoms = exp.audit.read(exp.OUT/'authenticated_electrons.json')
    records = exp.audit.read(exp.ROOT/'lap_f021_phys4_singleton_gradients_receipt.json')['records']
    path = exp.OUT/'original_forward_values.json'
    values = exp.audit.read(path) if path.exists() else {}
    md = exp.audit.read(bundle.root/'mrks/dispersion.json')
    features = bundle.mrks()
    for row in records:
        budget.check(600)
        task, spec = row['binding']['task'], row['binding']['record']
        key = task+'_'+exp.previous.key(spec)
        if key in values:
            continue
        if task == 'ae17':
            value = atomic_value(model, shadow, bundle, spec)
            value['original_loss'] = exp.audit.chemistry._factor('AE17')*np.sqrt(value['signed_error_kcal']**2+1e-20)
            expected = row['losses']['ae17']
        else:
            material = features[features.positions[spec]]
            raw = material['DensityDescriptorsN10'].cuda().float()
            weights = material['weights'].cuda().float()
            with torch.no_grad():
                prediction = integrated_energy(LapEnergy(model), raw, weights, chunk_size=4096)
                prediction = _add_mrks_dispersion_once(prediction, material['name'], md, True)
                target = material['Exc'].to(prediction).reshape(1)
                loss = batch_exc([material['name']], prediction.reshape(1), target)
            value = {'prediction_Ha': float(prediction), 'reference_production_Ha': float(target[0]),
                'reference_storage_Ha': float(material['Exc']), 'signed_error_Ha': float(prediction-target[0]),
                'original_loss': float(loss)}
            expected = row['losses']['exc']
            del material, raw, weights, prediction, target, loss
        assert abs(value['original_loss']-expected) <= 1e-6*max(1, abs(expected))
        value['source_cache_sha256'] = row['sha256']
        values[key] = value
        exp.audit.write(path, values)
        torch.cuda.empty_cache()
        print('F023_FORWARDS', len(values), len(records), flush=True)
    rows = []
    for row in records:
        task, spec = row['binding']['task'], row['binding']['record']
        value = values[task+'_'+exp.previous.key(spec)]
        identity = spec['identity'] if task=='ae17' else spec
        info = atoms['ae17' if task=='ae17' else 'mrks'][identity]
        rows.append({'task': 'ae17' if task=='ae17' else 'exc', 'identity': identity,
            'variant': spec['variant'] if task=='ae17' else 'central', 'source_id': info['source_id'],
            'Ne': info['Ne'], 'n_AO': info.get('n_ao', 'not applicable'),
            'source_sha256': info['source_sha256'], 'reference_provenance': info['reference_provenance'],
            'original_scalar_residual': value['signed_error_kcal'] if task=='ae17' else value['signed_error_Ha'],
            'residual_units': 'kcal/mol' if task=='ae17' else 'hartree',
            'original_loss_contribution': value['original_loss'], 'size_weight': info['Ne']**-.5})
    exp.table('electron_counts', rows)
    exp.table('original_physical_errors', rows)
    # Electron metadata for all 112 historical AE variants, without evaluating outside S128.
    all_variants = {exp.previous.key(r['ae17']): r['ae17'] for r in exp.f010._batches()}
    exp.table('atomic_variant_identity_census', [{'identity': s['identity'], 'variant': s['variant'],
        **{k: atoms['ae17'][s['identity']][k] for k in ('source_id','Z','ionic_charge','Ne','g09_sha256','xyz_sha256')},
        'in_S128': ('ae17_'+exp.previous.key(s)) in values} for s in all_variants.values()])
    return values


def native():
    started = time.perf_counter()
    budget = exp.Budget()
    model, shadow, run, bundle = setup()
    before = run.existing.digest(model)
    rng = exp.b8._capture_rng()
    original = copy.deepcopy(model.state_dict())
    parent = torch.load(exp.f010.PARENT_CHECKPOINT, map_location='cpu', weights_only=False)
    parameters = exp.f010._check_model(model)
    optimizer = exp.audit.geometry.arm._new_optimizer(parameters)
    exp.f010.continuation.prepare_loaded_optimizer(optimizer, parent['optimizer'], 128, 1e-4)
    moments = copy.deepcopy(optimizer.state_dict())
    moment_before = exp.f010.continuation.moment_sha(moments)
    assert moment_before == exp.dual.MOMENT
    exp.b8._restore_rng(parent['rng'])
    baseline_rng = exp.b8._capture_rng()
    protocol = exp.read('protocol')
    prefix = exp.f010._batches()[:128]
    census = np.load(exp.OUT/'cached_arrays.npz')
    assert exp.audit.sha(exp.OUT/'cached_arrays.npz') == exp.read('provenance')['cached_arrays_sha256']
    names = tuple(parameters)
    entry = prefix[0]
    dispersion = bundle.chemistry_dispersions()
    md = exp.audit.read(bundle.root/'mrks/dispersion.json')
    H = next(r for r in exp.audit.read(exp.ROOT/'relchem_joint_epoch_evaluation_manifest.json')['rows']
             if r['task']=='ae17' and bundle.reactions[r['identity']]['components']==['H_ae17'])
    H = {**H, 'database':'AE17'}
    system = bundle.mrks().operator_system(entry['mrks_id'],device='cuda',dtype=torch.float32,chunk_size=4096)
    factories = run.existing.core.make_mrks_objective_factories(model, system, point_chunk_size=256,
        exc_chunk_size=4096, dispersions=md)
    records = []
    base_H = atomic_value(model, shadow, bundle, H)
    assert abs(base_H['absolute_error_kcal']-protocol['hydrogen_baseline_kcal']) < 1e-6
    def losses():
        h = atomic_value(model, shadow, bundle, H)
        ae = atomic_value(model, shadow, bundle, entry['ae17'])
        with torch.no_grad():
            e, o = [float(f()) for f in factories]
        return {'H_error': h['absolute_error_kcal'], 'H_signed_error':h['signed_error_kcal'],
            'AE17_singleton':exp.audit.chemistry._factor('AE17')*np.sqrt(ae['signed_error_kcal']**2+1e-20),
            'Exc_singleton':e,'operator_singleton':o}
    try:
        # Live parity for the unchanged original physical entries.
        values, live, _sys, live_seconds = exp.b8.other_tasks(model,shadow,bundle,entry,dispersion,md)
        gaps = {}
        for task in ('ae17','exc','op'):
            actual = np.concatenate([live[task][n].detach().double().cpu().numpy().ravel() for n in names])
            reference = census[task][0]
            gaps[task] = float(np.linalg.norm(actual-reference)/max(np.linalg.norm(reference),1e-30))
            assert gaps[task] <= 1e-10
        del live
        baseline = losses()
        assert abs(baseline['AE17_singleton']-values['ae17']) < 1e-10
        assert abs(baseline['Exc_singleton']-values['exc']) < 1e-10
        assert abs(baseline['operator_singleton']-values['op']) < 1e-10
        delta_vectors = {}
        for label in ('old','new'):
            budget.check(600)
            model.load_state_dict(original)
            optimizer.load_state_dict(copy.deepcopy(moments))
            exp.b8._restore_rng(baseline_rng)
            flat = {'relchem':census['chemistry'][0]/exp.b8.LAMBDAS['relchem'],
                **{t:census[t if label=='old' else t+'_new'][0] for t in ('ae17','exc','op')}}
            raw = {}
            for task, vector in flat.items():
                offset = 0
                raw[task] = {}
                for name,p in parameters.items():
                    raw[task][name] = torch.from_numpy(np.array(vector[offset:offset+p.numel()],copy=True)).to(p.device).reshape_as(p)
                    offset += p.numel()
                assert offset == 9446
            before_step = {n:p.detach().clone() for n,p in parameters.items()}
            joint = run.adamw_step(model,optimizer,raw,exp.b8.LAMBDAS)
            assert {int(s['step']) for s in optimizer.state.values()} == {129}
            delta = np.concatenate([(parameters[n].detach().double()-before_step[n].double()).cpu().numpy().ravel() for n in names])
            assert np.isfinite(delta).all()
            delta_vectors[label] = delta
            finite = losses()
            checked = constraints(model,shadow)
            records.append({'arm':label,'native_step_norm':float(np.linalg.norm(delta)),
                'JW_S_linear_change':float(census['JW_S']@delta),
                'legacy_S_linear_change':float(census['legacy_S']@delta),
                **{t+'_prefix_linear_change':float(census[t].mean(axis=0)@delta) for t in ('ae17','exc','op')},
                **{k+'_before':v for k,v in baseline.items()}, **{k+'_after':v for k,v in finite.items()},
                **{k+'_finite_change':finite[k]-v for k,v in baseline.items()}, 'constraints_passed':checked['passed']})
            del raw,joint,before_step
            torch.cuda.empty_cache()
        exp.table('native_step_preflight', records)
        bound = exp.read('legacy_conflict_bounds')
        projection_improved = bound['new']['legacy_dot_physics']-bound['old']['legacy_dot_physics'] > max(
            protocol['mechanism_dot_absolute_resolution'],protocol['mechanism_dot_relative_resolution']*abs(bound['old']['legacy_dot_physics']))
        hydrogen_improved = records[0]['H_error_after']-records[1]['H_error_after'] > protocol['mechanism_H_comparison_resolution_kcal']
        mechanism = bool(projection_improved or hydrogen_improved)
        exp.write('preflight_decision', {'numerical_pass':True,'mechanism_pass':mechanism,
            'projection_improved':projection_improved,'native_H_improved':hydrogen_improved,
            'live_gradient_relative_gaps':gaps,'live_profile_seconds':live_seconds,
            'native_probe_seconds':time.perf_counter()-started,
            'native_displacement_cosine':exp.dual.cosine(delta_vectors['old'],delta_vectors['new']),
            'native_H_error_difference_new_minus_old':records[1]['H_error_after']-records[0]['H_error_after'],
            'training_authorized':False, 'resource_qualification':'PENDING' if mechanism else 'NOT RUN — mechanism gate failed',
            'optimizer_updates_retained':0})
        np.savez_compressed(exp.OUT/'native_displacements.npz',**delta_vectors)
    finally:
        model.load_state_dict(original)
        optimizer.load_state_dict(moments)
        exp.b8._restore_rng(rng)
        assert run.existing.digest(model)==before
        assert exp.f010.continuation.moment_sha(optimizer.state_dict())==moment_before
        assert run.existing.equal(rng,exp.b8._capture_rng())
        restored = {'model':True,'buffers':True,'optimizer_moments':True,'rng':True,'retained_updates':0,
            'parent_checkpoint_unchanged':exp.audit.sha(exp.f010.PARENT_CHECKPOINT)==exp.dual.CHECKPOINT}
        exp.write('restoration',restored)
        census.close()
        bundle.close()
        budget.check()
    print(json.dumps(exp.read('preflight_decision'),indent=2),flush=True)


if __name__=='__main__':
    native()
