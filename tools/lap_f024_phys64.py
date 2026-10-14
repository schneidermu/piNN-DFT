"""One matched physics-only continuation; production evaluators remain unchanged."""
import argparse
import time

import lap_f023_size as old
import numpy as np
import torch
from lap_f021_phys4_eval import constraints
from lap_f023_preflight import atomic_value

audit, b8, f010 = old.audit, old.b8, old.f010
ROOT = old.ROOT
OUT = ROOT.parent / 'lap_f024_phys64_20261014'
PREFIX = 'lap_f024_phys64_'
ANCESTOR = 'f23933534e09b8b8641a6c2085239c5d276bc846'
TASKS = ('ae17', 'exc', 'op')
CONTROL_SHA = 'eeb9a3d5099c8db7b961b95ca11db08c4a037339c1037c3cf6ced900edc5e4dd'
CONTROL = f010.OUT / 'ordinary_sgd_adamw/checkpoint_64.pt'


def read(name):
    return audit.read(ROOT / (PREFIX + name + '.json'))


def write(name, value):
    audit.write(ROOT / (PREFIX + name + '.json'), value)


def table(name, rows):
    audit.table(PREFIX + name + '.csv', rows)


def elapsed():
    return time.time() - read('protocol')['wall_origin_unix']


def check(reserve=0):
    if elapsed() + reserve >= 10800:
        raise TimeoutError('F024 whole scientific wall budget/reserve exhausted')


def flat(mapping):
    return np.concatenate([x.detach().double().cpu().numpy().reshape(-1) for x in mapping.values()])


def combine(raw, parameters):
    return {n: sum((raw[t][n].detach().double() * b8.LAMBDAS[t] for t in TASKS),
                   torch.zeros_like(p, dtype=torch.float64)) for n, p in parameters.items()}


def setup(path=None):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = b8.verify_sources()
    model, shadow, run = audit.existing._load_eval_model(source, path or f010.PARENT_CHECKPOINT)
    bundle = audit.TrainingBundle(ROOT.parent / 'publication_dataset_v1')
    parameters = f010._check_model(model)
    assert tuple(parameters) == tuple(read('protocol')['parameter_names'])
    return model, shadow, run, bundle, parameters


def freeze():
    import subprocess
    subprocess.run(['git', 'merge-base', '--is-ancestor', ANCESTOR, 'HEAD'], cwd=ROOT, check=True)
    source = b8.verify_sources()
    batches = f010._batches()
    parent = torch.load(f010.PARENT_CHECKPOINT, map_location='cpu', weights_only=False)
    assert audit.sha(f010.PARENT_CHECKPOINT) == old.dual.CHECKPOINT
    assert audit.sha(CONTROL) == CONTROL_SHA
    assert f010.continuation.moment_sha(parent['optimizer']) == old.dual.MOMENT
    names = audit.read(audit.geometry.prior.OUT / 'b8_t128/result.json')['parameter_names']
    sources = {p: audit.sha(ROOT / p) for p in source['source_hashes']}
    sources.update({p: audit.sha(ROOT / p) for p in (
        'tools/lap_b8_adamw.py', 'tools/lap_f010_uniform_normalized_chemistry.py',
        'tools/lap_b8_swa_continuation.py', 'train_models/lap_fixed_adamw.py',
        'train_models/lap_moo_training.py', 'train_models/lap_operator.py',
        'tools/lap_f024_phys64.py')})
    physical = [{'position': i, 'ae17': r['ae17'], 'mrks_id': r['mrks_id']}
                for i, r in enumerate(batches[:64])]
    payload = {'experiment_id': 'f024-physics-only-64-v1', 'ancestor': ANCESTOR,
        'checkpoint_sha256': old.dual.CHECKPOINT, 'control_sha256': CONTROL_SHA,
        'moment_sha256': old.dual.MOMENT, 'manifest_sha256': f010.continuation.MANIFEST_SHA,
        'physical_stream': physical, 'physical_stream_sha256': old.previous.key(physical),
        'parameter_names': names, 'coordinates': 9446, 'lambdas': {t: b8.LAMBDAS[t] for t in TASKS},
        'lr': 2.5e-5, 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': .01,
        'foreach': False, 'scheduler': None, 'clipping': None, 'chemistry_presentations': 0,
        'updates': 64, 'cumulative_start': 128, 'cumulative_final': 192,
        'exc_chunk': 4096, 'operator_chunk': 256, 'source_hashes': sources,
        'operator_effect_floor': .02918097603147132, 'H_resolution_kcal': 1e-6,
        'precision_relative_bound': float(16*np.finfo(np.float32).eps), 'F64_sum_atol': 1e-10,
        'evaluation_policy': 'E251/E17 one fixed variant per identity; full90; no development data',
        'scientific_wall_budget_seconds': 10800, 'Clean28': 'FORBIDDEN', 'SCF': 'FORBIDDEN'}
    path = ROOT / (PREFIX + 'protocol.json')
    if path.exists():
        saved = read('protocol')
        assert {k: v for k, v in saved.items() if k != 'wall_origin_unix'} == payload
    else:
        payload['wall_origin_unix'] = time.time()
        write('protocol', payload)
    OUT.mkdir(exist_ok=True)
    return source


def save(path, model, optimizer, cursor, logs, history, gradients):
    temporary = path.with_suffix('.partial.pt')
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'rng': b8._capture_rng(), 'cursor': cursor, 'logs': logs, 'history': history,
        'native_gradients': gradients, 'protocol_sha256': audit.sha(ROOT / (PREFIX+'protocol.json')),
        'source_hashes': read('protocol')['source_hashes'], 'experiment_id': 'f024-physics-only-64-v1'}, temporary)
    temporary.replace(path)


def preflight():
    started = time.perf_counter()
    model, shadow, run, bundle, parameters = setup()
    parent = torch.load(f010.PARENT_CHECKPOINT, map_location='cpu', weights_only=False)
    optimizer = f010.continuation._new_optimizer(parameters)
    f010.continuation.prepare_loaded_optimizer(optimizer, parent['optimizer'], 128, 1e-4)
    before = run.existing.digest(model)
    moments = f010.continuation.moment_sha(optimizer.state_dict())
    rng = b8._capture_rng()
    rows, timings = [], []
    specs = audit.chemistry._selected(bundle)
    for position in (0, 31):
        check(120)
        entry = f010._batches()[position]
        losses, raw, _, seconds = b8.other_tasks(model, shadow, bundle, entry,
            bundle.chemistry_dispersions(), audit.read(bundle.root/'mrks/dispersion.json'))
        for task in TASKS:
            label, record = ('ae17', entry['ae17']) if task == 'ae17' else ('mrks', entry['mrks_id'])
            data, receipt = old.census.cache(label, record)
            reference = data[task]
            gap = float(np.linalg.norm(flat(raw[task])-reference))
            bound = 1e-10 + read('protocol')['precision_relative_bound']*np.linalg.norm(reference)
            assert gap <= bound
            assert abs(losses[task]-receipt['losses'][task]) <= 1e-6*max(1,abs(losses[task]))
            rows.append({'position': position, 'task': task, 'gradient_gap': gap,
                         'bound': float(bound), 'loss': losses[task], 'source_cache_sha256':receipt['sha256']})
            data.close()
        g = combine(raw, parameters)
        independent = sum(b8.LAMBDAS[t]*flat(raw[t]) for t in TASKS)
        assert np.allclose(flat(g), independent, atol=1e-10, rtol=1e-10)
        assert all(torch.isfinite(v).all() and torch.isfinite(v.float()).all() for v in g.values())
        timings.append(seconds)
        del raw, g
    # One frozen evaluation reaction profiles the required endpoint pass, not a training presentation.
    tick = time.perf_counter()
    reaction = b8._load_reaction(bundle, 'relchem', specs[0])
    b8._singleton(model, shadow, reaction, bundle.chemistry_dispersions())
    chemistry_seconds = time.perf_counter()-tick
    del reaction
    b8._restore_rng(rng)
    assert before == run.existing.digest(model)
    assert moments == f010.continuation.moment_sha(optimizer.state_dict()) == old.dual.MOMENT
    constraint = constraints(model, shadow)
    reserve = 251*max(chemistry_seconds, 2)*2 + 90*max(timings)*1.75 + 600
    projection = reserve + 64*max(timings)*1.5
    qualified = elapsed()+projection < 10800
    write('preflight', {'passed': True, 'rows': rows, 'relative_chemistry_training_calls': 0,
        'model_restored': True, 'moments_restored':True, 'RNG_restored':True,
        'counter':128, 'constraints':constraint, 'seconds':time.perf_counter()-started})
    write('resource_budget', {'wall_limit':10800,'wall_used':elapsed(), 'GPU':torch.cuda.get_device_name(),
        'physical_profile_seconds':timings, 'chemistry_profile_seconds':chemistry_seconds,
        'endpoint_reserve_seconds':reserve, 'projected_remaining_seconds':projection,
        'resource_qualified':qualified, 'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30})
    assert qualified, 'Projected complete F024 cannot fit the frozen budget'
    bundle.close()


def train():
    assert read('preflight')['passed'] and read('resource_budget')['resource_qualified']
    model, shadow, _run, bundle, parameters = setup()
    parent = torch.load(f010.PARENT_CHECKPOINT, map_location='cpu', weights_only=False)
    optimizer = f010.continuation._new_optimizer(parameters)
    f010.continuation.prepare_loaded_optimizer(optimizer, parent['optimizer'], 128, 1e-4)
    initial_m = {n: optimizer.state[p]['exp_avg'].detach().double().cpu().numpy().copy() for n,p in parameters.items()}
    initial_v = {n: optimizer.state[p]['exp_avg_sq'].detach().double().cpu().numpy().copy() for n,p in parameters.items()}
    parent_flat = flat(parameters)
    latest = OUT/'resumable.pt'
    cursor, logs, history, gradients = 0, [], [], []
    if latest.exists():
        saved = torch.load(latest,map_location='cpu',weights_only=False)
        assert saved['protocol_sha256']==audit.sha(ROOT/(PREFIX+'protocol.json'))
        assert saved['source_hashes']==read('protocol')['source_hashes']
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        b8._restore_rng(saved['rng'])
        cursor,logs,history,gradients = (saved[k] for k in ('cursor','logs','history','native_gradients'))
    else:
        b8._restore_rng(parent['rng'])
        save(OUT/'checkpoint_0.pt',model,optimizer,0,logs,history,gradients)
        save(latest,model,optimizer,0,logs,history,gradients)
    new_m = {n: np.zeros_like(x) for n,x in initial_m.items()}
    new_v = {n: np.zeros_like(x) for n,x in initial_v.items()}
    # Resume accounting from retained native gradients; never advance the data cursor twice.
    for g in gradients:
        for n in parameters:
            new_m[n] = .9*new_m[n]+.1*g[n]
            new_v[n] = .999*new_v[n]+.001*g[n]**2
    H = next(r for r in audit.read(ROOT/'relchem_joint_epoch_evaluation_manifest.json')['rows']
             if r['task']=='ae17' and bundle.reactions[r['identity']]['components']==['H_ae17'])
    reserve = read('resource_budget')['endpoint_reserve_seconds']
    for t in range(cursor,64):
        check(reserve)
        tick=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
        entry=f010._batches()[t]; before=flat(parameters)
        losses,raw,_,_ = b8.other_tasks(model,shadow,bundle,entry,bundle.chemistry_dispersions(),
                                      audit.read(bundle.root/'mrks/dispersion.json'))
        joint=combine(raw,parameters)
        native={n:g.float().double().cpu().numpy() for n,g in joint.items()}
        assert all(np.isfinite(g).all() for g in native.values())
        optimizer.zero_grad(set_to_none=True)
        for n,p in parameters.items(): p.grad=joint[n].to(p)
        optimizer.step()
        done=t+1; cumulative=128+done
        assert {int(s['step']) for s in optimizer.state.values()}=={cumulative}
        assert all(torch.isfinite(p).all() for p in parameters.values())
        for n in parameters:
            new_m[n]=.9*new_m[n]+.1*native[n]
            new_v[n]=.999*new_v[n]+.001*native[n]**2
        im={n:.9**done*x for n,x in initial_m.items()}; iv={n:.999**done*x for n,x in initial_v.items()}
        actual_m={n:optimizer.state[p]['exp_avg'] for n,p in parameters.items()}
        actual_v={n:optimizer.state[p]['exp_avg_sq'] for n,p in parameters.items()}
        mf=flat(actual_m); vf=flat(actual_v)
        recm=np.concatenate([(im[n]+new_m[n]).ravel() for n in parameters])
        recv=np.concatenate([(iv[n]+new_v[n]).ravel() for n in parameters])
        update_history=[]; update_new=[]
        for n,p in parameters.items():
            denom=np.sqrt(optimizer.state[p]['exp_avg_sq'].double().cpu().numpy()/(1-.999**cumulative))+1e-8
            scale=-2.5e-5/(1-.9**cumulative)
            update_history.append((scale*im[n]/denom).ravel())
            update_new.append((scale*new_m[n]/denom).ravel())
        displacement=flat(parameters)-before
        decay=-2.5e-5*.01*before
        hstep=np.concatenate(update_history); nstep=np.concatenate(update_new)
        history.append({'local_update':done,'counter':cumulative,'beta1_power':.9**done,'beta2_power':.999**done,
            'inherited_m_norm':float(np.linalg.norm(np.concatenate([x.ravel() for x in im.values()]))),
            'new_m_norm':float(np.linalg.norm(np.concatenate([x.ravel() for x in new_m.values()]))),
            'inherited_v_norm':float(np.linalg.norm(np.concatenate([x.ravel() for x in iv.values()]))),
            'new_v_norm':float(np.linalg.norm(np.concatenate([x.ravel() for x in new_v.values()]))),
            'm_recurrence_relative_gap':float(np.linalg.norm(mf-recm)/max(np.linalg.norm(mf),1e-30)),
            'v_recurrence_relative_gap':float(np.linalg.norm(vf-recv)/max(np.linalg.norm(vf),1e-30)),
            'history_update_norm':float(np.linalg.norm(hstep)), 'fresh_update_norm':float(np.linalg.norm(nstep)),
            'decay_displacement_norm':float(np.linalg.norm(decay)),
            'update_decomposition_gap':float(np.linalg.norm(displacement-hstep-nstep-decay))})
        assert history[-1]['m_recurrence_relative_gap']<2e-5
        assert history[-1]['v_recurrence_relative_gap']<2e-5
        raw_norm={t:b8._grad_norm(raw[t]) for t in TASKS}
        gn=b8._grad_norm(joint)
        logs.append({'local_update':done,'counter':cumulative,'ae17_identity':entry['ae17']['identity'],
            'ae17_variant':entry['ae17']['variant'],'mrks_id':entry['mrks_id'], **losses,
            **{t+'_raw_norm':raw_norm[t] for t in TASKS},
            **{t+'_weighted_norm':raw_norm[t]*b8.LAMBDAS[t] for t in TASKS},
            'physical_norm':gn,'cancellation_ratio':gn/sum(raw_norm[t]*b8.LAMBDAS[t] for t in TASKS),
            'step_norm':float(np.linalg.norm(displacement)), 'parent_displacement':float(np.linalg.norm(flat(parameters)-parent_flat)),
            'seconds':time.perf_counter()-tick,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
            'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,'finite':True})
        gradients.append(native)
        del raw,joint
        save(latest,model,optimizer,done,logs,history,gradients)
        if done in (32,64):
            constraint=constraints(model,shadow)
            write('constraints_'+str(done),constraint)
            save(OUT/f'checkpoint_{done}.pt',model,optimizer,done,logs,history,gradients)
            write('H_'+str(done),atomic_value(model,shadow,bundle,{**H,'database':'AE17'}))
        table('training_log',logs); table('moment_history',history)
        print('F024_UPDATE',done,logs[-1]['seconds'],flush=True)
    bundle.close()
    write('training_receipt',{'updates':64,'counter':192,'physical_exposures':64,'relative_chemistry_exposures':0,
        'training_active_seconds':sum(r['seconds'] for r in logs),
        'checkpoints':{str(n):audit.sha(OUT/f'checkpoint_{n}.pt') for n in (0,32,64)}})


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=('freeze','preflight','train'))
    stage=parser.parse_args().stage
    if stage=='freeze': freeze()
    elif stage=='preflight': preflight()
    else: train()


if __name__=='__main__': main()
