"""One prefix-matched PHYS4 continuation; reuse the qualified F010 physics."""
import os

os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import hashlib
import json
import subprocess
import time
from collections import Counter

import lap_f015_audit as audit
import numpy as np
import torch

ROOT = audit.ROOT
OUT = ROOT.parent / 'lap_f021_phys4_20261013'
b8 = audit.geometry.b8
f010 = audit.geometry.f010
OFFSETS = (0, 31, 62, 93)
TOTAL = 128
BUDGET = 36000
PARENT = 'b2d23f66984b7434c926cd40686df25559b7134c'
MOMENT = '4659e23cce1cf2c66f701b0b445b897d58838739ca15cffc9300fa0fdc1a40d0'
TASKS = ('ae17', 'exc', 'op')


def write(name, payload):
    audit.write(ROOT / ('lap_f021_phys4_' + name + '.json'), payload)


def read(name):
    return audit.read(ROOT / ('lap_f021_phys4_' + name + '.json'))


def table(name, rows):
    audit.table('lap_f021_phys4_' + name + '.csv', rows)


def positions(t):
    return tuple((t + offset) % TOTAL for offset in OFFSETS)


def key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class Budget:
    def __init__(self):
        OUT.mkdir(exist_ok=True)
        self.start = time.perf_counter()
        self.path = OUT / 'budget_usage.json'
        self.prior = audit.read(self.path)['seconds'] if self.path.exists() else 0.0

    def used(self):
        return self.prior + time.perf_counter() - self.start

    def check(self, reserve=0):
        used = self.used()
        audit.write(self.path, {'seconds': used, 'limit': BUDGET})
        if used + reserve >= BUDGET:
            raise TimeoutError('F021-PHYS4 budget exhausted with mandatory evaluation reserve')


def freeze():
    subprocess.run(['git', 'merge-base', '--is-ancestor', PARENT, 'HEAD'], cwd=ROOT, check=True)
    batches = f010._batches()
    receipt = audit.read(ROOT / 'lap_f010_normalization_receipt.json')
    assert receipt['c'] == audit.geometry.C010
    assert audit.geometry.arm.scales_sha256(receipt['scales']) == audit.read(ROOT / 'lap_f013_gradient_receipts.json')['a_d_sha256']
    assert audit.sha(f010.PARENT_CHECKPOINT) == audit.geometry.STATES[0]['sha256']
    saved = torch.load(f010.PARENT_CHECKPOINT, map_location='cpu', weights_only=False)
    assert f010.continuation.moment_sha(saved['optimizer']) == MOMENT
    prefix = batches[:TOTAL]
    rows = []
    for t in range(TOTAL):
        for slot, p in enumerate(positions(t)):
            rows.append({'update': t, 'slot': slot, 'source_position': p,
                         'ae17_identity': prefix[p]['ae17']['identity'],
                         'ae17_variant': prefix[p]['ae17']['variant'], 'mrks_id': prefix[p]['mrks_id']})
    assert Counter(r['source_position'] for r in rows) == dict.fromkeys(range(TOTAL), 4)
    source = b8.verify_sources()
    pins = {relative: audit.sha(ROOT / relative) for relative in source['source_hashes']}
    pins.update({str(p.relative_to(ROOT)): audit.sha(p) for p in
                 (ROOT / 'tools/lap_f021_phys4.py', ROOT / 'tools/lap_f021_phys4_run.py', ROOT / 'tools/lap_f021_phys4_eval.py')})
    systems = {r['id']: r for r in map(json.loads, (ROOT.parent / 'publication_dataset_v1/mrks/systems.jsonl').read_text().splitlines())}
    first_position = {}
    for p, row in enumerate(prefix):
        first_position.setdefault(row['mrks_id'], p)
    small = sorted(first_position, key=lambda identity: (systems[identity]['n_grid'] * systems[identity]['n_ao'], identity))[:2]
    reference_positions = [first_position[i] for i in small] * 2
    payload = {'experiment': 'F021-PHYS4', 'parent': PARENT, 'budget_seconds': BUDGET,
               'checkpoint_sha256': audit.sha(f010.PARENT_CHECKPOINT), 'moments_sha256': MOMENT,
               'manifest_sha256': f010.continuation.MANIFEST_SHA, 'prefix_sha256': key(prefix),
               'prefix_mapping_sha256': key(rows), 'physical_modulo': TOTAL, 'offsets': list(OFFSETS),
               'updates': TOTAL, 'c010': audit.geometry.C010, 'lambdas': b8.LAMBDAS,
               'lr': 2.5e-5, 'betas': [0.9, 0.999], 'eps': 1e-8, 'weight_decay': .01,
               'foreach': False, 'scheduler': None, 'clipping': None,
               'source_hashes': pins, 'dataset_sha256': b8.DATA_SHA,
               'evaluation': 'one-fixed-variant-per-identity; 251/17/90',
               'physical_gradient_storage': 'F64; unchanged native singleton derivatives widened',
               'F64_reduction_relative_tolerance': 1e-10,
               'native_scalar_autograd_bound': '1e-10 + 16*eps32*mean(singleton_gradient_norm)',
               'diagnostic_positions': [0, 31], 'scalar_reference_positions': reference_positions,
               'checkpoints': [0, 64, 128],
               'baseline': b8.BASELINE, 'clean28': 'once at t128 only if all four strict ratios <1'}
    path = ROOT / 'lap_f021_phys4_protocol.json'
    if path.exists():
        assert read('protocol') == payload, 'Frozen F021-PHYS4 protocol changed'
    else:
        OUT.mkdir(exist_ok=True)
        write('protocol', payload)
        table('prefix_manifest', rows)
        write('exposure_parity', {'passed': True, 'source_position_counts': Counter(r['source_position'] for r in rows),
                                 'chemistry_presentations': 1024, 'physical_presentations': 512,
                                 'unique_ae17_records': len({key(r['ae17']) for r in prefix}),
                                 'unique_ae17_identities': len({r['ae17']['identity'] for r in prefix}),
                                 'unique_mrks': len({r['mrks_id'] for r in prefix}),
                                 'prefix_sha256': key(prefix), 'full_manifest_sha256': f010.continuation.MANIFEST_SHA})
    return prefix, source, receipt['scales']


def loaded(source):
    model, shadow, run = audit.existing._load_eval_model(source, f010.PARENT_CHECKPOINT)
    shadow.load_state_dict(model.state_dict(), strict=True)
    assert all(m.p == 0 for m in model.modules() if isinstance(m, torch.nn.Dropout))
    parameters = f010._check_model(model)
    names = tuple(parameters)
    expected = tuple(audit.read(audit.geometry.prior.OUT / 'b8_t128/result.json')['parameter_names'])
    assert names == expected and sum(p.numel() for p in parameters.values()) == 9446
    return model, shadow, run, names


def mrks(model, bundle, identity, dispersion):
    run = b8._import_run()
    system = bundle.mrks().operator_system(identity, device='cuda', dtype=torch.float32, chunk_size=4096)
    factories = run.existing.core.make_mrks_objective_factories(model, system, point_chunk_size=256,
                                                            exc_chunk_size=4096, dispersions=dispersion)
    losses, raw = {}, {}
    for task, factory in zip(('exc', 'op'), factories, strict=True):
        values, grads = run.existing.core.compute_isolated_task_gradients(model, {task: factory}, task_order=(task,))
        material = run.existing.core.materialize_task_zeros(model, grads, task_order=(task,))
        losses[task] = values[task]
        raw[task] = {n: g.double() for n, g in material[task].items()}
    return losses, raw


def flat(gradient, names):
    assert tuple(gradient) == names
    return np.concatenate([gradient[n].detach().double().cpu().numpy().reshape(-1) for n in names])


def physical_mean(model, shadow, bundle, prefix, t, dispersion, md):
    losses = dict.fromkeys(TASKS, 0.0)
    accum = None
    for p in positions(t):
        values, gradients, _system, _seconds = b8.other_tasks(model, shadow, bundle, prefix[p], dispersion, md)
        if accum is None:
            accum = {task: {n: torch.zeros_like(g, dtype=torch.float64) for n, g in gradients[task].items()} for task in TASKS}
        for task in TASKS:
            losses[task] += values[task] / 4
            for n, g in gradients[task].items():
                assert g.dtype == torch.float64
                accum[task][n].add_(g, alpha=.25)
        del gradients
    return losses, accum


def census(prefix, source, budget):
    model, shadow, run, names = loaded(source)
    before = run.existing.digest(model)
    rng = b8._capture_rng()
    bundle = audit.TrainingBundle(ROOT.parent / 'publication_dataset_v1')
    dispersion = bundle.chemistry_dispersions()
    md = audit.read(bundle.root / 'mrks/dispersion.json')
    cache = OUT / 'singleton'
    cache.mkdir(exist_ok=True)
    records = [('ae17', r['ae17']) for r in prefix] + [('mrks', r['mrks_id']) for r in prefix]
    distinct = {(task, key(record)): record for task, record in records}
    bindings = []
    for (task, digest), record in distinct.items():
        budget.check(120)
        path = cache / (task + '_' + digest + '.npz')
        meta = path.with_suffix('.json')
        source_record = (bundle.reactions[record['identity']] if task == 'ae17' else bundle.systems[record])
        bound = {'task': task, 'record': record, 'source_metadata_sha256': key(source_record),
                 'checkpoint_sha256': read('protocol')['checkpoint_sha256'],
                 'parameter_order_sha256': key(names)}
        if path.exists():
            receipt = audit.read(meta)
            assert receipt['binding'] == bound and receipt['sha256'] == audit.sha(path)
            bindings.append(receipt)
            continue
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        if task == 'ae17':
            reaction = b8._load_reaction(bundle, 'ae17', record)
            loss, gradient = b8._singleton(model, shadow, reaction, dispersion)
            vectors = {'ae17': flat(gradient, names)}
            losses = {'ae17': float(loss)}
            del reaction, gradient
        else:
            losses, raw = mrks(model, bundle, record, md)
            vectors = {t: flat(raw[t], names) for t in ('exc', 'op')}
            del raw
        assert all(np.isfinite(v).all() and v.shape == (9446,) for v in vectors.values())
        assert all(np.isfinite(v) for v in losses.values())
        torch.cuda.synchronize()
        np.savez_compressed(path, **vectors)
        receipt = {'binding': bound, 'sha256': audit.sha(path), 'losses': losses,
                   'seconds': time.perf_counter() - start, 'peak_allocated': torch.cuda.max_memory_allocated(),
                   'peak_reserved': torch.cuda.max_memory_reserved()}
        audit.write(meta, receipt)
        bindings.append(receipt)
        torch.cuda.empty_cache()
        print('F021_CENSUS', task, len(bindings), len(distinct), receipt['seconds'], flush=True)
    assert run.existing.digest(model) == before
    b8._restore_rng(rng)
    assert run.existing.equal(rng, b8._capture_rng())
    write('singleton_gradients_receipt', {'complete': True, 'records': bindings, 'parameter_order': names,
                                         'model_restored': True, 'rng_restored': True, 'optimizer_updates': 0})
    bundle.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('freeze', 'census'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    budget = Budget()
    try:
        prefix, source, _scales = freeze()
        if args.stage == 'census':
            census(prefix, source, budget)
    finally:
        budget.check()


if __name__ == '__main__':
    main()
