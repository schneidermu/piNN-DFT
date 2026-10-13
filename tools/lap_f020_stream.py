"""Read-only, resumable F020 stream-variant reconstruction."""
import os

os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import hashlib
import json
import subprocess
import time
from collections import Counter

import lap_f015_audit as audit
import lap_f019_fixed_db as f019
import numpy as np
import torch

ROOT = audit.ROOT
OUT = ROOT.parent / 'lap_f020_stream_20261013'
BUDGET = 1800


def write(name, value):
    audit.write(ROOT / name, value)


def source_hash(meta, variant, bundle):
    records = meta['variants'][variant]
    shards = sorted({r['shard'] for r in records})
    content = {'reaction': meta, 'variant': variant,
               'shards': {s: bundle.checksums[s] for s in shards}}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def execute():
    if (OUT / 'scientific_execution_closed.json').exists():
        raise SystemExit('F020 scientific execution closed after frozen stochastic gate failure')
    started = time.perf_counter()
    consumed = audit.read(ROOT / 'lap_f020_decision.json')['elapsed_seconds'] if (ROOT / 'lap_f020_decision.json').exists() else 0.0
    b8 = audit.geometry.b8
    subprocess.run(['git', 'merge-base', '--is-ancestor',
                    '6032da0ecc36394b1530bb73e12d3a0d8c10a125', 'HEAD'], cwd=ROOT, check=True)
    old_receipts = audit.read(ROOT / 'lap_f013_gradient_receipts.json')
    spec = audit.geometry.STATES[0]
    state, scales, names = audit.load(spec)
    assert b8.sha256_file(spec['checkpoint']) == spec['sha256']
    saved = torch.load(spec['checkpoint'], map_location='cpu', weights_only=False)
    assert audit.geometry.f010.continuation.moment_sha(saved['optimizer']) == '4659e23cce1cf2c66f701b0b445b897d58838739ca15cffc9300fa0fdc1a40d0'
    assert audit.geometry.arm.scales_sha256(scales) == old_receipts['a_d_sha256']
    batches = audit.geometry.f010._batches()
    by_identity = {}
    presentations = []
    for batch in batches:
        for slot, row in enumerate(batch['relchem']):
            presentations.append({'position': batch['cursor'], 'slot': slot, **row})
            by_identity.setdefault(row['identity'], []).append(row)
    assert len(presentations) == 2008 and len(by_identity) == 251
    assert all(len(rows) == 8 and len({r['variant'] for r in rows}) == 1 for rows in by_identity.values())
    unique = [by_identity[r['identity']][0] for r in state['rows']]
    assert Counter(r['database'] for r in unique) == f019.COUNTS
    matching = [i for i, (s, e) in enumerate(zip(unique, state['rows'], strict=True)) if s['variant'] == e['variant']]
    missing = [i for i in range(251) if i not in matching]
    assert len(matching) == 39 and len(missing) == 212
    protocol = {'experiment': 'F020', 'parent': '6032da0ecc36394b1530bb73e12d3a0d8c10a125',
                'budget_seconds': BUDGET, 'gradient_rtol': 1e-10, 'gradient_atol': 1e-10,
                'scalar_atol': 1e-8, 'stream_sha256': audit.geometry.f010.continuation.MANIFEST_SHA,
                'checkpoint_sha256': spec['sha256'], 'counts': f019.COUNTS,
                'positions': audit.read(ROOT / 'lap_f019_protocol.json')['positions'],
                'profile_rows': [next(i for i in missing if unique[i]['database'] == d) for d in sorted(f019.COUNTS)],
                'training': False, 'optimizer_updates': 0, 'clean28_access': False}
    path = ROOT / 'lap_f020_protocol.json'
    if path.exists():
        assert audit.read(path) == protocol
    else:
        write(path.name, protocol)
    bundle = audit.TrainingBundle(ROOT.parent / 'publication_dataset_v1')
    inventory = []
    for row in presentations:
        meta = bundle.reactions[row['identity']]
        assert meta['database'] == row['database'] and meta['reaction_id'] == row['reaction_id']
        assert row['variant'] in meta['component_paths']
        inventory.append({**row, 'coefficients': json.dumps(meta['coefficients']),
                          'source_hash': source_hash(meta, row['variant'], bundle),
                          'target_metadata_hash': hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()})
    audit.table('lap_f020_stream_variant_inventory.csv', inventory)
    write('lap_f020_variant_match_audit.json', {'unique': 251, 'presentations': 2008,
                                              'matched': 39, 'mismatched': 212, 'stream': 'VERIFIED'})
    OUT.mkdir(exist_ok=True)
    dispersion = bundle.chemistry_dispersions()
    model, shadow, _run = audit.existing._load_eval_model(b8.verify_sources(), spec['checkpoint'])
    shadow.load_state_dict(model.state_dict(), strict=True)
    parameters = _run.existing.named_trainable_parameters(shadow)
    assert tuple(parameters) == names and sum(p.numel() for p in parameters.values()) == 9446
    model_hash = _run.existing.digest(model)
    profile = []
    order = protocol['profile_rows'] + [i for i in missing if i not in protocol['profile_rows']]
    parity = []
    for i in order:
        target = OUT / f'row_{i:03d}.npz'
        metadata = OUT / f'row_{i:03d}.json'
        identity = unique[i]
        bound = {'identity': identity, 'source_hash': source_hash(bundle.reactions[identity['identity']], identity['variant'], bundle),
                 'checkpoint_sha256': spec['sha256'], 'parameter_order_sha256': hashlib.sha256(json.dumps(names).encode()).hexdigest()}
        if target.exists():
            receipt = audit.read(metadata)
            assert receipt['binding'] == bound and audit.sha(target) == receipt['sha256']
            continue
        elapsed = consumed + time.perf_counter() - started
        if elapsed >= BUDGET - 30:
            break
        if len(profile) == len(protocol['profile_rows']):
            estimate = max(r['seconds'] for r in profile) * (len(missing) - len(list(OUT.glob('row_*.npz'))))
            write('lap_f020_runtime_projection.json', {'profile': profile, 'remaining_upper_projection_seconds': estimate,
                                                      'elapsed': elapsed, 'budget': BUDGET})
            # The projection includes cold loading and independent parity calls.
            # Bound actual cumulative execution, rather than treating it as a lower bound.
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        row_start = time.perf_counter()
        reaction = b8._load_reaction(bundle, 'relchem', identity)
        prediction, reference = audit.chemistry._prediction(shadow, reaction, dispersion)
        grads = torch.autograd.grad(prediction, tuple(parameters.values()), allow_unused=True)
        flat = audit.chemistry._flat(names, grads)
        residual = float((prediction - reference).detach().cpu())
        ref = float(reference.detach().cpu())
        assert flat.shape == (9446,) and np.isfinite(flat).all() and np.isfinite(residual)
        if i in protocol['profile_rows']:
            loss, qualified = b8._singleton(model, shadow, reaction, dispersion)
            factor = audit.chemistry._factor(identity['database'])
            expected = flat * audit.geometry.f010.fact.deriv_l1(residual) * factor
            actual = np.concatenate([qualified[n].detach().cpu().numpy().reshape(-1) for n in names])
            gap = float(np.linalg.norm(expected - actual) / max(np.linalg.norm(actual), 1e-30))
            assert gap <= 1e-10
            parity.append({'row': i, 'database': identity['database'], 'relative_gradient_gap': gap,
                           'scalar_gap': abs(float(loss) - np.sqrt(residual**2 + 1e-20) * factor)})
        np.savez_compressed(target, gradient=flat, residual=residual, reference=ref)
        audit.write(metadata, {'binding': bound, 'sha256': audit.sha(target)})
        del reaction, prediction, reference, grads
        torch.cuda.synchronize()
        profile.append({'row': i, 'seconds': time.perf_counter() - row_start,
                        'peak_allocated': torch.cuda.max_memory_allocated(),
                        'peak_reserved': torch.cuda.max_memory_reserved()})
        print('F020_ROW', i, len(list(OUT.glob('row_*.npz'))), flush=True)
    audit.table('lap_f020_jacobian_parity.csv', parity) if parity else None
    count = len(list(OUT.glob('row_*.npz')))
    assert _run.existing.digest(model) == model_hash
    write('lap_f020_decision.json', {'classification': 'C' if count < 212 else 'INCOMPLETE_DOWNSTREAM',
                                    'historical_stream': 'VERIFIED', 'unique': 251, 'presentations': 2008,
                                    'eligible_archived_rows': 39, 'reused_rows': 0,
                                    'new_rows': count, 'optimizer_updates': 0,
                                    'training': False, 'elapsed_seconds': consumed + time.perf_counter() - started,
                                    'cache': OUT.name, 'profile': profile, 'gpu': torch.cuda.get_device_name(),
                                    'model_restored': True, 'dependent_stages': 'NOT RUN'})
    bundle.close()



if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    execute()
