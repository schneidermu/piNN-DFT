"""Bounded Exc chunk benchmark and fixed calibration; no optimizer updates."""
import gc
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_lap_microbatch as run

OUT = run.ROOT.parent / 'lap_adamw_90update_20261008'


def main():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    OUT.mkdir(exist_ok=False)
    bundle = run.PublicationDataset(run.DATA)
    assert bundle.manifest['logical_sha256'] == run.DATA_SHA
    prior = run.read(run.OLD / 'protocol.json')
    model, shadow = run.model_at(prior['initial_state'])
    ordered = sorted(bundle.systems.values(), key=lambda r: (r['n_grid'], r['id']))
    selected = [ordered[i] for i in (0, len(ordered) // 2, len(ordered) - 1)]
    tolerance = {'energy_atol_Ha': 5e-7, 'loss_rtol': 2e-6,
                 'gradient_relative_l2': 5e-6, 'gradient_relative_max': 5e-6}
    run.write(OUT / 'benchmark_protocol.json', {'systems': selected, 'chunks': [256, 1024, 4096],
              'tolerance': tolerance, 'max_allocated_fraction': .8,
              'model_sha256': prior['initial_state']['state_sha256']})
    dispersion = run.read(run.DATA / 'mrks/dispersion.json')
    original = run.existing.core.integrated_energy
    results = []
    for row in selected:
        system = bundle.mrks().operator_system(row['id'], device='cuda', dtype=torch.float32, chunk_size=4096)
        baseline = None
        for chunk in (256, 1024, 4096):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            energies, calls = [], []
            def capture(*args, _energies=energies, **kwargs):
                value = original(*args, **kwargs)
                _energies.append(float(value.detach()))
                return value
            run.existing.core.integrated_energy = capture
            hook = model.register_forward_pre_hook(lambda *args, _calls=calls: _calls.append(1))
            exc, _ = run.existing.core.make_mrks_objective_factories(
                model, system, point_chunk_size=256, exc_chunk_size=chunk, dispersions=dispersion)
            torch.cuda.synchronize()
            began = time.perf_counter()
            try:
                values, gradients = run.existing.core.compute_isolated_task_gradients(model, {'exc': exc}, task_order=('exc',))
                materialized = run.existing.core.materialize_task_zeros(model, gradients, task_order=('exc',))
                vector = torch.cat([g.flatten().double() for g in materialized['exc'].values()])
                torch.cuda.synchronize()
                seconds = time.perf_counter() - began
            finally:
                hook.remove()
                run.existing.core.integrated_energy = original
            item = {'system': system.name, 'id': row['id'], 'n_grid': row['n_grid'], 'chunk': chunk,
                    'seconds': seconds, 'energy_Ha': energies[0], 'loss': values['exc'],
                    'allocated_bytes': torch.cuda.max_memory_allocated(), 'reserved_bytes': torch.cuda.max_memory_reserved(),
                    'chunks': math.ceil(row['n_grid'] / chunk), 'model_forward_calls': len(calls)}
            array = vector.cpu().numpy()
            path = OUT / f"exc_{system.name}_{chunk}.npy"
            np.save(path, array)
            item['gradient_sha256'] = run.file_sha256(path)
            if baseline is None:
                baseline = (item, array.copy())
            first, reference = baseline
            item['energy_absolute_error_Ha'] = abs(item['energy_Ha'] - first['energy_Ha'])
            item['loss_relative_error'] = abs(item['loss'] - first['loss']) / abs(first['loss'])
            item['gradient_relative_l2'] = float(np.linalg.norm(array - reference) / np.linalg.norm(reference))
            item['gradient_relative_max'] = float(np.max(np.abs(array - reference)) / np.max(np.abs(reference)))
            item['parity'] = (item['energy_absolute_error_Ha'] <= tolerance['energy_atol_Ha']
                              and item['loss_relative_error'] <= tolerance['loss_rtol']
                              and item['gradient_relative_l2'] <= tolerance['gradient_relative_l2']
                              and item['gradient_relative_max'] <= tolerance['gradient_relative_max'])
            results.append(item)
            run.write(OUT / 'exc_benchmark.json', results)
            print('EXC_BENCHMARK', item, flush=True)
            if not item['parity']:
                raise RuntimeError('Frozen Exc parity gate failed; no calibration/training')
            del vector, gradients, materialized
        del system
    safe = [c for c in (256, 1024, 4096) if all(r['allocated_bytes'] < .8 * torch.cuda.get_device_properties(0).total_memory
                                                for r in results if r['chunk'] == c)]
    best = min(safe, key=lambda c: sum(r['seconds'] for r in results if r['chunk'] == c))
    manifest = run.sampling(bundle, 90)
    run.write(OUT / 'sampling_manifest.json', manifest)
    # Eight databases first, then four ordinary draws; mRKS size quantiles.
    rows = list(bundle.reactions.values())
    databases = sorted({r['database'] for r in rows if r['task'] == 'relchem'})
    panel = []
    for i in range(12):
        entry = dict(manifest[i])
        if i < len(databases):
            reaction = min((r for r in rows if r['task'] == 'relchem' and r['database'] == databases[i]), key=lambda r: r['id'])
            entry['relchem'] = {'identity': reaction['id'], 'database': reaction['database'],
                                'reaction_id': reaction['reaction_id'], 'variant': sorted(reaction['variants'])[i % 8], 'weight': 1.0}
        entry['mrks_id'] = ordered[round(i * (len(ordered) - 1) / 11)]['id']
        panel.append(entry)
    run.write(OUT / 'calibration_manifest.json', panel)
    records = []
    for entry in panel:
        record, raw = run.measure(model, shadow, bundle, entry, bundle.chemistry_dispersions(), dispersion, best)
        records.append(record)
        run.write(OUT / 'calibration_records.json', records)
        print('CALIBRATION', len(records), record['norms'], flush=True)
        del raw
    norms = np.array([[r['norms'][t] for t in run.TASKS] for r in records])
    scales, coefficients = run.fixed_coefficients(norms)
    calibration = {'scales': scales, 'lambda': coefficients, 'epsilon': 1e-12, 'normalization': 4.0}
    calibration['manifest_sha256'] = run.file_sha256(OUT / 'sampling_manifest.json')
    calibration['calibration_manifest_sha256'] = run.file_sha256(OUT / 'calibration_manifest.json')
    calibration['statistics'] = {t: dict(zip(('p25', 'median', 'p75', 'p90', 'max'), np.percentile(norms[:, j], [25, 50, 75, 90, 100]).tolist()))
                                 for j, t in enumerate(run.TASKS)}
    calibration['old_calibration'] = run.read(run.OUT / 'calibration.json')
    run.write(OUT / 'calibration.json', calibration)
    protocol = {'dataset_sha256': run.DATA_SHA, 'initial_state': prior['initial_state'],
                'initialization': 'seed11 P536; 536 PBE predopt steps; no additional displacement; never S5',
                'adamw': prior['adamw'], 'exc_chunk_size': best, 'operator_chunk_size': 256,
                'total_updates': 90, 'initial_lr': 1e-6, 'final_lr': 1e-7,
                'scheduler': 'CosineAnnealingLR T_max=90', 'no_svrg': True,
                'calibration_manifest_sha256': calibration['calibration_manifest_sha256'],
                'sampling': 'Independent uniform chemistry identity and variant; balanced shuffled mRKS90 cycle',
                'chemistry_objective': 'Stochastic singleton objective; evaluation: fixed one variant per identity, never exhaustive',
                'task_order': list(run.TASKS), 'checkpoints': [0, 10, 45, 90]}
    run.write(OUT / 'protocol.json', protocol)
    bundle.close()
    print('FROZEN_90_PROTOCOL', best, calibration['lambda'], flush=True)


if __name__ == '__main__':
    main()
