"""Fixed F023 metadata authentication and cached S128 geometry; no training here."""
import argparse
import hashlib
import json
import re
import subprocess
import tarfile
import time

import h5py
import lap_f021_phys4 as previous
import lap_f021_phys4_run as census
import lap_f022_scalarization as dual
import numpy as np
import torch

ROOT = previous.ROOT
OUT = ROOT.parent / 'lap_f023_size_20261014'
PARENT = '1c83a34fbea1aaa3a748cb23aca01ccf7e68bf0f'
audit, b8, f010 = previous.audit, previous.b8, previous.f010
BUDGET = 36000
# Atomic number is defined by periodic-table order; ionic charge comes from G09.
ATOMIC_NUMBERS = dict(zip(('H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F',
                          'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl'), range(1, 18), strict=True))


def read(name):
    return audit.read(ROOT / ('lap_f023_' + name + '.json'))


def write(name, payload):
    audit.write(ROOT / ('lap_f023_' + name + '.json'), payload)


def table(name, rows):
    audit.table('lap_f023_' + name + '.csv', rows)


class Budget:
    def __init__(self):
        self.origin = read('protocol')['wall_origin_unix']

    def used(self):
        return time.time() - self.origin

    def check(self, reserve=0):
        value = self.used()
        audit.write(OUT / 'budget_usage.json', {'wall_seconds': value, 'budget_seconds': BUDGET})
        if value + reserve >= BUDGET:
            raise TimeoutError('F023 fixed wall budget/reserve exhausted')


def freeze():
    subprocess.run(['git', 'merge-base', '--is-ancestor', PARENT, 'HEAD'], cwd=ROOT, check=True)
    OUT.mkdir(exist_ok=True)
    paths = ['tools/lap_f023_size.py', 'tools/lap_b8_adamw.py',
        'tools/lap_f010_uniform_normalized_chemistry.py', 'train_models/lap_fixed_adamw.py',
        'train_models/NN_models_lap.py', 'train_models/lap_moo_training.py',
        'train_models/lap_operator.py', 'train_models/build_lap_operator_data.py',
        'train_models/publication_data/loader.py', 'train_models/optuna_joint.py']
    protocol = {'experiment': 'F023', 'parent': PARENT, 'architecture': 'R0', 'coordinates': 9446,
        'checkpoint_sha256': dual.CHECKPOINT, 'moment_sha256': dual.MOMENT,
        'manifest_sha256': f010.continuation.MANIFEST_SHA, 'A_d_sha256': dual.SCALES,
        'dataset_logical_sha256': b8.DATA_SHA, 'size_exponent': .5,
        'calibration_population': 'S128; exact original positions 0..127 with repeats and variants',
        'chemistry_population': 'S251 stream Jacobian for calibration/geometry; E251 fixed variants for scientific evaluation',
        'c010': audit.geometry.C010, 'lambdas': b8.LAMBDAS, 'lr': 2.5e-5,
        'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': .01, 'foreach': False,
        'scheduler': None, 'clipping': None, 'updates': 128, 'arms': 1, 'checkpoints': [0, 64, 128],
        'scientific_baselines': b8.BASELINE, 'hydrogen_baseline_kcal': 1.3739066446900097,
        'operator_size_weight': 1., 'wall_budget_seconds': BUDGET,
        'gradient_rtol': 1e-10, 'gradient_atol': 1e-10,
        'native_scalar_gradient_bound': '1e-10 +16*eps32*mean(singleton gradient norms)',
        'chemistry_scalar_parity_kcal': 1e-6, 'electron_count_tolerance': 1e-6,
        'mechanism_dot_relative_resolution': 1e-5, 'mechanism_dot_absolute_resolution': 1e-8,
        'mechanism_H_comparison_resolution_kcal': 1e-6,
        'Clean28': 'only final t128 after all original P536 gates and H preservation; no t64 access',
        'source_hashes': {p: audit.sha(ROOT / p) for p in paths}}
    path = ROOT / 'lap_f023_protocol.json'
    if path.exists():
        saved = read('protocol')
        protocol['wall_origin_unix'] = saved['wall_origin_unix']
        assert saved == protocol, 'Frozen F023 protocol/source changed'
    else:
        protocol['wall_origin_unix'] = time.time()
        write('protocol', protocol)
    return protocol


def electron_count(nuclear_charges, charge):
    z = np.asarray(nuclear_charges)
    if z.ndim != 1 or len(z) == 0 or not np.isfinite(z).all() or (z <= 0).any() or not np.equal(z, z.astype(int)).all():
        raise ValueError('Exact positive integer nuclear charges required')
    if not isinstance(charge, int):
        raise TypeError('Exact integer ionic charge required')
    value = int(z.sum()) - charge
    if value <= 0:
        raise ValueError('Electron count must be strictly positive')
    return value


def authenticate():
    protocol = freeze()
    budget = Budget()
    budget.check(600)
    _prefix, source, _scales = previous.freeze()
    assert protocol['source_hashes']['train_models/lap_operator.py'] == source['source_hashes']['train_models/lap_operator.py']
    saved = torch.load(f010.PARENT_CHECKPOINT, map_location='cpu', weights_only=False)
    assert f010.continuation.moment_sha(saved['optimizer']) == dual.MOMENT
    bundle = audit.TrainingBundle(ROOT.parent / 'publication_dataset_v1')
    root = bundle.root
    manifest = bundle.manifest
    files = {r['file']: r['sha256'] for r in manifest['files']}
    geometry_path = root / 'provenance/training_geometry_identities.json'
    assert audit.sha(geometry_path) == files['provenance/training_geometry_identities.json']
    geometries = audit.read(geometry_path)
    official = ROOT.parent / 'publication_dataset_build_20261007/mn_databases.tar.gz'
    sources = audit.read(root / 'provenance/sources.json')
    assert audit.sha(official) == sources['minnesota_geometries']['sha256']
    atoms = {r['id']: r for r in bundle.reactions.values() if r['task'] == 'ae17'}
    atomic = {}
    with tarfile.open(official) as archive:
        for identity, row in atoms.items():
            assert len(row['components']) == 1 and row['coefficients'] == [1.]
            component = row['components'][0]
            geom = geometries[component]
            xyz = archive.extractfile(geom['xyz_source']).read()
            g09 = archive.extractfile(geom['g09_source']).read()
            assert hashlib.sha256(xyz).hexdigest() == geom['xyz_sha256']
            assert hashlib.sha256(g09).hexdigest() == geom['g09_sha256']
            lines = xyz.decode().splitlines()
            assert int(lines[0]) == 1 and lines[2].split()[0] == geom['Elements'][0]
            match = re.search(r'^\s*(-?\d+)[\s,]+(\d+)\s*$', g09.decode(), re.MULTILINE)
            assert match and int(match[1]) == geom['Charge'] and int(match[2])-1 == geom['UHF']
            Z = ATOMIC_NUMBERS[geom['Elements'][0]]
            Ne = electron_count([Z], geom['Charge'])
            if component == 'H_ae17':
                assert Z == 1 and geom['Charge'] == 0 and Ne == 1
            atomic[identity] = {'identity': identity, 'source_id': component, 'Z': Z, 'ionic_charge': geom['Charge'],
                'Ne': Ne, 'reference_kcal': row['target_kcal_mol'], 'source_sha256': row['source_group_sha256'],
                'reference_provenance': 'native F32 AE17 reference; official Minnesota XYZ/G09 charge identity',
                'g09_sha256': geom['g09_sha256'], 'xyz_sha256': geom['xyz_sha256']}
    molecule = {}
    handles = {}
    try:
        for identity, row in bundle.systems.items():
            budget.check(600)
            if row['shard'] not in handles:
                handles[row['shard']] = h5py.File(root / row['shard'], 'r')
            handle = handles[row['shard']]
            metadata = json.loads(handle[row['group']].attrs['source_metadata_json'])
            assert metadata['record_sha256'] == row['source_record_sha256']
            assert metadata['system_name'] == row['source_id'] and metadata['nao'] == row['n_ao']
            mol = metadata['molecule']
            Ne = electron_count(mol['atom_charges'], mol['charge'])
            measured = float(metadata['verification']['electron_count_from_dmks_overlap'])
            assert np.isfinite(measured) and abs(measured-Ne) <= 1e-6
            molecule[identity] = {'identity': identity, 'source_id': row['source_id'], 'Ne': Ne,
                'trace_dm_S': measured, 'charge': mol['charge'], 'n_ao': row['n_ao'],
                'source_sha256': row['source_sha256'], 'record_sha256': row['source_record_sha256'],
                'source_metadata_sha256': previous.key(metadata), 'npz_sha256': metadata['source']['npz_file_sha256'],
                'reference_provenance': metadata['target_preservation']['Exc']}
    finally:
        for handle in handles.values():
            handle.close()
    assert len(atomic) == 17 and len(molecule) == 90
    census_receipt = audit.read(ROOT / 'lap_f021_phys4_singleton_gradients_receipt.json')
    order = census_receipt['parameter_order']
    cache_hashes = {}
    for row in census_receipt['records']:
        binding = row['binding']
        assert binding['checkpoint_sha256'] == dual.CHECKPOINT and binding['parameter_order_sha256'] == previous.key(tuple(order))
        record = binding['record']
        numerical = bundle.reactions[record['identity']] if binding['task'] == 'ae17' else bundle.systems[record]
        assert previous.key(numerical) == binding['source_metadata_sha256']
        path = previous.OUT / 'singleton' / (binding['task'] + '_' + previous.key(record) + '.npz')
        assert audit.sha(path) == row['sha256']
        cache_hashes[str(path.relative_to(previous.OUT))] = row['sha256']
    bundle.close()
    audit.write(OUT / 'authenticated_electrons.json', {'ae17': atomic, 'mrks': molecule})
    write('provenance', {'authenticated': True, 'checkpoint_sha256': dual.CHECKPOINT, 'moment_sha256': dual.MOMENT,
        'parameter_order': order, 'manifest_sha256': protocol['manifest_sha256'], 'A_d_sha256': dual.SCALES,
        'dataset_logical_sha256': b8.DATA_SHA, 'source_hashes': protocol['source_hashes'],
        'electron_metadata_sha256': audit.sha(OUT / 'authenticated_electrons.json'),
        'official_atomic_archive_sha256': audit.sha(official), 'reused_gradient_cache_hashes': cache_hashes,
        'authenticated_atoms': 17, 'authenticated_mrks': 90, 'Clean28_accessed': False,
        'electron_method': 'official XYZ/G09 nuclear/ionic charges; saved Tr(dm_ks S) verification cross-checked against molecule charge',
        'new_density_or_SCF': False})
    return atomic, molecule


def cached():
    atomic, molecule = authenticate()
    prefix = f010._batches()[:128]
    arrays = {t: [] for t in ('ae17', 'exc', 'op')}
    wa, we = [], []
    for row in prefix:
        with census.cache('ae17', row['ae17'])[0] as data:
            arrays['ae17'].append(data['ae17'].copy())
        with census.cache('mrks', row['mrks_id'])[0] as data:
            for t in ('exc', 'op'):
                arrays[t].append(data[t].copy())
        wa.append(atomic[row['ae17']['identity']]['Ne']**-.5)
        we.append(molecule[row['mrks_id']]['Ne']**-.5)
    arrays = {t: np.asarray(v, dtype=np.float64) for t, v in arrays.items()}
    weights = {'ae17': np.asarray(wa), 'exc': np.asarray(we)}
    scales = {t: float(np.linalg.norm(arrays[t].mean(axis=0)) / np.linalg.norm((arrays[t]*weights[t][:, None]).mean(axis=0))) for t in weights}
    weighted = {t: arrays[t]*weights[t][:, None]*scales[t] for t in weights}
    weighted['op'] = arrays['op'].copy()
    for t in weights:
        assert np.isfinite(scales[t]) and scales[t] > 0
        assert abs(np.linalg.norm(weighted[t].mean(axis=0))/np.linalg.norm(arrays[t].mean(axis=0))-1) < 1e-10
    np.testing.assert_array_equal(arrays['op'], weighted['op'])
    chemistry_batches = census.chemistry_batches(prefix) * b8.LAMBDAS['relchem'] * audit.geometry.C010
    physical = {label: sum(b8.LAMBDAS[t]*v[t] for t in ('ae17', 'exc', 'op')) for label, v in [('old', arrays), ('new', weighted)]}
    joint = {label: chemistry_batches + p for label, p in physical.items()}
    # The prefix's chemistry mean is a distinct population; S251 diagnostics are also retained.
    with np.load(dual.stream.OUT / 'stream_jacobian.npz') as data:
        J, errors = data['gradients'].copy(), data['residual'].copy()
    rows = audit.read(dual.stream.OUT / 'stream_rows.json')
    gradients, _cj, _cl, _f, _penalty = dual.chemistry(J, errors, [r['database'] for r in rows], audit.read(ROOT/'lap_f010_normalization_receipt.json')['scales'])
    full_chemistry = b8.LAMBDAS['relchem']*audit.geometry.C010*gradients['JW']
    gL = gradients['legacy']
    means = {label: p.mean(axis=0) for label, p in physical.items()}
    calibrated_joint = {label: full_chemistry+p for label, p in means.items()}
    denominator = float(b8.LAMBDAS['relchem']*audit.geometry.C010*np.linalg.norm(gL)*np.linalg.norm(gradients['JW']))
    bounds = {label: {'legacy_dot_physics': float(gL @ p),
        'legacy_dot_joint': float(gL @ calibrated_joint[label]),
        'max_favorable_chemistry_dot': denominator, 'R_conflict': max(0., -float(gL @ p))/denominator} for label, p in means.items()}
    write('legacy_conflict_bounds', {'physical_population': 'S128', 'chemistry_population': 'S251', **bounds})
    write('calibration', {'population': 'S128', 'c_A': scales['ae17'], 'c_E': scales['exc'],
        'norms': {t: {'old': float(np.linalg.norm(arrays[t].mean(axis=0))),
            'size_unscaled': float(np.linalg.norm((arrays[t]*weights[t][:, None]).mean(axis=0))),
            'new_calibrated': float(np.linalg.norm(weighted[t].mean(axis=0))),
            'old_new_cosine': dual.cosine(arrays[t].mean(axis=0), weighted[t].mean(axis=0))} for t in weights},
        'physical_old_new_cosine': dual.cosine(means['old'], means['new']),
        'joint_old_new_cosine_S251': dual.cosine(calibrated_joint['old'], calibrated_joint['new']),
        'new_loss_weights': {'ae17': {k: scales['ae17']*v['Ne']**-.5 for k,v in atomic.items()},
            'exc': {k: scales['exc']*v['Ne']**-.5 for k,v in molecule.items()}}})
    geometry = []
    for label in ('old', 'new'):
        for target, g in {**gradients, **{t: arrays[t].mean(axis=0) for t in arrays}}.items():
            G = calibrated_joint[label]
            geometry.append({'population': 'chemistry S251; physical S128', 'direction': label, 'target': target,
                'cosine': dual.cosine(g, G), 'dot': float(g @ G), 'negative_gradient_derivative': -float(g @ G),
                'joint_norm': float(np.linalg.norm(G)), 'physical_norm': float(np.linalg.norm(means[label]))})
    table('gradient_geometry', geometry)
    fidelity = []
    for label, value in joint.items():
        mean = value.mean(axis=0)
        for i, vector in enumerate(value):
            fidelity.append({'position': i, 'arm': label, 'cosine': dual.cosine(vector, mean),
                'positive_dot': bool(vector @ mean > 0), 'norm_ratio': float(np.linalg.norm(vector)/np.linalg.norm(mean)),
                'relative_squared_error': float(np.linalg.norm(vector-mean)**2/np.linalg.norm(mean)**2),
                'F32_representable': bool(np.isfinite(vector.astype(np.float32)).all())})
    table('stochastic_fidelity', fidelity)
    np.savez_compressed(OUT/'cached_arrays.npz', **arrays, **{t+'_new': weighted[t] for t in weighted},
        chemistry=chemistry_batches, joint_old=joint['old'], joint_new=joint['new'], JW_S=gradients['JW'], legacy_S=gL)
    p = read('provenance')
    p['cached_arrays_sha256'] = audit.sha(OUT/'cached_arrays.npz')
    write('provenance', p)
    print(json.dumps({'c_A': scales['ae17'], 'c_E': scales['exc'], 'bounds': bounds,
        'cosine_A': dual.cosine(arrays['ae17'].mean(axis=0), weighted['ae17'].mean(axis=0)),
        'cosine_E': dual.cosine(arrays['exc'].mean(axis=0), weighted['exc'].mean(axis=0))}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('freeze', 'metadata', 'cached'))
    args = parser.parse_args()
    {'freeze': freeze, 'metadata': authenticate, 'cached': cached}[args.stage]()
