"""Database-stratified relchem importance sampling versus preserved IID AdamW.

Arm B draws one of eight databases uniformly, then one identity uniformly inside
it, and multiplies that singleton relchem gradient by p/q = 8*n_d/251 once.
AE17 identities, mRKS systems, coefficients, and AdamW settings stay fixed.
"""
import argparse
import copy
import hashlib
import random
import shutil
import subprocess
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_lap_microbatch as run
from tools.evaluate_microbatch_endpoint import evaluate
from train_models.lap_fixed_adamw import TASKS, eligible, weighted_gradient

DB_STRAT_SEED = 202610093
POPULATION = 251
N_DATABASES = 8
UPDATES = 90
CHECKPOINTS = (0, 20, 59, 70, 80, 90)
INITIAL_SHA = '3c2e45d6a86cccf80d7b4dc458e67f9b406935ed60d4e57433bb441a6df88da6'
DATA_SHA = '61c221a19b9987717e69cac182ad545241f8807db4126c0949a99992e4c210ef'
EVAL_MANIFEST_SHA = '132bd550161be6bc6761f8df1de8d3755c0b0debab53663208c3a9bbf56cb805'
EVALUATOR_SHA = 'b92a00bc8f51149e29be846bbadbedc74c47fd172b975f1fd0888fd4caf6da3f'
TRAINER_SHA = '2e837c3a88ca3d4737c3c3397dcfbb98adeb7f6917adc8c693cecf205dc37e9e'
CONTROL_MANIFEST_SHA = 'e84e237d88449edfae5c68f7caecd85791b57f60a4b6239a37e1ae1350a71089'
CONTROL_CALIBRATION_SHA = '4d97df1ec78aa01a2c9380a86a323b2f0b46828dfc5975f5e3a63c75f4440096'
BEST_KNOWN_CLEAN28 = 8.619694172
OUTPUT = run.ROOT.parent / 'lap_dbstrat_importance_20261009'
CONTROL = run.ROOT.parent / 'lap_adamw_lr_sweep_20261008' / '1e-4'
TRAJECTORY = run.ROOT.parent / 'lap_iid_adamw_t59_t90_20261009'
EXPECTED_COUNTS = {
    'ABDE4': 4, 'DBH76': 70, 'EA13': 11, 'IP13': 13,
    'MGAE109': 104, 'NCCE31': 28, 'PA8': 8, 'pTC13': 13,
}
LAMBDAS = {
    'relchem': 0.017015480965588553,
    'ae17': 0.00005141254618347414,
    'exc': 0.000015094644512009712,
    'op': 0.33597561607048215,
}
BASELINE = {
    'relchem': 1.2354710978866068,
    'ae17': 24.901047104016875,
    'exc': 92.17480502000551,
    'op': 0.03314821681605566,
}
PHYSICS = {
    'lap_training.py': 'ca789703e536f9311178ada07fd41d92ba7f79afd35b25ad240647747b3e9b77',
    'lap_moo_training.py': 'e3dafbe76279e13b73f2a6dd5f2547294902387d21fc1b78b4ce917430495113',
    'lap_operator.py': '7fae09c0857187875377fbf5c1c7c8d19b1afc869942310f75f5462f7bd2e864',
    'lap_fixed_adamw.py': '9effc074bd18236ad7af4c4e05fa04be1754d35e228b9a0567163c4b0bd5eb0a',
}
IID_CLEAN28_FILES = {
    0: 'baseline_0_validation.json',
    59: 'endpoint_59_validation.json',
    70: 'endpoint_70_validation.json',
    80: 'endpoint_80_validation.json',
    90: 'endpoint_90_validation.json',
}


def draw(seed, domain):
    """SHA256 domain stream. Independent of Python's process hash."""
    digest = hashlib.sha256(f'{seed}:{domain}'.encode()).digest()
    return random.Random(int.from_bytes(digest, 'big'))


def importance_fraction(database_n):
    if database_n not in EXPECTED_COUNTS.values():
        raise ValueError('Unknown relchem database size')
    return Fraction(N_DATABASES * database_n, POPULATION)


def f64_weight(numerator, denominator, device):
    if denominator != POPULATION or numerator % N_DATABASES:
        raise ValueError('Importance weight must be 8*n_d/251')
    numer = torch.tensor(numerator, dtype=torch.float64, device=device)
    denom = torch.tensor(denominator, dtype=torch.float64, device=device)
    return numer / denom


def relchem_population(rows):
    selected = [row for row in rows if row['task'] == 'relchem']
    counts = dict(Counter(row['database'] for row in selected))
    if len(selected) != POPULATION or counts != EXPECTED_COUNTS:
        raise ValueError(f'Relchem population {counts} differs from the verified 251-identity table')
    if any(len(row['variants']) != 8 for row in selected):
        raise ValueError('Every relchem identity needs eight quadrature variants')
    return selected


def stratified_relchem(rows, index, seed=DB_STRAT_SEED):
    """Uniform database, then uniform identity, then uniform variant, with replacement."""
    if index < 0:
        raise ValueError('Nonnegative cursor required')
    groups = {}
    for row in relchem_population(rows):
        groups.setdefault(row['database'], []).append(row)
    databases = sorted(groups)
    database = draw(seed, f'relchem:{index}:database').choice(databases)
    ordered = sorted(groups[database], key=lambda row: row['id'])
    record = draw(seed, f'relchem:{index}:identity').choice(ordered)
    variants = sorted(record['variants'])
    variant = draw(seed, f'relchem:{index}:variant').choice(variants)
    database_n = len(ordered)
    return {
        'identity': record['id'], 'database': database, 'reaction_id': record['reaction_id'],
        'variant': variant, 'weight': 1.0, 'database_n': database_n,
        'importance_numerator': N_DATABASES * database_n, 'importance_denominator': POPULATION,
        'seed': seed, 'sampler': 'uniform-database-then-uniform-identity',
    }


def build_manifest(relchem_rows, control_rows):
    if len(control_rows) != UPDATES:
        raise ValueError('Control stream must contain 90 frozen updates')
    manifest = []
    for old in control_rows:
        manifest.append({
            'cursor': old['cursor'],
            'relchem': stratified_relchem(relchem_rows, old['cursor']),
            'ae17': copy.deepcopy(old['ae17']),
            'mrks_id': old['mrks_id'],
        })
    return manifest


def validate_manifest(manifest, reactions, systems, control_rows):
    if len(manifest) != UPDATES or [row['cursor'] for row in manifest] != list(range(UPDATES)):
        raise ValueError('Manifest must be 90 sequential update rows')
    changed = False
    for row, old in zip(manifest, control_rows, strict=True):
        if set(row) != {'cursor', 'relchem', 'ae17', 'mrks_id'}:
            raise ValueError('Each update needs exactly one relchem, one AE17, and one mRKS sample')
        if row['ae17'] != old['ae17'] or row['mrks_id'] != old['mrks_id']:
            raise ValueError('AE17 or mRKS stream differs from the IID control')
        if row['mrks_id'] not in systems:
            raise ValueError('Sampled mRKS system is outside the frozen population')
        ae_source = reactions[row['ae17']['identity']]
        if ae_source['task'] != 'ae17' or row['ae17']['variant'] not in ae_source['variants']:
            raise ValueError('AE17 variant does not belong to the copied identity')
        rel = row['relchem']
        source = reactions[rel['identity']]
        if source['task'] != 'relchem' or source['database'] != rel['database']:
            raise ValueError('Stratified identity is outside its declared database')
        if rel['variant'] not in source['variants']:
            raise ValueError('Selected quadrature variant does not belong to the identity')
        database_n = EXPECTED_COUNTS[rel['database']]
        if rel['weight'] != 1.0 or rel['database_n'] != database_n:
            raise ValueError('Frequency weight or database size was rewritten')
        if (rel['importance_numerator'], rel['importance_denominator']) != (N_DATABASES * database_n, POPULATION):
            raise ValueError('Importance weight is not 8*n_d/251')
        changed = changed or rel['identity'] != old['relchem']['identity'] or rel['variant'] != old['relchem']['variant']
    if not changed:
        raise ValueError('Refusing an unchanged copy of the IID relchem stream')
    return True


def expectation_identity(gradients):
    """Exact sum_i q(i) w(i) g_i == sum_i p(i) g_i over one synthetic vector per identity."""
    if len(gradients) != POPULATION:
        raise ValueError('Unbiasedness check enumerates all 251 identities')
    counts = Counter(database for database, _gradient in gradients)
    if dict(counts) != EXPECTED_COUNTS:
        raise ValueError('Synthetic population does not match the verified database sizes')
    under_q = Fraction(0)
    under_p = Fraction(0)
    probability = Fraction(1, POPULATION)
    for database, gradient in gradients:
        database_n = EXPECTED_COUNTS[database]
        proposal = Fraction(1, N_DATABASES * database_n)
        weight = importance_fraction(database_n)
        if proposal * weight != probability:
            raise ValueError('q(i) * importance_weight(i) is not 1/251')
        under_q += proposal * weight * gradient
        under_p += probability * gradient
    if under_q != under_p:
        raise ValueError('Importance-corrected proposal expectation differs from IID')
    return under_q


def reused_calibration(original, manifest_sha, original_sha):
    if original_sha != CONTROL_CALIBRATION_SHA:
        raise ValueError('Historical calibration file hash changed')
    if {task: original['lambda'][task] for task in TASKS} != LAMBDAS:
        raise ValueError('Historical coefficients differ from the frozen lambda values')
    updated = copy.deepcopy(original)
    updated['manifest_sha256'] = manifest_sha
    updated['provenance'] = {
        'kind': 'reuse of a historical calibration, not a new calibration',
        'original_calibration_sha256': original_sha,
        'original_manifest_sha256': original['manifest_sha256'],
        'original_calibration': original,
        'coefficients_unchanged': True,
    }
    authorized = calibration_changes(updated, original)
    if authorized != {'manifest_sha256', 'provenance'}:
        raise ValueError(f'Calibration changed unauthorized fields: {sorted(authorized)}')
    return updated


def calibration_changes(updated, original):
    keys = set(updated) | set(original)
    return {key for key in keys if updated.get(key) != original.get(key)}


def apply_importance(raw, entry, coefficients):
    """Multiply the qualified relchem gradient once. Leave AE17, Exc, and operator untouched."""
    rel = entry['relchem']
    numerator, denominator = rel['importance_numerator'], rel['importance_denominator']
    if rel.get('weight') != 1.0:
        raise ValueError('Refusing to treat the manifest frequency weight as the importance weight')
    if (numerator, denominator) != (N_DATABASES * rel['database_n'], POPULATION):
        raise ValueError('Manifest importance ratio does not match 8*n_d/251')
    sample = next(iter(raw['relchem'].values()))
    weight = f64_weight(numerator, denominator, sample.device)
    if not all(torch.isfinite(value).all() for task in TASKS for value in raw[task].values()):
        raise FloatingPointError('Nonfinite qualified task gradient')
    coefficients = {task: coefficients[task] for task in TASKS}
    uncorrected_joint = weighted_gradient(raw, coefficients)
    snapshots = {name: value.detach().clone() for name, value in raw['relchem'].items()}
    corrected_rel = {name: value.detach().double() * weight for name, value in raw['relchem'].items()}
    for name, value in snapshots.items():
        if not torch.equal(raw['relchem'][name], value):
            raise RuntimeError('Importance correction mutated the raw relchem gradient')
        if not torch.equal(corrected_rel[name], value.double() * weight):
            raise RuntimeError('Corrected relchem gradient is not the raw gradient times the weight')
    updated = dict(raw)
    updated['relchem'] = corrected_rel
    for task in ('ae17', 'exc', 'op'):
        for name, value in raw[task].items():
            if updated[task][name].data_ptr() != value.data_ptr():
                raise RuntimeError('Importance correction replaced a non-relchem gradient')
    corrected_joint = weighted_gradient(updated, coefficients)
    raw_norm = torch.cat([value.flatten().double() for value in snapshots.values()]).norm()
    corrected_norm = torch.cat([value.flatten().double() for value in corrected_rel.values()]).norm()
    return updated, {
        'importance_weight': float(weight),
        'relchem_gradient_norm_raw': float(raw_norm),
        'relchem_gradient_norm_corrected': float(corrected_norm),
        'joint_gradient_norm_uncorrected': _flat_norm(uncorrected_joint),
        'joint_gradient_norm_corrected': _flat_norm(corrected_joint),
    }


def _flat_norm(gradients):
    return float(torch.cat([value.detach().flatten().double() for value in gradients.values()]).norm())


def make_corrected_measure(native, coefficients):
    def wrapped(model, shadow, bundle, entry, dispersion, mrks_dispersion, exc_chunk_size=None):
        record, raw = native(model, shadow, bundle, entry, dispersion, mrks_dispersion, exc_chunk_size)
        if record.get('importance_applied_once'):
            raise RuntimeError('Importance correction already applied')
        updated, info = apply_importance(raw, entry, coefficients)
        record = dict(record)
        record.update(info)
        record['importance_applied_once'] = True
        return record, updated
    return wrapped


def _git_revision():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=run.ROOT, text=True).strip()


def _source_hashes():
    paths = [
        'train_lap_microbatch.py', 'train_models/lap_training.py', 'train_models/lap_moo_training.py',
        'train_models/lap_operator.py', 'train_models/lap_fixed_adamw.py', 'train_models/lap_vxc.py',
        'train_models/NN_models_lap.py', 'train_models/lap_chemistry_sampling.py',
        'dft_functionals/PBE.py', 'dft_functionals/constants.py',
        'tools/evaluate_microbatch_endpoint.py', 'tools/lap_dbstrat_importance.py',
    ]
    return {path: run.file_sha256(run.ROOT / path) for path in paths}


def _protected_hashes():
    paths = {
        'control_sampling_manifest': CONTROL / 'sampling_manifest.json',
        'control_calibration': CONTROL / 'calibration.json',
        'control_protocol': CONTROL / 'protocol.json',
        'control_checkpoint_0': CONTROL / 'ordinary_sgd_adamw' / 'checkpoint_0.pt',
        'control_latest': CONTROL / 'ordinary_sgd_adamw' / 'latest.pt',
        'trajectory_checkpoint_90': TRAJECTORY / 'ordinary_sgd_adamw' / 'checkpoint_90.pt',
        'trajectory_sampling_manifest': TRAJECTORY / 'sampling_manifest.json',
        'evaluation_manifest': TRAJECTORY / 'evaluation_manifest.json',
        'baseline_validation': TRAJECTORY / 'baseline_0_validation.json',
        'initial_model': Path(run.read(CONTROL / 'protocol.json')['initial_state']['path']),
    }
    return {name: run.file_sha256(path) for name, path in paths.items()}


def _iid_clean28():
    values = {}
    for cursor, name in IID_CLEAN28_FILES.items():
        metrics = run.read(TRAJECTORY / name)['metrics']
        clean = [row for row in metrics['reaction_rows'] if row['clean']]
        if len(clean) != 28:
            raise ValueError('Historical Clean28 panel is not 28 reactions')
        reconstructed = float(np.mean([row['weighted_absolute_error'] for row in clean]))
        if abs(reconstructed - metrics['clean28']) > 1e-12:
            raise ValueError('Historical Clean28 does not reconstruct from contributions')
        values[cursor] = metrics['clean28']
    return values


def prepare():
    if run.file_sha256(run.ROOT / 'train_lap_microbatch.py') != TRAINER_SHA:
        raise RuntimeError('Qualified trainer hash changed; refusing to continue')
    if run.file_sha256(run.ROOT / 'tools' / 'evaluate_microbatch_endpoint.py') != EVALUATOR_SHA:
        raise RuntimeError('Qualified evaluator hash changed; refusing to continue')
    for name, expected in PHYSICS.items():
        if run.file_sha256(run.ROOT / 'train_models' / name) != expected:
            raise RuntimeError(f'Physics source hash changed: {name}')
    protocol = run.read(CONTROL / 'protocol.json')
    if protocol['initial_state']['state_sha256'] != INITIAL_SHA or protocol['dataset_sha256'] != DATA_SHA:
        raise RuntimeError('Control initialization or dataset SHA differs')
    if protocol['initial_lr'] != 1e-4 or protocol['scheduler'] != 'constant' or protocol['exc_chunk_size'] != 4096:
        raise RuntimeError('Control AdamW schedule or Exc chunk differs')
    if run.file_sha256(CONTROL / 'sampling_manifest.json') != CONTROL_MANIFEST_SHA:
        raise RuntimeError('Control sampling manifest hash differs')
    if run.file_sha256(TRAJECTORY / 'evaluation_manifest.json') != EVAL_MANIFEST_SHA:
        raise RuntimeError('Fixed evaluation manifest hash differs')
    calibration = run.read(CONTROL / 'calibration.json')
    if run.file_sha256(CONTROL / 'calibration.json') != CONTROL_CALIBRATION_SHA:
        raise RuntimeError('Control calibration hash differs')
    control_rows = run.read(CONTROL / 'sampling_manifest.json')
    trajectory = torch.load(TRAJECTORY / 'ordinary_sgd_adamw' / 'checkpoint_90.pt', map_location='cpu', weights_only=False)
    if len(trajectory['logs']) != UPDATES or any(row['sample'] != control_rows[i] for i, row in enumerate(trajectory['logs'])):
        raise RuntimeError('Executed IID logs do not match the frozen control manifest')
    iid = _iid_clean28()
    reference = run.read(TRAJECTORY / 'baseline_0_chemistry_one_variant.json')['objectives']
    reference.update(run.read(TRAJECTORY / 'baseline_0_mrks.json')['objectives'])
    if reference != BASELINE:
        raise RuntimeError('Frozen P536 scientific reference differs from the historical receipts')
    bundle = run.PublicationDataset(run.DATA)
    try:
        if bundle.manifest['logical_sha256'] != DATA_SHA:
            raise RuntimeError('Dataset logical SHA differs')
        rows = list(bundle.reactions.values())
        relchem_population(rows)
        manifest = build_manifest(rows, control_rows)
        validate_manifest(manifest, bundle.reactions, bundle.systems, control_rows)
        model, _shadow = run.model_at(protocol['initial_state'])
        if run.existing.digest(model) != INITIAL_SHA:
            raise RuntimeError('Loaded P536 tensor SHA differs')
        order = [{'name': name, 'shape': list(value.shape), 'numel': int(value.numel())}
                 for name, value in run.existing.named_trainable_parameters(model).items()]
        saved_order = run.read(CONTROL / 'ordinary_sgd_adamw' / 'parameter_order.json')
        if order != saved_order or sum(item['numel'] for item in order) != 9446:
            raise RuntimeError('Trainable parameter order differs from the IID control')
    finally:
        bundle.close()
    if (OUTPUT / 'protocol.json').exists():
        existing_protocol = run.read(OUTPUT / 'protocol.json')
        if 'initial_state' not in existing_protocol or 'adamw' not in existing_protocol:
            raise RuntimeError('Experiment protocol is missing the trainer initialization fields')
        if run.read(OUTPUT / 'sampling_manifest.json') != manifest:
            raise RuntimeError('Experiment directory already exists with a different manifest')
        return manifest
    if OUTPUT.exists():
        raise RuntimeError('Partial experiment directory exists; refusing to overwrite it')
    OUTPUT.mkdir(parents=True)
    run.write(OUTPUT / 'sampling_manifest.json', manifest)
    manifest_sha = run.file_sha256(OUTPUT / 'sampling_manifest.json')
    calibration_path = OUTPUT / 'calibration.json'
    run.write(calibration_path, reused_calibration(calibration, manifest_sha, CONTROL_CALIBRATION_SHA))
    if run.read(calibration_path)['manifest_sha256'] != manifest_sha:
        raise RuntimeError('Rebound calibration is not bound to the new manifest')
    shutil.copyfile(TRAJECTORY / 'evaluation_manifest.json', OUTPUT / 'evaluation_manifest.json')
    if run.file_sha256(OUTPUT / 'evaluation_manifest.json') != EVAL_MANIFEST_SHA:
        raise RuntimeError('Copied evaluation manifest hash differs')
    receipt = {
        'experiment': 'dbstrat-importance-v1', 'arm': 'B', 'updates': UPDATES,
        'initial_state': protocol['initial_state'], 'dataset_sha256': DATA_SHA,
        'adamw': protocol['adamw'], 'lr': 1e-4, 'scheduler': 'constant', 'exc_chunk_size': 4096,
        'lambdas': LAMBDAS, 'seed': DB_STRAT_SEED,
        'sampler': 'uniform database, then uniform identity, then uniform variant, with replacement',
        'importance_weight': '8*n_d/251 applied once to relchem only',
        'control_manifest_sha256': CONTROL_MANIFEST_SHA,
        'new_manifest_sha256': manifest_sha,
        'historical_calibration_sha256': CONTROL_CALIBRATION_SHA,
        'new_calibration_sha256': run.file_sha256(calibration_path),
        'evaluation_manifest_sha256': EVAL_MANIFEST_SHA,
        'source_hashes': _source_hashes(), 'protected_hashes': _protected_hashes(),
        'code_revision': _git_revision(), 'iid_clean28': iid,
        'scientific_baseline': BASELINE, 'best_known_clean28': BEST_KNOWN_CLEAN28,
        'control_retrained': False,
    }
    run.write(OUTPUT / 'protocol.json', receipt)
    return manifest


def _manual_joint(raw, coefficients):
    names = tuple(raw['relchem'])
    return {name: sum(raw[task][name].double() * coefficients[task] for task in TASKS) for name in names}


def preflight():
    prepare()
    receipt_path = OUTPUT / 'preflight.json'
    if receipt_path.exists() and run.read(receipt_path).get('passed'):
        print('PREFLIGHT_ALREADY_PASSED', flush=True)
        return
    if (OUTPUT / 'ordinary_sgd_adamw' / 'latest.pt').exists():
        raise RuntimeError('Refusing preflight after an optimizer update exists')
    protocol = run.read(CONTROL / 'protocol.json')
    calibration = run.read(OUTPUT / 'calibration.json')
    coefficients = {task: calibration['lambda'][task] for task in TASKS}
    manifest = run.read(OUTPUT / 'sampling_manifest.json')
    model, shadow = run.model_at(protocol['initial_state'])
    before = run.existing.digest(model)
    if before != INITIAL_SHA:
        raise RuntimeError('Preflight model is not corrected P536')
    parameters = run.existing.named_trainable_parameters(model)
    optimizer = torch.optim.AdamW(parameters.values(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8,
                                  weight_decay=0.01, foreach=False)
    control_groups = torch.load(CONTROL / 'ordinary_sgd_adamw' / 'checkpoint_0.pt', map_location='cpu', weights_only=False)
    if control_groups['cursor'] != 0 or control_groups['optimizer']['state']:
        raise RuntimeError('Control t0 checkpoint is not a fresh AdamW initialization')
    expected = control_groups['optimizer']['param_groups'][0]
    actual = optimizer.param_groups[0]
    for key in ('lr', 'betas', 'eps', 'weight_decay', 'foreach'):
        if tuple(actual[key]) != tuple(expected[key]) if key == 'betas' else actual[key] != expected[key]:
            raise RuntimeError(f'AdamW initialization differs: {key}')
    if optimizer.state:
        raise RuntimeError('Fresh AdamW moments are not empty')
    bundle = run.PublicationDataset(run.DATA)
    try:
        dispersion = bundle.chemistry_dispersions()
        parity = run.parity(model, shadow, bundle, manifest[0], dispersion)
        record, raw = run.measure(model, shadow, bundle, manifest[0], dispersion,
                                  run.read(run.DATA / 'mrks' / 'dispersion.json'), 4096)
        if record.get('importance_applied_once'):
            raise RuntimeError('Qualified measure applied the importance weight')
        corrected, info = apply_importance(raw, manifest[0], coefficients)
        joint = weighted_gradient(corrected, coefficients)
        manual = _manual_joint(corrected, coefficients)
        if tuple(joint) != tuple(parameters) or any(not torch.equal(joint[name], manual[name]) for name in joint):
            raise RuntimeError('Manual F64 scalarization differs from weighted_gradient')
        for name, value in parameters.items():
            cast = joint[name].to(value.dtype)
            if cast.dtype != torch.float32 or joint[name].shape != value.shape or not torch.equal(cast, joint[name].to(torch.float32)):
                raise RuntimeError('Optimizer-boundary cast differs')
            if value.grad is not None:
                raise RuntimeError('Preflight assigned an optimizer gradient')
        if run.existing.digest(model) != before:
            raise RuntimeError('Preflight changed the model')
        evaluator = (run.ROOT / 'tools' / 'evaluate_microbatch_endpoint.py').read_text(encoding='utf-8')
        if 'importance_weight' in evaluator or 'def measure' in evaluator:
            raise RuntimeError('Evaluator contains an importance-correction path')
    finally:
        bundle.close()
    del optimizer
    run.write(receipt_path, {
        'passed': True, 'optimizer_updated': False, 'parity': parity, 'first_sample': manifest[0],
        'gradient': info, 'model_sha256': before, 'protected_hashes': _protected_hashes(),
    })
    print('PREFLIGHT_PASS', info['importance_weight'], info['relchem_gradient_norm_raw'], flush=True)


def _save_checkpoint(cursor):
    arm = OUTPUT / 'ordinary_sgd_adamw'
    target = arm / f'checkpoint_{cursor}.pt'
    latest = arm / 'latest.pt'
    current = torch.load(latest, map_location='cpu', weights_only=False)['cursor']
    if current == cursor and not target.exists():
        shutil.copyfile(latest, target)
    if current != cursor and not target.exists():
        raise RuntimeError(f'Safe pause at {current}; no changed-setting retry')
    saved = torch.load(target, map_location='cpu', weights_only=False)
    if saved['cursor'] != cursor or saved['scheduler'] is not None:
        raise RuntimeError('Copied checkpoint does not match the native latest state')
    return saved


def train():
    prepare()
    if not (OUTPUT / 'preflight.json').exists() or not run.read(OUTPUT / 'preflight.json').get('passed'):
        raise RuntimeError('Preflight must pass before the first optimizer update')
    protected = _protected_hashes()
    if protected != run.read(OUTPUT / 'protocol.json')['protected_hashes']:
        raise RuntimeError('A protected control artifact changed before training')
    calibration = run.read(OUTPUT / 'calibration.json')
    coefficients = {task: calibration['lambda'][task] for task in TASKS}
    manifest = run.read(OUTPUT / 'sampling_manifest.json')
    wrapped = make_corrected_measure(run.measure, coefficients)
    for stop in (20, 59, 70, 80, 90):
        latest = OUTPUT / 'ordinary_sgd_adamw' / 'latest.pt'
        current = 0 if not latest.exists() else torch.load(latest, map_location='cpu', weights_only=False)['cursor']
        if current < stop:
            original_measure = run.measure
            run.measure = wrapped
            try:
                run.train(OUTPUT, UPDATES, stop_at=stop, runtime_seconds=7200,
                          learning_rate=1e-4, constant_lr=True, diagnostics=True)
            finally:
                run.measure = original_measure
        _save_checkpoint(stop)
    initial_path = OUTPUT / 'ordinary_sgd_adamw' / 'checkpoint_0.pt'
    if not initial_path.exists():
        raise RuntimeError('Trainer did not save the native t0 checkpoint')
    saved = torch.load(OUTPUT / 'ordinary_sgd_adamw' / 'checkpoint_90.pt', map_location='cpu', weights_only=False)
    if saved['cursor'] != UPDATES or len(saved['logs']) != UPDATES:
        raise RuntimeError('Training did not complete exactly 90 updates')
    for index, record in enumerate(saved['logs']):
        if not record.get('importance_applied_once') or record['learning_rate'] != 1e-4:
            raise RuntimeError('Update missing the single importance correction or the frozen learning rate')
        if record['sample']['ae17'] != manifest[index]['ae17'] or record['sample']['mrks_id'] != manifest[index]['mrks_id']:
            raise RuntimeError('Training consumed a different AE17 or mRKS sample')
        if abs(record['joint_gradient_norm_corrected'] - record['weighted_gradient_norm']) > 1e-9:
            raise RuntimeError('Logged joint norm differs from the corrected scalarization')
    initial = torch.load(OUTPUT / 'ordinary_sgd_adamw' / 'checkpoint_0.pt', map_location='cpu', weights_only=False)
    if initial['cursor'] != 0 or initial['logs'] or initial['optimizer']['state'] or initial['scheduler'] is not None:
        raise RuntimeError('t0 checkpoint is not a fresh optimizer state')
    model, _shadow = run.model_at(run.read(CONTROL / 'protocol.json')['initial_state'])
    model.load_state_dict(initial['model'])
    if run.existing.digest(model) != INITIAL_SHA:
        raise RuntimeError('t0 model tensor SHA differs from corrected P536')
    if _protected_hashes() != protected:
        raise RuntimeError('Training changed a protected control artifact')
    print('TRAIN_COMPLETE 90 updates', flush=True)


def _endpoint_path(cursor, stage):
    suffix = '_one_variant' if stage == 'chemistry' else ''
    return OUTPUT / f'endpoint_{cursor}_{stage}{suffix}.json'


def _endpoint_done(stage, current):
    if stage == 'validation':
        return 'metrics' in current
    return bool(current.get('complete'))


def _run_endpoint(cursor, stage):
    path = _endpoint_path(cursor, stage)
    previous = -1
    while True:
        checkpoint_sha = run.file_sha256(OUTPUT / 'ordinary_sgd_adamw' / f'checkpoint_{cursor}.pt')
        if path.exists():
            current = run.read(path)
            if _endpoint_done(stage, current) and current.get('checkpoint_sha256') == checkpoint_sha:
                return current
        evaluate(OUTPUT, cursor, stage, 3600)
        current = run.read(path)
        if _endpoint_done(stage, current):
            return current
        progress = len(current.get('rows', {}))
        if progress <= previous:
            raise RuntimeError(f'Endpoint {stage} at t{cursor} made no progress')
        previous = progress


def _reuse_t0_validation():
    checkpoint = OUTPUT / 'ordinary_sgd_adamw' / 'checkpoint_0.pt'
    model, _shadow = run.model_at(run.read(CONTROL / 'protocol.json')['initial_state'])
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'])
    if run.existing.digest(model) != INITIAL_SHA:
        raise RuntimeError('Cannot reuse t0 validation: model tensor differs')
    historical = run.read(TRAJECTORY / 'baseline_0_validation.json')
    if historical['metrics']['clean28'] != _iid_clean28()[0]:
        raise RuntimeError('Historical t0 Clean28 receipt is inconsistent')
    path = _endpoint_path(0, 'validation')
    if path.exists():
        return
    receipt = copy.deepcopy(historical)
    receipt['checkpoint_sha256'] = run.file_sha256(checkpoint)
    receipt['reused_from'] = {
        'artifact': str(TRAJECTORY / 'baseline_0_validation.json'),
        'reason': 'identical corrected P536 tensor and frozen validation protocol',
        'historical_checkpoint_sha256': historical['checkpoint_sha256'],
    }
    run.write(path, receipt)


def evaluate_experiment():
    if not (OUTPUT / 'ordinary_sgd_adamw' / 'checkpoint_90.pt').exists():
        raise RuntimeError('Training checkpoints are missing')
    if run.measure.__name__ != 'measure':
        raise RuntimeError('Importance patch is still installed during evaluation')
    _reuse_t0_validation()
    validations = {cursor: _run_endpoint(cursor, 'validation') for cursor in CHECKPOINTS}
    best = min(CHECKPOINTS, key=lambda cursor: validations[cursor]['metrics']['clean28'])
    science_cursors = [best] if best == 90 else [best, 90]
    science = {cursor: {'chemistry': _run_endpoint(cursor, 'chemistry'), 'mrks': _run_endpoint(cursor, 'mrks')}
               for cursor in science_cursors}
    run.write(OUTPUT / 'evaluation_selection.json', {'best_clean28_cursor': best, 'science_cursors': science_cursors})
    print('EVALUATION_COMPLETE', best, flush=True)
    return validations, science


def _reaction_catalog():
    bundle = run.PublicationDataset(run.DATA)
    try:
        return {row['source_id']: row for row in bundle.validation_reactions.values()}
    finally:
        bundle.close()


def enrich_validation(metrics, catalog):
    clean = [row for row in metrics['reaction_rows'] if row['clean']]
    if len(clean) != 28:
        raise ValueError('Clean28 panel must contain 28 reactions')
    rows = []
    for row in metrics['reaction_rows']:
        source = catalog[row['reaction_id']]
        signed = row['signed_error_kcal_mol']
        absolute = abs(signed)
        weight = source['diet_weight']
        contribution = row['weighted_absolute_error']
        if absolute and abs(contribution - absolute * weight) > 1e-8:
            raise ValueError('Diet weight does not reconstruct the weighted absolute error')
        enriched = dict(row)
        enriched.update(absolute_error_kcal_mol=absolute, diet_weight=weight,
                        reference_energy_kcal_mol=source['reference_energy_kcal_mol'],
                        predicted_energy_kcal_mol=source['reference_energy_kcal_mol'] + signed,
                        score_contribution=contribution / 28 if row['clean'] else None)
        rows.append(enriched)
    score = float(np.mean([row['weighted_absolute_error'] for row in clean]))
    pieces = [row['score_contribution'] for row in rows if row['clean']]
    if abs(score - metrics['clean28']) > 1e-12 or abs(sum(pieces) - metrics['clean28']) > 1e-12:
        raise ValueError('Clean28 does not reconstruct from preserved contributions')
    return {'clean28': metrics['clean28'], 'full30_diagnostic': metrics['full30'],
            'full30_selection_allowed': False, 'reactions': rows}


def classify(best_clean28, iid_at_best, eligible_best, beats_known):
    """One trajectory cannot prove a better sampler. GO still requires eligibility."""
    if iid_at_best is None:
        return 'NO-GO'
    improved = best_clean28 < iid_at_best
    material = best_clean28 <= iid_at_best - 0.05
    if eligible_best and material and beats_known:
        return 'GO'
    if improved:
        return 'PARTIAL'
    return 'NO-GO'


def _next_experiment(label):
    if label == 'GO':
        return ('One independent replication from the same corrected P536, with a new frozen '
                'DB_STRAT seed, the same 90-update budget, coefficients, and AE17/mRKS streams.')
    if label == 'PARTIAL':
        return ('One independent DB_STRAT seed from the same P536, stopped at the cursor where '
                'this trajectory had its best Clean28, with one fixed-variant/full90 audit.')
    return ('No further database-stratified training. One IID AdamW seed from the same P536, '
            '90 updates at LR 1e-4, to test whether the late Clean28 rebound is stream-specific.')


def _science(cursor):
    chemistry = run.read(_endpoint_path(cursor, 'chemistry'))
    mrks = run.read(_endpoint_path(cursor, 'mrks'))
    if not chemistry.get('complete') or not mrks.get('complete'):
        raise RuntimeError(f'Scientific endpoint at t{cursor} is incomplete')
    if chemistry.get('evaluation_manifest_sha256') != EVAL_MANIFEST_SHA:
        raise RuntimeError('Scientific endpoint used a different variant manifest')
    objectives = {**chemistry['objectives'], **mrks['objectives']}
    ratios = {task: objectives[task] / BASELINE[task] for task in TASKS}
    return {'objectives': objectives, 'ratios': ratios, 'eligible': bool(eligible(objectives, BASELINE))}


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    protocol = run.read(OUTPUT / 'protocol.json')
    manifest = run.read(OUTPUT / 'sampling_manifest.json')
    saved = torch.load(OUTPUT / 'ordinary_sgd_adamw' / 'checkpoint_90.pt', map_location='cpu', weights_only=False)
    control = torch.load(TRAJECTORY / 'ordinary_sgd_adamw' / 'checkpoint_90.pt', map_location='cpu', weights_only=False)
    catalog = _reaction_catalog()
    iid = _iid_clean28()
    validations = {}
    for cursor in CHECKPOINTS:
        metrics = run.read(_endpoint_path(cursor, 'validation'))['metrics']
        validations[cursor] = enrich_validation(metrics, catalog)
    best = min(CHECKPOINTS, key=lambda cursor: validations[cursor]['clean28'])
    selection = run.read(OUTPUT / 'evaluation_selection.json')
    if selection['best_clean28_cursor'] != best:
        raise RuntimeError('Scientific selection cursor does not match the lowest Clean28')
    science = {cursor: _science(cursor) for cursor in selection['science_cursors']}
    best_science = science[best]
    if best in iid:
        label = classify(validations[best]['clean28'], iid[best], best_science['eligible'],
                         validations[best]['clean28'] < BEST_KNOWN_CLEAN28)
    elif any(validations[cursor]['clean28'] < iid[cursor] for cursor in iid):
        label = 'PARTIAL'
    else:
        label = 'NO-GO'
    counts = Counter(row['relchem']['database'] for row in manifest)
    weights = [row['relchem']['importance_numerator'] / row['relchem']['importance_denominator'] for row in manifest]
    updates = []
    for index, record in enumerate(saved['logs']):
        other = control['logs'][index]
        updates.append({
            'cursor': index + 1, 'database': record['sample']['relchem']['database'],
            'identity': record['sample']['relchem']['identity'],
            'importance_weight': record['importance_weight'],
            'relchem_norm_raw': record['relchem_gradient_norm_raw'],
            'relchem_norm_corrected': record['relchem_gradient_norm_corrected'],
            'joint_norm_uncorrected': record['joint_gradient_norm_uncorrected'],
            'joint_norm_corrected': record['joint_gradient_norm_corrected'],
            'weighted_gradient_norm': record['weighted_gradient_norm'],
            'step_norm': record['step_norm'],
            'displacement_from_t0': record['parameter_displacement_from_t0'],
            'control_database': other['sample']['relchem']['database'],
            'control_relchem_norm': other['norms']['relchem'],
            'control_joint_norm': other['weighted_gradient_norm'],
            'control_step_norm': other['step_norm'],
            'control_displacement_from_t0': other['parameter_displacement_from_t0'],
            'total_seconds': record['total_seconds'],
            'peak_allocated_bytes': record['peak_allocated_bytes'],
            'peak_reserved_bytes': record['peak_reserved_bytes'],
        })
    seconds = float(sum(row['total_seconds'] for row in updates))
    peak = max(row['peak_allocated_bytes'] for row in updates) / 1024**3
    reserved = max(row['peak_reserved_bytes'] for row in updates) / 1024**3
    figure = plt.figure(figsize=(7.2, 4.2))
    axis = figure.add_subplot(111)
    known = [cursor for cursor in CHECKPOINTS if cursor in iid]
    axis.plot(known, [iid[cursor] for cursor in known], marker='o', label='IID control')
    axis.plot(list(CHECKPOINTS), [validations[cursor]['clean28'] for cursor in CHECKPOINTS],
              marker='s', label='DB-stratified')
    axis.set_xlabel('Optimizer update')
    axis.set_ylabel('Clean28 (kcal/mol)')
    axis.set_title('Clean28 trajectory')
    axis.legend()
    figure.tight_layout()
    figure.savefig(run.ROOT / 'lap_dbstrat_importance_clean28.png', dpi=140)
    plt.close(figure)
    payload = {
        'decision': label, 'best_cursor': best, 'best_clean28': validations[best]['clean28'],
        'iid_clean28': iid, 'validations': {str(cursor): validations[cursor] for cursor in CHECKPOINTS},
        'science': {str(cursor): science[cursor] for cursor in science},
        'baseline': BASELINE, 'best_known_clean28': BEST_KNOWN_CLEAN28,
        'database_counts': dict(counts), 'expected_database_count': UPDATES / N_DATABASES,
        'importance_weights': {'min': min(weights), 'max': max(weights)},
        'updates': updates, 'logged_training_seconds': seconds,
        'peak_allocated_gib': peak, 'peak_reserved_gib': reserved,
        'new_optimizer_updates': UPDATES, 'protocol': protocol,
        'checkpoint_sha256': {str(cursor): run.file_sha256(OUTPUT / 'ordinary_sgd_adamw' / f'checkpoint_{cursor}.pt')
                              for cursor in CHECKPOINTS},
        'next_experiment': _next_experiment(label),
        'limitations': [
            'One stochastic trajectory is not proof of a superior sampler.',
            'Cursor-aligned IID and DB-stratified samples are different reactions.',
            'An unbiased singleton correction does not imply the same AdamW trajectory.',
            'The fixed-panel relchem endpoint is not automatically the singleton-gradient expectation.',
            'No repeated-sample variance estimate was measured.',
        ],
    }
    run.write(run.ROOT / 'lap_dbstrat_importance_metrics.json', payload)
    run.write(OUTPUT / 'provenance.json', payload['protocol'])
    _write_markdown(payload)
    print('REPORT', label, best, validations[best]['clean28'], flush=True)
    return payload


def _fmt(value):
    return f'{value:.9f}'


def _write_markdown(payload):
    iid = payload['iid_clean28']
    validations = {int(key): value for key, value in payload['validations'].items()}
    best = payload['best_cursor']
    lines = [
        '# Database-stratified importance sampling versus IID AdamW',
        '',
        f"Decision: **{payload['decision']}**.",
        '',
        ('Arm B ran 90 new AdamW updates from corrected P536. Arm A was the preserved IID LR=1e-4 trajectory and was not retrained. '
         'The relchem draw is uniform over eight databases, then uniform over identities in the drawn database. '
         'The qualified singleton relchem gradient is multiplied once by `8*n_d/251`. AE17, Exc, operator, coefficients, and AdamW settings are unchanged.'),
        '',
        '## Sampling integrity',
        '',
        'Database counts in the 90 frozen stratified updates, against the uniform-database expectation 11.25:',
        '',
        '| Database | Identities | Sampled | Expected | Importance weight |',
        '|---|---:|---:|---:|---:|',
    ]
    for database, size in EXPECTED_COUNTS.items():
        lines.append(f"| {database} | {size} | {payload['database_counts'].get(database, 0)} | 11.25 | {N_DATABASES * size / POPULATION:.6f} |")
    raw = np.array([row['relchem_norm_raw'] for row in payload['updates']])
    corrected = np.array([row['relchem_norm_corrected'] for row in payload['updates']])
    joint = np.array([row['joint_norm_corrected'] for row in payload['updates']])
    joint_raw = np.array([row['joint_norm_uncorrected'] for row in payload['updates']])
    step = np.array([row['step_norm'] for row in payload['updates']])
    control_step = np.array([row['control_step_norm'] for row in payload['updates']])
    lines.extend([
        '',
        f"Importance weights used on the trajectory span {_fmt(payload['importance_weights']['min'])} to {_fmt(payload['importance_weights']['max'])}.",
        f"Median raw relchem norm { _fmt(float(np.median(raw))) }, median corrected relchem norm {_fmt(float(np.median(corrected)))}.",
        f"Median joint norm before correction {_fmt(float(np.median(joint_raw)))}, after correction {_fmt(float(np.median(joint)))}.",
        f"Median native AdamW step norm {_fmt(float(np.median(step)))}; cursor-aligned IID control median {_fmt(float(np.median(control_step)))}.",
        'These cursor-aligned norms are not paired samples. The correction changes gradient scale; AdamW moments do not preserve that scale as a proportional step.',
        '',
        '## Clean28',
        '',
        'Clean28 is the mean Diet-weighted absolute error of the 28 leakage-clean reactions, in kcal/mol. It is not full30 WTMAD-2. Full30 was not used for selection.',
        '',
        '| Cursor | Old IID Clean28 | New DB-strat Clean28 | Difference |',
        '|---|---:|---:|---:|',
    ])
    for cursor in CHECKPOINTS:
        new = validations[cursor]['clean28']
        if cursor not in iid:
            lines.append(f'| {cursor} | Not yet established | {_fmt(new)} | — |')
        else:
            lines.append(f'| {cursor} | {_fmt(iid[cursor])} | {_fmt(new)} | {_fmt(new - iid[cursor])} |')
    lines.extend(['', 'A negative difference favors the new sampler. The historical IID t20 value is not established and is not imputed.', ''])
    best_rows = {row['reaction_id']: row for row in validations[best]['reactions'] if row['clean']}
    if best in iid:
        control_name = IID_CLEAN28_FILES[best]
        control_metrics = enrich_validation(run.read(TRAJECTORY / control_name)['metrics'], _reaction_catalog())
        control_rows = {row['reaction_id']: row for row in control_metrics['reactions'] if row['clean']}
        deltas = sorted(((best_rows[key]['score_contribution'] - control_rows[key]['score_contribution'], key)
                         for key in best_rows), key=lambda item: item[0])
        improved = sum(delta < 0 for delta, _key in deltas)
        lines.append(f'At t{best}, {improved} of 28 clean reactions have a lower score contribution than IID at the same cursor.')
        lines.append('Largest contribution decreases versus that IID checkpoint:')
        for delta, key in deltas[:5]:
            lines.append(f'- {key}: {delta:+.9f}')
        lines.append('Largest contribution increases:')
        for delta, key in deltas[-5:]:
            lines.append(f'- {key}: {delta:+.9f}')
        lines.append('')
    lines.extend(['## Scientific objectives', ''])
    lines.append('| Checkpoint | relchem/t0 | AE17/t0 | Exc/t0 | operator/t0 | Eligible |')
    lines.append('|---|---:|---:|---:|---:|---|')
    for cursor, result in payload['science'].items():
        ratios = result['ratios']
        lines.append('| t{} | {} | {} | {} | {} | {} |'.format(
            cursor, _fmt(ratios['relchem']), _fmt(ratios['ae17']), _fmt(ratios['exc']),
            _fmt(ratios['op']), 'yes' if result['eligible'] else 'no'))
    lines.extend([
        '',
        f"Reference t0 objectives remain relchem {BASELINE['relchem']}, AE17 {BASELINE['ae17']}, Exc {BASELINE['exc']}, operator {BASELINE['op']}.",
        'Eligibility requires every finite ratio to be strictly below 1. A lower Clean28 without eligibility is not a promoted functional.',
        '',
        '## Runtime',
        '',
        (f"Logged synchronized update time: {payload['logged_training_seconds']:.3f}s over {payload['new_optimizer_updates']} new updates. "
         f"Peak live CUDA {payload['peak_allocated_gib']:.3f} GiB; peak reserved {payload['peak_reserved_gib']:.3f} GiB."),
        '',
        '## Interpretation',
        '',
    ])
    for note in payload['limitations']:
        lines.append(f'- {note}')
    lines.extend([
        '',
        f"Recommended next experiment, not executed: {payload['next_experiment']}",
        '',
        f"Code revision at execution: `{payload['protocol']['code_revision']}`.",
        f"Initial tensor SHA256: `{INITIAL_SHA}`.",
        f"Dataset logical SHA256: `{DATA_SHA}`.",
        f"New manifest SHA256: `{payload['protocol']['new_manifest_sha256']}`.",
        f"Historical calibration SHA256: `{CONTROL_CALIBRATION_SHA}`.",
        '',
    ])
    for cursor, sha in payload['checkpoint_sha256'].items():
        lines.append(f'- t{cursor} checkpoint SHA256: `{sha}`')
    lines.append('')
    (run.ROOT / 'lap_dbstrat_importance_report.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'preflight', 'train', 'evaluate', 'report'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.stage == 'prepare':
        prepare()
        print('PREPARE_PASS', flush=True)
    elif args.stage == 'preflight':
        preflight()
    elif args.stage == 'train':
        train()
    elif args.stage == 'evaluate':
        evaluate_experiment()
    else:
        report()
