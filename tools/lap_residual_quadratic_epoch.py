"""One 251-update residual-aware relchem arm matched to historical J251.

The only training change is the singleton relchem gradient multiplier. AE17,
Exc, and the AO operator keep their qualified gradients. J251 is not retrained.
"""
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO.parent / 'lap_residual_quadratic_epoch_20261009'
PROTOCOL_ID = 'residual-quadratic-j251-v1'
EPS = 1e-20
C = 0.5
BUDGET_SECONDS = 100 * 60
PARITY_INDICES = (0, 125, 250)
MILESTONES = (90, 175, 251)
HISTORICAL_BEST_CLEAN28 = 8.619694171538775
BREAKTHROUGH_CLEAN28 = 7.0
BASELINE = {
    'relchem': 1.2354710978866068,
    'ae17': 24.901047104016875,
    'exc': 92.17480502000551,
    'op': 0.03314821681605566,
}
LAMBDAS = {
    'relchem': 0.017015480965588553,
    'ae17': 0.00005141254618347414,
    'exc': 0.000015094644512009712,
    'op': 0.33597561607048215,
}
TASKS = ('relchem', 'ae17', 'exc', 'op')
MANIFEST_SHA = '72b827655ce4421f2fc933c082cda1c2dc3e5d9903ec29e0c5e3d62e82ac37f6'
EVAL_SHA = '132bd550161be6bc6761f8df1de8d3755c0b0debab53663208c3a9bbf56cb805'
CALIBRATION_SHA = '4d97df1ec78aa01a2c9380a86a323b2f0b46828dfc5975f5e3a63c75f4440096'
INITIAL_FILE_SHA = '0ca0f77367214c171e6edf0576bdb043d64479245ad1a0e539fa11fb0cee542d'
INITIAL_TENSOR_SHA = '3c2e45d6a86cccf80d7b4dc458e67f9b406935ed60d4e57433bb441a6df88da6'
DATA_SHA = '61c221a19b9987717e69cac182ad545241f8807db4126c0949a99992e4c210ef'
P536_CLEAN28 = 9.553190635871177
J251_CLEAN28 = 9.346760658714116
J251_RATIOS = {
    'relchem': 0.979707731825112,
    'ae17': 0.08101214513606354,
    'exc': 0.10110475652753241,
    'op': 0.9381212072698767,
}
B80_RATIOS = {
    'relchem': 1.0348695780913695,
    'ae17': 0.08072795193861575,
    'exc': 0.05905810287343182,
    'op': 0.9440697052032282,
}
SOURCE_HASHES = {
    'train_lap_microbatch.py': '2e837c3a88ca3d4737c3c3397dcfbb98adeb7f6917adc8c693cecf205dc37e9e',
    'train_models/lap_fixed_adamw.py': '9effc074bd18236ad7af4c4e05fa04be1754d35e228b9a0567163c4b0bd5eb0a',
    'train_models/lap_chemistry_sampling.py': '313f90bcd0fe438f3e82ca00e7625c7bee502797b772532fef16adb2adad9fad',
    'train_models/lap_moo_training.py': 'e3dafbe76279e13b73f2a6dd5f2547294902387d21fc1b78b4ce917430495113',
    'train_models/lap_training.py': 'ca789703e536f9311178ada07fd41d92ba7f79afd35b25ad240647747b3e9b77',
    'train_models/lap_vxc.py': 'ca22ac54ae9ffe4d2593f1e7072b11210277b0a321701d5240f564803711c287',
    'train_models/lap_operator.py': '7fae09c0857187875377fbf5c1c7c8d19b1afc869942310f75f5462f7bd2e864',
    'tools/relchem_joint_epoch.py': '3ed79ce4591d62d6a6c22c4b9da28a932fcf13af926d95c4e843a94ee0d71c00',
    'tools/evaluate_microbatch_endpoint.py': 'b92a00bc8f51149e29be846bbadbedc74c47fd172b975f1fd0888fd4caf6da3f',
    'train_models/NN_models_lap.py': '862f1a0989b188a9543a49947550521541c7012179ebda0929c92ed824c09cb0',
    'train_models/optuna_joint.py': '59c66d798ed7fd74b05caba271b590cca138d53336114edc820b25c9187846f5',
    'train_models/reaction_energy_calculation.py': 'd88424489fb020d6f71c78f6eaae8913b3f8601e60c370252d55e86bcab70003',
    'train_models/publication_data/loader.py': '27415bd43f68e64467eec21b1ddf03ac1fc540d93c077483b1585f547579bb6f',
    'train_models/publication_data/contracts.py': '47e6c5f1fb7a7889320adbfa4c969126fbeb67bc58095450907dfe8aac6b87cd',
    'lap_symmetric_moo_trajectory_arena.py': 'ff25e840b1268715c5f3ef0abaa3c3d129750d4fa3e5766dfb93fd70dd59bc7e',
}
GATE = 'NOT EVALUATED — ACCURACY GATE'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def loss_old(residual, factor, eps=EPS):
    if factor <= 0 or not math.isfinite(factor):
        raise ValueError('Database factor must be finite and positive')
    return factor * math.sqrt(residual * residual + eps)


def loss_new(residual, factor, scale, eps=EPS):
    if scale <= 0 or not math.isfinite(scale):
        raise ValueError('Residual scale must be finite and positive')
    root = math.sqrt(residual * residual + eps)
    return C * factor * (root + (residual * residual) / (2.0 * scale))


def absolute_residual(loss, factor, eps=EPS):
    if factor <= 0 or not math.isfinite(factor) or not math.isfinite(loss):
        raise ValueError('Loss and database factor must be finite')
    return math.sqrt(max((loss / factor) ** 2 - eps, 0.0))


def surrogate_loss(loss, factor, scale, eps=EPS):
    if scale <= 0 or factor <= 0:
        raise ValueError('Scale and database factor must be positive')
    return C * (loss + (loss * loss / factor - factor * eps) / (2.0 * scale))


def gradient_factor(loss, factor, scale):
    if scale <= 0 or factor <= 0 or not math.isfinite(loss):
        raise ValueError('Factor inputs must be finite and positive')
    value = C * (1.0 + loss / (factor * scale))
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Gradient multiplier must be finite and positive')
    return value


def median_scale(residuals):
    values = [float(v) for v in residuals]
    if len(values) != 251 or any(not math.isfinite(v) or v < 0.0 for v in values):
        raise ValueError('Scale calibration requires 251 finite nonnegative residuals')
    scale = sorted(values)[125]
    if scale != statistics.median(values) or not math.isfinite(scale) or scale <= 0.0:
        raise ValueError('Residual scale must be the positive middle residual')
    return float(scale)


def scale_relchem(raw, factor):
    multiplier = float(factor)
    if not math.isfinite(multiplier) or multiplier <= 0.0 or 'relchem' not in raw:
        raise ValueError('Relchem scaling requires a positive finite multiplier')
    scaled = {task: dict(gradients) for task, gradients in raw.items()}
    scaled['relchem'] = {name: gradient * multiplier for name, gradient in raw['relchem'].items()}
    return scaled


def validate_manifest(rows):
    if len(rows) != 251:
        raise ValueError('Training manifest must contain 251 entries')
    seen = set()
    for index, row in enumerate(rows):
        if row.get('cursor') != index or not row.get('mrks_id'):
            raise ValueError('Manifest cursor or mRKS id mismatch')
        for task in ('relchem', 'ae17'):
            item = row.get(task) or {}
            if not item.get('identity') or not item.get('variant') or not item.get('database'):
                raise ValueError('Manifest chemistry identity is incomplete')
        if row['ae17']['database'] != 'AE17':
            raise ValueError('AE17 stream database changed')
        seen.add(row['relchem']['identity'])
    if len(seen) != 251:
        raise ValueError('Relchem training identities are not unique')
    return rows


def require_checkpoint(saved, scale, calibration_sha):
    if saved.get('protocol_id') != PROTOCOL_ID:
        raise ValueError('Checkpoint protocol mismatch')
    if saved.get('manifest_sha256') != MANIFEST_SHA:
        raise ValueError('Checkpoint manifest mismatch')
    if saved.get('scale_hex') != float(scale).hex():
        raise ValueError('Checkpoint residual scale mismatch')
    if saved.get('calibration_sha256') != calibration_sha:
        raise ValueError('Checkpoint calibration mismatch')
    if saved.get('lambdas') != LAMBDAS:
        raise ValueError('Checkpoint coefficients mismatch')
    logs = saved.get('logs') or []
    if saved.get('cursor') != len(logs):
        raise ValueError('Checkpoint cursor does not match the log chain')
    for index, row in enumerate(logs):
        if row['cursor'] != index + 1 or row['manifest_index'] != index:
            raise ValueError('Training cursor chain mismatch')
        if index and logs[index - 1]['after_sha256'] != row['before_sha256']:
            raise ValueError('Model hash chain mismatch')
    return saved


def scientifically_eligible(objectives):
    if set(objectives) != set(TASKS):
        return False
    ratios = []
    for task in TASKS:
        value = objectives[task] / BASELINE[task]
        if not math.isfinite(value) or value >= 1.0:
            return False
        ratios.append(value)
    return dict(zip(TASKS, ratios, strict=True))


def beats_historical(clean28):
    return math.isfinite(clean28) and clean28 < HISTORICAL_BEST_CLEAN28


def classify(result):
    updates = result.get('updates', 0)
    milestones = result.get('milestones', {})
    ready = (result.get('parity_passed') and not result.get('numerical_failure')
             and updates == 251 and all(cursor in milestones for cursor in MILESTONES))
    if not ready:
        return 'PARTIAL'
    screened = [cursor for cursor in MILESTONES if beats_historical(milestones[cursor]['clean28'])]
    if not screened:
        return 'NO-GO'
    qualified = result.get('qualified', {})
    if any(cursor not in qualified for cursor in screened):
        return 'PARTIAL'
    eligible = [cursor for cursor in screened if qualified[cursor].get('eligible')]
    if not eligible:
        return 'NO-GO'
    best = min(eligible, key=lambda cursor: (milestones[cursor]['clean28'], cursor))
    if milestones[best]['clean28'] < BREAKTHROUGH_CLEAN28:
        return 'BREAKTHROUGH'
    return 'PROGRESS'


def database_factors():
    sys.path.insert(0, str(REPO / 'train_models'))
    from optuna_joint import FCHEM_DB_WEIGHTS, FREQ_WEIGHTS, MEAN_WEIGHT
    return {
        database: FCHEM_DB_WEIGHTS[database] * FREQ_WEIGHTS[database] / MEAN_WEIGHT
        for database in FCHEM_DB_WEIGHTS
    }


def _load(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding='utf-8'))


def _dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def _jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    lines = [line for line in path.read_text(encoding='utf-8').splitlines() if line]
    rows = []
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
            path.write_text(''.join(f'{item}\n' for item in lines[:-1]), encoding='utf-8')
    return rows


def _append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(row, allow_nan=False) + '\n')


def gpu_seconds():
    total = sum(row['seconds'] for row in _jsonl(OUT / 'calibration_rows.jsonl'))
    scale = _load(OUT / 'scale.json', {})
    total += float(scale.get('overhead_seconds', 0.0))
    preflight = _load(OUT / 'preflight.json', {})
    total += float(preflight.get('gpu_seconds', 0.0))
    status = _load(OUT / 'training_status.json', {})
    total += sum(float(row['seconds']) for row in status.get('logs', []))
    for path in sorted(OUT.glob('validation_*.json')):
        total += float(_load(path, {}).get('gpu_seconds', 0.0))
    total += sum(row['seconds'] for row in _jsonl(OUT / 'qualification_time.jsonl'))
    return float(total)


def verify_sources():
    protocol = _load(REPO / 'relchem_joint_epoch_protocol.json')
    for relative, expected in protocol['source_hashes'].items():
        if sha256_file(REPO / relative) != expected or SOURCE_HASHES[relative] != expected:
            raise SystemExit(f'Scientific source changed: {relative}')
    for relative, expected in SOURCE_HASHES.items():
        if sha256_file(REPO / relative) != expected:
            raise SystemExit(f'Scientific source changed: {relative}')
    if sha256_file(REPO / 'relchem_joint_epoch_sampling_manifest.json') != MANIFEST_SHA:
        raise SystemExit('Training manifest changed')
    if sha256_file(REPO / 'relchem_joint_epoch_evaluation_manifest.json') != EVAL_SHA:
        raise SystemExit('Evaluation manifest changed')
    calibration = Path(r'C:\Dev\readWFN_share_ms\lap_relchem_joint_epoch_20261009\J\calibration.json')
    if sha256_file(calibration) != CALIBRATION_SHA:
        raise SystemExit('J251 calibration file changed')
    loaded = _load(calibration)['lambda']
    if any(loaded[task] != LAMBDAS[task] for task in TASKS):
        raise SystemExit('J251 task coefficients changed')
    initial = protocol['initial_state']
    if (initial['file_sha256'] != INITIAL_FILE_SHA or initial['state_sha256'] != INITIAL_TENSOR_SHA
            or sha256_file(initial['path']) != INITIAL_FILE_SHA or protocol['dataset_sha256'] != DATA_SHA):
        raise SystemExit('Initial checkpoint or dataset provenance changed')
    if protocol['lr'] != 1e-4 or protocol['scheduler'] is not None or protocol['total_updates'] != 251:
        raise SystemExit('J251 optimizer protocol changed')
    return protocol


def execute():
    import random
    import time

    import numpy as np
    import torch

    sys.path.insert(0, str(REPO))

    import train_lap_microbatch as run
    from train_models.lap_fixed_adamw import weighted_gradient

    protocol = verify_sources()
    if OUT.exists() and (OUT / 'protocol.json').exists():
        existing = _load(OUT / 'protocol.json')
        if existing.get('protocol_id') != PROTOCOL_ID:
            raise SystemExit('Scratch directory belongs to a different experiment')
    prior_stop = _load(OUT / 'stop.json', {}).get('reason', '')
    if prior_stop and any(token in prior_stop for token in ('Nonfinite', 'parity', 'Parity', 'mismatch', 'changed')):
        raise SystemExit(prior_stop)
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required for this arm')
    manifest = validate_manifest(_load(REPO / 'relchem_joint_epoch_sampling_manifest.json'))
    factors = database_factors()
    _dump(OUT / 'protocol.json', {
        'protocol_id': PROTOCOL_ID,
        'loss': 'L_new = (a/2) * (sqrt(r^2+eps) + r^2/(2s))',
        'c': C,
        'epsilon': EPS,
        'normalization': (
            'c=0.5 anchors derivative sensitivity at the initial training-median residual. '
            'It does not guarantee identical joint-gradient norms or identical AdamW steps.'
        ),
        'lambdas': LAMBDAS,
        'adamw': protocol['adamw'],
        'lr': protocol['lr'],
        'scheduler': None,
        'total_updates': 251,
        'exc_chunk_size': protocol['exc_chunk_size'],
        'operator_chunk_size': protocol['operator_chunk_size'],
        'chemistry_chunk': protocol['chemistry_chunk'],
        'initial_state': protocol['initial_state'],
        'dataset_sha256': DATA_SHA,
        'sampling_manifest_sha256': MANIFEST_SHA,
        'evaluation_manifest_sha256': EVAL_SHA,
        'calibration_file_sha256': CALIBRATION_SHA,
        'source_hashes': SOURCE_HASHES,
        'j251_retrained': False,
    })

    def note_peak():
        torch.cuda.synchronize()
        memory = _load(OUT / 'memory.json', {'peak_allocated_bytes': 0, 'peak_reserved_bytes': 0})
        memory['peak_allocated_bytes'] = max(memory['peak_allocated_bytes'], torch.cuda.max_memory_allocated())
        memory['peak_reserved_bytes'] = max(memory['peak_reserved_bytes'], torch.cuda.max_memory_reserved())
        _dump(OUT / 'memory.json', memory)
        return memory

    def stop(reason):
        _dump(OUT / 'stop.json', {'reason': reason, 'gpu_seconds': gpu_seconds()})
        raise SystemExit(reason)

    def budget_blocked(extra=0.0):
        return gpu_seconds() + extra >= BUDGET_SECONDS

    scale_path = OUT / 'scale.json'
    if not scale_path.exists():
        if budget_blocked():
            stop('GPU budget exhausted before scale calibration')
        rows = _jsonl(OUT / 'calibration_rows.jsonl')
        if any(row['index'] != index for index, row in enumerate(rows)):
            stop('Calibration receipt is not a prefix of the training manifest')
        bundle = run.PublicationDataset(run.DATA)
        model = shadow = None
        try:
            if bundle.manifest['logical_sha256'] != DATA_SHA:
                stop('Dataset logical SHA changed')
            model, shadow = run.model_at(protocol['initial_state'])
            if run.existing.digest(model) != INITIAL_TENSOR_SHA:
                stop('P536 tensor changed during calibration load')
            dispersion = bundle.chemistry_dispersions()
            for index in range(len(rows), 251):
                if budget_blocked():
                    stop('GPU budget exhausted during scale calibration')
                entry = manifest[index]['relchem']
                if bundle.reactions[entry['identity']]['database'] != entry['database']:
                    stop('Training database label disagrees with the dataset')
                timer = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
                reaction = _load_reaction(run, bundle, 'train_relchem', entry['identity'], entry['variant'])
                with torch.no_grad():
                    loss = float(run.chemistry(model, shadow, reaction, dispersion)())
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - timer
                if not math.isfinite(loss):
                    stop('Nonfinite calibration loss')
                factor = factors[entry['database']]
                residual = absolute_residual(loss, factor)
                _append_jsonl(OUT / 'calibration_rows.jsonl', {
                    'index': index, 'identity': entry['identity'], 'database': entry['database'],
                    'variant': entry['variant'], 'loss_old': loss, 'database_factor': factor,
                    'abs_residual': residual, 'seconds': elapsed,
                })
                note_peak()
                del reaction
            if run.existing.digest(model) != INITIAL_TENSOR_SHA or sha256_file(protocol['initial_state']['path']) != INITIAL_FILE_SHA:
                stop('Calibration changed the initial model')
        finally:
            if model is not None:
                del model, shadow
            bundle.close()
            torch.cuda.empty_cache()
        rows = _jsonl(OUT / 'calibration_rows.jsonl')
        scale = median_scale([row['abs_residual'] for row in rows])
        for row in rows:
            row['gradient_factor'] = gradient_factor(row['loss_old'], row['database_factor'], scale)
        _dump(OUT / 'calibration_residuals.json', {
            'population': 251, 'epsilon': EPS, 's': scale, 's_hex': scale.hex(),
            'input_checkpoint_file_sha256': INITIAL_FILE_SHA,
            'input_tensor_sha256': INITIAL_TENSOR_SHA,
            'training_manifest_sha256': MANIFEST_SHA, 'rows': rows,
        })
        _dump(scale_path, {
            's': scale, 's_hex': scale.hex(), 'overhead_seconds': 0.0,
            'calibration_sha256': sha256_file(OUT / 'calibration_residuals.json'),
            'input_checkpoint_file_sha256': INITIAL_FILE_SHA,
            'training_manifest_sha256': MANIFEST_SHA, 'clean28_used': False,
        })
    scale_info = _load(scale_path)
    scale = float.fromhex(scale_info['s_hex'])
    if scale != median_scale([row['abs_residual'] for row in _load(OUT / 'calibration_residuals.json')['rows']]):
        stop('Frozen residual scale does not match the calibration receipt')

    if _load(OUT / 'preflight.json', {}).get('passed') is False:
        stop('Gradient parity previously failed')
    if not _load(OUT / 'preflight.json', {}).get('passed'):
        if budget_blocked():
            stop('GPU budget exhausted before preflight')
        bundle = run.PublicationDataset(run.DATA)
        model = shadow = None
        timer = time.perf_counter()
        try:
            torch.cuda.reset_peak_memory_stats()
            file_before = sha256_file(protocol['initial_state']['path'])
            model, shadow = run.model_at(protocol['initial_state'])
            digest_before = run.existing.digest(model)
            parameters = run.existing.named_trainable_parameters(model)
            coordinates = sum(parameter.numel() for parameter in parameters.values())
            if (type(model).__name__ != 'pcPBELMLOptimizerV2Lap' or coordinates != 9446
                    or int(model.lap_architecture_version) != 1
                    or any('tau' in name.lower() for name in parameters)):
                stop('Model architecture does not match the qualified P536 pilot')
            dispersion = bundle.chemistry_dispersions()
            checks = []
            retained = {}
            for index in PARITY_INDICES:
                entry = manifest[index]['relchem']
                reaction = _load_reaction(run, bundle, 'train_relchem', entry['identity'], entry['variant'])
                qualified_loss, qualified = run.chemistry(model, shadow, reaction, dispersion).value_and_grad()
                shadow.load_state_dict(model.state_dict(), strict=True)
                names = run.existing.named_trainable_parameters(shadow)
                loss_a = run.existing.make_reaction_objective(
                    shadow, reaction, device='cuda', dtype=torch.float64, dispersions=dispersion)()
                factor = factors[entry['database']]
                loss_direct = C * (loss_a + (loss_a.square() / factor - factor * EPS) / (2.0 * scale))
                leaves = tuple(names.values())
                old = torch.autograd.grad(loss_a, leaves, retain_graph=True, allow_unused=True)
                direct = torch.autograd.grad(loss_direct, leaves, allow_unused=True)
                multiplier = gradient_factor(float(loss_a.detach()), factor, scale)
                packed_old = _pack(names, old)
                packed_direct = _pack(names, direct)
                packed_qualified = _pack(names, qualified)
                reference = float(packed_direct.norm())
                discrepancy = float((packed_direct - multiplier * packed_old).norm())
                relative = discrepancy / reference if reference > 0 else (0.0 if discrepancy == 0 else math.inf)
                qualified_relative = _relative(packed_qualified, packed_old)
                scalar_gap = abs(float(loss_direct.detach()) - surrogate_loss(float(loss_a.detach()), factor, scale))
                if (packed_direct.numel() != 9446 or not torch.isfinite(packed_direct).all()
                        or not torch.isfinite(packed_old).all() or relative > 1e-10
                        or qualified_relative > 1e-10 or scalar_gap > 1e-10
                        or abs(float(loss_a.detach()) - qualified_loss) > 1e-10 * max(1.0, abs(qualified_loss))):
                    _dump(OUT / 'preflight.json', {'passed': False, 'index': index, 'relative_l2': relative,
                                                   'qualified_relative_l2': qualified_relative})
                    stop(f'Gradient parity failed at manifest index {index}')
                checks.append({'index': index, 'identity': entry['identity'], 'variant': entry['variant'],
                               'database': entry['database'], 'coordinates': 9446, 'relative_l2': relative,
                               'qualified_relative_l2': qualified_relative, 'factor': multiplier,
                               'loss_old': float(loss_a.detach()), 'loss_new': float(loss_direct.detach()),
                               'all_finite': True})
                if index == 0:
                    retained = {'qualified': {name: value.detach().clone() for name, value in qualified.items()},
                                'factor': multiplier}
                del reaction, loss_a, loss_direct, old, direct
            entry = manifest[0]
            record, raw = run.measure(model, shadow, bundle, entry, dispersion,
                                      run.read(run.DATA / 'mrks/dispersion.json'), exc_chunk_size=4096)
            multiplier = gradient_factor(record['losses']['relchem'], factors[entry['relchem']['database']], scale)
            if abs(multiplier - retained['factor']) > 1e-8 * max(1.0, abs(multiplier)):
                stop('Joint preflight chemistry factor disagrees with direct parity')
            transformed = scale_relchem(raw, multiplier)
            for task in ('ae17', 'exc', 'op'):
                for name, gradient in raw[task].items():
                    if transformed[task][name] is not gradient:
                        stop('Joint preflight changed a non-relchem gradient')
            joint = weighted_gradient(transformed, LAMBDAS)
            joint_norm = float(torch.cat([value.reshape(-1) for value in joint.values()]).norm())
            relchem_relative = _relative(_pack(run.existing.named_trainable_parameters(model), transformed['relchem']),
                                         multiplier * _pack(run.existing.named_trainable_parameters(model), raw['relchem']))
            qualified_match = _relative(_pack(run.existing.named_trainable_parameters(model), raw['relchem']),
                                        _pack(run.existing.named_trainable_parameters(model), retained['qualified']))
            if (not math.isfinite(joint_norm) or joint_norm <= 0 or relchem_relative > 1e-10
                    or qualified_match > 1e-8
                    or run.existing.digest(model) != digest_before
                    or sha256_file(protocol['initial_state']['path']) != file_before):
                stop('Joint-gradient preflight failed')
            torch.cuda.synchronize()
            note_peak()
            _dump(OUT / 'preflight.json', {
                'passed': True, 'optimizer_step': False, 'gpu_seconds': time.perf_counter() - timer,
                'parity': checks, 'joint_norm': joint_norm, 'joint_factor': multiplier,
                'joint_factor_applications': 1, 'relchem_scale_relative_l2': relchem_relative,
                'measure_vs_qualified_relative_l2': qualified_match,
                'other_tasks_identical': True, 'model_unchanged': True, 'checkpoint_unchanged': True,
                'initial_tensor_sha256': digest_before,
            })
        finally:
            if model is not None:
                del model, shadow
            bundle.close()
            torch.cuda.empty_cache()

    target = OUT / 'ordinary_sgd_adamw'
    target.mkdir(exist_ok=True)
    latest = target / 'latest.pt'
    torch.manual_seed(41)
    np.random.seed(41)
    random.seed(41)
    model, shadow = run.model_at(protocol['initial_state'])
    parameters = run.existing.named_trainable_parameters(model)
    optimizer = torch.optim.AdamW(parameters.values(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8,
                                 weight_decay=0.01, foreach=False)
    logs, cursor = [], 0
    if latest.exists():
        saved = torch.load(latest, map_location='cpu', weights_only=False)
        require_checkpoint(saved, scale, scale_info['calibration_sha256'])
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        run.existing.restore_rng_state(saved['rng'])
        logs, cursor = saved['logs'], saved['cursor']
        if logs and logs[0]['before_sha256'] != INITIAL_TENSOR_SHA:
            stop('Log chain does not start at corrected P536')
        if cursor and {int(state['step']) for state in optimizer.state.values()} != {cursor}:
            stop('Optimizer moment counter does not match the checkpoint cursor')
    else:
        _save(run, latest, model, optimizer, 0, logs, scale, scale_info['calibration_sha256'])
        _save(run, target / 'checkpoint_0.pt', model, optimizer, 0, logs, scale, scale_info['calibration_sha256'])
    bundle = run.PublicationDataset(run.DATA)
    try:
        if bundle.manifest['logical_sha256'] != DATA_SHA:
            stop('Dataset logical SHA changed')
        dispersion = bundle.chemistry_dispersions()
        mrks_dispersion = run.read(run.DATA / 'mrks/dispersion.json')
        for index in range(cursor, 251):
            if budget_blocked(180.0):
                stop('GPU budget cannot cover the remaining updates and Clean28 evaluations')
            in_vram = [row['seconds'] for row in logs if row['peak_reserved_bytes'] <= 16 * 1024 ** 3]
            if len(in_vram) >= 5:
                sample = sorted(in_vram)
                median_time = sample[len(sample) // 2]
                if gpu_seconds() + median_time * (251 - cursor) + 180.0 > BUDGET_SECONDS:
                    stop('Projected runtime exceeds the 100-minute GPU budget')
            entry = manifest[index]
            timer = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            before = run.existing.digest(model)
            if index == 0 and before != INITIAL_TENSOR_SHA:
                stop('First update did not start from corrected P536')
            before_weights = {name: value.detach().clone() for name, value in parameters.items()}
            record, raw = run.measure(model, shadow, bundle, entry, dispersion, mrks_dispersion, exc_chunk_size=4096)
            loss_old_value = float(record['losses']['relchem'])
            factor = factors[entry['relchem']['database']]
            multiplier = gradient_factor(loss_old_value, factor, scale)
            new_loss = surrogate_loss(loss_old_value, factor, scale)
            if not all(math.isfinite(value) for value in (loss_old_value, new_loss, multiplier)):
                stop('Nonfinite relchem loss or multiplier')
            original_relchem = raw['relchem']
            transformed = scale_relchem(raw, multiplier)
            for task in ('ae17', 'exc', 'op'):
                if any(transformed[task][name] is not gradient for name, gradient in raw[task].items()):
                    stop('A non-relchem gradient was replaced')
            new_norm = float(torch.cat([value.reshape(-1) for value in transformed['relchem'].values()]).norm())
            old_norm = record['norms']['relchem']
            if abs(new_norm - multiplier * old_norm) > 1e-8 * max(1.0, new_norm):
                stop('Relchem multiplier was not applied exactly once')
            joint = run.adamw_step(model, optimizer, transformed, LAMBDAS)
            torch.cuda.synchronize()
            after = run.existing.digest(model)
            displacement = torch.cat([(value.detach() - before_weights[name]).reshape(-1).double()
                                      for name, value in parameters.items()]).norm()
            steps = {int(state['step']) for state in optimizer.state.values()}
            finite = bool(torch.isfinite(displacement).all() and all(torch.isfinite(value).all() for value in parameters.values())
                          and math.isfinite(float(displacement)) and steps == {index + 1})
            peak_allocated = torch.cuda.max_memory_allocated()
            peak_reserved = torch.cuda.max_memory_reserved()
            note_peak()
            if not finite:
                stop('Nonfinite parameter or AdamW state')
            logs.append({
                'cursor': index + 1, 'manifest_index': index,
                'identity': entry['relchem']['identity'], 'database': entry['relchem']['database'],
                'variant': entry['relchem']['variant'], 'ae17_identity': entry['ae17']['identity'],
                'ae17_variant': entry['ae17']['variant'], 'mrks_id': entry['mrks_id'],
                'system': record['system'], 'loss_old': loss_old_value, 'loss_new': new_loss,
                'factor': multiplier, 'original_norms': record['norms'], 'new_relchem_norm': new_norm,
                'joint_norm': float(torch.cat([value.reshape(-1) for value in joint.values()]).norm()),
                'displacement_norm': float(displacement), 'before_sha256': before, 'after_sha256': after,
                'seconds': time.perf_counter() - timer, 'peak_allocated_bytes': peak_allocated,
                'peak_reserved_bytes': peak_reserved, 'finite': True, 'other_gradients_unscaled': True,
                'factor_applications': 1, 'adamw_step': index + 1,
            })
            cursor = index + 1
            _save(run, latest, model, optimizer, cursor, logs, scale, scale_info['calibration_sha256'])
            if cursor % 25 == 0 or cursor in MILESTONES or cursor == 251:
                milestone = target / f'checkpoint_{cursor}.pt'
                if not milestone.exists():
                    _save(run, milestone, model, optimizer, cursor, logs, scale, scale_info['calibration_sha256'])
            _dump(OUT / 'training_status.json', {'cursor': cursor, 'complete': cursor == 251, 'logs': logs})
            print('RQ_UPDATE', cursor, logs[-1]['seconds'], logs[-1]['loss_old'], logs[-1]['factor'], flush=True)
            del raw, original_relchem, transformed, joint, before_weights
            torch.cuda.empty_cache()
        if sha256_file(protocol['initial_state']['path']) != INITIAL_FILE_SHA:
            stop('Initial checkpoint file changed during training')
        if sha256_file(REPO / 'relchem_joint_epoch_sampling_manifest.json') != MANIFEST_SHA:
            stop('Training manifest changed during training')
    finally:
        bundle.close()
    if cursor != 251:
        stop(_load(OUT / 'stop.json', {}).get('reason', 'Training stopped before 251 updates'))
    _evaluate_clean28(run, protocol, scale, scale_info['calibration_sha256'])
    _qualify_if_needed(run)
    stale_stop = OUT / 'stop.json'
    if stale_stop.exists():
        stale_stop.unlink()


def _load_reaction(run, bundle, split, identity, variant):
    import torch
    reaction = bundle.chemistry(split).load_variant(identity, variant)
    reaction = run.existing.lap_training.tensor_record(reaction, 'cuda', torch.float64)
    if len(reaction['Grid']) > 131072:
        reaction['model_point_chunk_size'] = 16384
    return reaction


def _pack(parameters, gradients):
    import torch
    pieces = []
    pairs = ((parameter, gradients[name]) for name, parameter in parameters.items()) if isinstance(gradients, dict) else zip(parameters.values(), gradients, strict=True)
    for parameter, value in pairs:
        pieces.append(torch.zeros(parameter.numel(), dtype=torch.float64, device=parameter.device)
                      if value is None else value.detach().reshape(-1).double())
    return torch.cat(pieces)


def _relative(left, right):
    reference = float(right.norm())
    discrepancy = float((left - right).norm())
    if reference == 0.0:
        return 0.0 if discrepancy == 0.0 else math.inf
    return discrepancy / reference


def _save(run, path, model, optimizer, cursor, logs, scale, calibration_sha):
    import copy

    import torch
    path = Path(path)
    if not path.resolve().is_relative_to(OUT.resolve()):
        raise SystemExit('Refusing to write a checkpoint outside the experiment directory')
    temporary = path.with_suffix('.tmp')
    torch.save({
        'model': {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        'optimizer': copy.deepcopy(optimizer.state_dict()),
        'rng': run.existing.capture_rng_state(), 'cursor': cursor, 'logs': list(logs),
        'protocol_id': PROTOCOL_ID, 'manifest_sha256': MANIFEST_SHA, 'scale_hex': float(scale).hex(),
        'calibration_sha256': calibration_sha, 'lambdas': LAMBDAS, 'scheduler': None,
    }, temporary)
    temporary.replace(path)


def _evaluate_clean28(run, protocol, scale, calibration_sha):
    import time

    import torch

    from tools.evaluate_microbatch_endpoint import validation
    missing = any(not (OUT / f'validation_{cursor}.json').exists() for cursor in MILESTONES)
    if missing and gpu_seconds() >= BUDGET_SECONDS:
        _dump(OUT / 'stop.json', {'reason': 'GPU budget exhausted before Clean28', 'gpu_seconds': gpu_seconds()})
        raise SystemExit('GPU budget exhausted before Clean28')
    bundle = run.PublicationDataset(run.DATA)
    try:
        for cursor in MILESTONES:
            destination = OUT / f'validation_{cursor}.json'
            if destination.exists():
                continue
            if gpu_seconds() >= BUDGET_SECONDS:
                _dump(OUT / 'stop.json', {'reason': 'GPU budget exhausted during Clean28', 'gpu_seconds': gpu_seconds()})
                raise SystemExit('GPU budget exhausted during Clean28')
            timer = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            model, shadow = run.model_at(protocol['initial_state'])
            del shadow
            checkpoint = OUT / 'ordinary_sgd_adamw' / f'checkpoint_{cursor}.pt'
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            require_checkpoint(saved, scale, calibration_sha)
            model.load_state_dict(saved['model'])
            before = run.existing.digest(model)
            metrics = validation(model, bundle)
            clean_rows = [row for row in metrics['reaction_rows'] if row['clean']]
            if (len(clean_rows) != 28 or not math.isfinite(metrics['clean28'])
                    or run.existing.digest(model) != before):
                raise SystemExit('Clean28 evaluation changed the model or the population')
            torch.cuda.synchronize()
            _dump(destination, {
                'cursor': cursor, 'clean28': metrics['clean28'], 'reaction_rows': clean_rows,
                'dispersion': 'PBE0-D3(BJ)', 'full30_discarded': True, 'full30_used_for_selection': False,
                'gpu_seconds': time.perf_counter() - timer, 'model_sha256': before,
                'checkpoint_sha256': sha256_file(checkpoint),
            })
            memory = _load(OUT / 'memory.json', {'peak_allocated_bytes': 0, 'peak_reserved_bytes': 0})
            memory['peak_allocated_bytes'] = max(memory['peak_allocated_bytes'], torch.cuda.max_memory_allocated())
            memory['peak_reserved_bytes'] = max(memory['peak_reserved_bytes'], torch.cuda.max_memory_reserved())
            _dump(OUT / 'memory.json', memory)
            del model
            print('RQ_CLEAN28', cursor, metrics['clean28'], flush=True)
    finally:
        bundle.close()


def _qualify_if_needed(run):
    import time
    milestones = {cursor: _load(OUT / f'validation_{cursor}.json') for cursor in MILESTONES}
    screened = [cursor for cursor in MILESTONES if beats_historical(milestones[cursor]['clean28'])]
    _dump(OUT / 'accuracy_gate.json', {'screened_cursors': screened, 'historical_best': HISTORICAL_BEST_CLEAN28})
    if not screened:
        return
    source = REPO / 'relchem_joint_epoch_evaluation_manifest.json'
    destination = OUT / 'evaluation_manifest.json'
    if sha256_file(source) != EVAL_SHA:
        raise SystemExit('Evaluation manifest changed')
    if not destination.exists():
        destination.write_bytes(source.read_bytes())
    if sha256_file(destination) != EVAL_SHA:
        raise SystemExit('Scratch evaluation manifest changed')
    from tools.evaluate_microbatch_endpoint import evaluate
    for cursor in screened:
        for stage, expected in (('chemistry', 268), ('mrks', 90)):
            path = OUT / f"endpoint_{cursor}_{stage}{'_one_variant' if stage == 'chemistry' else ''}.json"
            if _load(path, {}).get('complete') and len(_load(path)['rows']) == expected:
                continue
            if gpu_seconds() >= BUDGET_SECONDS:
                _dump(OUT / 'stop.json', {'reason': 'GPU budget exhausted during qualification',
                                          'gpu_seconds': gpu_seconds()})
                raise SystemExit('GPU budget exhausted during qualification')
            timer = time.perf_counter()
            evaluate(OUT, cursor, stage, max(1, min(3600, int(BUDGET_SECONDS - gpu_seconds()))))
            _append_jsonl(OUT / 'qualification_time.jsonl', {
                'cursor': cursor, 'stage': stage, 'seconds': time.perf_counter() - timer,
            })


def _distribution(values):
    ordered = sorted(float(value) for value in values)
    if not ordered or any(not math.isfinite(value) for value in ordered):
        raise ValueError('Multiplier distribution must be finite')

    def quantile(fraction):
        position = (len(ordered) - 1) * fraction
        low = math.floor(position)
        high = math.ceil(position)
        return ordered[low] if low == high else ordered[low] * (high - position) + ordered[high] * (position - low)
    return {'count': len(ordered), 'min': ordered[0], 'p25': quantile(0.25), 'median': statistics.median(ordered),
            'p75': quantile(0.75), 'max': ordered[-1],
            'above_one': sum(value > 1.0 for value in ordered),
            'below_one': sum(value < 1.0 for value in ordered)}


def _controls():
    p536 = _load(Path(r'C:\Dev\readWFN_share_ms\lap_relchem_joint_epoch_20261009\R\endpoint_0_validation.json'))
    j251 = _load(Path(r'C:\Dev\readWFN_share_ms\lap_relchem_joint_epoch_20261009\J\endpoint_251_validation.json'))
    metrics = _load(REPO / 'relchem_joint_epoch_metrics.json')
    arm = metrics['results']['J'] if 'results' in metrics and 'J' in metrics['results'] else None
    if arm is None:
        for value in metrics.values():
            if isinstance(value, dict) and isinstance(value.get('J'), dict) and 'ratios' in value['J']:
                arm = value['J']
                break
    b80 = _load(REPO / 'iid_adamw_lr_stabilization_metrics.json')['audits']['B80']
    if (p536['metrics']['clean28'] != P536_CLEAN28 or j251['metrics']['clean28'] != J251_CLEAN28
            or arm['ratios'] != J251_RATIOS or b80['ratios_t0'] != B80_RATIOS):
        raise SystemExit('Frozen control receipt does not match the predeclared values')
    return {
        'p536_rows': [row for row in p536['metrics']['reaction_rows'] if row['clean']],
        'j251_rows': [row for row in j251['metrics']['reaction_rows'] if row['clean']],
        'j251_objectives': arm['objectives'], 'b80_objectives': b80['objectives'],
    }


def _reaction_analysis(rows, controls):
    diet = {}
    for line in (REPO.parent / 'publication_dataset_v1' / 'validation' / 'diet30_reactions.jsonl').read_text(encoding='utf-8').splitlines():
        if line:
            row = json.loads(line)
            diet[row['source_id']] = row['database']
    p536 = {row['reaction_id']: row for row in controls['p536_rows']}
    j251 = {row['reaction_id']: row for row in controls['j251_rows']}
    enriched = []
    for row in rows:
        enriched.append({
            'reaction_id': row['reaction_id'], 'database': diet[row['reaction_id']],
            'signed_error_kcal_mol': row['signed_error_kcal_mol'],
            'weighted_absolute_error': row['weighted_absolute_error'],
            'delta_weighted_vs_p536': row['weighted_absolute_error'] - p536[row['reaction_id']]['weighted_absolute_error'],
            'delta_weighted_vs_j251': row['weighted_absolute_error'] - j251[row['reaction_id']]['weighted_absolute_error'],
            'delta_signed_vs_p536': row['signed_error_kcal_mol'] - p536[row['reaction_id']]['signed_error_kcal_mol'],
            'delta_signed_vs_j251': row['signed_error_kcal_mol'] - j251[row['reaction_id']]['signed_error_kcal_mol'],
        })
    subsets = {}
    for row in enriched:
        bucket = subsets.setdefault(row['database'], {'count': 0, 'weighted_sum': 0.0})
        bucket['count'] += 1
        bucket['weighted_sum'] += row['weighted_absolute_error']
    largest = sorted(enriched, key=lambda row: row['weighted_absolute_error'], reverse=True)[:5]
    return {'rows': enriched, 'subsets': subsets, 'largest_five': largest,
            'improved_vs_j251': sum(row['delta_weighted_vs_j251'] < 0 for row in enriched),
            'worse_vs_j251': sum(row['delta_weighted_vs_j251'] > 0 for row in enriched)}


def _database_means(path):
    payload = _load(path)
    grouped = {}
    for row in payload['rows'].values():
        if row['task'] == 'relchem':
            grouped.setdefault(row['database'], []).append(row['loss'])
    return {database: {'count': len(values), 'mean': sum(values) / len(values),
                       'contribution': sum(values) / 251.0} for database, values in sorted(grouped.items())}


def render():
    status = _load(OUT / 'training_status.json', {})
    logs = status.get('logs', [])
    preflight = _load(OUT / 'preflight.json', {})
    scale = _load(OUT / 'scale.json', {})
    milestones = {}
    for cursor in MILESTONES:
        payload = _load(OUT / f'validation_{cursor}.json')
        if payload:
            milestones[cursor] = payload
    qualified = {}
    for cursor in MILESTONES:
        chemistry = _load(OUT / f'endpoint_{cursor}_chemistry_one_variant.json', {})
        mrks = _load(OUT / f'endpoint_{cursor}_mrks.json', {})
        if chemistry.get('complete') and mrks.get('complete'):
            objectives = {**chemistry['objectives'], **mrks['objectives']}
            ratios = scientifically_eligible(objectives)
            qualified[cursor] = {'objectives': objectives, 'ratios': ratios or {
                task: objectives[task] / BASELINE[task] for task in TASKS}, 'eligible': bool(ratios)}
    result = {'updates': status.get('cursor', 0), 'parity_passed': bool(preflight.get('passed')),
              'numerical_failure': 'Nonfinite' in _load(OUT / 'stop.json', {}).get('reason', ''),
              'milestones': milestones, 'qualified': qualified}
    decision = classify(result)
    controls = _controls() if milestones else None
    analyses = {cursor: _reaction_analysis(payload['reaction_rows'], controls)
                for cursor, payload in milestones.items()} if controls else {}
    calibration_rows = _load(OUT / 'calibration_residuals.json', {}).get('rows', [])
    summary = {
        'decision': decision, 's': scale.get('s'), 's_hex': scale.get('s_hex'),
        'calibration_sha256': scale.get('calibration_sha256'),
        'updates': status.get('cursor', 0), 'gpu_seconds': gpu_seconds() if OUT.exists() else 0.0,
        'memory': _load(OUT / 'memory.json', {}), 'stop': _load(OUT / 'stop.json'),
        'preflight': preflight,
        'calibration_multiplier_distribution': _distribution([row['gradient_factor'] for row in calibration_rows]) if calibration_rows else None,
        'training_multiplier_distribution': _distribution([row['factor'] for row in logs]) if logs else None,
        'milestones': {str(cursor): {'clean28': payload['clean28']} for cursor, payload in milestones.items()},
        'qualified': {str(cursor): value for cursor, value in qualified.items()},
        'analyses': {str(cursor): value for cursor, value in analyses.items()},
        'other_gradients_unscaled': all(row.get('other_gradients_unscaled') for row in logs) if logs else None,
        'factor_applications': sorted({row['factor_applications'] for row in logs}) if logs else [],
        'adamw_steps': [row['adamw_step'] for row in logs[-1:]] if logs else [],
    }
    if qualified:
        summary['database_means'] = {
            str(cursor): _database_means(OUT / f'endpoint_{cursor}_chemistry_one_variant.json') for cursor in qualified
        }
    _dump(OUT / 'summary.json', summary)
    _write_report(summary)
    return summary


def _fmt(value):
    return f'{value:.9f}' if isinstance(value, float) else str(value)


def _write_report(summary):
    decision = summary['decision']
    lines = [
        '# Residual-aware quadratic chemistry loss',
        '',
        f'**Decision: {decision}**',
        '',
        ('One new 251-update four-task AdamW arm, matched to historical J251. '
         'The only intended change is the singleton relchem gradient multiplier '
         '`f = 0.5 * (1 + L_A / (a * s))`. J251 was not retrained.'),
        '',
        '## Frozen scale and parity',
        '',
        f"Training-only residual scale `s = {summary.get('s')}`.",
        f"Calibration receipt SHA256: `{summary.get('calibration_sha256')}`.",
        ('This median uses the 251 J251 training-manifest variants at corrected P536. '
         'Clean28 did not enter the scale, the learning rate, or the four task coefficients.'),
        '',
        ('`c = 0.5` anchors the derivative ratio near 1 at the initial median residual. '
         'It does not make the joint-gradient norm or the AdamW step identical to J251.'),
        '',
    ]
    preflight = summary.get('preflight') or {}
    if preflight.get('parity'):
        lines.append('| Manifest index | Relative L2 | Qualified-path relative L2 | Factor |')
        lines.append('|---:|---:|---:|---:|')
        for row in preflight['parity']:
            lines.append(f"| {row['index']} | {row['relative_l2']:.3e} | {row['qualified_relative_l2']:.3e} | {row['factor']:.6f} |")
        lines.append('')
        lines.append(f"Joint preflight norm `{preflight.get('joint_norm')}`, factor applications "
                     f"`{preflight.get('joint_factor_applications')}`, optimizer step `{preflight.get('optimizer_step')}`.")
        lines.append('')
    for label, key in (('P536 calibration multipliers', 'calibration_multiplier_distribution'),
                       ('Training-update multipliers', 'training_multiplier_distribution')):
        distribution = summary.get(key)
        if distribution:
            lines.append(f"{label}: min {distribution['min']:.6f}, median {distribution['median']:.6f}, "
                         f"max {distribution['max']:.6f}, above 1: {distribution['above_one']}, "
                         f"below 1: {distribution['below_one']}.")
    lines.extend(['', '## Evidence that only relchem changed', ''])
    lines.append(f"Other-task gradients left identical: `{summary.get('other_gradients_unscaled')}`.")
    lines.append(f"Relchem factor applications per update: `{summary.get('factor_applications')}`.")
    lines.append('AE17, Exc, and operator gradients were reused from `measure`. '
                 'Historical lambdas were applied once by the existing scalarization. '
                 'The single F32 cast remains inside `adamw_step`.')
    lines.extend(['', '## Runtime', ''])
    lines.append(f"Completed optimizer updates: `{summary.get('updates')}`.")
    lines.append(f"Cumulative new GPU time: `{summary.get('gpu_seconds', 0.0):.3f}` seconds.")
    memory = summary.get('memory') or {}
    allocated = memory.get('peak_allocated_bytes')
    reserved = memory.get('peak_reserved_bytes')
    if allocated:
        lines.append(f'Peak allocated CUDA memory: `{allocated}` bytes ({allocated / 1024**3:.3f} GiB).')
        lines.append(f'Peak reserved CUDA memory: `{reserved}` bytes ({reserved / 1024**3:.3f} GiB).')
    if summary.get('stop'):
        lines.append(f"Stop record: `{summary['stop']}`")
    lines.extend(['', '## Comparison', '',
                  '| Model | Updates | Clean28 | Relchem ratio | AE17 ratio | Exc ratio | Op ratio | Eligible |',
                  '|---|---:|---:|---:|---:|---:|---:|---|',
                  f'| P536 | 0 | {_fmt(P536_CLEAN28)} | 1 | 1 | 1 | 1 | No, strict threshold |',
                  f"| J251 control | 251 | {_fmt(J251_CLEAN28)} | {_fmt(J251_RATIOS['relchem'])} | {_fmt(J251_RATIOS['ae17'])} | {_fmt(J251_RATIOS['exc'])} | {_fmt(J251_RATIOS['op'])} | Yes |",
                  f"| Historical best | 80 | {_fmt(HISTORICAL_BEST_CLEAN28)} | {_fmt(B80_RATIOS['relchem'])} | {_fmt(B80_RATIOS['ae17'])} | {_fmt(B80_RATIOS['exc'])} | {_fmt(B80_RATIOS['op'])} | No |"])
    for cursor in MILESTONES:
        measured = summary['milestones'].get(str(cursor))
        qualified = summary['qualified'].get(str(cursor))
        if not measured:
            lines.append(f'| New t{cursor} | {cursor} | not measured | not measured | not measured | not measured | not measured | not measured |')
            continue
        if qualified:
            ratios = qualified['ratios']
            cells = ' | '.join(_fmt(ratios[task]) for task in TASKS)
            eligible = 'Yes' if qualified['eligible'] else 'No'
        elif decision == 'NO-GO':
            cells = ' | '.join([GATE] * 4)
            eligible = GATE
        else:
            cells = ' | '.join(['not measured'] * 4)
            eligible = 'not measured'
        lines.append(f"| New t{cursor} | {cursor} | {_fmt(measured['clean28'])} | {cells} | {eligible} |")
    lines.extend(['', '## Clean28 reactions', ''])
    if not summary.get('analyses'):
        lines.append('Clean28 reaction rows were not available.')
    else:
        for cursor, analysis in summary['analyses'].items():
            lines.append(f'### t{cursor}')
            lines.append('')
            lines.append(f"Improved versus J251: {analysis['improved_vs_j251']}. "
                         f"Worse versus J251: {analysis['worse_vs_j251']}.")
            lines.append('')
            lines.append('| Subset | Count | Weighted sum |')
            lines.append('|---|---:|---:|')
            for name, bucket in sorted(analysis['subsets'].items()):
                lines.append(f"| {name} | {bucket['count']} | {bucket['weighted_sum']:.6f} |")
            lines.append('')
            lines.append('Largest five weighted contributions:')
            lines.append('')
            for row in analysis['largest_five']:
                lines.append(f"- `{row['reaction_id']}` ({row['database']}): signed {row['signed_error_kcal_mol']:.6f}, "
                             f"weighted {row['weighted_absolute_error']:.6f}, "
                             f"delta weighted vs P536 {row['delta_weighted_vs_p536']:.6f}, "
                             f"delta weighted vs J251 {row['delta_weighted_vs_j251']:.6f}.")
            lines.append('')
    lines.extend(['## Scientific databases', ''])
    if summary.get('database_means'):
        for cursor, grouped in summary['database_means'].items():
            lines.append(f'### t{cursor}')
            lines.append('')
            lines.append('| Database | Count | Mean singleton | Contribution |')
            lines.append('|---|---:|---:|---:|')
            for name, bucket in grouped.items():
                lines.append(f"| {name} | {bucket['count']} | {bucket['mean']:.6f} | {bucket['contribution']:.6f} |")
            lines.append('')
    else:
        lines.append(GATE if decision == 'NO-GO' else 'Scientific database means were not evaluated.')
    t251 = summary['milestones'].get('251', {}).get('clean28')
    helped = 'not measured'
    if t251 is not None:
        helped = 'yes' if t251 < J251_CLEAN28 else 'no'
    lines.extend(['', '## Did the quadratic term help?', '',
                  f'Matched t251 Clean28 versus frozen J251: `{helped}`.',
                  ('A lower training surrogate is not evidence of a better functional. '
                   'The accuracy gate is Clean28 strictly below the exact B80 receipt '
                   f'`{HISTORICAL_BEST_CLEAN28}`, followed by all four scientific ratios strictly below 1.'),
                  '',
                  '## Provenance', '',
                  f'Initial file SHA256 `{INITIAL_FILE_SHA}`.',
                  f'Initial tensor SHA256 `{INITIAL_TENSOR_SHA}`.',
                  f'Training manifest SHA256 `{MANIFEST_SHA}`.',
                  f'Evaluation manifest SHA256 `{EVAL_SHA}`.',
                  f'J251 calibration SHA256 `{CALIBRATION_SHA}`.',
                  'Production physics files were hashed against the J251 protocol before training.',
                  '',
                  '## Next recommendation', '',
                  _recommendation(decision),
                  '',
                  'This recommendation was not executed.',
                  ''])
    (REPO / 'lap_residual_quadratic_epoch_report.md').write_text('\n'.join(lines), encoding='utf-8')
    public = {key: summary[key] for key in (
        'decision', 's', 's_hex', 'calibration_sha256', 'updates', 'gpu_seconds', 'memory', 'stop',
        'calibration_multiplier_distribution', 'training_multiplier_distribution', 'milestones',
        'qualified', 'other_gradients_unscaled', 'factor_applications')}
    public['preflight'] = {key: preflight.get(key) for key in (
        'passed', 'optimizer_step', 'joint_norm', 'joint_factor', 'joint_factor_applications',
        'relchem_scale_relative_l2', 'measure_vs_qualified_relative_l2', 'parity')}
    public['analyses'] = summary.get('analyses')
    public['database_means'] = summary.get('database_means')
    public['controls'] = {'p536_clean28': P536_CLEAN28, 'j251_clean28': J251_CLEAN28,
                          'historical_best_clean28': HISTORICAL_BEST_CLEAN28,
                          'j251_ratios': J251_RATIOS, 'b80_ratios': B80_RATIOS}
    _dump(REPO / 'lap_residual_quadratic_epoch_metrics.json', public)


def _recommendation(decision):
    if decision == 'BREAKTHROUGH':
        return ('Freeze the eligible checkpoint and, in a separately authorized experiment, '
                'repeat the same loss once with a new seed. Do not retune `c` or `s` on this result.')
    if decision == 'PROGRESS':
        return ('The matched arm cleared the historical Clean28 bar while remaining eligible, '
                'but it did not reach 7 kcal/mol. A later confirmatory seed is the useful next test. '
                'Do not search quadratic coefficients on this seed.')
    if decision == 'NO-GO':
        return ('Do not sweep `c`, `s`, or another seed of this residual-squared multiplier. '
                'The matched arm did not put an eligible functional below the historical Clean28 best. '
                'The useful next study is a read-only attribution of the stubborn ABDE4, pTC13, and PA8 '
                'training reactions, with no further optimization.')
    return ('Resume this same arm from the last matching checkpoint if the GPU budget and a valid '
            'checkpoint remain. Do not start a second loss, seed, or coefficient search.')


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    if args.report_only:
        render()
        return
    try:
        execute()
    except SystemExit:
        raise
    except (FloatingPointError, RuntimeError, AssertionError, ValueError) as error:
        if OUT.exists():
            _dump(OUT / 'stop.json', {'reason': f'{type(error).__name__}: {error}', 'gpu_seconds': gpu_seconds()})
        raise
    finally:
        if OUT.exists():
            render()


if __name__ == '__main__':
    main()
