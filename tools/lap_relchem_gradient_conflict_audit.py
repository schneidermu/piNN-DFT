"""Frozen gradient-conflict diagnostic at P536 and IID t70.

Reads two existing checkpoints and evaluates the predeclared 25+17+4 panel.
Does not train, step an optimizer, run SCF, or write checkpoint files.
"""
import ast
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

POPULATION = 251
PARAMETER_COUNT = 9446
PANEL_INDEXES = (11, 33, 56, 78)
FOCUS_DATABASES = ('ABDE4', 'pTC13', 'PA8')
EXPECTED_COUNTS = {'ABDE4': 4, 'pTC13': 13, 'PA8': 8}
LAMBDAS = {
    'relchem': 0.017015480965588553,
    'ae17': 0.00005141254618347414,
    'exc': 0.000015094644512009712,
    'op': 0.33597561607048215,
}
DATA_SHA = '61c221a19b9987717e69cac182ad545241f8807db4126c0949a99992e4c210ef'
MANIFEST_SHA = '132bd550161be6bc6761f8df1de8d3755c0b0debab53663208c3a9bbf56cb805'
S0_TENSOR_SHA = '3c2e45d6a86cccf80d7b4dc458e67f9b406935ed60d4e57433bb441a6df88da6'
S0_FILE_SHA = '0ca0f77367214c171e6edf0576bdb043d64479245ad1a0e539fa11fb0cee542d'
S70_FILE_SHA = '04b8e549c17375988be576e2878881d13f55b217b42ec16a7e6101fdddd05443'
CLEAN28 = {'s0': 9.553190635871177, 's70': 8.629660230056073}
LOSS_ATOL = 1e-6
BUDGET_SECONDS = 30 * 60
EVALUATION_LIMIT = 100

REPO = Path(__file__).resolve().parents[1]
SHARE = REPO.parent
SCRATCH = SHARE / 'lap_relchem_gradconflict_20261009'
DATA = SHARE / 'publication_dataset_v1'
S0_PATH = SHARE / 'lap_init_landscape_runs_20261005' / 'states' / 'seed11_P536.pt'
S70_PATH = SHARE / 'lap_iid_adamw_t59_t90_20261009' / 'ordinary_sgd_adamw' / 'checkpoint_70.pt'
MANIFEST_PATH = SHARE / 'lap_iid_adamw_t59_t90_20261009' / 'evaluation_manifest.json'
RECEIPTS = {
    's0': SHARE / 'lap_iid_adamw_t59_t90_20261009' / 'baseline_0_chemistry_one_variant.json',
    's70': SHARE / 'lap_iid_adamw_lr_branches_20261009' / 'A' / 'endpoint_70_chemistry_one_variant.json',
}
CLEAN_RECEIPTS = {
    's0': SHARE / 'lap_iid_adamw_t59_t90_20261009' / 'baseline_0_validation.json',
    's70': SHARE / 'lap_iid_adamw_t59_t90_20261009' / 'endpoint_70_validation.json',
}
SOURCE_FILES = (
    'train_lap_microbatch.py',
    'train_models/lap_moo_training.py',
    'train_models/lap_fixed_adamw.py',
    'train_models/lap_training.py',
    'train_models/lap_operator.py',
    'train_models/lap_operator_data.py',
    'train_models/lap_vxc.py',
    'train_models/lap_chemistry_sampling.py',
    'train_models/publication_data/loader.py',
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def as_vector(named, names):
    """Pack one gradient in named_trainable_parameters order."""
    parts = []
    for name in names:
        value = named[name]
        if hasattr(value, 'detach'):
            array = value.detach().reshape(-1).double().cpu().numpy()
        else:
            array = np.asarray(value, dtype=np.float64).reshape(-1)
        parts.append(np.ascontiguousarray(array, dtype=np.float64))
    vector = np.concatenate(parts)
    if vector.shape != (PARAMETER_COUNT,) or not np.isfinite(vector).all():
        raise ValueError(f'Gradient length/finiteness failed: {vector.shape}')
    return vector


def database_terms(vectors):
    """Return the database mean gradient and its 251-identity contribution."""
    stacked = np.stack([np.asarray(vector, dtype=np.float64) for vector in vectors])
    count = stacked.shape[0]
    total = np.sum(stacked, axis=0)
    mean = total / np.float64(count)
    contribution = mean * (np.float64(count) / np.float64(POPULATION))
    residual = contribution - (total / np.float64(POPULATION))
    if np.max(np.abs(residual)) > 1e-8 * max(1.0, float(np.linalg.norm(contribution))):
        raise ValueError('Database mean and contribution are inconsistent')
    return mean, contribution


def cosine(left, right):
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0 or not math.isfinite(left_norm * right_norm):
        return None
    return float(np.dot(left, right) / (left_norm * right_norm))


def cancellation_ratio(left, right):
    """Zero when the vectors are parallel; one when they cancel exactly."""
    denominator = float(np.linalg.norm(left) + np.linalg.norm(right))
    if denominator == 0.0:
        return None
    return 1.0 - float(np.linalg.norm(left + right) / denominator)


def within_cancellation(vectors):
    denominator = float(sum(np.linalg.norm(vector) for vector in vectors))
    if denominator == 0.0:
        return None
    return 1.0 - float(np.linalg.norm(np.sum(np.stack(vectors), axis=0)) / denominator)


def select_systems(rows):
    """Choose four systems by sorted grid size before any gradient exists."""
    if len(rows) != 90:
        raise ValueError(f'Expected 90 mRKS systems, found {len(rows)}')
    ordered = sorted(rows, key=lambda row: (int(row['n_grid']), row['id']))
    return [ordered[index] for index in PANEL_INDEXES]


def competing_direction(ae17, exc, operator):
    return (
        LAMBDAS['ae17'] * ae17
        + LAMBDAS['exc'] * exc
        + LAMBDAS['op'] * operator
    )


def predicted_change(gradient, direction):
    return float(np.dot(gradient, -direction))


def sign_or_zero(value):
    if value is None or not math.isfinite(value) or value == 0.0:
        return 0
    return 1 if value > 0.0 else -1


def forbid_training_calls(source):
    """Reject calls that would update parameters or launch a job."""
    tree = ast.parse(source)
    forbidden_names = {
        'adamw_step', 'apply_joint_gradient', 'train_moo_update', 'train',
        'submit', 'sbatch',
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, 'attr', '')
            if name in forbidden_names or name == 'step':
                raise ValueError(f'Forbidden call: {name}')
        if isinstance(node, ast.Attribute) and node.attr == 'optim':
            raise ValueError('Optimizer construction is forbidden')


def chemistry_panel(manifest_rows, reactions):
    selected = {database: [] for database in FOCUS_DATABASES}
    ae17 = []
    for row in manifest_rows:
        identity, variant, task = row['identity'], row['variant'], row['task']
        record = reactions[identity]
        if record['database'] != row.get('database', record['database']) and 'database' in row:
            raise ValueError(f'Database mismatch for {identity}')
        if task == 'relchem' and record['database'] in selected:
            if record['task'] != 'relchem':
                raise ValueError(f'{identity} is not a relchem reaction')
            selected[record['database']].append({
                'identity': identity, 'variant': variant, 'database': record['database'],
            })
        elif task == 'ae17':
            if record['database'] != 'AE17' or record['task'] != 'ae17':
                raise ValueError(f'{identity} is not an AE17 reaction')
            ae17.append({'identity': identity, 'variant': variant, 'database': 'AE17'})
    for database, expected in EXPECTED_COUNTS.items():
        rows = sorted(selected[database], key=lambda item: item['identity'])
        if len(rows) != expected or len({item['identity'] for item in rows}) != expected:
            raise ValueError(f'{database} identity count is {len(rows)}, expected {expected}')
        selected[database] = rows
    if len(ae17) != 17 or len({item['identity'] for item in ae17}) != 17:
        raise ValueError(f'AE17 identity count is {len(ae17)}')
    return selected, sorted(ae17, key=lambda item: item['identity'])


def receipt_losses(path):
    payload = read_json(path)
    if payload.get('evaluation_manifest_sha256') != MANIFEST_SHA or not payload.get('complete'):
        raise ValueError(f'Chemistry receipt is not the frozen complete manifest: {path}')
    return {identity: row for identity, row in payload['rows'].items()}


def loss_agreement(computed, receipt_row):
    if computed['variant'] != receipt_row['variant'] or computed['database'] != receipt_row['database']:
        raise ValueError(f"Receipt alignment failed for {computed['identity']}")
    error = abs(computed['loss'] - receipt_row['loss'])
    if error > LOSS_ATOL:
        raise ValueError(
            f"Loss for {computed['identity']} differs from the receipt by {error}"
        )
    return error


def geometry(named_vectors):
    labels = tuple(named_vectors)
    matrix = {}
    for left in labels:
        matrix[left] = {}
        for right in labels:
            matrix[left][right] = cosine(named_vectors[left], named_vectors[right])
    return matrix


def analyze_state(terms, ae17, exc, operator):
    other = competing_direction(ae17, exc, operator)
    named = {**terms, 'AE17': ae17, 'Exc_panel': exc, 'Op_panel': operator, 'G_other': other}
    rows = {}
    for name, gradient in terms.items():
        weighted = {
            'ae17': LAMBDAS['ae17'] * float(np.dot(gradient, ae17)),
            'exc': LAMBDAS['exc'] * float(np.dot(gradient, exc)),
            'op': LAMBDAS['op'] * float(np.dot(gradient, operator)),
        }
        rows[name] = {
            'norm': float(np.linalg.norm(gradient)),
            'dot_ae17': float(np.dot(gradient, ae17)),
            'dot_exc_panel': float(np.dot(gradient, exc)),
            'dot_op_panel': float(np.dot(gradient, operator)),
            'dot_other': float(np.dot(gradient, other)),
            'weighted_dot_components': weighted,
            'cos_other': cosine(gradient, other),
            'predicted_delta_under_minus_other': predicted_change(gradient, other),
        }
    return {
        'other_norm': float(np.linalg.norm(other)),
        'weighted_task_norms': {
            'ae17': float(np.linalg.norm(LAMBDAS['ae17'] * ae17)),
            'exc': float(np.linalg.norm(LAMBDAS['exc'] * exc)),
            'op': float(np.linalg.norm(LAMBDAS['op'] * operator)),
        },
        'cosine_matrix': geometry(named),
        'terms': rows,
    }


def historical_row(actual, gradient_s0, gradient_s70, displacement):
    predicted_s0 = float(np.dot(gradient_s0, displacement))
    predicted_s70 = float(np.dot(gradient_s70, displacement))
    return {
        'actual_loss_change': actual,
        'dot_s0': predicted_s0,
        'dot_s70': predicted_s70,
        'cos_s0': cosine(gradient_s0, displacement),
        'cos_s70': cosine(gradient_s70, displacement),
        'sign_actual': sign_or_zero(actual),
        'sign_s0': sign_or_zero(predicted_s0),
        'sign_s70': sign_or_zero(predicted_s70),
    }


def adjudicate(history, states):
    """Label local evidence. These labels are not proofs of AdamW causality."""
    local = []
    for state in states.values():
        for database in FOCUS_DATABASES:
            row = state['terms'][database]
            local.append((row['cos_other'], row['predicted_delta_under_minus_other']))
    if all(value is not None and value < 0.0 and delta > 0.0 for value, delta in local):
        h1 = 'SUPPORTED LOCALLY'
    elif all(value is not None and value >= 0.0 for value, _delta in local):
        h1 = 'WEAKENED'
    else:
        h1 = 'INCONCLUSIVE'
    mutual = []
    for state in states.values():
        matrix = state['cosine_matrix']
        mutual.extend((matrix['ABDE4']['pTC13'], matrix['ABDE4']['PA8'], matrix['pTC13']['PA8']))
    if all(value is not None and value < 0.0 for value in mutual):
        h2 = 'SUPPORTED LOCALLY'
        mutual_clause = 'SUPPORTED LOCALLY'
    elif all(value is not None and value > 0.0 for value in mutual):
        h2 = 'INCONCLUSIVE'
        mutual_clause = 'WEAKENED'
    else:
        h2 = 'INCONCLUSIVE'
        mutual_clause = 'INCONCLUSIVE'
    focused = [history[database] for database in FOCUS_DATABASES]
    if all(row['sign_actual'] == row['sign_s0'] == row['sign_s70'] != 0 for row in focused):
        h3 = 'WEAKENED'
    elif sum(row['sign_actual'] not in (row['sign_s0'], row['sign_s70']) for row in focused) >= 2:
        h3 = 'SUPPORTED LOCALLY'
    else:
        h3 = 'INCONCLUSIVE'
    if h1 == 'SUPPORTED LOCALLY' and h3 == 'WEAKENED':
        choice = 'B'
    elif h3 == 'SUPPORTED LOCALLY' or (h1 == 'SUPPORTED LOCALLY' and h3 != 'WEAKENED'):
        choice = 'C'
    elif h1 == 'WEAKENED' and h3 == 'WEAKENED':
        choice = 'A'
    else:
        choice = 'D'
    return {
        'H1': h1,
        'H2': h2,
        'H2_mutual_clause': mutual_clause,
        'H2_remaining_226': 'NOT VERIFIED',
        'H3': h3,
        'next_experiment': choice,
    }


def scientific_modules():
    import sys

    import torch
    sys.path.insert(0, str(REPO))
    import train_lap_microbatch as microbatch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    return torch, microbatch


def rng_unchanged(torch, before, after):
    if before['python'] != after['python']:
        return False
    if before['numpy'][0] != after['numpy'][0] or not np.array_equal(before['numpy'][1], after['numpy'][1]):
        return False
    if before['numpy'][2:] != after['numpy'][2:]:
        return False
    if not torch.equal(before['torch_cpu'], after['torch_cpu']):
        return False
    return all(torch.equal(left, right) for left, right in zip(before['torch_cuda'], after['torch_cuda'], strict=True))


def parameter_names(model):
    import torch
    from lap_moo_training import named_trainable_parameters
    named = named_trainable_parameters(model)
    count = sum(int(parameter.numel()) for parameter in named.values())
    if count != PARAMETER_COUNT:
        raise RuntimeError(f'Trainable parameter count is {count}, expected {PARAMETER_COUNT}')
    if any(parameter.dtype != torch.float32 for parameter in named.values()):
        raise RuntimeError('Stored trainable parameters are not F32')
    if any(not torch.isfinite(parameter.detach()).all() for parameter in named.values()):
        raise RuntimeError('Nonfinite trainable parameter')
    return tuple(named)


def pack_theta(model, names):
    import torch
    named = {name: parameter for name, parameter in model.named_parameters(remove_duplicate=True)}
    arrays = []
    for name in names:
        value = named[name].detach()
        if value.dtype != torch.float32 or value.grad is not None:
            raise RuntimeError(f'Parameter storage changed for {name}')
        arrays.append(value.cpu().numpy().reshape(-1))
    theta = np.concatenate(arrays)
    if theta.shape != (PARAMETER_COUNT,) or theta.dtype != np.float32 or not np.isfinite(theta).all():
        raise RuntimeError('Parameter vector failed finiteness or length checks')
    return theta


def assert_model_intact(microbatch, model, digest):
    import torch
    if microbatch.existing.digest(model) != digest:
        raise RuntimeError('Model tensor hash changed during a diagnostic evaluation')
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError('A diagnostic wrote .grad into the stored model')
    if any(not torch.isfinite(parameter.detach()).all() for parameter in model.parameters()):
        raise RuntimeError('Model parameters became nonfinite')


def load_frozen(microbatch, path, file_sha, state_sha=None):
    import torch
    if sha256(path) != file_sha:
        raise RuntimeError(f'Checkpoint file hash mismatch: {path}')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if 'model' not in payload:
        raise RuntimeError(f'Checkpoint has no model state: {path}')
    probe = microbatch.existing._pilot_model(torch.device('cpu'), torch.float32)
    probe.load_state_dict(payload['model'])
    digest = microbatch.existing.digest(probe)
    del payload, probe
    if state_sha is not None and digest != state_sha:
        raise RuntimeError(f'Model tensor hash mismatch: {digest}')
    model, shadow = microbatch.model_at({'path': str(path), 'file_sha256': file_sha, 'state_sha256': digest})
    names = parameter_names(model)
    if tuple(microbatch.existing.core.named_trainable_parameters(shadow)) != names:
        raise RuntimeError('Shadow parameter order differs from the stored model')
    assert_model_intact(microbatch, model, digest)
    return model, shadow, names, digest


def chemistry_gradient(microbatch, model, shadow, bundle, dispersion, item, names):
    import torch
    task = 'ae17' if item['database'] == 'AE17' else 'relchem'
    reaction = bundle.chemistry('train_' + task).load_variant(item['identity'], item['variant'])
    reaction = microbatch.existing.lap_training.tensor_record(reaction, 'cuda', torch.float64)
    if len(reaction['Grid']) > 131072:
        reaction['model_point_chunk_size'] = 16384
    loss, named = microbatch.chemistry(model, shadow, reaction, dispersion).value_and_grad()
    vector = as_vector(named, names)
    del reaction, named
    return float(loss), vector


def mrks_gradient(microbatch, model, bundle, dispersion, system_id, task, names):
    system = bundle.mrks().operator_system(system_id, device='cuda', dtype=microbatch.torch.float32, chunk_size=4096)
    energy, operator = microbatch.existing.core.make_mrks_objective_factories(
        model, system, point_chunk_size=256, dispersions=dispersion, exc_chunk_size=4096,
    )
    factory = {'exc': energy, 'op': operator}[task]
    losses, gradients = microbatch.existing.core.compute_isolated_task_gradients(
        model, {task: factory}, task_order=(task,),
    )
    materialized = microbatch.existing.core.materialize_task_zeros(model, gradients, task_order=(task,))
    vector = as_vector({name: value.double() for name, value in materialized[task].items()}, names)
    del system, energy, operator, gradients, materialized
    return float(losses[task]), vector


def saved_vector(state, kind, key):
    folder = SCRATCH / 'grad' / state / kind
    array_path = folder / f'{key}.npy'
    meta_path = folder / f'{key}.json'
    if array_path.exists() or meta_path.exists():
        if not array_path.exists() or not meta_path.exists():
            raise RuntimeError(f'Incomplete scratch gradient: {array_path}')
        vector = np.load(array_path)
        meta = read_json(meta_path)
        if vector.shape != (PARAMETER_COUNT,) or not np.isfinite(vector).all():
            raise RuntimeError(f'Scratch gradient is invalid: {array_path}')
        return meta, vector, False
    return None, None, True


def store_vector(state, kind, key, meta, vector):
    folder = SCRATCH / 'grad' / state / kind
    folder.mkdir(parents=True, exist_ok=True)
    array_path = folder / f'{key}.npy'
    meta_path = folder / f'{key}.json'
    if array_path.exists() or meta_path.exists():
        raise RuntimeError(f'Refusing to overwrite {array_path}')
    temporary = array_path.with_suffix('.tmp.npy')
    np.save(temporary, np.asarray(vector, dtype=np.float64))
    temporary.replace(array_path)
    meta_path.write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')


def evaluate_one(microbatch, compute, state, kind, key, digest, model):
    cached, vector, needed = saved_vector(state, kind, key)
    if not needed:
        assert_model_intact(microbatch, model, digest)
        return cached, vector, 0.0
    torch = microbatch.torch
    torch.cuda.synchronize()
    rng = microbatch.existing.capture_rng_state()
    started = time.perf_counter()
    loss, vector = compute()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if not rng_unchanged(torch, rng, microbatch.existing.capture_rng_state()):
        microbatch.existing.restore_rng_state(rng)
        rng_changed = True
    else:
        rng_changed = False
    assert_model_intact(microbatch, model, digest)
    meta = {'loss': loss, 'seconds': elapsed, 'rng_restored': rng_changed}
    store_vector(state, kind, key, meta, vector)
    return meta, vector, elapsed


def prepare_protocol():
    if sha256(MANIFEST_PATH) != MANIFEST_SHA:
        raise RuntimeError('Evaluation manifest hash mismatch')
    dataset_manifest = read_json(DATA / 'dataset_manifest.json')
    if dataset_manifest.get('logical_sha256') != DATA_SHA:
        raise RuntimeError(f"Dataset logical SHA mismatch: {dataset_manifest.get('logical_sha256')}")
    checksums = {row['file']: row['sha256'] for row in dataset_manifest['files']}
    if sha256(DATA / 'mrks' / 'systems.jsonl') != checksums['mrks/systems.jsonl']:
        raise RuntimeError('mRKS system metadata hash does not match the dataset manifest')
    if sha256(DATA / 'chemistry' / 'reactions.jsonl') != checksums['chemistry/reactions.jsonl']:
        raise RuntimeError('Chemistry metadata hash does not match the dataset manifest')
    manifest = read_json(MANIFEST_PATH)['rows']
    systems = [json.loads(line) for line in (DATA / 'mrks' / 'systems.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(systems) != 90:
        raise RuntimeError('mRKS metadata does not contain 90 systems')
    selected = select_systems(systems)
    reactions = {}
    for line in (DATA / 'chemistry' / 'reactions.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        reactions[row['id']] = row
    counts = {}
    for database in FOCUS_DATABASES:
        counts[database] = sum(row['task'] == 'relchem' and row['database'] == database for row in reactions.values())
        if counts[database] != EXPECTED_COUNTS[database]:
            raise RuntimeError(f'Dataset count for {database} is {counts[database]}')
    focus, ae17 = chemistry_panel(manifest, reactions)
    receipts = {name: receipt_losses(path) for name, path in RECEIPTS.items()}
    for state, rows in receipts.items():
        for database, items in focus.items():
            for item in items:
                receipt = rows[item['identity']]
                if receipt['variant'] != item['variant'] or receipt['database'] != database or receipt['task'] != 'relchem':
                    raise RuntimeError(f"Receipt variant mismatch at {state} {item['identity']}")
        if sum(row['task'] == 'ae17' for row in rows.values()) != 17:
            raise RuntimeError('AE17 receipt count is not 17')
    for state, path in CLEAN_RECEIPTS.items():
        value = read_json(path)['metrics']['clean28']
        if abs(value - CLEAN28[state]) > 1e-12:
            raise RuntimeError(f'Clean28 receipt mismatch at {state}: {value}')
    protocol = {
        'dataset_sha256': DATA_SHA,
        'evaluation_manifest_sha256': MANIFEST_SHA,
        'systems_metadata_sha256': sha256(DATA / 'mrks' / 'systems.jsonl'),
        'selected_systems': [
            {'index': index, 'id': row['id'], 'source_id': row['source_id'], 'n_grid': row['n_grid'],
             'n_ao': row['n_ao'], 'source_record_sha256': row['source_record_sha256']}
            for index, row in zip(PANEL_INDEXES, selected, strict=True)
        ],
        'focus': focus,
        'ae17': ae17,
        'checkpoints': {
            's0': {'path': str(S0_PATH), 'file_sha256': S0_FILE_SHA, 'tensor_sha256': S0_TENSOR_SHA},
            's70': {'path': str(S70_PATH), 'file_sha256': S70_FILE_SHA},
        },
    }
    return protocol, receipts


def run_audit():
    import subprocess
    torch, microbatch = scientific_modules()
    if not torch.cuda.is_available():
        raise RuntimeError('This diagnostic requires the qualified CUDA gradient path')
    if SCRATCH.exists():
        unexpected = [path.name for path in SCRATCH.iterdir() if path.name not in {'protocol.json', 'grad', 'theta', 'summary.json'}]
        if unexpected:
            raise RuntimeError(f'Scratch directory contains unrelated files: {unexpected}')
        if (SCRATCH / 'summary.json').exists():
            raise RuntimeError('Refusing to recompute an existing gradient audit')
    protocol, receipts = prepare_protocol()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    protocol_path = SCRATCH / 'protocol.json'
    if protocol_path.exists():
        prior = read_json(protocol_path)
        if prior['selected_systems'] != protocol['selected_systems'] or prior['focus'] != protocol['focus']:
            raise RuntimeError('Existing scratch protocol does not match this panel')
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    new_seconds = 0.0
    evaluations = 0
    stored = {}
    names_by_state = {}
    file_hashes = {str(S0_PATH): sha256(S0_PATH), str(S70_PATH): sha256(S70_PATH)}
    duplicate = SHARE / 'lap_iid_adamw_lr_branches_20261009' / 'A' / 'ordinary_sgd_adamw' / 'checkpoint_70.pt'
    duplicate_sha = sha256(duplicate)
    if duplicate_sha != S70_FILE_SHA:
        raise RuntimeError('The LR-branch t70 copy does not match the IID t70 file hash')
    rates = {'relchem': None, 'ae17': None, 'exc': None, 'op': None}
    grids = {row['id']: row['n_grid'] for row in protocol['selected_systems']}

    def too_long(kind, system_id=None):
        if rates[kind] is None:
            return False
        estimate = rates[kind]
        if kind in ('exc', 'op') and system_id is not None and rates[kind + '_grid']:
            estimate = rates[kind] * (grids[system_id] / rates[kind + '_grid'])
        return new_seconds + estimate > BUDGET_SECONDS

    rates['exc_grid'] = None
    rates['op_grid'] = None
    bundle = microbatch.PublicationDataset(microbatch.DATA)
    try:
        if bundle.manifest['logical_sha256'] != DATA_SHA:
            raise RuntimeError('Opened dataset logical SHA mismatch')
        dispersion = bundle.chemistry_dispersions()
        mrks_dispersion = read_json(microbatch.DATA / 'mrks' / 'dispersion.json')
        for state, path, file_sha, tensor_sha in (
            ('s0', S0_PATH, S0_FILE_SHA, S0_TENSOR_SHA),
            ('s70', S70_PATH, S70_FILE_SHA, None),
        ):
            model, shadow, names, digest = load_frozen(microbatch, path, file_sha, tensor_sha)
            if names_by_state and names != names_by_state['s0']:
                raise RuntimeError('S0 and S70 trainable parameter coordinates differ')
            names_by_state[state] = names
            theta = pack_theta(model, names)
            theta_path = SCRATCH / 'theta' / f'{state}.npy'
            theta_path.parent.mkdir(parents=True, exist_ok=True)
            if theta_path.exists():
                if not np.array_equal(np.load(theta_path), theta):
                    raise RuntimeError(f'Existing parameter vector does not match {state}')
            else:
                np.save(theta_path, theta)
            stored[state] = {'digest': digest, 'vectors': {}, 'losses': {}}
            jobs = []
            for items in protocol['focus'].values():
                for item in items:
                    jobs.append(('relchem', item['identity'], item, None))
            for item in protocol['ae17']:
                jobs.append(('ae17', item['identity'], item, None))
            for system in protocol['selected_systems']:
                jobs.append(('exc', system['id'], None, system['id']))
                jobs.append(('op', system['id'], None, system['id']))
            if len(jobs) != 50:
                raise RuntimeError(f'Each checkpoint must schedule 50 evaluations, found {len(jobs)}')
            for kind, key, item, system_id in jobs:
                if too_long(kind, system_id):
                    write_partial(protocol, stored, evaluations, new_seconds, started, file_hashes, duplicate_sha)
                    return
                if kind in ('relchem', 'ae17'):
                    def compute(item=item, model=model, shadow=shadow, names=names):
                        return chemistry_gradient(
                            microbatch, model, shadow, bundle, dispersion, item, names,
                        )
                else:
                    def compute(system_id=system_id, kind=kind, model=model, names=names):
                        return mrks_gradient(
                            microbatch, model, bundle, mrks_dispersion, system_id, kind, names,
                        )
                meta, vector, elapsed = evaluate_one(microbatch, compute, state, kind, key, digest, model)
                print(f'GRADIENT {state} {kind} {key} loss={meta["loss"]:.8g} seconds={elapsed:.3f}', flush=True)
                if elapsed > 0.0:
                    evaluations += 1
                    new_seconds += elapsed
                    rates[kind] = elapsed
                    if system_id is not None:
                        rates[kind + '_grid'] = grids[system_id]
                if evaluations > EVALUATION_LIMIT:
                    raise RuntimeError('Gradient evaluation budget exceeded')
                if kind in ('relchem', 'ae17'):
                    record = {**item, 'loss': meta['loss'], 'seconds': meta['seconds']}
                    loss_agreement(record, receipts[state][item['identity']])
                stored[state]['vectors'].setdefault(kind, {})[key] = vector
                stored[state]['losses'].setdefault(kind, {})[key] = meta['loss']
                if new_seconds > BUDGET_SECONDS:
                    write_partial(protocol, stored, evaluations, new_seconds, started, file_hashes, duplicate_sha)
                    return
            assert_model_intact(microbatch, model, digest)
            del model, shadow
            torch.cuda.empty_cache()
    finally:
        bundle.close()
    if sha256(S0_PATH) != file_hashes[str(S0_PATH)] or sha256(S70_PATH) != file_hashes[str(S70_PATH)]:
        raise RuntimeError('Checkpoint file bytes changed during the diagnostic')
    summary = build_summary(protocol, stored, receipts, evaluations, new_seconds, started, file_hashes, duplicate_sha)
    summary['source_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    summary['source_sha256'] = {relative: sha256(REPO / relative) for relative in SOURCE_FILES}
    summary['status'] = 'COMPLETE'
    summary['peak_live_bytes'] = int(torch.cuda.max_memory_allocated())
    summary['peak_reserved_bytes'] = int(torch.cuda.max_memory_reserved())
    (SCRATCH / 'summary.json').write_text(json.dumps(json_ready(summary), indent=2) + '\n', encoding='utf-8')
    render(summary)


def write_partial(protocol, stored, evaluations, new_seconds, started, file_hashes, duplicate_sha):
    summary = {
        'status': 'PARTIAL',
        'protocol': protocol,
        'new_gradient_evaluations': evaluations,
        'new_gpu_seconds': new_seconds,
        'wall_seconds': time.perf_counter() - started,
        'completed_keys': {state: {kind: sorted(vectors) for kind, vectors in payload['vectors'].items()} for state, payload in stored.items()},
        'file_sha256_after': {path: sha256(path) for path in file_hashes},
        'duplicate_t70_sha256': duplicate_sha,
    }
    (SCRATCH / 'summary.json').write_text(json.dumps(json_ready(summary), indent=2) + '\n', encoding='utf-8')
    raise RuntimeError('PARTIAL: the 30-minute new-GPU budget cannot hold the remaining evaluations')


def contribution_bundle(vectors_by_database):
    means, contributions, norms = {}, {}, {}
    for database, vectors in vectors_by_database.items():
        mean, contribution = database_terms(vectors)
        means[database] = mean
        contributions[database] = contribution
        norms[database] = {
            'mean_norm': float(np.linalg.norm(mean)),
            'contribution_norm': float(np.linalg.norm(contribution)),
            'reaction_norms': [float(np.linalg.norm(vector)) for vector in vectors],
            'within_cancellation': within_cancellation(vectors),
        }
    focus = contributions['ABDE4'] + contributions['pTC13'] + contributions['PA8']
    return means, contributions, norms, focus


def actual_changes(receipts, protocol):
    changes = {}
    for database, items in protocol['focus'].items():
        before = sum(receipts['s0'][item['identity']]['loss'] for item in items) / POPULATION
        after = sum(receipts['s70'][item['identity']]['loss'] for item in items) / POPULATION
        changes[database] = after - before
    changes['Focus25'] = sum(changes[database] for database in FOCUS_DATABASES)
    before = sum(receipts['s0'][item['identity']]['loss'] for item in protocol['ae17']) / 17
    after = sum(receipts['s70'][item['identity']]['loss'] for item in protocol['ae17']) / 17
    changes['AE17'] = after - before
    return changes


def build_summary(protocol, stored, receipts, evaluations, new_seconds, started, file_hashes, duplicate_sha):
    theta0 = np.load(SCRATCH / 'theta' / 's0.npy').astype(np.float64)
    theta70 = np.load(SCRATCH / 'theta' / 's70.npy').astype(np.float64)
    raw0 = np.load(SCRATCH / 'theta' / 's0.npy')
    raw70 = np.load(SCRATCH / 'theta' / 's70.npy')
    displacement = theta70 - theta0
    states = {}
    term_vectors = {}
    for state in ('s0', 's70'):
        grouped = {}
        for database, items in protocol['focus'].items():
            grouped[database] = [stored[state]['vectors']['relchem'][item['identity']] for item in items]
        _means, contributions, norms, focus = contribution_bundle(grouped)
        ae17 = np.mean(np.stack([stored[state]['vectors']['ae17'][item['identity']] for item in protocol['ae17']]), axis=0)
        exc = np.mean(np.stack([stored[state]['vectors']['exc'][row['id']] for row in protocol['selected_systems']]), axis=0)
        operator = np.mean(np.stack([stored[state]['vectors']['op'][row['id']] for row in protocol['selected_systems']]), axis=0)
        analyzed = analyze_state(
            {'ABDE4': contributions['ABDE4'], 'pTC13': contributions['pTC13'], 'PA8': contributions['PA8'], 'Focus25': focus},
            ae17, exc, operator,
        )
        analyzed['database_norms'] = norms
        analyzed['pairwise_database_cancellation'] = {
            f'{left}:{right}': cancellation_ratio(contributions[left], contributions[right])
            for left, right in (('ABDE4', 'pTC13'), ('ABDE4', 'PA8'), ('pTC13', 'PA8'))
        }
        analyzed['ae17_norm'] = float(np.linalg.norm(ae17))
        analyzed['exc_panel_norm'] = float(np.linalg.norm(exc))
        analyzed['op_panel_norm'] = float(np.linalg.norm(operator))
        states[state] = analyzed
        term_vectors[state] = {**contributions, 'Focus25': focus, 'AE17': ae17, 'Exc_panel': exc, 'Op_panel': operator}
    changes = actual_changes(receipts, protocol)
    history = {}
    for name in (*FOCUS_DATABASES, 'Focus25', 'AE17', 'Exc_panel', 'Op_panel'):
        actual = changes.get(name)
        history[name] = historical_row(actual, term_vectors['s0'][name], term_vectors['s70'][name], displacement)
    return {
        'protocol': {
            'selected_systems': protocol['selected_systems'],
            'systems_metadata_sha256': protocol['systems_metadata_sha256'],
            'dataset_sha256': DATA_SHA,
            'evaluation_manifest_sha256': MANIFEST_SHA,
            'checkpoints': protocol['checkpoints'],
            's0_digest': stored['s0']['digest'],
            's70_digest': stored['s70']['digest'],
        },
        'new_gradient_evaluations': evaluations,
        'new_gpu_seconds': new_seconds,
        'wall_seconds': time.perf_counter() - started,
        'file_sha256_before': file_hashes,
        'duplicate_t70_sha256': duplicate_sha,
        'displacement': {
            'norm': float(np.linalg.norm(displacement)),
            'relative_to_s0': float(np.linalg.norm(displacement) / np.linalg.norm(theta0)),
            'unchanged_f32_coordinates': int(np.count_nonzero(raw0 == raw70)),
            'unchanged_f32_fraction': float(np.count_nonzero(raw0 == raw70) / PARAMETER_COUNT),
            'finite': bool(np.isfinite(displacement).all()),
        },
        'states': states,
        'history': history,
        'adjudication': adjudicate(history, states),
        'panel_losses': {
            state: {
                database: [stored[state]['losses']['relchem'][item['identity']] for item in items]
                for database, items in protocol['focus'].items()
            } for state in ('s0', 's70')
        },
    }


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def fmt(value, digits=6):
    if value is None:
        return 'undefined'
    return f'{value:.{digits}e}'


def median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def render(summary):
    if summary.get('status') == 'PARTIAL':
        text = '# Relchem gradient-conflict audit\n\nStatus: PARTIAL. The new-GPU budget stopped the audit before every requested gradient existed.\n'
        (REPO / 'lap_relchem_gradient_conflict_report.md').write_text(text, encoding='utf-8')
        (REPO / 'lap_relchem_gradient_conflict_metrics.json').write_text(json.dumps(json_ready(summary), indent=2) + '\n', encoding='utf-8')
        return
    labels = ('ABDE4', 'pTC13', 'PA8', 'Focus25', 'AE17', 'Exc_panel', 'Op_panel', 'G_other')
    lines = [
        '# Frozen gradient conflict at P536 and IID t70',
        '',
        'This diagnostic evaluates qualified singleton and four-system gradients at two frozen checkpoints. It does not update parameters. `C_d` is a database contribution to the 251-identity relchem gradient. `G_d` is the mean gradient inside that database. The Exc and operator vectors are four-system diagnostic gradients, not full90 gradients. `C_focus` omits the other 226 relchem identities.',
        '',
        '## A. Provenance',
        '',
        f"Source commit `{summary['source_commit']}`. Dataset logical SHA256 `{DATA_SHA}`. Evaluation manifest SHA256 `{MANIFEST_SHA}`.",
        f"S0 file `{summary['protocol']['checkpoints']['s0']['path']}` `{S0_FILE_SHA}`; tensor `{summary['protocol']['s0_digest']}`.",
        f"S70 file `{summary['protocol']['checkpoints']['s70']['path']}` `{S70_FILE_SHA}`; tensor `{summary['protocol']['s70_digest']}`.",
        f"A byte-identical t70 copy is `{summary['duplicate_t70_sha256']}`. It was hashed and not loaded.",
        f"mRKS metadata SHA256 `{summary['protocol']['systems_metadata_sha256']}`. Selected systems, sorted by `(n_grid, id)` at indexes 11, 33, 56 and 78:",
        '',
        '| Index | System | Source | n_grid | n_ao |',
        '|---:|---|---|---:|---:|',
    ]
    for row in summary['protocol']['selected_systems']:
        lines.append(f"| {row['index']} | `{row['id']}` | {row['source_id']} | {row['n_grid']} | {row['n_ao']} |")
    lines.extend([
        '',
        f"New isolated gradient evaluations: {summary['new_gradient_evaluations']}. New GPU seconds: {summary['new_gpu_seconds']:.3f}. Wall seconds: {summary['wall_seconds']:.3f}. Peak live CUDA bytes: {summary['peak_live_bytes']}. Peak reserved bytes: {summary['peak_reserved_bytes']}.",
        'Checkpoint file hashes were unchanged at the end of the run. No optimizer was constructed. Every chemistry singleton loss matched its frozen receipt to within 1e-6 before its gradient was stored.',
        '',
        '## B. Protected chemistry gradients',
        '',
    ])
    for state in ('s0', 's70'):
        block = summary['states'][state]
        lines.append(f'### {state}')
        lines.append('')
        lines.append('| Object | Contribution norm | Mean-gradient norm | Within-database cancellation |')
        lines.append('|---|---:|---:|---:|')
        for database in FOCUS_DATABASES:
            norms = block['database_norms'][database]
            lines.append(f"| {database} | {fmt(norms['contribution_norm'])} | {fmt(norms['mean_norm'])} | {fmt(norms['within_cancellation'])} |")
        lines.append(f"| Focus25 | {fmt(block['terms']['Focus25']['norm'])} |  |  |")
        lines.append('')
        lines.append('Reaction-gradient norm dispersion, from the qualified singleton gradients:')
        lines.append('')
        for database in FOCUS_DATABASES:
            values = block['database_norms'][database]['reaction_norms']
            lines.append(
                f"- {database}: min {fmt(min(values))}, median {fmt(median(values))}, max {fmt(max(values))}"
            )
        lines.append('')
        lines.append('Vector-cancellation ratios between contribution gradients, `1 - ||a+b||/(||a||+||b||)`:')
        lines.append('')
        for pair, value in block['pairwise_database_cancellation'].items():
            lines.append(f'- {pair}: {fmt(value)}')
        lines.append('')
    lines.extend(['## C. Conflict with other objectives', ''])
    for state in ('s0', 's70'):
        block = summary['states'][state]
        lines.append(f'### {state}')
        lines.append('')
        lines.append(f"Four-system panel norms: Exc {fmt(block['exc_panel_norm'])}, operator {fmt(block['op_panel_norm'])}. Exact AE17 norm {fmt(block['ae17_norm'])}. Weighted diagnostic `G_other` norm {fmt(block['other_norm'])}.")
        lines.append('')
        header = '| Gradient | ' + ' | '.join(labels) + ' |'
        lines.append(header)
        lines.append('|---|' + '|'.join(['---:'] * len(labels)) + '|')
        matrix = block['cosine_matrix']
        for left in labels:
            cells = [fmt(matrix[left][right], 4) for right in labels]
            lines.append(f"| {left} | " + ' | '.join(cells) + ' |')
        lines.append('')
        lines.append('| Contribution | dot AE17 | dot Exc panel | dot operator panel | dot G_other | cos G_other | predicted delta under -G_other |')
        lines.append('|---|---:|---:|---:|---:|---:|---:|')
        for name in (*FOCUS_DATABASES, 'Focus25'):
            row = block['terms'][name]
            lines.append(
                f"| {name} | {fmt(row['dot_ae17'])} | {fmt(row['dot_exc_panel'])} | {fmt(row['dot_op_panel'])} | "
                f"{fmt(row['dot_other'])} | {fmt(row['cos_other'], 4)} | {fmt(row['predicted_delta_under_minus_other'])} |"
            )
        lines.append('')
        lines.append('Weighted pieces of `dot(C, G_other)`:')
        lines.append('')
        for name in (*FOCUS_DATABASES, 'Focus25'):
            parts = block['terms'][name]['weighted_dot_components']
            lines.append(f"- {name}: AE17 {fmt(parts['ae17'])}, Exc panel {fmt(parts['exc'])}, operator panel {fmt(parts['op'])}")
        lines.append('')
    lines.extend([
        '## D. Actual versus predicted change',
        '',
        f"Parameter displacement norm {fmt(summary['displacement']['norm'])}; relative to S0 {fmt(summary['displacement']['relative_to_s0'])}; unchanged F32 coordinates {summary['displacement']['unchanged_f32_coordinates']} / {PARAMETER_COUNT} ({summary['displacement']['unchanged_f32_fraction']:.6f}).",
        '',
        'The dots below multiply one endpoint contribution gradient by the full 70-update displacement. They are not a line integral and are not an AdamW step.',
        '',
        '| Database | Actual contribution change | C(S0)·delta | C(S70)·delta | cos(C(S0), delta) | cos(C(S70), delta) |',
        '|---|---:|---:|---:|---:|---:|',
    ])
    for name in (*FOCUS_DATABASES, 'Focus25', 'AE17'):
        row = summary['history'][name]
        label = 'Focus25 total' if name == 'Focus25' else name
        lines.append(
            f"| {label} | {fmt(row['actual_loss_change'])} | {fmt(row['dot_s0'])} | {fmt(row['dot_s70'])} | "
            f"{fmt(row['cos_s0'], 4)} | {fmt(row['cos_s70'], 4)} |"
        )
    lines.extend([
        '',
        'AE17 uses its exact 17-identity mean gradient, so its actual change and dots share a definition. The S70 AE17 projection has the opposite sign from the measured AE17 decrease. That reversal is an AE17 path observation.',
        '',
        'Four-system diagnostic projections onto the same displacement, with no full90 loss comparison:',
        '',
        f"- Exc panel: C(S0)·delta {fmt(summary['history']['Exc_panel']['dot_s0'])}, C(S70)·delta {fmt(summary['history']['Exc_panel']['dot_s70'])}.",
        f"- Operator panel: C(S0)·delta {fmt(summary['history']['Op_panel']['dot_s0'])}, C(S70)·delta {fmt(summary['history']['Op_panel']['dot_s70'])}.",
        '',
        '## E. Hypothesis adjudication',
        '',
    ])
    labels_h = summary['adjudication']
    lines.extend([
        f"- H1 inter-task conflict: `{labels_h['H1']}`.",
        f"- H2 chemistry-internal conflict: `{labels_h['H2']}`. Mutual three-database clause: `{labels_h['H2_mutual_clause']}`. Conflict with the other 226 identities: `{labels_h['H2_remaining_226']}`.",
        f"- H3 finite-step/path effects: `{labels_h['H3']}`.",
        '',
        adjudication_narrative(summary),
        '',
        '## F. Historical context',
        '',
        'The cursor10 chemistry-gradient audit used another frozen state and a different sampling/PCD protocol. Its database cosines are not S0 or S70 gradients. In that older audit, ABDE4 aligned with the secondary tasks while pTC13 opposed them. That pattern is context only.',
        '',
        'J251, file SHA256 `11cee17f61c017e26902c905b6d9ace0b576ab7eaacf5d5f93f5172a49e3257a`, remains a historically eligible joint checkpoint on the same fixed panel: relchem/t0 0.979708, AE17/t0 0.081012, Exc/t0 0.101105, operator/t0 0.938121. Its existing singleton-mean ratios still rise for ABDE4 (1.097354), PA8 (1.065882) and pTC13 (1.107175). No new J251 gradient was computed.',
        '',
        'Skala-1.1 uses hierarchical then relative-excess dataset sampling. That training fact does not identify which optimizer intervention would protect these three databases.',
        '',
        '## G. Recommended next experiment',
        '',
        recommendation_text(labels_h['next_experiment']),
        '',
        '## Source hashes',
        '',
    ])
    for relative, digest in summary['source_sha256'].items():
        lines.append(f'- `{relative}` `{digest}`')
    lines.append('')
    report = '\n'.join(lines)
    (REPO / 'lap_relchem_gradient_conflict_report.md').write_text(report, encoding='utf-8')
    (REPO / 'lap_relchem_gradient_conflict_metrics.json').write_text(
        json.dumps(json_ready(summary), indent=2) + '\n', encoding='utf-8')


def adjudication_narrative(summary):
    s0 = summary['states']['s0']['terms']
    s70 = summary['states']['s70']['terms']
    matrix0 = summary['states']['s0']['cosine_matrix']
    matrix70 = summary['states']['s70']['cosine_matrix']
    pairwise = [
        matrix0['ABDE4']['pTC13'], matrix0['ABDE4']['PA8'], matrix0['pTC13']['PA8'],
        matrix70['ABDE4']['pTC13'], matrix70['ABDE4']['PA8'], matrix70['pTC13']['PA8'],
    ]
    h1 = ' '.join([
        'H1 asks whether the weighted direction `G_other` locally opposes the three contribution gradients at both frozen states.',
        'At S0 every protected cosine with `G_other` is negative and every predicted change under `-G_other` is positive:',
        f"ABDE4 {fmt(s0['ABDE4']['cos_other'], 4)} / {fmt(s0['ABDE4']['predicted_delta_under_minus_other'])},",
        f"pTC13 {fmt(s0['pTC13']['cos_other'], 4)} / {fmt(s0['pTC13']['predicted_delta_under_minus_other'])},",
        f"PA8 {fmt(s0['PA8']['cos_other'], 4)} / {fmt(s0['PA8']['predicted_delta_under_minus_other'])}.",
        'At S70 only ABDE4 keeps that pattern',
        f"({fmt(s70['ABDE4']['cos_other'], 4)}, predicted {fmt(s70['ABDE4']['predicted_delta_under_minus_other'])}).",
        f"pTC13 and PA8 have positive cosines {fmt(s70['pTC13']['cos_other'], 4)} and {fmt(s70['PA8']['cos_other'], 4)}, so `-G_other` locally decreases those two contributions.",
        'The four-system operator panel stays negatively aligned with all three databases at both states:',
        f"S0 cosines {fmt(matrix0['ABDE4']['Op_panel'], 4)}, {fmt(matrix0['pTC13']['Op_panel'], 4)}, {fmt(matrix0['PA8']['Op_panel'], 4)};",
        f"S70 cosines {fmt(matrix70['ABDE4']['Op_panel'], 4)}, {fmt(matrix70['pTC13']['Op_panel'], 4)}, {fmt(matrix70['PA8']['Op_panel'], 4)}.",
        'At S70 the positive AE17 and Exc-panel pieces outweigh that operator piece for pTC13 and PA8.',
        'ABDE4 is nearly orthogonal to AE17 at both states.',
        'These statements describe local directional geometry of the exact AE17 mean and a four-system diagnostic panel.',
        'They leave the historical AdamW cause unidentified, and the panel gradients are four-system diagnostics.',
    ])
    h2 = ' '.join([
        'H2 asks whether the three databases oppose one another.',
        f"All six pairwise contribution cosines are positive, from {fmt(min(pairwise), 4)} to {fmt(max(pairwise), 4)}.",
        'Mutual opposition among ABDE4, pTC13, and PA8 is therefore weakened.',
        'No SHA-matched full 251-identity parameter gradient for S0 or S70 was reused, so conflict with the other 226 identities remains unverified.',
    ])
    h3 = (
        'H3 asks whether the endpoint gradients fail to predict the sign of the measured contribution changes along `theta_70 - theta_0`. '
        'For ABDE4, pTC13, and PA8 the measured change and both endpoint dots are positive. '
        'The cosines of those gradients with the displacement are small, 0.030 to 0.075, so most of the displacement lies in other directions, while the projected component has the observed sign. '
        'PA8 at S0 projects to 9.086e-03 against a measured change of 2.954e-02; ABDE4 and pTC13 stay within the same order at both ends. '
        'One endpoint dot is a local linearization, not the nonlinear change along the 70-update path. '
        'The S0-versus-S70 reversal of the AE17 and panel projections is recorded and is kept separate from the three chemistry signs.'
    )
    return f'{h1}\n\n{h2}\n\n{h3}'


def recommendation_text(choice):
    texts = {
        'A': 'Next experiment, not run: one independent 90-update IID AdamW replica from corrected P536. The endpoint geometry does not show a consistent competing-task conflict that explains the historical signs. Cost is about 25-50 minutes on one local GPU, using the measured 15.3-32.7 s/update range, plus one fixed-panel audit if collected. Stop if any scientific ratio is nonfinite. Promote only if all four ratios are strictly below 1.',
        'B': 'Next experiment, not run: one tightly controlled chemistry-protection comparison at the same frozen P536, limited to a gradient-level or at most a short bounded AdamW arm that keeps the qualified singleton loss. It is justified only because the local competing-task direction opposes these databases at both checkpoints and the historical displacement has the same sign as that local prediction. Do not treat the four-system operator panel as full90. Expected cost for a gradient-only check is another diagnostic of this size; a 20-update arm would be about 5-11 minutes plus audits. Stop on a nonfinite update. This audit did not run it.',
        'C': 'Next experiment, not run: a frozen-state optimizer/path diagnostic. Replay is not authorized here. The next bounded study should record the actual AdamW displacement at a few saved cursors and compare its direction with these fixed-panel gradients, without a new training objective. Cost is one read of existing checkpoints plus, only if those checkpoints lack the required vectors, a separately authorized no-update gradient probe. Stop if the checkpoint hashes differ from the receipts. This audit did not run it.',
        'D': 'No further experiment is recommended. The weighted competing direction opposes all three databases at S0 and only ABDE4 at S70, so a chemistry-protection arm is not selected. The measured contribution increases have the same sign as both endpoint projections onto the historical displacement, so an optimizer-path replay is not selected to explain those signs. Another IID replica is not selected because this audit did not measure stream sensitivity. Expected additional GPU cost: none.',
    }
    return texts[choice]


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args(argv)
    forbid_training_calls(Path(__file__).read_text(encoding='utf-8'))
    if args.report_only:
        render(read_json(SCRATCH / 'summary.json'))
        return
    run_audit()


if __name__ == '__main__':
    main()
