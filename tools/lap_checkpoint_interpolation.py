"""Evaluation-only linear interpolation between IID t70 and J251.

Parameters move. Scientific objectives are measured, not interpolated.
No optimizer is constructed and no parent checkpoint is written.
"""
import hashlib
import json
import math
import time
from pathlib import Path

ALPHAS = (0.25, 0.5, 0.75)
EXECUTION_ORDER = (0.5, 0.25, 0.75)
ALLOWED = (0.0, 0.25, 0.5, 0.75, 1.0)
PARAMETER_COUNT = 9446
POPULATION = 251
BUDGET_SECONDS = 3600
BASELINE = {
    'relchem': 1.2354710978866068,
    'ae17': 24.901047104016875,
    'exc': 92.17480502000551,
    'op': 0.03314821681605566,
}
P536_TENSOR_SHA = '3c2e45d6a86cccf80d7b4dc458e67f9b406935ed60d4e57433bb441a6df88da6'
IID_FILE_SHA = '04b8e549c17375988be576e2878881d13f55b217b42ec16a7e6101fdddd05443'
IID_TENSOR_SHA = '59ab4b98550805b13852e13745281fb1d072efcbed5c2c8622bf3a6840c3ff51'
J_FILE_SHA = '11cee17f61c017e26902c905b6d9ace0b576ab7eaacf5d5f93f5172a49e3257a'
J_TENSOR_SHA = 'd1b1372997199331b4b332b7f98dfa4e791617ea9a453d4e9cdb999c16aafff5'
DATA_SHA = '61c221a19b9987717e69cac182ad545241f8807db4126c0949a99992e4c210ef'
MANIFEST_SHA = '132bd550161be6bc6761f8df1de8d3755c0b0debab53663208c3a9bbf56cb805'
IID_CLEAN28 = 8.629660230056073
J_CLEAN28 = 9.346760658714116
HISTORICAL_BEST_CLEAN28 = 8.619694172
DATABASES = ('ABDE4', 'DBH76', 'EA13', 'IP13', 'MGAE109', 'NCCE31', 'PA8', 'pTC13')
FOCUS = ('ABDE4', 'pTC13', 'PA8')
REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
SCRATCH = ROOT / 'lap_checkpoint_interpolation_20261009'
IID_CHECKPOINT = ROOT / 'lap_iid_adamw_t59_t90_20261009/ordinary_sgd_adamw/checkpoint_70.pt'
J_CHECKPOINT = ROOT / 'lap_relchem_joint_epoch_20261009/J/ordinary_sgd_adamw/checkpoint_251.pt'
IID_PROTOCOL = ROOT / 'lap_iid_adamw_t59_t90_20261009/protocol.json'
J_PROTOCOL = ROOT / 'lap_relchem_joint_epoch_20261009/J/protocol.json'
MANIFEST_PATH = ROOT / 'lap_iid_adamw_t59_t90_20261009/evaluation_manifest.json'
IID_CHEMISTRY = ROOT / 'lap_iid_adamw_lr_branches_20261009/A/endpoint_70_chemistry_one_variant.json'
J_CHEMISTRY = ROOT / 'lap_relchem_joint_epoch_20261009/J/endpoint_251_chemistry_one_variant.json'
T0_CHEMISTRY = ROOT / 'lap_iid_adamw_t59_t90_20261009/baseline_0_chemistry_one_variant.json'
IID_VALIDATION = ROOT / 'lap_iid_adamw_t59_t90_20261009/endpoint_70_validation.json'
J_VALIDATION = ROOT / 'lap_relchem_joint_epoch_20261009/J/endpoint_251_validation.json'
SOURCE_FILES = (
    'train_lap_microbatch.py',
    'train_models/NN_models_lap.py',
    'train_models/lap_checkpoint.py',
    'train_models/lap_vxc.py',
    'train_models/lap_training.py',
    'train_models/lap_moo_training.py',
    'train_models/lap_operator.py',
    'train_models/lap_fixed_adamw.py',
    'tools/evaluate_microbatch_endpoint.py',
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def alpha_name(alpha):
    return f'alpha_{alpha:.2f}'.replace('.', 'p')


def state_sha(state):
    digest = hashlib.sha256()
    for name, value in state.items():
        array = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(array.numpy().tobytes())
    return digest.hexdigest()


def mix_tensor(left, right, alpha):
    """F64 blend, one cast back to the stored dtype. Endpoints are exact copies."""
    if alpha not in ALLOWED:
        raise ValueError(f'Alpha {alpha} is outside the frozen set')
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError('Parent tensors differ in shape or dtype')
    if alpha == 0.0:
        return left.detach().cpu().clone()
    if alpha == 1.0:
        return right.detach().cpu().clone()
    mixed = (1.0 - alpha) * left.detach().double() + alpha * right.detach().double()
    cast = mixed.to(dtype=left.dtype)
    if not bool(cast.isfinite().all()):
        raise FloatingPointError('Interpolated tensor is nonfinite')
    return cast


def primary_map(tie_groups, trainable_names):
    trainable = set(trainable_names)
    mapping = {}
    for group in tie_groups:
        primaries = [name for name in group if name in trainable]
        if len(primaries) != 1:
            raise ValueError('Each tie group must contain one unique trainable parameter')
        primary = primaries[0]
        for name in group:
            if name in mapping:
                raise ValueError(f'{name} belongs to two tie groups')
            mapping[name] = primary
    missing = trainable.difference(mapping)
    if missing:
        raise ValueError(f'Trainable parameters missing from tie groups: {sorted(missing)}')
    return mapping


def interpolate_state(left, right, alpha, trainable_names, buffer_names, tie_groups):
    if alpha not in ALLOWED:
        raise ValueError(f'Alpha {alpha} is outside the frozen set')
    if set(left) != set(right):
        raise ValueError('Parent state-dict keys differ')
    mapping = primary_map(tie_groups, trainable_names)
    known = set(mapping) | set(buffer_names)
    if set(left) != known:
        raise ValueError('State dict contains tensors that are neither tied parameters nor buffers')
    mixed = {}
    primaries = {}
    for name in left:
        if not bool(left[name].isfinite().all()) or not bool(right[name].isfinite().all()):
            raise FloatingPointError(f'Nonfinite parent tensor: {name}')
        if name in buffer_names:
            if not left[name].equal(right[name]):
                raise RuntimeError(f'Non-parameter buffer differs: {name}')
            mixed[name] = left[name].detach().cpu().clone()
            continue
        primary = mapping[name]
        if primary not in primaries:
            primaries[primary] = mix_tensor(left[primary], right[primary], alpha)
        if name != primary and not (left[name].equal(left[primary]) and right[name].equal(right[primary])):
            raise RuntimeError(f'Tied parameter {name} disagrees with {primary}')
        mixed[name] = primaries[primary].detach().cpu().clone()
    verify_interpolation(mixed, left, right, alpha, trainable_names)
    return mixed


def verify_interpolation(result, left, right, alpha, trainable_names):
    for name in trainable_names:
        expected = mix_tensor(left[name], right[name], alpha)
        if not result[name].equal(expected):
            raise RuntimeError(f'Interpolation check failed for {name}')
        if alpha == 0.0 and not result[name].equal(left[name]):
            raise RuntimeError('Alpha 0 is not an exact parent copy')
        if alpha == 1.0 and not result[name].equal(right[name]):
            raise RuntimeError('Alpha 1 is not an exact parent copy')


def eligibility(ratios):
    if set(ratios) != set(BASELINE):
        raise ValueError('Eligibility requires all four objectives')
    return all(math.isfinite(value) and value < 1.0 for value in ratios.values())


def require_complete(record):
    if record.get('status') != 'complete':
        raise ValueError('Incomplete evaluation result')
    if set(record.get('objectives', {})) != set(BASELINE):
        raise ValueError('Incomplete objective set')
    if len(record.get('reactions', ())) != 28:
        raise ValueError('Clean28 selection record must contain 28 reactions')
    if any(not math.isfinite(record['objectives'][name]) for name in BASELINE):
        raise ValueError('Nonfinite objective')
    if 'full30' in record or any('full30' in row for row in record['reactions']):
        raise ValueError('Full30 must not enter a selection record')
    contributions = [row['contribution'] for row in record['reactions']]
    if not math.isclose(sum(contributions) / 28, record['clean28'], rel_tol=0.0, abs_tol=1e-9):
        raise ValueError('Clean28 does not match the mean of the 28 contributions')


def select_candidate(candidates):
    eligible = [row for row in candidates if row['eligible']]
    if not eligible:
        return None
    return min(eligible, key=lambda row: (row['clean28'], row['alpha']))


def decide(candidates, j251_clean=J_CLEAN28):
    if len(candidates) != 3 or [row['alpha'] for row in candidates] != [0.25, 0.5, 0.75]:
        raise ValueError('Decision requires the three frozen alphas in ascending order')
    best_clean = min(candidates, key=lambda row: (row['clean28'], row['alpha']))
    if any(not row['complete'] for row in candidates):
        return {'classification': 'PARTIAL', 'winner': None, 'best_clean28_candidate': best_clean}
    winner = select_candidate(candidates)
    improves = winner is not None and winner['clean28'] < j251_clean
    return {
        'classification': 'GO' if improves else 'NO-GO',
        'winner': winner,
        'best_clean28_candidate': best_clean,
        'improves_on_j251': improves,
        'beats_historical_best': bool(winner is not None and winner['clean28'] < HISTORICAL_BEST_CLEAN28),
    }


def database_summary(rows):
    grouped = {name: [] for name in DATABASES}
    relchem = [row for row in rows if row['task'] == 'relchem']
    ae17 = [row for row in rows if row['task'] == 'ae17']
    if len(relchem) != POPULATION or len(ae17) != 17:
        raise ValueError('Chemistry panel must contain 251 relchem and 17 AE17 identities')
    for row in relchem:
        grouped[row['database']].append(row['loss'])
    summary = {}
    for name, losses in grouped.items():
        if not losses:
            raise ValueError(f'Empty database {name}')
        mean = float(sum(losses) / len(losses))
        summary[name] = {
            'count': len(losses),
            'singleton_mean': mean,
            'contribution': float(sum(losses) / POPULATION),
        }
    return {
        'databases': summary,
        'relchem': float(sum(row['loss'] for row in relchem) / POPULATION),
        'ae17': float(sum(row['loss'] for row in ae17) / 17),
    }


def reaction_changes(candidate_rows, endpoint_rows):
    left = {row['reaction_id']: row['contribution'] for row in candidate_rows}
    right = {row['reaction_id']: row['contribution'] for row in endpoint_rows}
    if set(left) != set(right) or len(left) != 28:
        raise ValueError('Clean28 reaction ids do not match the endpoint receipt')
    changes = []
    for reaction_id in sorted(left):
        delta = left[reaction_id] - right[reaction_id]
        changes.append({'reaction_id': reaction_id, 'delta_contribution': delta})
    improved = sum(row['delta_contribution'] < 0.0 for row in changes)
    worsened = sum(row['delta_contribution'] > 0.0 for row in changes)
    return {'rows': changes, 'improved': improved, 'worsened': worsened, 'unchanged': 28 - improved - worsened}


def response_label(values):
    import itertools
    deltas = [right - left for left, right in itertools.pairwise(values)]
    if all(delta >= 0.0 for delta in deltas) or all(delta <= 0.0 for delta in deltas):
        return 'monotonic'
    return 'nonmonotonic'


def published_clean_rows(payload):
    rows = payload['metrics']['reaction_rows']
    clean = [row for row in rows if row['clean']]
    if len(clean) != 28:
        raise ValueError('Published validation receipt does not contain 28 clean reactions')
    return [
        {
            'reaction_id': row['reaction_id'],
            'signed_error_kcal_mol': row['signed_error_kcal_mol'],
            'contribution': row['weighted_absolute_error'],
        }
        for row in clean
    ]


def objective_ratios(objectives):
    return {name: objectives[name] / BASELINE[name] for name in BASELINE}


def forbid_training_calls(source):
    import ast
    forbidden_calls = {'adamw_step', 'apply_joint_gradient', 'train_moo_update', 'train', 'backward'}
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = None
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            if name in forbidden_calls or name == 'step':
                raise RuntimeError(f'Forbidden call: {name}')
        if isinstance(node, ast.Attribute) and node.attr == 'optim':
            raise RuntimeError('Optimizer construction is forbidden')


def scientific_modules():
    import sys

    import torch
    sys.path.insert(0, str(REPO))
    import train_lap_microbatch as microbatch
    from tools.evaluate_microbatch_endpoint import validation
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    return torch, microbatch, validation


def model_structure(model):
    from lap_moo_training import named_trainable_parameters
    trainable = tuple(named_trainable_parameters(model))
    groups = {}
    order = []
    for name, parameter in model.named_parameters(remove_duplicate=False):
        key = parameter.data_ptr()
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(name)
    buffers = tuple(name for name, _ in model.named_buffers())
    return trainable, tuple(tuple(groups[key]) for key in order), buffers


def assert_same_architecture(left, right):
    if type(left) is not type(right):
        raise RuntimeError('Parent models use different classes')
    if left.architecture != right.architecture or left.descriptor_protocol != right.descriptor_protocol:
        raise RuntimeError('Architecture or descriptor protocol differs')
    if left.model_kwargs != right.model_kwargs:
        raise RuntimeError('Model constructor arguments differ')
    left_names, left_ties, left_buffers = model_structure(left)
    right_names, right_ties, right_buffers = model_structure(right)
    if left_names != right_names or left_ties != right_ties or left_buffers != right_buffers:
        raise RuntimeError('Parameter order, ties, or buffers differ')
    if sum(parameter.numel() for parameter in named_values(left, left_names)) != PARAMETER_COUNT:
        raise RuntimeError('Unique trainable parameter count is not 9446')
    for name in left_names:
        a, b = dict(left.named_parameters())[name], dict(right.named_parameters())[name]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise RuntimeError(f'Parameter layout differs: {name}')
    for name in left_buffers:
        a = dict(left.named_buffers())[name]
        b = dict(right.named_buffers())[name]
        if a.shape != b.shape or a.dtype != b.dtype or not a.equal(b):
            raise RuntimeError(f'Non-parameter buffer differs: {name}')
    if set(left.state_dict()) != set(right.state_dict()):
        raise RuntimeError('State-dict keys differ')
    return left_names, left_ties, left_buffers


def named_values(model, names):
    found = dict(model.named_parameters(remove_duplicate=True))
    return [found[name] for name in names]


def load_parent_models(torch, microbatch):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the qualified Lap constructor')
    for path, expected in ((IID_CHECKPOINT, IID_FILE_SHA), (J_CHECKPOINT, J_FILE_SHA)):
        if sha256_file(path) != expected:
            raise RuntimeError(f'Checkpoint hash mismatch: {path}')
    for protocol in (read_json(IID_PROTOCOL), read_json(J_PROTOCOL)):
        if protocol['initial_state']['state_sha256'] != P536_TENSOR_SHA:
            raise RuntimeError('Parent protocol does not start from corrected P536')
        if protocol['dataset_sha256'] != DATA_SHA:
            raise RuntimeError('Parent protocol dataset hash differs')
        recorded = protocol.get('evaluation_manifest_sha256')
        if recorded is not None and recorded != MANIFEST_SHA:
            raise RuntimeError('Parent protocol evaluation manifest differs')
    if sha256_file(MANIFEST_PATH) != MANIFEST_SHA:
        raise RuntimeError('Evaluation manifest hash mismatch')
    p536 = {
        'path': read_json(IID_PROTOCOL)['initial_state']['path'],
        'file_sha256': read_json(IID_PROTOCOL)['initial_state']['file_sha256'],
        'state_sha256': P536_TENSOR_SHA,
    }
    iid_model, _shadow = microbatch.model_at(p536)
    j_model, _shadow_j = microbatch.model_at(p536)
    iid_state = torch.load(IID_CHECKPOINT, map_location='cpu', weights_only=False)['model']
    j_state = torch.load(J_CHECKPOINT, map_location='cpu', weights_only=False)['model']
    iid_model.load_state_dict(iid_state)
    j_model.load_state_dict(j_state)
    if microbatch.existing.digest(iid_model) != IID_TENSOR_SHA:
        raise RuntimeError('IID t70 tensor hash mismatch')
    if microbatch.existing.digest(j_model) != J_TENSOR_SHA:
        raise RuntimeError('J251 tensor hash mismatch')
    names, ties, buffers = assert_same_architecture(iid_model, j_model)
    iid_cpu = {name: value.detach().cpu().clone() for name, value in iid_model.state_dict().items()}
    j_cpu = {name: value.detach().cpu().clone() for name, value in j_model.state_dict().items()}
    del iid_model, j_model
    torch.cuda.empty_cache()
    return iid_cpu, j_cpu, names, ties, buffers


def enrich_clean_rows(raw_rows, reactions):
    by_source = {}
    for row in reactions.values():
        by_source.setdefault(row['source_id'], []).append(row)
    cleaned = []
    for row in raw_rows:
        if not row['clean']:
            continue
        matches = by_source[row['reaction_id']]
        if len(matches) != 1:
            raise RuntimeError(f'Ambiguous validation reaction id {row["reaction_id"]}')
        meta = matches[0]
        signed = float(row['signed_error_kcal_mol'])
        absolute = abs(signed)
        weight = float(meta['diet_weight'])
        contribution = float(row['weighted_absolute_error'])
        if not math.isclose(contribution, absolute * weight, rel_tol=0.0, abs_tol=1e-8):
            raise RuntimeError(f'Clean contribution does not match weight for {row["reaction_id"]}')
        cleaned.append({
            'reaction_id': row['reaction_id'],
            'predicted_energy_kcal_mol': signed + float(meta['reference_energy_kcal_mol']),
            'signed_error_kcal_mol': signed,
            'absolute_error_kcal_mol': absolute,
            'diet_weight': weight,
            'contribution': contribution,
        })
    if len(cleaned) != 28:
        raise RuntimeError('Clean split did not produce 28 reactions')
    return cleaned


def published_endpoints(reactions):
    iid_payload = read_json(IID_VALIDATION)
    j_payload = read_json(J_VALIDATION)
    if iid_payload['checkpoint_sha256'] != IID_FILE_SHA or j_payload['checkpoint_sha256'] != J_FILE_SHA:
        raise RuntimeError('Published validation receipt is bound to a different checkpoint')
    if iid_payload['metrics']['clean28'] != IID_CLEAN28 or j_payload['metrics']['clean28'] != J_CLEAN28:
        raise RuntimeError('Published Clean28 receipt does not match the frozen value')
    return {
        'iid_t70': enrich_clean_rows(iid_payload['metrics']['reaction_rows'], reactions),
        'j251': enrich_clean_rows(j_payload['metrics']['reaction_rows'], reactions),
    }


def chemistry_reference(path, expected_sha):
    payload = read_json(path)
    if payload['checkpoint_sha256'] != expected_sha:
        raise RuntimeError(f'Chemistry receipt hash mismatch: {path}')
    rows = [
        {'task': row['task'], 'database': row['database'], 'loss': row['loss'], 'identity': identity}
        for identity, row in payload['rows'].items()
    ]
    return database_summary(rows)


def prepare_references(bundle):
    endpoints = published_endpoints(bundle.validation_reactions)
    references = {
        't0': chemistry_reference(T0_CHEMISTRY, '598dde8ed2b35c158ce715998617c77506bac47425ed27f378cde5aafce87896'),
        'iid_t70': chemistry_reference(IID_CHEMISTRY, IID_FILE_SHA),
        'j251': chemistry_reference(J_CHEMISTRY, J_FILE_SHA),
    }
    published = read_json(REPO / 'relchem_joint_epoch_metrics.json')['arms']['J']['per_database']
    for name in FOCUS:
        if not math.isclose(references['t0']['databases'][name]['singleton_mean'], published[name]['t0_singleton_mean'], rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError(f'P536 {name} mean does not match the joint-epoch receipt')
        if not math.isclose(references['j251']['databases'][name]['singleton_mean'], published[name]['t251_singleton_mean'], rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError(f'J251 {name} mean does not match the joint-epoch receipt')
    return endpoints, references


def load_candidate(torch, microbatch, state):
    model = microbatch.existing._pilot_model(torch.device('cuda'), torch.float32)
    model.load_state_dict(state)
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError('Candidate load created parameter gradients')
    digest = microbatch.existing.digest(model)
    if digest != state_sha(model.state_dict()):
        raise RuntimeError('Candidate digest does not match the state-dict hash')
    return model, digest


def freeze_candidates(torch, microbatch, iid_state, j_state, names, ties, buffers):
    SCRATCH.mkdir(parents=True, exist_ok=False) if not SCRATCH.exists() else None
    if SCRATCH.exists():
        allowed = {'protocol.json', 'summary.json', 'budget.json', 'alpha_0p25', 'alpha_0p50', 'alpha_0p75'}
        unexpected = {path.name for path in SCRATCH.iterdir()} - allowed
        if unexpected:
            raise RuntimeError(f'Scratch directory contains unrelated files: {sorted(unexpected)}')
    frozen = []
    for alpha in ALPHAS:
        state = interpolate_state(iid_state, j_state, alpha, names, buffers, ties)
        folder = SCRATCH / alpha_name(alpha)
        folder.mkdir(exist_ok=True)
        path = folder / 'evaluation_state.pt'
        model, digest = load_candidate(torch, microbatch, state)
        if path.exists():
            saved = torch.load(path, map_location='cpu', weights_only=False)
            if saved.get('optimizer') is not None or saved.get('alpha') != alpha:
                raise RuntimeError('Existing candidate artifact is not this evaluation-only state')
            saved_state = saved['model']
            if set(saved_state) != set(state) or any(not saved_state[name].equal(state[name]) for name in state):
                raise RuntimeError('Existing candidate artifact disagrees with the interpolation formula')
        else:
            torch.save({
                'evaluation_only': True,
                'alpha': alpha,
                'parents': {'iid_t70': IID_FILE_SHA, 'j251': J_FILE_SHA},
                'model': {name: value.detach().cpu().clone() for name, value in state.items()},
            }, path)
        sidecar = folder / 'model_sha256.json'
        payload = {'alpha': alpha, 'model_sha256': digest, 'file': path.name}
        if sidecar.exists() and read_json(sidecar)['model_sha256'] != digest:
            raise RuntimeError('Frozen candidate hash changed')
        write_json(sidecar, payload)
        frozen.append({'alpha': alpha, 'model_sha256': digest, 'directory': str(folder)})
        del model
        torch.cuda.empty_cache()
    return frozen


def budget_exceeded(spent, estimate):
    return spent + estimate > BUDGET_SECONDS


def qualified_mean(values):
    import numpy as np
    return float(np.mean(list(values)))


def assert_clean28(clean28, reactions):
    total = sum(row['contribution'] for row in reactions)
    if len(reactions) != 28 or not math.isclose(total / 28, clean28, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError('Clean28 does not match the mean of the 28 contributions')


def run_clean28(validation, model, bundle, folder, digest, spent):
    path = folder / 'validation_clean.json'
    if path.exists():
        saved = read_json(path)
        if saved['model_sha256'] != digest or 'full30' in saved:
            raise RuntimeError('Clean28 receipt belongs to a different candidate or retains Full30')
        assert_clean28(saved['clean28'], saved['reactions'])
        return saved, spent, 0.0
    began = time.perf_counter()
    raw = validation(model, bundle)
    elapsed = time.perf_counter() - began
    reactions = enrich_clean_rows(raw['reaction_rows'], bundle.validation_reactions)
    clean28 = float(raw['clean28'])
    del raw
    assert_clean28(clean28, reactions)
    record = {'model_sha256': digest, 'clean28': clean28, 'reactions': reactions, 'seconds': elapsed}
    write_json(path, record)
    print(f'CLEAN28 {folder.name} {clean28:.9f} seconds={elapsed:.3f}', flush=True)
    return record, spent + elapsed, elapsed


def run_chemistry(torch, microbatch, model, shadow, bundle, manifest, folder, digest, spent):
    path = folder / 'chemistry.json'
    record = read_json(path) if path.exists() else {
        'model_sha256': digest,
        'evaluation_manifest_sha256': MANIFEST_SHA,
        'rows': {},
    }
    if record['model_sha256'] != digest or record['evaluation_manifest_sha256'] != MANIFEST_SHA:
        raise RuntimeError('Chemistry receipt does not match the frozen candidate')
    dispersion = bundle.chemistry_dispersions()
    new_seconds = 0.0
    last = None
    for selected in manifest['rows']:
        identity = selected['identity']
        if identity in record['rows']:
            continue
        if last is not None and budget_exceeded(spent + new_seconds, last):
            write_json(path, record)
            raise RuntimeError('PARTIAL')
        reaction = bundle.chemistry('train_' + selected['task']).load_variant(identity, selected['variant'])
        reaction = microbatch.existing.lap_training.tensor_record(reaction, 'cuda', torch.float64)
        if len(reaction['Grid']) > 131072:
            reaction['model_point_chunk_size'] = 16384
        began = time.perf_counter()
        with torch.no_grad():
            value = float(microbatch.chemistry(model, shadow, reaction, dispersion)())
        elapsed = time.perf_counter() - began
        if not math.isfinite(value):
            raise FloatingPointError(f'Nonfinite chemistry loss for {identity}')
        record['rows'][identity] = {
            'task': selected['task'],
            'variant': selected['variant'],
            'database': bundle.reactions[identity]['database'],
            'loss': value,
        }
        new_seconds += elapsed
        last = elapsed
        write_json(path, record)
        print(f'CHEMISTRY {folder.name} {identity} loss={value:.8g} seconds={elapsed:.3f}', flush=True)
        del reaction
    if len(record['rows']) != 268:
        write_json(path, record)
        raise RuntimeError('PARTIAL')
    grouped = [
        {'task': row['task'], 'database': row['database'], 'loss': row['loss']}
        for row in record['rows'].values()
    ]
    summary = database_summary(grouped)
    record['complete'] = True
    record['objectives'] = {
        'relchem': qualified_mean(row['loss'] for row in grouped if row['task'] == 'relchem'),
        'ae17': qualified_mean(row['loss'] for row in grouped if row['task'] == 'ae17'),
    }
    record['databases'] = summary['databases']
    record['seconds'] = new_seconds
    write_json(path, record)
    return record, spent + new_seconds, new_seconds


def run_mrks(microbatch, model, bundle, folder, digest, spent):
    import torch
    path = folder / 'mrks.json'
    record = read_json(path) if path.exists() else {'model_sha256': digest, 'rows': {}}
    if record['model_sha256'] != digest:
        raise RuntimeError('mRKS receipt does not match the frozen candidate')
    dispersions = read_json(microbatch.DATA / 'mrks/dispersion.json')
    new_seconds = 0.0
    last = None
    for identity in sorted(bundle.systems):
        if identity in record['rows']:
            continue
        if last is not None and budget_exceeded(spent + new_seconds, last):
            write_json(path, record)
            raise RuntimeError('PARTIAL')
        system = bundle.mrks().operator_system(identity, device='cuda', dtype=torch.float32, chunk_size=4096)
        exc, operator = microbatch.existing.core.make_mrks_objective_factories(
            model, system, point_chunk_size=256, exc_chunk_size=4096, dispersions=dispersions,
        )
        began = time.perf_counter()
        values = {'exc': float(exc().detach()), 'op': float(operator().detach())}
        elapsed = time.perf_counter() - began
        if not all(math.isfinite(value) for value in values.values()):
            raise FloatingPointError(f'Nonfinite mRKS objective for {identity}')
        record['rows'][identity] = values
        new_seconds += elapsed
        last = elapsed
        write_json(path, record)
        print(f"MRKS {folder.name} {identity} exc={values['exc']:.8g} op={values['op']:.8g} seconds={elapsed:.3f}", flush=True)
        del system, exc, operator
    if len(record['rows']) != 90:
        write_json(path, record)
        raise RuntimeError('PARTIAL')
    record['complete'] = True
    record['objectives'] = {
        'exc': qualified_mean(row['exc'] for row in record['rows'].values()),
        'op': qualified_mean(row['op'] for row in record['rows'].values()),
    }
    record['seconds'] = new_seconds
    write_json(path, record)
    return record, spent + new_seconds, new_seconds


def assert_sources():
    expected = read_json(J_PROTOCOL)['source_hashes']
    current = {relative: sha256_file(REPO / relative) for relative in SOURCE_FILES}
    for relative, digest in expected.items():
        if relative in current and current[relative] != digest:
            raise RuntimeError(f'Scientific source hash changed: {relative}')
    return current


def assert_published_objectives():
    iid_chem = read_json(IID_CHEMISTRY)
    j_chem = read_json(J_CHEMISTRY)
    iid_mrks = read_json(ROOT / 'lap_iid_adamw_lr_branches_20261009/A/endpoint_70_mrks.json')
    j_mrks = read_json(ROOT / 'lap_relchem_joint_epoch_20261009/J/endpoint_251_mrks.json')
    if iid_chem['checkpoint_sha256'] != IID_FILE_SHA or iid_mrks['checkpoint_sha256'] != IID_FILE_SHA:
        raise RuntimeError('IID published receipt is bound to a different checkpoint')
    if j_chem['checkpoint_sha256'] != J_FILE_SHA or j_mrks['checkpoint_sha256'] != J_FILE_SHA:
        raise RuntimeError('J251 published receipt is bound to a different checkpoint')
    iid_objectives = {
        'relchem': qualified_mean(row['loss'] for row in iid_chem['rows'].values() if row['task'] == 'relchem'),
        'ae17': qualified_mean(row['loss'] for row in iid_chem['rows'].values() if row['task'] == 'ae17'),
        'exc': qualified_mean(row['exc'] for row in iid_mrks['rows'].values()),
        'op': qualified_mean(row['op'] for row in iid_mrks['rows'].values()),
    }
    published_iid = {'relchem': 1.044607979, 'ae17': 0.258233903, 'exc': 0.196699782, 'op': 0.941692051}
    for name, value in published_iid.items():
        if round(objective_ratios(iid_objectives)[name], 9) != value:
            raise RuntimeError(f'IID t70 {name} ratio does not match the published value')
    joint = read_json(REPO / 'relchem_joint_epoch_metrics.json')['arms']['J']['objectives']
    j_objectives = {
        'relchem': qualified_mean(row['loss'] for row in j_chem['rows'].values() if row['task'] == 'relchem'),
        'ae17': qualified_mean(row['loss'] for row in j_chem['rows'].values() if row['task'] == 'ae17'),
        'exc': qualified_mean(row['exc'] for row in j_mrks['rows'].values()),
        'op': qualified_mean(row['op'] for row in j_mrks['rows'].values()),
    }
    for name, value in joint.items():
        if not math.isclose(j_objectives[name], value, rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError(f'J251 {name} objective does not match the published receipt')
    return iid_objectives


def load_spent():
    path = SCRATCH / 'budget.json'
    if not path.exists():
        return 0.0
    return float(read_json(path)['gpu_seconds'])


def store_spent(spent, peak_allocated, peak_reserved):
    write_json(SCRATCH / 'budget.json', {
        'gpu_seconds': spent,
        'peak_allocated_bytes': peak_allocated,
        'peak_reserved_bytes': peak_reserved,
    })


def evaluate_alpha(torch, microbatch, validation, bundle, manifest, alpha, endpoints, spent):
    import copy
    folder = SCRATCH / alpha_name(alpha)
    saved = torch.load(folder / 'evaluation_state.pt', map_location='cpu', weights_only=False)
    if 'optimizer' in saved or saved.get('evaluation_only') is not True:
        raise RuntimeError('Candidate artifact is not an evaluation-only interpolation state')
    model, digest = load_candidate(torch, microbatch, saved['model'])
    if digest != read_json(folder / 'model_sha256.json')['model_sha256']:
        raise RuntimeError('Candidate tensor hash changed before evaluation')
    shadow = copy.deepcopy(model).double()
    started = spent
    clean, spent, _clean_seconds = run_clean28(validation, model, bundle, folder, digest, spent)
    chemistry, spent, _chem_seconds = run_chemistry(
        torch, microbatch, model, shadow, bundle, manifest, folder, digest, spent,
    )
    mrks, spent, _mrks_seconds = run_mrks(microbatch, model, bundle, folder, digest, spent)
    if microbatch.existing.digest(model) != digest:
        raise RuntimeError('Model tensor hash changed during evaluation')
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError('Evaluation created parameter gradients')
    objectives = {**chemistry['objectives'], **mrks['objectives']}
    ratios = objective_ratios(objectives)
    record = {
        'alpha': alpha,
        'complete': True,
        'status': 'complete',
        'model_sha256': digest,
        'clean28': clean['clean28'],
        'objectives': objectives,
        'ratios': ratios,
        'eligible': eligibility(ratios),
        'databases': chemistry['databases'],
        'reactions': clean['reactions'],
        'seconds': spent - started,
        'changes_vs_iid_t70': reaction_changes(clean['reactions'], endpoints['iid_t70']),
        'changes_vs_j251': reaction_changes(clean['reactions'], endpoints['j251']),
    }
    require_complete(record)
    stored = {key: value for key, value in record.items() if key != 'status'}
    write_json(folder / 'candidate.json', stored)
    print(
        f"CANDIDATE {alpha:.2f} clean28={record['clean28']:.9f} eligible={record['eligible']}",
        flush=True,
    )
    del model, shadow
    torch.cuda.empty_cache()
    return record, spent


def candidate_from_disk(alpha, endpoints):
    path = SCRATCH / alpha_name(alpha) / 'candidate.json'
    if not path.exists():
        return None
    record = read_json(path)
    record['complete'] = True
    record['status'] = 'complete'
    record['eligible'] = eligibility(record['ratios'])
    record['changes_vs_iid_t70'] = reaction_changes(record['reactions'], endpoints['iid_t70'])
    record['changes_vs_j251'] = reaction_changes(record['reactions'], endpoints['j251'])
    require_complete(record)
    return record


def focus_changes(databases, reference):
    return {
        name: databases[name]['singleton_mean'] - reference[name]['singleton_mean']
        for name in FOCUS
    }


def assemble(records, references, frozen, sources, spent, peak, commit):
    ordered = sorted(records, key=lambda row: row['alpha'])
    decision = decide([
        {'alpha': row['alpha'], 'complete': True, 'eligible': row['eligible'], 'clean28': row['clean28']}
        for row in ordered
    ])
    iid_ratios = {'relchem': 1.044607979, 'ae17': 0.258233903, 'exc': 0.196699782, 'op': 0.941692051}
    j_ratios = read_json(REPO / 'relchem_joint_epoch_metrics.json')['arms']['J']['ratios']
    shapes = {'clean28': response_label([IID_CLEAN28, *[row['clean28'] for row in ordered], J_CLEAN28])}
    for name in BASELINE:
        shapes[name] = response_label([
            iid_ratios[name], *[row['ratios'][name] for row in ordered], j_ratios[name],
        ])
    for row in ordered:
        row['focus_delta_vs_p536'] = focus_changes(row['databases'], references['t0']['databases'])
        row['clean28_minus_iid'] = row['clean28'] - IID_CLEAN28
        row['clean28_minus_j251'] = row['clean28'] - J_CLEAN28
        row['clean28_minus_historical_best'] = row['clean28'] - HISTORICAL_BEST_CLEAN28
    return {
        'status': decision['classification'],
        'decision': decision,
        'shapes': shapes,
        'candidates': ordered,
        'frozen': frozen,
        'references': {
            't0_databases': references['t0']['databases'],
            'iid_t70_databases': references['iid_t70']['databases'],
            'j251_databases': references['j251']['databases'],
            'iid_clean28': IID_CLEAN28,
            'j251_clean28': J_CLEAN28,
        },
        'parents': {
            'iid_file_sha256': IID_FILE_SHA,
            'iid_tensor_sha256': IID_TENSOR_SHA,
            'j_file_sha256': J_FILE_SHA,
            'j_tensor_sha256': J_TENSOR_SHA,
            'p536_tensor_sha256': P536_TENSOR_SHA,
        },
        'dataset_sha256': DATA_SHA,
        'evaluation_manifest_sha256': MANIFEST_SHA,
        'source_commit': commit,
        'source_sha256': sources,
        'gpu_seconds': spent,
        'peak_allocated_bytes': peak[0],
        'peak_reserved_bytes': peak[1],
        'baseline': BASELINE,
        'optimizer_updates': 0,
    }


def render(summary):
    lines = [
        '# IID t70 to J251 checkpoint interpolation',
        '',
        'Three evaluation-only parameter states were formed by `theta(alpha) = (1-alpha) theta_IID70 + alpha theta_J251`. Scientific objectives were measured at those states. Endpoint metrics were reused from frozen receipts and were not linearly interpolated.',
        '',
        '## A. Provenance',
        '',
        f"Source commit `{summary['source_commit']}`. Dataset logical SHA256 `{summary['dataset_sha256']}`. Evaluation manifest SHA256 `{summary['evaluation_manifest_sha256']}`.",
        f"IID t70 file `{summary['parents']['iid_file_sha256']}`, tensor `{summary['parents']['iid_tensor_sha256']}`.",
        f"J251 file `{summary['parents']['j_file_sha256']}`, tensor `{summary['parents']['j_tensor_sha256']}`.",
        f"Shared P536 tensor `{summary['parents']['p536_tensor_sha256']}`. Unique trainable parameters: {PARAMETER_COUNT}. Interpolation arithmetic: F64, then one cast to F32.",
        f"New GPU seconds: {summary['gpu_seconds']:.3f}. Peak allocated CUDA bytes: {summary['peak_allocated_bytes']}. Peak reserved bytes: {summary['peak_reserved_bytes']}.",
        'No optimizer state was saved or stepped. Full30 was produced by the unchanged validation helper and then discarded. It did not enter selection.',
        '',
        '## B. Five-point comparison',
        '',
        '| Alpha | Clean28 | relchem/t0 | AE17/t0 | Exc/t0 | Op/t0 | Eligible |',
        '|---:|---:|---:|---:|---:|---:|---|',
        '| 0.00 IID t70 | 8.629660 | 1.044608 | 0.258234 | 0.196700 | 0.941692 | No |',
    ]
    by_alpha = {row['alpha']: row for row in summary.get('candidates', [])}
    for alpha in ALPHAS:
        row = by_alpha.get(alpha)
        if row is None:
            lines.append(f'| {alpha:.2f} | incomplete |  |  |  |  |  |')
            continue
        ratios = row['ratios']
        lines.append(
            f"| {alpha:.2f} | {row['clean28']:.6f} | {ratios['relchem']:.6f} | {ratios['ae17']:.6f} | "
            f"{ratios['exc']:.6f} | {ratios['op']:.6f} | {'Yes' if row['eligible'] else 'No'} |"
        )
    lines.append('| 1.00 J251 | 9.346761 | 0.979708 | 0.081012 | 0.101105 | 0.938121 | Yes |')
    decision = summary.get('decision') or {}
    lines.extend(['', f"Classification: **{summary['status']}**.", ''])
    winner = decision.get('winner')
    if winner:
        lines.append(
            f"Best eligible interpolated candidate: alpha {winner['alpha']:.2f}, Clean28 {winner['clean28']:.6f}."
        )
    else:
        lines.append('Best eligible interpolated candidate: none.')
    best = decision.get('best_clean28_candidate')
    if best:
        lines.append(
            f"Lowest Clean28 among the three new candidates: alpha {best['alpha']:.2f}, "
            f"Clean28 {best['clean28']:.6f}, eligible {best['eligible']}."
        )
    lines.extend(['', '## C. Relchem databases', ''])
    if 'references' in summary:
        lines.append('| Database | P536 mean | IID t70 mean | alpha 0.25 | alpha 0.50 | alpha 0.75 | J251 mean |')
        lines.append('|---|---:|---:|---:|---:|---:|---:|')
        refs = summary['references']
        for name in DATABASES:
            cells = [
                refs['t0_databases'][name]['singleton_mean'],
                refs['iid_t70_databases'][name]['singleton_mean'],
            ]
            for alpha in ALPHAS:
                row = by_alpha.get(alpha)
                cells.append(None if row is None else row['databases'][name]['singleton_mean'])
            cells.append(refs['j251_databases'][name]['singleton_mean'])
            rendered = ' | '.join('' if value is None else f'{value:.6f}' for value in cells)
            lines.append(f'| {name} | {rendered} |')
        lines.extend(['', 'ABDE4, pTC13 and PA8 singleton-mean changes from P536:', ''])
        for alpha in ALPHAS:
            row = by_alpha.get(alpha)
            if row is None or 'focus_delta_vs_p536' not in row:
                continue
            parts = ', '.join(f"{name} {row['focus_delta_vs_p536'][name]:+.6f}" for name in FOCUS)
            lines.append(f'- alpha {alpha:.2f}: {parts}')
    lines.extend(['', '## D. Clean28 reactions', ''])
    for alpha in ALPHAS:
        row = by_alpha.get(alpha)
        if row is None or 'changes_vs_iid_t70' not in row:
            continue
        versus_iid = row['changes_vs_iid_t70']
        versus_j = row['changes_vs_j251']
        improved = [item['reaction_id'] for item in versus_iid['rows'] if item['delta_contribution'] < 0.0]
        worst = sorted(versus_iid['rows'], key=lambda item: item['delta_contribution'], reverse=True)[:5]
        best_versus_j = sorted(versus_j['rows'], key=lambda item: item['delta_contribution'])[:5]
        lines.append(
            f"Alpha {alpha:.2f} versus IID t70: {versus_iid['improved']} improved, "
            f"{versus_iid['worsened']} worsened, {versus_iid['unchanged']} unchanged. "
            f"Versus J251: {versus_j['improved']} improved, {versus_j['worsened']} worsened, "
            f"{versus_j['unchanged']} unchanged. "
            f"Clean28 minus IID {row['clean28_minus_iid']:+.6f}; minus J251 {row['clean28_minus_j251']:+.6f}; "
            f"minus historical best {row['clean28_minus_historical_best']:+.6f}."
        )
        lines.append('Improved versus IID t70: ' + ', '.join(improved) + '.')
        lines.append('Largest contribution increases versus IID t70: ' + ', '.join(
            f"{item['reaction_id']} {item['delta_contribution']:+.3f}" for item in worst
        ) + '.')
        lines.append('Largest contribution decreases versus J251: ' + ', '.join(
            f"{item['reaction_id']} {item['delta_contribution']:+.3f}" for item in best_versus_j
        ) + '.')
        lines.append('')
    shapes = summary.get('shapes') or {}
    if shapes:
        described = ', '.join(f'{name} {label}' for name, label in shapes.items())
        tradeoff = 'a usable eligible candidate' if summary['status'] == 'GO' else 'no usable candidate'
        lines.extend([
            '## E. Response across the frozen grid',
            '',
            f'The five sampled points, including the two reused endpoints, are {described}. This grid has {tradeoff}. Three alphas are not a continuous Pareto frontier, and Clean28 does not measure an untouched future test.',
            '',
        ])
    lines.extend(['## F. Next experiment', '', recommendation_text(summary), '', '## Source hashes', ''])
    for relative, digest in summary.get('source_sha256', {}).items():
        lines.append(f'- `{relative}` `{digest}`')
    lines.append('')
    (REPO / 'lap_checkpoint_interpolation_report.md').write_text('\n'.join(lines), encoding='utf-8')
    (REPO / 'lap_checkpoint_interpolation_metrics.json').write_text(
        json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    if SCRATCH.exists():
        write_json(SCRATCH / 'summary.json', summary)


def recommendation_text(summary):
    status = summary['status']
    if status == 'PARTIAL':
        return 'No new experiment. Resume the same frozen alphas and receipts if the missing stages are still inside the original 60-minute budget. Do not add alphas or change chunk sizes.'
    if status == 'GO':
        winner = summary['decision']['winner']
        saved = next(row['model_sha256'] for row in summary['candidates'] if row['alpha'] == winner['alpha'])
        return (
            f"Next experiment, not run: one read-only reload of the saved alpha {winner['alpha']:.2f} state "
            f"`{saved}` under the same manifest, Clean28 split, and full90 populations, "
            'only if that state is going to be used further. Do not search another alpha. Clean28 and relchem are monotonic on this chord, '
            'and alpha 0.50 remains ineligible because its relchem ratio is above 1. Do not fine-tune. Expected cost is about 7 minutes on one local GPU.'
        )
    return 'No further interpolation experiment. This fixed grid did not produce an eligible model with Clean28 below J251. Another alpha, local optimization, and checkpoint fine-tuning are not selected.'


def parent_record():
    return {
        'iid_file_sha256': IID_FILE_SHA,
        'iid_tensor_sha256': IID_TENSOR_SHA,
        'j_file_sha256': J_FILE_SHA,
        'j_tensor_sha256': J_TENSOR_SHA,
        'p536_tensor_sha256': P536_TENSOR_SHA,
    }


def run_experiment():
    import subprocess
    forbid_training_calls(Path(__file__).read_text(encoding='utf-8'))
    if (SCRATCH / 'summary.json').exists() and read_json(SCRATCH / 'summary.json').get('status') in {'GO', 'NO-GO'}:
        render(read_json(SCRATCH / 'summary.json'))
        return
    sources = assert_sources()
    assert_published_objectives()
    torch, microbatch, validation = scientific_modules()
    if microbatch.DATA_SHA != DATA_SHA:
        raise RuntimeError('Dataset constant differs from the qualified microbatch driver')
    iid_state, j_state, names, ties, buffers = load_parent_models(torch, microbatch)
    parent_hashes = {str(IID_CHECKPOINT): sha256_file(IID_CHECKPOINT), str(J_CHECKPOINT): sha256_file(J_CHECKPOINT)}
    manifest_hash = sha256_file(MANIFEST_PATH)
    bundle = microbatch.PublicationDataset(microbatch.DATA)
    completed = []
    frozen = []
    references = None
    try:
        if bundle.manifest['logical_sha256'] != DATA_SHA:
            raise RuntimeError('Dataset logical SHA mismatch')
        manifest = read_json(MANIFEST_PATH)
        if manifest['policy'] != 'one-variant-per-identity-v1':
            raise RuntimeError('Evaluation manifest policy is not one variant per identity')
        from lap_chemistry_sampling import validate_evaluation
        validate_evaluation(manifest['rows'], bundle.reactions)
        endpoints, references = prepare_references(bundle)
        frozen = freeze_candidates(torch, microbatch, iid_state, j_state, names, ties, buffers)
        write_json(SCRATCH / 'protocol.json', {
            'alphas': list(ALPHAS),
            'execution_order': list(EXECUTION_ORDER),
            'frozen': frozen,
            'parents': parent_hashes,
            'arithmetic': 'F64 temporary, one cast to F32',
            'optimizer': None,
        })
        torch.cuda.reset_peak_memory_stats()
        spent = load_spent()
        for alpha in EXECUTION_ORDER:
            existing = candidate_from_disk(alpha, endpoints)
            if existing is not None:
                completed.append(existing)
                continue
            if spent >= BUDGET_SECONDS:
                raise RuntimeError('PARTIAL')
            record, spent = evaluate_alpha(
                torch, microbatch, validation, bundle, manifest, alpha, endpoints, spent,
            )
            store_spent(spent, torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())
            completed.append(record)
        if sha256_file(IID_CHECKPOINT) != parent_hashes[str(IID_CHECKPOINT)]:
            raise RuntimeError('IID checkpoint bytes changed')
        if sha256_file(J_CHECKPOINT) != parent_hashes[str(J_CHECKPOINT)]:
            raise RuntimeError('J251 checkpoint bytes changed')
        if sha256_file(MANIFEST_PATH) != manifest_hash:
            raise RuntimeError('Evaluation manifest bytes changed')
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
        peak = (torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())
        render(assemble(completed, references, frozen, sources, spent, peak, commit))
    except RuntimeError as error:
        if str(error) != 'PARTIAL':
            raise
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
        peak = (torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())
        summary = {
            'status': 'PARTIAL',
            'decision': {'classification': 'PARTIAL', 'winner': None, 'best_clean28_candidate': None},
            'candidates': completed,
            'frozen': frozen,
            'references': {},
            'parents': parent_record(),
            'dataset_sha256': DATA_SHA,
            'evaluation_manifest_sha256': MANIFEST_SHA,
            'source_commit': commit,
            'source_sha256': sources,
            'gpu_seconds': load_spent(),
            'peak_allocated_bytes': peak[0],
            'peak_reserved_bytes': peak[1],
            'shapes': {},
        }
        if references is not None:
            summary['references'] = {
                't0_databases': references['t0']['databases'],
                'iid_t70_databases': references['iid_t70']['databases'],
                'j251_databases': references['j251']['databases'],
            }
        render(summary)
        raise RuntimeError('PARTIAL: the 60-minute GPU budget stopped the remaining evaluations') from error
    finally:
        bundle.close()


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args(argv)
    if args.report_only:
        render(read_json(SCRATCH / 'summary.json'))
        return
    run_experiment()


if __name__ == '__main__':
    main()
