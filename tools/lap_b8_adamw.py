"""One 251-update AdamW arm with eight balanced relchem reactions per update.

J251 remains the frozen batch-1 control. This arm uses the same loss, coefficients,
learning rate, and AdamW settings. It does not retrain J251. Equal update counts
are not equal data: J251 saw 251 chemistry presentations and this arm sees 2,008.
"""
import ast
import copy
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
OUT = REPO.parent / 'lap_b8_adamw_20261010'
PROTOCOL_ID = 'b8-adamw-j251-v1'
OFFSETS = (0, 31, 62, 93, 124, 155, 186, 217)
TASKS = ('relchem', 'ae17', 'exc', 'op')
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
J251_RATIOS = {
    'relchem': 0.979707731825112,
    'ae17': 0.08101214513606354,
    'exc': 0.10110475652753241,
    'op': 0.9381212072698767,
}
P536_CLEAN28 = 9.553190635871177
J251_CLEAN28 = 9.346760658714116
HISTORICAL_BEST_CLEAN28 = 8.619694171538775
BEST_ELIGIBLE_CLEAN28 = 9.168650862151432
P536_FILE = '0ca0f77367214c171e6edf0576bdb043d64479245ad1a0e539fa11fb0cee542d'
P536_TENSOR = '3c2e45d6a86cccf80d7b4dc458e67f9b406935ed60d4e57433bb441a6df88da6'
DATA_SHA = '61c221a19b9987717e69cac182ad545241f8807db4126c0949a99992e4c210ef'
J251_MANIFEST_SHA = '72b827655ce4421f2fc933c082cda1c2dc3e5d9903ec29e0c5e3d62e82ac37f6'
EVAL_MANIFEST_SHA = '132bd550161be6bc6761f8df1de8d3755c0b0debab53663208c3a9bbf56cb805'
CALIBRATION_SHA = '4d97df1ec78aa01a2c9380a86a323b2f0b46828dfc5975f5e3a63c75f4440096'
PARITY_INDICES = (0, 125, 250)
MILESTONES = (0, 64, 128, 192, 251)
CLEAN_MILESTONES = (64, 128, 192, 251)
BUDGET_SECONDS = 180 * 60
EVAL_RESERVE_SECONDS = 2400
RELATIVE_L2_LIMIT = 1e-10
OTHER_TASK_L2_LIMIT = 1e-8
GATE = 'NOT EVALUATED — ACCURACY GATE'


def dump(value):
    return json.dumps(value, indent=2) + '\n'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(dump(value), encoding='utf-8', newline='\n')
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def build_batches(j251_rows):
    """Eight cyclic passes through the frozen J251 permutation."""
    if len(j251_rows) != 251:
        raise ValueError('J251 manifest must contain 251 updates')
    if tuple(OFFSETS) != (0, 31, 62, 93, 124, 155, 186, 217):
        raise ValueError('B8 offsets are frozen')
    batches = []
    for cursor in range(251):
        relchem = []
        for offset in OFFSETS:
            source = j251_rows[(cursor + offset) % 251]['relchem']
            relchem.append({key: source[key] for key in ('identity', 'database', 'reaction_id', 'variant', 'weight')})
        base = j251_rows[cursor]
        batches.append({
            'cursor': cursor,
            'relchem': relchem,
            'ae17': {key: base['ae17'][key] for key in ('identity', 'database', 'reaction_id', 'variant', 'weight')},
            'mrks_id': base['mrks_id'],
        })
    return batches


def presentation_counts(batches):
    counts = {}
    for batch in batches:
        seen = set()
        for spec in batch['relchem']:
            identity = spec['identity']
            if identity in seen:
                raise ValueError('A B8 batch repeats a relchem identity')
            seen.add(identity)
            counts[identity] = counts.get(identity, 0) + 1
    return counts


def assert_batch_properties(batches, j251_rows):
    if len(batches) != 251 or len(j251_rows) != 251:
        raise ValueError('B8 requires exactly 251 updates')
    counts = presentation_counts(batches)
    if len(counts) != 251 or any(value != 8 for value in counts.values()):
        raise ValueError('Each relchem identity must appear exactly eight times')
    variants = {}
    base_databases = {}
    got_databases = {}
    for cursor, (batch, source) in enumerate(zip(batches, j251_rows, strict=True)):
        if len(batch['relchem']) != 8:
            raise ValueError('Each update needs eight relchem reactions')
        if batch['relchem'][0] != source['relchem'] or batch['ae17'] != source['ae17']:
            raise ValueError('J251 base reaction or AE17 stream changed')
        if batch['mrks_id'] != source['mrks_id'] or batch['cursor'] != cursor:
            raise ValueError('J251 mRKS stream or cursor changed')
        base_databases[source['relchem']['database']] = base_databases.get(source['relchem']['database'], 0) + 1
        for offset, spec in zip(OFFSETS, batch['relchem'], strict=True):
            expected = j251_rows[(cursor + offset) % 251]['relchem']
            if spec != expected:
                raise ValueError('B8 entry is not the frozen cyclic permutation')
            variants.setdefault(spec['identity'], set()).add(spec['variant'])
            got_databases[spec['database']] = got_databases.get(spec['database'], 0) + 1
    if any(len(values) != 1 for values in variants.values()):
        raise ValueError('A relchem identity changed quadrature variant')
    if got_databases != {name: count * 8 for name, count in base_databases.items()}:
        raise ValueError('Database exposure is not eight times the J251 population')
    if sum(got_databases.values()) != 2008:
        raise ValueError('B8 must contain 2008 relchem presentations')
    return {'identities': 251, 'presentations_each': 8, 'presentations': 2008, 'databases': got_databases}


def combine_gradients(pairs, names=None, zero_templates=None):
    """F64 arithmetic mean. One reaction is returned at weight 1, not divided by 8."""
    if not pairs:
        raise ValueError('empty chemistry batch')
    zero_templates = {} if zero_templates is None else dict(zero_templates)
    first_names = tuple(pairs[0][1])
    if len(first_names) != len(set(first_names)):
        raise ValueError('duplicate parameter name')
    if names is None:
        names = first_names
        for _, grad in pairs[1:]:
            if tuple(grad) != names:
                raise ValueError('parameter order changed')
    else:
        names = tuple(names)
        if len(names) != len(set(names)):
            raise ValueError('duplicate parameter name')
    for _, grad in pairs:
        for grad_tensor in grad.values():
            if grad_tensor.dtype != torch.float64:
                raise ValueError('F64 chemistry gradients required')
    if len(pairs) == 1 and not zero_templates and names == first_names:
        value, grad = pairs[0]
        if not math.isfinite(float(value)) or any(not torch.isfinite(piece).all() for piece in grad.values()):
            raise FloatingPointError('nonfinite chemistry singleton')
        return float(value), {name: grad[name].detach() for name in names}
    count = len(pairs)
    totals = {}
    for name in names:
        present = [grad[name] for _, grad in pairs if name in grad]
        if len(present) != count:
            if name not in zero_templates:
                raise ValueError('incomplete parameter set')
            template = zero_templates[name]
        else:
            template = present[0]
        if template.dtype != torch.float64:
            raise ValueError('F64 chemistry gradients required')
        total = torch.zeros_like(template)
        for piece in present:
            if piece.shape != total.shape:
                raise ValueError('parameter shape changed')
            total.add_(piece)
        total.div_(count)
        totals[name] = total
    losses = [float(value) for value, _ in pairs]
    if not all(math.isfinite(value) for value in losses):
        raise FloatingPointError('nonfinite chemistry loss')
    if any(not torch.isfinite(piece).all() for piece in totals.values()):
        raise FloatingPointError('nonfinite chemistry gradient')
    return sum(losses) / count, totals


def relative_l2(left, right):
    names = tuple(left)
    if tuple(right) != names:
        raise ValueError('gradient names or order differ')
    actual = torch.cat([left[name].detach().reshape(-1).double() for name in names])
    reference = torch.cat([right[name].detach().reshape(-1).double() for name in names])
    denominator = torch.linalg.vector_norm(reference)
    if float(denominator) == 0.0:
        return 0.0 if torch.equal(actual, reference) else math.inf
    return float(torch.linalg.vector_norm(actual - reference) / denominator)


def scalarize(raw, coefficients):
    from train_models.lap_fixed_adamw import weighted_gradient
    if tuple(raw) != TASKS or tuple(coefficients) != TASKS:
        raise ValueError('coefficients must be applied in the frozen task order')
    return weighted_gradient(raw, coefficients)


def scientifically_eligible(ratios):
    if set(ratios) != set(TASKS):
        return False
    return all(math.isfinite(ratios[task]) and ratios[task] < 1.0 for task in TASKS)


def beats_historical(clean28):
    return math.isfinite(clean28) and clean28 < HISTORICAL_BEST_CLEAN28


def classify(payload):
    """One label. Accuracy-only is used when Clean28 passes and a scientific objective fails."""
    clean = payload.get('clean28') or {}
    if not payload.get('complete') or set(clean) != set(CLEAN_MILESTONES):
        return 'PARTIAL'
    if any(not math.isfinite(clean[cursor]) for cursor in CLEAN_MILESTONES):
        return 'PARTIAL'
    screened = [cursor for cursor in CLEAN_MILESTONES if beats_historical(clean[cursor])]
    if not screened:
        return 'NO-GO' if payload.get('diagnostic_complete') else 'PARTIAL'
    eligible = []
    for cursor in screened:
        science = (payload.get('science') or {}).get(str(cursor)) or (payload.get('science') or {}).get(cursor)
        if not science or science.get('status') == 'incomplete':
            return 'PARTIAL'
        if science.get('status') != 'complete':
            return 'PARTIAL'
        ratios = science.get('ratios') or {}
        if scientifically_eligible(ratios):
            eligible.append(cursor)
    if not eligible:
        return 'ACCURACY-ONLY PROGRESS'
    best = min(eligible, key=lambda cursor: (clean[cursor], cursor))
    if clean[best] < 7.0:
        return 'BREAKTHROUGH'
    return 'STRONG PROGRESS'


def _manifest_paths():
    root = REPO.parent / 'lap_relchem_joint_epoch_20261009'
    return (
        REPO / 'relchem_joint_epoch_sampling_manifest.json',
        root / 'sampling_manifest.json',
        root / 'J' / 'sampling_manifest.json',
    )


def _evaluation_paths():
    root = REPO.parent / 'lap_relchem_joint_epoch_20261009'
    return (
        REPO / 'relchem_joint_epoch_evaluation_manifest.json',
        root / 'evaluation_manifest.json',
        root / 'J' / 'evaluation_manifest.json',
    )


def load_j251_rows():
    rows = None
    for path in _manifest_paths():
        if not path.is_file():
            raise ValueError('J251 training manifest is missing: ' + str(path))
        if sha256_file(path) != J251_MANIFEST_SHA:
            raise ValueError('J251 training manifest hash changed: ' + str(path))
        parsed = read_json(path)
        if rows is None:
            rows = parsed
        elif parsed != rows:
            raise ValueError('J251 training manifest copies differ')
    return rows


def assert_split_disjoint(batches):
    dataset = REPO.parent / 'publication_dataset_v1'
    splits = read_json(dataset / 'splits.json')
    if 'test' in splits or 'future' in splits:
        raise ValueError('Training manifest resolves against a test or future split')
    identities = set(presentation_counts(batches))
    if identities != set(splits['train_relchem']['ids']):
        raise ValueError('B8 identities are not the frozen train_relchem population')
    diet_ids = set()
    for line in (dataset / 'validation' / 'diet30_reactions.jsonl').read_text(encoding='utf-8').splitlines():
        if line:
            diet_ids.add(json.loads(line)['source_id'])
    if identities & diet_ids:
        raise ValueError('A validation identity entered the B8 manifest')


def _import_run():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import train_lap_microbatch as run
    return run


def _stop(reason):
    write_json(OUT / 'stop.json', {'reason': reason})
    raise SystemExit(reason)


def _load_reaction(bundle, task, spec):
    run = _import_run()
    reaction = bundle.chemistry('train_' + task).load_variant(spec['identity'], spec['variant'])
    reaction = run.existing.lap_training.tensor_record(reaction, 'cuda', torch.float64)
    if len(reaction['Grid']) > 131072:
        reaction['model_point_chunk_size'] = 16384
    database = bundle.reactions[spec['identity']]['database']
    if database != spec['database'] or bundle.reactions[spec['identity']]['task'] != task:
        raise ValueError('Manifest database or task does not match the dataset')
    return reaction


def _singleton(model, shadow, reaction, dispersion):
    run = _import_run()
    value, grad = run.chemistry(model, shadow, reaction, dispersion).value_and_grad()
    if any(piece.dtype != torch.float64 for piece in grad.values()):
        raise ValueError('Qualified chemistry gradient left F64')
    return value, grad


def qualified_mean(model, shadow, bundle, specs, dispersion):
    """Sequential qualified singletons at one frozen parameter state, then an F64 mean."""
    torch.cuda.synchronize()
    started = time.perf_counter()
    pairs = []
    infos = []
    for spec in specs:
        reaction = _load_reaction(bundle, 'relchem', spec)
        value, grad = _singleton(model, shadow, reaction, dispersion)
        flat = torch.cat([piece.detach().reshape(-1) for piece in grad.values()])
        infos.append({
            'identity': spec['identity'], 'variant': spec['variant'], 'database': spec['database'],
            'loss': float(value), 'grad_norm': float(torch.linalg.vector_norm(flat)),
        })
        pairs.append((value, {name: piece.detach() for name, piece in grad.items()}))
        del reaction, grad, flat
        torch.cuda.empty_cache()
    loss, grad = combine_gradients(pairs)
    del pairs
    torch.cuda.synchronize()
    return loss, grad, infos, time.perf_counter() - started


def reference_mean(model, shadow, bundle, specs, dispersion):
    """Independent sum-then-divide, used only to check the B8 implementation."""
    torch.cuda.synchronize()
    started = time.perf_counter()
    accumulator = None
    names = None
    loss_sum = 0.0
    for spec in specs:
        reaction = _load_reaction(bundle, 'relchem', spec)
        value, grad = _singleton(model, shadow, reaction, dispersion)
        if accumulator is None:
            names = tuple(grad)
            accumulator = {name: torch.zeros_like(piece) for name, piece in grad.items()}
        if tuple(grad) != names:
            raise ValueError('reference parameter order changed')
        for name in names:
            accumulator[name].add_(grad[name])
        loss_sum += float(value)
        del reaction, grad
        torch.cuda.empty_cache()
    count = len(specs)
    for name in names:
        accumulator[name].div_(count)
    torch.cuda.synchronize()
    return loss_sum / count, accumulator, time.perf_counter() - started


def other_tasks(model, shadow, bundle, entry, dispersion, mrks_dispersion):
    """Unchanged AE17, Exc, and operator path from the J251 measurement."""
    run = _import_run()
    torch.cuda.synchronize()
    started = time.perf_counter()
    reaction = _load_reaction(bundle, 'ae17', entry['ae17'])
    ae_loss, ae_grad = _singleton(model, shadow, reaction, dispersion)
    del reaction
    torch.cuda.empty_cache()
    system = bundle.mrks().operator_system(entry['mrks_id'], device='cuda', dtype=torch.float32, chunk_size=4096)
    exc_factory, op_factory = run.existing.core.make_mrks_objective_factories(
        model, system, point_chunk_size=256, dispersions=mrks_dispersion, exc_chunk_size=4096)
    losses = {'ae17': float(ae_loss)}
    raw = {'ae17': ae_grad}
    for task, objective in (('exc', exc_factory), ('op', op_factory)):
        value, grads = run.existing.core.compute_isolated_task_gradients(model, {task: objective}, task_order=(task,))
        materialized = run.existing.core.materialize_task_zeros(model, grads, task_order=(task,))
        losses[task] = float(value[task])
        raw[task] = {name: piece.double() for name, piece in materialized[task].items()}
        del value, grads, materialized
        torch.cuda.empty_cache()
    name = system.name
    del system, exc_factory, op_factory
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return losses, raw, name, time.perf_counter() - started


def _grad_norm(grad):
    flat = torch.cat([piece.detach().reshape(-1) for piece in grad.values()])
    return float(torch.linalg.vector_norm(flat))


def _capture_rng():
    from train_models.lap_moo_training import capture_rng_state
    return capture_rng_state()


def _restore_rng(state):
    from train_models.lap_moo_training import restore_rng_state
    restore_rng_state(state)


def save_checkpoint(path, model, optimizer, cursor, manifest_sha, protocol_sha, logs):
    path = Path(path)
    if not path.resolve().is_relative_to(OUT.resolve()):
        raise ValueError('Checkpoint path is outside the B8 scratch directory')
    payload = {
        'model': {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        'optimizer': copy.deepcopy(optimizer.state_dict()),
        'rng': _capture_rng(),
        'cursor': cursor,
        'logs': list(logs),
        'protocol_id': PROTOCOL_ID,
        'manifest_sha256': manifest_sha,
        'protocol_sha256': protocol_sha,
        'lambdas': dict(LAMBDAS),
        'offsets': list(OFFSETS),
        'scheduler': None,
    }
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_checkpoint(path, model, optimizer, manifest_sha, protocol_sha):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved.get('protocol_id') != PROTOCOL_ID or saved.get('scheduler') is not None:
        raise ValueError('Checkpoint belongs to a different experiment')
    if saved.get('manifest_sha256') != manifest_sha or saved.get('protocol_sha256') != protocol_sha:
        raise ValueError('Checkpoint manifest or protocol hash mismatch')
    if saved.get('lambdas') != LAMBDAS or tuple(saved.get('offsets')) != OFFSETS:
        raise ValueError('Checkpoint loss protocol mismatch')
    model.load_state_dict(saved['model'])
    optimizer.load_state_dict(saved['optimizer'])
    _restore_rng(saved['rng'])
    return saved['cursor'], list(saved['logs'])


def verify_sources():
    protocol = read_json(REPO / 'relchem_joint_epoch_protocol.json')
    for relative, digest in protocol['source_hashes'].items():
        if sha256_file(REPO / relative) != digest:
            raise ValueError('Frozen scientific source changed: ' + relative)
    if protocol['sampling_manifest_sha256'] != J251_MANIFEST_SHA:
        raise ValueError('Protocol no longer points at the frozen J251 manifest')
    if protocol['evaluation_manifest_sha256'] != EVAL_MANIFEST_SHA:
        raise ValueError('Protocol no longer points at the frozen evaluation manifest')
    for path in _evaluation_paths():
        if not path.is_file() or sha256_file(path) != EVAL_MANIFEST_SHA:
            raise ValueError('Evaluation manifest hash changed: ' + str(path))
    calibration_path = REPO.parent / 'lap_relchem_joint_epoch_20261009' / 'J' / 'calibration.json'
    if sha256_file(calibration_path) != CALIBRATION_SHA:
        raise ValueError('J251 calibration file changed')
    calibration = read_json(calibration_path)
    if any(calibration['lambda'][task] != LAMBDAS[task] for task in TASKS):
        raise ValueError('Task coefficients differ from the J251 calibration receipt')
    if protocol['lr'] != 1e-4 or protocol['scheduler'] is not None or protocol['total_updates'] != 251:
        raise ValueError('J251 optimizer protocol is not the expected frozen arm')
    initial = protocol['initial_state']
    if initial['file_sha256'] != P536_FILE or initial['state_sha256'] != P536_TENSOR:
        raise ValueError('J251 initial state does not match corrected P536')
    if sha256_file(initial['path']) != P536_FILE:
        raise ValueError('Corrected P536 checkpoint file changed')
    return protocol


def _protocol_payload(manifest_sha, source_protocol):
    return {
        'protocol_id': PROTOCOL_ID,
        'j251_retrained': False,
        'question': 'Does a balanced eight-reaction relchem mean improve Clean28 under fixed AdamW?',
        'limitation': '251 B8 updates present 2008 chemistry examples; J251 presented 251. Update counts are not equal compute.',
        'batch_definition': 'B8[t] = P[(t+offset) mod 251] for offsets (0, 31, 62, 93, 124, 155, 186, 217)',
        'unbiasedness': (
            'At frozen parameters, a uniformly random update index has the full fixed-variant '
            '251-reaction mean as its expectation. The executed schedule is deterministic and '
            'parameters move, so individual updates are not conditionally unbiased.'
        ),
        'loss': 'original qualified singleton mean; weight 1/8; no residual multiplier; no Skala; no Huber',
        'lambdas': dict(LAMBDAS),
        'adamw': {'lr': 1e-4, 'betas': [0.9, 0.999], 'eps': 1e-8, 'weight_decay': 0.01, 'foreach': False},
        'scheduler': None,
        'updates': 251,
        'offsets': list(OFFSETS),
        'initial_state': source_protocol['initial_state'],
        'dataset_sha256': DATA_SHA,
        'j251_manifest_sha256': J251_MANIFEST_SHA,
        'b8_manifest_sha256': manifest_sha,
        'evaluation_manifest_sha256': EVAL_MANIFEST_SHA,
        'calibration_sha256': CALIBRATION_SHA,
        'source_hashes': source_protocol['source_hashes'],
        'exc_chunk_size': 4096,
        'operator_chunk_size': 256,
        'chemistry_chunk': 16384,
        'budget_seconds': BUDGET_SECONDS,
        'eval_reserve_seconds': EVAL_RESERVE_SECONDS,
    }


def gpu_seconds():
    total = 0.0
    preflight = OUT / 'preflight.json'
    if preflight.is_file():
        total += float(read_json(preflight).get('gpu_seconds') or 0.0)
    status = OUT / 'training_status.json'
    if status.is_file():
        total += sum(float(row['seconds']) for row in read_json(status).get('logs', []))
    for cursor in CLEAN_MILESTONES:
        path = OUT / f'validation_{cursor}.json'
        if path.is_file():
            total += float(read_json(path).get('gpu_seconds') or 0.0)
    diagnostic = OUT / 'diagnostic_relchem.json'
    if diagnostic.is_file():
        total += float(read_json(diagnostic).get('gpu_seconds') or 0.0)
    qualification = OUT / 'qualification_time.json'
    if qualification.is_file():
        total += float(read_json(qualification).get('gpu_seconds') or 0.0)
    return total


def _require_budget(extra):
    if gpu_seconds() + extra > BUDGET_SECONDS:
        _stop('GPU budget exhausted')


def _check_model(model):
    run = _import_run()
    parameters = run.existing.named_trainable_parameters(model)
    coordinates = sum(parameter.numel() for parameter in parameters.values())
    version = int(model.lap_architecture_version)
    if (type(model).__name__ != 'pcPBELMLOptimizerV2Lap' or coordinates != 9446 or version != 1
            or any('tau' in name.lower() for name in parameters)):
        _stop('Model architecture does not match the qualified P536 pilot')
    return parameters


def _parity_batch(model, shadow, bundle, batch, dispersion):
    reference_loss, reference_grad, reference_seconds = reference_mean(
        model, shadow, bundle, batch['relchem'], dispersion)
    loss, grad, infos, seconds = qualified_mean(model, shadow, bundle, batch['relchem'], dispersion)
    discrepancy = relative_l2(grad, reference_grad)
    scalar_gap = abs(loss - reference_loss) / max(abs(reference_loss), 1e-30)
    if discrepancy > RELATIVE_L2_LIMIT or scalar_gap > RELATIVE_L2_LIMIT:
        _stop(f'B8 gradient parity failed at cursor {batch["cursor"]}: L2 {discrepancy}')
    if sum(piece.numel() for piece in grad.values()) != 9446:
        _stop('B8 gradient does not cover 9446 coordinates')
    return {
        'index': batch['cursor'], 'relative_l2': discrepancy, 'scalar_relative': scalar_gap,
        'coordinates': sum(piece.numel() for piece in grad.values()), 'loss': loss,
        'reference_seconds': reference_seconds, 'implementation_seconds': seconds,
        'norm_of_mean': _grad_norm(grad),
        'mean_of_norms': sum(row['grad_norm'] for row in infos) / len(infos),
    }, grad


def preflight(protocol, batches, manifest_sha):
    run = _import_run()
    torch.cuda.synchronize()
    started = time.perf_counter()
    file_before = sha256_file(protocol['initial_state']['path'])
    model = None
    shadow = None
    bundle = None
    peak_allocated = 0
    peak_reserved = 0
    rows = []
    joint_info = None
    try:
        model, shadow = run.model_at(protocol['initial_state'])
        digest_before = run.existing.digest(model)
        bundle = run.PublicationDataset(run.DATA)
        if bundle.manifest['logical_sha256'] != DATA_SHA:
            _stop('Dataset logical hash changed')
        _check_model(model)
        dispersion = bundle.chemistry_dispersions()
        mrks_dispersion = read_json(run.DATA / 'mrks' / 'dispersion.json')
        for index in PARITY_INDICES:
            batch = batches[index]
            row, grad = _parity_batch(model, shadow, bundle, batch, dispersion)
            if index == 0:
                wrapped_loss, wrapped_grad, _, _ = qualified_mean(
                    model, shadow, bundle, [batch['relchem'][0]], dispersion)
                direct_reaction = _load_reaction(bundle, 'relchem', batch['relchem'][0])
                direct_loss, direct_grad = _singleton(model, shadow, direct_reaction, dispersion)
                b1 = relative_l2(wrapped_grad, direct_grad)
                b1_scalar = abs(wrapped_loss - float(direct_loss)) / max(abs(float(direct_loss)), 1e-30)
                if b1 > RELATIVE_L2_LIMIT or b1_scalar > RELATIVE_L2_LIMIT:
                    _stop('B1 wrapper does not reproduce the J251 singleton')
                row['b1_relative_l2'] = b1
                del direct_reaction, direct_grad
                _other_losses, other_raw, system_name, other_seconds = other_tasks(
                    model, shadow, bundle, batch, dispersion, mrks_dispersion)
                peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated())
                peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved())
                measured, measured_raw = run.measure(
                    model, shadow, bundle, load_j251_rows()[0], dispersion, mrks_dispersion, exc_chunk_size=4096)
                other_l2 = {
                    task: relative_l2(other_raw[task], measured_raw[task]) for task in ('ae17', 'exc', 'op')
                }
                if any(value > OTHER_TASK_L2_LIMIT for value in other_l2.values()):
                    _stop('AE17, Exc, or operator gradient differs from the J251 measurement path')
                raw = {'relchem': grad, 'ae17': other_raw['ae17'], 'exc': other_raw['exc'], 'op': other_raw['op']}
                pointers = {task: {name: piece.data_ptr() for name, piece in raw[task].items()} for task in TASKS[1:]}
                joint = scalarize(raw, LAMBDAS)
                if any(raw[task][name].data_ptr() != pointers[task][name] for task in TASKS[1:] for name in raw[task]):
                    _stop('Scalarization mutated a non-relchem gradient')
                manual = {
                    name: sum(raw[task][name].double() * LAMBDAS[task] for task in TASKS) for name in joint
                }
                coefficient_l2 = relative_l2(joint, manual)
                if coefficient_l2 > RELATIVE_L2_LIMIT:
                    _stop('Task coefficients were not applied exactly once')
                parameters = _check_model(model)
                if any(piece.dtype != torch.float32 for piece in parameters.values()):
                    _stop('AdamW parameter storage is not F32')
                if any(piece.dtype != torch.float64 for piece in joint.values()) or not math.isfinite(_grad_norm(joint)):
                    _stop('Joint gradient is not a finite F64 vector')
                if _grad_norm(joint) == 0.0:
                    _stop('Joint gradient norm is zero')
                row.update({
                    'other_seconds': other_seconds, 'other_task_relative_l2': other_l2,
                    'b1_relative_l2': b1, 'joint_norm': _grad_norm(joint), 'joint_dtype': 'float64',
                    'parameter_dtype': 'float32', 'coefficient_relative_l2': coefficient_l2,
                    'system': system_name, 'measured_relchem_ignored': float(measured['losses']['relchem']),
                    'optimizer_step': False,
                })
                joint_info = row
                del measured_raw, joint, manual
            else:
                _, _, _, other_seconds = other_tasks(model, shadow, bundle, batch, dispersion, mrks_dispersion)
                row['other_seconds'] = other_seconds
            peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated())
            peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved())
            rows.append({key: value for key, value in row.items() if key != 'grad'})
            del grad
            torch.cuda.empty_cache()
        if run.existing.digest(model) != digest_before or sha256_file(protocol['initial_state']['path']) != file_before:
            _stop('Preflight changed the model or the P536 checkpoint')
    finally:
        if bundle is not None:
            bundle.close()
        del model, shadow
        torch.cuda.empty_cache()
    samples = [row['implementation_seconds'] + row['other_seconds'] for row in rows]
    median_update = sorted(samples)[len(samples) // 2]
    spent = time.perf_counter() - started
    projected = spent + median_update * 251 + EVAL_RESERVE_SECONDS
    receipt = {
        'passed': True, 'optimizer_step': False, 'gpu_seconds': spent, 'batches': rows,
        'joint': {key: joint_info[key] for key in (
            'joint_norm', 'joint_dtype', 'parameter_dtype', 'coefficient_relative_l2', 'other_task_relative_l2',
            'b1_relative_l2')},
        'projection': {
            'sample_update_seconds': samples, 'median_update_seconds': median_update,
            'projected_seconds': projected, 'reserve_seconds': EVAL_RESERVE_SECONDS,
            'proceed': projected <= BUDGET_SECONDS,
        },
        'model_unchanged': True, 'checkpoint_unchanged': True, 'initial_tensor_sha256': digest_before,
        'manifest_sha256': manifest_sha,
        'peak_allocated_bytes': peak_allocated, 'peak_reserved_bytes': peak_reserved,
    }
    write_json(OUT / 'preflight.json', receipt)
    print('B8_PROJECT', median_update, projected, flush=True)
    if not receipt['projection']['proceed']:
        _stop('Projected runtime exceeds the 180-minute GPU budget')
    return receipt


def _new_optimizer(parameters):
    return torch.optim.AdamW(
        list(parameters.values()), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, foreach=False)


def _log_row(cursor, batch, infos, relchem_loss, relchem_grad, other_losses, other_raw, joint, displacement, before, after, seconds):
    norms = [row['grad_norm'] for row in infos]
    losses = [row['loss'] for row in infos]
    mean_of_norms = sum(norms) / len(norms)
    norm_of_mean = _grad_norm(relchem_grad)
    return {
        'cursor': cursor, 'manifest_index': batch['cursor'],
        'reactions': infos, 'loss_mean': relchem_loss, 'loss_min': min(losses), 'loss_max': max(losses),
        'batch_weight': 0.125, 'presentations': 8,
        'per_reaction_grad_norms': norms, 'norm_of_mean': norm_of_mean, 'mean_of_norms': mean_of_norms,
        'norm_ratio': norm_of_mean / mean_of_norms if mean_of_norms else None,
        'ae17_loss': other_losses['ae17'], 'exc_loss': other_losses['exc'], 'op_loss': other_losses['op'],
        'ae17_norm': _grad_norm(other_raw['ae17']), 'exc_norm': _grad_norm(other_raw['exc']),
        'op_norm': _grad_norm(other_raw['op']), 'joint_norm': _grad_norm(joint),
        'displacement_norm': displacement, 'lr': 1e-4, 'seconds': seconds,
        'before_sha256': before, 'after_sha256': after,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'finite': True, 'divided_by': 8,
    }


def train(protocol, batches, manifest_sha, protocol_sha):
    run = _import_run()
    random.seed(41)
    np.random.seed(41)
    torch.manual_seed(41)
    model, shadow = run.model_at(protocol['initial_state'])
    parameters = _check_model(model)
    optimizer = _new_optimizer(parameters)
    target = OUT / 'ordinary_sgd_adamw'
    target.mkdir(parents=True, exist_ok=True)
    latest = target / 'latest.pt'
    logs = []
    cursor = 0
    if latest.is_file():
        cursor, logs = restore_checkpoint(latest, model, optimizer, manifest_sha, protocol_sha)
        if cursor != len(logs):
            _stop('Checkpoint cursor does not match its log')
        if logs and logs[0]['before_sha256'] != P536_TENSOR:
            _stop('Training did not start from corrected P536')
        for index in range(len(logs) - 1):
            if logs[index]['after_sha256'] != logs[index + 1]['before_sha256']:
                _stop('Model hash chain broke')
        if cursor > 0:
            steps = {int(state['step']) for state in optimizer.state.values()}
            if steps != {cursor}:
                _stop('AdamW step counter does not match the cursor')
    else:
        save_checkpoint(target / 'checkpoint_0.pt', model, optimizer, 0, manifest_sha, protocol_sha, logs)
        save_checkpoint(latest, model, optimizer, 0, manifest_sha, protocol_sha, logs)
    bundle = run.PublicationDataset(run.DATA)
    try:
        if bundle.manifest['logical_sha256'] != DATA_SHA:
            _stop('Dataset logical hash changed')
        dispersion = bundle.chemistry_dispersions()
        mrks_dispersion = read_json(run.DATA / 'mrks' / 'dispersion.json')
        for index in range(cursor, 251):
            if gpu_seconds() + EVAL_RESERVE_SECONDS >= BUDGET_SECONDS:
                _stop('GPU budget exhausted')
            if index >= 10:
                sample = sorted(row['seconds'] for row in logs)
                median_time = sample[len(sample) // 2]
                projected = gpu_seconds() + median_time * (251 - index) + EVAL_RESERVE_SECONDS
                if projected > BUDGET_SECONDS:
                    _stop('Projected runtime exceeds the 180-minute GPU budget')
            batch = batches[index]
            if batch['relchem'][0] != load_j251_rows()[index]['relchem']:
                _stop('B8 manifest no longer preserves the J251 base reaction')
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            before = run.existing.digest(model)
            before_weights = {name: parameter.detach().clone() for name, parameter in parameters.items()}
            relchem_loss, relchem_grad, infos, _ = qualified_mean(
                model, shadow, bundle, batch['relchem'], dispersion)
            other_losses, other_raw, system_name, _ = other_tasks(
                model, shadow, bundle, batch, dispersion, mrks_dispersion)
            raw = {
                'relchem': relchem_grad, 'ae17': other_raw['ae17'], 'exc': other_raw['exc'], 'op': other_raw['op'],
            }
            if any(not math.isfinite(value) for value in (relchem_loss, *other_losses.values())):
                _stop('Nonfinite loss')
            joint = run.adamw_step(model, optimizer, raw, LAMBDAS)
            steps = {int(state['step']) for state in optimizer.state.values()}
            if steps != {index + 1}:
                _stop('AdamW did not take exactly one step')
            delta = torch.cat([
                (parameter.detach() - before_weights[name]).double().reshape(-1) for name, parameter in parameters.items()
            ])
            torch.cuda.synchronize()
            row = _log_row(
                index + 1, batch, infos, relchem_loss, relchem_grad, other_losses, other_raw, joint,
                float(torch.linalg.vector_norm(delta)), before, run.existing.digest(model), time.perf_counter() - started)
            row['system'] = system_name
            row['mrks_id'] = batch['mrks_id']
            row['ae17_identity'] = batch['ae17']['identity']
            row['ae17_variant'] = batch['ae17']['variant']
            logs.append(row)
            del raw, joint, relchem_grad, other_raw, before_weights, delta
            torch.cuda.empty_cache()
            cursor = index + 1
            save_checkpoint(latest, model, optimizer, cursor, manifest_sha, protocol_sha, logs)
            milestone = target / f'checkpoint_{cursor}.pt'
            if cursor in MILESTONES and not milestone.exists():
                save_checkpoint(milestone, model, optimizer, cursor, manifest_sha, protocol_sha, logs)
            write_json(OUT / 'training_status.json', {'cursor': cursor, 'complete': cursor == 251, 'logs': logs})
            print('B8_UPDATE', cursor, row['seconds'], row['loss_mean'], row['norm_ratio'], flush=True)
    finally:
        bundle.close()
    return cursor


def _evaluate_clean28(protocol):
    run = _import_run()
    from tools.evaluate_microbatch_endpoint import validation
    bundle = run.PublicationDataset(run.DATA)
    try:
        for cursor in CLEAN_MILESTONES:
            path = OUT / f'validation_{cursor}.json'
            if path.is_file():
                continue
            _require_budget(90)
            checkpoint = OUT / 'ordinary_sgd_adamw' / f'checkpoint_{cursor}.pt'
            model, _shadow = run.model_at(protocol['initial_state'])
            model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'])
            before = run.existing.digest(model)
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = validation(model, bundle)
            torch.cuda.synchronize()
            if run.existing.digest(model) != before:
                _stop('Validation changed the checkpoint weights')
            rows = [row for row in result['reaction_rows'] if row['clean']]
            if len(rows) != 28:
                _stop('Clean28 did not contain 28 reactions')
            payload = {
                'cursor': cursor, 'clean28': result['clean28'], 'reaction_rows': rows,
                'full30_discarded': True, 'full30_used_for_selection': False,
                'gpu_seconds': time.perf_counter() - started, 'optimizer_step': False,
                'checkpoint_sha256': sha256_file(checkpoint),
            }
            write_json(path, payload)
            print('B8_CLEAN28', cursor, result['clean28'], flush=True)
            del model
            torch.cuda.empty_cache()
    finally:
        bundle.close()


def _chemistry_scalar(protocol, cursor, task):
    run = _import_run()
    evaluation = read_json(REPO / 'relchem_joint_epoch_evaluation_manifest.json')
    if evaluation['policy'] != 'one-variant-per-identity-v1':
        _stop('Evaluation manifest policy changed')
    checkpoint = OUT / 'ordinary_sgd_adamw' / f'checkpoint_{cursor}.pt'
    model, shadow = run.model_at(protocol['initial_state'])
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'])
    before = run.existing.digest(model)
    bundle = run.PublicationDataset(run.DATA)
    rows = []
    started = time.perf_counter()
    try:
        dispersion = bundle.chemistry_dispersions()
        selected = [row for row in evaluation['rows'] if row['task'] == task]
        expected = 251 if task == 'relchem' else 17
        if len(selected) != expected or len({row['identity'] for row in selected}) != expected:
            _stop('Evaluation manifest does not contain one variant per identity')
        for spec in selected:
            if gpu_seconds() > BUDGET_SECONDS:
                _stop('GPU budget exhausted')
            reaction = _load_reaction(bundle, task, {
                'identity': spec['identity'], 'variant': spec['variant'],
                'database': bundle.reactions[spec['identity']]['database'],
            })
            with torch.no_grad():
                value = float(run.chemistry(model, shadow, reaction, dispersion)())
            if not math.isfinite(value):
                _stop('Nonfinite chemistry scalar')
            rows.append({
                'identity': spec['identity'], 'variant': spec['variant'], 'task': task,
                'database': bundle.reactions[spec['identity']]['database'], 'loss': value,
            })
            del reaction
        if run.existing.digest(model) != before:
            _stop('Scalar evaluation changed the checkpoint weights')
    finally:
        bundle.close()
    objective = sum(row['loss'] for row in rows) / len(rows)
    return {
        'cursor': cursor, 'task': task, 'objective': objective, 'ratio': objective / BASELINE[task],
        'rows': rows, 'gpu_seconds': time.perf_counter() - started, 'gradients': False,
    }


def _database_report(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row['database'], []).append(row['loss'])
    report = {}
    for name, values in sorted(grouped.items()):
        report[name] = {
            'count': len(values), 'mean': sum(values) / len(values), 'contribution': sum(values) / 251,
        }
    return report


def _qualify_or_diagnose(protocol):
    clean = {cursor: read_json(OUT / f'validation_{cursor}.json')['clean28'] for cursor in CLEAN_MILESTONES}
    screened = [cursor for cursor in CLEAN_MILESTONES if beats_historical(clean[cursor])]
    if not screened:
        best = min(CLEAN_MILESTONES, key=lambda cursor: (clean[cursor], cursor))
        diagnostic = _chemistry_scalar(protocol, best, 'relchem')
        diagnostic['databases'] = _database_report(diagnostic['rows'])
        diagnostic['role'] = 'diagnostic_not_qualification'
        diagnostic['j251_relchem_ratio'] = J251_RATIOS['relchem']
        write_json(OUT / 'diagnostic_relchem.json', diagnostic)
        print('B8_DIAG', best, diagnostic['ratio'], flush=True)
        return
    science = {}
    qualification_seconds = 0.0
    for cursor in screened:
        relchem = _chemistry_scalar(protocol, cursor, 'relchem')
        qualification_seconds += relchem['gpu_seconds']
        record = {'status': 'complete', 'ratios': {'relchem': relchem['ratio']}, 'objectives': {'relchem': relchem['objective']}}
        if not (math.isfinite(relchem['ratio']) and relchem['ratio'] < 1.0):
            record['stopped_at'] = 'relchem'
            science[str(cursor)] = record
            continue
        ae17 = _chemistry_scalar(protocol, cursor, 'ae17')
        qualification_seconds += ae17['gpu_seconds']
        record['ratios']['ae17'] = ae17['ratio']
        record['objectives']['ae17'] = ae17['objective']
        if not (math.isfinite(ae17['ratio']) and ae17['ratio'] < 1.0):
            record['stopped_at'] = 'ae17'
            science[str(cursor)] = record
            continue
        _require_budget(90)
        from tools.evaluate_microbatch_endpoint import evaluate
        if not (OUT / 'evaluation_manifest.json').is_file():
            source = REPO / 'relchem_joint_epoch_evaluation_manifest.json'
            (OUT / 'evaluation_manifest.json').write_bytes(source.read_bytes())
        torch.cuda.synchronize()
        started = time.perf_counter()
        evaluate(OUT, cursor, 'mrks', 3600)
        qualification_seconds += time.perf_counter() - started
        mrks = read_json(OUT / f'endpoint_{cursor}_mrks.json')
        if not mrks.get('complete'):
            record['status'] = 'incomplete'
            science[str(cursor)] = record
            write_json(OUT / 'science.json', science)
            _stop('Scientific mRKS evaluation incomplete')
        for task in ('exc', 'op'):
            record['objectives'][task] = mrks['objectives'][task]
            record['ratios'][task] = mrks['objectives'][task] / BASELINE[task]
        science[str(cursor)] = record
    write_json(OUT / 'qualification_time.json', {'gpu_seconds': qualification_seconds})
    write_json(OUT / 'science.json', science)


def _clean_rows(payload):
    rows = payload['reaction_rows'] if 'reaction_rows' in payload else payload['metrics']['reaction_rows']
    return [row for row in rows if row.get('clean', True)]


def _reaction_analysis(cursor):
    current = _clean_rows(read_json(OUT / f'validation_{cursor}.json'))
    p536 = {row['reaction_id']: row for row in _clean_rows(read_json(
        REPO.parent / 'lap_relchem_joint_epoch_20261009' / 'R' / 'endpoint_0_validation.json'))}
    j251 = {row['reaction_id']: row for row in _clean_rows(read_json(
        REPO.parent / 'lap_relchem_joint_epoch_20261009' / 'J' / 'endpoint_251_validation.json'))}
    diet = {}
    diet_path = REPO.parent / 'publication_dataset_v1' / 'validation' / 'diet30_reactions.jsonl'
    for line in diet_path.read_text(encoding='utf-8').splitlines():
        if line:
            row = json.loads(line)
            diet[row['source_id']] = row['database']
    analyzed = []
    for row in current:
        analyzed.append({
            'reaction_id': row['reaction_id'], 'database': diet.get(row['reaction_id'], 'UNJOINED'),
            'signed_error_kcal_mol': row['signed_error_kcal_mol'],
            'weighted_absolute_error': row['weighted_absolute_error'],
            'delta_weighted_vs_p536': row['weighted_absolute_error'] - p536[row['reaction_id']]['weighted_absolute_error'],
            'delta_weighted_vs_j251': row['weighted_absolute_error'] - j251[row['reaction_id']]['weighted_absolute_error'],
        })
    return analyzed


def _summary():
    preflight = read_json(OUT / 'preflight.json') if (OUT / 'preflight.json').is_file() else None
    status = read_json(OUT / 'training_status.json') if (OUT / 'training_status.json').is_file() else {'cursor': 0, 'logs': []}
    clean = {}
    analyses = {}
    for cursor in CLEAN_MILESTONES:
        path = OUT / f'validation_{cursor}.json'
        if path.is_file():
            clean[cursor] = read_json(path)['clean28']
            analyses[cursor] = _reaction_analysis(cursor)
    diagnostic = read_json(OUT / 'diagnostic_relchem.json') if (OUT / 'diagnostic_relchem.json').is_file() else None
    science = read_json(OUT / 'science.json') if (OUT / 'science.json').is_file() else {}
    logs = status.get('logs', [])
    ratios = [row['norm_ratio'] for row in logs if row.get('norm_ratio') is not None]
    peak_allocated = max((row['peak_allocated_bytes'] for row in logs), default=0)
    peak_reserved = max((row['peak_reserved_bytes'] for row in logs), default=0)
    if preflight:
        peak_allocated = max(peak_allocated, int(preflight.get('peak_allocated_bytes') or 0))
        peak_reserved = max(peak_reserved, int(preflight.get('peak_reserved_bytes') or 0))
    memory_path = OUT / 'memory.json'
    if memory_path.is_file():
        memory = read_json(memory_path)
        peak_allocated = max(peak_allocated, memory.get('peak_allocated_bytes', 0))
        peak_reserved = max(peak_reserved, memory.get('peak_reserved_bytes', 0))
    protocol = read_json(OUT / 'protocol.json') if (OUT / 'protocol.json').is_file() else {}
    payload = {
        'complete': status.get('complete') is True and set(clean) == set(CLEAN_MILESTONES),
        'clean28': clean,
        'science': science,
        'diagnostic_complete': diagnostic is not None and diagnostic.get('task') == 'relchem',
    }
    if payload['complete'] and any(beats_historical(value) for value in clean.values()) and not science:
        payload['complete'] = False
    return {
        'decision': classify(payload) if protocol else 'PARTIAL',
        'manifest_sha256': protocol.get('b8_manifest_sha256'),
        'updates': status.get('cursor', 0),
        'presentations': sum(row.get('presentations', 0) for row in logs),
        'gpu_seconds': gpu_seconds(),
        'peak_allocated_bytes': peak_allocated,
        'peak_reserved_bytes': peak_reserved,
        'preflight': preflight,
        'clean28': clean,
        'analyses': analyses,
        'diagnostic': diagnostic,
        'science': science,
        'norm_ratios': ratios,
        'stop': read_json(OUT / 'stop.json')['reason'] if (OUT / 'stop.json').is_file() else None,
        'identity_counts': None,
    }


def _fmt(value):
    if isinstance(value, str):
        return value
    if value is None:
        return 'NA'
    return f'{value:.9f}'


def _write_report(summary):
    preflight = summary['preflight'] or {}
    batches = preflight.get('batches', [])
    lines = [
        '# Balanced B8 chemistry AdamW',
        '',
        f"**Classification: {summary['decision']}**",
        '',
        'One new 251-update AdamW arm. The only intended change from J251 is the relchem',
        'estimator: eight qualified singleton gradients, averaged with equal weight 1/8.',
        'J251 was not retrained. Matched update counts are not matched data: J251 used 251',
        'chemistry presentations and B8 uses 2008. A difference cannot be attributed only to',
        'variance reduction. Successive batches are cyclic shifts of one permutation, not IID draws.',
        'At frozen parameters, the mean over a uniform random update index equals the full',
        'fixed-variant 251-reaction gradient. Parameters move during the run, so that equality',
        'is not a conditional-unbiasedness claim at each optimizer update.',
        '',
        '## Manifest',
        '',
        f"B8 manifest SHA256: `{summary['manifest_sha256']}`.",
        'Offsets: `(0, 31, 62, 93, 124, 155, 186, 217)`.',
        'Each of the 251 J251 identities is scheduled eight times, always with its original',
        'J251 quadrature variant. The first reaction of update t is the J251 reaction at t.',
        'AE17 and mRKS entries are the J251 entries. Database exposure is eight times the',
        'J251 population. No validation or future-test identity is in the training manifest.',
        '',
        '## Parity',
        '',
    ]
    if not batches:
        lines.append('Real B8 parity did not finish.')
    for batch in batches:
        lines.append(
            f"Index {batch['index']}: relative L2 `{batch['relative_l2']:.3e}`, "
            f"scalar relative `{batch['scalar_relative']:.3e}`, coordinates `{batch['coordinates']}`."
        )
    joint = preflight.get('joint') or {}
    if joint:
        lines.append(
            f"B1 singleton relative L2 `{joint.get('b1_relative_l2')}`. "
            f"Other-task relative L2 `{joint.get('other_task_relative_l2')}`. "
            f"Coefficient check `{joint.get('coefficient_relative_l2')}`. "
            f"Joint dtype `{joint.get('joint_dtype')}`, parameter dtype `{joint.get('parameter_dtype')}`, "
            f"optimizer step `{preflight.get('optimizer_step')}`."
        )
    lines.extend(['', '## Other tasks', '',
                   'AE17, Exc, and the operator gradient are computed by the J251 measurement calls.',
                   'They are not averaged over eight systems. Scalarization applies the historical',
                   'coefficients once. The F32 cast remains inside `adamw_step`.',
                   '', '## Runtime', '',
                   f"Completed optimizer updates: `{summary['updates']}`.",
                   f"Relchem presentations: `{summary['presentations']}`.",
                   f"Cumulative new GPU time: `{summary['gpu_seconds']:.3f}` seconds.",
                   f"Peak allocated CUDA memory: `{summary['peak_allocated_bytes']}` bytes.",
                   f"Peak reserved CUDA memory: `{summary['peak_reserved_bytes']}` bytes.",
                   ''])
    if preflight.get('projection'):
        projection = preflight['projection']
        lines.append(
            f"Preflight median update estimate: `{projection['median_update_seconds']:.3f}` seconds. "
            f"Projected total including the evaluation reserve: `{projection['projected_seconds']:.3f}`."
        )
    lines.extend(['', '## Clean28', '',
                   '| Model | Updates | Clean28 | Relchem ratio | AE17 ratio | Exc ratio | Op ratio | Eligible |',
                   '|---|---:|---:|---:|---:|---:|---:|---|',
                   f'| P536 | 0 | {_fmt(P536_CLEAN28)} | 1 | 1 | 1 | 1 | No, strict threshold |',
                   f'| J251 control | 251 | {_fmt(J251_CLEAN28)} | {_fmt(J251_RATIOS["relchem"])} | {_fmt(J251_RATIOS["ae17"])} | {_fmt(J251_RATIOS["exc"])} | {_fmt(J251_RATIOS["op"])} | Yes |',
                   f'| Historical best | 80 | {_fmt(HISTORICAL_BEST_CLEAN28)} | 1.034869578 | 0.080727952 | 0.059058103 | 0.944069705 | No |',
                   f'| Best eligible interpolation |  | {_fmt(BEST_ELIGIBLE_CLEAN28)} | 0.991430353 | 0.122272970 | 0.054614319 | 0.929237644 | Yes |'])
    for cursor in CLEAN_MILESTONES:
        clean = summary['clean28'].get(cursor)
        science = summary['science'].get(str(cursor))
        if clean is None:
            lines.append(f'| New t{cursor} | {cursor} | NA | NA | NA | NA | NA | NA |')
            continue
        if science and science.get('status') == 'complete' and set(science.get('ratios', {})) == set(TASKS):
            ratios = science['ratios']
            eligible = scientifically_eligible(ratios)
            lines.append(
                f"| New t{cursor} | {cursor} | {_fmt(clean)} | {_fmt(ratios['relchem'])} | {_fmt(ratios['ae17'])} | "
                f"{_fmt(ratios['exc'])} | {_fmt(ratios['op'])} | {eligible} |"
            )
        elif science and 'relchem' in science.get('ratios', {}):
            ratios = science['ratios']
            cells = []
            for task in TASKS:
                cells.append(_fmt(ratios[task]) if task in ratios else 'NOT EVALUATED — EARLIER OBJECTIVE FAILED')
            lines.append(f"| New t{cursor} | {cursor} | {_fmt(clean)} | {' | '.join(cells)} | No |")
        elif beats_historical(clean):
            lines.append(f'| New t{cursor} | {cursor} | {_fmt(clean)} | NA | NA | NA | NA | NA |')
        else:
            lines.append(f'| New t{cursor} | {cursor} | {_fmt(clean)} | {GATE} | {GATE} | {GATE} | {GATE} | {GATE} |')
    lines.extend(['', 'Differences below use the exact receipts. t64, t128, and t192 are not matched to J251 t251.', ''])
    for cursor in CLEAN_MILESTONES:
        clean = summary['clean28'].get(cursor)
        if clean is None:
            continue
        lines.append(
            f't{cursor}: Clean28 `{_fmt(clean)}`, versus P536 `{clean - P536_CLEAN28:+.6f}`, '
            f'versus J251 `{clean - J251_CLEAN28:+.6f}`, versus historical best `{clean - HISTORICAL_BEST_CLEAN28:+.6f}`.'
        )
        analyzed = summary['analyses'].get(cursor) or []
        improved = sum(row['delta_weighted_vs_j251'] < 0 for row in analyzed)
        worsened = sum(row['delta_weighted_vs_j251'] > 0 for row in analyzed)
        lines.append(f'Improved versus J251: {improved}. Worsened versus J251: {worsened}.')
        largest = sorted(analyzed, key=lambda row: row['weighted_absolute_error'], reverse=True)[:5]
        for row in largest:
            lines.append(
                f"- `{row['reaction_id']}` ({row['database']}): weighted `{row['weighted_absolute_error']:.6f}`, "
                f"signed `{row['signed_error_kcal_mol']:.6f}`, delta J251 `{row['delta_weighted_vs_j251']:+.6f}`."
            )
    diagnostic = summary['diagnostic']
    lines.extend(['', '## Full251 diagnostic', ''])
    if diagnostic:
        lines.append(
            f"Best Clean28 checkpoint t{diagnostic['cursor']} fixed-panel relchem ratio "
            f"`{diagnostic['ratio']:.9f}` (objective `{diagnostic['objective']:.9f}`). "
            'This is a no-gradient population diagnostic, not scientific qualification. '
            f"J251 relchem ratio is `{J251_RATIOS['relchem']:.9f}`."
        )
        lines.append('')
        lines.append('| Database | Count | Mean loss | Contribution |')
        lines.append('|---|---:|---:|---:|')
        for name, row in diagnostic['databases'].items():
            lines.append(f"| {name} | {row['count']} | {row['mean']:.9f} | {row['contribution']:.9f} |")
    elif summary['science']:
        lines.append('Accuracy screening passed, so the diagnostic was not used. Scientific ratios are in the table.')
    else:
        lines.append('Not run.')
    ratios = summary['norm_ratios']
    lines.extend(['', '## Gradient diagnostics', ''])
    if ratios:
        ordered = sorted(ratios)
        lines.append(
            f"`norm(mean(g_i)) / mean(norm(g_i))` over {len(ordered)} updates: "
            f"min `{ordered[0]:.6f}`, median `{ordered[len(ordered) // 2]:.6f}`, max `{ordered[-1]:.6f}`."
        )
        lines.append('The ratio compares those two norms. It is not a variance estimate.')
    else:
        lines.append('No training updates were recorded.')
    lines.extend(['', '## Provenance', '',
                   f'P536 file `{P536_FILE}`.',
                   f'P536 tensor `{P536_TENSOR}`.',
                   f'J251 manifest `{J251_MANIFEST_SHA}`.',
                   f'Evaluation manifest `{EVAL_MANIFEST_SHA}`.',
                   f'Calibration `{CALIBRATION_SHA}`.',
                   'Production physics files were hashed against the J251 protocol before training.',
                   '', '## Next recommendation', ''])
    lines.extend(_recommendation(summary))
    lines.append('')
    lines.append('This recommendation was not executed.')
    (REPO / 'lap_b8_adamw_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _recommendation(summary):
    diagnostic = summary['diagnostic']
    if summary['decision'] == 'PARTIAL':
        return ['Resume this same arm from the last matching checkpoint if the budget and checkpoint remain. Do not start a second batch size, seed, or loss.']
    if summary['decision'] in ('BREAKTHROUGH', 'STRONG PROGRESS'):
        return ['Freeze the eligible checkpoint. A later confirmatory seed would be a separate experiment. Do not retune coefficients or the learning rate from this result.']
    if summary['decision'] == 'ACCURACY-ONLY PROGRESS':
        return ['Clean28 moved without scientific eligibility. Do not promote the checkpoint. The next study should explain the failed scientific objective before any longer run.']
    if not diagnostic:
        return ['Do not launch a follow-up until the authorized diagnostic or qualification exists.']
    ratio = diagnostic['ratio']
    if ratio < 1.0 and ratio < J251_RATIOS['relchem']:
        return [(
            'Full251 relchem improved relative to both P536 and J251 while Clean28 stayed above the historical best. '
            'That is a generalization mismatch, not proof that the population estimator failed. '
            'Rank a longer fixed-loss, learning-rate-scheduled continuation of this same B8 loss ahead of SVRG. '
            'Do not search another batch size first. Do not launch it.'
        )]
    if ratio < 1.0:
        return [(
            'Full251 relchem beat P536 but did not beat J251. Balanced B8 did not improve the matched population objective. '
            'Rank an SVRG-style control variate ahead of a longer B8 schedule. Do not launch it.'
        )]
    return [(
        'Full251 relchem did not beat P536. B8 did not remove the chemistry-optimization bottleneck on this seed. '
        'Rank SVRG ahead of a longer fixed-loss B8 schedule. Do not launch it.'
    )]


def _write_metrics(summary):
    diagnostic = summary['diagnostic']
    diagnostic_public = None
    if diagnostic:
        diagnostic_public = {
            'cursor': diagnostic['cursor'], 'objective': diagnostic['objective'], 'ratio': diagnostic['ratio'],
            'databases': diagnostic['databases'], 'role': diagnostic['role'], 'gradients': False,
        }
    ratios = summary['norm_ratios']
    ordered = sorted(ratios)
    payload = {
        'decision': summary['decision'],
        'b8_manifest_sha256': summary['manifest_sha256'],
        'j251_manifest_sha256': J251_MANIFEST_SHA,
        'updates': summary['updates'],
        'presentations': summary['presentations'],
        'identity_presentations': 8,
        'gpu_seconds': summary['gpu_seconds'],
        'peak_allocated_bytes': summary['peak_allocated_bytes'],
        'peak_reserved_bytes': summary['peak_reserved_bytes'],
        'clean28': {str(key): value for key, value in summary['clean28'].items()},
        'science': summary['science'],
        'diagnostic': diagnostic_public,
        'norm_ratio': None if not ordered else {
            'count': len(ordered), 'min': ordered[0], 'median': ordered[len(ordered) // 2], 'max': ordered[-1],
        },
        'preflight': None if not summary['preflight'] else {
            'passed': summary['preflight'].get('passed'),
            'projection': summary['preflight'].get('projection'),
            'batches': summary['preflight'].get('batches'),
            'joint': summary['preflight'].get('joint'),
        },
        'analyses': {str(key): value for key, value in summary['analyses'].items()},
        'stop': summary['stop'],
        'controls': {
            'p536_clean28': P536_CLEAN28, 'j251_clean28': J251_CLEAN28,
            'historical_best_clean28': HISTORICAL_BEST_CLEAN28,
            'best_eligible_clean28': BEST_ELIGIBLE_CLEAN28, 'j251_ratios': J251_RATIOS,
        },
    }
    write_json(REPO / 'lap_b8_adamw_metrics.json', payload)


def render():
    if not OUT.exists():
        return
    summary = _summary()
    _write_report(summary)
    _write_metrics(summary)


def execute():
    if OUT.exists():
        existing = OUT / 'protocol.json'
        if existing.is_file() and read_json(existing).get('protocol_id') != PROTOCOL_ID:
            raise SystemExit('Scratch directory belongs to a different experiment')
        if not existing.is_file() and any(OUT.iterdir()):
            raise SystemExit('Scratch directory exists and is not this experiment')
    source_protocol = verify_sources()
    prior = read_json(OUT / 'stop.json')['reason'] if (OUT / 'stop.json').is_file() else ''
    if prior and any(token in prior for token in ('Nonfinite', 'parity', 'Parity', 'mismatch', 'changed')):
        raise SystemExit(prior)
    j251_rows = load_j251_rows()
    batches = build_batches(j251_rows)
    properties = assert_batch_properties(batches, j251_rows)
    assert_split_disjoint(batches)
    manifest_text = dump(batches)
    manifest_sha = sha256_text(manifest_text)
    payload = _protocol_payload(manifest_sha, source_protocol)
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_path = OUT / 'protocol.json'
    if protocol_path.is_file() and read_json(protocol_path) != payload:
        raise SystemExit('Frozen B8 protocol changed')
    manifest_path = OUT / 'b8_manifest.json'
    if manifest_path.is_file():
        if json.loads(manifest_path.read_text(encoding='utf-8')) != batches:
            raise SystemExit('Frozen B8 manifest changed')
        if sha256_file(manifest_path) != manifest_sha:
            manifest_path.write_text(manifest_text, encoding='utf-8', newline='\n')
        if sha256_file(manifest_path) != manifest_sha:
            raise SystemExit('Frozen B8 manifest changed')
    else:
        manifest_path.write_text(manifest_text, encoding='utf-8', newline='\n')
    if not protocol_path.is_file():
        write_json(protocol_path, payload)
    protocol_sha = sha256_file(protocol_path)
    write_json(OUT / 'manifest_properties.json', properties)
    if not (OUT / 'preflight.json').is_file():
        preflight(source_protocol, batches, manifest_sha)
    elif not read_json(OUT / 'preflight.json').get('passed'):
        _stop('Gradient parity previously failed')
    elif not read_json(OUT / 'preflight.json')['projection']['proceed']:
        _stop('Projected runtime exceeds the 180-minute GPU budget')
    status_path = OUT / 'training_status.json'
    cursor = read_json(status_path).get('cursor', 0) if status_path.is_file() else 0
    if cursor < 251:
        train(source_protocol, batches, manifest_sha, protocol_sha)
    status = read_json(OUT / 'training_status.json')
    if status.get('complete'):
        _evaluate_clean28(source_protocol)
        clean = {item: read_json(OUT / f'validation_{item}.json')['clean28'] for item in CLEAN_MILESTONES}
        screened = [item for item in CLEAN_MILESTONES if beats_historical(clean[item])]
        diagnostic_done = (OUT / 'diagnostic_relchem.json').is_file()
        science_done = (OUT / 'science.json').is_file()
        if (screened and not science_done) or (not screened and not diagnostic_done):
            _qualify_or_diagnose(source_protocol)
    stale = OUT / 'stop.json'
    if stale.is_file() and read_json(OUT / 'training_status.json').get('complete'):
        stale.unlink()


def main():
    if '--report-only' in sys.argv:
        render()
        return
    try:
        execute()
    except SystemExit:
        raise
    except (FloatingPointError, RuntimeError, AssertionError, ValueError) as error:
        if OUT.exists():
            write_json(OUT / 'stop.json', {'reason': type(error).__name__ + ': ' + str(error)})
        raise
    finally:
        if OUT.exists():
            render()


if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    main()


def _ast_guard():
    """Imported by tests. Training must not call backward, optimizer.step, or .to()."""
    tree = ast.parse(Path(__file__).read_text(encoding='utf-8'))
    forbidden = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {'backward', 'step', 'to'}:
            forbidden.append(node.func.attr)
    return forbidden
