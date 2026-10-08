"""Population-matched fixed calibration and fresh-AdamW direction audit only."""
import copy
import hashlib
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_lap_microbatch as run
from train_models.lap_fixed_adamw import weighted_gradient

OLD = run.ROOT.parent / 'lap_adamw_90update_20261008'
OUT = run.ROOT.parent / 'lap_population_adamw_90update_20261008'
TASKS = run.TASKS


def draws(bundle, count, seed):
    rows, systems = list(bundle.reactions.values()), sorted(bundle.systems)
    result = []
    for i in range(count):
        value = int.from_bytes(hashlib.sha256(f'{seed}:mrks:{i}'.encode()).digest(), 'big')
        result.append({'cursor': i, **{t: run.sample_pair(rows, t, i, seed) for t in ('relchem', 'ae17')},
                       'mrks_id': random.Random(value).choice(systems)})
    return result


def bootstrap(norms, seed=930339, repetitions=10000):
    rng = np.random.default_rng(seed)
    medians = np.median(norms[rng.integers(0, len(norms), (repetitions, len(norms)))], axis=1)
    return {t: {'median_95pct_interval': np.percentile(medians[:, j], [2.5, 97.5]).tolist(),
                'coefficient_95pct_interval': np.percentile(1 / (4 * np.maximum(medians[:, j], 1e-12)), [2.5, 97.5]).tolist()}
            for j, t in enumerate(TASKS)}


def geometry(matrix, coefficients):
    weights = matrix.new_tensor([coefficients[t] for t in TASKS])
    joint = weights @ matrix
    length = joint.norm()
    if not torch.isfinite(length) or length <= 0:
        raise FloatingPointError('Undefined weighted-gradient direction')
    task_norms = matrix.norm(dim=1)
    scales = weights * task_norms
    unit = joint / length
    return joint, {'weighted_norm': float(length), 'norm_shares': (scales / scales.sum()).tolist(),
                   'task_dots_unit_weighted_direction': (matrix @ unit).tolist(),
                   'normalized_task_progress': ((matrix @ unit) / task_norms).tolist(),
                   'signed_fraction_of_joint_squared_norm': (weights * (matrix @ joint) / length.square()).tolist()}


def cosine(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


def main():
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    bundle = run.PublicationDataset(run.DATA)
    assert bundle.manifest['logical_sha256'] == run.DATA_SHA
    old_protocol = run.read(run.ROOT / 'microbatch_90update_protocol.json')
    old_coefficients = run.read(OLD / 'calibration.json')['lambda']
    calibration_manifest = draws(bundle, 48, 930337)
    audit_manifest = draws(bundle, 8, 930338)
    run.write(OUT / 'calibration_manifest.json', calibration_manifest)
    run.write(OUT / 'audit_manifest.json', audit_manifest)
    run.write(OUT / 'calibration_protocol.json', {
        'count': 48, 'seed': 930337, 'audit_count': 8, 'audit_seed': 930338,
        'bootstrap_seed': 930339, 'bootstrap_repetitions': 10000,
        'initial_state': old_protocol['initial_state'], 'dataset_sha256': run.DATA_SHA,
        'sampling': 'IID uniform task identity, IID uniform variant; IID uniform mRKS system',
        'rare_databases': 'Report only; no forced coverage or allocation',
        'rule': 'lambda=1/(4*max(median_norm,1e-12))',
        'count_rationale': '48 is the smallest requested panel; uncertainty is reported, no adaptive additions',
        'gate': {'median_relchem_norm_share_gain_min': 2.0,
                 'median_weighted_direction_angle_deg_min': 5.0,
                 'alternative_median_adamw_displacement_angle_deg_min': 5.0},
        'trainer_source_sha256': run.file_sha256(run.ROOT / 'train_lap_microbatch.py'),
        'physics_sources_sha256': old_protocol['physics_sources_sha256'],
        'tool_sha256': run.file_sha256(__file__)})
    model, shadow = run.model_at(old_protocol['initial_state'])
    parameters = run.existing.named_trainable_parameters(model)
    run.write(OUT / 'parameter_order.json', [{'name': n, 'shape': list(p.shape), 'numel': p.numel()} for n, p in parameters.items()])
    assert sum(p.numel() for p in parameters.values()) == 9446
    frozen = copy.deepcopy(model.state_dict())
    initial_sha = run.existing.digest(model)
    chem_dispersion = bundle.chemistry_dispersions()
    mrks_dispersion = run.read(run.DATA / 'mrks/dispersion.json')
    records = []
    for entry in calibration_manifest:
        record, raw = run.measure(model, shadow, bundle, entry, chem_dispersion, mrks_dispersion, 4096)
        matrix = torch.stack([torch.cat([g.flatten() for g in raw[t].values()]) for t in TASKS])
        record['task_gram'] = (matrix @ matrix.T).cpu().tolist()
        records.append(record)
        run.write(OUT / 'calibration_records.json', records)
        assert run.existing.digest(model) == initial_sha
        print('POPULATION_CALIBRATION', len(records), record['norms'], flush=True)
        del raw, matrix
    norms = np.array([[r['norms'][t] for t in TASKS] for r in records])
    scales, coefficients = run.fixed_coefficients(norms)
    calibration = {'scales': scales, 'lambda': coefficients, 'epsilon': 1e-12, 'C': 4.0,
                   'count': 48, 'statistics': {t: dict(zip(('p25', 'median', 'p75', 'p90', 'max'),
                       np.percentile(norms[:, j], [25, 50, 75, 90, 100]).tolist())) for j, t in enumerate(TASKS)},
                   'bootstrap': bootstrap(norms), 'old_lambda': old_coefficients}
    # Native task norms and signed contributions to the sum are available from Gram matrices.
    calibration['panel_comparison'] = {}
    for label, weights in (('old', old_coefficients), ('new', coefficients)):
        w = np.array([weights[t] for t in TASKS])
        shares = norms * w / (norms * w).sum(axis=1, keepdims=True)
        contributions = []
        for record in records:
            gram = np.array(record['task_gram'])
            contributions.append(w * (gram @ w) / (w @ gram @ w))
        calibration['panel_comparison'][label] = {'median_norm_shares': dict(zip(TASKS, np.median(shares, axis=0).tolist())),
            'median_signed_fraction_of_joint_squared_norm': dict(zip(TASKS, np.median(contributions, axis=0).tolist()))}
    run.write(OUT / 'calibration_statistics.json', calibration)
    audit = []
    for entry in audit_manifest:
        record, raw = run.measure(model, shadow, bundle, entry, chem_dispersion, mrks_dispersion, 4096)
        matrix = torch.stack([torch.cat([g.flatten() for g in raw[t].values()]) for t in TASKS])
        arrays, directions, steps = {t: matrix[j].cpu().numpy() for j, t in enumerate(TASKS)}, {}, {}
        row = {'sample': entry, 'norms': record['norms']}
        for label, weights in (('old', old_coefficients), ('new', coefficients)):
            joint, details = geometry(matrix, weights)
            # Independently check the existing dictionary-based weighted aggregation.
            reconstructed = torch.cat([g.flatten() for g in weighted_gradient(raw, weights).values()])
            torch.testing.assert_close(reconstructed, joint, rtol=1e-12, atol=1e-12)
            before = torch.cat([p.detach().flatten().double() for p in parameters.values()])
            optimizer = torch.optim.AdamW(parameters.values(), lr=1e-6, **{**old_protocol['adamw'], 'betas': tuple(old_protocol['adamw']['betas'])})
            run.adamw_step(model, optimizer, raw, weights)
            after = torch.cat([p.detach().flatten().double() for p in parameters.values()])
            displacement = before - after  # Positive descent-space displacement.
            details.update(displacement_norm=float(displacement.norm()),
                task_first_order_loss_change=(-matrix @ displacement).tolist(),
                normalized_task_progress_actual_displacement=((matrix @ displacement) / (matrix.norm(dim=1) * displacement.norm())).tolist())
            model.load_state_dict(frozen)
            assert run.existing.digest(model) == initial_sha
            row[label] = details
            directions[label], steps[label] = joint, displacement
            arrays[label + '_joint'], arrays[label + '_descent_displacement'] = joint.cpu().numpy(), displacement.cpu().numpy()
            del optimizer
        row['weighted_direction_cosine'] = cosine(directions['old'], directions['new'])
        row['adamw_displacement_cosine'] = cosine(steps['old'], steps['new'])
        row['weighted_direction_angle_degrees'] = float(np.degrees(np.arccos(np.clip(row['weighted_direction_cosine'], -1, 1))))
        row['adamw_displacement_angle_degrees'] = float(np.degrees(np.arccos(np.clip(row['adamw_displacement_cosine'], -1, 1))))
        path = OUT / f"direction_audit_{entry['cursor']}.npz"
        np.savez(path, **arrays)
        row['vectors_sha256'] = run.file_sha256(path)
        row['restoration_max_abs_difference'] = 0.0
        audit.append(row)
        run.write(OUT / 'direction_audit.json', audit)
        print('DIRECTION_AUDIT', len(audit), row['weighted_direction_angle_degrees'], row['adamw_displacement_angle_degrees'], flush=True)
        del matrix, raw, directions, steps
    old_share = np.median([x['old']['norm_shares'][0] for x in audit])
    new_share = np.median([x['new']['norm_shares'][0] for x in audit])
    weighted_angle = float(np.median([x['weighted_direction_angle_degrees'] for x in audit]))
    adamw_angle = float(np.median([x['adamw_displacement_angle_degrees'] for x in audit]))
    passed = new_share / old_share >= 2 and (weighted_angle >= 5 or adamw_angle >= 5)
    gate = {'passed': bool(passed), 'relchem_median_norm_share_gain': float(new_share / old_share),
            'median_weighted_direction_angle_degrees': weighted_angle, 'median_adamw_displacement_angle_degrees': adamw_angle,
            'old_median_norm_shares': np.median([x['old']['norm_shares'] for x in audit], axis=0).tolist(),
            'new_median_norm_shares': np.median([x['new']['norm_shares'] for x in audit], axis=0).tolist(),
            'no_pareto_gate': True, 'initial_model_restored_exactly': True}
    run.write(OUT / 'calibration_direction_gate.json', gate)
    if passed:
        shutil.copyfile(OLD / 'sampling_manifest.json', OUT / 'sampling_manifest.json')
        calibration['manifest_sha256'] = run.file_sha256(OUT / 'sampling_manifest.json')
        calibration['calibration_manifest_sha256'] = run.file_sha256(OUT / 'calibration_manifest.json')
        run.write(OUT / 'calibration.json', calibration)
        protocol = {k: old_protocol[k] for k in ('initial_state', 'dataset_sha256', 'adamw', 'exc_chunk_size', 'operator_chunk_size')}
        protocol.update(total_updates=90, initial_lr=1e-6, final_lr=1e-7, scheduler='CosineAnnealingLR T_max=90',
                        only_change='Population-matched fixed task coefficients', source_commit='855686bbc52ae0c861935104a345dd703d7e6b79',
                        parent_executable_protocol_sha256=old_protocol['executable_protocol_sha256'],
                        calibration_direction_gate_sha256=run.file_sha256(OUT / 'calibration_direction_gate.json'))
        run.write(OUT / 'protocol.json', protocol)
    bundle.close()
    print('POPULATION_CALIBRATION_GATE', gate, flush=True)


if __name__ == '__main__':
    main()
