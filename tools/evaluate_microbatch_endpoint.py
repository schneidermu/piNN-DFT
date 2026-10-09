"""Resumable no-parameter-backward endpoint evaluation of frozen objectives."""
import argparse
import copy
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_lap_microbatch as run
from train_models.lap_chemistry_sampling import POLICY, validate_evaluation
from train_models.lap_vxc import LapEnergy, sigma_from_gradients


def validation(model, bundle):
    double = copy.deepcopy(model).double().eval()
    energy = LapEnergy(double)
    energies = {}
    with torch.no_grad():
        for key, row in bundle.validation_species.items():
            group = bundle.handles.open(row['shard'])[row['group']]
            features, weights = group['features'][...], group['weights'][...]
            pieces = []
            for start in range(0, len(weights), 4096):
                f = torch.from_numpy(features[start:start + 4096]).to('cuda', torch.float64)
                w = torch.from_numpy(weights[start:start + 4096]).to('cuda', torch.float64)
                value = energy(f[:, :2], sigma_from_gradients(f[:, 2:8].reshape(-1, 2, 3)), f[:, 8:])
                pieces.append(float((value * w).sum().cpu()))
            energies[key] = float(group['nonxc'][()]) + sum(pieces) + row['primary_dispersion_hartree']
    assert len(energies) == 84 and np.isfinite(list(energies.values())).all()
    clean = set(bundle.splits['diet30_clean_validation']['ids'])
    rows = []
    for key, row in bundle.validation_reactions.items():
        predicted = sum(c['coefficient'] * energies[c['species_id']] for c in row['components']) * 627.5095
        error = predicted - row['reference_energy_kcal_mol']
        rows.append({'reaction_id': row['source_id'], 'signed_error_kcal_mol': error,
                     'weighted_absolute_error': abs(error) * row['diet_weight'], 'clean': key in clean})
    return {'clean28': float(np.mean([r['weighted_absolute_error'] for r in rows if r['clean']])),
            'full30': float(np.mean([r['weighted_absolute_error'] for r in rows])),
            'full30_selection_allowed': False, 'dispersion': 'PBE0-D3(BJ)', 'reaction_rows': rows}


def evaluate(folder, cursor, stage, seconds):
    checkpoint = folder / 'ordinary_sgd_adamw' / f'checkpoint_{cursor}.pt'
    protocol = run.read(folder / 'protocol.json')
    evaluation = run.read(folder / 'evaluation_manifest.json') if stage == 'chemistry' else None
    if evaluation is not None and evaluation['policy'] != POLICY:
        raise ValueError('Obsolete exhaustive chemistry evaluation protocol')
    model, shadow = run.model_at(protocol['initial_state'])
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'])
    before = run.existing.digest(model)
    bundle = run.PublicationDataset(run.DATA)
    assert bundle.manifest['logical_sha256'] == run.DATA_SHA
    suffix = '_one_variant' if stage == 'chemistry' else ''
    path = folder / f'endpoint_{cursor}_{stage}{suffix}.json'
    result = run.read(path) if path.exists() else {'checkpoint_sha256': run.file_sha256(checkpoint), 'rows': {}}
    assert result['checkpoint_sha256'] == run.file_sha256(checkpoint)
    began = time.perf_counter()
    try:
        if stage == 'validation':
            if 'metrics' not in result:
                result['metrics'] = validation(model, bundle)
        elif stage == 'mrks':
            for identity in sorted(bundle.systems):
                if identity in result['rows']:
                    continue
                if time.perf_counter() - began > seconds:
                    break
                system = bundle.mrks().operator_system(identity, device='cuda', dtype=torch.float32, chunk_size=4096)
                exc, op = run.existing.core.make_mrks_objective_factories(
                    model, system, point_chunk_size=256, exc_chunk_size=protocol['exc_chunk_size'],
                    dispersions=run.read(run.DATA / 'mrks/dispersion.json'))
                # Operator requires local density derivatives, but no parameter backward.
                values = {'exc': float(exc().detach()), 'op': float(op().detach())}
                assert all(math.isfinite(v) for v in values.values())
                result['rows'][identity] = values
                run.write(path, result)
                print('ENDPOINT', cursor, stage, len(result['rows']), flush=True)
                del system, exc, op
            result['complete'] = len(result['rows']) == 90
            if result['complete']:
                result['objectives'] = {t: float(np.mean([r[t] for r in result['rows'].values()])) for t in ('exc', 'op')}
        elif stage == 'chemistry':
            dispersion = bundle.chemistry_dispersions()
            validate_evaluation(evaluation['rows'], bundle.reactions)
            manifest_sha = run.file_sha256(folder / 'evaluation_manifest.json')
            if result.get('evaluation_manifest_sha256', manifest_sha) != manifest_sha:
                raise ValueError('Evaluation variant manifest changed')
            result['evaluation_manifest_sha256'] = manifest_sha
            for selected in evaluation['rows']:
                identity, variant, task = (selected[k] for k in ('identity', 'variant', 'task'))
                if identity in result['rows']:
                    continue
                if time.perf_counter() - began > seconds:
                    break
                reaction = bundle.chemistry('train_' + task).load_variant(identity, variant)
                reaction = run.existing.lap_training.tensor_record(reaction, 'cuda', torch.float64)
                if len(reaction['Grid']) > 131072:
                    reaction['model_point_chunk_size'] = 16384
                with torch.no_grad():
                    value = float(run.chemistry(model, shadow, reaction, dispersion)())
                assert math.isfinite(value)
                result['rows'][identity] = {'task': task, 'variant': variant,
                                            'database': bundle.reactions[identity]['database'], 'loss': value}
                run.write(path, result)
                del reaction
            result['complete'] = len(result['rows']) == 268
            if result['complete']:
                result['objectives'] = {t: float(np.mean([r['loss'] for r in result['rows'].values() if r['task'] == t]))
                                        for t in ('relchem', 'ae17')}
            result['definition'] = 'Fixed one-variant-per-identity mean of qualified singleton losses'
        assert run.existing.digest(model) == before
        result['elapsed_seconds_this_invocation'] = time.perf_counter() - began
        result['model_unchanged'] = True
        run.write(path, result)
    finally:
        bundle.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--cursor', type=int, required=True)
    parser.add_argument('--stage', choices=('validation', 'mrks', 'chemistry'), required=True)
    parser.add_argument('--seconds', type=int, default=1800)
    args = parser.parse_args()
    if not 0 < args.seconds <= 3600:
        raise ValueError('Standalone evaluation cap must be at most 60 minutes')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    evaluate(args.folder, args.cursor, args.stage, args.seconds)
