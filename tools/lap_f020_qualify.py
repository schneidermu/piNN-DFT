"""F020 frozen-cycle algebra on authenticated stream Jacobian rows."""
import json
import time

import lap_f020_stream as stream
import numpy as np
import torch


def execute():
    if (stream.OUT / 'scientific_execution_closed.json').exists():
        raise SystemExit('F020 scientific execution closed; report regeneration only')
    audit = stream.audit
    b8 = audit.geometry.b8
    protocol = audit.read(stream.ROOT / 'lap_f020_protocol.json')
    state, scales, _names = audit.load(audit.geometry.STATES[0])
    batches = audit.geometry.f010._batches()
    population = {r['identity']: r for batch in batches for r in batch['relchem']}
    specimens = [population[row['identity']] for row in state['rows']]
    if len(list(stream.OUT.glob('row_*.npz'))) < 212:
        print('INCOMPLETE: missing reaction rows')
        return
    began = time.perf_counter()
    prior = audit.read(stream.ROOT / 'lap_f020_decision.json')
    bundle = audit.TrainingBundle(stream.ROOT.parent / 'publication_dataset_v1')
    model, shadow, _run = audit.existing._load_eval_model(b8.verify_sources(), audit.geometry.STATES[0]['checkpoint'])
    shadow.load_state_dict(model.state_dict(), strict=True)
    dispersion = bundle.chemistry_dispersions()
    reused_parity = []
    for s, e in zip(specimens, state['rows'], strict=True):
        if s['variant'] != e['variant']:
            continue
        assert prior['elapsed_seconds'] + time.perf_counter() - began < stream.BUDGET - 30
        reaction = b8._load_reaction(bundle, 'relchem', s)
        with torch.no_grad():
            prediction, reference = audit.chemistry._prediction(shadow, reaction, dispersion)
        gap = abs(float(prediction - reference) - e['residual'])
        assert gap <= 1e-8 and float(reference) == e['reference']
        reused_parity.append({'identity': s['identity'], 'variant': s['variant'],
                              'scalar_gap': gap, 'reference_exact': True,
                              'source_hash': stream.source_hash(bundle.reactions[s['identity']], s['variant'], bundle)})
        del reaction, prediction, reference
    audit.write(stream.OUT / 'reused_row_parity.json', reused_parity)
    vectors, residuals, mapping = [], [], []
    for index, (s, e) in enumerate(zip(specimens, state['rows'], strict=True)):
        if s['variant'] == e['variant']:
            vector, residual = state['jacobian'][index], e['residual']
            origin = 'authenticated evaluation cache'
        else:
            path = stream.OUT / f'row_{index:03d}.npz'
            if not path.exists():
                print('INCOMPLETE', index)
                return
            receipt = audit.read(path.with_suffix('.json'))
            assert audit.sha(path) == receipt['sha256']
            assert receipt['binding']['identity'] == s
            with np.load(path) as payload:
                vector, residual = payload['gradient'].copy(), float(payload['residual'])
                assert float(payload['reference']) == e['reference']
            origin = receipt['sha256']
        vectors.append(vector)
        residuals.append(residual)
        mapping.append({'row': index, **s, 'origin': origin,
                        'source_hash': stream.source_hash(bundle.reactions[s['identity']], s['variant'], bundle)})
    matrix = np.array(vectors, dtype=np.float64)
    assert matrix.shape == (251, 9446) and np.isfinite(matrix).all()
    np.savez_compressed(stream.OUT / 'stream_jacobian.npz', gradients=matrix, residual=residuals)
    audit.write(stream.OUT / 'stream_rows.json', mapping)
    db = [s['database'] for s in specimens]
    w = stream.f019.weights(db)
    coefficients = audit.geometry.arm.jw_coefficients(residuals, scales, db)
    singleton = coefficients[:, None] * matrix * 251
    gj = singleton.mean(axis=0)
    gd = (w[:, None] * singleton).mean(axis=0)
    lj = np.sqrt(np.square(residuals) + 1e-20) / np.array([scales[d] for d in db])
    lookup = {s['identity']: i for i, s in enumerate(specimens)}
    indices = [[lookup[r['identity']] for r in batch['relchem']] for batch in batches]
    old = np.array([singleton[ix].mean(axis=0) for ix in indices])
    new = np.array([(singleton[ix] * w[ix, None]).mean(axis=0) for ix in indices])
    pieces = np.array([singleton[np.array(db) == d].mean(axis=0) / 8 for d in sorted(stream.f019.COUNTS)])
    parity = {'JW_gradient_l2': float(np.linalg.norm(old.mean(axis=0) - gj)),
              'DB_gradient_l2': float(np.linalg.norm(new.mean(axis=0) - gd)),
              'DB_database_reconstruction_l2': float(np.linalg.norm(pieces.sum(axis=0) - gd)),
              'JW_scalar_gap': abs(float(np.mean([lj[ix].mean() for ix in indices]) - lj.mean())),
              'DB_scalar_gap': abs(float(np.mean([(w[ix] * lj[ix]).mean() for ix in indices]) - (w * lj).mean()))}
    assert all(v <= 1e-10 for v in parity.values())
    stream.write('lap_f020_cycle_gradient_parity.json', {'status': 'PASS', **parity})
    c = audit.geometry.C010 * float(np.linalg.norm(gj) / np.linalg.norm(gd))
    np.savez_compressed(stream.OUT / 'full_gradients.npz', jw_S=gj, db_S=gd,
                        singleton=singleton, old_batches=old, new_batches=new)
    assert np.isfinite(c) and c > 0
    coeff_e = audit.geometry.arm.jw_coefficients(state['residuals'], scales, db)
    singleton_e = coeff_e[:, None] * state['jacobian'] * 251
    loss_e = np.sqrt(np.square(state['residuals']) + 1e-20) / np.array([scales[d] for d in db])
    stream.write('lap_f020_stream_calibration.json', {'c019_S': c, 'c019_E': 27.576519733036633,
                                                    'JW_norm_S': float(np.linalg.norm(gj)),
                                                    'DB_norm_S': float(np.linalg.norm(gd)),
                                                    'cosine_JW_DB_S': stream.f019.cosine(gj, gd),
                                                    'JW_S': float(lj.mean()), 'DB_S': float((w * lj).mean()),
                                                    'JW_E': float(loss_e.mean()), 'DB_E': float((w * loss_e).mean()),
                                                    'JW_cosine_S_E': stream.f019.cosine(gj, singleton_e.mean(axis=0)),
                                                    'DB_cosine_S_E': stream.f019.cosine(gd, (w[:, None] * singleton_e).mean(axis=0))})
    comparison = []
    for d in sorted(stream.f019.COUNTS):
        mask = np.array(db) == d
        comparison.append({'database': d, 'loss_S': float(lj[mask].mean()), 'loss_E': float(loss_e[mask].mean()),
                           'gradient_cosine_S_E': stream.f019.cosine(singleton[mask].mean(axis=0), singleton_e[mask].mean(axis=0))})
    audit.table('lap_f020_population_comparison.csv', comparison)
    records = []
    physical = sum(audit.geometry.b8.LAMBDAS[t] * state['components'][t] for t in ('ae17', 'exc', 'op'))
    composite_old = physical + audit.geometry.b8.LAMBDAS['relchem'] * audit.geometry.C010 * gj
    composite_new = physical + audit.geometry.b8.LAMBDAS['relchem'] * c * gd
    for position in protocol['positions']:
        records.append({'position': position, 'cosine_F010': stream.f019.cosine(old[position], gj),
                        'cosine_F019': stream.f019.cosine(new[position], gd),
                        'norm_ratio_F010': float(np.linalg.norm(old[position]) / np.linalg.norm(gj)),
                        'norm_ratio_F019': float(np.linalg.norm(new[position]) / np.linalg.norm(gd)),
                        'relative_error_F010': float(np.linalg.norm(old[position] - gj) / np.linalg.norm(gj)),
                        'relative_error_F019': float(np.linalg.norm(new[position] - gd) / np.linalg.norm(gd)),
                        'composite_cosine_F010_frozen_physical': stream.f019.cosine(physical + audit.geometry.b8.LAMBDAS['relchem'] * audit.geometry.C010 * old[position], composite_old),
                        'composite_cosine_F019_frozen_physical': stream.f019.cosine(physical + audit.geometry.b8.LAMBDAS['relchem'] * c * new[position], composite_new),
                        'maximum_weight': float(w[indices[position]].max())})
    audit.table('lap_f020_stochastic_fidelity.csv', records)
    positive = sum(r['cosine_F019'] > 0 for r in records)
    p10_old = float(np.quantile([r['cosine_F010'] for r in records], .1))
    p10_new = float(np.quantile([r['cosine_F019'] for r in records], .1))
    passes = positive >= 24 and p10_new >= p10_old - .10
    decision = audit.read(stream.ROOT / 'lap_f020_decision.json')
    decision.update({'classification': 'B' if not passes else 'C', 'new_rows': 212, 'reused_rows': 39,
                     'c019_S': c, 'positive_dot': positive, 'p10_F010': p10_old, 'p10_F019': p10_new,
                     'stochastic_fidelity': 'PASS' if passes else 'FAIL', 'finite_step': 'NOT RUN',
                     'reason': 'Stochastic fidelity failed; dependent finite-step stage forbidden' if not passes
                               else 'Finite-step stage remains to be performed',
                     'elapsed_seconds': prior['elapsed_seconds'] + time.perf_counter() - began})
    stream.write('lap_f020_decision.json', decision)
    if not passes:
        audit.write(stream.OUT / 'scientific_execution_closed.json',
                    {'classification': 'B', 'stochastic_fidelity': 'FAIL', 'optimizer_updates': 0})
    stream.write('lap_f020_jacobian_reconstruction.json', {'status': 'PASS', 'shape': [251, 9446],
                 'reused': 39, 'new': 212, 'sha256': audit.sha(stream.OUT / 'stream_jacobian.npz'),
                 'mapping_sha256': audit.sha(stream.OUT / 'stream_rows.json')})
    print(json.dumps({k: v for k, v in decision.items() if k != 'profile'}))
    bundle.close()


if __name__ == '__main__':
    execute()
