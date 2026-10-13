"""F015: frozen training-only chemistry and constrained local linear algebra.

No optimizer construction/step, no validation evaluator, no checkpoint writes.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'train_models'), str(ROOT)]
import lap_f011_normalized_chemistry as existing  # noqa: F401 -- qualified helper re-export
import lap_f013_geometry_audit as geometry
import lap_training_metric_gradient_audit as chemistry  # noqa: F401 -- qualified helper re-export
import numpy as np
import torch
from scipy.linalg import qr, svd
from scipy.optimize import minimize, nnls

from train_models.publication_data import PublicationDataset
from train_models.publication_data.loader import Handles

OUT = ROOT.parent / 'lap_f015_audit_20261013'
RADIUS = 0.0006498670299681904
S5 = ROOT.parent / 'replay_trial_19_simple4_s5_two_step_40_10_500_snapshots_from280_gc_svelu_mirror/checkpoints/trial_19_selected.pt'
S5_SHA = '946d93f12656415288d4cb98b92183a203ad78fa13b57c8acaeef7ac5d1776de'
STATES = [s for s in geometry.STATES if s['name'] in ('b8_t128', 'f010_t128', 'f010_t251', 'f012_t128', 'f012_t251')]
LOCAL = ('b8_t128', 'f010_t128', 'f010_t251', 'f012_t128')


class TrainingBundle(PublicationDataset):
    """F015-only metadata boundary; reuse canonical training access methods."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        protocol = verify()
        assert sha(self.root / 'dataset_manifest.json') == protocol['pins'][str(self.root / 'dataset_manifest.json')]
        self.manifest = read(self.root / 'dataset_manifest.json')
        self.checksums = {r['file']: r['sha256'] for r in self.manifest['files']}
        self.reactions = self._index('chemistry/reactions.jsonl')
        self.species = self._index('chemistry/species.jsonl')
        self.systems = self._index('mrks/systems.jsonl')
        assert len(self.reactions) == 268 and len(self.systems) == 90
        self.splits = {'train_relchem': {'ids': sorted(k for k, v in self.reactions.items() if v['task'] == 'relchem')},
                       'train_ae17': {'ids': sorted(k for k, v in self.reactions.items() if v['task'] == 'ae17')},
                       'train_mrks': {'ids': sorted(self.systems)}}
        assert len(self.splits['train_relchem']['ids']) == 251 and len(self.splits['train_ae17']['ids']) == 17
        self.handles = Handles(self.root, self.checksums)

    def validation(self, *_args, **_kwargs):
        raise PermissionError('F015 is training-only; no validation metadata/evaluator')


def sha(path):
    return geometry.b8.sha256_file(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def table(name, rows):
    assert rows
    with (ROOT / name).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def freeze():
    OUT.mkdir(exist_ok=True)
    if (ROOT / 'lap_f015_protocol.json').exists():
        verify()
        print('F015_FROZEN existing immutable protocol verified', flush=True)
        return
    assert sha(S5) == S5_SHA
    receipts = read(ROOT / 'lap_f013_gradient_receipts.json')
    pins = {str(S5): S5_SHA}
    for spec in STATES:
        receipt = next(r for r in receipts['states'] if r['checkpoint'] == spec['name'])
        assert sha(spec['checkpoint']) == spec['sha256'] == receipt['checkpoint_sha256']
        assert sha(spec['reactions']) == receipt['reactions_sha256']
        pins[str(spec['checkpoint'])] = spec['sha256']
        pins[str(spec['reactions'])] = receipt['reactions_sha256']
        pins[str(spec['reactions'].with_suffix('.json'))] = sha(spec['reactions'].with_suffix('.json'))
        if spec['physical'].suffix == '.npz':
            assert sha(spec['physical']) == receipt['physical_sha256']
            pins[str(spec['physical'])] = receipt['physical_sha256']
            pins[str(spec['physical'].with_suffix('.json'))] = sha(spec['physical'].with_suffix('.json'))
        else:
            for task in ('ae17', 'exc', 'op'):
                path = spec['physical'] / f'{task}.npy.npz'
                pins[str(path)] = sha(path)
    old = ROOT.parent / 'lap_s5_ae17_runs_20261004'
    old_protocol = read(old / 'protocol.json')
    old_result = read(old / 'results.json')
    assert sha(old / 's5_rows.jsonl') == old_result['s5_rows_sha256']
    for name in ('protocol.json', 'results.json', 's5_rows.jsonl'):
        pins[str(old / name)] = sha(old / name)
    for path, value in old_protocol['pins'].items():
        assert sha(path) == value
        pins[path] = value
    for path in (ROOT / 'relchem_joint_epoch_evaluation_manifest.json', ROOT / 'lap_f010_normalization_receipt.json',
                 geometry.prior.OUT / 'b8_t128/result.json', ROOT.parent / 'publication_dataset_v1/dataset_manifest.json'):
        pins[str(path)] = sha(path)
    sources = {str(path): sha(path) for folder in ('train_models', 'dft_functionals') for path in (ROOT / folder).rglob('*.py')}
    # Bound all reused diagnostic helpers too; none of their execution entrypoints is invoked.
    sources.update({str(p): sha(p) for p in (ROOT / 'tools').glob('lap_f01*.py') if 'f015' not in p.name})
    payload = {'experiment': 'F015', 'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               'optimizer_updates': 0, 'training': False, 'radius': RADIUS, 'finite_scales': [.25, 1.0],
               'models': [s['name'] for s in STATES] + ['s5'], 'local_models': list(LOCAL), 'pins': pins, 'sources': sources,
               'chemistry': '251 identities, one fixed J251 evaluation variant each; JW = mean(abs(e)/A_d)',
               'quadratic_weights': 'W_rr=1/(251*A_database); diagnostic squared residual, NOT JW',
               'rank_relative_threshold': 1e-8, 'rank_sensitivity_thresholds': [1e-6, 1e-10],
               'solver': 'SLSQP on exact span(J rows, raw constraints), scaled unit ball; independent convex KKT/dual certificate',
               'tolerances': {'energy_kcal_abs': 1e-6, 'jvp_relative': 5e-3, 'jvp_absolute_kcal': 1e-6,
                              'kkt_relative': 1e-6, 'feasibility_unit': 1e-9, 'gap_relative': 1e-7,
                              'input_fd_relative': 1e-4, 'input_fd_absolute': 1e-7, 'anchor_absolute': 2e-7},
               'descriptor_thresholds': [.95, .99, .999], 'descriptor_stride': 101, 'descriptor_chunk': 8192,
               'density_regimes': [1e-6, 1e-3], 'positive_stat_weights': 'max(w,0); electrons=max(w,0)*max(rho_total,0)',
               'sensitivity': 'dimensionless learned outputs; q derivative scaled from physical Laplacian; tau derivative and tau*derivative',
               'conditional_information': 'deterministic nearest neighbours in bounded 8-dimensional Lap descriptors; stride101 physical samples; no trained model',
               'no_development_or_test_evaluation': True, 's5_operator_comparison': False}
    path = ROOT / 'lap_f015_protocol.json'
    if path.exists():
        assert read(path) == payload
    else:
        write(path, payload)
    verify()
    print('F015_FROZEN all checkpoints/caches available; zero updates', flush=True)


def verify():
    protocol = read(ROOT / 'lap_f015_protocol.json')
    for path, value in {**protocol['pins'], **protocol['sources']}.items():
        assert sha(path) == value, path
    return protocol


def load(spec):
    scales = read(ROOT / 'lap_f010_normalization_receipt.json')['scales']
    state = geometry.load_state(spec, scales)
    names = tuple(read(geometry.prior.OUT / 'b8_t128/result.json')['parameter_names'])
    saved = torch.load(spec['checkpoint'], map_location='cpu', weights_only=False)
    assert sum(saved['model'][n].numel() for n in names) == 9446
    # Qualified named_trainable_parameters sorts names; state_dict traversal does not.
    assert tuple(sorted(n for n in saved['model'] if n in names)) == names
    assert len({(r['identity'], r['variant']) for r in state['rows']}) == 251
    return state, scales, names


def cache_checks():
    verify()
    with (ROOT / 'lap_f013_database_cosines.csv').open() as stream:
        old_db = list(csv.DictReader(stream))
    receipts = read(ROOT / 'lap_f013_gradient_receipts.json')['states']
    output = []
    for spec in STATES:
        state, _scales, names = load(spec)
        ref = next(r for r in receipts if r['checkpoint'] == spec['name'])
        assert abs(state['jw_scalar'] - ref['jw_scalar']) < 1e-12
        for row in [r for r in old_db if r['checkpoint'] == spec['name']]:
            vector = state['pieces'][row['database']]
            norm = float(np.linalg.norm(vector))
            cosine = float(vector @ state['components']['jw'] / norm / np.linalg.norm(state['components']['jw']))
            assert abs(norm - float(row['norm'])) <= 1e-12 * max(1., norm)
            assert abs(cosine - float(row['cosine_jw'])) < 1e-12
        output.append({'checkpoint': spec['name'], 'jw_scalar_gap': abs(state['jw_scalar'] - ref['jw_scalar']),
                       'eight_database_gradient_norms_cosines_match': True, 'canonical_parameter_order_match': True,
                       'parameter_names': list(names), 'reaction_order': [(r['identity'], r['variant']) for r in state['rows']]})
    write(ROOT / 'lap_f015_cache_parity.json', output)


def stats(errors):
    e = np.asarray(errors, dtype=np.float64)
    return {'mae': float(np.abs(e).mean()), 'rmse': float(np.sqrt(np.square(e).mean())),
            'median_abs': float(np.median(np.abs(e))), 'p90_abs': float(np.quantile(np.abs(e), .9)),
            'p95_abs': float(np.quantile(np.abs(e), .95)), 'signed_mean': float(e.mean())}


def solve_probe(jacobian, errors, weights, constraints, radius=RADIUS):
    """Convex quadratic in the exact row span; x is delta/radius in that basis."""
    j, e, w = (np.asarray(a, dtype=np.float64) for a in (jacobian, errors, weights))
    g = np.asarray(constraints, dtype=np.float64)
    assert np.isfinite(j).all() and np.all(w > 0) and np.isfinite(g).all()
    units = g / np.linalg.norm(g, axis=1)[:, None]
    # QR without numerical truncation retains every row/constraint direction.
    basis = qr(np.concatenate([j, units], axis=0).T, mode='economic')[0]
    a = units @ basis
    b = radius * j @ basis
    h, f = b.T @ (w[:, None] * b), b.T @ (w * e)
    scale = max(float(np.linalg.norm(f)), 1e-30)
    h, f = h / scale, f / scale
    result = minimize(lambda x: float(f @ x + .5 * x @ h @ x), np.zeros(len(f)),
                      jac=lambda x: f + h @ x, method='SLSQP',
                      constraints=[{'type': 'ineq', 'fun': lambda x: -a @ x, 'jac': lambda x: -a},
                                   {'type': 'ineq', 'fun': lambda x: .5 * (1 - x @ x), 'jac': lambda x: -x}],
                      options={'maxiter': 2000, 'ftol': 1e-13})
    x = result.x
    active = np.where(np.abs(a @ x) < 1e-7)[0]
    ball = bool(abs(np.linalg.norm(x) - 1) < 1e-7)
    columns = [a[i] for i in active] + ([x] if ball else [])
    dual = nnls(np.stack(columns, axis=1), -(f + h @ x))[0] if columns else np.empty(0)
    lam = np.zeros(len(a))
    lam[active] = dual[:len(active)]
    mu = float(dual[-1]) if ball else 0.
    stationarity = float(np.linalg.norm(f + h @ x + a.T @ lam + mu * x))
    rhs = f + a.T @ lam
    inv = np.linalg.lstsq(h + mu * np.eye(len(x)), rhs, rcond=1e-14)[0]
    dual_value = -.5 * rhs @ inv - .5 * mu
    primal_value = float(f @ x + .5 * x @ h @ x)
    gap = float(primal_value - dual_value)
    feasibility = float(max(0., np.max(a @ x), np.linalg.norm(x) - 1))
    delta = radius * (basis @ x)
    certificate = {'solver_success': bool(result.success), 'iterations': int(result.nit), 'message': str(result.message),
                   'kkt_relative': stationarity, 'feasibility_unit': feasibility, 'dual_gap_scaled': gap,
                   'active': active.tolist(), 'ball_active': ball, 'multipliers': lam.tolist(), 'ball_multiplier': mu,
                   'step_norm': float(np.linalg.norm(delta)), 'initial_quadratic': float(.5 * np.dot(w * e, e)),
                   'predicted_quadratic': float(.5 * np.dot(w * (e + j @ delta), e + j @ delta)),
                   'span_reconstruction_relative': float(np.linalg.norm(j - (j @ basis) @ basis.T) / np.linalg.norm(j)),
                   'hessian_regularized_condition': float(np.linalg.cond(h + mu * np.eye(len(x))))}
    certificate['qualified'] = bool(feasibility <= 1e-9 and stationarity <= 1e-6 and abs(gap) <= 1e-7)
    return delta, certificate


def local():
    verify()
    projections, rows, protection = {}, [], []
    s5_rows = read(OUT / 'matched_s5_rows.json') if (OUT / 'matched_s5_rows.json').exists() else None
    for spec in STATES:
        if spec['name'] not in LOCAL:
            continue
        state, scales, _names = load(spec)
        j, e = state['jacobian'], np.array(state['residuals'])
        u, s, _vt = svd(j, full_matrices=False, lapack_driver='gesdd')
        retained = s >= s[0] * 1e-8
        project = u[:, retained] @ (u[:, retained].T @ (-e))
        record = {'singular_values': s.tolist(), 'rank': int(retained.sum()),
                  'rank_sensitivity': {str(t): int((s >= s[0] * t).sum()) for t in (1e-6, 1e-10)},
                  'condition_resolved': float(s[0] / s[retained][-1]),
                  'error_correction_projected_fraction': float(np.dot(project, project) / np.dot(e, e)),
                  'error_correction_min_norm_step': float(np.linalg.norm((u[:, retained].T @ (-e)) / s[retained])),
                  's5_projection': None}
        if s5_rows:
            lookup = {r['identity']: r for r in s5_rows}
            target = np.array([lookup[r['identity']]['residual'] for r in state['rows']]) - e
            projected = u[:, retained] @ (u[:, retained].T @ target)
            norm = float(np.linalg.norm((u[:, retained].T @ target) / s[retained]))
            record['s5_projection'] = {'squared_fraction': float(projected @ projected / (target @ target)),
                                       'orthogonal_residual_norm': float(np.linalg.norm(target - projected)),
                                       'minimum_norm_step': norm, 'step_over_radius': norm / RADIUS}
        projections[spec['name']] = record
        w = np.array([1 / (251 * scales[d]) for d in state['databases']])
        constraints = [state['components'][t] for t in ('ae17', 'exc', 'op')]
        names_g = ['ae17', 'exc', 'op']
        delta, cert = solve_probe(j, e, w, constraints)
        assert cert['qualified'], cert
        predicted = e + j @ delta
        worsens = [d for d in geometry.PROBLEM_DB if state['pieces'][d] @ delta > 0]
        for label in ('base', 'protected') if worsens else ('base',):
            if label == 'protected':
                constraints += [state['pieces'][d] for d in geometry.PROBLEM_DB]
                names_g += list(geometry.PROBLEM_DB)
                delta, cert = solve_probe(j, e, w, constraints)
                assert cert['qualified'], cert
                predicted = e + j @ delta
            np.save(OUT / f'{spec["name"]}_{label}_delta.npy', delta)
            write(OUT / f'{spec["name"]}_{label}_certificate.json', {**cert, 'constraint_names': names_g.copy()})
            for db in sorted(scales):
                mask = np.array(state['databases']) == db
                old, new = stats(e[mask]), stats(predicted[mask])
                rows.append({'checkpoint': spec['name'], 'direction': label, 'database': db,
                             'mae_before': old['mae'], 'mae_linear': new['mae'], 'rmse_before': old['rmse'],
                             'rmse_linear': new['rmse'], 'jw_change_linear': float(np.mean(np.abs(predicted[mask]) - np.abs(e[mask])) / scales[db]),
                             'db_first_order_dot': float(state['pieces'][db] @ delta), 'step_norm': cert['step_norm'],
                             **{f'{t}_predicted_change': float(state['components'][t] @ delta) for t in ('ae17', 'exc', 'op')}})
            protection.append({'checkpoint': spec['name'], 'direction': label, 'weighted_ls_before': cert['initial_quadratic'],
                               'weighted_ls_linear': cert['predicted_quadratic'], 'kkt_relative': cert['kkt_relative'],
                               'dual_gap': cert['dual_gap_scaled'], 'feasibility': cert['feasibility_unit'],
                               'active': ';'.join(names_g[i] for i in cert['active']), 'protected_trigger': ';'.join(worsens)})
        print('F015_LOCAL', spec['name'], 'rank', record['rank'], 'qualified', flush=True)
    write(ROOT / 'lap_f015_jacobian_rank_and_projection.json', projections)
    table('lap_f015_local_attainability.csv', rows)
    table('lap_f015_database_protection.csv', protection)
    verify()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'local', 'check'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    {'freeze': freeze, 'local': local, 'check': cache_checks}[args.stage]()
