"""Write the F013 tables and plots from the frozen audit summary."""
import csv
import hashlib
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import lap_b8_adamw as b8
import lap_f013_geometry_audit as audit
import lap_training_metric_gradient_audit as j251

SUMMARY = audit.OUT / 'summary.json'
PROBE = audit.OUT / 'probes'
TASKS = ('jw', 'ae17', 'exc', 'op')
PUBLISHED = {
    'b8_t128': {'jw': 0.1564146381387076, 'ae17': 3.27546495584018, 'exc': 14.679044990002591, 'op': 0.030933124717599843},
    'f010_t64': {'jw': 0.14681873308372703, 'ae17': 2.3782119154678547, 'exc': 4.704043037478069, 'op': 0.029475733365122544},
    'f010_t128': {'jw': 0.14236986905644153, 'ae17': 2.2512007100644498, 'exc': 4.879897803270238, 'op': 0.029448852920882538},
    'f010_t251': {'jw': 0.13582209542516546, 'ae17': 1.4194527798078582, 'exc': 6.065543632527309, 'op': 0.029510273039158826},
    'f012_t64': {'jw': 0.1467451222187472, 'ae17': 2.078141107940078, 'exc': 4.986269839180046, 'op': 0.029360617791770065},
    'f012_t128': {'jw': 0.14249476779865852, 'ae17': 5.407950311254037, 'exc': 15.14046524175436, 'op': 0.028969947544150682},
    'f012_t251': {'jw': 0.13638789330754067, 'ae17': 1.208499592777915, 'exc': 7.839060830215015, 'op': 0.029382694695032924},
}


def _write_csv(path, rows):
    with path.open('w', newline='\n', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def render():
    b8._import_run()
    summary = b8.read_json(SUMMARY)
    scales = b8.read_json(b8.REPO / 'lap_f010_normalization_receipt.json')['scales']
    margins = []
    receipts = []
    cosine_rows = []
    for spec in audit.STATES:
        state = audit.load_state(spec, scales)
        units = [audit.unit(state['components'][task])[0] for task in TASKS]
        solved = audit.maxmin(units)
        relchem = sum(
            j251._factor(database) * ((1e-20 + residual * residual) ** 0.5)
            for residual, database in zip(state['residuals'], state['databases'], strict=True)
        ) / 251.0
        identity = state['f012'] - state['f010'] - b8.LAMBDAS['op'] * state['components']['op']
        physical = {}
        if spec['physical'].suffix == '.npz':
            physical = b8.read_json(spec['physical'].with_name('physical.json'))
        elif spec['name'] == 'b8_t128':
            physical = PUBLISHED['b8_t128']
        receipts.append({
            'checkpoint': spec['name'],
            'checkpoint_sha256': spec['sha256'],
            'reactions_sha256': _sha(spec['reactions']),
            'physical_sha256': _sha(spec['physical']) if spec['physical'].suffix == '.npz' else _sha(spec['physical'] / 'ae17.npy.npz'),
            'jw_scalar': state['jw_scalar'],
            'jw_published': PUBLISHED[spec['name']]['jw'],
            'jw_abs_diff': abs(state['jw_scalar'] - PUBLISHED[spec['name']]['jw']),
            'relchem_scalar': relchem,
            'ae17_published': physical.get('ae17'),
            'exc_published': physical.get('exc'),
            'op_published': physical.get('op'),
            'f012_identity_l2': float(np.linalg.norm(identity)),
            'gamma': solved['gamma'],
            'composite_margin_f010': min(float(np.dot(unit, state['f010'] / np.linalg.norm(state['f010']))) for unit in units),
            'composite_margin_f012': min(float(np.dot(unit, state['f012'] / np.linalg.norm(state['f012']))) for unit in units),
            'unit_eigenvalues': np.linalg.eigvalsh(np.stack(units) @ np.stack(units).T).clip(min=0).tolist(),
        })
        matrix = np.eye(4)
        for left, task_left in enumerate(TASKS):
            for right, task_right in enumerate(TASKS):
                matrix[left, right] = float(np.dot(units[left], units[right]))
                cosine_rows.append({
                    'checkpoint': spec['name'], 'left': task_left, 'right': task_right,
                    'cosine': matrix[left, right],
                    'raw_norm_left': float(np.linalg.norm(state['components'][task_left])),
                })
        margins.append((spec['name'], solved['gamma'], matrix))
    _write_csv(ROOT / 'lap_f013_primary_cosines.csv', cosine_rows)
    _write_csv(ROOT / 'lap_f013_database_cosines.csv', summary['databases'])
    feasibility = []
    for name, certificate in summary['certificates'].items():
        receipt = next(item for item in receipts if item['checkpoint'] == name)
        for label in ('four', 'four_plus_problem_db', 'databases_plus_physical'):
            item = certificate[label]
            feasibility.append({
                'checkpoint': name, 'system': label, 'gamma': item['gamma'],
                'active': ' '.join(item['active']),
                'composite_margin_f010': receipt['composite_margin_f010'],
                'composite_margin_f012': receipt['composite_margin_f012'],
            })
    _write_csv(ROOT / 'lap_f013_feasibility_results.csv', feasibility)
    _write_csv(ROOT / 'lap_f013_parameter_block_results.csv', summary['blocks'])
    _write_csv(ROOT / 'lap_f013_adamw_counterfactuals.csv', summary['adamw'])
    batches = [b8.read_json(path) for path in sorted(PROBE.glob('minibatch_*.json'))]
    batch_rows = []
    for row in batches:
        batch_rows.append({
            'checkpoint': row['checkpoint'], 'cursor': row['cursor'], 'mrks_id': row['mrks_id'],
            'jw_cosine': row['jw_cosine'], 'jw_norm_ratio': row['jw_norm_ratio'],
            'f010_cosine': row['f010_cosine'], 'f012_cosine': row['f012_cosine'],
            **{key: row[key] for key in row if key.endswith(('_dot_jw', '_dot_ae17', '_dot_exc', '_dot_op'))},
        })
    _write_csv(ROOT / 'lap_f013_minibatch_full_gradient_comparison.csv', batch_rows)
    finite_rows = []
    for path in sorted(PROBE.glob('finite_*.json')):
        row = b8.read_json(path)
        for task in TASKS:
            observed = row['observed'][task] - row['baseline'][task]
            predicted = row['predicted'][task]
            finite_rows.append({
                'checkpoint': row['checkpoint'], 'direction': row['direction'], 'scale': row['scale'],
                'task': task, 'predicted': predicted, 'observed': observed,
                'ratio': observed / predicted if abs(predicted) > 1e-15 else None,
                'restored': row['restored'], 'spin_max_abs': row['constraints']['spin_max_abs'],
                'passthrough_max_abs': row['constraints']['passthrough_max_abs'],
            })
    _write_csv(ROOT / 'lap_f013_finite_step_validation.csv', finite_rows)
    b8.write_json(ROOT / 'lap_f013_feasibility_certificates.json', summary['certificates'])
    b8.write_json(ROOT / 'lap_f013_gradient_receipts.json', {
        'optimizer_updates': 0,
        'parameter_dimension': 9446,
        'a_d_sha256': '3233a7b3194e68af447bb63edda0d154325da7d051e0df6fe60a7a788cebc73d',
        'manifest_sha256': 'b249964d410f0db4a59867de53bb35c6442ec23aa81eefb0b96b50242d94f6bc',
        'states': receipts,
    })
    _plots(margins, summary, batches, finite_rows)
    print('F013_RENDER_DONE', flush=True)


def _plots(margins, summary, batches, finite_rows):
    figure, axes = plt.subplots(2, 4, figsize=(14, 7))
    for axis, (name, _gamma, matrix) in zip(axes.ravel(), margins, strict=False):
        image = axis.imshow(matrix, vmin=-1, vmax=1, cmap='coolwarm')
        axis.set_title(name, fontsize=9)
        axis.set_xticks(range(4), TASKS, fontsize=7)
        axis.set_yticks(range(4), TASKS, fontsize=7)
    axes.ravel()[-1].axis('off')
    figure.colorbar(image, ax=axes.ravel().tolist(), fraction=0.02)
    figure.savefig(ROOT / 'lap_f013_cosine_matrices.png', dpi=140, bbox_inches='tight')
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4))
    gammas = [item['gamma'] for item in summary['rows']]
    axis.bar(range(len(gammas)), gammas, color='#3d5a80')
    axis.set_xticks(range(len(gammas)), [item['checkpoint'] for item in summary['rows']], rotation=30, ha='right')
    axis.set_ylabel('normalized max-min margin')
    axis.axhline(0.01, color='#9b2226', lw=0.8)
    figure.savefig(ROOT / 'lap_f013_margins.png', dpi=140, bbox_inches='tight')
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4))
    systems = ('four', 'four_plus_problem_db', 'databases_plus_physical')
    names = list(summary['certificates'])
    width = 0.25
    for index, system in enumerate(systems):
        values = [summary['certificates'][name][system]['gamma'] for name in names]
        axis.bar(np.arange(len(names)) + (index - 1) * width, values, width=width, label=system)
    axis.set_xticks(range(len(names)), names, rotation=30, ha='right')
    axis.legend(fontsize=7)
    axis.set_ylabel('normalized margin')
    figure.savefig(ROOT / 'lap_f013_database_feasibility.png', dpi=140, bbox_inches='tight')
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4))
    subspaces = ('full', 'exchange', 'correlation', 'normalization')
    for index, subspace in enumerate(subspaces):
        values = [row['gamma'] or 0.0 for row in summary['blocks'] if row['subspace'] == subspace]
        axis.plot(range(len(values)), values, marker='o', label=subspace)
    axis.set_xticks(range(len(names)), names, rotation=30, ha='right')
    axis.legend(fontsize=7)
    axis.set_ylabel('subspace margin')
    figure.savefig(ROOT / 'lap_f013_block_feasibility.png', dpi=140, bbox_inches='tight')
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4))
    chosen = [row for row in summary['adamw'] if row['policy'] == ('f012' if row['checkpoint'].startswith('f012') else 'f010')]
    axis.bar(range(len(chosen)), [row['margin_exact'] for row in chosen], color='#9b2226', label='AdamW cone margin')
    axis.plot(range(len(chosen)), [next(item['gamma'] for item in summary['rows'] if item['checkpoint'] == row['checkpoint']) for row in chosen], 'o-', color='#1b4332', label='common-descent margin')
    axis.axhline(0, color='black', lw=0.6)
    axis.set_xticks(range(len(chosen)), [row['checkpoint'] for row in chosen], rotation=30, ha='right')
    axis.legend(fontsize=7)
    figure.savefig(ROOT / 'lap_f013_full_vs_adamw.png', dpi=140, bbox_inches='tight')
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.scatter([row['cursor'] for row in batches], [row['f012_cosine'] for row in batches], c='#3d5a80', label='composite')
    axis.scatter([row['cursor'] for row in batches], [row['jw_cosine'] for row in batches], c='#bb3e03', marker='x', label='JW only')
    axis.axhline(0, color='black', lw=0.6)
    axis.set_ylabel('cosine with full gradient')
    axis.set_xlabel('predeclared minibatch cursor')
    axis.legend(fontsize=7)
    figure.savefig(ROOT / 'lap_f013_minibatch_angles.png', dpi=140, bbox_inches='tight')
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 6))
    predicted = [row['predicted'] for row in finite_rows]
    observed = [row['observed'] for row in finite_rows]
    axis.scatter(predicted, observed, s=18, c='#3d5a80')
    limit = max(abs(value) for value in predicted + observed)
    axis.plot([-limit, limit], [-limit, limit], color='#9b2226', lw=0.8)
    axis.set_xlabel('first-order prediction')
    axis.set_ylabel('observed change')
    figure.savefig(ROOT / 'lap_f013_finite_step.png', dpi=140, bbox_inches='tight')
    plt.close(figure)


if __name__ == '__main__':
    render()
