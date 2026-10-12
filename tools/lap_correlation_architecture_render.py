"""Write the correlation-audit tables and plots from the frozen scratch cube."""
import csv
import hashlib

import matplotlib

matplotlib.use('Agg')
import lap_b8_adamw as b8
import lap_correlation_architecture_audit as audit
import matplotlib.pyplot as plt
import numpy as np

REPO = b8.REPO
NAMES = tuple(spec['name'] for spec in audit.CHECKS)


def _rows(name):
    return b8.read_json(audit.OUT / name / 'systems.json')['systems']


def _write(path, rows):
    path = REPO / path
    with path.open('w', encoding='utf-8', newline='\n') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _pooled(rows, key):
    electrons = sum(row['electrons'] for row in rows)
    return sum(row[key] * row['electrons'] for row in rows) / electrons


def tables():
    distribution = []
    positive = []
    decomposition = []
    counterfactual = []
    for name in NAMES:
        payload = b8.read_json(audit.OUT / name / 'systems.json')
        for row in payload['systems']:
            distribution.append({
                'checkpoint': name, 'system': row['system'], 'points': row['points'],
                'electrons': row['electrons'], 'nonpositive_weights': row['nonpositive_weights'],
                'gc_min': row['gc_min'], 'gc_max': row['gc_max'],
                'gc_mean_point': row['gc_mean_point'], 'gc_mean_electron': row['gc_mean_electron'],
                'fraction_gc_gt_1_point': row['fraction_gc_gt_1_point'],
                'fraction_gc_gt_1_electron': row['fraction_gc_gt_1_electron'],
                'fraction_gc_gt_1p01_electron': row['fraction_gc_gt_1p01_electron'],
                'fraction_gc_gt_1p1_electron': row['fraction_gc_gt_1p1_electron'],
                'fraction_gc_gt_1p5_electron': row['fraction_gc_gt_1p5_electron'],
                'fraction_gc_lt_0_electron': row['fraction_gc_lt_0_electron'],
                'fraction_gc_near_1_electron': row['fraction_gc_near_1_electron'],
                'nonfinite': row['nonfinite'],
            })
            ratio = 0.0 if row['Ec_abs'] == 0.0 else row['Ec_positive'] / row['Ec_abs']
            positive.append({
                'checkpoint': name, 'system': row['system'], 'positive_points': row['positive_points'],
                'positive_electron_fraction': row['positive_electron_fraction'],
                'Ec_positive_Ha': row['Ec_positive'], 'Ec_negative_Ha': row['Ec_negative'],
                'Ec_Ha': row['Ec'], 'Ec_positive_over_abs': ratio,
                'case_B_electron_fraction': row['case_B_electron_fraction'],
                'mechanism_A_Ha': row['mechanisms']['A'], 'mechanism_B_Ha': row['mechanisms']['B'],
                'mechanism_C_Ha': row['mechanisms']['C'], 'mechanism_D_Ha': row['mechanisms']['D'],
            })
            decomposition.append({
                'checkpoint': name, 'system': row['system'],
                'Ec_Ha': row['Ec'], 'production_Exc_Ha': row['production_Exc'],
                'smooth_Ec_Ha': row['smooth_Ec'], 'smooth_Exc_Ha': row['smooth_Exc'],
                'decomposition_max_abs': row['decomposition_max_abs'],
                'beta_min': row['beta_min'], 'beta_max': row['beta_max'],
                'gamma_min': row['gamma_min'], 'gamma_max': row['gamma_max'],
                'activation_gap_Ha': (row['smooth_Exc'] - row['production_Exc']) - (row['smooth_Ec'] - row['Ec']),
            })
            counterfactual.append({
                'checkpoint': name, 'system': row['system'],
                'delta_Ec_Ha': row['smooth_Ec'] - row['Ec'],
                'delta_Exc_Ha': row['smooth_Exc'] - row['production_Exc'],
                'smooth_Ec_positive_Ha': row['smooth_Ec_positive'],
                'Ec_positive_Ha': row['Ec_positive'],
            })
        print(name, 'pooled_gt1', _pooled(payload['systems'], 'fraction_gc_gt_1_electron'),
              'equal_gt1', np.mean([row['fraction_gc_gt_1_electron'] for row in payload['systems']]),
              'pooled_gt1p01', _pooled(payload['systems'], 'fraction_gc_gt_1p01_electron'),
              'nonpositive_weights', sum(row['nonpositive_weights'] for row in payload['systems']),
              'points', sum(row['points'] for row in payload['systems']))
    constraints = []
    for name in NAMES:
        for row in b8.read_json(audit.OUT / name / 'constraints.json')['rows']:
            constraints.append({'checkpoint': name, **row, 'gc_slope_at_0p5_1_1p5': ' '.join(f'{value:.16g}' for value in row['gc_slope_at_0p5_1_1p5'])})
    blocks = [row for row in b8.read_json(audit.OUT / 'saved_blocks.json')['rows'] if row['task'] in ('jw', 'relchem', 'exc', 'op')]
    ec = b8.read_json(audit.OUT / 'ec_block_shares.json')['rows']
    sensitivity = []
    for row in blocks:
        sensitivity.append({
            'checkpoint': row['checkpoint'], 'task': row['task'], 'norm': row['norm'],
            'exchange_squared_fraction': row['exchange_squared_fraction'],
            'correlation_squared_fraction': row['correlation_squared_fraction'],
            'normalization_squared_fraction': row['normalization_squared_fraction'],
            'exchange_norm': row['exchange_norm'], 'correlation_norm': row['correlation_norm'],
            'exchange_norm_per_parameter': row['exchange_norm_per_parameter'],
            'correlation_norm_per_parameter': row['correlation_norm_per_parameter'],
            'smooth_correlation_norm': '', 'cosine_original_smooth': '',
        })
    for row in ec:
        sensitivity.append({
            'checkpoint': row['checkpoint'], 'task': 'Ec_15panel_mean',
            'norm': row['correlation_norm'] + row['exchange_norm'],
            'exchange_squared_fraction': row['shares']['exchange'],
            'correlation_squared_fraction': row['shares']['correlation'],
            'normalization_squared_fraction': row['shares']['normalization'],
            'exchange_norm': row['exchange_norm'], 'correlation_norm': row['correlation_norm'],
            'exchange_norm_per_parameter': '', 'correlation_norm_per_parameter': '',
            'smooth_correlation_norm': row['smooth_correlation_norm'],
            'cosine_original_smooth': row['cosine'],
        })
    slopes = b8.read_json(audit.OUT / 'slopes.json')['rows']
    saturation = []
    for row in slopes:
        sample = np.load(audit.OUT / row['checkpoint'] / 'sample.npz')
        g = sample['g']
        z = np.where(g >= 1.0, g, 1.0 + np.log(np.maximum(g, 1e-30)))
        saturation.append({
            **row,
            'sample_points': len(g),
            'sample_point_mean_abs_smooth_slope': float(np.mean(np.abs(np.tanh(2.0 * (z - 1.0))))),
            'sample_median_gc': float(np.median(g)),
            'population_slope': '15-system electron-weighted full grids',
            'population_sample': '90-system stride-400 point sample',
        })
    panel = b8.read_json(audit.OUT / 'panel.json')['rows']
    operator = []
    for row in panel:
        operator.append({
            'checkpoint': row['checkpoint'], 'system': row['system'],
            'operator_loss': row['op'], 'smooth_operator_loss': row['smooth_op'],
            'delta_operator_loss': row['smooth_op'] - row['op'],
            'Ec_Ha': row['Ec'], 'smooth_Ec_Ha': row['smooth_Ec'],
            'max_epsilon_c': row['max_eps'],
            'ec_gradient_norm': row['ec_gradient_norm'],
            'smooth_ec_gradient_norm': row['smooth_ec_gradient_norm'],
        })
    written = [
        _write('lap_correlation_gc_distribution.csv', distribution),
        _write('lap_correlation_positive_energy.csv', positive),
        _write('lap_correlation_energy_decomposition.csv', decomposition),
        _write('lap_correlation_physical_constraints.csv', constraints),
        _write('lap_correlation_parameter_sensitivity.csv', sensitivity),
        _write('lap_correlation_activation_saturation.csv', saturation),
        _write('lap_correlation_operator_sensitivity.csv', operator),
        _write('lap_correlation_smooth_gc_counterfactual.csv', counterfactual),
    ]
    return written


def _sample(name):
    payload = np.load(audit.OUT / name / 'sample.npz')
    return payload


def plots():
    labels = {'b8_t128': 'B8 t128', 'f010_t251': 'F010 t251', 'f011_t251': 'F011 t251'}
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    for name, label in labels.items():
        g = _sample(name)['g']
        axis.hist(g, bins=80, range=(0.98, 1.12), density=True, histtype='step', linewidth=1.6, label=label)
    axis.set_xlabel(r'$G_c$')
    axis.set_ylabel('Point density')
    axis.set_title('Original $G_c$ on the 90-system mRKS stride-400 sample')
    axis.legend()
    figure.tight_layout()
    figure.savefig(REPO / 'lap_correlation_gc_distribution.png', dpi=140)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    for name, label in labels.items():
        eps = _sample(name)['eps']
        axis.hist(np.log10(-eps), bins=80, density=True, histtype='step', linewidth=1.6, label=label)
    axis.set_xlabel(r'$\log_{10}(-\epsilon_c)$ / Hartree per particle')
    axis.set_ylabel('Point density')
    axis.set_title(r'Local correlation energy, 90-system stride-400 sample')
    axis.legend()
    figure.tight_layout()
    figure.savefig(REPO / 'lap_correlation_local_energy.png', dpi=140)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    names = list(NAMES)
    negative = []
    positive = []
    for name in names:
        rows = _rows(name)
        negative.append(-sum(row['Ec_negative'] for row in rows))
        positive.append(sum(row['Ec_positive'] for row in rows))
    positions = np.arange(len(names))
    axis.bar(positions - 0.18, negative, width=0.36, label=r'$-E_c^-$')
    axis.bar(positions + 0.18, positive, width=0.36, label=r'$E_c^+$')
    axis.set_xticks(positions, names, rotation=20)
    axis.set_ylabel('Hartree, sum over 90 mRKS systems')
    axis.set_title('Integrated correlation contributions, full grids')
    axis.legend()
    figure.tight_layout()
    figure.savefig(REPO / 'lap_correlation_positive_negative.png', dpi=140)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    sample = _sample('b8_t128')
    g, pw, h, rho = sample['g'], sample['pw'], sample['h'], sample['rho']
    mask = (h > 1e-8) & (pw < 0) & (rho > 1e-4)
    crit = -pw[mask] / h[mask]
    order = np.argsort(rho[mask])[:: max(len(crit) // 4000, 1)]
    axis.scatter(crit[order], g[mask][order], s=6, alpha=0.35, linewidths=0)
    limit = [1.0, 1.4]
    axis.plot(limit, limit, color='black', linewidth=1.0)
    axis.set_xlim(1.0, 1.4)
    axis.set_ylim(0.98, 1.14)
    axis.set_xlabel(r'$G_c^{\mathrm{crit}}=-\epsilon_c^{\mathrm{PW92}}/H$')
    axis.set_ylabel(r'$G_c$')
    axis.set_title('B8 t128, stride-400 points with rho > 1e-4 bohr^-3')
    figure.tight_layout()
    figure.savefig(REPO / 'lap_correlation_gc_versus_critical.png', dpi=140)
    plt.close(figure)

    z = np.linspace(0.0, 2.0, 400)
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.plot(z, np.where(z >= 1.0, z, np.exp(z - 1.0)), label='Shifted ELU (production)')
    axis.plot(z, audit.smooth_gc(z, 4.0), label='Smooth bound, k=4 (counterfactual)')
    axis.set_xlabel('Preactivation z')
    axis.set_ylabel(r'$G_c$')
    axis.set_title('Frozen-weight activation substitution, not a trained model')
    axis.legend()
    figure.tight_layout()
    figure.savefig(REPO / 'lap_correlation_activation_comparison.png', dpi=140)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    blocks = {row['task']: row for row in b8.read_json(audit.OUT / 'saved_blocks.json')['rows'] if row['checkpoint'] == 'b8_t128'}
    ec = next(row for row in b8.read_json(audit.OUT / 'ec_block_shares.json')['rows'] if row['checkpoint'] == 'b8_t128')
    tasks = ['jw', 'relchem', 'exc', 'op']
    exchange = [blocks[task]['exchange_squared_fraction'] for task in tasks] + [ec['shares']['exchange']]
    correlation = [blocks[task]['correlation_squared_fraction'] for task in tasks] + [ec['shares']['correlation']]
    labels_x = ['JW', 'relchem', 'Exc', 'operator', r'$E_c$ panel']
    positions = np.arange(len(labels_x))
    axis.bar(positions - 0.18, exchange, width=0.36, label='Exchange block')
    axis.bar(positions + 0.18, correlation, width=0.36, label='Correlation block')
    axis.set_xticks(positions, labels_x)
    axis.set_ylabel('Squared-norm fraction')
    axis.set_ylim(0, 1)
    axis.set_title('B8 t128: training objectives use 90-system means; $E_c$ uses the 15-system panel')
    axis.legend()
    figure.tight_layout()
    figure.savefig(REPO / 'lap_correlation_exchange_response.png', dpi=140)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.plot(z, np.where(z >= 1.0, 1.0, np.exp(z - 1.0)), label='Shifted ELU derivative')
    axis.plot(z, -np.tanh(2.0 * (z - 1.0)), label='Smooth derivative, k=4')
    axis.axvline(1.0, color='black', linewidth=0.8)
    axis.set_xlabel('Preactivation z')
    axis.set_ylabel(r'$dG_c/dz$')
    axis.set_title('Derivative at the PBE anchor z=1 is 1 for ELU and 0 for the smooth bound')
    axis.legend()
    figure.tight_layout()
    figure.savefig(REPO / 'lap_correlation_activation_derivatives.png', dpi=140)
    plt.close(figure)


def execute():
    written = tables()
    plots()
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in written}
    for path in REPO.glob('lap_correlation_*.png'):
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    b8.write_json(audit.OUT / 'artifact_hashes.json', hashes)
    print('RENDER_DONE', len(hashes))


if __name__ == '__main__':
    execute()
