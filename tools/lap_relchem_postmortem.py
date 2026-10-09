"""CPU-only reconstruction of existing fixed-panel relchem endpoints.

Reads frozen evaluation receipts. Does not train, evaluate, or allocate CUDA.
"""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

SHARE = Path(__file__).resolve().parents[2]
REPO = Path(__file__).resolve().parents[1]
POPULATION = 251
DATABASES = ('ABDE4', 'DBH76', 'EA13', 'IP13', 'MGAE109', 'NCCE31', 'PA8', 'pTC13')
BASELINE = 1.2354710978866068
MANIFEST_SHA = '132bd550161be6bc6761f8df1de8d3755c0b0debab53663208c3a9bbf56cb805'
EXPECTED_RATIOS = {
    'iid_t59': 1.013391210,
    'iid_t70': 1.044607979,
    'iid_t80': 1.019138724,
    'iid_t90': 1.026268288,
    'dbstrat_t70': 1.031209044,
    'dbstrat_t90': 1.018877965,
    'lr3e-5_t80': 1.034869578,
}
EXPECTED_CLEAN28 = {
    't0': 9.553190636,
    'iid_t70': 8.629660230,
    'dbstrat_t70': 8.909938464,
    'lr3e-5_t80': 8.619694172,
}
IID_ROOT = SHARE / 'lap_iid_adamw_t59_t90_20261009'
BRANCH_ROOT = SHARE / 'lap_iid_adamw_lr_branches_20261009'
DB_ROOT = SHARE / 'lap_dbstrat_importance_20261009'
CHEMISTRY = {
    't0': IID_ROOT / 'baseline_0_chemistry_one_variant.json',
    'iid_t59': IID_ROOT / 'endpoint_59_chemistry_one_variant.json',
    'iid_t70': BRANCH_ROOT / 'A' / 'endpoint_70_chemistry_one_variant.json',
    'iid_t80': BRANCH_ROOT / 'A' / 'endpoint_80_chemistry_one_variant.json',
    'iid_t90': IID_ROOT / 'endpoint_90_chemistry_one_variant.json',
    'dbstrat_t70': DB_ROOT / 'endpoint_70_chemistry_one_variant.json',
    'dbstrat_t90': DB_ROOT / 'endpoint_90_chemistry_one_variant.json',
    'lr3e-5_t80': BRANCH_ROOT / 'B' / 'endpoint_80_chemistry_one_variant.json',
}
MANIFEST = IID_ROOT / 'evaluation_manifest.json'
IID_MANIFEST = IID_ROOT / 'sampling_manifest.json'
DB_MANIFEST = DB_ROOT / 'sampling_manifest.json'
IID_CHECKPOINT = IID_ROOT / 'ordinary_sgd_adamw' / 'checkpoint_90.pt'
DB_METRICS = REPO / 'lap_dbstrat_importance_metrics.json'
CLEAN28 = {
    't0': IID_ROOT / 'baseline_0_validation.json',
    'iid_t70': IID_ROOT / 'endpoint_70_validation.json',
    'dbstrat_t70': DB_ROOT / 'endpoint_70_validation.json',
    'lr3e-5_t80': BRANCH_ROOT / 'B' / 'endpoint_80_validation.json',
}
DIET_REACTIONS = SHARE / 'publication_dataset_v1' / 'validation' / 'diet30_reactions.jsonl'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def relchem_rows(path, manifest):
    payload = read_json(path)
    if payload.get('evaluation_manifest_sha256') != MANIFEST_SHA or not payload.get('complete'):
        raise RuntimeError(f'Endpoint manifest or completeness mismatch: {path}')
    ae17 = [row for row in payload['rows'].values() if row['task'] == 'ae17']
    if len(ae17) != 17:
        raise RuntimeError(f'AE17 panel is not 17 identities: {path}')
    selected = {identity: row for identity, row in payload['rows'].items() if row['task'] == 'relchem'}
    if set(selected) != set(manifest) or len(selected) != POPULATION:
        raise RuntimeError(f'Relchem identity set differs from the frozen manifest: {path}')
    for identity, row in selected.items():
        if row['variant'] != manifest[identity]['variant']:
            raise RuntimeError(f'Fixed variant mismatch for {identity} in {path}')
    reported = payload['objectives']['relchem']
    reconstructed = math.fsum(row['loss'] for row in selected.values()) / POPULATION
    if abs(reconstructed - reported) > 1e-12:
        raise RuntimeError(f'Reconstructed relchem differs from the receipt: {path}')
    return selected, reported


def database_table(panel, baseline, reported):
    rows = []
    for database in DATABASES:
        identities = [identity for identity, row in panel.items() if row['database'] == database]
        losses = [panel[identity]['loss'] for identity in identities]
        base = [baseline[identity]['loss'] for identity in identities]
        deltas = [loss - old for loss, old in zip(losses, base, strict=True)]
        positive = math.fsum(delta for delta in deltas if delta > 0) / POPULATION
        negative = math.fsum(delta for delta in deltas if delta < 0) / POPULATION
        rows.append({
            'database': database,
            'n': len(identities),
            'mean_loss': math.fsum(losses) / len(identities),
            'contribution': math.fsum(losses) / POPULATION,
            'baseline_mean': math.fsum(base) / len(identities),
            'baseline_contribution': math.fsum(base) / POPULATION,
            'delta_contribution': math.fsum(deltas) / POPULATION,
            'improved_identities': sum(delta < 0 for delta in deltas),
            'worsened_identities': sum(delta > 0 for delta in deltas),
            'positive_contribution': positive,
            'negative_contribution': negative,
        })
    total = math.fsum(row['contribution'] for row in rows)
    delta = math.fsum(row['delta_contribution'] for row in rows)
    if abs(total - reported) > 1e-12:
        raise RuntimeError('Database contributions do not reproduce the relchem objective')
    if abs(delta - (reported - BASELINE)) > 1e-12:
        raise RuntimeError('Database deltas do not reproduce the relchem change')
    return rows


def identity_deltas(panel, baseline):
    rows = []
    for identity, row in panel.items():
        old = baseline[identity]
        if row['database'] != old['database'] or row['variant'] != old['variant']:
            raise RuntimeError(f'Identity alignment failed for {identity}')
        rows.append({
            'identity': identity,
            'database': row['database'],
            'variant': row['variant'],
            'baseline_loss': old['loss'],
            'loss': row['loss'],
            'difference': row['loss'] - old['loss'],
        })
    return rows


def concentration(deltas):
    positive = sorted((row for row in deltas if row['difference'] > 0), key=lambda row: row['difference'], reverse=True)
    total_positive = math.fsum(row['difference'] for row in positive) / POPULATION
    net = math.fsum(row['difference'] for row in deltas) / POPULATION
    summary = {'net': net, 'positive_mass': total_positive, 'n_worsened': len(positive), 'n_improved': sum(row['difference'] < 0 for row in deltas)}
    for count in (5, 10, 20):
        piece = math.fsum(row['difference'] for row in positive[:count]) / POPULATION
        summary[f'top_{count}_positive'] = piece
        summary[f'top_{count}_share_of_positive'] = None if total_positive == 0 else piece / total_positive
        summary[f'top_{count}_share_of_net'] = None if net == 0 else piece / net
    return summary


def exposure(manifest):
    counts = Counter(row['relchem']['database'] for row in manifest)
    by_identity = Counter(row['relchem']['identity'] for row in manifest)
    per_database = []
    for database in DATABASES:
        identities = [row['relchem']['identity'] for row in manifest if row['relchem']['database'] == database]
        repeats = Counter(identities)
        per_database.append({
            'database': database,
            'samples': counts[database],
            'unique_identities': len(repeats),
            'repeated_identities': sum(count > 1 for count in repeats.values()),
            'max_repeats': max(repeats.values(), default=0),
        })
    return {
        'rows': len(manifest),
        'unique_identities': len(by_identity),
        'databases': per_database,
        'variant_counts': dict(Counter(row['relchem']['variant'] for row in manifest)),
        'identity_counts': dict(by_identity),
    }


def exposure_association(identity_count, deltas):
    bins = []
    for label, predicate in (('unseen_in_90', lambda count: count == 0), ('seen_once', lambda count: count == 1), ('seen_more_than_once', lambda count: count > 1)):
        chosen = [row for row in deltas if predicate(identity_count.get(row['identity'], 0))]
        bins.append({
            'bin': label,
            'n': len(chosen),
            'mean_difference': None if not chosen else math.fsum(row['difference'] for row in chosen) / len(chosen),
            'net_contribution': math.fsum(row['difference'] for row in chosen) / POPULATION,
        })
    return bins


def iid_norms(path):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    saved = torch.load(path, map_location='cpu', weights_only=False)
    return [{
        'cursor': index,
        'database': record['sample']['relchem']['database'],
        'identity': record['sample']['relchem']['identity'],
        'raw_relchem_norm': record['norms']['relchem'],
        'joint_norm': record['weighted_gradient_norm'],
    } for index, record in enumerate(saved['logs'])]


def clean28(path):
    metrics = read_json(path)['metrics']
    clean = [row for row in metrics['reaction_rows'] if row['clean']]
    if len(clean) != 28:
        raise RuntimeError(f'Clean28 panel is not 28 reactions: {path}')
    reconstructed = math.fsum(row['weighted_absolute_error'] for row in clean) / 28
    if abs(reconstructed - metrics['clean28']) > 1e-12:
        raise RuntimeError(f'Clean28 does not reconstruct: {path}')
    return metrics['clean28'], clean


def diet_labels():
    labels = {}
    for line in DIET_REACTIONS.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        labels[row['source_id']] = row['database']
    return labels


def variant_coverage(manifest, rows):
    summary = []
    for row in rows:
        draws = [item['relchem'] for item in manifest if item['relchem']['identity'] == row['identity']]
        summary.append({
            'identity': row['identity'],
            'draws': len(draws),
            'fixed_variant_draws': sum(item['variant'] == row['variant'] for item in draws),
        })
    return {
        'n': len(summary),
        'absent': sum(item['draws'] == 0 for item in summary),
        'fixed_variant_seen': sum(item['fixed_variant_draws'] > 0 for item in summary),
    }


def cited_objectives(reconstructed_ratios):
    """Read previously stored four-objective ratios. Do not recompute them."""
    lr = read_json(REPO / 'iid_adamw_lr_stabilization_metrics.json')
    db = read_json(DB_METRICS)
    cited = {
        'iid_t59': lr['audits']['A59']['ratios_t0'],
        'iid_t70': lr['audits']['A70']['ratios_t0'],
        'iid_t80': lr['audits']['A80']['ratios_t0'],
        'iid_t90': lr['audits']['A90']['ratios_t0'],
        'lr3e-5_t80': lr['audits']['B80']['ratios_t0'],
        'dbstrat_t70': db['science']['70']['ratios'],
        'dbstrat_t90': db['science']['90']['ratios'],
    }
    for name, ratios in cited.items():
        if abs(ratios['relchem'] - reconstructed_ratios[name]) > 1e-9:
            raise RuntimeError(f'Cited relchem ratio disagrees with the endpoint reconstruction: {name}')
    return cited


def analyze():
    manifest_rows = read_json(MANIFEST)['rows']
    manifest = {row['identity']: row for row in manifest_rows if row['task'] == 'relchem'}
    if len(manifest) != POPULATION:
        raise RuntimeError('Frozen manifest does not contain 251 relchem identities')
    panels = {}
    reported = {}
    for name, path in CHEMISTRY.items():
        panels[name], reported[name] = relchem_rows(path, manifest)
    if abs(reported['t0'] - BASELINE) > 1e-15:
        raise RuntimeError('P536 relchem receipt differs from the frozen reference')
    ratios = {name: reported[name] / reported['t0'] for name in EXPECTED_RATIOS}
    for name, expected in EXPECTED_RATIOS.items():
        if abs(ratios[name] - expected) > 5e-10:
            raise RuntimeError(f'Reported ratio mismatch for {name}: {ratios[name]} vs {expected}')
    databases = {name: set(row['database'] for row in panel.values()) for name, panel in panels.items()}
    if any(found != set(DATABASES) for found in databases.values()):
        raise RuntimeError('A checkpoint uses a different relchem database partition')
    tables = {name: database_table(panels[name], panels['t0'], reported[name]) for name in panels if name != 't0'}
    deltas = {name: identity_deltas(panels[name], panels['t0']) for name in panels if name != 't0'}
    ranked = {}
    for name, rows in deltas.items():
        worsening = sorted(rows, key=lambda row: row['difference'], reverse=True)[:20]
        improving = sorted(rows, key=lambda row: row['difference'])[:20]
        ranked[name] = {'worsening': worsening, 'improving': improving, 'concentration': concentration(rows)}
    both = {}
    for cursor in (70, 90):
        iid = {row['identity']: row for row in deltas[f'iid_t{cursor}']}
        other = {row['identity']: row for row in deltas[f'dbstrat_t{cursor}']}
        shared = []
        db_only = []
        for identity, row in iid.items():
            partner = other[identity]
            if row['difference'] > 0 and partner['difference'] > 0:
                shared.append({**row, 'dbstrat_difference': partner['difference']})
            if row['difference'] > 0 and partner['difference'] < 0:
                db_only.append({**row, 'dbstrat_difference': partner['difference']})
        both[str(cursor)] = {
            'shared_worsening_count': len(shared),
            'shared_worsening_top20': sorted(shared, key=lambda row: row['difference'] + row['dbstrat_difference'], reverse=True)[:20],
            'improved_by_dbstrat_while_iid_worsened_count': len(db_only),
            'improved_by_dbstrat_while_iid_worsened_top20': sorted(db_only, key=lambda row: row['difference'] - row['dbstrat_difference'], reverse=True)[:20],
        }
    iid_sampling = exposure(read_json(IID_MANIFEST))
    db_sampling = exposure(read_json(DB_MANIFEST))
    association = {
        'iid_t70': exposure_association(iid_sampling['identity_counts'], deltas['iid_t70']),
        'dbstrat_t70': exposure_association(db_sampling['identity_counts'], deltas['dbstrat_t70']),
    }
    iid_gradient = iid_norms(IID_CHECKPOINT)
    db_gradient = [{
        'cursor': row['cursor'] - 1,
        'database': row['database'],
        'raw_relchem_norm': row['relchem_norm_raw'],
        'corrected_relchem_norm': row['relchem_norm_corrected'],
    } for row in read_json(DB_METRICS)['updates']]
    clean_values = {}
    clean_rows = {}
    for name, path in CLEAN28.items():
        value, rows = clean28(path)
        if abs(value - EXPECTED_CLEAN28[name]) > 5e-9:
            raise RuntimeError(f'Clean28 mismatch for {name}: {value}')
        clean_values[name] = value
        clean_rows[name] = rows
    labels = diet_labels()
    clean_categories = {}
    for name, rows in clean_rows.items():
        grouped = {}
        for row in rows:
            label = labels[row['reaction_id']]
            bucket = grouped.setdefault(label, {'n': 0, 'contribution_sum': 0.0})
            bucket['n'] += 1
            bucket['contribution_sum'] += row['weighted_absolute_error'] / 28
        clean_categories[name] = grouped
    worst = sorted(deltas['iid_t70'], key=lambda row: row['difference'], reverse=True)[:20]
    coverage = {
        'iid_top20_worsening': variant_coverage(read_json(IID_MANIFEST), worst),
        'dbstrat_on_those_identities': variant_coverage(read_json(DB_MANIFEST), worst),
        'iid_probability_of_missing_abde4': (247 / 251) ** 90,
    }
    other = cited_objectives(ratios)
    labeled_inputs = {
        'evaluation_manifest': MANIFEST,
        'iid_sampling_manifest': IID_MANIFEST,
        'dbstrat_sampling_manifest': DB_MANIFEST,
        'dbstrat_metrics': DB_METRICS,
        'iid_metrics': REPO / 'iid_adamw_t59_t90_metrics.json',
        'lr_metrics': REPO / 'iid_adamw_lr_stabilization_metrics.json',
        'iid_checkpoint_90': IID_CHECKPOINT,
        'diet30_reactions': DIET_REACTIONS,
    }
    labeled_inputs.update({f'chemistry_{name}': path for name, path in CHEMISTRY.items()})
    labeled_inputs.update({f'clean28_{name}': path for name, path in CLEAN28.items()})
    inputs = {name: {'path': str(path), 'sha256': sha256(path)} for name, path in labeled_inputs.items()}
    return {
        'reported_relchem': reported,
        'ratios': ratios,
        'database_tables': tables,
        'ranked': ranked,
        'matched_overlap': both,
        'sampling': {
            'iid': {key: value for key, value in iid_sampling.items() if key != 'identity_counts'},
            'dbstrat': {key: value for key, value in db_sampling.items() if key != 'identity_counts'},
            'association_t70': association,
            'iid_norm_count': len(iid_gradient),
            'iid_norm_median': median(row['raw_relchem_norm'] for row in iid_gradient),
            'dbstrat_raw_norm_median': median(row['raw_relchem_norm'] for row in db_gradient),
            'dbstrat_corrected_norm_median': median(row['corrected_relchem_norm'] for row in db_gradient),
        },
        'other_objective_ratios': other,
        'fixed_variant_coverage': coverage,
        'clean28': clean_values,
        'clean28_diet_subset_contribution_sums': clean_categories,
        'inputs': inputs,
        'loss_definition': (
            'Qualified singleton loss from optuna_joint.batch_fchem: '
            'database_weight * frequency_weight / mean_weight * sqrt(MSE + 1e-20). '
            'The fixed-panel relchem objective is the equal-identity mean of these 251 scalars. '
            'Database and frequency factors were not applied again.'
        ),
    }


def median(values):
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def reaction_table(rows):
    lines = [
        '| Identity | Database | Variant | P536 loss | Checkpoint loss | Difference |',
        '|---|---|---|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(
            f"| `{row['identity']}` | {row['database']} | {row['variant']} | "
            f"{row['baseline_loss']:.6f} | {row['loss']:.6f} | {row['difference']:+.6f} |"
        )
    return lines


def render(result):
    consistent_negative = []
    consistent_positive = []
    for database in DATABASES:
        deltas = [
            next(row['delta_contribution'] for row in rows if row['database'] == database)
            for rows in result['database_tables'].values()
        ]
        if all(delta < 0 for delta in deltas):
            consistent_negative.append(database)
        if all(delta > 0 for delta in deltas):
            consistent_positive.append(database)
    iid70_conc = result['ranked']['iid_t70']['concentration']
    iid_db = {row['database']: row for row in result['database_tables']['iid_t70']}
    top_overlap = len(
        {row['identity'] for row in result['ranked']['iid_t70']['worsening']}
        & {row['identity'] for row in result['ranked']['dbstrat_t70']['worsening']}
    )
    lines = [
        '# Relchem postmortem: why the fixed panel stays above P536',
        '',
        'This is an offline reconstruction of existing fixed-variant chemistry receipts. No model was evaluated again.',
        '',
        result['loss_definition'],
        '',
        'The stored P536 relchem objective is 1.2354710978866068. Recalculated ratios agree with the published nine-decimal receipts to within 5e-10. AE17, Exc, and operator ratios below are copied from the existing audit JSON files and were not recomputed.',
        '',
        '| Checkpoint | Relchem | AE17 | Exc | Operator |',
        '|---|---:|---:|---:|---:|',
    ]
    for name in ('iid_t59', 'iid_t70', 'iid_t80', 'iid_t90', 'dbstrat_t70', 'dbstrat_t90', 'lr3e-5_t80'):
        ratios = result['other_objective_ratios'][name]
        lines.append(f"| {name} | {ratios['relchem']:.9f} | {ratios['ae17']:.9f} | {ratios['exc']:.9f} | {ratios['op']:.9f} |")
    lines.extend(['',
        '## Finding',
        '',
        f"The relchem ratio stays above 1 at every audited checkpoint because a minority of identities become much worse in the qualified singleton loss. At IID t70, {iid70_conc['n_improved']} of 251 identities improve and {iid70_conc['n_worsened']} worsen. The worsening mass is {iid70_conc['positive_mass']:.4f} and the net rise is {iid70_conc['net']:.4f}. The largest 20 positive reactions account for {iid70_conc['top_20_share_of_positive']:.1%} of the worsening mass and {iid70_conc['top_20_share_of_net']:.1f} times the net rise, because the improvements cancel most of the damage.",
        '',
        'The net rise is carried by the small databases ABDE4, pTC13, and PA8. At IID t70 their contributions to the change are '
        f"{iid_db['ABDE4']['delta_contribution']:+.4f}, {iid_db['pTC13']['delta_contribution']:+.4f}, and {iid_db['PA8']['delta_contribution']:+.4f}. "
        f"Together they exceed the net rise. Databases with a negative delta at every audited checkpoint: {', '.join(consistent_negative)}. Databases with a positive delta at every audited checkpoint: {', '.join(consistent_positive)}. Population size is not the source of the regression: the large databases are in the improving group.",
        '',
        f"The same three small databases stay above P536 under database-stratified sampling. At t70, {result['matched_overlap']['70']['shared_worsening_count']} identities worsen in both streams, and {top_overlap} of the 20 largest IID worsenings are also among the 20 largest DB-strat worsenings. DB-strat reduces the ABDE4 delta relative to IID but does not make ABDE4, pTC13, or PA8 better than P536. The failure is therefore not an IID-only omission, and it is not spread evenly over all 251 identities.",
        '',
        '## What the objective change is made of',
        '',
        'A positive database delta raises the relchem objective. Contribution is the sum of qualified singleton losses in that database divided by 251. Mean loss is the same sum divided by the database size. These are not ordinary unweighted MAE values.',
        '',
    ])
    focus = ('iid_t59', 'iid_t70', 'iid_t80', 'iid_t90', 'dbstrat_t70', 'dbstrat_t90', 'lr3e-5_t80')
    lines.append('| Checkpoint | Relchem | Ratio to P536 | Net delta | Top-20 share of worsening mass | Identities worsened |')
    lines.append('|---|---:|---:|---:|---:|---:|')
    for name in focus:
        block = result['ranked'][name]['concentration']
        lines.append('| {} | {:.12f} | {:.9f} | {:+.6e} | {:.3f} | {} / 251 |'.format(
            name, result['reported_relchem'][name], result['ratios'][name], block['net'],
            block['top_20_share_of_positive'], block['n_worsened']))
    lines.extend(['', '### Eight-database change from P536', ''])
    header = '| Database | n | ' + ' | '.join(focus) + ' |'
    lines.append(header)
    lines.append('|---|---:|' + '|'.join(['---:'] * len(focus)) + '|')
    by_checkpoint = {name: {row['database']: row for row in rows} for name, rows in result['database_tables'].items()}
    for database in DATABASES:
        cells = [f"{by_checkpoint[name][database]['delta_contribution']:+.4e}" for name in focus]
        lines.append(f"| {database} | {by_checkpoint['iid_t70'][database]['n']} | " + ' | '.join(cells) + ' |')
    lines.extend(['', '### IID t70 and DB-strat t70 in the qualified-loss units', ''])
    lines.append('| Database | n | IID mean | IID contribution | IID delta | DB mean | DB contribution | DB delta | DB minus IID | IID improved | DB improved | IID positive | IID negative | DB positive | DB negative |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    iid70 = by_checkpoint['iid_t70']
    db70 = by_checkpoint['dbstrat_t70']
    for database in DATABASES:
        left, right = iid70[database], db70[database]
        lines.append(
            f"| {database} | {left['n']} | {left['mean_loss']:.4f} | {left['contribution']:.4f} | {left['delta_contribution']:+.4e} | "
            f"{right['mean_loss']:.4f} | {right['contribution']:.4f} | {right['delta_contribution']:+.4e} | "
            f"{right['delta_contribution'] - left['delta_contribution']:+.4e} | "
            f"{left['improved_identities']}/{left['n']} | {right['improved_identities']}/{right['n']} | "
            f"{left['positive_contribution']:+.4e} | {left['negative_contribution']:+.4e} | "
            f"{right['positive_contribution']:+.4e} | {right['negative_contribution']:+.4e} |"
        )
    signs = []
    for database in DATABASES:
        deltas = [by_checkpoint[name][database]['delta_contribution'] for name in result['database_tables']]
        if all(delta < 0 for delta in deltas):
            signs.append(f'{database} improves at every audited checkpoint')
        elif all(delta > 0 for delta in deltas):
            signs.append(f'{database} worsens at every audited checkpoint')
    lines.extend(['', 'Sign of the database delta across every audited checkpoint:', ''])
    lines.extend(f'- {item}' for item in signs)
    lines.extend(['', '### How concentrated the worsening is', ''])
    lines.append('| Checkpoint | Top 5 / positive mass | Top 10 / positive mass | Top 20 / positive mass | Top 20 / net rise |')
    lines.append('|---|---:|---:|---:|---:|')
    for name in focus:
        block = result['ranked'][name]['concentration']
        lines.append('| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |'.format(
            name, block['top_5_share_of_positive'], block['top_10_share_of_positive'],
            block['top_20_share_of_positive'], block['top_20_share_of_net']))
    for name in ('iid_t70', 'dbstrat_t70', 'iid_t90', 'dbstrat_t90'):
        lines.extend(['', f'### Twenty largest worsenings at {name}', ''])
        lines.extend(reaction_table(result['ranked'][name]['worsening']))
        lines.extend(['', f'### Twenty largest improvements at {name}', ''])
        lines.extend(reaction_table(result['ranked'][name]['improving']))
    overlap = result['matched_overlap']['70']
    overlap90 = result['matched_overlap']['90']
    lines.extend([
        '',
        '## IID versus database-stratified relchem',
        '',
        f"At t70, {overlap['shared_worsening_count']} identities are worse than P536 in both streams, and {overlap['improved_by_dbstrat_while_iid_worsened_count']} are worse under IID but improved under DB-strat.",
        f"At t90 those counts are {overlap90['shared_worsening_count']} and {overlap90['improved_by_dbstrat_while_iid_worsened_count']}.",
        '',
        '### Largest reactions worse in both streams at t70',
        '',
        '| Identity | Database | Variant | P536 loss | IID loss | DB-strat loss | IID difference | DB-strat difference |',
        '|---|---|---|---:|---:|---:|---:|---:|',
    ])
    for row in overlap['shared_worsening_top20']:
        lines.append(
            f"| `{row['identity']}` | {row['database']} | {row['variant']} | {row['baseline_loss']:.6f} | {row['loss']:.6f} | "
            f"{row['baseline_loss'] + row['dbstrat_difference']:.6f} | {row['difference']:+.6f} | {row['dbstrat_difference']:+.6f} |"
        )
    lines.extend(['', '### Worse under IID t70 and improved under DB-strat t70', '',
                   '| Identity | Database | Variant | P536 loss | IID loss | DB-strat loss | IID difference | DB-strat difference |',
                   '|---|---|---|---:|---:|---:|---:|---:|'])
    for row in overlap['improved_by_dbstrat_while_iid_worsened_top20']:
        lines.append(
            f"| `{row['identity']}` | {row['database']} | {row['variant']} | {row['baseline_loss']:.6f} | {row['loss']:.6f} | "
            f"{row['baseline_loss'] + row['dbstrat_difference']:.6f} | {row['difference']:+.6f} | {row['dbstrat_difference']:+.6f} |"
        )
    lines.extend([
        '',
        '## Sampling exposure',
        '',
        'Counts below are the 90 executed relchem draws. They are not claims about the earlier P536 predopt phase.',
        '',
        '| Database | IID samples | IID unique | DB-strat samples | DB-strat unique |',
        '|---|---:|---:|---:|---:|',
    ])
    iid_db = {row['database']: row for row in result['sampling']['iid']['databases']}
    db_db = {row['database']: row for row in result['sampling']['dbstrat']['databases']}
    for database in DATABASES:
        lines.append(f"| {database} | {iid_db[database]['samples']} | {iid_db[database]['unique_identities']} | {db_db[database]['samples']} | {db_db[database]['unique_identities']} |")
    lines.extend(['', 'Mean qualified-loss change at t70 by whether the identity appeared in that arm\'s 90 draws:', ''])
    for arm, rows in result['sampling']['association_t70'].items():
        rendered = ', '.join(f"{row['bin']} n={row['n']} mean delta={row['mean_difference']:+.4f}" for row in rows)
        lines.append(f'- {arm}: {rendered}')
    lines.extend([
        '',
        f"Median logged raw relchem gradient norm over 90 updates: IID {result['sampling']['iid_norm_median']:.3f}; DB-strat raw {result['sampling']['dbstrat_raw_norm_median']:.3f}; DB-strat after multiplying by the importance weight once {result['sampling']['dbstrat_corrected_norm_median']:.3f}. The IID median matches the paired control series already stored in the DB-strat metrics. The larger DB-strat raw norms follow the databases that sampler draws: ABDE4, PA8, and pTC13 have much larger raw norms than MGAE109 and DBH76.",
        '',
        f"Of the 20 largest IID t70 worsenings, {result['fixed_variant_coverage']['iid_top20_worsening']['absent']} identities are absent from the IID manifest and {result['fixed_variant_coverage']['iid_top20_worsening']['fixed_variant_seen']} include the frozen evaluation variant. On the DB-strat manifest, {result['fixed_variant_coverage']['dbstrat_on_those_identities']['absent']} of those identities are absent and {result['fixed_variant_coverage']['dbstrat_on_those_identities']['fixed_variant_seen']} include the frozen variant. The fixed-panel losses therefore moved without repeated training on the evaluated grid. Under uniform identity sampling, the probability of drawing no ABDE4 identity in 90 updates is {(247 / 251) ** 90:.3f}. That makes one empty ABDE4 stream plausible. It does not show that those identities were absent from the earlier P536 predopt phase.",
        '',
        'The exposure bins are descriptive. In the IID stream, unseen identities account for the net rise and seen identities improve on average. In the DB-strat stream the repeated identities are the small databases, and those repeats have a positive mean change. That association is confounded by which databases receive the repeats. It is not a causal estimate of exposure.',
        '',
        '## Clean28',
        '',
        'Clean28 is the mean Diet-weighted absolute error of 28 leakage-clean reactions. It is not the 251-identity relchem objective. Identifiers differ (`ACONF-10` versus `reaction_<hash>`), so the panels were not joined.',
        '',
        '| Checkpoint | Clean28 |',
        '|---|---:|',
    ])
    for name, value in result['clean28'].items():
        lines.append(f'| {name} | {value:.9f} |')
    lines.extend([
        '',
        'Clean28 falls at every cited checkpoint while the 251-identity relchem mean rises, so Clean28 is not a proxy for relchem eligibility. Diet30 subset labels on that external panel are stored in the metrics. They are not the eight training databases, and similarly named reactions were not treated as the same identity.',
        '',
        '## Skala-1.1 v6, from arXiv:2506.14665v6',
        '',
        'The current manuscript identifies Skala-1.1, trained on about 400,000 energy differences, as superseding the earlier Skala-1.0 recipe. Facts below are from the fetched v6 text.',
        '',
        'Pretraining evaluates the functional on fixed B3LYP densities and precomputed non-XC total-energy components, including D3 (main text Sec. 2.1 and Supplement B.1, Eq. 29). The loss is the expectation of squared reaction-energy error divided by `1e-4 Eh + |reference reaction energy|` (B.1, Eq. 31). That is not the qualified singleton loss used here, which is a database-and-frequency-weighted square root of a one-reaction MSE.',
        '',
        'Sampling is two-level: a dataset is drawn with probability `p_i`, then a reaction is drawn uniformly inside it (B.2). Initial `p_i` is proportional to category weight times dataset size. Nine categories are named. The numeric category weights and target proportions are NOT VERIFIED; the extracted B.2 says they are solved to hit prescribed proportions but does not list them. Every 25,000 steps, relative excess loss against baseline DFT MAEs multiplies those probabilities by `exp(0.025 * REL)` and renormalizes them (B.2, Eqs. 32-33). This is not uniform-database sampling followed by an inverse-probability correction. The B.2 title includes model selection. An explicit checkpoint-picking rule beyond the REL diagnostic was NOT VERIFIED in the extracted section.',
        '',
        'Optimization uses Muon on hidden matrices and Adam on biases and the final layer, separate cosine schedules with 50,000 warmup steps, 1,000,000 pretraining steps, a gradient-clipping threshold of 0.0001, and EMA decay 0.9999 (B.4, Eq. 34 and Table 2). Peak learning rates are 0.0007 for Muon and 0.00015 for Adam. The numeric cosine floor is NOT VERIFIED. Ablations used 8 A100 GPUs with one reaction per GPU. This is not constant-LR AdamW for 90 updates.',
        '',
        'Fine-tuning runs 20,000 steps on the model\'s own SCF densities, with the optimizer reset, constant learning rate 1e-5, the same weighted loss, and sampling probabilities frozen from step 1,000,000 (B.5). No gradient is propagated through SCF. That procedure is not authorized here and was not run.',
        '',
        '| Component | Skala-1.1 v6 | Current piNN-DFT 90-update arms | Potential relevance |',
        '|---|---|---|---|',
        '| Density source | Fixed B3LYP densities, not B3LYP energies, in pretraining; the model\'s own SCF densities in fine-tuning (Sec. 2.1, B.1, B.5) | Frozen Minnesota reaction grids for the 251-identity objective. Clean28 uses frozen PBE0 densities and PBE0-D3(BJ). The SCF functional that generated the Minnesota grids is NOT VERIFIED here | Both pretraining stages evaluate a fixed density. The density sources are not the same |',
        '| Reaction regression loss | Weighted MSE, Eq. 31 | Qualified weighted sqrt-MSE singleton, then an equal-identity mean | The objective being audited is not Skala\'s reaction MSE |',
        '| Sampling probabilities | Hierarchical, then REL-adaptive | IID uniform identity, or uniform database plus `p/q` | Skala\'s adaptation changes effort toward weak datasets; our correction preserves the IID singleton expectation |',
        '| Chemical dataset balance | Category targets, then excess-loss updates | Frozen Minnesota database/frequency factors inside the singleton loss | Our factors are constant weights, not an online sampler |',
        '| Optimizer | Muon plus Adam, warmup and cosine | Constant AdamW, lr 1e-4 or one reduced-lr branch | No evidence yet that the optimizer family explains the relchem ratio |',
        '| Training duration | 1,000,000 pretraining steps plus 20,000 fine-tuning steps | 536 PBE predopt updates, then 90 AdamW updates | The audited failure is inside the short AdamW phase |',
        '| Model stabilization | EMA 0.9999, gradient clipping | No EMA and no clipping change in these arms | NOT VERIFIED as a cause of the relchem ratio |',
        '| SCF fine-tuning | 20,000 on-policy SCF steps | Not authorized | Cannot explain these fixed-density relchem numbers |',
        '| Scientific constraints | Enhancement-factor constraints including uniform scaling, size consistency, and a Lieb-Oxford bound | Tau-free Laplacian NN-PBE with the existing PBE constraints | Both constrain the XC form; the constraint sets are not the same |',
        '',
        'Skala demonstrates a long hierarchical pretraining run and a separate SCF fine-tune on a much larger reaction collection. Our receipts demonstrate that 90 AdamW updates improve AE17, Exc, and the operator while the fixed 251-identity relchem mean rises. The comparison suggests hypotheses. It does not show that copying Skala\'s sampler would lower this relchem objective.',
        '',
        result['recommendation'],
        '',
        '## Input receipts',
        '',
    ])
    for name, item in sorted(result['inputs'].items()):
        lines.append(f"- `{name}`: `{item['path']}` `{item['sha256']}`")
    lines.append('')
    return '\n'.join(lines)


def recommendation(result):
    iid = result['ranked']['iid_t70']['concentration']
    db = result['ranked']['dbstrat_t70']['concentration']
    return '\n'.join([
        '## Next experiment, not executed',
        '',
        f"IID t70 net relchem change {iid['net']:+.6e}, with {iid['n_worsened']} identities worse and the largest 20 explaining {iid['top_20_share_of_positive']:.1%} of the positive mass. DB-strat t70 net change {db['net']:+.6e}, top-20 share {db['top_20_share_of_positive']:.1%}.",
        '',
        'Candidate A asks whether an independent 90-update IID AdamW stream from the same corrected P536 reproduces a relchem ratio above 1 driven by ABDE4, pTC13, and PA8. It keeps the qualified singleton loss, the task coefficients, and the uniform identity sampler, so it does not change the intended training gradient. The earlier IID continuation averaged 15.3 s/update over 31 updates. The DB-strat run averaged 32.7 s/update over 90 updates. A new 90-update arm is therefore about 25-50 minutes on one local GPU. One fixed-variant chemistry audit plus one 90-system mRKS audit, using the existing evaluator, adds roughly 6 minutes if those receipts are collected. The main risk is a second negative that still does not identify a mechanism. The stream-specific hypothesis is falsified if the new checkpoint with the best Clean28 still has relchem/P536 at least 1, with positive deltas for ABDE4, pTC13, and PA8. It is supported only if all four scientific ratios are finite and strictly below 1. One bounded arm can test it.',
        '',
        'Candidate B would emphasize databases with excess relchem loss, in the spirit of Skala\'s relative-excess update. Without an importance correction it changes the expected training gradient. The database-stratified arm already raised ABDE4 draws from 0 to 8, pTC13 from 4 to 15, and PA8 from 5 to 9, with the inverse-probability correction, and those three deltas stayed positive. Skala\'s baseline targets and normalizers for these eight databases are not available and are not invented here. B is a different experiment from A, and this postmortem does not justify running it next. Its cost would be another 90-update trajectory. It would be falsified if the fixed-panel relchem ratio did not fall relative to a paired IID replica.',
        '',
        'Candidate C would change the qualified loss weights or the four task coefficients. That changes the objective whose eligibility is being judged. The present deltas are already inside the weighted singleton loss, and the reduced-learning-rate branch, which continued the same IID stream at 3e-5, still left ABDE4, pTC13, and PA8 above P536. A short coefficient trial could be bounded, but it would not answer whether the current objective can pass.',
        '',
        'Selected next experiment: **A. One independent 90-update IID AdamW replica from corrected P536, with unchanged coefficients, the qualified singleton loss, and the existing uniform identity sampler.** Stop on a nonfinite update, a manifest or coefficient change, or a nonfinite scientific ratio. Do not promote a checkpoint unless relchem, AE17, Exc, and the operator are all strictly below their P536 values. Do not start B or C from this result. If A reproduces the same three-database pattern, the next question is why those qualified losses rise under AdamW, not another sampler.',
    ])


def main():
    result = analyze()
    result['recommendation'] = recommendation(result)
    metrics_path = REPO / 'lap_relchem_postmortem_metrics.json'
    report_path = REPO / 'lap_relchem_postmortem_report.md'
    payload = {key: value for key, value in result.items() if key != 'recommendation'}
    payload['recommendation'] = 'A: one independent 90-update IID AdamW replica from corrected P536'
    metrics_path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    report_path.write_text(render(result), encoding='utf-8')
    print('POSTMORTEM', result['ratios']['iid_t70'], result['ranked']['iid_t70']['concentration']['top_20_share_of_positive'])


if __name__ == '__main__':
    main()
