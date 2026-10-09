"""Frozen-model audit of three chemistry losses at P536 and IID t70.

Compares the qualified singleton loss, a literal Skala reaction-energy loss,
and a normalized Huber alternative. Does not train, step an optimizer, run SCF,
or write a checkpoint.
"""
import argparse
import ast
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

K = 627.5095
SMOOTH = 1e-20
FLOOR_H = 1e-4
DELTA = 0.1
DERIVATIVE_FLOOR = 1e-8
LOSS_ATOL = 1e-6
FORMULA_ATOL = 1e-8
PARITY_RTOL = 1e-4
PARAMETER_COUNT = 9446
POPULATION = 251
PANEL_COUNT = 40
BUDGET_SECONDS = 30 * 60
ORIGINAL_BACKWARDS = 80
DIRECT_BACKWARD_LIMIT = 6
SEED = '202610094'
REL_DATABASES = (
    'ABDE4', 'DBH76', 'EA13', 'IP13', 'MGAE109', 'NCCE31', 'PA8', 'pTC13',
)
FOCUS_DATABASES = ('ABDE4', 'pTC13', 'PA8')
CONTROL_DATABASES = ('DBH76', 'EA13', 'IP13', 'MGAE109', 'NCCE31')
EXPECTED_COUNTS = {
    'ABDE4': 4, 'DBH76': 70, 'EA13': 11, 'IP13': 13,
    'MGAE109': 104, 'NCCE31': 28, 'PA8': 8, 'pTC13': 13,
}
DIRECT_DATABASES = ('ABDE4', 'pTC13', 'DBH76')
STUBBORN = (
    'reaction_499996e5b8084d7129c856a5',
    'reaction_c258198ad576955c8267a32b',
    'reaction_18a4cfbaf87fde8157323b2d',
    'reaction_dce81e2069e365d6f6f375e6',
    'reaction_2711e62a8820baac39b75e55',
)
LAMBDAS_OLD = {
    'relchem': 0.017015480965588553,
    'ae17': 0.00005141254618347414,
    'exc': 0.000015094644512009712,
    'op': 0.33597561607048215,
}
DATA_SHA = '61c221a19b9987717e69cac182ad545241f8807db4126c0949a99992e4c210ef'
MANIFEST_SHA = '132bd550161be6bc6761f8df1de8d3755c0b0debab53663208c3a9bbf56cb805'
S0_FILE_SHA = '0ca0f77367214c171e6edf0576bdb043d64479245ad1a0e539fa11fb0cee542d'
S0_TENSOR_SHA = '3c2e45d6a86cccf80d7b4dc458e67f9b406935ed60d4e57433bb441a6df88da6'
S70_FILE_SHA = '04b8e549c17375988be576e2878881d13f55b217b42ec16a7e6101fdddd05443'
S70_TENSOR_SHA = '59ab4b98550805b13852e13745281fb1d072efcbed5c2c8622bf3a6840c3ff51'
SOURCE_SHA = {
    'train_lap_microbatch.py': '2e837c3a88ca3d4737c3c3397dcfbb98adeb7f6917adc8c693cecf205dc37e9e',
    'train_models/optuna_joint.py': '59c66d798ed7fd74b05caba271b590cca138d53336114edc820b25c9187846f5',
    'train_models/lap_training.py': 'ca789703e536f9311178ada07fd41d92ba7f79afd35b25ad240647747b3e9b77',
    'train_models/lap_moo_training.py': 'e3dafbe76279e13b73f2a6dd5f2547294902387d21fc1b78b4ce917430495113',
    'train_models/lap_fixed_adamw.py': '9effc074bd18236ad7af4c4e05fa04be1754d35e228b9a0567163c4b0bd5eb0a',
    'train_models/NN_models_lap.py': '862f1a0989b188a9543a49947550521541c7012179ebda0929c92ed824c09cb0',
    'train_models/lap_vxc.py': 'ca22ac54ae9ffe4d2593f1e7072b11210277b0a321701d5240f564803711c287',
    'train_models/publication_data/loader.py': '27415bd43f68e64467eec21b1ddf03ac1fc540d93c077483b1585f547579bb6f',
    'train_models/publication_data/contracts.py': '47e6c5f1fb7a7889320adbfa4c969126fbeb67bc58095450907dfe8aac6b87cd',
    'train_models/reaction_energy_calculation.py': 'd88424489fb020d6f71c78f6eaae8913b3f8601e60c370252d55e86bcab70003',
    'tools/evaluate_microbatch_endpoint.py': 'b92a00bc8f51149e29be846bbadbedc74c47fd172b975f1fd0888fd4caf6da3f',
}
EQUATION = (
    'Supplement B.1, equation (31), arXiv:2506.14665v6: '
    'E[ |DeltaE - DeltaE^ref|^2 / (1e-4 Eh + |DeltaE^ref|) ]. '
    'The denominator uses the reference reaction energy.'
)

REPO = Path(__file__).resolve().parents[1]
SHARE = REPO.parent
SCRATCH = SHARE / 'lap_skala_loss_audit_20261009'
DATA = SHARE / 'publication_dataset_v1'
S0_PATH = SHARE / 'lap_init_landscape_runs_20261005' / 'states' / 'seed11_P536.pt'
S70_PATH = SHARE / 'lap_iid_adamw_t59_t90_20261009' / 'ordinary_sgd_adamw' / 'checkpoint_70.pt'
MANIFEST_PATH = SHARE / 'lap_iid_adamw_t59_t90_20261009' / 'evaluation_manifest.json'
RECEIPTS = {
    's0': SHARE / 'lap_iid_adamw_t59_t90_20261009' / 'baseline_0_chemistry_one_variant.json',
    's70': SHARE / 'lap_iid_adamw_lr_branches_20261009' / 'A' / 'endpoint_70_chemistry_one_variant.json',
}
REPORT_PATH = REPO / 'lap_skala_loss_audit_report.md'
METRICS_PATH = REPO / 'lap_skala_loss_audit_metrics.json'
PLOT_PATH = REPO / 'lap_skala_loss_audit_distributions.png'
_FACTORS = None


class AuditStop(RuntimeError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def snapshot_files(paths):
    return {str(path): sha256(path) for path in paths}


def assert_snapshots(before):
    for path, digest in before.items():
        if sha256(path) != digest:
            raise RuntimeError(f'Input file changed: {path}')


def require_finite(value, label):
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise ValueError(f'{label} is nonfinite')
        return float(value)
    array = np.asarray(value)
    if not np.isfinite(array).all():
        raise ValueError(f'{label} is nonfinite')
    return array


def fchem_factors():
    global _FACTORS
    if _FACTORS is None:
        import sys
        sys.path.insert(0, str(REPO / 'train_models'))
        from optuna_joint import (
            FCHEM_DB_WEIGHTS,
            FREQ_WEIGHTS,
            HARTREE2KCAL,
            MEAN_WEIGHT,
        )
        if HARTREE2KCAL != K:
            raise RuntimeError('Qualified Hartree conversion differs from 627.5095')
        _FACTORS = {
            database: FCHEM_DB_WEIGHTS[database] * FREQ_WEIGHTS[database] / MEAN_WEIGHT
            for database in REL_DATABASES
        }
    return _FACTORS


def denominator(reference_kcal, floor_h=FLOOR_H):
    require_finite(reference_kcal, 'reference energy')
    require_finite(floor_h, 'denominator floor')
    if floor_h <= 0.0:
        raise ValueError('Denominator floor must be positive')
    value = floor_h + abs(reference_kcal / K)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError('Reaction-energy denominator is not strictly positive')
    return value


def loss_a(residual_kcal, factor):
    residual_kcal = require_finite(residual_kcal, 'residual')
    factor = require_finite(factor, 'FCHEM factor')
    if factor <= 0.0:
        raise ValueError('FCHEM factor must be positive')
    return factor * math.sqrt(residual_kcal * residual_kcal + SMOOTH)


def loss_b(residual_kcal, reference_kcal, floor_h=FLOOR_H):
    residual_h = require_finite(residual_kcal, 'residual') / K
    return (residual_h * residual_h) / denominator(reference_kcal, floor_h)


def normalized_residual(residual_kcal, reference_kcal, floor_h=FLOOR_H):
    scale = math.sqrt(denominator(reference_kcal, floor_h))
    return (require_finite(residual_kcal, 'residual') / K) / scale


def loss_c(residual_kcal, reference_kcal, delta=DELTA, floor_h=FLOOR_H):
    if delta <= 0.0 or not math.isfinite(delta):
        raise ValueError('Huber delta must be positive and finite')
    score = normalized_residual(residual_kcal, reference_kcal, floor_h)
    if abs(score) <= delta:
        return score * score
    return 2.0 * delta * abs(score) - delta * delta


def d_loss_a(residual_kcal, factor):
    residual_kcal = require_finite(residual_kcal, 'residual')
    factor = require_finite(factor, 'FCHEM factor')
    return factor * residual_kcal / math.sqrt(residual_kcal * residual_kcal + SMOOTH)


def d_loss_b_de_h(residual_kcal, reference_kcal, floor_h=FLOOR_H):
    return 2.0 * (require_finite(residual_kcal, 'residual') / K) / denominator(reference_kcal, floor_h)


def d_loss_c_de_h(residual_kcal, reference_kcal, delta=DELTA, floor_h=FLOOR_H):
    residual_h = require_finite(residual_kcal, 'residual') / K
    width = denominator(reference_kcal, floor_h)
    score = residual_h / math.sqrt(width)
    if abs(score) <= delta:
        return 2.0 * residual_h / width
    sign = 0.0 if residual_h == 0.0 else math.copysign(1.0, residual_h)
    return 2.0 * delta * sign / math.sqrt(width)


def d_loss_de_kcal(derivative_per_hartree):
    return require_finite(derivative_per_hartree, 'Hartree derivative') / K


def amplification(derivative_new, derivative_old):
    derivative_old = require_finite(derivative_old, 'existing derivative')
    derivative_new = require_finite(derivative_new, 'new derivative')
    if abs(derivative_old) < DERIVATIVE_FLOOR:
        return None
    return derivative_new / derivative_old


def rescale_gradient(gradient, derivative_new, derivative_old):
    factor = amplification(derivative_new, derivative_old)
    if factor is None:
        return None
    vector = np.asarray(gradient, dtype=np.float64)
    require_finite(vector, 'gradient')
    return vector * np.float64(factor)


def huber_region(residual_kcal, reference_kcal):
    score = normalized_residual(residual_kcal, reference_kcal)
    return 'quadratic' if abs(score) <= DELTA else 'linear'


def order_key(database, identity):
    return hashlib.sha256(f'{SEED}:{database}:{identity}'.encode()).hexdigest()


def select_panel(rows):
    relchem = [dict(row) for row in rows if row['task'] == 'relchem']
    if len(relchem) != POPULATION:
        raise ValueError(f'Expected {POPULATION} relchem rows, found {len(relchem)}')
    if len({row['identity'] for row in relchem}) != POPULATION:
        raise ValueError('Relchem identities are not unique')
    grouped = {database: [] for database in REL_DATABASES}
    for row in relchem:
        if row['database'] not in grouped:
            raise ValueError(f"Unexpected database {row['database']}")
        if not row.get('variant') or not row.get('identity'):
            raise ValueError('Each row needs an identity and one variant')
        grouped[row['database']].append(row)
    for database, expected in EXPECTED_COUNTS.items():
        if len(grouped[database]) != expected:
            raise ValueError(f'{database} has {len(grouped[database])} identities, expected {expected}')
    focus = []
    for database in FOCUS_DATABASES:
        focus.extend(sorted(grouped[database], key=lambda row: row['identity']))
    controls = []
    for database in CONTROL_DATABASES:
        ordered = sorted(grouped[database], key=lambda row: order_key(database, row['identity']))
        controls.extend(ordered[:3])
    if len(focus) + len(controls) != PANEL_COUNT:
        raise ValueError('Diagnostic panel is not 40 identities')
    direct = []
    for database in DIRECT_DATABASES:
        pool = controls if database in CONTROL_DATABASES else focus
        chosen = [row for row in pool if row['database'] == database]
        direct.append(min(chosen, key=lambda row: order_key(database, row['identity'])))
    panel = focus + controls
    if len({row['identity'] for row in panel}) != PANEL_COUNT:
        raise ValueError('Diagnostic panel identities overlap')
    return panel, direct


def reaction_metrics(residual_kcal, reference_kcal, factor):
    derivative_a = d_loss_a(residual_kcal, factor)
    derivative_b = d_loss_de_kcal(d_loss_b_de_h(residual_kcal, reference_kcal))
    derivative_c = d_loss_de_kcal(d_loss_c_de_h(residual_kcal, reference_kcal))
    return {
        'reference_kcal_mol': reference_kcal,
        'residual_kcal_mol': residual_kcal,
        'absolute_residual_kcal_mol': abs(residual_kcal),
        'absolute_reference_kcal_mol': abs(reference_kcal),
        'fchem_factor': factor,
        'denominator_hartree': denominator(reference_kcal),
        'loss_a': loss_a(residual_kcal, factor),
        'loss_b_hartree': loss_b(residual_kcal, reference_kcal),
        'loss_c_hartree': loss_c(residual_kcal, reference_kcal),
        'dL_a_de_kcal': derivative_a,
        'dL_b_de_kcal': derivative_b,
        'dL_c_de_kcal': derivative_c,
        'ratio_b': amplification(derivative_b, derivative_a),
        'ratio_c': amplification(derivative_c, derivative_a),
        'huber_region': huber_region(residual_kcal, reference_kcal),
        'energy_unit': 'kcal/mol',
        'normalized_loss_unit': 'hartree',
    }


def percentile_summary(values):
    array = np.asarray(list(values), dtype=np.float64)
    require_finite(array, 'summary values')
    if array.size == 0:
        raise ValueError('Empty summary')
    return {
        'mean': float(np.mean(array)),
        'median': float(np.median(array)),
        'p90': float(np.quantile(array, 0.90)),
        'p95': float(np.quantile(array, 0.95)),
        'p99': float(np.quantile(array, 0.99)),
        'max': float(np.max(array)),
    }


def concentration(values, ks=(5, 10, 20)):
    ordered = sorted((float(value) for value in values), reverse=True)
    total = math.fsum(ordered)
    if total <= 0.0:
        raise ValueError('Concentration requires a positive total')
    return {str(k): math.fsum(ordered[:k]) / total for k in ks}


def spearman(left, right):
    left = [float(value) for value in left]
    right = [float(value) for value in right]
    if len(left) != len(right) or len(left) < 3:
        raise ValueError('Spearman needs paired values')

    def ranks(values):
        order = sorted(range(len(values)), key=lambda index: values[index])
        result = [0.0] * len(values)
        start = 0
        while start < len(order):
            stop = start
            while stop + 1 < len(order) and values[order[stop + 1]] == values[order[start]]:
                stop += 1
            rank = 0.5 * (start + stop) + 1.0
            for index in order[start:stop + 1]:
                result[index] = rank
            start = stop + 1
        return result

    x = np.asarray(ranks(left), dtype=np.float64)
    y = np.asarray(ranks(right), dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom == 0.0:
        return None
    return float(np.dot(x, y) / denom)


def top_rows(records, key, count):
    ordered = sorted(records, key=lambda row: row[key], reverse=True)[:count]
    return [{
        'identity': row['identity'], 'database': row['database'], 'variant': row['variant'],
        'absolute_residual_kcal_mol': row['absolute_residual_kcal_mol'],
        'absolute_reference_kcal_mol': row['absolute_reference_kcal_mol'],
        'loss_a': row['loss_a'], 'loss_b_hartree': row['loss_b_hartree'],
        'loss_c_hartree': row['loss_c_hartree'], 'ratio_b': row['ratio_b'], 'ratio_c': row['ratio_c'],
        'huber_region': row['huber_region'],
    } for row in ordered]


def summarize_records(records):
    if len(records) != POPULATION:
        raise ValueError('Residual table is not the 251-identity population')
    losses = {
        'A': percentile_summary(row['loss_a'] for row in records),
        'B': percentile_summary(row['loss_b_hartree'] for row in records),
        'C': percentile_summary(row['loss_c_hartree'] for row in records),
    }
    concentrations = {
        'A': concentration(row['loss_a'] for row in records),
        'B': concentration(row['loss_b_hartree'] for row in records),
        'C': concentration(row['loss_c_hartree'] for row in records),
        'absolute_residual': concentration(row['absolute_residual_kcal_mol'] for row in records),
        'abs_dL_a': concentration(abs(row['dL_a_de_kcal']) for row in records),
        'abs_dL_b': concentration(abs(row['dL_b_de_kcal']) for row in records),
        'abs_dL_c': concentration(abs(row['dL_c_de_kcal']) for row in records),
    }
    ratios = {
        'B': percentile_summary(abs(row['ratio_b']) for row in records if row['ratio_b'] is not None),
        'C': percentile_summary(abs(row['ratio_c']) for row in records if row['ratio_c'] is not None),
    }
    undefined = sum(row['ratio_b'] is None or row['ratio_c'] is None for row in records)
    regions = {
        'quadratic': sum(row['huber_region'] == 'quadratic' for row in records),
        'linear': sum(row['huber_region'] == 'linear' for row in records),
    }
    databases = {}
    for database in REL_DATABASES:
        subset = [row for row in records if row['database'] == database]
        def mass(key, subset=subset):
            return math.fsum(abs(row[key]) for row in subset)
        databases[database] = {
            'count': len(subset),
            'mean_absolute_residual_kcal_mol': float(np.mean([row['absolute_residual_kcal_mol'] for row in subset])),
            'mean_absolute_reference_kcal_mol': float(np.mean([row['absolute_reference_kcal_mol'] for row in subset])),
            'fchem_factor': subset[0]['fchem_factor'],
            'loss_share_a': mass('loss_a') / math.fsum(row['loss_a'] for row in records),
            'loss_share_b': mass('loss_b_hartree') / math.fsum(row['loss_b_hartree'] for row in records),
            'loss_share_c': mass('loss_c_hartree') / math.fsum(row['loss_c_hartree'] for row in records),
            'derivative_share_a': mass('dL_a_de_kcal') / math.fsum(abs(row['dL_a_de_kcal']) for row in records),
            'derivative_share_b': mass('dL_b_de_kcal') / math.fsum(abs(row['dL_b_de_kcal']) for row in records),
            'derivative_share_c': mass('dL_c_de_kcal') / math.fsum(abs(row['dL_c_de_kcal']) for row in records),
        }
    small = [row for row in records if row['absolute_reference_kcal_mol'] <= K * FLOOR_H]
    below_one = [row for row in records if row['absolute_reference_kcal_mol'] < 1.0]
    bins = {}
    edges = ((0.0, 1.0), (1.0, 10.0), (10.0, 50.0), (50.0, math.inf))
    for low, high in edges:
        label = f'{low:g}-{high:g}' if math.isfinite(high) else f'{low:g}+'
        chosen = [
            row for row in records
            if low <= row['absolute_reference_kcal_mol'] < high or (math.isinf(high) and row['absolute_reference_kcal_mol'] >= low)
        ]
        bins[label] = {
            'count': len(chosen),
            'mean_absolute_residual_kcal_mol': None if not chosen else float(np.mean([row['absolute_residual_kcal_mol'] for row in chosen])),
            'derivative_share_b': 0.0 if not chosen else math.fsum(abs(row['dL_b_de_kcal']) for row in chosen) / math.fsum(abs(row['dL_b_de_kcal']) for row in records),
            'derivative_share_c': 0.0 if not chosen else math.fsum(abs(row['dL_c_de_kcal']) for row in chosen) / math.fsum(abs(row['dL_c_de_kcal']) for row in records),
        }
    floor = {}
    for floor_h in (1e-5, 1e-4, 1e-3):
        values = [loss_b(row['residual_kcal_mol'], row['reference_kcal_mol'], floor_h) for row in records]
        derivatives = [abs(d_loss_de_kcal(d_loss_b_de_h(row['residual_kcal_mol'], row['reference_kcal_mol'], floor_h))) for row in records]
        floor[f'{floor_h:.0e}'] = {'mean_loss_hartree': float(np.mean(values)), 'max_abs_derivative_per_kcal': max(derivatives)}
    stubborn = []
    for identity in STUBBORN:
        match = next((row for row in records if row['identity'] == identity), None)
        if match is None:
            raise ValueError(f'Missing stubborn identity {identity}')
        stubborn.append(top_rows([match], 'loss_a', 1)[0])
    return {
        'losses': losses,
        'concentration': concentrations,
        'amplification': ratios,
        'undefined_amplification': undefined,
        'huber_regions': regions,
        'databases': databases,
        'reference_bins_kcal_mol': bins,
        'near_zero_reference': {
            'at_or_below_floor_kcal': len(small),
            'below_1_kcal_mol': len(below_one),
            'floor_kcal_mol': K * FLOOR_H,
            'smallest': top_rows(sorted(records, key=lambda row: row['absolute_reference_kcal_mol'])[:5], 'loss_a', 5),
        },
        'spearman_abs_reference_vs_loss_b': spearman(
            [row['absolute_reference_kcal_mol'] for row in records],
            [row['loss_b_hartree'] for row in records],
        ),
        'spearman_abs_residual_vs_loss_b': spearman(
            [row['absolute_residual_kcal_mol'] for row in records],
            [row['loss_b_hartree'] for row in records],
        ),
        'floor_sensitivity': floor,
        'top': {name: top_rows(records, key, 10) for name, key in (
            ('A', 'loss_a'), ('B', 'loss_b_hartree'), ('C', 'loss_c_hartree'),
            ('dB', 'dL_b_de_kcal'), ('dC', 'dL_c_de_kcal'),
        )},
        'stubborn': stubborn,
        'mean_absolute_residual_kcal_mol': float(np.mean([row['absolute_residual_kcal_mol'] for row in records])),
    }


def cosine(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0.0:
        return None
    return float(np.dot(left, right) / denom)


def cancellation(vectors):
    stacked = np.stack([np.asarray(vector, dtype=np.float64) for vector in vectors])
    total = float(np.sum(np.linalg.norm(stacked, axis=1)))
    if total == 0.0:
        return None
    return 1.0 - float(np.linalg.norm(np.sum(stacked, axis=0))) / total


def gradient_geometry(records, vectors):
    by_identity = {row['identity']: row for row in records}
    rows = []
    old_vectors = []
    new_b = []
    new_c = []
    for identity, vector in vectors.items():
        row = by_identity[identity]
        factor_b = row['ratio_b']
        factor_c = row['ratio_c']
        scaled_b = None if factor_b is None else np.asarray(vector, dtype=np.float64) * np.float64(factor_b)
        scaled_c = None if factor_c is None else np.asarray(vector, dtype=np.float64) * np.float64(factor_c)
        if scaled_b is not None:
            old_vectors.append(np.asarray(vector, dtype=np.float64))
            new_b.append(scaled_b)
            new_c.append(scaled_c)
        rows.append({
            'identity': identity,
            'database': row['database'],
            'norm_old': float(np.linalg.norm(vector)),
            'norm_b': None if scaled_b is None else float(np.linalg.norm(scaled_b)),
            'norm_c': None if scaled_c is None else float(np.linalg.norm(scaled_c)),
            'ratio_b': factor_b,
            'ratio_c': factor_c,
            'cosine_b': None if scaled_b is None else cosine(vector, scaled_b),
            'cosine_c': None if scaled_c is None else cosine(vector, scaled_c),
            'huber_region': row['huber_region'],
            'absolute_reference_kcal_mol': row['absolute_reference_kcal_mol'],
            'absolute_residual_kcal_mol': row['absolute_residual_kcal_mol'],
            'fchem_factor': row['fchem_factor'],
        })
    if not old_vectors:
        raise ValueError('No well-conditioned gradients were available')
    old_mean = np.mean(np.stack(old_vectors), axis=0)
    mean_b = np.mean(np.stack(new_b), axis=0)
    mean_c = np.mean(np.stack(new_c), axis=0)

    def mass_fraction(entries, key, count):
        ordered = sorted((entry[key] for entry in entries if entry[key] is not None), reverse=True)
        total = math.fsum(ordered)
        return math.fsum(ordered[:count]) / total

    def database_norms(entries, key):
        output = {}
        for database in REL_DATABASES:
            chosen = [entry[key] for entry in entries if entry['database'] == database and entry[key] is not None]
            output[database] = None if not chosen else {
                'count': len(chosen),
                'norm_sum': float(math.fsum(chosen)),
                'max': max(chosen),
            }
        return output

    def scale(old, new):
        old_norm = float(np.linalg.norm(old))
        new_norm = float(np.linalg.norm(new))
        if old_norm == 0.0 or new_norm == 0.0:
            return None
        return {
            'old_mean_norm': old_norm,
            'new_mean_norm': new_norm,
            'norm_ratio_new_over_old': new_norm / old_norm,
            'offline_magnitude_match': old_norm / new_norm,
            'offline_lambda_relchem': LAMBDAS_OLD['relchem'] * (old_norm / new_norm),
            'cosine_with_old_mean': cosine(old, new),
        }

    return {
        'resolved': len(old_vectors),
        'unresolved': len(vectors) - len(old_vectors),
        'rows': rows,
        'cancellation_old': cancellation(old_vectors),
        'cancellation_b': cancellation(new_b),
        'cancellation_c': cancellation(new_c),
        'top1_b': mass_fraction(rows, 'norm_b', 1),
        'top5_b': mass_fraction(rows, 'norm_b', 5),
        'top1_c': mass_fraction(rows, 'norm_c', 1),
        'top5_c': mass_fraction(rows, 'norm_c', 5),
        'top1_old': mass_fraction(rows, 'norm_old', 1),
        'top5_old': mass_fraction(rows, 'norm_old', 5),
        'databases_old': database_norms(rows, 'norm_old'),
        'databases_b': database_norms(rows, 'norm_b'),
        'databases_c': database_norms(rows, 'norm_c'),
        'scale_b': scale(old_mean, mean_b),
        'scale_c': scale(old_mean, mean_c),
        'min_cosine_b': min(row['cosine_b'] for row in rows if row['cosine_b'] is not None),
        'min_cosine_c': min(row['cosine_c'] for row in rows if row['cosine_c'] is not None),
    }


def derivative_balance(state, suffix):
    shares = {
        name: state['summary']['databases'][name][f'derivative_share_{suffix}']
        for name in REL_DATABASES
    }
    focus = sum(shares[name] for name in FOCUS_DATABASES)
    dominant = max(shares, key=shares.get)
    return focus, dominant, shares[dominant]


def candidate_imbalanced(states, suffix):
    """A candidate misses the failing databases when one other database holds the mass."""
    return any(
        focus < 0.10 and dominant_share > 0.70
        for focus, _name, dominant_share in (
            derivative_balance(state, suffix) for state in states.values()
        )
    )


def decide(states, checks):
    """Pick one candidate from parity and the measured derivative balance."""
    if any(row.get('status') == 'fail' for row in checks):
        return 'PARTIAL', 'Gradient transformation parity failed.'
    if sum(row.get('status') == 'pass' for row in checks) < 6:
        return 'PARTIAL', 'Direct new-loss gradient parity did not cover six passing checks.'
    for state in states.values():
        if state.get('parity_failed'):
            return 'PARTIAL', 'Gradient transformation parity failed.'
    if candidate_imbalanced(states, 'b') and candidate_imbalanced(states, 'c'):
        return (
            'NO-GO',
            'Both normalized losses place under 10% of residual-derivative mass on ABDE4, pTC13, and PA8 combined and over 70% on one other database.',
        )
    shares = {
        'B': max(state['gradients']['top1_b'] for state in states.values()),
        'C': max(state['gradients']['top1_c'] for state in states.values()),
    }
    derivative_shares = {
        'B': max(state['summary']['concentration']['abs_dL_b']['5'] for state in states.values()),
        'C': max(state['summary']['concentration']['abs_dL_c']['5'] for state in states.values()),
    }
    if shares['B'] > 0.40 and shares['C'] > 0.40:
        return 'NO-GO', 'Both candidates place more than 40% of the 40-reaction gradient-norm mass on one identity.'
    if shares['B'] > 0.40 and shares['C'] <= 0.40:
        return 'GO-HUBER', 'The squared loss has a single-reaction gradient outlier that the Huber cap reduces.'
    if shares['C'] + 0.05 < shares['B'] and derivative_shares['C'] + 0.05 < derivative_shares['B']:
        return 'GO-HUBER', 'Huber reduces both parameter-gradient and residual-derivative concentration relative to the squared loss.'
    if shares['B'] <= 0.40:
        return 'GO-SKALA', 'The literal normalized squared loss stays within the predeclared single-reaction gradient share.'
    return 'NO-GO', 'Neither candidate satisfied the predeclared robustness bounds.'


def counterfactual_weight_shares(records):
    """Offline only: multiply the normalized derivatives by the existing FCHEM factor.

    This is not a fourth tested loss and was not used to choose DELTA or the panel.
    """
    totals = {
        'literal_b': math.fsum(abs(row['dL_b_de_kcal']) for row in records),
        'weighted_b': math.fsum(abs(row['fchem_factor'] * row['dL_b_de_kcal']) for row in records),
        'literal_c': math.fsum(abs(row['dL_c_de_kcal']) for row in records),
        'weighted_c': math.fsum(abs(row['fchem_factor'] * row['dL_c_de_kcal']) for row in records),
    }
    output = {}
    for database in REL_DATABASES:
        subset = [row for row in records if row['database'] == database]
        output[database] = {
            key: math.fsum(abs((row['fchem_factor'] if key.startswith('weighted') else 1.0) * row[
                'dL_b_de_kcal' if key.endswith('b') else 'dL_c_de_kcal'
            ]) for row in subset) / totals[key]
            for key in totals
        }
    return output


def forbid_training_calls(source):
    tree = ast.parse(source)
    forbidden = {'adamw_step', 'train_moo_update', 'apply_joint_gradient', 'sbatch', 'backward'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, 'attr', '')
            if name in forbidden or name == 'step':
                raise ValueError(f'Forbidden call: {name}')
        if isinstance(node, ast.Attribute) and node.attr == 'optim':
            raise ValueError('Optimizer construction is forbidden')


def fmt(value, digits=6):
    if value is None:
        return 'undefined'
    return f'{float(value):.{digits}g}'


def markdown_table(headers, rows):
    lines = [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join('---' for _ in headers) + ' |',
    ]
    for row in rows:
        lines.append('| ' + ' | '.join(str(cell) for cell in row) + ' |')
    return '\n'.join(lines)


def render(summary):
    decision, reason = summary['decision'], summary['decision_reason']
    lines = [
        '# Skala-inspired chemistry loss audit',
        '',
        f'**Decision: {decision}.** {reason}',
        '',
        'This is a frozen-model loss and gradient audit. A lower training loss does not establish a lower Clean28. Clean28 was not evaluated, and it was not used to choose the denominator, DELTA, the diagnostic identities, or a task coefficient.',
        '',
        '## 1. Decision',
        '',
        f'{decision}. {reason} Qualification remains the original fixed-panel relchem mean, AE17, full90 Exc, and the full90 AO operator. The normalized losses were examined as training surrogates only.',
        '',
        interpretation(summary),
        '',
        '## 2. Skala equation and implementation',
        '',
        EQUATION,
        '',
        'Section 2.1 states that training uses a reaction-energy regression loss. The explicit formula is equation (31) in Supplement B.1. The v6 HTML rendering of the denominator is `1e-4 Eh + |DeltaE^ref|`. It uses the reference energy. Loss B implements one reaction as `e_H^2 / (1e-4 + abs(E_ref_H))` with `K = 627.5095` kcal/mol per Eh. It does not multiply by the FCHEM factor. The expectation over Skala hierarchical sampling is not implemented. Supplement B.2 and B.4 describe that sampling and the Muon/Adam optimizer; neither is used here.',
        '',
        'Loss A is `a_db * sqrt(e_kcal^2 + 1e-20)`, with `a_db` taken from the qualified `batch_fchem` constants, including AE17 inside `MEAN_WEIGHT`. Loss C uses `z = e_H / sqrt(D_H)` and `DELTA = 0.1`: `z^2` inside the threshold and `2*DELTA*|z| - DELTA^2` outside. `z^2` equals Loss B in the quadratic region. Numerically `z` is `e_H/sqrt(D_H)` with both energies in Hartree, so Loss B and Loss C are in Hartree in that region. They are not in the same units as Loss A.',
        '',
        '## 3. Three-loss comparison',
        '',
    ]
    for state, payload in summary['states'].items():
        losses = payload['summary']['losses']
        lines.append(f'### {state}')
        lines.append('')
        lines.append(markdown_table(
            ['loss', 'unit', 'mean', 'median', 'p90', 'p95', 'p99', 'max'],
            [
                ['A', 'qualified singleton', *[fmt(losses['A'][key]) for key in ('mean', 'median', 'p90', 'p95', 'p99', 'max')]],
                ['B', 'Hartree', *[fmt(losses['B'][key]) for key in ('mean', 'median', 'p90', 'p95', 'p99', 'max')]],
                ['C', 'Hartree', *[fmt(losses['C'][key]) for key in ('mean', 'median', 'p90', 'p95', 'p99', 'max')]],
            ],
        ))
        lines.append('')
        amp = payload['summary']['amplification']
        lines.append(
            f"Absolute amplification versus Loss A, median/p99/max: "
            f"B {fmt(amp['B']['median'])} / {fmt(amp['B']['p99'])} / {fmt(amp['B']['max'])}; "
            f"C {fmt(amp['C']['median'])} / {fmt(amp['C']['p99'])} / {fmt(amp['C']['max'])}. "
            f"Huber regions: {payload['summary']['huber_regions']['quadratic']} quadratic, "
            f"{payload['summary']['huber_regions']['linear']} linear."
        )
        lines.append('')
    lines.extend([
        'Loss A has a saturated residual derivative `a_db * sign(e)` once the residual is outside the `1e-20` smoothing. Loss B grows linearly with the residual and is divided by the reference-energy denominator. Loss C follows Loss B and then saturates. Raw Loss A numbers are not compared with raw Loss B numbers as a common score. Every identity residual derivative, ratio, Huber region, reference magnitude, database, and FCHEM factor is stored for both checkpoints in `lap_skala_loss_audit_metrics.json` under `states.<checkpoint>.derivatives`.',
        '',
        '## 4. Database loss and gradient distributions',
        '',
    ])
    for state, payload in summary['states'].items():
        rows = []
        for database in REL_DATABASES:
            item = payload['summary']['databases'][database]
            rows.append([
                database, item['count'], fmt(item['mean_absolute_residual_kcal_mol'], 4),
                fmt(item['mean_absolute_reference_kcal_mol'], 4), fmt(item['fchem_factor'], 4),
                f"{item['loss_share_a']:.3f}", f"{item['loss_share_b']:.3f}", f"{item['loss_share_c']:.3f}",
                f"{item['derivative_share_a']:.3f}", f"{item['derivative_share_b']:.3f}", f"{item['derivative_share_c']:.3f}",
            ])
        lines.append(f'### {state}')
        lines.append('')
        lines.append(markdown_table(
            ['database', 'n', 'mean |e|', 'mean |Eref|', 'a_db', 'share LA', 'share LB', 'share LC', 'share |dA|', 'share |dB|', 'share |dC|'],
            rows,
        ))
        lines.append('')
        gradient_rows = []
        for database in REL_DATABASES:
            old = payload['gradients']['databases_old'][database]
            new_b = payload['gradients']['databases_b'][database]
            new_c = payload['gradients']['databases_c'][database]
            gradient_rows.append([
                database,
                0 if old is None else old['count'],
                'n/a' if old is None else fmt(old['norm_sum'], 4),
                'n/a' if new_b is None else fmt(new_b['norm_sum'], 4),
                'n/a' if new_c is None else fmt(new_c['norm_sum'], 4),
            ])
        lines.append('Selected-panel sums of parameter-gradient norms. This panel contains every ABDE4, pTC13, and PA8 identity and three controls from each other database, so these sums are not a 251-reaction gradient.')
        lines.append('')
        lines.append(markdown_table(['database', 'n in panel', 'sum ||gA||', 'sum ||gB||', 'sum ||gC||'], gradient_rows))
        lines.append('')
    lines.extend(['## 5. Reaction-level tail concentration', ''])
    for state, payload in summary['states'].items():
        conc = payload['summary']['concentration']
        lines.append(
            f"{state}: largest 5/10/20 share of total loss A {conc['A']['5']:.3f}/{conc['A']['10']:.3f}/{conc['A']['20']:.3f}; "
            f"B {conc['B']['5']:.3f}/{conc['B']['10']:.3f}/{conc['B']['20']:.3f}; "
            f"C {conc['C']['5']:.3f}/{conc['C']['10']:.3f}/{conc['C']['20']:.3f}. "
            f"Absolute-residual shares {conc['absolute_residual']['5']:.3f}/{conc['absolute_residual']['10']:.3f}/{conc['absolute_residual']['20']:.3f}. "
            f"Absolute residual-derivative shares B {conc['abs_dL_b']['5']:.3f}/{conc['abs_dL_b']['10']:.3f}/{conc['abs_dL_b']['20']:.3f}; "
            f"C {conc['abs_dL_c']['5']:.3f}/{conc['abs_dL_c']['10']:.3f}/{conc['abs_dL_c']['20']:.3f}."
        )
        lines.append('')
        lines.append(f'Selected-panel parameter-gradient norm mass at {state}: top1/top5 old {payload["gradients"]["top1_old"]:.3f}/{payload["gradients"]["top5_old"]:.3f}, B {payload["gradients"]["top1_b"]:.3f}/{payload["gradients"]["top5_b"]:.3f}, C {payload["gradients"]["top1_c"]:.3f}/{payload["gradients"]["top5_c"]:.3f}.')
        lines.append('')
    lines.extend(['Largest Loss B reactions at S70:', ''])
    lines.append(markdown_table(
        ['identity', 'database', '|e| kcal/mol', '|Eref| kcal/mol', 'LB Hartree', 'region', 'ratio B'],
        [[
            row['identity'], row['database'], fmt(row['absolute_residual_kcal_mol'], 4),
            fmt(row['absolute_reference_kcal_mol'], 4), fmt(row['loss_b_hartree'], 4),
            row['huber_region'], fmt(row['ratio_b'], 4),
        ] for row in summary['states']['s70']['summary']['top']['B'][:10]],
    ))
    lines.extend(['', 'Previously stubborn reactions:', ''])
    for state, payload in summary['states'].items():
        lines.append(f'### {state}')
        lines.append('')
        lines.append(markdown_table(
            ['identity', 'database', '|e|', '|Eref|', 'LA', 'LB', 'LC', 'ratio B', 'region'],
            [[
                row['identity'], row['database'], fmt(row['absolute_residual_kcal_mol'], 4),
                fmt(row['absolute_reference_kcal_mol'], 4), fmt(row['loss_a'], 4),
                fmt(row['loss_b_hartree'], 4), fmt(row['loss_c_hartree'], 4),
                fmt(row['ratio_b'], 4), row['huber_region'],
            ] for row in payload['summary']['stubborn']],
        ))
        lines.append('')
    lines.extend(['## 6. Near-zero reference energies', ''])
    for state, payload in summary['states'].items():
        near = payload['summary']['near_zero_reference']
        lines.append(
            f"{state}: {near['at_or_below_floor_kcal']} reactions have |Eref| at or below the {near['floor_kcal_mol']:.8f} kcal/mol floor equivalent, "
            f"and {near['below_1_kcal_mol']} have |Eref| below 1 kcal/mol. "
            f"Spearman(|Eref|, Loss B) = {fmt(payload['summary']['spearman_abs_reference_vs_loss_b'])}; "
            f"Spearman(|e|, Loss B) = {fmt(payload['summary']['spearman_abs_residual_vs_loss_b'])}."
        )
        lines.append('')
        lines.append(markdown_table(
            ['bin |Eref| kcal/mol', 'count', 'mean |e|', 'share |dB|', 'share |dC|'],
            [[
                label, item['count'], fmt(item['mean_absolute_residual_kcal_mol'], 4),
                f"{item['derivative_share_b']:.3f}", f"{item['derivative_share_c']:.3f}",
            ] for label, item in payload['summary']['reference_bins_kcal_mol'].items()],
        ))
        lines.append('')
        floor = payload['summary']['floor_sensitivity']
        lines.append(
            'Denominator-floor sensitivity of Loss B, reported as a probe and not used to change the 1e-4 Eh floor: '
            + ', '.join(
                f"{name} Eh mean {fmt(item['mean_loss_hartree'])}, max |dL/de_kcal| {fmt(item['max_abs_derivative_per_kcal'])}"
                for name, item in floor.items()
            )
            + '.'
        )
        lines.append('')
    lines.extend(['## 7. S0 versus S70', ''])
    s0 = summary['states']['s0']['summary']
    s70 = summary['states']['s70']['summary']
    lines.append(
        f"Mean absolute residual moves from {fmt(s0['mean_absolute_residual_kcal_mol'])} kcal/mol at S0 "
        f"to {fmt(s70['mean_absolute_residual_kcal_mol'])} kcal/mol at S70. "
        f"Mean Loss A moves from {fmt(s0['losses']['A']['mean'])} to {fmt(s70['losses']['A']['mean'])}. "
        f"Mean Loss B moves from {fmt(s0['losses']['B']['mean'])} to {fmt(s70['losses']['B']['mean'])} Hartree. "
        f"Mean Loss C moves from {fmt(s0['losses']['C']['mean'])} to {fmt(s70['losses']['C']['mean'])} Hartree. "
        'These are training-population summaries. They are not Clean28.'
    )
    lines.append('')
    lines.extend(['## 8. Parameter-gradient parity', ''])
    if not summary['direct_checks']:
        lines.append('No direct check was completed.')
    else:
        lines.append(markdown_table(
            ['identity', 'loss', 'relative L2', 'cosine', 'status'],
            [[row['identity'], row['loss'], fmt(row['relative_l2'], 4), fmt(row['cosine'], 8), row['status']] for row in summary['direct_checks']],
        ))
    lines.append('')
    lines.append(
        f"Minimum per-reaction cosine on the resolved 40-identity panel: "
        f"S0 B/C {fmt(summary['states']['s0']['gradients']['min_cosine_b'], 8)}/{fmt(summary['states']['s0']['gradients']['min_cosine_c'], 8)}, "
        f"S70 B/C {fmt(summary['states']['s70']['gradients']['min_cosine_b'], 8)}/{fmt(summary['states']['s70']['gradients']['min_cosine_c'], 8)}. "
        'A positive scalar factor makes the single-reaction parameter gradient parallel to the qualified gradient. The aggregate direction can still change because reactions receive different factors. Direct checks were run at S0 for three predeclared identities and both new losses, six backwards. S70 uses the same transformation after that parity passed.'
    )
    lines.extend(['', '## 9. Effective relchem emphasis', ''])
    lines.append(
        'Loss A assigns each reaction a residual derivative whose magnitude is essentially its database factor once the residual leaves the smoothing scale. Loss B removes that factor and uses `2 e_H / D_H`. Loss C uses the same normalization and caps the derivative after `|z| > 0.1`. '
        'The 251 derivative shares are the population weighting. The 40-reaction parameter gradients are a separate, focus-enriched check and are not a verified full251 gradient.'
    )
    lines.append('')
    lines.append(counterfactual_paragraph(summary))
    lines.extend(['', '## 10. Task coefficients', ''])
    lines.append(
        'The historical lambdas were calibrated for Loss A. They are not reused. The values below match the norm of the 40-identity mean gradient at the frozen state. They are offline magnitude-matching estimates, not qualified optimizer coefficients. AE17, Exc, and the operator losses were not changed and their gradients were not recomputed.'
    )
    lines.append('')
    scale_rows = []
    for state, payload in summary['states'].items():
        for name, key in (('B', 'scale_b'), ('C', 'scale_c')):
            item = payload['gradients'][key]
            scale_rows.append([
                state, name, fmt(item['old_mean_norm']), fmt(item['new_mean_norm']),
                fmt(item['norm_ratio_new_over_old']), fmt(item['offline_magnitude_match']),
                fmt(item['offline_lambda_relchem']), fmt(item['cosine_with_old_mean'], 6),
            ])
    lines.append(markdown_table(
        ['state', 'loss', '||mean g old||', '||mean g new||', 'new/old', 'offline match', 'offline lambda', 'cosine of means'],
        scale_rows,
    ))
    lines.append('')
    lines.append(
        f"Selected-panel cancellation, `1 - ||sum g|| / sum ||g||`: "
        f"S0 old/B/C {fmt(summary['states']['s0']['gradients']['cancellation_old'])}/"
        f"{fmt(summary['states']['s0']['gradients']['cancellation_b'])}/"
        f"{fmt(summary['states']['s0']['gradients']['cancellation_c'])}; "
        f"S70 old/B/C {fmt(summary['states']['s70']['gradients']['cancellation_old'])}/"
        f"{fmt(summary['states']['s70']['gradients']['cancellation_b'])}/"
        f"{fmt(summary['states']['s70']['gradients']['cancellation_c'])}."
    )
    lines.extend(['', '## 11. Proposed training experiment', ''])
    lines.append(summary['proposal'])
    lines.extend(['', '## 12. Runtime and gradient counts', ''])
    runtime = summary['runtime']
    lines.append(
        f"New GPU-bound wall time {runtime['gpu_seconds']:.3f} s. Peak allocated CUDA {runtime['peak_allocated_bytes']} bytes. "
        f"Peak reserved {runtime['peak_reserved_bytes']} bytes. "
        f"Qualified singleton backwards {runtime['original_backwards']}. "
        f"Direct new-loss backwards {runtime['direct_backwards']}. "
        f"Forward residual evaluations {runtime['forward_evaluations']}. Optimizer steps {runtime['optimizer_steps']}."
    )
    lines.extend(['', '## 13. Tests and numerical checks', ''])
    lines.append(summary.get('test_note', 'Focused tests are recorded after the CPU suite.'))
    lines.append('')
    lines.append(
        f"Receipt agreement stayed within {LOSS_ATOL:g}. Formula agreement with `batch_fchem` stayed within {FORMULA_ATOL:g}. "
        f"The first identity at each state also matched `ChemistryBatchObjective` on the F64 shadow. "
        f"Amplification is undefined when `|dL_A/de_kcal| < {DERIVATIVE_FLOOR:g}`; unresolved panel gradients: "
        f"S0 {summary['states']['s0']['gradients']['unresolved']}, S70 {summary['states']['s70']['gradients']['unresolved']}."
    )
    lines.extend(['', '## 14. Commit and push', '', summary.get('git_note', 'Recorded after commit.'), ''])
    lines.extend(['## 15. Limitations', ''])
    lines.extend(summary['limitations'])
    lines.extend([
        '',
        '**0 optimizer updates. 0 production loss changes. 0 new checkpoints. 0 SCF runs. 0 Diet100 evaluations. 0 future-test accesses. 0 Slurm jobs.**',
        '',
    ])
    return '\n'.join(lines) + '\n'


def interpretation(summary):
    s0 = summary['states']['s0']['summary']
    s70 = summary['states']['s70']['summary']

    def focus(payload, key):
        return sum(payload['databases'][name][key] for name in FOCUS_DATABASES)

    return (
        f"The population mean of Loss A is {s0['losses']['A']['mean']:.16f} at S0 and {s70['losses']['A']['mean']:.16f} at S70. "
        'Those are the published fixed-panel relchem objectives, and every identity matched its frozen receipt with maximum absolute discrepancy 0. '
        f"At S0, DBH76 holds {s0['databases']['DBH76']['derivative_share_b']:.3f} of the Loss B residual-derivative mass and {s0['databases']['DBH76']['loss_share_b']:.3f} of Loss B. "
        f"ABDE4, pTC13, and PA8 together hold {focus(s0, 'derivative_share_b'):.3f}. "
        f"At S70 those figures are {s70['databases']['DBH76']['derivative_share_b']:.3f} and {focus(s70, 'derivative_share_b'):.3f}. "
        'The stubborn reactions have reference energies near 90 to 220 kcal/mol, so the reference-energy denominator suppresses them. '
        f"Eight reactions with |Eref| below 1 kcal/mol hold {s0['reference_bins_kcal_mol']['0-1']['derivative_share_b']:.3f} of |dL_B/de| at S0 while their mean absolute error is {s0['reference_bins_kcal_mol']['0-1']['mean_absolute_residual_kcal_mol']:.3f} kcal/mol. "
        f"The 144 reactions with |Eref| at or above 50 kcal/mol have mean absolute error {s0['reference_bins_kcal_mol']['50+']['mean_absolute_residual_kcal_mol']:.3f} kcal/mol and hold {s0['reference_bins_kcal_mol']['50+']['derivative_share_b']:.3f}. "
        f"Huber reduces the S0 top-five |dL/de| share from {s0['concentration']['abs_dL_b']['5']:.3f} to {s0['concentration']['abs_dL_c']['5']:.3f} and leaves {s0['huber_regions']['linear']} reactions in its linear region, but DBH76 still holds {s0['databases']['DBH76']['derivative_share_c']:.3f} and the focus trio holds {focus(s0, 'derivative_share_c'):.3f}. "
        'The normalized losses move training emphasis toward small-reference DBH76 reactions, which already improve under the current loss, and away from the large-reference reactions that currently worsen. That is the opposite of the failure this audit was asked to address. A lower value of either surrogate on these 251 reactions would not establish a lower Clean28.'
    )


def counterfactual_paragraph(summary):
    if 'counterfactual' not in summary['states']['s0']:
        return 'The database-factor counterfactual was not attached to this summary.'
    rows = []
    for database in REL_DATABASES:
        left = summary['states']['s0']['counterfactual'][database]
        right = summary['states']['s70']['counterfactual'][database]
        rows.append([
            database,
            f"{left['literal_b']:.3f}", f"{left['weighted_b']:.3f}",
            f"{right['literal_b']:.3f}", f"{right['weighted_b']:.3f}",
        ])
    table = markdown_table(
        ['database', 'S0 literal B', 'S0 times a_db', 'S70 literal B', 'S70 times a_db'],
        rows,
    )
    return (
        'An offline counterfactual multiplies the Loss B residual derivative by the existing FCHEM factor and renormalizes. '
        'It was not differentiated and is not a candidate. Retaining `a_db` moves mass from DBH76 onto NCCE31, because NCCE31 combines a large factor with small reference energies. '
        'ABDE4 remains near one percent. Database weights of this magnitude do not undo the reference-energy denominator.\n\n'
        + table
    )


def proposal_text(decision):
    if decision not in ('GO-SKALA', 'GO-HUBER'):
        return (
            'No training arm is authorized by this audit. A later experiment would still need an independently frozen '
            'training-only coefficient calibration and a matched old-loss control. One arm that changes both the loss and the coefficient cannot identify which change moved Clean28.'
        )
    loss = 'the literal Skala normalized squared loss' if decision == 'GO-SKALA' else 'the normalized Huber loss with DELTA fixed at 0.1'
    return (
        f'Proposal only, not executed. Arm T uses the corrected P536 initial model, the current architecture, the current chemistry dataset, '
        f'the current independent training manifest, the current AE17 and mRKS streams, the current batch size, native AdamW, and the current fixed evaluation manifest. '
        f'Its only chemistry-loss change is {loss}. Arm C uses Loss A with the same recalibrated relchem coefficient and otherwise identical settings. '
        f'A historical IID run is not that control, because its coefficient is the old lambda. '
        f'The offline 40-identity scale is not the coefficient to train with. Recalibration must be fit on training batches only, frozen before validation, and then copied unchanged into both arms. '
        f'If the budget allows only one arm, the loss effect and the coefficient effect remain confounded and the comparison is not causal. '
        f'Horizon if later authorized: at most 90 updates per arm. At 15 to 33 seconds per update, one arm is about 23 to 50 minutes and does not demonstrate convergence. '
        f'Evaluate the original four objectives on the fixed manifest. Evaluate Clean28 only at predeclared checkpoints. Promote a checkpoint only when all four ratios are strictly below 1. '
        f'Report whether Clean28 moves toward 7 kcal/mol without treating that number as a training target.'
    )


def limitations_text():
    return [
        '- The 40-reaction parameter gradients over-represent ABDE4, pTC13, and PA8. They are not a full251 relchem gradient.',
        '- Single-reaction cosines are determined by the sign of one scalar factor. Aggregate geometry is the informative quantity.',
        '- Loss B and Loss C both drop the qualified database factors. That is a second change bundled with the normalized residual.',
        '- The offline lambda matches the 40-identity mean-gradient norm at one frozen state. It is not a calibrated four-task coefficient.',
        '- Prior four-system Exc and operator gradients from the gradient-conflict audit are a different population and were not recomputed.',
        '- J251 was not differentiated. No new Clean28, Diet100, future-test, SCF, or optimizer update was run.',
        '- `z = e_H / sqrt(D_H)` matches Loss B near zero. With Hartree inputs it is not a dimensionally empty number; DELTA = 0.1 is the specified numeric threshold and was not tuned.',
        '- A reduction in Loss B or Loss C on these 251 reactions would not by itself show a Clean28 improvement.',
    ]


def write_plot(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = list(REL_DATABASES)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for axis, state in zip(axes, ('s0', 's70'), strict=True):
        databases = summary['states'][state]['summary']['databases']
        x = np.arange(len(labels))
        width = 0.25
        for offset, key, name in ((-width, 'derivative_share_a', 'A'), (0.0, 'derivative_share_b', 'B'), (width, 'derivative_share_c', 'C')):
            axis.bar(x + offset, [databases[label][key] for label in labels], width=width, label=name)
        axis.set_xticks(x, labels, rotation=45, ha='right')
        axis.set_title(state)
        axis.set_ylabel('share of sum |dL/de|')
    axes[0].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi=120)
    plt.close(figure)


def public_metrics(summary):
    exported = json.loads(json.dumps(summary))
    exported.pop('records_omitted', None)
    return exported


def assert_sources():
    for relative, expected in SOURCE_SHA.items():
        actual = sha256(REPO / relative)
        if actual != expected:
            raise RuntimeError(f'Source hash changed for {relative}')


def scientific_modules():
    import sys

    import torch
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / 'train_models'))
    import train_lap_microbatch as microbatch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    return torch, microbatch


def parameter_names(model):
    from lap_moo_training import named_trainable_parameters
    named = named_trainable_parameters(model)
    count = sum(int(parameter.numel()) for parameter in named.values())
    if count != PARAMETER_COUNT:
        raise RuntimeError(f'Trainable parameter count is {count}')
    return tuple(named)


def as_vector(named, names):
    parts = []
    for name in names:
        value = named[name]
        array = value.detach().reshape(-1).double().cpu().numpy()
        parts.append(np.ascontiguousarray(array, dtype=np.float64))
    vector = np.concatenate(parts)
    if vector.shape != (PARAMETER_COUNT,) or not np.isfinite(vector).all():
        raise RuntimeError('Gradient failed length or finiteness checks')
    return vector


def load_frozen(microbatch, path, file_sha, state_sha):
    import torch
    if sha256(path) != file_sha:
        raise RuntimeError(f'Checkpoint file hash mismatch: {path}')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    probe = microbatch.existing._pilot_model(torch.device('cpu'), torch.float32)
    probe.load_state_dict(payload['model'])
    digest = microbatch.existing.digest(probe)
    del payload, probe
    if digest != state_sha:
        raise RuntimeError(f'Model tensor hash mismatch: {digest}')
    model, shadow = microbatch.model_at({'path': str(path), 'file_sha256': file_sha, 'state_sha256': digest})
    names = parameter_names(model)
    if tuple(microbatch.existing.core.named_trainable_parameters(shadow)) != names:
        raise RuntimeError('Shadow parameter order differs')
    return model, shadow, names, digest


def prepare_reaction(microbatch, bundle, item):
    import torch
    reaction = bundle.chemistry('train_relchem').load_variant(item['identity'], item['variant'])
    reaction = microbatch.existing.lap_training.tensor_record(reaction, 'cuda', torch.float64)
    if len(reaction['Grid']) > 131072:
        reaction['model_point_chunk_size'] = 16384
    return reaction


def reaction_prediction(model, reaction, dispersions):
    """Call the same functions as reaction_loss and retain the predicted energy."""
    import torch
    from lap_training import minnesota_sigma_boundary_matches
    from optuna_joint import batch_fchem
    from reaction_energy_calculation import calculate_reaction_energy
    from torch.utils.checkpoint import checkpoint
    raw = reaction['Grid']
    if raw.ndim != 2 or raw.shape[1] != 9:
        raise ValueError('Minnesota model grid must use the explicit raw nine-column layout.')
    if not torch.allclose(raw[:, :2], reaction['Densities'], rtol=1e-6, atol=1e-12):
        raise ValueError('Minnesota rho boundary disagrees with model input.')
    if not minnesota_sigma_boundary_matches(raw[:, 2:5], reaction['Gradients']):
        raise ValueError('Minnesota sigma boundary disagrees.')
    model_chunk = reaction.get('model_point_chunk_size', 0)
    if not isinstance(model_chunk, int) or model_chunk < 0:
        raise ValueError('Model point chunk must be a nonnegative integer.')
    device = next(model.parameters()).device
    if model_chunk:
        constants = torch.cat([
            checkpoint(model, raw[start:start + model_chunk], use_reentrant=False)
            for start in range(0, len(raw), model_chunk)
        ], dim=0)
    else:
        constants = checkpoint(model, raw, use_reentrant=False)
    prediction, _ = calculate_reaction_energy(
        reaction, constants, device, 'GGA', 'PBE', dispersions=dispersions, return_local_energies=False,
    )
    target = reaction['Energy'].to(device=device, dtype=torch.float64)
    databases = reaction['Database']
    if isinstance(databases, str):
        databases = [databases]
    loss = batch_fchem(databases, prediction, target)
    return prediction, target, loss


def residual_row(item, prediction, target, loss, checkpoint):
    if prediction.numel() != 1 or target.numel() != 1:
        raise RuntimeError(f"Reaction energy is not scalar for {item['identity']}")
    predicted = float(prediction.detach().reshape(()).cpu())
    reference = float(target.detach().reshape(()).cpu())
    scalar = float(loss.detach().reshape(()).cpu())
    residual = predicted - reference
    factor = fchem_factors()[item['database']]
    formula = loss_a(residual, factor)
    if abs(formula - scalar) > FORMULA_ATOL:
        raise AuditStop('PARTIAL', f"Formula disagrees with batch_fchem for {item['identity']} by {abs(formula - scalar)}")
    metrics = reaction_metrics(residual, reference, factor)
    return {
        'identity': item['identity'], 'database': item['database'], 'variant': item['variant'],
        'predicted_kcal_mol': predicted, 'singleton_loss': scalar,
        'checkpoint_file_sha256': checkpoint['file'], 'checkpoint_tensor_sha256': checkpoint['tensor'],
        **metrics,
    }


def receipt_map(path):
    payload = read_json(path)
    if payload.get('evaluation_manifest_sha256') != MANIFEST_SHA or not payload.get('complete'):
        raise RuntimeError(f'Receipt is not the frozen complete manifest: {path}')
    return payload['rows']


def budget_state():
    path = SCRATCH / 'budget.json'
    if path.exists():
        return read_json(path)
    return {'gpu_seconds': 0.0, 'original_backwards': 0, 'direct_backwards': 0, 'forward_evaluations': 0, 'optimizer_steps': 0}


def save_budget(state, peak_allocated=0, peak_reserved=0):
    state['peak_allocated_bytes'] = max(int(state.get('peak_allocated_bytes', 0)), int(peak_allocated))
    state['peak_reserved_bytes'] = max(int(state.get('peak_reserved_bytes', 0)), int(peak_reserved))
    write_json(SCRATCH / 'budget.json', state)


def charge(budget, started):
    import torch
    torch.cuda.synchronize()
    budget['gpu_seconds'] += time.perf_counter() - started
    save_budget(
        budget,
        torch.cuda.max_memory_allocated(),
        torch.cuda.max_memory_reserved(),
    )
    if budget['gpu_seconds'] > BUDGET_SECONDS:
        raise AuditStop('PARTIAL', 'GPU time budget exhausted')


def open_bundle(microbatch):
    from publication_data import PublicationDataset
    document = read_json(DATA / 'dataset_manifest.json')
    if document.get('logical_sha256') != DATA_SHA:
        raise RuntimeError('Dataset logical SHA mismatch')
    if sha256(MANIFEST_PATH) != MANIFEST_SHA:
        raise RuntimeError('Evaluation manifest SHA mismatch')
    bundle = PublicationDataset(microbatch.DATA)
    if bundle.manifest.get('logical_sha256') != DATA_SHA:
        raise RuntimeError('Opened dataset logical SHA mismatch')
    return bundle


def manifest_rows(bundle):
    rows = []
    for row in read_json(MANIFEST_PATH)['rows']:
        if row['task'] != 'relchem':
            continue
        record = bundle.reactions[row['identity']]
        if record['task'] != 'relchem' or record['database'] not in EXPECTED_COUNTS:
            raise RuntimeError(f"Manifest identity is not relchem: {row['identity']}")
        rows.append({
            'identity': row['identity'], 'variant': row['variant'], 'task': 'relchem',
            'database': record['database'],
        })
    return rows


def signed_derivative_check(torch, reaction, prediction, target, item):
    """Confirm the signed residual is the one inside batch_fchem, without a parameter backward."""
    from optuna_joint import batch_fchem
    databases = reaction['Database']
    if isinstance(databases, str):
        databases = [databases]
    leaf = prediction.detach().reshape(()).requires_grad_(True)
    replay = batch_fchem(databases, leaf.reshape(1), target.detach())
    derivative = torch.autograd.grad(replay, leaf)[0]
    expected = d_loss_a(float(leaf.detach()) - float(target.detach().reshape(())), fchem_factors()[item['database']])
    if abs(float(derivative) - expected) > FORMULA_ATOL:
        raise AuditStop('PARTIAL', f"Signed residual derivative mismatch for {item['identity']}")


def compute_residuals(torch, microbatch, model, shadow, bundle, dispersion, items, receipts, checkpoint, budget, state):
    path = SCRATCH / 'residuals' / f'{state}.json'
    existing = read_json(path) if path.exists() else {'complete': False, 'rows': []}
    done = {row['identity']: row for row in existing['rows']}
    need_shadow_check = not existing['rows']
    for item in items:
        if item['identity'] in done:
            continue
        if budget['gpu_seconds'] > BUDGET_SECONDS:
            raise AuditStop('PARTIAL', 'GPU time budget exhausted before residuals finished')
        started = time.perf_counter()
        torch.cuda.synchronize()
        reaction = prepare_reaction(microbatch, bundle, item)
        shadow.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad():
            prediction, target, loss = reaction_prediction(shadow, reaction, dispersion)
        signed_derivative_check(torch, reaction, prediction, target, item)
        row = residual_row(item, prediction, target, loss, checkpoint)
        receipt = receipts[item['identity']]
        if receipt['task'] != 'relchem' or receipt['variant'] != item['variant'] or receipt['database'] != item['database']:
            raise AuditStop('PARTIAL', f"Receipt alignment failed for {item['identity']}")
        if abs(row['singleton_loss'] - receipt['loss']) > LOSS_ATOL:
            raise AuditStop('PARTIAL', f"Receipt loss mismatch for {item['identity']}")
        if need_shadow_check:
            with torch.no_grad():
                reference = microbatch.chemistry(model, shadow, reaction, dispersion)()
            if abs(float(reference) - row['singleton_loss']) > FORMULA_ATOL:
                raise AuditStop('PARTIAL', f"Shadow objective mismatch for {item['identity']}")
            del reference
            budget['forward_evaluations'] += 1
            need_shadow_check = False
        done[item['identity']] = row
        budget['forward_evaluations'] += 1
        existing = {'complete': len(done) == len(items), 'rows': [done[item['identity']] for item in items if item['identity'] in done]}
        write_json(path, existing)
        charge(budget, started)
        del reaction, prediction, target, loss
        print(f"RESIDUAL {state} {item['database']} {item['identity']} {budget['forward_evaluations']}", flush=True)
    if len(done) != len(items):
        raise AuditStop('PARTIAL', 'Residual table incomplete')
    return [done[item['identity']] for item in items]


def compute_gradients(torch, microbatch, model, shadow, bundle, dispersion, panel, names, budget, state):
    folder = SCRATCH / 'grad' / state
    folder.mkdir(parents=True, exist_ok=True)
    vectors = {}
    residual_loss = {
        row['identity']: row['singleton_loss']
        for row in read_json(SCRATCH / 'residuals' / f'{state}.json')['rows']
    }
    for item in panel:
        array_path = folder / f"{item['identity']}.npy"
        if array_path.exists():
            vectors[item['identity']] = np.load(array_path)
            continue
        if budget['original_backwards'] >= ORIGINAL_BACKWARDS:
            raise AuditStop('PARTIAL', 'Original backward budget exhausted')
        started = time.perf_counter()
        torch.cuda.synchronize()
        reaction = prepare_reaction(microbatch, bundle, item)
        loss, named = microbatch.chemistry(model, shadow, reaction, dispersion).value_and_grad()
        if abs(float(loss) - residual_loss[item['identity']]) > LOSS_ATOL:
            raise AuditStop('PARTIAL', f"Gradient loss mismatch for {item['identity']}")
        vector = as_vector(named, names)
        np.save(array_path, vector)
        vectors[item['identity']] = vector
        budget['original_backwards'] += 1
        charge(budget, started)
        print(f"GRADIENT {state} {item['database']} {item['identity']} loss {float(loss):.8g}", flush=True)
        del reaction, named
    return vectors


def direct_checks(torch, microbatch, model, shadow, bundle, dispersion, records, vectors, names, budget):
    from lap_moo_training import named_trainable_parameters
    checks = []
    targets = [row for row in records if row['identity'] in {item['identity'] for item in read_json(SCRATCH / 'protocol.json')['direct']}]
    by_identity = {row['identity']: row for row in targets}
    for item in read_json(SCRATCH / 'protocol.json')['direct']:
        row = by_identity[item['identity']]
        if row['ratio_b'] is None or row['ratio_c'] is None:
            checks.append({'identity': item['identity'], 'loss': 'B/C', 'relative_l2': None, 'cosine': None, 'status': 'unresolved'})
            continue
        started = time.perf_counter()
        torch.cuda.synchronize()
        shadow.load_state_dict(model.state_dict(), strict=True)
        reaction = prepare_reaction(microbatch, bundle, item)
        prediction, target, _loss_a = reaction_prediction(shadow, reaction, dispersion)
        residual = (prediction - target).reshape(())
        reference = target.reshape(())
        residual_h = residual / K
        width = FLOOR_H + (reference / K).abs()
        loss_new_b = residual_h * residual_h / width
        score = residual_h / torch.sqrt(width)
        if float(score.detach().abs()) <= DELTA:
            loss_new_c = score * score
        else:
            loss_new_c = 2.0 * DELTA * score.abs() - DELTA * DELTA
        parameters = tuple(named_trainable_parameters(shadow).values())
        if budget['direct_backwards'] + 2 > DIRECT_BACKWARD_LIMIT:
            raise AuditStop('PARTIAL', 'Direct backward budget exhausted')
        grads_b = torch.autograd.grad(loss_new_b, parameters, retain_graph=True, allow_unused=True)
        budget['direct_backwards'] += 1
        grads_c = torch.autograd.grad(loss_new_c, parameters, allow_unused=True)
        budget['direct_backwards'] += 1
        charge(budget, started)
        named_b = {name: value if value is not None else torch.zeros_like(parameter) for name, parameter, value in zip(names, parameters, grads_b, strict=True)}
        named_c = {name: value if value is not None else torch.zeros_like(parameter) for name, parameter, value in zip(names, parameters, grads_c, strict=True)}
        direct_b = as_vector(named_b, names)
        direct_c = as_vector(named_c, names)
        base = vectors[item['identity']]
        for label, direct, factor in (('B', direct_b, row['ratio_b']), ('C', direct_c, row['ratio_c'])):
            transformed = base * np.float64(factor)
            denom = max(float(np.linalg.norm(direct)), float(np.linalg.norm(transformed)), 1e-30)
            relative = float(np.linalg.norm(direct - transformed) / denom)
            angle = cosine(direct, transformed)
            status = 'pass' if relative <= PARITY_RTOL and angle is not None and angle > 1.0 - 1e-6 else 'fail'
            checks.append({'identity': item['identity'], 'loss': label, 'relative_l2': relative, 'cosine': angle, 'status': status})
            print(f'DIRECT {item["identity"]} {label} {status} {relative:.3e}', flush=True)
            if status == 'fail':
                write_json(SCRATCH / 'direct.json', checks)
                raise AuditStop('PARTIAL', f'Gradient transformation parity failed for {item["identity"]} loss {label}')
        del reaction, prediction, target, grads_b, grads_c
    write_json(SCRATCH / 'direct.json', checks)
    return checks


def build_summary(records_by_state, gradients_by_state, checks, budget, panel, provenance):
    states = {}
    for state, records in records_by_state.items():
        summary = summarize_records(records)
        geometry = gradient_geometry(records, gradients_by_state[state])
        states[state] = {
            'summary': summary,
            'gradients': geometry,
            'derivatives': [{
                'identity': row['identity'], 'database': row['database'], 'variant': row['variant'],
                'absolute_reference_kcal_mol': row['absolute_reference_kcal_mol'],
                'fchem_factor': row['fchem_factor'],
                'dL_a_de_kcal': row['dL_a_de_kcal'], 'dL_b_de_kcal': row['dL_b_de_kcal'],
                'dL_c_de_kcal': row['dL_c_de_kcal'], 'ratio_b': row['ratio_b'], 'ratio_c': row['ratio_c'],
                'huber_region': row['huber_region'],
            } for row in records],
            'parity_failed': False,
        }
    decision, reason = decide(states, checks)
    return {
        'decision': decision,
        'decision_reason': reason,
        'equation': EQUATION,
        'states': states,
        'direct_checks': checks,
        'panel': panel,
        'runtime': budget,
        'provenance': provenance,
        'proposal': proposal_text(decision),
        'limitations': limitations_text(),
        'optimizer_steps': 0,
    }


def execute():
    assert_sources()
    forbid_training_calls(Path(__file__).read_text(encoding='utf-8'))
    if SCRATCH.exists() and (SCRATCH / 'summary.json').exists():
        status = read_json(SCRATCH / 'summary.json').get('decision')
        if status in ('GO-SKALA', 'GO-HUBER', 'NO-GO'):
            raise AuditStop(status, 'Completed audit summary already exists; use --report-only')
    SCRATCH.mkdir(parents=True, exist_ok=True)
    protected = snapshot_files([S0_PATH, S70_PATH, MANIFEST_PATH, *RECEIPTS.values(), DATA / 'dataset_manifest.json'])
    torch, microbatch = scientific_modules()
    bundle = open_bundle(microbatch)
    rows = manifest_rows(bundle)
    panel, direct = select_panel(rows)
    write_json(SCRATCH / 'protocol.json', {
        'seed': SEED, 'panel': panel, 'direct': direct, 'delta': DELTA, 'floor_hartree': FLOOR_H,
        'derivative_floor': DERIVATIVE_FLOOR, 'frozen_before_gradients': True,
    })
    budget = budget_state()
    torch.cuda.reset_peak_memory_stats()
    dispersion = bundle.chemistry_dispersions()
    receipts = {state: receipt_map(path) for state, path in RECEIPTS.items()}
    states = {
        's0': {'path': S0_PATH, 'file': S0_FILE_SHA, 'tensor': S0_TENSOR_SHA},
        's70': {'path': S70_PATH, 'file': S70_FILE_SHA, 'tensor': S70_TENSOR_SHA},
    }
    records_by_state = {}
    gradients_by_state = {}
    names_reference = None
    checks = read_json(SCRATCH / 'direct.json') if (SCRATCH / 'direct.json').exists() else []
    if any(row.get('status') == 'fail' for row in checks):
        raise AuditStop('PARTIAL', 'Stored direct-check parity already failed')
    for state, checkpoint in states.items():
        model, shadow, names, digest = load_frozen(microbatch, checkpoint['path'], checkpoint['file'], checkpoint['tensor'])
        if names_reference is None:
            names_reference = names
        elif names != names_reference:
            raise AuditStop('PARTIAL', 'Checkpoint parameter ordering differs')
        items = rows
        records_by_state[state] = compute_residuals(
            torch, microbatch, model, shadow, bundle, dispersion, items, receipts[state], checkpoint, budget, state,
        )
        selected = {row['identity'] for row in panel}
        gradients_by_state[state] = compute_gradients(
            torch, microbatch, model, shadow, bundle, dispersion, panel, names, budget, state,
        )
        if set(gradients_by_state[state]) != selected:
            raise AuditStop('PARTIAL', 'Gradient panel does not match the frozen selection')
        if state == 's0' and not checks:
            checks = direct_checks(
                torch, microbatch, model, shadow, bundle, dispersion,
                records_by_state[state], gradients_by_state[state], names, budget,
            )
        if microbatch.existing.digest(model) != digest or sha256(checkpoint['path']) != checkpoint['file']:
            raise AuditStop('PARTIAL', 'Checkpoint changed during the audit')
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise AuditStop('PARTIAL', 'Diagnostic wrote .grad into the stored model')
        del model, shadow
        torch.cuda.empty_cache()
    assert_snapshots(protected)
    provenance = {
        'source_sha256': SOURCE_SHA,
        'dataset_logical_sha256': DATA_SHA,
        'manifest_sha256': MANIFEST_SHA,
        's0_file_sha256': S0_FILE_SHA,
        's0_tensor_sha256': S0_TENSOR_SHA,
        's70_file_sha256': S70_FILE_SHA,
        's70_tensor_sha256': S70_TENSOR_SHA,
        'parameter_count': PARAMETER_COUNT,
    }
    summary = build_summary(records_by_state, gradients_by_state, checks, budget, panel, provenance)
    write_json(SCRATCH / 'summary.json', summary)
    write_json(METRICS_PATH, public_metrics(summary))
    REPORT_PATH.write_text(render(summary), encoding='utf-8')
    write_plot(summary)
    print(f"DECISION {summary['decision']}", flush=True)
    return summary


def regenerate():
    summary = read_json(SCRATCH / 'summary.json')
    for payload in summary['states'].values():
        payload['counterfactual'] = counterfactual_weight_shares(payload['derivatives'])
    decision, reason = decide(summary['states'], summary['direct_checks'])
    summary['decision'] = decision
    summary['decision_reason'] = reason
    summary['test_note'] = (
        '15 CPU pytest tests in tests/test_lap_skala_loss_audit.py passed. '
        'Ruff and compileall passed on the audit module and its tests. '
        'git diff --check passed on the new files. No broad GPU or SCF suite was run.'
    )
    summary['proposal'] = proposal_text(decision)
    summary['limitations'] = limitations_text()
    write_json(SCRATCH / 'summary.json', summary)
    write_json(METRICS_PATH, public_metrics(summary))
    REPORT_PATH.write_text(render(summary), encoding='utf-8')
    write_plot(summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    if args.report_only:
        regenerate()
        return
    try:
        execute()
    except AuditStop as exc:
        write_json(SCRATCH / 'stop.json', {
            'decision': exc.status, 'decision_reason': str(exc), 'runtime': budget_state(),
        })
        REPORT_PATH.write_text(
            f'# Skala-inspired chemistry loss audit\n\n**Decision: {exc.status}.** {exc}\n\n'
            'Valid stage receipts remain in the scratch directory.\n',
            encoding='utf-8',
        )
        print(f'STOP {exc.status} {exc}', flush=True)
        raise SystemExit(2) from exc


if __name__ == '__main__':
    main()
