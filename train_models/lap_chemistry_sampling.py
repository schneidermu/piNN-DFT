"""Permanent one-variant chemistry policy and deterministic identity epochs."""
import hashlib
import random

POLICY = 'one-variant-per-identity-v1'


def stream(seed, domain):
    return random.Random(int(hashlib.sha256(f'{seed}:{domain}'.encode()).hexdigest(), 16))


def selected(row, variant):
    return {'identity': row['id'], 'database': row['database'], 'reaction_id': row['reaction_id'],
                'variant': variant, 'weight': 1.0}


def fixed_evaluation(reactions, seed=202610091):
    rows = []
    for identity, row in sorted(reactions.items()):
        variant = stream(seed, 'evaluation:' + identity).choice(sorted(row['variants']))
        rows.append({'identity': identity, 'variant': variant, 'task': row['task']})
    validate_evaluation(rows, reactions)
    return {'policy': POLICY, 'seed': seed, 'rows': rows}


def validate_evaluation(rows, reactions):
    ids = [r['identity'] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(reactions):
        raise ValueError('Exactly one selected variant per chemical identity is mandatory')
    for row in rows:
        source = reactions[row['identity']]
        if row['variant'] not in source['variants'] or row['task'] != source['task']:
            raise ValueError('Invalid selected variant/task')


def epoch_samples(reactions, systems, epoch=0, seed=202610092):
    identities = sorted(i for i, r in reactions.items() if r['task'] == 'relchem')
    ae = sorted(i for i, r in reactions.items() if r['task'] == 'ae17')
    stream(seed, f'training:epoch:{epoch}:order').shuffle(identities)
    samples, cycle = [], []
    for index, identity in enumerate(identities):
        if index % len(systems) == 0:
            cycle = sorted(systems)
            stream(seed, f'training:epoch:{epoch}:mrks:{index // len(systems)}').shuffle(cycle)
        row = dict(reactions[identity], id=identity)
        variant = stream(seed, f'training:epoch:{epoch}:variant:{identity}').choice(sorted(row['variants']))
        rng = stream(seed, f'training:epoch:{epoch}:ae:{index}')
        a = rng.choice(ae)
        ae_row = dict(reactions[a], id=a)
        samples.append({'cursor': index, 'relchem': selected(row, variant),
                            'ae17': selected(ae_row, rng.choice(sorted(ae_row['variants']))),
                            'mrks_id': cycle[index % len(systems)]})
    return samples
