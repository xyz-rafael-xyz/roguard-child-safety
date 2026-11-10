from collections import Counter

import pytest

np = pytest.importorskip('numpy')

from roguard.trajectory_model import RESPONSE_LABELS
from training.response_sampling_v10 import MixedResponseSampler, STRATA


def population():
    rows = [{'case_id': 'dev-excluded', 'partition': 'dev'}]
    for origin, language in STRATA:
        for label in RESPONSE_LABELS:
            n = 2 if origin == 'historical' else 10 if label == 'appropriate' else 5
            for j in range(n):
                rows.append({'case_id': f'{origin}:{language}:{label}:{j}',
                             'origin': origin, 'language': language,
                             'gold': label, 'partition': 'train'})
    return rows, list(range(1, len(rows)))


def test_every_update_mixes_strata_and_full_pass_covers_generated_once():
    rows, indices = population()
    sampler = MixedResponseSampler(rows, indices)
    batches = sampler.batches(np.random.default_rng(1059))
    assert batches.shape == (sampler.batches_per_pass, 8) == (15, 8)
    assert batches.size == sampler.presentations_per_pass == 120
    visits = Counter(batches.ravel())
    assert set(visits) == set(indices)
    for batch in batches:
        assert len(set(batch)) == 8
        assert Counter((rows[i]['origin'], rows[i]['language']) for i in batch) == {
            key: 2 for key in STRATA}
    for i in indices:
        assert visits[i] == (1 if rows[i]['origin'] == 'generated' else 3)
    assert visits[0] == 0 and sampler.weights[0] == 0
    # Complete historical cycles and one generated pass give equal objective
    # mass to every group despite unequal label and source populations.
    mass = Counter()
    for i, count in visits.items():
        r = rows[i]
        mass[(r['origin'], r['language'], r['gold'])] += count * sampler.weights[i]
    assert len(mass) == 20
    assert list(mass.values()) == pytest.approx([6.] * 20)


def test_partial_historical_cycle_preserves_coverage_and_no_batch_duplicates():
    rows, _ = population()
    # Historical size 12 does not divide generated size 30.
    for language in ('ro', 'uk'):
        for j in range(2):
            rows.append({'case_id': f'extra:{language}:{j}', 'origin': 'historical',
                         'language': language, 'gold': 'appropriate', 'partition': 'train'})
    sampler = MixedResponseSampler(rows, range(1, len(rows)))
    batches = sampler.batches(np.random.default_rng(17))
    visits = Counter(batches.ravel())
    for i, row in enumerate(rows[1:], 1):
        assert visits[i] in ({2, 3} if row['origin'] == 'historical' else {1})
    assert all(len(set(batch)) == 8 for batch in batches)


def test_seed_reproduces_sequence_without_repeating_epoch_order():
    rows, indices = population()
    sampler = MixedResponseSampler(rows, indices)
    a, b = np.random.default_rng(1059), np.random.default_rng(1059)
    first, second = sampler.batches(a), sampler.batches(a)
    assert np.array_equal(first, sampler.batches(b))
    assert np.array_equal(second, sampler.batches(b))
    assert not np.array_equal(first, second)
    with pytest.raises(ValueError):
        sampler.weights[1] = 0


@pytest.mark.parametrize('bad', ['dev', 'duplicate_index', 'duplicate_id', 'negative',
                               'outside', 'boolean', 'float', 'empty_id', 'unknown_gold',
                               'unknown_origin', 'unknown_language', 'missing_group',
                               'unequal_generated', 'odd_historical', 'empty'])
def test_rejects_leakage_or_invalid_sampling_population(bad):
    rows, indices = population()
    if bad == 'dev': indices.append(0)
    elif bad == 'duplicate_index': indices.append(indices[0])
    elif bad == 'duplicate_id': rows[2]['case_id'] = rows[1]['case_id']
    elif bad == 'negative': indices[0] = -1
    elif bad == 'outside': indices[0] = len(rows)
    elif bad == 'boolean': indices[0] = True
    elif bad == 'float': indices[0] = 1.0
    elif bad == 'empty_id': rows[1]['case_id'] = ''
    elif bad == 'unknown_gold': rows[1]['gold'] = 'other'
    elif bad == 'unknown_origin': rows[1]['origin'] = 'other'
    elif bad == 'unknown_language': rows[1]['language'] = 'en'
    elif bad == 'missing_group':
        indices = [i for i in indices if not (rows[i]['origin'] == 'historical'
                   and rows[i]['language'] == 'ro' and rows[i]['gold'] == 'boundary')]
    elif bad == 'unequal_generated':
        removed = [i for i in indices if rows[i]['origin'] == 'generated'][:2]
        indices = [i for i in indices if i not in removed]
    elif bad == 'odd_historical': indices = indices[1:]
    elif bad == 'empty': indices = []
    with pytest.raises(ValueError):
        MixedResponseSampler(rows, indices)
