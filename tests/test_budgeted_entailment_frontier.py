import itertools
import random

import pytest

pytest.importorskip('numpy')
from training.budgeted_entailment_frontier_v9 import frontier
from training.entailment_frontier_v9 import cuts


def brute_force(rows):
    groups = sorted({r['group'] for r in rows})
    best = None
    for support, revision in itertools.product(cuts([r['support_score'] for r in rows]),
                                               cuts([r['revision_score'] for r in rows])):
        rates = []
        total_misses = 0
        for group in groups:
            population = [r for r in rows if r['group'] == group]
            positives = sum(r['gold'] for r in population)
            misses = sum(r['gold'] == 1 and r['support_score'] < support and r['revision_score'] < revision for r in population)
            if misses * 100 > positives:
                break
            total_misses += misses
            negatives = len(population)-positives
            false = sum(r['gold'] == 0 and (r['support_score'] >= support or r['revision_score'] >= revision) for r in population)
            rates.append(false/negatives)
        else:
            rank = (max(rates), sum(rates), total_misses)
            if best is None or rank < best:
                best = rank
    return best


@pytest.mark.parametrize('seed', range(6))
def test_exact_frontier_matches_exhaustive_search_with_ties_and_small_groups(seed):
    rng = random.Random(seed)
    rows = [dict(group=g, gold=gold, support_score=rng.choice([0., .2, .8, 1.]),
                 revision_score=rng.choice([0., .2, .8, 1.]))
            for g, positive in [('formal', 200), ('conversation', 100), ('small', 44)]
            for gold, size in [(1, positive), (0, 40)] for _ in range(size)]
    # Avoid forcing every threshold to zero: retain nontrivial OR frontiers.
    for row in rows:
        if row['gold'] and row['support_score'] == row['revision_score'] == 0:
            row['revision_score'] = .2
    rows.extend(dict(group=g, gold=0, support_score=0., revision_score=0.)
                for g in ('formal', 'conversation', 'small'))
    result = frontier(rows)
    assert tuple(result['rank']) == brute_force(rows)
    assert result['rank'][0] < 1
    assert result['metrics']['small']['missed'] == 0
    assert all(m['missed'] * 100 <= m['positive'] for m in result['metrics'].values())


def test_one_outlier_can_be_missed_only_when_positive_count_allows_it():
    rows = [dict(group='formal', gold=1, support_score=.9, revision_score=0.) for _ in range(99)]
    rows += [dict(group='formal', gold=1, support_score=.01, revision_score=0.)]
    rows += [dict(group='formal', gold=0, support_score=.1, revision_score=0.) for _ in range(30)]
    result = frontier(rows)
    assert result['counts_pass'] and result['metrics']['formal']['missed'] == 1
    assert result['metrics']['formal']['false_reviews'] == 0
    rows.pop(0)
    result = frontier(rows)
    assert result['metrics']['formal']['missed'] == 0
    assert result['metrics']['formal']['false_reviews'] == 30
