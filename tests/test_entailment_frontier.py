import itertools
import math
import random

import pytest

pytest.importorskip('numpy')

from training.entailment_frontier_v9 import cuts, frontier


def row(support, revision, gold, group='a'):
    return dict(support_score=support, revision_score=revision, gold=gold, group=group)


def test_separate_cutoffs_resolve_incompatible_score_scales():
    values = [row(.8, .01, 1), row(.1, .03, 1), row(.6, .02, 0)]
    result = frontier(values)
    assert result['counts_pass']
    assert .6 < result['support_threshold'] <= .8
    assert .02 < result['revision_threshold'] <= .03


def test_frontier_matches_exhaustive_search_including_ties_and_endpoints():
    rng = random.Random(123)
    for _ in range(40):
        values = [row(rng.choice([0., .1, .4, .9, 1.]), rng.choice([0., .2, .3, .8, 1.]), gold, group)
                  for group in ('a', 'b') for gold in (0, 0, 1, 1)]
        ranks = []
        for s, r in itertools.product(cuts([v['support_score'] for v in values]), cuts([v['revision_score'] for v in values])):
            predicted = [v['support_score'] >= s or v['revision_score'] >= r for v in values]
            if any(v['gold'] and not p for v, p in zip(values, predicted)):
                continue
            rates = [sum(p for v, p in zip(values, predicted) if not v['gold'] and v['group'] == g) / 2 for g in ('a', 'b')]
            ranks.append((max(rates), sum(rates)))
        assert tuple(frontier(values)['rank']) == min(ranks)


@pytest.mark.parametrize('bad', [math.nan, math.inf, -.1, 1.1, True])
def test_invalid_probability_rejected(bad):
    with pytest.raises(ValueError, match='probability'):
        frontier([row(bad, .5, 1), row(.1, .1, 0)])


def test_missing_negative_group_rejected():
    with pytest.raises(ValueError, match='both classes'):
        frontier([row(.9, .2, 1)])
