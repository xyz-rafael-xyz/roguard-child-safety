import json
import random
from collections import defaultdict
from types import SimpleNamespace

import pytest

from training.referent_grounding_v9 import generate
from training.shared_tail_training_v9 import plan_batches
from training.composite_tail_cache_v9 import CompositeReader


def test_referents_keep_context_dependence_and_disjoint_wording():
    for language in ('ro', 'uk'):
        seen = set()
        for partition, count in (('train', 600), ('dev', 52)):
            rows = generate(language, partition)
            assert len(rows) == count
            families = defaultdict(list)
            for row in rows:
                key = json.dumps(row['turns'], ensure_ascii=False, sort_keys=True)
                assert key not in seen
                seen.add(key)
                if row.get('context_family'):
                    families[row['context_family']].append(row)
                else:
                    assert row['review'] == 0
            assert all(len(v) == 6 for v in families.values())
            for family in families.values():
                assert [r['review'] for r in family] == [0, 1, 0, 1, 1, 1]
            batches = plan_batches(rows, random.Random(1177), 8)
            assert sorted(i for ids, _, _ in batches for i in ids) == list(range(count))


def test_composite_reader_preserves_order_identity_and_partial_dev_batches():
    reader = CompositeReader.__new__(CompositeReader)
    def fake(prefix, count):
        entries = [dict(partition='dev', row=dict(language='ro')) for _ in range(count)]
        return SimpleNamespace(entries=entries, read_many=lambda indices: [(prefix, i) for i in indices])
    reader.readers = [fake('a', 3), fake('b', 4)]
    reader.locations = [(0, i) for i in range(3)]+[(1, i) for i in range(4)]
    assert reader.read_many([5, 0, 3, 0]) == [('b', 2), ('a', 0), ('b', 0), ('a', 0)]
    assert list(reader.development_batches('ro', 2)) == [[0, 1], [2], [3, 4], [5, 6]]
    for invalid in (-1, 7, True):
        with pytest.raises(ValueError):
            reader.read_many([invalid])


def test_headroom_frontier_balances_both_error_budgets():
    pytest.importorskip('numpy')
    from training.balanced_headroom_frontier_v9 import frontier
    # At .8: 1% misses uses the whole recall budget. At .3: 1% false
    # reviews uses only one fifth of that budget and preserves all concerns.
    rows = [dict(gold=1, score=.9, group='a') for _ in range(99)]
    rows += [dict(gold=1, score=.4, group='a')]
    rows += [dict(gold=0, score=.1, group='a') for _ in range(99)]
    rows += [dict(gold=0, score=.6, group='a')]
    selected = frontier(rows)
    assert selected['counts_pass']
    assert selected['metrics']['a']['missed'] == 0
    assert selected['metrics']['a']['false_reviews'] == 1
    assert selected['rank'][0] == .2
    # Small groups may never hide a missed concern inside rounded rates.
    small = [dict(gold=1, score=.2, group='small'), dict(gold=0, score=.8, group='small')]
    result = frontier(small)
    assert not result['counts_pass']
    with pytest.raises(ValueError):
        frontier([dict(gold=0, score=float('nan'), group='bad')])
