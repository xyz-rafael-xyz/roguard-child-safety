import pytest

pytest.importorskip('numpy')
from training.budgeted_native_frontier_v9 import frontier


def test_each_group_has_its_own_miss_budget():
    rows = [dict(group='large', gold=1, score=.9) for _ in range(99)]
    rows += [dict(group='large', gold=1, score=.01)]
    rows += [dict(group='small', gold=1, score=.5) for _ in range(44)]
    rows += [dict(group=g, gold=0, score=.1) for g in ('large','small') for _ in range(30)]
    result = frontier(rows)
    assert result['counts_pass']
    assert result['metrics']['large']['missed'] == 1
    assert result['metrics']['small']['missed'] == 0
    rows[-1]['score'] = .5
    result = frontier(rows)
    assert result['metrics']['small']['missed'] == 0
    assert result['metrics']['small']['false_reviews'] == 1


def test_equal_false_review_counts_choose_fewer_misses():
    rows = [dict(group='g', gold=1, score=.9) for _ in range(99)]
    rows += [dict(group='g', gold=1, score=.2)]
    rows += [dict(group='g', gold=0, score=.1) for _ in range(30)]
    assert frontier(rows)['rank'] == [0., 0., 0]


def test_no_cutoff_can_separate_opposed_labels_at_identical_scores():
    rows = [dict(group='g', gold=g, score=.5) for g in (0,1) for _ in range(100)]
    result = frontier(rows)
    assert result['metrics']['g']['missed'] == 0
    assert result['metrics']['g']['false_reviews'] == 100
    assert not result['counts_pass']


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-0.1,1.1,True])
def test_invalid_scores_fail_before_cutoff_selection(value):
    with pytest.raises(ValueError):
        frontier([dict(group='g',gold=0,score=value),dict(group='g',gold=1,score=.8)])
