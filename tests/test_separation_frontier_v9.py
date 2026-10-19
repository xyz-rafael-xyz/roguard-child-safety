import pytest

pytest.importorskip('numpy')

from training.separation_frontier_v9 import frontier


def population(negative, positive):
    return [dict(group='one', gold=0, score=negative),
            dict(group='one', gold=1, score=positive)]


def test_same_perfect_accuracy_prefers_wider_separation_and_is_order_independent():
    narrow = frontier(population(.99, 1.))
    rows = population(.1, .9)
    wide = frontier(rows)
    assert wide['rank'] < narrow['rank']
    assert wide['threshold'] == pytest.approx(.5)
    assert wide['score_margin'] == pytest.approx(.4)
    assert frontier(list(reversed(rows))) == wide


def test_accuracy_has_priority_and_identical_scores_cannot_separate():
    perfect = frontier(population(.499, .501))
    imperfect = frontier(population(.9, .1))
    equal = frontier(population(.5, .5))
    assert perfect['rank'] < imperfect['rank']
    assert imperfect['separation_interval'] is None
    assert not equal['counts_pass']


def test_one_group_cannot_hide_an_error_in_another():
    rows = population(.01, .99)+[dict(group='two', gold=0, score=.9), dict(group='two', gold=1, score=.8)]
    assert not frontier(rows)['counts_pass']
