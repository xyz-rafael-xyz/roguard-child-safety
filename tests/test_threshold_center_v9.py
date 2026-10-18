import pytest
from training.threshold_center_v9 import center_threshold


def test_center_preserves_decisions_and_avoids_positive_boundary():
    scores = [.1, .3, .56132089, .8, .99]
    old = .56132088
    new = center_threshold(scores, old)
    assert [s >= new for s in scores] == [s >= old for s in scores]
    assert min(abs(s-new) for s in scores) > .1
    assert [s-1e-5 >= new for s in scores] == [s >= old for s in scores]


@pytest.mark.parametrize('scores,limit', [([.5,.5],.5),([.2,.3],.1),([.2,.3],.8)])
def test_uniform_decision_population_is_unchanged(scores,limit):
    assert center_threshold(scores,limit) == limit


@pytest.mark.parametrize('scores,limit', [([], .5),([float('nan')],.5),([1.1],.5),([.2],float('nan'))])
def test_invalid_values_fail(scores,limit):
    with pytest.raises(ValueError):center_threshold(scores,limit)
