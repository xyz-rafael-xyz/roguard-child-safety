import pytest
from training.conservative_fusion_v9 import full_coverage_gap


def test_all_development_concerns_are_covered_with_a_gap():
    cutoff=full_coverage_gap([0,1,0,1],[.1,.2,.3,.4])
    assert .1 < cutoff < .2
    assert all(s>=cutoff for s in [.2,.4])
    assert full_coverage_gap([1,0],[0.,.9])==0


@pytest.mark.parametrize('scores',[[None,.2],[float('nan'),.2],[1.1,.2]])
def test_no_silent_abstention_or_invalid_score(scores):
    with pytest.raises(ValueError):full_coverage_gap([1,0],scores)


def test_no_positive_population_rejected():
    with pytest.raises(ValueError):full_coverage_gap([0,0],[.1,.2])
