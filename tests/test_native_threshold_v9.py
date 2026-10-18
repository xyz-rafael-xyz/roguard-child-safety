import pytest
from training.native_threshold_v9 import select, margin_threshold


def population():
    rows, truth, scores = [], [], []
    for source in ('formal', 'v7_training_frame_projection'):
        rows += [dict(language='ro', source=source)]*200
        truth += [0]*100+[1]*100
        scores += [.4]+[.01]*99+[.2]+[.9]*99
    return rows, truth, scores


def test_margin_uses_one_percent_budget_and_prioritizes_worst_stratum_recall():
    rows, truth, scores = population()
    result = select(rows, truth, scores, {'ro': .5}, 'ro')
    assert result['selected']['name'] == 'margin_1pct'
    assert result['selected']['thresholds']['ro'] == pytest.approx(.0101)
    for group in ('ro_formal', 'ro_conversational'):
        assert result['selected']['development'][group]['missed'] == 0
        assert result['selected']['development'][group]['false_reviews'] == 1


def test_unachievable_guarded_budget_does_not_invent_a_valid_threshold():
    rows, truth, scores = population()
    scores = [1. if gold == 0 else score for gold, score in zip(truth, scores)]
    with pytest.raises(ValueError, match='guarded threshold'):
        margin_threshold(rows, truth, scores, 'ro')
    result = select(rows, truth, scores, {'ro': .5}, 'ro')
    assert result['selected']['name'] == 'original'
    assert not result['any_development_eligible']


def test_development_tie_prefers_original_without_consulting_calibration():
    rows, truth, scores = population()
    result = select(rows, truth, scores, {'ro': .0101}, 'ro')
    assert result['selected']['name'] == 'original'
    assert result['variants'][0]['development'] == result['variants'][1]['development']
