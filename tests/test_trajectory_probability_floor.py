"""Conformal probability ordering must survive tiny positive model scores."""
import pytest

from roguard.trajectory_sets import WORLDS, calibrate, predict_set
from roguard.assurance import digest


PIPELINE = 'a'*64


def distribution(p):
    values = [0.]*len(WORLDS)
    values[0] = p; values[1] = p/10; values[2] = 1-p-p/10
    return values


def rows(probabilities, families=9):
    return [{'case_id': lang+str(i), 'split': 'calibration', 'language': lang,
             'family': str(i), 'pipeline_digest': PIPELINE, 'world': WORLDS[0],
             'probabilities': list(probabilities)} for lang in ('ro', 'uk') for i in range(families)]


def test_positive_tiny_cutoffs_do_not_collapse_to_the_full_ontology():
    values = distribution(1e-30)
    legacy = calibrate(rows(values), pipeline_digest=PIPELINE)
    stable = calibrate(rows(values), pipeline_digest=PIPELINE, method='probability_floor')
    assert legacy['quantiles']['ro'] == 1.
    assert stable['probability_floors']['ro'] == 1e-30
    old = predict_set(values, legacy, language='ro', pipeline_digest=PIPELINE)
    new = predict_set(values, stable, language='ro', pipeline_digest=PIPELINE)
    assert len(old['hypotheses']) == 432
    assert new['hypotheses'] == [WORLDS[0], WORLDS[2]]
    assert new['coverage_assumptions_verified'] is False
    assert stable['real_world_coverage_established'] is False


def test_family_minima_and_finite_sample_rank_are_preserved():
    values = distribution(.01); records = rows(values)
    for lang in ('ro', 'uk'):
        records.append({'case_id': lang+'extra', 'split': 'calibration', 'language': lang,
            'family': '0', 'pipeline_digest': PIPELINE, 'world': WORLDS[0], 'probabilities': distribution(1e-25)})
    stable = calibrate(records, pipeline_digest=PIPELINE, method='probability_floor')
    assert stable['families_per_language'] == {'ro': 9, 'uk': 9}
    assert stable['probability_floors'] == {'ro': 1e-25, 'uk': 1e-25}
    small = calibrate(rows(values, 8), pipeline_digest=PIPELINE, method='probability_floor')
    assert small['probability_floors'] == {'ro': 0., 'uk': 0.}
    assert len(predict_set(values, small, language='uk', pipeline_digest=PIPELINE)['hypotheses']) == 432


def test_equivalence_to_legacy_rule_away_from_cancellation():
    records = rows(distribution(.25), families=19)
    old = calibrate(records, pipeline_digest=PIPELINE)
    new = calibrate(records, pipeline_digest=PIPELINE, method='probability_floor')
    for probability in (.125, .25, .5):
        values = distribution(probability)
        assert predict_set(values, old, language='ro', pipeline_digest=PIPELINE)['hypotheses'] == predict_set(
            values, new, language='ro', pipeline_digest=PIPELINE)['hypotheses']


def test_floor_artifact_identity_and_limits_are_checked():
    values = distribution(1e-20)
    calibration = calibrate(rows(values), pipeline_digest=PIPELINE, method='probability_floor')
    with pytest.raises(ValueError, match='another pipeline'):
        predict_set(values, calibration, language='ro', pipeline_digest='b'*64)
    calibration['probability_floors']['ro'] = .5
    with pytest.raises(ValueError, match='integrity'):
        predict_set(values, calibration, language='ro', pipeline_digest=PIPELINE)
    calibration['probability_floors']['ro'] = -1.
    calibration['digest'] = digest({k: v for k, v in calibration.items() if k != 'digest'})
    with pytest.raises(ValueError, match='quantile'):
        predict_set(values, calibration, language='ro', pipeline_digest=PIPELINE)
    with pytest.raises(ValueError, match='Unknown'):
        calibrate(rows(values), pipeline_digest=PIPELINE, method='unregistered')


def test_unsupported_and_empty_sets_remain_explicit():
    calibration = calibrate(rows(distribution(.5)), pipeline_digest=PIPELINE, method='probability_floor')
    flat = [1/432]*432
    for supported in (False, True):
        result = predict_set(flat, calibration, language='uk', pipeline_digest=PIPELINE, supported=supported)
        assert result['reason'] == 'UNSUPPORTED_OR_EMPTY_SET'
        assert len(result['hypotheses']) == 432
