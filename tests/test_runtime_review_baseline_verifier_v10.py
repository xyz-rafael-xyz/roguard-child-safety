"""Incomplete extraction, corrupt arrays and inaccurate parity reports fail the audit."""
from copy import deepcopy

import pytest

from eval.verify_runtime_review_baseline_v10 import require_complete, check_population, argmaxes


def test_completion_guard_accepts_only_finished_reference_extraction():
    report = dict(complete=True, decision='REFERENCE_EXTRACTION_ONLY_NO_MODEL_QUALIFICATION',
        training_performed=False, held_out_test_opened=False, qualified=False,
        languages={'ro': {}, 'uk': {}})
    require_complete(report)
    for name, value in [('complete', False), ('decision', 'PASS'), ('training_performed', True),
                        ('held_out_test_opened', True), ('qualified', True), ('languages', {'ro': {}})]:
        changed = deepcopy(report); changed[name] = value
        with pytest.raises(ValueError, match='incomplete or has invalid scope'):
            require_complete(changed)


@pytest.fixture
def population():
    np = pytest.importorskip('numpy')
    rows = [dict(case_id=str(i), partition='train' if i < 2 else 'dev', group='fixture', gold=1)
            for i in range(4)]
    x = np.zeros((4, 2560), np.float32); z = np.zeros((4, 19), np.float32)
    warm = np.zeros((2, 19), np.float32); old_x = x.copy(); old_z = z.copy(); reference = warm.astype(np.float64)
    observations = [dict(feature_index=i, case_id=str(i), partition=r['partition'],
        native_replay_error=.0001, historical_feature_difference=0., historical_binary_decision_change=0)
        for i,r in enumerate(rows)]
    spec = {'mode':'binary', 'threshold':.5}
    report = dict(complete=True, rows=4, checked=4, development_checked=2,
        maximum_native_replay_error=.0001, maximum_warm_start_logit_error=0.,
        maximum_historical_feature_difference=0., historical_feature_differences_above_001=0,
        historical_binary_decision_changes=0, runtime_source_groups_checked=2, maximum_runtime_score_error=0.)
    return [rows, observations, x, z, warm, reference, old_x, old_z, spec, spec.copy(), report]


def test_all_rows_and_both_partitions_are_recounted(population):
    result = check_population(*population)
    assert result['checked'] == 4 and result['development_checked'] == 2
    assert result['paired_v9'] == {s:dict(rows=2, historical_errors=0, runtime_errors=0, fixes=0, introductions=0)
                                  for s in ('train', 'dev')}
    assert result['warm_start_development_binary_errors'] == 0


@pytest.mark.parametrize('fault', ['test_partition', 'missing_observation', 'order', 'identity',
    'partition', 'boolean_index', 'native_nan', 'native_tolerance', 'wrong_feature_difference',
    'wrong_decision_change', 'feature_shape', 'feature_dtype', 'logit_nan', 'warm_shape',
    'warm_tolerance', 'warm_argmax', 'warm_decision', 'wrong_count', 'wrong_maximum',
    'wrong_groups', 'wrong_feature_count', 'serving_tolerance', 'serving_nan'])
def test_corrupted_or_incomplete_population_is_rejected(population, fault):
    np = pytest.importorskip('numpy')
    rows, observations, x, z, warm, reference, old_x, old_z, spec, warm_spec, report = population
    if fault == 'test_partition': rows[0]['partition'] = 'test'
    elif fault == 'missing_observation': observations.pop()
    elif fault == 'order': observations.reverse()
    elif fault == 'identity': observations[0]['case_id'] = 'other'
    elif fault == 'partition': observations[0]['partition'] = 'dev'
    elif fault == 'boolean_index': observations[0]['feature_index'] = False
    elif fault == 'native_nan': observations[0]['native_replay_error'] = float('nan')
    elif fault == 'native_tolerance': observations[0]['native_replay_error'] = .0011
    elif fault == 'wrong_feature_difference': observations[0]['historical_feature_difference'] = .00001
    elif fault == 'wrong_decision_change': observations[0]['historical_binary_decision_change'] = 1
    elif fault == 'feature_shape': population[2] = x[:, :2]
    elif fault == 'feature_dtype': population[2] = x.astype(np.float64)
    elif fault == 'logit_nan': z[0, 0] = np.nan
    elif fault == 'warm_shape': population[4] = warm[:1]
    elif fault == 'warm_tolerance': warm[0, 0] = .00002
    elif fault == 'warm_argmax': warm[0, 1] = .000001
    elif fault == 'warm_decision': warm[0, -1] = -.000001
    elif fault == 'wrong_count': report['checked'] = 3
    elif fault == 'wrong_maximum': report['maximum_native_replay_error'] = 0.
    elif fault == 'wrong_groups': report['runtime_source_groups_checked'] = 1
    elif fault == 'wrong_feature_count': report['historical_feature_differences_above_001'] = 1
    elif fault == 'serving_tolerance': report['maximum_runtime_score_error'] = 1e-9
    else: report['maximum_runtime_score_error'] = float('nan')
    with pytest.raises(ValueError):
        check_population(*population)


def test_historical_changes_are_reported_without_false_equivalence(population):
    x = population[2]; x[0, 0] = .002
    difference = float(x[0, 0]); population[1][0]['historical_feature_difference'] = difference
    population[-1].update(maximum_historical_feature_difference=difference, historical_feature_differences_above_001=1)
    result = check_population(*population)
    assert result['historical_feature_differences_above_001'] == 1


def test_seven_heads_use_their_own_argmax_ranges():
    np = pytest.importorskip('numpy')
    values = np.zeros((1, 19))
    values[0, [2, 5, 9, 12, 14, 16, 18]] = 1
    assert argmaxes(values).tolist() == [[2, 2, 3, 2, 1, 1, 1]]
