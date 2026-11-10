"""Response coverage and malformed-output protection in the research gate."""
from copy import deepcopy

import pytest

from roguard.trajectory_gate import assess
from roguard.trajectory_data import FORMAT, input_identity
from roguard.trajectory_model import RESPONSE_LABELS


def response_rows():
    return [{'case_id': lang+':'+label, 'language': lang, 'gold': label,
             'prediction': {'label': label}} for lang in ('ro', 'uk') for label in RESPONSE_LABELS]


def gate(rows):
    payload = {'language': 'ro', 'turns': [{'role': 'minor', 'text': 'Un mesaj.'}]}
    case = dict(payload, format=FORMAT, case_id='example', split='challenge', family='example',
                author='synthetic-author', generator='synthetic-generator', synthetic=True,
                checkpoints=[{'focus': 0, 'review': 0}])
    prediction = {'checkpoint_id': 'example:0', 'language': 'ro', 'focus': 0,
                  'input_digest': input_identity(payload), 'predicted': 0}
    return assess([case], [prediction], response_rows=rows)


def test_complete_correct_response_matrix_does_not_qualify_release():
    result = gate(response_rows())
    assert not any(f.startswith('RESPONSE_') for f in result['engineering_failures'])
    assert result['response_evaluation']['rows'] == 10
    assert result['response_evaluation']['missing_groups'] == []
    assert result['promotion_pass'] is False and result['stable_release_ready'] is False
    assert 'INDEPENDENT_SEALED_EVALUATION_REQUIRED' in result['external_requirements']


@pytest.mark.parametrize('rows', [response_rows()[:5], response_rows()[:-1], response_rows()[:1]])
def test_correct_but_incomplete_response_population_cannot_pass(rows):
    result = gate(rows)
    assert 'RESPONSE_LANGUAGE_CATEGORY_COVERAGE_MISSING' in result['engineering_failures']
    assert result['response_evaluation']['missing_groups']


@pytest.mark.parametrize('value', [False, 'true', 1, None])
def test_explicit_invalid_output_cannot_pass_by_matching_label(value):
    rows = response_rows(); rows[0]['prediction']['valid'] = value
    result = gate(rows)
    assert 'RESPONSE_EVALUATION_ERRORS' in result['engineering_failures']
    assert result['response_evaluation']['invalid_rows'] == 1


def test_duplicate_and_unknown_rows_do_not_create_coverage():
    rows = response_rows(); rows[-1] = deepcopy(rows[0])
    result = gate(rows)
    assert result['response_evaluation']['invalid_rows'] == 1
    assert 'uk:unsupported' in result['response_evaluation']['missing_groups']
    rows = response_rows(); rows[0]['gold'] = rows[0]['prediction']['label'] = 'unknown'
    result = gate(rows)
    assert result['response_evaluation']['invalid_rows'] == 1
    assert 'RESPONSE_EVALUATION_ERRORS' in result['engineering_failures']


def test_wrong_known_label_is_an_error_without_losing_gold_group_coverage():
    rows = response_rows(); rows[0]['prediction']['label'] = 'privacy'
    result = gate(rows)
    assert result['response_evaluation']['errors'] == 1
    assert result['response_evaluation']['invalid_rows'] == 0
    assert not result['response_evaluation']['missing_groups']
    assert 'RESPONSE_EVALUATION_ERRORS' in result['engineering_failures']


@pytest.mark.parametrize('rows', [None, [], {}, [None], [{'gold': 'appropriate', 'prediction': {'label': 'appropriate'}}]])
def test_missing_or_malformed_response_data_never_passes(rows):
    result = gate(rows)
    assert any(f.startswith('RESPONSE_') for f in result['engineering_failures'])
