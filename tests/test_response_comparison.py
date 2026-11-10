from copy import deepcopy

import pytest

from eval.assess_response_attention_v10 import compare
from roguard.trajectory_model import RESPONSE_LABELS


def population():
    return [{'case_id': lang+':'+label, 'language': lang, 'draft': label,
             'context': '', 'gold': label, 'prediction': {'label': label}}
            for lang in ('ro', 'uk') for label in RESPONSE_LABELS]


def test_pairing_exposes_regression_despite_equal_total_accuracy():
    expected = population(); old = deepcopy(expected); new = deepcopy(expected)
    old[0]['prediction']['label'] = expected[1]['gold']
    new[1]['prediction']['label'] = expected[0]['gold']
    result = compare(expected, old, new)
    assert result['baseline_errors'] == result['candidate_errors'] == 1
    assert result['fixed'] == result['introduced'] == 1


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'duplicate', 'input', 'gold', 'invalid'])
def test_population_or_prediction_changes_are_rejected(mutation):
    expected = population(); new = deepcopy(expected)
    if mutation == 'missing': new.pop()
    elif mutation == 'extra': new.append({**deepcopy(new[0]), 'case_id': 'extra'})
    elif mutation == 'duplicate': new.append(deepcopy(new[0]))
    elif mutation == 'input': new[0]['context'] = 'changed'
    elif mutation == 'gold': new[0]['gold'] = new[1]['gold']
    elif mutation == 'invalid': new[0]['prediction']['valid'] = False
    with pytest.raises(ValueError): compare(expected, expected, new)


def test_order_is_irrelevant_but_every_case_is_required():
    rows = population()
    result = compare(rows, rows[::-1], rows[2:]+rows[:2])
    assert result['rows'] == 10
    assert result['candidate_errors'] == result['introduced'] == 0
