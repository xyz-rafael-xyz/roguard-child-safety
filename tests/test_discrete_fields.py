import copy
import pytest

from roguard.assurance import policy_digest
from roguard.evidence import ONTOLOGY
from roguard.native_review import prepare_input
from roguard.discrete_fields import review


def example():
    payload = dict(language='ro', turns=[dict(role='minor', text='Exemplu fictiv pentru test.')])
    _, focus, identity = prepare_input(payload)
    values = dict(subject='SELF', time='CURRENT', assertion='AFFIRMED', source='DIRECT', relation='NEW', concern='1')
    report = dict(format='roguard-native-fields-v1', model_identity='a'*64, input_digest=identity,
        focus=focus, language='ro', policy_digest=policy_digest(), interpretation_is_head_argmax=True,
        interpretation=values, field_probabilities={f: [float(v == label) for label in ONTOLOGY[f]] for f, v in values.items()},
        score=.01, predicted=0, probabilities_are_calibrated=False, text_in_report=False,
        establishes_safety=False, permits_external_action=False, permits_notification=False)
    return payload, report


def test_discrete_rule_uses_interpretation_not_parent_threshold():
    payload, report = example()
    result = review(payload, report, expected_model_identity='a'*64)
    assert result['predicted'] == 1
    assert result['source_model_identity'] == 'a'*64
    assert result['model_identity'] != report['model_identity']
    assert result['permits_notification'] is False


@pytest.mark.parametrize('key,value', [('focus', True), ('input_digest', 'b'*64),
    ('model_identity', 'b'*64), ('policy_digest', 'b'*64), ('language', 'uk'),
    ('permits_notification', True), ('probabilities_are_calibrated', True)])
def test_source_binding_and_claims_are_checked(key, value):
    payload, report = example()
    report[key] = value
    with pytest.raises(ValueError):
        review(payload, report, expected_model_identity='a'*64)


def test_inconsistent_interpretation_is_rejected():
    payload, report = example()
    report['interpretation']['concern'] = '0'
    with pytest.raises(ValueError):
        review(payload, report, expected_model_identity='a'*64)


def test_context_abstention_remains_a_review_request():
    payload, report = example()
    report.update(field_probabilities=None, interpretation=None, score=None, predicted=None)
    result = review(payload, report, expected_model_identity='a'*64)
    assert result['predicted'] is None and result['review_requested'] is True


def test_invalid_probabilities_are_rejected():
    payload, report = example()
    report['field_probabilities']['concern'][0] = float('nan')
    with pytest.raises(ValueError):
        review(payload, report, expected_model_identity='a'*64)
