from copy import deepcopy
import pytest
from training.diagnose_interpreter_v9 import action, corrections, diagnose


def attributes(**changes):
    result = dict(concern='1', subject='SELF', source='DIRECT', assertion='AFFIRMED', relation='NEW', time='CURRENT')
    result.update(changes)
    return result


def test_alternative_repairs_preserve_inconsistent_scope_review():
    gold = attributes()
    prediction = attributes(concern='0', subject='FICTIONAL', source='FICTION')
    assert action(prediction) == 0 and action(gold) == 1
    assert corrections(prediction, gold) == [('concern', 'subject'), ('concern', 'source')]


def test_benign_fiction_requires_both_scope_fields_corrected():
    prediction = attributes()
    gold = attributes(subject='FICTIONAL', source='FICTION')
    assert corrections(prediction, gold) == [('subject', 'source')]


def test_strict_diagnosis_preserves_abstention_and_rejects_forged_policy():
    rows = [dict(case_id='one', language='uk', attributes=attributes(), source='synthetic')]
    predictions = [dict(case_id='one', language='uk', gold=1, predicted=None,
                        interpretation=attributes(), valid_label_tokens=False)]
    report = diagnose(rows, predictions)
    assert report['errors'][0]['invalid_schema']
    assert report['errors'][0]['minimal_policy_repairs'] == []
    assert report['strict_evaluation']['groups']['uk_formal']['strict_accuracy'] == 0
    bad = deepcopy(predictions)
    bad[0].update(predicted=0, valid_label_tokens=True)
    with pytest.raises(ValueError, match='strict decoded policy'):
        diagnose(rows, bad)
