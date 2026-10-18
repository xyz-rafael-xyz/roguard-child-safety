from copy import deepcopy

import pytest
import roguard.interpreter as runtime
from roguard.assurance import answer, verify
from roguard.evidence import ONTOLOGY
from roguard.interpreter_assurance import plan_review, verify_review_plan


@pytest.fixture
def inputs(monkeypatch):
    monkeypatch.setattr(runtime, '_read_archive', lambda *args: {})
    state = dict(values={field: labels[0] for field, labels in ONTOLOGY.items()}, valid=True)
    class Engine:
        def __init__(self, *args):
            pass
        def decode(self, turns):
            values = deepcopy(state['values'])
            return dict(interpretation=values, valid_label_tokens=state['valid'],
                        conditional_field_probabilities={field: [float(v == values[field]) for v in labels]
                                                         for field, labels in ONTOLOGY.items()})
    monkeypatch.setattr(runtime, '_MLXEngine', Engine)
    payload = dict(language='ro', turns=[dict(role='minor', text='Exercițiul de azi.')])
    def report(identity='a'*64):
        return runtime.InterpreterModel('unused', 'unused', expected_archive_sha256=identity).predict(payload)
    return payload, state, report


def test_actual_runtime_report_shape_connects_to_verifiable_plan(inputs):
    payload, _, report = inputs
    result = plan_review(payload, [report()], expected_model_identities=['a'*64])
    assert verify(result['certificate'])
    assert result['certificate']['possible_actions'] == ['CONTINUE']
    assert not result['permits_external_action'] and not result['permits_notification']
    assert not result['hypothesis_coverage_verified'] and not result['reports_authenticated']
    assert 'Exercițiul' not in str(result)


def test_joint_worlds_are_preserved_without_cross_model_hybrids(inputs):
    payload, state, report = inputs
    a = report()
    state['values'].update(concern='1', assertion='UNCERTAIN', subject='OTHER', source='REPORTED')
    b = report('b'*64)
    result = plan_review(payload, [a,b], expected_model_identities=['a'*64,'b'*64])
    certificate = result['certificate']
    assert len(certificate['input']['hypotheses']) == 2
    assert certificate['status'] == 'CLARIFICATION_REQUIRED'
    field = certificate['tree']['question']
    resolved = answer(certificate, {field:b['interpretation'][field]}, expected_certificate_digest=certificate['certificate_digest'])
    assert not resolved['permits_external_action']


def test_abstention_retains_full_uncertainty(inputs):
    payload, state, report = inputs
    state['valid'] = False
    result = plan_review(payload, [report()], expected_model_identities=['a'*64])
    assert result['includes_abstention']
    assert len(result['certificate']['input']['hypotheses']) == 432
    assert result['certificate']['status'] == 'CLARIFICATION_REQUIRED'
    assert verify(result['certificate'])


@pytest.mark.parametrize('fault', ['input_digest','contract_digest','model_identity','focus','predicted'])
def test_stale_foreign_or_inconsistent_report_rejected(inputs, fault):
    payload, _, make_report = inputs
    report = make_report()
    report[fault] = 1 if fault in ('focus','predicted') else 'b'*64
    with pytest.raises(ValueError):
        plan_review(payload, [report], expected_model_identities=['a'*64])


def test_later_turns_remain_outside_binding(inputs):
    payload, _, report = inputs
    value = report()
    before = plan_review(payload, [value], expected_model_identities=['a'*64])
    payload['turns'].append({'invalid':'future'})
    after = plan_review(payload, [value], expected_model_identities=['a'*64], focus=0)
    assert before == after


def test_missing_or_duplicate_models_cannot_look_like_consensus(inputs):
    payload, _, report = inputs
    value = report()
    with pytest.raises(ValueError, match='missing'):
        plan_review(payload, [value], expected_model_identities=['a'*64,'b'*64])
    with pytest.raises(ValueError, match='repeated'):
        plan_review(payload, [value,value], expected_model_identities=['a'*64])


def test_wrapper_provenance_is_verified_as_well_as_certificate(inputs):
    payload, _, report = inputs
    reports = [report()]
    plan = plan_review(payload, reports, expected_model_identities=['a'*64])
    assert verify_review_plan(plan, payload, reports, expected_model_identities=['a'*64])
    plan['source_report_digests'] = ['b'*64]
    assert verify(plan['certificate'])
    with pytest.raises(ValueError, match='bound reports'):
        verify_review_plan(plan, payload, reports, expected_model_identities=['a'*64])
