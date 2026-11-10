"""Memory, provenance, abstention and durable-session integration checks."""
from copy import deepcopy
from pathlib import Path

import pytest

from roguard.assurance import digest
from roguard.committee_trajectory import CommitteeTrajectoryModel
from roguard.gemma4_committee import Gemma4ReviewCommittee, COMPONENTS
from roguard.interpreter import prepare_input
from roguard.trajectory_data import input_identity
from roguard.trajectory_model import TrajectoryModel
from roguard.trajectory_neural import prepare_memory
from roguard.trajectory_sets import WORLDS, calibrate
from roguard.trajectory_store import ReviewSessionStore


def pipeline(callback=None, budget=1200):
    calls = []; models = {}
    for lang in ('ro', 'uk'):
        model = object.__new__(Gemma4ReviewCommittee)
        model.language = lang; model.identity = digest(lang)
        model.component_identities = {k: digest(lang+k) for k in COMPONENTS}; model._engine = None
        def predict(payload, model=model):
            calls.append(deepcopy(payload))
            if callback is not None:
                return callback(payload, model)
            return output(payload, model)
        model.predict = predict; models[lang] = model
    selector = TrajectoryModel(Path(__file__).resolve().parents[1]/'research/v10-evidence/student-03/student.json.gz')
    return CommitteeTrajectoryModel(models, selector, budget=budget), calls, selector


def output(payload, model, predicted=0):
    return {'input_digest': prepare_input(payload)[2], 'model_identity': model.identity,
        'predicted': predicted, 'score': predicted, 'review_requested': predicted != 0,
        'components': {k: {'field_logits': [0.]*19, 'predicted': predicted, 'score': .5} for k in COMPONENTS},
        'component_identities': model.component_identities, 'component_disagreement': False}


def test_memory_and_original_prefix_are_bound_separately():
    model, calls, selector = pipeline()
    payload = {'language': 'ro', 'turns': [{'role': 'minor', 'text': 'Prima propoziție. A doua propoziție!'},
                                         {'role': 'minor', 'text': 'Viitorul nu este consultat.'}]}
    result = model.predict(payload, focus=0)
    original = {'language': 'ro', 'turns': payload['turns'][:1]}
    _, expected = prepare_memory(original, selector, 1200, preserve_complete=True)
    assert calls == [expected]
    assert result['input_digest'] == input_identity(original)
    assert result['inference_input_digest'] == input_identity(expected)
    assert result['predictor_identity'] == model.predictor_identity()
    assert result['focus'] == 0 and result['used_memory'] is True
    assert 'interpretation_set' not in result
    assert len(result['probabilities']) == 432
    assert all(result[k] is False for k in ('qualified', 'establishes_safety', 'permits_external_action', 'permits_notification'))


def test_selector_is_owned_and_identity_tracks_memory_configuration():
    model, _, selector = pipeline()
    original = deepcopy(model.selector.spec)
    selector.spec['min_relevance'] = .999
    assert model.selector.spec == original
    other, _, _ = pipeline(budget=1000)
    assert model.identity != other.identity and model.predictor_identity() != other.predictor_identity()


def test_overflow_abstains_without_model_inference():
    model, calls, _ = pipeline(budget=128)
    result = model.predict({'language': 'uk', 'turns': [{'role': 'minor', 'text': 'А'*200}]})
    assert calls == [] and result['predicted'] is None and result['review_requested'] is True
    assert result['inference_input_digest'] is None and result['components'] is None
    assert result['reason'] == 'EVIDENCE_VIEW_EXCEEDS_COMMITTEE_CONTRACT'


def test_calibration_mismatch_and_planning_are_rejected_before_inference():
    model, calls, _ = pipeline()
    calibration = calibrate([], pipeline_digest='f'*64, method='probability_floor')
    payload = {'language': 'ro', 'turns': [{'role': 'minor', 'text': 'Un mesaj.'}]}
    with pytest.raises(ValueError):
        model.predict(payload, calibration=calibration)
    with pytest.raises(ValueError):
        model.predict(payload, plan=True)
    assert calls == []
    calibration = calibrate([], pipeline_digest=model.pipeline_digest(), method='probability_floor')
    result = model.predict(payload, calibration=calibration)
    assert result['interpretation_set']['hypotheses'] == WORLDS
    assert model.predictor_identity(calibration=calibration) != model.predictor_identity()
    planned = model.predict(payload, calibration=calibration, plan=True)
    assert 'assurance' in planned
    assert planned['predictor_identity'] != result['predictor_identity']


def test_language_switch_releases_the_other_encoder_and_failures_propagate():
    model, _, _ = pipeline()
    model.components['ro']._engine = object()
    model.predict({'language': 'ro', 'turns': [{'role': 'minor', 'text': 'Un mesaj.'}]})
    model.predict({'language': 'uk', 'turns': [{'role': 'minor', 'text': 'Повідомлення.'}]})
    assert model.components['ro']._engine is None
    def failure(payload, model):
        raise RuntimeError('inference failed')
    failed, _, _ = pipeline(failure)
    with pytest.raises(RuntimeError, match='inference failed'):
        failed.predict({'language': 'ro', 'turns': [{'role': 'minor', 'text': 'Un mesaj.'}]})


def test_wrong_component_binding_is_rejected():
    def wrong(payload, model):
        result = output(payload, model); result['input_digest'] = '0'*64; return result
    model, _, _ = pipeline(wrong)
    with pytest.raises(ValueError, match='another prefix'):
        model.predict({'language': 'ro', 'turns': [{'role': 'minor', 'text': 'Un mesaj.'}]})


def test_durable_store_keeps_earlier_review_after_negative_result(tmp_path):
    def prediction(payload, model):
        return output(payload, model, int(len(payload['turns']) == 1))
    model, calls, _ = pipeline(prediction); binding = model.predictor_identity()
    store = ReviewSessionStore(tmp_path/'sessions.sqlite'); store.create('session', 'ro', model_identity=binding)
    result = store.append('session', [{'role': 'minor', 'text': 'Primul mesaj.'}, {'role': 'minor', 'text': 'Al doilea mesaj.'}],
        expected_revision=0, request_id='request', model_identity=binding, predictor=model.predict)
    assert len(calls) == 2
    assert [r['predicted'] for r in result['predictions']] == [1, 0]
    assert result['session']['session_review_required'] is True
    reopened = ReviewSessionStore(tmp_path/'sessions.sqlite').read('session')
    assert reopened['session_review_required'] is True and len(reopened['unresolved_review_references']) == 1
