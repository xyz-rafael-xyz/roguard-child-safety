"""Durable review invariants, including failures and competing writers."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
import subprocess
import sys
from threading import Barrier

import pytest

from roguard.trajectory_data import input_identity
from roguard.trajectory_store import ReviewSessionStore, SessionConflict


def turn(text, role='minor'):
    return {'role': role, 'text': text}


def predict(payload):
    text = payload['turns'][-1]['text']
    return {'input_digest': input_identity(payload),
            'predicted': {'review': 1, 'unknown': None}.get(text, 0),
            'permits_external_action': True, 'unresolved_review_references': []}


@pytest.fixture
def store(tmp_path):
    value = ReviewSessionStore(tmp_path/'sessions.sqlite')
    value.create('conversation', 'ro', model_identity='test-pipeline')
    return value


def append(store, turns, revision=0, request='append', predictor=predict):
    return store.append('conversation', turns, expected_revision=revision,
                        request_id=request, model_identity='test-pipeline', predictor=predictor)


def test_every_minor_prefix_is_reviewed_and_persisted_across_process_restart(store):
    visited = []
    def tracking(payload):
        visited.append(len(payload['turns']))
        return predict(payload)
    result = append(store, [turn('review'), turn('okay', 'assistant'), turn('fine')], predictor=tracking)
    assert visited == [1, 3]
    assert result['session']['session_review_required']
    assert result['session']['last_prediction']['predicted'] == 0
    assert not result['session']['permits_external_action']
    code = ('import json,sys; from roguard.trajectory_store import ReviewSessionStore; '
            'print(json.dumps(ReviewSessionStore(sys.argv[1]).read("conversation")))')
    completed = subprocess.run([sys.executable, '-c', code, store.path],
        check=True, capture_output=True, text=True, env=os.environ.copy())
    resumed = json.loads(completed.stdout)
    assert resumed == result['session']
    assert resumed['unresolved_review_references'][0]['focus'] == 0


def test_abstention_remains_open_after_negative_and_assistant_turn(store):
    first = append(store, [turn('unknown'), turn('fine')])
    next_result = append(store, [turn('all done', 'assistant')], 1, 'next')
    assert next_result['predictions'] == []
    assert next_result['session']['session_review_required']
    assert first['session']['unresolved_review_references'] == next_result['session']['unresolved_review_references']
    assert first['session']['unresolved_review_references'][0]['reason'] == 'MODEL_ABSTAINED'


def test_retry_is_exact_and_does_not_repeat_inference_even_after_later_writes(store):
    calls = []
    def tracking(payload):
        calls.append(payload)
        return predict(payload)
    first = append(store, [turn('review')], predictor=tracking)
    append(store, [turn('fine')], 1, 'next')
    assert append(store, [turn('review')], predictor=tracking) == first
    assert len(calls) == 1
    assert store.read('conversation')['revision'] == 2
    with pytest.raises(SessionConflict, match='reused'):
        append(store, [turn('changed')])
    with pytest.raises(SessionConflict, match='Stale'):
        append(store, [turn('fine')], 0, 'stale')


def test_failed_batch_never_commits_partial_history_or_consumes_request(store):
    def fail_second(payload):
        if len(payload['turns']) == 2:
            raise RuntimeError('model unavailable')
        return predict(payload)
    with pytest.raises(RuntimeError, match='unavailable'):
        append(store, [turn('review'), turn('fine')], predictor=fail_second)
    assert store.read('conversation')['turn_count'] == 0
    assert store.read('conversation')['revision'] == 0
    assert append(store, [turn('review'), turn('fine')])['committed_revision'] == 1


@pytest.mark.parametrize('result', [
    {'predicted': 0}, {'predicted': True}, {'predicted': '0'},
    {'predicted': 2}, {}, None, {'predicted': 0, 'input_digest': 'forged'},
])
def test_invalid_or_unbound_prediction_cannot_commit(store, result):
    def invalid(payload):
        if isinstance(result, dict) and result and 'input_digest' not in result:
            if result == {'predicted': 0}:
                return result
            return {**result, 'input_digest': input_identity(payload)}
        return result
    with pytest.raises(ValueError):
        append(store, [turn('review')], predictor=invalid)
    assert store.read('conversation')['revision'] == 0


def test_two_independent_connections_cannot_overwrite_each_other(store):
    barrier = Barrier(2)
    def race(index):
        second = ReviewSessionStore(store.path)
        def waiting(payload):
            barrier.wait(timeout=10)
            return predict(payload)
        try:
            return append(second, [turn('review')], request='writer-'+str(index), predictor=waiting)
        except SessionConflict:
            return 'conflict'
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(race, (0, 1)))
    assert sum(r == 'conflict' for r in results) == 1
    assert store.read('conversation')['revision'] == 1
    assert len(store.audit('conversation')['turns']) == 1


def test_same_request_race_commits_once_and_returns_same_response(store):
    barrier = Barrier(2)
    def race(_):
        def waiting(payload):
            barrier.wait(timeout=10)
            return predict(payload)
        return append(ReviewSessionStore(store.path), [turn('review')], predictor=waiting)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(race, (0, 1)))
    assert results[0] == results[1]
    assert store.read('conversation')['revision'] == 1


def test_resolution_is_bound_to_current_context_and_audited(store):
    first = append(store, [turn('review'), turn('unknown')])['session']
    reference = first['unresolved_review_references'][0]['reference']
    newer = append(store, [turn('fine')], 1, 'next')['session']
    args = dict(expected_revision=1, expected_context_digest=first['context_digest'],
                request_id='resolve', reviewer_id='host-reviewer', reason='Reviewed original context')
    with pytest.raises(SessionConflict, match='stale'):
        store.resolve('conversation', reference, **args)
    args.update(expected_revision=2, expected_context_digest=newer['context_digest'])
    resolved = store.resolve('conversation', reference, **args)
    assert resolved['session']['session_review_required']  # abstention remains open
    assert len(resolved['session']['unresolved_review_references']) == 1
    assert store.resolve('conversation', reference, **args) == resolved
    audit = store.audit('conversation')
    assert audit['resolved'][0]['reviewer_id'] == 'host-reviewer'
    assert audit['resolved'][0]['resolution_context_digest'] == newer['context_digest']
    assert audit['turns'][0] == turn('review')


def test_reentrant_resolution_cannot_discard_concurrent_append(store):
    first = append(store, [turn('review')])['session']
    def resolve_during_inference(payload):
        store.resolve('conversation', first['unresolved_review_references'][0]['reference'],
            expected_revision=1, expected_context_digest=first['context_digest'],
            request_id='resolve', reviewer_id='reviewer', reason='Handled by host')
        return predict(payload)
    with pytest.raises(SessionConflict, match='advanced'):
        append(store, [turn('unknown')], 1, 'next', resolve_during_inference)
    assert store.read('conversation')['turn_count'] == 1
    committed = append(store, [turn('unknown')], 2, 'next-rebased')
    assert committed['session']['session_review_required']


def test_model_binding_limits_and_caller_mutation(store):
    with pytest.raises(SessionConflict):
        store.create('conversation', 'uk', model_identity='test-pipeline')
    with pytest.raises(SessionConflict):
        store.append('conversation', [turn('fine')], expected_revision=0,
            request_id='different', model_identity='other-model', predictor=predict)
    with pytest.raises(ValueError):
        append(store, [turn('fine')]*513)
    turns = [turn('review')]
    def mutate(payload):
        result = predict(payload)
        payload['turns'][0]['text'] = 'mutated'
        turns[0]['text'] = 'caller mutation'
        return result
    append(store, turns, predictor=mutate)
    assert store.audit('conversation')['turns'] == [turn('review')]
