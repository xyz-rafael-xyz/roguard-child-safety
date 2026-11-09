import copy
import gzip
import hashlib
import json
import types
from pathlib import Path

import pytest
np = pytest.importorskip('numpy')
from roguard import native_tail_structured as s
from roguard.native_tail import DetailedPrediction, NativeTail
from roguard.v10_closure import NativeTailClosure


def payload(lang='ro', text='Eu văd un exemplu.'):
    return {'language': lang, 'turns': [{'role': 'assistant', 'text': 'Context neutru.'},
                                      {'role': 'minor', 'text': text}]}


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    assets = {}; calls = []; state = {'fail': False, 'mutate': None, 'closing': False}
    def add(name, raw):
        path = tmp_path / (name + '.json'); path.write_bytes(raw); assets[name] = path
    for lang in ('ro', 'uk'):
        raw = (Path(__file__).parent / 'fixtures/role-residual-specs' / (lang + '-v9.json')).read_bytes()
        add(lang + '_spec9', raw)
        v = json.loads(raw); v['adapter_sha256'] = 'a' * 64
        add(lang + '_native_spec', json.dumps(v).encode()); add(lang + '_manifest', b'{}')
    class Selector:
        def __init__(self): self.identity = s.SELECTOR_SHA; self.spec = {'min_relevance': None}
        def relevance(self, text, language, role): return 1.
    selector_spec = {'min_relevance': None}; add('selector', gzip.compress(json.dumps(selector_spec).encode(), mtime=0))
    selector_sha = hashlib.sha256(assets['selector'].read_bytes()).hexdigest()
    monkeypatch.setattr(s, 'SELECTOR_SHA', selector_sha); monkeypatch.setattr(s, 'TrajectoryModel', Selector)
    closure = object.__new__(NativeTailClosure); closure.receipt = {'synthetic': 'approved-test-fixture'}
    closure.identity = s.digest(closure.receipt)
    closure.asset_path = lambda name: assets[name]
    def asset_sha(name):
        if name.endswith('_adapter9'): return s.WEIGHTS9[name[:2]]
        if name.endswith('_native_adapter'): return 'a' * 64
        return hashlib.sha256(assets[name].read_bytes()).hexdigest()
    closure.asset_sha = asset_sha
    def verify():
        if state['closing']: raise ValueError('synthetic closing asset mutation')
        return True
    closure.verify = closure.cheap_verify = verify
    components = {}
    for lang in ('ro', 'uk'):
        component = object.__new__(NativeTail); component.language = lang
        component._closure = types.SimpleNamespace(parent=closure, language=lang)
        component._parents = {route: types.SimpleNamespace(identity=asset_sha(lang + '_' + key),
            _spec=json.loads(assets[lang + '_' + key].read_bytes()))
            for route, key in [('native_tail', 'native_spec'), ('original_v9', 'spec9')]}
        components[lang] = component
    def native_batch(component, payloads, *, focuses):
        calls.append((component.language, copy.deepcopy(payloads), list(focuses)))
        if state['fail']: raise RuntimeError('native batch failed')
        details = []
        for value, focus in zip(payloads, focuses):
            route = 'original_v9_unsupported' if 'UNSUPPORTED' in value['turns'][-1]['text'] else 'original_v9' if 'FALLBACK' in value['turns'][-1]['text'] else 'native_tail'
            parent = component._parents['native_tail' if route == 'native_tail' else 'original_v9']
            z = None if route == 'original_v9_unsupported' else tuple(float(v) for v in np.linspace(-.5, .8, 19, dtype=np.float32))
            if z is not None: z = (-0., *z[1:])
            score = None if z is None else float(s._score(z, parent._spec)); pred = None if score is None else int(score >= parent._spec['threshold'])
            response = dict(format='roguard-gemma4-tail-review-v1', language=component.language, focus=focus,
                input_digest=s.input_identity(value), model_identity=parent.identity, score=score, predicted=pred,
                decision='clarify' if pred is None else 'review' if pred else 'no_signal',
                reason='INPUT_EXCEEDS_MODEL_CONTEXT' if pred is None else 'MODEL_REVIEW_PROPOSAL' if pred else 'NO_SIGNAL_DETECTED',
                review_requested=pred != 0, score_is_calibrated=False, text_in_report=False, establishes_safety=False,
                permits_external_action=False, permits_notification=False)
            detail = DetailedPrediction(response, z, route, parent.identity)
            details.append(state['mutate'](detail) if state['mutate'] else detail)
        if state.get('late'): state['closing'] = True
        return details
    monkeypatch.setattr(NativeTail, 'predict_detailed_many', native_batch)
    def make(lang='ro', mode='direct-first'):
        return s.NativeTailStructuredWrapper({lang: components[lang]}, Selector(), closure,
                                            active_language=lang, mode=mode)
    return make, calls, state, components, closure


def test_genuine96_batches_preserve_order_and_serial_outputs(fixture):
    make, calls, *_ = fixture; wrapper = make()
    requests = [dict(payload=payload(text=f'Exemplul {i}.')) for i in range(193)]
    batch = wrapper.predict_many(requests)
    assert [len(c[1]) for c in calls] == [96, 96, 1]
    calls.clear()
    serial = [wrapper.predict(r['payload']) for r in requests[:3]]
    assert serial == batch[:3]
    assert all(r['closure_verification']['state'] == 'PASS' and r['qualified'] is False for r in batch)
    assert batch[0]['field_logits'][0].hex() == (-0.).hex()


def test_uniform_unicode_evidence_focus_and_single_batch(fixture):
    make, calls, *_ = fixture; wrapper = make(mode='uniform')
    p = payload(text='  Eu va\u0306d 👩🏽‍💻.\nA doua propoziție!  ')
    p['turns'].append({'role': 'minor', 'text': 'Viitorul nu se citește.'}); saved = copy.deepcopy(p)
    out = wrapper.predict_many([dict(payload=p, focus=1), dict(payload=payload())])
    assert len(calls) == 1 and len(calls[0][1]) == 2 and p == saved
    assert all(r['used_memory'] for r in out)
    assert 'Viitorul' not in json.dumps(calls)
    for segment in out[0]['inference_to_original_segments']:
        original = p['turns'][segment['source_turn']]['text'][segment['source_start']:segment['source_end']]
        inferred = calls[0][1][0]['turns'][segment['inference_turn']]['text'][segment['inference_start']:segment['inference_end']]
        assert original == inferred


def test_unsupported_retry_and_memory_overflow_no_fake_confidence(fixture):
    make, calls, *_ = fixture; wrapper = make()
    r = wrapper.predict(payload(text='UNSUPPORTED'))
    assert len(calls) == 2 and r['direct_unsupported_response']['predicted'] is None
    assert r['predicted'] is None and len(r['interpretation_set']['hypotheses']) == 432
    calls.clear(); uniform = make(mode='uniform'); r = uniform.predict(payload(text='x' * 1300))
    assert calls == [] and r['route'] == 'memory_unsupported' and r['evidence'] == []
    assert len(r['interpretation_set']['hypotheses']) == 432


def test_uk_legacy_binary_remains_separate_and_bilingual_identity(fixture):
    make, calls, *_ = fixture; ro = make(); uk = make('uk')
    assert ro.identity == uk.identity
    a = ro.predict(payload()); b = uk.predict(payload('uk', 'FALLBACK'))
    assert a['semantic_score_relation']['score_mass_equivalence_required'] is True
    assert b['semantic_score_relation']['score_mass_equivalence_required'] is False
    assert b['route'] == 'original_v9' and b['score'] == b['native_response']['score']
    assert abs(sum(b['probabilities']) - 1) < 1e-14


def test_errors_and_late_mutation_fail_entire_batch(fixture):
    make, calls, state, *_ = fixture; wrapper = make()
    state['fail'] = True
    with pytest.raises(RuntimeError, match='native batch failed'): wrapper.predict_many([dict(payload=payload())] * 2)
    assert len(calls) == 1 and not wrapper._active
    state['fail'] = False; state['late'] = True
    with pytest.raises(BaseExceptionGroup, match='Structured inference and closing failed') as error:
        wrapper.predict(payload())
    assert all('closing asset mutation' in str(e) for e in error.value.exceptions)
    assert not wrapper._active


@pytest.mark.parametrize('mutation', [
    lambda d: d._replace(spec_sha256='0' * 64),
    lambda d: d._replace(logits=list(d.logits)),
    lambda d: d._replace(response=dict(d.response, permits_notification=True)),
    lambda d: d._replace(response=dict(d.response, input_digest='0' * 64)),
])
def test_same_call_full_record_must_match(fixture, mutation):
    make, calls, state, *_ = fixture; wrapper = make(); state['mutate'] = mutation
    with pytest.raises(ValueError): wrapper.predict(payload())
    assert len(calls) == 1


def test_calibration_bound_to_mode_and_configuration_mutation(fixture):
    make, calls, *_ = fixture; wrapper = make(mode='uniform')
    artifact = dict(format='roguard-joint-calibration-v10-research2', pipeline_digest=wrapper.identity,
                    policy_digest=s.policy_digest(), probability_floors={'ro': 0., 'uk': 0.})
    artifact['digest'] = s.digest(artifact)
    assert len(wrapper.predict(payload(), calibration=artifact)['interpretation_set']['hypotheses']) == 432
    with pytest.raises(ValueError): make().predict(payload(), calibration=artifact)
    calls.clear(); wrapper.mode = 'direct-first'
    with pytest.raises(ValueError, match='configuration'): wrapper.predict(payload())
    assert not calls


def test_inactive_language_rejected_before_native(fixture):
    make, calls, *_ = fixture
    with pytest.raises(s.InactiveLanguageError): make().predict_many([dict(payload=payload()), dict(payload=payload('uk'))])
    assert not calls
