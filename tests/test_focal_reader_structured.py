"""Structured focal adapter tests: real routing/set arithmetic, fake native details."""
import copy
import gzip
import hashlib
import json
import types
from pathlib import Path

import pytest
np = pytest.importorskip('numpy')
from roguard import focal_reader_structured as s
from roguard import focal_reader_residual as f
from roguard.v10_closure import FocalReaderClosure


def payload(lang='ro', text='Eu văd un exemplu.'):
    return {'language': lang, 'turns': [
        {'role': 'assistant', 'text': 'Context neutru.'},
        {'role': 'minor', 'text': text}]}


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    assets = {}; calls = []; returned = []
    state = {'fail': False, 'mutate': None, 'closing': False,
             'reader_present': True, 'focal_present': True, 'batch_mutate': None}
    def add(name, raw):
        path = tmp_path / (name + '.json')
        path.write_bytes(raw); assets[name] = path
    for lang in ('ro', 'uk'):
        add(lang + '_spec9', (Path(__file__).parent / 'fixtures/role-residual-specs' /
                            (lang + '-v9.json')).read_bytes())
        for role in f.ARRAY_ROLES:
            add(lang + '_' + role, json.dumps({'synthetic': lang + role}).encode())
        manifest = dict(format=f.FORMAT, language=lang, qualified=False,
            parent_adapter_sha256=f.V9[lang], parent_spec_sha256=f.V9_SPEC[lang],
            arrays={role: hashlib.sha256(assets[lang + '_' + role].read_bytes()).hexdigest()
                    for role in f.ARRAY_ROLES}, fit_report_sha256='b' * 64,
            reader_report_sha256='c' * 64, feature_contract=copy.deepcopy(f.FEATURE_CONTRACT))
        add(lang + '_manifest', json.dumps(manifest).encode())
    class Selector:
        def __init__(self):
            self.identity = s.SELECTOR_SHA; self.spec = {'min_relevance': None}
        def relevance(self, text, language, role):
            return 1.
    add('selector', gzip.compress(json.dumps({'min_relevance': None}).encode(), mtime=0))
    selector_sha = hashlib.sha256(assets['selector'].read_bytes()).hexdigest()
    monkeypatch.setattr(s, 'SELECTOR_SHA', selector_sha)
    monkeypatch.setattr(s, 'TrajectoryModel', Selector)
    closure = object.__new__(FocalReaderClosure)
    closure.receipt = {'synthetic': 'externally-approved-test-inventory'}
    closure.identity = s.digest(closure.receipt)
    closure.asset_path = lambda name: assets[name]
    def asset_sha(name):
        if name.endswith('_adapter9'):
            return s.WEIGHTS9[name[:2]]
        return hashlib.sha256(assets[name].read_bytes()).hexdigest()
    closure.asset_sha = asset_sha
    state['verification_calls'] = 0
    def verify():
        state['verification_calls'] += 1
        if state['closing']:
            raise ValueError('synthetic closing asset mutation')
        return True
    closure.verify = closure.cheap_verify = verify
    components = {}
    for lang in ('ro', 'uk'):
        component = object.__new__(f.FocalReaderResidualModel)
        component.language = lang
        component.identity = asset_sha(lang + '_manifest')
        component._manifest = json.loads(assets[lang + '_manifest'].read_bytes())
        component._parent = types.SimpleNamespace(identity=asset_sha(lang + '_spec9'),
            _spec=json.loads(assets[lang + '_spec9'].read_bytes()))
        components[lang] = component
    def native_batch(component, payloads, *, focuses):
        calls.append((component.language, copy.deepcopy(payloads), list(focuses)))
        if state['fail']:
            raise RuntimeError('native detail batch failed')
        if state.get('reenter'):
            state['reenter']()
        details = []
        for value, focus in zip(payloads, focuses, strict=True):
            unsupported = ('UNSUPPORTED' in value['turns'][-1]['text'] or
                           (state.get('unsupported_first_batch') and len(calls) == 1))
            z = None if unsupported else tuple(float(v) for v in np.linspace(-.5, .8, 19, dtype=np.float32))
            if z is not None:
                # Distinct per-input evidence catches accidental batch reordering.
                marker = sum(ord(x) for x in value['turns'][-1]['text']) % 1024
                z = (-0., float(np.float32(marker / 1024)), *z[2:])
            score = None if z is None else float(s._score(z, component._parent._spec))
            pred = None if score is None else int(score >= component._parent._spec['threshold'])
            response = dict(format='roguard-canonical-focal-reader-review-v1',
                language=component.language, focus=focus, input_digest=s.input_identity(value),
                model_identity=component.identity, score=score, predicted=pred,
                decision='clarify' if pred is None else 'review' if pred else 'no_signal',
                reason='INPUT_EXCEEDS_MODEL_CONTEXT' if pred is None else 'MODEL_REVIEW_PROPOSAL' if pred else 'NO_SIGNAL_DETECTED',
                review_requested=pred != 0, score_is_calibrated=False, text_in_report=False,
                establishes_safety=False, permits_external_action=False, permits_notification=False)
            reader = None if unsupported else ((1., 2., 3., 4., 5., 1.)
                        if state['reader_present'] else (0.,) * 6)
            focal = None if unsupported else ((.25,) * 1024
                        if state['focal_present'] else (0.,) * 1024)
            detail = f.FocalReaderPrediction(response, z, reader, focal,
                None if unsupported else state['focal_present'])
            details.append(state['mutate'](detail) if state['mutate'] else detail)
        returned.extend(details)
        if state.get('late'):
            state['closing'] = True
        return state['batch_mutate'](details) if state['batch_mutate'] else details
    monkeypatch.setattr(f.FocalReaderResidualModel, 'predict_details', native_batch)
    def make(lang='ro', mode='direct-first'):
        return s.FocalReaderStructuredWrapper({lang: components[lang]}, Selector(), closure,
                                             active_language=lang, mode=mode)
    return make, calls, state, components, closure, returned


def test_real96_chunking_restores_order_and_preserves_signed_zero(fixture):
    make, calls, _, _, _, returned = fixture; wrapper = make()
    requests = [dict(payload=payload(text=f'Exemplul {i}.')) for i in range(193)]
    result = wrapper.predict_many(requests)
    assert [len(batch[1]) for batch in calls] == [96, 96, 1]
    assert [p['turns'][-1]['text'] for _, batch, _ in calls for p in batch] == [r['payload']['turns'][-1]['text'] for r in requests]
    assert [r['native_response']['input_digest'] for r in result] == [s.input_identity(r['payload']) for r in requests]
    assert [r['field_logits'] for r in result] == [r.logits for r in returned]
    calls.clear()
    assert [wrapper.predict(r['payload']) for r in requests[:3]] == result[:3]
    assert result[0]['field_logits'][0].hex() == (-0.).hex()
    assert all(r['closure_verification']['state'] == 'PASS' and not r['qualified'] for r in result)
    returned[0].response['permits_external_action'] = True
    assert result[0]['native_response']['permits_external_action'] is False


def test_ro_policy_mass_and_uk_binary_are_separate_same_call(fixture):
    make, *_ = fixture; ro = make(); uk = make('uk')
    assert ro.identity == uk.identity
    a = ro.predict(payload()); b = uk.predict(payload('uk', 'Український приклад.'))
    assert a['semantic_score_relation']['score_mass_equivalence_required'] is True
    assert abs(a['semantic_score_relation']['semantic_review_mass'] - a['score']) <= s.MASS_ATOL
    assert b['semantic_score_relation']['score_mass_equivalence_required'] is False
    assert b['semantic_score_relation']['semantic_vs_binary_difference'] > .01
    assert b['score'] == b['native_response']['score'] == float(s._score(b['field_logits'], uk.specs['uk']))
    assert abs(sum(b['probabilities']) - 1.) < 1e-14
    assert len(b['probabilities']) == 432


@pytest.mark.parametrize('reader_present,focal_present', [(False, True), (True, False), (False, False)])
def test_neutral_missing_blocks_do_not_disable_supported_prediction(fixture, reader_present, focal_present):
    make, calls, state, *_ = fixture
    state.update(reader_present=reader_present, focal_present=focal_present)
    result = make().predict(payload())
    assert len(calls) == 1 and result['field_logits'] is not None
    assert result['route'] == 'focal_reader' and result['predicted'] is not None
    assert result['probabilities_source'] == 'SAME_FORWARD_ROUTE_LOGITS'


def test_unicode_nonfinal_focus_exact_evidence_and_one_memory_batch(fixture):
    make, calls, *_ = fixture; wrapper = make(mode='uniform')
    p = payload(text='  Eu va\u0306d 👩🏽‍💻.\nA doua propoziție!  ')
    p['turns'].append(dict(role='minor', text='FUTURE_SENTINEL'))
    saved = copy.deepcopy(p)
    result = wrapper.predict_many([dict(payload=p, focus=1), dict(payload=payload())])
    assert len(calls) == 1 and len(calls[0][1]) == 2 and saved == p
    assert 'FUTURE_SENTINEL' not in json.dumps(calls)
    assert result[0]['focus'] == 1 and all(r['used_memory'] for r in result)
    for segment in result[0]['inference_to_original_segments']:
        original = p['turns'][segment['source_turn']]['text'][segment['source_start']:segment['source_end']]
        inferred = calls[0][1][0]['turns'][segment['inference_turn']]['text'][segment['inference_start']:segment['inference_end']]
        assert original == inferred
    assert result[0]['evidence_scope'] == 'retained_source_units_not_verified_semantic_support'


def test_unsupported_retry_and_overflow_return_full432_without_confidence(fixture):
    make, calls, *_ = fixture; wrapper = make()
    result = wrapper.predict(payload(text='UNSUPPORTED'))
    assert len(calls) == 2 and result['direct_unsupported_response']['predicted'] is None
    assert result['route'] == 'focal_reader_unsupported' and result['predicted'] is None
    assert result['score'] is None and len(result['interpretation_set']['hypotheses']) == 432
    assert result['probabilities_source'] == 'UNSUPPORTED_UNIFORM_PLACEHOLDER'
    calls.clear(); result = make(mode='uniform').predict(payload(text='x' * 1300))
    assert not calls and result['route'] == 'memory_unsupported' and result['evidence'] == []
    assert len(result['interpretation_set']['hypotheses']) == 432


def test_calibration_binds_exact_pipeline_and_memory_mode(fixture):
    make, calls, *_ = fixture; uniform = make(mode='uniform')
    artifact = dict(format='roguard-joint-calibration-v10-research2', pipeline_digest=uniform.identity,
                    policy_digest=s.policy_digest(), probability_floors={'ro': 0., 'uk': 0.})
    artifact['digest'] = s.digest(artifact)
    assert len(uniform.predict(payload(), calibration=artifact)['interpretation_set']['hypotheses']) == 432
    with pytest.raises(ValueError):
        make().predict(payload(), calibration=artifact)
    calls.clear(); uniform.mode = 'direct-first'
    with pytest.raises(ValueError, match='configuration'):
        uniform.predict(payload())
    assert not calls


@pytest.mark.parametrize('mutation', [
    lambda d: d._replace(logits=list(d.logits)),
    lambda d: d._replace(logits=(.1, *d.logits[1:])),
    lambda d: d._replace(reader_features=(1., 0., 0., 0., 0., 0.)),
    lambda d: d._replace(focal_present=False),
    lambda d: d._replace(focal_present=1),
    lambda d: d._replace(focal_mean=list(d.focal_mean)),
    lambda d: d._replace(response=dict(d.response, permits_notification=True)),
    lambda d: d._replace(response=dict(d.response, input_digest='0' * 64)),
    lambda d: d._replace(response=dict(d.response, model_identity='0' * 64)),
    lambda d: d._replace(response=dict(d.response, focus=0)),
    lambda d: d._replace(response=dict(d.response, score=d.response['score'] + 1e-12)),
])
def test_full_same_call_features_and_public_metadata_fail_closed(fixture, mutation):
    make, calls, state, *_ = fixture; wrapper = make(); state['mutate'] = mutation
    with pytest.raises(ValueError):
        wrapper.predict(payload())
    assert len(calls) == 1 and not wrapper._active


@pytest.mark.parametrize('change', [lambda d: d[:-1], lambda d: d + d[:1]])
def test_batch_cardinality_exact(fixture, change):
    make, calls, state, *_ = fixture; wrapper = make(); state['batch_mutate'] = change
    with pytest.raises(ValueError, match='Complete native batch'):
        wrapper.predict_many([dict(payload=payload()), dict(payload=payload())])
    assert len(calls) == 1 and not wrapper._active


def test_failure_closing_and_reentrancy_release_batch_state(fixture):
    make, calls, state, *_ = fixture; wrapper = make(); state['fail'] = True
    with pytest.raises(RuntimeError, match='native detail batch failed'):
        wrapper.predict_many([dict(payload=payload())] * 2)
    assert len(calls) == 1 and not wrapper._active
    state['fail'] = False; state['reenter'] = lambda: wrapper.predict(payload())
    with pytest.raises(ValueError, match='already active'):
        wrapper.predict(payload())
    assert not wrapper._active
    state.pop('reenter'); state['late'] = True
    with pytest.raises(BaseExceptionGroup, match='inference and closing failed') as caught:
        wrapper.predict(payload())
    assert len(caught.value.exceptions) == 2
    assert all('closing asset mutation' in str(e) for e in caught.value.exceptions)
    assert not wrapper._active


def test_inactive_language_rejected_before_any_component(fixture):
    make, calls, *_ = fixture
    with pytest.raises(s.InactiveLanguageError):
        make().predict_many([dict(payload=payload()), dict(payload=payload('uk'))])
    assert not calls


def test_component_or_parent_manifest_mutation_rejected_before_forward(fixture):
    make, calls, _, components, *_ = fixture; wrapper = make()
    components['ro']._manifest['fit_report_sha256'] = 'd' * 64
    with pytest.raises(ValueError, match='manifest differs'):
        wrapper.predict(payload())
    assert not calls


def test_unsupported_direct_can_recover_only_from_actual_memory_forward(fixture):
    make, calls, state, *_ = fixture; wrapper = make()
    state['unsupported_first_batch'] = True
    result = wrapper.predict(payload())
    assert len(calls) == 2 and result['used_memory'] is True
    assert result['direct_unsupported_response']['predicted'] is None
    assert result['route'] == 'focal_reader' and result['predicted'] is not None
    assert result['probabilities_source'] == 'SAME_FORWARD_ROUTE_LOGITS'
    assert result['inference_input_digest'] == s.input_identity(calls[1][1][0])
    assert result['native_response']['input_digest'] == result['inference_input_digest']
