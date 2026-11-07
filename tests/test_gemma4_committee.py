"""Committee input, disagreement, state sharing and failure recovery checks."""
from copy import deepcopy
import hashlib
import itertools
import json
from types import SimpleNamespace

import pytest

from roguard.assurance import policy_digest
from roguard.gemma4_contract import BASE_MODEL, BASE_REVISION, POLICY_SHA256
from roguard.gemma4_review import FORMAT, LORA, Gemma4ReviewModel
from roguard.gemma4_committee import Gemma4ReviewCommittee, COMPONENTS
import roguard.gemma4_committee as committee_module


def configurations(tmp_path, language='ro'):
    models = {}
    for i, name in enumerate(COMPONENTS):
        spec = {'format': FORMAT, 'language': language, 'base_model': BASE_MODEL,
            'base_revision': BASE_REVISION, 'policy_sha256': POLICY_SHA256,
            'policy_digest': policy_digest(), 'adapter_sha256': str(i+1)*64,
            'mode': 'binary', 'couplings': None, 'threshold': .5, 'boundary': 24, 'lora': LORA}
        path = tmp_path/(language+'-'+name+'.json'); path.write_text(json.dumps(spec))
        models[name] = Gemma4ReviewModel(path, tmp_path/(name+'.safetensors'), tmp_path/'base',
                                       expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return models


def component_rows(votes):
    return {name: {'score': .8 if vote else .2, 'predicted': vote,
                   'field_logits': [0.]*18+[.8 if vote else .2]} for name, vote in zip(COMPONENTS, votes)}


def fake_engine(monkeypatch, output):
    calls = []
    class Engine:
        def __init__(self, *args):
            calls.append('load')
        def scores(self, prefixes, specs):
            calls.append(deepcopy(prefixes))
            return deepcopy(output)
    monkeypatch.setattr(committee_module, '_CommitteeEngine', Engine)
    # Isolate orchestration from the optional numerical backend in portable CI.
    monkeypatch.setattr(committee_module, '_score', lambda z, spec: z[-1])
    return calls


def test_all_votes_and_disagreement_are_exposed_without_authority(tmp_path, monkeypatch):
    combinations = list(itertools.product((0, 1), repeat=3))
    calls = fake_engine(monkeypatch, [component_rows(v) for v in combinations])
    model = Gemma4ReviewCommittee(**configurations(tmp_path))
    payload = {'language': 'ro', 'turns': [{'role': 'minor', 'text': 'Un exemplu.'}]}
    results = model.predict_many([payload]*8)
    assert calls[0] == 'load'
    for votes, row in zip(combinations, results):
        assert row['predicted'] == int(all(votes))
        assert row['review_requested'] == bool(all(votes))
        assert row['component_disagreement'] == (len(set(votes)) > 1)
        assert row['score_kind'] == 'binary_unanimity_indicator'
        assert row['model_identity'] == model.identity
        assert all(row[k] is False for k in ('score_is_calibrated', 'qualified', 'text_in_report',
                                            'establishes_safety', 'permits_external_action', 'permits_notification'))
        assert 'Un exemplu' not in json.dumps(row)


def test_focus_is_bound_and_token_overflow_abstains(tmp_path, monkeypatch):
    calls = fake_engine(monkeypatch, [None])
    model = Gemma4ReviewCommittee(**configurations(tmp_path))
    payload = {'language': 'ro', 'turns': [{'role': 'minor', 'text': 'Acum.'}, {'invalid': 'future'}]}
    result = model.predict(payload, focus=0)
    assert calls[1] == [{'language': 'ro', 'turns': [{'role': 'minor', 'text': 'Acum.'}]}]
    assert result['focus'] == 0 and result['predicted'] is None and result['review_requested'] is True
    assert result['components'] is None and result['component_disagreement'] is None
    assert result['reason'] == 'INPUT_EXCEEDS_MODEL_CONTEXT'
    assert payload['turns'][1] == {'invalid': 'future'}


def test_bad_batch_and_language_are_rejected_before_loading(tmp_path, monkeypatch):
    calls = fake_engine(monkeypatch, [])
    models = configurations(tmp_path); model = Gemma4ReviewCommittee(**models)
    payload = {'language': 'uk', 'turns': [{'role': 'minor', 'text': 'Приклад.'}]}
    for value, focuses in (([], None), ([payload], []), ([payload]*129, None), ([payload], None)):
        with pytest.raises(ValueError):
            model.predict_many(value, focuses=focuses)
    assert calls == []
    mixed = dict(models); mixed['attention'] = configurations(tmp_path, 'uk')['attention']
    with pytest.raises(ValueError):
        Gemma4ReviewCommittee(**mixed)


@pytest.mark.parametrize('problem', ['missing', 'nan', 'score', 'decision', 'boolean'])
def test_incomplete_or_inconsistent_component_output_is_rejected(tmp_path, monkeypatch, problem):
    result = component_rows((1, 1, 1))
    if problem == 'missing':
        del result['readout']
    elif problem == 'nan':
        result['readout']['score'] = float('nan')
    elif problem == 'score':
        result['readout']['score'] = .7
    elif problem == 'decision':
        result['readout']['predicted'] = 0
    else:
        result['readout']['predicted'] = True
    fake_engine(monkeypatch, [result])
    model = Gemma4ReviewCommittee(**configurations(tmp_path))
    with pytest.raises(ValueError):
        model.predict({'language': 'ro', 'turns': [{'role': 'minor', 'text': 'Exemplu.'}]})


def test_shared_encoder_and_inactive_heads_cannot_change():
    np = pytest.importorskip('numpy')
    original = {name: np.ones((2, 2), dtype=np.float32) for name in
                ('base.model.layers.24.self_attn.q_proj.lora_b',
                 'base.model.layers.23.self_attn.q_proj.lora_b',
                 'heads.ro.first.weight', 'heads.uk.first.weight')}
    readout = deepcopy(original); attention = deepcopy(original)
    readout['heads.ro.first.weight'] += 1
    attention['base.model.layers.24.self_attn.q_proj.lora_b'] += 1
    committee_module._validate_components(original, readout, attention, 'ro')
    for kind, name in (('readout', 'base.model.layers.24.self_attn.q_proj.lora_b'),
                       ('attention', 'base.model.layers.23.self_attn.q_proj.lora_b'),
                       ('attention', 'heads.uk.first.weight')):
        variants = {'readout': deepcopy(readout), 'attention': deepcopy(attention)}
        variants[kind][name] += 1
        with pytest.raises(ValueError):
            committee_module._validate_components(original, variants['readout'], variants['attention'], 'ro')


def test_shared_replay_matches_full_models_and_restores_after_failure(monkeypatch):
    mx = pytest.importorskip('mlx.core')
    np = pytest.importorskip('numpy')
    from mlx.utils import tree_flatten
    from mlx_lm.models.gemma4_text import Model, ModelArgs
    from mlx_lm.models.cache import make_prompt_cache
    from roguard.gemma4_review import _make_model, _load_weights, _score
    mx.set_default_device(mx.cpu); mx.random.seed(1028)
    base = Model(ModelArgs(hidden_size=16, num_hidden_layers=4, intermediate_size=32,
        num_attention_heads=2, head_dim=8, global_head_dim=8, num_key_value_heads=1,
        num_kv_shared_layers=2, vocab_size=64, vocab_size_per_layer_input=64,
        hidden_size_per_layer_input=8, sliding_window=4,
        layer_types=['sliding_attention', 'full_attention']*2))
    lora = dict(rank=2, scale=4., dropout=0., keys=['self_attn.q_proj', 'self_attn.o_proj'])
    model = _make_model(base, boundary=2, width=12, lora=lora)
    original = {k: np.asarray(v).copy() for k, v in tree_flatten(model.trainable_parameters())}
    readout = deepcopy(original); attention = deepcopy(original)
    readout['heads.ro.last.weight'][-1] += .1
    rng = np.random.default_rng(1028)
    for key, value in attention.items():
        if key.endswith('.lora_b'):
            attention[key] = rng.normal(0, .02, value.shape).astype(np.float32)
    attention['heads.ro.last.bias'][-1] += .3
    committee_module._validate_components(original, readout, attention, 'ro', boundary=2, layers=4)
    tensors = dict(v9=original, readout=readout, attention=attention)
    specs = {k: {'language': 'ro', 'mode': 'binary', 'threshold': .5} for k in COMPONENTS}
    tokens = [1, 2, 3, 4]; expected = {}
    for name in COMPONENTS:
        _load_weights(model, tensors[name])
        hidden = model.base.model(mx.array([tokens]))[:, -1, :]
        logits = model.heads['ro'](hidden); mx.eval(logits)
        expected[name] = np.asarray(logits)[0].copy()
    _load_weights(model, original)
    engine = committee_module._CommitteeEngine.__new__(committee_module._CommitteeEngine)
    engine.engine = SimpleNamespace(mx=mx, model=model, tokenizer=None, make_cache=make_prompt_cache)
    engine.tensors = tensors; engine.boundary = 2
    monkeypatch.setattr(committee_module, 'PREFIX_TOKENS', [1, 2])
    monkeypatch.setattr(committee_module, 'classifier_tokens', lambda tokenizer, p: p['tokens'])
    def verify_restored():
        assert all(np.array_equal(np.asarray(v), original[k]) for k, v in tree_flatten(model.trainable_parameters()))
    result = engine.scores([{'tokens': tokens}], specs)[0]
    for name in COMPONENTS:
        assert np.max(np.abs(np.array(result[name]['field_logits'])-expected[name])) < 1e-5
        assert abs(result[name]['score']-_score(expected[name], specs[name])) < 1e-6
    verify_restored()
    real_replay = committee_module.replay; calls = []
    def fail_attention(*args):
        calls.append(None)
        if len(calls) == 2:
            raise RuntimeError('injected attention inference failure')
        return real_replay(*args)
    monkeypatch.setattr(committee_module, 'replay', fail_attention)
    with pytest.raises(RuntimeError, match='injected'):
        engine.scores([{'tokens': tokens}], specs)
    verify_restored()
