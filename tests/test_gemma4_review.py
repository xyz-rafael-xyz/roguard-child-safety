import hashlib
import json

import pytest

from roguard.assurance import policy_digest
from roguard.gemma4_contract import BASE_MODEL, BASE_REVISION, POLICY_SHA256
from roguard.gemma4_review import FORMAT, LORA, Gemma4ReviewModel


def artifact(tmp_path, **changes):
    spec = dict(format=FORMAT, language='ro', base_model=BASE_MODEL, base_revision=BASE_REVISION,
        policy_sha256=POLICY_SHA256, policy_digest=policy_digest(), adapter_sha256='0'*64,
        mode='binary', couplings=None, threshold=.5, boundary=24, lora=LORA)
    spec.update(changes)
    path = tmp_path/'review.json'
    path.write_text(json.dumps(spec))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_review_validates_whole_batch_and_binds_selected_prefix_without_text(tmp_path, monkeypatch):
    import roguard.gemma4_review as module
    path, expected = artifact(tmp_path)
    calls = []
    class Engine:
        def __init__(self, *args):
            calls.append('load')
        def scores(self, prefixes, spec):
            calls.append(prefixes)
            return [.8, None][:len(prefixes)]
    monkeypatch.setattr(module, '_Engine', Engine)
    model = Gemma4ReviewModel(path, tmp_path/'adapter', tmp_path/'base', expected_sha256=expected)
    payload = dict(language='ro', turns=[dict(role='minor', text='Aș vrea să vorbesc cu cineva.'), {'invalid': 'future'}])
    with pytest.raises(ValueError):
        model.predict_many([payload, dict(language='uk', turns=[dict(role='minor', text='Привіт.')])], focuses=[0, 0])
    assert calls == []
    output = model.predict_many([payload, payload], focuses=[0, 0])
    assert len(calls[-1][0]['turns']) == 1
    assert payload['turns'][1] == {'invalid': 'future'}
    assert output[0]['predicted'] == 1 and output[1]['predicted'] is None
    assert output[0]['input_digest'] == output[1]['input_digest']
    assert output[1]['review_requested'] is True and output[1]['decision'] == 'clarify'
    assert all(row['model_identity'] == expected and row['focus'] == 0 for row in output)
    for row in output:
        assert all(row[key] is False for key in ('text_in_report', 'establishes_safety', 'permits_external_action', 'permits_notification', 'score_is_calibrated'))
        assert 'vorbesc' not in json.dumps(row)
    with pytest.raises(ValueError):
        Gemma4ReviewModel(path, tmp_path/'adapter', tmp_path/'base', expected_sha256='f'*64)


@pytest.mark.parametrize('change', [dict(mode='unknown'), dict(threshold=float('nan')), dict(language='en'),
    dict(boundary=True), dict(mode='coupled_policy'), dict(policy_digest='f'*64), dict(adapter_sha256='bad')])
def test_review_rejects_invalid_contract_before_loading(tmp_path, change):
    path, expected = artifact(tmp_path, **change)
    with pytest.raises(ValueError):
        Gemma4ReviewModel(path, tmp_path/'adapter', tmp_path/'base', expected_sha256=expected)


def test_portable_policy_scores_match_research_for_all_modes():
    np = pytest.importorskip('numpy')
    from roguard.gemma4_review import _score
    from roguard.native_fields import SIZES
    from training.shared_tail_qualification_v9 import scores_from_logits
    rng = np.random.default_rng(166)
    logits = rng.normal(size=(64, sum(SIZES)+2))*5
    couplings = dict(scope=np.exp(rng.normal(size=(3, 3))).tolist(), assertion_relation=np.exp(rng.normal(size=(4, 2))).tolist())
    for mode in ('binary', 'policy', 'coupled_policy'):
        spec = dict(mode=mode, couplings=couplings)
        assert np.max(np.abs(np.array([_score(row, spec) for row in logits])-scores_from_logits(logits, spec))) < 1e-12


def test_runtime_prompt_matches_frozen_training_and_ignores_object_key_order():
    from roguard.gemma4_contract import classifier_tokens, POLICY
    from training.gemma4_prompt_v9 import classifier_tokens as training_tokens
    from training.probe_instruction_v8 import POLICY_V2
    assert POLICY == POLICY_V2
    class Tokenizer:
        def encode(self, text, **kwargs):
            return [ord(text)]
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs['enable_thinking'] is False
            return list(json.dumps(messages, ensure_ascii=False).encode())
    tokenizer = Tokenizer()
    row = dict(language='ro', turns=[dict(role='minor', text='s\u0326tiu'), dict(role='minor', text='Acum vorbesc despre mine.')], focus=1)
    expected = training_tokens(tokenizer, row, [48, 49])
    reordered = dict(language='ro', turns=[dict(text=t['text'], role=t['role']) for t in row['turns']])
    assert classifier_tokens(tokenizer, reordered) == expected


def test_runtime_adapter_and_prefix_path_match_full_trained_model():
    mx = pytest.importorskip('mlx.core')
    import numpy as np
    from copy import deepcopy
    from mlx.utils import tree_flatten
    from mlx_lm.models.gemma4_text import Model, ModelArgs
    from mlx_lm.models.cache import make_prompt_cache
    from roguard.gemma4_attention import capture_with_prefix, replay
    from roguard.gemma4_review import _make_model, _load_weights
    from roguard.native_fields import SIZES
    from training.shared_tail_training_v9 import make_classifier
    mx.set_default_device(mx.cpu)
    mx.random.seed(166)
    base = Model(ModelArgs(hidden_size=16, num_hidden_layers=4, intermediate_size=32,
        num_attention_heads=2, head_dim=8, global_head_dim=8, num_key_value_heads=1,
        num_kv_shared_layers=2, vocab_size=64, vocab_size_per_layer_input=64,
        hidden_size_per_layer_input=8, sliding_window=4,
        layer_types=['sliding_attention', 'full_attention']*2))
    independent = deepcopy(base)
    rng = np.random.default_rng(166)
    head = {key: rng.normal(0, .2, shape).astype(np.float32) for key, shape in
        [('0.weight', (12, 16)), ('0.bias', (12,)), ('3.weight', (sum(SIZES)+2, 12)), ('3.bias', (sum(SIZES)+2,))]}
    lora = dict(rank=2, scale=4., dropout=0., keys=['self_attn.q_proj', 'self_attn.o_proj'])
    trained = make_classifier(base, {'ro': head, 'uk': head}, boundary=2, lora=lora)
    weights = {key: np.asarray(value).copy() for key, value in tree_flatten(trained.trainable_parameters())}
    for key, value in weights.items():
        if key.endswith('.lora_b'):
            weights[key] = rng.normal(0, .01, value.shape).astype(np.float32)
    trained.load_weights([(key, mx.array(value)) for key, value in weights.items()], strict=False)
    runtime = _make_model(independent, boundary=2, width=12, lora=lora)
    _load_weights(runtime, weights)
    cache = make_prompt_cache(runtime.base)
    mx.eval(runtime.base.model(mx.array([[1, 2]]), cache=cache))
    for language, tokens in [('ro', [1, 2, 3]), ('uk', [1, 2, 3, 4, 5, 6])]:
        state = capture_with_prefix(runtime.base, mx.array([tokens[2:]]), cache, start=2)
        actual = runtime.heads[language](replay(runtime.base, state)[:, 0, :])
        expected = trained.heads[language](trained.base.model(mx.array([tokens]))[:, -1, :])
        mx.eval(actual, expected)
        assert float(mx.max(mx.abs(actual-expected)).item()) < 2e-5
    with pytest.raises(ValueError, match='inventory'):
        _load_weights(runtime, {})
    bad = dict(weights)
    key = next(iter(bad))
    bad[key] = bad[key].astype(np.float64)
    with pytest.raises(ValueError, match='tensor'):
        _load_weights(runtime, bad)


def test_bundle_dispatch_requires_supported_device_and_local_assets(tmp_path):
    from roguard.model_bundle import load_bundle
    path, artifact_hash = artifact(tmp_path)
    bundle = tmp_path/'bundle.json'
    spec = dict(format='roguard-gemma4-tail-review-bundle-v1', artifact_sha256=artifact_hash,
        assets=dict(artifact=path.name, adapter='adapter.safetensors', base='base'))
    bundle.write_text(json.dumps(spec))
    expected = hashlib.sha256(bundle.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='Apple Silicon'):
        load_bundle(bundle, expected_sha256=expected, device='cpu')
    model = load_bundle(bundle, expected_sha256=expected, device='mps')
    assert model.identity == artifact_hash and model._engine is None
    spec['assets']['adapter'] = '../escape.safetensors'
    bundle.write_text(json.dumps(spec))
    with pytest.raises(ValueError):
        load_bundle(bundle, expected_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(), device='mps')
