"""The diagnostic must preserve native execution and restore instrumentation."""
import pytest

from eval.diagnose_review_execution_layers_v10 import observe_layers, selected_cases


def test_layer_observation_restores_native_calls_after_failure():
    class Layer:
        def __call__(self, value):
            return value+1
    layers = [Layer(), Layer()]
    original = Layer.__call__
    observed = []
    with pytest.raises(RuntimeError, match='probe failed'):
        with observe_layers(layers, lambda i, result: observed.append((i, result))):
            assert layers[1](4) == 5
            assert Layer()(8) == 9
            raise RuntimeError('probe failed')
    assert observed == [(1, 5)]
    assert Layer.__call__ is original
    assert layers[0](2) == 3


def test_selection_preserves_control_and_outliers_without_train_leakage():
    rows = [dict(feature_index=i, partition='dev', paths={
        name: dict(feature_error=float(i), logit_error=float(9-i))
        for name in ('prefix', 'uncached')}) for i in range(10)]
    rows[4]['paths']['prefix']['full_replay_error'] = 100.
    rows.append(dict(feature_index=100, partition='train', paths={}))
    assert [r['feature_index'] for r in selected_cases(rows)] == [0, 1, 4, 8, 9]
    with pytest.raises(ValueError, match='duplicate'):
        selected_cases(rows+[rows[0]])


def test_native_gemma_instrumentation_is_transparent_on_cpu():
    mx = pytest.importorskip('mlx.core')
    np = pytest.importorskip('numpy')
    from mlx_lm.models.gemma4_text import Model, ModelArgs
    from mlx_lm.models.cache import make_prompt_cache
    from roguard.gemma4_attention import capture, capture_with_prefix, replay
    from copy import deepcopy
    mx.set_default_device(mx.cpu); mx.random.seed(1084)
    model = Model(ModelArgs(hidden_size=16, num_hidden_layers=4, intermediate_size=32,
        num_attention_heads=2, head_dim=8, global_head_dim=8, num_key_value_heads=1,
        num_kv_shared_layers=2, vocab_size=64, vocab_size_per_layer_input=64,
        hidden_size_per_layer_input=8, sliding_window=8,
        layer_types=['sliding_attention', 'full_attention']*2))
    model.eval()
    tokens = mx.array([[1, 2, 3, 4, 5, 6]])
    prefix = make_prompt_cache(model)
    mx.eval(model.model(tokens[:, :3], cache=prefix))
    original = type(model.model.layers[0]).__call__
    for cached in (False, True):
        def run():
            return model.model(tokens[:, 3:] if cached else tokens,
                               cache=deepcopy(prefix) if cached else None)
        expected = run(); mx.eval(expected)
        seen = []
        def observe(index, result):
            hidden, kv, offset = result
            mx.eval(hidden, kv)
            seen.append((index, np.asarray(hidden[:, -1, :]).copy(),
                         np.asarray(kv[0][:, :, 2, :]).copy(), offset))
        with observe_layers(model.model.layers, observe):
            actual = run(); mx.eval(actual)
        assert type(model.model.layers[0]).__call__ is original
        assert [item[0] for item in seen] == [0, 1, 2, 3]
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)
        state = (capture_with_prefix(model, tokens[:, 3:], prefix, start=2) if cached
                 else capture(model, tokens, start=2))
        mx.eval(state)
        baseline = replay(model, state); mx.eval(baseline)
        seen.clear()
        with observe_layers(model.model.layers, observe):
            traced = replay(model, state); mx.eval(traced)
        assert type(model.model.layers[0]).__call__ is original
        assert [item[0] for item in seen] == [2, 3]
        np.testing.assert_allclose(traced, baseline, atol=1e-6, rtol=0)
        np.testing.assert_allclose(traced[:, 0, :], expected[:, -1, :], atol=1e-5, rtol=0)
