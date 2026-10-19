import pytest


def test_portable_readout_matches_fitted_classifier_and_rejects_bad_features():
    np = pytest.importorskip('numpy')
    sklearn = pytest.importorskip('sklearn.linear_model')
    from training.gemma4_readout_v9 import normalize, scores
    rng = np.random.default_rng(141)
    x = rng.normal(size=(100, 12))
    y = (x[:, 0] - x[:, 1] > 0).astype(int)
    fitted = sklearn.LogisticRegression(C=.1, max_iter=500).fit(normalize(x), y)
    head = dict(weight=fitted.coef_[0].tolist(), intercept=float(fitted.intercept_[0]))
    assert np.max(np.abs(scores(x, head) - fitted.predict_proba(normalize(x))[:, 1])) < 1e-12
    with pytest.raises(ValueError):
        scores(np.zeros((2, 12)), head)
    with pytest.raises(ValueError):
        scores(np.full((2, 12), np.nan), head)
    with pytest.raises(ValueError):
        scores(x[:, :11], head)


def test_frozen_context_features_match_full_and_cached_gemma4():
    mx = pytest.importorskip('mlx.core')
    from mlx_lm.models.gemma4_text import Model, ModelArgs
    from mlx_lm.models.cache import make_prompt_cache
    from copy import deepcopy
    from training.gemma4_readout_v9 import last_features
    mx.set_default_device(mx.cpu)
    mx.random.seed(141)
    model = Model(ModelArgs(hidden_size=16, num_hidden_layers=2, intermediate_size=32,
        num_attention_heads=2, head_dim=8, global_head_dim=8, num_key_value_heads=1,
        num_kv_shared_layers=0, vocab_size=64, vocab_size_per_layer_input=64,
        hidden_size_per_layer_input=8, layer_types=['sliding_attention', 'full_attention']))
    tokens = mx.array([[1, 2, 3, 4]])
    full = model.model(tokens)[:, -1, :]
    cache = make_prompt_cache(model)
    mx.eval(model.model(tokens[:, :2], cache=cache))
    actual = last_features(model, tokens[:, 2:], cache=deepcopy(cache))
    mx.eval(full, actual)
    assert float(mx.max(mx.abs(full-actual)).item()) < 1e-5
