from copy import deepcopy
import pytest


@pytest.mark.parametrize('shared_layers',[0,2])
def test_variable_length_batch_matches_individual_full_context_without_mutating_prefix(shared_layers):
    mx = pytest.importorskip('mlx.core')
    from mlx_lm.models.gemma4_text import Model,ModelArgs
    from mlx_lm.models.cache import make_prompt_cache
    from training.gemma4_batch_features_v9 import batch_features
    mx.set_default_device(mx.cpu)
    mx.random.seed(143)
    model = Model(ModelArgs(hidden_size=16,num_hidden_layers=4,intermediate_size=32,
        num_attention_heads=2,head_dim=8,global_head_dim=8,num_key_value_heads=1,
        num_kv_shared_layers=shared_layers,vocab_size=64,vocab_size_per_layer_input=64,
        hidden_size_per_layer_input=8,sliding_window=4,
        layer_types=['sliding_attention','full_attention']*2))
    prefix = [1,2,3]
    tails = [[4],[4,5],[6,7],[6,7,8,9,10]]
    cache = make_prompt_cache(model)
    mx.eval(model.model(mx.array([prefix]),cache=cache))
    before = deepcopy(cache)
    actual = batch_features(model,tails,cache)
    expected = mx.concatenate([model.model(mx.array([prefix+row]))[:,-1,:] for row in tails])
    mx.eval(actual,expected)
    assert float(mx.max(mx.abs(actual-expected)).item()) < 1e-5
    for old,new in zip(before,cache):
        assert old.offset == new.offset
        for a,b in zip(old.state,new.state):
            assert bool(mx.array_equal(a,b).item())
    uncached = batch_features(model,[prefix+row for row in tails])
    mx.eval(uncached)
    assert float(mx.max(mx.abs(uncached-expected)).item()) < 1e-5
    with pytest.raises(ValueError):
        batch_features(model,[[]],cache)
    with pytest.raises(ValueError):
        batch_features(model,[[64]],cache)
