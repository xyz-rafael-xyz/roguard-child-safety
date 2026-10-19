from copy import deepcopy

import pytest


@pytest.mark.parametrize('continuation', [[4], [4, 5], [4, 5, 6, 7, 8]])
def test_common_prefix_capture_matches_full_sequence_without_mutation(continuation):
    mx = pytest.importorskip('mlx.core')
    from mlx_lm.models.gemma4_text import Model, ModelArgs
    from mlx_lm.models.cache import make_prompt_cache
    from training.gemma4_shared_prefix_v9 import capture_with_prefix
    from training.gemma4_shared_tail_v9 import replay
    mx.set_default_device(mx.cpu)
    mx.random.seed(164)
    model = Model(ModelArgs(hidden_size=16, num_hidden_layers=6, intermediate_size=32,
        num_attention_heads=2, head_dim=8, global_head_dim=8, num_key_value_heads=1,
        num_kv_shared_layers=4, vocab_size=64, vocab_size_per_layer_input=64,
        hidden_size_per_layer_input=8, sliding_window=4,
        layer_types=['sliding_attention', 'full_attention']*3))
    prefix = [1, 2, 3]
    cache = make_prompt_cache(model)
    mx.eval(model.model(mx.array([prefix]), cache=cache))
    original = deepcopy(cache)
    tails = [continuation, [v+1 for v in continuation]]
    state = capture_with_prefix(model, mx.array(tails), cache, start=2)
    actual = replay(model, state)
    expected = model.model(mx.array([prefix+v for v in tails]))[:, -1:, :]
    mx.eval(actual, expected)
    assert float(mx.max(mx.abs(actual-expected)).item()) < 2e-5
    assert state['length'] == len(prefix)+len(continuation)
    for old, new in zip(original, cache):
        assert old.meta_state == new.meta_state
        assert all(bool(mx.array_equal(a, b).item()) for a, b in zip(old.state, new.state))
    invalid = make_prompt_cache(model)
    mx.eval(model.model(mx.array([[1, 2, 3, 4]]), cache=invalid))
    with pytest.raises(ValueError, match='sliding window'):
        capture_with_prefix(model, mx.array([[5]]), invalid, start=2)
