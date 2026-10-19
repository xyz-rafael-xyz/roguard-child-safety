import pytest


def setup_model():
    mx = pytest.importorskip('mlx.core')
    from mlx_lm.models.gemma4_text import Model, ModelArgs
    mx.set_default_device(mx.cpu)
    mx.random.seed(163)
    model = Model(ModelArgs(hidden_size=16, num_hidden_layers=6, intermediate_size=32,
        num_attention_heads=2, head_dim=8, global_head_dim=8, num_key_value_heads=1,
        num_kv_shared_layers=4, vocab_size=64, vocab_size_per_layer_input=64,
        hidden_size_per_layer_input=8, sliding_window=4,
        layer_types=['sliding_attention', 'full_attention']*3))
    return mx, model


@pytest.mark.parametrize('length,start', [(1, 2), (7, 2), (11, 4)])
def test_final_token_replay_matches_full_sequence_before_and_after_tail_changes(length, start):
    from training.gemma4_shared_tail_v9 import capture, replay
    mx, model = setup_model()
    tokens = mx.array([list(range(1, length+1)), list(range(2, length+2))])
    state = capture(model, tokens, start=start)
    full = model.model(tokens)[:, -1:, :]
    actual = replay(model, state)
    mx.eval(full, actual)
    assert float(mx.max(mx.abs(full-actual)).item()) < 2e-5
    for layer in model.model.layers[start:]:
        layer.self_attn.q_proj.weight = layer.self_attn.q_proj.weight+.01
        layer.mlp.down_proj.weight = layer.mlp.down_proj.weight-.02
    changed = model.model(tokens)[:, -1:, :]
    actual = replay(model, state)
    mx.eval(changed, actual)
    assert float(mx.max(mx.abs(changed-actual)).item()) < 2e-5


def test_shared_tail_gradients_match_full_sequence():
    from training.gemma4_shared_tail_v9 import capture, replay
    mx, model = setup_model()
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7]])
    model.freeze()
    for layer in model.model.layers[2:]:
        layer.unfreeze()
    state = capture(model, tokens, start=2)
    target = mx.arange(16)/16
    def full_loss(m):
        return mx.square(m.model(tokens)[:, -1:, :]-target).mean()
    def tail_loss(m):
        return mx.square(replay(m, state)-target).mean()
    full, full_grad = nn.value_and_grad(model, full_loss)(model)
    tail, tail_grad = nn.value_and_grad(model, tail_loss)(model)
    mx.eval(full, tail, full_grad, tail_grad)
    assert abs(float(full.item())-float(tail.item())) < 2e-5
    left, right = dict(tree_flatten(full_grad)), dict(tree_flatten(tail_grad))
    assert left.keys() == right.keys()
    assert left
    for key in left:
        assert float(mx.max(mx.abs(left[key]-right[key])).item()) < 2e-4, key


def test_rejects_tail_that_has_its_own_key_value_projection():
    from training.gemma4_shared_tail_v9 import capture
    mx, model = setup_model()
    with pytest.raises(ValueError, match='frozen prefix KV'):
        capture(model, mx.array([[1, 2]]), start=1)
