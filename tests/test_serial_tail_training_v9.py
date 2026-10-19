import pytest


def test_training_uses_single_example_attention_and_preserves_joint_gradients(monkeypatch):
    mx = pytest.importorskip('mlx.core')
    np = pytest.importorskip('numpy')
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm.models.gemma4_text import Model, ModelArgs
    from training.gemma4_shared_tail_v9 import capture
    from training.native_fields_v9 import SIZES
    from training.serial_tail_training_v9 import make_classifier
    import training.gemma4_shared_tail_v9 as attention
    mx.set_default_device(mx.cpu)
    mx.random.seed(186)
    base = Model(ModelArgs(hidden_size=16, num_hidden_layers=6, intermediate_size=32,
        num_attention_heads=2, head_dim=8, global_head_dim=8, num_key_value_heads=1,
        num_kv_shared_layers=4, vocab_size=64, vocab_size_per_layer_input=64,
        hidden_size_per_layer_input=8, sliding_window=4,
        layer_types=['sliding_attention', 'full_attention']*3))
    states = [capture(base, mx.array([[1, 2, 3, i]]), start=2) for i in (4, 5, 6)]
    heads = {lang: {'0.weight': np.ones((3, 16), np.float32)*.1,
        '0.bias': np.zeros(3, np.float32), '3.weight': np.ones((sum(SIZES)+2, 3), np.float32)*.1,
        '3.bias': np.zeros(sum(SIZES)+2, np.float32)} for lang in ('ro', 'uk')}
    calls = []
    original = attention.replay
    def observed(model, state):
        calls.append(state['hidden'].shape[0])
        return original(model, state)
    monkeypatch.setattr(attention, 'replay', observed)
    model = make_classifier(base, heads, boundary=2,
        lora=dict(rank=2, scale=4., dropout=0., keys=['self_attn.q_proj', 'self_attn.o_proj']))
    model.train()
    train = model(states, ['ro']*3)
    mx.eval(train)
    assert calls == [1, 1, 1]
    calls.clear()
    model.eval()
    development = model(states, ['ro']*3)
    mx.eval(development)
    assert calls == [3]
    assert float(mx.max(mx.abs(train-development)).item()) < 1e-5
    model.train()
    def loss(m):
        values = m(states, ['ro']*3)
        # Include a cross-example term: serial attention must retain gradients
        # through the complete minibatch objective, without detaching examples.
        return mx.square(values[0]-values[1]).mean()+mx.square(values[2]-.2).mean()
    value, gradients = nn.value_and_grad(model, loss)(model)
    mx.eval(value, gradients)
    flat = dict(tree_flatten(gradients))
    assert all(bool(mx.all(mx.isfinite(g)).item()) for g in flat.values())
    assert any(float(mx.max(mx.abs(g)).item()) > 0 for name, g in flat.items() if '.lora_' in name)
