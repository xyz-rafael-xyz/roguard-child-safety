"""Exercise prefix reuse with real hybrid attention and recurrent MLX blocks."""
import pytest


@pytest.mark.parametrize('checkpoint', [False, True])
def test_cached_loss_and_gradients_survive_updates_and_repeated_inputs(checkpoint):
    mx = pytest.importorskip('mlx.core')
    nn = pytest.importorskip('mlx.nn')
    from mlx.utils import tree_flatten
    from mlx_lm.models.qwen3_5 import Model, ModelArgs
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from training.qwen_cached_loss_v9 import make_loss

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    layer_type = original_call = None
    try:
        mx.random.seed(994)
        model = Model(ModelArgs(model_type='qwen3_5', text_config=dict(
            model_type='qwen3_5_text', hidden_size=64, intermediate_size=128,
            num_hidden_layers=8, num_attention_heads=2, num_key_value_heads=1,
            head_dim=32, vocab_size=128, linear_num_value_heads=2,
            linear_num_key_heads=2, linear_key_head_dim=16,
            linear_value_head_dim=16, full_attention_interval=4)))
        model.freeze()
        linear_to_lora_layers(model, 1, dict(rank=2, scale=2., dropout=0.))
        if checkpoint:
            from mlx_lm.tuner.trainer import grad_checkpoint
            layer_type = type(model.layers[0])
            original_call = layer_type.__call__
            grad_checkpoint(model.layers[0])
        sequences = [[1, 2, 3, 4, 5, 0], [1, 2, 3, 8, 9, 11, 1]]
        cached, common = make_loss(model, sequences)
        assert common == 3

        def full(model, batch, lengths):
            for layer in model.layers[:-1]:
                layer.eval()
            position = lengths[:, 0] - 1
            hidden = model.language_model.model(batch[:, :-1])
            logits = model.language_model.lm_head(hidden[:, position[0], :])
            return nn.losses.cross_entropy(logits, batch[:, position[0] + 1], reduction='mean'), mx.array(1)

        import mlx.optimizers as optim
        optimizer = optim.SGD(learning_rate=.01)
        for index in [0, 1, 0, 1]:
            model.train()
            tokens = sequences[index]
            batch = mx.array([tokens])
            lengths = mx.array([[len(tokens) - 1, len(tokens)]])
            expected, gradient = nn.value_and_grad(model, full)(model, batch, lengths)
            actual, cached_gradient = nn.value_and_grad(model, cached)(model, batch, lengths)
            mx.eval(expected, actual, gradient, cached_gradient)
            assert abs(expected[0].item() - actual[0].item()) < 1e-5
            left, right = dict(tree_flatten(gradient)), dict(tree_flatten(cached_gradient))
            assert left.keys() == right.keys()
            for key in left:
                assert mx.max(mx.abs(left[key] - right[key])).item() < 1e-5, key
            optimizer.update(model, cached_gradient)
            mx.eval(model.parameters(), optimizer.state)
    finally:
        if layer_type is not None:
            layer_type.__call__ = original_call
        mx.set_default_device(previous)
