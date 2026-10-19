import random

import pytest

from training.context_transition_v9 import generate
from training.shared_tail_training_v9 import plan_batches


def test_shared_tail_batches_keep_history_families_and_complete_coverage():
    rows = generate('ro', 'train')+generate('uk', 'train')
    batches = plan_batches(rows, random.Random(1165))
    assert sorted(i for batch, _, _ in batches for i in batch) == list(range(len(rows)))
    assert all(len(indices) <= 8 for indices, _, _ in batches)
    assert sum(len(pairs) for _, pairs, _ in batches) == 432
    assert sum(len(invariant) for _, _, invariant in batches) == 1728


def test_differentiable_readout_matches_portable_export_and_loss_has_finite_gradients():
    mx = pytest.importorskip('mlx.core')
    import numpy as np
    from training.gemma4_structured_v9 import portable_logits
    from training.native_fields_v9 import SIZES
    from training.shared_tail_training_v9 import make_head, supervised_loss
    mx.set_default_device(mx.cpu)
    rng = np.random.default_rng(1165)
    tensors = {key: rng.normal(0, .2, shape).astype(np.float32) for key, shape in
        [('0.weight', (12, 16)), ('0.bias', (12,)), ('3.weight', (sum(SIZES)+2, 12)), ('3.bias', (sum(SIZES)+2,))]}
    features = rng.normal(size=(4, 16)).astype(np.float32)
    actual = make_head(tensors)(mx.array(features))
    mx.eval(actual)
    assert np.max(np.abs(np.asarray(actual)-portable_logits(features, tensors))) < 2e-5
    y = mx.array([0, 1, 1, 1])
    fields = mx.zeros((4, len(SIZES)), dtype=mx.int32)
    weights = mx.ones((4,))
    def objective(values):
        return supervised_loss(values, y, fields, weights, [(0, 1)], [(2, 3)])
    loss, gradients = mx.value_and_grad(objective)(actual)
    mx.eval(loss, gradients)
    assert bool(mx.isfinite(loss).item())
    assert bool(mx.all(mx.isfinite(gradients)).item())
    equal = mx.repeat(actual[:1], 2, axis=0)
    first = supervised_loss(equal, y[:2], fields[:2], weights[:2], [], [])
    second = supervised_loss(equal, y[:2], fields[:2], weights[:2], [], [(0, 1)])
    assert abs(float(first.item())-float(second.item())) < 1e-6


def test_only_tail_adapters_and_language_heads_train_with_mixed_length_states():
    mx = pytest.importorskip('mlx.core')
    import mlx.nn as nn
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_lm.models.gemma4_text import Model, ModelArgs
    from training.gemma4_shared_tail_v9 import capture
    from training.native_fields_v9 import SIZES
    from training.shared_tail_training_v9 import make_classifier
    mx.set_default_device(mx.cpu)
    mx.random.seed(1165)
    base = Model(ModelArgs(hidden_size=16, num_hidden_layers=4, intermediate_size=32,
        num_attention_heads=2, head_dim=8, global_head_dim=8, num_key_value_heads=1,
        num_kv_shared_layers=2, vocab_size=64, vocab_size_per_layer_input=64,
        hidden_size_per_layer_input=8, sliding_window=4,
        layer_types=['sliding_attention', 'full_attention']*2))
    tokens = [[1, 2], [1, 2, 3, 4, 5], [2, 3]]
    languages = ['ro', 'uk', 'ro']
    states = [capture(base, mx.array([row]), start=2) for row in tokens]
    rng = np.random.default_rng(1165)
    head = {key: rng.normal(0, .2, shape).astype(np.float32) for key, shape in
        [('0.weight', (12, 16)), ('0.bias', (12,)), ('3.weight', (sum(SIZES)+2, 12)), ('3.bias', (sum(SIZES)+2,))]}
    model = make_classifier(base, {'ro': head, 'uk': head}, boundary=2,
        lora=dict(rank=2, scale=4., dropout=0., keys=['self_attn.q_proj', 'self_attn.o_proj']))
    names = dict(tree_flatten(model.trainable_parameters()))
    assert names and any(n.startswith('heads.ro.') for n in names) and any(n.startswith('heads.uk.') for n in names)
    assert all(n.startswith('heads.') or n.endswith(('.lora_a', '.lora_b')) for n in names)
    actual = model(states, languages)
    expected = mx.concatenate([model.heads[lang](model.base.model(mx.array([row]))[:, -1, :]) for row, lang in zip(tokens, languages)])
    mx.eval(actual, expected)
    assert float(mx.max(mx.abs(actual-expected)).item()) < 2e-5
    def objective(m):
        return nn.losses.cross_entropy(m(states, languages)[:, -2:], mx.array([0, 1, 0])).mean()
    loss, gradients = nn.value_and_grad(model, objective)(model)
    mx.eval(loss, gradients)
    assert all(bool(mx.all(mx.isfinite(v)).item()) for _, v in tree_flatten(gradients))
