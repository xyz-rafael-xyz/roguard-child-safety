"""Bound shared-KV response-tail computation for the registered research fit."""
import re

from roguard.gemma4_attention import _backbone, capture, capture_with_prefix, replay
from training.probe_assertion_query_v10 import tokens_for


def zero_tail_outputs(base, *, boundary=24):
    import mlx.core as mx
    body = _backbone(base, boundary)
    for layer in body.layers[boundary:]:
        for name in ('q_proj', 'o_proj'):
            projection = getattr(layer.self_attn, name)
            if not hasattr(projection, 'lora_a') or not hasattr(projection, 'lora_b'):
                raise ValueError('Expected existing query/output LoRA')
            projection.lora_b = mx.zeros_like(projection.lora_b)


def make_response_tail(base, head, *, boundary=24):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    class ResponseTail(nn.Module):
        def __init__(self):
            super().__init__(); self.base = base; self.head = head
        def __call__(self, states):
            if not isinstance(states, list) or not states:
                raise ValueError('Expected nonempty response states')
            return mx.concatenate([self.head(replay(self.base, state)[:, 0, :]) for state in states])
    model = ResponseTail(); body = _backbone(base, boundary)
    expected = {'head.last.weight', 'head.last.bias'} | {
        f'base.model.layers.{i}.self_attn.{projection}.lora_{factor}'
        for i in range(boundary, len(body.layers)) for projection in ('q_proj', 'o_proj') for factor in ('a', 'b')}
    actual = dict(tree_flatten(model.trainable_parameters()))
    if set(actual) != expected:
        raise ValueError('Response fit would change an unregistered parameter')
    if actual['head.last.weight'].shape[0] != 5 or actual['head.last.bias'].shape != (5,):
        raise ValueError('Expected a five-category response head')
    return model


def response_state(engine, query, payload):
    """Capture only frozen layers; do not zero the adaptable tail during replay."""
    mx = engine.mx
    tokens, _ = tokens_for(engine.tokenizer, query.prompt, payload)
    if not tokens or len(tokens) > 1024:
        raise ValueError('Response query exceeds the registered envelope')
    common = len(query.prefix_tokens)
    if len(tokens) > common and tokens[:common] == query.prefix_tokens:
        state = capture_with_prefix(engine.model.base, mx.array([tokens[common:]]), query.prefix, start=24)
    else:
        state = capture(engine.model.base, mx.array([tokens]), start=24)
    mx.eval(state)
    return state, tokens


def load_response_weights(model, tensors):
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten
    expected = dict(tree_flatten(model.trainable_parameters()))
    if set(tensors) != set(expected) or any(
            value.shape != expected[name].shape or value.dtype != np.float32 or not np.isfinite(value).all()
            for name, value in tensors.items()):
        raise ValueError('Invalid response-tail checkpoint inventory')
    model.load_weights([(name, mx.array(value)) for name, value in tensors.items()], strict=False)


def make_response_optimizer(config):
    """Keep the optional head learning rate separate from attention updates."""
    import mlx.optimizers as optim
    tail = optim.AdamW(learning_rate=config['learning_rate'], weight_decay=config['weight_decay'])
    if 'head_learning_rate' not in config:
        return tail, [(tail, config['learning_rate'])]
    head = optim.AdamW(learning_rate=config['head_learning_rate'], weight_decay=config['weight_decay'])
    optimizer = optim.MultiOptimizer([head, tail], [lambda name, _: name.startswith('head.')])
    return optimizer, [(head, config['head_learning_rate']), (tail, config['learning_rate'])]
