"""Capture shared-tail states while reusing an immutable common prompt prefix."""
from copy import deepcopy

from training.gemma4_shared_tail_v9 import _backbone


def capture_with_prefix(model, tokens, prefix_cache, *, start):
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache, RotatingKVCache
    body = _backbone(model, start)
    if tokens.ndim != 2 or not tokens.shape[0] or not tokens.shape[1]:
        raise ValueError('Expected nonempty equal-length continuations')
    if not prefix_cache or any(type(v) not in (KVCache, RotatingKVCache) or v.empty() for v in prefix_cache):
        raise ValueError('Expected populated native prefix caches')
    offsets = {v.offset for v in prefix_cache}
    if len(offsets) != 1:
        raise ValueError('Prefix positions differ across layers')
    offset = next(iter(offsets))
    if not 0 < offset < body.window_size:
        raise ValueError('Common prefix must fit inside the sliding window')
    first_shared = len(body.layers)-body.config.num_kv_shared_layers
    if len(prefix_cache) != first_shared:
        raise ValueError('Prefix cache layer count differs')
    cache = deepcopy(prefix_cache)
    for layer in cache:
        keys, values = layer.state
        if keys.shape[0] != 1 or values.shape[0] != 1 or keys.shape[2] != offset or values.shape[2] != offset:
            raise ValueError('Expected one untrimmed prefix')
        layer.state = (mx.repeat(keys, tokens.shape[0], axis=0), mx.repeat(values, tokens.shape[0], axis=0))
    cache += [None]*(len(body.layers)-len(cache))
    h = body.embed_tokens(tokens)*body.embed_scale
    if body.hidden_size_per_layer_input:
        per_layer = body._project_per_layer_inputs(h, body._get_per_layer_inputs(tokens))
        per_layer = [per_layer[:, :, i, :] for i in range(len(body.layers))]
    else:
        per_layer = [None]*len(body.layers)
    masks = body._make_masks(h, cache)
    intermediates = [(None, None)]*len(body.layers)
    for index in range(start):
        kvs, position = intermediates[body.previous_kvs[index]]
        h, kvs, position = body.layers[index](h, masks[index], cache[index],
            per_layer_input=per_layer[index], shared_kv=kvs, offset=position)
        intermediates[index] = (kvs, position)
    shared = {index: tuple(mx.stop_gradient(v) for v in intermediates[index][0]) for index in set(body.previous_kvs[start:])}
    last_masks = []
    for mask in masks[start:]:
        if isinstance(mask, str):
            if mask != 'causal':
                raise ValueError('Unknown attention mask')
            last_masks.append(None)
        else:
            last_masks.append(None if mask is None else mx.stop_gradient(mask[..., -1:, :]))
    return dict(start=start, layers=len(body.layers), length=offset+tokens.shape[1],
        previous_kvs=tuple(body.previous_kvs[start:]), hidden=mx.stop_gradient(h[:, -1:, :]), shared=shared,
        per_layer=[None if v is None else mx.stop_gradient(v[:, -1:, :]) for v in per_layer[start:]], masks=last_masks)
