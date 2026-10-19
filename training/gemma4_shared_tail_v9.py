"""Exact final-token replay through Gemma4's shared-KV decoder tail.

Only tail parameters may change after capture. Earlier layers, embeddings,
per-layer input projections and their KV states must stay frozen. This helper
does not truncate text or approximate the attention keys and values.
"""


def _backbone(model, start):
    lm = getattr(model, 'language_model', model)
    if lm.model_type != 'gemma4_text':
        raise ValueError('Expected Gemma4 text architecture')
    body = lm.model
    if type(start) is not int or not 0 < start < len(body.layers):
        raise ValueError('Invalid shared tail boundary')
    for index in range(start, len(body.layers)):
        if body.layers[index].self_attn.has_kv or body.previous_kvs[index] >= start:
            raise ValueError('Every adapted layer must reuse frozen prefix KV states')
    return body


def capture(model, tokens, *, start):
    """Run the frozen body and save only the tail's required final-token inputs."""
    import mlx.core as mx
    body = _backbone(model, start)
    if tokens.ndim != 2 or not tokens.shape[0] or not tokens.shape[1]:
        raise ValueError('Expected a nonempty, unpadded token batch')
    h = body.embed_tokens(tokens)*body.embed_scale
    if body.hidden_size_per_layer_input:
        per_layer = body._project_per_layer_inputs(h, body._get_per_layer_inputs(tokens))
        per_layer = [per_layer[:, :, i, :] for i in range(len(body.layers))]
    else:
        per_layer = [None]*len(body.layers)
    masks = body._make_masks(h, [None]*len(body.layers))
    intermediates = [(None, None)]*len(body.layers)
    for index in range(start):
        kvs, offset = intermediates[body.previous_kvs[index]]
        h, kvs, offset = body.layers[index](h, masks[index], None,
            per_layer_input=per_layer[index], shared_kv=kvs, offset=offset)
        intermediates[index] = (kvs, offset)
    sources = sorted(set(body.previous_kvs[start:]))
    shared = {index: tuple(mx.stop_gradient(v) for v in intermediates[index][0]) for index in sources}
    final_masks = []
    for mask in masks[start:]:
        if isinstance(mask, str):
            if mask != 'causal':
                raise ValueError('Unknown attention mask')
            final_masks.append(None)  # The last query may see every earlier key.
        else:
            final_masks.append(None if mask is None else mx.stop_gradient(mask[..., -1:, :]))
    return dict(start=start, layers=len(body.layers), length=tokens.shape[1],
        hidden=mx.stop_gradient(h[:, -1:, :]), shared=shared, masks=final_masks,
        per_layer=[None if v is None else mx.stop_gradient(v[:, -1:, :]) for v in per_layer[start:]],
        previous_kvs=tuple(body.previous_kvs[start:]))


def replay(model, state):
    """Recompute the final token through the adaptable tail and final norm."""
    body = _backbone(model, state['start'])
    if state['layers'] != len(body.layers) or state['previous_kvs'] != tuple(body.previous_kvs[state['start']:]):
        raise ValueError('Cached tail architecture differs')
    h = state['hidden']
    for local, index in enumerate(range(state['start'], len(body.layers))):
        source = body.previous_kvs[index]
        h, _, _ = body.layers[index](h, state['masks'][local], None,
            per_layer_input=state['per_layer'][local], shared_kv=state['shared'][source],
            offset=state['length']-1)
    return body.norm(h)
