"""Batch equal-length continuations of one frozen Gemma4 prompt prefix.

Mixed lengths are grouped without padding. This preserves the independent
attention path, including its special handling of one-token continuations.
The supplied prefix cache is cloned and never advanced by this operation.
"""
from copy import deepcopy


def batch_features(model, continuations, prefix_cache=None):
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache, RotatingKVCache
    lm = getattr(model, 'language_model', model)
    if lm.model_type != 'gemma4_text':
        raise ValueError('Expected Gemma4 text architecture')
    if not continuations or len(continuations) > 16:
        raise ValueError('Expected one to sixteen independent continuations')
    vocabulary = lm.args.vocab_size
    for row in continuations:
        if not row or any(type(v) is not int or not 0 <= v < vocabulary for v in row):
            raise ValueError('Expected nonempty valid token sequences')
    groups = {}
    for i,row in enumerate(continuations):
        groups.setdefault(len(row),[]).append(i)
    output = [None]*len(continuations)
    for indices in groups.values():
        cache = deepcopy(prefix_cache)
        if cache is not None:
            for layer in cache:
                if type(layer) not in (KVCache,RotatingKVCache):
                    raise ValueError('Unsupported prefix cache type')
                if layer.empty():
                    continue
                keys,values = layer.state
                if keys.ndim != 4 or values.ndim != 4 or keys.shape[0] != 1 or values.shape[0] != 1:
                    raise ValueError('Prefix cache must represent one conversation')
                layer.state = (mx.repeat(keys,len(indices),axis=0),mx.repeat(values,len(indices),axis=0))
        hidden = lm.model(mx.array([continuations[i] for i in indices]),cache=cache)
        for position,index in enumerate(indices):
            output[index] = hidden[position,-1,:]
    return mx.stack(output)
