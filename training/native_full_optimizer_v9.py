"""Layer-wise learning rates for full native encoder adaptation."""
import math


def parameter_groups(model, *, top_lr=5e-6, decay=.85):
    if (type(top_lr) not in (int, float) or not math.isfinite(top_lr) or top_lr <= 0
            or type(decay) not in (int, float) or not math.isfinite(decay) or not 0 < decay <= 1):
        raise ValueError('Invalid layer learning rates')
    base = model.base_model
    layers = list(base.encoder.layer)
    if not layers:
        raise ValueError('Expected native encoder layers')
    modules = [('embeddings', [base.embeddings], len(layers))]
    modules.extend((f'encoder.{i}', [layer], len(layers)-1-i) for i, layer in enumerate(layers))
    head = [model.classifier]
    if getattr(base, 'pooler', None) is not None:
        head.append(base.pooler)
    modules.append(('head_and_pooler', head, 0))
    result = []
    seen = set()
    for name, members, depth in modules:
        parameters = [p for module in members for p in module.parameters()]
        identities = [id(p) for p in parameters]
        if len(set(identities)) != len(identities) or seen.intersection(identities):
            raise ValueError('A parameter belongs to multiple optimization groups')
        seen.update(identities)
        for parameter in parameters:
            parameter.requires_grad_(True)
        rate = top_lr * decay**depth
        result.append(dict(name=name, params=parameters, lr=rate, base_lr=rate))
    if seen != {id(p) for p in model.parameters()}:
        raise ValueError('Full native adaptation must cover every parameter')
    return result


def schedule(optimizer, scale):
    if type(scale) not in (int, float) or not math.isfinite(scale) or not 0 <= scale <= 1:
        raise ValueError('Invalid learning-rate schedule scale')
    for group in optimizer.param_groups:
        group['lr'] = group['base_lr'] * scale
