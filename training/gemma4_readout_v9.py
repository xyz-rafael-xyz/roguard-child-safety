"""Frozen contextual representations and a portable binary linear readout.

This is a learned classifier, not an attempt to reinterpret malformed generated
answers. The foundation model receives the same direct-answer policy prompt.
"""
import math


def last_features(model, tokens, cache=None):
    lm = getattr(model, 'language_model', model)
    if lm.model_type != 'gemma4_text':
        raise ValueError('Expected Gemma 4 text architecture')
    return lm.model(tokens, cache=cache)[:, -1, :]


def normalize(features):
    import numpy as np
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError('Expected finite two-dimensional contextual features')
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError('Zero contextual representation')
    return values / norms * math.sqrt(values.shape[1])


def scores(features, head):
    import numpy as np
    values = normalize(features)
    weight = np.asarray(head['weight'], dtype=np.float64)
    intercept = head['intercept']
    if (weight.shape != (values.shape[1],) or not np.isfinite(weight).all()
            or type(intercept) not in (int, float) or not math.isfinite(intercept)):
        raise ValueError('Invalid contextual readout')
    margin = values @ weight + intercept
    # Stable sigmoid in both tails; parameters never come from executable pickle.
    return np.exp(-np.logaddexp(0., -margin))
