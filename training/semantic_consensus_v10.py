"""Monotone consensus of neural binary and ontology-policy scores."""
import numpy as np
from training.native_fields_v9 import SIZES, policy_mass


def score_features(logits, couplings):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != sum(SIZES)+2 or not np.isfinite(logits).all():
        raise ValueError('Invalid semantic logits')
    def softmax(values):
        exp = np.exp(values-values.max(axis=1, keepdims=True))
        return exp/exp.sum(axis=1, keepdims=True)
    pieces = np.split(logits[:, :sum(SIZES)], np.cumsum(SIZES)[:-1], axis=1)
    factors = [softmax(v) for v in pieces]
    probabilities = np.column_stack((softmax(logits[:, -2:])[:, 1],
        policy_mass(factors), policy_mass(factors, couplings)))
    p = np.clip(probabilities, 1e-6, 1-1e-6)
    return np.log(p)-np.log1p(-p)


def predict(features, parameters):
    weights = np.asarray(parameters['weights'], dtype=np.float64)
    intercept = parameters['intercept']
    if weights.shape != (3,) or (weights < 0).any() or not np.isfinite(weights).all() or not np.isfinite(intercept):
        raise ValueError('Invalid monotone consensus parameters')
    z = np.clip(features@weights+intercept, -700, 700)
    return 1/(1+np.exp(-z))
