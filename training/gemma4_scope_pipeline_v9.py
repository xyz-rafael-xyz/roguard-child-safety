"""Numeric research scorer for a frozen contextual/native/scope composition."""
import numpy as np

from training.diagnose_contextual_complementarity_v9 import blend
from training.gemma4_structured_v9 import portable_logits
from training.native_fields_v9 import SIZES, policy_mass
from training.score_scope_gate_v9 import score as scope_score


def score(features, rows, native, tensors, head, mixing, gate):
    native = np.asarray(native, dtype=np.float64)
    if len(rows) != len(features) or native.shape != (len(rows),) or not np.isfinite(native).all():
        raise ValueError('Mismatched or invalid scoring inputs')
    if any(row['language'] != head['language'] for row in rows):
        raise ValueError('Wrong language for selected head')
    logits = portable_logits(features, tensors).astype(np.float64)
    def softmax(values):
        exp = np.exp(values-values.max(axis=1, keepdims=True))
        return exp/exp.sum(axis=1, keepdims=True)
    if head['mode'] == 'binary':
        semantic = softmax(logits[:, -2:])[:, 1]
    elif head['mode'] in ('policy', 'coupled_policy'):
        fields = np.split(logits[:, :sum(SIZES)], np.cumsum(SIZES)[:-1], axis=1)
        semantic = policy_mass([softmax(v) for v in fields], head['couplings'] if head['mode'] == 'coupled_policy' else None)
    else:
        raise ValueError('Unsupported semantic scoring mode')
    mixed = np.array([blend(float(n), float(s), mixing['native_weight'], mixing['mode']) for n, s in zip(native, semantic)])
    final = scope_score(gate, rows, mixed)
    if not np.isfinite(final).all():
        raise ValueError('Nonfinite composed scores')
    return dict(native=native, semantic=semantic, mixed=mixed, score=final)
