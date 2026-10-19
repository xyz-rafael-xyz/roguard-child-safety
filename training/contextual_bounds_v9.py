"""Conservative early decisions for the existing monotone score composition.

These bounds preserve a classifier's decision; they make no claim that its
decision is correct. An unresolved bound requires the remaining model score.
"""
import math
from numbers import Real

from training.diagnose_contextual_complementarity_v9 import blend


def _unit(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Expected a finite score in [0,1]')
    return float(value)


def interval(multiplier, native_weight, native_score=None):
    multiplier = _unit(multiplier)
    weight = _unit(native_weight)
    if native_score is None:
        return 0., multiplier
    native = _unit(native_score)
    return (blend(native, 0., weight, 'logit')*multiplier,
            blend(native, 1., weight, 'logit')*multiplier)


def early_decision(multiplier, native_weight, threshold, native_score=None):
    """Return 0/1 only if all possible contextual scores have that decision."""
    threshold = _unit(threshold)
    low, high = interval(multiplier, native_weight, native_score)
    # Leave a numerical margin around the cutoff; boundary cases use the full path.
    padding = 1e-12
    if high+padding < threshold:
        return 0
    if low-padding >= threshold:
        return 1
    return None
