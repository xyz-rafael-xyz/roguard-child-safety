"""Center the highest score gap that preserves the registered recall budget.

No absolute probability epsilon is subtracted: very small probabilities can
still contain a useful ordering. Installed inference must separately reproduce
every selected decision; this gap is not an assumed bound on device error.
"""
import math
from training.review_fusion_v9 import strata


def recall_gap_threshold(rows, truth, scores):
    if not rows or len(rows) != len(truth) or len(rows) != len(scores):
        raise ValueError('Threshold inputs must have equal nonzero lengths')
    values = [float(value) for value in scores]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError('Expected finite probability scores')
    if any(value not in (0, 1) for value in truth):
        raise ValueError('Expected binary policy labels')
    ceilings = []
    for kind in ('formal', 'conversational'):
        positives = sorted(value for row, label, value in zip(rows, truth, values)
                           if strata(row) == kind and label == 1)
        if not positives:
            raise ValueError('Both recall strata require positive examples')
        ceilings.append(positives[len(positives) // 100])
    ceiling = min(ceilings)
    below = [value for value in values if value < ceiling]
    left = max(below, default=0.)
    midpoint = left + (ceiling - left) / 2
    # If rounding eliminates the gap, the boundary still retains all tied
    # positives. Actual installed decision parity remains mandatory.
    return midpoint if left < midpoint <= ceiling else ceiling
