"""Center a threshold without changing any supplied development decision."""
import math


def center_threshold(scores, threshold):
    values = sorted(set(float(v) for v in scores))
    if not values or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Invalid threshold population')
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError('Invalid probability')
    below = [v for v in values if v < threshold]
    above = [v for v in values if v >= threshold]
    # Preserve all-positive/all-negative boundaries rather than fabricate a
    # second class. Actual installed parity remains mandatory in every case.
    if not below or not above:
        return float(threshold)
    centered = below[-1] + (above[0] - below[-1]) / 2
    if not below[-1] < centered <= above[0]:
        return float(threshold)
    return centered
