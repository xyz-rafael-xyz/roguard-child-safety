"""Break equal DEV error-count ranks by distance to the decision boundary."""
import math

from training.balanced_headroom_frontier_v9 import frontier as error_frontier


def frontier(rows):
    result = error_frontier(rows)
    # The existing minimax selector is retained for imperfect candidates. When
    # there are no errors, exactly one attainable partition separates all labels.
    # Center that interval and maximize its radius across checkpoints/modes.
    # This is additive score robustness on DEV, not calibrated uncertainty.
    if result['rank'] == [0., 0., 0]:
        lower = max(r['score'] for r in rows if r['gold'] == 0)
        upper = min(r['score'] for r in rows if r['gold'] == 1)
        threshold = lower + (upper-lower)/2
        if not lower < threshold <= upper:
            threshold = upper
        radius = min(threshold-lower, upper-threshold)
        assert math.isfinite(radius) and radius >= 0
        result.update(threshold=threshold, separation_interval=[lower, upper],
            score_margin=radius, rank=result['rank']+[-radius])
    else:
        result.update(separation_interval=None, score_margin=0., rank=result['rank']+[0.])
    return result
