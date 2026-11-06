"""Sorted-count implementation of the existing development threshold selector.

It enumerates exactly the same cutoffs and preserves the original minimax,
summed-budget, error-count and first-cutoff tie breaking. No search approximation.
"""
import math

from training.entailment_frontier_v9 import cuts


def frontier(rows):
    import numpy as np
    if not rows:
        raise ValueError('Empty development population')
    for row in rows:
        if (type(row['gold']) is not int or row['gold'] not in (0, 1)
                or not isinstance(row['group'], str) or not row['group']
                or type(row['score']) not in (int, float) or not math.isfinite(row['score'])
                or not 0 <= row['score'] <= 1):
            raise ValueError('Invalid development row')
    scores = np.array([r['score'] for r in rows]); gold = np.array([r['gold'] for r in rows], dtype=bool)
    thresholds = np.array(cuts(scores.tolist()))
    risks, all_errors, counts = [], np.zeros(len(thresholds), dtype=np.int64), {}
    for group in sorted({r['group'] for r in rows}):
        mask = np.array([r['group'] == group for r in rows])
        positive, negative = int((mask & gold).sum()), int((mask & ~gold).sum())
        if not positive or not negative:
            raise ValueError('Each development group needs both classes')
        missed = np.searchsorted(np.sort(scores[mask & gold]), thresholds, side='left')
        false = negative-np.searchsorted(np.sort(scores[mask & ~gold]), thresholds, side='left')
        risks.extend((100*missed/positive, 20*false/negative))
        all_errors += missed+false
        counts[group] = positive, negative, missed, false
    highest = np.maximum.reduce(risks)
    summed = np.zeros(len(thresholds))
    for risk in risks:
        summed += risk
    index = int(np.lexsort((np.arange(len(thresholds)), all_errors, summed, highest))[0])
    metrics = {}
    for group, (positive, negative, missed, false) in counts.items():
        miss, fp = int(missed[index]), int(false[index])
        metrics[group] = {'positive': positive, 'negative': negative, 'missed': miss,
            'false_reviews': fp, 'recall': 1-miss/positive, 'false_review_rate': fp/negative,
            'passes': miss*100 <= positive and fp*20 <= negative}
    rank = [float(highest[index]), float(summed[index]), int(all_errors[index])]
    result = {'threshold': float(thresholds[index]), 'rank': rank,
              'metrics': metrics, 'counts_pass': all(v['passes'] for v in metrics.values())}
    if rank == [0., 0., 0]:
        lower, upper = float(scores[~gold].max()), float(scores[gold].min())
        threshold = lower+(upper-lower)/2
        if not lower < threshold <= upper:
            threshold = upper
        radius = min(threshold-lower, upper-threshold)
        result.update(threshold=threshold, separation_interval=[lower, upper], score_margin=radius, rank=rank+[-radius])
    else:
        result.update(separation_interval=None, score_margin=0., rank=rank+[0.])
    return result
