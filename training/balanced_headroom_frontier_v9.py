"""DEV-only minimax utilization of the unchanged recall/false-review budgets."""
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
    scores = np.array([r['score'] for r in rows])
    gold = np.array([r['gold'] for r in rows], dtype=bool)
    masks = {g: np.array([r['group'] == g for r in rows]) for g in sorted({r['group'] for r in rows})}
    if any(not gold[m].any() or gold[m].all() for m in masks.values()):
        raise ValueError('Each development group needs both classes')
    best = None
    for threshold in cuts(scores.tolist()):
        predicted = scores >= threshold
        metrics, risks = {}, []
        for group, mask in masks.items():
            positive, negative = int((mask & gold).sum()), int((mask & ~gold).sum())
            missed, false = int((mask & gold & ~predicted).sum()), int((mask & ~gold & predicted).sum())
            passes = missed*100 <= positive and false*20 <= negative
            metrics[group] = dict(positive=positive, negative=negative, missed=missed,
                false_reviews=false, recall=1-missed/positive, false_review_rate=false/negative, passes=passes)
            risks.extend((100*missed/positive, 20*false/negative))
        # First minimize the largest fraction of either error budget used,
        # then aggregate utilization and error count. No CAL threshold tuning.
        rank = [max(risks), sum(risks), sum(m['missed']+m['false_reviews'] for m in metrics.values())]
        if best is None or rank < best['rank']:
            best = dict(threshold=threshold, rank=rank, metrics=metrics,
                counts_pass=all(m['passes'] for m in metrics.values()))
    return best
