"""Choose a native development cutoff under the existing per-group 99/5 gates."""
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
    gold = np.array([r['gold'] for r in rows], dtype=bool)
    scores = np.array([r['score'] for r in rows])
    groups = {g:np.array([r['group'] == g for r in rows]) for g in sorted({r['group'] for r in rows})}
    if any(not gold[m].any() or gold[m].all() for m in groups.values()):
        raise ValueError('Each group requires both classes')
    best = selected = None
    for cutoff in cuts(scores.tolist()):
        predicted = scores >= cutoff
        metrics = {}
        for group, mask in groups.items():
            positive = int((mask & gold).sum())
            negative = int((mask & ~gold).sum())
            missed = int((mask & gold & ~predicted).sum())
            false = int((mask & ~gold & predicted).sum())
            if missed * 100 > positive:
                break
            metrics[group] = dict(positive=positive, negative=negative, missed=missed,
                                  false_reviews=false, false_review_rate=false/negative,
                                  recall=(positive-missed)/positive, passes=false*20 <= negative)
        else:
            rates = [m['false_review_rate'] for m in metrics.values()]
            rank = (max(rates), sum(rates), sum(m['missed'] for m in metrics.values()))
            if best is None or rank < best:
                best = rank
                selected = dict(threshold=cutoff, rank=list(rank), metrics=metrics,
                                counts_pass=all(m['passes'] for m in metrics.values()))
    return selected
