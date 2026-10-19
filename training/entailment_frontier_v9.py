"""Exact development frontier for two review reasons joined by Boolean OR.

This diagnoses whether separate cutoffs can satisfy the existing review gates.
It never reads calibration or reserved evaluation inputs and changes no labels.
"""
import math


def cuts(values):
    """One interior cutoff for every attainable >= partition on [0, 1]."""
    ordered = sorted(set(values))
    result = [ordered[0] / 2]
    for lower, upper in zip(ordered, ordered[1:]):
        middle = lower + (upper - lower) / 2
        result.append(middle if lower < middle < upper else upper)
    if ordered[-1] < 1:
        middle = ordered[-1] + (1 - ordered[-1]) / 2
        result.append(middle if middle > ordered[-1] else 1.)
    return result


def frontier(rows):
    """Minimize worst group FP rate, then sum, with zero development misses.

For each support cutoff the highest revision cutoff covering the remaining
positives is optimal: lowering it can only add false reviews. Enumerating these
points therefore finds the global optimum for this fixed OR decision family.
The optimum is a development diagnosis, not an estimate of generalization.
"""
    import numpy as np

    if not rows:
        raise ValueError('Empty development population')
    for row in rows:
        if type(row['gold']) is not int or row['gold'] not in (0, 1):
            raise ValueError('Invalid label')
        if not isinstance(row['group'], str) or not row['group']:
            raise ValueError('Invalid group')
        for key in ('support_score', 'revision_score'):
            value = row[key]
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Invalid probability')
    gold = np.array([r['gold'] for r in rows], dtype=bool)
    support = np.array([r['support_score'] for r in rows])
    revision = np.array([r['revision_score'] for r in rows])
    groups = sorted({r['group'] for r in rows})
    masks = {g: np.array([r['group'] == g for r in rows]) for g in groups}
    if any(not gold[m].any() or gold[m].all() for m in masks.values()):
        raise ValueError('Each group requires both classes')
    revision_cuts = np.array(cuts(revision.tolist()))
    best = None
    selected = None
    for support_cutoff in cuts(support.tolist()):
        support_positive = support >= support_cutoff
        uncovered = gold & ~support_positive
        if uncovered.any():
            boundary = revision[uncovered].min()
            index = int(np.searchsorted(revision_cuts, boundary, side='right')) - 1
            revision_cutoff = float(revision_cuts[index])
        else:
            revision_cutoff = float(revision_cuts[-1])
        predicted = support_positive | (revision >= revision_cutoff)
        assert not (gold & ~predicted).any()
        metrics = {}
        for group, mask in masks.items():
            negative = int((mask & ~gold).sum())
            false = int((mask & ~gold & predicted).sum())
            metrics[group] = dict(rows=int(mask.sum()), positive=int((mask & gold).sum()),
                                  negative=negative, missed=0, false_reviews=false,
                                  recall=1., false_review_rate=false / negative,
                                  passes=false / negative <= .05)
        rates = [g['false_review_rate'] for g in metrics.values()]
        rank = (max(rates), sum(rates))
        if best is None or rank < best:
            best = rank
            selected = dict(support_threshold=support_cutoff, revision_threshold=revision_cutoff,
                            rank=list(rank), metrics=metrics, counts_pass=all(g['passes'] for g in metrics.values()))
    return selected
