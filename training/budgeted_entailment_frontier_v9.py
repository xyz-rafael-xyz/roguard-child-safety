"""Exact development cutoff search under the unchanged 99% recall requirement.

This is a diagnostic alternative to the zero-miss selection used by the frozen
training runs. It does not modify those runs or inspect calibration/test data.
"""
import math

from training.entailment_frontier_v9 import cuts


def frontier(rows):
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
    masks = {g: np.array([r['group'] == g for r in rows]) for g in sorted({r['group'] for r in rows})}
    if any(not gold[m].any() or gold[m].all() for m in masks.values()):
        raise ValueError('Each group requires both classes')
    positives = {g: int((m & gold).sum()) for g, m in masks.items()}
    revision_cuts = np.array(cuts(revision.tolist()))
    best = selected = None
    for support_cutoff in cuts(support.tolist()):
        support_positive = support >= support_cutoff
        revision_cutoff = float(revision_cuts[-1])
        for group, mask in masks.items():
            remaining = np.sort(revision[mask & gold & ~support_positive])
            allowance = positives[group] // 100
            if len(remaining) > allowance:
                # At most allowance unsupported positives may fall below the
                # revision cutoff. Equal scores are never split by identity.
                boundary = remaining[allowance]
                index = int(np.searchsorted(revision_cuts, boundary, side='right')) - 1
                revision_cutoff = min(revision_cutoff, float(revision_cuts[index]))
        # Recover any additional positives without adding a false review.
        # Among the negatives not already selected by support, preserve every
        # currently negative decision and lower revision within that partition.
        retained_negatives = revision[~gold & ~support_positive & (revision < revision_cutoff)]
        if len(retained_negatives):
            index = int(np.searchsorted(revision_cuts, retained_negatives.max(), side='right'))
            revision_cutoff = float(revision_cuts[index])
        else:
            revision_cutoff = float(revision_cuts[0])
        predicted = support_positive | (revision >= revision_cutoff)
        metrics = {}
        for group, mask in masks.items():
            positive = positives[group]
            negative = int((mask & ~gold).sum())
            missed = int((mask & gold & ~predicted).sum())
            false = int((mask & ~gold & predicted).sum())
            assert missed * 100 <= positive
            metrics[group] = dict(positive=positive, negative=negative, missed=missed, false_reviews=false,
                                  recall=(positive-missed)/positive, false_review_rate=false/negative,
                                  passes=missed*100 <= positive and false*20 <= negative)
        rates = [m['false_review_rate'] for m in metrics.values()]
        rank = (max(rates), sum(rates), sum(m['missed'] for m in metrics.values()))
        if best is None or rank < best:
            best = rank
            selected = dict(support_threshold=support_cutoff, revision_threshold=revision_cutoff,
                            rank=list(rank), metrics=metrics, counts_pass=all(m['passes'] for m in metrics.values()))
    return selected
