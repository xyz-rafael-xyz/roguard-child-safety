"""Prioritize semantic loss among checkpoints passing unchanged review gates."""
import math


def selection_rank(groups, field_loss, step, coupled):
    if (set(groups) != {'formal', 'conversational'} or not math.isfinite(field_loss)
            or field_loss < 0 or type(step) is not int or step < 0):
        raise ValueError('Invalid semantic selection evidence')
    worst = max(value['false_review_rate'] for value in groups.values())
    total = sum(value['false_review_rate'] for value in groups.values())
    passes = all(value['passes'] is True for value in groups.values())
    if passes:
        return [0, field_loss, worst, total, step, int(coupled)]
    return [1, worst, total, field_loss, step, int(coupled)]
