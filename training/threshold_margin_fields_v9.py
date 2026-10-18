"""Choose a numerical gap without excluding any originally retained concern."""
import math


def margin_threshold(scores, recall_threshold, margin=1e-4):
    values=sorted(set(float(v) for v in scores))
    if (not values or not math.isfinite(recall_threshold) or not 0<=recall_threshold<=1
            or not math.isfinite(margin) or margin<=0
            or any(not math.isfinite(v) or not 0<=v<=1 for v in values)):
        raise ValueError('Invalid numerical threshold inputs')
    retained=[v for v in values if v>=recall_threshold]
    if not retained:raise ValueError('Recall threshold retains no examples')
    ceiling=retained[0]
    boundaries=sorted(set([0.,*values]))
    eligible=[left+(right-left)/2 for left,right in zip(boundaries,boundaries[1:])
              if right<=ceiling and right-left>=2*margin]
    # Zero reviews every valid probability, including exactly zero. That can
    # fail the false-review gate, but cannot conceal a numerically fragile cut.
    return max(eligible) if eligible else 0.
