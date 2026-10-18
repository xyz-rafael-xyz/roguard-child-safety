import math
import pytest
from training.threshold_margin_v9 import select


def test_shared_threshold_respects_each_stratum_and_ties():
    rows = []
    scores = []
    for source, values in [('formal', [.9]*4+[.2]*96),
                           ('v7_training_frame_projection', [.8]*2+[.3]*98)]:
        rows.extend(dict(language='uk', source=source) for _ in values)
        scores.extend(values)
    threshold = select(rows, [0]*len(rows), scores, 'uk')
    assert threshold == math.nextafter(.9, math.inf)
    assert sum(s >= threshold for s in scores[:100]) <= 3
    assert sum(s >= threshold for s in scores[100:]) <= 3


def test_threshold_selection_does_not_use_positive_scores():
    rows = [dict(language='ro', source=source) for source in
            ('formal', 'v7_training_frame_projection') for _ in range(101)]
    truth = ([0]*100+[1])*2
    scores = ([.1]*100+[.99])*2
    first = select(rows, truth, scores, 'ro')
    scores[100] = scores[201] = .001
    assert select(rows, truth, scores, 'ro') == first
    with pytest.raises(ValueError, match='Missing development negatives'):
        select(rows, truth, scores, 'uk')
