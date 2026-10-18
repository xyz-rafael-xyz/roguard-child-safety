import math
import pytest


def rows():
    return [dict(source=source) for source in ('formal', 'formal',
        'v7_training_frame_projection', 'v7_training_frame_projection')]


@pytest.mark.parametrize('scale', [1., 1e-9, 1e-30])
def test_gap_preserves_separation_at_tiny_probability_scales(scale):
    from training.recall_gap_threshold_v9 import recall_gap_threshold
    scores = [0.1*scale, 0.8*scale, 0.2*scale, 0.7*scale]
    cutoff = recall_gap_threshold(rows(), [0, 1, 0, 1], scores)
    assert 0.2*scale < cutoff < 0.7*scale
    assert [int(score >= cutoff) for score in scores] == [0, 1, 0, 1]


def test_zero_positive_cannot_be_repaired_by_threshold_choice():
    from training.recall_gap_threshold_v9 import recall_gap_threshold
    assert recall_gap_threshold(rows(), [0, 1, 0, 1], [0., 0., .2, .7]) == 0.


def test_tied_boundary_preserves_registered_recall():
    from training.recall_gap_threshold_v9 import recall_gap_threshold
    population = [dict(source=kind) for kind in ('formal', 'v7_training_frame_projection') for _ in range(100)]
    cutoff = recall_gap_threshold(population, [1]*200, [0., .1] + [.9]*98 + [.2]*100)
    assert sum(score < cutoff for score in [0., .1] + [.9]*98) == 1
    assert cutoff <= .2


@pytest.mark.parametrize('score', [float('nan'), float('inf'), -.1, 1.1])
def test_invalid_scores_rejected(score):
    from training.recall_gap_threshold_v9 import recall_gap_threshold
    with pytest.raises(ValueError):
        recall_gap_threshold(rows(), [0, 1, 0, 1], [score, .8, .2, .7])
