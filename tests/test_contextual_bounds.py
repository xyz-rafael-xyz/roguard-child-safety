import random

import pytest

from training.contextual_bounds_v9 import early_decision, interval
from training.diagnose_contextual_complementarity_v9 import blend


def test_bounds_enclose_dense_extreme_and_random_contextual_scores():
    rng = random.Random(1501)
    for _ in range(100):
        native = rng.choice([0., 1., 1e-15, 1-1e-15, rng.random()])
        weight = rng.choice([0., 1., .25, .75, rng.random()])
        multiplier = rng.random()
        low, high = interval(multiplier, weight, native)
        for semantic in [0., 1., 1e-15, 1-1e-15]+[rng.random() for _ in range(20)]:
            value = blend(native, semantic, weight, 'logit')*multiplier
            assert low-1e-15 <= value <= high+1e-15
            for threshold in (low, high, rng.random()):
                answer = early_decision(multiplier, weight, threshold, native)
                if answer is not None:
                    assert answer == int(value >= threshold)


def test_scope_only_exit_and_boundary_fallback():
    assert early_decision(.01, .25, .1) == 0
    assert early_decision(.1, .25, .1) is None
    assert early_decision(.1-5e-13, .25, .1) is None
    assert early_decision(1., .75, .1, 1.) == 1
    assert early_decision(1., .75, .1, 0.) == 0


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -.1, 1.1, True])
def test_invalid_bounds_rejected(value):
    with pytest.raises(ValueError):
        interval(value, .25)
    with pytest.raises(ValueError):
        interval(.5, .25, value)
