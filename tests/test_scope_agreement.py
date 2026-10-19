import random

from training.qualify_scope_agreement_v9 import agreement_score


def test_numeric_agreement_matches_both_frozen_cutoffs():
    rng = random.Random(154)
    for _ in range(500):
        a, b = rng.uniform(1e-6, .999999), rng.uniform(1e-6, .999999)
        for x, y in ((rng.random(), rng.random()), (a, b), (a, 0.), (0., b), (1., 1.)):
            value = agreement_score(x, y, a, b)
            assert 0 <= value <= 1
            assert (value >= .5) == (x >= a and y >= b)
