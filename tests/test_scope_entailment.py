import random

from training.fit_scope_entailment_v9 import relative_or


def test_normalized_or_preserves_original_reason_decisions():
    rng = random.Random(153)
    for _ in range(500):
        support, revision = rng.random(), rng.random()
        limits = dict(support=rng.uniform(1e-6, .999999), revision=rng.uniform(1e-6, .999999))
        for s, r in ((support, revision), (limits['support'], 0.), (0., limits['revision']), (0., 0.), (1., 1.)):
            score = relative_or(s, r, limits)
            assert 0 <= score <= 1
            assert (score >= .5) == (s >= limits['support'] or r >= limits['revision'])
