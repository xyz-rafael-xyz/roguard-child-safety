"""Numerical equivalence for the faster research threshold search."""
import random
import unittest

try:
    import numpy
except ImportError:
    numpy = None

from training.reference_frontier_v10 import frontier
from training.separation_frontier_v9 import frontier as original


@unittest.skipIf(numpy is None, 'optional NumPy training dependency')
class ReferenceFrontierTest(unittest.TestCase):
    def test_every_cut_and_tie_matches_original(self):
        for seed in range(30):
            rng = random.Random(seed)
            rows = [{'gold': i % 2, 'group': str(i // 20),
                     'score': rng.choice([0., 1., .25, .5, .75]) if seed % 2 else rng.random()}
                    for i in range(100)]
            with self.subTest(seed=seed):
                self.assertEqual(frontier(rows), original(rows))

    def test_perfect_separation_and_adjacent_floats(self):
        for lower, upper in ((.2, .8), (0., 1.), (.5, float(numpy.nextafter(.5, 1.)))):
            rows = [{'gold': 0, 'group': 'one', 'score': lower},
                    {'gold': 1, 'group': 'one', 'score': upper}]
            self.assertEqual(frontier(rows), original(rows))

    def test_rejects_group_without_both_labels(self):
        with self.assertRaises(ValueError):
            frontier([{'gold': 0, 'group': 'one', 'score': .2},
                      {'gold': 1, 'group': 'two', 'score': .8}])


if __name__ == '__main__':
    unittest.main()
