import itertools
import math
import random

import pytest

from roguard.evidence import ONTOLOGY
from training.policy_decode_v9 import actions, decode
from training.semantic_interpreter_v9 import FIELDS


def test_conditional_bounds_cover_exhaustive_correlated_distribution():
    for seed in range(8):
        rng = random.Random(seed)
        table = {}
        def conditional(prefix, field):
            if prefix not in table:
                raw = [rng.random()**3 for _ in ONTOLOGY[field]]
                table[prefix] = [v/sum(raw) for v in raw]
            return table[prefix]
        # Materialize first to make the oracle independent of traversal order.
        exact = 0.0
        for world in itertools.product(*(ONTOLOGY[f] for f in FIELDS)):
            mass = 1.0
            for i, label in enumerate(world):
                mass *= conditional(world[:i], FIELDS[i])[ONTOLOGY[FIELDS[i]].index(label)]
            exact += mass * next(iter(actions(world)))
        for budget in (0, 1, 3, 8, 64):
            result = decode(conditional, max_queries=budget)
            assert result['lower_review_mass']-1e-12 <= exact <= result['upper_review_mass']+1e-12
            if result['predicted'] is not None:
                assert result['predicted'] == int(exact >= .5)
            assert result['queries'] <= budget


def test_invalid_output_retains_mass_and_cannot_certify():
    result = decode(lambda p, f: None)
    assert result['predicted'] is None
    assert result['invalid_mass'] == 1
    assert (result['lower_review_mass'], result['upper_review_mass']) == (0, 1)


def test_ordinary_content_collapses_without_querying_other_fields():
    calls = []
    def oracle(prefix, field):
        calls.append(field)
        assert field == 'concern'
        return [.99, .01]
    result = decode(oracle)
    assert result['predicted'] == 0
    assert calls == ['concern']
    assert math.isclose(result['unresolved_mass'], .01)


@pytest.mark.parametrize('bad', [[.1, .1], [float('nan'), 0], [-.1, 1.1], [True, False]])
def test_invalid_probabilities_rejected(bad):
    with pytest.raises(ValueError):
        decode(lambda p, f: bad)
