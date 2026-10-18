"""Policy-sensitive training weights; no claim of calibrated risk or optimality."""
import itertools
from roguard.evidence import ONTOLOGY, _complete_action
from training.semantic_interpreter_v9 import FIELDS


def weights(world):
    review = _complete_action(world) != 'CONTINUE'
    result = []
    for field in FIELDS:
        alternatives = [label for label in ONTOLOGY[field] if label != world[field]]
        changed = sum((_complete_action(dict(world, **{field: label})) != 'CONTINUE') != review
                      for label in alternatives)
        result.append(1 + 2*changed/len(alternatives))
    return result


def table():
    return [weights(dict(zip(FIELDS, values)))
            for values in itertools.product(*(ONTOLOGY[field] for field in FIELDS))]
