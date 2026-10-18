from itertools import product
from roguard.evidence import ONTOLOGY, _complete_action
from training.semantic_interpreter_v9 import FIELDS
from training.policy_field_weights_v9 import table


def test_weights_follow_exact_binary_policy_interventions():
    values = table()
    worlds = list(product(*(ONTOLOGY[f] for f in FIELDS)))
    assert len(values) == len(worlds) == 432
    for index, labels in enumerate(worlds):
        world = dict(zip(FIELDS, labels))
        flattened = 0
        for field in FIELDS:
            flattened = flattened*len(ONTOLOGY[field])+ONTOLOGY[field].index(world[field])
        assert flattened == index
        baseline = _complete_action(world) != 'CONTINUE'
        for i, field in enumerate(FIELDS):
            possible = []
            for label in ONTOLOGY[field]:
                if label == world[field]:
                    continue
                altered = dict(world, **{field: label})
                possible.append((_complete_action(altered) != 'CONTINUE') != baseline)
            assert values[index][i] == 1+2*sum(possible)/len(possible)
            assert 1 <= values[index][i] <= 3
        # Time changes the type of support proposal, never the binary review gate.
        assert values[index][FIELDS.index('time')] == 1
