import itertools
import random
import unittest
from roguard.evidence import ONTOLOGY, _complete_action
from training.policy_probability_v9 import review_mass


class PolicyProbabilityTests(unittest.TestCase):
    def test_all_deterministic_worlds(self):
        for values in itertools.product(*ONTOLOGY.values()):
            world=dict(zip(ONTOLOGY,values))
            marginals={name:{v:float(v==world[name]) for v in choices} for name,choices in ONTOLOGY.items()}
            self.assertEqual(review_mass(marginals),float(_complete_action(world)!='CONTINUE'))

    def test_marginal_mass_matches_exhaustive_world_sum(self):
        rng=random.Random(903)
        for _ in range(50):
            marginals={}
            for name,choices in ONTOLOGY.items():
                weights=[rng.random() for _ in choices];total=sum(weights)
                marginals[name]={v:w/total for v,w in zip(choices,weights)}
            expected=0
            for values in itertools.product(*ONTOLOGY.values()):
                world=dict(zip(ONTOLOGY,values));probability=1
                for name,val in world.items():probability*=marginals[name][val]
                if _complete_action(world)!='CONTINUE':expected+=probability
            self.assertAlmostEqual(review_mass(marginals),expected,places=12)
