from roguard.evidence import ONTOLOGY
from training.semantic_boosted_v9 import semantic_vector


def test_binary_score_and_all_its_interactions_are_excluded():
    width = sum(map(len, ONTOLOGY.values()))
    original = list(range(2 + 7*width))
    altered = original.copy()
    for index in [0, 1, *range(2+2*width, 2+3*width), *range(2+5*width, 2+6*width)]:
        altered[index] = -999
    assert semantic_vector(original) == semantic_vector(altered)
    assert len(semantic_vector(original)) == 5*width
    assert semantic_vector(None) is None
