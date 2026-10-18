import pytest

from roguard.evidence import ONTOLOGY
from training.three_expert_fusion_v9 import combine_vectors


def test_distinct_semantics_and_agreement_features():
    width = sum(map(len, ONTOLOGY.values()))
    native = [.4, -.4] + [.25] * width + [0.] * width + [.1] * width
    decoder = [.4, -.4] + [.5] * width + [1.] * width + [.2] * width
    result = combine_vectors(native, decoder)
    assert len(result) == 2 + 7 * width
    assert result[:len(native)] == native
    assert result[len(native):-width] == decoder[2:]
    assert result[-width:] == [.125] * width
    assert combine_vectors(native, None) is None


def test_disagreement_about_direct_model_is_rejected():
    width = 2 + 3 * sum(map(len, ONTOLOGY.values()))
    left = [.25] * width
    right = left.copy()
    right[0] = .5
    with pytest.raises(ValueError, match='binding'):
        combine_vectors(left, right)
    with pytest.raises(ValueError, match='binding'):
        combine_vectors(left[:-1], left)
