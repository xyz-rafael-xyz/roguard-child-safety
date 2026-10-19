from copy import deepcopy
import pytest
from roguard.evidence import ONTOLOGY
from training.repaired_three_expert_v9 import combine


def parts():
    probabilities = {key: [1. / len(labels)] * len(labels) for key, labels in ONTOLOGY.items()}
    return (dict(score=.2), dict(field_probabilities=probabilities),
            dict(valid_label_tokens=True, conditional_field_probabilities=deepcopy(probabilities),
                 interpretation={key: labels[0] for key, labels in ONTOLOGY.items()}))


def test_repaired_score_updates_both_experts_interactions():
    native, semantic, interpreted = parts()
    first = combine(native, semantic, interpreted)
    native['score'] = .8
    second = combine(native, semantic, interpreted)
    width = sum(map(len, ONTOLOGY.values()))
    assert len(second) == 2 + 7 * width
    assert second[0] == .8 and second[1] > first[1]
    assert second[2:2 + 2 * width] == first[2:2 + 2 * width]
    assert second[-width:] == first[-width:]
    for start in (2 + 2 * width, 2 + 5 * width):
        assert second[start:start + width] == pytest.approx([4 * v for v in first[start:start + width]])


def test_gold_text_and_identifiers_are_never_predictive_features():
    values = parts()
    expected = combine(*values)
    for value in values:
        value.update(gold=1, case_id='changed', text='untrusted', source='other')
    assert combine(*values) == expected


def test_invalid_decoder_output_preserves_abstention():
    native, semantic, interpreted = parts()
    interpreted['valid_label_tokens'] = False
    assert combine(native, semantic, interpreted) is None
