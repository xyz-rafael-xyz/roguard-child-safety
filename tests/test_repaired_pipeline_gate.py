from copy import deepcopy
import pytest
from training.verify_repaired_pipeline_v9 import require_counts, counts


def groups():
    return {kind: dict(positive=100, negative=20, missed=1, false_reviews=1, abstentions=0)
            for kind in ('formal', 'conversational')}


def test_gate_accepts_exact_boundaries():
    require_counts(groups())


@pytest.mark.parametrize('key,value', [('missed', 2), ('false_reviews', 2), ('positive', 0),
                                     ('abstentions', 1), ('missed', True), ('negative', -1)])
def test_gate_rejects_failed_or_malformed_counts(key, value):
    values = groups()
    values['conversational'][key] = value
    with pytest.raises(ValueError):
        require_counts(values)


def test_gate_cannot_drop_a_group():
    with pytest.raises(ValueError):
        require_counts({'formal': groups()['formal']})


def test_stress_abstentions_are_counted_as_errors():
    result = counts([dict(gold=1, predicted=None), dict(gold=0, predicted=None)])
    assert result['missed'] == result['false_reviews'] == 1
    assert result['abstentions'] == 2 and result['passes'] is False
