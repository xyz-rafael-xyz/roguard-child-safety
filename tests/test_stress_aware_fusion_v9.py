from collections import defaultdict
import pytest
from training.needs_repair_data_v9 import generate
from training.stress_aware_fusion_v9 import assignments, stress_family


def test_stress_folds_hold_out_whole_content_families_across_scope_variants():
    rows = generate('dev')
    groups = defaultdict(set)
    values = assignments([], rows)
    for row, fold in zip(rows, values):
        groups[stress_family(row)].add(fold)
    assert len(rows) == 216 and len(groups) == 8
    assert all(len(folds) == 1 for folds in groups.values())
    assert len(set(values)) > 1
    reversed_values = assignments([], list(reversed(rows)))
    assert values == list(reversed(reversed_values))


def test_unknown_stress_content_cannot_enter_family_validation():
    with pytest.raises(ValueError, match='exactly one'):
        stress_family({'turns': [{'text': 'Unknown family'}]})
