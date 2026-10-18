import pytest
from training.reserved_metrics_v9 import counts, input_key, unique_records


def test_abstentions_fail_both_classes():
    result = counts([dict(gold=1, predicted=None), dict(gold=0, predicted=None)])
    assert result['missed'] == result['false_reviews'] == 1
    assert result['abstentions'] == 2 and not result['passes']


def test_exact_gate_boundary_and_missing_class():
    rows = [dict(gold=1, predicted=int(i != 0)) for i in range(100)]
    rows += [dict(gold=0, predicted=int(i == 0)) for i in range(20)]
    assert counts(rows)['passes']
    rows[1]['predicted'] = 0
    assert not counts(rows)['passes']
    assert not counts([dict(gold=0, predicted=0)])['passes']


def test_repeated_inputs_do_not_inflate_counts_and_disagreement_rejected():
    row = dict(input_digest='same', gold=1, predicted=1)
    assert unique_records([row, dict(row)]) == [row]
    with pytest.raises(ValueError, match='inconsistent'):
        unique_records([row, dict(row, predicted=0)])


def test_future_turn_does_not_change_input_identity():
    row = dict(language='ro', focus=0, turns=[dict(role='minor', text='Salut')])
    future = dict(row, turns=row['turns']+[dict(role='minor', text='future')])
    assert input_key(row) == input_key(future)
    assert input_key(dict(row, language='uk')) != input_key(row)
