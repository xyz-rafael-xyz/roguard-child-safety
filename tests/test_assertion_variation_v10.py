"""Guard source scope and the intended minimal change in matched augmentation."""
from copy import deepcopy

import pytest

from training.evidence_v8 import ASSERTIONS, unit
from training.prepare_assertion_variation_v10 import build, decompose, negation, normalized, transform


def parent(lang='ro', assertion='UNCERTAIN', layout=0):
    # Inner negation must remain inside the unchanged claim.
    core = 'Nu mai pot dormi.' if lang == 'ro' else 'Я більше не можу спати.'
    row = {'case_id': 'train-'+lang+'-fixture', 'language': lang,
           'source': 'controlled_composition_v8', 'focus': 2,
           'turns': [{'role': 'minor', 'text': 'prior'}, {'role': 'assistant', 'text': 'reply'},
                     {'role': 'minor', 'text': unit(lang, 'train', core, 0, 0, ASSERTIONS.index(assertion), layout)}],
           'attributes': {'subject': 'SELF', 'time': 'CURRENT', 'assertion': assertion,
                          'source': 'DIRECT', 'relation': 'RETRACTS' if assertion == 'RETRACTED' else 'NEW', 'concern': '1'}}
    recovered = {'feature_index': 0, 'token_sha256': 'a'*64, 'fields': [0, 0, ASSERTIONS.index(assertion), 0, int(assertion == 'RETRACTED'), 1],
                 'gold': int(assertion in ('AFFIRMED', 'UNCERTAIN'))}
    return row, recovered


@pytest.mark.parametrize('language', ['ro', 'uk'])
@pytest.mark.parametrize('assertion', ASSERTIONS)
@pytest.mark.parametrize('layout', range(3))
def test_all_variants_preserve_core_scope_context_and_labels(language, assertion, layout):
    row, _ = parent(language, assertion, layout); original = deepcopy(row)
    before, core, tail = decompose(row); changed_texts = set(); prequote = []
    for op in range(4):
        for placement in ('before', 'after'):
            changed = transform(row, op, placement)
            assert row == original
            assert changed['attributes'] == original['attributes']
            assert changed['turns'][:2] == original['turns'][:2]
            text = changed['turns'][2]['text']; changed_texts.add(text)
            assert text.startswith(before) and text.endswith(tail)
            assert text.count('«'+core+'»') == 1
            prequote.append(negation(text.partition('«')[0]))
    assert len(changed_texts) == 8
    assert sum(prequote) == 2  # Identical marker balance for every stance.


@pytest.mark.parametrize('fault', ['dev', 'test', 'source', 'empty_quote', 'multiple_quotes', 'operator', 'closing_quote'])
def test_parent_restrictions_fail_closed(fault):
    row, _ = parent()
    if fault in ('dev', 'test'): row['case_id'] = fault+'-ro-fixture'
    elif fault == 'source': row['source'] = 'other'
    else:
        text = row['turns'][2]['text']
        if fault == 'empty_quote': text = text.replace('«Nu mai pot dormi.»', '«»')
        elif fault == 'multiple_quotes': text += ' «extra»'
        elif fault == 'operator': text = text.replace('Nu știu dacă este adevărat:', 'different:')
        elif fault == 'closing_quote': text = text.replace('»', '')
        row['turns'][2]['text'] = text
    with pytest.raises(ValueError): transform(row, 0, 'before')


def test_control_repeats_exact_input_with_paired_treatment_labels():
    row, recovered = parent(); treatment, control, selected = build([(row, recovered)], set())
    assert len(treatment) == len(control) == 8 and len(selected) == 1
    for actual, reference in zip(treatment, control):
        assert reference['turns'] == row['turns']
        assert actual['pair_id'] == reference['pair_id']
        assert actual['attributes'] == reference['attributes'] == row['attributes']
        assert actual['review'] == reference['review'] == recovered['gold']
        assert actual['parent'] == reference['parent']
        assert actual['partition'] == reference['partition'] == 'train'
        assert reference['literal_pre_quote_negation'] is True
        assert actual['literal_pre_quote_negation'] == (actual['factors']['operator_index'] >= 2 and actual['factors']['quote_placement'] == 'before')


def test_duplicate_or_preexisting_treatment_cannot_enter_corpus():
    row, recovered = parent()
    text = transform(row, 0, 'before')['turns'][2]['text']
    with pytest.raises(ValueError, match='overlaps'):
        build([(row, recovered)], {normalized(text)})
    with pytest.raises(ValueError, match='Duplicate'):
        build([(row, recovered), (row, recovered)], set())
