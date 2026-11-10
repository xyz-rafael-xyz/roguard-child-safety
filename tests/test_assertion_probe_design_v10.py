"""Probe siblings and evaluation examples cannot silently enter fitting."""
from copy import deepcopy
import itertools

import pytest

from training.assertion_probe_design_v10 import ASSERTIONS, expected_partition, validate_probe_rows


def fixture():
    plan = dict(format='roguard-assertion-probe-design-1', languages=['ro', 'uk'], assertions=list(ASSERTIONS),
        operator_families={'o1': 'fit', 'o2': 'fit', 'o3': 'probe'},
        meaning_families={'m1': 'fit', 'm2': 'fit', 'm3': 'probe'},
        paired_meaning_ids={m: [m+':0', m+':1'] for m in ('m1', 'm2', 'm3')},
        context_variants=['neutral', 'role_foil'], withheld_combinations=[['o1', 'm1']],
        training_admitted=False, held_out_test_opened=False)
    rows = []
    for op, meaning in itertools.product(plan['operator_families'], plan['meaning_families']):
        for pair, assertion, lang, context in itertools.product(plan['paired_meaning_ids'][meaning], ASSERTIONS,
                                                               plan['languages'], plan['context_variants']):
            identity = ':'.join((op, meaning, pair, assertion, lang, context))
            attrs = dict(subject='SELF', time='CURRENT', assertion=assertion, source='DIRECT',
                         relation='RETRACTS' if assertion == 'RETRACTED' else 'NEW', concern=pair[-1])
            rows.append(dict(case_id=identity, operator_family=op, meaning_family=meaning,
                paired_meaning_id=pair, context_variant=context, language=lang,
                partition=expected_partition(plan, op, meaning), proposed_attributes=attrs,
                proposed_gold=int(pair[-1] == '1' and assertion != 'NEGATED'),
                training_admitted=False, independently_annotated=False,
                source_focus=0,
                payload=dict(language=lang, turns=[dict(role='minor', text=identity)])))
    return plan, rows


def test_complete_split_has_all_folds_and_never_admits_training():
    plan, rows = fixture(); before = deepcopy(rows)
    report = validate_probe_rows(rows, plan)
    assert report['counts'] == dict(fit=96, probe_combination=32, probe_operator=64, probe_meaning=64, probe_both=32)
    assert report['rows'] == report['unique_normalized_inputs'] == 288 and report['blocks'] == 9
    assert report['bilingual_sibling_groups'] == 144
    assert not any(report[k] for k in ('training_admitted', 'semantic_labels_validated', 'token_overlap_checked',
                                      'held_out_test_opened', 'independently_validated', 'qualified'))
    assert rows == before


@pytest.mark.parametrize('fold', ['probe_operator', 'probe_meaning', 'probe_combination', 'probe_both'])
def test_each_probe_fold_rejected_if_marked_fit(fold):
    plan, rows = fixture()
    row = next(r for r in rows if r['partition'] == fold)
    row['partition'] = 'fit'
    with pytest.raises(ValueError, match='Partition differs'): validate_probe_rows(rows, plan)


def test_legacy_all_train_ingestion_is_rejected():
    plan, rows = fixture()
    for row in rows: row['partition'] = 'train'
    with pytest.raises(ValueError, match='Partition differs'): validate_probe_rows(rows, plan)


@pytest.mark.parametrize('mutation', ['drop_translation', 'drop_role', 'duplicate', 'foreign_cell', 'sibling_label', 'language'])
def test_incomplete_or_altered_siblings_rejected(mutation):
    plan, rows = fixture()
    if mutation == 'drop_translation': rows = [r for r in rows if r['language'] != 'uk']
    if mutation == 'drop_role': rows = [r for r in rows if r['context_variant'] != 'role_foil']
    if mutation == 'duplicate': rows.append(deepcopy(rows[0]))
    if mutation == 'foreign_cell': rows[0]['operator_family'] = 'foreign'
    if mutation == 'sibling_label': rows[0]['proposed_gold'] = 1-rows[0]['proposed_gold']
    if mutation == 'language': rows[0]['payload']['language'] = 'uk'
    with pytest.raises(ValueError): validate_probe_rows(rows, plan)


def test_payload_normalization_and_role_sensitive_overlap():
    plan, rows = fixture()
    protected = deepcopy(rows[0]['payload'])
    protected['turns'][0]['text'] = '  '+protected['turns'][0]['text'].upper()+'  '
    with pytest.raises(ValueError, match='protected'): validate_probe_rows(rows, plan, protected_payloads=[protected])
    protected['turns'][0]['role'] = 'assistant'
    assert validate_probe_rows(rows, plan, protected_payloads=[protected])['protected_unique_inputs_checked'] == 1


def test_same_input_cannot_be_disguised_as_a_different_family():
    plan, rows = fixture()
    a = next(r for r in rows if r['partition'] == 'fit')
    b = next(r for r in rows if r['partition'] == 'probe_operator' and r['language'] == a['language'])
    b['payload'] = deepcopy(a['payload'])
    with pytest.raises(ValueError, match='Repeated normalized input'): validate_probe_rows(rows, plan)


@pytest.mark.parametrize('destination', ['protected', 'probe'])
def test_compatibility_unicode_cannot_hide_overlap(destination):
    plan, rows = fixture()
    original = next(r for r in rows if r['partition'] == 'fit')
    disguised = deepcopy(original['payload'])
    disguised['turns'][0]['text'] = ''.join(chr(ord(c)+0xFEE0) if '!' <= c <= '~' else c
                                           for c in disguised['turns'][0]['text'])
    if destination == 'protected':
        with pytest.raises(ValueError, match='protected'):
            validate_probe_rows(rows, plan, protected_payloads=[disguised])
    else:
        probe = next(r for r in rows if r['partition'] == 'probe_operator' and r['language'] == original['language'])
        probe['payload'] = disguised
        with pytest.raises(ValueError, match='Repeated normalized input'): validate_probe_rows(rows, plan)


@pytest.mark.parametrize('mutation', ['lost_fit_margin', 'duplicated_combo', 'unknown_combo', 'no_probe', 'shared_meaning', 'admitted'])
def test_invalid_family_design_rejected(mutation):
    plan, rows = fixture()
    if mutation == 'lost_fit_margin': plan['withheld_combinations'].append(['o1', 'm2'])
    if mutation == 'duplicated_combo': plan['withheld_combinations'].append(['o1', 'm1'])
    if mutation == 'unknown_combo': plan['withheld_combinations'].append(['o3', 'm1'])
    if mutation == 'no_probe': plan['operator_families']['o3'] = 'fit'
    if mutation == 'shared_meaning': plan['paired_meaning_ids']['m2'][0] = 'm1:0'
    if mutation == 'admitted': plan['training_admitted'] = True
    with pytest.raises(ValueError): validate_probe_rows(rows, plan)


def test_rows_cannot_claim_independent_annotation_or_admission():
    for field in ('training_admitted', 'independently_annotated'):
        plan, rows = fixture(); rows[0][field] = True
        with pytest.raises(ValueError, match='cannot grant'): validate_probe_rows(rows, plan)


@pytest.mark.parametrize('mutation', ['future_turn', 'assistant_focus', 'bool_focus'])
def test_focal_prefix_cannot_be_ambiguous_or_include_future_turns(mutation):
    plan, rows = fixture()
    if mutation == 'future_turn': rows[0]['payload']['turns'].append(dict(role='assistant', text='Later context'))
    if mutation == 'assistant_focus': rows[0]['payload']['turns'][0]['role'] = 'assistant'
    if mutation == 'bool_focus': rows[0]['source_focus'] = False
    with pytest.raises(ValueError, match='focal minor'): validate_probe_rows(rows, plan)


@pytest.mark.parametrize('mutation', ['value', 'bool'])
def test_numeric_field_proposals_cannot_disagree_with_canonical_named_labels(mutation):
    from roguard.evidence import ONTOLOGY
    from roguard.native_fields import FIELDS
    plan, rows = fixture()
    for row in rows:
        row['proposed_fields'] = [ONTOLOGY[k].index(row['proposed_attributes'][k]) for k in FIELDS]
    assert validate_probe_rows(rows, plan)['verification'] == 'PASS'
    rows[0]['proposed_fields'][0] = 1 if mutation == 'value' else False
    with pytest.raises(ValueError, match='Numeric field proposals'):
        validate_probe_rows(rows, plan)
