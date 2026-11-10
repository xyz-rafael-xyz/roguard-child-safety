"""Verify generated probe partitions without admitting their semantic labels.

This module does not turn an author proposal into TRAIN data. The fitter must
perform separate admission and explicitly select the fit partition. The probe
folds are developer diagnostics, not independent or sealed release evaluation.
"""
from collections import Counter
import itertools
import re
import unicodedata

from roguard.assurance import digest
from roguard.evidence import ONTOLOGY
from roguard.native_fields import FIELDS
from roguard.trajectory_data import validate_conversation


FOLDS = ('fit', 'probe_operator', 'probe_meaning', 'probe_combination', 'probe_both')
ASSERTIONS = ('AFFIRMED', 'NEGATED', 'UNCERTAIN', 'RETRACTED')


def require(value, message):
    if not value:
        raise ValueError(message)


def normalized_payload_identity(payload):
    validate_conversation(payload)
    return digest({'language': payload['language'], 'turns': [
        {'role': turn['role'], 'text': re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', turn['text'])).strip().casefold()}
        for turn in payload['turns']]})


def validate_plan(plan):
    require(plan.get('format') == 'roguard-assertion-probe-design-1', 'Unknown probe design')
    require(plan.get('languages') == ['ro', 'uk'] and plan.get('assertions') == list(ASSERTIONS),
            'Bilingual assertion coverage differs')
    require(plan.get('training_admitted') is False and plan.get('held_out_test_opened') is False,
            'A probe design cannot grant training admission or open TEST')
    for key in ('operator_families', 'meaning_families'):
        groups = plan[key]
        require(isinstance(groups, dict) and groups and set(groups.values()) == {'fit', 'probe'}
                and all(isinstance(k, str) and k for k in groups), 'Invalid family margins')
    contexts = plan['context_variants']
    require(isinstance(contexts, list) and contexts and len(contexts) == len(set(contexts))
            and all(isinstance(c, str) and c for c in contexts), 'Invalid context variants')
    pairs = plan['paired_meaning_ids']
    require(set(pairs) == set(plan['meaning_families']), 'Meaning inventory differs from family margins')
    all_ids = []
    for ids in pairs.values():
        require(isinstance(ids, list) and ids and all(isinstance(i, str) and i for i in ids), 'Invalid paired meanings')
        all_ids.extend(ids)
    require(len(all_ids) == len(set(all_ids)), 'Paired meaning identity reused across families')
    withheld = plan['withheld_combinations']
    require(isinstance(withheld, list) and withheld, 'Missing combination probes')
    require(all(isinstance(pair, list) and len(pair) == 2 and all(isinstance(v, str) for v in pair)
                for pair in withheld), 'Invalid withheld combination')
    require(len(withheld) == len({tuple(pair) for pair in withheld}), 'Repeated withheld combination')
    for op, meaning in withheld:
        require(plan['operator_families'].get(op) == plan['meaning_families'].get(meaning) == 'fit',
                'Combination probe must have both margins represented in fitting')
    for op, margin in plan['operator_families'].items():
        if margin == 'fit':
            require(any(margin2 == 'fit' and [op, meaning] not in withheld
                        for meaning, margin2 in plan['meaning_families'].items()), 'Fit operator has no fitting block')
    for meaning, margin in plan['meaning_families'].items():
        if margin == 'fit':
            require(any(margin2 == 'fit' and [op, meaning] not in withheld
                        for op, margin2 in plan['operator_families'].items()), 'Fit meaning has no fitting block')


def expected_partition(plan, operator, meaning):
    require(operator in plan['operator_families'] and meaning in plan['meaning_families'], 'Unknown family')
    op = plan['operator_families'][operator] == 'fit'
    core = plan['meaning_families'][meaning] == 'fit'
    if op and core:
        return 'probe_combination' if [operator, meaning] in plan['withheld_combinations'] else 'fit'
    if op:
        return 'probe_meaning'
    return 'probe_operator' if core else 'probe_both'


def validate_probe_rows(rows, plan, *, protected_payloads=()):
    """Reconstruct every block; reject loss, duplication and mislabeled folds."""
    validate_plan(plan)
    require(isinstance(rows, list) and rows, 'No probe rows')
    expected = set()
    for op, meaning in itertools.product(plan['operator_families'], plan['meaning_families']):
        expected.update((op, meaning, pair, assertion, language, context)
            for pair, assertion, language, context in itertools.product(
                plan['paired_meaning_ids'][meaning], ASSERTIONS, plan['languages'], plan['context_variants']))
    protected = {normalized_payload_identity(p) for p in protected_payloads}
    seen, case_ids, inputs, siblings = set(), set(), {}, {}
    counts, blocks = Counter(), {}
    for row in rows:
        identity = row['case_id']
        require(isinstance(identity, str) and identity and identity not in case_ids, 'Repeated or invalid case identity')
        case_ids.add(identity)
        attributes = row['proposed_attributes']
        require(set(attributes) == set(ONTOLOGY) and all(attributes[k] in values for k, values in ONTOLOGY.items()),
                'Invalid proposed field labels')
        if 'proposed_fields' in row:
            encoded = row['proposed_fields']
            require(isinstance(encoded, list) and all(type(v) is int for v in encoded)
                    and encoded == [ONTOLOGY[k].index(attributes[k]) for k in FIELDS],
                    'Numeric field proposals disagree with named labels')
        require(type(row['proposed_gold']) is int and row['proposed_gold'] in (0, 1), 'Invalid proposed binary label')
        require(row.get('training_admitted') is False and row.get('independently_annotated') is False,
                'Structural checking cannot grant semantic admission or independent annotation')
        key = (row['operator_family'], row['meaning_family'], row['paired_meaning_id'],
               attributes['assertion'], row['language'], row['context_variant'])
        require(key in expected and key not in seen, 'Unexpected or repeated probe cell')
        seen.add(key)
        fold = expected_partition(plan, key[0], key[1])
        require(row['partition'] == fold, 'Partition differs from independently reconstructed family split')
        require(row.get('proposal_fold', fold) == fold, 'Proposal fold disagrees with actual partition')
        require(row['payload']['language'] == row['language'], 'Payload language differs')
        payload_id = normalized_payload_identity(row['payload'])
        require(type(row['source_focus']) is int and row['source_focus'] == len(row['payload']['turns'])-1
                and row['payload']['turns'][-1]['role'] == 'minor',
                'Probe input must end at its declared focal minor turn')
        require(payload_id not in protected, 'Input overlaps a protected population')
        require(payload_id not in inputs, 'Repeated normalized input inside or across probe blocks')
        inputs[payload_id] = identity
        sibling = key[:4]+key[5:]
        labels = digest([attributes, row['proposed_gold']])
        require(sibling not in siblings or siblings[sibling] == labels, 'Bilingual sibling proposals differ')
        siblings[sibling] = labels
        counts[fold] += 1
        blocks[(key[0], key[1])] = fold
    require(seen == expected, 'Missing probe cells; every registered sibling remains required')
    require(set(counts) == set(FOLDS), 'Missing a fitting or probe population')
    return dict(verification='PASS', rows=len(rows), unique_normalized_inputs=len(inputs),
        blocks=len(blocks), counts=dict(counts), bilingual_sibling_groups=len(siblings),
        protected_unique_inputs_checked=len(protected), design_sha256=digest(plan), rows_sha256=digest(rows),
        training_admitted=False, semantic_labels_validated=False, token_overlap_checked=False,
        held_out_test_opened=False, independently_validated=False, qualified=False,
        scope='Complete declared block assignment, sibling coverage and normalized role-preserving input checks only. '
              'Family margins are held out from the proposed additional fitting, not proven absent from prior '
              'model training. Semantic review, token checks, complete protected-source audit and new registration remain required.')
