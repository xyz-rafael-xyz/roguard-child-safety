"""Validate paired wording inputs against exact original feature positions."""
from collections import Counter

from roguard.assurance import digest
from roguard.evidence import ONTOLOGY
from roguard.gemma4_contract import PREFIX_TOKENS
from roguard.trajectory_data import prefix


def validate_inputs(treatment, controls, inputs, metadata):
    if not treatment or len(treatment) != len(controls) or len(treatment) != len(inputs):
        raise ValueError('Incomplete paired input inventory')
    result = {lang: [] for lang in ('ro', 'uk')}; seen = set(); token_keys = set(); factors = {}
    protected = {lang: {r['token_sha256'] for r in metadata[lang]} for lang in result}
    for changed, control, record in zip(treatment, controls, inputs):
        lang = changed['language']; parent = changed['parent']
        index = parent['feature_index']
        if lang not in result or type(index) is not int or not 0 <= index < len(metadata[lang]):
            raise ValueError('Invalid original feature position')
        original = metadata[lang][index]
        if (original['partition'] != 'train' or parent['language'] != lang
                or any(parent[k] != original[k] for k in ('case_id', 'language', 'gold', 'fields', 'token_sha256'))):
            raise ValueError('Control does not bind an original TRAIN feature row')
        if (record['parent'] != parent or control['parent'] != parent
                or record['case_id'] != changed['case_id']
                or record['language'] != lang or control['language'] != lang
                or not (changed['pair_id'] == control['pair_id'] == record['pair_id'])):
            raise ValueError('Treatment/control identity mismatch')
        if changed['arm'] != 'treatment' or control['arm'] != 'control':
            raise ValueError('Arm roles differ')
        for row in (changed, control):
            if row['case_id'] in seen:
                raise ValueError('Duplicate example identity')
            seen.add(row['case_id'])
            if (row['partition'] != 'train' or row['review'] != original['gold']
                    or [ONTOLOGY[k].index(row['attributes'][k]) for k in ONTOLOGY] != original['fields']):
                raise ValueError('Paired labels or partitions differ')
            if row['focus'] != len(row['turns'])-1:
                raise ValueError('Expected focal-last examples')
        if (record['partition'] != 'train' or record['gold'] != original['gold']
                or record['fields'] != original['fields'] or record['fields'][:2] != [0, 0]):
            raise ValueError('Runtime inputs exceed curated scope or have wrong labels')
        factor = changed['factors']; op = factor['operator_index']; placement = factor['quote_placement']
        if (type(op) is not int or op not in range(4) or placement not in ('before', 'after')
                or factor != {'operator_index': op, 'quote_placement': placement, 'operator_negation_marker': op >= 2}
                or factor != control['factors'] or factor != record['factors']):
            raise ValueError('Experimental factors differ')
        factors.setdefault((lang, index), []).append((op, placement))
        tokens = record['tokens']
        if (record['payload'] != prefix(changed, changed['focus']) or record['token_sha256'] != digest(tokens)
                or record['token_count'] != len(tokens) or not 0 < len(tokens) <= 1024
                or any(type(t) is not int or t < 0 for t in tokens)
                or tokens[:len(PREFIX_TOKENS)] != PREFIX_TOKENS):
            raise ValueError('Runtime payload or token binding differs')
        key = (lang, record['token_sha256'])
        if key in token_keys or record['token_sha256'] in protected[lang]:
            raise ValueError('Treatment tokens overlap another input')
        token_keys.add(key)
        if changed['turns'][:-1] != control['turns'][:-1]:
            raise ValueError('Treatment changes preceding context')
        left = changed['turns'][-1]['text']; right = control['turns'][-1]['text']
        if (left.count('«') != 1 or left.count('»') != 1 or right.count('«') != 1 or right.count('»') != 1
                or left.split('«')[1].split('»')[0] != right.split('«')[1].split('»')[0]):
            raise ValueError('Treatment changes the quoted core')
        result[lang].append({'row': changed, 'control': control, 'input': record, 'parent_index': index})
    if not all(result.values()):
        raise ValueError('Both languages are required')
    required = Counter((i, p) for i in range(4) for p in ('before', 'after'))
    if any(Counter(values) != required for values in factors.values()):
        raise ValueError('A parent is missing or repeats a factor cell')
    return result
