"""Paired decision tests: invariance, directional transitions and abstention."""
from __future__ import annotations
import copy
import unicodedata
from .flow import identifier

KNOWN = frozenset({'ALLOW', 'BLOCK', 'REVIEW', 'REVIEW_SUPPORT_SIGNAL', 'NO_MODEL_SIGNAL', 'INSUFFICIENT_EVIDENCE'})
UNRESOLVED = frozenset({'REVIEW', 'INSUFFICIENT_EVIDENCE'})


def evaluate_consistency(pairs, predictor):
    if not isinstance(pairs, list) or not 1 <= len(pairs) <= 200 or not callable(predictor):
        raise ValueError('Provide 1 to 200 pairs and a predictor')
    names = set()
    for pair in pairs:
        if not isinstance(pair, dict) or not {'pair_id', 'relation', 'before', 'after'} <= pair.keys() or set(pair) - {'pair_id', 'relation', 'before', 'after', 'expected'}:
            raise ValueError('Invalid consistency pair')
        name = identifier(pair['pair_id'])
        if name in names or pair['relation'] not in ('invariant', 'transition'):
            raise ValueError('Duplicate pair or unknown relation')
        names.add(name)
        expected = pair.get('expected')
        if expected is not None and (not isinstance(expected, list) or len(expected) != 2 or any(not isinstance(v, str) or v not in KNOWN for v in expected)):
            raise ValueError('Expected decisions must be a pair of supported codes')
        if pair['relation'] == 'transition' and (expected is None or expected[0] == expected[1] or any(v in UNRESOLVED for v in expected)):
            raise ValueError('Transitions require two different resolved expected decisions')
        if pair['relation'] == 'invariant' and expected is not None and expected[0] != expected[1]:
            raise ValueError('Invariant expected decisions must match')
    results = []
    for pair in pairs:
        error = False; decisions = []
        for side in ('before', 'after'):
            try:
                value = predictor(copy.deepcopy(pair[side]))
                if not isinstance(value, str) or value not in KNOWN:
                    raise ValueError('Invalid predictor output')
                decisions.append(value)
            except Exception:
                decisions.append('INSUFFICIENT_EVIDENCE'); error = True
        unresolved = error or any(value in UNRESOLVED for value in decisions)
        expected = pair.get('expected')
        relation_met = decisions[0] == decisions[1] if pair['relation'] == 'invariant' else decisions == expected
        reference_met = decisions == expected if expected is not None else None
        passed = relation_met and reference_met is not False
        results.append({'pair_id': pair['pair_id'], 'relation': pair['relation'], 'decisions': decisions,
                        'status': 'UNRESOLVED' if unresolved else 'PASS' if passed else 'FAIL',
                        'relation_met': None if unresolved else relation_met,
                        'reference_met': None if unresolved else reference_met, 'predictor_error': error})
    return {'pairs': results, 'total': len(results), 'passed': sum(r['status'] == 'PASS' for r in results),
            'failed': sum(r['status'] == 'FAIL' for r in results),
            'unresolved': sum(r['status'] == 'UNRESOLVED' for r in results),
            'equivalent_meaning_verified': False, 'agreement_establishes_accuracy': False, 'text_in_report': False}


def surface_pairs(payloads):
    """Small declared surface-equivalence hypotheses; no automatic translation."""
    from .conversation import validate
    if not isinstance(payloads, list) or not 1 <= len(payloads) <= 20:
        raise ValueError('Provide 1 to 20 conversations')
    pairs = []
    for index, payload in enumerate(payloads):
        validate(payload)
        changes = {'upper': str.upper, 'spaces': lambda t: t.replace(' ', '  '),
                   'unicode': lambda t: unicodedata.normalize('NFD', t), 'punctuation': lambda t: t + ' ...'}
        if payload['language'] == 'ro':
            changes['diacritics'] = lambda t: ''.join(c for c in unicodedata.normalize('NFD', t) if not unicodedata.combining(c))
        elif payload['language'] == 'uk':
            changes['mixed_i'] = lambda t: t.replace('і', 'i')
        for name, transform in changes.items():
            after = copy.deepcopy(payload)
            for turn in after['turns']:
                turn['text'] = transform(turn['text'])
            pairs.append({'pair_id': f'case_{index}_{name}', 'relation': 'invariant',
                          'before': copy.deepcopy(payload), 'after': after})
    return pairs


def conversation_consistency(payloads, scorer=None):
    from .conversation import analyze_conversation, default_scorer
    scorer = scorer or default_scorer()
    result = evaluate_consistency(surface_pairs(payloads), lambda p: analyze_conversation(p, scorer)['signal']['decision'])
    result['backend'] = scorer.backend
    result['semantic_head_sha256'] = getattr(scorer, 'head_sha256', None)
    return result
