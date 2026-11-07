"""Bounded extractive views with original offsets and complete source lineage.

The host owns the transcript. Selection is a proposal, never proof that omitted
text is irrelevant. No generated summary substitutes for source text.
"""
from __future__ import annotations

from copy import deepcopy
import math
import re

from .assurance import digest
from .trajectory_data import validate_conversation, input_identity

MEMORY_CONTRACT = 'sentence-offsets-focal-mandatory-v2'

def units(payload):
    validate_conversation(payload)
    result = []
    for index, turn in enumerate(payload['turns']):
        # Sentence/paragraph candidates keep exact original code-point offsets.
        for match in re.finditer(r'[^\n.!?…]*(?:[.!?…]+|(?=\n)|$)', turn['text']):
            a, b = match.start(), match.end()
            while a < b and turn['text'][a].isspace():
                a += 1
            if a < b:
                result.append({'turn': index, 'start': a, 'end': b,
                               'role': turn['role'], 'text': turn['text'][a:b]})
    if len(result) > 8192:
        raise ValueError('Too many candidate evidence spans')
    return result


def select(payload, scorer=None, *, budget=3000, strategy='learned', pinned=(), min_relevance=None):
    if type(budget) is not int or not 128 <= budget <= 12000:
        raise ValueError('Evidence budget must be 128 to 12000 characters')
    if strategy not in ('learned', 'recent', 'lexical'):
        raise ValueError('Unknown selection strategy')
    if min_relevance is not None and (type(min_relevance) not in (int,float) or not math.isfinite(min_relevance) or not 0<=min_relevance<=1):
        raise ValueError('Invalid learned relevance threshold')
    candidates = units(payload)
    focus = len(payload['turns'])-1
    focal_words = set(re.findall(r'\w+', payload['turns'][-1]['text'].casefold()))
    mandatory = set()
    pinned = set(pinned)
    if any(type(i) is not int or not 0 <= i <= focus for i in pinned):
        raise ValueError('Invalid pinned source turn')
    for i, candidate in enumerate(candidates):
        if candidate['turn'] == focus or candidate['turn'] in pinned:
            mandatory.add(i)
    scores = []
    for candidate in candidates:
        if strategy == 'learned':
            if scorer is None:
                raise ValueError('Learned selection requires a scorer')
            value = scorer(candidate['text'], payload['language'], candidate['role'])
        elif strategy == 'lexical':
            words = set(re.findall(r'\w+', candidate['text'].casefold()))
            value = len(words & focal_words)/max(1, len(words | focal_words))
        else:
            value = candidate['turn']
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('Nonfinite evidence score')
        scores.append(value)
    chosen = set(mandatory)
    cost = lambda i: len(candidates[i]['text'])+32  # conservative role/turn framing
    used = sum(cost(i) for i in chosen)
    overflow = used > budget
    if not overflow:
        for i in sorted(range(len(candidates)), key=lambda i: (-scores[i], -candidates[i]['turn'], i)):
            if strategy=='learned' and min_relevance is not None and scores[i]<min_relevance:
                continue
            if i not in chosen and used+cost(i) <= budget:
                chosen.add(i)
                used += cost(i)
    selected = [deepcopy(candidates[i]) for i in sorted(chosen)] if not overflow else []
    return {'format': 'roguard-memory-v10-research1', 'contract': MEMORY_CONTRACT,
            'input_digest': input_identity(payload),
            'language': payload['language'], 'focus': focus, 'strategy': strategy,
            'budget': budget, 'min_relevance': min_relevance, 'used_characters': used if not overflow else 0,
            'spans': selected, 'pinned_turns': sorted(pinned), 'consulted_turns': list(range(focus+1)),
            'omitted_spans': len(candidates)-len(selected), 'mandatory_overflow': overflow,
            'semantic_sufficiency_verified': False, 'permits_external_action': False}


def verify_view(payload, view):
    if view.get('contract') != MEMORY_CONTRACT:
        raise ValueError('Unknown memory extraction contract')
    if view['input_digest'] != input_identity(payload) or view['language'] != payload['language'] or view['focus'] != len(payload['turns'])-1:
        raise ValueError('Memory view belongs to another prefix')
    if view['consulted_turns'] != list(range(len(payload['turns']))):
        raise ValueError('Incomplete selection provenance')
    if type(view['budget']) is not int or not 128<=view['budget']<=12000 or type(view['mandatory_overflow']) is not bool:
        raise ValueError('Invalid memory budget declaration')
    if view['mandatory_overflow'] and view['spans']:
        raise ValueError('Overflow cannot return partial mandatory evidence')
    used=sum(len(s['text'])+32 for s in view['spans'])
    if used!=view['used_characters'] or used>view['budget']:
        raise ValueError('Memory exceeds its declared budget')
    previous = None
    candidate_keys={(s['turn'],s['start'],s['end']) for s in units(payload)}
    selected_keys=set()
    for span in view['spans']:
        key = (span['turn'], span['start'], span['end'])
        if any(type(x) is not int for x in key) or (previous is not None and key <= previous):
            raise ValueError('Invalid span ordering')
        if not 0 <= key[0] < len(payload['turns']):
            raise ValueError('Future or unknown source turn')
        turn = payload['turns'][key[0]]
        if not 0 <= key[1] < key[2] <= len(turn['text']) or span['text'] != turn['text'][key[1]:key[2]] or span['role'] != turn['role']:
            raise ValueError('Source span differs')
        previous = key
        if key not in candidate_keys:
            raise ValueError('Memory span is not an extracted source unit')
        selected_keys.add(key)
    pinned=view.get('pinned_turns',[])
    if not isinstance(pinned,list) or any(type(t) is not int or not 0<=t<len(payload['turns']) for t in pinned):
        raise ValueError('Invalid pinned turns')
    mandatory={k for k in candidate_keys if k[0] in set(pinned+[len(payload['turns'])-1])}
    mandatory_cost=sum(k[2]-k[1]+32 for k in mandatory)
    if view['mandatory_overflow'] != (mandatory_cost>view['budget']):
        raise ValueError('Incorrect mandatory overflow declaration')
    if not view['mandatory_overflow'] and not mandatory<=selected_keys:
        raise ValueError('Memory omits mandatory focal or pinned evidence')
    return True


def render(view):
    return '\n'.join(f"[{s['role']}] {s['text']}" for s in view['spans'])


def derive_memory(ledger, artifact_id, payload, view, source_ids):
    """Inherit every consulted source, including sources omitted from the view.

    source_ids must be host-registered FlowLedger artifacts in turn order.
    Live permission/revocation checks remain with the existing FlowLedger API.
    """
    verify_view(payload, view)
    if len(source_ids) != len(payload['turns']) or len(set(source_ids)) != len(source_ids):
        raise ValueError('Require one distinct registered source per turn')
    text = render(view)
    if not text or len(text) > 6000:
        raise ValueError('Memory does not fit the existing ledger artifact limit')
    # Callers bind registered source IDs to this payload. The ledger checks that
    # every parent exists and inherits their complete root restrictions.
    for source in source_ids:
        ledger.describe(source)  # preflight unknown sources before mutation
    parents=list(source_ids)
    if len(parents)>32:
        groups=[]
        for i in range(0,len(parents),32):
            group='v10_'+digest([artifact_id,view['input_digest'],i])[:40]
            groups.append((group,parents[i:i+32]))
        for group,_ in groups:
            try:ledger.describe(group)
            except KeyError:pass
            else:raise ValueError('Provenance group already exists')
        for group,members in groups:
            ledger.derive(group,'Selection source lineage',members,kind='memory')
        parents=[g for g,_ in groups]
    ledger.derive(artifact_id, text, parents, kind='memory', complete=True)
    return {'artifact_id': artifact_id, 'input_digest': view['input_digest'],
            'view_digest': digest(view), 'source_ids': list(source_ids),
            'source_binding_host_supplied': True, 'permits_external_action': False}
