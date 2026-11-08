"""Structural role-token alignment for the optional experimental residual.

No encoder, filesystem discovery, labels, or data admission. Pure functions
preserve the reviewed research alignment AST; see the lineage fixture.
"""
from collections import Counter
import hashlib
import json
import unicodedata

RESEARCH_SOURCE_SHA256 = "d684b8b592eed2e5241945211bf689097ca09a3bb44b58eb4dba3b7650bdbc5c"
ROLE_POOLS = ('current_minor', 'prior_minor', 'prior_nonminor')
OTHER_ROLES = ('assistant', 'adult', 'other', 'unknown')

def require(ok, message):
    if not ok:
        raise ValueError(message)

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def serialize_with_spans(prefix):
    """Construct exactly json.dumps' frozen layout and track content interiors.

    Spans refer to NFC text after JSON escaping, excluding its surrounding quote
    delimiters. Each escaped codepoint retains a normalized-text source offset.
    No unescaping/replacement is applied to the string fed to the tokenizer.
    """
    turns = [dict(role=t['role'], text=unicodedata.normalize('NFC', t['text'])) for t in prefix['turns']]
    chunks = []; spans = []; position = 0
    def add(value):
        nonlocal position
        chunks.append(value); position += len(value)
    def turn(index):
        t = turns[index]
        add('{"role": ' + json.dumps(t['role'], ensure_ascii=False) + ', "text": "')
        start = position
        mapping = []
        for offset, char in enumerate(t['text']):
            escaped = json.dumps(char, ensure_ascii=False)[1:-1]
            char_start = position; add(escaped)
            mapping.append(dict(serialized_start=char_start, serialized_end=position,
                                normalized_text_start=offset, normalized_text_end=offset + 1))
        end = position; add('"}')
        group = ('current_minor' if index == len(turns) - 1 else
                 'prior_minor' if t['role'] == 'minor' else 'prior_nonminor')
        spans.append(dict(turn_index=index, role=t['role'], pool=group, start=start, end=end,
                          normalized_text=t['text'], escaped_text=''.join(chunks)[start:end],
                          normalized_codepoint_map=mapping))
    add('{"prior_turns": [')
    for index in range(len(turns) - 1):
        if index: add(', ')
        turn(index)
    add('], "focal_message": '); turn(len(turns) - 1); add('}')
    result = ''.join(chunks)
    expected = json.dumps(dict(prior_turns=turns[:-1], focal_message=turns[-1]), ensure_ascii=False)
    require(result == expected, 'Tracked serialization differs from serving JSON')
    return result, spans

def select_offsets(offsets, token_ids, spans, special_ids, prefix_length):
    """Assign only non-special tokens wholly inside one content span.

    Offset bounds are Python Unicode-codepoint positions in the rendered chat.
    Repeated/overlapping offsets from byte/Unicode tokenization are permitted;
    each token position is assigned at most once. A content/delimiter straddle
    remains in classifier input but is excluded from every pooling arm.
    """
    pools = {name: [] for name in ROLE_POOLS}; exclusions = []; assignments = []
    require(len(offsets) == len(token_ids), 'Offset/token count differs')
    for index, (token_id, offset) in enumerate(zip(token_ids, offsets)):
        start, end = offset
        require(type(start) is int and type(end) is int and 0 <= start <= end, 'Invalid tokenizer offset')
        inside = [span for span in spans if span['start'] <= start < end <= span['end']]
        overlaps = [span for span in spans if start < span['end'] and end > span['start']]
        require(len(inside) <= 1, 'Overlapping content ownership spans')
        if token_id in special_ids:
            reason = 'SPECIAL_TOKEN_ID'
        elif start == end:
            reason = 'EMPTY_OFFSET'
        elif inside:
            span = inside[0]
            require(index >= prefix_length, 'Content assigned to invariant cached prefix')
            pools[span['pool']].append(index)
            assignments.append(dict(token_index=index, turn_index=span['turn_index'], role=span['role'],
                                    pool=span['pool'], rendered_start=start, rendered_end=end))
            continue
        elif overlaps:
            reason = 'STRADDLES_CONTENT_BOUNDARY'
        else:
            reason = 'OUTSIDE_CONTENT'
        exclusions.append(dict(token_index=index, reason=reason, overlaps_content=bool(overlaps),
                               rendered_start=start, rendered_end=end))
    return pools, assignments, exclusions

def three_contiguous_pools(content_positions):
    """Near-equal contiguous thirds of the ordered eligible content stream.

    Sizes differ by at most1; leading buckets receive the remainder. No roles,
    speaker labels or text meanings enter this control. Delimiter gaps remain
    gaps in absolute token positions; contiguity is in eligible-content order.
    """
    require(content_positions == sorted(set(content_positions)), 'Content stream must be unique and ordered')
    base, remainder = divmod(len(content_positions), 3)
    pools = []; start = 0
    for i in range(3):
        size = base + int(i < remainder)
        pools.append(content_positions[start:start + size]); start += size
    require(start == len(content_positions), 'Control pool coverage differs')
    return pools

def align(payload, tokenizer, focus=None):
    from roguard.interpreter import prepare_input
    from roguard.gemma4_contract import POLICY, PREFIX_TOKENS, classifier_tokens
    prefix, selected_focus, identity = prepare_input(payload, focus)
    canonical_payload, spans = serialize_with_spans(prefix)
    messages = [dict(role='system', content=POLICY), dict(role='user', content=canonical_payload),
                dict(role='assistant', content='0')]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False)
    require(rendered.count(canonical_payload) == 1, 'Serving template did not preserve one exact payload')
    shift = rendered.index(canonical_payload)
    absolute_spans = [dict(span, start=span['start'] + shift, end=span['end'] + shift) for span in spans]
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    native_full = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, enable_thinking=False)
    require(encoded['input_ids'] == native_full, 'Offset encoding changed native full-template token IDs')
    native_classifier = classifier_tokens(tokenizer, prefix)
    require(native_full[:len(native_classifier)] == native_classifier, 'Classifier prefix changed')
    token_ids = encoded['input_ids'][:len(native_classifier)]
    offsets = encoded['offset_mapping'][:len(native_classifier)]
    require(token_ids == native_classifier and token_ids[:len(PREFIX_TOKENS)] == PREFIX_TOKENS,
            'Classifier IDs or fixed cached prefix differ')
    require(all(end <= len(rendered) for start, end in offsets), 'Offsets exceed rendered chat')
    special_ids = set(tokenizer.all_special_ids) | {key for key, value in tokenizer.added_tokens_decoder.items() if value.special}
    pools, assignments, excluded = select_offsets(offsets, token_ids, absolute_spans, special_ids, len(PREFIX_TOKENS))
    content_positions = sorted(index for pool in pools.values() for index in pool)
    require(len(content_positions) == len(set(content_positions)), 'Role masks overlap')
    control = three_contiguous_pools(content_positions)
    role_presence = [any(span['pool'] == name for span in spans) for name in ROLE_POOLS]
    pool_validity = [bool(pools[name]) for name in ROLE_POOLS]
    return dict(format='roguard-role-token-alignment-1', language=prefix['language'], focus=selected_focus,
        serving_input_identity=identity, normalized_content_identity=digest([(span['role'], span['normalized_text']) for span in spans]),
        classifier_token_ids=token_ids, classifier_token_sha256=digest(token_ids), classifier_token_count=len(token_ids),
        classifier_ids_unchanged=True, fixed_prefix_length=len(PREFIX_TOKENS),
        rendered_chat=rendered, rendered_classifier_offsets=[list(value) for value in offsets],
        serialized_payload=canonical_payload, payload_rendered_start=shift, content_spans=absolute_spans,
        role_pool_names=list(ROLE_POOLS), role_pool_positions=[pools[name] for name in ROLE_POOLS],
        role_pool_validity=pool_validity, shared_structural_role_presence=role_presence,
        role_blind_pool_positions=control, role_blind_pool_validity=[bool(pool) for pool in control],
        final_token_index=len(token_ids) - 1, final_token_only_auxiliary_positions=[[len(token_ids) - 1] for _ in range(3)],
        eligible_content_positions=content_positions, assignments=assignments, exclusions=excluded,
        exclusion_counts=dict(Counter(row['reason'] for row in excluded)),
        structural_roles_present=dict(Counter(span['role'] for span in spans)),
        content_span_without_eligible_tokens=[span['turn_index'] for span in spans
            if not any(a['turn_index'] == span['turn_index'] for a in assignments)],
        research_600_token_envelope=len(token_ids) <= 600, serving_1024_token_limit=len(token_ids) <= 1024,
        token_truncation_applied=False, encoder_loaded=False, head_loaded=False, gold_metadata_used=False,
        training_admitted=False, qualified=False)
