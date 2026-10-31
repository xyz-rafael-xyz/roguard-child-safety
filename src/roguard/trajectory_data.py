"""Validated conversation studies. Labels never enter the model input."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import unicodedata

from .assurance import digest
from .evidence import ONTOLOGY

FORMAT = 'roguard-trajectory-v10-research1'


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)+'\n' for row in rows), encoding='utf-8')


def validate_conversation(payload):
    if not isinstance(payload, dict) or set(payload) != {'language', 'turns'}:
        raise ValueError('Expected only language and turns')
    turns = payload['turns']
    if payload['language'] not in ('ro', 'uk') or not isinstance(turns, list) or not 1 <= len(turns) <= 512:
        raise ValueError('Expected ro/uk and 1 to 512 turns')
    total = 0
    for turn in turns:
        if not isinstance(turn, dict) or set(turn) != {'role', 'text'}:
            raise ValueError('A turn contains only role and text')
        if turn['role'] not in ('minor', 'assistant', 'other') or not isinstance(turn['text'], str) or not 1 <= len(turn['text'].strip()) <= 6000:
            raise ValueError('Invalid role or turn text')
        total += len(turn['text'])
    if total > 500_000:
        raise ValueError('Conversation exceeds character limit')
    return payload


def prefix(case, focus):
    payload = validate_conversation({'language': case['language'], 'turns': case['turns']})
    if type(focus) is not int or not 0 <= focus < len(payload['turns']):
        raise ValueError('Invalid focal turn')
    return {'language': payload['language'], 'turns': payload['turns'][:focus+1]}


def validate_case(row):
    required = {'format', 'case_id', 'language', 'turns', 'split', 'family', 'author', 'generator', 'synthetic', 'checkpoints'}
    if not isinstance(row, dict) or not required <= set(row) or row['format'] != FORMAT:
        raise ValueError('Invalid trajectory case')
    validate_conversation({'language': row['language'], 'turns': row['turns']})
    if row['split'] not in ('train', 'dev', 'calibration', 'test', 'challenge') or type(row['synthetic']) is not bool:
        raise ValueError('Invalid split or provenance')
    for key in ('case_id', 'family', 'author', 'generator'):
        if not isinstance(row[key], str) or not row[key].strip():
            raise ValueError('Missing provenance: '+key)
    if not isinstance(row['checkpoints'], list) or not row['checkpoints']:
        raise ValueError('No annotated checkpoints')
    focuses = set()
    for point in row['checkpoints']:
        focus = point['focus']
        prefix(row, focus)
        if focus in focuses or (type(point['review']) is not int or point['review'] not in (0, 1)):
            raise ValueError('Duplicate checkpoint or invalid label')
        focuses.add(focus)
        if 'outside_ontology' in point and type(point['outside_ontology']) is not bool:
            raise ValueError('Outside-ontology declaration must be boolean')
        world = point.get('world')
        if world is not None and (not isinstance(world, dict) or set(world) != set(ONTOLOGY) or any(world[k] not in v for k, v in ONTOLOGY.items())):
            raise ValueError('Invalid annotated interpretation')
        spans = point.get('evidence', [])
        for span in spans:
            t, start, end = span['turn'], span['start'], span['end']
            if any(type(v) is not int for v in (t, start, end)) or not 0 <= t <= focus or not 0 <= start < end <= len(row['turns'][t]['text']):
                raise ValueError('Invalid or future evidence span')
    return row


def input_identity(payload):
    return digest(validate_conversation(payload))


def audit_splits(rows, *, independent_test=False):
    """Reject exact prefix leakage and family leakage, including translated pairs.

    Independent-test mode additionally requires disjoint authors and generators.
    These are declared identifiers; this cannot verify a person's independence.
    """
    seen_ids, inputs, families, authors, generators = set(), {}, {}, {}, {}
    counts = {}
    for row in rows:
        validate_case(row)
        if row['case_id'] in seen_ids:
            raise ValueError('Duplicate case ID')
        seen_ids.add(row['case_id'])
        split = row['split']
        counts[split] = counts.get(split, 0)+1
        for key, table in [('family', families), ('author', authors), ('generator', generators)]:
            table.setdefault(row[key], set()).add(split)
        for point in row['checkpoints']:
            p = prefix(row, point['focus'])
            normalized = [(t['role'], re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', t['text']).casefold()).strip()) for t in p['turns']]
            identity = digest([p['language'], normalized])
            if identity in inputs and inputs[identity] != split:
                raise ValueError('Normalized input overlaps splits')
            inputs[identity] = split
    if any(len(v) != 1 for v in families.values()):
        raise ValueError('Scenario family overlaps splits')
    if independent_test:
        for table in (authors, generators):
            if any('test' in v and len(v) > 1 for v in table.values()):
                raise ValueError('Independent test shares author or generator')
        if not counts.get('test'):
            raise ValueError('Independent test is absent')
    return {'cases': counts, 'unique_prefixes': len(inputs), 'families': len(families),
            'content_digest': digest(rows), 'independence_verified': False,
            'declared_independent_test_separation_checked': independent_test}


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()
