"""Validate two annotation returns and prepare blind disagreement adjudication."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import random

from roguard.assurance import digest
from roguard.trajectory_annotation import render, validate_packet
from roguard.trajectory_data import file_sha
from roguard.trajectory_review import annotation_agreement, validate_annotations


def coordinate(packet_a, key_a, completed_a, packet_b, key_b, completed_b, *, adjudicator_id):
    for packet in (packet_a, packet_b): validate_packet(packet)
    identifiers = [packet_a['annotator_id'], packet_b['annotator_id'], adjudicator_id]
    if (not isinstance(adjudicator_id, str) or not adjudicator_id.strip()
            or len(set(identifiers)) != 3):
        raise ValueError('Distinct annotator and adjudicator identifiers are required')
    left = validate_annotations(packet_a, key_a, completed_a)
    right = validate_annotations(packet_b, key_b, completed_b)
    def sources(packet, key):
        result = {}
        for row in packet['items']:
            binding = key['key'][row['item_id']]; identifier = binding['checkpoint_id']
            if identifier in result or binding['input_digest'] != row['input_digest']:
                raise ValueError('Malformed coordinator input mapping')
            result[identifier] = row
        return result
    a, b = sources(packet_a, key_a), sources(packet_b, key_b)
    if set(a) != set(b) or any(a[k]['input_digest'] != b[k]['input_digest'] or a[k]['conversation'] != b[k]['conversation'] for k in a):
        raise ValueError('Annotators did not receive the same conversation prefixes')
    agreement = annotation_agreement(left, right)
    unresolved = [identifier for identifier in left if any(
        labels[identifier]['uncertain'] or labels[identifier]['review'] is None for labels in (left, right))]
    agreement['unresolved_checkpoints'] = sorted(unresolved)
    agreement['requires_adjudication'] = sorted(set(agreement['requires_adjudication']) | set(unresolved))
    conflicts = agreement['requires_adjudication']; rows = []; mapping = {}
    for identifier in conflicts:
        row = deepcopy(a[identifier]); token = digest([adjudicator_id, identifier, row['input_digest']])[:24]
        if token in mapping: raise ValueError('Adjudication item collision')
        row['item_id'] = token; rows.append(row)
        mapping[token] = {'checkpoint_id': identifier, 'input_digest': row['input_digest']}
    random.Random(1049).shuffle(rows)
    packet = {'format': 'roguard-v10-blind-annotation-1', 'annotator_id': adjudicator_id,
              'instructions': 'Give an independent third judgment of each displayed prefix before discussing earlier annotations. Null means unresolved, never negative. Do not infer event truth.',
              'items': rows}
    key = {'packet_digest': digest(packet), 'annotator_id': adjudicator_id, 'key': mapping}
    report = {'format': 'roguard-v10-annotation-coordination-1', 'complete': True,
              'agreement': agreement, 'validated_annotations_a': left, 'validated_annotations_b': right,
              'independence_verified': False, 'human_identity_verified': False,
              'gold_labels_changed': False, 'stable_release_ready': False,
              'next_step': 'Obtain an independent adjudicator return and record final decisions with reasons; no automatic majority vote or model promotion.'}
    return report, packet if rows else None, key if rows else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    names = ['packet_a', 'key_a', 'completed_a', 'packet_b', 'key_b', 'completed_b']
    for name in names: p.add_argument('--'+name.replace('_', '-'), type=Path, required=True)
    p.add_argument('--adjudicator-id', required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise ValueError('Refusing to overwrite a coordination result')
    records = [json.loads(getattr(a, name).read_text(encoding='utf-8')) for name in names]
    report, packet, key = coordinate(*records, adjudicator_id=a.adjudicator_id)
    report['input_sha256'] = {name: file_sha(getattr(a, name)) for name in names}
    report['source_sha256'] = file_sha(__file__)
    a.output.mkdir(parents=True)
    (a.output/'coordinator-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    if packet is not None:
        for name, value in (('adjudication-packet.json', packet), ('adjudication-PRIVATE-key.json', key)):
            (a.output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        (a.output/'adjudication.html').write_text(render(packet), encoding='utf-8')
    (a.output/'README.txt').write_text('Keep this directory with the coordinator. Only adjudication.html goes to the adjudicator. It hides both earlier answers and every model output. Validate the adjudicator return against its packet/key, then document final resolutions and reasons. Matching opaque identifiers cannot verify real participant independence. No labels or model defaults were changed.\n', encoding='utf-8')
    print(json.dumps({'items': report['agreement']['items'], 'requires_adjudication': len(report['agreement']['requires_adjudication']), 'gold_labels_changed': False, 'stable_release_ready': False}))


if __name__ == '__main__': main()
