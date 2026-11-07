"""Create a self-contained, offline form from a blinded annotation packet."""
from __future__ import annotations

import argparse
from importlib.resources import files
import json
from pathlib import Path
import re

from .evidence import ONTOLOGY
from .trajectory_data import input_identity, validate_conversation


def validate_packet(packet):
    if (not isinstance(packet, dict) or set(packet) != {'format', 'annotator_id', 'instructions', 'items'}
            or packet['format'] != 'roguard-v10-blind-annotation-1'
            or not isinstance(packet['annotator_id'], str) or not packet['annotator_id'].strip()
            or not isinstance(packet['instructions'], str)
            or not isinstance(packet['items'], list) or not packet['items']):
        raise ValueError('Expected a nonempty blinded annotation packet')
    seen = set()
    fields = {'item_id', 'conversation', 'input_digest', 'review', 'world', 'outside_ontology', 'evidence', 'uncertain', 'notes'}
    for row in packet['items']:
        if not isinstance(row, dict) or set(row) != fields:
            raise ValueError('Unexpected annotation fields; do not include keys or model outputs')
        identifier = row['item_id']
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError('Invalid or duplicate annotation ID')
        seen.add(identifier); validate_conversation(row['conversation'])
        if row['input_digest'] != input_identity(row['conversation']):
            raise ValueError('Annotation input digest differs')
        if (any(row[k] is not None for k in ('review', 'world', 'outside_ontology', 'uncertain'))
                or row['evidence'] != [] or row['notes'] != ''):
            raise ValueError('Generate the form from the blank packet; resume answers in the form')
    return packet


def render(packet):
    validate_packet(packet)
    def embedded(value):
        return json.dumps(value, ensure_ascii=False, allow_nan=False).replace('&', '\\u0026').replace('<', '\\u003c')
    template = files('roguard').joinpath('trajectory-annotation.html').read_text(encoding='utf-8')
    script = files('roguard').joinpath('trajectory-annotation.js').read_text(encoding='utf-8')
    substitutions = {'__PACKET_JSON__': embedded(packet), '__ONTOLOGY_JSON__': embedded(ONTOLOGY), '__APPLICATION_SCRIPT__': script}
    return re.sub('|'.join(map(re.escape, substitutions)), lambda match: substitutions[match.group()], template)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('packet', type=Path); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    html = render(json.loads(a.packet.read_text(encoding='utf-8')))
    with a.output.open('x', encoding='utf-8') as stream:
        stream.write(html)
    print('Created offline annotation form. Distribute only this HTML to its assigned annotator.')


if __name__ == '__main__':
    main()
