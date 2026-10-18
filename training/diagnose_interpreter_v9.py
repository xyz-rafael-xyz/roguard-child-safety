"""Audit structured predictions and attribute policy errors to label corrections.

Corrections use supplied synthetic labels for offline diagnosis only. They are
not model predictions, causal explanations, or evidence of deployable accuracy.
"""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

from roguard.evidence import ONTOLOGY, disposition
from roguard.review_evaluation import evaluate, target_failures
from training.prepare_instruction_sft_v8 import target
from training.semantic_interpreter_v9 import FIELDS


def action(attributes):
    if set(attributes) != set(FIELDS) or any(attributes[f] not in ONTOLOGY[f] for f in FIELDS):
        raise ValueError('Invalid interpretation')
    return int(disposition({f: {'label': attributes[f]} for f in FIELDS})[0] != 'CONTINUE')


def corrections(predicted, gold):
    """All inclusion-minimal field repairs that restore the supplied gold action."""
    desired = action(gold)
    if action(predicted) == desired:
        return []
    different = [f for f in FIELDS if predicted[f] != gold[f]]
    repairs = []
    for size in range(1, len(different)+1):
        for fields in itertools.combinations(different, size):
            if any(set(previous) <= set(fields) for previous in repairs):
                continue
            fixed = dict(predicted)
            fixed.update({f: gold[f] for f in fields})
            if action(fixed) == desired:
                repairs.append(fields)
    return repairs


def diagnose(rows, predictions, reference=None):
    expected = [dict(case_id=r['case_id'], language=r['language'], gold=target(r),
                     kind='conversational' if r.get('source') == 'v7_training_frame_projection' else 'formal')
                for r in rows]
    strict = evaluate(expected, predictions)
    by_id = {p['case_id']: p for p in predictions}
    reference_by_id = None
    if reference is not None:
        evaluate(expected, reference)
        reference_by_id = {p['case_id']: p for p in reference}
    fields = {f: Counter() for f in FIELDS}
    repairs = Counter()
    transitions = Counter()
    errors = []
    for row, truth in zip(rows, expected):
        p = by_id[row['case_id']]
        if type(p.get('valid_label_tokens')) is not bool:
            raise ValueError('Missing strict schema validity')
        if p.get('gold') != truth['gold'] or type(p.get('gold')) is not int:
            raise ValueError('Prediction gold mismatch')
        predicted_action = action(p['interpretation'])
        if action(row['attributes']) != truth['gold']:
            raise ValueError('Gold policy disagrees with supplied attributes')
        if p['predicted'] != (predicted_action if p['valid_label_tokens'] else None):
            raise ValueError('Decision does not follow strict decoded policy')
        for f in FIELDS:
            fields[f][row['attributes'][f]+' -> '+p['interpretation'][f]] += 1
        if reference_by_id is not None:
            ref = reference_by_id[row['case_id']]
            if 'gold' in ref and ref['gold'] != truth['gold']:
                raise ValueError('Reference gold mismatch')
            before = 'correct' if ref['predicted'] == truth['gold'] else 'incorrect'
            after = 'correct' if p['predicted'] == truth['gold'] else 'incorrect'
            transitions[before+' -> '+after] += 1
        if p['predicted'] == truth['gold']:
            continue
        fixes = corrections(p['interpretation'], row['attributes'])
        for fix in fixes:
            repairs[' + '.join(fix)] += 1
        errors.append(dict(case_id=row['case_id'], language=row['language'], kind=truth['kind'],
                           gold=truth['gold'], predicted=p['predicted'],
                           invalid_schema=not p['valid_label_tokens'],
                           differing_fields=[f for f in FIELDS if p['interpretation'][f] != row['attributes'][f]],
                           minimal_policy_repairs=[list(f) for f in fixes]))
    return dict(synthetic_only=True, strict_evaluation=strict,
                target_failures=target_failures(strict), field_confusions=fields,
                minimal_policy_repairs=repairs, paired_reference_transitions=transitions,
                errors=errors, stable_release_ready=False,
                caveat='Gold-label repairs are offline diagnostics, never corrected accuracy. '
                       'Repairs may overlap; schema validity remains a separate error.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    def read(path):
        return [json.loads(line) for line in path.read_text().splitlines()]
    result = diagnose(read(args.data), read(args.predictions), read(args.reference) if args.reference else None)
    result['input_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (args.data, args.predictions, args.reference) if p is not None}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('errors', 'field_confusions')}, indent=2))


if __name__ == '__main__':
    main()
