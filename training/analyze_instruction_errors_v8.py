"""Summarize development errors by annotated context without printing messages."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from roguard.review_evaluation import evaluate
from training.prepare_instruction_sft_v8 import target


def analyze(rows, predictions):
    expected = [dict(case_id=r['case_id'], language=r['language'], gold=target(r)) for r in rows]
    evaluate(expected, predictions)  # Require exact ID coverage and valid decisions.
    indexed = {r['case_id']: r for r in predictions}
    groups = defaultdict(lambda: dict(cases=0, concerns=0, benign=0, missed=0, false_reviews=0, unresolved=0))
    for row in rows:
        prediction = indexed[row['case_id']]['predicted']
        gold = target(row)
        fields = dict(row['attributes'], input_source=row.get('source', 'unknown'))
        for field, value in fields.items():
            key = '/'.join((row['language'], field, str(value)))
            counts = groups[key]
            counts['cases'] += 1
            counts['concerns'] += gold == 1
            counts['benign'] += gold == 0
            counts['missed'] += gold == 1 and prediction != 1
            counts['false_reviews'] += gold == 0 and prediction != 0
            counts['unresolved'] += prediction is None
    return {'development_diagnostic_only': True, 'groups': dict(sorted(groups.items())),
            'limitation': 'Overlapping, correlated slices; not independent tests or causal explanations.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.data.read_text(encoding='utf-8').splitlines()]
    predictions = [json.loads(line) for line in args.predictions.read_text(encoding='utf-8').splitlines()]
    result = analyze(rows, predictions)
    result['input_sha256'] = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for name, path in (('data', args.data), ('predictions', args.predictions))}
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(result, output, indent=2)
        output.write('\n')


if __name__ == '__main__':
    main()
