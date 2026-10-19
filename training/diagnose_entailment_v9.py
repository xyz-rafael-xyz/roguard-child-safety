"""Summarize development-only errors by policy reason and language attributes."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from training.entailment_frontier_v9 import frontier
from training.policy_entailment_v9 import targets
from training.train_bare_distress_v9 import digest, read
from training.prepare_instruction_sft_v8 import target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recall-budget', action='store_true', help='Diagnose the existing 99% recall gate rather than zero misses')
    args = parser.parse_args()
    selected_frontier = frontier
    frontier_module = Path('training/entailment_frontier_v9.py')
    frontier_key = 'full_coverage_two_cutoff_frontier'
    if args.recall_budget:
        from training.budgeted_entailment_frontier_v9 import frontier as selected_frontier
        frontier_module = Path('training/budgeted_entailment_frontier_v9.py')
        frontier_key = 'recall_budget_two_cutoff_frontier'
    root = args.candidate
    report = json.loads((root/'development.json').read_text())
    if report.get('complete') is not True:
        raise ValueError('Candidate must be complete before diagnosis')
    spec = json.loads((root/'review.json').read_text())
    if digest(root/'review.json') != report['artifact_sha256']:
        raise ValueError('Artifact binding differs')
    hashes = {str(root/name): digest(root/name) for name in ('development.json', 'review.json', 'train.jsonl.gz')}
    training = Counter((r['language'], *targets(r)) for r in read(root/'train.jsonl.gz'))
    languages = {}
    for language in ('ro', 'uk'):
        population = []
        groups = defaultdict(lambda: dict(rows=0, positives=0, negatives=0, misses=0, false_reviews=0))
        weakest = []
        splits = (language+'_dev', language+'_scope_dev') + (('uk_stress',) if language == 'uk' else ())
        for split in splits:
            inputs = root/(split+'.jsonl.gz')
            predictions = root/(split+'-predictions.jsonl')
            hashes[str(inputs)] = digest(inputs)
            hashes[str(predictions)] = digest(predictions)
            rows = read(inputs)
            values = [json.loads(line) for line in predictions.read_text().splitlines()]
            if len(rows) != len(values):
                raise ValueError('Incomplete development predictions')
            limits = spec['thresholds'][language]
            if not isinstance(limits, dict):
                limits = dict(support=limits, revision=limits)
            for row, value in zip(rows, values):
                gold = int(row['review']) if 'review' in row else target(row)
                predicted = int(value['support_score'] >= limits['support'] or value['revision_score'] >= limits['revision'])
                if row['case_id'] != value['case_id'] or value['gold'] != gold or value['predicted'] != predicted:
                    raise ValueError('Development prediction binding differs')
                group = ('conversational' if row.get('source') == 'v7_training_frame_projection' else 'formal') if split == language+'_dev' else split
                population.append(dict(value, group=group))
                factor = targets(row)
                attributes = row['attributes']
                for key in ('all', 'reason:'+str(factor), 'assertion:'+attributes['assertion'],
                            'subject:'+attributes['subject'], 'time:'+attributes['time']):
                    cell = groups[group+'/'+key]
                    cell['rows'] += 1
                    cell['positives'] += gold
                    cell['negatives'] += 1-gold
                    cell['misses'] += int(gold == 1 and predicted == 0)
                    cell['false_reviews'] += int(gold == 0 and predicted == 1)
                if gold:
                    weakest.append(dict(case_id=row['case_id'], group=group, attributes=attributes,
                                        support_score=value['support_score'], revision_score=value['revision_score'],
                                        score=value['score']))
        languages[language] = dict(selected_counts=dict(sorted(groups.items())),
                                   lowest_positive_scores=sorted(weakest, key=lambda v: v['score'])[:12])
        languages[language][frontier_key] = selected_frontier(population)
    result = dict(format='roguard-entailment-development-diagnosis-v1', source_sha256=digest(Path(__file__)),
                  input_sha256=hashes, frontier_source_sha256=digest(frontier_module),
                  minimum_development_recall=.99 if args.recall_budget else 1.,
                  synthetic_only=True, calibration_read=False, fresh_test_read=False,
                  model_changed=False, training_reason_counts=[dict(language=l, support=s, revision=r, rows=n)
                                                              for (l, s, r), n in sorted(training.items())],
                  languages=languages)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps({l: v[frontier_key] for l, v in languages.items()}))


if __name__ == '__main__':
    main()
