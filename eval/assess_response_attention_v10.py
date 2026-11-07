"""Match every response diagnostic to its input and prior-checker result."""
import argparse
import json
from pathlib import Path

from roguard.trajectory_data import file_sha, read_jsonl
from roguard.trajectory_gate import _response_assessment


def compare(expected, baseline, candidate):
    fields = ('language', 'context', 'draft', 'gold')
    def indexed(rows):
        result = {r['case_id']: r for r in rows}
        if len(result) != len(rows):
            raise ValueError('Duplicate response case ID')
        return result
    gold = indexed(expected); old = indexed(baseline); new = indexed(candidate)
    if not gold or set(gold) != set(old) or set(gold) != set(new):
        raise ValueError('Response population is incomplete or contains extra cases')
    for identifier, row in gold.items():
        for prediction in (old[identifier], new[identifier]):
            if any(prediction.get(k) != row.get(k) for k in fields):
                raise ValueError('Matched response input or gold label differs')
    for rows in (baseline, candidate):
        summary, failures = _response_assessment(rows)
        if summary['invalid_rows'] or summary['missing_groups']:
            raise ValueError('Invalid or incomplete bilingual response results')
    pairs = []
    for identifier, row in gold.items():
        before = old[identifier]['prediction']['label'] == row['gold']
        after = new[identifier]['prediction']['label'] == row['gold']
        pairs.append({'case_id': identifier, 'language': row['language'],
                      'baseline_correct': before, 'candidate_correct': after})
    return {
        'rows': len(pairs), 'distinct_language_drafts': len({(r['language'], r['draft']) for r in expected}),
        'baseline_errors': sum(not r['baseline_correct'] for r in pairs),
        'candidate_errors': sum(not r['candidate_correct'] for r in pairs),
        'fixed': sum(not r['baseline_correct'] and r['candidate_correct'] for r in pairs),
        'introduced': sum(r['baseline_correct'] and not r['candidate_correct'] for r in pairs),
        'changed_cases': [r for r in pairs if r['baseline_correct'] != r['candidate_correct']],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'data', 'parent', 'context', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Refusing to overwrite response assessment')
    report = json.loads((a.study/'report.json').read_text())
    if not report['complete'] or file_sha(a.study/'selection.json') != report['selection_sha256']:
        raise ValueError('Incomplete response run or changed selection')
    paths = {'development': (a.data/'dev.jsonl', a.parent/'development-predictions.jsonl', a.study/'development-predictions.jsonl'),
             'context': (a.context/'inputs.jsonl', a.context/'baseline26-predictions.jsonl', a.study/'context-predictions.jsonl')}
    metrics = {name: compare(*(read_jsonl(path) for path in sources)) for name, sources in paths.items()}
    for name, result in metrics.items():
        if result['candidate_errors'] != report['metrics'][name]['errors'] or result['rows'] != report['metrics'][name]['rows']:
            raise ValueError('Recorded response metrics differ from actual predictions')
    record = {'complete': True, 'source_sha256': file_sha(__file__),
              'report_sha256': file_sha(a.study/'report.json'), 'selection_sha256': report['selection_sha256'],
              'inputs': {str(path): file_sha(path) for sources in paths.values() for path in sources},
              'metrics': metrics, 'decision': report['decision'],
              'baseline': 'response head26, not the v9 binary classifier',
              'limitation': 'Exposed synthetic development diagnostics; context variants share drafts and are not independent samples.',
              'independent_validation': False, 'stable_release_ready': False}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({name: {k: v for k, v in m.items() if k != 'changed_cases'} for name, m in metrics.items()}, indent=2))


if __name__ == '__main__':
    main()
