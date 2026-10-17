"""Compare saved decisions on identical inputs, rejecting denominator drift."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from roguard.evidence import disposition
from training.review_data_v8 import input_key
from training.check_review_candidate_v8 import check_counts


def read(candidate, split, language):
    candidate = Path(candidate)
    dataset = candidate / (split + '.jsonl.gz')
    predictions = candidate / (split + '-predictions.jsonl')
    rows = [json.loads(line) for line in gzip.decompress(dataset.read_bytes()).decode().splitlines()]
    records = [json.loads(line) for line in predictions.read_text().splitlines()]
    ids = [r['case_id'] for r in rows]
    pred_ids = [r['case_id'] for r in records]
    if len(set(ids)) != len(ids) or len(set(pred_ids)) != len(pred_ids) or set(ids) != set(pred_ids):
        raise ValueError('Prediction coverage must match unique dataset IDs exactly')
    lookup = {r['case_id']: r for r in records}
    result = {}
    for row in rows:
        p = lookup[row['case_id']]
        gold = int(disposition({k: {'label': v} for k, v in row['attributes'].items()})[0] != 'CONTINUE')
        if p['language'] != row['language'] or type(p['gold']) is not int or p['gold'] != gold:
            raise ValueError('Prediction language or gold differs from dataset')
        if type(p['predicted']) is not int or p['predicted'] not in (0, 1):
            raise ValueError('Expected binary decisions')
        if row['language'] != language:
            continue
        key = input_key(row)
        if key in result:
            raise ValueError('Duplicate normalized input')
        group = language + ('_conversational' if row.get('source') == 'v7_training_frame_projection' else '_formal')
        result[key] = (gold, p['predicted'], group)
    return result, {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (dataset, predictions)}


def compare(baseline, candidate, language):
    report = {'language': language, 'synthetic_regression_only': True, 'stable_release_ready': False, 'splits': {}}
    for split in ('dev', 'cal'):
        before, bh = read(baseline, split, language)
        after, ah = read(candidate, split, language)
        if not before or set(before) != set(after):
            raise ValueError('Cannot compare different evaluation populations')
        groups = {}
        for key, (gold, pred, group) in after.items():
            old_gold, old_pred, old_group = before[key]
            if gold != old_gold or group != old_group:
                raise ValueError('Labels or groups changed between candidates')
            g = groups.setdefault(group, {'positive': 0, 'negative': 0, 'missed': 0, 'false_reviews': 0,
                                          'baseline_missed': 0, 'baseline_false_reviews': 0, 'fixed': 0, 'regressed': 0})
            g['positive' if gold else 'negative'] += 1
            g['missed'] += int(gold == 1 and pred == 0)
            g['false_reviews'] += int(gold == 0 and pred == 1)
            g['baseline_missed'] += int(gold == 1 and old_pred == 0)
            g['baseline_false_reviews'] += int(gold == 0 and old_pred == 1)
            g['fixed'] += int(old_pred != gold and pred == gold)
            g['regressed'] += int(old_pred == gold and pred != gold)
        report['splits'][split] = {'groups': groups, 'target_failures': check_counts(groups, (language+'_formal', language+'_conversational')),
                                   'baseline_sha256': bh, 'candidate_sha256': ah}
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--language', choices=('ro', 'uk'), required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report = compare(a.baseline, a.candidate, a.language)
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
