"""Retrospective threshold feasibility; never selects a deployment threshold."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path


def feasibility(groups):
    ceilings = []
    for rows in groups.values():
        if any(r['gold'] not in (0, 1) or not math.isfinite(r['score']) or not 0 <= r['score'] <= 1 for r in rows):
            raise ValueError('Invalid label or score')
        positive = sorted(r['score'] for r in rows if r['gold'])
        if not positive or len(positive) == len(rows):
            raise ValueError('Both classes are required in every group')
        # Prediction is score >= threshold. At most floor(P/100) misses.
        ceilings.append(positive[len(positive) // 100])
    if not ceilings:
        raise ValueError('No groups')
    threshold = min(ceilings)
    metrics = {}
    for group, rows in groups.items():
        p = sum(r['gold'] for r in rows)
        metrics[group] = dict(positive=p, negative=len(rows)-p,
            missed=sum(r['gold'] == 1 and r['score'] < threshold for r in rows),
            false_reviews=sum(r['gold'] == 0 and r['score'] >= threshold for r in rows))
    return {'retrospective_only': True, 'deployment_threshold_selected': False,
            'largest_threshold_preserving_99_percent_recall_in_every_group': threshold,
            'metrics': metrics,
            'single_threshold_can_meet_targets_on_inspected_rows': all(m['false_reviews']*100 <= 5*m['negative'] for m in metrics.values())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--language', choices=('ro', 'uk'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    groups, hashes = {}, {}
    for split in ('dev', 'cal'):
        path = args.candidate/(split+'.jsonl.gz')
        raw = path.read_bytes()
        hashes[path.name] = hashlib.sha256(raw).hexdigest()
        inputs = {r['case_id']: r for r in map(json.loads, gzip.decompress(raw).decode().splitlines())}
        path = args.candidate/(split+'-predictions.jsonl')
        raw = path.read_bytes()
        hashes[path.name] = hashlib.sha256(raw).hexdigest()
        seen = set()
        for row in map(json.loads, raw.decode().splitlines()):
            key = row['case_id']
            if key in seen or key not in inputs:
                raise ValueError('Duplicate or unknown prediction')
            seen.add(key)
            if row['language'] != inputs[key]['language']:
                raise ValueError('Prediction language mismatch')
            if row['language'] != args.language:
                continue
            kind = 'conversational' if inputs[key].get('source') == 'v7_training_frame_projection' else 'formal'
            groups.setdefault(split+'_'+kind, []).append(row)
        if seen != set(inputs):
            raise ValueError('Incomplete predictions')
    result = feasibility(groups)
    result.update(candidate=str(args.candidate), language=args.language, input_sha256=hashes,
                  limitation='Inspected synthetic regression data; not fresh evaluation or real-world accuracy.')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
