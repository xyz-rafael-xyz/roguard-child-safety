"""Development-only sensitivity selection under a fixed 3% false-review budget.

Preparation never opens calibration predictions. Audit replays a frozen threshold;
neither stage trains a model or establishes independent generalization evidence.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path

from training.check_review_candidate_v8 import audit_predictions, check_counts
from training.train_native_review_v9 import counts


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(candidate, split, thresholds):
    rows = [json.loads(line) for line in gzip.decompress((candidate/(split+'.jsonl.gz')).read_bytes()).splitlines()]
    predictions = [json.loads(line) for line in (candidate/(split+'-predictions.jsonl')).read_text().splitlines()]
    indexed = {p['case_id']: p for p in predictions}
    if len(indexed) != len(predictions) or len(rows) != len(predictions) or {r['case_id'] for r in rows} != set(indexed):
        raise ValueError('Prediction coverage differs')
    truth = [indexed[r['case_id']]['gold'] for r in rows]
    scores = [indexed[r['case_id']]['score'] for r in rows]
    metrics = counts(rows, truth, scores, thresholds)
    errors = audit_predictions(candidate/(split+'-predictions.jsonl'), rows, metrics, thresholds)
    if errors:
        raise ValueError(str(errors))
    return rows, truth, scores


def select(rows, truth, scores, language):
    """Lowest shared threshold meeting <=3% dev false reviews in each stratum."""
    boundaries = []
    for conversational in (False, True):
        negatives = sorted((scores[i] for i, r in enumerate(rows)
                            if r['language'] == language and truth[i] == 0
                            and (r.get('source') == 'v7_training_frame_projection') == conversational), reverse=True)
        if not negatives:
            raise ValueError('Missing development negatives')
        limit = 3*len(negatives)//100
        boundary = math.nextafter(negatives[limit], math.inf)
        if boundary > 1:
            raise ValueError('No valid threshold can satisfy the budget')
        boundaries.append(boundary)
    return max(boundaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'audit'))
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--language', choices=('ro', 'uk'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    original = json.loads(gzip.decompress((args.candidate/'review.json.gz').read_bytes()))['thresholds']
    groups = tuple(args.language+'_'+kind for kind in ('formal', 'conversational'))
    files = ['review.json.gz', 'encoder.json.gz', 'dev.jsonl.gz', 'dev-predictions.jsonl', 'training-history.jsonl.gz']
    hashes = {name: sha(args.candidate/name) for name in files if (args.candidate/name).exists()}
    if args.stage == 'prepare':
        rows, truth, scores = read(args.candidate, 'dev', original)
        thresholds = dict(original, **{args.language: select(rows, truth, scores, args.language)})
        metrics = counts(rows, truth, scores, thresholds)
        record = dict(created_utc=datetime.now(timezone.utc).isoformat(), synthetic_only=True,
                      candidate=str(args.candidate), language=args.language, source_hashes=hashes,
                      thresholds=thresholds, original_thresholds=original, development=metrics,
                      development_failures=check_counts(metrics, groups),
                      method='Lowest shared threshold allowing at most 3% false reviews separately in each development stratum. The 2 percentage-point headroom is a heuristic, not a statistical guarantee.',
                      model_selection='Parent selected checkpoint remains fixed. No calibration scores used for threshold selection.',
                      reserved_probes_used=False, stable_release_ready=False,
                      source_sha256=sha(Path(__file__)))
        args.output.mkdir(exist_ok=False)
        (args.output/'selection.json').write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps(record, indent=2))
    else:
        selection = args.output/'selection.json'
        record = json.loads(selection.read_text())
        if (record['candidate'] != str(args.candidate) or record['language'] != args.language
                or record['source_hashes'] != hashes or record['source_sha256'] != sha(Path(__file__))):
            raise ValueError('Frozen threshold inputs changed')
        rows, truth, scores = read(args.candidate, 'cal', original)
        metrics = counts(rows, truth, scores, record['thresholds'])
        predictions = [dict(case_id=r['case_id'], language=r['language'], gold=y, score=s,
                            predicted=int(s >= record['thresholds'][r['language']])) for r, y, s in zip(rows, truth, scores)]
        path = args.output/'cal-predictions.jsonl'
        with path.open('x') as out:
            out.write(''.join(json.dumps(p)+'\n' for p in predictions))
        failures = ['development/'+f for f in record['development_failures']]
        failures += ['calibration/'+f for f in check_counts(metrics, groups)]
        failures += audit_predictions(path, rows, metrics, record['thresholds'])
        result = dict(selection_sha256=sha(selection), calibration=metrics, failures=failures,
                      threshold_counts_pass=not failures, stable_release_ready=False,
                      reserved_probes_used=False, synthetic_only=True,
                      limitation='Reused regression partitions; complete candidate provenance, fresh probes and runtime verification still required.')
        (args.output/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
