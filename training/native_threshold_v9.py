"""Predeclared development-only threshold margin for fixed native checkpoints."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path

from training.check_review_candidate_v8 import audit_predictions, check_counts
from training.threshold_margin_v9 import read
from training.train_native_review_v9 import counts


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def margin_threshold(rows, truth, scores, language):
    boundaries = []
    for conversational in (False, True):
        negatives = sorted((scores[i] for i, row in enumerate(rows)
            if row['language'] == language and truth[i] == 0
            and (row.get('source') == 'v7_training_frame_projection') == conversational), reverse=True)
        if not negatives or any(not math.isfinite(v) or not 0 <= v <= 1 for v in negatives):
            raise ValueError('Missing or invalid development negatives')
        boundary = negatives[len(negatives)//100]+.0001
        if boundary > 1:
            raise ValueError('No guarded threshold can satisfy the 1% budget')
        boundaries.append(boundary)
    return max(boundaries)


def select(rows, truth, scores, original, language):
    groups = tuple(language+'_'+kind for kind in ('formal', 'conversational'))
    options = [('original', dict(original))]
    unavailable = None
    try:
        options.append(('margin_1pct', dict(original, **{language: margin_threshold(rows, truth, scores, language)})))
    except ValueError as error:
        unavailable = str(error)
    variants, eligible = [], []
    for index, (name, thresholds) in enumerate(options):
        metrics = counts(rows, truth, scores, thresholds)
        failures = check_counts(metrics, groups)
        active = [metrics[g] for g in groups]
        rank = None if failures else (-min(v['recall'] for v in active), max(v['false_review_rate'] for v in active),
                                     sum(v['false_review_rate'] for v in active), index)
        variants.append(dict(name=name, thresholds=thresholds, development=metrics, failures=failures, rank=rank))
        if rank is not None:
            eligible.append((rank, index))
    selected = variants[min(eligible)[1] if eligible else 0]
    return dict(selected=selected, variants=variants, margin_unavailable=unavailable,
                any_development_eligible=bool(eligible))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('prepare', 'audit'))
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--language', choices=('ro', 'uk'), required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    artifact = json.loads(gzip.decompress((a.candidate/'review.json.gz').read_bytes()))
    if artifact.get('format') != 'roguard-native-full-review-experiment-v1' or artifact['language'] != a.language:
        raise ValueError('Expected the selected native-language checkpoint')
    sources = {name: sha(a.candidate/name) for name in ('review.json.gz', 'dev.jsonl.gz',
                                                      'dev-predictions.jsonl', 'training-history.jsonl.gz')}
    protocol_hash = sha(Path('research/v9-stabilization/native-threshold-34/protocol.json'))
    if a.stage == 'prepare':
        rows, truth, scores = read(a.candidate, 'dev', artifact['thresholds'])
        selected = select(rows, truth, scores, artifact['thresholds'], a.language)
        record = dict(created_utc=datetime.now(timezone.utc).isoformat(), parent=str(a.candidate), language=a.language,
            source_hashes=sources, source_sha256=sha(Path(__file__)), protocol_sha256=protocol_hash, selection_partition='development',
            calibration_predictions_opened=False, reserved_probes_used=False, stable_release_ready=False, **selected)
        a.output.mkdir(exist_ok=False)
        (a.output/'selection.json').write_text(json.dumps(record, indent=2)+'\n')
        artifact['thresholds'] = record['selected']['thresholds']
        (a.output/'review.json.gz').write_bytes(gzip.compress(json.dumps(artifact, separators=(',', ':')).encode(), mtime=0))
        records = [dict(case_id=r['case_id'], language=r['language'], gold=y, score=s,
                        predicted=int(s >= artifact['thresholds'][r['language']])) for r, y, s in zip(rows, truth, scores)]
        (a.output/'dev-predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        print(json.dumps(record, indent=2))
    else:
        selection = a.output/'selection.json'
        record = json.loads(selection.read_text())
        if (record['parent'] != str(a.candidate) or record['language'] != a.language or record['source_hashes'] != sources
                or record['source_sha256'] != sha(Path(__file__)) or record['protocol_sha256'] != protocol_hash):
            raise ValueError('Frozen threshold inputs changed')
        thresholds = record['selected']['thresholds']
        exported = json.loads(gzip.decompress((a.output/'review.json.gz').read_bytes()))
        if exported != dict(artifact, thresholds=thresholds):
            raise ValueError('Exported threshold artifact differs')
        rows, truth, scores = read(a.candidate, 'cal', artifact['thresholds'])
        metrics = counts(rows, truth, scores, thresholds)
        predictions = [dict(case_id=r['case_id'], language=r['language'], gold=y, score=s,
                            predicted=int(s >= thresholds[r['language']])) for r, y, s in zip(rows, truth, scores)]
        destination = a.output/'cal-predictions.jsonl'
        with destination.open('x') as stream:
            stream.write(''.join(json.dumps(r)+'\n' for r in predictions))
        failures = ['development/'+v for v in record['selected']['failures']]
        failures += ['calibration/'+v for v in check_counts(metrics, tuple(a.language+'_'+kind for kind in ('formal', 'conversational')))]
        failures += audit_predictions(destination, rows, metrics, thresholds)
        result = dict(selection_sha256=sha(selection), artifact_sha256=sha(a.output/'review.json.gz'),
            checkpoint_sha256=artifact['checkpoint_sha256'], calibration=metrics, failures=failures,
            calibration_source_hashes={name: sha(a.candidate/name) for name in ('cal.jsonl.gz', 'cal-predictions.jsonl')},
            predictions_sha256=sha(destination),
            threshold_counts_pass=not failures, reserved_probes_used=False, stable_release_ready=False,
            synthetic_regression_only=True, additional_requirements=['installed inference replay', 'fresh actual-runtime evaluation'])
        (a.output/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
