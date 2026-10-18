"""Select semantic policy aggregation from a fixed jointly trained encoder.

The parent was selected for its direct review head. This separate experiment
tests its auxiliary fields without changing the encoder or using calibration
labels to choose an aggregation rule.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

from roguard.assurance import policy_digest
from roguard.evidence import disposition
from training.native_fields_v9 import FIELDS, fit_couplings, policy_mass
from training.review_fusion_v9 import metrics, strata, threshold
from training.threshold_margin_fields_v9 import margin_threshold


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    data = Path(path).read_bytes()
    if str(path).endswith('.gz'):
        data = gzip.decompress(data)
    return [json.loads(line) for line in data.splitlines()]


def choose(rows, truth, probabilities, couplings):
    trials = []
    for mode, factors in (('independent', None), ('pair-coupled', couplings)):
        scores = policy_mass(probabilities, factors)
        cutoff = margin_threshold(scores, threshold(rows, truth, scores))
        observed = metrics(rows, truth, scores, cutoff)
        rank = [int(not all(v['passes'] for v in observed.values())),
                max(v['false_review_rate'] for v in observed.values()),
                sum(v['false_review_rate'] for v in observed.values()),
                int(factors is not None)]
        trials.append(dict(mode=mode, threshold=cutoff, metrics=observed, rank=rank))
    return min(trials, key=lambda item: item['rank']), trials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('fit', 'audit'))
    parser.add_argument('--language', choices=('ro', 'uk'), required=True)
    args = parser.parse_args()
    import numpy as np
    root = Path('research/v9-stabilization')
    experiment = root / 'multitask-policy-65'
    for name, expected in json.loads((experiment / 'source-freeze.json').read_text()).items():
        if sha(name) != expected:
            raise ValueError('Frozen semantic selection source changed: ' + name)
    parent = root / 'multitask-review-61' / args.language
    study = experiment / args.language
    parent_selection = json.loads((parent / 'selection.json').read_text())
    heads = parent / 'auxiliary-heads.json.gz'
    if (sha(heads) != parent_selection['auxiliary_heads_sha256']
            or sha(parent / 'review.json.gz') != parent_selection['artifact_sha256']):
        raise ValueError('Jointly trained parent identity changed')
    spec = json.loads(gzip.decompress(heads.read_bytes()))
    if spec['native_artifact_sha256'] != sha(parent / 'review.json.gz'):
        raise ValueError('Auxiliary heads belong to a different encoder')
    split = 'dev' if args.stage == 'fit' else 'cal'
    rows = read(parent / (split + '.jsonl.gz'))
    predictions = read(parent / (split + '-predictions.jsonl'))
    truth = np.array([int(disposition({f: {'label': v} for f, v in r['attributes'].items()})[0]
                          != 'CONTINUE') for r in rows])
    if ([r['case_id'] for r in rows] != [r['case_id'] for r in predictions]
            or truth.tolist() != [r['gold'] for r in predictions]):
        raise ValueError('Parent prediction coverage or labels differ')
    probabilities = [np.array([r['field_probabilities'][f] for r in predictions]) for f in FIELDS]
    if args.stage == 'fit':
        study.mkdir(parents=True, exist_ok=False)
        training = read(parent / 'train.jsonl.gz')
        groups = [('augmented' if r['source'] == 'joint_semantic_paraphrases_v9' else strata(r),
                   int(disposition({f: {'label': v} for f, v in r['attributes'].items()})[0]
                       != 'CONTINUE')) for r in training]
        sizes = Counter(groups)
        if len(sizes) != 6:
            raise ValueError('Expected six training source/class groups')
        couplings = fit_couplings(training, [len(groups) / (6 * sizes[g]) for g in groups])
        best, trials = choose(rows, truth, probabilities, couplings)
        factors = None if best['mode'] == 'independent' else couplings
        spec.update(threshold=best['threshold'], couplings=factors, policy_digest=policy_digest(),
                    distribution=best['mode'] + ' categorical fields; uncalibrated')
        artifact = study / 'selected-heads.json.gz'
        artifact.write_bytes(gzip.compress(json.dumps(spec).encode(), mtime=0))
        selection = dict(created_utc=datetime.now(timezone.utc).isoformat(), selected=best, trials=trials,
            artifact_sha256=sha(artifact), parent_selection_sha256=sha(parent / 'selection.json'),
            parent_heads_sha256=sha(heads), parent_development_predictions_sha256=sha(parent / 'dev-predictions.jsonl'),
            data_hashes={s: sha(parent / (s + '.jsonl.gz')) for s in ('train', 'dev')},
            calibration_used_for_selection=False, reserved_probes_used=False, stable_release_ready=False)
        (study / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    else:
        selection = json.loads((study / 'selection.json').read_text())
        if (selection['artifact_sha256'] != sha(study / 'selected-heads.json.gz')
                or selection['parent_selection_sha256'] != sha(parent / 'selection.json')
                or selection['parent_heads_sha256'] != sha(heads)):
            raise ValueError('Selected semantic policy identity changed')
        spec = json.loads(gzip.decompress((study / 'selected-heads.json.gz').read_bytes()))
    scores = policy_mass(probabilities, spec['couplings'])
    observed = metrics(rows, truth, scores, spec['threshold'])
    if args.stage == 'fit' and observed != selection['selected']['metrics']:
        raise ValueError('Exported semantic policy changed decisions')
    with (study / (split + '-predictions.jsonl')).open('x') as stream:
        for row, value, score in zip(rows, predictions, scores):
            record = dict(value, score=float(score), predicted=int(score >= spec['threshold']))
            stream.write(json.dumps(record) + '\n')
    if args.stage == 'audit':
        audit = dict(selection_sha256=sha(study / 'selection.json'), calibration=observed,
            counts_pass=all(v['passes'] for v in observed.values()) and
                        all(v['passes'] for v in selection['selected']['metrics'].values()),
            calibration_data_sha256=sha(parent / 'cal.jsonl.gz'),
            parent_calibration_predictions_sha256=sha(parent / 'cal-predictions.jsonl'),
            synthetic_only=True, reserved_probes_used=False, stable_release_ready=False)
        (study / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(dict(language=args.language, split=split, metrics=observed)), flush=True)


if __name__ == '__main__':
    main()
