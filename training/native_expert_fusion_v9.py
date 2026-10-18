"""Grouped development fusion of direct-native and semantic-native experts.

The six field vectors are native marginals, not conditional decoder outputs.
No text, case identifier or source-group identifier is a predictive feature.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from roguard.evidence import ONTOLOGY, disposition
from training.review_fusion_v9 import (REGULARIZATION, sha, read, features, folds,
    strata, fit, probability, metrics)
from training.recall_gap_threshold_v9 import recall_gap_threshold

FIELD_STUDIES = {'ro': 'semantic-selection-71', 'uk': 'scope-repair-66'}


def load_split(root, language, split):
    candidate = root / ('native-full-22-' + language)
    fields = root / FIELD_STUDIES[language] / language
    data = root / FIELD_STUDIES[language] / 'encoders' / language
    paths = dict(rows=candidate / (split + '.jsonl.gz'),
                 native=candidate / (split + '-predictions.jsonl'),
                 fields=fields / (split + '-predictions.jsonl'))
    if sha(paths['rows']) != sha(data / (split + '.jsonl.gz')):
        raise ValueError('Experts use different evaluation inputs')
    rows, native, semantic = [read(paths[key]) for key in ('rows', 'native', 'fields')]
    identities = [r['case_id'] for r in rows]
    if (len(set(identities)) != len(rows) or identities != [r['case_id'] for r in native]
            or identities != [r['case_id'] for r in semantic]):
        raise ValueError('Native expert coverage differs')
    x, y = [], []
    for row, direct, field in zip(rows, native, semantic):
        gold = int(disposition({k: {'label': v} for k, v in row['attributes'].items()})[0] != 'CONTINUE')
        if (direct['gold'] != gold or field['gold'] != gold or row['language'] != language
                or direct['language'] != language or field['language'] != language):
            raise ValueError('Native expert labels or languages differ')
        p = field['field_probabilities']
        labels = {f: values[max(range(len(values)), key=p[f].__getitem__)] for f, values in ONTOLOGY.items()}
        # Reuse the same arithmetic feature construction, while explicitly
        # retaining native-marginal semantics in this separate experiment.
        vector = features(direct, dict(valid_label_tokens=True,
            conditional_field_probabilities=p, interpretation=labels))
        x.append(vector); y.append(gold)
    return rows, x, y, {key: sha(path) for key, path in paths.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'audit'))
    parser.add_argument('--language', choices=('ro', 'uk'), required=True)
    args = parser.parse_args()
    root = Path('research/v9-stabilization'); output = root/'native-expert-fusion-75'/args.language
    protocol = root/'native-expert-fusion-75/protocol.json'
    parent_paths = [root/('native-full-22-'+args.language)/'review.json.gz', root/FIELD_STUDIES[args.language]/args.language/'selection.json']
    parent_hashes = {str(path.relative_to(root)): sha(path) for path in parent_paths}
    if args.stage == 'prepare':
        rows, x, y, hashes = load_split(root, args.language, 'dev')
        assignments = folds(rows)
        if set(assignments) != set(range(5)):
            raise ValueError('Connected development groups do not cover all five folds')
        kinds = [strata(r) for r in rows]; trials = []
        for regularization in REGULARIZATION:
            scores = [None]*len(rows)
            for fold in range(5):
                train = [i for i in range(len(rows)) if assignments[i] != fold and x[i] is not None]
                model = fit([x[i] for i in train], [y[i] for i in train], [kinds[i] for i in train], regularization)
                for i in range(len(rows)):
                    if assignments[i] == fold:
                        scores[i] = probability(model, x[i])
            limit = recall_gap_threshold(rows, y, scores); counts = metrics(rows, y, scores, limit)
            # Unchanged gate eligibility precedes false-review ranking.
            rank = (not all(v['passes'] for v in counts.values()), max(v['false_review_rate'] for v in counts.values()),
                    sum(v['false_review_rate'] for v in counts.values()), -regularization)
            trials.append(dict(regularization=regularization, threshold=limit, metrics=counts, rank=rank, scores=scores))
        selected = min(trials, key=lambda v: v['rank'])
        valid = [i for i in range(len(rows)) if x[i] is not None]
        fitted = fit([x[i] for i in valid], [y[i] for i in valid], [kinds[i] for i in valid], selected['regularization'])
        output.mkdir(parents=True, exist_ok=False)
        record = dict(created_utc=datetime.now(timezone.utc).isoformat(), language=args.language,
            selection_population='Grouped out-of-fold development predictions; base models already used development for selection.',
            calibration_opened=False, reserved_probes_used=False, stable_release_ready=False,
            source_sha256=sha(__file__), protocol_sha256=sha(protocol), development_hashes=hashes, parent_hashes=parent_hashes,
            fold_counts=dict(Counter(assignments)), selected={k: v for k, v in selected.items() if k != 'scores'},
            trials=[{k: v for k, v in trial.items() if k != 'scores'} for trial in trials], model=fitted)
        (output/'selection.json').write_text(json.dumps(record, indent=2)+'\n')
        (output/'oof-predictions.jsonl').write_text(''.join(json.dumps(dict(case_id=r['case_id'], gold=t, score=s,
            fold=f, predicted=None if s is None else int(s >= selected['threshold'])))+'\n' for r,t,s,f in zip(rows,y,selected['scores'],assignments)))
        print(json.dumps(record['selected'], indent=2))
    else:
        selection = output/'selection.json'; record = json.loads(selection.read_text())
        if (record['source_sha256'] != sha(__file__) or record['protocol_sha256'] != sha(protocol)
                or record['parent_hashes'] != parent_hashes):
            raise ValueError('Frozen fusion recipe changed')
        _, _, _, hashes = load_split(root, args.language, 'dev')
        if hashes != record['development_hashes']:
            raise ValueError('Development evidence changed')
        rows, x, y, hashes = load_split(root, args.language, 'cal')
        scores = [probability(record['model'], value) for value in x]
        counts = metrics(rows, y, scores, record['selected']['threshold'])
        result = dict(selection_sha256=sha(selection), calibration_hashes=hashes, calibration=counts,
                      counts_pass=all(v['passes'] for v in counts.values()) and all(v['passes'] for v in record['selected']['metrics'].values()),
                      stable_release_ready=False, reserved_probes_used=False,
                      requirements_remaining=['Installed composite runtime parity', 'Frozen reserved actual-runtime probes', 'Product integration and release checks'])
        with (output/'cal-predictions.jsonl').open('x') as stream:
            stream.write(''.join(json.dumps(dict(case_id=r['case_id'], gold=t, score=s,
                predicted=None if s is None else int(s >= record['selected']['threshold'])))+'\n' for r,t,s in zip(rows,y,scores)))
        (output/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
