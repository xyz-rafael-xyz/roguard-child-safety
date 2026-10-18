"""Grouped development fusion of direct, semantic-native and interpreter experts.

Native field marginals and decoder conditional vectors remain distinct features.
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


def combine_vectors(native_fields, interpreter):
    if interpreter is None:
        return None
    width = sum(len(v) for v in ONTOLOGY.values())
    expected = 2 + 3 * width
    if (not isinstance(native_fields, list) or not isinstance(interpreter, list)
            or len(native_fields) != expected or len(interpreter) != expected
            or native_fields[:2] != interpreter[:2]):
        raise ValueError('Three-expert feature binding differs')
    # Conditional decoder probabilities and unconditional native marginals
    # remain separate features; their products express agreement, not a joint
    # probability or a calibrated confidence estimate.
    agreement = [a*b for a,b in zip(native_fields[2:2+width], interpreter[2:2+width])]
    return native_fields + interpreter[2:] + agreement


def load_split(root, language, split):
    from training.native_expert_fusion_v9 import load_split as native_split
    from training.review_fusion_v9 import load_split as interpreter_split
    rows, native, truth, left = native_split(root, language, split)
    other, interpreted, labels, right = interpreter_split(root, language, split)
    if (left['rows'] != right['rows'] or left['native'] != right['native']
            or [r['case_id'] for r in rows] != [r['case_id'] for r in other]
            or truth != labels or len(native) != len(interpreted)):
        raise ValueError('Three-expert input populations differ')
    x = [combine_vectors(a,b) for a,b in zip(native, interpreted)]
    hashes = dict(left)
    hashes['interpreter'] = right['structured']
    return rows, x, truth, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'audit'))
    parser.add_argument('--language', choices=('ro', 'uk'), required=True)
    args = parser.parse_args()
    root = Path('research/v9-stabilization'); output = root/'three-expert-fusion-76'/args.language
    protocol = root/'three-expert-fusion-76/protocol.json'
    parent_paths = [root/('native-full-22-'+args.language)/'review.json.gz', root/FIELD_STUDIES[args.language]/args.language/'selection.json', root/'interpreter-full-21/selection.json']
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
