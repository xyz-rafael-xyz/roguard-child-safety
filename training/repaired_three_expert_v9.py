"""Combine a repaired native model with the two previously frozen experts."""
import argparse
from collections import Counter
import json
from pathlib import Path

from roguard.evidence import ONTOLOGY
from training.review_fusion_v9 import read, sha, folds, features, metrics, strata
from training.three_expert_fusion_v9 import combine_vectors
from training.boosted_fusion_v9 import CONFIGS, fit, probability
from training.conservative_fusion_v9 import full_coverage_gap
from training.prepare_instruction_sft_v8 import target

ROOT = Path('research/v9-stabilization')
STUDY = ROOT / 'repaired-three-expert-97'
PARENTS = {85: 'needs-repair-85', 95: 'bare-distress-95'}


def combine(native, field, interpreted):
    probabilities = field['field_probabilities']
    labels = {key: values[max(range(len(values)), key=probabilities[key].__getitem__)]
              for key, values in ONTOLOGY.items()}
    left = features(native, dict(valid_label_tokens=True,
        conditional_field_probabilities=probabilities, interpretation=labels))
    right = features(native, interpreted)
    return combine_vectors(left, right)


def load(parent, split):
    native = ROOT / PARENTS[parent] / 'uk'
    paths = dict(rows=native / (split + '.jsonl.gz'),
        native=native / (split + '-predictions.jsonl'),
        fields=ROOT / 'scope-repair-66/uk' / (split + '-predictions.jsonl'),
        interpreter=ROOT / 'interpreter-full-21' / split / 'predictions.jsonl')
    rows = read(paths['rows'])
    original = read(ROOT / 'native-full-22-uk' / (split + '.jsonl.gz'))
    if rows != original:
        raise ValueError('Repaired-model evaluation inputs differ from original inputs')
    maps = {}
    for name in ('native', 'fields', 'interpreter'):
        records = read(paths[name])
        maps[name] = {r['case_id']: r for r in records}
        if len(maps[name]) != len(records):
            raise ValueError('Duplicate component prediction')
    ids = [r['case_id'] for r in rows]
    if len(set(ids)) != len(ids) or any(set(maps[k]) != set(ids) for k in ('native', 'fields')):
        raise ValueError('Incomplete native/field coverage')
    x, y = [], []
    for row in rows:
        parts = [maps[name][row['case_id']] for name in ('native', 'fields', 'interpreter')]
        gold = target(row)
        if row['language'] != 'uk' or any(p['language'] != 'uk' or p['gold'] != gold for p in parts):
            raise ValueError('Component label or language differs')
        x.append(combine(*parts)); y.append(gold)
    return rows, x, y, {key: sha(path) for key, path in paths.items()}


def cutoff(truth, scores):
    valid = [(y, s) for y, s in zip(truth, scores) if s is not None]
    return full_coverage_gap([y for y, _ in valid], [s for _, s in valid])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('fit', 'audit'))
    parser.add_argument('--parent', type=int, choices=PARENTS, required=True)
    args = parser.parse_args()
    for name, expected in json.loads((STUDY / 'source-freeze.json').read_text()).items():
        if sha(name) != expected:
            raise ValueError('Frozen experiment source differs: ' + name)
    parent_paths = [ROOT / PARENTS[args.parent] / 'uk/review.json.gz',
        ROOT / 'scope-repair-66/uk/selection.json', ROOT / 'interpreter-full-21/selection.json']
    parent_hashes = {str(p.relative_to(ROOT)): sha(p) for p in parent_paths}
    output = STUDY / str(args.parent)
    if args.stage == 'fit':
        rows, x, y, hashes = load(args.parent, 'dev')
        assignments = folds(rows)
        if set(assignments) != set(range(5)):
            raise ValueError('Missing connected development fold')
        kinds = [strata(r) for r in rows]
        trials = []
        for regularization in CONFIGS:
            scores = [None] * len(rows)
            for fold in range(5):
                indices = [i for i in range(len(rows)) if assignments[i] != fold and x[i] is not None]
                model = fit([x[i] for i in indices], [y[i] for i in indices],
                            [kinds[i] for i in indices], regularization)
                for i in range(len(rows)):
                    if assignments[i] == fold:
                        scores[i] = probability(model, x[i])
            threshold = cutoff(y, scores)
            counts = metrics(rows, y, scores, threshold)
            rank = (not all(v['passes'] for v in counts.values()),
                    max(v['false_review_rate'] for v in counts.values()),
                    sum(v['false_review_rate'] for v in counts.values()),
                    CONFIGS[regularization][0], -CONFIGS[regularization][1])
            trials.append(dict(regularization=regularization, threshold=threshold,
                               metrics=counts, rank=rank, scores=scores))
        selected = min(trials, key=lambda t: t['rank'])
        valid = [i for i in range(len(rows)) if x[i] is not None]
        model = fit([x[i] for i in valid], [y[i] for i in valid],
                    [kinds[i] for i in valid], selected['regularization'])
        fitted = [probability(model, v) for v in x]
        threshold = cutoff(y + y, selected['scores'] + fitted)
        record = dict(language='uk', parent=args.parent, parent_hashes=parent_hashes,
            model=model, threshold=threshold, development_hashes=hashes,
            development=metrics(rows, y, fitted, threshold),
            oof=metrics(rows, y, selected['scores'], threshold),
            selected_regularization=selected['regularization'], fold_counts=dict(Counter(assignments)),
            trials=[{k: v for k, v in t.items() if k != 'scores'} for t in trials],
            selection='Connected development OOF selection; final cutoff covers valid OOF/refit concerns',
            calibration_used_for_selection=False, calibration_diagnosis_informed_experiment=True,
            synthetic_only=True, independent_accuracy_evaluation=False,
            reserved_probes_used=False, stable_release_ready=False)
        output.mkdir(exist_ok=False)
        (output / 'selection.json').write_text(json.dumps(record, indent=2) + '\n')
        for name, scores in [('dev', fitted), ('oof', selected['scores'])]:
            (output / (name + '-predictions.jsonl')).write_text(''.join(
                json.dumps(dict(case_id=r['case_id'], language='uk', gold=t, score=s,
                    predicted=None if s is None else int(s >= threshold))) + '\n'
                for r, t, s in zip(rows, y, scores)))
        print(json.dumps({k: record[k] for k in ('threshold', 'development', 'oof')}), flush=True)
    else:
        selection = output / 'selection.json'
        record = json.loads(selection.read_text())
        _, _, _, hashes = load(args.parent, 'dev')
        if record['development_hashes'] != hashes or record['parent_hashes'] != parent_hashes:
            raise ValueError('Frozen candidate evidence differs')
        rows, x, y, hashes = load(args.parent, 'cal')
        scores = [probability(record['model'], v) for v in x]
        counts = metrics(rows, y, scores, record['threshold'])
        report = dict(selection_sha256=sha(selection), calibration_hashes=hashes,
            calibration=counts,
            counts_pass=all(c['passes'] for group in (record['development'], record['oof'], counts) for c in group.values()),
            retained_stress_not_yet_checked=True, reserved_probes_used=False, stable_release_ready=False,
            remaining=['Actual retained-stress inference', 'Installed composite parity',
                       'Frozen reserved evaluation', 'Bilingual integration and release checks'])
        (output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
        (output / 'cal-predictions.jsonl').write_text(''.join(
            json.dumps(dict(case_id=r['case_id'], language='uk', gold=t, score=s,
                predicted=None if s is None else int(s >= record['threshold']))) + '\n'
            for r, t, s in zip(rows, y, scores)))
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
