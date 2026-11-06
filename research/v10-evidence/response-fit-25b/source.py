"""Fit only registered response training examples; freeze before old diagnostics."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import time

from roguard.trajectory_data import file_sha, read_jsonl, write_jsonl
from roguard.trajectory_model import TrajectoryModel, RESPONSE_LABELS
from training.response_readout_v10 import response_features, predict
from training.train_trajectory_v10 import matrix, save_model


def summarize(rows, outputs):
    import numpy as np
    groups = defaultdict(list)
    for row, prediction in zip(rows, outputs):
        groups[row['language']+':'+row['gold']].append(prediction['label'] != row['gold'])
    if len(rows) != len(outputs) or set(groups) != {l+':'+c for l in ('ro', 'uk') for c in RESPONSE_LABELS}:
        raise ValueError('Incomplete bilingual category evaluation')
    errors = sum(sum(v) for v in groups.values())
    ce = float(np.mean([-np.log(max(p.get('probabilities', p.get('scores'))[r['gold']], 1e-30)) for r, p in zip(rows, outputs)]))
    maximum = max(sum(v)/len(v) for v in groups.values())
    return {'rows': len(rows), 'unique_drafts': len({(r['language'], r['draft']) for r in rows}),
        'correct': len(rows)-errors, 'errors': errors, 'cross_entropy': ce,
        'groups': {k: {'errors': sum(v), 'total': len(v)} for k, v in groups.items()},
        'selection_rank': [maximum, errors, ce]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'baseline', 'study'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.study.exists():
        raise ValueError('Refusing to overwrite response fitting')
    source = json.loads((a.data/'report.json').read_text())
    if not source['complete'] or source['amendment_sha256'] != file_sha('research/v10-evidence/protocol-amendment-23.json'):
        raise ValueError('Response curriculum differs from registration')
    for name, expected in source['artifacts'].items():
        if file_sha(a.data/(name+'.jsonl')) != expected:
            raise ValueError('Response data binding differs')
    train = read_jsonl(a.data/'train.jsonl'); dev = read_jsonl(a.data/'dev.jsonl')
    if any(r['partition'] != 'train' for r in train) or any(r['partition'] != 'dev' for r in dev):
        raise ValueError('Only registered training and development partitions may be fitted/selected')
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    a.study.mkdir(parents=True); started = time.monotonic()
    protocol = {'format': 'roguard-response-readout-fit-1', 'source_sha256': file_sha(Path(__file__)),
        'readout_source_sha256': file_sha('training/response_readout_v10.py'),
        'data_report_sha256': file_sha(a.data/'report.json'), 'baseline_sha256': file_sha(a.baseline),
        'amendment_sha256': file_sha('research/v10-evidence/protocol-amendment-23.json'),
        'old_diagnostic_used_for_fitting_or_selection': False, 'independent_test': False, 'stable_release_ready': False}
    (a.study/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (a.study/'source.py').write_bytes(Path(__file__).read_bytes())
    (a.study/'readout-source.py').write_bytes(Path('training/response_readout_v10.py').read_bytes())
    candidates = []; selected = None
    for representation in ('draft-only', 'separate-context-draft'):
        x = matrix([response_features(r['draft'], r['language'], r['context'], representation) for r in train])
        d = matrix([response_features(r['draft'], r['language'], r['context'], representation) for r in dev])
        for c in (1., 8.):
            model = LogisticRegression(C=c, max_iter=800, solver='lbfgs', random_state=1025)
            model.fit(x, [r['gold'] for r in train])
            if np.any(model.n_iter_ >= 800):
                raise ValueError('Response training did not converge')
            spec = {'format': 'roguard-response-readout-research-1', 'representation': representation,
                'head': {'classes': model.classes_.tolist(), 'weights': model.coef_.tolist(), 'bias': model.intercept_.tolist()},
                'data_report_sha256': protocol['data_report_sha256'], 'regularization_C': c,
                'qualified': False, 'independent_validation': False, 'synthetic_only': True}
            name = representation+'-C'+str(int(c)); path = a.study/(name+'.json.gz')
            save_model(path, spec)
            # Check serialized, standard-library inference rather than the live estimator.
            restored = json.loads(gzip.decompress(path.read_bytes()))
            outputs = [predict(restored, r) for r in dev]
            actual = np.array([[v['probabilities'][label] for label in model.classes_] for v in outputs])
            expected = model.predict_proba(d)
            error = float(np.max(np.abs(actual-expected)))
            if error > 1e-12 or any(p['label'] != label for p, label in zip(outputs, model.predict(d))):
                raise ValueError('Exported response inference differs from fitted estimator')
            record = {'name': name, 'artifact_sha256': file_sha(path), 'maximum_export_probability_error': error,
                      'development': summarize(dev, outputs)}
            candidates.append(record)
            write_jsonl(a.study/(name+'-dev-predictions.jsonl'), [{**r, 'prediction': p} for r, p in zip(dev, outputs)])
            if selected is None or record['development']['selection_rank'] < selected['development']['selection_rank']:
                selected = record; (a.study/'selected.json.gz').write_bytes(path.read_bytes())
    selection = {'complete': True, 'selected': selected, 'candidates': candidates,
                 'diagnostic_opened': False, 'stable_release_ready': False}
    selection_path = a.study/'selection.json'; selection_path.write_text(json.dumps(selection, indent=2)+'\n')
    frozen_selection_sha = file_sha(selection_path)
    # Selection is now immutable before opening the original exposed diagnostic.
    diagnostic = read_jsonl(a.data/'old-diagnostic.jsonl')
    spec = json.loads(gzip.decompress((a.study/'selected.json.gz').read_bytes()))
    baseline = TrajectoryModel(a.baseline); comparisons = {}
    for name, rows in (('development', dev), ('old-diagnostic', diagnostic)):
        output = [predict(spec, r) for r in rows]
        original = [baseline.review_response(r['draft'], r['language'], r['context']) for r in rows]
        comparisons[name] = {'candidate': summarize(rows, output), 'baseline': summarize(rows, original),
            'fixed': sum(b['label'] != r['gold'] and n['label'] == r['gold'] for r, b, n in zip(rows, original, output)),
            'introduced': sum(b['label'] == r['gold'] and n['label'] != r['gold'] for r, b, n in zip(rows, original, output))}
        write_jsonl(a.study/(name+'-selected-predictions.jsonl'), [{**r, 'prediction': p, 'baseline': b} for r, p, b in zip(rows, output, original)])
    if file_sha(selection_path) != frozen_selection_sha:
        raise ValueError('Selection changed after diagnostic access')
    report = {'complete': True, 'selection_sha256': frozen_selection_sha,
        'selected_artifact_sha256': file_sha(a.study/'selected.json.gz'), 'comparisons': comparisons,
        'decision': 'REJECT' if any(v['candidate']['errors'] for v in comparisons.values()) else 'ENGINEERING_PASS_ONLY',
        'seconds': time.monotonic()-started, 'independent_test': False, 'stable_release_ready': False,
        'limitations': ['Developer-authored synthetic wording', 'Single-category examples',
                        'No independent native review', 'No calibrated response-safety or factual-support guarantee']}
    (a.study/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
