"""Converged L2 response readouts on fixed study-32 features and partitions."""
import argparse
import json
from pathlib import Path
import time

from roguard.assurance import digest
from roguard.trajectory_data import file_sha, read_jsonl, write_jsonl
from roguard.trajectory_model import RESPONSE_LABELS
from training.train_contrast_response_v10 import payload
from training.train_neural_response_v10 import prediction_rows
from training.train_response_readout_v10 import summarize


def normalized(x):
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2560 or not np.isfinite(x).all():
        raise ValueError('Invalid response representation')
    return x/np.sqrt(np.maximum(np.mean(x*x, axis=1, keepdims=True), 1e-12))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'original_study', 'original_features', 'contrast_study', 'contrast_features', 'context_study', 'context_features', 'study', 'private'):
        p.add_argument('--'+name.replace('_', '-'), dest=name, type=Path, required=True)
    p.add_argument('--draft-only', action='store_true', help='Registered study36: deduplicated zero-context behavior view')
    a = p.parse_args()
    if a.study.exists() or a.private.exists():
        raise ValueError('Refusing to overwrite regularized response training')
    source = json.loads((a.contrast_study/'report.json').read_text())
    source_protocol = json.loads((a.contrast_study/'protocol.json').read_text())
    selection = json.loads((a.contrast_study/'selection.json').read_text())
    original_selection = json.loads((a.original_study/'selection.json').read_text())
    original_report = json.loads((a.original_study/'report.json').read_text())
    data = json.loads((a.data/'report.json').read_text())
    if (not source['complete'] or not original_report['complete']
            or file_sha(a.original_study/'selection.json') != original_report['selection_sha256']
            or file_sha(a.original_features/'features.npy') != original_selection['features_sha256']
            or file_sha(a.contrast_study/'selection.json') != source['selection_sha256']
            or file_sha(a.contrast_features/'additional-features.npy') != selection['additional_features_sha256']
            or file_sha(a.contrast_study/'additional-train.jsonl') != source_protocol['additional_train_sha256']):
        raise ValueError('Response feature or selection binding differs')
    for name in ('train', 'dev'):
        if file_sha(a.data/(name+'.jsonl')) != data['artifacts'][name]:
            raise ValueError('Original response fitting data differs')
    rows = read_jsonl(a.data/'train.jsonl')+read_jsonl(a.data/'dev.jsonl')+read_jsonl(a.contrast_study/'additional-train.jsonl')
    identities = read_jsonl(a.contrast_study/'feature-inputs.jsonl')
    if len(rows) != 676 or len(identities) != len(rows) or any(
            i != {'case_id': r['case_id'], 'partition': r['partition'], 'input_sha256': digest(payload(r))}
            for r, i in zip(rows, identities)):
        raise ValueError('Response feature order/input identity differs')
    amendment = Path('research/v10-evidence/protocol-amendment-'+('34' if a.draft_only else '33')+'.json'); plan = json.loads(amendment.read_text())
    a.study.mkdir(parents=True); a.private.mkdir(parents=True)
    protocol = {'format': 'roguard-isolated-response-fit-1' if a.draft_only else 'roguard-regularized-response-fit-1', 'source_sha256': file_sha(Path(__file__)),
        'amendment_sha256': file_sha(amendment), 'contrast_report_sha256': file_sha(a.contrast_study/'report.json'),
        'original_report_sha256': file_sha(a.original_study/'report.json'),
        'feature_inputs_sha256': file_sha(a.contrast_study/'feature-inputs.jsonl'),
        'labels': list(RESPONSE_LABELS), 'prompt': source_protocol['prompt'],
        'normalization': 'float64 per-row RMS, epsilon 1e-12', 'diagnostic_used_for_selection': False,
        'draft_only': a.draft_only, 'contextual_factual_support_established': False,
        'independent_validation': False, 'stable_release_ready': False}
    (a.study/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (a.study/'source.py').write_bytes(Path(__file__).read_bytes())
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    started = time.monotonic()
    x = normalized(np.concatenate((np.load(a.original_features/'features.npy', allow_pickle=False),
                                    np.load(a.contrast_features/'additional-features.npy', allow_pickle=False))))
    if len(x) != len(rows):
        raise ValueError('Response feature count differs')
    train = np.array([i for i, r in enumerate(rows) if r['partition'] == 'train'])
    dev = np.array([i for i, r in enumerate(rows) if r['partition'] == 'dev'])
    if len(train) != 596 or len(dev) != 80:
        raise ValueError('Response partition counts differ')
    original_dev = dev.copy(); isolated = {}
    if a.draft_only:
        for i, row in enumerate(rows):
            key = (row['language'], row['draft'])
            if not row['context']:
                if key in isolated:
                    raise ValueError('Duplicate isolated draft input')
                isolated[key] = i
        if len(isolated) != 204 or any((r['language'], r['draft']) not in isolated for r in rows):
            raise ValueError('Incomplete zero-context draft representations')
        train = np.array([i for i in train if not rows[i]['context']])
        dev = np.array([i for i in dev if not rows[i]['context']])
        if len(train) != 164 or len(dev) != 40:
            raise ValueError('Isolated draft partition counts differ')
    gold = np.array([RESPONSE_LABELS.index(r['gold']) for r in rows]); dev_rows = [rows[i] for i in dev]
    candidates = []; selected = None; config = plan['candidates']
    for c in config['C']:
        model = LogisticRegression(C=c, class_weight=config['class_weight'], solver=config['solver'],
            max_iter=config['maximum_iterations'], tol=config['tolerance'], random_state=config['seed'])
        model.fit(x[train], gold[train])
        if not np.array_equal(model.classes_, np.arange(5)):
            raise ValueError('Unexpected classifier class order')
        artifact = a.private/('C-'+str(c)+'.npz')
        np.savez(artifact, weights=model.coef_, bias=model.intercept_)
        restored = np.load(artifact, allow_pickle=False)
        logits = x[dev]@restored['weights'].T+restored['bias']; predictions = prediction_rows(logits)
        expected = model.predict_proba(x[dev])
        actual = np.array([[p['probabilities'][label] for label in RESPONSE_LABELS] for p in predictions])
        error = float(np.max(np.abs(expected-actual)))
        if error > 1e-10 or not np.array_equal(logits.argmax(axis=1), model.predict(x[dev])):
            raise ValueError('Serialized logistic head differs from sklearn')
        converged = bool(np.all(model.n_iter_ < config['maximum_iterations']))
        record = {'C': c, 'converged': converged, 'iterations': model.n_iter_.tolist(),
            'development': summarize(dev_rows, predictions), 'head_sha256': file_sha(artifact),
            'maximum_export_probability_error': error}
        candidates.append(record)
        if converged and (selected is None or record['development']['selection_rank'] < selected['development']['selection_rank']):
            selected = record; (a.private/'selected.npz').write_bytes(artifact.read_bytes())
        print(json.dumps(record), flush=True)
    if selected is None:
        raise ValueError('No regularized fit converged')
    selection_path = a.study/'selection.json'
    selection_path.write_text(json.dumps({'complete': True, 'selected': selected, 'candidates': candidates,
        'diagnostic_opened_for_scoring': False, 'stable_release_ready': False}, indent=2)+'\n')
    selection_sha = file_sha(selection_path); restored = np.load(a.private/'selected.npz', allow_pickle=False)
    # Diagnostics are opened only after the serialized selection is frozen.
    if (file_sha(a.data/'old-diagnostic.jsonl') != data['artifacts']['old-diagnostic']
            or file_sha(a.contrast_features/'diagnostic-features.npy') != source['diagnostic_features_sha256']):
        raise ValueError('Old diagnostic data or representation differs')
    diagnostic = read_jsonl(a.data/'old-diagnostic.jsonl')
    dx = normalized(np.load(a.contrast_features/'diagnostic-features.npy', allow_pickle=False))
    context_report = json.loads((a.context_study/'report.json').read_text())
    context_protocol = json.loads((a.context_study/'protocol.json').read_text())
    if (not context_report['complete'] or file_sha(a.context_study/'inputs.jsonl') != context_protocol['input_sha256']
            or file_sha(a.context_features/'features.npy') != context_report['features_sha256']
            or context_protocol['prompt'] != protocol['prompt']):
        raise ValueError('Frozen context diagnostic binding differs')
    contexts = read_jsonl(a.context_study/'inputs.jsonl'); cx = normalized(np.load(a.context_features/'features.npy', allow_pickle=False))
    evaluation_dev_rows = dev_rows; evaluation_dev_x = x[dev]
    if a.draft_only:
        if any(r['context'] for r in diagnostic):
            raise ValueError('Diagnostic lacks an isolated draft representation')
        evaluation_dev_rows = [rows[i] for i in original_dev]
        def isolate(population):
            return x[[isolated[(r['language'], r['draft'])] for r in population]]
        evaluation_dev_x = isolate(evaluation_dev_rows); cx = isolate(contexts)
    metrics = {}
    for name, population, values in (('development', evaluation_dev_rows, evaluation_dev_x), ('old-diagnostic', diagnostic, dx), ('context34', contexts, cx)):
        if len(population) != len(values):
            raise ValueError('Diagnostic population and features differ')
        logits = values@restored['weights'].T+restored['bias']; outputs = prediction_rows(logits)
        metrics[name] = summarize(population, outputs)
        write_jsonl(a.study/(name+'-predictions.jsonl'), [{**r, 'prediction': pred, 'logits': z.tolist()}
                    for r, pred, z in zip(population, outputs, logits)])
    if file_sha(selection_path) != selection_sha:
        raise ValueError('Selection changed after diagnostic scoring')
    report = {'complete': True, 'selected': selected, 'selection_sha256': selection_sha, 'metrics': metrics,
        'context_report_sha256': file_sha(a.context_study/'report.json'), 'seconds': time.monotonic()-started,
        'draft_only': a.draft_only, 'contextual_factual_support_established': False,
        'decision': 'REJECT' if any(m['errors'] for m in metrics.values()) else 'RESPONSE_ENGINEERING_PASS_ONLY',
        'fresh_encoder_inference': False, 'independent_validation': False, 'stable_release_ready': False}
    (a.study/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
