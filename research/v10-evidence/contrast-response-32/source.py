"""Fit response heads after a registered matched-pair TRAIN augmentation."""
import argparse
import json
from pathlib import Path
import time

from eval.run_trajectory_v10 import LocalFrozenV9
from roguard.assurance import digest
from roguard.gemma4_review import _Engine
from roguard.trajectory_data import file_sha, read_jsonl, write_jsonl
from roguard.trajectory_model import RESPONSE_LABELS, TrajectoryModel
from training.contrast_response_data_v10 import additional_rows
from training.query_inference_v10 import QueryView
from training.train_neural_response_v10 import PROMPT, make_head, prediction_rows
from training.train_response_readout_v10 import summarize


def payload(row):
    turns = [{'role': 'minor', 'text': row['context']}] if row['context'] else []
    return {'language': row['language'], 'turns': turns+[{'role': 'assistant', 'text': row['draft']}]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'candidate', 'data', 'baseline', 'parent_study', 'parent_features', 'study', 'private'):
        p.add_argument('--'+name.replace('_', '-'), dest=name, type=Path, required=True)
    a = p.parse_args()
    if a.study.exists() or a.private.exists():
        raise ValueError('Refusing to overwrite contrast experiment')
    data = json.loads((a.data/'report.json').read_text())
    for name in ('train', 'dev', 'old-diagnostic'):
        if file_sha(a.data/(name+'.jsonl')) != data['artifacts'][name]:
            raise ValueError('Original response data differs')
    source = json.loads((a.parent_study/'report.json').read_text())
    parent_protocol = json.loads((a.parent_study/'protocol.json').read_text())
    parent_selection = json.loads((a.parent_study/'selection.json').read_text())
    if (not source['complete'] or parent_protocol['prompt'] != PROMPT
            or file_sha(a.parent_study/'selection.json') != source['selection_sha256']
            or file_sha(a.parent_features/'features.npy') != parent_selection['features_sha256']):
        raise ValueError('Parent response feature binding differs')
    original = read_jsonl(a.data/'train.jsonl')+read_jsonl(a.data/'dev.jsonl')
    diagnostic_screen = read_jsonl(a.data/'old-diagnostic.jsonl')
    extra, audit = additional_rows(original+diagnostic_screen)
    identities = read_jsonl(a.parent_study/'feature-inputs.jsonl')
    if len(original) != 420 or len(identities) != 420 or any(
            i != {'case_id': r['case_id'], 'partition': r['partition'], 'input_sha256': digest(payload(r))}
            for r, i in zip(original, identities)):
        raise ValueError('Parent feature order/input identity differs')
    amendment = Path('research/v10-evidence/protocol-amendment-30.json')
    plan = json.loads(amendment.read_text())
    a.study.mkdir(parents=True); a.private.mkdir(parents=True)
    write_jsonl(a.study/'additional-train.jsonl', extra)
    protocol = {'format': 'roguard-contrast-response-fit-1', 'source_sha256': file_sha(Path(__file__)),
        'data_source_sha256': file_sha('training/contrast_response_data_v10.py'),
        'amendment_sha256': file_sha(amendment), 'parent_report_sha256': file_sha(a.parent_study/'report.json'),
        'parent_features_sha256': parent_selection['features_sha256'], 'prompt': PROMPT,
        'additional_train_sha256': file_sha(a.study/'additional-train.jsonl'), 'augmentation_audit': audit,
        'diagnostic_used_for_fit_or_selection': False, 'independent_validation': False, 'stable_release_ready': False}
    (a.study/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (a.study/'source.py').write_bytes(Path(__file__).read_bytes())
    (a.study/'data-source.py').write_bytes(Path('training/contrast_response_data_v10.py').read_bytes())
    import numpy as np
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from safetensors.numpy import save_file, load_file
    started = time.monotonic()
    frozen = LocalFrozenV9(a.base, a.candidate, a.study/'v9-binding')
    engine = _Engine(*frozen.models['ro']._args); query = QueryView(engine, PROMPT)
    parent_x = np.load(a.parent_features/'features.npy', allow_pickle=False)
    if parent_x.shape != (420, 2560) or not np.isfinite(parent_x).all():
        raise ValueError('Invalid parent response representations')
    checked = set(); parent_parity = {}
    for i, row in enumerate(original):
        group = row['language']+':'+row['partition']
        if group not in checked:
            value = query(payload(row), verify_full=True)
            error = float(np.max(np.abs(value-parent_x[i])))
            if error > 1e-3:
                raise ValueError('Reused response representation differs from fresh inference')
            parent_parity[group] = error; checked.add(group)
    def extract(rows):
        values = []
        for i, row in enumerate(rows):
            group = row['language']+':'+row['partition']
            values.append(query(payload(row), verify_full=group not in checked)); checked.add(group)
            mx.clear_cache()
            if (i+1) % 32 == 0:
                print(json.dumps({'phase': 'features', 'rows': i+1, 'total': len(rows)}), flush=True)
        return np.asarray(values, dtype=np.float32)
    extra_x = extract(extra); np.save(a.private/'additional-features.npy', extra_x, allow_pickle=False)
    rows = original+extra; x = np.concatenate((parent_x, extra_x))
    write_jsonl(a.study/'feature-inputs.jsonl', [
        {'case_id': r['case_id'], 'partition': r['partition'], 'input_sha256': digest(payload(r))} for r in rows])
    train = np.array([i for i, r in enumerate(rows) if r['partition'] == 'train'])
    dev = np.array([i for i, r in enumerate(rows) if r['partition'] == 'dev'])
    if len(train) != 596 or len(dev) != 80 or {digest(payload(rows[i])) for i in train} & {digest(payload(rows[i])) for i in dev}:
        raise ValueError('Unexpected contrast fitting population')
    gold = np.array([RESPONSE_LABELS.index(r['gold']) for r in rows], dtype=np.int32)
    weights = mx.array((len(train)/(5*np.bincount(gold[train], minlength=5))).astype(np.float32))
    dev_rows = [rows[i] for i in dev]; history = []; selected = None; optimization = plan['optimization']
    for configuration in plan['candidates']:
        mx.random.seed(optimization['seed']); rng = np.random.default_rng(optimization['seed'])
        head = make_head(configuration); mx.eval(head.parameters())
        optimizer = optim.AdamW(learning_rate=configuration['learning_rate'], weight_decay=optimization['weight_decay'])
        def loss_fn(model, values, labels):
            return (nn.losses.cross_entropy(model(values), labels)*weights[labels]).mean()
        gradient_fn = nn.value_and_grad(head, loss_fn)
        for epoch in range(optimization['epochs']+1):
            if epoch:
                order = rng.permutation(train)
                for start in range(0, len(order), optimization['batch_size']):
                    chosen = order[start:start+optimization['batch_size']]
                    loss, gradient = gradient_fn(head, mx.array(x[chosen]), mx.array(gold[chosen]))
                    gradient, norm = optim.clip_grad_norm(gradient, optimization['gradient_norm_clip']); mx.eval(loss, norm, gradient)
                    if not bool(mx.isfinite(loss).item()) or not bool(mx.isfinite(norm).item()):
                        raise ValueError('Nonfinite response training')
                    optimizer.update(head, gradient); mx.eval(head.parameters(), optimizer.state)
            z = head(mx.array(x[dev])); mx.eval(z); values = np.asarray(z).copy()
            metrics = summarize(dev_rows, prediction_rows(values.astype(np.float64)))
            directory = a.private/configuration['name']; directory.mkdir(exist_ok=True)
            path = directory/f'epoch-{epoch}.safetensors'
            save_file({k: np.asarray(v).copy() for k, v in tree_flatten(head.parameters())}, str(path))
            record = {'configuration': configuration, 'epoch': epoch, 'development': metrics, 'head_sha256': file_sha(path)}
            history.append(record)
            if selected is None or metrics['selection_rank'] < selected['development']['selection_rank']:
                selected = record; (a.private/'selected.safetensors').write_bytes(path.read_bytes())
                np.save(a.private/'selected-development-logits.npy', values, allow_pickle=False)
            print(json.dumps({'phase': 'fit', 'configuration': configuration['name'], 'epoch': epoch, 'errors': metrics['errors']}), flush=True)
    selection = {'complete': True, 'selected': selected, 'epochs': history,
        'additional_features_sha256': file_sha(a.private/'additional-features.npy'),
        'diagnostic_scores_used_for_selection': False}
    path = a.study/'selection.json'; path.write_text(json.dumps(selection, indent=2)+'\n'); selection_sha = file_sha(path)
    chosen = make_head(selected['configuration'])
    chosen.load_weights([(k, mx.array(v)) for k, v in load_file(a.private/'selected.safetensors').items()])
    z = chosen(mx.array(x[dev])); mx.eval(z); dev_z = np.asarray(z).copy()
    if not np.array_equal(dev_z, np.load(a.private/'selected-development-logits.npy', allow_pickle=False)):
        raise ValueError('Selected checkpoint reload differs')
    diagnostic = read_jsonl(a.data/'old-diagnostic.jsonl'); dx = extract(diagnostic)
    np.save(a.private/'diagnostic-features.npy', dx, allow_pickle=False)
    dz = chosen(mx.array(dx)); mx.eval(dz)
    baseline = TrajectoryModel(a.baseline); comparisons = {}
    for name, population, values in (('development', dev_rows, dev_z), ('old-diagnostic', diagnostic, np.asarray(dz))):
        predictions = prediction_rows(values.astype(np.float64))
        old = [baseline.review_response(r['draft'], r['language'], r['context']) for r in population]
        comparisons[name] = {'candidate': summarize(population, predictions), 'baseline': summarize(population, old),
            'fixed': sum(b['label'] != r['gold'] and n['label'] == r['gold'] for r, b, n in zip(population, old, predictions)),
            'introduced': sum(b['label'] == r['gold'] and n['label'] != r['gold'] for r, b, n in zip(population, old, predictions))}
        write_jsonl(a.study/(name+'-predictions.jsonl'), [
            {**r, 'prediction': n, 'baseline': b, 'logits': v.tolist()} for r, n, b, v in zip(population, predictions, old, values)])
    if file_sha(path) != selection_sha:
        raise ValueError('Selection changed after diagnostic scoring')
    report = {'complete': True, 'selected': selected, 'selection_sha256': selection_sha, 'comparisons': comparisons,
        'parent_feature_fresh_parity': parent_parity, 'maximum_full_query_parity_error': query.maximum_full_parity_error,
        'selected_reload_maximum_logit_error': 0., 'diagnostic_features_sha256': file_sha(a.private/'diagnostic-features.npy'),
        'decision': 'REJECT' if any(v['candidate']['errors'] for v in comparisons.values()) else 'RESPONSE_ENGINEERING_PASS_ONLY',
        'seconds': time.monotonic()-started, 'independent_validation': False, 'stable_release_ready': False}
    (a.study/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
