"""Separate class-balanced assertion training on fixed original TRAIN/DEV views."""
import argparse
import json
from pathlib import Path
import time

from roguard.trajectory_data import file_sha, read_jsonl, write_jsonl
from training.probe_assertion_query_v10 import LABELS


def make_head(configuration):
    import mlx.core as mx
    import mlx.nn as nn
    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            width = 5120 if configuration['view'] == 'dual' else 2560
            if configuration['hidden']:
                self.first = nn.Linear(width, 128); self.last = nn.Linear(128, 4)
            else:
                self.last = nn.Linear(width, 4)
        def __call__(self, x):
            if x.ndim != 2 or x.shape[1] != 5120:
                raise ValueError('Expected paired frozen representations')
            views = mx.split(x, 2, axis=1)
            if configuration['view'] != 'dual':
                views = [views[0 if configuration['view'] == 'v9' else 1]]
            x = mx.concatenate([v/mx.sqrt(mx.maximum(mx.mean(mx.square(v), axis=1, keepdims=True), 1e-12)) for v in views], axis=1)
            if configuration['hidden']:
                x = nn.gelu(self.first(x))
            return self.last(x)
    if configuration['view'] not in ('v9', 'query', 'dual') or configuration['hidden'] not in (0, 128):
        raise ValueError('Unregistered assertion-head configuration')
    return Head()


def metrics(logits, gold):
    import numpy as np
    values = logits.astype(np.float64)
    if values.shape != (len(gold), 4) or not np.isfinite(values).all():
        raise ValueError('Invalid assertion logits')
    ex = np.exp(values-values.max(axis=1, keepdims=True)); probabilities = ex/ex.sum(axis=1, keepdims=True)
    predicted = probabilities.argmax(axis=1); groups = {}; class_ce = []
    for i, label in enumerate(LABELS):
        mask = gold == i
        if not mask.any():
            raise ValueError('Missing assertion class')
        groups[label] = {'total': int(mask.sum()), 'errors': int((predicted[mask] != i).sum())}
        class_ce.append(float(-np.log(np.maximum(probabilities[mask, i], 1e-30)).mean()))
    errors = int((predicted != gold).sum())
    confusion = np.zeros((4, 4), dtype=int)
    np.add.at(confusion, (gold, predicted), 1)
    summary = {'rows': len(gold), 'errors': errors, 'groups': groups, 'confusion': confusion.tolist(),
        'mean_class_cross_entropy': float(np.mean(class_ce)),
        'selection_rank': [max(v['errors']/v['total'] for v in groups.values()), errors, float(np.mean(class_ce))]}
    return summary, probabilities


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('features', 'feature_study', 'study', 'private'):
        p.add_argument('--'+name.replace('_', '-'), dest=name, type=Path, required=True)
    a = p.parse_args()
    if a.study.exists() or a.private.exists():
        raise ValueError('Refusing to overwrite assertion training')
    source = json.loads((a.feature_study/'report.json').read_text())
    source_protocol = json.loads((a.feature_study/'protocol.json').read_text())
    if not source['complete'] or source_protocol['format'] != 'roguard-query-reference-features-1':
        raise ValueError('Expected completed fixed query representations')
    for lang, record in source['languages'].items():
        for suffix, sha in record['artifacts'].items():
            if file_sha(a.features/(lang+'-'+suffix)) != sha:
                raise ValueError('Assertion input feature binding differs')
    amendment = Path('research/v10-evidence/protocol-amendment-28.json')
    optimization = json.loads(amendment.read_text())['optimization']
    configurations = [{'name': view+('-mlp128' if hidden else '-linear'), 'view': view,
                       'hidden': hidden, 'learning_rate': .0003 if hidden else .001}
                      for view in ('v9', 'query', 'dual') for hidden in (0, 128)]
    a.study.mkdir(parents=True); a.private.mkdir(parents=True)
    protocol = {'format': 'roguard-assertion-specialist-fit-1', 'source_sha256': file_sha(Path(__file__)),
        'amendment_sha256': file_sha(amendment), 'feature_report_sha256': file_sha(a.feature_study/'report.json'),
        'configurations': configurations, 'optimization': optimization, 'labels': list(LABELS),
        'binary_decision_changed': False, 'calibration_or_regression_loaded': False, 'stable_release_ready': False}
    (a.study/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (a.study/'source.py').write_bytes(Path(__file__).read_bytes())
    import numpy as np
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from safetensors.numpy import save_file, load_file
    mx.set_memory_limit(4*1024**3); mx.set_cache_limit(128*1024**2)
    started = time.monotonic(); report = {'complete': False, 'languages': {}, 'binary_decision_changed': False,
        'calibration_or_regression_loaded': False, 'stable_release_ready': False}
    def persist():
        report['seconds'] = time.monotonic()-started
        (a.study/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    for lang in ('ro', 'uk'):
        rows = read_jsonl(a.features/(lang+'-rows.jsonl'))
        x = np.load(a.features/(lang+'-features.npy'), allow_pickle=False)
        original = np.load(a.features/(lang+'-logits.npy'), allow_pickle=False)
        if x.shape != (len(rows), 5120) or original.shape != (len(rows), 19) or not np.isfinite(x).all() or not np.isfinite(original).all():
            raise ValueError('Malformed assertion feature tensors')
        train = np.array([i for i, r in enumerate(rows) if r['partition'] == 'train'])
        dev = np.array([i for i, r in enumerate(rows) if r['partition'] == 'dev'])
        if len(train)+len(dev) != len(rows) or {rows[i]['query_token_sha256'] for i in train} & {rows[i]['query_token_sha256'] for i in dev}:
            raise ValueError('Assertion TRAIN/DEV binding differs')
        labels = np.array([r['fields'][2] for r in rows], dtype=np.int32)
        counts = np.bincount(labels[train], minlength=4)
        if len(counts) != 4 or not (counts > 0).all():
            raise ValueError('Expected all four TRAIN assertion classes')
        weights = mx.array((len(train)/(4*counts)).astype(np.float32))
        baseline, _ = metrics(original[dev, 6:10], labels[dev])
        record = {'training_rows': len(train), 'development_rows': len(dev), 'training_class_counts': counts.tolist(),
                  'development_class_counts': np.bincount(labels[dev], minlength=4).tolist(),
                  'baseline': baseline, 'epochs': [], 'selected': None}
        report['languages'][lang] = record
        for configuration in configurations:
            mx.random.seed(optimization['seed']); rng = np.random.default_rng(optimization['seed'])
            head = make_head(configuration); mx.eval(head.parameters())
            optimizer = optim.AdamW(learning_rate=configuration['learning_rate'], weight_decay=optimization['weight_decay'])
            def loss_fn(model, features, gold):
                return (nn.losses.cross_entropy(model(features), gold)*weights[gold]).mean()
            value_grad = nn.value_and_grad(head, loss_fn)
            for epoch in range(optimization['epochs']+1):
                if epoch:
                    order = rng.permutation(train)
                    for start in range(0, len(train), optimization['batch_size']):
                        chosen = order[start:start+optimization['batch_size']]
                        loss, gradient = value_grad(head, mx.array(x[chosen]), mx.array(labels[chosen]))
                        gradient, norm = optim.clip_grad_norm(gradient, optimization['gradient_norm_clip']); mx.eval(loss, norm, gradient)
                        if not bool(mx.isfinite(loss).item()) or not bool(mx.isfinite(norm).item()):
                            raise ValueError('Nonfinite assertion optimization')
                        optimizer.update(head, gradient); mx.eval(head.parameters(), optimizer.state)
                parts = []
                for start in range(0, len(dev), 256):
                    z = head(mx.array(x[dev[start:start+256]])); mx.eval(z); parts.append(np.asarray(z).copy())
                logits = np.concatenate(parts); summary, probabilities = metrics(logits, labels[dev])
                directory = a.private/lang/configuration['name']; directory.mkdir(parents=True, exist_ok=True)
                path = directory/('epoch-'+str(epoch)+'.safetensors')
                save_file({k: np.asarray(v).copy() for k, v in tree_flatten(head.parameters())}, str(path))
                result = {'configuration': configuration, 'epoch': epoch, **summary, 'head_sha256': file_sha(path)}
                record['epochs'].append(result)
                if record['selected'] is None or result['selection_rank'] < record['selected']['selection_rank']:
                    record['selected'] = result
                    (a.private/(lang+'-selected.safetensors')).write_bytes(path.read_bytes())
                    np.save(a.private/(lang+'-selected-development-logits.npy'), logits, allow_pickle=False)
                    spec = {'format': 'roguard-assertion-readout-research-1', 'language': lang,
                        'configuration': configuration, 'labels': list(LABELS), 'head_sha256': file_sha(path),
                        'feature_report_sha256': protocol['feature_report_sha256'], 'query_prompt': source_protocol['prompt'],
                        'qualified': False, 'binary_decision_changed': False}
                    (a.study/(lang+'-head.json')).write_text(json.dumps(spec, indent=2)+'\n')
                    write_jsonl(a.study/(lang+'-development-predictions.jsonl'), [
                        {'case_id': rows[i]['case_id'], 'language': lang, 'gold': LABELS[labels[i]],
                         'predicted': LABELS[int(p.argmax())], 'probabilities': p.tolist(),
                         'token_sha256': rows[i]['token_sha256'], 'query_token_sha256': rows[i]['query_token_sha256']}
                        for i, p in zip(dev, probabilities)])
                persist()
                print(json.dumps({'language': lang, 'candidate': configuration['name'], 'epoch': epoch,
                                  'errors': summary['errors'], 'maximum_class_error_rate': summary['selection_rank'][0]}), flush=True)
            del head, optimizer, value_grad; mx.clear_cache()
        selected = make_head(record['selected']['configuration'])
        selected.load_weights([(k, mx.array(v)) for k, v in load_file(a.private/(lang+'-selected.safetensors')).items()])
        parts = []
        for start in range(0, len(dev), 256):
            z = selected(mx.array(x[dev[start:start+256]])); mx.eval(z); parts.append(np.asarray(z).copy())
        if not np.array_equal(np.concatenate(parts), np.load(a.private/(lang+'-selected-development-logits.npy'), allow_pickle=False)):
            raise ValueError('Selected assertion checkpoint reload differs')
        record['selected_reload_maximum_error'] = 0.; record['complete'] = True; persist()
        del x, original, selected; mx.clear_cache()
    report['complete'] = True
    report['decision'] = 'REJECT' if any(v['selected']['errors'] for v in report['languages'].values()) else 'ASSERTION_DEVELOPMENT_PASS_ONLY'
    persist(); print(json.dumps({k: v['selected'] for k, v in report['languages'].items()}, indent=2))


if __name__ == '__main__':
    main()
