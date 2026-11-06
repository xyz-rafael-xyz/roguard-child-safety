"""Train frozen-representation heads under the registered base-reference study."""
import argparse
import json
from pathlib import Path
import time

from roguard.trajectory_data import file_sha, read_jsonl, write_jsonl
from training.base_reference_head_v10 import make_head
from training.native_fields_v9 import SIZES, policy_mass
from training.reference_frontier_v10 import frontier


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('features', 'feature_study', 'study', 'private'):
        p.add_argument('--'+key.replace('_', '-'), dest=key, type=Path, required=True)
    p.add_argument('--curriculum', action='store_true', help='Use the registered scope-20 curriculum and old/new weighting.')
    p.add_argument('--query', action='store_true', help='Use the registered query-conditioned study-24 features.')
    a = p.parse_args()
    if a.curriculum and a.query:
        raise ValueError('Query and curriculum are different registered experiments')
    if a.study.exists() or a.private.exists():
        raise ValueError('Refusing to overwrite a reference-head fit')
    extraction = json.loads((a.feature_study/'report.json').read_text())
    if not extraction['complete']:
        raise ValueError('Paired feature extraction is incomplete')
    base_amendment = Path('research/v10-evidence/protocol-amendment-15.json')
    amendment = Path('research/v10-evidence/protocol-amendment-17.json') if a.curriculum else base_amendment
    if a.query:
        amendment = Path('research/v10-evidence/protocol-amendment-22.json')
    feature_protocol = json.loads((a.feature_study/'protocol.json').read_text())
    expected_format = 'roguard-scope-reference-features-1' if a.curriculum else 'roguard-base-reference-features-1'
    if a.query:
        expected_format = 'roguard-query-reference-features-1'
    if feature_protocol['amendment_sha256'] != file_sha(amendment) or feature_protocol['format'] != expected_format:
        raise ValueError('Feature extraction and registered plan differ')
    plan = json.loads(base_amendment.read_text()); optimization = dict(plan['optimization'])
    if a.curriculum:
        if feature_protocol['input_clarification_sha256'] != file_sha('research/v10-evidence/protocol-amendment-18.json'):
            raise ValueError('Curriculum effective-input audit differs')
        optimization['seed'] = 1020
    if a.query:
        optimization['seed'] = 1024
    for lang, record in extraction['languages'].items():
        for suffix, expected in record['artifacts'].items():
            if file_sha(a.features/(lang+'-'+suffix)) != expected:
                raise ValueError('Paired feature binding differs')
    a.study.mkdir(parents=True); a.private.mkdir(parents=True)
    protocol = {'format': 'roguard-query-reference-fit-1' if a.query else 'roguard-base-reference-fit-1',
        'source_sha256': file_sha(Path(__file__)), 'head_source_sha256': file_sha('training/base_reference_head_v10.py'),
        'frontier_source_sha256': file_sha('training/reference_frontier_v10.py'),
        'amendment_sha256': file_sha(amendment), 'feature_report_sha256': file_sha(a.feature_study/'report.json'),
        'optimization': optimization, 'registered_candidates': plan['registered_candidates'],
        'base_configuration_amendment_sha256': file_sha(base_amendment),
        'equal_old_new_training_mass': a.curriculum,
        'calibration_or_test_loaded': False, 'stable_release_ready': False}
    if a.query:
        protocol['query_prompt'] = feature_protocol['prompt']
        protocol['query_helper_sha256'] = feature_protocol['query_helper_sha256']
    (a.study/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (a.study/'source.py').write_bytes(Path(__file__).read_bytes())
    (a.study/'head-source.py').write_bytes(Path('training/base_reference_head_v10.py').read_bytes())
    (a.study/'frontier-source.py').write_bytes(Path('training/reference_frontier_v10.py').read_bytes())
    import numpy as np
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from safetensors.numpy import save_file
    mx.set_memory_limit(4*1024**3); mx.set_cache_limit(128*1024**2)
    started = time.monotonic()
    report = {'complete': False, 'languages': {}, 'calibration_or_test_loaded': False,
              'stable_release_ready': False}
    def persist():
        report['seconds'] = time.monotonic()-started
        (a.study/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    for lang in ('ro', 'uk'):
        rows = read_jsonl(a.features/(lang+'-rows.jsonl'))
        x = np.load(a.features/(lang+'-features.npy'), allow_pickle=False)
        if x.shape != (len(rows), 5120) or not np.isfinite(x).all():
            raise ValueError('Invalid paired representations')
        train = np.array([i for i, r in enumerate(rows) if r['partition'] == 'train'])
        dev = np.array([i for i, r in enumerate(rows) if r['partition'] == 'dev'])
        if not len(train) or not len(dev) or len(train)+len(dev) != len(rows):
            raise ValueError('Only TRAIN and DEV are permitted')
        if {rows[i]['token_sha256'] for i in train} & {rows[i]['token_sha256'] for i in dev}:
            raise ValueError('Exact model input overlaps TRAIN and DEV')
        if a.query and {rows[i]['query_token_sha256'] for i in train} & {rows[i]['query_token_sha256'] for i in dev}:
            raise ValueError('Exact query input overlaps TRAIN and DEV')
        labels = np.array([r['fields'] for r in rows], dtype=np.int32)
        gold = np.array([r['gold'] for r in rows], dtype=np.int32)
        row_weights = np.ones(len(rows), dtype=np.float32)
        if a.curriculum:
            new = np.array([r.get('curriculum', False) for r in rows], dtype=bool)
            if new[dev].any() or not new[train].any() or new[train].all():
                raise ValueError('Curriculum must add TRAIN only while retaining original TRAIN and DEV')
            for flag in (False, True):
                indices = train[new[train] == flag]
                row_weights[indices] = len(train)/(2*len(indices))
        original_spec = json.loads((a.feature_study/'v9-binding'/(lang+'-review.json')).read_text())
        weights = []
        for j, size in enumerate(SIZES):
            counts = np.bincount(labels[train, j], weights=row_weights[train], minlength=size)
            values = np.minimum(4., np.sqrt(len(train)/np.maximum(1, counts)/size))
            values /= np.average(values[labels[train, j]], weights=row_weights[train])
            weights.append(mx.array(values.astype(np.float32)))
        record = {'training_rows': len(train), 'development_rows': len(dev), 'epochs': [], 'selected': None}
        report['languages'][lang] = record
        def evaluate(head, configuration, epoch):
            head.eval(); parts = []
            for offset in range(0, len(dev), 256):
                logits = head(mx.array(x[dev[offset:offset+256]])); mx.eval(logits)
                parts.append(np.asarray(logits).copy())
            z = np.concatenate(parts).astype(np.float64)
            if not np.isfinite(z).all():
                raise ValueError('Nonfinite development logits')
            def softmax(values):
                exp = np.exp(values-values.max(axis=1, keepdims=True))
                return exp/exp.sum(axis=1, keepdims=True)
            factors = [softmax(v) for v in np.split(z[:, :sum(SIZES)], np.cumsum(SIZES)[:-1], axis=1)]
            predicted_fields = np.column_stack([v.argmax(axis=1) for v in factors])
            joint_errors = int(np.any(predicted_fields != labels[dev], axis=1).sum())
            ce = float(np.mean([-np.log(np.maximum(v[np.arange(len(dev)), labels[dev, j]], 1e-30)).mean()
                                for j, v in enumerate(factors)]))
            scores = {'binary': softmax(z[:, -2:])[:, 1], 'policy': policy_mass(factors),
                      'coupled_policy': policy_mass(factors, original_spec['couplings'])}
            choices = []
            for mode, values in scores.items():
                population = [{'case_id': rows[i]['case_id'], 'language': lang,
                               'group': rows[i]['group'], 'gold': int(gold[i]), 'score': float(value)}
                              for i, value in zip(dev, values)]
                boundary = frontier(population)
                binary = (values >= boundary['threshold']).astype(int)
                errors = int((binary != gold[dev]).sum())
                choices.append((errors, -boundary['score_margin'], mode, boundary, binary, population))
            errors, negative_margin, mode, boundary, binary, population = min(choices, key=lambda r: r[:2])
            rank = [errors, joint_errors, ce, negative_margin]
            summary = {'candidate': configuration['name'], 'epoch': epoch, 'binary_errors': errors,
                'joint_argmax_errors': joint_errors, 'field_errors': (predicted_fields != labels[dev]).sum(axis=0).tolist(),
                'mean_field_cross_entropy': ce, 'mode': mode, 'threshold': boundary['threshold'],
                'score_margin': -negative_margin, 'selection_rank': rank}
            record['epochs'].append(summary)
            tensors = {k: np.asarray(v).copy() for k, v in tree_flatten(head.parameters())}
            destination = a.private/lang/configuration['name']; destination.mkdir(parents=True, exist_ok=True)
            checkpoint = destination/('epoch-'+str(epoch)+'.safetensors')
            save_file(tensors, str(checkpoint))
            if record['selected'] is None or rank < record['selected']['selection_rank']:
                selected = a.private/(lang+'-selected.safetensors'); selected.write_bytes(checkpoint.read_bytes())
                summary['head_sha256'] = file_sha(selected); record['selected'] = dict(summary)
                spec = {'format': 'roguard-query-reference-head-1' if a.query else 'roguard-base-reference-head-1', 'language': lang,
                        'configuration': configuration, 'head_sha256': file_sha(selected),
                        'v9_identity': file_sha(a.feature_study/'v9-binding'/(lang+'-review.json')),
                        'mode': mode, 'threshold': boundary['threshold'], 'couplings': original_spec['couplings'],
                        'qualified': False}
                if a.query:
                    spec['query_prompt'] = feature_protocol['prompt']
                    spec['feature_report_sha256'] = protocol['feature_report_sha256']
                (a.study/(lang+'-head.json')).write_text(json.dumps(spec, indent=2)+'\n')
                for row, value, logits in zip(population, binary, z):
                    row['predicted'] = int(value); row['field_logits'] = logits.tolist()
                write_jsonl(a.study/(lang+'-development-predictions.jsonl'), population)
                np.save(a.private/(lang+'-selected-dev-logits.npy'), z, allow_pickle=False)
            persist(); head.train()
            print(json.dumps({'language': lang, **summary}), flush=True)
        for configuration in plan['registered_candidates']:
            mx.random.seed(optimization['seed']); rng = np.random.default_rng(optimization['seed'])
            head = make_head(configuration); mx.eval(head.parameters())
            optimizer = optim.AdamW(learning_rate=configuration['learning_rate'], weight_decay=optimization['weight_decay'])
            def loss_fn(model, features, fields, binary, sample_weights):
                output = model(features); terms = []; offset = 0
                for j, size in enumerate(SIZES):
                    terms.append(nn.losses.cross_entropy(output[:, offset:offset+size], fields[:, j])*weights[j][fields[:, j]])
                    offset += size
                return (sample_weights*(optimization['binary_ce_weight']*nn.losses.cross_entropy(output[:, -2:], binary)
                        +optimization['semantic_ce_weight']*mx.stack(terms).mean(axis=0))).mean()
            value_grad = nn.value_and_grad(head, loss_fn)
            evaluate(head, configuration, 0)
            for epoch in range(1, optimization['epochs']+1):
                order = rng.permutation(train); losses = []
                for start in range(0, len(train), optimization['batch_size']):
                    chosen = order[start:start+optimization['batch_size']]
                    loss, gradient = value_grad(head, mx.array(x[chosen]), mx.array(labels[chosen]), mx.array(gold[chosen]),
                                                mx.array(row_weights[chosen]))
                    gradient, norm = optim.clip_grad_norm(gradient, optimization['gradient_norm_clip'])
                    mx.eval(loss, norm, gradient)
                    if not bool(mx.isfinite(loss).item()) or not bool(mx.isfinite(norm).item()):
                        raise ValueError('Nonfinite optimization step')
                    optimizer.update(head, gradient); mx.eval(head.parameters(), optimizer.state)
                    losses.append(float(loss.item()))
                evaluate(head, configuration, epoch)
                record['epochs'][-1]['mean_training_loss'] = float(np.mean(losses)); persist()
            del head, optimizer, value_grad; mx.clear_cache()
        record['complete'] = True; persist(); del x
    report['complete'] = True; persist()
    print(json.dumps({lang: record['selected'] for lang, record in report['languages'].items()}, indent=2))


if __name__ == '__main__':
    main()
