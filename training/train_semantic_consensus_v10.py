"""Fit preregistered score consensus on train, select on original development."""
import argparse
import json
from pathlib import Path

from roguard.trajectory_data import file_sha, read_jsonl, write_jsonl
from training.separation_frontier_v9 import frontier
from training.semantic_consensus_v10 import score_features, predict
from training.shared_tail_training_v9 import make_head


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('features', 'feature_study', 'selected', 'selected_study', 'study'):
        p.add_argument('--'+key.replace('_', '-'), dest=key, type=Path, required=True)
    a = p.parse_args()
    if a.study.exists():
        raise ValueError('Refusing to overwrite consensus fit')
    extraction = json.loads((a.feature_study/'report.json').read_text())
    selection = json.loads((a.selected_study/'report.json').read_text())
    if not extraction['complete'] or not selection['complete']:
        raise ValueError('Incomplete parent study')
    for lang, record in extraction['languages'].items():
        for suffix, sha in record['artifacts'].items():
            if file_sha(a.features/(lang+'-'+suffix)) != sha:
                raise ValueError('Training representation differs')
    amendment = Path('research/v10-evidence/protocol-amendment-10.json')
    configurations = json.loads(amendment.read_text())['registered_candidates']
    a.study.mkdir(parents=True)
    protocol = {'format': 'roguard-semantic-consensus-study-1',
        'source_sha256': file_sha(Path(__file__)),
        'score_source_sha256': file_sha(Path('training/semantic_consensus_v10.py')),
        'amendment_sha256': file_sha(amendment),
        'feature_report_sha256': file_sha(a.feature_study/'report.json'),
        'selection_report_sha256': file_sha(a.selected_study/'report.json'),
        'calibration_or_test_loaded': False, 'stable_release_ready': False}
    (a.study/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (a.study/'source.py').write_bytes(Path(__file__).read_bytes())
    (a.study/'score-source.py').write_bytes(Path('training/semantic_consensus_v10.py').read_bytes())
    import numpy as np
    import mlx.core as mx
    from scipy.optimize import minimize
    from scipy.special import expit
    from safetensors.numpy import load_file
    report = {'complete': False, 'languages': {}, 'stable_release_ready': False}
    artifact = {'format': 'roguard-semantic-consensus-v10-research1',
                'languages': {}, 'qualified': False, 'permits_external_action': False}
    for lang in ('ro', 'uk'):
        spec = json.loads((a.selected_study/(lang+'-review.json')).read_text())
        checkpoint = a.selected/(lang+'-selected.safetensors')
        if file_sha(checkpoint) != spec['adapter_sha256']:
            raise ValueError('Selected adapter binding differs')
        tensors = load_file(checkpoint)
        head = make_head({k: tensors['heads.'+lang+'.'+v] for k, v in
            {'0.weight': 'first.weight', '0.bias': 'first.bias',
             '3.weight': 'last.weight', '3.bias': 'last.bias'}.items()})
        head.eval()
        rows = read_jsonl(a.features/(lang+'-rows.jsonl'))
        x = np.load(a.features/(lang+'-features.npy'), mmap_mode='r')
        z = []
        for start in range(0, len(x), 128):
            values = head(mx.array(x[start:start+128])); mx.eval(values)
            z.append(np.asarray(values).copy())
        f = score_features(np.concatenate(z), spec['couplings'])
        train = np.array([i for i, r in enumerate(rows) if r['partition'] == 'train'])
        dev = np.array([i for i, r in enumerate(rows) if r['partition'] == 'dev'])
        y = np.array([r['gold'] for r in rows], dtype=np.float64)
        from collections import Counter
        strata = [(rows[i]['group'], rows[i]['gold']) for i in train]
        counts = Counter(strata)
        weights = np.array([1/(len(counts)*counts[s]) for s in strata])
        fitted = []; best = None
        for anchor_name in configurations['anchors']:
            anchor = np.array([1., 0., 0., 0.] if anchor_name == 'binary' else [0., 0., 1., 0.])
            for penalty in configurations['regularization']:
                def loss(theta):
                    score = f[train]@theta[:3]+theta[3]
                    residual = weights*(expit(score)-y[train])
                    value = np.sum(weights*(np.logaddexp(0, score)-y[train]*score))
                    difference = theta-anchor
                    gradient = np.r_[f[train].T@residual, residual.sum()]+penalty*difference
                    return value+penalty/2*np.sum(difference*difference), gradient
                result = minimize(loss, anchor, jac=True, method='L-BFGS-B',
                    bounds=[(0, None)]*3+[(None, None)], options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
                if not result.success or not np.isfinite(result.x).all():
                    raise ValueError('Consensus optimization did not converge')
                parameters = {'weights': result.x[:3].tolist(), 'intercept': float(result.x[3])}
                scores = predict(f[dev], parameters)
                population = [{'case_id': rows[i]['case_id'], 'language': lang,
                    'group': rows[i]['group'], 'gold': int(y[i]), 'score': float(score)}
                    for i, score in zip(dev, scores)]
                boundary = frontier(population)
                for row in population:
                    row['predicted'] = int(row['score'] >= boundary['threshold'])
                errors = sum(r['predicted'] != r['gold'] for r in population)
                rank = [errors, -boundary['score_margin'], -penalty]
                record = {'anchor': anchor_name, 'regularization': penalty, **parameters,
                    'binary_errors': errors, 'threshold': boundary['threshold'],
                    'score_margin': boundary['score_margin'], 'selection_rank': rank}
                fitted.append(record)
                if best is None or rank < best['selection_rank']:
                    best = record
                    write_jsonl(a.study/(lang+'-development-predictions.jsonl'), population)
        report['languages'][lang] = {'registered_fits': fitted, 'selected': best,
            'training_rows': len(train), 'development_rows': len(dev)}
        artifact['languages'][lang] = {**best, 'adapter_sha256': spec['adapter_sha256'],
            'review_spec_sha256': file_sha(a.selected_study/(lang+'-review.json'))}
    report['complete'] = True
    (a.study/'consensus.json').write_text(json.dumps(artifact, indent=2)+'\n')
    report['consensus_sha256'] = file_sha(a.study/'consensus.json')
    (a.study/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({lang:r['selected'] for lang,r in report['languages'].items()}, indent=2))


if __name__ == '__main__':
    main()
