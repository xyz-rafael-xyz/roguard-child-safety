"""Conditional CPU-only nonlinear readout using study141's verified features."""
from datetime import datetime,timezone
import gzip
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

from training.budgeted_native_frontier_v9 import frontier
from training.gemma4_readout_v9 import normalize
from training.gemma4_kernel_v9 import margins,scores
from training.prepare_instruction_sft_v8 import target


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import numpy as np
    from sklearn.svm import SVC
    from threadpoolctl import threadpool_limits
    root = Path('research/v9-stabilization')
    study = root/'gemma4-kernel-142'
    parent = root/'gemma4-readout-141'
    for freeze in (study/'source-freeze.json',parent/'source-freeze.json'):
        for path,expected in json.loads(freeze.read_text()).items():
            if digest(path) != expected:
                raise ValueError('Frozen source differs: '+path)
    software = json.loads((study/'software.json').read_text())
    for name,expected in software.items():
        if version(name) != expected:
            raise ValueError('Fitting dependency differs: '+name)
    original = json.loads((parent/'actual/report.json').read_text())
    if not original.get('complete') or original['source_freeze_sha256'] != digest(parent/'source-freeze.json'):
        raise ValueError('Incomplete or changed parent study')
    output = study/'actual'
    output.mkdir(exist_ok=False)
    if original['development_passes']:
        (output/'report.json').write_text(json.dumps(dict(complete=True,skipped=True,
            reason='Study141 passes development; prioritize its remaining verification'),indent=2)+'\n')
        return
    started = time.monotonic()
    rows = [json.loads(l) for l in gzip.decompress((parent/'inputs.jsonl.gz').read_bytes()).decode().splitlines()]
    chunks = []
    offset = 0
    for chunk in original['chunks']:
        start,stop = chunk['start'],chunk['stop']
        if start != offset or stop <= start or stop > len(rows):
            raise ValueError('Feature chunks are not contiguous')
        identity = hashlib.sha256(json.dumps([r['token_sha256'] for r in rows[start:stop]]).encode()).hexdigest()
        path = parent/'actual'/(f'chunk-{start:06d}-features.npz')
        if (identity != chunk['input_sha256'] or digest(path) != chunk['feature_sha256']
                or chunk['source_freeze_sha256'] != digest(parent/'source-freeze.json')):
            raise ValueError('Feature chunk provenance differs')
        features = np.load(path,allow_pickle=False)['features']
        if features.shape != (stop-start,original['feature_width']) or not np.isfinite(features).all():
            raise ValueError('Feature dimensions or values differ')
        chunks.append(features)
        offset = stop
    if offset != len(rows):
        raise ValueError('Incomplete feature population')
    features = np.concatenate(chunks)
    protocol = json.loads((study/'protocol.json').read_text())
    report = dict(complete=False,parent_report_sha256=digest(parent/'actual/report.json'),
        source_freeze_sha256=digest(study/'source-freeze.json'),candidates={},selected={},
        synthetic_only=True,independently_annotated=False,fresh_test_inferred=False,
        calibration_inferred=False,new_foundation_inference=False,stable_release_ready=False)
    for lang in ('ro','uk'):
        train = [i for i,r in enumerate(rows) if r['row']['language']==lang and r['partition']=='train']
        dev = [i for i,r in enumerate(rows) if r['row']['language']==lang and r['partition']=='dev']
        y = np.array([target(rows[i]['row']) for i in train])
        truth = [target(rows[i]['row']) for i in dev]
        x = normalize(features[train])
        previous = original['selected'][lang]
        selected = dict(kind='linear',parent_language=lang,frontier=previous['frontier'],C=previous['C'])
        # Exact ties retain the already-selected linear classifier.
        best = tuple(previous['frontier']['rank'])
        candidates = []
        for c in protocol['candidates_C']:
            for factor in protocol['gamma_width_factors']:
                gamma = factor/features.shape[1]
                with threadpool_limits(limits=4):
                    fitted = SVC(C=c,gamma=gamma,kernel='rbf',class_weight='balanced',
                        tol=1e-5,max_iter=100000,cache_size=256,random_state=1142).fit(x,y)
                    if fitted.fit_status_ != 0:
                        raise ValueError('Kernel solver did not converge')
                    head = dict(dual_coefficients=fitted.dual_coef_[0].tolist(),
                        intercept=float(fitted.intercept_[0]),gamma=float(fitted._gamma))
                    support = fitted.support_vectors_
                    reference = fitted.decision_function(normalize(features[dev]))
                    exported = margins(features[dev],head,support)
                    error = float(np.max(np.abs(reference-exported)))
                    if error > 1e-7:
                        raise ValueError('Portable kernel margin differs from fitted classifier')
                    probabilities = scores(features[dev],head,support)
                population = [dict(gold=int(g),score=float(s),group=rows[i]['group']) for i,g,s in zip(dev,truth,probabilities)]
                optimum = frontier(population)
                entry = dict(kind='rbf',C=c,gamma=gamma,gamma_width_factor=factor,
                    support_vectors=len(support),frontier=optimum,export_margin_error=error)
                candidates.append(entry)
                rank = tuple(optimum['rank'])
                if rank < best:
                    best = rank
                    selected = entry
                    path = output/(lang+'-support-features.npz')
                    np.savez_compressed(path,support=support)
                    head.update(format='roguard-gemma4-kernel-readout-experiment-v1',language=lang,
                        normalization='l2-times-sqrt-width',score='0.5+atan(margin)/pi; not a probability',
                        threshold=optimum['threshold'],support_file=path.name,support_sha256=digest(path),
                        source_freeze_sha256=digest(study/'source-freeze.json'),
                        base_manifest_sha256=digest(parent/'base-manifest.json'))
                    (output/(lang+'-readout.json')).write_text(json.dumps(head,indent=2)+'\n')
                    predictions = [dict(case_id=rows[i]['row']['case_id'],language=lang,group=rows[i]['group'],
                        gold=int(g),score=float(s),predicted=int(s>=optimum['threshold'])) for i,g,s in zip(dev,truth,probabilities)]
                    (output/(lang+'-dev-predictions.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in predictions))
                print('FIT',lang,json.dumps(entry),flush=True)
        report['candidates'][lang] = candidates
        report['selected'][lang] = selected
    report.update(complete=True,finished_utc=datetime.now(timezone.utc).isoformat(),
        seconds=time.monotonic()-started,
        development_passes=all(r['frontier']['counts_pass'] for r in report['selected'].values()))
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('FINISHED',json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
