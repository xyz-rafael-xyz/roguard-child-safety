"""Extract frozen Gemma4 features, then fit and select binary readouts on DEV."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
from importlib.metadata import distribution, version
import json
from pathlib import Path
import time
import warnings

from training.gemma4_prompt_v9 import classifier_tokens
from training.gemma4_readout_v9 import last_features, normalize, scores
from training.prepare_instruction_sft_v8 import target
from training.budgeted_native_frontier_v9 import frontier


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    study = Path('research/v9-stabilization/gemma4-readout-141')
    for name, expected in json.loads((study/'source-freeze.json').read_text()).items():
        if digest(name) != expected:
            raise ValueError('Source changed: '+name)
    manifest = json.loads((study/'base-manifest.json').read_text())
    if args.base.name != manifest['revision']:
        raise ValueError('Base revision differs')
    for name, entry in manifest['files'].items():
        if digest(args.base/name) != entry['sha256']:
            raise ValueError('Base file differs: '+name)
    software = json.loads((study/'software.json').read_text())
    for name, expected in software['versions'].items():
        if version(name) != expected:
            raise ValueError('Runtime version differs: '+name)
    package = Path(distribution('mlx-lm').locate_file('mlx_lm'))
    for name, expected in software['implementation_sha256'].items():
        if digest(package/name) != expected:
            raise ValueError('MLX implementation differs: '+name)
    import numpy as np
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    protocol = json.loads((study/'protocol.json').read_text())
    output = study/'actual'
    output.mkdir(exist_ok=True)
    if (output/'report.json').exists():
        raise ValueError('Completed study must not be overwritten')
    mx.set_memory_limit(10*1024**3)
    mx.set_cache_limit(256*1024**2)
    started = time.monotonic()
    model, tokenizer = load(str(args.base), tokenizer_config={'trust_remote_code':False})
    model.set_dtype(mx.float32)
    label_ids = [tokenizer.encode(str(i), add_special_tokens=False) for i in (0,1)]
    if any(len(v) != 1 for v in label_ids):
        raise ValueError('Expected one-token labels')
    rows = [json.loads(line) for line in gzip.decompress((study/'inputs.jsonl.gz').read_bytes()).decode().splitlines()]
    encoded = [classifier_tokens(tokenizer, r['row'], [v[0] for v in label_ids]) for r in rows]
    for r, ids in zip(rows, encoded):
        if (len(ids) != r['token_count'] or hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest() != r['token_sha256']):
            raise ValueError('Runtime prompt differs from preflight')
    common = 0
    for column in zip(*encoded):
        if len(set(column)) != 1:
            break
        common += 1
    common = min(common, min(map(len, encoded))-1)
    cache = make_prompt_cache(model)
    if common:
        mx.eval(model.language_model.model(mx.array([encoded[0][:common]]), cache=cache))
    feature_chunks = []
    max_error = 0.
    chunks = []
    for start in range(0, len(rows), protocol['feature_chunk_size']):
        stop = min(start+protocol['feature_chunk_size'], len(rows))
        path = output/(f'chunk-{start:06d}-features.npz')
        record_path = output/(f'chunk-{start:06d}.json')
        identity = hashlib.sha256(json.dumps([r['token_sha256'] for r in rows[start:stop]]).encode()).hexdigest()
        if path.exists() and record_path.exists():
            record = json.loads(record_path.read_text())
            if record['input_sha256'] != identity or record['feature_sha256'] != digest(path) or record['source_freeze_sha256'] != digest(study/'source-freeze.json'):
                raise ValueError('Cached representation provenance differs')
            features = np.load(path, allow_pickle=False)['features']
            if features.shape[0] != stop-start or not np.isfinite(features).all():
                raise ValueError('Invalid cached features')
            max_error = max(max_error, record['maximum_cache_relative_error'])
        else:
            values = []
            for i in range(start, stop):
                hidden = last_features(model, mx.array([encoded[i][common:]]), cache=deepcopy(cache)).astype(mx.float32)
                mx.eval(hidden)
                if i < 2:
                    full = last_features(model, mx.array([encoded[i]])).astype(mx.float32)
                    relative = mx.max(mx.abs(full-hidden))/mx.maximum(1.,mx.max(mx.abs(full)))
                    mx.eval(relative)
                    max_error = max(max_error, float(relative.item()))
                    if float(relative.item()) > .0001:
                        raise ValueError('Cached/full contextual features differ')
                    del full, relative
                values.append(np.asarray(hidden[0]).copy())
                del hidden
                mx.clear_cache()
            features = np.stack(values)
            if not np.isfinite(features).all():
                raise ValueError('Nonfinite contextual features')
            np.savez_compressed(path, features=features)
            record = dict(start=start, stop=stop, input_sha256=identity,
                feature_sha256=digest(path), source_freeze_sha256=digest(study/'source-freeze.json'),
                maximum_cache_relative_error=max_error)
            record_path.write_text(json.dumps(record,indent=2)+'\n')
        chunks.append(record)
        feature_chunks.append(features)
        print('FEATURES',stop,'/',len(rows),'seconds',round(time.monotonic()-started,2),flush=True)
    features = np.concatenate(feature_chunks)
    peak = mx.get_peak_memory()
    del model, cache
    mx.clear_cache()
    report = dict(complete=False, rows=len(rows), feature_width=features.shape[1],
        source_freeze_sha256=digest(study/'source-freeze.json'), chunks=chunks,
        maximum_cache_relative_error=max_error, peak_mlx_bytes=peak,
        selected={}, candidates={}, synthetic_only=True, independently_annotated=False,
        fresh_test_inferred=False, calibration_inferred=False, stable_release_ready=False,
        software_versions={name:version(name) for name in ('numpy','scikit-learn','mlx','mlx-lm')})
    for lang in ('ro','uk'):
        train = [i for i,r in enumerate(rows) if r['row']['language']==lang and r['partition']=='train']
        dev = [i for i,r in enumerate(rows) if r['row']['language']==lang and r['partition']=='dev']
        y = np.array([target(rows[i]['row']) for i in train])
        truth = [target(rows[i]['row']) for i in dev]
        x = normalize(features[train])
        choices = []
        best = None
        for c in protocol['candidates_C']:
            with warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                fitted = LogisticRegression(C=c, class_weight='balanced', max_iter=2000, tol=1e-7, random_state=1141).fit(x,y)
            head = dict(weight=fitted.coef_[0].tolist(), intercept=float(fitted.intercept_[0]))
            probabilities = scores(features[dev],head)
            reference = fitted.predict_proba(normalize(features[dev]))[:,1]
            export_error = float(np.max(np.abs(reference-probabilities)))
            if export_error > 1e-10:
                raise ValueError('Portable readout differs from fitted classifier')
            population = [dict(gold=int(g),score=float(s),group=rows[i]['group']) for i,g,s in zip(dev,truth,probabilities)]
            selected = frontier(population)
            entry = dict(C=c, frontier=selected, export_score_error=export_error,
                training_rows=len(train), development_rows=len(dev), iterations=fitted.n_iter_.tolist())
            choices.append(entry)
            rank = tuple(selected['rank'])+(c,)
            if best is None or rank < best:
                best = rank
                report['selected'][lang] = entry
                head.update(format='roguard-gemma4-linear-readout-experiment-v1',language=lang,
                    normalization='l2-times-sqrt-width',C=c,threshold=selected['threshold'],
                    base_manifest_sha256=digest(study/'base-manifest.json'),
                    source_freeze_sha256=digest(study/'source-freeze.json'))
                (output/(lang+'-readout.json')).write_text(json.dumps(head,indent=2)+'\n')
                predictions = [dict(case_id=rows[i]['row']['case_id'],language=lang,group=rows[i]['group'],
                    gold=int(g),score=float(s),predicted=int(s>=selected['threshold'])) for i,g,s in zip(dev,truth,probabilities)]
                (output/(lang+'-dev-predictions.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in predictions))
            print('FIT',lang,json.dumps(entry),flush=True)
        report['candidates'][lang] = choices
    report.update(complete=True, finished_utc=datetime.now(timezone.utc).isoformat(),
        seconds_this_invocation=time.monotonic()-started,
        development_passes=all(r['frontier']['counts_pass'] for r in report['selected'].values()))
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('FINISHED',json.dumps({k:v for k,v in report.items() if k!='chunks'}),flush=True)


if __name__ == '__main__':
    main()
