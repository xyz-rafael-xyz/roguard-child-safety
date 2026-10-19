"""Separate prior-population extraction for the frozen Ukrainian scope candidate."""
import argparse
from datetime import datetime,timezone
import gzip,hashlib,json
from importlib.metadata import distribution,version
from pathlib import Path
import time
from training.gemma4_prompt_v9 import classifier_tokens
from training.gemma4_batch_features_v9 import batch_features


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args()
    root=Path('research/v9-stabilization');study=root/'gemma4-qualified-scope-148';parent=root/'gemma4-readout-141'
    for freeze in (study/'source-freeze.json',parent/'source-freeze.json'):
        for name,h in json.loads(freeze.read_text()).items():
            if digest(name)!=h:raise ValueError('Frozen source differs: '+name)
    base=json.loads((study/'base-manifest.json').read_text())
    if a.base.name!=base['revision']:raise ValueError('Wrong base revision')
    for name,entry in base['files'].items():
        if digest(a.base/name)!=entry['sha256']:raise ValueError('Base file differs: '+name)
    software=json.loads((study/'software.json').read_text())
    for name,v in software['versions'].items():
        if version(name)!=v:raise ValueError('Dependency differs: '+name)
    package=Path(distribution('mlx-lm').locate_file('mlx_lm'))
    for name,h in software['implementation_sha256'].items():
        if digest(package/name)!=h:raise ValueError('MLX implementation differs')
    import numpy as np
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    protocol=json.loads((study/'protocol.json').read_text())
    candidate=json.loads((root/'expanded-scope-gate-147/actual/report.json').read_text())
    if not candidate['complete'] or not candidate['languages']['uk']['selected']['frontier']['counts_pass']:
        raise ValueError('Development must qualify first')
    output=study/'actual';output.mkdir(exist_ok=False)
    started=time.monotonic();mx.set_memory_limit(10*1024**3);mx.set_cache_limit(256*1024**2)
    model,tokenizer=load(str(a.base),tokenizer_config={'trust_remote_code':False});model.set_dtype(mx.float32)
    ids=[tokenizer.encode(str(i),add_special_tokens=False) for i in (0,1)]
    if any(len(v)!=1 for v in ids):raise ValueError('Class labels differ')
    def read(path):return [json.loads(l) for l in gzip.decompress(path.read_bytes()).decode().splitlines()]
    rows=read(study/'inputs.jsonl.gz');old=read(parent/'inputs.jsonl.gz')
    references=[old[i] for i in protocol['reference_indices']]
    def encode(entries):
        result=[]
        for row in entries:
            encoded=classifier_tokens(tokenizer,row['row'],[v[0] for v in ids])
            h=hashlib.sha256(json.dumps(encoded,separators=(',',':')).encode()).hexdigest()
            if h!=row['token_sha256'] or len(encoded)!=row['token_count']:raise ValueError('Runtime prompt differs')
            result.append(encoded)
        return result
    encoded=encode(rows);reference_tokens=encode(references)
    common=0
    for column in zip(*(encoded+reference_tokens)):
        if len(set(column))!=1:break
        common+=1
    common=min(common,min(map(len,encoded+reference_tokens))-1)
    cache=make_prompt_cache(model)
    if common:mx.eval(model.language_model.model(mx.array([encoded[0][:common]]),cache=cache))
    original=json.loads((parent/'actual/report.json').read_text())
    if not original.get('complete'):raise ValueError('Incomplete reference representations')
    records={r['start']:r for r in original['chunks']};loaded={};reference_features=[]
    for i in protocol['reference_indices']:
        start=i//200*200
        if start not in loaded:
            record=records[start];path=parent/'actual'/f'chunk-{start:06d}-features.npz'
            identity=hashlib.sha256(json.dumps([r['token_sha256'] for r in old[start:record['stop']]]).encode()).hexdigest()
            if digest(path)!=record['feature_sha256'] or identity!=record['input_sha256'] or record['source_freeze_sha256']!=digest(parent/'source-freeze.json'):
                raise ValueError('Reference feature provenance differs')
            loaded[start]=np.load(path,allow_pickle=False)['features']
        reference_features.append(loaded[start][i-start])
    reference_features=np.stack(reference_features);del loaded
    def infer(values,size):
        result=[]
        for start in range(0,len(values),size):
            hidden=batch_features(model,[v[common:] for v in values[start:start+size]],cache).astype(mx.float32)
            mx.eval(hidden);result.extend(np.asarray(hidden).copy());del hidden;mx.clear_cache()
        result=np.stack(result)
        if not np.isfinite(result).all():raise ValueError('Nonfinite features')
        return result
    benchmarks={}
    for size in (4,1):
        tick=time.monotonic()
        try:
            actual=infer(reference_tokens,size)
            error=float(np.max(np.max(np.abs(actual-reference_features),axis=1)/np.maximum(1.,np.max(np.abs(reference_features),axis=1))))
            benchmarks[str(size)]=dict(seconds=time.monotonic()-tick,relative_error=error,passes=error<=1e-4)
        except Exception as exc:
            benchmarks[str(size)]=dict(seconds=time.monotonic()-tick,passes=False,error_type=type(exc).__name__)
            mx.clear_cache()
            if size==1:raise
    if not benchmarks['1']['passes']:raise ValueError('Canonical reference features differ')
    size=4 if benchmarks['4']['passes'] and benchmarks['4']['seconds']<benchmarks['1']['seconds'] else 1
    (output/'batching.json').write_text(json.dumps(dict(selected_batch_size=size,references=100,comparisons=benchmarks,prior_extraction_only=True),indent=2)+'\n')
    print('BATCHING',size,json.dumps(benchmarks),flush=True)
    chunks=[]
    for start in range(0,len(rows),protocol['chunk_size']):
        stop=min(start+protocol['chunk_size'],len(rows));features=infer(encoded[start:stop],size)
        path=output/f'chunk-{start:06d}-features.npz';np.savez_compressed(path,features=features)
        identity=hashlib.sha256(json.dumps([r['token_sha256'] for r in rows[start:stop]]).encode()).hexdigest()
        record=dict(start=start,stop=stop,input_sha256=identity,feature_sha256=digest(path),source_freeze_sha256=digest(study/'source-freeze.json'))
        (output/f'chunk-{start:06d}.json').write_text(json.dumps(record,indent=2)+'\n');chunks.append(record)
        print('FEATURES',stop,'/',len(rows),'seconds',round(time.monotonic()-started,2),flush=True)
    report=dict(complete=True,finished_utc=datetime.now(timezone.utc).isoformat(),rows=len(rows),feature_width=features.shape[1],
        source_freeze_sha256=digest(study/'source-freeze.json'),chunks=chunks,batch_size=size,benchmarks=benchmarks,
        peak_mlx_bytes=mx.get_peak_memory(),seconds=time.monotonic()-started,only_training_inferred=False,
        fresh_test_inferred=False,calibration_inferred=True,synthetic_only=True,stable_release_ready=False)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('FINISHED',json.dumps({k:v for k,v in report.items() if k!='chunks'}),flush=True)


if __name__=='__main__':main()
