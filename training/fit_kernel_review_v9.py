"""CPU-only nonlinear-head control using verified existing feature caches.

Cache input hashes and the originating encoder record are checked. Any candidate
still requires independent encoder replay before promotion; cached evidence is
not a substitute for integrated inference.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from roguard.context import render
from training.review_data_v8 import input_key
from training.prepare_instruction_sft_v8 import target
from training.fit_review_v8 import group_metrics


def read(path):
    return [json.loads(l) for l in gzip.decompress(path.read_bytes()).decode().splitlines()]


def cached_features(directory, split, rows):
    parts=[]
    for field,texts in [('context',[render(r['turns'][:r['focus']+1]) for r in rows]),
                        ('focus',[r['turns'][r['focus']]['text'] for r in rows])]:
        unique=list(dict.fromkeys(texts));identity=hashlib.sha256(json.dumps(unique,ensure_ascii=False).encode()).hexdigest()
        with np.load(directory/(split+'-'+field+'-features.npz'),allow_pickle=False) as data:
            if str(data['identity'])!=identity or data['values'].shape!=(len(unique),768):raise ValueError('Cache/input mismatch')
            lookup={t:i for i,t in enumerate(unique)};parts.append(data['values'][[lookup[t] for t in texts]])
    return np.concatenate(parts,axis=1).astype(np.float64)


def cutoff(rows, truth, scores):
    thresholds={}
    for lang in ('ro','uk'):
        limits=[]
        for native in (False,True):
            values=sorted(float(s) for r,y,s in zip(rows,truth,scores) if r['language']==lang and y and (r.get('source')=='v7_training_frame_projection')==native)
            if not values:raise ValueError('Missing development group')
            limits.append(values[int(.01*len(values))])
        thresholds[lang]=min(limits)*(1-1e-4)
    return thresholds


def exported_scores(spec, matrix):
    support=np.asarray(spec['support'],dtype=np.float64);dual=np.asarray(spec['dual'],dtype=np.float64)
    result=[];norm=(support*support).sum(1)
    for offset in range(0,len(matrix),64):
        x=matrix[offset:offset+64];distance=np.maximum(0,(x*x).sum(1)[:,None]+norm[None,:]-2*x@support.T)
        logits=np.exp(-spec['gamma']*distance)@dual+spec['intercept']
        result.extend(1/(1+np.exp(-np.clip(logits,-700,700))))
    return np.asarray(result)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from sklearn.svm import SVC
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=2)
    root=Path('research/v8-stabilization');source=root/'review-01';a.output.mkdir(exist_ok=False)
    encoder_path=Path('research/v8-evidence/semantic-05/semantic.json.gz')
    identity=hashlib.sha256(encoder_path.read_bytes()).hexdigest()
    if json.loads((source/'development.json').read_text())['encoder_sha256']!=identity:raise ValueError('Encoder record differs')
    rows={};matrix={};truth={}
    for split in ('train','dev','cal'):
        original=read(source/(split+'.jsonl.gz'));features=cached_features(source,split,original)
        by_key={input_key(r):i for i,r in enumerate(original)}
        if split=='train':indices=list(by_key.values());rows[split]=[original[i] for i in indices]
        else:
            rows[split]=read(root/'review-09-ro'/(split+'.jsonl.gz'));indices=[by_key[input_key(r)] for r in rows[split]]
        matrix[split]=features[indices];truth[split]=np.array([target(r) for r in rows[split]])
        (a.output/(split+'.jsonl.gz')).write_bytes(gzip.compress((''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows[split])).encode(),mtime=0))
    train_keys=set(map(input_key,rows['train']))
    if train_keys & set(map(input_key,rows['dev']+rows['cal'])):raise ValueError('Training overlap')
    (a.output/'training-history.jsonl.gz').write_bytes((a.output/'train.jsonl.gz').read_bytes())
    record=dict(synthetic_only=True,stable_release_ready=False,reserved_probes_used=False,training_rows=len(rows['train']),
                encoder_sha256=identity,requires_fresh_encoder_replay=True,selection_partition='development',candidates=[])
    best=None
    for c,gamma in ((1.,.01),(10.,.01),(1.,.05),(10.,.05)):
        start=time.monotonic();model=SVC(C=c,gamma=gamma,class_weight='balanced',cache_size=128).fit(matrix['train'],truth['train'])
        scores=1/(1+np.exp(-np.clip(model.decision_function(matrix['dev']),-700,700)))
        thresholds=cutoff(rows['dev'],truth['dev'],scores);metrics=group_metrics(rows['dev'],truth['dev'],scores,thresholds)
        entry=dict(C=c,gamma=gamma,seconds=time.monotonic()-start,development=metrics,thresholds=thresholds,support_vectors=len(model.support_vectors_))
        record['candidates'].append(entry);print(json.dumps(entry),flush=True)
        rank=(max(v['false_review_rate'] for v in metrics.values()),sum(v['false_review_rate'] for v in metrics.values()))
        if best is None or rank<best:
            best=rank;selected=model;record['selected']=entry
        (a.output/'development.json').write_text(json.dumps(record,indent=2)+'\n')
    spec=dict(format='roguard-kernel-review-experiment-v1',encoder_sha256=identity,
              support=selected.support_vectors_.tolist(),dual=selected.dual_coef_[0].tolist(),
              intercept=float(selected.intercept_[0]),gamma=selected._gamma,thresholds=record['selected']['thresholds'])
    raw=gzip.compress(json.dumps(spec,separators=(',',':')).encode(),mtime=0);(a.output/'review.json.gz').write_bytes(raw)
    spec=json.loads(gzip.decompress(raw));parity={}
    for split in ('dev','cal'):
        scores=exported_scores(spec,matrix[split]);reference=1/(1+np.exp(-np.clip(selected.decision_function(matrix[split]),-700,700)))
        error=float(np.max(np.abs(scores-reference)));parity[split]=error
        if error>1e-8:raise ValueError('Exported kernel does not reproduce scores')
        predictions=[dict(case_id=r['case_id'],language=r['language'],gold=int(y),score=float(s),predicted=int(s>=spec['thresholds'][r['language']])) for r,y,s in zip(rows[split],truth[split],scores)]
        (a.output/(split+'-predictions.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in predictions))
        if split=='cal':record['calibration']=group_metrics(rows[split],truth[split],scores,spec['thresholds'])
    from training.check_review_candidate_v8 import check_counts
    record['failures']=['dev/'+v for v in check_counts(record['selected']['development'])]+['cal/'+v for v in check_counts(record['calibration'])]
    record['artifact_sha256']=hashlib.sha256(raw).hexdigest();record['export_parity']=parity
    (a.output/'development.json').write_text(json.dumps(record,indent=2)+'\n');print('FINISHED',json.dumps(record['calibration']),flush=True)


if __name__=='__main__':main()
