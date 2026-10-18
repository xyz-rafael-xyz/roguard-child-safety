"""Fit small interpretation heads, freezing selection before calibration."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path

from training.native_fields_v9 import FIELDS,SIZES,targets,network,probabilities,policy_mass
from training.review_fusion_v9 import threshold,metrics,strata
from roguard.evidence import disposition
from roguard.assurance import policy_digest


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('fit','audit'))
    p.add_argument('--language',choices=('ro','uk'),required=True);a=p.parse_args()
    import numpy as np
    import torch
    torch.set_num_threads(4)
    root=Path('research/v9-stabilization');study=root/'native-fields-43'/a.language
    parent=root/('native-full-22-'+a.language)
    source_hashes={name:sha(Path('training')/name) for name in
        ('native_fields_v9.py','train_native_fields_v9.py','extract_native_fields_v9.py')}
    def load(split):
        path=parent/(split+'.jsonl.gz');meta=json.loads((study/(split+'-features.json')).read_text())
        fp=study/(split+'-features.npz')
        if meta['source_sha256']!=sha(path) or meta['features_sha256']!=sha(fp):raise ValueError('Feature evidence changed')
        rows=[json.loads(l) for l in gzip.decompress(path.read_bytes()).splitlines()]
        with np.load(fp,allow_pickle=False) as arrays:
            if arrays['case_ids'].tolist()!=[r['case_id'] for r in rows]:raise ValueError('Feature identities differ')
            x=arrays['features'].copy()
        y=np.array(targets(rows),dtype=np.int64)
        binary=np.array([int(disposition({f:{'label':v} for f,v in r['attributes'].items()})[0]!='CONTINUE') for r in rows])
        return rows,x,y,binary
    def infer(model,x):
        model.eval();outputs=[]
        with torch.inference_mode():
            for offset in range(0,len(x),256):outputs.append(model(torch.from_numpy(x[offset:offset+256])))
        logits=torch.cat(outputs)
        p=probabilities(torch,logits)
        return p,policy_mass(p)
    def predictions(path,rows,binary,p,scores,limit):
        with path.open('x') as stream:
            for i,row in enumerate(rows):
                stream.write(json.dumps(dict(case_id=row['case_id'],language=a.language,gold=int(binary[i]),
                    predicted=int(scores[i]>=limit),score=float(scores[i]),
                    field_probabilities={field:values[i].tolist() for field,values in zip(FIELDS,p)}))+'\n')
    if a.stage=='fit':
        rows,x,y,binary=load('train');dev,dx,dy,db=load('dev')
        mean=x.mean(axis=0);scale=x.std(axis=0);scale[scale<1e-6]=1
        x=np.ascontiguousarray((x-mean)/scale,dtype=np.float32)
        dx=np.ascontiguousarray((dx-mean)/scale,dtype=np.float32)
        groups=[(strata(r),int(v)) for r,v in zip(rows,binary)];frequencies=Counter(groups)
        if len(frequencies)!=4:raise ValueError('Training needs all four source/class groups')
        weights=torch.tensor([1/frequencies[g] for g in groups],dtype=torch.float64)
        xt=torch.from_numpy(x);yt=torch.from_numpy(y);trials=[];best=None
        for hidden in (0,256):
            torch.manual_seed(943+hidden)
            model=network(torch,x.shape[1],hidden)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
            for epoch in range(1,21):
                model.train();indices=torch.multinomial(weights,len(rows),replacement=True)
                for start in range(0,len(indices),256):
                    ix=indices[start:start+256];optimizer.zero_grad(set_to_none=True)
                    logits=torch.split(model(xt[ix]),SIZES,dim=-1)
                    loss=sum(torch.nn.functional.cross_entropy(v,yt[ix,i]) for i,v in enumerate(logits))/len(FIELDS)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite head training loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
                if epoch%5:continue
                p,scores=infer(model,dx);limit=threshold(dev,db,scores);counts=metrics(dev,db,scores,limit)
                ce=float(np.mean([-np.log(np.maximum(v[np.arange(len(dev)),dy[:,i]],1e-30)).mean() for i,v in enumerate(p)]))
                rank=[int(not all(c['passes'] for c in counts.values())),max(c['false_review_rate'] for c in counts.values()),
                      sum(c['false_review_rate'] for c in counts.values()),ce,hidden,epoch]
                entry=dict(hidden=hidden,epoch=epoch,threshold=float(limit),metrics=counts,unweighted_field_loss=ce,rank=rank)
                trials.append(entry);print(json.dumps(entry),flush=True)
                if best is None or rank<best['rank']:
                    best=entry
                    spec=dict(format='roguard-native-fields-experiment-v1',language=a.language,hidden=hidden,dimensions=x.shape[1],
                        native_artifact_sha256=sha(parent/'review.json.gz'),mean=mean.tolist(),scale=scale.tolist(),
                        weights={k:v.detach().tolist() for k,v in model.state_dict().items()},threshold=float(limit),
                        fields=FIELDS,sizes=SIZES,distribution='independent categorical fields; uncalibrated',policy_digest=policy_digest())
                    (study/'selected-heads.json.gz').write_bytes(gzip.compress(json.dumps(spec,separators=(',',':')).encode(),mtime=0))
        record=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=source_hashes,trials=trials,selected=best,
            artifact_sha256=sha(study/'selected-heads.json.gz'),native_artifact_sha256=sha(parent/'review.json.gz'),
            data_hashes={s:sha(parent/(s+'.jsonl.gz')) for s in ('train','dev')},
            feature_hashes={s:sha(study/(s+'-features.npz')) for s in ('train','dev')},
            calibration_used_for_head_selection=False,reserved_probes_used=False,stable_release_ready=False)
        write(study/'selection.json',record)
    else:
        record=json.loads((study/'selection.json').read_text())
        if record['source_sha256']!=source_hashes or record['artifact_sha256']!=sha(study/'selected-heads.json.gz'):
            raise ValueError('Selected head recipe or weights changed')
    # Reload the actual exported heads for both development replay and calibration.
    spec=json.loads(gzip.decompress((study/'selected-heads.json.gz').read_bytes()))
    model=network(torch,spec['dimensions'],spec['hidden'])
    model.load_state_dict({k:torch.tensor(v,dtype=torch.float32) for k,v in spec['weights'].items()},strict=True)
    split='dev' if a.stage=='fit' else 'cal';rows,x,y,binary=load(split)
    x=np.ascontiguousarray((x-np.asarray(spec['mean'],dtype=np.float32))/np.asarray(spec['scale'],dtype=np.float32),dtype=np.float32)
    p,scores=infer(model,x);counts=metrics(rows,binary,scores,spec['threshold'])
    if a.stage=='fit' and counts!=record['selected']['metrics']:raise ValueError('Exported head decisions changed')
    predictions(study/(split+'-predictions.jsonl'),rows,binary,p,scores,spec['threshold'])
    if a.stage=='audit':
        write(study/'audit.json',dict(selection_sha256=sha(study/'selection.json'),calibration=counts,
            counts_pass=all(c['passes'] for c in counts.values()) and all(c['passes'] for c in record['selected']['metrics'].values()),
            synthetic_only=True,stable_release_ready=False,reserved_probes_used=False,
            remaining=['Actual installed runtime','Reserved actual-runtime evaluation','Product integration and release checks']))


if __name__=='__main__':main()
