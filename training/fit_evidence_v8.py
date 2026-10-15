"""Fit six heads using train/dev/cal only; freeze before final evaluation."""
import argparse,gzip,hashlib,json,math,time
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from training.evidence_v8 import load
from roguard.evidence import ONTOLOGY


def metrics(rows,heads,matrix):
    out={}
    for task,model in heads.items():
        pred=model.predict(matrix)
        out[task]={lang:float(f1_score([r['attributes'][task] for r in rows if r['language']==lang],
            [v for r,v in zip(rows,pred) if r['language']==lang],labels=list(ONTOLOGY[task]),average='macro',zero_division=0)) for lang in ('ro','uk')}
    return out


def semantic_features(texts,encoder,path):
    unique=list(dict.fromkeys(texts));identity=hashlib.sha256(json.dumps(unique,ensure_ascii=False).encode()).hexdigest()
    if path.exists():
        cached=np.load(path)
        if str(cached['identity'])!=identity:raise ValueError('Feature cache differs')
        values=cached['values']
    else:
        start=time.monotonic();parts=[]
        for offset in range(0,len(unique),128):
            parts.append(encoder.encode(unique[offset:offset+128],batch_size=16))
            if offset%512==0:print('encoding',path.name,offset,len(unique),round(time.monotonic()-start),flush=True)
        values=np.concatenate(parts);np.savez_compressed(path,identity=identity,values=values)
    lookup={t:i for i,t in enumerate(unique)}
    return values[[lookup[t] for t in texts]]


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('research/v8-evidence/data'));p.add_argument('--output',type=Path,required=True);p.add_argument('--backend',choices=['character','semantic'],required=True);p.add_argument('--context',action='store_true');p.add_argument('--augment',type=Path);p.add_argument('--base');p.add_argument('--device',default='cpu');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);rows={s:load(a.data,s) for s in ('train','dev','cal')}
    if a.augment:
        rows['train'] += [json.loads(line) for line in gzip.decompress(a.augment.read_bytes()).decode().splitlines()]
    texts={s:[r['turns'][r['focus']]['text'] for r in rs] for s,rs in rows.items()}
    record={'backend':a.backend,'synthetic_only':True,'data_freeze_sha256':hashlib.sha256((a.data/'data-freeze.json').read_bytes()).hexdigest(),'candidates':[]}
    if a.augment:record['augmentation_sha256']=hashlib.sha256(a.augment.read_bytes()).hexdigest()
    if a.backend=='character':
        vec=TfidfVectorizer(analyzer='char',ngram_range=(2,5),min_df=2,max_features=14000,sublinear_tf=True)
        matrix={'train':vec.fit_transform(texts['train'])};matrix.update({s:vec.transform(texts[s]) for s in ('dev','cal')})
    else:
        from roguard.semantic import SemanticEncoder
        import torch
        torch.set_num_threads(4);encoder=SemanticEncoder(a.base,a.device)
        matrix={s:semantic_features(texts[s],encoder,a.output/(s+'-features.npz')) for s in rows}
        if a.context:
            from roguard.context import render
            for split in rows:
                full=semantic_features([render(r['turns'][:r['focus']+1]) for r in rows[split]],encoder,a.output/(split+'-context-features.npz'))
                matrix[split]=np.concatenate([full,matrix[split]],axis=1)
    best=None
    for c in (.1,1.,10.):
        heads={};start=time.monotonic()
        for task in ONTOLOGY:
            model=LogisticRegression(C=c,max_iter=500,class_weight='balanced',random_state=810)
            model.fit(matrix['train'],[r['attributes'][task] for r in rows['train']]);heads[task]=model
        scores=metrics(rows['dev'],heads,matrix['dev']);rank=(min(v for m in scores.values() for v in m.values()),sum(v for m in scores.values() for v in m.values()))
        entry={'C':c,'development_macro_f1':scores,'seconds':time.monotonic()-start};record['candidates'].append(entry);print(json.dumps(entry),flush=True)
        if best is None or rank>best:best=rank;chosen=heads;chosen_c=c
        (a.output/'development.json').write_text(json.dumps(record,indent=2)+'\n')
    payload={'format':'roguard-evidence-v1','backend':a.backend,'heads':{},'C':chosen_c,'synthetic_only':True,'training_input':'bounded_context_and_paragraph' if a.context else 'independent_paragraph','representation':'context_plus_focus' if a.context else 'paragraph','data_freeze_sha256':record['data_freeze_sha256']}
    for task,model in chosen.items():
        classes=model.classes_.tolist();prob=model.predict_proba(matrix['cal']);q={}
        for lang in ('ro','uk'):
            q[lang]=[]
            for i,label in enumerate(classes):
                residuals=sorted(1-float(p[i]) for r,p in zip(rows['cal'],prob) if r['language']==lang and r['attributes'][task]==label)
                rank=math.ceil((len(residuals)+1)*.95);q[lang].append(residuals[rank-1] if rank<=len(residuals) else 1.)
        payload['heads'][task]={'classes':classes,'weights':model.coef_.tolist(),'intercept':model.intercept_.tolist(),'quantiles':q,'calibration':'language_task_class_conditional_alpha_0.05'}
    if a.backend=='character':payload['features']=list(zip(vec.get_feature_names_out(),vec.idf_.tolist()))
    artifact=a.output/(a.backend+'.json.gz');artifact.write_bytes(gzip.compress(json.dumps(payload,separators=(',',':')).encode(),mtime=0))
    record['selected_C']=chosen_c;record['artifact_sha256']=hashlib.sha256(artifact.read_bytes()).hexdigest()
    (a.output/'development.json').write_text(json.dumps(record,indent=2)+'\n')
    print('FROZEN',record['artifact_sha256'],flush=True)

if __name__=='__main__':main()
