"""Fit context heads without opening final tests; retain every development candidate."""
import argparse, gzip, hashlib, json, time
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from training.context_v7 import generate, render, digest, LABELS
from roguard.semantic import SemanticEncoder

def unique(rows):
    seen=set();out=[]
    for r in rows:
        key=(r['language'],render(r['turns']))
        if key not in seen:seen.add(key);out.append(r)
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(exist_ok=False)
    from training.context_extra_v7 import augment
    rows={s:unique(augment(generate(s)) if s=='train' else generate(s)) for s in ('train','dev','cal')}
    record={'synthetic_only':True,'unique_rows':{s:len(r) for s,r in rows.items()},'digests':{s:digest(r) for s,r in rows.items()},'candidates':[]}
    texts={s:[render(r['turns']) for r in rs] for s,rs in rows.items()}
    encoder=SemanticEncoder(a.base,a.device);features={}
    for split in rows:
        start=time.monotonic();features[split]=np.concatenate([encoder.encode(texts[split],batch_size=16),encoder.encode([render([r['turns'][-1]]) for r in rows[split]],batch_size=16)],axis=1);print(split,len(rows[split]),'seconds',time.monotonic()-start,flush=True)
    np.savez_compressed(a.output/'features.npz',**features)
    for backend in ('character','semantic'):
        vectorizer=TfidfVectorizer(analyzer='char',ngram_range=(2,5),min_df=2,max_features=40000,sublinear_tf=True)
        matrix={s:vectorizer.fit_transform(texts[s]) if s=='train' else vectorizer.transform(texts[s]) for s in rows} if backend=='character' else features
        for c in (.1,1,10):
            heads={};scores={};metrics={}
            for task in ('frame','signal'):
                model=LogisticRegression(C=c,max_iter=1500,class_weight='balanced',random_state=710)
                model.fit(matrix['train'],[r[task] for r in rows['train']]);heads[task]=model
                pred=model.predict(matrix['dev']);metrics[task]={l:float(f1_score([r[task] for r in rows['dev'] if r['language']==l],[v for r,v in zip(rows['dev'],pred) if r['language']==l],average='macro')) for l in ('ro','uk')}
            key=(min(metrics['frame'].values()),sum(metrics['frame'].values()),min(metrics['signal'].values()))
            entry=dict(backend=backend,C=c,development=metrics);record['candidates'].append(entry);print(json.dumps(entry),flush=True)
            if c==.1 or key>best:
                best=key;chosen=heads;chosen_c=c
        payload={'format':'roguard-context-v1','backend':backend,'heads':{},'C':chosen_c,'synthetic_only':True,'representation':'context_plus_focus' if backend=='semantic' else 'context'}
        for task,model in chosen.items():
            classes=[int(v) if task=='signal' else str(v) for v in model.classes_]
            head=dict(classes=classes,weights=model.coef_.tolist(),intercept=model.intercept_.tolist(),quantiles={})
            probabilities=model.predict_proba(matrix['cal'])
            # Per-language pooled split conformal quantile, alpha=0.1 (not class-conditional).
            for l in ('ro','uk'):
                residuals=sorted(1-float(p[classes.index(r[task])]) for r,p in zip(rows['cal'],probabilities) if r['language']==l)
                rank=int(np.ceil((len(residuals)+1)*.9))
                q=residuals[rank-1] if rank<=len(residuals) else 1.0
                head['quantiles'][l]=[q]*len(classes)
            head['calibration_method']='language_conditional_pooled_classes'
            payload['heads'][task]=head
        if backend=='character':payload['features']=[[n,float(v)] for n,v in zip(vectorizer.get_feature_names_out(),vectorizer.idf_)];payload['ngram_range']=[2,5]
        data=gzip.compress(json.dumps(payload,ensure_ascii=False,separators=(',',':')).encode(),mtime=0)
        (a.output/(backend+'.json.gz')).write_bytes(data)
    (a.output/'development.json').write_text(json.dumps(record,indent=2)+'\n')
    (a.output/'freeze.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.output.iterdir() if p.suffix!='.npz'},indent=2)+'\n')
if __name__=='__main__':main()
