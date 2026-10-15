"""Frozen held-out evaluation. No fitting; retain per-case predictions and hashes."""
import argparse,gzip,hashlib,json,math,time
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from training.evidence_v8 import load
from training.fit_evidence_v8 import semantic_features
from roguard.context import render
from roguard.evidence import ONTOLOGY,disposition


def probabilities(spec,matrix):
    results=[]
    logits={k:np.asarray(matrix@np.array(h['weights']).T)+np.array(h['intercept']) for k,h in spec['heads'].items()}
    for k,z in logits.items():
        if z.shape[1]==1:z=np.concatenate([np.zeros_like(z),z],axis=1)
        z=z-z.max(axis=1,keepdims=True);p=np.exp(z);logits[k]=p/p.sum(axis=1,keepdims=True)
    return logits


def predictions(spec,prob,rows):
    out=[]
    for i,r in enumerate(rows):
        attrs={}
        for task,h in spec['heads'].items():
            p=prob[task][i];labels=[c for c,v,q in zip(h['classes'],p,h['quantiles'][r['language']]) if 1-v<=q+1e-12]
            winner=h['classes'][int(p.argmax())]
            attrs[task]={'label':labels[0] if len(labels)==1 and labels[0]==winner else None,'prediction_set':labels,'argmax':winner}
        out.append({'supported':True,'reason':'MODEL_PROPOSAL','attributes':attrs})
    return out


def evaluate(rows,preds):
    result={}
    for lang in ('ro','uk'):
        selected=[(r,p) for r,p in zip(rows,preds) if r['language']==lang];n=len(selected)
        tasks={}
        for task in ONTOLOGY:
            chosen=[(r,p) for r,p in selected if p['attributes'][task]['label'] is not None]
            tasks[task]={'argmax_macro_f1':float(f1_score([r['attributes'][task] for r,p in selected],[p['attributes'][task]['argmax'] for r,p in selected],labels=list(ONTOLOGY[task]),average='macro',zero_division=0)),
                         'decided':len(chosen),'decided_errors':sum(r['attributes'][task]!=p['attributes'][task]['label'] for r,p in chosen)}
        joint=sum(all(p['attributes'][k]['argmax']==r['attributes'][k] for k in ONTOLOGY) for r,p in selected)
        gold=lambda r:disposition({k:{'label':v} for k,v in r['attributes'].items()})[0]
        benign=[(r,p) for r,p in selected if gold(r)=='CONTINUE'];concern=[(r,p) for r,p in selected if gold(r)=='SUPPORT_AND_REVIEW']
        result[lang]={'n':n,'attributes':tasks,'joint_argmax_exact':joint,
          'benign':len(benign),'benign_review':sum(disposition(p['attributes'])[0]!='CONTINUE' for r,p in benign),
          'concern':len(concern),'concern_review':sum(disposition(p['attributes'])[0]!='CONTINUE' for r,p in concern),
          'action_exact':sum(disposition(p['attributes'])[0]==gold(r) for r,p in selected),
          'challenger_units':sum(p.get('challenger_used',False) for r,p in selected)}
    # Controlled assertion contrast, all other focus attributes/context held.
    groups={}
    for r,p in zip(rows,preds):
        group=(r['language'],r['pair_group'],r['case_id'].split('-')[-1])
        groups.setdefault(group,[]).append((r,p))
    pairs=[]
    for items in groups.values():
        affirmed=next(((r,p) for r,p in items if r['attributes']['assertion']=='AFFIRMED'),None)
        negated=next(((r,p) for r,p in items if r['attributes']['assertion']=='NEGATED'),None)
        if affirmed and negated:pairs.append((affirmed,negated))
    result['assertion_pairs']={'pairs':len(pairs),'both_argmax_correct':sum(a[1]['attributes']['assertion']['argmax']=='AFFIRMED' and b[1]['attributes']['assertion']['argmax']=='NEGATED' for a,b in pairs)}
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('research/v8-evidence/data'));p.add_argument('--split',choices=['dev','test'],default='test');p.add_argument('--character',type=Path,required=True);p.add_argument('--semantic',type=Path,required=True);p.add_argument('--base',required=True);p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);rows=load(a.data,a.split);texts=[r['turns'][r['focus']]['text'] for r in rows];allpred={};summary={'synthetic_only':True,'split':a.split,'models':{},'results':{}}
    from roguard.semantic import SemanticEncoder
    import torch
    torch.set_num_threads(4)
    for backend,path in [('character',a.character),('semantic',a.semantic)]:
        start=time.monotonic();spec=json.loads(gzip.decompress(path.read_bytes()));summary['models'][backend]=hashlib.sha256(path.read_bytes()).hexdigest()
        if backend=='character':
            vec=TfidfVectorizer(analyzer='char',ngram_range=(2,5),sublinear_tf=True,vocabulary={n:i for i,(n,v) in enumerate(spec['features'])});vec.idf_=np.array([v for n,v in spec['features']]);matrix=vec.transform(texts)
        else:
            from roguard.evidence import EvidenceModel
            encoder=EvidenceModel('semantic',artifact=path,base_path=a.base,device=a.device).encoder
            matrix=semantic_features(texts,encoder,a.output/'focus-features.npz')
            if spec.get('representation')=='context_plus_focus':
                full=semantic_features([render(r['turns'][:r['focus']+1]) for r in rows],encoder,a.output/'context-features.npz');matrix=np.concatenate([full,matrix],axis=1)
        preds=predictions(spec,probabilities(spec,matrix),rows);allpred[backend]=preds;summary['results'][backend]=evaluate(rows,preds);summary['results'][backend]['batch_seconds']=time.monotonic()-start
    # Invoke the exact production merge using replay models, not an evaluator copy.
    from roguard.evidence import EvidenceCascade
    class Replay:
        identity='evaluation-replay'
        def __init__(self,p):self.p=p
        def predict(self,texts,language):return [self.p for _ in texts]
    merged=[EvidenceCascade(Replay(a),Replay(b)).predict(['replay'],r['language'])[0] for r,a,b in zip(rows,allpred['character'],allpred['semantic'])]
    allpred['cascade']=merged;summary['results']['cascade']=evaluate(rows,merged)
    records=[{'case_id':r['case_id'],'predictions':{k:v[i] for k,v in allpred.items()}} for i,r in enumerate(rows)]
    (a.output/'predictions.jsonl.gz').write_bytes(gzip.compress(('\n'.join(json.dumps(r,ensure_ascii=False) for r in records)+'\n').encode(),mtime=0))
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
