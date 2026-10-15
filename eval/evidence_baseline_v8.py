"""Compare v7 focus-turn routing on the frozen v8 test, using exact policy code."""
import argparse,gzip,hashlib,json,time
from pathlib import Path
import numpy as np
from training.evidence_v8 import load
from training.fit_evidence_v8 import semantic_features
from roguard.context import ContextModel,analyze_context,render
from roguard.evidence import disposition
from eval.evidence_v8 import probabilities

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 import torch
 torch.set_num_threads(4);model=ContextModel('joint',base_path=a.base,device=a.device);rows=load('research/v8-evidence/data','test')
 full=[render(r['turns']) for r in rows];focus=[render([r['turns'][-1]]) for r in rows]
 features=np.concatenate([semantic_features(full,model.encoder,a.output/'full-features.npz'),semantic_features(focus,model.encoder,a.output/'focus-features.npz')],axis=1)
 prob=probabilities(model.spec,features);records=[]
 class Replay:
  backend='joint';identity=model.identity
  def __init__(self,p):self.p=p
  def predict(self,texts,language):return [self.p for _ in texts]
 for i,r in enumerate(rows):
  heads={}
  for task,h in model.spec['heads'].items():
   ps=prob[task][i];winner=h['classes'][int(ps.argmax())];labels=[c for c,v,q in zip(h['classes'],ps,h['quantiles'][r['language']]) if 1-v<=q+1e-12]
   heads[task]={'label':labels[0] if len(labels)==1 and labels[0]==winner else None,'prediction_set':labels,'argmax':winner}
  prediction={'supported':True,'heads':heads}
  # Only the final evidence entry is scored. Earlier replay entries are ignored.
  result=analyze_context({k:r[k] for k in ('language','turns')},Replay(prediction))['evidence'][-1]
  records.append({'case_id':r['case_id'],'language':r['language'],'gold_action':disposition({k:{'label':v} for k,v in r['attributes'].items()})[0],'review':result['review_recommended'],'prediction':prediction})
 summary={'model_sha256':model.identity,'synthetic_only':True,'focus_turn_routing_only':True,'languages':{}}
 for lang in ('ro','uk'):
  rs=[r for r in records if r['language']==lang];benign=[r for r in rs if r['gold_action']=='CONTINUE'];concern=[r for r in rs if r['gold_action']=='SUPPORT_AND_REVIEW']
  summary['languages'][lang]={'benign':len(benign),'benign_review':sum(r['review'] for r in benign),'concern':len(concern),'concern_review':sum(r['review'] for r in concern)}
 (a.output/'predictions.jsonl.gz').write_bytes(gzip.compress(('\n'.join(json.dumps(r,ensure_ascii=False) for r in records)+'\n').encode(),mtime=0));(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
