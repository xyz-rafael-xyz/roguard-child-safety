"""Direct review head experiment: train/dev only; existing tests remain untouched."""
import argparse,gzip,hashlib,json,random
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from roguard.evidence import EvidenceModel,disposition
from roguard.context import render
from training.evidence_v8 import load
from training.adapt_evidence_v8 import natural
from training.fit_evidence_v8 import semantic_features


def labels(rows):return np.array([int(disposition({k:{'label':v} for k,v in r['attributes'].items()})[0]!='CONTINUE') for r in rows])


def group_metrics(rows,y,p,thresholds):
 out={}
 for language in ['ro','uk']:
  for kind in ['formal','conversational']:
   ix=np.array([i for i,r in enumerate(rows) if r['language']==language and ('conversational' if r.get('source')=='v7_training_frame_projection' else 'formal')==kind]);truth=y[ix];review=p[ix]>=thresholds[language]
   positive=truth==1;negative=truth==0
   out[language+'_'+kind]={'positive':int(positive.sum()),'negative':int(negative.sum()),'missed':int((~review & positive).sum()),'false_reviews':int((review & negative).sum()),'recall':float(review[positive].mean()),'false_review_rate':float(review[negative].mean())}
 return out


def main():
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
 import torch
 torch.set_num_threads(4);root=Path('research/v8-evidence');rng=random.Random(860)
 train=[json.loads(l) for l in gzip.decompress((root/'semantic-05/selected-training.jsonl.gz').read_bytes()).decode().splitlines()]
 dev_native=natural('dev');rng.shuffle(dev_native);cal_native=natural('cal');rng.shuffle(cal_native)
 rows={'train':train,'dev':load(root/'data','dev')+dev_native[:1000],'cal':load(root/'data','cal')+cal_native[:1000]}
 model=EvidenceModel('semantic',artifact=root/'semantic-05/semantic.json.gz',base_path=a.base,device=a.device)
 matrices={};targets={};record={'encoder_sha256':model.identity,'synthetic_only':True,'selection_partition':'development','candidates':[]}
 for split,rs in rows.items():
  raw='\n'.join(json.dumps(r,ensure_ascii=False) for r in rs)+'\n';(a.output/(split+'.jsonl.gz')).write_bytes(gzip.compress(raw.encode(),mtime=0))
  full=semantic_features([render(r['turns'][:r['focus']+1]) for r in rs],model.encoder,a.output/(split+'-context-features.npz'))
  focus=semantic_features([r['turns'][r['focus']]['text'] for r in rs],model.encoder,a.output/(split+'-focus-features.npz'))
  matrices[split]=np.concatenate([full,focus],axis=1);targets[split]=labels(rs)
 best=None
 for c in [.01,.1,1.,10.]:
  head=LogisticRegression(C=c,class_weight='balanced',max_iter=1000,random_state=860).fit(matrices['train'],targets['train']);probs=head.predict_proba(matrices['dev'])[:,1];thresholds={}
  for language in ['ro','uk']:
   limits=[]
   for kind in ['formal','conversational']:
    ix=[i for i,r in enumerate(rows['dev']) if r['language']==language and ('conversational' if r.get('source')=='v7_training_frame_projection' else 'formal')==kind and targets['dev'][i]==1]
    sorted_scores=np.sort(probs[ix]);limits.append(float(sorted_scores[int(.01*len(ix))]))
   thresholds[language]=min(limits)
  metrics=group_metrics(rows['dev'],targets['dev'],probs,thresholds);rank=(max(v['false_review_rate'] for v in metrics.values()),sum(v['false_review_rate'] for v in metrics.values()))
  entry={'C':c,'thresholds':thresholds,'development':metrics};record['candidates'].append(entry);print(json.dumps(entry),flush=True)
  if best is None or rank<best:best=rank;chosen=head;chosen_c=c;chosen_thresholds=thresholds
 spec={'format':'roguard-review-experiment-v1','encoder_sha256':model.identity,'weights':chosen.coef_[0].tolist(),'intercept':float(chosen.intercept_[0]),'thresholds':chosen_thresholds,'C':chosen_c}
 (a.output/'review-head.json').write_text(json.dumps(spec,separators=(',',':'))+'\n')
 record['selected_C']=chosen_c;record['calibration_audit']=group_metrics(rows['cal'],targets['cal'],chosen.predict_proba(matrices['cal'])[:,1],chosen_thresholds);record['artifact_sha256']=hashlib.sha256((a.output/'review-head.json').read_bytes()).hexdigest()
 (a.output/'development.json').write_text(json.dumps(record,indent=2)+'\n');print('FINISHED',json.dumps(record['calibration_audit']),flush=True)
if __name__=='__main__':main()
