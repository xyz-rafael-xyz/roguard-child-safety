"""Replay saved predictions after the uncertainty bug fix; never a fresh test."""
import gzip,hashlib,json
from pathlib import Path
from roguard.evidence import EvidenceCascade,disposition
from training.evidence_v8 import load
from eval.evidence_v8 import evaluate

class Replay:
    identity='saved-prediction'
    def __init__(self,p):self.p=p
    def predict(self,texts,language):return [self.p for _ in texts]

def main():
 root=Path('research/v8-evidence');path=root/'uncertainty-policy-replay.json'
 if path.exists():raise ValueError('Refusing overwrite')
 rows=load(root/'data','test');out={'kind':'posthoc_policy_regression_replay','runtime_sha256':hashlib.sha256(Path('src/roguard/evidence.py').read_bytes()).hexdigest(),'controlled':{},'prose':{}}
 for folder in ['final-test','mixed-regression','continued-regression']:
  records=[json.loads(l) for l in gzip.decompress((root/folder/'predictions.jsonl.gz').read_bytes()).decode().splitlines()]
  for row,record in zip(rows,records):assert row['case_id']==record['case_id']
  results={name:evaluate(rows,[r['predictions'][name] for r in records]) for name in ['character','semantic']}
  merged=[EvidenceCascade(Replay(r['predictions']['character']),Replay(r['predictions']['semantic'])).predict(['saved'],row['language'])[0] for row,r in zip(rows,records)]
  results['cascade']=evaluate(rows,merged);out['controlled'][folder]=results
 for filename in ['prose-results.json','prose-regression-04.json','prospective-results-04.json','confirmation-results-05.json']:
  original=json.loads((root/filename).read_text());summary={}
  for model in ['character','semantic','cascade','v7_joint']:
   summary[model]={}
   for language in ['ro','uk']:
    cases=[c for c in original['cases'] if c['language']==language]
    def reviewed(c):
     if model=='v7_joint':return c['outputs'][model]['review']
     # Prose cascade retains original raw inputs in per-head provenance; replay
     # direct semantic and character routing only, and explicitly omit cascade.
     return disposition(c['outputs'][model]['focus']['attributes'])[0]!='CONTINUE'
    if model=='cascade':continue
    summary[model][language]={}
    for label,name in [('CONTINUE','benign'),('REVIEW','concern')]:
     selected=[c for c in cases if c['expected']==label]
     summary[model][language][name]=len(selected);summary[model][language][name+'_review']=sum(reviewed(c) for c in selected)
  out['prose'][filename]=summary
 path.write_text(json.dumps(out,indent=2)+'\n')
 for name,results in out['controlled'].items():
  print(name,{lang:{k:results['semantic'][lang][k] for k in ['benign_review','concern_review']} for lang in ['ro','uk']})
 print(json.dumps(out['prose']['confirmation-results-05.json']['semantic']))
if __name__=='__main__':main()
