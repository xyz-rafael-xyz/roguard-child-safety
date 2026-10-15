"""Replay frozen focus predictions through the guarded production workflow."""
import gzip,hashlib,json
from pathlib import Path
from roguard.evidence import GuardedEvidence,analyze_evidence,disposition
from training.evidence_v8 import load

class Replay:
    backend='joint'
    def __init__(self,p,identity):self.p=p;self.identity=identity
    def predict(self,texts,language):return [self.p for _ in texts]

def main():
 root=Path('research/v8-evidence');out=root/'guarded-controlled-regression.json'
 if out.exists():raise ValueError('Refusing overwrite')
 rows=load(root/'data','test')
 def read(p):return [json.loads(l) for l in gzip.decompress(p.read_bytes()).decode().splitlines()]
 primary=read(root/'continued-regression/predictions.jsonl.gz');baseline=read(root/'v7-baseline/predictions.jsonl.gz');records=[]
 for row,p,b in zip(rows,primary,baseline):
  assert row['case_id']==p['case_id']==b['case_id']
  model=GuardedEvidence(Replay(p['predictions']['semantic'],'semantic-05'),Replay(b['prediction'],'v7-joint'))
  focus=analyze_evidence({'language':row['language'],'turns':[row['turns'][row['focus']]]},model)['evidence'][-1]
  records.append({'language':row['language'],'gold':disposition({k:{'label':v} for k,v in row['attributes'].items()})[0],'review':focus['proposed_action']!='CONTINUE','baseline_review':b['review']})
 results={}
 for language in ['ro','uk']:
  results[language]={}
  for label,name in [('CONTINUE','benign'),('SUPPORT_AND_REVIEW','concern')]:
   rs=[r for r in records if r['language']==language and r['gold']==label]
   results[language][name]=len(rs);results[language][name+'_review']=sum(r['review'] for r in rs)
  results[language]['released_reviews_lost']=sum(r['baseline_review'] and not r['review'] for r in records if r['language']==language)
 result={'kind':'saved_prediction_production_policy_replay','fresh_evaluation':False,'runtime_sha256':hashlib.sha256(Path('src/roguard/evidence.py').read_bytes()).hexdigest(),'languages':results}
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
