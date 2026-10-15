"""Compare actual runtime evidence cascade and released v7 joint model on prose."""
import argparse,gzip,hashlib,json,time
from pathlib import Path
from roguard.evidence import EvidenceModel,EvidenceCascade,GuardedEvidence,analyze_evidence
from roguard.context import ContextModel,analyze_context

def main():
 p=argparse.ArgumentParser();p.add_argument('--probe',default='prose-probe');p.add_argument('--base',required=True);p.add_argument('--device',default='cpu');p.add_argument('--character',type=Path,required=True);p.add_argument('--semantic',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 root=Path('research/v8-evidence');path=root/(a.probe+'.jsonl');freeze=json.loads((root/(a.probe+'-freeze.json')).read_text())
 if hashlib.sha256(path.read_bytes()).hexdigest()!=freeze['sha256']:raise ValueError('Probe freeze differs')
 if a.output.exists():raise ValueError('Refusing overwrite')
 import torch
 torch.set_num_threads(4)
 primary=EvidenceModel('character',artifact=a.character);semantic=EvidenceModel('semantic',artifact=a.semantic,base_path=a.base,device=a.device)
 models={'character':primary,'semantic':semantic,'cascade':EvidenceCascade(primary,semantic),'v7_joint':ContextModel('joint',base_path=a.base,device=a.device)}
 models['guarded']=GuardedEvidence(semantic,models['v7_joint'])
 rows=[json.loads(l) for l in path.read_text().splitlines()];result={'synthetic_only':True,'probe_sha256':freeze['sha256'],'models':{},'cases':[]}
 for row in rows:
  payload={k:row[k] for k in ('language','turns')};outputs={}
  for name,model in models.items():
   start=time.monotonic();report=analyze_context(payload,model) if name=='v7_joint' else analyze_evidence(payload,model)
   focus=report['evidence'][-1];review=focus['review_recommended'] if name=='v7_joint' else focus['proposed_action']!='CONTINUE'
   outputs[name]={'review':review,'seconds':time.monotonic()-start,'focus':focus}
  result['cases'].append({'case_id':row['case_id'],'language':row['language'],'expected':row['expected_focus_routing'],'outputs':outputs})
 for name,model in models.items():
  result['models'][name]={'identity':model.identity,'languages':{}}
  for lang in ('ro','uk'):
   rs=[r for r in result['cases'] if r['language']==lang];benign=[r for r in rs if r['expected']=='CONTINUE'];concern=[r for r in rs if r['expected']=='REVIEW']
   result['models'][name]['languages'][lang]={'benign':len(benign),'benign_review':sum(r['outputs'][name]['review'] for r in benign),'concern':len(concern),'concern_review':sum(r['outputs'][name]['review'] for r in concern)}
 a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result['models'],indent=2))
if __name__=='__main__':main()
