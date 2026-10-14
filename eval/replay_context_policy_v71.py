"""Recheck policy-only changes against saved prefix predictions; no model refit.

The original model outputs are preserved. Case IDs, prefix indices and original
turn lengths must match before replay. This is regression, not a new model test.
"""
import argparse,hashlib,json
from pathlib import Path
from roguard.context import analyze_context,render


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--prefix',default='candidate09');parser.add_argument('--include-finals',action='store_true');args=parser.parse_args()
 root=Path('research/v7-1-context')
 from training.context_factorial_final_v7 import generate as factorial
 from training.context_session_probe_v7 import generate as session
 for suite in ('regression','fresh','stress','session')+(('final2','final3') if args.include_finals else ()):
  source=root/('candidate09-final2-regression.json' if suite=='final2' else 'candidate09-final3.json' if suite=='final3' else f'candidate08-{suite}.json');old=json.loads(source.read_text());destination=root/f'{args.prefix}-{suite}-replay.json'
  if destination.exists():raise ValueError('Do not overwrite regression evidence')
  rows=[r for r in factorial() if r['surface']=='original'] if suite=='regression' else session() if suite=='session' else [json.loads(line) for line in (root/f'{suite}-test.jsonl').read_text().splitlines()]
  prior={o['case_id']:o for o in old['outputs']};cells={};outputs=[]
  assert {r['case_id'] for r in rows}==set(prior)
  for row in rows:
   saved=prior[row['case_id']];indices=[i for i,t in enumerate(row['turns']) if t['role']=='minor']
   assert [e['turn_index'] for e in saved['evidence']]==indices
   for e in saved['evidence']:assert e['candidate_reference']['end']==len(row['turns'][e['turn_index']]['text'])
   class Cached:
    backend='joint';identity=old['model_sha256']
    def predict(self,texts,language):
     assert language==row['language'] and texts==[render(row['turns'][:i+1]) for i in indices]
     return [{'supported':e['supported'],'reason':e['reason'],'heads':{'frame':{'label':e['frame'],'prediction_set':e['frame_set']},'signal':{'label':e['mention'],'prediction_set':e['mention_set']}}} for e in saved['evidence']]
   report=analyze_context({'language':row['language'],'turns':row['turns']},Cached())
   kind=('benign' if row['kind']=='benign' else 'personal' if row['kind']=='personal' else 'quoted' if row['kind']=='quoted' else 'other_concern') if suite=='session' else 'benign' if not row['signal'] else 'personal' if row['frame'] in ('SELF_CURRENT','SELF_PAST') else 'quoted' if row['frame']=='QUOTED' else 'other_concern'
   cell=cells.setdefault(row['language']+'/'+kind,{'count':0,'review':0});cell['count']+=1;cell['review']+=int(report['decision']=='REVIEW');outputs.append({'case_id':row['case_id'],'report':report})
  record={'synthetic_only':True,'method':'policy replay on saved model prefix predictions','source':source.name,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'policy_sha256':hashlib.sha256(Path('src/roguard/context.py').read_bytes()).hexdigest(),'rules_sha256':hashlib.sha256(Path('src/roguard/context_signals.py').read_bytes()).hexdigest(),'cells':cells,'outputs':outputs}
  destination.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(suite,cells)
if __name__=='__main__':main()
