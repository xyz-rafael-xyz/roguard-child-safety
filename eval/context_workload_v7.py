"""End-to-end review workload on the frozen factorial subset, no fitting."""
import argparse,json,time
from pathlib import Path
from training.context_factorial_final_v7 import generate
from roguard.context import ContextModel,analyze_context
from roguard.conversation import ConversationScorer,analyze_conversation

def main():
 p=argparse.ArgumentParser();p.add_argument('--backend',action='append',choices=['joint','v6']);p.add_argument('--base',required=True);p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise ValueError('Preserve workload result')
 rows=[r for r in generate() if r['surface']=='original'];record={'synthetic_only':True,'subset':'all original-case rows, all three presentation forms','rows':len(rows),'backends':{}}
 for backend in (a.backend or ('joint','v6')):
  model=ContextModel('joint',a.base,a.device) if backend=='joint' else ConversationScorer('adapted',a.base,a.device);cells={};times=[]
  for r in rows:
   start=time.monotonic();payload={'language':r['language'],'turns':r['turns']}
   result=analyze_context(payload,model) if backend=='joint' else analyze_conversation(payload,model)
   decision=result['decision'] if backend=='joint' else result['signal']['decision'];review=decision in ('REVIEW','REVIEW_SUPPORT_SIGNAL','INSUFFICIENT_EVIDENCE');times.append((time.monotonic()-start)*1000)
   kind='benign' if not r['signal'] else 'personal' if r['frame'] in ('SELF_CURRENT','SELF_PAST') else 'quoted' if r['frame']=='QUOTED' else 'other_concern'
   key=r['language']+'/'+kind;cell=cells.setdefault(key,{'rows':0,'review':0});cell['rows']+=1;cell['review']+=int(review)
  record['backends'][backend]={'cells':cells,'latency_ms_p50':sorted(times)[len(times)//2],'latency_ms_p95':sorted(times)[int(len(times)*.95)],'device':a.device,'model_load_excluded':True}
  a.output.write_text(json.dumps(record,indent=2)+'\n');print(backend,cells,flush=True);del model
if __name__=='__main__':main()
