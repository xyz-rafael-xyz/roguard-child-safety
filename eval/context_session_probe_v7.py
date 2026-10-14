"""Actual incremental workflow probe; no aggregate safety-accuracy claim."""
import argparse,json,time
from pathlib import Path
from training.context_session_probe_v7 import generate
from roguard.context import ContextModel,analyze_context
from roguard.conversation import ConversationScorer,analyze_conversation

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise ValueError('Preserve frozen session probe')
 record={'synthetic_only':True,'author':'same project author, no independent native review','rows':generate(),'backends':{}}
 for backend in ('joint','v6'):
  model=ContextModel('joint',a.base,a.device) if backend=='joint' else ConversationScorer('adapted',a.base,a.device);outputs=[];metrics={}
  for row in record['rows']:
   payload={'language':row['language'],'turns':row['turns']};start=time.monotonic();r=analyze_context(payload,model) if backend=='joint' else analyze_conversation(payload,model);decision=r['decision'] if backend=='joint' else r['signal']['decision'];review=decision in ('REVIEW','REVIEW_SUPPORT_SIGNAL','INSUFFICIENT_EVIDENCE')
   key=row['language']+'/'+row['kind'];cell=metrics.setdefault(key,{'count':0,'review':0});cell['count']+=1;cell['review']+=int(review);outputs.append({'case_id':row['case_id'],'report':r,'latency_ms':(time.monotonic()-start)*1000})
  record['backends'][backend]={'metrics':metrics,'outputs':outputs};a.output.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(backend,metrics,flush=True);del model
if __name__=='__main__':main()
