"""Evaluate a separate synthetic source without selecting or changing models."""
import argparse,hashlib,json,subprocess
from pathlib import Path
from roguard.conversation import ConversationScorer, configuration, render, language_uncertainty
from roguard.message_model import SyntheticMessageModel
from roguard.semantic import head_scores
from eval.run_conversations_v4 import summarize

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');args=p.parse_args()
 if args.output.exists():raise ValueError('Preserve earlier diagnostic')
 source=json.loads(args.input.read_text());rows=source['rows'];rows.sort(key=lambda r:r['language'])
 if not rows:raise ValueError('No valid rows')
 config=configuration();fast=ConversationScorer();semantic=ConversationScorer('semantic',args.base,args.device)
 cs=[];ss=[];old=[]
 for lang in ('ro','uk'):
  subset=[r for r in rows if r['language']==lang];texts=[render(r['turns']) for r in subset]
  cs.extend(fast.scores(texts,lang));ss.extend(head_scores(semantic.encoder.encode(texts),semantic.head).tolist())
  baseline=SyntheticMessageModel.packaged(lang);old.extend(baseline.score(r['turns'][-1]['text']) for r in subset)
 mix=[config['semantic_weight']*s+(1-config['semantic_weight'])*c for s,c in zip(ss,cs)]
 oc={lang:{'cutoff':SyntheticMessageModel.packaged(lang).cutoff} for lang in ('ro','uk')}
 reports={name:summarize(rows,scores,settings) for name,scores,settings in [('v3_last_message',old,oc),('fast',cs,config['fast']),('semantic',ss,config['semantic']),('hybrid',mix,config['hybrid'])]}
 result={'source_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
  'rows':len(rows),'reports':reports,'language_quality_validated':False,'labels_independently_validated':False,
  'language_guard_abstentions':sum(bool(language_uncertainty(r['turns'],r['language'])) for r in rows),
  'predictions':[{'id':r['id'],'label':r['label'],'scores':dict(v3=old[i],fast=cs[i],semantic=ss[i],hybrid=mix[i])} for i,r in enumerate(rows)]}
 with args.output.open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(reports,indent=2))
if __name__=='__main__':main()
