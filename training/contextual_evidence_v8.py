"""Training-only scope-separated conversations, preserving explicit labels."""
import gzip,json,random
from pathlib import Path
from training.evidence_v8 import load

root=Path('research/v8-evidence/data')
rows=[json.loads(l) for l in gzip.decompress((root/'train-paraphrases.jsonl.gz').read_bytes()).decode().splitlines()]
out=[]
for i,r in enumerate(rows):
 out.append(r)
 # Move subject and time context to an earlier turn in a second conversation.
 v=json.loads(json.dumps(r));text=v['turns'][2]['text'];start=text.index('«');end=text.rindex('»')+1
 # Find assertion boundary before opening quote (colon); preserve its sentence.
 before=text[:start];boundary=max(before.rfind('.'),before.rfind('!'),before.rfind('?'))+1
 focus=text[boundary:end].strip();scope=(text[:boundary]+' '+text[end:]).strip()
 if not scope or not focus:raise ValueError('Could not separate authored scope')
 v['turns'][0]['text']=v['turns'][0]['text']+' '+scope
 v['turns'][2]['text']=focus;v['case_id']+='-scope';v['source']='training_scope_separation_v8';out.append(v)
path=root/'train-contextual.jsonl.gz'
if path.exists():raise ValueError('Refusing overwrite')
path.write_bytes(gzip.compress(('\n'.join(json.dumps(r,ensure_ascii=False) for r in out)+'\n').encode(),mtime=0))
print('Contextual augmentation rows',len(out))
