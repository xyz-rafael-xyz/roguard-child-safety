"""Mixed conversational/structured multi-task LoRA; select on development only."""
import argparse,gzip,hashlib,json,math,random,time
from pathlib import Path
import numpy as np
import torch
from peft import LoraConfig,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict
from sklearn.metrics import f1_score
from training.evidence_v8 import load
from training.context_transfer_v71 import generate
from training.fit_context_v7 import unique
from roguard.context import render
from roguard.evidence import ONTOLOGY
from roguard.semantic import SemanticEncoder,normalize_semantic_text


def natural(split):
    out=[]
    for r in unique(generate(split)):
        f=r['frame']
        if f=='CORRECTION' or r.get('surface')=='transliterated':continue
        subject='OTHER' if f=='OTHER_PERSON' else 'FICTIONAL' if f=='QUOTED' else 'SELF'
        out.append(dict(case_id='native-'+r['case_id'],language=r['language'],turns=r['turns'],focus=len(r['turns'])-1,
                        attributes={'subject':subject,'time':'PAST' if f=='SELF_PAST' else 'CURRENT',
                                    'assertion':'UNCERTAIN' if f=='UNCERTAIN' else 'AFFIRMED',
                                    'source':'REPORTED' if subject=='OTHER' else 'FICTION' if subject=='FICTIONAL' else 'DIRECT',
                                    'relation':'NEW','concern':str(r['signal'])},fictional=True,source='v7_training_frame_projection'))
    return out


def main():
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--device',default='cpu');p.add_argument('--epochs',type=int,default=2);p.add_argument('--structured-rows',type=int,default=5000);p.add_argument('--conversational-rows',type=int,default=3000);p.add_argument('--initial-evidence',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
 torch.set_num_threads(4);torch.manual_seed(830);rng=random.Random(830)
 root=Path('research/v8-evidence');structured=load(root/'data','train')+[json.loads(l) for l in gzip.decompress((root/'data/train-contextual.jsonl.gz').read_bytes()).decode().splitlines()];rng.shuffle(structured)
 native=natural('train');rng.shuffle(native);train=structured[:a.structured_rows]+native[:a.conversational_rows]
 dev_formal=load(root/'data','dev');native_dev=natural('dev');rng.shuffle(native_dev);native_dev=native_dev[:1000];dev=dev_formal+native_dev
 cal=load(root/'data','cal')+natural('cal')[:1000]
 heads_path=a.initial_evidence or root/'semantic-03/semantic.json.gz';adapter_path=a.initial_evidence or Path('src/roguard/context-joint.json.gz')
 old=json.loads(gzip.decompress(heads_path.read_bytes()));prior=json.loads(gzip.decompress(adapter_path.read_bytes()))
 encoder=SemanticEncoder(a.base,a.device);model=get_peft_model(encoder.model,LoraConfig(r=4,lora_alpha=8,lora_dropout=.05,target_modules=['Wqkv']))
 set_peft_model_state_dict(model,{k:torch.tensor(v,dtype=torch.float32) for k,v in prior['lora']['tensors'].items()})
 heads=torch.nn.ModuleDict();classes={}
 for name,h in old['heads'].items():
  classes[name]=h['classes'];head=torch.nn.Linear(1536,len(h['classes']))
  w=torch.tensor(h['weights']);b=torch.tensor(h['intercept'])
  if len(h['classes'])==2:w=torch.cat([torch.zeros_like(w),w]);b=torch.cat([torch.zeros_like(b),b])
  with torch.no_grad():head.weight.copy_(w);head.bias.copy_(b)
  heads[name]=head
 heads.to(a.device)
 def token_ok(r):return len(encoder.tokenizer(normalize_semantic_text(render(r['turns'][:r['focus']+1])))['input_ids'])<=256
 before=len(train);train=[r for r in train if token_ok(r)];dev=[r for r in dev if token_ok(r)];cal=[r for r in cal if token_ok(r)]
 raw=('\n'.join(json.dumps(r,ensure_ascii=False) for r in train)+'\n').encode();(a.output/'selected-training.jsonl.gz').write_bytes(gzip.compress(raw,mtime=0))
 record={'initial_lora_sha256':hashlib.sha256(adapter_path.read_bytes()).hexdigest(),'initial_heads_sha256':hashlib.sha256(heads_path.read_bytes()).hexdigest(),'train_rows':len(train),'excluded_overlength':before-len(train),'training_sha256':hashlib.sha256(raw).hexdigest(),'epochs':[],'synthetic_only':True,'test_data_used_for_selection':False}
 params=[v for v in model.parameters() if v.requires_grad]+list(heads.parameters());opt=torch.optim.AdamW(params,lr=3e-5)
 def forward(rows):
  full=[render(r['turns'][:r['focus']+1]) for r in rows];focus=[r['turns'][r['focus']]['text'] for r in rows]
  tokens=encoder.tokenizer([normalize_semantic_text(t) for t in full+focus],padding=True,truncation=False,return_tensors='pt')
  if tokens['input_ids'].shape[1]>256:raise ValueError('Semantic limit')
  tokens={k:v.to(a.device) for k,v in tokens.items()};h=model(**tokens).last_hidden_state.float();mask=tokens['attention_mask'].unsqueeze(-1);pooled=torch.nn.functional.normalize((h*mask).sum(1)/mask.sum(1),dim=1)*10;features=torch.cat([pooled[:len(rows)],pooled[len(rows):]],dim=1)
  return {k:head(features) for k,head in heads.items()}
 def score(rows):
  out={k:[] for k in heads}
  with torch.no_grad():
   for offset in range(0,len(rows),16):
    batch=forward(rows[offset:offset+16])
    for k,logits in batch.items():out[k].extend(torch.softmax(logits,dim=1).cpu().tolist())
  return out
 best=None
 for epoch in range(1,a.epochs+1):
  model.train();heads.train();order=list(train);rng.shuffle(order);losses=[];start=time.monotonic()
  for offset in range(0,len(order),8):
   batch=order[offset:offset+8];opt.zero_grad();logits=forward(batch)
   loss=sum(torch.nn.functional.cross_entropy(v,torch.tensor([classes[k].index(r['attributes'][k]) for r in batch],device=a.device)) for k,v in logits.items())/len(heads)
   loss.backward();torch.nn.utils.clip_grad_norm_(params,1);opt.step();losses.append(float(loss.detach().cpu()))
   if offset%800==0:print('epoch',epoch,'row',offset,'loss',losses[-1],flush=True)
  model.eval();heads.eval();scores=score(dev);metrics={};native_metrics={}
  for k in heads:
   for group,target in [('formal',metrics),('native',native_metrics)]:
    target[k]={}
    for lang in ('ro','uk'):
     indices=[i for i,r in enumerate(dev) if r['language']==lang and (r.get('source')=='v7_training_frame_projection')==(group=='native')]
     target[k][lang]=float(f1_score([dev[i]['attributes'][k] for i in indices],[classes[k][int(np.argmax(scores[k][i]))] for i in indices],labels=classes[k] if group=='formal' else None,average='macro',zero_division=0))
  rank=(min(v for group in (metrics,native_metrics) for vals in group.values() for v in vals.values()),sum(v for group in (metrics,native_metrics) for vals in group.values() for v in vals.values()))
  entry={'epoch':epoch,'loss':sum(losses)/len(losses),'seconds':time.monotonic()-start,'formal_development':metrics,'conversational_development':native_metrics};record['epochs'].append(entry);print(json.dumps(entry),flush=True)
  if best is None or rank>best:
   best=rank;calp=score(cal);spec=json.loads(json.dumps(old));spec['lora']={'rank':4,'alpha':8,'tensors':{k:v.detach().cpu().tolist() for k,v in get_peft_model_state_dict(model).items()}};spec['adapted']=True
   for k,head in heads.items():
    w=head.weight.detach().cpu().numpy();b=head.bias.detach().cpu().numpy()
    if len(classes[k])==2:w=np.array([w[1]-w[0]]);b=np.array([b[1]-b[0]])
    h=spec['heads'][k];h['weights']=w.tolist();h['intercept']=b.tolist()
    for lang in ('ro','uk'):
     qs=[]
     for ci,label in enumerate(classes[k]):
      residuals=sorted(1-p[ci] for r,p in zip(cal,calp[k]) if r['language']==lang and r['attributes'][k]==label);n=math.ceil((len(residuals)+1)*.95);qs.append(residuals[n-1] if n<=len(residuals) else 1.)
     h['quantiles'][lang]=qs
   (a.output/'semantic.json.gz').write_bytes(gzip.compress(json.dumps(spec,separators=(',',':')).encode(),mtime=0));record['selected_epoch']=epoch
  (a.output/'development.json').write_text(json.dumps(record,indent=2)+'\n')
 print('FINISHED',flush=True)
if __name__=='__main__':main()
