"""Joint frame/mention LoRA experiment, selected only on development data."""
import argparse,gzip,json,random,time
from pathlib import Path
import numpy as np
import torch
from peft import LoraConfig,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict
from sklearn.metrics import f1_score
from training.context_v7 import render,digest
from training.context_v71 import generate
from training.context_extra_v7 import augment
from training.fit_context_v7 import unique
from roguard.semantic import SemanticEncoder,normalize_semantic_text

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--device',default='cpu');p.add_argument('--initial',type=Path,default=Path('src/roguard/context-joint.json.gz'));p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
 torch.set_num_threads(4);torch.manual_seed(710);random.seed(710)
 pool=[r for r in augment(generate('train')) if r['surface']!='transliterated'];prefixes=[r for r in pool if r['source']=='derived_intermediate_prefix'];train=unique([r for r in pool if r['source']!='derived_intermediate_prefix']);random.shuffle(train);train=train[:4000]+prefixes+[r for r in pool if r['source']=='v71_authored']
 dev=unique(generate('dev'));cal=unique(generate('cal'))
 (a.output/'selected-training.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in train))
 old=json.loads(gzip.decompress(a.initial.read_bytes()))
 encoder=SemanticEncoder(a.base,a.device);model=get_peft_model(encoder.model,LoraConfig(r=4,lora_alpha=8,lora_dropout=.05,target_modules=['Wqkv']))
 set_peft_model_state_dict(model,{k:torch.tensor(v,dtype=torch.float32) for k,v in old['lora']['tensors'].items()})
 frame=torch.nn.Linear(1536,7).to(a.device);signal=torch.nn.Linear(1536,1).to(a.device)
 with torch.no_grad():
  for key,head in [('frame',frame),('signal',signal)]:head.weight.copy_(torch.tensor(old['heads'][key]['weights'],device=a.device));head.bias.copy_(torch.tensor(old['heads'][key]['intercept'],device=a.device))
 params=[v for v in model.parameters() if v.requires_grad]+list(frame.parameters())+list(signal.parameters());opt=torch.optim.AdamW(params,lr=0.00005)
 classes=old['heads']['frame']['classes'];record={'synthetic_only':True,'train_rows':len(train),'train_digest':digest(train),'joint_objective':'frame_cross_entropy + 1.0 * mention_binary_cross_entropy','epochs':[]}
 def forward(rows):
  full=[render(r['turns']) for r in rows];focus=[render([r['turns'][-1]]) for r in rows]
  tokens=encoder.tokenizer([normalize_semantic_text(t) for t in full+focus],padding=True,truncation=False,return_tensors='pt')
  if tokens['input_ids'].shape[1]>256:raise ValueError('Token limit')
  tokens={k:v.to(a.device) for k,v in tokens.items()};hidden=model(**tokens).last_hidden_state.float();mask=tokens['attention_mask'].unsqueeze(-1)
  pooled=torch.nn.functional.normalize((hidden*mask).sum(1)/mask.sum(1),dim=1)*10;features=torch.cat([pooled[:len(rows)],pooled[len(rows):]],dim=1)
  return frame(features),signal(features).flatten()
 def scores(rows):
  f=[];s=[]
  with torch.no_grad():
   for offset in range(0,len(rows),8):
    ff,ss=forward(rows[offset:offset+8]);f.extend(torch.softmax(ff,dim=1).cpu().tolist());s.extend(torch.sigmoid(ss).cpu().tolist())
  return f,[[1-v,v] for v in s]
 best=(-1,-1)
 for epoch in range(1,3):
  model.train();frame.train();signal.train();order=list(train);random.Random(710+epoch).shuffle(order);start=time.monotonic();losses=[]
  for offset in range(0,len(order),8):
   batch=order[offset:offset+8];opt.zero_grad();ff,ss=forward(batch)
   loss=torch.nn.functional.cross_entropy(ff,torch.tensor([classes.index(r['frame']) for r in batch],device=a.device))+torch.nn.functional.binary_cross_entropy_with_logits(ss,torch.tensor([r['signal'] for r in batch],device=a.device,dtype=torch.float32))
   loss.backward();torch.nn.utils.clip_grad_norm_(params,1);opt.step();losses.append(float(loss.detach().cpu()))
   if offset%800==0:print('epoch',epoch,'row',offset,'loss',losses[-1],flush=True)
  model.eval();frame.eval();signal.eval();f,s=scores(dev);metrics={l:float(f1_score([r['frame'] for r in dev if r['language']==l],[classes[int(np.argmax(v))] for r,v in zip(dev,f) if r['language']==l],average='macro')) for l in ('ro','uk')};mention_metrics={l:float(f1_score([r['signal'] for r in dev if r['language']==l],[int(np.argmax(v)) for r,v in zip(dev,s) if r['language']==l],average='macro')) for l in ('ro','uk')};rank=(min(list(metrics.values())+list(mention_metrics.values())),sum(metrics.values())+sum(mention_metrics.values()))
  entry=dict(epoch=epoch,loss=sum(losses)/len(losses),seconds=time.monotonic()-start,development_frame_macro_f1=metrics,development_mention_macro_f1=mention_metrics);record['epochs'].append(entry);print(json.dumps(entry),flush=True)
  if rank>best:
   best=rank;cf,cs=scores(cal);payload=json.loads(json.dumps(old));payload['backend']='joint';payload['representation']='context_plus_focus';payload['lora']={'rank':4,'alpha':8,'tensors':{k:v.detach().cpu().tolist() for k,v in get_peft_model_state_dict(model).items()}}
   for key,head,probabilities in [('frame',frame,cf),('signal',signal,cs)]:
    h=payload['heads'][key];h['weights']=head.weight.detach().cpu().tolist();h['intercept']=head.bias.detach().cpu().tolist();labels=h['classes']
    for lang in ('ro','uk'):
     residuals=sorted(1-v[labels.index(r[key])] for r,v in zip(cal,probabilities) if r['language']==lang);rankq=int(np.ceil((len(residuals)+1)*.9));q=residuals[rankq-1] if rankq<=len(residuals) else 1.;h['quantiles'][lang]=[q]*len(labels)
   (a.output/'joint.json.gz').write_bytes(gzip.compress(json.dumps(payload,separators=(',',':')).encode(),mtime=0));record['selected_epoch']=epoch
  (a.output/'development.json').write_text(json.dumps(record,indent=2)+'\n')
if __name__=='__main__':main()
