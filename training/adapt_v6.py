"""Registered LoRA development experiments; test split is never opened here."""
import argparse,copy,hashlib,json,random,subprocess,time
from pathlib import Path
import numpy as np
import torch
from peft import LoraConfig,get_peft_model
from roguard.semantic import SemanticEncoder,normalize_semantic_text
from roguard.conversation import render,configuration
from training.conversations_v4b import generate as old
from training.conversations_v6 import generate,digest

def metrics(rows,scores,cuts):
    out={}
    for lang in ('ro','uk'):
        pairs=[(r['label'],s) for r,s in zip(rows,scores) if r['language']==lang]; c=cuts[lang]
        tp=sum(y==1 and s>=c for y,s in pairs); fp=sum(y==0 and s>=c for y,s in pairs)
        p=sum(y==1 for y,s in pairs); n=len(pairs)-p
        out[lang]=dict(recall=tp/p,fpr=fp/n,balanced_accuracy=(tp/p+1-fp/n)/2,tp=tp,fp=fp,positives=p,negatives=n,cutoff=c)
    return out

def select(rows,scores):
    cuts={}
    for lang in ('ro','uk'):
        options=[]
        for c in np.arange(.1,.901,.02):
            m=metrics(rows,scores,{'ro':c,'uk':c})[lang]
            key=(m['fpr']<=.1,m['recall'] if m['fpr']<=.1 else -m['fpr'],m['balanced_accuracy'],-abs(c-.5))
            options.append((key,float(c)))
        cuts[lang]=max(options)[1]
    return cuts,metrics(rows,scores,cuts)

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve previous experiment')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise ValueError('Commit study first')
    a.output.mkdir(parents=True);torch.set_num_threads(4);random.seed(20261026);np.random.seed(20261026);torch.manual_seed(20261026)
    consumed=old('train');random.Random(20261026).shuffle(consumed)
    train=consumed[:320]+generate('train');dev=generate('dev')
    record=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),train_digest=digest(train),dev_digest=digest(dev),train_rows=len(train),dev_rows=len(dev),synthetic_only=True,candidates=[])
    basehead=json.loads(Path('src/roguard/conversation-head.json').read_text()); best=None
    for lr in (.0001,.0003):
        torch.manual_seed(20261026);encoder=SemanticEncoder(a.base,a.device)
        targets=[name for name,module in encoder.model.named_modules() if isinstance(module,torch.nn.Linear) and (name.endswith('Wqkv') or name.endswith('query') or name.endswith('value'))]
        if not targets:raise ValueError('No registered attention projections found')
        model=get_peft_model(encoder.model,LoraConfig(r=4,lora_alpha=8,lora_dropout=.05,target_modules=targets))
        head=torch.nn.Linear(len(basehead['weights']),1).to(a.device)
        with torch.no_grad():head.weight.copy_(torch.tensor(basehead['weights'],device=a.device)[None,:]);head.bias.fill_(basehead['intercept'])
        params=[x for x in model.parameters() if x.requires_grad]+list(head.parameters());opt=torch.optim.AdamW(params,lr=lr)
        def forward(rows):
            tokens=encoder.tokenizer([normalize_semantic_text(render(r['turns'])) for r in rows],padding=True,truncation=False,return_tensors='pt')
            if tokens['input_ids'].shape[1]>256:raise ValueError('Oversized training input')
            tokens={k:v.to(a.device) for k,v in tokens.items()};h=model(**tokens).last_hidden_state.float();mask=tokens['attention_mask'].unsqueeze(-1)
            return head(torch.nn.functional.normalize((h*mask).sum(1)/mask.sum(1),dim=1)*10).flatten()
        for epoch in range(1,4):
            model.train();head.train();order=list(range(len(train)));random.Random(20261026+epoch).shuffle(order);start=time.time();losses=[]
            for offset in range(0,len(order),8):
                batch=[train[i] for i in order[offset:offset+8]];opt.zero_grad();logits=forward(batch);loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.tensor([r['label'] for r in batch],device=a.device,dtype=torch.float32));loss.backward();torch.nn.utils.clip_grad_norm_(params,1);opt.step();losses.append(float(loss.detach().cpu()))
            model.eval();head.eval();scores=[]
            with torch.no_grad():
                for offset in range(0,len(dev),8):scores.extend(torch.sigmoid(forward(dev[offset:offset+8])).cpu().tolist())
            cuts,m=select(dev,scores);rank=(min(x['balanced_accuracy'] for x in m.values()),sum(x['balanced_accuracy'] for x in m.values()))
            row=dict(lr=lr,epoch=epoch,loss=sum(losses)/len(losses),seconds=time.time()-start,dev=m,scores=scores);record['candidates'].append(row);print(json.dumps({k:v for k,v in row.items() if k!='scores'}),flush=True)
            if best is None or rank>best:
                best=rank;model.save_pretrained(a.output/'adapter');(a.output/'head.json').write_text(json.dumps(dict(weights=head.weight.detach().cpu().flatten().tolist(),intercept=float(head.bias.detach().cpu()),cutoffs=cuts)));record['selected']=dict(lr=lr,epoch=epoch,dev=m)
            (a.output/'development.json').write_text(json.dumps(record,indent=2))
        del model,encoder,head,opt
    files={str(p.relative_to(a.output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in a.output.rglob('*') if p.is_file()}
    (a.output/'freeze.json').write_text(json.dumps(files,indent=2))
if __name__=='__main__':main()
