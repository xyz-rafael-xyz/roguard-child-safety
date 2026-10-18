"""Evaluate an explicitly selected direct-review checkpoint on a supplied dataset."""
import argparse,gzip,hashlib,json
from pathlib import Path
import numpy as np
import torch
from peft import LoraConfig,get_peft_model,set_peft_model_state_dict
from roguard.context import render
from roguard.semantic import SemanticEncoder
from training.fit_review_v8 import labels,group_metrics
from training.fit_evidence_v8 import semantic_features


def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--base',required=True);p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True);p.add_argument('--language',choices=('ro','uk'));a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);spec=json.loads(gzip.decompress(a.artifact.read_bytes()));raw=a.data.read_bytes();rows=[json.loads(l) for l in (gzip.decompress(raw) if a.data.suffix=='.gz' else raw).decode().splitlines()]
    if a.language:rows=[r for r in rows if r['language']==a.language]
    if not rows:raise ValueError('No evaluation rows for selected language')
    for r in rows:r.setdefault('focus',len(r['turns'])-1)
    adapter=spec['lora'];rank=adapter['rank'];alpha=adapter['alpha'];targets=adapter.get('targets',['Wqkv'])
    if rank not in (4,16) or alpha!=2*rank or targets not in (['Wqkv'],['Wqkv','Wo','Wi']):raise ValueError('Unsupported research adapter architecture')
    encoder=SemanticEncoder(a.base,a.device);model=get_peft_model(encoder.model,LoraConfig(r=rank,lora_alpha=alpha,lora_dropout=.05,target_modules=targets));set_peft_model_state_dict(model,{k:torch.tensor(v,dtype=torch.float32) for k,v in spec['lora']['tensors'].items()});encoder.model=model.to(a.device).eval()
    if spec.get('format')=='roguard-attention-review-experiment-v1':
        from training.attention_pool_v9 import attention_pool
        from roguard.semantic import normalize_semantic_text
        queries=torch.tensor(spec['pooling_queries'],dtype=torch.float32,device=a.device)
        parts=[]
        with torch.inference_mode():
            for start in range(0,len(rows),16):
                batch=rows[start:start+16]
                texts=[normalize_semantic_text(render(r['turns'][:r['focus']+1])) for r in batch]
                texts += [normalize_semantic_text(r['turns'][r['focus']]['text']) for r in batch]
                tokens=encoder.tokenizer(texts,padding=True,truncation=False,return_tensors='pt')
                if tokens['input_ids'].shape[1]>256:raise ValueError('Input exceeds token limit')
                tokens={k:v.to(a.device) for k,v in tokens.items()}
                hidden=model(**tokens).last_hidden_state.float()
                pooled=attention_pool(hidden,tokens['attention_mask'],queries)
                parts.append(torch.cat([pooled[:len(batch)],pooled[len(batch):]],dim=1).cpu().numpy())
        features=np.concatenate(parts)
    else:
        full=semantic_features([render(r['turns'][:r['focus']+1]) for r in rows],encoder,a.output/'context-features.npz');focus=semantic_features([r['turns'][r['focus']]['text'] for r in rows],encoder,a.output/'focus-features.npz')
        features=np.concatenate([full,focus],axis=1)
    if spec.get('format')=='roguard-policy-review-experiment-v1':
        from training.policy_probability_v9 import review_mass
        probabilities={}
        for task,head in spec['heads'].items():
            logits=features@np.asarray(head['weights']).T+np.asarray(head['intercept'])
            values=np.exp(logits-logits.max(axis=1,keepdims=True))
            values=values/values.sum(axis=1,keepdims=True)
            probabilities[task]={label:values[:,i] for i,label in enumerate(head['classes'])}
        scores=np.clip(review_mass(probabilities),1e-6,1-1e-6)
    else:
        logits=features@np.array(spec['weights'])+spec['intercept']
        scores=1/(1+np.exp(-np.clip(logits,-700,700)))
    truth=np.array([r['review'] if 'review' in r else int(labels([r])[0]) for r in rows]);thresholds=spec['thresholds']
    predictions=[{'case_id':r['case_id'],'language':r['language'],'score':float(s),'gold':int(y),'predicted':int(s>=thresholds[r['language']])} for r,s,y in zip(rows,scores,truth)]
    (a.output/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
    metrics={}
    for lang in ((a.language,) if a.language else ('ro','uk')):
        ps=[r for r in predictions if r['language']==lang];metrics[lang]={'positive':sum(r['gold']==1 for r in ps),'negative':sum(r['gold']==0 for r in ps),'missed':sum(r['gold']==1 and r['predicted']==0 for r in ps),'false_reviews':sum(r['gold']==0 and r['predicted']==1 for r in ps)}
    record={'artifact_sha256':hashlib.sha256(a.artifact.read_bytes()).hexdigest(),'input_sha256':hashlib.sha256(raw).hexdigest(),'metrics':metrics,'language_filter':a.language,'thresholds':thresholds,'synthetic_only':True}
    (a.output/'metrics.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
if __name__=='__main__':main()
