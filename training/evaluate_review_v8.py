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
    p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--base',required=True);p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);spec=json.loads(gzip.decompress(a.artifact.read_bytes()));raw=a.data.read_bytes();rows=[json.loads(l) for l in (gzip.decompress(raw) if a.data.suffix=='.gz' else raw).decode().splitlines()]
    for r in rows:r.setdefault('focus',len(r['turns'])-1)
    encoder=SemanticEncoder(a.base,a.device);model=get_peft_model(encoder.model,LoraConfig(r=4,lora_alpha=8,lora_dropout=.05,target_modules=['Wqkv']));set_peft_model_state_dict(model,{k:torch.tensor(v,dtype=torch.float32) for k,v in spec['lora']['tensors'].items()});encoder.model=model.to(a.device).eval()
    full=semantic_features([render(r['turns'][:r['focus']+1]) for r in rows],encoder,a.output/'context-features.npz');focus=semantic_features([r['turns'][r['focus']]['text'] for r in rows],encoder,a.output/'focus-features.npz')
    logits=np.concatenate([full,focus],axis=1)@np.array(spec['weights'])+spec['intercept'];scores=1/(1+np.exp(-np.clip(logits,-700,700)));truth=np.array([r['review'] if 'review' in r else int(labels([r])[0]) for r in rows]);thresholds=spec['thresholds']
    predictions=[{'case_id':r['case_id'],'language':r['language'],'score':float(s),'gold':int(y),'predicted':int(s>=thresholds[r['language']])} for r,s,y in zip(rows,scores,truth)]
    (a.output/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
    metrics={}
    for lang in ('ro','uk'):
        ps=[r for r in predictions if r['language']==lang];metrics[lang]={'positive':sum(r['gold']==1 for r in ps),'negative':sum(r['gold']==0 for r in ps),'missed':sum(r['gold']==1 and r['predicted']==0 for r in ps),'false_reviews':sum(r['gold']==0 and r['predicted']==1 for r in ps)}
    record={'artifact_sha256':hashlib.sha256(a.artifact.read_bytes()).hexdigest(),'input_sha256':hashlib.sha256(raw).hexdigest(),'metrics':metrics,'thresholds':thresholds,'synthetic_only':True}
    (a.output/'metrics.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
if __name__=='__main__':main()
