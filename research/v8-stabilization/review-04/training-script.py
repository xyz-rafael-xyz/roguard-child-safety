"""Adapt the encoder for review decisions, selecting only on development data."""
import argparse
import gzip
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from roguard.context import render
from roguard.semantic import SemanticEncoder, normalize_semantic_text
from training.adapt_evidence_v8 import natural
from training.evidence_v8 import load
from training.fit_review_v8 import labels, group_metrics


def thresholds_for(rows, truth, scores):
    thresholds = {}
    for lang in ('ro', 'uk'):
        limits = []
        for native in (False, True):
            values = sorted(float(scores[i]) for i, r in enumerate(rows)
                            if r['language'] == lang and bool(truth[i]) and
                            (r.get('source') == 'v7_training_frame_projection') == native)
            if not values:
                raise ValueError('Missing positive development group')
            limits.append(values[int(.01 * len(values))])
        thresholds[lang] = min(limits) * (1 - 1e-4)  # Small numerical guard for exported inference.
    return thresholds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--initial-review', type=Path)
    parser.add_argument('--contrasts', action='store_true')
    parser.add_argument('--learning-rate',type=float,default=1e-4)
    parser.add_argument('--label-smoothing',type=float,default=0.0)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(4); torch.manual_seed(870)
    rng = random.Random(870)
    root = Path('research/v8-evidence')
    formal = load(root/'data', 'train')
    formal += [json.loads(l) for l in gzip.decompress((root/'data/train-contextual.jsonl.gz').read_bytes()).decode().splitlines()]
    rng.shuffle(formal)
    native = natural('train'); rng.shuffle(native)
    train = formal[:8000] + native[:8000]
    if args.contrasts:
        from training.review_contrasts_v8 import generate
        train += generate()
    # Reuse exact declared development/calibration samples from the prior study.
    rows = {'train': train}
    for split in ('dev', 'cal'):
        rows[split] = [json.loads(l) for l in gzip.decompress(Path('research/v8-stabilization/review-01', split+'.jsonl.gz').read_bytes()).decode().splitlines()]
    from training.review_data_v8 import isolate_partitions
    rows, removed = isolate_partitions(rows)
    (args.output/'overlap-exclusions.json').write_text(json.dumps(removed,indent=2)+'\n')
    initial = root/'semantic-05/semantic.json.gz'
    spec = json.loads(gzip.decompress(initial.read_bytes()))
    if args.initial_review:
        spec = json.loads(gzip.decompress(args.initial_review.read_bytes()))
    encoder = SemanticEncoder(args.base, args.device)
    model = get_peft_model(encoder.model, LoraConfig(r=4, lora_alpha=8, lora_dropout=.05, target_modules=['Wqkv']))
    set_peft_model_state_dict(model, {k: torch.tensor(v, dtype=torch.float32) for k, v in spec['lora']['tensors'].items()})
    head = torch.nn.Linear(1536, 1).to(args.device)
    old = spec if args.initial_review else json.loads(Path('research/v8-stabilization/review-01/review-head.json').read_text())
    with torch.no_grad():
        head.weight.copy_(torch.tensor([old['weights']], device=args.device))
        head.bias.copy_(torch.tensor([old['intercept']], device=args.device))
    features = {}; targets = {}; record = {'synthetic_only': True, 'selection_partition': 'development', 'learning_rate': args.learning_rate, 'label_smoothing': args.label_smoothing, 'additional_authored_contrasts': args.contrasts, 'initial_encoder_sha256': hashlib.sha256(initial.read_bytes()).hexdigest(), 'initial_review_sha256': hashlib.sha256(args.initial_review.read_bytes()).hexdigest() if args.initial_review else None, 'epochs': [], 'excluded_overlength': {}}
    for split, rs in rows.items():
        kept = []; tokens = []
        for r in rs:
            full = normalize_semantic_text(render(r['turns'][:r['focus']+1]))
            focus = normalize_semantic_text(r['turns'][r['focus']]['text'])
            encoded = encoder.tokenizer([full, focus], padding=False, truncation=False)
            if max(map(len, encoded['input_ids'])) > 256:
                continue
            kept.append(r)
            tokens.append([{k:v[i] for k,v in encoded.items()} for i in range(2)])
        record['excluded_overlength'][split] = len(rs)-len(kept)
        rows[split] = kept; features[split] = tokens; targets[split] = labels(kept)
        raw = ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in kept).encode()
        (args.output/(split+'.jsonl.gz')).write_bytes(gzip.compress(raw,mtime=0))
    def forward(split, indices):
        tokens = encoder.tokenizer.pad([features[split][i][j] for j in (0,1) for i in indices], padding=True, return_tensors='pt')
        tokens = {k:v.to(args.device) for k,v in tokens.items()}
        hidden = model(**tokens).last_hidden_state.float()
        mask = tokens['attention_mask'].unsqueeze(-1)
        pooled = torch.nn.functional.normalize((hidden*mask).sum(1)/mask.sum(1),dim=1)*10
        return head(torch.cat([pooled[:len(indices)],pooled[len(indices):]],dim=1)).squeeze(-1)
    def score(split):
        model.eval();head.eval(); result=[]
        with torch.no_grad():
            for start in range(0,len(rows[split]),16):
                result.extend(torch.sigmoid(forward(split,list(range(start,min(start+16,len(rows[split])))))).cpu().tolist())
        return np.array(result)
    params = [v for v in model.parameters() if v.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.learning_rate)
    best = None
    for epoch in range(1,args.epochs+1):
        model.train();head.train();start=time.monotonic();order=list(range(len(rows['train'])));rng.shuffle(order);losses=[]
        for offset in range(0,len(order),8):
            ix=order[offset:offset+8]; opt.zero_grad()
            logits=forward('train',ix)
            target=torch.tensor(targets['train'][ix],dtype=torch.float32,device=args.device)
            target=target*(1-args.label_smoothing)+.5*args.label_smoothing
            loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,target)
            loss.backward();torch.nn.utils.clip_grad_norm_(params,1);opt.step();losses.append(float(loss.detach().cpu()))
            if offset%800==0:print('epoch',epoch,'row',offset,'of',len(order),'loss',losses[-1],flush=True)
        scores=score('dev'); thresholds=thresholds_for(rows['dev'],targets['dev'],scores)
        metrics=group_metrics(rows['dev'],targets['dev'],scores,thresholds)
        entry={'epoch':epoch,'loss':sum(losses)/len(losses),'seconds':time.monotonic()-start,'thresholds':thresholds,'development':metrics}
        record['epochs'].append(entry);print(json.dumps(entry),flush=True)
        rank=(max(v['false_review_rate'] for v in metrics.values()),sum(v['false_review_rate'] for v in metrics.values()))
        if best is None or rank<best:
            best=rank;record['selected_epoch']=epoch;chosen_thresholds=thresholds
            artifact={'format':'roguard-direct-review-experiment-v1','representation':'context_plus_focus','weights':head.weight.detach().cpu()[0].tolist(),'intercept':float(head.bias.detach().cpu()[0]),'thresholds':thresholds,'lora':{'rank':4,'alpha':8,'tensors':{k:v.detach().cpu().tolist() for k,v in get_peft_model_state_dict(model).items()}}}
            (args.output/'review.json.gz').write_bytes(gzip.compress(json.dumps(artifact,separators=(',',':')).encode(),mtime=0))
        (args.output/'development.json').write_text(json.dumps(record,indent=2)+'\n')
    chosen=json.loads(gzip.decompress((args.output/'review.json.gz').read_bytes()))
    set_peft_model_state_dict(model,{k:torch.tensor(v,dtype=torch.float32) for k,v in chosen['lora']['tensors'].items()})
    with torch.no_grad():
        head.weight.copy_(torch.tensor([chosen['weights']],device=args.device));head.bias.copy_(torch.tensor([chosen['intercept']],device=args.device))
    record['calibration_audit']=group_metrics(rows['cal'],targets['cal'],score('cal'),chosen_thresholds)
    record['artifact_sha256']=hashlib.sha256((args.output/'review.json.gz').read_bytes()).hexdigest()
    (args.output/'development.json').write_text(json.dumps(record,indent=2)+'\n')
    print('FINISHED',json.dumps(record['calibration_audit']),flush=True)

if __name__=='__main__':main()
