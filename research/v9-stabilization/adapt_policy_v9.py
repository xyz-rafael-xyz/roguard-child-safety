"""Train interpretation heads through the fixed review policy.

All six interpretation heads shape the shared representation during training.
Review mass is computed from the interpretation heads under an explicitly
independent marginal approximation. It is not calibrated confidence.
"""
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
from roguard.evidence import ONTOLOGY
from training.policy_probability_v9 import review_mass
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
    parser.add_argument('--structured-rows',type=int,default=8000)
    parser.add_argument('--conversational-rows',type=int,default=8000)
    parser.add_argument('--initial-evidence', type=Path, default=Path('research/v8-evidence/semantic-05/semantic.json.gz'))
    parser.add_argument('--evaluation-reference', type=Path, required=True)
    parser.add_argument('--head-learning-rate', type=float, default=0.0005)
    parser.add_argument('--contrasts', action='store_true')
    parser.add_argument('--short-contrasts', action='store_true')
    parser.add_argument('--context-interventions', action='store_true')
    parser.add_argument('--hesitancy-contrasts', action='store_true')
    parser.add_argument('--uk-scope-contrasts', action='store_true')
    parser.add_argument('--scope-curriculum', action='store_true')
    parser.add_argument('--train-language', choices=('ro','uk'))
    parser.add_argument('--balance-short', action='store_true')
    parser.add_argument('--learning-rate',type=float,default=1e-4)
    parser.add_argument('--label-smoothing',type=float,default=0.0)
    parser.add_argument('--auxiliary-weight',type=float,default=1.0)
    args = parser.parse_args()
    if not 0 < args.auxiliary_weight <= 10:
        raise ValueError('Auxiliary loss weight must be in (0, 10]')
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(4); torch.manual_seed(870)
    rng = random.Random(870)
    root = Path('research/v8-evidence')
    formal = load(root/'data', 'train')
    formal += [json.loads(l) for l in gzip.decompress((root/'data/train-contextual.jsonl.gz').read_bytes()).decode().splitlines()]
    rng.shuffle(formal)
    native = natural('train'); rng.shuffle(native)
    train = formal[:args.structured_rows] + native[:args.conversational_rows]
    if args.contrasts:
        from training.review_contrasts_v8 import generate
        train += generate()
    if args.short_contrasts:
        from training.short_review_contrasts_v8 import generate as short_generate
        train += short_generate()
    if args.context_interventions:
        from training.context_interventions_v8 import generate as interventions
        train += interventions()
    if args.hesitancy_contrasts:
        from training.disclosure_hesitancy_v8 import generate as hesitancy
        train += hesitancy()
    if args.uk_scope_contrasts:
        from training.uk_scope_contrasts_v8 import generate as uk_scope
        train += uk_scope()
    if args.scope_curriculum:
        from training.scope_curriculum_v9 import generate as scope_curriculum
        train += scope_curriculum()
    if args.train_language:
        train=[r for r in train if r['language']==args.train_language]
    # Reuse exact declared development/calibration samples from the prior study.
    rows = {'train': train}
    for split in ('dev', 'cal'):
        rows[split] = [json.loads(l) for l in gzip.decompress((args.evaluation_reference / (split+'.jsonl.gz')).read_bytes()).decode().splitlines()]
    from training.review_data_v8 import isolate_partitions, input_key
    prior_path=args.initial_evidence.parent/'training-history.jsonl.gz'
    if not prior_path.exists():prior_path=args.initial_evidence.parent/'selected-training.jsonl.gz'
    prior_training=[json.loads(l) for l in gzip.decompress(prior_path.read_bytes()).decode().splitlines()]
    conservative_history=args.evaluation_reference/'training-history.jsonl.gz'
    prior_training += [json.loads(l) for l in gzip.decompress(conservative_history.read_bytes()).decode().splitlines()]
    evaluation_keys={input_key(r) for split in ('dev','cal') for r in rows[split]}
    if evaluation_keys.intersection(input_key(r) for r in prior_training):
        raise ValueError('Initial encoder history overlaps fixed evaluation inputs')
    # Evaluation ownership is immutable. Remove colliding new training rows.
    omitted=[r['case_id'] for r in rows['train'] if input_key(r) in evaluation_keys]
    rows['train']=[r for r in rows['train'] if input_key(r) not in evaluation_keys]
    rows, removed = isolate_partitions(rows,prior_training)
    if removed['dev'] or removed['cal']:raise ValueError('Evaluation population changed')
    removed['new_training_collisions']=omitted
    (args.output/'overlap-exclusions.json').write_text(json.dumps(removed,indent=2)+'\n')
    initial=args.initial_evidence
    spec=json.loads(gzip.decompress(initial.read_bytes()))
    encoder = SemanticEncoder(args.base, args.device)
    model = get_peft_model(encoder.model, LoraConfig(r=4, lora_alpha=8, lora_dropout=.05, target_modules=['Wqkv']))
    set_peft_model_state_dict(model, {k: torch.tensor(v, dtype=torch.float32) for k, v in spec['lora']['tensors'].items()})
    features = {}; targets = {}; record = {'synthetic_only': True, 'selection_partition': 'development', 'learning_rate': args.learning_rate, 'label_smoothing': args.label_smoothing, 'additional_authored_contrasts': args.contrasts, 'short_contrasts': args.short_contrasts, 'context_interventions': args.context_interventions, 'hesitancy_contrasts': args.hesitancy_contrasts, 'uk_scope_contrasts': args.uk_scope_contrasts, 'train_language': args.train_language, 'balance_short': args.balance_short, 'initial_encoder_sha256': hashlib.sha256(initial.read_bytes()).hexdigest(), 'head_learning_rate': args.head_learning_rate, 'epochs': [], 'excluded_overlength': {}}
    record['scope_curriculum'] = args.scope_curriculum
    record['history_includes_conservative_predecessor_union'] = True
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
    history={input_key(r):r for r in prior_training}
    history.update({input_key(r):r for r in rows['train']})
    history_raw=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in history.values()).encode()
    (args.output/'training-history.jsonl.gz').write_bytes(gzip.compress(history_raw,mtime=0))
    record['training_history_unique_inputs']=len(history)
    record['training_history_sha256']=hashlib.sha256(history_raw).hexdigest()
    classes = {task: list(values) for task, values in ONTOLOGY.items()}
    auxiliaries = torch.nn.ModuleDict({task: torch.nn.Linear(1536, len(values)) for task, values in classes.items()}).to(args.device)
    with torch.no_grad():
        for task, module in auxiliaries.items():
            previous=spec['heads'][task]
            weight=torch.tensor(previous['weights'],dtype=torch.float32,device=args.device)
            bias=torch.tensor(previous['intercept'],dtype=torch.float32,device=args.device)
            if len(previous['classes'])==2 and weight.shape[0]==1:
                weight=torch.cat([-weight/2,weight/2],dim=0)
                bias=torch.cat([-bias/2,bias/2],dim=0)
            order=[previous['classes'].index(value) for value in classes[task]]
            module.weight.copy_(weight[order]);module.bias.copy_(bias[order])
    auxiliary_targets = {task: torch.tensor([values.index(row['attributes'][task]) for row in rows['train']], dtype=torch.long, device=args.device) for task, values in classes.items()}
    record['auxiliary_supervision'] = {'weight': args.auxiliary_weight, 'classes': classes, 'loss': 'mean six-task cross entropy', 'inference_uses_auxiliary_heads': True, 'head_initialization': 'trained semantic heads', 'probability_assumption': 'independent marginals'}
    weights=np.ones(len(rows['train']),dtype=np.float32)
    if args.balance_short:
        from collections import Counter
        groups=[(r['language'],len(r['turns'])==1 and len(r['turns'][0]['text'].split())<=12,int(y)) for r,y in zip(rows['train'],targets['train'])]
        counts=Counter(groups)
        weights=np.array([len(groups)/(len(counts)*counts[g]) for g in groups],dtype=np.float32)
        record['training_balance_groups']={str(k):v for k,v in counts.items()}
    def forward(split, indices, with_auxiliary=False):
        tokens = encoder.tokenizer.pad([features[split][i][j] for j in (0,1) for i in indices], padding=True, return_tensors='pt')
        tokens = {k:v.to(args.device) for k,v in tokens.items()}
        hidden = model(**tokens).last_hidden_state.float()
        mask = tokens['attention_mask'].unsqueeze(-1)
        pooled = torch.nn.functional.normalize((hidden*mask).sum(1)/mask.sum(1),dim=1)*10
        representation = torch.cat([pooled[:len(indices)],pooled[len(indices):]],dim=1)
        task_logits = {task: module(representation) for task,module in auxiliaries.items()}
        probabilities={task:torch.softmax(values,dim=-1) for task,values in task_logits.items()}
        marginals={task:{value:probabilities[task][:,i] for i,value in enumerate(classes[task])} for task in classes}
        probability=review_mass(marginals).clamp(1e-6,1-1e-6)
        logits=torch.logit(probability)
        return (logits,task_logits) if with_auxiliary else logits
    def score(split):
        model.eval();auxiliaries.eval(); result=[]
        with torch.no_grad():
            for start in range(0,len(rows[split]),16):
                result.extend(torch.sigmoid(forward(split,list(range(start,min(start+16,len(rows[split])))))).cpu().tolist())
        return np.array(result)
    def save_scores(split, scores, thresholds):
        predictions=[{'case_id':r['case_id'],'language':r['language'],'score':float(p),'gold':int(y),'predicted':int(p>=thresholds[r['language']])} for r,y,p in zip(rows[split],targets[split],scores)]
        (args.output/(split+'-predictions.jsonl')).write_text(''.join(json.dumps(p)+'\n' for p in predictions))
    encoder_params=[v for v in model.parameters() if v.requires_grad]
    params=encoder_params+list(auxiliaries.parameters())
    opt=torch.optim.AdamW([{'params':encoder_params,'lr':args.learning_rate},{'params':list(auxiliaries.parameters()),'lr':args.head_learning_rate}])
    best = None
    for epoch in range(1,args.epochs+1):
        model.train();auxiliaries.train();start=time.monotonic();order=list(range(len(rows['train'])));rng.shuffle(order);losses=[]
        for offset in range(0,len(order),8):
            ix=order[offset:offset+8]; opt.zero_grad()
            logits, auxiliary_logits=forward('train',ix,with_auxiliary=True)
            target=torch.tensor(targets['train'][ix],dtype=torch.float32,device=args.device)
            target=target*(1-args.label_smoothing)+.5*args.label_smoothing
            loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,target,reduction='none')*torch.tensor(weights[ix],device=args.device)).mean()
            auxiliary_loss = torch.stack([torch.nn.functional.cross_entropy(auxiliary_logits[task], auxiliary_targets[task][ix]) for task in classes]).mean()
            loss = loss + args.auxiliary_weight * auxiliary_loss
            loss.backward();torch.nn.utils.clip_grad_norm_(params,1);opt.step();losses.append(float(loss.detach().cpu()))
            if offset%800==0:print('epoch',epoch,'row',offset,'of',len(order),'loss',losses[-1],flush=True)
        scores=score('dev'); thresholds=thresholds_for(rows['dev'],targets['dev'],scores)
        metrics=group_metrics(rows['dev'],targets['dev'],scores,thresholds)
        entry={'epoch':epoch,'loss':sum(losses)/len(losses),'seconds':time.monotonic()-start,'thresholds':thresholds,'development':metrics}
        record['epochs'].append(entry);print(json.dumps(entry),flush=True)
        selected_metrics=[v for k,v in metrics.items() if args.train_language is None or k.startswith(args.train_language+'_')]
        rank=(max(v['false_review_rate'] for v in selected_metrics),sum(v['false_review_rate'] for v in selected_metrics))
        artifact={'format':'roguard-policy-review-experiment-v1','representation':'context_plus_focus','heads':{task:{'classes':classes[task],'weights':module.weight.detach().cpu().tolist(),'intercept':module.bias.detach().cpu().tolist()} for task,module in auxiliaries.items()},'thresholds':thresholds,'lora':{'rank':4,'alpha':8,'tensors':{k:v.detach().cpu().tolist() for k,v in get_peft_model_state_dict(model).items()}}}
        compressed=gzip.compress(json.dumps(artifact,separators=(',',':')).encode(),mtime=0)
        (args.output/f'epoch-{epoch}-review.json.gz').write_bytes(compressed)
        if best is None or rank<best:
            best=rank;record['selected_epoch']=epoch;chosen_thresholds=thresholds
            save_scores('dev',scores,thresholds)
            (args.output/'review.json.gz').write_bytes(compressed)
        (args.output/'development.json').write_text(json.dumps(record,indent=2)+'\n')
    chosen=json.loads(gzip.decompress((args.output/'review.json.gz').read_bytes()))
    set_peft_model_state_dict(model,{k:torch.tensor(v,dtype=torch.float32) for k,v in chosen['lora']['tensors'].items()})
    with torch.no_grad():
        for task,module in auxiliaries.items():
            module.weight.copy_(torch.tensor(chosen['heads'][task]['weights'],device=args.device))
            module.bias.copy_(torch.tensor(chosen['heads'][task]['intercept'],device=args.device))
    calibration_scores=score('cal')
    save_scores('cal',calibration_scores,chosen_thresholds)
    record['calibration_audit']=group_metrics(rows['cal'],targets['cal'],calibration_scores,chosen_thresholds)
    record['artifact_sha256']=hashlib.sha256((args.output/'review.json.gz').read_bytes()).hexdigest()
    (args.output/'development.json').write_text(json.dumps(record,indent=2)+'\n')
    print('FINISHED',json.dumps(record['calibration_audit']),flush=True)

if __name__=='__main__':main()
