"""Reason-covered continuation with separate development-fitted support/revision cutoffs."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import gzip
import json
from pathlib import Path
import time
from training.train_bare_distress_v9 import digest,read
from training.train_paired_nli_v9 import premise
from training.policy_entailment_v9 import HYPOTHESES,targets
from training.verify_candidate_pipeline_v9 import counts

STUDY=Path('research/v9-stabilization/reason-coverage-123')


def plan_batches(train,rng):
    pair_groups=defaultdict(list)
    for i,r in enumerate(train):
        if r.get('source') in ('paired_scope_v9','revision_contrasts_v9') and r.get('pair_group'):pair_groups[r['pair_group']].append(i)
    paired=set();units=[]
    for group in pair_groups.values():
        if len(group)!=2:raise ValueError('Incomplete original pair')
        units.append(group);paired.update(group)
    singles=[i for i in range(len(train)) if i not in paired];rng.shuffle(singles)
    units.extend(singles[i:i+2] for i in range(0,len(singles),2));rng.shuffle(units)
    batches=[]
    for start in range(0,len(units),2):
        original=[i for unit in units[start:start+2] for i in unit];indices=[2*i+t for i in original for t in range(2)]
        pairs=[]
        positions={value:i for i,value in enumerate(indices)}
        for unit in units[start:start+2]:
            if len(unit)==2 and train[unit[0]].get('source') in ('paired_scope_v9','revision_contrasts_v9') and train[unit[0]].get('pair_group') and train[unit[0]].get('pair_group')==train[unit[1]].get('pair_group'):
                for task in range(2):
                    a0,a1=[2*i+task for i in unit]
                    if targets(train[a0//2])[a0%2]!=targets(train[a1//2])[a1%2]:
                        neg,pos=(a0,a1) if targets(train[a0//2])[a0%2]==0 else (a1,a0);pairs.append((positions[neg],positions[pos]))
        batches.append((indices,pairs))
    return batches


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--device',default='mps');a=p.parse_args()
    for path,expected in json.loads((STUDY/'source-freeze.json').read_text()).items():
        if digest(Path(path))!=expected:raise ValueError('Source freeze changed: '+path)
    import numpy as np
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    from safetensors.torch import load_file,save_file
    from roguard.nli_v21 import MODEL_ID,REVISION,BASE_HASHES,verify_base
    from roguard.native_assets import reject_native_overrides
    from training.prepare_instruction_sft_v8 import target
    from training.conservative_fusion_v9 import full_coverage_gap
    verify_base(a.base);reject_native_overrides(a.base,BASE_HASHES)
    if a.base.name!=REVISION:raise ValueError('Base revision differs')
    parent=Path('research/v9-stabilization/policy-entailment-121/candidate');data=STUDY/'data';record=json.loads((parent/'development.json').read_text())
    spec=json.loads((parent/'review.json').read_text())
    if record.get('complete') is not True or digest(parent/'selected-model.safetensors')!=spec['checkpoint_sha256'] or digest(parent/'review.json')!=record['artifact_sha256']:raise ValueError('Incomplete parent candidate')
    output=STUDY/'candidate';output.mkdir(exist_ok=False)
    (output/'parent-selection.json').write_text(json.dumps(dict(recorded_utc=datetime.now(timezone.utc).isoformat(),parent_artifact_sha256=record['artifact_sha256'],parent_report_sha256=digest(parent/'development.json'),parent_checkpoint_sha256=spec['checkpoint_sha256'],input_sha256={name:digest(data/name) for name in ['training-history.jsonl.gz']+[split+'.jsonl.gz' for split in ('train','ro_dev','uk_dev','ro_scope_dev','uk_scope_dev','uk_stress','ro_cal','uk_cal','ro_exposed','uk_exposed')]}),indent=2)+'\n')
    rows={s:read(data/(s+'.jsonl.gz')) for s in ('train','ro_dev','uk_dev','ro_scope_dev','uk_scope_dev','uk_stress','ro_cal','uk_cal','ro_exposed','uk_exposed')}
    train=rows['train'];tokenizer=AutoTokenizer.from_pretrained(a.base,local_files_only=True,trust_remote_code=False)
    if tokenizer.padding_side!='right':raise ValueError('Right padding required')
    def encode(values):
        result=tokenizer([premise(r) for r in values for _ in HYPOTHESES],list(HYPOTHESES)*len(values),truncation=False,padding=False)
        if any(len(v)>512 for v in result['input_ids']):raise ValueError('No silent truncation')
        return result
    tokens={s:encode(values) for s,values in rows.items()}
    fresh=[]
    for lang in ('ro','uk'):
        fresh += [json.loads(l) for l in (Path('research/v9-stabilization/paired-scope-117/data')/(lang+'-test.jsonl')).read_text().splitlines()]
    protected={tuple(v) for s,t in tokens.items() if s!='train' for v in t['input_ids']} | {tuple(v) for v in encode(fresh)['input_ids']}
    if {tuple(v) for v in tokens['train']['input_ids']}&protected:raise ValueError('Token exposure overlap')
    for split,values in rows.items():(output/(split+'.jsonl.gz')).write_bytes((data/(split+'.jsonl.gz')).read_bytes())
    (output/'training-history.jsonl.gz').write_bytes((data/'training-history.jsonl.gz').read_bytes())
    (output/'overlap-audit.json').write_text(json.dumps(dict(protected_token_overlap=0,fresh_token_overlap=0,new_training_inputs=True,inherits121_history=True,hypotheses=list(HYPOTHESES)),indent=2)+'\n')
    torch.set_num_threads(4);torch.manual_seed(1123);rng=np.random.default_rng(1123)
    truth={s:np.array([int(r['review']) if 'review' in r else target(r) for r in values]) for s,values in rows.items()}
    factor_truth=np.array([v for r in train for v in targets(r)])
    assert all(int(any(targets(r)))==int(y) for r,y in zip(train,truth['train']))
    model=AutoModelForSequenceClassification.from_pretrained(a.base,local_files_only=True,trust_remote_code=False,use_safetensors=True).float().to(a.device)
    model.load_state_dict(load_file(str(parent/'selected-model.safetensors')),strict=True)
    for parameter in model.parameters():parameter.requires_grad_(False)
    for module in [*model.base_model.encoder.layer[-4:],model.pooler,model.classifier]:
        for parameter in module.parameters():parameter.requires_grad_(True)
    def batch(split,indices):return {k:v.to(a.device) for k,v in tokenizer.pad([{k:v[i] for k,v in tokens[split].items()} for i in indices],return_tensors='pt',pad_to_multiple_of=8).items()}
    def score(split):
        model.eval();values=[]
        with torch.inference_mode():
            for start in range(0,2*len(rows[split]),8):
                values.extend(torch.softmax(model(**batch(split,list(range(start,min(start+8,2*len(rows[split])))))).logits.float(),dim=-1)[:,0].cpu().tolist())
                if a.device=='mps' and start%200==0:torch.mps.empty_cache()
        values=np.array(values).reshape(-1,2)
        if not np.isfinite(values).all():raise ValueError('Nonfinite scores')
        return values
    def predictions(split,scores,limit):
        return [dict(case_id=r['case_id'],language=r['language'],gold=int(y),support_score=float(v[0]),revision_score=float(v[1]),score=float(max(v)),predicted=int(v[0]>=limit['support'] or v[1]>=limit['revision'])) for r,y,v in zip(rows[split],truth[split],scores)]
    def metrics(split,scores,limit):
        values=predictions(split,scores,limit)
        if split in ('ro_dev','uk_dev','ro_cal','uk_cal'):
            return {name:counts([v for r,v in zip(rows[split],values) if (r.get('source')=='v7_training_frame_projection')==flag]) for name,flag in [('formal',False),('conversational',True)]}
        return {'all':counts(values)}
    selection_splits=['ro_dev','uk_dev','ro_scope_dev','uk_scope_dev','uk_stress'];best=None;chosen=None
    report=dict(format='roguard-reason-coverage-training-v1',base=MODEL_ID,revision=REVISION,hypotheses=list(HYPOTHESES),training_rows=len(train),training_queries=2*len(train),parent_artifact_sha256=record['artifact_sha256'],parent_selection_sha256=digest(output/'parent-selection.json'),source_freeze_sha256=digest(STUDY/'source-freeze.json'),epochs=[],synthetic_only=True,independently_annotated=False,seed=1123,training_epochs=2,learning_rate=5e-6,threshold_selection="exact-two-reason-full-coverage-development-frontier",neutral_probability_retained=True,test_used_for_selection=False,exposed115_used_for_selection=False)
    def validate(epoch):
        nonlocal best,chosen
        scores={s:score(s) for s in selection_splits};thresholds={}
        for lang in ('ro','uk'):
            splits=[s for s in selection_splits if s.startswith(lang)]
            from training.entailment_frontier_v9 import frontier
            population=[]
            for split in splits:
                for row,y,values in zip(rows[split],truth[split],scores[split]):
                    group=(('conversational' if row.get('source')=='v7_training_frame_projection' else 'formal') if split==lang+'_dev' else split)
                    population.append(dict(gold=int(y),support_score=float(values[0]),revision_score=float(values[1]),group=group))
            optimum=frontier(population)
            thresholds[lang]=dict(support=optimum['support_threshold'],revision=optimum['revision_threshold'])
        groups={s:metrics(s,v,thresholds[s[:2]]) for s,v in scores.items()};flat=[g for values in groups.values() for g in values.values()]
        rank=(max(g['false_review_rate'] for g in flat),sum(g['false_review_rate'] for g in flat))
        entry=dict(epoch=epoch,thresholds=thresholds,metrics=groups,rank=rank);report['epochs'].append(entry);print(json.dumps(entry),flush=True)
        checkpoint=output/('epoch-'+str(epoch)+'.safetensors');save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(checkpoint))
        if best is None or rank<best:
            best=rank;chosen=entry;report['selected_epoch']=epoch
            import shutil
            shutil.copyfile(checkpoint,output/'selected-model.safetensors')
            artifact=dict(format='roguard-reason-entailment-artifact-v1',base=MODEL_ID,revision=REVISION,thresholds=thresholds,hypotheses=list(HYPOTHESES),checkpoint_sha256=digest(output/'selected-model.safetensors'),base_hashes=BASE_HASHES,score='separate-entailment-thresholds-including-neutral',preprocessing='normalized-prefix-and-focus-json-v1')
            (output/'review.json').write_text(json.dumps(artifact,indent=2)+'\n')
            for s,v in scores.items():(output/(s+'-predictions.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in predictions(s,v,thresholds[s[:2]])))
        (output/'development.json').write_text(json.dumps(report,indent=2)+'\n')
    validate(0)
    strata=[(r['language'],task,int(targets(r)[task])) for r in train for task in range(2)]
    frequencies=Counter(strata)
    if len(frequencies)!=8:raise ValueError('Missing language/task/class stratum')
    weights=np.array([1/frequencies[g] for g in strata]);weights/=weights.mean()
    # Pair batches keep both hypotheses for each row, and complete counterfactual pairs together.
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=5e-6,weight_decay=.01);model.train()
    for epoch in range(2):
        batches=plan_batches(train,rng)
        for step,(indices,pairs) in enumerate(batches):
            warm=max(1,int(.05*len(batches)));scale=min((step+1)/warm,max(.1,(len(batches)-step)/(len(batches)-warm)))
            for group in optimizer.param_groups:group['lr']=5e-6*scale
            optimizer.zero_grad(set_to_none=True);logits=model(**batch('train',indices)).logits.float();labels=torch.tensor([0 if factor_truth[i] else 2 for i in indices],device=a.device)
            losses=torch.nn.functional.cross_entropy(logits,labels,reduction='none');w=torch.tensor(weights[indices],dtype=logits.dtype,device=a.device);loss=(losses*w).mean()
            if pairs:
                margin=logits[:,0]-torch.logsumexp(logits[:,1:],dim=-1)
                loss=loss+.2*torch.nn.functional.softplus(1.-(margin[[p for n,p in pairs]]-margin[[n for n,p in pairs]])).mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
            if a.device=='mps' and step%25==0:torch.mps.empty_cache()
            if step%100==0:print('step',step,'/',len(batches),'loss',float(loss.detach().cpu()),flush=True)
            if step+1 in {len(batches)//2,len(batches)}:validate(epoch+(.5 if step+1<len(batches) else 1));model.train()
    model.load_state_dict(load_file(str(output/'selected-model.safetensors')),strict=True);audits={}
    for split in ('ro_cal','uk_cal','ro_exposed','uk_exposed'):
        scores=score(split);limit=chosen['thresholds'][split[:2]];audits[split]=metrics(split,scores,limit)
        (output/(split+'-predictions.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in predictions(split,scores,limit)))
    passed=all(g['passes'] for groups in chosen['metrics'].values() for g in groups.values()) and all(g['passes'] for groups in audits.values() for g in groups.values())
    report.update(complete=True,audit=audits,counts_pass=passed,fresh_test_not_inferred=True,stable_release_ready=False,finished_utc=datetime.now(timezone.utc).isoformat(),artifact_sha256=digest(output/'review.json'))
    (output/'development.json').write_text(json.dumps(report,indent=2)+'\n');print('FINISHED',json.dumps(audits),flush=True)


if __name__=='__main__':main()
