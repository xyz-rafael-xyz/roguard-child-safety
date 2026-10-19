"""Transfer an entailment encoder with paired scope supervision in both languages."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import gzip
import json
from pathlib import Path
import time
from training.train_bare_distress_v9 import read,digest,texts
from training.verify_candidate_pipeline_v9 import counts

STUDY=Path('research/v9-stabilization/paired-nli-119')
HYPOTHESIS='In the focal message, a real person expresses distress, a need for emotional support or a safety concern, including a withdrawal of an earlier disclosure.'


def premise(row):
    focus=row.get('focus',len(row['turns'])-1)
    from roguard.semantic import normalize_semantic_text
    return json.dumps(dict(previous_messages=[dict(t,text=normalize_semantic_text(t['text'])) for t in row['turns'][:focus]],focal_message=dict(row['turns'][focus],text=normalize_semantic_text(row['turns'][focus]['text']))),ensure_ascii=False,separators=(',',':'))


def loss_function(logits,labels,pairs=()):
    import torch
    import torch.nn.functional as F
    targets=torch.where(labels.bool(),0,2)
    loss=F.cross_entropy(logits,targets)
    if pairs:
        margin=logits[:,0]-torch.logsumexp(logits[:,1:],dim=-1)
        loss=loss+.2*F.softplus(1.-(margin[[p for n,p in pairs]]-margin[[n for n,p in pairs]])).mean()
    return loss


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--device',default='mps');a=p.parse_args()
    for path,expected in json.loads((STUDY/'source-freeze.json').read_text()).items():
        if digest(Path(path))!=expected:raise ValueError('Frozen source differs: '+path)
    import numpy as np
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    from safetensors.torch import save_file,load_file
    from roguard.nli_v21 import verify_base,MODEL_ID,REVISION,BASE_HASHES
    from roguard.native_assets import reject_native_overrides
    from training.prepare_instruction_sft_v8 import target
    from training.conservative_fusion_v9 import full_coverage_gap
    verify_base(a.base);reject_native_overrides(a.base,BASE_HASHES)
    if a.base.name!=REVISION:raise ValueError('Revision differs')
    torch.set_num_threads(4);torch.manual_seed(1119);rng=np.random.default_rng(1119)
    output=STUDY/'candidate';output.mkdir(exist_ok=False)
    data=Path('research/v9-stabilization/paired-scope-117')
    parents={'ro':Path('research/v9-stabilization/native-full-22-ro'),'uk':Path('research/v9-stabilization/bare-distress-95/uk')}
    rows={};fresh=[];history=[];authored=[]
    for lang in ('ro','uk'):
        for split in ('dev','cal'):rows[lang+'_'+split]=read(parents[lang]/(split+'.jsonl.gz'))
        rows[lang+'_scope_dev']=[json.loads(l) for l in (data/'data'/(lang+'-dev.jsonl')).read_text().splitlines()]
        # Exposed115 is audit only. Training files from117 contain both languages after completion.
        rows[lang+'_exposed']=read(data/lang/'exposed.jsonl.gz')
        history+=read(parents[lang]/'training-history.jsonl.gz')
        authored += [json.loads(l) for l in (data/'data'/(lang+'-train.jsonl')).read_text().splitlines()]
        fresh += [json.loads(l) for l in (data/'data'/(lang+'-test.jsonl')).read_text().splitlines()]
    from training.needs_repair_data_v9 import generate as needs
    rows['uk_stress']=needs('dev')
    tokenizer=AutoTokenizer.from_pretrained(a.base,local_files_only=True,trust_remote_code=False)
    if tokenizer.padding_side!='right':raise ValueError('Right padding required')
    def encode(values):
        result=tokenizer([premise(r) for r in values],[HYPOTHESIS]*len(values),truncation=False,padding=False)
        if any(len(v)>512 for v in result['input_ids']):raise ValueError('Input too long, no truncation')
        return result
    def keys(values):return {tuple(v) for v in encode(values)['input_ids']}
    protected_keys=keys(fresh+[r for values in rows.values() for r in values]);fresh_keys=keys(fresh)
    # No prior project adaptation is inherited by this model. Historical data is a pool, not a checkpoint lineage.
    pool={}
    for row,ids in zip(history,encode(history)['input_ids']):
        key=tuple(ids)
        if key in protected_keys:continue
        if key in pool and target(pool[key])!=target(row):raise ValueError('Contradictory replay labels')
        pool[key]=row
    train=[];seen=set()
    for row,ids in zip(authored,encode(authored)['input_ids']):
        key=tuple(ids)
        if key in protected_keys or key in seen:raise ValueError('Authored overlap')
        seen.add(key);train.append(row)
    authored_count=len(train)
    for lang in ('ro','uk'):
        for native in (False,True):
            for gold in (0,1):
                candidates=[r for key,r in pool.items() if key not in seen and r['language']==lang and (r.get('source')=='v7_training_frame_projection')==native and target(r)==gold]
                if len(candidates)<512:raise ValueError('Missing balanced replay stratum')
                for i in rng.choice(len(candidates),512,replace=False):
                    row=candidates[int(i)];train.append(row)
                    seen.update(keys([row]))
    if keys(train)&protected_keys:raise ValueError('Training leakage')
    rows['train']=train
    (output/'overlap-audit.json').write_text(json.dumps(dict(training_protected_token_overlap=0,training_fresh_test_token_overlap=0,project_checkpoint_inherited=False,training_rows=len(train),authored_rows=authored_count),indent=2)+'\n')
    for split,values in rows.items():(output/(split+'.jsonl.gz')).write_bytes(gzip.compress(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in values).encode(),mtime=0))
    (output/'training-history.jsonl.gz').write_bytes((output/'train.jsonl.gz').read_bytes())
    tokens={s:encode(v) for s,v in rows.items()};truth={s:np.array([int(r['review']) if 'review' in r else target(r) for r in v]) for s,v in rows.items()}
    model=AutoModelForSequenceClassification.from_pretrained(a.base,local_files_only=True,trust_remote_code=False,use_safetensors=True).float().to(a.device)
    if model.config.id2label!={0:'entailment',1:'neutral',2:'contradiction'}:raise ValueError('NLI labels differ')
    for parameter in model.parameters():parameter.requires_grad_(False)
    for module in [*model.base_model.encoder.layer[-4:],model.pooler,model.classifier]:
        for parameter in module.parameters():parameter.requires_grad_(True)
    def batch(split,indices):return {k:v.to(a.device) for k,v in tokenizer.pad([{k:v[i] for k,v in tokens[split].items()} for i in indices],return_tensors='pt',pad_to_multiple_of=8).items()}
    def score(split):
        model.eval();values=[]
        with torch.inference_mode():
            for start in range(0,len(rows[split]),8):
                values.extend(torch.softmax(model(**batch(split,list(range(start,min(start+8,len(rows[split])))))).logits.float(),dim=-1)[:,0].cpu().tolist())
                if a.device=='mps' and start%200==0:torch.mps.empty_cache()
        if not np.isfinite(values).all():raise ValueError('Nonfinite inference')
        return values
    def records(split,scores,cutoff):return [dict(case_id=r['case_id'],language=r['language'],gold=int(y),score=float(s),predicted=int(s>=cutoff)) for r,y,s in zip(rows[split],truth[split],scores)]
    def metrics(split,scores,cutoff):
        values=records(split,scores,cutoff)
        if split.endswith(('_dev','_cal')) and not split.endswith('scope_dev'):
            return {kind:counts([v for r,v in zip(rows[split],values) if (r.get('source')=='v7_training_frame_projection')==flag]) for kind,flag in [('formal',False),('conversational',True)]}
        return {'all':counts(values)}
    selected_splits=['ro_dev','ro_scope_dev','uk_dev','uk_scope_dev','uk_stress'];best=None;chosen=None
    report=dict(format='roguard-paired-nli-training-v1',base=MODEL_ID,revision=REVISION,hypothesis=HYPOTHESIS,source_freeze_sha256=digest(STUDY/'source-freeze.json'),epochs=[],synthetic_only=True,independently_annotated=False,training_rows=len(train),authored_rows=authored_count,seed=1119,learning_rate=1e-5,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),neutral_probability_retained=True,test_used_for_selection=False,exposed115_used_for_selection=False)
    def validate(epoch):
        nonlocal best,chosen
        scores={s:score(s) for s in selected_splits};thresholds={}
        for lang in ('ro','uk'):
            splits=[s for s in selected_splits if s.startswith(lang)]
            thresholds[lang]=full_coverage_gap([int(v) for s in splits for v in truth[s]],[v for s in splits for v in scores[s]])
        groups={s:metrics(s,v,thresholds[s[:2]]) for s,v in scores.items()};flat=[g for values in groups.values() for g in values.values()]
        rank=(max(g['false_review_rate'] for g in flat),sum(g['false_review_rate'] for g in flat))
        entry=dict(epoch=epoch,thresholds=thresholds,metrics=groups,rank=rank);report['epochs'].append(entry);print(json.dumps(entry),flush=True)
        checkpoint=output/('epoch-'+str(epoch)+'.safetensors');save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(checkpoint))
        if best is None or rank<best:
            best=rank;chosen=entry;report['selected_epoch']=epoch
            import shutil
            shutil.copyfile(checkpoint,output/'selected-model.safetensors')
            spec=dict(format='roguard-paired-nli-artifact-v1',base=MODEL_ID,revision=REVISION,thresholds=thresholds,hypothesis=HYPOTHESIS,checkpoint_sha256=digest(output/'selected-model.safetensors'),base_hashes=BASE_HASHES,score='softmax-entailment-including-neutral',preprocessing='normalized-prefix-and-focus-json-v1')
            (output/'review.json').write_text(json.dumps(spec,indent=2)+'\n')
            for s,v in scores.items():(output/(s+'-predictions.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in records(s,v,thresholds[s[:2]])))
        (output/'development.json').write_text(json.dumps(report,indent=2)+'\n')
    groups=defaultdict(list)
    for i,r in enumerate(train[:authored_count]):
        if r.get('pair_group'):groups[r['pair_group']].append(i)
    pairs=[];paired=set()
    for group in groups.values():
        if len(group)!=2 or sorted(truth['train'][i] for i in group)!=[0,1]:raise ValueError('Malformed pair')
        pair=sorted(group,key=lambda i:truth['train'][i]);pairs.append(pair);paired.update(pair)
    singles=[i for i in range(len(train)) if i not in paired]
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5,weight_decay=.01)
    total_steps=2*((len(pairs)+3)//4+(len(singles)+7)//8);step=0
    # Starting base is a diagnostic control, never selected without the same development checks.
    validate(0)
    for epoch in (1,2):
        shuffled=[pairs[int(i)] for i in rng.permutation(len(pairs))];batches=[]
        for offset in range(0,len(shuffled),4):
            selected=shuffled[offset:offset+4];batches.append(([i for pair in selected for i in pair],[(i*2,i*2+1) for i in range(len(selected))]))
        order=rng.permutation(singles).tolist();batches.extend((order[i:i+8],[]) for i in range(0,len(order),8));rng.shuffle(batches)
        model.train();start=time.monotonic()
        for bi,(indices,pair_ix) in enumerate(batches):
            warm=max(1,int(.05*total_steps));scale=min((step+1)/warm,max(.1,(total_steps-step)/(total_steps-warm)))
            for group in optimizer.param_groups:group['lr']=1e-5*scale
            optimizer.zero_grad(set_to_none=True);logits=model(**batch('train',indices)).logits.float()
            loss=loss_function(logits,torch.tensor(truth['train'][indices],device=a.device),pair_ix)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();step+=1
            if a.device=='mps' and bi%25==0:torch.mps.empty_cache()
            if bi%100==0:print('epoch',epoch,'batch',bi,'/',len(batches),'loss',float(loss.detach().cpu()),flush=True)
            if bi+1 in {len(batches)//2,len(batches)}:
                validate(epoch-.5 if bi+1<len(batches) else epoch);model.train()
        print('epoch_seconds',time.monotonic()-start,flush=True)
    model.load_state_dict(load_file(str(output/'selected-model.safetensors')),strict=True);audits={}
    for split in ('ro_cal','uk_cal','ro_exposed','uk_exposed'):
        scores=score(split);cutoff=chosen['thresholds'][split[:2]];audits[split]=metrics(split,scores,cutoff)
        (output/(split+'-predictions.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in records(split,scores,cutoff)))
    passed=all(g['passes'] for groups in chosen['metrics'].values() for g in groups.values()) and all(g['passes'] for groups in audits.values() for g in groups.values())
    report.update(complete=True,audit=audits,counts_pass=passed,fresh_test_not_inferred=True,stable_release_ready=False,finished_utc=datetime.now(timezone.utc).isoformat(),artifact_sha256=digest(output/'review.json'))
    (output/'development.json').write_text(json.dumps(report,indent=2)+'\n');print('FINISHED',json.dumps(audits),flush=True)


if __name__=='__main__':main()
