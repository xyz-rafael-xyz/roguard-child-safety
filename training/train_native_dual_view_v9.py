"""Fresh pretrained native encoder with complete-content and quote-scope views."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import gzip
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time
from training.train_bare_distress_v9 import BASES,digest,read
from training.native_quote_view_v9 import PREPROCESSING,texts
from training.verify_candidate_pipeline_v9 import counts

STUDY=Path('research/v9-stabilization/native-dual-view-150')
PARENTS={lang:Path('research/v9-stabilization/native-scope-operators-131')/lang for lang in ('ro','uk')}


def paired_loss(logits,labels,pairs=()):
    import torch.nn.functional as F
    loss=F.cross_entropy(logits,labels)
    if pairs:
        margin=logits[:,1]-logits[:,0]
        # Pair order is explicitly negative, positive; no dependence on row sorting.
        loss=loss+.2*F.softplus(1.-(margin[[p for n,p in pairs]]-margin[[n for n,p in pairs]])).mean()
    return loss


def main():
    p=argparse.ArgumentParser();p.add_argument('--language',choices=BASES,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--device',default='mps');a=p.parse_args()
    for path,expected in json.loads((STUDY/'source-freeze.json').read_text()).items():
        if digest(Path(path))!=expected:raise ValueError('Frozen source differs: '+path)
    for name,expected in json.loads((STUDY/'software.json').read_text()).items():
        if version(name)!=expected:raise ValueError('Dependency version differs: '+name)
    import numpy as np
    import torch
    from safetensors.torch import load_file,save_file
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    from training.prepare_instruction_sft_v8 import target
    from training.budgeted_native_frontier_v9 import frontier
    from roguard.native_assets import PADDING_MULTIPLE,reject_native_overrides
    lang=a.language;parent=PARENTS[lang];output=STUDY/lang;output.mkdir(exist_ok=False)
    model_id,revision,weight_name,weight_hash=BASES[lang]
    if a.base.name!=revision or digest(a.base/weight_name)!=weight_hash:raise ValueError('Base identity differs')
    manifest_path=Path('research/v9-stabilization/native-full-22-base-manifest.json');manifest=json.loads(manifest_path.read_text())['models'][lang]
    reject_native_overrides(a.base,manifest['files'])
    for name,entry in manifest['files'].items():
        if digest(a.base/name)!=entry['sha256']:raise ValueError('Base/tokenizer changed')
    torch.set_num_threads(4);torch.manual_seed(1150);rng=np.random.default_rng(1150)
    tokenizer=AutoTokenizer.from_pretrained(a.base,local_files_only=True,trust_remote_code=False)
    data=parent
    rows={split:read(parent/(split+'.jsonl.gz')) for split in ('dev','cal','scope_dev','exposed')}
    if lang=='uk':rows['stress']=read(parent/'stress.jsonl.gz')
    fresh_path=Path('research/v9-stabilization/paired-scope-117/data')/(lang+'-test.jsonl')
    fresh=[json.loads(line) for line in fresh_path.read_text().splitlines()]
    protected=[r for values in rows.values() for r in values]+fresh
    (output/'initialization.json').write_text(json.dumps(dict(recorded_utc=datetime.now(timezone.utc).isoformat(),base=model_id,revision=revision,base_weight_sha256=weight_hash,initialization='Original pinned pretrained encoder plus new random binary head; no prior task checkpoint',parent_training_input_sha256=digest(parent/'train.jsonl.gz'),preprocessing=PREPROCESSING,seed=1150),indent=2)+'\n')
    def encode(values):
        pairs=[texts(r) for r in values]
        result=tokenizer([p[0] for p in pairs],[p[1] for p in pairs],padding=False,truncation=False)
        if any(len(ids)>512 for ids in result['input_ids']):raise ValueError('No silent truncation')
        return result
    def keys(values):return {tuple(v) for v in encode(values)['input_ids']}
    protected_keys=keys(protected)
    history=[]  # This run does not inherit any previous task-trained checkpoint.
    candidates=read(parent/'train.jsonl.gz')
    for name in ('indirect-distress-135','freeform-scope-136'):
        candidates+=read(Path('research/v9-stabilization')/name/(lang+'-train.jsonl.gz'))
    if any(r['language']!=lang for r in candidates):raise ValueError('Training language differs')
    train=[];dedup={};excluded=[]
    for row,ids in zip(candidates,encode(candidates)['input_ids']):
        key=tuple(ids);gold=target(row)
        if key in protected_keys:raise ValueError('Training overlaps protected native-token inputs')
        if key in dedup:
            if dedup[key]!=gold:raise ValueError('Contradictory duplicate')
            excluded.append(row['case_id']);continue
        dedup[key]=gold;train.append(row)
    authored_count=sum(r.get('source') in ('paired_scope_v9','revision_contrasts_v9','morphology_scope_v9','scope_operators_v9','indirect_distress_v9','freeform_scope_v9') for r in train)
    expected=json.loads((STUDY/'protocol.json').read_text())['training_rows'][lang]
    if len(train)!=expected:raise ValueError('Training population differs from preflight')
    rows['train']=train
    all_history={texts(r):r for r in history+train}
    (output/'training-history.jsonl.gz').write_bytes(gzip.compress(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in all_history.values()).encode(),mtime=0))
    for split,values in rows.items():
        (output/(split+'.jsonl.gz')).write_bytes(gzip.compress(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in values).encode(),mtime=0))
    (output/'overlap-audit.json').write_text(json.dumps(dict(inherited_task_checkpoint=False,training_protected_token_overlap=0,excluded_duplicates=excluded,history_rows=len(history),training_rows=len(train)),indent=2)+'\n')
    tokens={s:encode(v) for s,v in rows.items()}
    truth={s:np.array([int(r['review']) if 'review' in r else target(r) for r in v]) for s,v in rows.items()}
    model=AutoModelForSequenceClassification.from_pretrained(a.base,num_labels=2,local_files_only=True,trust_remote_code=False,attn_implementation='eager',use_safetensors=weight_name.endswith('.safetensors'),weights_only=True).float().to(a.device)
    from training.native_full_optimizer_v9 import parameter_groups,schedule
    optimization_groups=parameter_groups(model,top_lr=2e-5,decay=.9)
    def batch(split,indices):return {k:v.to(a.device) for k,v in tokenizer.pad([{k:v[i] for k,v in tokens[split].items()} for i in indices],return_tensors='pt',pad_to_multiple_of=PADDING_MULTIPLE).items()}
    def score(split):
        model.eval();values=[]
        with torch.inference_mode():
            for start in range(0,len(rows[split]),8):
                values.extend(torch.softmax(model(**batch(split,list(range(start,min(start+8,len(rows[split])))))).logits.float(),dim=-1)[:,1].cpu().tolist())
                if a.device=='mps' and start%200==0:torch.mps.empty_cache()
        if not np.isfinite(values).all():raise ValueError('Nonfinite scores')
        return values
    def records(split,scores,cutoff):return [dict(case_id=r['case_id'],language=lang,gold=int(y),score=float(s),predicted=int(s>=cutoff)) for r,y,s in zip(rows[split],truth[split],scores)]
    def metrics(split,scores,cutoff):
        values=records(split,scores,cutoff)
        if split in ('dev','cal'):
            return {kind:counts([v for r,v in zip(rows[split],values) if (r.get('source')=='v7_training_frame_projection')==flag]) for kind,flag in [('formal',False),('conversational',True)]}
        return {'all':counts(values)}
    selection_splits=['dev','scope_dev']+(['stress'] if lang=='uk' else [])
    report=dict(format='roguard-native-dual-view-training-v1',language=lang,synthetic_only=True,independently_annotated=False,exposed115_diagnosis_informed_design=True,exposed115_used_for_selection=False,test_used_for_selection=False,pretrained_initialization=True,preprocessing=PREPROCESSING,initialization_sha256=digest(output/'initialization.json'),source_freeze_sha256=digest(STUDY/'source-freeze.json'),training_rows=len(train),authored_rows=authored_count,full_encoder_adaptation=True,layer_decay=.9,optimizer_groups=[dict(name=g['name'],base_lr=g['base_lr'],parameters=sum(p.numel() for p in g['params'])) for g in optimization_groups],minimum_development_recall=.99,selection='worst-group false reviews, sum rates, then fewer misses; each group recall>=.99',epochs=[],learning_rate=2e-5,seed=1150,objective='CE plus0.2softplus(1-positive_margin+negative_margin) on counterfactual pairs; CE on replay',trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    best=None;chosen=None
    def validate(epoch):
        nonlocal best,chosen
        scores={s:score(s) for s in selection_splits}
        population=[]
        for split in selection_splits:
            for row,y,value in zip(rows[split],truth[split],scores[split]):
                group=('conversational' if row.get('source')=='v7_training_frame_projection' else 'formal') if split=='dev' else split
                population.append(dict(gold=int(y),score=float(value),group=group))
        optimum=frontier(population)
        cutoff=optimum['threshold']
        groups={s:metrics(s,scores[s],cutoff) for s in selection_splits}
        flat=[g for split in groups.values() for g in split.values()]
        rank=(max(g['false_review_rate'] for g in flat),sum(g['false_review_rate'] for g in flat),sum(g['missed'] for g in flat))
        entry=dict(epoch=epoch,threshold=cutoff,metrics=groups,rank=rank)
        report['epochs'].append(entry);print(json.dumps(entry),flush=True)
        # Keep every checked tensor checkpoint for audit; selection uses development only.
        checkpoint=output/('epoch-'+str(epoch)+'.safetensors')
        save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(checkpoint))
        if best is None or rank<best:
            best=rank;chosen=entry;report['selected_epoch']=epoch
            import shutil
            shutil.copyfile(checkpoint,output/'selected-model.safetensors')
            spec=dict(format='roguard-native-dual-view-review-experiment-v1',base=model_id,revision=revision,language=lang,thresholds={lang:cutoff},checkpoint_sha256=digest(output/'selected-model.safetensors'),base_manifest_sha256=digest(manifest_path),preprocessing=PREPROCESSING)
            (output/'review.json.gz').write_bytes(gzip.compress(json.dumps(spec,separators=(',',':')).encode(),mtime=0))
            for s,v in scores.items():(output/(s+'-predictions.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in records(s,v,cutoff)))
        (output/'development.json').write_text(json.dumps(report,indent=2)+'\n')
    validate(0)
    pair_groups=defaultdict(list)
    for i,r in enumerate(train):
        if r.get('source') in ('paired_scope_v9','revision_contrasts_v9','morphology_scope_v9','scope_operators_v9','indirect_distress_v9','freeform_scope_v9') and r.get('pair_group'):pair_groups[r['pair_group']].append(i)
    pairs=[];paired_indices=set()
    for indices in pair_groups.values():
        if len(indices)!=2:continue
        if sorted(int(truth['train'][i]) for i in indices)!=[0,1]:raise ValueError('Invalid contrastive pair')
        pair=sorted(indices,key=lambda i:truth['train'][i]);pairs.append(pair);paired_indices.update(pair)
    singles=[i for i in range(len(train)) if i not in paired_indices]
    optimizer=torch.optim.AdamW(optimization_groups,weight_decay=.01)
    total_steps=2*((len(pairs)+3)//4+(len(singles)+7)//8);step=0
    for epoch in (1,2):
        shuffled=[pairs[int(i)] for i in rng.permutation(len(pairs))]
        batches=[]
        for offset in range(0,len(shuffled),4):
            selected=shuffled[offset:offset+4];batches.append(([i for pair in selected for i in pair],[(i*2,i*2+1) for i in range(len(selected))]))
        order=rng.permutation(singles).tolist()
        batches.extend((order[i:i+8],[]) for i in range(0,len(order),8))
        rng.shuffle(batches);model.train();start=time.monotonic();losses=[]
        for bi,(indices,pair_ix) in enumerate(batches):
            warm=max(1,int(.05*total_steps));scale=min((step+1)/warm,max(.1,(total_steps-step)/(total_steps-warm)))
            schedule(optimizer,scale)
            optimizer.zero_grad(set_to_none=True)
            loss=paired_loss(model(**batch('train',indices)).logits.float(),torch.tensor(truth['train'][indices],device=a.device),pair_ix)
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();step+=1;losses.append(float(loss.detach().cpu()))
            if a.device=='mps' and bi%25==0:torch.mps.empty_cache()
            if bi%100==0:print(lang,'epoch',epoch,'batch',bi,'/',len(batches),'loss',losses[-1],flush=True)
            if bi+1 in {len(batches)//2,len(batches)}:
                validate(epoch-.5 if bi+1<len(batches) else epoch);model.train()
        print('epoch_seconds',time.monotonic()-start,flush=True)
    model.load_state_dict(load_file(str(output/'selected-model.safetensors')),strict=True)
    threshold=chosen['threshold'];audits={}
    for split in ('cal','exposed'):
        scores=score(split);audits[split]=metrics(split,scores,threshold)
        (output/(split+'-predictions.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in records(split,scores,threshold)))
    passed=all(g['passes'] for group in chosen['metrics'].values() for g in group.values()) and all(g['passes'] for group in audits.values() for g in group.values())
    report.update(complete=True,finished_utc=datetime.now(timezone.utc).isoformat(),artifact_sha256=digest(output/'review.json.gz'),audit=audits,counts_pass=passed,stable_release_ready=False,fresh_test_not_inferred=True)
    (output/'development.json').write_text(json.dumps(report,indent=2)+'\n');print('FINISHED',json.dumps(report['audit']),flush=True)


if __name__=='__main__':main()
