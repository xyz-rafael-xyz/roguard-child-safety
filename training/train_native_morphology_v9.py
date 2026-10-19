"""Native continuation adding the frozen balanced grammatical-gender contrasts."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import time
from training.train_bare_distress_v9 import BASES,digest,texts,read
from training.verify_candidate_pipeline_v9 import counts

STUDY=Path('research/v9-stabilization/native-morphology-128')
PARENTS={lang:Path('research/v9-stabilization/native-coverage-125')/lang for lang in ('ro','uk')}


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
    import numpy as np
    import torch
    from safetensors.torch import load_file,save_file
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    from training.prepare_instruction_sft_v8 import target
    from training.conservative_fusion_v9 import full_coverage_gap
    from roguard.native_assets import PREPROCESSING,PADDING_MULTIPLE,reject_native_overrides
    lang=a.language;parent=PARENTS[lang];output=STUDY/lang;output.mkdir(exist_ok=False)
    model_id,revision,weight_name,weight_hash=BASES[lang]
    if a.base.name!=revision or digest(a.base/weight_name)!=weight_hash:raise ValueError('Base identity differs')
    manifest_path=Path('research/v9-stabilization/native-full-22-base-manifest.json');manifest=json.loads(manifest_path.read_text())['models'][lang]
    reject_native_overrides(a.base,manifest['files'])
    for name,entry in manifest['files'].items():
        if digest(a.base/name)!=entry['sha256']:raise ValueError('Base/tokenizer changed')
    torch.set_num_threads(4);torch.manual_seed(1128);rng=np.random.default_rng(1128)
    tokenizer=AutoTokenizer.from_pretrained(a.base,local_files_only=True,trust_remote_code=False)
    data=Path('research/v9-stabilization/morphology-coverage-127')
    rows={split:read(parent/(split+'.jsonl.gz')) for split in ('dev','cal','scope_dev','exposed')}
    if lang=='uk':rows['stress']=read(parent/'stress.jsonl.gz')
    fresh_path=Path('research/v9-stabilization/paired-scope-117/data')/(lang+'-test.jsonl')
    fresh=[json.loads(line) for line in fresh_path.read_text().splitlines()]
    protected=[r for values in rows.values() for r in values]+fresh
    parent_spec=json.loads(gzip.decompress((parent/'review.json.gz').read_bytes()))
    parent_report=json.loads((parent/'development.json').read_text())
    if parent_report.get('complete') is not True or parent_report.get('artifact_sha256')!=digest(parent/'review.json.gz'):raise ValueError('Incomplete parent artifact')
    if digest(parent/'selected-model.safetensors')!=parent_spec['checkpoint_sha256']:raise ValueError('Parent checkpoint differs')
    (output/'parent-selection.json').write_text(json.dumps(dict(recorded_utc=datetime.now(timezone.utc).isoformat(),parent_artifact_sha256=digest(parent/'review.json.gz'),parent_checkpoint_sha256=parent_spec['checkpoint_sha256'],parent_report_sha256=digest(parent/'development.json'),parent_training_input_sha256=digest(parent/'train.jsonl.gz'),added_training_input_sha256=digest(data/(lang+'-train.jsonl.gz'))),indent=2)+'\n')
    def encode(values):
        pairs=[texts(r) for r in values]
        result=tokenizer([p[0] for p in pairs],[p[1] for p in pairs],padding=False,truncation=False)
        if any(len(ids)>512 for ids in result['input_ids']):raise ValueError('No silent truncation')
        return result
    def keys(values):return {tuple(v) for v in encode(values)['input_ids']}
    protected_keys=keys(protected)
    history=read(parent/'training-history.jsonl.gz')
    fresh_keys=keys(fresh);historical_keys=keys(history)
    if fresh_keys&historical_keys:raise ValueError('Fresh test overlaps inherited training')
    if protected_keys&historical_keys:raise ValueError('Protected evaluation overlaps inherited training')
    candidates=read(parent/'train.jsonl.gz')+read(data/(lang+'-train.jsonl.gz'))
    if any(r['language']!=lang for r in candidates):raise ValueError('Training language differs')
    train=[];dedup={};excluded=[]
    for row,ids in zip(candidates,encode(candidates)['input_ids']):
        key=tuple(ids);gold=target(row)
        if key in protected_keys:raise ValueError('Training overlaps protected native-token inputs')
        if key in dedup:
            if dedup[key]!=gold:raise ValueError('Contradictory duplicate')
            excluded.append(row['case_id']);continue
        dedup[key]=gold;train.append(row)
    authored_count=sum(r.get('source') in ('paired_scope_v9','revision_contrasts_v9','morphology_scope_v9') for r in train)
    rows['train']=train
    all_history={texts(r):r for r in history+train}
    (output/'training-history.jsonl.gz').write_bytes(gzip.compress(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in all_history.values()).encode(),mtime=0))
    for split,values in rows.items():
        (output/(split+'.jsonl.gz')).write_bytes(gzip.compress(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in values).encode(),mtime=0))
    (output/'overlap-audit.json').write_text(json.dumps(dict(fresh_test_inherited_token_overlap=0,training_protected_token_overlap=0,excluded_duplicates=excluded,history_rows=len(history),training_rows=len(train)),indent=2)+'\n')
    tokens={s:encode(v) for s,v in rows.items()}
    truth={s:np.array([int(r['review']) if 'review' in r else target(r) for r in v]) for s,v in rows.items()}
    model=AutoModelForSequenceClassification.from_pretrained(a.base,num_labels=2,local_files_only=True,trust_remote_code=False,attn_implementation='eager',use_safetensors=weight_name.endswith('.safetensors'),weights_only=True).float().to(a.device)
    parent_spec=json.loads(gzip.decompress((parent/'review.json.gz').read_bytes()))
    if digest(parent/'selected-model.safetensors')!=parent_spec['checkpoint_sha256']:raise ValueError('Parent checkpoint differs')
    model.load_state_dict(load_file(str(parent/'selected-model.safetensors')),strict=True)
    for parameter in model.parameters():parameter.requires_grad_(False)
    for module in [*model.base_model.encoder.layer[-4:],model.classifier]:
        for parameter in module.parameters():parameter.requires_grad_(True)
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
    report=dict(format='roguard-native-morphology-training-v1',language=lang,synthetic_only=True,independently_annotated=False,exposed115_diagnosis_informed_design=True,exposed115_used_for_selection=False,test_used_for_selection=False,parent_artifact_sha256=digest(parent/'review.json.gz'),source_freeze_sha256=digest(STUDY/'source-freeze.json'),training_rows=len(train),authored_rows=authored_count,epochs=[],learning_rate=5e-6,seed=1128,objective='CE plus0.2softplus(1-positive_margin+negative_margin) on counterfactual pairs; CE on replay',trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    best=None;chosen=None
    def validate(epoch):
        nonlocal best,chosen
        scores={s:score(s) for s in selection_splits}
        cutoff=full_coverage_gap([int(y) for s in selection_splits for y in truth[s]],[v for s in selection_splits for v in scores[s]])
        groups={s:metrics(s,scores[s],cutoff) for s in selection_splits}
        flat=[g for split in groups.values() for g in split.values()]
        rank=(max(g['false_review_rate'] for g in flat),sum(g['false_review_rate'] for g in flat))
        entry=dict(epoch=epoch,threshold=cutoff,metrics=groups,rank=rank)
        report['epochs'].append(entry);print(json.dumps(entry),flush=True)
        # Keep every checked tensor checkpoint for audit; selection uses development only.
        checkpoint=output/('epoch-'+str(epoch)+'.safetensors')
        save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(checkpoint))
        if best is None or rank<best:
            best=rank;chosen=entry;report['selected_epoch']=epoch
            import shutil
            shutil.copyfile(checkpoint,output/'selected-model.safetensors')
            spec=dict(format='roguard-native-full-review-experiment-v1',base=model_id,revision=revision,language=lang,thresholds={lang:cutoff},checkpoint_sha256=digest(output/'selected-model.safetensors'),base_manifest_sha256=digest(manifest_path),preprocessing=PREPROCESSING)
            (output/'review.json.gz').write_bytes(gzip.compress(json.dumps(spec,separators=(',',':')).encode(),mtime=0))
            for s,v in scores.items():(output/(s+'-predictions.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in records(s,v,cutoff)))
        (output/'development.json').write_text(json.dumps(report,indent=2)+'\n')
    validate(0)
    pair_groups=defaultdict(list)
    for i,r in enumerate(train):
        if r.get('source') in ('paired_scope_v9','revision_contrasts_v9','morphology_scope_v9') and r.get('pair_group'):pair_groups[r['pair_group']].append(i)
    pairs=[];paired_indices=set()
    for indices in pair_groups.values():
        if len(indices)!=2:continue
        if sorted(int(truth['train'][i]) for i in indices)!=[0,1]:raise ValueError('Invalid contrastive pair')
        pair=sorted(indices,key=lambda i:truth['train'][i]);pairs.append(pair);paired_indices.update(pair)
    singles=[i for i in range(len(train)) if i not in paired_indices]
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=5e-6,weight_decay=.01)
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
            for group in optimizer.param_groups:group['lr']=5e-6*scale
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
