"""Full-scope Ukrainian repair with replay-only teacher distillation."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

from training.train_bare_distress_v9 import BASES, digest, texts, read, checkpoint_offsets, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', choices=tuple(BASES), required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='mps')
    args = parser.parse_args()
    import numpy as np
    import torch
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from training.prepare_instruction_sft_v8 import target
    from training.check_review_candidate_v8 import inspect
    for name,expected in json.loads(Path('research/v9-stabilization/scope-distillation-102/source-freeze.json').read_text()).items():
        if digest(Path(name))!=expected:raise ValueError('Frozen source changed: '+name)
    language = args.language
    model_id, revision, weight_name, weight_hash = BASES[language]
    if args.base.name != revision or digest(args.base/weight_name) != weight_hash:
        raise ValueError('Pinned base weight identity mismatch')
    base_manifest_path=Path('research/v9-stabilization/native-full-22-base-manifest.json')
    base_manifest=json.loads(base_manifest_path.read_text())['models'][language]
    if base_manifest['model']!=model_id or base_manifest['revision']!=revision:
        raise ValueError('Base manifest identity mismatch')
    from roguard.native_assets import reject_native_overrides, PADDING_MULTIPLE, PREPROCESSING
    reject_native_overrides(args.base,base_manifest['files'])
    for name,expected in base_manifest['files'].items():
        if digest(args.base/name)!=expected['sha256']:
            raise ValueError('Pinned base/tokenizer file differs: '+name)
    torch.set_num_threads(4); torch.manual_seed(1102)
    rng = np.random.default_rng(1102)
    root = Path('research/v8-stabilization')
    parent=Path('research/v9-stabilization/bare-distress-95/uk')
    teacher_parent=Path('research/v9-stabilization/needs-repair-85/uk')
    if language!='uk':raise ValueError('Study102 is Ukrainian-only')
    from training.needs_repair_data_v9 import generate as needs
    from training.scope_distillation_data_v9 import generate, anchored_loss
    from training.verify_reserved_native_v9 import PROBES
    from training.prepare_scope_repair_v9 import read as read_any
    original=read(parent/'training-history.jsonl.gz')
    replay_history=read(teacher_parent/'training-history.jsonl.gz')
    selected=rng.choice(len(replay_history),size=3072,replace=False).tolist()
    pool=[replay_history[i] for i in selected]+generate()
    rows={split:read(parent/(split+'.jsonl.gz')) for split in ('dev','cal')}
    rows['stress']=needs('dev')
    evaluation_keys={texts(r) for split in rows.values() for r in split}
    for filename,expected in PROBES.values():
        path=root/filename
        if digest(path)!=expected:raise ValueError('Reserved overlap-audit identity differs')
        evaluation_keys.update(texts(r) for r in read_any(path) if r['language']==language)
    seen={};excluded=[];train=[]
    for row in pool:
        key=texts(row)
        if key in evaluation_keys:
            excluded.append(row['case_id']);continue
        if key in seen:
            if seen[key]!=target(row):raise ValueError('Contradictory training labels')
            excluded.append(row['case_id']);continue
        seen[key]=target(row);train.append(row)
    rows['train']=train
    if {texts(r) for r in rows['dev']} & {texts(r) for r in rows['cal']}:
        raise ValueError('Development/calibration overlap after native normalization')
    args.output.mkdir(exist_ok=False)
    for split, values in rows.items():
        (args.output/(split+'.jsonl.gz')).write_bytes(gzip.compress(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in values).encode(),mtime=0))
    history={texts(r):r for r in original+train}
    (args.output/'training-history.jsonl.gz').write_bytes(gzip.compress(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in history.values()).encode(),mtime=0))
    (args.output/'training-exclusions.json').write_text(json.dumps(excluded,indent=2)+'\n')
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True, trust_remote_code=False)
    if tokenizer.padding_side != 'right':
        raise ValueError('Native training requires right padding')
    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=2, local_files_only=True,
                trust_remote_code=False, attn_implementation='eager', use_safetensors=weight_name.endswith('.safetensors'), weights_only=True).float().to(args.device)
    parent_spec=json.loads(gzip.decompress((parent/'review.json.gz').read_bytes()))
    if digest(parent/'selected-model.safetensors')!=parent_spec['checkpoint_sha256']:raise ValueError('Parent checkpoint differs')
    model.load_state_dict(load_file(str(parent/'selected-model.safetensors')),strict=True)
    for parameter in model.parameters():parameter.requires_grad_(False)
    for module in [*model.base_model.encoder.layer[-4:],model.classifier]:
        for parameter in module.parameters():parameter.requires_grad_(True)
    tokens = {}; truth = {}
    for split, values in rows.items():
        pairs = [texts(r) for r in values]
        tokens[split] = tokenizer([p[0] for p in pairs], [p[1] for p in pairs], padding=False, truncation=False)
        if max(map(len,tokens[split]['input_ids'])) > 512:
            raise ValueError('Input exceeds 512 tokens; no silent truncation')
        truth[split] = np.array([target(r) for r in values])
    def batch(split, indices):
        return {k:v.to(args.device) for k,v in tokenizer.pad([{k:v[i] for k,v in tokens[split].items()} for i in indices],return_tensors='pt',pad_to_multiple_of=PADDING_MULTIPLE).items()}
    def score(split):
        model.eval(); scores=[]
        with torch.inference_mode():
            for offset in range(0,len(rows[split]),8):
                logits=model(**batch(split,list(range(offset,min(offset+8,len(rows[split])))))).logits.float()
                scores.extend(torch.softmax(logits,dim=-1)[:,1].cpu().tolist())
                if args.device == 'mps' and (offset//8+1)%25 == 0:
                    torch.mps.empty_cache()
        if not np.isfinite(scores).all():
            raise ValueError('Nonfinite inference scores')
        return np.array(scores)
    def save_scores(split,scores,thresholds):
        records=[dict(case_id=r['case_id'],language=language,gold=int(y),score=float(s),predicted=int(s>=thresholds[language])) for r,y,s in zip(rows[split],truth[split],scores)]
        (args.output/(split+'-predictions.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in records))
    anchor_mask=np.array([r.get('source')!='scope_distillation_v9' for r in train])
    teacher_spec=json.loads(gzip.decompress((teacher_parent/'review.json.gz').read_bytes()))
    teacher_checkpoint=teacher_parent/'selected-model.safetensors'
    if digest(teacher_checkpoint)!=teacher_spec['checkpoint_sha256']:raise ValueError('Teacher checkpoint differs')
    model.load_state_dict(load_file(str(teacher_checkpoint)),strict=True);model.eval()
    teacher_probabilities=torch.full((len(train),2),.5,dtype=torch.float32)
    replay_indices=np.flatnonzero(anchor_mask).tolist()
    with torch.inference_mode():
        for offset in range(0,len(replay_indices),8):
            indices=replay_indices[offset:offset+8]
            values=torch.softmax(model(**batch('train',indices)).logits.float()/2.,dim=-1).cpu()
            teacher_probabilities[indices]=values
            if args.device=='mps' and (offset//8+1)%25==0:torch.mps.empty_cache()
    model.load_state_dict(load_file(str(parent/'selected-model.safetensors')),strict=True)
    groups=[(r.get('source') in ('v7_training_frame_projection','distress_repair_v9','needs_repair_v9','bare_distress_repair_v9','scope_distillation_v9'),int(y),r['attributes']['assertion']=='RETRACTED') for r,y in zip(train,truth['train'])]
    frequencies=Counter(groups)
    if len(frequencies)!=8:
        raise ValueError('Missing formal/native label stratum')
    weights=np.array([1/frequencies[g] for g in groups]);weights/=weights.sum()
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-6,weight_decay=.01)
    total_steps=2*math.ceil(len(train)/8);warmup=max(1,int(.05*total_steps))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min((step+1)/warmup,max(0.,(total_steps-step)/(total_steps-warmup))))
    record=dict(synthetic_only=True,selection_partition='development',base=model_id,revision=revision,train_language=language,
                training_rows=len(train),epochs=[],learning_rate=3e-6,full_parameter_training=False,
                focus_boundary_training_rows=sum(r.get('source')=='focus_boundary_contrasts_v9' for r in train),
                time_scope_training_rows=sum(r.get('source')=='time_scope_contrasts_v9' for r in train),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                sampling='Equal expected mass per formal/conversational x label x retraction; authored repair conversational; replacement; seed1102.',
                base_weights_sha256=weight_hash)
    record['parent_artifact_sha256']=digest(parent/'review.json.gz')
    record['teacher_artifact_sha256']=digest(teacher_parent/'review.json.gz')
    record['objective']='Cross entropy without label smoothing plus0.5*T^2*KL(teacher85||student), T=2, only historical replay rows; new authored rows never distilled'
    record['teacher_replay_rows']=len(replay_indices)
    record['calibration_diagnosis_informed_training']=True
    record['authored_repair_rows']=sum(r.get('source') in ('distress_repair_v9','needs_repair_v9','bare_distress_repair_v9','scope_distillation_v9') for r in train)
    record['stress_rows']=len(rows['stress'])
    record['stress_data_sha256']=digest(args.output/'stress.jsonl.gz')
    record['base_manifest_sha256']=digest(base_manifest_path)
    record['validation_schedule']='Half-epoch and end-of-epoch, selected solely by development counts.'
    record['padding_multiple']=PADDING_MULTIPLE
    record['mps_cache_release_interval']=25
    import shutil
    chosen_thresholds=parent_spec['thresholds']
    chosen_scores=score('dev');chosen_stress_scores=score('stress')
    initial_metrics=counts(rows['dev'],truth['dev'],chosen_scores,chosen_thresholds)
    initial_false=float(np.mean(chosen_stress_scores[truth['stress']==0]>=chosen_thresholds[language]))
    active=[v for k,v in initial_metrics.items() if k.startswith(language+'_')]
    best=(max([v['false_review_rate'] for v in active]+[initial_false]),sum(v['false_review_rate'] for v in active)+initial_false)
    record['epochs'].append(dict(epoch=0,thresholds=chosen_thresholds,development=initial_metrics,stress_false_review_rate=initial_false,loss=None,unmodified_warm_start=True))
    record['selected_epoch']=0
    shutil.copyfile(parent/'selected-model.safetensors',args.output/'selected-model.safetensors')
    shutil.copyfile(parent/'review.json.gz',args.output/'review.json.gz')
    save_scores('dev',chosen_scores,chosen_thresholds)
    (args.output/'development.json').write_text(json.dumps(record,indent=2)+'\n')
    print('Teacher replay and unmodified warm-start validation complete',flush=True)
    visited=set()
    checks=checkpoint_offsets(len(train))
    for epoch in range(1,3):
        order=rng.choice(len(train),size=len(train),replace=True,p=weights).tolist()
        model.train();losses=[];start=time.monotonic()
        for offset in range(0,len(order),8):
            indices=order[offset:offset+8];optimizer.zero_grad(set_to_none=True)
            logits=model(**batch('train',indices)).logits.float()
            loss=anchored_loss(logits,torch.tensor(truth['train'][indices],device=args.device),teacher_probabilities[indices].to(args.device),torch.tensor(anchor_mask[indices],device=args.device))
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();scheduler.step()
            losses.append(float(loss.detach().cpu()));visited.update(indices)
            if args.device == 'mps' and (offset//8+1)%25 == 0:
                torch.mps.empty_cache()
            if offset%1600==0:print('epoch',epoch,'row',offset,'of',len(train),'loss',losses[-1],flush=True)
            completed=min(offset+8,len(order))
            if completed not in checks:
                continue
            checkpoint_epoch=epoch if completed==len(order) else epoch-.5
            scores=score('dev');limits=[]
            for native in (False,True):
                positives=sorted(float(scores[i]) for i,r in enumerate(rows['dev']) if truth['dev'][i]==1 and (r.get('source')=='v7_training_frame_projection')==native)
                if not positives:raise ValueError('Missing development positives')
                limits.append(positives[int(.01*len(positives))])
            from training.conservative_fusion_v9 import full_coverage_gap
            stress_scores=score('stress')
            thresholds={language:full_coverage_gap(truth['dev'].tolist()+truth['stress'].tolist(),scores.tolist()+stress_scores.tolist())}
            metrics=counts(rows['dev'],truth['dev'],scores,thresholds)
            active=[v for k,v in metrics.items() if k.startswith(language+'_')]
            stress_false=float(np.mean(stress_scores[truth['stress']==0]>=thresholds[language]))
            rank=(max([v['false_review_rate'] for v in active]+[stress_false]),sum(v['false_review_rate'] for v in active)+stress_false)
            entry=dict(epoch=checkpoint_epoch,training_epoch=epoch,rows_completed=completed,loss=sum(losses)/len(losses),seconds=time.monotonic()-start,thresholds=thresholds,development=metrics,stress_false_review_rate=stress_false,stress_concern_recall=1.0)
            record['epochs'].append(entry);print(json.dumps(entry),flush=True)
            if best is None or rank<best:
                best=rank;record['selected_epoch']=checkpoint_epoch;chosen_thresholds=thresholds;chosen_scores=scores.copy();chosen_stress_scores=stress_scores.copy()
                checkpoint=args.output/'selected-model.safetensors'
                save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(checkpoint))
                artifact=dict(format='roguard-native-full-review-experiment-v1',base=model_id,revision=revision,
                              language=language,thresholds=thresholds,checkpoint_sha256=digest(checkpoint),
                              base_manifest_sha256=digest(base_manifest_path),
                              preprocessing=PREPROCESSING)
                (args.output/'review.json.gz').write_bytes(gzip.compress(json.dumps(artifact,separators=(',',':')).encode(),mtime=0))
                save_scores('dev',scores,thresholds)
            record['unique_training_indices_seen']=len(visited)
            (args.output/'development.json').write_text(json.dumps(record,indent=2)+'\n')
            model.train()
    model.load_state_dict(load_file(str(args.output/'selected-model.safetensors')),strict=True)
    replay=score('dev');error=float(np.max(np.abs(replay-chosen_scores)))
    if error>1e-5:raise ValueError('Selected checkpoint prediction replay differs')
    stress_replay=score('stress');stress_error=float(np.max(np.abs(stress_replay-chosen_stress_scores)))
    if stress_error>1e-5:raise ValueError('Stress checkpoint replay differs')
    save_scores('stress',stress_replay,chosen_thresholds)
    record['stress_calibration_used']=False
    record['stress_false_review_rate']=float(np.mean(stress_replay[truth['stress']==0]>=chosen_thresholds[language]))
    record['stress_missed']=int(np.sum(stress_replay[truth['stress']==1]<chosen_thresholds[language]))
    record['selected_checkpoint_dev_replay_max_error']=error
    scores=score('cal');save_scores('cal',scores,chosen_thresholds)
    record['calibration_audit']=counts(rows['cal'],truth['cal'],scores,chosen_thresholds)
    record['artifact_sha256']=digest(args.output/'review.json.gz')
    (args.output/'development.json').write_text(json.dumps(record,indent=2)+'\n')
    check=inspect(args.output,language)
    if record['stress_false_review_rate']>.05 or record['stress_missed']:
        check['failures'].append('Authored development stress gate failed')
        check['study_prerequisites_pass']=False
    (args.output/'promotion-check.json').write_text(json.dumps(check,indent=2)+'\n')
    print('FINISHED',json.dumps(check),flush=True)


if __name__ == '__main__':
    main()
