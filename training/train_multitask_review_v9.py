"""Train binary review and six semantic objectives over one native encoder."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

from roguard.evidence import disposition
from roguard.native_fields import _native_states
from roguard.native_review import NativeReviewModel, _TorchEngine, prepare_input
from roguard.native_assets import PADDING_MULTIPLE
from training.native_fields_v9 import FIELDS, SIZES, network, probabilities, targets
from training.review_fusion_v9 import metrics, threshold
from training.threshold_margin_fields_v9 import margin_threshold


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return [json.loads(line) for line in gzip.decompress(Path(path).read_bytes()).splitlines()]
def write(path, value): path.write_text(json.dumps(value, indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--language',choices=('ro','uk'),required=True)
    parser.add_argument('--base',type=Path,required=True);args=parser.parse_args()
    import numpy as np
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file, load_file
    torch.set_num_threads(4);torch.manual_seed(961+(args.language=='uk'))
    root=Path('research/v9-stabilization');experiment=root/'multitask-review-61'
    for name, expected in json.loads((experiment/'source-freeze.json').read_text()).items():
        if sha(name)!=expected: raise ValueError('Frozen multitask recipe changed: '+name)
    study=experiment/args.language
    if (study/'selection.json').exists(): raise ValueError('Training already selected')
    parent=root/'multitask-fields-54/encoders'/args.language
    heads=root/'multitask-fields-54'/args.language/'selected-heads.json.gz'
    parent_selection=json.loads((root/'multitask-fields-54'/args.language/'selection.json').read_text())
    if parent_selection['artifact_sha256']!=sha(heads): raise ValueError('Warm-start heads changed')
    spec=json.loads(gzip.decompress(heads.read_bytes()))
    wrapper=NativeReviewModel(parent/'review.json.gz',parent/'selected-model.safetensors',args.base,
        root/'native-full-22-base-manifest.json',expected_artifact_sha256=spec['native_artifact_sha256'],device='mps')
    engine=_TorchEngine(*wrapper._engine_args)
    # Reuse the earlier binary classifier and pooler, then adapt them jointly.
    binary_parent=root/('native-full-22-'+args.language)
    descriptor=json.loads(gzip.decompress((binary_parent/'review.json.gz').read_bytes()))
    if sha(binary_parent/'selected-model.safetensors')!=descriptor['checkpoint_sha256']:
        raise ValueError('Binary warm-start checkpoint changed')
    prefix='bert.pooler.' if args.language=='ro' else 'roberta.pooler.'
    with safe_open(str(binary_parent/'selected-model.safetensors'),framework='pt',device='cpu') as source:
        state={name:source.get_tensor(name) for name in source.keys()
               if name.startswith(('classifier.',prefix))}
    expected={name for name in engine.model.state_dict() if name.startswith(('classifier.',prefix))}
    if set(state)!=expected: raise ValueError('Binary warm-start tensor inventory differs')
    engine.model.load_state_dict(state,strict=False);del state
    for parameter in engine.model.parameters(): parameter.requires_grad_(False)
    layers=engine.model.base_model.encoder.layer[-4:]
    decision_modules=[engine.model.classifier]
    if engine.model.base_model.pooler is not None: decision_modules.append(engine.model.base_model.pooler)
    for module in [*layers,*decision_modules]:
        for parameter in module.parameters(): parameter.requires_grad_(True)
    head=network(torch,1536,spec['hidden']).float().to('mps')
    head.load_state_dict({k:torch.tensor(v,dtype=torch.float32,device='mps') for k,v in spec['weights'].items()},strict=True)
    mean=torch.tensor(spec['mean'],device='mps');scale=torch.tensor(spec['scale'],device='mps')
    rows={};encoded={};labels={};truth={}
    def load(split):
        rows[split]=read(study/(split+'.jsonl.gz'))
        pairs=[prepare_input(dict(language=r['language'],turns=r['turns']),r['focus'])[0] for r in rows[split]]
        encoded[split]=engine.tokenizer([p[0] for p in pairs],[p[1] for p in pairs],padding=False,truncation=False)
        if max(map(len,encoded[split]['input_ids']))>512: raise ValueError('No truncation permitted')
        labels[split]=np.array(targets(rows[split]),dtype=np.int64)
        truth[split]=np.array([int(disposition({f:{'label':v} for f,v in r['attributes'].items()})[0]!='CONTINUE') for r in rows[split]])
    def forward(split,indices):
        batch=engine.tokenizer.pad([{k:v[i] for k,v in encoded[split].items()} for i in indices],
            return_tensors='pt',pad_to_multiple_of=PADDING_MULTIPLE)
        mask=torch.zeros(batch['input_ids'].shape,dtype=torch.float32)
        for j,i in enumerate(indices):
            ids=encoded[split].sequence_ids(i);mask[j,:len(ids)]=torch.tensor([v==1 for v in ids])
        if (mask.sum(dim=1)==0).any(): raise ValueError('Empty focal span')
        hidden,logits=_native_states(engine,{k:v.to('mps') for k,v in batch.items()},include_logits=True)
        mask=mask.to('mps');pooled=(hidden*mask[:,:,None]).sum(dim=1)/mask.sum(dim=1)[:,None]
        return head((torch.cat((hidden[:,0,:],pooled),dim=1)-mean)/scale),logits
    def infer(split):
        engine.model.eval();head.eval();parts=[[] for _ in FIELDS];scores=[]
        with torch.inference_mode():
            for offset in range(0,len(rows[split]),8):
                fields,binary=forward(split,list(range(offset,min(offset+8,len(rows[split])))))
                for destination,value in zip(parts,probabilities(torch,fields)): destination.append(value)
                scores.append(torch.softmax(binary.float(),dim=-1)[:,1].cpu().numpy())
                if (offset//8+1)%25==0: torch.mps.empty_cache()
        return [np.concatenate(v) for v in parts],np.concatenate(scores).astype(np.float64)
    load('train');load('dev')
    def family(row):
        return ('augmented' if row['source']=='joint_semantic_paraphrases_v9' else
                'conversational' if row['source']=='v7_training_frame_projection' else 'formal')
    groups=[(family(row),int(label)) for row,label in zip(rows['train'],truth['train'])];sizes=Counter(groups)
    if len(sizes)!=6: raise ValueError('Six source/class groups required')
    weights=torch.tensor([1/sizes[g] for g in groups],dtype=torch.float64)
    encoder_parameters=[p for layer in layers for p in layer.parameters()]
    decision_parameters=[p for module in decision_modules for p in module.parameters()]
    all_parameters=[*encoder_parameters,*decision_parameters,*head.parameters()]
    optimizer=torch.optim.AdamW([dict(params=encoder_parameters,lr=1e-5),
        dict(params=[*decision_parameters,*head.parameters()],lr=1e-4)],weight_decay=.01)
    trials=[];best=None;best_scores=None;best_spec=None
    def evaluate(step):
        nonlocal best,best_scores,best_spec
        p,scores=infer('dev');cutoff=margin_threshold(scores,threshold(rows['dev'],truth['dev'],scores))
        observed=metrics(rows['dev'],truth['dev'],scores,cutoff)
        field_loss=float(np.mean([-np.log(np.maximum(v[np.arange(len(rows['dev'])),labels['dev'][:,i]],1e-30)).mean() for i,v in enumerate(p)]))
        rank=[int(not all(v['passes'] for v in observed.values())),max(v['false_review_rate'] for v in observed.values()),
              sum(v['false_review_rate'] for v in observed.values()),field_loss,step]
        entry=dict(step=step,threshold=cutoff,metrics=observed,field_loss=field_loss,rank=rank,
            field_correct={f:int((v.argmax(axis=1)==labels['dev'][:,i]).sum()) for i,(f,v) in enumerate(zip(FIELDS,p))})
        trials.append(entry);print(json.dumps(entry),flush=True)
        if best is None or rank<best['rank']:
            best=entry;best_scores=scores.copy()
            checkpoint=study/'selected-model.safetensors'
            save_file({k:v.detach().cpu().contiguous() for k,v in engine.model.state_dict().items()},str(checkpoint))
            artifact=json.loads(gzip.decompress((parent/'review.json.gz').read_bytes()))
            artifact.update(checkpoint_sha256=sha(checkpoint),thresholds={args.language:cutoff})
            (study/'review.json.gz').write_bytes(gzip.compress(json.dumps(artifact).encode(),mtime=0))
            best_spec=dict(spec,native_artifact_sha256=sha(study/'review.json.gz'),
                weights={k:v.detach().cpu().tolist() for k,v in head.state_dict().items()})
            (study/'auxiliary-heads.json.gz').write_bytes(gzip.compress(json.dumps(best_spec).encode(),mtime=0))
    evaluate(0)
    indices=torch.multinomial(weights,len(groups),replacement=True).tolist();steps=math.ceil(len(indices)/8)
    warmup=max(1,math.ceil(steps*.05));start=time.monotonic()
    for step,offset in enumerate(range(0,len(indices),8),1):
        engine.model.eval()
        for module in [*layers,*decision_modules]: module.train()
        if hasattr(engine.model,'dropout'): engine.model.dropout.train()
        head.train();optimizer.zero_grad(set_to_none=True)
        factor=min(step/warmup,max(0.,(steps-step)/(steps-warmup)))
        for group,rate in zip(optimizer.param_groups,(1e-5,1e-4)): group['lr']=factor*rate
        ix=indices[offset:offset+8];fields,binary=forward('train',ix)
        expected=torch.from_numpy(labels['train'][ix]).to('mps')
        auxiliary=sum(torch.nn.functional.cross_entropy(v,expected[:,i]) for i,v in enumerate(torch.split(fields,SIZES,dim=-1)))/len(FIELDS)
        loss=auxiliary+torch.nn.functional.cross_entropy(binary,torch.from_numpy(truth['train'][ix]).to('mps'))
        if not torch.isfinite(loss): raise ValueError('Nonfinite multitask loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(all_parameters,1);optimizer.step()
        if step%100==0: print('step',step,'of',steps,'loss',float(loss.detach().cpu()),'seconds',time.monotonic()-start,flush=True)
        del loss,auxiliary,fields,binary,expected
        if step%25==0: torch.mps.empty_cache()
        if step in {math.ceil(steps/2),steps}: evaluate(step)
    engine.model.load_state_dict(load_file(str(study/'selected-model.safetensors')),strict=True)
    head.load_state_dict({k:torch.tensor(v,dtype=torch.float32,device='mps') for k,v in best_spec['weights'].items()},strict=True)
    replay_fields,replay=infer('dev');error=float(np.max(np.abs(replay-best_scores)))
    if error>1e-5 or metrics(rows['dev'],truth['dev'],replay,best['threshold'])!=best['metrics']:
        raise ValueError('Selected binary checkpoint did not reload consistently')
    selection=dict(created_utc=datetime.now(timezone.utc).isoformat(),selected=best,trials=trials,
        artifact_sha256=sha(study/'review.json.gz'),checkpoint_sha256=sha(study/'selected-model.safetensors'),
        auxiliary_heads_sha256=sha(study/'auxiliary-heads.json.gz'),parent_heads_sha256=sha(heads),
        binary_warm_start_sha256=descriptor['checkpoint_sha256'],selected_checkpoint_dev_replay_max_error=error,
        data_hashes={s:sha(study/(s+'.jsonl.gz')) for s in ('train','dev')},
        calibration_used_for_selection=False,reserved_probes_used=False,stable_release_ready=False)
    write(study/'selection.json',selection)
    for split in ('dev','cal'):
        if split=='cal': load('cal')
        p,scores=(replay_fields,replay) if split=='dev' else infer('cal')
        observed=metrics(rows[split],truth[split],scores,best['threshold'])
        with (study/(split+'-predictions.jsonl')).open('x') as stream:
            for i,row in enumerate(rows[split]):
                stream.write(json.dumps(dict(case_id=row['case_id'],language=args.language,gold=int(truth[split][i]),
                    score=float(scores[i]),predicted=int(scores[i]>=best['threshold']),
                    field_probabilities={f:v[i].tolist() for f,v in zip(FIELDS,p)}))+'\n')
        if split=='cal':
            write(study/'development.json',dict(selection,calibration_audit=observed,
                selection_sha256=sha(study/'selection.json'),
                counts_pass=all(v['passes'] for v in observed.values()) and all(v['passes'] for v in best['metrics'].values()),
                field_correct={f:int((v.argmax(axis=1)==labels[split][:,i]).sum()) for i,(f,v) in enumerate(zip(FIELDS,p))}))
            print('CALIBRATION',json.dumps(observed),flush=True)


if __name__=='__main__': main()
