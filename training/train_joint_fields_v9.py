"""Jointly adapt two encoder layers and six semantic heads on synthetic training."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

from roguard.assurance import policy_digest
from roguard.evidence import disposition
from roguard.native_fields import _native_states
from roguard.native_review import NativeReviewModel, _TorchEngine, prepare_input
from roguard.native_assets import PADDING_MULTIPLE
from training.native_fields_v9 import FIELDS, SIZES, network, probabilities, targets, policy_mass, fit_couplings
from training.review_fusion_v9 import metrics, threshold
from training.threshold_margin_fields_v9 import margin_threshold


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return [json.loads(line) for line in gzip.decompress(Path(path).read_bytes()).splitlines()]
def write(path,value): path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--language',choices=('ro','uk'),required=True)
    parser.add_argument('--base',type=Path,required=True);args=parser.parse_args()
    import numpy as np
    import torch
    from safetensors.torch import save_file, load_file
    torch.set_num_threads(4);torch.manual_seed(954+(args.language=='uk'))
    root=Path('research/v9-stabilization');experiment=root/'multitask-fields-54'
    for name,expected in json.loads((experiment/'source-freeze.json').read_text()).items():
        if sha(Path(name))!=expected:raise ValueError('Frozen joint-field source or data changed: '+name)
    study=experiment/args.language;study.mkdir(exist_ok=False)
    parent=experiment/'encoders'/args.language
    original=root/'pretrained-fields-47/encoders'/args.language
    original_heads=root/'pretrained-fields-47'/args.language/'selected-heads.json.gz'
    original_spec=json.loads(gzip.decompress(original_heads.read_bytes()))
    if original_spec['native_artifact_sha256']!=sha(original/'review.json.gz'):
        raise ValueError('Warm-start encoder identity differs')
    wrapper=NativeReviewModel(original/'review.json.gz',original/'selected-model.safetensors',args.base,
        root/'native-full-22-base-manifest.json',expected_artifact_sha256=sha(original/'review.json.gz'),device='mps')
    engine=_TorchEngine(*wrapper._engine_args)
    for parameter in engine.model.parameters():parameter.requires_grad_(False)
    layers=engine.model.base_model.encoder.layer[-2:]
    for layer in layers:
        for parameter in layer.parameters():parameter.requires_grad_(True)
    head=network(torch,1536,original_spec['hidden']).float()
    head.load_state_dict({k:torch.tensor(v,dtype=torch.float32) for k,v in original_spec['weights'].items()},strict=True)
    head=head.to('mps')
    mean=torch.tensor(original_spec['mean'],device='mps',dtype=torch.float32)
    scale=torch.tensor(original_spec['scale'],device='mps',dtype=torch.float32)
    rows={};encoded={};truth={};labels={}

    def load(split):
        rows[split]=read(parent/(split+'.jsonl.gz'))
        pairs=[prepare_input(dict(language=r['language'],turns=r['turns']),r['focus'])[0] for r in rows[split]]
        encoded[split]=engine.tokenizer([p[0] for p in pairs],[p[1] for p in pairs],padding=False,truncation=False)
        if max(map(len,encoded[split]['input_ids']))>512:raise ValueError('No truncation permitted')
        labels[split]=np.array(targets(rows[split]),dtype=np.int64)
        truth[split]=np.array([int(disposition({f:{'label':v} for f,v in r['attributes'].items()})[0]!='CONTINUE') for r in rows[split]])

    def features(split,indices):
        batch=engine.tokenizer.pad([{k:v[i] for k,v in encoded[split].items()} for i in indices],
            return_tensors='pt',pad_to_multiple_of=PADDING_MULTIPLE)
        mask=torch.zeros(batch['input_ids'].shape,dtype=torch.float32)
        for j,i in enumerate(indices):
            ids=encoded[split].sequence_ids(i);mask[j,:len(ids)]=torch.tensor([v==1 for v in ids])
        if (mask.sum(dim=1)==0).any():raise ValueError('Empty focal span')
        hidden,_=_native_states(engine,{k:v.to('mps') for k,v in batch.items()})
        mask=mask.to('mps');pooled=(hidden*mask[:,:,None]).sum(dim=1)/mask.sum(dim=1)[:,None]
        return (torch.cat((hidden[:,0,:],pooled),dim=1)-mean)/scale

    def infer(split):
        engine.model.eval();head.eval();parts=[[] for _ in FIELDS]
        with torch.inference_mode():
            for offset in range(0,len(rows[split]),8):
                indices=list(range(offset,min(offset+8,len(rows[split]))))
                p=probabilities(torch,head(features(split,indices)))
                for destination,value in zip(parts,p):destination.append(value)
                if (offset//8+1)%25==0:torch.mps.empty_cache()
        return [np.concatenate(values) for values in parts]

    load('train');load('dev')
    def family(row):
        return ('augmented' if row['source']=='joint_semantic_paraphrases_v9' else
                'conversational' if row['source']=='v7_training_frame_projection' else 'formal')
    group=[(family(row),int(label)) for row,label in zip(rows['train'],truth['train'])]
    sizes=Counter(group)
    if len(sizes)!=6:raise ValueError('All six training source/class groups required')
    weights=torch.tensor([1/sizes[g] for g in group],dtype=torch.float64)
    couplings=fit_couplings(rows['train'],[len(group)/(6*sizes[g]) for g in group])
    parameters=[p for p in engine.model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW([dict(params=parameters,lr=1e-5),dict(params=head.parameters(),lr=1e-4)],weight_decay=.01)
    trials=[];best=None;best_spec=None;best_p=None

    def evaluate(step):
        nonlocal best,best_spec,best_p
        p=infer('dev')
        loss=float(np.mean([-np.log(np.maximum(v[np.arange(len(rows['dev'])),labels['dev'][:,i]],1e-30)).mean() for i,v in enumerate(p)]))
        for mode,factors in (('independent',None),('pair-coupled',couplings)):
            scores=policy_mass(p,factors);cutoff=margin_threshold(scores,threshold(rows['dev'],truth['dev'],scores))
            groups=metrics(rows['dev'],truth['dev'],scores,cutoff)
            rank=[int(not all(v['passes'] for v in groups.values())),max(v['false_review_rate'] for v in groups.values()),
                  sum(v['false_review_rate'] for v in groups.values()),loss,step,int(factors is not None)]
            entry=dict(step=step,coupling_mode=mode,threshold=cutoff,metrics=groups,unweighted_field_loss=loss,rank=rank,
                       minimum_development_threshold_distance=float(np.min(np.abs(scores-cutoff))))
            trials.append(entry);print(json.dumps(entry),flush=True)
            if best is None or rank<best['rank']:
                best=entry;best_p=[v.copy() for v in p]
                checkpoint=parent/'selected-model.safetensors'
                save_file({k:v.detach().cpu().contiguous() for k,v in engine.model.state_dict().items()},str(checkpoint))
                descriptor=json.loads(gzip.decompress((original/'review.json.gz').read_bytes()))
                descriptor['checkpoint_sha256']=sha(checkpoint)
                (parent/'review.json.gz').write_bytes(gzip.compress(json.dumps(descriptor).encode(),mtime=0))
                best_spec=dict(original_spec,native_artifact_sha256=sha(parent/'review.json.gz'),
                    weights={k:v.detach().cpu().tolist() for k,v in head.state_dict().items()},threshold=cutoff,
                    couplings=factors,policy_digest=policy_digest(),
                    distribution=('independent' if factors is None else 'pair-coupled')+' categorical fields; uncalibrated')
                (study/'selected-heads.json.gz').write_bytes(gzip.compress(json.dumps(best_spec,separators=(',',':')).encode(),mtime=0))

    evaluate(0)
    indices=torch.multinomial(weights,len(group),replacement=True).tolist()
    steps=math.ceil(len(indices)/8);warmup=max(1,math.ceil(steps*.05));start=time.monotonic()
    for step,offset in enumerate(range(0,len(indices),8),1):
        engine.model.eval()
        for layer in layers:layer.train()
        head.train();optimizer.zero_grad(set_to_none=True)
        factor=min(step/warmup,max(0.,(steps-step)/(steps-warmup)))
        for parameter_group,base_lr in zip(optimizer.param_groups,(1e-5,1e-4)):parameter_group['lr']=base_lr*factor
        ix=indices[offset:offset+8]
        outputs=torch.split(head(features('train',ix)),SIZES,dim=-1)
        target=torch.from_numpy(labels['train'][ix]).to('mps')
        loss=sum(torch.nn.functional.cross_entropy(value,target[:,i]) for i,value in enumerate(outputs))/len(FIELDS)
        if not torch.isfinite(loss):raise ValueError('Nonfinite joint training loss')
        loss.backward();torch.nn.utils.clip_grad_norm_([*parameters,*head.parameters()],1);optimizer.step()
        if step%100==0:print('step',step,'of',steps,'loss',float(loss.detach().cpu()),'seconds',time.monotonic()-start,flush=True)
        del loss,outputs,target
        if step%25==0:torch.mps.empty_cache()
        if step in {math.ceil(steps/2),steps}:evaluate(step)
    engine.model.load_state_dict(load_file(str(parent/'selected-model.safetensors')),strict=True)
    head.load_state_dict({k:torch.tensor(v,dtype=torch.float32,device='mps') for k,v in best_spec['weights'].items()},strict=True)
    replay=infer('dev')
    error=max(float(np.max(np.abs(a-b))) for a,b in zip(replay,best_p))
    if error>1e-5:raise ValueError('Selected joint model did not reload exactly enough')
    selection=dict(created_utc=datetime.now(timezone.utc).isoformat(),selected=best,trials=trials,
        artifact_sha256=sha(study/'selected-heads.json.gz'),native_artifact_sha256=sha(parent/'review.json.gz'),
        checkpoint_sha256=sha(parent/'selected-model.safetensors'),warm_start_heads_sha256=sha(original_heads),
        data_hashes={s:sha(parent/(s+'.jsonl.gz')) for s in ('train','dev')},
        source_freeze_sha256=sha(experiment/'source-freeze.json'),selected_field_reload_max_error=error,
        calibration_used_for_selection=False,reserved_probes_used=False,stable_release_ready=False)
    write(study/'selection.json',selection)
    for split in ('dev','cal'):
        if split=='cal':load('cal')
        p=replay if split=='dev' else infer('cal');scores=policy_mass(p,best_spec['couplings'])
        groups=metrics(rows[split],truth[split],scores,best_spec['threshold'])
        if split=='dev' and groups!=best['metrics']:raise ValueError('Reloaded joint decisions differ')
        with (study/(split+'-predictions.jsonl')).open('x') as stream:
            for i,row in enumerate(rows[split]):
                stream.write(json.dumps(dict(case_id=row['case_id'],language=args.language,gold=int(truth[split][i]),
                    score=float(scores[i]),predicted=int(scores[i]>=best_spec['threshold']),
                    field_probabilities={f:v[i].tolist() for f,v in zip(FIELDS,p)}))+'\n')
        if split=='cal':
            write(study/'audit.json',dict(selection_sha256=sha(study/'selection.json'),calibration=groups,
                counts_pass=all(v['passes'] for v in groups.values()) and all(v['passes'] for v in best['metrics'].values()),
                synthetic_only=True,reserved_probes_used=False,stable_release_ready=False))
            print('CALIBRATION',json.dumps(groups),flush=True)


if __name__=='__main__':main()
