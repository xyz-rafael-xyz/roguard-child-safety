"""Study145: expanded formal and indirect-distress semantic supervision."""
from datetime import datetime,timezone
import gzip
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

from training.gemma4_readout_v9 import normalize
from training.gemma4_structured_v9 import network,portable_logits
from training.native_fields_v9 import FIELDS,SIZES,targets,probabilities,policy_mass,fit_couplings
from training.prepare_instruction_sft_v8 import target
from training.budgeted_native_frontier_v9 import frontier


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import numpy as np
    import torch
    import torch.nn.functional as F
    from safetensors.torch import save_file
    from threadpoolctl import threadpool_limits
    root=Path('research/v9-stabilization')
    study=root/'gemma4-structured-expanded-145';parent=root/'gemma4-readout-141'
    expansion=root/'gemma4-expansion-144'
    for freeze in (study/'source-freeze.json',parent/'source-freeze.json',expansion/'source-freeze.json'):
        for path,expected in json.loads(freeze.read_text()).items():
            if digest(path)!=expected:raise ValueError('Frozen source differs: '+path)
    for name,expected in json.loads((study/'software.json').read_text()).items():
        if version(name)!=expected:raise ValueError('Dependency differs: '+name)
    prior=json.loads((parent/'actual/report.json').read_text())
    kernel_path=root/'gemma4-structured-143/actual/report.json'
    kernel=json.loads(kernel_path.read_text())
    if not prior.get('complete') or not kernel.get('complete'):raise ValueError('Incomplete parents')
    output=study/'actual';output.mkdir(exist_ok=False)
    if prior.get('development_passes') or kernel.get('development_passes'):
        (output/'report.json').write_text(json.dumps(dict(complete=True,skipped=True,reason='Prior classifier qualifies for remaining verification'),indent=2)+'\n')
        return
    rows=[json.loads(l) for l in gzip.decompress((parent/'inputs.jsonl.gz').read_bytes()).decode().splitlines()]
    pieces=[];offset=0
    for entry in prior['chunks']:
        start,stop=entry['start'],entry['stop']
        if start!=offset or stop<=start or stop>len(rows):raise ValueError('Noncontiguous feature chunks')
        path=parent/'actual'/f'chunk-{start:06d}-features.npz'
        identity=hashlib.sha256(json.dumps([r['token_sha256'] for r in rows[start:stop]]).encode()).hexdigest()
        if (digest(path)!=entry['feature_sha256'] or identity!=entry['input_sha256']
                or entry['source_freeze_sha256']!=digest(parent/'source-freeze.json')):
            raise ValueError('Feature provenance differs')
        value=np.load(path,allow_pickle=False)['features']
        if value.shape!=(stop-start,prior['feature_width']) or not np.isfinite(value).all():
            raise ValueError('Invalid features')
        pieces.append(value);offset=stop
    if offset!=len(rows):raise ValueError('Incomplete features')
    features=np.concatenate(pieces)
    extra_report=json.loads((expansion/'actual/report.json').read_text())
    if not extra_report.get('complete') or extra_report['source_freeze_sha256']!=digest(expansion/'source-freeze.json'):
        raise ValueError('Incomplete or changed expansion')
    additions=[json.loads(l) for l in gzip.decompress((expansion/'inputs.jsonl.gz').read_bytes()).decode().splitlines()]
    extra_pieces=[];offset=0
    for entry in extra_report['chunks']:
        start,stop=entry['start'],entry['stop']
        if start!=offset or stop<=start or stop>len(additions):raise ValueError('Expansion chunks are not contiguous')
        path=expansion/'actual'/f'chunk-{start:06d}-features.npz'
        identity=hashlib.sha256(json.dumps([r['token_sha256'] for r in additions[start:stop]]).encode()).hexdigest()
        if digest(path)!=entry['feature_sha256'] or identity!=entry['input_sha256'] or entry['source_freeze_sha256']!=digest(expansion/'source-freeze.json'):
            raise ValueError('Expansion feature provenance differs')
        value=np.load(path,allow_pickle=False)['features']
        if value.shape!=(stop-start,prior['feature_width']) or not np.isfinite(value).all():raise ValueError('Invalid expanded features')
        extra_pieces.append(value);offset=stop
    if offset!=len(additions) or any(r['partition']!='train' for r in additions):raise ValueError('Only complete training additions permitted')
    if {r['token_sha256'] for r in rows}&{r['token_sha256'] for r in additions}:raise ValueError('Expansion duplicates an existing input')
    rows+=additions;features=np.concatenate([features,*extra_pieces])
    train=[i for i,r in enumerate(rows) if r['partition']=='train']
    dev={lang:[i for i,r in enumerate(rows) if r['partition']=='dev' and r['row']['language']==lang] for lang in ('ro','uk')}
    expected=json.loads((study/'protocol.json').read_text())
    if len(train)!=expected['training_rows'] or sum(map(len,dev.values()))!=expected['development_rows']:
        raise ValueError('Expanded training/development population differs')
    train_rows=[rows[i]['row'] for i in train]
    main_y=np.array([target(r) for r in train_rows])
    field_y=np.array(targets(train_rows))
    groups=[(r['language'],int(y)) for r,y in zip(train_rows,main_y)]
    counts={group:groups.count(group) for group in set(groups)}
    if set(counts)!={('ro',0),('ro',1),('uk',0),('uk',1)}:raise ValueError('Missing language/label stratum')
    weights=np.array([len(train)/(4*counts[g]) for g in groups],dtype=np.float32)
    couplings=fit_couplings(train_rows,weights.tolist())
    torch.set_num_threads(4);torch.manual_seed(1143);rng=np.random.default_rng(1143)
    x=torch.tensor(normalize(features[train]),dtype=torch.float32)
    y=torch.tensor(main_y);fields=torch.tensor(field_y);sample_weights=torch.tensor(weights)
    dev_x={lang:torch.tensor(normalize(features[indices]),dtype=torch.float32) for lang,indices in dev.items()}
    model=network(torch,features.shape[1],hidden=128).float()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    protocol=json.loads((study/'protocol.json').read_text())
    started=time.monotonic()
    report=dict(complete=False,parent_linear_report_sha256=digest(parent/'actual/report.json'),
        parent_semantic_report_sha256=digest(kernel_path),expanded_feature_report_sha256=digest(expansion/'actual/report.json'),source_freeze_sha256=digest(study/'source-freeze.json'),
        selected={},epochs=[],training_rows=len(train),synthetic_only=True,independently_annotated=False,
        fresh_test_inferred=False,calibration_inferred=False,new_foundation_inference=False,
        stable_release_ready=False,training_language_label_weights={str(g):len(train)/(4*n) for g,n in counts.items()})
    best={}
    for lang in ('ro','uk'):
        previous=kernel['selected'][lang]
        report['selected'][lang]=dict(kind='parent143',parent_selection=previous,frontier=previous['frontier'])
        best[lang]=tuple(previous['frontier']['rank'])
    def evaluate(epoch):
        model.eval()
        checkpoint=output/(f'epoch-{epoch}.safetensors')
        save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(checkpoint))
        entry=dict(epoch=epoch,languages={},checkpoint_sha256=digest(checkpoint))
        tensors={k:v.detach().numpy().copy() for k,v in model.state_dict().items()}
        for lang,indices in dev.items():
            with torch.inference_mode():
                logits=model(dev_x[lang])
                binary=torch.softmax(logits[:,-2:],dim=-1)[:,1].numpy()
                marginal=probabilities(torch,logits[:,:sum(SIZES)])
            with threadpool_limits(limits=4):
                exported=portable_logits(features[indices],tensors)
            error=float(np.max(np.abs(exported-logits.numpy())))
            if error>1e-4:raise ValueError('Portable structured head logits differ')
            variants=dict(binary=binary,policy=policy_mass(marginal),coupled_policy=policy_mass(marginal,couplings))
            result={}
            for mode,values in variants.items():
                population=[dict(gold=target(rows[i]['row']),score=float(s),group=rows[i]['group']) for i,s in zip(indices,values)]
                selected=frontier(population);result[mode]=selected
                rank=tuple(selected['rank'])
                if rank<best[lang]:
                    best[lang]=rank
                    checkpoint=output/(lang+'-readout.safetensors')
                    save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(checkpoint))
                    spec=dict(format='roguard-gemma4-structured-readout-experiment-v1',language=lang,
                        hidden=128,dimensions=features.shape[1],normalization='l2-times-sqrt-width',mode=mode,
                        threshold=selected['threshold'],couplings=couplings,fields=FIELDS,sizes=SIZES,
                        epoch=epoch,checkpoint_sha256=digest(checkpoint),source_freeze_sha256=digest(study/'source-freeze.json'))
                    (output/(lang+'-readout.json')).write_text(json.dumps(spec,indent=2)+'\n')
                    report['selected'][lang]=dict(kind='structured',epoch=epoch,mode=mode,frontier=selected,checkpoint_sha256=digest(checkpoint))
                    predictions=[dict(case_id=rows[i]['row']['case_id'],language=lang,group=rows[i]['group'],
                        gold=target(rows[i]['row']),score=float(s),predicted=int(s>=selected['threshold'])) for i,s in zip(indices,values)]
                    (output/(lang+'-dev-predictions.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in predictions))
            entry['languages'][lang]=dict(export_logit_error=error,variants=result)
        report['epochs'].append(entry)
        (output/'development-progress.json').write_text(json.dumps(report,indent=2)+'\n')
        print('EVALUATE',json.dumps(entry),flush=True)
    evaluate(0)
    for epoch in range(1,protocol['epochs']+1):
        model.train();order=rng.permutation(len(train));losses=[]
        for start in range(0,len(order),128):
            ix=order[start:start+128]
            optimizer.zero_grad(set_to_none=True)
            output_logits=model(x[ix])
            primary=F.cross_entropy(output_logits[:,-2:],y[ix],reduction='none')
            auxiliary=torch.stack([F.cross_entropy(v,fields[ix,j],reduction='none')
                for j,v in enumerate(torch.split(output_logits[:,:sum(SIZES)],SIZES,dim=-1))]).mean(dim=0)
            loss=((primary+auxiliary)*sample_weights[ix]).mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite structured loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            losses.append(float(loss.detach()))
        if epoch%10==0:print('EPOCH',epoch,'loss',float(np.mean(losses)),flush=True)
        if epoch in protocol['evaluation_epochs']:evaluate(epoch)
    report.update(complete=True,finished_utc=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-started,
        development_passes=all(r['frontier']['counts_pass'] for r in report['selected'].values()))
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('FINISHED',json.dumps(dict(selected=report['selected'],development_passes=report['development_passes'],seconds=report['seconds'])),flush=True)


if __name__=='__main__':main()
