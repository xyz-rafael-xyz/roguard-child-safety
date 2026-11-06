"""Train registered semantic heads while preserving the v9 encoder/adapters."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import time

from roguard.trajectory_data import file_sha, read_jsonl, write_jsonl
from training.native_fields_v9 import SIZES, policy_mass
from training.separation_frontier_v9 import frontier
from training.shared_tail_training_v9 import make_head


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('features','feature_study','candidate','study','private'):
        parser.add_argument('--'+key.replace('_','-'),dest=key,type=Path,required=True)
    args=parser.parse_args()
    if args.study.exists() or args.private.exists():raise SystemExit('Refusing to overwrite a semantic fit')
    extraction=json.loads((args.feature_study/'report.json').read_text())
    if not extraction.get('complete'):raise ValueError('Feature extraction is incomplete')
    for lang,info in extraction['languages'].items():
        for suffix,expected in info['artifacts'].items():
            if file_sha(args.features/(lang+'-'+suffix))!=expected:raise ValueError('Representation binding differs')
    amendment=Path('research/v10-evidence/protocol-amendment-08.json')
    configurations=json.loads(amendment.read_text())['registered_candidates']
    args.study.mkdir(parents=True);args.private.mkdir(parents=True)
    protocol={'format':'roguard-v10-semantic-fit-1','created_utc':datetime.now(timezone.utc).isoformat(),
              'amendment_sha256':file_sha(amendment),'feature_report_sha256':file_sha(args.feature_study/'report.json'),
              'source_sha256':file_sha(Path(__file__)),'epochs':12,'seed':1011,'batch_size':64,
              'calibration_or_test_loaded':False,'stable_release_ready':False}
    (args.study/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (args.study/'source.py').write_bytes(Path(__file__).read_bytes())
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_flatten
    from safetensors.numpy import load_file, save_file
    mx.set_memory_limit(4*1024**3);mx.set_cache_limit(128*1024**2)
    report={'complete':False,'languages':{},'calibration_or_test_loaded':False,'stable_release_ready':False}
    started=time.monotonic()
    for lang in ('ro','uk'):
        info=extraction['languages'][lang]
        x=np.load(args.features/(lang+'-features.npy'),allow_pickle=False)
        teacher=np.load(args.features/(lang+'-logits.npy'),allow_pickle=False)
        rows=read_jsonl(args.features/(lang+'-rows.jsonl'))
        if len(rows)!=info['rows'] or x.shape!=(len(rows),info['dimensions']) or teacher.shape!=(len(rows),sum(SIZES)+2):
            raise ValueError('Incomplete or malformed feature arrays')
        train=np.array([i for i,r in enumerate(rows) if r['partition']=='train'])
        dev=np.array([i for i,r in enumerate(rows) if r['partition']=='dev'])
        if not len(train) or not len(dev) or len(train)+len(dev)!=len(rows):raise ValueError('Invalid partitions')
        labels=np.array([r['fields'] for r in rows],dtype=np.int32)
        review=np.array([r['gold'] for r in rows],dtype=np.int32)
        original=load_file(args.candidate/'actual'/(lang+'-selected.safetensors'))
        original_spec=json.loads((args.feature_study/'v9-binding'/(lang+'-review.json')).read_text())
        mapping={'0.weight':'first.weight','0.bias':'first.bias','3.weight':'last.weight','3.bias':'last.bias'}
        initial={k:original['heads.'+lang+'.'+v] for k,v in mapping.items()}
        weights=[]
        for field,size in enumerate(SIZES):
            counts=np.bincount(labels[train,field],minlength=size)
            w=np.minimum(4.,np.sqrt(len(train)/np.maximum(1,counts)/size))
            w/=np.mean(w[labels[train,field]])
            weights.append(mx.array(w.astype(np.float32)))
        record={'training_rows':len(train),'development_rows':len(dev),'epochs':[],'selected':None}
        report['languages'][lang]=record

        def persist():
            report['seconds']=time.monotonic()-started
            (args.study/'report.json').write_text(json.dumps(report,indent=2)+'\n')

        def evaluate(head,config,epoch):
            head.eval();parts=[]
            for offset in range(0,len(dev),256):
                z=head(mx.array(x[dev[offset:offset+256]]));mx.eval(z);parts.append(np.asarray(z).copy())
            z=np.concatenate(parts).astype(np.float64)
            if not np.isfinite(z).all():raise ValueError('Nonfinite development logits')
            pieces=np.split(z[:,:sum(SIZES)],np.cumsum(SIZES)[:-1],axis=1)
            def softmax(v):
                p=np.exp(v-v.max(axis=1,keepdims=True));return p/p.sum(axis=1,keepdims=True)
            factors=[softmax(v) for v in pieces]
            predictions=np.column_stack([v.argmax(axis=1) for v in factors])
            joint_errors=int(np.any(predictions!=labels[dev],axis=1).sum())
            field_errors=(predictions!=labels[dev]).sum(axis=0).tolist()
            field_ce=float(np.mean([-np.log(np.maximum(v[np.arange(len(dev)),labels[dev,j]],1e-30)).mean() for j,v in enumerate(factors)]))
            modes={'binary':softmax(z[:,-2:])[:,1], 'policy':policy_mass(factors),
                   'coupled_policy':policy_mass(factors,original_spec['couplings'])}
            choices=[]
            for mode,scores in modes.items():
                population=[{'case_id':rows[i]['case_id'],'language':lang,'group':rows[i]['group'],
                             'gold':int(review[i]),'score':float(score)} for i,score in zip(dev,scores)]
                f=frontier(population);binary=(scores>=f['threshold']).astype(int)
                errors=int((binary!=review[dev]).sum())
                choices.append((errors,-f['score_margin'],mode,f,population,binary))
            errors,negative_margin,mode,f,population,binary=min(choices,key=lambda c:c[:2])
            rank=[errors,joint_errors,field_ce,negative_margin]
            summary={'candidate':config['name'],'epoch':epoch,'binary_errors':errors,
                     'joint_argmax_errors':joint_errors,'field_errors':field_errors,
                     'mean_field_cross_entropy':field_ce,'mode':mode,'threshold':f['threshold'],
                     'score_margin':f['score_margin'],'selection_rank':rank}
            record['epochs'].append(summary)
            destination=args.private/lang/config['name'];destination.mkdir(parents=True,exist_ok=True)
            tensors={k:np.asarray(v).copy() for k,v in tree_flatten(head.parameters())}
            save_file(tensors,str(destination/('epoch-'+str(epoch)+'.safetensors')))
            if record['selected'] is None or rank<record['selected']['selection_rank']:
                combined=dict(original)
                for key,value in tensors.items():combined['heads.'+lang+'.'+key]=value
                checkpoint=args.private/(lang+'-selected.safetensors');save_file(combined,str(checkpoint))
                metadata=dict(original_spec,adapter_sha256=file_sha(checkpoint),mode=mode,threshold=f['threshold'])
                (args.study/(lang+'-review.json')).write_text(json.dumps(metadata,indent=2)+'\n')
                summary['adapter_sha256']=file_sha(checkpoint);record['selected']=dict(summary)
                for row,value in zip(population,binary):row['predicted']=int(value)
                write_jsonl(args.study/(lang+'-development-predictions.jsonl'),population)
                np.save(args.private/(lang+'-selected-dev-logits.npy'),z,allow_pickle=False)
            persist();head.train()
            print(json.dumps({'language':lang,**summary}),flush=True)

        for config in configurations:
            mx.random.seed(1011);rng=np.random.default_rng(1011)
            head=make_head(initial);mx.eval(head.parameters())
            optimizer=optim.AdamW(learning_rate=config['learning_rate'],weight_decay=.01)
            coefficient=config['teacher_kl']
            def loss_fn(model,features,y,fields,teacher_logits):
                output=model(features);offset=0;terms=[];divergences=[]
                for j,size in enumerate(SIZES):
                    values=output[:,offset:offset+size];old=teacher_logits[:,offset:offset+size]
                    terms.append(nn.losses.cross_entropy(values,fields[:,j])*weights[j][fields[:,j]])
                    old_logp=old-mx.logsumexp(old,axis=-1,keepdims=True)
                    logp=values-mx.logsumexp(values,axis=-1,keepdims=True)
                    divergences.append((mx.exp(old_logp)*(old_logp-logp)).sum(axis=-1))
                    offset+=size
                old=teacher_logits[:,-2:];values=output[:,-2:]
                old_logp=old-mx.logsumexp(old,axis=-1,keepdims=True)
                logp=values-mx.logsumexp(values,axis=-1,keepdims=True)
                divergences.append((mx.exp(old_logp)*(old_logp-logp)).sum(axis=-1))
                return (nn.losses.cross_entropy(values,y)+mx.stack(terms).mean(axis=0)
                        +coefficient*mx.stack(divergences).mean(axis=0)).mean()
            value_grad=nn.value_and_grad(head,loss_fn)
            evaluate(head,config,0)
            for epoch in range(1,13):
                losses=[]
                for start in range(0,len(train),64):
                    if start==0:order=rng.permutation(train)
                    chosen=order[start:start+64]
                    loss,gradient=value_grad(head,mx.array(x[chosen]),mx.array(review[chosen]),
                        mx.array(labels[chosen]),mx.array(teacher[chosen]))
                    gradient,norm=optim.clip_grad_norm(gradient,1.);mx.eval(loss,norm,gradient)
                    if not bool(mx.isfinite(loss).item()) or not bool(mx.isfinite(norm).item()):raise ValueError('Nonfinite optimization step')
                    optimizer.update(head,gradient);mx.eval(head.parameters(),optimizer.state)
                    losses.append(float(loss.item()))
                evaluate(head,config,epoch);record['epochs'][-1]['training_loss']=float(np.mean(losses));persist()
            del head,optimizer,value_grad;mx.clear_cache()
        record['complete']=True;persist()
    report.update(complete=True,finished_utc=datetime.now(timezone.utc).isoformat());persist()
    print(json.dumps({lang:r['selected'] for lang,r in report['languages'].items()},indent=2))


if __name__=='__main__':main()
