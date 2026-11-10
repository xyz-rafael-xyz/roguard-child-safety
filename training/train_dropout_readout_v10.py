"""CPU semantic heads with matched hidden-dropout controls on the larger corpus."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import time
import shutil

from training.coupled_policy_loss_v10 import make_coupled_policy_mass, policy_binary_cross_entropy, LOSS_EPSILON

from roguard.trajectory_data import file_sha, read_jsonl, write_jsonl
from training.native_fields_v9 import SIZES, policy_mass
from training.separation_frontier_v9 import frontier
from training.shared_tail_training_v9 import make_head
from training.dropout_readout_v10 import make_head as make_dropout_head



def validate_inputs(args, extraction, plan):
    import numpy as np
    if plan['device']!='cpu' or plan['epochs']!=12 or plan['seed']!=1011 or plan['batch_size']!=64:
        raise ValueError('Registered CPU schedule differs')
    expected=[{'name':name,'learning_rate':.0001,'teacher_kl':.2,'policy_ce':{'ro':.25,'uk':1.},'dropout':rate}
              for name,rate in (('dropout-control',0.),('dropout-tenth',.1),('dropout-three-tenths',.3))]
    if plan['registered_candidates']!=expected or plan['policy_loss_epsilon']!=LOSS_EPSILON:
        raise ValueError('Registered policy configurations differ')
    if set(extraction['languages'])!={'ro','uk'}:raise ValueError('Wrong languages')
    bindings={'feature_report_sha256':file_sha(args.feature_study/'report.json'),
        'helpers':{name:file_sha(name) for name in ('training/dropout_readout_v10.py','training/coupled_policy_loss_v10.py',
            'training/shared_tail_training_v9.py','training/native_fields_v9.py',
            'training/separation_frontier_v9.py','training/balanced_headroom_frontier_v9.py',
            'src/roguard/evidence.py')}}
    total_train=total_dev=0
    for lang,info in extraction['languages'].items():
        row_path=args.features/(lang+'-rows.jsonl');rows=read_jsonl(row_path)
        x=np.load(args.features/(lang+'-features.npy'),mmap_mode='r',allow_pickle=False)
        teacher=np.load(args.features/(lang+'-logits.npy'),mmap_mode='r',allow_pickle=False)
        for suffix,expected_hash in info['artifacts'].items():
            if file_sha(args.features/(lang+'-'+suffix))!=expected_hash:raise ValueError('Feature binding differs')
        if (x.shape!=(len(rows),info['dimensions']) or teacher.shape!=(len(rows),sum(SIZES)+2)
                or x.dtype!=np.float32 or teacher.dtype!=np.float32 or not np.isfinite(x).all()
                or not np.isfinite(teacher).all()):raise ValueError('Invalid feature arrays')
        if len(rows)!=info['rows'] or len({r['case_id'] for r in rows})!=len(rows):raise ValueError('Invalid row inventory')
        for row in rows:
            if (row['language']!=lang or row['partition'] not in ('train','dev')
                    or type(row['gold']) is not int or row['gold'] not in (0,1)
                    or len(row['fields'])!=len(SIZES)
                    or any(type(v) is not int or not 0<=v<size for v,size in zip(row['fields'],SIZES))):
                raise ValueError('Invalid training/development metadata')
        counts={split:sum(r['partition']==split for r in rows) for split in ('train','dev')}
        if counts!=info['partitions']:raise ValueError('Population counts differ')
        total_train+=counts['train'];total_dev+=counts['dev']
        spec_path=args.feature_study/'v9-binding'/(lang+'-review.json')
        spec=json.loads(spec_path.read_text());weights=args.candidate/'actual'/(lang+'-selected.safetensors')
        if spec['adapter_sha256']!=file_sha(weights):raise ValueError('Initial v9 checkpoint differs')
        bindings[lang]={'feature_artifacts':info['artifacts'],'partitions':counts,
            'v9_spec_sha256':file_sha(spec_path),'v9_weights_sha256':file_sha(weights)}
    if (total_train,total_dev)!=(plan['training_rows'],plan['development_rows']):raise ValueError('Registered populations differ')
    if bindings!=plan['input_bindings']:raise ValueError('Registered source/input bindings differ')
    return bindings

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('features','feature_study','candidate','study','private'):
        parser.add_argument('--'+key.replace('_','-'),dest=key,type=Path,required=True)
    parser.add_argument('--amendment',type=Path,required=True)
    parser.add_argument('--preflight-report',type=Path)
    args=parser.parse_args()
    if args.study.exists() or args.private.exists():raise SystemExit('Refusing to overwrite a semantic fit')
    extraction=json.loads((args.feature_study/'report.json').read_text())
    if not extraction.get('complete'):raise ValueError('Feature extraction is incomplete')
    for lang,info in extraction['languages'].items():
        for suffix,expected in info['artifacts'].items():
            if file_sha(args.features/(lang+'-'+suffix))!=expected:raise ValueError('Representation binding differs')
    amendment=args.amendment
    plan=json.loads(amendment.read_text());configurations=plan['registered_candidates']
    bindings=validate_inputs(args,extraction,plan)
    if args.preflight_report:
        if args.preflight_report.exists():raise ValueError('Refusing to overwrite preflight')
        record={'complete':True,'source_sha256':file_sha(__file__),'amendment_sha256':file_sha(amendment),
            'bindings':bindings,'training_rows':plan['training_rows'],'development_rows':plan['development_rows'],
            'device':'cpu','encoder_loaded':False,'training_launched':False,'qualified':False}
        args.preflight_report.parent.mkdir(parents=True,exist_ok=True)
        args.preflight_report.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2));return
    if shutil.disk_usage(args.private.parent).free<8*1024**3:raise RuntimeError('Disk reserve reached')
    args.study.mkdir(parents=True);args.private.mkdir(parents=True)
    import fcntl
    writer_lock=(args.private/'writer.lock').open('a');fcntl.flock(writer_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol={'format':'roguard-v10-dropout-semantic-fit-1','created_utc':datetime.now(timezone.utc).isoformat(),
              'amendment_sha256':file_sha(amendment),'feature_report_sha256':file_sha(args.feature_study/'report.json'),
              'source_sha256':file_sha(Path(__file__)),'epochs':12,'seed':1011,'batch_size':64,
              'bindings':bindings,'registered_candidates':configurations,'device':'cpu','encoder_loaded':False,
              'policy_loss_epsilon':LOSS_EPSILON,'policy_definition_unchanged':True,
              'calibration_or_test_loaded':False,'stable_release_ready':False}
    (args.study/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (args.study/'source.py').write_bytes(Path(__file__).read_bytes())
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_flatten
    from safetensors.numpy import load_file, save_file
    if mx.default_device()!=mx.cpu:raise ValueError('This experiment must use CPU')
    report={'complete':False,'languages':{},'calibration_or_test_loaded':False,'stable_release_ready':False,
            'device':'cpu','encoder_loaded':False,'training_rows':plan['training_rows'],
            'development_rows':plan['development_rows'],'policy_loss_epsilon':LOSS_EPSILON}
    started=time.monotonic()
    for lang in ('ro','uk'):
        info=extraction['languages'][lang]
        x=np.load(args.features/(lang+'-features.npy'),mmap_mode='r',allow_pickle=False)
        teacher=np.load(args.features/(lang+'-logits.npy'),mmap_mode='r',allow_pickle=False)
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
        policy_score=make_coupled_policy_mass(mx,original_spec['couplings'])
        maximum_initial_logit_error=0.
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
            pending=args.study/'report.tmp';pending.write_text(json.dumps(report,indent=2)+'\n')
            pending.replace(args.study/'report.json')

        def evaluate(head,config,epoch):
            nonlocal maximum_initial_logit_error
            head.eval();parts=[]
            for offset in range(0,len(dev),256):
                z=head(mx.array(x[dev[offset:offset+256]]));mx.eval(z);parts.append(np.asarray(z).copy())
            z=np.concatenate(parts).astype(np.float64)
            if not np.isfinite(z).all():raise ValueError('Nonfinite development logits')
            if epoch==0:
                error=float(np.max(np.abs(z-teacher[dev])))
                maximum_initial_logit_error=max(maximum_initial_logit_error,error)
                if error>1e-3:raise ValueError('Initial CPU head differs from bound v9 logits')
                record['maximum_initial_logit_error']=maximum_initial_logit_error
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
            if set(tensors)!=set(mapping.values()) or any(v.dtype!=np.float32 or not np.isfinite(v).all() for v in tensors.values()):
                raise ValueError('Invalid head checkpoint inventory or values')
            if shutil.disk_usage(args.private).free<8*1024**3:raise RuntimeError('Disk reserve reached')
            checkpoint=destination/('epoch-'+str(epoch)+'.safetensors')
            save_file(tensors,str(checkpoint))
            summary['checkpoint_sha256']=file_sha(checkpoint)
            summary['completed_training_updates']=epoch*((len(train)+63)//64)
            summary['completed_training_presentations']=epoch*len(train)
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
            head=make_dropout_head(initial,config['dropout']);mx.eval(head.parameters())
            optimizer=optim.AdamW(learning_rate=config['learning_rate'],weight_decay=.01)
            coefficient=config['teacher_kl'];policy_coefficient=config['policy_ce'][lang]
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
                per_row=(nn.losses.cross_entropy(values,y)+mx.stack(terms).mean(axis=0)
                         +coefficient*mx.stack(divergences).mean(axis=0))
                if policy_coefficient:
                    per_row=per_row+policy_coefficient*policy_binary_cross_entropy(mx,policy_score(output[:,:sum(SIZES)]),y)
                return per_row.mean()
            value_grad=nn.value_and_grad(head,loss_fn)
            evaluate(head,config,0)
            for epoch in range(1,13):
                losses=[]
                for start in range(0,len(train),64):
                    if shutil.disk_usage(args.private).free<8*1024**3:raise RuntimeError('Disk reserve reached')
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
        selected_weights=load_file(args.private/(lang+'-selected.safetensors'))
        selected_head=make_head({k:selected_weights['heads.'+lang+'.'+v] for k,v in mapping.items()});selected_head.eval()
        parts=[]
        for start in range(0,len(dev),256):
            values=selected_head(mx.array(x[dev[start:start+256]]));mx.eval(values);parts.append(np.asarray(values).copy())
        reloaded=np.concatenate(parts).astype(np.float64)
        saved=np.load(args.private/(lang+'-selected-dev-logits.npy'),allow_pickle=False)
        reload_error=float(np.max(np.abs(reloaded-saved)))
        if reload_error>1e-5 or not np.array_equal(reloaded.argmax(axis=1),saved.argmax(axis=1)):
            raise ValueError('Selected CPU head reload differs')
        record['selected_reload_maximum_logit_error']=reload_error
        record['complete']=True;persist()
        del selected_head,selected_weights,x,teacher;mx.clear_cache()
    if validate_inputs(args,extraction,plan)!=bindings:raise ValueError('Source bindings changed during training')
    report.update(complete=True,finished_utc=datetime.now(timezone.utc).isoformat(),
        decision='REJECT' if any(r['selected']['binary_errors'] or r['selected']['joint_argmax_errors'] for r in report['languages'].values()) else 'DEVELOPMENT_PASS_REQUIRES_FULL_REGRESSION');persist()
    print(json.dumps({lang:r['selected'] for lang,r in report['languages'].items()},indent=2))


if __name__=='__main__':main()
