"""Larger response-head fit with chunked extraction and frozen held-out evaluation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import time

from eval.run_trajectory_v10 import LocalFrozenV9
from roguard.assurance import digest
from roguard.gemma4_review import _Engine
from roguard.trajectory_data import file_sha, read_jsonl, write_jsonl
from roguard.trajectory_model import RESPONSE_LABELS
from training.query_inference_v10 import QueryView
from training.train_contrast_response_v10 import payload
from training.train_neural_response_v10 import PROMPT, make_head, prediction_rows
from training.train_response_attention_v10 import load_inputs
from training.train_response_readout_v10 import summarize


def population_metrics(rows, logits):
    import numpy as np
    predictions=prediction_rows(np.asarray(logits,dtype=np.float64));metrics={}
    for origin in ('historical','generated'):
        indices=[i for i,row in enumerate(rows) if row['origin']==origin]
        if indices:metrics[origin]=summarize([rows[i] for i in indices],[predictions[i] for i in indices])
    rank=[max(v['selection_rank'][0] for v in metrics.values()),
          max(v['errors']/v['rows'] for v in metrics.values()),
          sum(v['errors'] for v in metrics.values()),
          sum(v['cross_entropy'] for v in metrics.values())/len(metrics)]
    return {'populations':metrics,'selection_rank':rank,'errors':rank[2]}


def training_weights(rows, indices):
    import numpy as np
    counts=Counter((rows[i]['origin'],rows[i]['language'],rows[i]['gold']) for i in indices)
    if len(counts)!=20:raise ValueError('Missing TRAIN origin/language/category')
    selected=set(indices)
    return np.asarray([len(indices)/(20*counts[(rows[i]['origin'],rows[i]['language'],rows[i]['gold'])])
                       if i in selected else 0. for i in range(len(rows))],dtype=np.float32)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','candidate','data','parent_study','parent_private','contrast_study','contrast_private',
                 'context_study','generated_study','generated_private','study','private','amendment'):
        p.add_argument('--'+name.replace('_','-'),dest=name,type=Path,required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--preflight-report',type=Path)
    a=p.parse_args();plan=json.loads(a.amendment.read_text())
    old,old_x,_,_,_,parent,old_data,bindings=load_inputs(a)
    if file_sha(a.data/'old-diagnostic.jsonl')!=old_data['artifacts']['old-diagnostic']:raise ValueError('Old diagnostic binding differs')
    generated=json.loads((a.generated_study/'report.json').read_text())
    if not generated['complete'] or generated['format']!='roguard-compositional-response-1':raise ValueError('Wrong generated corpus')
    for split in ('train','dev','test'):
        if file_sha(a.generated_private/(split+'.jsonl'))!=generated['artifacts'][split]:raise ValueError('Generated partition changed')
    new=read_jsonl(a.generated_private/'train.jsonl')+read_jsonl(a.generated_private/'dev.jsonl')
    if any(row['partition'] not in ('train','dev') for row in new):raise ValueError('Unexpected fit partition')
    old=[{**row,'origin':'historical'} for row in old];new=[{**row,'origin':'generated'} for row in new];rows=old+new
    identities=[{'case_id':row['case_id'],'partition':row['partition'],'input_sha256':digest(payload(row))} for row in rows]
    if len({r['case_id'] for r in rows})!=len(rows):raise ValueError('Duplicate fit case')
    train=[i for i,row in enumerate(rows) if row['partition']=='train'];dev=[i for i,row in enumerate(rows) if row['partition']=='dev']
    if (len(train),len(dev))!=(plan['train_rows'],plan['dev_rows']):raise ValueError('Registered split counts differ')
    if {identities[i]['input_sha256'] for i in train}&{identities[i]['input_sha256'] for i in dev}:raise ValueError('Input leakage')
    protocol={'format':'roguard-large-response-fit-1','source_sha256':file_sha(__file__),
              'amendment_sha256':file_sha(a.amendment),'generated_report_sha256':file_sha(a.generated_study/'report.json'),
              'helpers':{path:file_sha(path) for path in ('training/query_inference_v10.py','training/train_neural_response_v10.py',
                        'training/train_response_attention_v10.py','training/train_contrast_response_v10.py','training/train_response_readout_v10.py')},
              'bindings':bindings,'prompt':PROMPT,'labels':list(RESPONSE_LABELS),'fit_input_sha256':digest(identities),
              'fit_label_sha256':digest([row['gold'] for row in rows]),'training_rows':len(train),'development_rows':len(dev),
              'optimization':plan['optimization'],'candidates':plan['candidates'],
              'test_opened':False,'independent_validation':False,'stable_release_ready':False}
    if a.preflight_report:
        if a.preflight_report.exists():raise ValueError('Preflight exists')
        a.preflight_report.parent.mkdir(parents=True,exist_ok=True)
        a.preflight_report.write_text(json.dumps({**protocol,'model_loaded':False},indent=2)+'\n');return
    if a.resume:
        if json.loads((a.study/'protocol.json').read_text())!=protocol:raise ValueError('Fit resume bindings differ')
        if (a.study/'selection.json').exists():raise ValueError('Selection already frozen; do not rerun fit')
        if any(a.private.glob('candidate-*')):raise ValueError('Partial optimizer run retained; do not overwrite checkpoints')
    else:
        if a.study.exists() or a.private.exists():raise ValueError('Refusing to overwrite fit')
        a.study.mkdir(parents=True);a.private.mkdir(parents=True)
        (a.study/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        (a.study/'source.py').write_bytes(Path(__file__).read_bytes());write_jsonl(a.study/'feature-inputs.jsonl',identities)
    import fcntl
    lock=(a.private/'writer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import numpy as np
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from safetensors.numpy import save_file,load_file
    if shutil.disk_usage(a.private).free<8*1024**3:raise RuntimeError('Disk reserve reached')
    started=time.monotonic();binding=a.study/'v9-binding'
    if binding.exists():
        number=1
        while (a.study/('v9-binding-resume-'+str(number))).exists():number+=1
        binding=a.study/('v9-binding-resume-'+str(number))
    frozen=LocalFrozenV9(a.base,a.candidate,binding)
    engine=_Engine(*frozen.models['ro']._args);query=QueryView(engine,PROMPT);checked=set();parent_parity={}
    for i,row in enumerate(old):
        group=row['language']+':'+row['partition']
        if group not in checked:
            value=query(payload(row),verify_full=True);error=float(np.max(np.abs(value-old_x[i])))
            if error>1e-3:raise ValueError('Parent feature parity failed')
            checked.add(group);parent_parity[group]=error
    def extract(population,name):
        folder=a.private/(name+'-chunks');folder.mkdir(exist_ok=True);arrays=[]
        for offset in range(0,len(population),32):
            if shutil.disk_usage(a.private).free<8*1024**3:raise RuntimeError('Disk reserve reached')
            batch=population[offset:offset+32];path=folder/(str(offset)+'.npy');meta=folder/(str(offset)+'.json')
            identity=digest([{'input':payload(row),'case_id':row['case_id']} for row in batch])
            if meta.exists():
                record=json.loads(meta.read_text())
                if record['input_sha256']!=identity or file_sha(path)!=record['array_sha256']:raise ValueError('Feature chunk changed')
                if not 0<=record['full_parity_error']<=1e-3:raise ValueError('Invalid retained parity result')
                query.maximum_full_parity_error=max(query.maximum_full_parity_error,record['full_parity_error'])
                values=np.load(path,allow_pickle=False)
            else:
                if path.exists():raise ValueError('Uncommitted feature chunk retained; inspect before resuming')
                values=[]
                for row in batch:
                    group=name+':'+row['language']+':'+row['partition']
                    values.append(query(payload(row),verify_full=group not in checked));checked.add(group);mx.clear_cache()
                values=np.asarray(values,dtype=np.float32);np.save(path,values,allow_pickle=False)
                meta.write_text(json.dumps({'input_sha256':identity,'array_sha256':file_sha(path),'rows':len(batch),
                                            'full_parity_error':query.maximum_full_parity_error},indent=2)+'\n')
            if values.shape!=(len(batch),2560) or not np.isfinite(values).all():raise ValueError('Invalid feature chunk')
            arrays.append(values)
            progress={'phase':name+'-features','completed':offset+len(batch),'total':len(population),'seconds':time.monotonic()-started}
            temp=a.study/'progress.tmp';temp.write_text(json.dumps(progress,indent=2)+'\n');temp.replace(a.study/'progress.json')
            print(json.dumps(progress),flush=True)
        return np.concatenate(arrays)
    new_x=extract(new,'fit');x=np.concatenate((old_x,new_x));np.save(a.private/'features.npy',x,allow_pickle=False)
    labels=np.asarray([RESPONSE_LABELS.index(row['gold']) for row in rows],dtype=np.int32)
    weights=training_weights(rows,train);dev_rows=[rows[i] for i in dev];history=[];selected=None;optimization=plan['optimization']
    if any(a.private.glob('candidate-*')):raise ValueError('Partial optimizer run retained; do not overwrite checkpoints')
    for configuration in plan['candidates']:
        mx.random.seed(optimization['seed']);rng=np.random.default_rng(optimization['seed']);head=make_head(configuration)
        if configuration.get('warm_start'):head.load_weights([(k,mx.array(v)) for k,v in load_file(a.parent_private/'selected.safetensors').items()])
        optimizer=optim.AdamW(learning_rate=configuration['learning_rate'],weight_decay=optimization['weight_decay'])
        def loss_fn(model,features,gold,row_weights):
            return (row_weights*nn.losses.cross_entropy(model(features),gold)).mean()
        value_grad=nn.value_and_grad(head,loss_fn);destination=a.private/('candidate-'+configuration['name']);destination.mkdir()
        for epoch in range(optimization['epochs']+1):
            if epoch:
                order=rng.permutation(train)
                for offset in range(0,len(order),optimization['batch_size']):
                    chosen=order[offset:offset+optimization['batch_size']]
                    loss,gradient=value_grad(head,mx.array(x[chosen]),mx.array(labels[chosen]),mx.array(weights[chosen]))
                    gradient,norm=optim.clip_grad_norm(gradient,optimization['gradient_norm_clip']);mx.eval(loss,norm,gradient)
                    if not bool(mx.isfinite(loss).item()) or not bool(mx.isfinite(norm).item()):raise ValueError('Nonfinite update')
                    optimizer.update(head,gradient);mx.eval(head.parameters(),optimizer.state)
            z=head(mx.array(x[dev]));mx.eval(z);logits=np.asarray(z).copy();summary=population_metrics(dev_rows,logits)
            path=destination/('epoch-'+str(epoch)+'.safetensors')
            save_file({k:np.asarray(v).copy() for k,v in tree_flatten(head.parameters())},str(path))
            record={'configuration':configuration,'epoch':epoch,'development':summary,'head_sha256':file_sha(path)};history.append(record)
            if selected is None or summary['selection_rank']<selected['development']['selection_rank']:
                selected=record;(a.private/'selected.safetensors').write_bytes(path.read_bytes())
                np.save(a.private/'selected-development-logits.npy',logits,allow_pickle=False)
            (a.study/'history.json').write_text(json.dumps({'selected':selected,'epochs':history},indent=2)+'\n')
            print(json.dumps({'phase':'fit','candidate':configuration['name'],'epoch':epoch,'errors':summary['errors'],
                              'selected_errors':selected['development']['errors'],'seconds':time.monotonic()-started}),flush=True)
    selection={'complete':True,'selected':selected,'epochs':history,'features_sha256':file_sha(a.private/'features.npy'),
               'parent_feature_parity':parent_parity,'test_opened':False}
    path=a.study/'selection.json';path.write_text(json.dumps(selection,indent=2)+'\n');selection_sha=file_sha(path)
    head=make_head(selected['configuration']);head.load_weights([(k,mx.array(v)) for k,v in load_file(a.private/'selected.safetensors').items()])
    baseline=make_head(parent['selected']['configuration']);baseline.load_weights([(k,mx.array(v)) for k,v in load_file(a.parent_private/'selected.safetensors').items()])
    z=head(mx.array(x[dev]));mx.eval(z)
    if not np.array_equal(np.asarray(z),np.load(a.private/'selected-development-logits.npy',allow_pickle=False)):raise ValueError('Selected reload differs')
    metrics={}
    def evaluate(name,population,features):
        newz=head(mx.array(features));oldz=baseline(mx.array(features));mx.eval(newz,oldz)
        newp=prediction_rows(np.asarray(newz,dtype=np.float64));oldp=prediction_rows(np.asarray(oldz,dtype=np.float64))
        metrics[name]={'candidate':summarize(population,newp),'baseline26':summarize(population,oldp),
            'fixed':sum(o['label']!=r['gold'] and n['label']==r['gold'] for r,o,n in zip(population,oldp,newp)),
            'introduced':sum(o['label']==r['gold'] and n['label']!=r['gold'] for r,o,n in zip(population,oldp,newp))}
        write_jsonl(a.private/(name+'-predictions.jsonl'),[{**r,'prediction':n,'baseline26':o} for r,n,o in zip(population,newp,oldp)])
    for origin in ('historical','generated'):
        indices=[i for i in dev if rows[i]['origin']==origin]
        evaluate(origin+'-development',[rows[i] for i in indices],x[indices])
    context_protocol=json.loads((a.context_study/'protocol.json').read_text())
    if file_sha(a.context_study/'inputs.jsonl')!=context_protocol['input_sha256']:raise ValueError('Context diagnostic changed')
    for name,population in (('context34',read_jsonl(a.context_study/'inputs.jsonl')),
                            ('old-diagnostic',read_jsonl(a.data/'old-diagnostic.jsonl'))):
        evaluate(name,population,extract(population,name))
    test_opened=False
    if all(m['candidate']['errors']==0 and m['introduced']==0 for m in metrics.values()):
        if file_sha(a.generated_private/'test.jsonl')!=generated['artifacts']['test']:raise ValueError('Held-out test changed')
        test=read_jsonl(a.generated_private/'test.jsonl');test_opened=True
        evaluate('generated-test',test,extract(test,'test'))
    if file_sha(path)!=selection_sha:raise ValueError('Selection changed after diagnostic access')
    spec={'format':'roguard-large-response-research-1','head_sha256':file_sha(a.private/'selected.safetensors'),
          'configuration':selected['configuration'],'prompt':PROMPT,'labels':list(RESPONSE_LABELS),
          'selection_sha256':selection_sha,'qualified':False}
    (a.study/'model.json').write_text(json.dumps(spec,indent=2)+'\n')
    report={'complete':True,'selection_sha256':selection_sha,'selected':selected,'metrics':metrics,
            'training_rows':len(train),'development_rows':len(dev),'held_out_test_opened':test_opened,
            'maximum_full_query_parity_error':query.maximum_full_parity_error,'selected_reload_error':0.,
            'seconds':time.monotonic()-started,'decision':'REJECT' if not test_opened or any(m['candidate']['errors'] or m['introduced'] for m in metrics.values()) else 'RESPONSE_ENGINEERING_PASS_ONLY',
            'independent_validation':False,'stable_release_ready':False}
    (a.study/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
