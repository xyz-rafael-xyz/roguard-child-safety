"""Registered streaming attention adaptation for the response-behavior checker."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

from eval.run_trajectory_v10 import LocalFrozenV9
from roguard.assurance import digest
from roguard.gemma4_review import _Engine
from roguard.trajectory_data import file_sha, read_jsonl, write_jsonl
from roguard.trajectory_model import RESPONSE_LABELS
from training.query_inference_v10 import QueryView
from training.response_attention_v10 import make_response_tail, zero_tail_outputs, response_state, load_response_weights, make_response_optimizer
from training.train_contrast_response_v10 import payload
from training.train_neural_response_v10 import PROMPT, make_head, prediction_rows
from training.train_response_readout_v10 import summarize


def load_inputs(a):
    import numpy as np
    data=json.loads((a.data/'report.json').read_text())
    parent=json.loads((a.parent_study/'report.json').read_text())
    parent_protocol=json.loads((a.parent_study/'protocol.json').read_text())
    parent_selection=json.loads((a.parent_study/'selection.json').read_text())
    contrast=json.loads((a.contrast_study/'report.json').read_text())
    contrast_protocol=json.loads((a.contrast_study/'protocol.json').read_text())
    contrast_selection=json.loads((a.contrast_study/'selection.json').read_text())
    if (not data['complete'] or not parent['complete'] or not contrast['complete']
            or parent_protocol['prompt']!=PROMPT or contrast_protocol['prompt']!=PROMPT
            or file_sha(a.parent_study/'selection.json')!=parent['selection_sha256']
            or file_sha(a.contrast_study/'selection.json')!=contrast['selection_sha256']
            or file_sha(a.parent_private/'features.npy')!=parent_selection['features_sha256']
            or file_sha(a.contrast_private/'additional-features.npy')!=contrast_selection['additional_features_sha256']
            or file_sha(a.parent_private/'selected.safetensors')!=parent['selected']['head_sha256']
            or parent['selected']['configuration']['hidden']!=0):
        raise ValueError('Incomplete or changed response initialization/features')
    for name in ('train','dev'):
        if file_sha(a.data/(name+'.jsonl'))!=data['artifacts'][name]:raise ValueError('Fixed response split differs')
    if file_sha(a.contrast_study/'additional-train.jsonl')!=contrast_protocol['additional_train_sha256']:
        raise ValueError('Contrast TRAIN differs')
    original=read_jsonl(a.data/'train.jsonl')+read_jsonl(a.data/'dev.jsonl')
    extra=read_jsonl(a.contrast_study/'additional-train.jsonl');rows=original+extra
    if any(r['partition']!='train' for r in extra):raise ValueError('Additional data must be TRAIN')
    identities=[{'case_id':r['case_id'],'partition':r['partition'],'input_sha256':digest(payload(r))} for r in rows]
    if (identities[:420]!=read_jsonl(a.parent_study/'feature-inputs.jsonl')
            or identities!=read_jsonl(a.contrast_study/'feature-inputs.jsonl')
            or len({r['case_id'] for r in rows})!=len(rows)):
        raise ValueError('Feature order or input identity differs')
    train=[i for i,r in enumerate(rows) if r['partition']=='train'];dev=[i for i,r in enumerate(rows) if r['partition']=='dev']
    if len(rows)!=676 or len(train)!=596 or len(dev)!=80 or {identities[i]['input_sha256'] for i in train}&{identities[i]['input_sha256'] for i in dev}:
        raise ValueError('Response partition count or separation differs')
    x=np.concatenate((np.load(a.parent_private/'features.npy',allow_pickle=False),
                      np.load(a.contrast_private/'additional-features.npy',allow_pickle=False)))
    if x.shape!=(676,2560) or not np.isfinite(x).all():raise ValueError('Malformed frozen features')
    bindings={'data_report_sha256':file_sha(a.data/'report.json'),
        'parent_report_sha256':file_sha(a.parent_study/'report.json'),
        'contrast_report_sha256':file_sha(a.contrast_study/'report.json'),
        'parent_features_sha256':parent_selection['features_sha256'],
        'additional_features_sha256':contrast_selection['additional_features_sha256'],
        'parent_head_sha256':parent['selected']['head_sha256'],
        'context_protocol_sha256':file_sha(a.context_study/'protocol.json')}
    return rows,x,train,dev,identities,parent,data,bindings


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','candidate','data','parent_study','parent_private','contrast_study','contrast_private','context_study','study','private'):
        p.add_argument('--'+name.replace('_','-'),dest=name,type=Path,required=True)
    p.add_argument('--preflight-report',type=Path)
    p.add_argument('--amendment',type=Path,default=Path('research/v10-evidence/protocol-amendment-46.json'))
    a=p.parse_args()
    if a.study.exists() or a.private.exists():raise ValueError('Refusing to overwrite response attention study')
    amendment=a.amendment;plan=json.loads(amendment.read_text());config=plan['optimization']
    fresh_head=plan.get('fresh_response_head',False)
    if not isinstance(fresh_head,bool):raise ValueError('Invalid response initialization option')
    rows,x,train,dev,identities,parent,data,bindings=load_inputs(a)
    if a.preflight_report:
        if a.preflight_report.exists():raise ValueError('Refusing to overwrite preflight')
        a.preflight_report.parent.mkdir(parents=True,exist_ok=True)
        record={'complete':True,'source_sha256':file_sha(__file__),'amendment_sha256':file_sha(amendment),
            'train_rows':len(train),'dev_rows':len(dev),'bindings':bindings,'model_loaded':False,'qualified':False}
        a.preflight_report.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2));return
    import numpy as np
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from safetensors.numpy import load_file,save_file
    from roguard.gemma4_attention import replay
    a.study.mkdir(parents=True);a.private.mkdir(parents=True)
    protocol={'format':'roguard-response-attention-fit-1','source_sha256':file_sha(__file__),
        'helper_sha256':file_sha('training/response_attention_v10.py'),'amendment_sha256':file_sha(amendment),
        'prompt':PROMPT,'labels':list(RESPONSE_LABELS),'optimization':config,'bindings':bindings,'fresh_response_head':fresh_head,
        'training_rows':596,'development_rows':80,'persistent_body_cache':False,
        'independent_validation':False,'stable_release_ready':False}
    (a.study/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (a.study/'source.py').write_bytes(Path(__file__).read_bytes())
    (a.study/'helper.py').write_bytes(Path('training/response_attention_v10.py').read_bytes())
    write_jsonl(a.study/'feature-inputs.jsonl',identities)
    started=time.monotonic();frozen=LocalFrozenV9(a.base,a.candidate,a.study/'v9-binding')
    engine=_Engine(*frozen.models['ro']._args);mx.random.seed(config['seed']);rng=np.random.default_rng(config['seed'])
    zero_tail_outputs(engine.model.base)
    head=make_head(parent['selected']['configuration'])
    head.load_weights([(k,mx.array(v)) for k,v in load_file(a.parent_private/'selected.safetensors').items()])
    head.eval();teacher=head(mx.array(x));mx.eval(teacher);teacher=np.asarray(teacher).copy()
    if fresh_head:
        mx.random.seed(config['seed']);head=make_head(parent['selected']['configuration']);head.eval()
    initial=head(mx.array(x));mx.eval(initial);initial=np.asarray(initial).copy()
    model=make_response_tail(engine.model.base,head);query=QueryView(engine,PROMPT)
    trainable_names=set(dict(tree_flatten(model.trainable_parameters())))
    if len(trainable_names)!=74:raise ValueError('Expected72 LoRA tensors and two response-head tensors')
    labels=np.array([RESPONSE_LABELS.index(r['gold']) for r in rows],dtype=np.int32)
    counts=Counter((rows[i]['language'],rows[i]['gold']) for i in train)
    if len(counts)!=10:raise ValueError('Missing TRAIN language/category')
    weights=np.array([len(train)/(10*counts[(r['language'],r['gold'])]) for r in rows],dtype=np.float32)
    dev_rows=[rows[i] for i in dev];history=[];selected=None;max_initial_feature=0.;max_full=0.
    def infer(population,reference_indices=None,check_full=False):
        nonlocal max_initial_feature,max_full
        values=[];checked=set();model.eval()
        for j,row in enumerate(population):
            state,tokens=response_state(engine,query,payload(row))
            h=replay(model.base,state)[:,0,:];z=model.head(h);mx.eval(h,z)
            if reference_indices is not None:
                error=float(np.max(np.abs(np.asarray(h)[0]-x[reference_indices[j]])))
                max_initial_feature=max(max_initial_feature,error)
                if error>1e-3:raise ValueError('Initial response features differ from bound parent')
            if check_full and row['language'] not in checked:
                full=model.base.model(mx.array([tokens]))[:,-1,:];mx.eval(full)
                error=float(mx.max(mx.abs(full-h)).item());max_full=max(max_full,error);checked.add(row['language'])
                if error>1e-3:raise ValueError('Actual response full/replay parity failed')
            values.append(np.asarray(z)[0].copy());del state,h,z;mx.clear_cache()
        return np.stack(values)
    # Verify one existing and one augmented TRAIN representation per language too.
    probes=[next(i for i in train if rows[i]['language']==lang and (i>=420)==additional)
            for lang in ('ro','uk') for additional in (False,True)]
    infer([rows[i] for i in probes],probes,True)
    def save_evaluation(epoch):
        nonlocal selected
        z=infer(dev_rows,dev if epoch==0 else None,True)
        outputs=prediction_rows(z.astype(np.float64));summary=summarize(dev_rows,outputs)
        if epoch==0:
            expected=prediction_rows(initial[dev].astype(np.float64))
            if any(a['label']!=b['label'] for a,b in zip(outputs,expected)) or (not fresh_head and summary['errors']!=3):
                raise ValueError('Initial response decisions differ from bound representations')
        path=a.private/('epoch-'+str(epoch)+'.safetensors')
        tensors={k:np.asarray(v).copy() for k,v in tree_flatten(model.trainable_parameters())}
        if set(tensors)!=trainable_names or any(not np.isfinite(v).all() for v in tensors.values()):
            raise ValueError('Response parameter inventory or finiteness changed')
        save_file(tensors,str(path));record={'epoch':epoch,'development':summary,'adapter_sha256':file_sha(path)};history.append(record)
        if selected is None or summary['selection_rank']<selected['development']['selection_rank']:
            selected=record;(a.private/'selected.safetensors').write_bytes(path.read_bytes())
            np.save(a.private/'selected-development-logits.npy',z,allow_pickle=False)
        (a.study/'progress.json').write_text(json.dumps({'selected':selected,'epochs':history,'seconds':time.monotonic()-started},indent=2)+'\n')
        print(json.dumps({'epoch':epoch,'development_errors':summary['errors'],'selected_epoch':selected['epoch'],'seconds':time.monotonic()-started}),flush=True)
    save_evaluation(0)
    optimizer,rate_groups=make_response_optimizer(config)
    def loss_fn(current,states,gold,row_weights,old):
        z=current(states);logp=z-mx.logsumexp(z,axis=-1,keepdims=True)
        old_logp=old-mx.logsumexp(old,axis=-1,keepdims=True)
        kl=(mx.exp(old_logp)*(old_logp-logp)).sum(axis=-1)
        return (row_weights*(nn.losses.cross_entropy(z,gold)+config['teacher_kl_weight']*kl)).mean()
    value_grad=nn.value_and_grad(model,loss_fn);step=0
    total=config['epochs']*((len(train)+config['batch_size']-1)//config['batch_size']);warm=max(1,int(.05*total))
    for epoch in range(1,config['epochs']+1):
        model.train();order=rng.permutation(train)
        for offset in range(0,len(order),config['batch_size']):
            chosen=order[offset:offset+config['batch_size']];states=[response_state(engine,query,payload(rows[i]))[0] for i in chosen]
            step+=1;scale=min(step/warm,max(.1,(total-step)/max(1,total-warm)))
            for group,rate in rate_groups:group.learning_rate=rate*scale
            loss,gradient=value_grad(model,states,mx.array(labels[chosen]),mx.array(weights[chosen]),mx.array(teacher[chosen]))
            gradient,norm=optim.clip_grad_norm(gradient,config['gradient_norm_clip']);mx.eval(loss,norm,gradient)
            if not bool(mx.isfinite(loss).item()) or not bool(mx.isfinite(norm).item()):raise ValueError('Nonfinite response update')
            optimizer.update(model,gradient);mx.eval(model.trainable_parameters(),optimizer.state)
            if step%25==0:print(json.dumps({'epoch':epoch,'step':step,'total_steps':total,'loss':float(loss.item()),'seconds':time.monotonic()-started}),flush=True)
            del states,gradient;mx.clear_cache()
        save_evaluation(epoch)
    selection={'complete':True,'selected':selected,'epochs':history,'maximum_initial_feature_error':max_initial_feature,
        'maximum_full_replay_error':max_full,'diagnostics_opened':False}
    selected_path=a.study/'selection.json';selected_path.write_text(json.dumps(selection,indent=2)+'\n');selection_sha=file_sha(selected_path)
    load_response_weights(model,load_file(a.private/'selected.safetensors'))
    z=infer(dev_rows,check_full=True);saved=np.load(a.private/'selected-development-logits.npy',allow_pickle=False)
    reload_error=float(np.max(np.abs(z-saved)))
    if reload_error>1e-5 or not np.array_equal(z.argmax(axis=1),saved.argmax(axis=1)):raise ValueError('Selected response reload differs')
    metrics={'development':summarize(dev_rows,prediction_rows(z.astype(np.float64)))}
    write_jsonl(a.study/'development-predictions.jsonl',[{**r,'prediction':p,'logits':v.tolist()} for r,p,v in zip(dev_rows,prediction_rows(z.astype(np.float64)),z)])
    context_protocol=json.loads((a.context_study/'protocol.json').read_text())
    if file_sha(a.context_study/'protocol.json')!=bindings['context_protocol_sha256'] or file_sha(a.context_study/'inputs.jsonl')!=context_protocol['input_sha256']:
        raise ValueError('Post-selection context diagnostic binding differs')
    context=read_jsonl(a.context_study/'inputs.jsonl')
    if len(context)!=120:raise ValueError('Expected every context34 diagnostic')
    cz=infer(context,check_full=True);cp=prediction_rows(cz.astype(np.float64));metrics['context']=summarize(context,cp)
    write_jsonl(a.study/'context-predictions.jsonl',[{**r,'prediction':p,'logits':v.tolist()} for r,p,v in zip(context,cp,cz)])
    if not any(m['errors'] for m in metrics.values()):
        if file_sha(a.data/'old-diagnostic.jsonl')!=data['artifacts']['old-diagnostic']:raise ValueError('Old diagnostic binding differs')
        diagnostic=read_jsonl(a.data/'old-diagnostic.jsonl');dz=infer(diagnostic,check_full=True);dp=prediction_rows(dz.astype(np.float64))
        metrics['old-diagnostic']=summarize(diagnostic,dp)
        write_jsonl(a.study/'old-diagnostic-predictions.jsonl',[{**r,'prediction':p,'logits':v.tolist()} for r,p,v in zip(diagnostic,dp,dz)])
    if file_sha(selected_path)!=selection_sha:raise ValueError('Selection changed after diagnostic access')
    spec={'format':'roguard-response-attention-research-1','adapter_sha256':file_sha(a.private/'selected.safetensors'),
        'base_model':'mlx-community/gemma-4-e4b-it-4bit','base_revision':'475b9088d29754a3379866cf5aeb6b41acd313c2',
        'labels':list(RESPONSE_LABELS),'prompt':PROMPT,'boundary':24,'head':parent['selected']['configuration'],
        'protocol_sha256':file_sha(a.study/'protocol.json'),'fresh_response_head':fresh_head,
        'selection_sha256':selection_sha,'qualified':False,'contextual_factual_support_established':False}
    (a.study/'model.json').write_text(json.dumps(spec,indent=2)+'\n')
    report={'complete':True,'selected':selected,'selection_sha256':selection_sha,'model_spec_sha256':file_sha(a.study/'model.json'),
        'metrics':metrics,'maximum_initial_feature_error':max_initial_feature,'maximum_full_replay_error':max_full,
        'selected_reload_maximum_logit_error':reload_error,'trainable_tensors':len(trainable_names),'seconds':time.monotonic()-started,
        'decision':'REJECT' if any(m['errors'] for m in metrics.values()) or 'old-diagnostic' not in metrics else 'RESPONSE_ENGINEERING_PASS_ONLY',
        'original_diagnostic_opened':'old-diagnostic' in metrics,'independent_validation':False,'stable_release_ready':False}
    (a.study/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
