"""Audit saved study65 checkpoint counters, selection and a training-log prefix."""
import argparse
import hashlib
import json
import math
from pathlib import Path

PROTOCOL_SHA='93a96fd4e4848d23531b40fdfb03562ebafe5a1c9cb50b8677b74db6a1d99cf2'
LABELS=['appropriate','boundary','privacy','pressure','unsupported']


def check(condition,message):
    if not condition:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def audit_history(progress,protocol):
    history=progress['epochs']
    check(1<=len(history)<=7,'Invalid epoch count')
    check([e['epoch'] for e in history]==list(range(len(history))),'Noncontiguous epochs')
    check(protocol['labels']==LABELS,'Unexpected labels')
    check(protocol['optimization']['epochs']==6 and protocol['optimization']['batch_size']==8,'Unexpected schedule')
    check(protocol['sampling']=={'kind':'four-stratum-cyclic-v1','examples_per_stratum':2,'batches_per_pass':3072,'presentations_per_pass':24576},'Sampling differs')
    check(protocol['optimizer_updates']==18432 and protocol['training_presentations']==147456,'Registered counters differ')
    for epoch in history:
        check(type(epoch['completed_training_updates']) is int and epoch['completed_training_updates']==epoch['epoch']*3072,'Checkpoint update count differs')
        check(type(epoch['completed_training_presentations']) is int and epoch['completed_training_presentations']==epoch['epoch']*24576,'Checkpoint presentation count differs')
        dev=epoch['development'];populations=dev['populations']
        check(set(populations)=={'historical','generated'},'Population missing')
        rates=[]
        for origin,p in populations.items():
            expected={f'{lang}:{label}':(8 if origin=='historical' else 128 if label=='appropriate' else 32) for lang in ('ro','uk') for label in LABELS}
            check(set(p['groups'])==set(expected),'Group missing')
            for key,g in p['groups'].items():
                check(g['total']==expected[key] and type(g['errors']) is int and 0<=g['errors']<=g['total'],'Invalid group count')
            check(p['rows']==sum(expected.values()) and p['errors']==sum(g['errors'] for g in p['groups'].values()),'Population counts differ')
            check(p['correct']==p['rows']-p['errors'],'Correct count differs')
            check(math.isfinite(p['cross_entropy']) and p['cross_entropy']>=0,'Invalid loss')
            worst=max(g['errors']/g['total'] for g in p['groups'].values());rates.append(worst)
            check(p['selection_rank']==[worst,p['errors'],p['cross_entropy']],'Population rank differs')
        rank=[max(rates),max(p['errors']/p['rows'] for p in populations.values()),sum(p['errors'] for p in populations.values()),sum(p['cross_entropy'] for p in populations.values())/2]
        check(dev['selection_rank']==rank and dev['errors']==rank[2],'Combined rank differs')
    selected=min(history,key=lambda e:e['development']['selection_rank'])
    check(progress['selected']==selected,'Wrong selected checkpoint')
    return selected


def audit_log(raw,progress):
    # An actively written trailing fragment is excluded, never parsed as a complete event.
    end=raw.rfind(b'\n')+1;prefix=raw[:end]
    events=[]
    for line in prefix.splitlines():
        if line.startswith(b'{'):events.append(json.loads(line))
        check(b'Traceback (most recent call last)' not in line,'Training traceback present')
    starts=[e for e in events if 'starting' in e]
    check(starts==[{'starting':'train-mixed-large-response-attention65'}],'Unexpected sequence starts')
    updates=[e for e in events if 'step' in e]
    check([e['step'] for e in updates]==list(range(25,25*(len(updates)+1),25)),'Missing or duplicated update log')
    for e in updates:
        check(e['total_steps']==18432 and 0<e['step']<=18432,'Logged schedule differs')
        check(e['epoch']==(e['step']-1)//3072+1,'Logged epoch differs')
        check(math.isfinite(e['loss']) and e['loss']>=0 and math.isfinite(e['seconds']),'Nonfinite logged loss/time')
    check(all(a['seconds']<b['seconds'] for a,b in zip(updates,updates[1:])),'Nonmonotone update times')
    latest=updates[-1]['step'] if updates else 0
    completed_updates=progress['epochs'][-1]['completed_training_updates']
    check(latest>=completed_updates//25*25,'Log does not cover completed checkpoint updates')
    check(latest<=min(18432,completed_updates+3072),'Log advances beyond the next checkpoint')
    timed=[e for e in events if 'step' in e or 'development_errors' in e]
    check(all(math.isfinite(e['seconds']) and e['seconds']>=0 for e in timed),'Invalid event time')
    check(all(a['seconds']<b['seconds'] for a,b in zip(timed,timed[1:])),'Nonmonotone event times')
    completed=[e for e in events if 'development_errors' in e]
    check([e['epoch'] for e in completed]==[e['epoch'] for e in progress['epochs']],'Progress/log checkpoint mismatch; retry observation')
    for event,epoch in zip(completed,progress['epochs'],strict=True):
        check(event['development_errors']==epoch['development']['errors'],'Logged error count differs')
        chosen=min(progress['epochs'][:epoch['epoch']+1],key=lambda e:e['development']['selection_rank'])
        check(event['selected_epoch']==chosen['epoch'],'Logged selection differs')
    return {'log_prefix_bytes':len(prefix),'log_prefix_sha256':hashlib.sha256(prefix).hexdigest(),
            'partial_trailing_log_line_excluded':end<len(raw),'logged_updates':len(updates),
            'latest_logged_update':updates[-1]['step'] if updates else 0}


def verify(root,private):
    study=root/'research/v10-evidence/mixed-large-response-attention-65';weights=private/'mixed-large-response-attention-65'
    check(sha(study/'protocol.json')==PROTOCOL_SHA,'Protocol changed')
    protocol=json.loads((study/'protocol.json').read_text())
    check(sha(study/'source.py')==protocol['source_sha256']==sha(root/'training/train_mixed_response_attention_v10.py'),'Trainer changed')
    check(sha(study/'helper.py')==protocol['helper_sha256']==sha(root/'training/response_attention_v10.py'),'Model helper changed')
    for path,want in protocol['helpers'].items():check(sha(root/path)==want,'Helper changed: '+path)
    raw=(study/'progress.json').read_bytes();progress=json.loads(raw)
    selected=audit_history(progress,protocol)
    for epoch in progress['epochs']:check(sha(weights/f"epoch-{epoch['epoch']}.safetensors")==epoch['adapter_sha256'],'Checkpoint hash differs')
    check(sha(weights/'selected.safetensors')==selected['adapter_sha256'],'Selected weights differ')
    log=private/'mixed-response-attention-sequence-65.log';log_raw=log.read_bytes();logged=audit_log(log_raw,progress)
    check((study/'progress.json').read_bytes()==raw,'Progress changed during observation; retry')
    check(log.read_bytes().startswith(log_raw),'Log prefix changed')
    return {'verification':'PASS','scope':'Recorded completed-checkpoint counts/ranks, actual saved weight identities, registered update/presentation counters and complete log prefix; no per-case or model qualification claim.',
            'verifier_sha256':sha(Path(__file__)),'protocol_sha256':PROTOCOL_SHA,'progress_sha256':hashlib.sha256(raw).hexdigest(),
            'completed_epoch':progress['epochs'][-1]['epoch'],'selected_epoch':selected['epoch'],
            'selected_development_errors':selected['development']['errors'],
            'completed_checkpoint_updates':progress['epochs'][-1]['completed_training_updates'],
            'completed_checkpoint_presentations':progress['epochs'][-1]['completed_training_presentations'],
            **logged,'model_qualified':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--private',type=Path,required=True)
    p.add_argument('--output',type=Path);a=p.parse_args()
    if a.output and a.output.exists():raise ValueError('Refusing to overwrite verification')
    result=verify(a.root,a.private)
    if a.output:
        with a.output.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
