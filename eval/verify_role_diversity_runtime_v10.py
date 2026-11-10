"""Audit revised TRAIN runtime references and selected95 DEV parity on CPU."""
import argparse
import json
import math
from pathlib import Path

from roguard.trajectory_data import file_sha,read_jsonl


def require_complete(report):
    if (report.get('complete') is not True
            or report.get('decision')!='REFERENCE_EXTRACTION_ONLY_REQUIRES_INDEPENDENT_AUDIT'
            or set(report.get('languages',{}))!={'ro','uk'}
            or any(report.get(k) is not False for k in ('training_performed','held_out_test_opened','qualified'))):
        raise ValueError('Reference extraction incomplete or scope differs')
    for language in report['languages'].values():
        if (language.get('complete') is not True or set(language.get('modes',{}))!={'v9','selected95'}
                or any(mode.get('complete') is not True for mode in language['modes'].values())):
            raise ValueError('Language/mode extraction incomplete')


def finite_error(value,limit):
    if type(value) not in (float,int) or not math.isfinite(value) or not 0<=value<=limit:
        raise ValueError('Numerical observation is invalid or exceeds tolerance')
    return float(value)


def compare_outputs(actual,expected,spec,*,limit,exact):
    import numpy as np
    from roguard.gemma4_review import _score
    for array in (actual,expected):
        if (array.ndim!=2 or array.shape[1]!=19 or len(array)==0
                or array.dtype!=np.float32 or not np.isfinite(array).all()):
            raise ValueError('Invalid logit comparison array')
    if actual.shape!=expected.shape:raise ValueError('Logit shapes differ')
    error=float(np.max(np.abs(actual-expected)))
    finite_error(error,limit)
    def classes(z):return np.column_stack([v.argmax(1) for v in np.split(z,[3,6,10,13,15,17],axis=1)])
    argmax=int(np.any(classes(actual)!=classes(expected),axis=1).sum())
    decisions=sum((_score(a,spec)>=spec['threshold'])!=(_score(b,spec)>=spec['threshold']) for a,b in zip(actual,expected))
    if exact and (argmax or decisions):raise ValueError('Reference argmax/decision parity failed')
    return dict(maximum_logit_error=error,argmax_rows_differing=argmax,fixed_threshold_decision_differences=decisions)


def check_observations(rows,observations,x,z,record,mode):
    import numpy as np
    n=len(rows)
    if not n or len(observations)!=n or any(r['partition']!='train' for r in rows):
        raise ValueError('TRAIN observation population differs')
    for array,shape in ((x,(n,2560)),(z,(n,19))):
        if array.shape!=shape or array.dtype!=np.float32 or not np.isfinite(array).all():
            raise ValueError('Retained feature/logit array differs')
    if mode not in ('v9','selected95'):raise ValueError('Unknown reference mode')
    seen=set();max_native=max_score=0.
    for i,(row,obs) in enumerate(zip(rows,observations)):
        if (type(obs['index']) is not int or obs['index']!=i or obs['case_id']!=row['case_id']
                or obs['token_sha256']!=row['token_sha256']):
            raise ValueError('Observation identity/order differs')
        max_native=max(max_native,finite_error(obs['native_replay_error'],1e-3))
        group=tuple(row['factors'][k] for k in ('scope','assertion','concern','bare'))
        if group not in seen:
            max_score=max(max_score,finite_error(obs['runtime_score_error'],1e-10));seen.add(group)
        elif obs['runtime_score_error'] is not None:
            raise ValueError('Serving probe is not the first row of a group')
    reported_native=finite_error(record['maximum_native_replay_error'],1e-3)
    finite_error(record['maximum_runtime_score_error'],1e-10)
    if ((mode=='v9' and reported_native!=max_native)
            or (mode=='selected95' and reported_native<max_native)
            or record['maximum_runtime_score_error']!=max_score
            or record['runtime_groups_checked']!=len(seen)
            or any(type(record[k]) is not int for k in ('runtime_groups_checked','train_checked','dev_checked'))
            or record['train_checked']!=n or record['complete'] is not True
            or (mode=='v9' and record['dev_checked']!=0)):
        raise ValueError('Reported mode metrics differ from retained observations')
    return dict(rows=n,train_native_observation_maximum=max_native,
                reported_combined_native_maximum=reported_native,runtime_score_observation_maximum=max_score,
                runtime_groups_checked=len(seen),dev_native_maximum_independently_recounted=False)


def verify_log(text,report):
    events=[];completion=[]
    for line in text.splitlines():
        try:obj=json.loads(line)
        except json.JSONDecodeError:continue
        if not isinstance(obj,dict):continue
        if 'stage' in obj and 'language' in obj:events.append(obj)
        if obj.get('decision')=='REFERENCE_EXTRACTION_ONLY_REQUIRES_INDEPENDENT_AUDIT':completion.append(obj)
    expected=[]
    for lang,dev_n in [('ro',4202),('uk',4635)]:
        for mode,stage,n in [('v9','train',1584),('selected95','train',1584),('selected95','dev',dev_n)]:
            checkpoints=list(range(100,n+1,100))
            if not checkpoints or checkpoints[-1]!=n:checkpoints.append(n)
            expected.extend((lang,mode,stage,i,n) for i in checkpoints)
    actual=[tuple(e[k] for k in ('language','mode','stage','checked','rows')) for e in events]
    if actual!=expected:raise ValueError('Extraction progress log is missing or reordered')
    times=[e['seconds'] for e in events]
    if any(type(t) not in (float,int) or not math.isfinite(t) or t<0 for t in times) or times!=sorted(times):
        raise ValueError('Progress time is invalid')
    if (len(completion)!=1 or completion[0].get('complete') is not True
            or completion[0]['seconds']!=report['seconds'] or report['seconds']<times[-1]):
        raise ValueError('Completion event differs from report')
    return len(events)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','private_root','audit_plan','output'):
        parser.add_argument('--'+name.replace('_','-'),dest=name,type=Path,required=True)
    parser.add_argument('--audit-plan-sha256',required=True)
    a=parser.parse_args()
    if a.output.exists():raise ValueError('Refusing to overwrite independent audit')
    if file_sha(a.audit_plan)!=a.audit_plan_sha256:raise ValueError('Audit plan differs')
    plan=json.loads(a.audit_plan.read_text())
    def check():
        for base,mapping in ((a.root,plan['public']),(a.private_root,plan['private'])):
            for name,sha in mapping.items():
                if file_sha(base/name)!=sha:raise ValueError('Audit source/binding differs: '+name)
        if file_sha(a.audit_plan)!=a.audit_plan_sha256:raise ValueError('Audit plan changed')
    check()
    if plan['cpu_logit_tolerance']!=1e-3:raise ValueError('CPU replay tolerance differs')
    study=a.root/plan['study'];data=a.private_root/plan['data']
    report_path=study/'report.json';report=json.loads(report_path.read_text());require_complete(report)
    sequence_path=a.root/plan['sequence'];sequence=json.loads(sequence_path.read_text())
    queue_path=a.private_root/plan['queue_plan'];queue=json.loads(queue_path.read_text())
    if (sequence['plan_sha256']!=file_sha(queue_path) or sequence['source_hashes']!=queue['source_hashes']
            or [s['name'] for s in sequence['completed_steps']]!=[s['name'] for s in queue['steps']]):
        raise ValueError('Completed sequence differs from registered queue')
    from scripts.wait_completed_sequence_exit_v10 import require_exit
    require_exit(sequence,plan['producer_queue_pid'])
    from training.role_diversity_runtime_v10 import load_registered
    extraction,languages=load_registered(a.root,a.private_root,a.root/plan['extraction_plan'],plan['extraction_plan_sha256'])
    protocol_path=study/'protocol.json';protocol=json.loads(protocol_path.read_text())
    extractor_sha=extraction['public']['training/extract_role_diversity_runtime_v10.py']
    expected_protocol=dict(format='roguard-role-diversity-runtime-1',plan_sha256=plan['extraction_plan_sha256'],
        source_sha256=extractor_sha,modes=['v9','selected95'],new_train_rows=3168,new_forward_inputs=6336,
        selected_development_rows=8837,native_replay_tolerance=1e-3,reference_logit_tolerance=1e-5,
        runtime_score_tolerance=1e-10,training_performed=False,held_out_test_opened=False,qualified=False)
    if protocol!=expected_protocol or file_sha(study/'source.py')!=extractor_sha:
        raise ValueError('Retained protocol/source differs')
    log_path=a.private_root/plan['log'];event_count=verify_log(log_path.read_text(),report)
    snapshot={p:file_sha(p) for p in (report_path,protocol_path,study/'source.py',sequence_path,log_path)}
    import numpy as np
    import mlx.core as mx
    from safetensors.numpy import load_file
    from training.shared_tail_training_v9 import make_head
    mx.set_default_device(mx.cpu)
    mapping={'0.weight':'first.weight','0.bias':'first.bias','3.weight':'last.weight','3.bias':'last.bias'}
    artifacts={};summaries={};projected=dev_count=0
    for lang,inputs in languages.items():
        language=report['languages'][lang];rows_path=data/(lang+'-rows.jsonl')
        rows=read_jsonl(rows_path)
        if rows!=inputs['train'] or file_sha(rows_path)!=language['rows_sha256']:
            raise ValueError('Retained TRAIN rows differ')
        snapshot[rows_path]=file_sha(rows_path);artifacts[rows_path.name]=snapshot[rows_path];modes={}
        for mode in ('v9','selected95'):
            record=language['modes'][mode]
            names={f'{lang}-{mode}-{suffix}' for suffix in ('features.npy','logits.npy','observations.jsonl')}
            if mode=='selected95':names.add(lang+'-selected95-dev-logits.npy')
            if set(record['artifacts'])!=names:raise ValueError('Mode artifact inventory differs')
            for name in sorted(names):
                path=data/name
                if file_sha(path)!=record['artifacts'][name]:raise ValueError('Retained artifact changed')
                snapshot[path]=file_sha(path);artifacts[name]=snapshot[path]
            x=np.load(data/f'{lang}-{mode}-features.npy',mmap_mode='r',allow_pickle=False)
            z=np.load(data/f'{lang}-{mode}-logits.npy',allow_pickle=False)
            obs=read_jsonl(data/f'{lang}-{mode}-observations.jsonl')
            summary=check_observations(rows,obs,x,z,record,mode)
            if summary['runtime_groups_checked']!=18:raise ValueError('Registered factor-group probe coverage differs')
            tensors=load_file(inputs['weights'][mode]);spec=inputs['specs'][mode]
            head=make_head({k:tensors['heads.'+lang+'.'+v] for k,v in mapping.items()});head.eval();pieces=[]
            for start in range(0,len(rows),256):
                actual=head(mx.array(x[start:start+256]));mx.eval(actual);pieces.append(np.asarray(actual).copy())
            replay=np.concatenate(pieces);summary['cpu_head_replay']=compare_outputs(replay,z,spec,limit=1e-3,exact=False)
            projected+=len(rows)
            if mode=='selected95':
                fresh=np.load(data/(lang+'-selected95-dev-logits.npy'),allow_pickle=False)
                parity=compare_outputs(fresh,inputs['reference'],spec,limit=1e-5,exact=True)
                if record['development_parity']!=parity or record['dev_checked']!=len(inputs['dev']):
                    raise ValueError('Reported development parity differs')
                summary['development_parity']=parity;summary['development_rows']=len(fresh);dev_count+=len(fresh)
            modes[mode]=summary
            del head,tensors,pieces,replay,actual,x,z;mx.clear_cache()
        summaries[lang]=modes
    if len(artifacts)!=16 or projected!=6336 or dev_count!=8837:raise ValueError('Incomplete independent audit population')
    load_registered(a.root,a.private_root,a.root/plan['extraction_plan'],plan['extraction_plan_sha256']);check()
    if any(file_sha(path)!=sha for path,sha in snapshot.items()):raise ValueError('Artifact changed during audit')
    proof=dict(verification='PASS',source_sha256=file_sha(__file__),audit_plan_sha256=a.audit_plan_sha256,
               extraction_plan_sha256=plan['extraction_plan_sha256'],report_sha256=snapshot[report_path],
               protocol_sha256=snapshot[protocol_path],sequence_sha256=snapshot[sequence_path],log_sha256=snapshot[log_path],
               artifact_count=16,artifacts=artifacts,cpu_head_rows=projected,development_rows_compared=dev_count,
               ordered_progress_events=event_count,languages=summaries,device='cpu',encoder_loaded=False,
               training_performed=False,training_admitted=False,held_out_test_opened=False,qualified=False,
               scope='All retained TRAIN features/logits/rows/observations and DEV logits with independent CPU head replay and exact registered DEV parity. TRAIN native and serving score differences are recounted observations, not new encoder inference. Selected95 DEV native maxima remain producer-only. No model or independent gold qualification.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(proof,f,indent=2);f.write('\n')
    print(json.dumps({k:proof[k] for k in ('verification','artifact_count','cpu_head_rows','development_rows_compared','ordered_progress_events','qualified')}))


if __name__=='__main__':main()
