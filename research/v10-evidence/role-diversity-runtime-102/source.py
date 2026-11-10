"""Extract original-v9 teachers and selected95 serving references without fitting."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import time

from roguard.trajectory_data import file_sha,write_jsonl
from training.role_diversity_runtime_v10 import load_registered,validate_array


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','private_root','base','plan','study','output'):
        p.add_argument('--'+name.replace('_','-'),dest=name,type=Path,required=True)
    p.add_argument('--plan-sha256',required=True)
    p.add_argument('--preflight-report',type=Path)
    a=p.parse_args()
    if a.study.exists() or a.output.exists():raise ValueError('Refusing to overwrite reference extraction')
    plan,languages=load_registered(a.root,a.private_root,a.plan,a.plan_sha256)
    if a.base.name!=plan['base_revision']:raise ValueError('Base snapshot differs')
    if a.preflight_report:
        result=dict(complete=True,plan_sha256=a.plan_sha256,source_sha256=file_sha(__file__),
                    new_train_rows=3168,selected_development_rows=8837,new_forward_inputs=6336,
                    encoder_loaded=False,training_performed=False,held_out_test_opened=False,qualified=False)
        a.preflight_report.parent.mkdir(parents=True,exist_ok=True)
        with a.preflight_report.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
        print(json.dumps(result));return
    from scripts.wait_completed_sequence_exit_v10 import require_exit
    require_exit(json.loads((a.root/plan['dependency_report']).read_text()),plan['dependency_pid'])
    def disk_check():
        if shutil.disk_usage(a.private_root).free<8*1024**3:raise ValueError('Disk reserve reached')
    disk_check()
    import numpy as np
    import mlx.core as mx
    from roguard.gemma4_review import _Engine,_score
    from roguard.gemma4_contract import PREFIX_TOKENS,classifier_tokens
    from roguard.gemma4_attention import capture_with_prefix,replay
    from training.runtime_attention_continuation_v10 import native_cached_features
    from training.extract_runtime_review_baseline_v10 import bounded_error
    from training.prepare_runtime_warm_start_v10 import compare_logits
    a.study.mkdir(parents=True);a.output.mkdir(parents=True)
    import fcntl
    lock=(a.output/'writer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=dict(format='roguard-role-diversity-runtime-1',plan_sha256=a.plan_sha256,
                  source_sha256=file_sha(__file__),modes=['v9','selected95'],
                  new_train_rows=3168,new_forward_inputs=6336,selected_development_rows=8837,
                  native_replay_tolerance=1e-3,reference_logit_tolerance=1e-5,runtime_score_tolerance=1e-10,
                  training_performed=False,held_out_test_opened=False,qualified=False)
    (a.study/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (a.study/'source.py').write_bytes(Path(__file__).read_bytes())
    report=dict(complete=False,languages={},training_performed=False,held_out_test_opened=False,qualified=False)
    started=time.monotonic()
    def persist():
        report['seconds']=time.monotonic()-started
        p=a.study/'report.tmp';p.write_text(json.dumps(report,indent=2)+'\n');p.replace(a.study/'report.json')
    for lang,data in languages.items():
        record=dict(complete=False,modes={});report['languages'][lang]=record
        write_jsonl(a.output/(lang+'-rows.jsonl'),data['train'])
        for mode in ('v9','selected95'):
            weights=data['weights'][mode];spec=data['specs'][mode]
            engine=_Engine(a.base,weights,file_sha(weights));model=engine.model;model.eval()
            prefix=engine.make_cache(model.base);mx.eval(model.base.model(mx.array([PREFIX_TOKENS]),cache=prefix))
            current=dict(complete=False,train_checked=0,dev_checked=0,maximum_native_replay_error=0.,
                         maximum_runtime_score_error=0.,runtime_groups_checked=0)
            record['modes'][mode]=current;persist()
            def forward(row,native=False):
                if classifier_tokens(engine.tokenizer,row['payload'])!=row['tokens']:
                    raise ValueError('Actual runtime tokenizer differs')
                suffix=mx.array([row['tokens'][len(PREFIX_TOKENS):]])
                state=capture_with_prefix(model.base,suffix,prefix,start=24);mx.eval(state)
                hidden=replay(model.base,state)[:,0,:];z=model.heads[lang](hidden);mx.eval(hidden,z)
                x=np.asarray(hidden)[0].copy();values=np.asarray(z)[0].copy()
                validate_array(x,(2560,));validate_array(values,(19,))
                error=0.
                if native:
                    direct=native_cached_features(model,suffix,prefix);mx.eval(direct)
                    error=bounded_error(x,np.asarray(direct)[0],1e-3);del direct
                del suffix,state,hidden,z;mx.clear_cache()
                return x,values,error
            xs=np.lib.format.open_memmap(a.output/f'{lang}-{mode}-features.npy',mode='w+',dtype=np.float32,shape=(1584,2560))
            zs=np.lib.format.open_memmap(a.output/f'{lang}-{mode}-logits.npy',mode='w+',dtype=np.float32,shape=(1584,19))
            groups=set()
            with (a.output/f'{lang}-{mode}-observations.jsonl').open('x') as observations:
                for i,row in enumerate(data['train']):
                    disk_check();x,z,error=forward(row,native=True);xs[i]=x;zs[i]=z
                    group=tuple(row['factors'][k] for k in ('scope','assertion','concern','bare'))
                    score_error=None
                    if group not in groups:
                        actual=engine.scores([row['payload']],spec)[0];score=_score(z,spec)
                        score_error=bounded_error(np.array([score]),np.array([actual]),1e-10)
                        if (score>=spec['threshold'])!=(actual>=spec['threshold']):raise ValueError('Served decision differs')
                        groups.add(group);current['runtime_groups_checked']=len(groups)
                        current['maximum_runtime_score_error']=max(current['maximum_runtime_score_error'],score_error)
                    observations.write(json.dumps(dict(index=i,case_id=row['case_id'],token_sha256=row['token_sha256'],
                        native_replay_error=error,runtime_score_error=score_error))+'\n')
                    current.update(train_checked=i+1,maximum_native_replay_error=max(current['maximum_native_replay_error'],error))
                    if (i+1)%100==0 or i+1==len(data['train']):
                        xs.flush();zs.flush();observations.flush();persist()
                        print(json.dumps(dict(language=lang,mode=mode,stage='train',checked=i+1,rows=len(data['train']),seconds=report['seconds'])),flush=True)
            xs.flush();zs.flush();del xs,zs
            if mode=='selected95':
                values=[];checked_groups=set()
                for i,row in enumerate(data['dev']):
                    disk_check();native=row['group'] not in checked_groups
                    _,z,error=forward(row,native=native);values.append(z)
                    if native:
                        checked_groups.add(row['group']);current['maximum_native_replay_error']=max(current['maximum_native_replay_error'],error)
                    # Reject drift immediately, preserving completed TRAIN references and progress.
                    compare_logits(z[None,:],data['reference'][i:i+1],spec,limit=1e-5,exact_decisions=True)
                    current['dev_checked']=i+1
                    if (i+1)%100==0 or i+1==len(data['dev']):
                        persist();print(json.dumps(dict(language=lang,mode=mode,stage='dev',checked=i+1,rows=len(data['dev']),seconds=report['seconds'])),flush=True)
                fresh=np.stack(values);current['development_parity']=compare_logits(fresh,data['reference'],spec,limit=1e-5,exact_decisions=True)
                np.save(a.output/(lang+'-selected95-dev-logits.npy'),fresh,allow_pickle=False)
                del fresh,values
            current.update(complete=True,artifacts={p.name:file_sha(p) for p in sorted(a.output.glob(f'{lang}-{mode}-*'))});persist()
            # The closure captures the engine/model; discard it before loading another encoder.
            del forward,engine,model,prefix;gc.collect();mx.clear_cache()
        record.update(complete=True,rows_sha256=file_sha(a.output/(lang+'-rows.jsonl')));persist()
    load_registered(a.root,a.private_root,a.plan,a.plan_sha256)
    report.update(complete=True,decision='REFERENCE_EXTRACTION_ONLY_REQUIRES_INDEPENDENT_AUDIT');persist()
    print(json.dumps(dict(complete=True,decision=report['decision'],seconds=report['seconds'])),flush=True)


if __name__=='__main__':main()
