"""Unregistered bounded native role-pooling resource prototype; no fitting.

All actual encoder execution requires an exact root-registered plan. Full suffix
states are transient; only seven pooled/final vectors per synthetic case persist.
"""
import argparse
from collections import Counter
from copy import deepcopy
import gc
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import resource
import os
import threading
import subprocess
import sys
import tempfile
import time
import types

FORMAT='roguard-role-pooling-resource-plan-1'
V9={'ro':'114d3e89191d06152cceccd080114998901a4f0bf6a9540de8e11dd8b6cf6b09',
    'uk':'0fee43a071d2c07c26e8b3d69cacb4933fd284bc64dbee0d18fe5481e1e4815f'}
SPECS={'ro':'6b2ea2979ffa43c735d8f8a6e3f7e1fbf6cf5c085889e8d484a13bf43026ccc5',
       'uk':'fc86fed9981b786149c4f6855c3d98f07067654cf2af05d40aef589345be34ed'}
ALIGNMENT_SHA='d684b8b592eed2e5241945211bf689097ca09a3bb44b58eb4dba3b7650bdbc5c'
VECTOR_NAMES=['native_final','current_minor','prior_minor','prior_nonminor','blind_third0','blind_third1','blind_third2']
CASES_SHA='14b46686a9ceceda17b01623cfc0eab1fcc8afb072b0ceeeb15b3272c5c0d370'
RECORDS_SHA='81f688e9a47f42b32d4ac8469b77145bc90edec244fc99f1af3c650ac56f22b1'
DESIGN_SHA='1edeffcd27354cf4b6f9e25da3c1d60eefd81c9ae70dffa588e28d14312e825f'
RESERVE=8*1024**3


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def canonical(value):return json.dumps(value,sort_keys=True,ensure_ascii=True,separators=(',',':'),allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(v) for v in Path(path).read_text().splitlines()]


def bindings(plan):
    for key,item in plan['files'].items():
        require(Path(item['path']).is_absolute() and sha(item['path'])==item['sha256'],'Changed bound file: '+key)


def origins(plan):
    root=Path(plan['project_root']).resolve();bound={Path(v['path']).resolve():v['sha256'] for v in plan['files'].values()}
    for path in (root/'src/roguard').rglob('*.py'):
        require(path.resolve() in bound and sha(path)==bound[path.resolve()],'Incomplete serving source inventory')
    for name,module in list(sys.modules.items()):
        if name=='roguard' or name.startswith('roguard.'):relative=Path('src')/Path(*name.split('.'))
        elif name=='training' or name.startswith('training.'):relative=Path(*name.split('.'))
        else:continue
        if getattr(module,'__file__',None) is None:
            require(name=='training' and all(Path(p).resolve()==root/'training' for p in module.__path__),'Unexpected namespace')
            continue
        expected=root/relative;expected=expected/'__init__.py' if expected.is_dir() else expected.with_suffix('.py')
        actual=Path(module.__file__).resolve()
        require(actual==expected.resolve() and actual in bound and sha(actual)==bound[actual],'Wrong imported source: '+name)


def validate_alignment(a):
    tokens=a['classifier_token_ids'];prefix=a['fixed_prefix_length']
    require(type(prefix) is int and prefix==387 and 387<len(tokens)<=600
        and all(type(i) is int and i>=0 for i in tokens)
        and a['classifier_token_count']==len(tokens) and a['classifier_token_sha256']==digest(tokens)
        and a['classifier_ids_unchanged'] is True and a['token_truncation_applied'] is False,'Invalid fixed serving sequence')
    roles=a['role_pool_positions'];controls=a['role_blind_pool_positions']
    require(len(roles)==len(controls)==3,'Expected three role and three control pools')
    for pool in roles+controls:
        require(pool==sorted(set(pool)) and all(type(i) is int and prefix<=i<len(tokens) for i in pool),'Invalid pool coordinates')
    eligible=sorted(i for pool in roles for i in pool)
    require(len(eligible)==len(set(eligible)) and eligible==a['eligible_content_positions']
        and sorted(i for pool in controls for i in pool)==eligible,'Role/control content coverage differs')
    q,r=divmod(len(eligible),3);start=0;wanted=[]
    for i in range(3):n=q+int(i<r);wanted.append(eligible[start:start+n]);start+=n
    require(controls==wanted and a['role_pool_validity']==[bool(p) for p in roles]
        and a['role_blind_pool_validity']==[bool(p) for p in controls]
        and a['final_token_index']==len(tokens)-1,'Mask/control/validity contract differs')
    return [[i-prefix for i in pool] for pool in roles+controls]


def preflight(path,expected):
    require(sha(path)==expected,'Changed resource plan');plan=read(path)
    require(plan['format']==FORMAT and plan['registered'] is True
        and plan['authorization']=='ROOT_REGISTERED_SYNTHETIC_RESOURCE_PILOT_ONLY'
        and plan['source_sha256']==sha(__file__) and plan['rows']==16
        and plan['maximum_tokens']==600 and plan['vectors']==VECTOR_NAMES
        and plan['training_performed'] is False and plan['accuracy_evaluation_permitted'] is False,
        'Unregistered resource pilot or changed scope')
    bindings(plan);paths={k:Path(v['path']) for k,v in plan['files'].items()}
    require(sha(paths['alignment_source'])==ALIGNMENT_SHA,'Wrong frozen alignment source')
    require(sha(paths['inventory'])==CASES_SHA and sha(paths['alignments'])==RECORDS_SHA and sha(paths['design'])==DESIGN_SHA,'Shared design/inventory changed')
    inventory=rows(paths['inventory']);records=rows(paths['alignments'])
    require(len(inventory)==len(records)==16 and len({r['case_id'] for r in inventory})==16,'Exact16 synthetic cases required')
    require(digest([r['case_id'] for r in inventory])==plan['ordered_case_ids_sha256'],'Case inventory order changed')
    require(Counter(r['payload']['language'] for r in inventory)==plan['language_counts'],'Language inventory changed')
    for row,record in zip(inventory,records):
        require(record['case_id']==row['case_id'] and row['synthetic'] is True and row['semantic_labels_present'] is False
            and '/' not in row['case_id'] and '..' not in row['case_id'],'Alignment case/source order differs')
        validate_alignment(record)
    for language in ('ro','uk'):
        require(plan['files'][language+'_weights']['sha256']==V9[language]
            and plan['files'][language+'_spec']['sha256']==SPECS[language],'Original-v9 head and exact fixed spec required')
        spec=read(paths[language+'_spec']);require(spec['language']==language and spec['adapter_sha256']==V9[language],'Wrong original spec')
    terminal=read(paths['prior120_terminal_report'])
    require(terminal['complete'] is True,'Prior encoder120 must be terminal before resource pilot')
    for package,pinned in plan['software_versions'].items():require(version(package)==pinned,'Software version changed: '+package)
    base=Path(plan['base_snapshot']);require(base.name==plan['base_revision'],'Base snapshot revision differs')
    for name,pin in plan['base_files'].items():
        require((base/name).stat().st_size==pin['bytes'] and sha(base/name)==pin['sha256'],'Pinned base changed: '+name)
    bindings(plan);origins(plan)
    require(sha(path)==expected and sha(__file__)==plan['source_sha256'],'Source/plan changed in preflight')
    return plan,paths,inventory,records


def disk_check(output,base_files):
    require(shutil.disk_usage(output.parent).free>=RESERVE,'Output reserve below8GiB')
    require(shutil.disk_usage(tempfile.gettempdir()).free>=RESERVE+sum(p['bytes'] for p in base_files.values()),'Temp base-copy plus8GiB reserve unavailable')


def cache_fingerprint(cache,np,mx=None):
    """Hash transient cache states and metadata; never serialize KV arrays."""
    h=hashlib.sha256()
    def visit(value):
        if isinstance(value,(list,tuple)):
            h.update(type(value).__name__.encode());h.update(str(len(value)).encode())
            for child in value:visit(child)
        elif isinstance(value,dict):
            for key in sorted(value):h.update(str(key).encode());visit(value[key])
        elif hasattr(value,'shape') and hasattr(value,'dtype'):
            original_dtype=str(value.dtype);original_shape=tuple(value.shape)
            if original_dtype in ('bfloat16','mlx.core.bfloat16'):
                require(mx is not None,'BF16 cache hashing requires explicit MLX conversion')
                # Every finite BF16 value (including signed zero) is exactly
                # representable in float32. Preserve original dtype in the hash.
                converted=value.astype(mx.float32);mx.eval(converted);a=np.asarray(converted)
            else:a=np.asarray(value)
            h.update(str(original_shape).encode());h.update(original_dtype.encode());h.update(a.tobytes())
        else:h.update(canonical(value).encode())
    for layer in cache:visit(layer.state);visit(layer.meta_state)
    return h.hexdigest()


def peak_rss():
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform=='darwin' else value*1024)


def seven_argmax(z):
    offsets=(0,3,6,10,13,15,17,19)
    return [max(range(b-a),key=lambda i:float(z[a+i])) for a,b in zip(offsets,offsets[1:])]


def enrich(result,row):
    role=result['role_pool_positions'];prior=sorted(role[1]+role[2]);n=len(role[1]);chrono=[prior[:n],prior[n:]]
    result.update(case_id=row['case_id'],fixture_family=row['fixture_family'],
        chronological_prior_control_positions=chrono,chronological_masks_identical=role[1:]==chrono,
        future_shared_six_metadata=[len(indices)/1024 for indices in role]+[float(flag) for flag in result['shared_structural_role_presence']],
        chronological_control_scope='Mask-only census; these2 extra vectors are NOT added to the7-vector engineering pilot.')
    return result


class DiskSampler:
    """Sample free-space minima, not an exact attributable temporary-copy peak."""
    def __init__(self,output):
        self.output=output;self.stop_event=threading.Event();self.samples=[];self.errors=[];self.thread=None
    def sample(self):
        self.samples.append(dict(seconds=time.monotonic(),output_free=shutil.disk_usage(self.output.parent).free,
            temp_free=shutil.disk_usage(tempfile.gettempdir()).free))
    def start(self):
        self.sample()
        def loop():
            while not self.stop_event.wait(.1):
                try:self.sample()
                except Exception as error:self.errors.append(str(error));return
        self.thread=threading.Thread(target=loop,daemon=True);self.thread.start()
    def finish(self):
        self.stop_event.set()
        if self.thread is not None:self.thread.join(timeout=2)
        require(self.thread is None or not self.thread.is_alive(),'Disk sampler failed to stop')
        self.sample();require(not self.errors,'Disk sampling failed: '+repr(self.errors))
        return dict(samples=self.samples,minimum_output_free=min(v['output_free'] for v in self.samples),
            minimum_temp_free=min(v['temp_free'] for v in self.samples),
            scope='100ms sampled free-space minima; shared-filesystem effects included, sub-sample peaks may be missed.')


def current_rss():
    result=subprocess.run(['ps','-o','rss=','-p',str(os.getpid())],capture_output=True,text=True,check=True,timeout=2)
    return int(result.stdout.strip())*1024


def compare(left,right,limit,np):
    require(left.shape==right.shape and np.isfinite(left).all() and np.isfinite(right).all(),'Nonfinite or mismatched parity arrays')
    error=float(np.max(np.abs(left.astype(np.float64)-right.astype(np.float64))))
    require(error<=limit,'Numerical parity exceeded: '+str(error));return error


def pooled_vectors(states,pools,mx,np):
    require(states.ndim==3 and states.shape[0]==1 and states.shape[2]==2560,'Invalid full suffix-state shape')
    # CPU float64 oracle reads the same transient states, not another encoder pass.
    begin=time.monotonic();host=np.asarray(states)[0];copy_seconds=time.monotonic()-begin
    native_seconds=cpu_seconds=0.
    require(host.dtype==np.float32 and np.isfinite(host).all(),'Invalid suffix-state dtype/values')
    vectors=[];errors=[]
    for positions in pools:
        if positions:
            begin=time.monotonic();native=mx.mean(states[0,mx.array(positions),:],axis=0);mx.eval(native)
            value=np.asarray(native).copy();native_seconds+=time.monotonic()-begin
            begin=time.monotonic();oracle=host[positions].astype(np.float64).mean(axis=0).astype(np.float32);cpu_seconds+=time.monotonic()-begin
        else:value=np.zeros(2560,dtype=np.float32);oracle=np.zeros(2560,dtype=np.float64)
        require(value.shape==(2560,) and value.dtype==np.float32 and np.isfinite(value).all(),'Invalid pooled vector')
        error=np.abs(value.astype(np.float64)-oracle)
        require(bool(np.all(error<=1e-5)),'Native pooling differs from float64 same-state mean')
        vectors.append(value);errors.append(float(np.max(error)))
    result=np.stack([host[-1].copy()]+vectors)
    require(result.shape==(7,2560) and np.isfinite(result).all(),'Seven-vector output invalid')
    return result,errors,dict(native_pool_seconds=native_seconds,cpu_state_copy_seconds=copy_seconds,cpu_float64_mean_seconds=cpu_seconds)


def run(plan_path,plan_sha,output):
    require(not output.exists(),'Refusing existing resource output')
    plan,paths,inventory,records=preflight(plan_path,plan_sha)
    import numpy as np
    import mlx.core as mx
    from roguard.gemma4_review import _Engine,_score
    from roguard.gemma4_contract import PREFIX_TOKENS,FILES
    from roguard.interpreter import prepare_input
    from roguard.gemma4_attention import capture_with_prefix,replay
    from training.runtime_attention_continuation_v10 import native_cached_features
    alignment=types.ModuleType('registered_role_alignment');alignment.__file__=str(paths['alignment_source'])
    raw=paths['alignment_source'].read_bytes();require(hashlib.sha256(raw).hexdigest()==ALIGNMENT_SHA,'Alignment bytes changed')
    exec(compile(raw,str(paths['alignment_source']),'exec'),alignment.__dict__)
    require(canonical(FILES)==canonical(plan['base_files']),'Actual pinned base contract differs')
    origins(plan);bindings(plan);disk_check(output,FILES)
    # Global16-case retokenization/role alignment happens before any encoder.
    with alignment.pinned_tokenizer(plan['base_snapshot']) as (tokenizer,provenance):
        for row,record in zip(inventory,records):
            actual=enrich(alignment.align(row['payload'],tokenizer,row.get('focus')),row)
            require(canonical(actual)==canonical(record),'Registered alignment differs before encoder')
    bindings(plan);origins(plan)
    require(sha(plan_path)==plan_sha and sha(__file__)==plan['source_sha256'],'Source/plan changed after tokenization')
    output.mkdir(parents=True,exist_ok=False)
    (output/'source.py').write_bytes(Path(__file__).read_bytes());(output/'registered-plan.json').write_bytes(Path(plan_path).read_bytes())
    report=dict(format='roguard-role-pooling-resource-result-1',complete=False,source_sha256=sha(__file__),
        plan_sha256=plan_sha,languages={},training_performed=False,accuracy_evaluation_performed=False,
        full_token_states_saved=False,encoder_loaded=True,qualified=False)
    started=time.monotonic()
    def persist():
        report['seconds']=time.monotonic()-started;tmp=output/'report.tmp';tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(output/'report.json')
    persist()
    saved_vectors=np.zeros((16,7,2560),dtype=np.float32);saved_logits=np.zeros((16,3,19),dtype=np.float32)
    expected_observations={}
    (output/'inventory.jsonl').write_bytes(paths['inventory'].read_bytes())
    (output/'alignment-records.jsonl').write_bytes(paths['alignments'].read_bytes())
    for language in ('ro','uk'):
        selected=[(i,row,record) for i,(row,record) in enumerate(zip(inventory,records)) if row['payload']['language']==language]
        current=dict(complete=False,rows=len(selected),checked=0,observations=[]);report['languages'][language]=current
        engine=model=prefix=states=suffix=state=qualified=logits=qualified_logits=vectors=direct=direct_logits=None
        sampler=DiskSampler(output)
        try:
            bindings(plan);origins(plan)
            require(sha(plan_path)==plan_sha and sha(__file__)==plan['source_sha256'],'Source/plan changed before encoder')
            disk_check(output,FILES);sampler.start();mx.reset_peak_memory();begin=time.monotonic()
            engine=_Engine(Path(plan['base_snapshot']),paths[language+'_weights'],V9[language]);model=engine.model;model.eval()
            current['load_seconds']=time.monotonic()-begin;current['rss_after_load']=current_rss()
            require(current['load_seconds']<=900 and peak_rss()<=12*1024**3,'Encoder load exceeds resource budget')
            prefix_begin=time.monotonic()
            prefix=engine.make_cache(model.base);mx.eval(model.base.model(mx.array([PREFIX_TOKENS]),cache=prefix))
            fingerprint=cache_fingerprint(prefix,np,mx);current['prefix_seconds']=time.monotonic()-prefix_begin
            spec=read(paths[language+'_spec']);origins(plan);bindings(plan)
            for index,row,record in selected:
                case_start=time.monotonic();disk_check(output,FILES)
                actual=enrich(alignment.align(row['payload'],engine.tokenizer,row.get('focus')),row)
                require(canonical(actual)==canonical(record),'Engine alignment differs')
                pools=validate_alignment(actual);tokens=actual['classifier_token_ids']
                suffix=mx.array([tokens[387:]])
                phase_start=time.monotonic();states=model.base.model(suffix,cache=deepcopy(prefix));mx.eval(states)
                native_seconds=time.monotonic()-phase_start
                require(states.shape==(1,len(tokens)-387,2560),'Full suffix state count differs')
                phase_start=time.monotonic();vectors,errors,pool_times=pooled_vectors(states,pools,mx,np)
                pooling_and_cpu_audit_seconds=time.monotonic()-phase_start
                head_start=time.monotonic();logits=model.heads[language](states[:,-1,:]);mx.eval(logits)
                native_head_seconds=time.monotonic()-head_start
                require(cache_fingerprint(prefix,np,mx)==fingerprint,'Prefix changed after native full states')
                phase_start=time.monotonic()
                direct=native_cached_features(model,suffix,prefix);direct_logits=model.heads[language](direct);mx.eval(direct,direct_logits)
                direct_error=compare(vectors[0],np.asarray(direct)[0],0.,np)
                require(cache_fingerprint(prefix,np,mx)==fingerprint,'Prefix changed after direct native reference')
                state=capture_with_prefix(model.base,suffix,deepcopy(prefix),start=24);mx.eval(state)
                qualified=replay(model.base,state)[:,0,:];qualified_logits=model.heads[language](qualified);mx.eval(qualified,qualified_logits)
                z=np.asarray(logits)[0].copy();q=np.asarray(qualified_logits)[0]
                require(z.shape==(19,) and z.dtype==np.float32,'Wrong head logit contract')
                feature_error=compare(vectors[0],np.asarray(qualified)[0],1e-3,np)
                head_error=compare(z,q,1e-3,np)
                d=np.asarray(direct_logits)[0].copy();compare(z,d,0.,np)
                require(seven_argmax(z)==seven_argmax(q)==seven_argmax(d),'Seven head argmax differs')
                require(cache_fingerprint(prefix,np,mx)==fingerprint,'Prefix changed after capture/replay reference')
                reference_seconds=time.monotonic()-phase_start
                phase_start=time.monotonic()
                serving_prefix,_,_=prepare_input(row['payload'],row.get('focus'))
                score=float(_score(z,spec));served=engine.scores([serving_prefix],spec)[0]
                score_error=compare(np.array([score]),np.array([served]),1e-10,np)
                require((score>=spec['threshold'])==(served>=spec['threshold']),'Fixed served decision differs')
                require(cache_fingerprint(prefix,np,mx)==fingerprint,'Shared immutable prefix changed')
                score_seconds=time.monotonic()-phase_start
                case=row['case_id'];saved_vectors[index]=vectors;saved_logits[index]=np.stack([z,d,q])
                np.save(output/'vectors.npy',saved_vectors,allow_pickle=False);np.save(output/'logits.npy',saved_logits,allow_pickle=False)
                observation=dict(case_id=case,language=language,token_sha256=actual['classifier_token_sha256'],
                    vector_names=VECTOR_NAMES,role_and_control_validity=[bool(p) for p in pools],
                    role_presence=actual['shared_structural_role_presence'],pool_counts=[len(p) for p in pools],
                    pool_mean_max_errors=errors,direct_native_final_error=direct_error,feature_parity_error=feature_error,head_parity_error=head_error,
                    seven_argmax_differences=0,served_decision_differences=0,global_positions=actual['role_pool_positions']+actual['role_blind_pool_positions'],
                    suffix_positions=pools,native_forward_seconds=native_seconds,pooling_and_cpu_audit_seconds=pooling_and_cpu_audit_seconds,pool_phase_times=pool_times,
                    references_seconds=reference_seconds,score_seconds=score_seconds,native_head_seconds=native_head_seconds,
                    score_parity_error=score_error,score=score,served_score=served,prefix_sha256=fingerprint,
                    seconds=time.monotonic()-case_start,mlx_active_memory=mx.get_active_memory(),
                    mlx_peak_memory=mx.get_peak_memory(),mlx_cache_memory=mx.get_cache_memory(),
                    disk_free=shutil.disk_usage(output).free,process_peak_rss_bytes=peak_rss(),process_current_rss_bytes=current_rss())
                (output/(case+'-observation.json')).write_text(json.dumps(observation,indent=2,allow_nan=False)+'\n')
                expected_observations[case]=observation;current['observations'].append(observation);current['checked']+=1
                require(time.monotonic()-begin<=900 and peak_rss()<=12*1024**3,'Resource time/RSS ceiling reached')
                require(sum(p.stat().st_size for p in output.iterdir())<=64*1024**2,'Resource output cap reached')
                states=suffix=state=qualified=logits=qualified_logits=vectors=z=q=d=direct=direct_logits=None
                mx.clear_cache();persist()
        except Exception as error:
            current['failure']=dict(type=type(error).__name__,message=str(error));persist();raise
        finally:
            # No output-buffer flush can prevent release; partial files remain.
            engine=model=prefix=states=suffix=state=qualified=logits=qualified_logits=vectors=direct=direct_logits=None
            z=q=d=None
            active=sys.exc_info()[1];cleanup=[]
            try:current['disk_sampling']=sampler.finish()
            except Exception as error:cleanup.append(str(error))
            for action in (gc.collect,mx.clear_cache):
                try:action()
                except Exception as error:cleanup.append(str(error))
            current['cleanup_failures']=cleanup
            try:persist()
            except Exception as error:cleanup.append('Persist cleanup report: '+str(error))
            if cleanup and active is None:raise RuntimeError('Cleanup failed: '+repr(cleanup))
        current['complete']=True;persist()
    bindings(plan);origins(plan)
    require(sha(plan_path)==plan_sha and sha(__file__)==plan['source_sha256'],'Final source/plan changed')
    require(sum(v['checked'] for v in report['languages'].values())==16,'Incomplete16-case run')
    snapshot={p.name:sha(p) for p in output.iterdir() if p.name not in ('report.json','report.tmp')}
    for filename,expected in [('vectors.npy',saved_vectors),('logits.npy',saved_logits)]:
        actual=np.load(output/filename,allow_pickle=False)
        require(actual.dtype==np.float32 and actual.shape==expected.shape and np.isfinite(actual).all()
            and actual.tobytes()==expected.tobytes(),'Saved vectors/logits differ')
    for case,expected in expected_observations.items():require(canonical(read(output/(case+'-observation.json')))==canonical(expected),'Saved observation differs')
    for name,source in [('source.py',Path(__file__)),('registered-plan.json',plan_path),('inventory.jsonl',paths['inventory']),('alignment-records.jsonl',paths['alignments'])]:
        require((output/name).read_bytes()==Path(source).read_bytes(),'Retained source/input snapshot differs')
    bindings(plan);origins(plan)
    require(sha(plan_path)==plan_sha and sha(__file__)==plan['source_sha256'],'Source/plan changed during output readback')
    require(snapshot=={p.name:sha(p) for p in output.iterdir() if p.name not in ('report.json','report.tmp')},'Artifact changed during readback')
    report['artifacts']=snapshot
    report['non_report_artifact_bytes']=sum((output/name).stat().st_size for name in snapshot)
    report['byte_count_scope']='Only hash-bound non-report artifacts; mutable report.json excluded.'
    report.update(complete=True,decision='BOUNDED_SYNTHETIC_RESOURCE_MEASUREMENT_ONLY_REQUIRES_AUDIT');persist();return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.plan,args.plan_sha256,args.output)
