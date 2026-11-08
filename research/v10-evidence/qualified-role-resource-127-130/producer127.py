"""Unregistered qualified-final75 and count-matched role resource prototype; no fitting.

All actual encoder execution requires an exact root-registered plan. Full suffix
states are transient; only six main vectors plus metadata/logit diagnostics per synthetic case persist.
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

FORMAT='roguard-qualified-final-resource-plan-1'
V9={'ro':'114d3e89191d06152cceccd080114998901a4f0bf6a9540de8e11dd8b6cf6b09',
    'uk':'0fee43a071d2c07c26e8b3d69cacb4933fd284bc64dbee0d18fe5481e1e4815f'}
PARENT75={'ro':'1c18db5e5fc26baf4c0739d053c0546338201d76fcb9061e47477c14650535e3',
          'uk':'466a1900655c85a6db809adb384d90fc634357a4f7b72f1415274c4787b3ecb7'}
SPECS={'ro':'5fdeae97843d873755ed41a02453b5c29823fd8028508c3e99d1586175c8f4a9',
       'uk':'bbf3fea8ee298c40d86701851fe045352ea1a924ba5287dd52b70e59ce25de27'}
ALIGNMENT_SHA='d684b8b592eed2e5241945211bf689097ca09a3bb44b58eb4dba3b7650bdbc5c'
VECTOR_NAMES=['qualified_final','focal','aligned_prior_minor','aligned_prior_nonminor','chronological_prior1','chronological_prior2']
LOGIT_NAMES=['qualified75','qualified75_repeat','nativefinal75_diagnostic']
CASES_SHA='14b46686a9ceceda17b01623cfc0eab1fcc8afb072b0ceeeb15b3272c5c0d370'
RECORDS_SHA='81f688e9a47f42b32d4ac8469b77145bc90edec244fc99f1af3c650ac56f22b1'
DESIGN_SHA='96d26e1978dce3c027388a2d540e9e41aa2e583bfc3ca447df795f027f66508a'
ADDENDUM_SHA='cd5d1e7562227560e2da6496d3155103334e955914ee1da4ee9c23010b41975b'
CONTROL_SHA='e7186eea9850bd39b9ef0df4999b505c6532d675c6ed7c4f5a412d93cb0c51f1'
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
        and plan['authorization']=='ROOT_REGISTERED_QUALIFIED_FINAL_RESOURCE_ONLY'
        and plan['source_sha256']==sha(__file__) and plan['rows']==16
        and plan['maximum_tokens']==600 and plan['vectors']==VECTOR_NAMES
        and plan['training_performed'] is False and plan['accuracy_evaluation_permitted'] is False,
        'Unregistered resource pilot or changed scope')
    bindings(plan);paths={k:Path(v['path']) for k,v in plan['files'].items()}
    require(sha(paths['alignment_source'])==ALIGNMENT_SHA,'Wrong frozen alignment source')
    require(sha(paths['architecture_addendum'])==ADDENDUM_SHA and sha(paths['control_amendment'])==CONTROL_SHA,'Qualified architecture/control contract changed')
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
            and plan['files'][language+'_parent_weights']['sha256']==PARENT75[language]
            and plan['files'][language+'_spec']['sha256']==SPECS[language],'Original-v9 backbone and unchanged75 head/spec required')
        spec=read(paths[language+'_spec']);require(spec['language']==language and spec['adapter_sha256']==PARENT75[language],'Wrong original spec')
    terminal=read(paths['prior120_terminal_report'])
    require(terminal['complete'] is True,'Prior encoder120 must be terminal before resource pilot')
    previous=read(paths['prior_native_terminal_report'])
    require(previous['complete'] is False and previous['source_sha256']=='97cc2debafde6915dc8035c0e4464e315c2faf16e58dd4c2705f4942a30c71d2'
        and previous['plan_sha256']==sha(paths['prior_native_plan'])
        and type(plan['prior_native_exit_code']) is int and plan['prior_native_exit_code']==1,'Prior native-only failure must remain bound')
    warm=read(paths['warm_start90_report'])
    require(warm['complete'] is True and warm['qualified'] is False
        and all(warm['languages'][lang]['artifacts'][lang+'-selected.safetensors']==PARENT75[lang] for lang in ('ro','uk')),'Unchanged75 provenance differs')
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


def native_pooled_vectors(states,pools,mx,np):
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
    result=np.stack(vectors)
    require(result.shape==(5,2560) and np.isfinite(result).all(),'Five native pools invalid')
    return result,host[-1].copy(),errors,dict(native_pool_seconds=native_seconds,cpu_state_copy_seconds=copy_seconds,cpu_float64_mean_seconds=cpu_seconds)


def runtime_alignment(alignment,payload,tokenizer,focus=None):
    """HF offsets plus independently identical actual runtime-wrapper token IDs.

    TokenizerWrapper forwards attributes but Python does not forward __call__.
    Unwrap only the pinned runtime wrapper for offset encoding; preserve the
    wrapper itself for the exact serving classifier tokenization check.
    """
    from mlx_lm.tokenizer_utils import TokenizerWrapper
    from roguard.interpreter import prepare_input
    from roguard.gemma4_contract import classifier_tokens
    offset_tokenizer=tokenizer._tokenizer if isinstance(tokenizer,TokenizerWrapper) else tokenizer
    require(callable(offset_tokenizer) and offset_tokenizer.is_fast,'Runtime offsets require a callable fast tokenizer')
    actual=alignment.align(payload,offset_tokenizer,focus)
    serving_prefix,_,_=prepare_input(payload,focus)
    runtime_ids=classifier_tokens(tokenizer,serving_prefix)
    require(actual['classifier_token_ids']==runtime_ids,'Underlying offset tokenizer differs from actual runtime wrapper IDs')
    return actual


def pool_contract(record,np):
    validate_alignment(record)
    roles=record['role_pool_positions'];prior=sorted(roles[1]+roles[2]);n=len(roles[1]);chronological=[prior[:n],prior[n:]]
    require(chronological==record['chronological_prior_control_positions'],'Frozen chronological masks differ')
    require([len(p) for p in chronological]==[len(p) for p in roles[1:]],'Chronological counts/emptiness differ')
    pools=roles+chronological
    metadata=np.asarray([len(p)/1024 for p in roles]+[float(v) for v in record['shared_structural_role_presence']],dtype=np.float32)
    require(metadata.shape==(6,) and np.isfinite(metadata).all(),'Shared metadata invalid')
    return [[i-387 for i in p] for p in pools],metadata


def parameter_proof(model,language,np,mx):
    from mlx.utils import tree_flatten
    excluded='heads.'+language+'.'
    all_parameters=dict(tree_flatten(model.parameters()))
    outside={name:[id(value),list(value.shape),str(value.dtype)] for name,value in all_parameters.items() if not name.startswith(excluded)}
    trainable={name:value for name,value in tree_flatten(model.trainable_parameters()) if not name.startswith(excluded)}
    head={name:value for name,value in tree_flatten(model.heads[language].parameters())}
    require(set(head)=={'first.weight','first.bias','last.weight','last.bias'},'Existing75 head class inventory differs')
    fingerprint=lambda values:cache_fingerprint([types.SimpleNamespace(state=values,meta_state=None)],np,mx)
    return dict(nonparent_parameter_identity_sha256=digest(outside),other_trainable_value_sha256=fingerprint(trainable),
        parent_head_value_sha256=fingerprint(head),frozen_base_values_hashed=False)


def install_parent_head(model,language,path,np,mx):
    from safetensors.numpy import load
    from mlx.utils import tree_flatten
    raw=Path(path).read_bytes();require(hashlib.sha256(raw).hexdigest()==PARENT75[language],'Original75 head bytes changed')
    tensors=load(raw);existing=dict(tree_flatten(model.heads[language].parameters()))
    require(set(existing)=={'first.weight','first.bias','last.weight','last.bias'},'Existing parenthead inventory differs')
    values={name:tensors['heads.'+language+'.'+name] for name in existing}
    for name,value in values.items():
        require(value.dtype==np.float32 and value.shape==existing[name].shape and np.isfinite(value).all(),'Invalid unchanged75 head tensor')
    before=parameter_proof(model,language,np,mx)
    model.heads[language].load_weights([(name,mx.array(value)) for name,value in values.items()],strict=True)
    model.eval();mx.eval(model.heads[language].parameters())
    installed=dict(tree_flatten(model.heads[language].parameters()))
    for name,value in values.items():
        actual=np.asarray(installed[name]);require(actual.dtype==np.float32 and actual.tobytes()==value.tobytes(),'Installed75 head differs from bound tensors')
    after=parameter_proof(model,language,np,mx)
    require(before['nonparent_parameter_identity_sha256']==after['nonparent_parameter_identity_sha256']
        and before['other_trainable_value_sha256']==after['other_trainable_value_sha256'],'Parent head installation changed other parameters')
    return after


def identity_parent_logits(parent,residual=None):
    import numpy as np
    require(isinstance(parent,np.ndarray) and parent.dtype==np.float32 and parent.shape==(19,) and np.isfinite(parent).all(),'Invalid qualified parent19 logits')
    require(residual is None,'No learned residual implemented or admitted')
    return parent


def qualified_projection(model,language,suffix,prefix,capture,replay,mx):
    state=capture(model.base,suffix,deepcopy(prefix),start=24);mx.eval(state)
    feature=replay(model.base,state)[:,0,:];logits=model.heads[language](feature);mx.eval(feature,logits)
    return feature,logits


def run(plan_path,plan_sha,output):
    require(not output.exists(),'Refusing existing qualified-resource output')
    plan,paths,inventory,records=preflight(plan_path,plan_sha)
    import numpy as np
    import mlx.core as mx
    from roguard.gemma4_review import _Engine,_score
    from roguard.gemma4_contract import PREFIX_TOKENS,FILES
    from roguard.interpreter import prepare_input
    from roguard.gemma4_attention import capture_with_prefix,replay
    alignment=types.ModuleType('registered_role_alignment');alignment.__file__=str(paths['alignment_source'])
    raw=paths['alignment_source'].read_bytes();require(hashlib.sha256(raw).hexdigest()==ALIGNMENT_SHA,'Alignment bytes changed')
    exec(compile(raw,str(paths['alignment_source']),'exec'),alignment.__dict__)
    require(canonical(FILES)==canonical(plan['base_files']),'Actual base contract differs')
    origins(plan);bindings(plan);disk_check(output,FILES)
    with alignment.pinned_tokenizer(plan['base_snapshot']) as (tokenizer,provenance):
        for row,record in zip(inventory,records):
            actual=enrich(alignment.align(row['payload'],tokenizer,row.get('focus')),row)
            require(canonical(actual)==canonical(record),'Frozen alignment differs before encoder');pool_contract(actual,np)
    bindings(plan);origins(plan)
    require(sha(plan_path)==plan_sha and sha(__file__)==plan['source_sha256'],'Source/plan changed before output')
    output.mkdir(parents=True,exist_ok=False)
    for name,path in [('source.py',Path(__file__)),('registered-plan.json',plan_path),('inventory.jsonl',paths['inventory']),('alignment-records.jsonl',paths['alignments'])]:
        (output/name).write_bytes(Path(path).read_bytes())
    report=dict(format='roguard-qualified-final-resource-result-1',complete=False,source_sha256=sha(__file__),plan_sha256=plan_sha,
        languages={},training_performed=False,accuracy_evaluation_performed=False,full_token_states_saved=False,
        encoder_loaded=True,qualified=False,learned_residual_implemented=False,baseline='Fresh qualified capture/replay unchanged75 head; original-v9 backbone',
        native_final_interchangeability_claim=False,vector_names=VECTOR_NAMES,logit_names=LOGIT_NAMES,
        alignment_record_scope='Copied frozen stage0 records describe the prior7-vector mask census; current6-vector schema uses their verified chronological masks and is defined by vector_names.')
    start=time.monotonic()
    def persist():
        report['seconds']=time.monotonic()-start;tmp=output/'report.tmp';tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(output/'report.json')
    persist();saved_vectors=np.zeros((16,6,2560),dtype=np.float32);saved_metadata=np.zeros((16,6),dtype=np.float32)
    saved_logits=np.zeros((16,3,19),dtype=np.float32);expected_observations={}
    for language in ('ro','uk'):
        selected=[(i,row,record) for i,(row,record) in enumerate(zip(inventory,records)) if row['payload']['language']==language]
        current=dict(complete=False,rows=len(selected),checked=0,observations=[]);report['languages'][language]=current
        engine=model=prefix=states=suffix=pooled=native_final=native_logits=qfeature=qlogits=repeat_feature=repeat_logits=None
        z=nz=rz=vectors=identity=None;sampler=DiskSampler(output)
        try:
            bindings(plan);origins(plan);require(sha(plan_path)==plan_sha and sha(__file__)==plan['source_sha256'],'Source/plan changed before encoder')
            disk_check(output,FILES);sampler.start();mx.reset_peak_memory();begin=time.monotonic()
            engine=_Engine(Path(plan['base_snapshot']),paths[language+'_weights'],V9[language]);model=engine.model;model.eval()
            current['load_seconds']=time.monotonic()-begin;current['rss_after_load']=current_rss()
            require(current['load_seconds']<=900 and peak_rss()<=12*1024**3,'Encoder load exceeds resource budget')
            phase=time.monotonic();proof=install_parent_head(model,language,paths[language+'_parent_weights'],np,mx)
            current['head_install_seconds']=time.monotonic()-phase;current['parameter_proof']=proof
            phase=time.monotonic();prefix=engine.make_cache(model.base);mx.eval(model.base.model(mx.array([PREFIX_TOKENS]),cache=prefix))
            fingerprint=cache_fingerprint(prefix,np,mx);current['prefix_seconds']=time.monotonic()-phase
            spec=read(paths[language+'_spec']);origins(plan);bindings(plan)
            for index,row,record in selected:
                case_start=time.monotonic();disk_check(output,FILES)
                actual=enrich(runtime_alignment(alignment,row['payload'],engine.tokenizer,row.get('focus')),row)
                require(canonical(actual)==canonical(record),'Engine token alignment differs')
                pools,metadata=pool_contract(actual,np);tokens=actual['classifier_token_ids'];suffix=mx.array([tokens[387:]])
                phase=time.monotonic();states=model.base.model(suffix,cache=deepcopy(prefix));mx.eval(states)
                native_seconds=time.monotonic()-phase;require(states.shape==(1,len(tokens)-387,2560),'Native suffix shape differs')
                phase=time.monotonic();pooled,native_final,pool_errors,pool_times=native_pooled_vectors(states,pools,mx,np)
                pool_total=time.monotonic()-phase
                phase=time.monotonic();native_logits=model.heads[language](states[:,-1,:]);mx.eval(native_logits);nz=np.asarray(native_logits)[0].copy()
                native_diagnostic_head_seconds=time.monotonic()-phase
                require(nz.dtype==np.float32 and nz.shape==(19,) and np.isfinite(nz).all(),'Invalid native diagnostic logits')
                require(cache_fingerprint(prefix,np,mx)==fingerprint,'Prefix changed during native pooling')
                states=native_logits=None;mx.clear_cache()  # release full states before qualified path
                phase=time.monotonic();qfeature,qlogits=qualified_projection(model,language,suffix,prefix,capture_with_prefix,replay,mx)
                qualified_seconds=time.monotonic()-phase
                qhost=np.asarray(qfeature)[0].copy();z=np.asarray(qlogits)[0].copy()
                require(qhost.shape==(2560,) and qhost.dtype==np.float32 and np.isfinite(qhost).all(),'Invalid qualified final')
                identity=identity_parent_logits(z);require(identity is z and identity.tobytes()==z.tobytes(),'Zero-residual baseline identity failed')
                require(cache_fingerprint(prefix,np,mx)==fingerprint,'Prefix changed during qualified baseline')
                vectors=np.concatenate([qhost[None,:],pooled],axis=0)
                require(vectors.shape==(6,2560) and vectors.dtype==np.float32 and np.isfinite(vectors).all(),'Six-vector inventory invalid')
                qfeature=qlogits=None;mx.clear_cache()
                phase=time.monotonic();repeat_feature,repeat_logits=qualified_projection(model,language,suffix,prefix,capture_with_prefix,replay,mx)
                rz=np.asarray(repeat_logits)[0].copy();repeat_feature_error=compare(qhost,np.asarray(repeat_feature)[0],0.,np)
                repeat_logit_error=compare(identity,rz,0.,np)
                require(seven_argmax(identity)==seven_argmax(rz),'Qualified repeat seven-head mismatch')
                require(cache_fingerprint(prefix,np,mx)==fingerprint,'Prefix changed during qualified repeat')
                repeat_seconds=time.monotonic()-phase;repeat_feature=repeat_logits=None;mx.clear_cache()
                phase=time.monotonic();serving_prefix,_,_=prepare_input(row['payload'],row.get('focus'))
                score=float(_score(identity,spec));served=engine.scores([serving_prefix],spec)[0]
                score_error=compare(np.asarray([score]),np.asarray([served]),1e-10,np)
                require((score>=spec['threshold'])==(served>=spec['threshold'])==(_score(rz,spec)>=spec['threshold']),'Qualified75 served decision differs')
                require(cache_fingerprint(prefix,np,mx)==fingerprint,'Prefix changed during runtime reference')
                runtime_seconds=time.monotonic()-phase
                native_score=float(_score(nz,spec));native_feature_drift=float(np.max(np.abs(native_final.astype(np.float64)-qhost.astype(np.float64))))
                native_logit_drift=float(np.max(np.abs(nz.astype(np.float64)-z.astype(np.float64))))
                saved_vectors[index]=vectors;saved_metadata[index]=metadata;saved_logits[index]=np.stack([identity,rz,nz])
                for name,values in [('vectors.npy',saved_vectors),('metadata.npy',saved_metadata),('logits.npy',saved_logits)]:np.save(output/name,values,allow_pickle=False)
                case=row['case_id'];observation=dict(case_id=case,language=language,token_sha256=actual['classifier_token_sha256'],
                    vector_names=VECTOR_NAMES,logit_names=LOGIT_NAMES,global_positions=actual['role_pool_positions']+actual['chronological_prior_control_positions'],
                    suffix_positions=pools,pool_counts=[len(p) for p in pools],pool_validity=[bool(p) for p in pools],
                    shared_metadata=metadata.tolist(),role_presence=actual['shared_structural_role_presence'],
                    chronological_masks_identical=actual['chronological_masks_identical'],pool_mean_max_errors=pool_errors,
                    native_states_released_before_qualified=True,qualified_repeat_feature_error=repeat_feature_error,
                    qualified_repeat_logit_error=repeat_logit_error,qualified_seven_argmax_differences=0,qualified_binary_differences=0,
                    zero_residual_same_object=True,zero_residual_bytes_identical=True,score=score,served_score=served,score_parity_error=score_error,
                    native_final_diagnostic=dict(feature_max_abs=native_feature_drift,logit_max_abs=native_logit_drift,
                        score=native_score,score_difference=native_score-score,
                        seven_argmax_differences=sum(a!=b for a,b in zip(seven_argmax(nz),seven_argmax(z))),
                        binary_difference=int((native_score>=spec['threshold'])!=(score>=spec['threshold'])),interchangeability_claim=False),
                    native_forward_seconds=native_seconds,pool_phase_times=pool_times,pooling_and_cpu_audit_seconds=pool_total,
                    qualified_baseline_seconds=qualified_seconds,primary_two_path_seconds=native_seconds+pool_times['native_pool_seconds']+qualified_seconds,
                    extra_native_diagnostic_head_seconds=native_diagnostic_head_seconds,extra_qualified_repeat_seconds=repeat_seconds,
                    extra_runtime_reference_seconds=runtime_seconds,seconds=time.monotonic()-case_start,prefix_sha256=fingerprint,
                    mlx_active_memory=mx.get_active_memory(),mlx_peak_memory=mx.get_peak_memory(),mlx_cache_memory=mx.get_cache_memory(),
                    process_peak_rss_bytes=peak_rss(),process_current_rss_bytes=current_rss(),disk_free=shutil.disk_usage(output).free)
                (output/(case+'-observation.json')).write_text(json.dumps(observation,indent=2,allow_nan=False)+'\n')
                expected_observations[case]=observation;current['observations'].append(observation);current['checked']+=1
                require(time.monotonic()-begin<=900 and peak_rss()<=12*1024**3,'Resource time/RSS ceiling reached')
                require(sum(p.stat().st_size for p in output.iterdir())<=64*1024**2,'Resource output cap reached')
                states=suffix=pooled=native_final=native_logits=qfeature=qlogits=repeat_feature=repeat_logits=None
                z=nz=rz=vectors=identity=qhost=None;mx.clear_cache();persist()
            require(parameter_proof(model,language,np,mx)==proof,'Parameters changed during resource forward passes')
        except Exception as error:
            current['failure']=dict(type=type(error).__name__,message=str(error));persist();raise
        finally:
            engine=model=prefix=states=suffix=pooled=native_final=native_logits=qfeature=qlogits=repeat_feature=repeat_logits=None
            z=nz=rz=vectors=identity=qhost=None;active=sys.exc_info()[1];cleanup=[]
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
    bindings(plan);origins(plan);require(sha(plan_path)==plan_sha and sha(__file__)==plan['source_sha256'],'Final source/plan changed')
    require(sum(v['checked'] for v in report['languages'].values())==16,'Incomplete16-case run')
    snapshot={p.name:sha(p) for p in output.iterdir() if p.name not in ('report.json','report.tmp')}
    for filename,expected in [('vectors.npy',saved_vectors),('metadata.npy',saved_metadata),('logits.npy',saved_logits)]:
        actual=np.load(output/filename,allow_pickle=False)
        require(actual.dtype==np.float32 and actual.shape==expected.shape and np.isfinite(actual).all() and actual.tobytes()==expected.tobytes(),'Saved qualified resource arrays differ')
    for case,expected in expected_observations.items():require(canonical(read(output/(case+'-observation.json')))==canonical(expected),'Saved observation differs')
    for name,path in [('source.py',Path(__file__)),('registered-plan.json',plan_path),('inventory.jsonl',paths['inventory']),('alignment-records.jsonl',paths['alignments'])]:
        require((output/name).read_bytes()==Path(path).read_bytes(),'Retained source/input snapshot differs')
    bindings(plan);origins(plan);require(sha(plan_path)==plan_sha and sha(__file__)==plan['source_sha256'],'Source/plan changed during output readback')
    require(snapshot=={p.name:sha(p) for p in output.iterdir() if p.name not in ('report.json','report.tmp')},'Artifact changed during readback')
    report['artifacts']=snapshot;report['non_report_artifact_bytes']=sum((output/name).stat().st_size for name in snapshot)
    report['byte_count_scope']='Only hash-bound non-report artifacts; mutable report.json excluded.'
    report.update(complete=True,decision='QUALIFIED75_IDENTITY_AND_TWO_PATH_RESOURCE_ONLY_NO_TRAINING')
    try:
        persist()
        final_report_raw=(output/'report.json').read_bytes()
        require(canonical(json.loads(final_report_raw))==canonical(report),'Final report readback differs')
        final_report_sha=hashlib.sha256(final_report_raw).hexdigest()
        bindings(plan);origins(plan)
        require(sha(plan_path)==plan_sha and sha(__file__)==plan['source_sha256'],'Source/plan changed after final report write')
        require(snapshot=={p.name:sha(p) for p in output.iterdir() if p.name not in ('report.json','report.tmp')},'Artifact changed after final report write')
        require(sha(output/'report.json')==final_report_sha,'Final report changed during closing checks')
    except Exception as error:
        report.update(complete=False,finalization_failure=dict(type=type(error).__name__,message=str(error)))
        persist();raise
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.plan,args.plan_sha256,args.output)
