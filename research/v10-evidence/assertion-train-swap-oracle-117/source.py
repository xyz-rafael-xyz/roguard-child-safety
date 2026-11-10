"""Unlaunched TRAIN-only oracle reachability diagnostic; never a learned fit.

Global stdlib preflight checks both languages before any numerical import or
head replay. Original88 DEV metadata is inspected only to identify and exclude
its partition; no DEV features, labels or logits are selected for computation.
"""
import argparse
import ast
from collections import Counter
import hashlib
import importlib.util
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import sys

FORMAT='roguard-train-oracle-feasibility-plan-1'
COUNTS={'ro':(17376,13174),'uk':(20390,15755)}
SIZES=(3,3,4,3,2,2)
THRESHOLDS={'ro':.7517333793124337,'uk':.9274248149759536}
WEIGHTS75={'ro':'1c18db5e5fc26baf4c0739d053c0546338201d76fcb9061e47477c14650535e3',
           'uk':'466a1900655c85a6db809adb384d90fc634357a4f7b72f1415274c4787b3ecb7'}
SPECS75={'ro':'5fdeae97843d873755ed41a02453b5c29823fd8028508c3e99d1586175c8f4a9',
         'uk':'bbf3fea8ee298c40d86701851fe045352ea1a924ba5287dd52b70e59ce25de27'}
PROJECTION_SHA='aec96156c0c029c56c490c89802bc22b57e7c3cbe8cbd06b19e11276e99a2959'
ADAPTER_SHA='12754dca395eff37f087a41b049cdea0f966427dc558d72d476f6de37a992fa2'
GELU_SHA='b89b4f643ff7dc2dcfdaab70fe11b0a7641de17740a1fc0ae8713006bf09850e'


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def select_train(path,language,expected):
    """Filter before touching labels. The full ordered(index,row) digest is bound."""
    selected=[];groups=Counter();seen=set();total=dev=0
    with Path(path).open() as f:
        for index,line in enumerate(f):
            row=json.loads(line);total+=1
            require(row.get('partition') in ('train','dev'),'Unexpected partition in old88 metadata')
            if row['partition']=='dev':
                dev+=1
                continue
            require(row.get('language')==language and isinstance(row.get('case_id'),str)
                    and row['case_id'] and row['case_id'] not in seen,'Invalid or duplicate TRAIN identity')
            seen.add(row['case_id'])
            require(isinstance(row.get('fields'),list) and len(row['fields'])==6
                    and all(type(v) is int and 0<=v<size for v,size in zip(row['fields'],SIZES))
                    and type(row.get('gold')) is int and row['gold'] in (0,1),'Invalid TRAIN labels')
            require(isinstance(row.get('group'),str) and row['group'],'Missing TRAIN group')
            selected.append([index,row]);groups[row['group']]+=1
    require(total==expected['total_metadata_rows'] and len(selected)==expected['train_rows'],
            'Original88 TRAIN/metadata population differs')
    require(digest(selected)==expected['train_order_sha256'] and dict(groups)==expected['train_groups'],
            'Exact filtered TRAIN order/group membership differs')
    return selected,dict(total_metadata_rows=total,train_rows=len(selected),dev_metadata_rows_skipped=dev,
                         train_order_sha256=digest(selected),train_groups=dict(groups))


def npy_header(path):
    with Path(path).open('rb') as f:
        require(f.read(6)==b'\x93NUMPY','Not a NumPy feature file')
        v=tuple(f.read(2));require(v in ((1,0),(2,0),(3,0)),'Unsupported NumPy header')
        length=int.from_bytes(f.read(2 if v==(1,0) else 4),'little')
        require(0<length<65536,'Invalid NumPy header length')
        return ast.literal_eval(f.read(length).decode('utf-8' if v==(3,0) else 'latin1'))


def head_header(path,language):
    with Path(path).open('rb') as f:
        length=int.from_bytes(f.read(8),'little');require(0<length<1024**2,'Invalid weight header length')
        header=json.loads(f.read(length))
    for name,shape in [('first.weight',[128,2560]),('first.bias',[128]),('last.weight',[19,128]),('last.bias',[19])]:
        record=header['heads.'+language+'.'+name]
        require(record['dtype']=='F32' and record['shape']==shape,'Original75 head shape/type differs')


def check_files(plan):
    for key,item in plan['files'].items():
        require(Path(item['path']).is_absolute() and sha(item['path'])==item['sha256'],'Bound file changed: '+key)


def bound_bytes(plan,key):
    record=plan['files'][key];data=Path(record['path']).read_bytes()
    require(hashlib.sha256(data).hexdigest()==record['sha256'],'Bound bytes changed before decode: '+key)
    return data


def preflight(plan_path,plan_sha):
    require(sha(plan_path)==plan_sha,'Registered feasibility plan changed')
    plan=read(plan_path)
    require(plan['format']==FORMAT and plan['registered'] is True
            and plan['authorization']=='ROOT_REGISTERED_TRAIN_ORACLE_FEASIBILITY_ONLY'
            and plan['source_sha256']==sha(__file__) and plan['risk_tolerance']==1e-12
            and plan['batch_size']==256 and plan['gelu_function_sha256']==GELU_SHA
            and plan['optimizer_run'] is False and plan['new_probe_scoring'] is False,
            'Unregistered source or changed numerical/scope contract')
    require(set(plan['languages'])=={'ro','uk'},'Both-language preflight is mandatory')
    check_files(plan)
    paths={key:Path(value['path']) for key,value in plan['files'].items()}
    require(sha(paths['serving_adapter'])==ADAPTER_SHA and sha(paths['probability_projection'])==PROJECTION_SHA,
            'Reviewed conservative component changed')
    require(paths['probability_projection'].resolve()==(paths['serving_adapter'].parent.parent/'coupling-projection-stage0-v2/projection.py').resolve(),
            'Adapter projection import path differs from binding')
    ledger_paths={Path(item['path']).resolve() for item in plan['files'].values()}
    root=Path(plan['project_root'])
    require(all(p.resolve() in ledger_paths for p in (root/'src/roguard').rglob('*.py')),
            'Entire portable serving source inventory must be bound')
    for package in ('numpy','torch','safetensors'):
        require(version(package)==plan['numerical_versions'][package],'Numerical package version changed: '+package)
    # Only audit metadata is read. Neither oldDEV arrays nor failed116 predictions are loaded.
    audit=read(paths['feature_audit89']);warm=read(paths['warm_start90_report'])
    adapter_report=read(paths['adapter_report']);review=read(paths['independent_adapter_review'])
    require(audit['verification']=='PASS' and audit['qualified'] is False
            and warm['complete'] is True and warm['qualified'] is False,'Missing original88/75 lineage')
    require(adapter_report['verification']=='PASS' and adapter_report['adapter_sha256']==ADAPTER_SHA
            and adapter_report['original_score_tolerance']==1e-12 and adapter_report['tolerance_relaxed'] is False
            and review['verification']=='PASS_SCOPED_NUMERICAL_GUARD' and review['qualified'] is False,
            'Missing conservative adapter test/review evidence')
    languages={}
    for language in ('ro','uk'):
        expected=plan['languages'][language]
        require((expected['total_metadata_rows'],expected['train_rows'])==COUNTS[language],'Frozen TRAIN count differs')
        for suffix in ('rows','features'):
            require(audit['artifacts'][language+'-'+suffix+('.jsonl' if suffix=='rows' else '.npy')]
                    ==plan['files'][language+'_'+suffix]['sha256'],'Audit89 feature/row binding differs')
        require(plan['files'][language+'_weights75']['sha256']==WEIGHTS75[language]
                and plan['files'][language+'_spec75']['sha256']==SPECS75[language]
                and warm['languages'][language]['artifacts'][language+'-selected.safetensors']==WEIGHTS75[language],
                'Not original75 runtime90 parent')
        spec=read(paths[language+'_spec75'])
        require(spec['adapter_sha256']==WEIGHTS75[language] and spec['language']==language
                and spec['mode']=='coupled_policy' and spec['threshold']==THRESHOLDS[language],
                'Fixed parent scorer/threshold differs')
        selected,summary=select_train(paths[language+'_rows'],language,expected)
        header=npy_header(paths[language+'_features'])
        require(header['shape']==(COUNTS[language][0],2560) and header['descr']=='<f4'
                and header['fortran_order'] is False,'Full original88 feature header differs')
        head_header(paths[language+'_weights75'],language)
        languages[language]=dict(selected=selected,summary=summary,spec=spec)
    require(sha(plan_path)==plan_sha and sha(__file__)==plan['source_sha256'],
            'Plan/source changed during global preflight')
    check_files(plan)
    return plan,paths,languages


def check_origins(plan):
    root=Path(plan['project_root']).resolve();bound={Path(r['path']).resolve():r['sha256'] for r in plan['files'].values()}
    for name,module in list(sys.modules.items()):
        if name=='roguard' or name.startswith('roguard.'):
            path=root/'src'/Path(*name.split('.'));path=path/'__init__.py' if path.is_dir() else path.with_suffix('.py')
            actual=Path(module.__file__).resolve()
            require(actual==path.resolve() and actual in bound and sha(actual)==bound[actual],'Unbound scorer import origin: '+name)


def cpu_head(features,tensors):
    """Exact-erf forward independently verified against MLX in evaluator116."""
    import numpy as np
    import torch
    torch.set_num_threads(2)
    with torch.no_grad():
        x=torch.from_numpy(np.array(features,dtype=np.float32,copy=True))
        t={key:torch.from_numpy(value.copy()) for key,value in tensors.items()}
        norm=(x*x).sum(1,keepdim=True).sqrt()
        require(bool(torch.isfinite(x).all()) and bool((norm>0).all()),'Invalid TRAIN feature row')
        x=x/norm*2560**.5
        h=torch.nn.functional.linear(x,t['first.weight'],t['first.bias'])
        h=h*(1+torch.erf(h/(2.**.5)))/2
        z=torch.nn.functional.linear(h,t['last.weight'],t['last.bias']).numpy().copy()
    require(z.shape==(len(features),19) and z.dtype==np.float32 and np.isfinite(z).all(),'Invalid TRAIN parent logits')
    return z


def run(plan_path,plan_sha,output):
    require(not output.exists(),'Refusing to overwrite feasibility output')
    plan,paths,languages=preflight(plan_path,plan_sha)  # Both languages, before any head replay.
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='2'
    import numpy as np
    from safetensors.numpy import load as load_safetensors
    from roguard.gemma4_review import _score
    module_spec=importlib.util.spec_from_file_location('registered_train_oracle_adapter',paths['serving_adapter'])
    adapter=importlib.util.module_from_spec(module_spec)
    exec(compile(bound_bytes(plan,'serving_adapter'),str(paths['serving_adapter']),'exec'),adapter.__dict__)
    check_files(plan);check_origins(plan)
    require(sha(plan_path)==plan_sha and sha(__file__)==plan['source_sha256'],
            'Plan/source changed before numerical replay')
    output.mkdir(parents=True,exist_ok=False)
    (output/'source.py').write_bytes(Path(__file__).read_bytes());(output/'registered-plan.json').write_bytes(Path(plan_path).read_bytes())
    report=dict(format='roguard-train-oracle-feasibility-1',complete=False,plan_sha256=plan_sha,
        source_sha256=sha(__file__),languages={},oracle_not_learned=True,encoder_loaded=False,
        optimizer_run=False,dev_feature_rows_scored=0,new_probe_rows_scored=0,qualified=False,
        parent_backend='Torch CPU exact-erf head replay; not newly verified native-serving parity',
        native_serving_parity_claimed=False,parent_threshold_diagnostic_band=1e-3,
        oracle_scope='Swap parent AFFIRMED/UNCERTAIN probabilities only on wrong A/U TRAIN predictions; this is one specified target, not exhaustive reachability over all conservative targets.',
        diagnostic_band_rule='Report all parent distances and count abs(score-threshold)<=1e-3; never filter rows or adapt thresholds.')
    def persist():
        temporary=output/'report.tmp';temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temporary.replace(output/'report.json')
    persist()
    for language in ('ro','uk'):
        info=languages[language];selected=info['selected'];spec=info['spec'];n=len(selected)
        record=dict(info['summary'],complete=False,rows_processed=0,groups={});report['languages'][language]=record
        with (output/(language+'-rows.jsonl')).open('x') as f:
            for train_index,(source_index,row) in enumerate(selected):
                f.write(json.dumps(dict(train_index=train_index,source_index=source_index,row=row),ensure_ascii=False,sort_keys=True)+'\n')
        decoded=load_safetensors(bound_bytes(plan,language+'_weights75'))
        tensors={key:decoded['heads.'+language+'.'+key] for key in ('first.weight','first.bias','last.weight','last.bias')}
        del decoded
        require(all(v.dtype==np.float32 and np.isfinite(v).all() for v in tensors.values()),'Invalid parent75 head tensors')
        features=np.load(paths[language+'_features'],mmap_mode='r',allow_pickle=False)
        parent=np.lib.format.open_memmap(output/(language+'-parent-logits.npy'),mode='w+',dtype=np.float32,shape=(n,19))
        corrected=np.lib.format.open_memmap(output/(language+'-oracle-logits.npy'),mode='w+',dtype=np.float32,shape=(n,19))
        with (output/(language+'-results.jsonl')).open('x') as results:
            for start in range(0,n,256):
                batch=selected[start:start+256]
                indices=[index for index,row in batch]
                # This is the only access to feature values; every index came from TRAIN filtering.
                logits=cpu_head(features[indices],tensors)
                for local,(source_index,row) in enumerate(batch):
                    index=start+local;z=logits[local];gold=row['fields'][2];before=int(z[6:10].argmax())
                    pair_in_scope=before in (0,2) and gold in (0,2)
                    eligible=pair_in_scope and before!=gold
                    parent_score=_score(z,spec)
                    if eligible:
                        target=adapter.field_probabilities(z)['assertion'];target[0],target[2]=target[2],target[0]
                        adapted=adapter.adapt(z,spec,target_assertion=target,score_tolerance=1e-12)
                        after_z=adapted['logits'];status=adapted['status'];reason=adapted['fallback_reason']
                    else:
                        after_z=z.copy();adapted=None;status='PARENT_ALREADY_CORRECT' if before==gold else 'OUT_OF_SCOPE_ASSERTION_PAIR';reason=None
                    after=int(after_z[6:10].argmax());after_score=_score(after_z,spec)
                    require(z[adapter.OTHER].tobytes()==after_z[adapter.OTHER].tobytes()
                            and abs(after_score-parent_score)<=1e-12
                            and (after_score>=spec['threshold'])==(parent_score>=spec['threshold']),
                            'TRAIN oracle broke serving invariants')
                    if not eligible or status=='EXACT_PARENT_FALLBACK':require(after_z.tobytes()==z.tobytes(),'Untouched/fallback TRAIN logits changed')
                    parent[index]=z;corrected[index]=after_z
                    original_fields=[int(x.argmax()) for x in np.split(z[:17],[3,6,10,13,15])]
                    oracle_fields=[int(x.argmax()) for x in np.split(after_z[:17],[3,6,10,13,15])]
                    result=dict(train_index=index,source_index=source_index,case_id=row['case_id'],language=language,
                        partition='train',group=row['group'],gold_fields=row['fields'],gold_binary=row['gold'],
                        parent_fields=original_fields,oracle_fields=oracle_fields,parent_assertion=before,oracle_assertion=after,
                        eligible_au_disagreement=eligible,au_pair_in_scope=pair_in_scope,status=status,fallback_reason=reason,
                        parent_score=parent_score,oracle_score=after_score,score_difference=after_score-parent_score,
                        parent_signed_distance_to_threshold=parent_score-spec['threshold'],
                        parent_absolute_distance_to_threshold=abs(parent_score-spec['threshold']),
                        parent_within_1e_3_threshold_band=abs(parent_score-spec['threshold'])<=1e-3,
                        parent_binary=int(parent_score>=spec['threshold']),oracle_binary=int(after_score>=spec['threshold']),
                        corrected_assertion=before!=gold and after==gold,introduced_assertion_error=before==gold and after!=gold,
                        other15_logits_bitwise_unchanged=True,full_parent_identity=z.tobytes()==after_z.tobytes(),
                        adapter_observation=None if adapted is None else {k:v for k,v in adapted.items() if k!='logits'},
                        oracle_not_learned=True)
                    results.write(json.dumps(result,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n')
                    for group in ('all','gold_assertion:'+str(gold),'source_group:'+row['group']):
                        totals=record['groups'].setdefault(group,dict(rows=0,eligible=0,corrected=0,introduced=0,
                            parent_assertion_errors=0,oracle_assertion_errors=0,parent_binary_errors=0,oracle_binary_errors=0,
                            maximum_score_difference=0.,minimum_parent_threshold_distance=None,
                            parent_within_1e_3_threshold_band=0,statuses={},fallback_reasons={}))
                        for key,value in [('rows',1),('eligible',int(eligible)),('corrected',int(result['corrected_assertion'])),
                            ('introduced',int(result['introduced_assertion_error'])),('parent_assertion_errors',int(before!=gold)),
                            ('oracle_assertion_errors',int(after!=gold)),('parent_binary_errors',int(result['parent_binary']!=row['gold'])),
                            ('oracle_binary_errors',int(result['oracle_binary']!=row['gold']))]:totals[key]+=value
                        totals['maximum_score_difference']=max(totals['maximum_score_difference'],abs(after_score-parent_score))
                        distance=abs(parent_score-spec['threshold'])
                        previous=totals['minimum_parent_threshold_distance']
                        totals['minimum_parent_threshold_distance']=distance if previous is None else min(previous,distance)
                        totals['parent_within_1e_3_threshold_band']+=int(distance<=1e-3)
                        totals['statuses'][status]=totals['statuses'].get(status,0)+1
                        if reason:totals['fallback_reasons'][reason]=totals['fallback_reasons'].get(reason,0)+1
                    record['rows_processed']=index+1
                parent.flush();corrected.flush();results.flush();persist()
        parent.flush();corrected.flush();del parent,corrected,features,tensors
        record['artifacts']={p.name:sha(p) for p in sorted(output.glob(language+'-*'))}
        record['complete']=True;persist()
    check_files(plan);check_origins(plan)
    require(sha(plan_path)==plan_sha and sha(__file__)==plan['source_sha256'],'Plan/source changed during feasibility run')
    require(sum(r['rows_processed'] for r in report['languages'].values())==sum(v[1] for v in COUNTS.values()),
            'Incomplete TRAIN population')
    report.update(complete=True,decision='TRAIN_ORACLE_REACHABILITY_ONLY_NOT_LEARNED_PERFORMANCE',
        parent_error_floor_unchanged='Preserving initial75 binary decisions cannot eliminate its14RO/2UK originalDEV and1ROcounterfoil error; no such rows were scored here.')
    persist();return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--output',type=Path);parser.add_argument('--preflight-report',type=Path)
    args=parser.parse_args()
    require((args.output is None)!=(args.preflight_report is None),'Select preflight or actual output, not both')
    if args.preflight_report is not None:
        plan,paths,languages=preflight(args.plan,args.plan_sha256)
        write(args.preflight_report,dict(verification='PASS_PREFLIGHT_ONLY',plan_sha256=args.plan_sha256,
            source_sha256=sha(__file__),languages={l:r['summary'] for l,r in languages.items()},
            numerical_head_replay=False,encoder_loaded=False,optimizer_run=False,qualified=False))
    else:run(args.plan,args.plan_sha256,args.output)


if __name__=='__main__':main()
