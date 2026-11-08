"""Independent saved128 audit. Registered execution only; no head/encoder replay."""
import argparse
from collections import defaultdict
import hashlib
import io
import json
import math
import shutil
from pathlib import Path
import sys
import numpy as np

FORMAT='roguard-train-positive-scale-saved-audit-1'
FIELDS=('subject','time','assertion','source','relation','concern')
WIDTHS=(3,3,4,3,2,2,2)
STARTS=(0,3,6,10,13,15,17)
ALPHAS=(1.,1.125,1.25,1.5,2.,3.,4.)
COUNTS={'ro':13174,'uk':15755}
ACTIVE=('subject','assertion','source','relation','concern')
PRODUCER_SHA='c9bccff28a36561211fddbd1722c66941ee53ad0a26f8f73a3c95f7f881d8d34'
WEIGHTS={'ro':'1c18db5e5fc26baf4c0739d053c0546338201d76fcb9061e47477c14650535e3','uk':'466a1900655c85a6db809adb384d90fc634357a4f7b72f1415274c4787b3ecb7'}
SPECS={'ro':'5fdeae97843d873755ed41a02453b5c29823fd8028508c3e99d1586175c8f4a9','uk':'bbf3fea8ee298c40d86701851fe045352ea1a924ba5287dd52b70e59ce25de27'}
THRESHOLDS={'ro':.7517333793124337,'uk':.9274248149759536}

def require(ok,message):
    if not ok:raise ValueError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()
def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def rows(path):return [json.loads(x) for x in Path(path).read_bytes().splitlines()]
def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    require(canonical(read(path))==canonical(value),'Audit JSON readback differs')
def verify(ledger):
    require(isinstance(ledger,dict) and ledger,'Missing bound ledger')
    for binding in ledger.values():require(sha(binding['path'])==binding['sha256'],'Bound bytes changed: '+str(binding['path']))
def array(binding):
    raw=Path(binding['path']).read_bytes()
    require(hashlib.sha256(raw).hexdigest()==binding['sha256'],'Array changed before decoding')
    return np.load(io.BytesIO(raw),allow_pickle=False)

def head_indices(values):
    return np.column_stack([values[:,start:start+width].argmax(1) for start,width in zip(STARTS,WIDTHS)])

def oracle_transform(parent,alpha):
    """Vectorized named-field transform; independent of producer row routine."""
    require(type(alpha) is float and alpha in ALPHAS,'Unregistered alpha')
    require(parent.ndim==2 and parent.shape[1]==19 and parent.dtype==np.float32 and np.isfinite(parent).all(),'Invalid parent matrix')
    output=parent.copy();status=np.full(len(parent),'identity',dtype=object)
    if alpha==1.:return output,status.tolist()
    status[:]='scaled'
    with np.errstate(over='ignore',under='ignore',invalid='ignore'):
        for name in ACTIVE:
            index=FIELDS.index(name);start,width=STARTS[index],WIDTHS[index]
            block=parent[:,start:start+width].astype(np.float64)
            output[:,start:start+width]=np.asarray((block-np.max(block,axis=1)[:,None])*alpha,dtype=np.float32)
    nonfinite=~np.isfinite(output).all(1)
    status[nonfinite]='fallback_nonfinite';output[nonfinite]=parent[nonfinite]
    changed=(head_indices(parent)!=head_indices(output)).any(1)
    for start,width in zip(STARTS,WIDTHS):
        before=parent[:,start:start+width];after=output[:,start:start+width]
        changed|=((before==before.max(1)[:,None])!=(after==after.max(1)[:,None])).any(1)
    changed &= ~nonfinite
    status[changed]='fallback_argmax_or_tie';output[changed]=parent[changed]
    require(output[:,3:6].tobytes()==parent[:,3:6].tobytes()
            and output[:,17:19].tobytes()==parent[:,17:19].tobytes(),'Protected slices changed')
    return output,status.tolist()

def map_world(z,spec):
    """Named pair-wise MAP with first-index tie rule, no producer import."""
    parts={name:z[start:start+width].astype(np.float64) for name,start,width in zip(FIELDS,STARTS,WIDTHS)}
    def best(left,right,weights):
        candidates=[(float(parts[left][i])+float(parts[right][j])+math.log(float(weights[i][j])),i,j)
            for i in range(len(parts[left])) for j in range(len(parts[right]))]
        winner=max(candidates,key=lambda v:v[0]);return winner[1:]
    subject,source=best('subject','source',spec['couplings']['scope'])
    assertion,relation=best('assertion','relation',spec['couplings']['assertion_relation'])
    return [subject,int(parts['time'].argmax()),assertion,source,relation,int(parts['concern'].argmax())]

def outcome(parent,candidate,row,spec,scorer,alpha,status,index,parent_score=None,parent_map=None):
    scores=[float(scorer(parent,spec)) if parent_score is None else parent_score,float(scorer(candidate,spec))]
    require(all(math.isfinite(x) and 0<=x<=1 for x in scores),'Invalid portable policy probability')
    before,after=scores;threshold=spec['threshold'];gold=row['gold'];truth=row['fields']
    pred=head_indices(candidate[None])[0].tolist()
    require(pred==head_indices(parent[None])[0].tolist(),'Unary argmax changed')
    bd,ad=int(before>=threshold),int(after>=threshold)
    wrong_before,wrong_after=bd!=gold,ad!=gold
    errors=[int(pred[i]!=truth[i]) for i in range(6)]
    nll=[];brier=[];ties=[]
    for column,(start,width) in enumerate(zip(STARTS,WIDTHS)):
        raw=candidate[start:start+width]
        ties.append([i for i,v in enumerate(raw) if v==max(raw)])
        if column==6:continue
        centered=raw.astype(np.float64)-float(max(raw));exponent=np.exp(centered)
        probabilities=exponent/float(exponent.sum())
        target=np.zeros(width,np.float64);target[truth[column]]=1.
        nll.append(float(math.log(float(exponent.sum()))-centered[truth[column]]))
        brier.append(float(np.dot(probabilities-target,probabilities-target)))
    ptrue=after if gold else 1-after
    return dict(case_id=row['case_id'],language=row['language'],group=row['group'],gold=gold,fields=truth,
        alpha=alpha,status=status,parent_score=before,score=after,parent_decision=bd,decision=ad,
        parent_margin=before-threshold,margin=after-threshold,parent_error=wrong_before,error=wrong_after,
        introduced=wrong_after and not wrong_before,fixed=wrong_before and not wrong_after,
        field_errors=errors,joint_error=any(errors),argmax7=pred,tie_sets=ties,
        coupled_map_changed=(map_world(parent,spec) if parent_map is None else parent_map)!=map_world(candidate,spec),
        field_nll=nll,field_brier=brier,binary_brier=float((after-gold)**2),
        binary_nll=float(-math.log(max(ptrue,np.finfo(np.float64).tiny))),binary_true_probability_zero=ptrue==0.,
        parent_within_diagnostic_band=abs(before-threshold)<=1e-3,train_index=index)

def compare_values(actual,expected,path=''):
    require(type(actual)==type(expected),'Retained value type differs: '+path)
    if isinstance(expected,dict):
        require(set(actual)==set(expected),'Retained key inventory differs: '+path)
        for key in expected:compare_values(actual[key],expected[key],path+'/'+key)
    elif isinstance(expected,list):
        require(len(actual)==len(expected),'Retained list length differs: '+path)
        for i,(a,b) in enumerate(zip(actual,expected)):compare_values(a,b,path+'/'+str(i))
    elif isinstance(expected,float):
        # Policy scores/margins and alpha are exact; only arithmetic diagnostics
        # may differ by a tiny absolute float64 reduction error.
        numerical=any(key in path for key in ('field_nll','field_brier','binary_nll','binary_brier'))
        require(math.isfinite(actual) and abs(actual-expected)<=(1e-10 if numerical else 0.),'Retained float differs: '+path)
    else:require(actual==expected,'Retained value differs: '+path)

def summarize(records):
    count=len(records);require(count>0,'Empty summary')
    def total(key):return sum(int(r[key]) for r in records)
    result=dict(rows=count,binary_errors=total('error'),parent_binary_errors=total('parent_error'),
        introduced=total('introduced'),fixed=total('fixed'),
        tp=sum(r['decision']==1 and r['gold']==1 for r in records),tn=sum(r['decision']==0 and r['gold']==0 for r in records),
        fp=sum(r['decision']==1 and r['gold']==0 for r in records),fn=sum(r['decision']==0 and r['gold']==1 for r in records),
        joint_errors=total('joint_error'),field_errors=[sum(r['field_errors'][i] for r in records) for i in range(6)],
        fallbacks=sum(r['status'] in ('fallback_nonfinite','fallback_argmax_or_tie') for r in records),
        scaled=sum(r['status']=='scaled' for r in records),coupled_map_changes=total('coupled_map_changed'),
        field_nll=[float(np.mean([r['field_nll'][i] for r in records])) for i in range(6)],
        field_brier=[float(np.mean([r['field_brier'][i] for r in records])) for i in range(6)],
        binary_nll=float(np.mean([r['binary_nll'] for r in records])),binary_brier=float(np.mean([r['binary_brier'] for r in records])),
        binary_true_probability_zeros=total('binary_true_probability_zero'),parent_within_diagnostic_band=total('parent_within_diagnostic_band'))
    return result

def groups(records):
    index=defaultdict(list)
    for row in records:
        for key in ('all','source_group:'+row['group'],'gold:'+str(row['gold']),
                    'assertion:'+str(row['fields'][2]),'status:'+row['status']):index[key].append(row)
    return {key:summarize(value) for key,value in sorted(index.items())}

def select(candidates):
    require(canonical([c['alpha'] for c in candidates])==canonical(list(ALPHAS)),'Seven ordered candidates required')
    initial=candidates[0];eligible=[c for c in candidates if c['groups']['all']['introduced']==0]
    require(initial in eligible,'Identity must have zero introductions')
    better=[c for c in eligible if c['groups']['all']['binary_errors']<initial['groups']['all']['binary_errors']]
    chosen=initial if not better else sorted(better,key=lambda c:(c['groups']['all']['binary_errors'],c['alpha']-1.,c['alpha']))[0]
    return dict(alpha=chosen['alpha'],binary_errors=chosen['groups']['all']['binary_errors'],improved=chosen is not initial,
        rule='Minimize allTRAIN binary errors among zero-introduced candidates; then closest to1 and numeric alpha. Identity if no strict gain.',candidate_artifacts=chosen['artifacts'])

def check_origins(producer):
    root=Path(producer['project_root']).resolve();bound={Path(v['path']).resolve() for v in producer['files'].values()}
    require(all(p.resolve() in bound for p in (root/'src/roguard').rglob('*.py')),'Incomplete serving source inventory')
    for name,module in list(sys.modules.items()):
        if name=='roguard' or name.startswith('roguard.'):
            p=root/'src'/Path(*name.split('.'));p=p/'__init__.py' if p.is_dir() else p.with_suffix('.py')
            require(Path(module.__file__).resolve()==p.resolve() and p.resolve() in bound,'Unbound scorer origin')

def preflight(plan_path,expected):
    raw=Path(plan_path).read_bytes();require(hashlib.sha256(raw).hexdigest()==expected,'Audit plan changed')
    plan=json.loads(raw)
    require(plan['format']==FORMAT and plan['registered'] is True and plan['held_out_test_opened'] is False
            and plan['authorization']=='TRAIN_ONLY_SAVED_OUTPUT_AUDIT' and plan['qualified'] is False
            and type(plan['producer_exit_code']) is int and plan['producer_exit_code']==0,
            'Not registered for saved-output audit')
    require(plan['source_sha256']==sha(__file__),'Executing auditor changed')
    verify(plan['files']);verify(plan['outputs'])
    p=lambda key:Path(plan['files'][key]['path'])
    producer=read(p('producer_plan'));report=read(p('producer_report'))
    require(producer['format']=='roguard-train-positive-scale-1' and producer['registered'] is True
            and producer['authorization']=='TRAIN_ONLY_BOUNDED_SCALING' and producer['held_out_test_opened'] is False,
            'Producer registration scope differs')
    require(producer['source_sha256']==plan['files']['producer_source']['sha256']==PRODUCER_SHA,
            'Producer must be corrected reviewed128 source')
    require(canonical(producer['alphas'])==canonical(list(ALPHAS))
            and canonical(producer['active_field_indices'])==canonical([0,2,3,4,5]),'Frozen transform family differs')
    verify(producer['files']);q=lambda key:Path(producer['files'][key]['path'])
    require(np.__version__==producer['numpy_version']==plan['numpy_version'],'Numerical NumPy pin differs')
    require(report['format']=='roguard-train-positive-scale-1' and report['complete'] is True
            and report['source_sha256']==PRODUCER_SHA and report['plan_sha256']==plan['files']['producer_plan']['sha256']
            and report['field_order']==list(FIELDS),'Incomplete or unbound producer report')
    for key in ('encoder_loaded','head_replayed','optimizer_run','qualified'):require(report[key] is False,'Producer scope differs')
    require(type(report['dev_rows_scored']) is int and type(report['new_probe_rows_scored']) is int
            and report['dev_rows_scored']==report['new_probe_rows_scored']==0,'Producer scored forbidden population')
    require(set(report['languages'])==set(COUNTS),'Producer language inventory incomplete')
    names={'registered-plan.json','source.py','selection-before-dev.json'}|{lang+'/row-order.json' for lang in COUNTS}
    names|={f'{lang}/alpha-{alpha}/{file}' for lang in COUNTS for alpha in ALPHAS for file in ('logits.npy','results.jsonl')}
    require(set(report['artifacts'])==set(plan['outputs'])==names,'Complete exact128 output inventory required')
    require(all(report['artifacts'][key]==value['sha256'] for key,value in plan['outputs'].items()),'Producer output digest differs')
    output_root=p('producer_report').parent.resolve()
    require(p('producer_report').name=='report.json','Producer report filename differs')
    for name,binding in plan['outputs'].items():
        require(Path(binding['path']).resolve()==output_root/name,'Producer output path escapes exact role')
    def actual_output_inventory():
        return {str(item.relative_to(output_root)) for item in output_root.rglob('*') if item.is_file()}
    require(actual_output_inventory()==names|{'report.json'},'Producer output inventory incomplete or contains failure/extra files')
    o=lambda name:Path(plan['outputs'][name]['path'])
    require(canonical(read(o('registered-plan.json')))==canonical(producer)
            and sha(o('source.py'))==PRODUCER_SHA,'Retained producer plan/source differs')
    upstream=read(q('report117'));audit=read(q('audit118'));oldplan=read(q('plan117'))
    require(upstream['complete'] is True and upstream['qualified'] is False
            and audit['format']=='roguard-train-oracle-saved-audit-1' and audit['verification']=='PASS'
            and audit['qualified'] is False and audit['parent_head_replayed'] is False,'117/118 scope incomplete')
    for key,source in [('producer_report_sha256','report117'),('producer_source_sha256','source117'),
                       ('producer_plan_sha256','plan117'),('source_sha256','source118'),('audit_plan_sha256','plan118')]:
        require(audit[key]==producer['files'][source]['sha256'],'118 provenance differs')
    require(upstream['source_sha256']==producer['files']['source117']['sha256']
            and upstream['plan_sha256']==producer['files']['plan117']['sha256'],'117 producer provenance differs')
    data={}
    for lang,n in COUNTS.items():
        original=rows(q(lang+'_original_rows'));saved=rows(q(lang+'_rows117'))
        selected=[(i,row) for i,row in enumerate(original) if row['partition']=='train']
        require(len(selected)==len(saved)==n,'Complete original TRAIN population required')
        require(all(row['partition'] in ('train','dev') for row in original),'Unexpected original partition')
        seen=set()
        for j,((index,row),retained) in enumerate(zip(selected,saved)):
            require(type(retained['train_index']) is int and retained['train_index']==j
                    and type(retained['source_index']) is int and retained['source_index']==index
                    and canonical(retained['row'])==canonical(row),'117 TRAIN row/order differs')
            require(row['language']==lang and isinstance(row['case_id'],str) and row['case_id'] not in seen,'TRAIN identity differs');seen.add(row['case_id'])
            require(isinstance(row['fields'],list) and len(row['fields'])==6 and all(type(v) is int and 0<=v<w for v,w in zip(row['fields'],WIDTHS))
                    and type(row['gold']) is int and row['gold'] in (0,1),'TRAIN label types differ')
        expected_rows=[row for _,row in selected]
        require(canonical(read(o(lang+'/row-order.json')))==canonical(expected_rows),'128 retained row order differs')
        order=digest([[i,row] for i,row in selected])
        require(order==upstream['languages'][lang]['train_order_sha256']==audit['languages'][lang]['train_order_sha256']
                and type(audit['languages'][lang]['rows_checked']) is int and audit['languages'][lang]['rows_checked']==n,'118 allTRAIN recount differs')
        for suffix,key in [('rows.jsonl',lang+'_rows117'),('parent-logits.npy',lang+'_logits117')]:
            filename=lang+'-'+suffix;h=producer['files'][key]['sha256']
            require(upstream['languages'][lang]['artifacts'][filename]==h
                    and audit['output_files']['output:'+filename]['sha256']==h,'117/118 parent artifact differs')
        for name,newkey in [('rows',lang+'_original_rows'),('weights75',lang+'_weights75'),('spec75',lang+'_spec75')]:
            require(oldplan['files'][lang+'_'+name]['sha256']==producer['files'][newkey]['sha256'],'Original117 input differs')
        require(producer['files'][lang+'_weights75']['sha256']==audit['languages'][lang]['weights75_sha256']==WEIGHTS[lang]
                and producer['files'][lang+'_spec75']['sha256']==audit['languages'][lang]['spec75_sha256']==SPECS[lang], 'Fixed75 weight/spec differs')
        spec=read(q(lang+'_spec75'));require(spec['mode']=='coupled_policy' and spec['language']==lang
            and spec['threshold']==THRESHOLDS[lang] and spec['adapter_sha256']==WEIGHTS[lang],'Fixed75 serving contract differs')
        parent=array(producer['files'][lang+'_logits117'])
        require(parent.shape==(n,19) and parent.dtype==np.float32 and np.isfinite(parent).all(),'Parent logit matrix differs')
        data[lang]=(expected_rows,parent,spec)
    verify(producer['files']);verify(plan['files']);verify(plan['outputs'])
    require(sha(plan_path)==expected and sha(__file__)==plan['source_sha256'],'Audit source/plan changed during preflight')
    return plan,producer,report,data,raw

def run(plan_path,expected,output):
    output=Path(output);require(not output.exists(),'Refusing overwrite')
    require(shutil.disk_usage(output.parent).free>=8*1024**3,'Eight GiB disk reserve')
    plan,producer,report,data,plan_raw=preflight(plan_path,expected)
    from roguard.gemma4_review import _score,FIELDS as serving_fields
    require(tuple(serving_fields)==FIELDS,'Serving ontology order changed')
    check_origins(producer)
    def barrier():
        verify(plan['files']);verify(plan['outputs']);verify(producer['files'])
        root=Path(plan['files']['producer_report']['path']).parent.resolve()
        require({str(item.relative_to(root)) for item in root.rglob('*') if item.is_file()}==set(plan['outputs'])|{'report.json'},
                'Producer output inventory changed or includes failure')
        require(sha(plan_path)==expected and sha(__file__)==plan['source_sha256'],'Auditor source/plan changed')
    barrier() # Global both-language guard after imports, before any row replay.
    audit={};selections={}
    for language,(input_rows,parent,spec) in data.items():
        parent_scores=[float(_score(row,spec)) for row in parent]
        parent_maps=[map_world(row,spec) for row in parent]
        candidates=report['languages'][language]['candidates']
        require(canonical([c['alpha'] for c in candidates])==canonical(list(ALPHAS)),'All seven scales must be retained in order')
        verified_candidates=[];row_replays=0
        for alpha,declared in zip(ALPHAS,candidates):
            folder=f'{language}/alpha-{alpha}';array_name=folder+'/logits.npy';rows_name=folder+'/results.jsonl'
            saved=array(plan['outputs'][array_name]);wanted,statuses=oracle_transform(parent,alpha)
            require(saved.dtype==np.float32 and saved.shape==wanted.shape and saved.tobytes()==wanted.tobytes(),
                    'Saved candidate is not exact independent transform: '+folder)
            require(declared['artifacts']=={name:plan['outputs'][name]['sha256'] for name in (array_name,rows_name)},'Candidate artifact mapping differs')
            retained=rows(plan['outputs'][rows_name]['path']);require(len(retained)==len(input_rows),'Saved allTRAIN row denominator differs')
            recomputed=[]
            for i,row in enumerate(input_rows):
                result=outcome(parent[i],saved[i],row,spec,_score,alpha,statuses[i],i,parent_scores[i],parent_maps[i])
                compare_values(retained[i],result);recomputed.append(result)
            summary=groups(recomputed);compare_values(declared['groups'],summary)
            verified_candidates.append(dict(alpha=alpha,groups=summary,artifacts=dict(declared['artifacts'])))
            row_replays+=len(recomputed)
        chosen=select(verified_candidates);compare_values(report['languages'][language]['selected'],chosen)
        selections[language]=chosen
        audit[language]=dict(train_rows=len(input_rows),scales=len(ALPHAS),candidate_rows_recounted=row_replays,
            candidates=verified_candidates,selected=chosen,parent_head_replayed=False)
    compare_values(read(plan['outputs']['selection-before-dev.json']['path']),selections)
    barrier();check_origins(producer)
    require(shutil.disk_usage(output.parent).free>=8*1024**3,'Eight GiB disk reserve')
    source_raw=Path(__file__).read_bytes();require(hashlib.sha256(source_raw).hexdigest()==plan['source_sha256'],'Auditor source changed beforecopy')
    output.mkdir(parents=True)
    for name,raw,h in [('source.py',source_raw,plan['source_sha256']),('registered-plan.json',plan_raw,expected)]:
        with (output/name).open('xb') as f:f.write(raw)
        require(sha(output/name)==h,'Auditor retained snapshot changed')
    final=dict(format=FORMAT,verification='PASS',audit_plan_sha256=expected,source_sha256=plan['source_sha256'],
        producer_report_sha256=plan['files']['producer_report']['sha256'],producer_plan_sha256=plan['files']['producer_plan']['sha256'],
        producer_source_sha256=PRODUCER_SHA,input_files=producer['files'],output_files=plan['outputs'],languages=audit,
        total_train_rows=sum(COUNTS.values()),total_candidate_rows_recounted=sum(COUNTS.values())*len(ALPHAS),
        selections=selections,encoder_loaded=False,parent_head_replayed=False,optimizer_run=False,
        dev_rows_scored=0,new_probe_rows_scored=0,calibration_claimed=False,held_out_test_opened=False,qualified=False,
        limitations=['Independent saved-transform/outcome/group/selection consistency;117 parent-head logits are not independently regenerated here.',
            'TRAIN-selected scales are not independent performance estimates, semantic correction, newnativeparity or inherited conformal calibration.',
            'Binary scores/decisions, transform bytes, all7argmax/ties, statuses and integer counts exact; float64 loss/Brier diagnostics use1e-10 absolute tolerance.'])
    try:
        save(output/'report.json',final);final_sha=sha(output/'report.json')
        barrier();check_origins(producer)
        require(sha(output/'source.py')==plan['source_sha256'] and sha(output/'registered-plan.json')==expected
                and sha(output/'report.json')==final_sha,'Auditor retained evidence changed')
    except BaseException as error:
        failure=dict(format=FORMAT,verification='FAIL',qualified=False,reason=str(error),audit_plan_sha256=expected)
        (output/'report.json').write_text(canonical(failure)+'\n')
        raise
    return final

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=run(args.plan,args.plan_sha256,args.output)
    print(json.dumps(dict(verification=result['verification'],rows=result['total_candidate_rows_recounted'],qualified=False)))
