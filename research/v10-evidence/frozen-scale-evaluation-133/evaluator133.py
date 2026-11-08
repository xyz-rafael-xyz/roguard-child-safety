"""Fixed TRAIN-chosen positive-scale evaluation; no fitting or alpha search.

Only original exposed DEV and the complete already-exposed larger bank are
scored. Unary labels remain unchanged; coupled MAP and policy decisions need
not. This establishes no conformal coverage, calibration or native parity.
"""
import argparse
from collections import Counter
import hashlib,io,json,math,shutil,sys,types
from importlib.metadata import version
from pathlib import Path

FORMAT='roguard-frozen-train-scale-evaluation-plan-1'
ALPHAS={'ro':2.0,'uk':1.5}
DEV={'ro':4202,'uk':4635}
TRAIN={'ro':13174,'uk':15755}
BANK={'fit':288,'probe_operator':384,'probe_meaning':384,'probe_combination':96,'probe_both':384,'counterfoil':4}
FIELDS=('subject','time','assertion','source','relation','concern')
SIZES=(3,3,4,3,2,2)
SCALE_SHA='c9bccff28a36561211fddbd1722c66941ee53ad0a26f8f73a3c95f7f881d8d34'
AUDIT_SHA='f8290afbd22edbbbe721c2159359c21197e90cb997be08519f3d48b14ffdfd4c'
BASELINE_SOURCE_SHA='55601bd5402bb3e24cfd0fd5d47ed655db9f90e083fbbbd25383c6ad5a64a6d0'
BASELINE_ROOT_SHA='4ae9db78633a02bf362876da65d6e95b948f43d887999b71fbc3d5db680bf7af'
DESIGN_SHA='d2d440d4233c90ae23e5a0053b12672b77b1b77d00acaba8537d5c0ddb19a8a5'
WEIGHTS75={'ro':'1c18db5e5fc26baf4c0739d053c0546338201d76fcb9061e47477c14650535e3','uk':'466a1900655c85a6db809adb384d90fc634357a4f7b72f1415274c4787b3ecb7'}
SPECS75={'ro':'5fdeae97843d873755ed41a02453b5c29823fd8028508c3e99d1586175c8f4a9','uk':'bbf3fea8ee298c40d86701851fe045352ea1a924ba5287dd52b70e59ce25de27'}
V9={'ro':'114d3e89191d06152cceccd080114998901a4f0bf6a9540de8e11dd8b6cf6b09','uk':'0fee43a071d2c07c26e8b3d69cacb4933fd284bc64dbee0d18fe5481e1e4815f'}
SPECSV9={'ro':'6b2ea2979ffa43c735d8f8a6e3f7e1fbf6cf5c085889e8d484a13bf43026ccc5','uk':'fc86fed9981b786149c4f6855c3d98f07067654cf2af05d40aef589345be34ed'}
REFERENCES=('unchanged-initial75','original-v9','frozen-scale75')
DISK_RESERVE=8*1024**3
MAX_OUTPUT=256*1024**2


def require(ok,message):
    if not ok:raise ValueError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def canonical(value):return json.dumps(value,ensure_ascii=True,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def rows(path):return [json.loads(v) for v in Path(path).read_bytes().splitlines()]
def identical(a,b,message):require(canonical(a)==canonical(b),message)
def bound(plan,key):
    b=plan['files'][key];raw=Path(b['path']).read_bytes();require(hashlib.sha256(raw).hexdigest()==b['sha256'],'Bound bytes changed: '+key);return raw
def checks(plan,plan_path,expected):
    require(sha(plan_path)==expected and sha(__file__)==plan['source_sha256'],'Evaluation source/plan changed')
    for key,b in plan['files'].items():require(Path(b['path']).is_absolute() and sha(b['path'])==b['sha256'],'Bound input changed: '+key)
def origins(plan):
    root=Path(plan['project_root']).resolve();bound_paths={Path(v['path']).resolve() for v in plan['files'].values()}
    for name,module in list(sys.modules.items()):
        if name.split('.')[0] not in ('roguard','training','eval','scripts') or not getattr(module,'__file__',None):continue
        path=Path(module.__file__).resolve();base=root/'src' if name.split('.')[0]=='roguard' else root
        expected=base.joinpath(*name.split('.'))
        require(path in (expected.with_suffix('.py').resolve(),(expected/'__init__.py').resolve()) and path in bound_paths,'Wrong imported project origin: '+name)
def module(plan,key,name):
    raw=bound(plan,key);m=types.ModuleType(name);m.__file__=plan['files'][key]['path'];exec(compile(raw,m.__file__,'exec'),m.__dict__);return m

def resource_check(output):
    require(shutil.disk_usage(output.parent).free>=DISK_RESERVE,'Eight GiB output reserve')
    if output.exists():require(sum(p.stat().st_size for p in output.rglob('*') if p.is_file())<=MAX_OUTPUT,'Evaluation output cap')

def validate_rows(values,language,new):
    expected=BANK if new else {'train':TRAIN[language],'dev':DEV[language]}
    require(dict(Counter(r['partition'] for r in values))==expected,'Complete population differs')
    ids=set()
    for row in values:
        require(row['language']==language and isinstance(row['case_id'],str) and row['case_id'] and row['case_id'] not in ids,'Language/case identity differs');ids.add(row['case_id'])
        require(isinstance(row['fields'],list) and len(row['fields'])==6 and all(type(v) is int and 0<=v<n for v,n in zip(row['fields'],SIZES))
            and type(row['gold']) is int and row['gold'] in (0,1),'Exact typed labels required')
        if new:
            identical(row['fields'],row['proposed_fields'],'New field proposal differs');identical(row['gold'],row['proposed_gold'],'New binary proposal differs')
            require(row['training_admitted'] is False and row['independently_annotated'] is False,'New label independence/admission scope differs')

def pointer(value,path):
    for part in path.strip('/').split('/'):value=value[part]
    return value

def lineage(plan):
    files=plan['files'];path=lambda key:Path(files[key]['path'])
    original=read(path('original_feature_report'));audit=read(path('original_feature_audit'));warm=read(path('initial_reference_report'))
    require(original['complete'] is True and original['qualified'] is False and audit['verification']=='PASS' and audit['qualified'] is False
        and warm['complete'] is True and warm['qualified'] is False,'Original88/89/warm90 lineage incomplete')
    require(audit['report_sha256']==files['original_feature_report']['sha256'],'Audit89 producer differs')
    for lang in ('ro','uk'):
        for suffix,filename in [('original_rows','rows.jsonl'),('original_logits','logits.npy')]:
            key=lang+'_'+suffix;name=lang+'-'+filename
            require(original['languages'][lang]['artifacts'][name]==audit['artifacts'][name]==files[key]['sha256'],'Original88/89 row/logit artifact differs')
        for suffix,filename in [('initial_weights','selected.safetensors'),('initial_dev_logits','selected-dev-logits.npy'),('initial_dev_rows','dev-rows.jsonl')]:
            require(warm['languages'][lang]['artifacts'][lang+'-'+filename]==files[lang+'_'+suffix]['sha256'],'Warm90 initial75 artifact differs')
        for suffix,expected in [('initial_weights',WEIGHTS75[lang]),('initial_spec',SPECS75[lang]),('v9_weights',V9[lang]),('v9_spec',SPECSV9[lang])]:
            require(files[lang+'_'+suffix]['sha256']==expected,'Unchanged75/v9 reference identity differs')
    baseline=read(path('baseline122_report'));bp=read(path('baseline122_plan'));root=read(path('baseline122_verification'))
    require(files['baseline122_source']['sha256']==BASELINE_SOURCE_SHA and files['baseline122_verification']['sha256']==BASELINE_ROOT_SHA,'Frozen122 proof identity differs')
    require(baseline['complete'] is True and baseline['qualified'] is False and baseline['source_sha256']==BASELINE_SOURCE_SHA
        and baseline['plan_sha256']==files['baseline122_plan']['sha256'],'Completed122 reference provenance differs')
    require(root['verification']=='PASS' and root['producer_report_sha256']==files['baseline122_report']['sha256']
        and root['rows_recounted']==6160 and root['panel_responses_recounted']==80
        and root['initial75_head_rows_independently_replayed']==3080 and root['all_group_metrics_exact'] is True and root['all_panel_metrics_exact'] is True,'Independent122 proof incomplete')
    for lang in ('ro','uk'):
        require(bp['files'][lang+'_rows']['sha256']==files[lang+'_bank_rows']['sha256'],'122 bank row order differs')
        for ref in ('unchanged-initial75','original-v9'):
            for kind in ('logits.npy','results.jsonl'):
                name=lang+'-'+ref+'-'+kind
                require(baseline['artifacts'][name]==files['baseline122_artifact:'+name]['sha256'],'122 retained baseline differs')
        require(bp['files'][lang+'_spec75']['sha256']==SPECS75[lang] and bp['files'][lang+'_specv9']['sha256']==SPECSV9[lang],'122 spec differs')
    for key in ('full_panel','panel_alias_map'):
        require(bp['files'][key]['sha256']==files[key]['sha256'],'Frozen40 panel mapping differs')
    for key in ('feature_audit121','auditor_source','producer_report120'):
        require(bp['files'][key]['sha256']==files['baseline122_lineage:'+key]['sha256'],'122-to121 feature lineage differs')
    feature_audit=read(path('baseline122_lineage:feature_audit121'))
    require(feature_audit['verification']=='PASS' and feature_audit['qualified'] is False
        and feature_audit['token_rows_replayed']==feature_audit['head_rows_replayed']==3080
        and feature_audit['source_sha256']==files['baseline122_lineage:auditor_source']['sha256']
        and feature_audit['producer_report_sha256']==files['baseline122_lineage:producer_report120']['sha256'],'121 token/head audit differs')
    for lang in ('ro','uk'):
        require(feature_audit['feature_artifacts'][lang+'-rows.jsonl']==files[lang+'_bank_rows']['sha256']
            and feature_audit['encoder_weights'][lang]==V9[lang],'121 bank rows/original-v9 differs')
    require(feature_audit['full_panel_sha256']==files['full_panel']['sha256'] and feature_audit['panel_alias_map_sha256']==files['panel_alias_map']['sha256'],'121 full40 map differs')
    return baseline


def selection_contract(plan):
    f=plan['files'];path=lambda k:Path(f[k]['path'])
    producer=read(path('scale128_report'));pplan=read(path('scale128_plan'));audit=read(path('audit129_report'));aplan=read(path('audit129_plan'))
    require(pplan['registered'] is True and pplan['authorization']=='TRAIN_ONLY_BOUNDED_SCALING' and pplan['source_sha256']==SCALE_SHA,'128 registered source contract differs')
    require(aplan['registered'] is True and aplan['source_sha256']==AUDIT_SHA and type(aplan['producer_exit_code']) is int and aplan['producer_exit_code']==0,'129 registered source/terminal contract differs')
    require(f['scale128_source']['sha256']==SCALE_SHA and f['audit129_source']['sha256']==AUDIT_SHA,'Frozen scale/auditor source differs')
    require(producer['format']=='roguard-train-positive-scale-1' and producer['complete'] is True
        and producer['source_sha256']==SCALE_SHA and producer['plan_sha256']==f['scale128_plan']['sha256']
        and producer['dev_rows_scored']==producer['new_probe_rows_scored']==0 and producer['qualified'] is False,'128 TRAIN-only freeze incomplete')
    require(audit['format']=='roguard-train-positive-scale-saved-audit-1' and audit['verification']=='PASS'
        and audit['source_sha256']==AUDIT_SHA and audit['audit_plan_sha256']==f['audit129_plan']['sha256']
        and audit['producer_source_sha256']==SCALE_SHA and audit['producer_plan_sha256']==f['scale128_plan']['sha256']
        and audit['producer_report_sha256']==f['scale128_report']['sha256']
        and audit['total_train_rows']==28929 and audit['total_candidate_rows_recounted']==202503,'Independent129 selection proof incomplete')
    for flag in ('encoder_loaded','parent_head_replayed','optimizer_run','calibration_claimed','held_out_test_opened','qualified'):
        require(audit[flag] is False,'129 scope changed: '+flag)
    require(type(audit['dev_rows_scored']) is int and type(audit['new_probe_rows_scored']) is int
        and audit['dev_rows_scored']==audit['new_probe_rows_scored']==0,'129 used evaluation rows')
    identical(audit['input_files'],pplan['files'],'129 input ledger differs')
    identical(audit['output_files'],aplan['outputs'],'129 output ledger differs')
    for name,h in producer['artifacts'].items():require(audit['output_files'][name]['sha256']==h,'129 producer artifact differs')
    selection=read(path('scale128_selection'))
    require(producer['artifacts']['selection-before-dev.json']==f['scale128_selection']['sha256']
        and plan['frozen_selection_sha256']==f['scale128_selection']['sha256'],'Selection snapshot differs')
    identical(selection,audit['selections'],'Audited selection differs')
    require(set(selection)=={'ro','uk'},'Both frozen scales required')
    for lang,alpha in ALPHAS.items():
        summary=audit['languages'][lang];n=TRAIN[lang]
        require(type(summary['train_rows']) is int and summary['train_rows']==n and type(summary['scales']) is int and summary['scales']==7
            and type(summary['candidate_rows_recounted']) is int and summary['candidate_rows_recounted']==n*7 and summary['parent_head_replayed'] is False,'129 per-language recount differs')
        identical([item['alpha'] for item in summary['candidates']],[1.,1.125,1.25,1.5,2.,3.,4.],'129 complete ordered candidate family differs')
        require(all(type(item['groups']['all']['rows']) is int and item['groups']['all']['rows']==n for item in summary['candidates']),'129 candidate denominator differs')
        identical(selection[lang],summary['selected'],'129 language selection differs')
        identical(selection[lang],producer['languages'][lang]['selected'],'128/129 selection record differs')
        require(type(selection[lang]['alpha']) is float and selection[lang]['alpha']==alpha,'TRAIN-chosen scale substituted')
        require(pplan['files'][lang+'_weights75']['sha256']==WEIGHTS75[lang] and pplan['files'][lang+'_spec75']['sha256']==SPECS75[lang]
            and pplan['files'][lang+'_original_rows']['sha256']==f[lang+'_original_rows']['sha256'],'128 parent/reference population differs')
    return selection


def preflight(plan_path,expected):
    require(sha(plan_path)==expected,'Evaluation plan changed');plan=read(plan_path)
    require(plan['format']==FORMAT and plan['registered'] is True and plan['authorization']=='ROOT_REGISTERED_FROZEN_SCALE_EVALUATION_ONLY'
        and plan['source_sha256']==sha(__file__) and plan['new_alpha_search_permitted'] is False and plan['encoder_permitted'] is False
        and plan['head_replay_permitted'] is False and plan['training_permitted'] is False,'Unregistered frozen-scale evaluation')
    identical(plan['frozen_alpha'],ALPHAS,'Frozen scales differ')
    require(plan['disk_reserve_bytes']==DISK_RESERVE and plan['maximum_output_bytes']==MAX_OUTPUT,'Resource contract differs')
    checks(plan,plan_path,expected)
    require(plan['files']['design']['sha256']==DESIGN_SHA,'Prospective design differs')
    require(version('numpy')==plan['numpy_version'],'NumPy version differs')
    selection=selection_contract(plan);baseline=lineage(plan);metadata={}
    for lang in ('ro','uk'):
        old=rows(plan['files'][lang+'_original_rows']['path']);new=rows(plan['files'][lang+'_bank_rows']['path']);validate_rows(old,lang,False);validate_rows(new,lang,True)
        indices=[i for i,r in enumerate(old) if r['partition']=='dev'];dev=[old[i] for i in indices]
        warm=rows(plan['files'][lang+'_initial_dev_rows']['path']);keys=('case_id','language','fields','gold','token_sha256')
        identical([[r[k] for k in keys] for r in dev],[[r[k] for k in keys] for r in warm],'Original DEV order differs from warm90')
        metadata[lang]=dict(original_rows=old,dev_indices=indices,dev_rows=dev,bank_rows=new)
    checks(plan,plan_path,expected);return plan,metadata,selection,baseline

def normalized_row(row,population,index):
    result=dict(row);result['group']=row.get('group',population+':'+row['partition'])
    require(isinstance(result['group'],str) and result['group'],'Missing source group')
    result['population']=population;result['evaluation_index']=index;return result


def axes(row,population,status):
    result={'all':'all','population':population,'partition':row['partition'],'gold':str(row['gold']),
        'gold_assertion':str(row['fields'][2]),'source_group':row['group'],'status':status}
    if population=='larger_bank':
        fallback=row.get('source_panel_row',{})
        for key in ('operator_family','meaning_family','register','core_type','block_id'):
            value=row.get(key,fallback.get(key));require(isinstance(value,str) and value,'Missing bank stratum: '+key);result[key]=value
        context=row.get('context_variant',fallback.get('context'));require(isinstance(context,str) and context,'Missing bank context');result['context']=context
    return result


def summarize(records,scale):
    result=scale.summarize(records)
    for name in ('75','v9'):
        result['introduced_vs_'+name]=sum(r['introduced_vs_'+name] for r in records)
        result['fixed_vs_'+name]=sum(r['fixed_vs_'+name] for r in records)
    result['unary_changes_vs_75']=sum(r['unary_changes_vs_75'] for r in records)
    result['tie_changes_vs_75']=sum(r['tie_changes_vs_75'] for r in records)
    return result


def grouped(records,source,population,scale):
    require(len(records)==len(source) and [r['case_id'] for r in records]==[r['case_id'] for r in source],'Filtered/reordered scoring rows')
    groups={}
    for record,row in zip(records,source):
        for key,value in axes(row,population,record['status']).items():groups.setdefault(key+':'+value,[]).append(record)
    return {key:summarize(values,scale) for key,values in sorted(groups.items())}


def evaluate_population(source,parent,v9,spec,v9spec,alpha,population,scale,oracle,scorer,np):
    n=len(source)
    for values in (parent,v9):require(values.dtype==np.float32 and values.shape==(n,19) and np.isfinite(values).all(),'Invalid evaluation reference logits')
    require(type(alpha) is float and alpha==ALPHAS[source[0]['language']],'Only exact frozen language alpha allowed')
    candidate=np.empty_like(parent);statuses=[]
    for i,z in enumerate(parent):candidate[i],status=scale.scale_logits(z,alpha);statuses.append(status)
    independent,independent_status=oracle.oracle_transform(parent,alpha)
    require(candidate.tobytes()==independent.tobytes() and statuses==independent_status,'Independent frozen-transform byte/status mismatch')
    normalized=[normalized_row(row,population,i) for i,row in enumerate(source)];records={name:[] for name in REFERENCES}
    for i,row in enumerate(normalized):
        parent_record=scale.row_result(parent[i],parent[i],row,spec,scorer,'identity',1.)
        v9_record=scale.row_result(v9[i],v9[i],row,v9spec,scorer,'identity',1.)
        candidate_record=scale.row_result(parent[i],candidate[i],row,spec,scorer,statuses[i],alpha)
        require(candidate_record['argmax7']==parent_record['argmax7'] and candidate_record['tie_sets']==parent_record['tie_sets']
            and candidate[i,3:6].tobytes()==parent[i,3:6].tobytes() and candidate[i,17:19].tobytes()==parent[i,17:19].tobytes(),
            'Frozen scale changed point labels/ties/protected logits')
        require(candidate_record['field_errors']==parent_record['field_errors'] and candidate_record['joint_error']==parent_record['joint_error'],'Point-label error counts changed')
        for name,result in zip(REFERENCES,(parent_record,v9_record,candidate_record)):
            result.update(reference=name,population=population,evaluation_index=i,
                parent_reference='unchanged-initial75' if name=='frozen-scale75' else name,
                introduced_vs_75=bool(result['error'] and not parent_record['error']),fixed_vs_75=bool(parent_record['error'] and not result['error']),
                introduced_vs_v9=bool(result['error'] and not v9_record['error']),fixed_vs_v9=bool(v9_record['error'] and not result['error']),
                unary_changes_vs_75=result['argmax7']!=parent_record['argmax7'],tie_changes_vs_75=result['tie_sets']!=parent_record['tie_sets'])
            records[name].append(result)
    return normalized,{'unchanged-initial75':parent,'original-v9':v9,'frozen-scale75':candidate},records


def panel_indices(source,panel,mapping,language):
    require(len(panel)==len(mapping['aliases'])==40,'Complete full40 panel required')
    lookup={r['case_id']:i for i,r in enumerate(source)};require(len(lookup)==len(source),'Duplicate bank identities')
    indices=[];metadata=[]
    for i,(original,alias) in enumerate(zip(panel,mapping['aliases'])):
        require(type(alias['panel_index']) is int and alias['panel_index']==i and alias['panel_example_id']==original['example_id'],'Panel map order differs')
        if original['language']!=language:continue
        index=lookup[alias['canonical_case_id']];row=source[index]
        require(row['language']==language and row['partition']==alias['canonical_partition'] and row['partition'] in ('probe_operator','probe_both','counterfoil'),'Panel mapping includes fitting row')
        identical(row['fields'],alias['proposed_fields'],'Panel fields differ');identical(row['gold'],alias['proposed_gold'],'Panel binary differs')
        identical(row['gold'],original['proposed_binary'],'Panel original binary differs');require(alias['labels_checked'] is True,'Panel labels not checked')
        require(alias['alias_kind'] in ('EXACT_ROLE_PRESERVING_NORMALIZED_BANK_ALIAS','UNIQUE_SUPPLEMENTAL_NONFITTING_CONTROL'),'Unknown alias kind')
        bank=alias['alias_kind']=='EXACT_ROLE_PRESERVING_NORMALIZED_BANK_ALIAS';require(bank==(row['partition']!='counterfoil'),'Panel alias type differs')
        indices.append(index);metadata.append(dict(panel_index=i,panel_example_id=original['example_id'],case_id=row['case_id'],bank_index=index,bank_alias=bank))
    require(len(indices)==len(set(indices))==20 and dict(Counter(source[i]['partition'] for i in indices))=={'probe_operator':8,'probe_both':8,'counterfoil':4},'Full20 per-language panel denominator differs')
    return indices,metadata


def compare122(groups,panel,reference,language,baseline):
    retained=baseline['languages'][language][reference]
    for part,count in BANK.items():
        current=groups['partition:'+part];old=retained['partition'][part]
        require(current['rows']==old['rows']==count,'122 partition count differs')
        for key in ('binary_errors','joint_errors'):
            require(current[key]==old[key],'122 reference errors differ')
        require(current['field_errors']==[old['field_errors'][f] for f in FIELDS]
            and current['field_errors'][2]==old['assertion_errors'],'122 reference field errors differ')
    retained_panel=baseline['panel'][reference][language]
    for key in ('rows','binary_errors','joint_errors','bank_aliases'):
        require(panel[key]==retained_panel[key],'122 fullpanel reference differs')
    require(panel['field_errors']==[retained_panel['field_errors'][f] for f in FIELDS],'122 fullpanel field errors differ')


def diagnostic_comparison(summaries):
    candidate=summaries['frozen-scale75'];result={}
    for ref in ('unchanged-initial75','original-v9'):
        baseline=summaries[ref]
        result[ref]={key:dict(rows=value['rows'],binary_error_delta=value['binary_errors']-baseline[key]['binary_errors'],
            field_errors_preserved=(value['field_errors']==baseline[key]['field_errors']),
            binary_nll_delta=value['binary_nll']-baseline[key]['binary_nll'],binary_brier_delta=value['binary_brier']-baseline[key]['binary_brier'],
            introduced=value['introduced_vs_75' if ref=='unchanged-initial75' else 'introduced_vs_v9'],
            fixed=value['fixed_vs_75' if ref=='unchanged-initial75' else 'fixed_vs_v9'])
            for key,value in candidate.items() if key in baseline and not key.startswith('status:')}
    return result


def write_json(path,value):
    raw=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    with Path(path).open('xb') as f:f.write(raw)
    require(Path(path).read_bytes()==raw,'Retained JSON write differs')


def run(plan_path,expected,output):
    output=Path(output);require(not output.exists(),'Refusing overwrite')
    plan,metadata,selection,baseline=preflight(plan_path,expected)
    import numpy as np
    from roguard.gemma4_review import _score
    scale=module(plan,'scale128_source','frozen_scale128');oracle=module(plan,'audit129_source','independent_transform129')
    require(tuple(scale.FIELDS)==FIELDS and tuple(oracle.FIELDS)==FIELDS,'Actual semantic field order differs')
    origins(plan);checks(plan,plan_path,expected)
    arrays={}
    def load(key):return np.load(io.BytesIO(bound(plan,key)),allow_pickle=False)
    for lang,meta in metadata.items():
        old=load(lang+'_original_logits');require(old.dtype==np.float32 and old.shape==(TRAIN[lang]+DEV[lang],19) and np.isfinite(old).all(),'Original88 retained logits invalid')
        arrays[lang]={
            'original_dev':(load(lang+'_initial_dev_logits'),old[meta['dev_indices']].copy()),
            'larger_bank':(load('baseline122_artifact:'+lang+'-unchanged-initial75-logits.npy'),load('baseline122_artifact:'+lang+'-original-v9-logits.npy'))}
        for population,(parent,v9) in arrays[lang].items():
            n=DEV[lang] if population=='original_dev' else sum(BANK.values())
            require(parent.dtype==v9.dtype==np.float32 and parent.shape==v9.shape==(n,19) and np.isfinite(parent).all() and np.isfinite(v9).all(),'Evaluation population arrays invalid')
    panel=rows(plan['files']['full_panel']['path']);mapping=read(plan['files']['panel_alias_map']['path']);panel_maps={}
    for lang in ('ro','uk'):panel_maps[lang]=panel_indices(metadata[lang]['bank_rows'],panel,mapping,lang)
    checks(plan,plan_path,expected);origins(plan);resource_check(output);output.mkdir()
    source_raw=bound({'files':{'self':{'path':__file__,'sha256':plan['source_sha256']}}},'self');plan_raw=Path(plan_path).read_bytes()
    require(hashlib.sha256(plan_raw).hexdigest()==expected,'Plan changed before snapshot')
    (output/'source.py').write_bytes(source_raw);(output/'registered-plan.json').write_bytes(plan_raw)
    proof=dict(frozen_alpha=ALPHAS,selection_sha256=plan['frozen_selection_sha256'],audit129_report_sha256=plan['files']['audit129_report']['sha256'],
        scale_source_sha256=SCALE_SHA,oracle_source_sha256=AUDIT_SHA,new_search=False,selection=selection)
    write_json(output/'frozen-selection-before-evaluation.json',proof)
    report=dict(format='roguard-frozen-train-scale-evaluation-1',complete=False,source_sha256=plan['source_sha256'],plan_sha256=expected,
        frozen_alpha=ALPHAS,selection_before_evaluation_sha256=sha(output/'frozen-selection-before-evaluation.json'),languages={},
        encoder_loaded=False,head_replayed=False,optimizer_run=False,new_alpha_search=False,train_rows_scored=0,held_out_test_opened=False,
        qualification=False,calibration_claimed=False,conformal_coverage_inherited=False,qualified=False)
    def persist():
        temp=output/'report.tmp';raw=(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode();temp.write_bytes(raw);temp.replace(output/'report.json')
        require((output/'report.json').read_bytes()==raw,'Report readback differs')
    persist();snapshot={p:sha(p) for p in output.iterdir() if p.name!='report.json'}
    try:
        for lang in ('ro','uk'):
            spec=json.loads(bound(plan,lang+'_initial_spec'));v9spec=json.loads(bound(plan,lang+'_v9_spec'))
            require(spec['adapter_sha256']==WEIGHTS75[lang] and v9spec['adapter_sha256']==V9[lang],'Scoring spec/head differs')
            language_result={};report['languages'][lang]=language_result
            for population,key in [('original_dev','dev_rows'),('larger_bank','bank_rows')]:
                source=metadata[lang][key];parent,v9=arrays[lang][population]
                normalized,z,records=evaluate_population(source,parent,v9,spec,v9spec,ALPHAS[lang],population,scale,oracle,_score,np)
                folder=output/lang/population;folder.mkdir(parents=True)
                write_json(folder/'row-order.json',[dict(case_id=r['case_id'],partition=r['partition'],fields=r['fields'],gold=r['gold'],
                    token_sha256=r['token_sha256']) for r in normalized]);snapshot[folder/'row-order.json']=sha(folder/'row-order.json')
                retained_records={};summaries={};panels={}
                for ref in REFERENCES:
                    npy=folder/(ref+'-logits.npy');np.save(npy,z[ref],allow_pickle=False);saved=np.load(npy,allow_pickle=False)
                    require(saved.dtype==z[ref].dtype and saved.shape==z[ref].shape and saved.tobytes()==z[ref].tobytes(),'Retained evaluation logits differ')
                    target=folder/(ref+'-results.jsonl')
                    with target.open('x') as f:
                        for record in records[ref]:f.write(canonical(record)+'\n')
                    retained=rows(target);identical(retained,records[ref],'Retained evaluation row results differ')
                    # Recompute complete outcomes from saved matrices, including both references.
                    reread=scale.row_result
                    for i,row in enumerate(normalized):
                        current_spec=v9spec if ref=='original-v9' else spec
                        base=parent[i] if ref=='frozen-scale75' else saved[i]
                        replay=reread(base,saved[i],row,current_spec,_score,retained[i]['status'],ALPHAS[lang] if ref=='frozen-scale75' else 1.)
                        for field,value in replay.items():identical(retained[i][field],value,'Saved outcome replay differs: '+field)
                    retained_records[ref]=retained;summaries[ref]=grouped(retained,normalized,population,scale)
                    snapshot[npy]=sha(npy);snapshot[target]=sha(target)
                    if population=='larger_bank':
                        indices,map_records=panel_maps[lang];mapped=[dict(mapping=meta,result=retained[index]) for index,meta in zip(indices,map_records)]
                        write_json(folder/(ref+'-full-panel.json'),mapped);snapshot[folder/(ref+'-full-panel.json')]=sha(folder/(ref+'-full-panel.json'))
                        summary=summarize([item['result'] for item in mapped],scale);summary.update(bank_aliases=16,supplemental_rows=4)
                        panels[ref]=summary
                        if ref!='frozen-scale75':compare122(summaries[ref],summary,ref,lang,baseline)
                candidate=summaries['frozen-scale75']['all:all'];baseline75=summaries['unchanged-initial75']['all:all']
                require(candidate['unary_changes_vs_75']==candidate['tie_changes_vs_75']==0
                    and candidate['field_errors']==baseline75['field_errors'] and candidate['joint_errors']==baseline75['joint_errors'],'Point-label conservation failed')
                language_result[population]=dict(rows=len(source),references=summaries,comparisons=diagnostic_comparison(summaries))
                if panels:language_result['full_panel']=panels
                resource_check(output);persist()
        require(sum(v['original_dev']['rows'] for v in report['languages'].values())==sum(DEV.values())
            and sum(v['larger_bank']['rows'] for v in report['languages'].values())==2*sum(BANK.values()),'Complete evaluation denominators differ')
        total_dev=sum(DEV.values());total_bank=2*sum(BANK.values())
        report.update(original_dev_rows_scored=total_dev,larger_bank_rows_scored=total_bank,unique_candidate_rows=total_dev+total_bank,
            reference_rows_scored=2*(total_dev+total_bank),panel_responses_recounted=120,unary_point_labels_and_ties_preserved=True,
            decision='FIXED_TRAIN_SCALE_EVALUATION_ONLY_NO_CALIBRATION_OR_QUALIFICATION',
            limitations=['Exactly one TRAIN-chosen scale per language; no DEV/new-bank search or125/131 model combination.',
                'Unary six-field tuples and legacy argmax/ties unchanged from75; coupled joint MAP and policy decisions may change.',
                'Full40 panel has32 aliases and8supplemental inputs already in3080; it is not40independent extra examples.',
                'NLL/Brier are descriptive. No inherited conformal coverage, calibrated uncertainty, native75 parity or independent human validation.'])
        checks(plan,plan_path,expected);origins(plan)
        require(all(sha(p)==h for p,h in snapshot.items()),'Retained evidence changed before completion')
        require((output/'source.py').read_bytes()==source_raw and (output/'registered-plan.json').read_bytes()==plan_raw,'Retained source/plan snapshots differ')
        report['artifacts']={str(p.relative_to(output)):h for p,h in snapshot.items()};report['complete']=True;persist()
        final_report_sha=sha(output/'report.json');checks(plan,plan_path,expected);origins(plan);resource_check(output)
        require(all(sha(p)==h for p,h in snapshot.items()) and sha(output/'report.json')==final_report_sha,'Evidence changed after final write')
        return report
    except BaseException as error:
        report.update(complete=False,error_type=type(error).__name__,error=str(error))
        try:persist()
        except BaseException:
            marker=output/'report.json'
            if marker.exists():marker.replace(output/'rejected-unreadable-report.json')
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.plan,a.plan_sha256,a.output)
