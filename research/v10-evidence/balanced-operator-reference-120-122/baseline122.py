"""Prospective fixed-reference profiling; no fitting or reference selection.

Requires separately registered completed120/121 evidence. Until then this module
supports synthetic helper tests only. Original-v9 uses audited retained logits;
unchanged initial75 uses exact-erf CPU head replay on the same v9 feature space.
"""
import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

FORMAT='roguard-balanced-baseline-profile-plan-1'
COUNTS={'fit':288,'probe_operator':384,'probe_meaning':384,'probe_combination':96,'probe_both':384,'counterfoil':4}
FIELDS=('subject','time','assertion','source','relation','concern')
SIZES=(3,3,4,3,2,2)
ASSERTIONS=('AFFIRMED','NEGATED','UNCERTAIN','RETRACTED')
TRANSFER=('probe_operator','probe_both')
V9={'ro':'114d3e89191d06152cceccd080114998901a4f0bf6a9540de8e11dd8b6cf6b09',
    'uk':'0fee43a071d2c07c26e8b3d69cacb4933fd284bc64dbee0d18fe5481e1e4815f'}
WEIGHTS75={'ro':'1c18db5e5fc26baf4c0739d053c0546338201d76fcb9061e47477c14650535e3',
           'uk':'466a1900655c85a6db809adb384d90fc634357a4f7b72f1415274c4787b3ecb7'}
SPECS75={'ro':'5fdeae97843d873755ed41a02453b5c29823fd8028508c3e99d1586175c8f4a9',
         'uk':'bbf3fea8ee298c40d86701851fe045352ea1a924ba5287dd52b70e59ce25de27'}
SPECSV9={'ro':'6b2ea2979ffa43c735d8f8a6e3f7e1fbf6cf5c085889e8d484a13bf43026ccc5',
         'uk':'fc86fed9981b786149c4f6855c3d98f07067654cf2af05d40aef589345be34ed'}
AUDITOR_SHA='288e7484da55467e315b7e3f2397206033c3b658a16aa4d3eea83df637272d2c'
EXTRACTOR_SHA='4476f0862bcd3d579e70f738bf6cf5a2eec978454aa3a7a04a8070166bb24de1'
CANONICAL_SHA='e5a660b024141b47adda8e3f5e110b4f49d9663195562b9599cf16355dddc31e'
ALIASES_SHA='3037b1bd4c304d571ce5eba2b4b1950c4942ae1bf2700ecab7fa234d51dbccfa'
PANEL_SHA='3781ab148c79454b3078c741e70f0bb0ad648f917e63113e04a324402e78335e'


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def canonical(value):return json.dumps(value,ensure_ascii=True,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode('ascii')).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]


def check_files(plan):
    for name,binding in plan['files'].items():
        require(Path(binding['path']).is_absolute() and sha(binding['path'])==binding['sha256'],'Bound file changed: '+name)


def check_reference_pins(files):
    for language in ('ro','uk'):
        for suffix,expected in [('weights75',WEIGHTS75[language]),('spec75',SPECS75[language]),('specv9',SPECSV9[language])]:
            require(files[language+'_'+suffix]['sha256']==expected,
                    'Fixed original-v9/unchanged75 reference differs;95 or alternate heads forbidden')


def validate_rows(values,language):
    require(len(values)==1540 and Counter(r['partition'] for r in values)==COUNTS,'Complete language/fold population differs')
    require(len({r['case_id'] for r in values})==1540,'Duplicate retained case identity')
    for row in values:
        require(row['language']==language and row['payload']['language']==language,'Retained language differs')
        require(isinstance(row['fields'],list) and len(row['fields'])==6
            and all(type(v) is int and 0<=v<size for v,size in zip(row['fields'],SIZES))
            and type(row['gold']) is int and row['gold'] in (0,1),'Retained labels require exact integers')
        require(type(row['source_focus']) is int and row['source_focus']==len(row['payload']['turns'])-1
            and row['payload']['turns'][-1]['role']=='minor','Final minor focus differs')
        require(canonical(row['fields'])==canonical(row['proposed_fields'])
            and canonical(row['gold'])==canonical(row['proposed_gold'])
            and row['training_admitted'] is False and row['independently_annotated'] is False,'Unreviewed labels or admission claim')


def preflight(path,expected_sha):
    """Global metadata/binding preflight, before numerical imports or head replay."""
    require(sha(path)==expected_sha,'Profile plan changed')
    plan=read(path)
    require(plan['format']==FORMAT and plan['registered'] is True
        and plan['authorization']=='ROOT_REGISTERED_FIXED_BASELINE_PROFILE_ONLY'
        and plan['source_sha256']==sha(__file__)
        and plan['references']==['original-v9','unchanged-initial75']
        and plan['primary_reference']=='unchanged-initial75'
        and plan['primary_transfer_partitions']==list(TRANSFER)
        and plan['primary_transfer_rows_per_language']==768
        and plan['training_performed'] is False and plan['selection_permitted'] is False
        and plan['held_out_test_opened'] is False,'Unregistered or changed profiling scope')
    check_reference_pins(plan['files']);check_files(plan)
    paths={name:Path(b['path']) for name,b in plan['files'].items()}
    require(sha(paths['canonical_rows'])==CANONICAL_SHA and sha(paths['panel_alias_map'])==ALIASES_SHA
        and sha(paths['full_panel'])==PANEL_SHA,'Reviewed3080/40 population changed')
    audit=read(paths['feature_audit121']);producer=read(paths['producer_report120']);extraction=read(paths['extraction_plan120'])
    require(audit['format']=='roguard-assertion-probe-features-audit-1' and audit['verification']=='PASS'
        and audit['source_sha256']==AUDITOR_SHA and audit['encoder_weights']==V9
        and audit['token_rows_replayed']==audit['head_rows_replayed']==3080
        and audit['extraction_plan_sha256']==sha(paths['extraction_plan120'])
        and audit['producer_report_sha256']==sha(paths['producer_report120'])
        and audit['full_panel_sha256']==PANEL_SHA and audit['panel_alias_map_sha256']==ALIASES_SHA
        and all(audit[k] is False for k in ('encoder_loaded','native_encoder_independently_replayed',
            'probe_accuracy_summaries_produced','training_performed','training_admitted','held_out_test_opened','qualified')),
        'Completed original-v9 feature audit121 is required')
    require(extraction['extractor_sha256']==EXTRACTOR_SHA and extraction['total_rows']==3080
        and producer['complete'] is True and producer['source_sha256']==EXTRACTOR_SHA
        and producer['plan_sha256']==sha(paths['extraction_plan120'])
        and producer['training_performed'] is False and producer['qualified'] is False,'Wrong feature producer lineage')
    require(audit['panel_alias_summary']==dict(full_panel_rows=40,main_bank_alias_rows=32,unique_supplemental_rows=8,
        all_panel_responses_represented=True,independent_extra_panel_examples=False),'Panel alias audit scope differs')
    warm=read(paths['warm_start90_report'])
    require(warm['complete'] is True and warm['qualified'] is False,'Original75 provenance incomplete')
    original=rows(paths['canonical_rows']);token_records=rows(paths['token_records120'])
    require(len(original)==len(token_records)==3080,'Canonical/token population differs')
    expected=[]
    for source,token in zip(original,token_records):
        require(token['case_id']==source['case_id'] and token['source_row_sha256']==digest(source),'Canonical token ordering differs')
        expected.append(dict(source,tokens=token['tokens'],token_sha256=token['token_sha256'],
            source_row_sha256=digest(source),payload_sha256=digest(source['payload']),fields=source['proposed_fields'],gold=source['proposed_gold']))
    language_rows={}
    for language in ('ro','uk'):
        current=rows(paths[language+'_rows']);validate_rows(current,language)
        wanted=[r for r in expected if r['language']==language]
        require(canonical(current)==canonical(wanted),'Exact audited row/proposal order differs')
        for key,suffix in [('rows','rows.jsonl'),('features','features.npy'),('v9_logits','logits.npy')]:
            h=sha(paths[language+'_'+key]);name=language+'-'+suffix
            require(audit['feature_artifacts'][name]==h and producer['languages'][language]['artifacts'][name]==h,
                    'Audited feature artifact differs: '+name)
        summary=audit['languages'][language]
        require(summary['rows']==1540 and summary['partitions']==COUNTS
            and summary['seven_head_argmax_rows_differing']==summary['binary_decision_differences']==0,
            'Audited original-v9 parity/population differs')
        require(sha(paths[language+'_weights75'])==WEIGHTS75[language]
            and sha(paths[language+'_spec75'])==SPECS75[language]
            and sha(paths[language+'_specv9'])==SPECSV9[language]
            and warm['languages'][language]['artifacts'][language+'-selected.safetensors']==WEIGHTS75[language],
            'Only unchanged75-from90 and original-v9 specs allowed;95 heads are excluded')
        for suffix,head_sha in [('spec75',WEIGHTS75[language]),('specv9',V9[language])]:
            spec=read(paths[language+'_'+suffix])
            require(spec['language']==language and spec['adapter_sha256']==head_sha,'Reference spec/head identity differs')
        language_rows[language]=current
    for package in ('numpy','torch','safetensors'):
        require(version(package)==plan['numerical_versions'][package],'Numerical version changed: '+package)
    root=Path(plan['project_root'])
    bound={Path(v['path']).resolve() for v in plan['files'].values()}
    require(all(p.resolve() in bound for p in (root/'src/roguard').rglob('*.py')),'Complete serving source inventory not bound')
    check_files(plan);require(sha(path)==expected_sha and sha(__file__)==plan['source_sha256'],'Plan/source changed during preflight')
    return plan,paths,language_rows


def check_origins(plan):
    root=Path(plan['project_root']).resolve();bound={Path(v['path']).resolve():v['sha256'] for v in plan['files'].values()}
    for name,module in list(sys.modules.items()):
        if name=='roguard' or name.startswith('roguard.'):
            expected=root/'src'/Path(*name.split('.'));expected=expected/'__init__.py' if expected.is_dir() else expected.with_suffix('.py')
            actual=Path(module.__file__).resolve()
            require(actual==expected.resolve() and actual in bound and sha(actual)==bound[actual],'Unbound scorer origin: '+name)


def cpu_head(features,tensors):
    """Same exact-erf CPU operation order used by independently audited116/117."""
    import numpy as np
    import torch
    torch.set_num_threads(2)
    pieces=[]
    with torch.no_grad():
        t={key:torch.from_numpy(value.copy()) for key,value in tensors.items()}
        for start in range(0,len(features),256):
            x=torch.from_numpy(np.array(features[start:start+256],dtype=np.float32,copy=True))
            norm=(x*x).sum(1,keepdim=True).sqrt()
            require(bool(torch.isfinite(x).all()) and bool(torch.isfinite(norm).all()) and bool((norm>0).all()),'Invalid baseline feature row')
            x=x/norm*2560**.5
            h=torch.nn.functional.linear(x,t['first.weight'],t['first.bias'])
            h=h*(1+torch.erf(h/(2.**.5)))/2
            pieces.append(torch.nn.functional.linear(h,t['last.weight'],t['last.bias']).numpy().copy())
    z=np.concatenate(pieces)
    require(z.shape==(len(features),19) and z.dtype==np.float32 and np.isfinite(z).all(),'Invalid initial75 CPU logits')
    return z


def predict_fields(logits):
    offsets=(0,3,6,10,13,15,17)
    return [max(range(b-a),key=lambda i:float(logits[a+i])) for a,b in zip(offsets,offsets[1:])]


def outcome(row,logits,score,threshold):
    prediction=predict_fields(logits);errors=[p!=g for p,g in zip(prediction,row['fields'])]
    binary=int(score>=threshold)
    return dict(case_id=row['case_id'],language=row['language'],partition=row['partition'],
        gold_fields=row['fields'],gold_binary=row['gold'],predicted_fields=prediction,
        field_errors=errors,assertion_error=errors[2],joint_error=any(errors),
        score=float(score),threshold=float(threshold),binary=binary,binary_error=binary!=row['gold'],
        distance_to_threshold=float(abs(score-threshold)),within_1e_3_threshold_band=abs(score-threshold)<=1e-3)


def labels(row):
    fallback=row.get('source_panel_row',{})
    result={'all':'all','partition':row['partition'],'gold_assertion':ASSERTIONS[row['fields'][2]],
            'block_id':row['block_id']}
    for key in ('register','operator_family','meaning_family','core_type'):
        result[key]=row.get(key,fallback.get(key))
        require(isinstance(result[key],str) and result[key],'Missing profiling stratum: '+key)
    result['context']=row.get('context_variant',fallback.get('context'))
    require(isinstance(result['context'],str) and result['context'],'Missing profiling context')
    if row['partition'] in TRANSFER:result['primary_transfer']='all'
    return result


def tally(result,groups):
    for axis,value in groups.items():
        m=result.setdefault(axis,{}).setdefault(value,dict(rows=0,field_errors={key:0 for key in FIELDS},
            binary_errors=0,joint_errors=0,assertion_errors=0,within_1e_3_threshold_band=0,minimum_threshold_distance=None))
        yield m


def summarize(values,source_rows):
    require(len(values)==len(source_rows),'No profiling rows may be filtered')
    summary={}
    for row,result in zip(source_rows,values):
        require(result['case_id']==row['case_id'],'Profile metric order changed')
        for metrics in tally(summary,labels(row)):
            metrics['rows']+=1
            for key,error in zip(FIELDS,result['field_errors']):metrics['field_errors'][key]+=int(error)
            for key in ('binary','joint','assertion'):metrics[key+'_errors']+=int(result[key+'_error'])
            metrics['within_1e_3_threshold_band']+=int(result['within_1e_3_threshold_band'])
            old=metrics['minimum_threshold_distance'];distance=result['distance_to_threshold']
            metrics['minimum_threshold_distance']=distance if old is None else min(old,distance)
    return summary


def panel_results(by_case,panel,mapping):
    require(len(panel)==len(mapping['aliases'])==40,'Full40 panel denominator differs')
    values=[]
    for index,(original,alias) in enumerate(zip(panel,mapping['aliases'])):
        require(type(alias['panel_index']) is int and alias['panel_index']==index
            and alias['panel_example_id']==original['example_id'],'Full40 panel mapping order differs')
        result=by_case[alias['canonical_case_id']]
        require(canonical(result['gold_fields'])==canonical(alias['proposed_fields'])
            and type(original['proposed_binary']) is int and result['gold_binary']==original['proposed_binary'],
            'Panel alias labels differ')
        values.append(dict(panel_index=index,panel_example_id=original['example_id'],canonical_case_id=result['case_id'],
            repeated_main_bank_alias=alias['alias_kind']=='EXACT_ROLE_PRESERVING_NORMALIZED_BANK_ALIAS',
            prediction=result))
    return values


def eligibility(reference_summaries):
    errors={reference:{language:reference_summaries[language][reference]['primary_transfer']['all']['assertion_errors']
        for language in ('ro','uk')} for reference in ('original-v9','unchanged-initial75')}
    for language in ('ro','uk'):
        for reference in errors:
            require(reference_summaries[language][reference]['primary_transfer']['all']['rows']==768,
                    'Primary transfer population must remain all768rows per language')
    failed=[language for language in ('ro','uk') if errors['unchanged-initial75'][language]==0]
    return dict(primary_reference='unchanged-initial75',partitions=list(TRANSFER),rows_per_language=768,
        assertion_errors=errors,strict_improvement_study_eligible=not failed,ineligible_languages=failed,
        zero_ceiling_flags={r:[l for l,n in counts.items() if n==0] for r,counts in errors.items()},
        decision='INELIGIBLE_ZERO_PARENT_ASSERTION_ERROR_FLOOR' if failed else 'PARENT_ERROR_HEADROOM_ONLY_NOT_TRAINING_ADMISSION',
        rule='Either initial75 language atzero makes the intended strict-improvement registration ineligible. No row deletion, easier-reference substitution, training or checkpoint selection follows.',
        qualified=False)


def run(plan_path,plan_sha,output):
    require(not output.exists(),'Refusing existing baseline output')
    plan,paths,language_rows=preflight(plan_path,plan_sha)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='2'
    import numpy as np
    from safetensors.numpy import load
    from roguard.gemma4_review import _score
    check_files(plan);check_origins(plan)
    arrays={}
    for language in ('ro','uk'):
        x=np.load(paths[language+'_features'],mmap_mode='r',allow_pickle=False)
        z=np.load(paths[language+'_v9_logits'],allow_pickle=False)
        require(x.shape==(1540,2560) and z.shape==(1540,19) and x.dtype==z.dtype==np.float32
            and np.isfinite(x).all() and np.isfinite(z).all(),'Audited arrays invalid')
        arrays[language]=(x,z)
    output.mkdir(parents=True,exist_ok=False)
    (output/'source.py').write_bytes(Path(__file__).read_bytes());(output/'registered-plan.json').write_bytes(Path(plan_path).read_bytes())
    report=dict(format='roguard-balanced-baseline-profile-1',complete=False,plan_sha256=plan_sha,
        source_sha256=sha(__file__),languages={},encoder_loaded=False,training_performed=False,
        selection_performed=False,held_out_test_opened=False,training_admitted=False,qualified=False,
        original_v9_backend='Audited retained original-v9 logits from120; audit121 independently replayed its heads.',
        initial75_backend='CPU exact-erf head replay on original-v9 encoder features; no newly established native75 parity.',
        threshold_band='All distances retained;1e-3 is diagnostic only and never filters examples or changes thresholds.',
        panel_scope='40mapped panel rows are a separate diagnostic report;32are bank aliases and are not added to3080unique-input metrics.')
    def persist():
        temp=output/'report.tmp';temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temp.replace(output/'report.json')
    persist();all_results={r:{} for r in ('original-v9','unchanged-initial75')};expected_logits={}
    for language in ('ro','uk'):
        x,teacher=arrays[language]
        raw=paths[language+'_weights75'].read_bytes()
        require(hashlib.sha256(raw).hexdigest()==WEIGHTS75[language],'Original75 weights changed before decode')
        decoded=load(raw)
        tensors={key:decoded['heads.'+language+'.'+key] for key in ('first.weight','first.bias','last.weight','last.bias')}
        initial=cpu_head(x,tensors)
        report['languages'][language]={}
        for reference,z,spec_key in [('original-v9',teacher,'specv9'),('unchanged-initial75',initial,'spec75')]:
            spec=read(paths[language+'_'+spec_key]);np.save(output/(language+'-'+reference+'-logits.npy'),z,allow_pickle=False)
            expected_logits[(language,reference)]=z.copy()
            values=[outcome(row,logits,_score(logits,spec),spec['threshold']) for row,logits in zip(language_rows[language],z)]
            with (output/(language+'-'+reference+'-results.jsonl')).open('x') as f:
                for value in values:f.write(json.dumps(value,sort_keys=True,allow_nan=False)+'\n')
            report['languages'][language][reference]=summarize(values,language_rows[language])
            all_results[reference].update({r['case_id']:r for r in values})
        persist()
    panel=rows(paths['full_panel']);mapping=read(paths['panel_alias_map']);report['panel']={}
    for reference,by_case in all_results.items():
        require(len(by_case)==3080,'All3080 unique reference outcomes must be retained')
        mapped=panel_results(by_case,panel,mapping)
        with (output/(reference+'-full40-panel.jsonl')).open('x') as f:
            for value in mapped:f.write(json.dumps(value,sort_keys=True,allow_nan=False)+'\n')
        summary={}
        for language in ('ro','uk'):
            current=[r for r in mapped if r['prediction']['language']==language]
            require(len(current)==20,'Per-language full panel denominator differs')
            summary[language]=dict(rows=20,bank_aliases=sum(r['repeated_main_bank_alias'] for r in current),
                field_errors={key:sum(int(r['prediction']['field_errors'][i]) for r in current) for i,key in enumerate(FIELDS)},
                binary_errors=sum(int(r['prediction']['binary_error']) for r in current),
                joint_errors=sum(int(r['prediction']['joint_error']) for r in current),
                assertion_errors=sum(int(r['prediction']['assertion_error']) for r in current))
        report['panel'][reference]=summary
    report['eligibility']=eligibility(report['languages'])
    check_files(plan);check_origins(plan)
    require(sha(plan_path)==plan_sha and sha(__file__)==plan['source_sha256'],'Profile plan/source changed')
    # Complete means retained bytes reproduce the exact full reported population.
    # Take one snapshot before readback and require it unchanged afterwards.
    artifact_snapshot={p.name:sha(p) for p in output.iterdir() if p.name not in ('report.json','report.tmp')}
    require((output/'source.py').read_bytes()==Path(__file__).read_bytes()
        and (output/'registered-plan.json').read_bytes()==Path(plan_path).read_bytes(),'Retained source/plan snapshot differs')
    retained_results={reference:{} for reference in all_results};retained_summaries={}
    for language in ('ro','uk'):
        retained_summaries[language]={}
        for reference,spec_key in [('original-v9','specv9'),('unchanged-initial75','spec75')]:
            saved=np.load(output/(language+'-'+reference+'-logits.npy'),allow_pickle=False)
            expected=expected_logits[(language,reference)]
            require(saved.shape==(1540,19) and saved.dtype==np.float32 and np.isfinite(saved).all()
                and np.array_equal(saved,expected),'Retained logits differ from complete produced array')
            spec=read(paths[language+'_'+spec_key])
            recomputed=[outcome(row,logits,_score(logits,spec),spec['threshold']) for row,logits in zip(language_rows[language],saved)]
            stored=rows(output/(language+'-'+reference+'-results.jsonl'))
            require(canonical(stored)==canonical(recomputed)
                and canonical(stored)==canonical([all_results[reference][r['case_id']] for r in language_rows[language]]),
                'Retained outcomes differ from complete logits/source order')
            retained_summaries[language][reference]=summarize(stored,language_rows[language])
            retained_results[reference].update({r['case_id']:r for r in stored})
    require(canonical(retained_summaries)==canonical(report['languages']),'Retained metrics differ')
    require(canonical(eligibility(retained_summaries))==canonical(report['eligibility']),'Retained gate differs')
    for reference,by_case in retained_results.items():
        expected_panel=panel_results(by_case,panel,mapping)
        stored_panel=rows(output/(reference+'-full40-panel.jsonl'))
        require(canonical(stored_panel)==canonical(expected_panel),'Retained full40 panel differs')
        for language in ('ro','uk'):
            selected=[r for r in stored_panel if r['prediction']['language']==language]
            recounted=dict(rows=len(selected),bank_aliases=sum(r['repeated_main_bank_alias'] for r in selected),
                field_errors={key:sum(int(r['prediction']['field_errors'][i]) for r in selected) for i,key in enumerate(FIELDS)},
                binary_errors=sum(int(r['prediction']['binary_error']) for r in selected),
                joint_errors=sum(int(r['prediction']['joint_error']) for r in selected),
                assertion_errors=sum(int(r['prediction']['assertion_error']) for r in selected))
            require(canonical(recounted)==canonical(report['panel'][reference][language]),'Retained panel metric differs')
    check_files(plan);check_origins(plan)
    require(sha(plan_path)==plan_sha and sha(__file__)==plan['source_sha256'],'Plan/source changed during readback')
    require(artifact_snapshot=={p.name:sha(p) for p in output.iterdir() if p.name not in ('report.json','report.tmp')},
            'Artifact changed during retained-output validation')
    report['artifacts']=artifact_snapshot
    report['retained_output_validation']=dict(logit_rows=6160,outcome_rows=6160,panel_rows=80,
        exact_saved_array_value_equality=True,metrics_recomputed_from_saved_logits=True,source_plan_snapshots_verified=True)
    report.update(complete=True,data_now_exposed_engineering_diagnostics=True,
        decision='COMPLETE_FIXED_REFERENCE_PROFILE_ONLY_NO_SELECTION_OR_FITTING')
    persist();return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();result=run(a.plan,a.plan_sha256,a.output)
    print(json.dumps(dict(complete=result['complete'],eligibility=result['eligibility'],qualified=False)))
