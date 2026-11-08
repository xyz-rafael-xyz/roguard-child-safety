"""Independently recount saved133; no encoder/head, alpha search or133 code import."""
import argparse,hashlib,io,json,math,types,shutil
from pathlib import Path
from collections import Counter
import numpy as np
SOURCE133='f135e5d15f60e957299345e6e380dd2ea66ea00bd0441dc2721ef6286bed7208'
ORACLE129='f8290afbd22edbbbe721c2159359c21197e90cb997be08519f3d48b14ffdfd4c'
ALPHAS={'ro':2.,'uk':1.5};DEV={'ro':4202,'uk':4635};TRAIN={'ro':13174,'uk':15755}
BANK={'fit':288,'probe_operator':384,'probe_meaning':384,'probe_combination':96,'probe_both':384,'counterfoil':4}
REFS=('unchanged-initial75','original-v9','frozen-scale75')
def require(v,message):
 if not v:raise ValueError(message)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def rows(path):return [json.loads(line) for line in Path(path).read_bytes().splitlines()]
def canonical(v):return json.dumps(v,ensure_ascii=True,sort_keys=True,separators=(',',':'),allow_nan=False)
def verify(ledger):
 for key,b in ledger.items():require(sha(b['path'])==b['sha256'],'Bound input changed: '+key)
def decoded(binding):
 raw=Path(binding['path']).read_bytes();require(hashlib.sha256(raw).hexdigest()==binding['sha256'],'Input changed before decode')
 return np.load(io.BytesIO(raw),allow_pickle=False)
def save(path,value):
 raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
 with Path(path).open('xb') as f:f.write(raw)
 require(Path(path).read_bytes()==raw,'Audit report readback differs')

def summary(records,oracle):
 result=oracle.summarize(records)
 for name in ('introduced_vs_75','fixed_vs_75','introduced_vs_v9','fixed_vs_v9','unary_changes_vs_75','tie_changes_vs_75'):
  result[name]=sum(int(r[name]) for r in records)
 return result

def grouping(source,records,population,oracle):
 """Independent set-defined memberships, not133's grouping function."""
 require(len(source)==len(records),'Group length differs')
 tags=[]
 for row,record in zip(source,records):
  require(row['case_id']==record['case_id'],'Group identity differs')
  current={'all:all','population:'+population,'partition:'+row['partition'],'gold:'+str(row['gold']),
   'gold_assertion:'+str(row['fields'][2]),'source_group:'+row['group'],'status:'+record['status']}
  if population=='larger_bank':
   fallback=row.get('source_panel_row',{})
   for name in ('operator_family','meaning_family','register','core_type','block_id'):
    value=row[name] if name in row else fallback[name]
    require(isinstance(value,str) and value,'Invalid stratum');current.add(name+':'+value)
   context=row['context_variant'] if 'context_variant' in row else fallback['context'];require(isinstance(context,str) and context,'Invalid context');current.add('context:'+context)
  tags.append(current)
 names=set().union(*tags)
 return {name:summary([record for record,tag in zip(records,tags) if name in tag],oracle) for name in sorted(names)}

def comparisons(groups):
 candidate=groups['frozen-scale75'];out={}
 for ref,suffix in [('unchanged-initial75','75'),('original-v9','v9')]:
  baseline=groups[ref];out[ref]={}
  for name in sorted(set(candidate)&set(baseline)):
   if name.startswith('status:'):continue
   a,b=candidate[name],baseline[name]
   out[ref][name]={'rows':a['rows'],'binary_error_delta':a['binary_errors']-b['binary_errors'],
    'field_errors_preserved':a['field_errors']==b['field_errors'],'binary_nll_delta':a['binary_nll']-b['binary_nll'],
    'binary_brier_delta':a['binary_brier']-b['binary_brier'],'introduced':a['introduced_vs_'+suffix],'fixed':a['fixed_vs_'+suffix]}
 return out

def reconstruct_results(source,parent,v9,candidate,statuses,spec,v9spec,population,language,oracle,scorer):
 normalized=[];results={ref:[] for ref in REFS}
 for i,row in enumerate(source):
  row=dict(row);row['group']=row.get('group',population+':'+row['partition']);normalized.append(row)
  outcomes=[]
  for ref,baseline,z,contract,alpha,status in [
   ('unchanged-initial75',parent[i],parent[i],spec,1.,'identity'),
   ('original-v9',v9[i],v9[i],v9spec,1.,'identity'),
   ('frozen-scale75',parent[i],candidate[i],spec,ALPHAS[language],statuses[i])]:
   result=oracle.outcome(baseline,z,row,contract,scorer,alpha,status,i);del result['train_index'];outcomes.append(result)
  baseline75,baselinev9,_=outcomes
  for ref,result in zip(REFS,outcomes):
   result.update(reference=ref,population=population,evaluation_index=i,parent_reference='unchanged-initial75' if ref=='frozen-scale75' else ref,
    introduced_vs_75=bool(result['error'] and not baseline75['error']),fixed_vs_75=bool(baseline75['error'] and not result['error']),
    introduced_vs_v9=bool(result['error'] and not baselinev9['error']),fixed_vs_v9=bool(baselinev9['error'] and not result['error']),
    unary_changes_vs_75=result['argmax7']!=baseline75['argmax7'],tie_changes_vs_75=result['tie_sets']!=baseline75['tie_sets'])
   results[ref].append(result)
 return normalized,results

def panel_recount(source,results,panel,mapping,language,oracle):
 require(len(panel)==len(mapping['aliases'])==40,'Full panel differs');lookup={r['case_id']:i for i,r in enumerate(source)};selected=[]
 for j in range(40):
  original,alias=panel[j],mapping['aliases'][j]
  require(type(alias['panel_index']) is int and alias['panel_index']==j and alias['panel_example_id']==original['example_id'],'Panel source/order differs')
  if original['language']!=language:continue
  index=lookup[alias['canonical_case_id']];row=source[index]
  require(row['language']==language and row['partition']==alias['canonical_partition'] and row['partition'] in ('probe_operator','probe_both','counterfoil'),'Panel fitting or wrong-language row')
  require(canonical(row['fields'])==canonical(alias['proposed_fields']) and canonical(row['gold'])==canonical(alias['proposed_gold'])==canonical(original['proposed_binary']) and alias['labels_checked'] is True,'Panel target differs')
  kind=alias['alias_kind'];require(kind in ('EXACT_ROLE_PRESERVING_NORMALIZED_BANK_ALIAS','UNIQUE_SUPPLEMENTAL_NONFITTING_CONTROL'),'Panel alias kind differs')
  bank=kind=='EXACT_ROLE_PRESERVING_NORMALIZED_BANK_ALIAS';require(bank==(row['partition']!='counterfoil'),'Panel alias partition differs')
  selected.append((index,dict(panel_index=j,panel_example_id=original['example_id'],case_id=row['case_id'],bank_index=index,bank_alias=bank)))
 require(len(selected)==len({i for i,_ in selected})==20 and Counter(source[i]['partition'] for i,_ in selected)==dict(probe_operator=8,probe_both=8,counterfoil=4),'Full panel denominator differs')
 panel_results={ref:[dict(mapping=meta,result=results[ref][i]) for i,meta in selected] for ref in REFS}
 summary_results={ref:dict(summary([r['result'] for r in values],oracle),bank_aliases=16,supplemental_rows=4) for ref,values in panel_results.items()}
 return panel_results,summary_results

def run(path,expected,out):
 out=Path(out);require(not out.exists(),'Refusing overwrite');raw=Path(path).read_bytes();require(hashlib.sha256(raw).hexdigest()==expected,'Audit plan changed');plan=json.loads(raw)
 require(plan['format']=='roguard-saved133-independent-audit-1' and plan['registered'] is True and plan['source_sha256']==sha(__file__)
  and type(plan['producer_exit_code']) is int and plan['producer_exit_code']==0 and plan['encoder_permitted'] is False and plan['head_replay_permitted'] is False,'Not registered saved-output audit')
 verify(plan['files']);verify(plan['outputs']);f=lambda key:Path(plan['files'][key]['path'])
 producer=read(f('producer_plan'));report=read(f('producer_report'));root=f('producer_report').parent
 require(producer['source_sha256']==report['source_sha256']==plan['files']['producer_source']['sha256']==SOURCE133 and report['plan_sha256']==plan['files']['producer_plan']['sha256'],'133 source/plan differs')
 require(report['complete'] is True and report['qualified'] is False and report['frozen_alpha']==ALPHAS and report['original_dev_rows_scored']==8837 and report['larger_bank_rows_scored']==3080 and report['unique_candidate_rows']==11917 and report['reference_rows_scored']==23834 and report['panel_responses_recounted']==120,'133 scope/denominator differs')
 for flag in ('encoder_loaded','head_replayed','optimizer_run','new_alpha_search','held_out_test_opened','calibration_claimed','conformal_coverage_inherited'):require(report[flag] is False,'133 scope differs')
 require(report['train_rows_scored']==0 and report['unary_point_labels_and_ties_preserved'] is True,'133 population/invariant differs')
 require(producer['registered'] is True and producer['authorization']=='ROOT_REGISTERED_FROZEN_SCALE_EVALUATION_ONLY' and producer['frozen_alpha']==ALPHAS,'133 registration differs')
 require(np.__version__==plan['numpy_version']==producer['numpy_version'],'NumPy pin differs')
 require(set(report['artifacts'])==set(plan['outputs']) and len(plan['outputs'])==37,'Complete37 output bindings required')
 for name,binding in plan['outputs'].items():require(Path(binding['path']).resolve()==root.resolve()/name and binding['sha256']==report['artifacts'][name],'Output path/role binding differs')
 for key,b in producer['files'].items():require(plan['files']['input:'+key]==b,'Consumed133 input binding differs')
 def barrier():
  verify(plan['files']);verify(plan['outputs'])
  require({str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}==set(plan['outputs'])|{'report.json'},'133 inventory/failure marker differs')
  require(sha(path)==expected and sha(__file__)==plan['source_sha256'],'Audit plan/source changed')
 barrier()
 require(sha(root/'source.py')==SOURCE133 and canonical(read(root/'registered-plan.json'))==canonical(producer),'Retained133 source/plan differs')
 oracle_raw=f('oracle129').read_bytes();require(hashlib.sha256(oracle_raw).hexdigest()==ORACLE129,'Independent oracle changed')
 oracle=types.ModuleType('saved133_oracle129');oracle.__file__=str(f('oracle129'));exec(compile(oracle_raw,str(f('oracle129')),'exec'),oracle.__dict__)
 from roguard.gemma4_review import _score
 oracle.check_origins(producer);barrier()
 keypath=lambda key:Path(producer['files'][key]['path']);data={}
 for lang in ('ro','uk'):
  old=rows(keypath(lang+'_original_rows'));bank=rows(keypath(lang+'_bank_rows'))
  require(Counter(r['partition'] for r in old)==dict(train=TRAIN[lang],dev=DEV[lang]) and Counter(r['partition'] for r in bank)==BANK,'OriginalDEV/bank population differs')
  require(len({r['case_id'] for r in old})==len(old) and len({r['case_id'] for r in bank})==len(bank),'Duplicate source row')
  for row in old+bank:
   require(row['language']==lang and type(row['gold']) is int and row['gold'] in (0,1) and len(row['fields'])==6 and all(type(v) is int and 0<=v<n for v,n in zip(row['fields'],oracle.WIDTHS)),'Original typed labels differ')
  for row in bank:require(canonical(row['fields'])==canonical(row['proposed_fields']) and canonical(row['gold'])==canonical(row['proposed_gold']),'Bank proposals differ')
  ix=[i for i,r in enumerate(old) if r['partition']=='dev'];dev=[old[i] for i in ix];warm=rows(keypath(lang+'_initial_dev_rows'));keys=('case_id','language','fields','gold','token_sha256')
  require(canonical([[r[k] for k in keys] for r in dev])==canonical([[r[k] for k in keys] for r in warm]),'OriginalDEV/warm75 order differs')
  oldz=decoded(producer['files'][lang+'_original_logits']);require(oldz.shape==(len(old),19) and oldz.dtype==np.float32 and np.isfinite(oldz).all(),'Original reference array differs')
  populations={'original_dev':(dev,decoded(producer['files'][lang+'_initial_dev_logits']),oldz[ix].copy()),'larger_bank':(bank,decoded(producer['files']['baseline122_artifact:'+lang+'-unchanged-initial75-logits.npy']),decoded(producer['files']['baseline122_artifact:'+lang+'-original-v9-logits.npy']))}
  for source,parent,v9 in populations.values():require(parent.shape==v9.shape==(len(source),19) and parent.dtype==v9.dtype==np.float32 and np.isfinite(parent).all() and np.isfinite(v9).all(),'Reference population array differs')
  data[lang]=(populations,read(keypath(lang+'_initial_spec')),read(keypath(lang+'_v9_spec')))
 token_hashes=[row['token_sha256'] for populations,_,_ in data.values() for source,_,_ in populations.values() for row in source]
 require(len(token_hashes)==len(set(token_hashes))==11917 and all(isinstance(h,str) and len(h)==64 for h in token_hashes),'Candidate token hashes are not all11917 distinct')
 token_order_sha=hashlib.sha256(canonical(token_hashes).encode()).hexdigest()
 panel=rows(keypath('full_panel'));mapping=read(keypath('panel_alias_map'));barrier();oracle.check_origins(producer)
 audit={};total=groups_checked=0
 for lang,(populations,spec,v9spec) in data.items():
  audit[lang]={}
  for population,(source,parent,v9) in populations.items():
   folder=root/lang/population;candidate,statuses=oracle.oracle_transform(parent,ALPHAS[lang]);expected_arrays=dict(zip(REFS,(parent,v9,candidate)))
   for ref,wanted in expected_arrays.items():
    saved=decoded(plan['outputs'][str((folder/(ref+'-logits.npy')).relative_to(root))]);require(saved.dtype==np.float32 and saved.shape==wanted.shape and saved.tobytes()==wanted.tobytes(),'Retained reference/transform bytes differ')
   expected_order=[{key:r[key] for key in ('case_id','partition','fields','gold','token_sha256')} for r in source];require(canonical(read(folder/'row-order.json'))==canonical(expected_order),'Retained roworder differs')
   normalized,results=reconstruct_results(source,parent,v9,candidate,statuses,spec,v9spec,population,lang,oracle,_score)
   complete_groups={}
   for ref in REFS:
    retained=rows(folder/(ref+'-results.jsonl'));oracle.compare_values(retained,results[ref]);total+=len(retained)
    complete_groups[ref]=grouping(normalized,results[ref],population,oracle);groups_checked+=len(complete_groups[ref])
    oracle.compare_values(report['languages'][lang][population]['references'][ref],complete_groups[ref])
   comp=comparisons(complete_groups);oracle.compare_values(report['languages'][lang][population]['comparisons'],comp)
   require(complete_groups['frozen-scale75']['all:all']['unary_changes_vs_75']==complete_groups['frozen-scale75']['all:all']['tie_changes_vs_75']==0,'Unary predictions changed')
   audit[lang][population]=dict(rows=len(source),references=complete_groups,comparisons=comp,all_saved_rows_checked=True)
   if population=='larger_bank':
    panel_results,panel_summaries=panel_recount(source,results,panel,mapping,lang,oracle)
    for ref in REFS:oracle.compare_values(read(folder/(ref+'-full-panel.json')),panel_results[ref])
    oracle.compare_values(report['languages'][lang]['full_panel'],panel_summaries);audit[lang]['full_panel']=panel_summaries
 barrier();oracle.check_origins(producer);require(total==35751,'Complete row recount differs')
 require(shutil.disk_usage(out.parent).free>=8*1024**3,'Eight GiB reserve')
 own=Path(__file__).read_bytes();require(hashlib.sha256(own).hexdigest()==plan['source_sha256'],'Audit source changed');out.mkdir()
 (out/'source.py').write_bytes(own);(out/'registered-plan.json').write_bytes(raw)
 require(sha(out/'source.py')==plan['source_sha256'] and sha(out/'registered-plan.json')==expected,'Audit retained snapshots differ')
 final=dict(format='roguard-saved133-independent-audit-1',verification='PASS',source_sha256=plan['source_sha256'],audit_plan_sha256=expected,producer_report_sha256=plan['files']['producer_report']['sha256'],producer_plan_sha256=plan['files']['producer_plan']['sha256'],producer_source_sha256=SOURCE133,oracle129_sha256=ORACLE129,all_input_files=plan['files'],all_output_files=plan['outputs'],candidate_rows_recounted=11917,unique_candidate_token_hashes=len(set(token_hashes)),ordered_candidate_token_hashes_sha256=token_order_sha,reference_rows_recounted=23834,total_saved_rows_recounted=total,group_summaries_recounted=groups_checked,panel_responses_recounted=120,languages=audit,encoder_loaded=False,head_replayed=False,new_alpha_search=False,qualified=False,limits=['Independent129 vectorized transforms and independently implemented129 row outcomes/summary;133 grouping/comparison/panel code not imported.','Same frozen runtime policy primitive; no independent native/head regeneration or calibration/coverage claim.','Exact arrays, point labels/ties/statuses/policy outcomes/integercounts; float64loss/Brier only allow1e-10 absolute reduction tolerance.','All references/rows and all40mapped panelresponses per reference retained;32panelaliases+8supplements are not extra independent inputs.'])
 try:
  save(out/'report.json',final);final_sha=sha(out/'report.json');barrier();oracle.check_origins(producer)
  require(sha(out/'source.py')==plan['source_sha256'] and sha(out/'registered-plan.json')==expected and sha(out/'report.json')==final_sha,'Audit evidence changed after write')
 except BaseException as error:
  (out/'report.json').write_text(json.dumps(dict(verification='FAIL',qualified=False,error=str(error))));raise
 return final
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=run(a.plan,a.plan_sha256,a.output);print(json.dumps({k:r[k] for k in ('verification','candidate_rows_recounted','reference_rows_recounted','group_summaries_recounted','panel_responses_recounted')}))
