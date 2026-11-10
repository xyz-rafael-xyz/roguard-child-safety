"""Independent saved-array and CPU75-head audit; no encoder or gold labels."""
import argparse,hashlib,importlib.util,json,math
from pathlib import Path
from importlib.metadata import version
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
def require(v,m):
 if not v:raise ValueError(m)
def finite(v):
 require(type(v) in (int,float) and math.isfinite(v) and v>=0,'Invalid nonnegative measurement')
 return v
def module(p):
 s=importlib.util.spec_from_file_location('bound130_'+Path(p).stem,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def run(path,expected,out):
 require(sha(path)==expected,'Audit plan changed');plan=read(path)
 require(plan['registered'] is True and plan['study']==130 and plan['encoder_permitted'] is False,'Unregistered audit')
 require(sha(__file__)==plan['source_sha256'],'Auditor changed')
 source_raw=Path(__file__).read_bytes();plan_raw=Path(path).read_bytes()
 require(hashlib.sha256(source_raw).hexdigest()==plan['source_sha256'] and hashlib.sha256(plan_raw).hexdigest()==expected,'Initial source/plan snapshot changed')
 def check():
  require(sha(path)==expected and sha(__file__)==plan['source_sha256'],'Audit identity changed')
  for k,b in plan['files'].items():require(sha(b['path'])==b['sha256'],'Audit input changed: '+k)
 check()
 for k,v in plan['versions'].items():require(version(k)==v,'Numerical version changed')
 f=lambda k:Path(plan['files'][k]['path'])
 producer=module(f('producer_source'))
 parent,paths,inventory,records=producer.preflight(f('producer_plan'),plan['files']['producer_plan']['sha256'])
 report=read(f('producer_report'));root=f('producer_report').parent
 require(report['complete'] is True and report['qualified'] is False and report['training_performed'] is False,'Producer incomplete')
 require(report['source_sha256']==sha(f('producer_source')) and report['plan_sha256']==sha(f('producer_plan')),'Producer lineage changed')
 expected_names={'source.py','registered-plan.json','inventory.jsonl','alignment-records.jsonl','vectors.npy','metadata.npy','logits.npy'}|{r['case_id']+'-observation.json' for r in inventory}
 require(set(report['artifacts'])==expected_names and {p.name for p in root.iterdir()}==expected_names|{'report.json'},'Artifact inventory differs')
 snapshot={n:sha(root/n) for n in expected_names};require(snapshot==report['artifacts'],'Artifact bytes changed')
 def barrier():
  check();producer.bindings(parent);producer.origins(parent)
  require({n:sha(root/n) for n in expected_names}==snapshot,'Saved artifact changed')
  require({p.name for p in root.iterdir()}==expected_names|{'report.json'},'Saved artifact inventory changed')
 require(report['accuracy_evaluation_performed'] is False and report['full_token_states_saved'] is False and report['learned_residual_implemented'] is False and report['native_final_interchangeability_claim'] is False,'Producer scope flags differ')
 require(report['vector_names']==producer.VECTOR_NAMES and report['logit_names']==producer.LOGIT_NAMES,'Producer vector/logit schema differs')
 require(sum((root/n).stat().st_size for n in expected_names)==report['non_report_artifact_bytes'],'Byte accounting differs')
 require(sum(p.stat().st_size for p in root.iterdir())<=64*1024**2,'Output cap exceeded')
 for name,bound in [('source.py',f('producer_source')),('registered-plan.json',f('producer_plan')),('inventory.jsonl',paths['inventory']),('alignment-records.jsonl',paths['alignments'])]:require((root/name).read_bytes()==bound.read_bytes(),'Snapshot differs')
 import numpy as np
 import torch
 from safetensors.numpy import load_file
 from roguard.gemma4_review import _score
 producer.origins(parent)
 arrays={}
 for name,shape in [('vectors.npy',(16,6,2560)),('metadata.npy',(16,6)),('logits.npy',(16,3,19))]:
  a=np.load(root/name,allow_pickle=False);require(a.shape==shape and a.dtype==np.float32 and np.isfinite(a).all(),'Saved array contract differs');arrays[name]=a
 alignment=module(paths['alignment_source']);tokens_checked=0
 barrier()
 with alignment.pinned_tokenizer(parent['base_snapshot']) as (tokenizer,provenance):
  for row,record in zip(inventory,records):
   actual=producer.enrich(alignment.align(row['payload'],tokenizer,row.get('focus')),row)
   require(canonical(actual)==canonical(record),'Retokenization differs');tokens_checked+=1
 cuts=[0,3,6,10,13,15,17,19]
 classes=lambda z:[int(z[a:b].argmax()) for a,b in zip(cuts[:-1],cuts[1:])]
 results={};cpu_outputs=np.empty((16,19),dtype=np.float32)
 barrier()
 for lang in ['ro','uk']:
  info=report['languages'][lang]
  require(info['complete'] is True and type(info['checked']) is int and info['checked']==8 and not info['cleanup_failures'],'Incomplete language')
  for key in ['load_seconds','head_install_seconds','prefix_seconds']:finite(info[key])
  sampling=info['disk_sampling'];samples=sampling['samples'];require(isinstance(samples,list) and samples,'Missing disk samples')
  for sample in samples:
   finite(sample['seconds'])
   for key in ['output_free','temp_free']:require(type(sample[key]) is int and sample[key]>=8*1024**3,'Disk sample reserve failed')
  require(all(a['seconds']<=b['seconds'] for a,b in zip(samples,samples[1:])),'Disk sample order differs')
  require(min(v['output_free'] for v in samples)==sampling['minimum_output_free'] and min(v['temp_free'] for v in samples)==sampling['minimum_temp_free'],'Disk minima recount differs')
  weights=load_file(str(paths[lang+'_parent_weights']));spec=read(paths[lang+'_spec']);maxlogit=0.;maxscore=0.;maskdiff=0;nativechanges=0;primary_times=[]
  selected=[i for i,r in enumerate(inventory) if r['payload']['language']==lang]
  require(len(info['observations'])==8,'Missing observations')
  for ordinal,i in enumerate(selected):
   row,record=inventory[i],records[i];obs=read(root/(row['case_id']+'-observation.json'))
   require(canonical(obs)==canonical(info['observations'][ordinal]),'Observation snapshot differs')
   require(obs['case_id']==row['case_id'] and obs['language']==lang and obs['token_sha256']==record['classifier_token_sha256'],'Observation identity differs')
   require(obs['vector_names']==producer.VECTOR_NAMES and obs['logit_names']==producer.LOGIT_NAMES,'Observation schema differs')
   require(obs['native_states_released_before_qualified'] is True,'Native state release observation differs')
   for key in ['qualified_seven_argmax_differences','qualified_binary_differences']:require(type(obs[key]) is int and obs[key]==0,'Qualified parity observation differs')
   for key in ['native_forward_seconds','pooling_and_cpu_audit_seconds','qualified_baseline_seconds','primary_two_path_seconds','extra_native_diagnostic_head_seconds','extra_qualified_repeat_seconds','extra_runtime_reference_seconds','seconds']:finite(obs[key])
   require(set(obs['pool_phase_times'])=={'native_pool_seconds','cpu_state_copy_seconds','cpu_float64_mean_seconds'},'Pool timing inventory differs')
   for value in obs['pool_phase_times'].values():finite(value)
   roles=record['role_pool_positions'];prior=sorted(roles[1]+roles[2]);n=len(roles[1]);pools=roles+[prior[:n],prior[n:]]
   require(obs['global_positions']==pools and obs['suffix_positions']==[[p-387 for p in pool] for pool in pools],'Pool coordinates differ')
   require(obs['pool_counts']==list(map(len,pools)) and obs['pool_validity']==[bool(p) for p in pools],'Pool counts differ')
   meta=np.array([len(p)/1024 for p in roles]+list(map(float,record['shared_structural_role_presence'])),dtype=np.float32)
   require(arrays['metadata.npy'][i].tobytes()==meta.tobytes() and obs['shared_metadata']==meta.tolist(),'Shared metadata differs')
   for j,pool in enumerate(pools):
    if not pool:require(not np.any(arrays['vectors.npy'][i,j+1]),'Nonzero empty pool')
   same=roles[1:]==pools[3:];require(obs['chronological_masks_identical'] is same,'Chronological identity differs');maskdiff+=not same
   require(len(obs['pool_mean_max_errors'])==5,'Five pool error observations required')
   for v in obs['pool_mean_max_errors']:require(type(v) in (int,float) and math.isfinite(v) and 0<=v<=1e-5,'Pool error observation exceeds bound')
   z,repeat,native=arrays['logits.npy'][i]
   require(z.tobytes()==repeat.tobytes() and obs['qualified_repeat_logit_error']==0 and obs['qualified_repeat_feature_error']==0,'Qualified repeat differs')
   require(obs['zero_residual_same_object'] is True and obs['zero_residual_bytes_identical'] is True,'Identity observation differs')
   with torch.no_grad():
    x=torch.from_numpy(arrays['vectors.npy'][i,0].copy()).reshape(1,2560);norm=(x*x).sum(1,keepdim=True).sqrt()
    require(bool(torch.isfinite(norm).all()) and bool((norm>0).all()),'Invalid final-vector norm')
    x=x/norm*math.sqrt(2560)
    w=lambda name:torch.from_numpy(weights['heads.'+lang+'.'+name])
    h=torch.nn.functional.linear(x,w('first.weight'),w('first.bias'));h=h*(1+torch.erf(h/math.sqrt(2)))/2
    cpu=torch.nn.functional.linear(h,w('last.weight'),w('last.bias')).numpy()[0].copy()
   error=float(np.max(np.abs(cpu.astype(float)-z.astype(float))));require(error<=1e-3 and classes(cpu)==classes(z),'Independent CPU75 head differs')
   score=float(_score(z,spec));cscore=float(_score(cpu,spec));native_score=float(_score(native,spec))
   require(score==obs['score'] and abs(score-obs['served_score'])<=1e-10,'Retained qualified score differs')
   require(abs(score-obs['served_score'])==obs['score_parity_error'],'Score error observation differs')
   require((cscore>=spec['threshold'])==(score>=spec['threshold'])==(obs['served_score']>=spec['threshold']),'Independent CPU decision differs')
   diagnostic=obs['native_final_diagnostic'];finite(diagnostic['feature_max_abs'])
   require(float(np.max(np.abs(native.astype(np.float64)-z.astype(np.float64))))==diagnostic['logit_max_abs'],'Native logit drift diagnostic differs')
   require(native_score==diagnostic['score'] and native_score-score==diagnostic['score_difference'],'Native score diagnostic differs')
   changes=sum(a!=b for a,b in zip(classes(native),classes(z)));require(changes==diagnostic['seven_argmax_differences'],'Native class diagnostic differs');nativechanges+=changes
   require(int((native_score>=spec['threshold'])!=(score>=spec['threshold']))==diagnostic['binary_difference'],'Native decision diagnostic differs')
   require(diagnostic['interchangeability_claim'] is False,'Native interchangeability claimed')
   for k in ['process_peak_rss_bytes','process_current_rss_bytes']:require(type(obs[k]) is int and 0<obs[k]<=12*1024**3,'RSS observation invalid')
   require(obs['disk_free']>=8*1024**3,'Observed disk reserve failed')
   primary=obs['native_forward_seconds']+obs['pool_phase_times']['native_pool_seconds']+obs['qualified_baseline_seconds']
   require(primary==obs['primary_two_path_seconds'] and 0<=primary<=obs['seconds'],'Primary timing sum differs')
   primary_times.append(primary);maxlogit=max(maxlogit,error);maxscore=max(maxscore,abs(cscore-score));cpu_outputs[i]=cpu
  require(maskdiff==3,'Expected count-matched structural split differs')
  require(info['load_seconds']+info['head_install_seconds']+info['prefix_seconds']+sum(o['seconds'] for o in info['observations'])<=900,'Recorded lower-bound time exceeds budget')
  results[lang]=dict(rows=8,maximum_cpu75_logit_error=maxlogit,maximum_cpu75_score_difference=maxscore,seven_argmax_differences=0,cpu_binary_decision_differences=0,chronological_masks_different=maskdiff,native_diagnostic_argmax_changes=nativechanges,primary_two_path_seconds=primary_times)
 check();producer.bindings(parent);require({n:sha(root/n) for n in expected_names}==snapshot,'Artifacts changed during audit')
 out=Path(out);require(not out.exists(),'Refusing overwrite');out.mkdir()
 (out/'source.py').write_bytes(source_raw);(out/'registered-plan.json').write_bytes(plan_raw)
 require(sha(out/'source.py')==plan['source_sha256'] and sha(out/'registered-plan.json')==expected,'Retained audit snapshot differs')
 np.save(out/'cpu75-logits.npy',cpu_outputs,allow_pickle=False);saved_cpu=np.load(out/'cpu75-logits.npy',allow_pickle=False)
 require(saved_cpu.dtype==np.float32 and saved_cpu.shape==(16,19) and np.isfinite(saved_cpu).all() and saved_cpu.tobytes()==cpu_outputs.tobytes(),'Audit CPU output readback differs')
 result=dict(verification='PASS',study=130,source_sha256=sha(__file__),plan_sha256=expected,producer_report_sha256=sha(f('producer_report')),token_rows_replayed=tokens_checked,head_rows_replayed=16,languages=results,encoder_loaded=False,training_performed=False,qualified=False,limitations=['Independent tokenizer and CPU75 head replay on retained qualified-final vectors; no native encoder replay.','Full states/reference features were not retained: pooling/repeated-feature/resource observations are recounted, not independently measured.','Only a lower bound on per-language elapsed time can be reconstructed; timings are producer observations.','Zero-residual same-object and installation/backbone proofs are producer observations; audit confirms saved outputs and pinned75 head behavior, not learned residual correctness.'])
 result['artifacts']={n:sha(out/n) for n in ['source.py','registered-plan.json','cpu75-logits.npy']}
 try:
  barrier()
  (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');require(canonical(read(out/'report.json'))==canonical(result),'Audit report readback differs')
  final_sha=sha(out/'report.json');barrier()
  require({n:sha(out/n) for n in result['artifacts']}==result['artifacts'] and sha(out/'report.json')==final_sha,'Closing audit artifacts changed')
 except BaseException as error:
  failure=dict(verification='FAIL',qualified=False,reason=str(error),plan_sha256=expected)
  (out/'report.json').write_text(json.dumps(failure,indent=2)+'\n');raise
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.plan,a.plan_sha256,a.output)))
