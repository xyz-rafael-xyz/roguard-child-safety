"""Prospective TRAIN-only bounded positive-scale experiment; no encoder/head replay."""
import argparse,hashlib,io,json,math,shutil,sys,time
from pathlib import Path
import numpy as np

FORMAT='roguard-train-positive-scale-1'
ALPHAS=(1.,1.125,1.25,1.5,2.,3.,4.)
COUNTS={'ro':13174,'uk':15755}
CUTS=(0,3,6,10,13,15,17,19)
FIELDS=('subject','time','assertion','source','relation','concern')
ACTIVE=(0,2,3,4,5)
DISK_RESERVE=8*1024**3
MAX_OUTPUT=1024**3
MAX_LINE=4096
WEIGHTS={'ro':'1c18db5e5fc26baf4c0739d053c0546338201d76fcb9061e47477c14650535e3','uk':'466a1900655c85a6db809adb384d90fc634357a4f7b72f1415274c4787b3ecb7'}
SPECS={'ro':'5fdeae97843d873755ed41a02453b5c29823fd8028508c3e99d1586175c8f4a9','uk':'bbf3fea8ee298c40d86701851fe045352ea1a924ba5287dd52b70e59ce25de27'}
THRESHOLDS={'ro':.7517333793124337,'uk':.9274248149759536}

def require(ok,message):
 if not ok:raise ValueError(message)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(v) for v in Path(p).read_text().splitlines()]
def save(p,v):
 with Path(p).open('x') as h:json.dump(v,h,ensure_ascii=False,indent=2,allow_nan=False);h.write('\n')
 require(digest(read(p))==digest(v),'JSON readback differs')
def resource_check(output,*,initial=False):
 output=Path(output);require(shutil.disk_usage(output.parent).free>=DISK_RESERVE+(MAX_OUTPUT if initial else 0),'Eight GiB disk reserve')
 if output.exists():require(sum(p.stat().st_size for p in output.rglob('*') if p.is_file())<=MAX_OUTPUT,'Bounded output storage exceeded')

def checks(plan,plan_path,plan_sha):
 require(sha(plan_path)==plan_sha and sha(__file__)==plan['source_sha256'],'Registered plan/source changed')
 for k,b in plan['files'].items():require(sha(b['path'])==b['sha256'],'Bound input changed: '+k)
def origins(plan):
 root=Path(plan['project_root']).resolve();bound={Path(b['path']).resolve() for b in plan['files'].values()}
 for name,module in list(sys.modules.items()):
  if name.split('.')[0] not in ('roguard','training','eval','scripts') or not getattr(module,'__file__',None):continue
  actual=Path(module.__file__).resolve();base=root/'src' if name.split('.')[0]=='roguard' else root;expected=base.joinpath(*name.split('.'))
  require(actual in (expected.with_suffix('.py').resolve(),(expected/'__init__.py').resolve()) and actual in bound,'Imported source origin differs: '+name)

def argmax7(z):return [int(np.argmax(z[CUTS[j]:CUTS[j+1]])) for j in range(7)]
def tie_sets(z):return [np.flatnonzero(z[CUTS[j]:CUTS[j+1]]==np.max(z[CUTS[j]:CUTS[j+1]])).tolist() for j in range(7)]
def scale_logits(parent,alpha):
 require(type(alpha) in (int,float) and math.isfinite(alpha) and alpha in ALPHAS,'Unregistered bounded scale')
 z=np.asarray(parent);require(z.shape==(19,) and z.dtype==np.float32 and np.isfinite(z).all(),'Invalid parent float32 logits')
 if alpha==1.:return z.copy(),'identity'
 out=z.copy()
 with np.errstate(over='ignore',under='ignore',invalid='ignore'):
  for j in ACTIVE:
   a,b=CUTS[j:j+2];x=z[a:b].astype(np.float64);out[a:b]=((x-x.max())*alpha).astype(np.float32)
 if not np.isfinite(out).all():return z.copy(),'fallback_nonfinite'
 if argmax7(out)!=argmax7(z) or tie_sets(out)!=tie_sets(z):return z.copy(),'fallback_argmax_or_tie'
 require(out[3:6].tobytes()==z[3:6].tobytes() and out[17:19].tobytes()==z[17:19].tobytes(),'Time or legacy changed')
 return out,'scaled'

def coupled_map(z,spec):
 # Independent pair maxima plus unchanged time and concern; no432-world enumeration.
 scope=z[:3,None].astype(np.float64)+z[None,10:13].astype(np.float64)+np.log(np.asarray(spec['couplings']['scope']))
 ar=z[6:10,None].astype(np.float64)+z[None,13:15].astype(np.float64)+np.log(np.asarray(spec['couplings']['assertion_relation']))
 s,q=np.unravel_index(scope.argmax(),scope.shape);a,r=np.unravel_index(ar.argmax(),ar.shape)
 return [int(s),int(np.argmax(z[3:6])),int(a),int(q),int(r),int(np.argmax(z[15:17]))]

def row_result(parent,candidate,row,spec,scorer,status,alpha):
 gold=row['gold'];fields=row['fields'];before=float(scorer(parent,spec));after=float(scorer(candidate,spec))
 require(math.isfinite(before) and math.isfinite(after) and 0<=before<=1 and 0<=after<=1,'Invalid fixed policy score')
 pred=argmax7(candidate);require(pred==argmax7(parent),'Unary labels changed')
 bd=int(before>=spec['threshold']);ad=int(after>=spec['threshold']);be=bd!=gold;ae=ad!=gold
 nll=[];brier=[]
 for j in range(6):
  x=candidate[CUTS[j]:CUTS[j+1]].astype(np.float64);x=x-x.max();lse=math.log(float(np.exp(x).sum()));p=np.exp(x-lse);y=np.eye(len(x))[fields[j]]
  nll.append(float(lse-x[fields[j]]));brier.append(float(np.square(p-y).sum()))
 truth=after if gold else 1-after
 return dict(case_id=row['case_id'],language=row['language'],group=row['group'],gold=gold,fields=fields,
  alpha=alpha,status=status,parent_score=before,score=after,parent_decision=bd,decision=ad,
  parent_margin=before-spec['threshold'],margin=after-spec['threshold'],parent_error=be,error=ae,
  introduced=bool(ae and not be),fixed=bool(be and not ae),field_errors=[int(v!=f) for v,f in zip(pred[:6],fields)],
  joint_error=pred[:6]!=fields,argmax7=pred,tie_sets=tie_sets(candidate),
  coupled_map_changed=coupled_map(parent,spec)!=coupled_map(candidate,spec),
  field_nll=nll,field_brier=brier,binary_brier=(after-gold)**2,
  binary_nll=-math.log(max(truth,np.finfo(np.float64).tiny)),binary_true_probability_zero=truth==0.,
  parent_within_diagnostic_band=abs(before-spec['threshold'])<=1e-3)

def summarize(records):
 require(records,'Empty candidate population');n=len(records)
 return dict(rows=n,binary_errors=sum(r['error'] for r in records),parent_binary_errors=sum(r['parent_error'] for r in records),
  introduced=sum(r['introduced'] for r in records),fixed=sum(r['fixed'] for r in records),
  tp=sum(r['decision']==1 and r['gold']==1 for r in records),tn=sum(r['decision']==0 and r['gold']==0 for r in records),
  fp=sum(r['decision']==1 and r['gold']==0 for r in records),fn=sum(r['decision']==0 and r['gold']==1 for r in records),
  joint_errors=sum(r['joint_error'] for r in records),field_errors=np.sum([r['field_errors'] for r in records],axis=0).tolist(),
  fallbacks=sum(r['status'].startswith('fallback_') for r in records),scaled=sum(r['status']=='scaled' for r in records),
  coupled_map_changes=sum(r['coupled_map_changed'] for r in records),
  field_nll=np.mean([r['field_nll'] for r in records],axis=0).tolist(),field_brier=np.mean([r['field_brier'] for r in records],axis=0).tolist(),
  binary_nll=float(np.mean([r['binary_nll'] for r in records])),binary_brier=float(np.mean([r['binary_brier'] for r in records])),
  binary_true_probability_zeros=sum(r['binary_true_probability_zero'] for r in records),parent_within_diagnostic_band=sum(r['parent_within_diagnostic_band'] for r in records))

def grouped(records):
 g={'all':records}
 for row in records:
  for key in ('source_group:'+row['group'],'gold:'+str(row['gold']),'assertion:'+str(row['fields'][2]),'status:'+row['status']):g.setdefault(key,[]).append(row)
 return {k:summarize(v) for k,v in sorted(g.items())}

def choose(candidates):
 require([r['alpha'] for r in candidates]==list(ALPHAS),'Complete fixed candidate inventory required')
 eligible=[r for r in candidates if r['groups']['all']['introduced']==0]
 require(eligible and candidates[0] in eligible,'Identity baseline must be eligible')
 chosen=min(eligible,key=lambda r:(r['groups']['all']['binary_errors'],abs(r['alpha']-1.),r['alpha']))
 if chosen['groups']['all']['binary_errors']>=candidates[0]['groups']['all']['binary_errors']:chosen=candidates[0]
 return dict(alpha=chosen['alpha'],binary_errors=chosen['groups']['all']['binary_errors'],improved=chosen is not candidates[0],
  rule='Minimize allTRAIN binary errors among zero-introduced candidates; then closest to1 and numeric alpha. Identity if no strict gain.',candidate_artifacts=chosen['artifacts'])

def validate_rows(original,retained,language):
 require(all(r['partition'] in ('train','dev') for r in original),'Unexpected original partition')
 expected=[(i,r) for i,r in enumerate(original) if r['partition']=='train']
 require(len(expected)==len(retained)==COUNTS[language],'Exact complete TRAIN membership required')
 seen=set()
 for j,((index,row),record) in enumerate(zip(expected,retained)):
  require(type(record['train_index']) is int and record['train_index']==j and type(record['source_index']) is int and record['source_index']==index and digest(record['row'])==digest(row),'TRAIN order or row identity changed')
  require(row['language']==language and row['case_id'] not in seen and row['partition']=='train','TRAIN identity differs');seen.add(row['case_id'])
  require(len(row['fields'])==6 and all(type(v) is int and 0<=v<n for v,n in zip(row['fields'],(3,3,4,3,2,2))) and type(row['gold']) is int and row['gold'] in (0,1),'Invalid TRAIN labels')
 return [r for _,r in expected]

def preflight(plan_path,expected):
 require(sha(plan_path)==expected,'Plan mismatch');plan=read(plan_path)
 require(plan['format']==FORMAT and plan['registered'] is True and plan['authorization']=='TRAIN_ONLY_BOUNDED_SCALING' and plan['held_out_test_opened'] is False,'Not registered')
 require(digest(plan['alphas'])==digest(list(ALPHAS)) and digest(plan['active_field_indices'])==digest(list(ACTIVE)),'Candidate family changed')
 require(plan['disk_reserve_bytes']==DISK_RESERVE and plan['maximum_output_bytes']==MAX_OUTPUT and plan['maximum_result_line_bytes']==MAX_LINE,'Resource contract changed')
 require(np.__version__==plan['numpy_version'],'Numerical NumPy version differs')
 checks(plan,plan_path,expected);path=lambda k:Path(plan['files'][k]['path'])
 report=read(path('report117'));audit=read(path('audit118'));oldplan=read(path('plan117'))
 require(report['complete'] is True and report['qualified'] is False and audit['verification']=='PASS' and audit['qualified'] is False,'Incomplete retained TRAIN evidence')
 for key,binding in [('producer_report_sha256','report117'),('producer_source_sha256','source117'),('producer_plan_sha256','plan117')]:require(audit[key]==sha(path(binding)),'118 producer lineage changed')
 require(audit['source_sha256']==sha(path('source118')) and audit['audit_plan_sha256']==sha(path('plan118')),'118 source/plan binding differs')
 require(report['source_sha256']==sha(path('source117')) and report['plan_sha256']==sha(path('plan117')),'117 producer binding differs')
 data={}
 for lang,n in COUNTS.items():
  for name in ('rows.jsonl','parent-logits.npy'):
   filename=lang+'-'+name;k=lang+('_rows117' if name=='rows.jsonl' else '_logits117');h=sha(path(k))
   require(report['languages'][lang]['artifacts'][filename]==h and audit['output_files']['output:'+filename]['sha256']==h,'117/118 retained artifact differs')
  require(plan['files'][lang+'_weights75']['sha256']==WEIGHTS[lang] and plan['files'][lang+'_spec75']['sha256']==SPECS[lang],'Only unchanged original75 allowed')
  for key in ('rows','weights75','spec75'):require(oldplan['files'][lang+'_'+key]['sha256']==plan['files'][lang+('_original_rows' if key=='rows' else '_'+key)]['sha256'],'117 input lineage differs')
  selected=validate_rows(rows(path(lang+'_original_rows')),rows(path(lang+'_rows117')),lang)
  order=digest([[record['source_index'],record['row']] for record in rows(path(lang+'_rows117'))])
  require(order==report['languages'][lang]['train_order_sha256']==audit['languages'][lang]['train_order_sha256'] and audit['languages'][lang]['rows_checked']==n,'118 TRAIN order/replay population differs')
  require(audit['languages'][lang]['weights75_sha256']==WEIGHTS[lang] and audit['languages'][lang]['spec75_sha256']==SPECS[lang],'118 fixed75 lineage differs')
  spec=read(path(lang+'_spec75'));require(spec['threshold']==THRESHOLDS[lang] and spec['language']==lang and spec['mode']=='coupled_policy' and spec['adapter_sha256']==WEIGHTS[lang],'Fixed75 specification differs')
  raw=path(lang+'_logits117').read_bytes();require(hashlib.sha256(raw).hexdigest()==plan['files'][lang+'_logits117']['sha256'],'117 logits changed before decode')
  z=np.load(io.BytesIO(raw),allow_pickle=False);require(z.shape==(n,19) and z.dtype==np.float32 and np.isfinite(z).all(),'Invalid TRAIN logits')
  data[lang]=(selected,z,spec)
 checks(plan,plan_path,expected)
 return plan,data

def run(plan_path,expected,output):
 output=Path(output);require(not output.exists(),'Refusing overwrite')
 plan,data=preflight(plan_path,expected)
 from roguard.gemma4_review import _score,FIELDS as serving_fields
 require(tuple(serving_fields)==FIELDS,'Actual serving ontology order differs')
 origins(plan);checks(plan,plan_path,expected)
 resource_check(output,initial=True);output.mkdir()
 source_bytes=Path(__file__).read_bytes();require(hashlib.sha256(source_bytes).hexdigest()==plan['source_sha256'],'Source changed before retained copy')
 save(output/'registered-plan.json',plan);(output/'source.py').write_bytes(source_bytes)
 require(sha(output/'source.py')==plan['source_sha256'],'Retained source copy differs')
 report=dict(format=FORMAT,field_order=list(FIELDS),complete=False,plan_sha256=expected,source_sha256=sha(__file__),languages={},encoder_loaded=False,head_replayed=False,optimizer_run=False,dev_rows_scored=0,new_probe_rows_scored=0,qualified=False,limitations=['TRAIN parameter selection only, not learned semantic correction or native75 parity.','Unary labels remain unchanged, including all nine preexisting wrong TRAIN tuples.','No conformal coverage/calibration inheritance; separate frozen evaluation and recalibration required.'])
 snapshot={output/'registered-plan.json':sha(output/'registered-plan.json'),output/'source.py':sha(output/'source.py')}
 try:
  for lang,(selected,parent,spec) in data.items():
   directory=output/lang;directory.mkdir();save(directory/'row-order.json',selected);snapshot[directory/'row-order.json']=sha(directory/'row-order.json');candidates=[]
   for alpha in ALPHAS:
    resource_check(output)
    target=directory/('alpha-'+str(alpha));target.mkdir();z=np.empty_like(parent);records=[]
    for i,(row,original) in enumerate(zip(selected,parent)):
     z[i],status=scale_logits(original,alpha);record=row_result(original,z[i],row,spec,_score,status,alpha);record['train_index']=i;records.append(record)
    np.save(target/'logits.npy',z,allow_pickle=False)
    saved=np.load(target/'logits.npy',allow_pickle=False);require(saved.dtype==z.dtype and saved.shape==z.shape and saved.tobytes()==z.tobytes(),'Candidate NPY readback differs')
    with (target/'results.jsonl').open('x') as h:
     for record in records:
      line=json.dumps(record,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n'
      require(len(line.encode('utf-8'))<=MAX_LINE,'Result line exceeds storage bound');h.write(line)
    retained=rows(target/'results.jsonl');require(digest(retained)==digest(records),'Candidate result readback differs')
    # Recompute outcomes from every saved row, not producer summary memory.
    for i,(row,original) in enumerate(zip(selected,parent)):
     rr=row_result(original,saved[i],row,spec,_score,retained[i]['status'],alpha);rr['train_index']=i
     require(digest(rr)==digest(retained[i]),'Saved outcome replay differs')
    resource_check(output)
    summary=grouped(retained);artifacts={name:sha(target/name) for name in ('logits.npy','results.jsonl')}
    for name,h in artifacts.items():snapshot[target/name]=h
    candidates.append(dict(alpha=alpha,groups=summary,artifacts={str((target/name).relative_to(output)):h for name,h in artifacts.items()}))
   report['languages'][lang]=dict(candidates=candidates,selected=choose(candidates))
  save(output/'selection-before-dev.json',{l:r['selected'] for l,r in report['languages'].items()});snapshot[output/'selection-before-dev.json']=sha(output/'selection-before-dev.json')
  resource_check(output)
  checks(plan,plan_path,expected);origins(plan);require(all(sha(p)==h for p,h in snapshot.items()),'Retained artifact changed')
  report.update(complete=True,artifacts={str(p.relative_to(output)):h for p,h in snapshot.items()},decision='TRAIN_SCALARS_FROZEN_REQUIRE_INDEPENDENT_AUDIT_BEFORE_DEV')
  save(output/'report.json',report);resource_check(output);require(all(sha(p)==h for p,h in snapshot.items()),'Final artifact changed')
  return report
 except BaseException as error:
  report.update(complete=False,error_type=type(error).__name__,error=str(error));save(output/'failure.json',report);raise

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.plan,a.plan_sha256,a.output)
