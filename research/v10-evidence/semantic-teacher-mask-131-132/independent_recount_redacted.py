from pathlib import Path
from collections import defaultdict
import hashlib,json
import numpy as np
from roguard.gemma4_review import _score
P=Path('[local-path-omitted:6832ad4dc34b]');D=P/'assertion-pilot-next';E=P/'semantic-mask-evaluation-132';F=P/'semantic-fit-mask-study-131';O=Path(__file__).parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
canon=lambda v:hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
tracked={}
def bind(p,digest=None):
 p=Path(p);s=sha(p);assert digest is None or digest==s;tracked[str(p)]=s;return p
def load(p,digest=None):return read(bind(p,digest))
def array(p,digest=None):return np.load(bind(p,digest),allow_pickle=False)
r=load(E/'report.json');plan=load(D/'semantic-fit-mask131-plan.json');ep=load(D/'semantic-mask-evaluation132-plan.json');fr=load(F/'report.json',r['fit_report_sha256']);selection=load(F/'selected-before-probes.json',r['selection_sha256'])
assert sha(D/'semantic-mask-evaluation132-plan.json')==r['evaluation_plan_sha256'] and r['verification']=='PASS'
assert r['qualified'] is False and r['decision']=='REJECT_ESCALATION'
for folder in [E,F]:
 for f in folder.rglob('*'):
  if f.is_file():bind(f)
def path(k):
 v=plan['files'][k];return bind(v['path'],v['sha256'])
def infer(z,spec):
 assert z.ndim==2 and z.shape[1]==19 and z.dtype==np.float32 and np.isfinite(z).all()
 cuts=[0,3,6,10,13,15,17]
 return np.column_stack([z[:,a:b].argmax(1) for a,b in zip(cuts,cuts[1:])]),np.array([int(_score(x,spec)>=spec['threshold']) for x in z])
def measures(z,rows,spec,refs):
 y=np.array([v.get('fields',v.get('proposed_fields')) for v in rows]);g=np.array([v.get('gold',v.get('proposed_gold')) for v in rows]);p,b=infer(z,spec);w=p!=y;bw=b!=g;ce=[]
 for j,(a,c) in enumerate(zip([0,3,6,10,13,15],[3,6,10,13,15,17])):
  q=z[:,a:c].astype(np.float64);q-=q.max(1,keepdims=True);q=np.exp(q);q/=q.sum(1,keepdims=True);ce.append(-np.log(np.maximum(q[np.arange(len(y)),y[:,j]],1e-30)))
 ce=np.array(ce).T
 def metric(ids):
  ids=np.asarray(ids);e=w[ids];be=bw[ids];out=dict(rows=len(ids),binary_errors=int(be.sum()),joint_errors=int(e.any(1).sum()),assertion_errors=int(e[:,2].sum()),field_errors=e.sum(0).tolist(),mean_field_ce=float(ce[ids].mean(0).mean()),references={})
  for name,(rp,rb) in refs.items():
   rw=rp[ids]!=y[ids];rbe=rb[ids]!=g[ids]
   out['references'][name]=dict(introduced_binary=int((be&~rbe).sum()),fixed_binary=int((~be&rbe).sum()),introduced_joint=int((e.any(1)&~rw.any(1)).sum()),fixed_joint=int((~e.any(1)&rw.any(1)).sum()),introduced_fields=(e&~rw).sum(0).tolist(),fixed_fields=(~e&rw).sum(0).tolist())
  if refs:out.update(introduced_v9=out['references']['v9']['introduced_binary'],fixed_v9=out['references']['v9']['fixed_binary'],introduced_initial75=out['references']['initial75']['introduced_binary'])
  else:out.pop('references')
  return out
 return metric
checks=0
max_ce=0.
def compare(m,saved):
 global checks,max_ce
 for k,v in m.items():
  if k=='mean_field_ce':
   delta=abs(v-saved[k]);max_ce=max(max_ce,delta);assert delta<=1e-12
  else:assert canon(v)==canon(saved[k]),(k,v,saved[k])
 checks+=1
CONTROL='new-mask_wrong_assertion';PRIMARY='new-mask_wrong_new_fit_semantics';names=[CONTROL,PRIMARY,'initial75','v9','frozen125-primary','frozen125-full_control'];totals={};dev={};groups_count=0
for lang in ('ro','uk'):
 rr=r['results'][lang];rows=[json.loads(x) for x in path(lang+'_new_rows').read_text().splitlines()];order=load(E/lang/'row-order.json');assert len(rows)==1540
 assert [x['case_id'] for x in rows]==[x['case_id'] for x in order]
 assert [x['source_row_sha256'] for x in rows]==[x['source_row_sha256'] for x in order]
 panel=load(E/lang/'panel-order.json');pi=[x['canonical_language_index'] for x in panel];assert len(set(pi))==20
 for v in panel:assert rows[v['canonical_language_index']]['case_id']==v['canonical_case_id']
 assert sum(v['bank_alias'] for v in panel)==16
 specs={n:load(path(lang+('_v9_spec' if n=='v9' else '_initial_spec'))) for n in names}
 saved={CONTROL:rr['arms'][CONTROL],PRIMARY:rr['arms'][PRIMARY],**rr['new_baselines'],'frozen125-primary':rr['frozen125_references']['primary'],'frozen125-full_control':rr['frozen125_references']['full_control']}
 zs={n:array(E/lang/(n+'-all-new.npy'),saved[n]['logits_sha256']) for n in names};refs={n:infer(zs[n],specs[n]) for n in ('v9','initial75')}
 ix=defaultdict(list)
 for i,row in enumerate(rows):
  ix['partition:'+row['partition']].append(i)
  for key in ('operator_family','meaning_family','paired_meaning_id','context_variant','panel_pair_id','contrast_group','register','core_type','block_id'):
   if row.get(key):ix[key+':'+row[key]].append(i)
  ix['assertion:'+row['proposed_attributes']['assertion']].append(i)
  if row.get('operator_family') and row.get('meaning_family'):ix['block:'+row['operator_family']+'::'+row['meaning_family']].append(i)
 totals[lang]={}
 for n in names:
  m=measures(zs[n],rows,specs[n],refs);table={k:m(ids) for k,ids in sorted(ix.items())};assert set(table)==set(saved[n]['groups'])
  for k,v in table.items():compare(v,saved[n]['groups'][k]);groups_count+=1
  pm=m(pi);pm.update(bank_aliases=16,supplemental_rows=4);compare(pm,saved[n]['full_panel']);totals[lang][n]=dict(groups=table,full_panel=pm)
 # Independently recount selected oldDEV from saved logits (no head execution).
 old=[json.loads(x) for x in path(lang+'_original_rows').read_text().splitlines()];di=[i for i,x in enumerate(old) if x['partition']=='dev'];dr=[old[i] for i in di]
 dz9=array(path(lang+'_original_logits'))[di];dzi=array(path(lang+'_initial_dev_logits'));devrefs={'v9':infer(dz9,specs['v9']),'initial75':infer(dzi,specs['initial75'])}
 dev[lang]={}
 for n in [CONTROL,PRIMARY]:
  s=selection[lang][n];z=array(F/lang/n/('epoch-'+str(s['epoch'])+'-dev.npy'),s['dev_logits_sha256']);m=measures(z,dr,specs[n],devrefs)(range(len(dr)));compare(m,saved[n]['dev']);dev[lang][n]=m
 m=measures(dzi,dr,specs['initial75'],devrefs)(range(len(dr)));compare(m,rr['initial75_dev']);dev[lang]['initial75']=m
 # Frozen oldDEV selected arrays, preserving historical comparison values.
 oldsel=load(P/'balanced-head-study-125/selected-before-probes.json')
 for n,arm,k in [('frozen125-primary',CONTROL,'primary'),('frozen125-full_control','control-full_teacher','full_control')]:
  s=oldsel[lang][arm];z=array(P/'balanced-head-study-125'/lang/arm/('epoch-'+str(s['epoch'])+'-dev.npy'),s['dev_logits_sha256']);m=measures(z,dr,specs[n],devrefs)(range(len(dr)));compare(m,rr['frozen125_references'][k]['dev']);dev[lang][n]=m
fail=[]
for lang in ('ro','uk'):
 t=totals[lang];p=t[PRIMARY];references=[('selected_control',CONTROL),('frozen125-primary','frozen125-primary'),('frozen125-control-full_teacher','frozen125-full_control')]
 for label,name in references:
  for part in ('probe_operator','probe_meaning','probe_combination','probe_both'):
   for key in ('binary_errors','joint_errors','assertion_errors'):
    if p['groups']['partition:'+part][key]>t[name]['groups']['partition:'+part][key]:fail.append(f'{lang}/{part}/{label}/{key}')
 pool=lambda n:sum(t[n]['groups']['partition:'+part]['assertion_errors'] for part in ('probe_operator','probe_both'))
 for label,name in references+[('initial75','initial75'),('v9','v9')]:
  if pool(PRIMARY)>=pool(name):fail.append(f'{lang}/no_strict_transfer_improvement/{label}')
 if pool(PRIMARY)>={'ro':106,'uk':102}[lang]:fail.append(lang+'/no_strict_frozen125_floor_improvement')
 for label,name in references+[('initial75','initial75')]:
  for key in ('binary_errors','joint_errors','introduced_v9'):
   if dev[lang][PRIMARY][key]>dev[lang][name][key]:fail.append(f'{lang}/original_dev/{label}/{key}')
 for key,limit in {'ro':dict(binary_errors=12,joint_errors=673,introduced_v9=2),'uk':dict(binary_errors=1,joint_errors=634,introduced_v9=0)}[lang].items():
  if dev[lang][PRIMARY][key]>limit:fail.append(f'{lang}/fixed_original_dev_ceiling/{key}')
 for key in ('binary_errors','joint_errors'):
  if p['full_panel'][key]:fail.append(f'{lang}/full_panel/{key}')
assert fail==r['gate']['failures'] and r['gate']['escalation_allowed']==(not fail)
assert len(r['replay_records'])==52 and sum(v['rows'] for v in r['replay_records'])==229762
assert all(sha(p)==v for p,v in tracked.items())
summary={l:dict(transfer_assertion_errors=sum(totals[l][PRIMARY]['groups']['partition:'+p]['assertion_errors'] for p in ('probe_operator','probe_both')),dev=dev[l][PRIMARY],panel=totals[l][PRIMARY]['full_panel']) for l in ('ro','uk')}
result=dict(verification='PASS',source_sha256=sha(__file__),producer_report_sha256=sha(E/'report.json'),saved_unique_input_predictions_recounted=18480,panel_responses_recounted=240,selected_dev_predictions_recounted=5*(4202+4635),groups_recounted=groups_count,metric_objects_checked=checks,maximum_mean_ce_difference=max_ce,all_integer_and_reference_metrics_exact=True,gate_decision='REJECT_ESCALATION',failed_gate_predicates=len(fail),failures=fail,summary=summary,inputs_and_outputs_unchanged=True,input_sha256=tracked,qualified=False,scope='Independent saved-logit policy-score/argmax/metric/gate recount. Pinned portable policy scorer reused; no encoder, head replay, optimizer replay, label adjudication, held-out TEST or independent semantic validation.')
(O/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('input_sha256','summary')}))
