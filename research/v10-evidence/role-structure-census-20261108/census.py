"""Read-only structural role census on all recovered72 inputs matched to88 metadata.
No tokenizer, numerical libraries, features, model predictions or label metrics.
"""
from pathlib import Path
import json,hashlib,os
from collections import Counter
P=Path(os.environ['ROGUARD_PRIVATE_ROOT']);R=Path(os.environ['ROGUARD_PUBLIC_ROOT']);O=Path(os.environ['ROGUARD_CENSUS_OUTPUT'])

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def classify(roles):
 assert all(r in ('minor','assistant','adult','other','unknown') for r in roles)
 if not roles:return 'NO_PRIOR'
 if all(r=='minor' for r in roles):return 'ONLY_PRIOR_MINOR'
 if all(r!='minor' for r in roles):return 'ONLY_PRIOR_NONMINOR'
 seen=False
 for r in roles:
  if r!='minor':seen=True
  elif seen:return 'NONMINOR_BEFORE_LATER_MINOR'
 return 'ALL_PRIOR_MINOR_BEFORE_NONMINOR'
assert classify([])=='NO_PRIOR'
assert classify(['minor','assistant'])=='ALL_PRIOR_MINOR_BEFORE_NONMINOR'
assert classify(['assistant','minor'])=='NONMINOR_BEFORE_LATER_MINOR'
assert classify(['minor','assistant','minor'])=='NONMINOR_BEFORE_LATER_MINOR'

def main():
 raw=P/'large-review-input-recovery-72/actual/inputs.jsonl';r72=R/'research/v10-evidence/large-review-input-recovery-72/report.json';r88=R/'research/v10-evidence/runtime-review-baseline-88/report.json'
 a,b=read(r72),read(r88);assert a['complete'] is True and b['complete'] is True
 assert sha(raw)==a['inputs_sha256'] and raw.stat().st_size==a['inputs_bytes']
 files={str(x):sha(x) for x in (raw,r72,r88,Path(__file__))};metas={};cursor={'ro':0,'uk':0}
 for lang,n in [('ro',17376),('uk',20390)]:
  p=P/f'runtime-review-baseline-88/{lang}-rows.jsonl';assert sha(p)==b['languages'][lang]['artifacts'][p.name]
  files[str(p)]=sha(p);metas[lang]=[json.loads(s) for s in p.read_text().splitlines()];assert len(metas[lang])==n
 write(O/'input-registration.json',dict(scope='ALL37766 recovered72 inputs matched to original88 rows, structural roles only',files=files,encoder_loaded=False,tokenizer_loaded=False,feature_or_logit_arrays_opened=False))
 totals={};seen=set();output=O/'rows.jsonl'
 with raw.open() as src,output.open('x') as dest:
  for line in src:
   row=json.loads(line);lang=row['language'];assert lang in cursor
   idx=cursor[lang];meta=metas[lang][idx]
   for key in ('case_id','language','partition','group','fields','gold','token_sha256'):assert canonical(row[key])==canonical(meta[key]),(key,lang,idx)
   assert row['feature_index']==idx and type(row['feature_index']) is int
   assert row['partition'] in ('train','dev');assert row['case_id'] not in seen;seen.add(row['case_id'])
   turns=row['payload']['turns'];assert row['payload']['language']==lang and turns and turns[-1]['role']=='minor'
   prior=[t['role'] for t in turns[:-1]];category=classify(prior)
   out=dict(case_id=row['case_id'],language=lang,partition=row['partition'],group=row['group'],feature_index=idx,token_sha256=row['token_sha256'],prior_roles=prior,prior_minor_turns=prior.count('minor'),prior_nonminor_turns=sum(r!='minor' for r in prior),category=category,structural_order_allows_different_matched_masks=category=='NONMINOR_BEFORE_LATER_MINOR')
   dest.write(canonical(out)+'\n')
   for axis,value in [('all','all'),('partition',row['partition']),('source_group',row['group'])]:
    counts=totals.setdefault(lang,{}).setdefault(axis,{}).setdefault(value,Counter());counts['rows']+=1;counts[category]+=1
   cursor[lang]+=1
 assert cursor=={'ro':17376,'uk':20390} and len(seen)==37766
 assert all(sha(Path(p))==h for p,h in files.items())
 # Independently recount saved row categories/identity population before completion.
 retained=[json.loads(s) for s in output.read_text().splitlines()];assert len(retained)==37766
 for lang in cursor:
  c=Counter(r['category'] for r in retained if r['language']==lang)
  expected={k:v for k,v in totals[lang]['all']['all'].items() if k!='rows'};assert c==expected
 write(O/'report.json',dict(status='COMPLETE_STRUCTURAL_CENSUS_ONLY',source_sha256=sha(Path(__file__)),input_registration_sha256=sha(O/'input-registration.json'),rows_sha256=sha(output),total_rows=37766,languages=totals,interpretation='Only nonminor-before-later-minor ordering can yield different aligned/count-matched-chronological prior masks. This is an upper bound on informative rows; excluded/empty content-token spans can still make those masks identical. No actual token-mask differences established.',limits=['Turn-level structure only; cached token digests matched to88 metadata, not retokenized.','No feature/logit arrays, head, encoder or accuracy metrics.','TRAIN and DEV remain separate groups; DEV already exposed. No TEST opened.','No rows filtered; source-group and partition counts retained.'],encoder_loaded=False,tokenizer_loaded=False,training_performed=False,qualified=False))
 print(json.dumps({l:totals[l]['all']['all'] for l in totals}))
if __name__=='__main__':main()
