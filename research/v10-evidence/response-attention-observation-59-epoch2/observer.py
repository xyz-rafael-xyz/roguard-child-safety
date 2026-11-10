import argparse,json,hashlib
from pathlib import Path
import numpy as np
from roguard.assurance import digest
from roguard.trajectory_data import read_jsonl,file_sha
from roguard.trajectory_model import RESPONSE_LABELS
p=argparse.ArgumentParser()
for name in ('study','private','baseline_study','baseline_private','output'):p.add_argument('--'+name.replace('_','-'),dest=name,type=Path,required=True)
p.add_argument('--epoch',type=int,required=True);a=p.parse_args()
if a.output.exists():raise ValueError('Refusing to overwrite epoch observation')
progress=json.loads((a.study/'progress.json').read_text());epoch=next(e for e in progress['epochs'] if e['epoch']==a.epoch)
if progress['selected']!=epoch:raise ValueError('This epoch no longer owns the saved selected logits')
weights=a.private/('epoch-'+str(a.epoch)+'.safetensors')
if file_sha(weights)!=epoch['adapter_sha256'] or file_sha(a.private/'selected.safetensors')!=epoch['adapter_sha256']:raise ValueError('Selected weights differ')
identity=[r for r in read_jsonl(a.study/'feature-inputs.jsonl') if r['partition']=='dev']
baseline=json.loads((a.baseline_study/'report.json').read_text());rows=[];source_hashes={}
for origin in ('historical','generated'):
 path=a.baseline_private/(origin+'-development-predictions.jsonl');part=read_jsonl(path);rows.extend(part);source_hashes[origin]=file_sha(path)
 if sum(r['baseline26']['label']!=r['gold'] for r in part)!=baseline['metrics'][origin+'-development']['baseline26']['errors']:raise ValueError('Baseline counts differ')
if len(identity)!=len(rows):raise ValueError('Incomplete DEV mapping')
for r,i in zip(rows,identity):
 payload={'language':r['language'],'turns':([{'role':'minor','text':r['context']}] if r['context'] else [])+[{'role':'assistant','text':r['draft']}]}
 if i!={'case_id':r['case_id'],'partition':'dev','input_sha256':digest(payload)}:raise ValueError('DEV identity differs')
raw=(a.private/'selected-development-logits.npy').read_bytes()
import io
z=np.load(io.BytesIO(raw),allow_pickle=False)
if z.shape!=(len(rows),5) or not np.isfinite(z).all():raise ValueError('Invalid recorded logits')
metrics={};errors=[]
for origin in ('historical','generated'):
 indices=[i for i,r in enumerate(rows) if r['origin']==origin];m={'rows':len(indices),'candidate_errors':0,'baseline26_errors':0,'fixed':0,'introduced':0}
 for i in indices:
  row=rows[i];label=RESPONSE_LABELS[int(z[i].argmax())];old=row['baseline26']['label'];gold=row['gold']
  m['candidate_errors']+=label!=gold;m['baseline26_errors']+=old!=gold;m['fixed']+=old!=gold and label==gold;m['introduced']+=old==gold and label!=gold
  if label!=gold:errors.append({'case_id':row['case_id'],'language':row['language'],'origin':origin,'gold':gold,'predicted':label,'baseline26':old})
 if m['candidate_errors']!=epoch['development']['populations'][origin]['errors']:raise ValueError('Epoch counts differ')
 metrics[origin]=m
if file_sha(a.private/'selected.safetensors')!=epoch['adapter_sha256']:raise ValueError('Selection changed during observation')
a.output.mkdir(parents=True)
(a.private/('observed-epoch-'+str(a.epoch)+'-development-logits.npy')).write_bytes(raw)
report={'complete':True,'epoch':a.epoch,'adapter_sha256':epoch['adapter_sha256'],'recorded_logits_sha256':hashlib.sha256(raw).hexdigest(),'source_prediction_hashes':source_hashes,'baseline_report_sha256':file_sha(a.baseline_study/'report.json'),'observer_sha256':file_sha(__file__),'metrics':metrics,'remaining_errors':errors,'training_complete':False,'selection_frozen':False,'selected_runtime_reload_verified':False,'diagnostics_opened_by_observation':False,'stable_release_ready':False}
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');(a.output/'observer.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(metrics,indent=2))
