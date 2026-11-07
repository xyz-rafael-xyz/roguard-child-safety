import hashlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from safetensors import safe_open
from training.train_large_response_attention_v10 import load_inputs
from training.train_large_response_v10 import population_metrics
from roguard.trajectory_data import file_sha,read_jsonl
from roguard.trajectory_model import RESPONSE_LABELS
import argparse
p=argparse.ArgumentParser(description='CPU diagnostic transplant of the epoch2 response head onto its original bound features.')
p.add_argument('--root',type=Path,required=True);p.add_argument('--private-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
args=p.parse_args();root=args.root;cache=args.private_root;public=root/'research/v10-evidence'
if args.output.exists():raise ValueError('Refusing overwrite')
command=json.loads((cache/'large-response-attention-59-command.json').read_text());a=SimpleNamespace(**{key[2:].replace('-','_'):Path(value) for key,value in zip(command[2::2],command[3::2])})
protocol=json.loads((a.study/'protocol.json').read_text());assert file_sha(root/'training/train_large_response_attention_v10.py')==protocol['source_sha256']
rows,x,train,dev,identities,parent,data,bindings=load_inputs(a);assert bindings==protocol['bindings']
observation=json.loads((public/'response-attention-observation-59-epoch2/report.json').read_text());weights=a.private/'epoch-2.safetensors';assert file_sha(weights)==observation['adapter_sha256']
with safe_open(weights,framework='numpy') as f:w=f.get_tensor('head.last.weight');b=f.get_tensor('head.last.bias')
assert w.shape==(5,2560) and b.shape==(5,) and w.dtype==b.dtype==np.float32
h=x[dev];z32=h@w.T+b;z64=h.astype(np.float64)@w.astype(np.float64).T+b.astype(np.float64)
assert np.array_equal(z32.argmax(1),z64.argmax(1));ordered=np.sort(z64,axis=1);minimum_margin=float((ordered[:,-1]-ordered[:,-2]).min())
assert minimum_margin>1e-4
original=a.parent_private/'selected.safetensors';assert file_sha(original)==parent['selected']['head_sha256']
with safe_open(original,framework='numpy') as f:ow=f.get_tensor('last.weight');ob=f.get_tensor('last.bias')
baseline=h.astype(np.float64)@ow.astype(np.float64).T+ob.astype(np.float64)
saved_path=a.private/'observed-epoch-2-development-logits.npy';assert file_sha(saved_path)==observation['recorded_logits_sha256'];full=np.load(saved_path,allow_pickle=False)
dev_rows=[rows[i] for i in dev]
old_rows=[]
for origin in ('historical','generated'):
 p=a.feature_private/(origin+'-development-predictions.jsonl');assert file_sha(p)==observation['source_prediction_hashes'][origin];old_rows.extend(read_jsonl(p))
assert [r['case_id'] for r in old_rows]==[r['case_id'] for r in dev_rows]
assert all(RESPONSE_LABELS[int(z.argmax())]==r['baseline26']['label'] for z,r in zip(baseline,old_rows))
metrics={name:population_metrics(dev_rows,z) for name,z in [('baseline26',baseline),('adapted_head_frozen_features',z64),('full_adapted_epoch2',full)]}
assert metrics['baseline26']['errors']==39 and metrics['full_adapted_epoch2']['errors']==5
gold=np.array([RESPONSE_LABELS.index(r['gold']) for r in dev_rows]);old=z64.argmax(1);new=full.argmax(1)
result={'complete':True,'source_sha256':file_sha(__file__),'epoch':2,'adapter_sha256':file_sha(weights),'feature_sha256':bindings['large_features_sha256'],'full_logits_sha256':file_sha(saved_path),'rows':len(dev),'metrics':metrics,'head_only_float32_float64_decisions_equal':True,'head_only_maximum_float32_float64_logit_difference':float(np.max(np.abs(z32-z64))),'head_only_minimum_top_two_logit_margin':minimum_margin,'full_vs_head_only':{'fixed':int(((old!=gold)&(new==gold)).sum()),'introduced':int(((old==gold)&(new!=gold)).sum())},'interpretation':'Diagnostic transplant of a coadapted head onto frozen pretraining features. It is not a separately trained candidate or a controlled estimate of a learned causal mechanism.','encoder_loaded':False,'training_or_selection_changed':False,'reserved_test_opened':False,'stable_release_ready':False}
args.output.mkdir(parents=True);(args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');(args.output/'source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps({'errors':{name:m['errors'] for name,m in metrics.items()},'full_vs_head_only':result['full_vs_head_only'],'minimum_margin':minimum_margin,'maximum_numeric_difference':result['head_only_maximum_float32_float64_logit_difference']},indent=2))
