"""Audit native-token duplicates and conflicting labels in prior evaluation only."""
import argparse
from pathlib import Path
import json,gzip,hashlib
from collections import defaultdict
from datetime import datetime,timezone
from transformers import AutoTokenizer
from training.train_bare_distress_v9 import texts
from training.prepare_instruction_sft_v8 import target
root=Path('research/v9-stabilization')
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--ro-base',type=Path,required=True)
parser.add_argument('--uk-base',type=Path,required=True)
parser.add_argument('--parent',type=Path,default=root/'native-morphology-128')
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
bases={'ro':args.ro_base,'uk':args.uk_base}

result={};inputs={}
for language,base in bases.items():
 tok=AutoTokenizer.from_pretrained(base,local_files_only=True,trust_remote_code=False)
 groups=defaultdict(list);total=0
 for split in ('dev','scope_dev','cal','exposed')+(('stress',) if language=='uk' else ()):
  p=args.parent/language/(split+'.jsonl.gz');inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
  rows=[json.loads(l) for l in gzip.decompress(p.read_bytes()).decode().splitlines()]
  pairs=[texts(r) for r in rows]
  values=tok([x[0] for x in pairs],[x[1] for x in pairs],padding=False,truncation=False)['input_ids']
  assert max(map(len,values))<=512
  for r,ids in zip(rows,values):
   key=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
   groups[key].append(dict(case_id=r['case_id'],split=split,gold=int(r['review'] if 'review' in r else target(r))))
  total+=len(rows)
 conflicting=[dict(token_input_sha256=k,rows=v) for k,v in groups.items() if len({r['gold'] for r in v})>1]
 result[language]=dict(rows=total,unique_inputs=len(groups),duplicate_groups=sum(len(v)>1 for v in groups.values()),duplicate_details=[dict(token_input_sha256=k,rows=v) for k,v in groups.items() if len(v)>1],contradictory_groups=conflicting)
 print(language,json.dumps({k:v for k,v in result[language].items() if k not in ('contradictory_groups','duplicate_details')}),'conflicts',len(conflicting),flush=True)
record=dict(format='roguard-prior-evaluation-token-label-audit-v1',created_utc=datetime.now(timezone.utc).isoformat(),populations='Nine prior evaluation populations only',parent=str(args.parent),source_sha256=hashlib.sha256(Path('training/audit_evaluation_inputs_v9.py').read_bytes()).hexdigest(),reserved_test_read=False,neural_inference_performed=False,input_sha256=inputs,results=result)
args.output.write_text(json.dumps(record,indent=2)+'\n')
