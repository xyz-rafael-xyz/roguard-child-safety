"""Amended-policy replay on saved predictions; never a fresh model test."""
import gzip,hashlib,json
from pathlib import Path
from training.evidence_v8 import load
from roguard.evidence import disposition
from eval.evidence_v8 import evaluate

root=Path('research/v8-evidence');rows=load(root/'data','test')
records=[json.loads(l) for l in gzip.decompress((root/'final-test/predictions.jsonl.gz').read_bytes()).decode().splitlines()]
assert [r['case_id'] for r in rows]==[r['case_id'] for r in records]
result={'kind':'policy_replay_not_fresh_test','policy_sha256':hashlib.sha256(Path('src/roguard/evidence.py').read_bytes()).hexdigest(),'results':{}}
for name in ('character','semantic','cascade'):result['results'][name]=evaluate(rows,[r['predictions'][name] for r in records])
base=[json.loads(l) for l in gzip.decompress((root/'v7-baseline/predictions.jsonl.gz').read_bytes()).decode().splitlines()]
assert [r['case_id'] for r in rows]==[r['case_id'] for r in base]
result['v7_baseline']={}
for lang in ('ro','uk'):
 items=[(r,p) for r,p in zip(rows,base) if r['language']==lang]
 gold=lambda r:disposition({k:{'label':v} for k,v in r['attributes'].items()})[0]
 benign=[(r,p) for r,p in items if gold(r)=='CONTINUE'];concerns=[(r,p) for r,p in items if gold(r)=='SUPPORT_AND_REVIEW']
 result['v7_baseline'][lang]={'benign':len(benign),'benign_review':sum(p['review'] for r,p in benign),'concern':len(concerns),'concern_review':sum(p['review'] for r,p in concerns)}
path=root/'policy-replay.json'
if path.exists():raise ValueError('Refusing overwrite')
path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:{l:{key:v for key,v in m.items() if key!='attributes'} for l,m in values.items() if l in ('ro','uk')} for k,values in result['results'].items()},indent=2))
print('v7',result['v7_baseline'])
