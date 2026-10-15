"""Validate frozen counts, exact train/test separation and augmentation lineage."""
import gzip,hashlib,json
from pathlib import Path
from training.evidence_v8 import load
from roguard.context import validate
from roguard.evidence import ONTOLOGY

root=Path('research/v8-evidence/data');allseen={};report={'splits':{},'training_additions':{}}
for split in ('train','dev','cal','test'):
 rows=load(root,split);seen=set()
 for r in rows:
  validate({k:r[k] for k in ('language','turns')})
  assert all(r['attributes'][k] in values for k,values in ONTOLOGY.items())
  key=json.dumps(r['turns'],ensure_ascii=False,sort_keys=True)
  assert key not in allseen,(split,allseen.get(key))
  seen.add(key)
 for key in seen:allseen[key]=split
 report['splits'][split]={'rows':len(rows),'unique_inputs':len(seen)}
# Contextual file includes the paraphrase rows; count them only once.
path=root/'train-contextual.jsonl.gz';rows=[json.loads(l) for l in gzip.decompress(path.read_bytes()).decode().splitlines()];new=set()
for r in rows:
 key=json.dumps(r['turns'],ensure_ascii=False,sort_keys=True)
 assert key not in allseen or allseen[key]=='train','Held-out leakage'
 if key not in allseen:new.add(key)
report['training_additions']={'rows':len(rows),'unique_additional_inputs':len(new),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
report['total_distinct_inputs']=len(allseen)+len(new);report['exact_cross_split_overlap']=0
out=root.parent/'data-audit.json'
if out.exists():raise ValueError('Refusing overwrite')
out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
