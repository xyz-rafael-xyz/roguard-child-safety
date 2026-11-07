"""Compare two captured selections on the same592 exposed DEV rows; no inference."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from roguard.trajectory_data import file_sha, write_jsonl
from roguard.trajectory_model import RESPONSE_LABELS
from training.train_large_response_attention_v10 import load_inputs

p=argparse.ArgumentParser()
for name in ('root','private','output'):p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();paired=a.output.with_name(a.output.stem+'-rows.jsonl')
assert not a.output.exists() and not paired.exists()
read=lambda path:json.loads(path.read_text())
study=a.root/'research/v10-evidence/mixed-large-response-attention-65'
protocol=read(study/'protocol.json');command_path=a.private/'mixed-large-response-attention-65-command.json';command=read(command_path)
args=SimpleNamespace(**{command[i][2:].replace('-','_'):Path(command[i+1]) for i in range(2,len(command),2)})
rows,x,train,dev,identities,_,_,bindings=load_inputs(args);del x
assert bindings==protocol['bindings']
assert identities==[json.loads(line) for line in (study/'feature-inputs.jsonl').read_text().splitlines()]
arrays=[];proofs=[]
for epoch,name in ((4,'mixed-response-epoch4-selected-logits-verification-65.json'),(6,'mixed-response-epoch6-selected-logits-verification-65.json')):
 proof=read(a.private/name);path=a.private/f'mixed-response-epoch{epoch}-selected-dev-logits-65.npy'
 assert proof['verified'] is True and proof['actual_protocol_sha256']==file_sha(study/'protocol.json')
 assert proof['command_sha256']==file_sha(command_path) and proof['selected_dev_logits_sha256']==file_sha(path)
 values=np.load(path,allow_pickle=False);assert values.shape==(592,5) and np.isfinite(values).all()
 arrays.append(values);proofs.append(proof)
old,new=[values.argmax(axis=1) for values in arrays];gold=np.array([RESPONSE_LABELS.index(rows[i]['gold']) for i in dev])
assert int((old!=gold).sum())==proofs[0]['errors']==12
assert int((new!=gold).sum())==proofs[1]['errors']==11
counts={'rows':592,'previous_errors':int((old!=gold).sum()),'current_errors':int((new!=gold).sum()),
 'fixes':int(((old!=gold)&(new==gold)).sum()),'introductions':int(((old==gold)&(new!=gold)).sum()),
 'both_wrong':int(((old!=gold)&(new!=gold)).sum()),'both_correct':int(((old==gold)&(new==gold)).sum()),
 'changed_predictions':int((old!=new).sum()),'changed_wrong_class':int(((old!=gold)&(new!=gold)&(old!=new)).sum())}
assert counts['fixes']-counts['introductions']==counts['previous_errors']-counts['current_errors']
records=[{'case_id':rows[i]['case_id'],'origin':rows[i]['origin'],'language':rows[i]['language'],'gold':rows[i]['gold'],
          'epoch4':RESPONSE_LABELS[int(before)],'epoch6':RESPONSE_LABELS[int(after)]} for i,before,after in zip(dev,old,new)]
write_jsonl(paired,records)
changes=[r for r in records if r['epoch4']!=r['epoch6']]
report={'verified':True,'source_sha256':file_sha(__file__),'protocol_sha256':file_sha(study/'protocol.json'),
 'epoch4_logits_sha256':proofs[0]['selected_dev_logits_sha256'],'epoch6_logits_sha256':proofs[1]['selected_dev_logits_sha256'],
 **counts,'changed_groups':sorted({r['origin']+':'+r['language']+':'+r['gold'] for r in changes}),
 'paired_records_sha256':file_sha(paired),'scope':'Per-case epoch4 versus epoch6 comparison on the same592 exposed development rows only; no comparison with v9, no broader regression or final qualification.',
 'model_inference_performed':False,'held_out_test_opened':False,'qualified':False}
with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(report,indent=2))
