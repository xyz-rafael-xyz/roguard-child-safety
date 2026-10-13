"""Export only a gate-passing fictional-data adapter to a bounded JSON artifact."""
import argparse,gzip,hashlib,json
from pathlib import Path
from safetensors.torch import load_file

def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    evaluation=json.loads(a.evaluation.read_text());freeze=json.loads((a.candidate/'freeze.json').read_text())
    if not evaluation['promotion_gate_passed'] or evaluation['freeze']!=freeze:raise ValueError('Promotion gate or freeze mismatch')
    for name,sha in freeze.items():
        if hashlib.sha256((a.candidate/name).read_bytes()).hexdigest()!=sha:raise ValueError('Changed frozen artifact')
    tensors=load_file(a.candidate/'adapter/adapter_model.safetensors');data={k:v.tolist() for k,v in tensors.items()}
    head=json.loads((a.candidate/'head.json').read_text());model=evaluation['models']['v6_adapter']
    payload=dict(tensors=data,head=head,model_id=model['model_id'],calibration={l:c['artifact'] for l,c in model['calibration'].items()},format='roguard-lora-json-v1')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    raw=gzip.compress(json.dumps(payload,sort_keys=True,separators=(',',':')).encode(),mtime=0)
    with a.output.open('xb') as stream:stream.write(raw)
    print(hashlib.sha256(raw).hexdigest())
if __name__=='__main__':main()
