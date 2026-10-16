"""Export/restore synthetic-study LoRA deltas as a bounded JSON artifact.

No base-model weights, local paths, executable objects, or pickle data are stored.
The upstream base remains a separate, pinned dependency.
"""
import argparse
import base64
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

FORMAT='roguard-instruction-lora-v1'
MAX_BYTES=64*1024*1024


def validate(spec):
    if spec.get('format')!=FORMAT or spec.get('base_model')!='mlx-community/Qwen3-4B-Instruct-2507-4bit':
        raise ValueError('Unsupported adapter format or base')
    config=spec['adapter_config']
    if config!={'fine_tune_type':'lora','num_layers':8,'lora_parameters':{'rank':8,'dropout':0.0,'scale':16.0}}:
        raise ValueError('Unexpected adapter architecture')
    dimensions={'self_attn.q_proj':(2560,4096),'self_attn.k_proj':(2560,1024),'self_attn.v_proj':(2560,1024),'self_attn.o_proj':(4096,2560),'mlp.gate_proj':(2560,9728),'mlp.up_proj':(2560,9728),'mlp.down_proj':(9728,2560)}
    shapes={f'model.layers.{layer}.{module}.lora_{side}':([widths[0],8] if side=='a' else [8,widths[1]]) for layer in range(28,36) for module,widths in dimensions.items() for side in ('a','b')}
    if not isinstance(spec.get('tensors'),dict) or set(spec['tensors'])!=set(shapes):
        raise ValueError('Missing or unexpected adapter tensors')
    total=0
    for name,value in spec['tensors'].items():
        if not re.fullmatch(r'model\.layers\.\d+\.[a-z_]+\.[a-z_]+\.lora_[ab]',name):
            raise ValueError('Unexpected tensor name')
        shape=value['shape']
        if shape!=shapes[name]:raise ValueError('Tensor shape differs from pinned base architecture')
        if len(shape)!=2 or any(type(n) is not int or not 1<=n<=20000 for n in shape):
            raise ValueError('Invalid tensor shape')
        if value['dtype'] not in ('float32','float16'):
            raise ValueError('Unsupported tensor dtype')
        expected=math.prod(shape)*(4 if value['dtype']=='float32' else 2)
        total+=expected
        if total>MAX_BYTES:raise ValueError('Adapter exceeds size limit')
        raw=base64.b64decode(value['data'],validate=True)
        if len(raw)!=expected:raise ValueError('Tensor byte length mismatch')
    return spec


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('export','restore'))
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import numpy as np
    from safetensors.numpy import load_file,save_file
    if a.mode=='export':
        tensors=load_file(str(a.input/'adapters.safetensors'))
        if any(not np.isfinite(v).all() for v in tensors.values()):raise ValueError('Nonfinite adapter weight')
        original=json.loads((a.input/'adapter_config.json').read_text(encoding='utf-8'))
        config={k:original[k] for k in ('fine_tune_type','num_layers','lora_parameters')}
        spec=dict(format=FORMAT,base_model='mlx-community/Qwen3-4B-Instruct-2507-4bit',
            base_revision='50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b',base_license='Apache-2.0',
            synthetic_training_only=True,stable_release_approved=False,adapter_config=config,
            original_safetensors_sha256=hashlib.sha256((a.input/'adapters.safetensors').read_bytes()).hexdigest(),
            tensors={k:dict(shape=list(v.shape),dtype=str(v.dtype),data=base64.b64encode(v.tobytes()).decode('ascii')) for k,v in tensors.items()})
        validate(spec)
        with a.output.open('xb') as f:f.write(gzip.compress(json.dumps(spec,separators=(',',':')).encode(),mtime=0))
    else:
        with gzip.open(a.input,'rb') as f:raw=f.read(MAX_BYTES*3+1)
        if len(raw)>MAX_BYTES*3:raise ValueError('Oversized adapter JSON')
        spec=validate(json.loads(raw))
        tensors={k:np.frombuffer(base64.b64decode(v['data']),dtype=v['dtype']).reshape(v['shape']).copy() for k,v in spec['tensors'].items()}
        if any(not np.isfinite(v).all() for v in tensors.values()):raise ValueError('Nonfinite adapter weight')
        a.output.mkdir(parents=True,exist_ok=False)
        save_file(tensors,str(a.output/'adapters.safetensors'))
        (a.output/'adapter_config.json').write_text(json.dumps(spec['adapter_config'],indent=2)+'\n',encoding='utf-8')
    print(a.mode,'complete:',a.output)

if __name__=='__main__':main()
