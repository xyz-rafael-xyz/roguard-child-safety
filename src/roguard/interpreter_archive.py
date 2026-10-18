"""Strict data-only decoder for the frozen v9 interpreter adapter archive."""
import base64
import math
import re

FORMAT='roguard-gemma-interpreter-lora-v1'

MAX_BYTES=64*1024*1024

BASES={
    'mlx-community/gemma-3-text-12b-it-4bit':('83eda451975103fff6f11e97bf75cd0f1c61af2c',3840,4096,2048,15360,40,48),
    'mlx-community/gemma-3-text-4b-it-4bit':('4f665a4c50ecfe4ecdc34056ab52fe3e3c4abf9e',2560,2048,1024,10240,26,34),
}

def validate(spec):
    if spec.get('format')!=FORMAT or spec.get('base_model') not in BASES:
        raise ValueError('Unsupported adapter format or base')
    revision,hidden,query,kv,intermediate,first,last=BASES[spec['base_model']]
    if spec.get('base_revision')!=revision:raise ValueError('Unexpected base revision')
    config=spec['adapter_config']
    if config!={'fine_tune_type':'lora','num_layers':8,'lora_parameters':{'rank':8,'dropout':0.0,'scale':16.0}}:
        raise ValueError('Unexpected adapter architecture')
    dimensions={'self_attn.q_proj':(hidden,query),'self_attn.k_proj':(hidden,kv),'self_attn.v_proj':(hidden,kv),'self_attn.o_proj':(query,hidden),'mlp.gate_proj':(hidden,intermediate),'mlp.up_proj':(hidden,intermediate),'mlp.down_proj':(intermediate,hidden)}
    shapes={f'language_model.model.layers.{layer}.{module}.lora_{side}':([widths[0],8] if side=='a' else [8,widths[1]]) for layer in range(first,last) for module,widths in dimensions.items() for side in ('a','b')}
    if not isinstance(spec.get('tensors'),dict) or set(spec['tensors'])!=set(shapes):
        raise ValueError('Missing or unexpected adapter tensors')
    total=0
    for name,value in spec['tensors'].items():
        if not re.fullmatch(r'language_model\.model\.layers\.\d+\.[a-z_]+\.[a-z_]+\.lora_[ab]',name):
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
