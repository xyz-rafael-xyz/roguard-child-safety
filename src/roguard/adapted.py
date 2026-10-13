"""Verified public LoRA artifact; pinned base, no executable model downloads."""
import gzip,hashlib,json,math
from importlib import resources
from .uncertainty import CalibratedDecisions

ADAPTER_SHA256='19557343192cee579aaf296ced0993a2a5b6a07631d0efce1b2551e670bd2389'

def load_adapted(encoder):
    import torch
    from peft import LoraConfig,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict
    raw=resources.read_binary('roguard','v6-adapter.json.gz')
    if hashlib.sha256(raw).hexdigest()!=ADAPTER_SHA256:raise ValueError('Adapter digest mismatch')
    data=json.loads(gzip.decompress(raw))
    if data['format']!='roguard-lora-json-v1':raise ValueError('Invalid adapter format')
    model=get_peft_model(encoder.model,LoraConfig(r=4,lora_alpha=8,lora_dropout=.05,target_modules=['Wqkv']))
    expected=get_peft_model_state_dict(model);weights=data['tensors']
    if set(weights)!=set(expected):raise ValueError('Adapter tensor names differ')
    state={}
    for key,value in weights.items():
        tensor=torch.tensor(value,dtype=torch.float32)
        if tensor.shape!=expected[key].shape or not torch.isfinite(tensor).all():raise ValueError('Invalid adapter tensor')
        state[key]=tensor
    set_peft_model_state_dict(model,state);encoder.model=model.to(encoder.device).eval()
    return data['head'],data['model_id'],{lang:CalibratedDecisions(a) for lang,a in data['calibration'].items()}
