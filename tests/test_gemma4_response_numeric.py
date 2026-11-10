"""Numerical readout and adapter-restoration tests on small CPU fixtures."""
import hashlib

import pytest


def dependencies():
    mx=pytest.importorskip('mlx.core');np=pytest.importorskip('numpy')
    pytest.importorskip('safetensors.numpy');mx.set_default_device(mx.cpu)
    return mx,np


@pytest.mark.parametrize('hidden',[0,128])
def test_packaged_response_head_matches_training_and_rejects_bad_weights(tmp_path,hidden):
    mx,np=dependencies()
    from mlx.utils import tree_flatten
    from safetensors.numpy import save_file
    from training.train_neural_response_v10 import make_head
    from roguard.gemma4_response import _Readout
    mx.random.seed(1058);head=make_head({'hidden':hidden});mx.eval(head.parameters())
    tensors={key:np.asarray(value).copy() for key,value in tree_flatten(head.parameters())}
    path=tmp_path/'head.safetensors';save_file(tensors,str(path));sha=hashlib.sha256(path.read_bytes()).hexdigest()
    packaged=_Readout(path,sha,hidden)
    for x in (np.zeros((1,2560),np.float32),np.random.default_rng(1058).normal(size=(1,2560)).astype(np.float32)):
        expected=head(mx.array(x));mx.eval(expected)
        assert np.array_equal(np.asarray(packaged.logits(x),dtype=np.float32),np.asarray(expected)[0])
    with pytest.raises(ValueError,match='identity'):_Readout(path,'0'*64,hidden)
    tensors['last.bias']=np.full((5,),np.nan,dtype=np.float32);save_file(tensors,str(path))
    with pytest.raises(ValueError,match='Invalid response weight'):_Readout(path,hashlib.sha256(path.read_bytes()).hexdigest(),hidden)


def test_packaged_base_view_restores_actual_gemma_adapters_after_failure():
    mx,np=dependencies()
    from mlx_lm.models.gemma4_text import Model,ModelArgs
    from roguard.gemma4_review import _make_model
    from roguard.gemma4_response import _without_adapters
    model=Model(ModelArgs(hidden_size=16,num_hidden_layers=42,intermediate_size=32,
        num_attention_heads=2,head_dim=8,global_head_dim=8,num_key_value_heads=1,
        num_kv_shared_layers=18,vocab_size=64,vocab_size_per_layer_input=64,
        hidden_size_per_layer_input=8,sliding_window=4,
        layer_types=['sliding_attention','full_attention']*21))
    prepared=_make_model(model,boundary=24,width=12,lora=dict(rank=2,scale=4.,dropout=0.,keys=['self_attn.q_proj','self_attn.o_proj']))
    modules=[getattr(layer.self_attn,name) for layer in prepared.base.layers[24:] for name in ('q_proj','o_proj')]
    for i,module in enumerate(modules):module.lora_b=mx.ones_like(module.lora_b)*(i+1)
    saved=[module.lora_b for module in modules]
    with pytest.raises(RuntimeError,match='deliberate'):
        with _without_adapters(prepared.base):
            assert all(float(mx.max(mx.abs(module.lora_b)).item())==0 for module in modules)
            raise RuntimeError('deliberate inference failure')
    assert all(module.lora_b is value for module,value in zip(modules,saved))
