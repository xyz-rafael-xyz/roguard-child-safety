"""Actual small-Gemma attention switching and packaged readout parity."""
import hashlib

import pytest


def model():
    mx=pytest.importorskip('mlx.core');np=pytest.importorskip('numpy');mx.set_default_device(mx.cpu)
    from mlx_lm.models.gemma4_text import Model,ModelArgs
    from roguard.gemma4_review import _make_model
    mx.random.seed(1060)
    base=Model(ModelArgs(hidden_size=16,num_hidden_layers=42,intermediate_size=32,
        num_attention_heads=2,head_dim=8,global_head_dim=8,num_key_value_heads=1,
        num_kv_shared_layers=18,vocab_size=64,vocab_size_per_layer_input=64,
        hidden_size_per_layer_input=8,sliding_window=4,
        layer_types=['sliding_attention','full_attention']*21))
    prepared=_make_model(base,boundary=24,width=12,
        lora=dict(rank=2,scale=4.,dropout=0.,keys=['self_attn.q_proj','self_attn.o_proj']))
    prepared.eval();return mx,np,prepared


def test_actual_adapters_change_predictions_match_cached_replay_and_restore_on_error():
    mx,np,prepared=model()
    from roguard.gemma4_response_attention import _adapter_slots,_replace_adapters
    from roguard.gemma4_response import _without_adapters
    from roguard.gemma4_attention import capture_with_prefix,replay
    from mlx_lm.models.cache import make_prompt_cache
    slots=_adapter_slots(prepared.base)
    original={key:getattr(module,attr) for key,(module,attr) in slots.items()}
    replacement={key:mx.ones_like(value)*.03 for key,value in original.items()}
    tokens=mx.array([[1,2,3,4,5]])
    old=prepared.base.model(tokens)[:,-1,:];mx.eval(old)
    cache=make_prompt_cache(prepared.base)
    with _without_adapters(prepared.base):mx.eval(prepared.base.model(tokens[:,:2],cache=cache))
    with pytest.raises(RuntimeError,match='deliberate'):
        with _replace_adapters(prepared.base,replacement):
            actual=prepared.base.model(tokens)[:,-1,:]
            state=capture_with_prefix(prepared.base,tokens[:,2:],cache,start=24)
            cached=replay(prepared.base,state)[:,0,:];mx.eval(actual,cached)
            assert np.max(np.abs(np.asarray(actual)-np.asarray(cached)))<1e-4
            assert not np.allclose(np.asarray(actual),np.asarray(old),atol=1e-5)
            raise RuntimeError('deliberate')
    assert all(getattr(module,attr) is original[key] for key,(module,attr) in slots.items())
    after=prepared.base.model(tokens)[:,-1,:];mx.eval(after)
    assert np.array_equal(np.asarray(after),np.asarray(old))
    bad=dict(replacement);bad.pop(next(iter(bad)))
    with pytest.raises(ValueError,match='inventory'):
        with _replace_adapters(prepared.base,bad):pytest.fail('Partial replacement was activated')
    assert all(getattr(module,attr) is original[key] for key,(module,attr) in slots.items())


def test_packaged_attention_readout_matches_training_and_validates_all_tensors(tmp_path):
    mx,np,prepared=model()
    from mlx.utils import tree_flatten
    from safetensors.numpy import save_file
    from roguard.gemma4_response_attention import _adapter_slots,_AdapterReadout
    from training.train_neural_response_v10 import make_head
    slots=_adapter_slots(prepared.base)
    head=make_head({'hidden':0});head.eval();mx.eval(head.parameters())
    tensors={key:np.asarray(getattr(module,attr)).copy() for key,(module,attr) in slots.items()}
    tensors.update({'head.'+key:np.asarray(value).copy() for key,value in tree_flatten(head.parameters())})
    path=tmp_path/'response.safetensors'
    def save(values):
        save_file(values,str(path));return hashlib.sha256(path.read_bytes()).hexdigest()
    readout=_AdapterReadout(path,save(tensors),prepared.base)
    for x in (np.zeros((1,2560),np.float32),np.random.default_rng(1060).normal(size=(1,2560)).astype(np.float32)):
        expected=head(mx.array(x));mx.eval(expected)
        assert np.array_equal(np.asarray(readout.logits(x),np.float32),np.asarray(expected)[0])
    with pytest.raises(ValueError,match='identity'):_AdapterReadout(path,'0'*64,prepared.base)
    for kind in ('missing','shape','dtype','nonfinite'):
        changed={key:value.copy() for key,value in tensors.items()};key=next(iter(slots))
        if kind=='missing':changed.pop(key)
        elif kind=='shape':changed[key]=changed[key][:1]
        elif kind=='dtype':changed[key]=changed[key].astype(np.float16)
        else:changed[key].flat[0]=np.nan
        with pytest.raises(ValueError):_AdapterReadout(path,save(changed),prepared.base)
