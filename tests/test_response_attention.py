"""Actual small-Gemma response-tail parity, gradients and frozen-state checks."""
import pytest


def setup():
    mx = pytest.importorskip('mlx.core'); np = pytest.importorskip('numpy')
    import mlx.nn as nn
    from mlx_lm.models.gemma4_text import Model, ModelArgs
    from roguard.gemma4_review import _make_model
    from training.response_attention_v10 import make_response_tail, zero_tail_outputs
    mx.set_default_device(mx.cpu); mx.random.seed(1048)
    base = Model(ModelArgs(hidden_size=16, num_hidden_layers=4, intermediate_size=32,
        num_attention_heads=2, head_dim=8, global_head_dim=8, num_key_value_heads=1,
        num_kv_shared_layers=2, vocab_size=64, vocab_size_per_layer_input=64,
        hidden_size_per_layer_input=8, sliding_window=4,
        layer_types=['sliding_attention', 'full_attention']*2))
    prepared = _make_model(base, boundary=2, width=12,
        lora=dict(rank=2, scale=4., dropout=0., keys=['self_attn.q_proj', 'self_attn.o_proj']))
    zero_tail_outputs(prepared.base, boundary=2)
    class Head(nn.Module):
        def __init__(self):super().__init__(); self.last=nn.Linear(16,5)
        def __call__(self,h):return self.last(h/mx.sqrt(mx.maximum(mx.mean(h*h,axis=1,keepdims=True),1e-12)))
    return mx, np, make_response_tail(prepared.base, Head(), boundary=2)


def test_response_replay_gradients_match_full_forward_and_leave_body_frozen():
    mx, np, model = setup()
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from roguard.gemma4_attention import capture
    tokens = [mx.array([[1,2,3]]), mx.array([[2,3,4,5,6]])]
    states = [capture(model.base,t,start=2) for t in tokens]
    labels = mx.array([1,4])
    def full(m):return mx.concatenate([m.head(m.base.model(t)[:,-1,:]) for t in tokens])
    def loss(m, cached):return nn.losses.cross_entropy(m(states) if cached else full(m),labels).mean()
    a,ag=nn.value_and_grad(model,loss)(model,True); b,bg=nn.value_and_grad(model,loss)(model,False)
    mx.eval(a,b,ag,bg)
    assert abs(float(a.item())-float(b.item()))<2e-5
    left,right=dict(tree_flatten(ag)),dict(tree_flatten(bg));assert set(left)==set(right)
    for name in left:assert float(mx.max(mx.abs(left[name]-right[name])).item())<3e-4,name
    trainable=set(dict(tree_flatten(model.trainable_parameters())))
    frozen={k:np.asarray(v).copy() for k,v in tree_flatten(model.parameters()) if k not in trainable}
    before={k:np.asarray(v).copy() for k,v in tree_flatten(model.trainable_parameters())}
    optim.SGD(learning_rate=.01).update(model,ag);mx.eval(model.parameters())
    assert any(not np.array_equal(np.asarray(v),before[k]) for k,v in tree_flatten(model.trainable_parameters()))
    assert all(np.array_equal(np.asarray(v),frozen[k]) for k,v in tree_flatten(model.parameters()) if k in frozen)
    actual,expected=model(states),full(model);mx.eval(actual,expected)
    assert float(mx.max(mx.abs(actual-expected)).item())<2e-5


def test_response_checkpoint_rejects_changed_inventory_and_roundtrips():
    mx,np,model=setup()
    from mlx.utils import tree_flatten
    from training.response_attention_v10 import load_response_weights
    tensors={k:np.asarray(v).copy() for k,v in tree_flatten(model.trainable_parameters())}
    load_response_weights(model,tensors)
    assert all(np.array_equal(np.asarray(v),tensors[k]) for k,v in tree_flatten(model.trainable_parameters()))
    bad=dict(tensors);bad['base.model.layers.0.fake']=np.ones((1,),dtype=np.float32)
    with pytest.raises(ValueError):load_response_weights(model,bad)
    bad=dict(tensors);bad['head.last.bias']=np.full((5,),np.nan,dtype=np.float32)
    with pytest.raises(ValueError):load_response_weights(model,bad)
