"""Tiny CPU-native tensors only; no encoder or real model weights."""
import importlib.util
from pathlib import Path
import pytest
np=pytest.importorskip('numpy')
mx=pytest.importorskip('mlx.core')
nn=pytest.importorskip('mlx.nn')
from mlx.utils import tree_flatten
from roguard import experimental_role_residual as m

def test_actual_mlx_cpu_head_copy_install():
 old=mx.default_device();mx.set_default_device(mx.cpu)
 try:
  class Head(nn.Module):
   def __init__(self):super().__init__();self.first=nn.Linear(3,2);self.last=nn.Linear(2,19)
   def __call__(self,q):return self.last(nn.gelu(self.first(q)))
  class Model(nn.Module):
   def __init__(self):super().__init__();self.heads={'ro':Head(),'uk':Head()};self.base=nn.Linear(3,3)
  model=Model();model.eval();mx.eval(model.parameters());before={k:np.asarray(v).copy() for k,v in tree_flatten(model.heads['ro'].parameters())}
  weights={'heads.ro.'+k:np.full(v.shape,.25,np.float32) for k,v in before.items()}
  original,proof=m.install75_and_retain_v9(model,'ro',weights,np,mx,tree_flatten)
  assert all(np.asarray(v).tobytes()==before[k].tobytes() for k,v in tree_flatten(original.parameters()))
  q=mx.ones((1,3));expected=original(q);mx.eval(expected);assert np.isfinite(np.asarray(expected)).all()
  assert all(np.asarray(v).tobytes()==weights['heads.ro.'+k].tobytes() for k,v in tree_flatten(model.heads['ro'].parameters()))
  assert proof['frozen_base_values_hashed'] is False
 finally:mx.clear_cache();mx.set_default_device(old)

def test_actual_mlx_cpu_global_pool_axis():
 old=mx.default_device();mx.set_default_device(mx.cpu)
 try:
  raw=np.arange(1*2*400*512,dtype=np.float32).reshape(1,2,400,512)/1024
  positions=[[387,389],[388,390],[]]
  actual=m.value_pools(mx.array(raw),positions,mx,np)
  expected=np.stack([raw[0,:,p,:].mean(0).reshape(1024) if p else np.zeros(1024,np.float32) for p in positions])
  assert actual.tobytes()==expected.tobytes()
 finally:mx.clear_cache();mx.set_default_device(old)
