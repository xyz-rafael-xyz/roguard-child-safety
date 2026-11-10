import copy,hashlib,importlib.util,json
import pytest
from pathlib import Path
np = pytest.importorskip('numpy')
torch = pytest.importorskip('torch')
from safetensors.numpy import save
from roguard import experimental_role_residual as m

class ReferenceResidual(torch.nn.Module):
 def __init__(self):
  super().__init__();self.first=torch.nn.Linear(5638,32);self.last=torch.nn.Linear(32,17)
 def forward(self,parent,q,f,prior,metadata,arm):
  def normalize(x):
   y=x.to(torch.float64);return (y/torch.linalg.vector_norm(y,dim=-1,keepdim=True).clamp_min(1e-6)*y.shape[-1]**.5).float()
  x=torch.cat((normalize(q),normalize(torch.cat((f[:,None,:],prior),1)).flatten(1),metadata),-1)
  delta=self.last(torch.nn.functional.gelu(self.first(x),approximate='none'))
  return torch.cat((parent[:,:17]+delta,parent[:,17:]),-1)
class training:Residual=ReferenceResidual
torch.set_num_threads(2)

def fixtures(n=8):
 rng=np.random.default_rng(712);weights={k:(rng.normal(size=v)*.025).astype(np.float32) for k,v in m.SHAPES.items()}
 x=dict(parent=rng.normal(size=(n,19)).astype(np.float32),q=rng.normal(size=(n,2560)).astype(np.float32),roles=rng.normal(size=(n,3,1024)).astype(np.float32),metadata=np.tile(np.array([12/1024,7/1024,13/1024,1,1,1],np.float32),(n,1)))
 return weights,x

def manifest(tensor_sha):
 return dict(format='roguard-role-residual-experiment-1',language='ro',architecture=m.ARCHITECTURE,tensor_sha256=tensor_sha,parent=dict(original_v9_adapter_sha256='1'*64,original_v9_spec_sha256='2'*64,head75_source_sha256='3'*64,spec_sha256='4'*64,base_revision='475b9088d29754a3379866cf5aeb6b41acd313c2'),normalization=copy.deepcopy(m.NORMALIZATION),alignment_source_sha256=m.ALIGNMENT_SHA,alignment_lineage_sha256=m.ALIGNMENT_LINEAGE_SHA,runtime_source_sha256=hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest(),maximum_tokens=613,fallback=dict(above_residual_envelope='original_v9',maximum_parent_tokens=1024,on_error=False,on_low_confidence=False),provenance=dict(selected_checkpoint_sha256='5'*64,training_plan_sha256='6'*64,feature_manifest_sha256='7'*64,selection_proof_sha256='8'*64),qualified=False)

def test_torch_feature_parity_trained_nonzero_residual():
 weights,x=fixtures(64);runtime=m.ResidualArrays(weights);model=training.Residual();model.load_state_dict({k:torch.from_numpy(v.copy()) for k,v in weights.items()})
 with torch.no_grad():expected=model(torch.from_numpy(x['parent']),torch.from_numpy(x['q']),torch.from_numpy(x['roles'][:,0]),torch.from_numpy(x['roles'][:,1:]),torch.from_numpy(x['metadata']),arm='aligned').numpy()
 actual=runtime(**x);error=float(np.max(np.abs(actual.astype(np.float64)-expected)))
 assert error<=1e-5,error
 assert actual[:,17:].tobytes()==x['parent'][:,17:].tobytes()

@pytest.mark.parametrize('mutation',['zero','cancellation','extreme_q','empty'])
def test_special_normalization_matches_torch(mutation):
 weights,x=fixtures()
 if mutation=='zero':weights['last.weight'][:]=0;weights['last.bias'][:]=0
 elif mutation=='cancellation':x['roles'][:]=0
 elif mutation=='extreme_q':x['q'][:]=np.float32(1e38)
 else:x['roles'][:,1]=0;x['metadata'][:,1]=0
 model=training.Residual();model.load_state_dict({k:torch.from_numpy(v.copy()) for k,v in weights.items()});model.eval()
 with torch.no_grad():expected=model(torch.from_numpy(x['parent']),torch.from_numpy(x['q']),torch.from_numpy(x['roles'][:,0]),torch.from_numpy(x['roles'][:,1:]),torch.from_numpy(x['metadata']),arm='aligned').numpy()
 actual=m.ResidualArrays(weights)(**x);np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=0)

def test_disabled_delegates_same_objects_without_residual():
 class Parent:
  def scores(self,p,s):self.args=(p,s);return result
 result=[.25,None];p=[object()];s=object();parent=Parent();engine=m.OptionalResidualEngine(parent,enabled=False)
 assert engine.scores(p,s) is result and parent.args==(p,s)

@pytest.mark.parametrize('length,route',[(613,'residual'),(614,'original_v9'),(1024,'original_v9'),(1025,'original_v9_unsupported')])
def test_boundaries(length,route):assert m.route_for_length(length,613)==route

def test_fallback_uses_original_threshold_and_identity():
 original=dict(language='ro',threshold=.8);parent=dict(language='ro',threshold=.3)
 a=m.render_result(.5,original,'input','original',2,'roguard-gemma4-tail-review-v1');b=m.render_result(.5,parent,'input','residual',2,'roguard-role-residual-review-v1')
 assert a['predicted']==0 and b['predicted']==1 and a['model_identity']=='original'
 u=m.render_result(None,original,'input','original',2,'roguard-gemma4-tail-review-v1');assert u['predicted'] is None and u['reason']=='INPUT_EXCEEDS_MODEL_CONTEXT'

def test_artifact_roundtrip_snapshot_and_small_inventory(tmp_path):
 weights,_=fixtures();raw=save(weights);assert len(raw)<1024*1024;tp=tmp_path/'residual.safetensors';tp.write_bytes(raw);v=manifest(hashlib.sha256(raw).hexdigest());mp=tmp_path/'manifest.json';mp.write_text(json.dumps(v,indent=2));h=hashlib.sha256(mp.read_bytes()).hexdigest()
 loaded=m.ResidualArrays.from_files(mp,tp,expected_manifest_sha256=h);assert sum(x.size for x in loaded._weights.values())==181009
 changed=loaded.manifest;changed['maximum_tokens']=999;assert loaded.manifest['maximum_tokens']==613
 for k in weights:assert loaded._weights[k].tobytes()==weights[k].tobytes()
 tp.write_bytes(raw+b'bad')
 with pytest.raises(ValueError,match='bytes'):m.ResidualArrays.from_files(mp,tp,expected_manifest_sha256=h)

@pytest.mark.parametrize('bad',['extra','shape','float64','nan'])
def test_weights_reject(bad):
 weights,_=fixtures()
 if bad=='extra':weights['unexpected']=np.ones(1,np.float32)
 elif bad=='shape':weights['first.bias']=weights['first.bias'][:-1]
 elif bad=='float64':weights['last.bias']=weights['last.bias'].astype(np.float64)
 else:weights['last.bias'][0]=np.nan
 with pytest.raises(ValueError):m.ResidualArrays(weights)

@pytest.mark.parametrize('field',['parent','q','roles','metadata'])
@pytest.mark.parametrize('bad',['nan','float64'])
def test_inputs_reject(field,bad):
 weights,x=fixtures()
 if bad=='nan':x[field].flat[0]=np.nan
 else:x[field]=x[field].astype(np.float64)
 with pytest.raises(ValueError):m.ResidualArrays(weights)(**x)

@pytest.mark.parametrize('key,value',[('maximum_tokens',True),('maximum_tokens',614),('qualified',True),('architecture','other'),('alignment_source_sha256','0'*64)])
def test_manifest_mutations_reject(key,value):
 v=manifest('1'*64);v[key]=value
 with pytest.raises(ValueError):m.validate_manifest(v)

def test_fallback_boolean_must_not_be_integer():
 v=manifest('1'*64);v['fallback']['on_error']=0
 with pytest.raises(ValueError):m.validate_manifest(v)

def test_reader_cap(tmp_path):
 p=tmp_path/'big';p.write_bytes(b'x'*(1024*1024+1))
 with pytest.raises(ValueError):m.bounded(p,1024*1024)

class FakeMX:
 float32=np.float32
 def __init__(self):self.clears=0
 def array(self,x):return np.asarray(x)
 def eval(self,*x):pass
 def mean(self,*a,**kw):return np.mean(*a,**kw)
 def zeros(self,*a,**kw):return np.zeros(*a,**kw)
 def clear_cache(self):self.clears+=1

def fake_enabled(tmp_path,monkeypatch):
 import types,sys
 import roguard.gemma4_contract as contract
 import roguard.gemma4_attention as attention
 import roguard.gemma4_review as review
 class FakeTokenizer: is_fast=True
 FakeTokenizer.__call__=lambda *args:None
 mx=FakeMX();calls=[];prefix=contract.PREFIX_TOKENS
 def ids(tokenizer,p):return prefix+[11]*(int(p['turns'][-1]['text'])-len(prefix))
 monkeypatch.setattr(contract,'classifier_tokens',ids)
 monkeypatch.setattr(review,'_score',lambda z,spec:float(z[0]))
 def capture(base,suffix,cache,start):
  length=len(prefix)+suffix.shape[1];calls.append(length);v=np.zeros((1,2,length,512),np.float32)
  return dict(start=24,layers=42,length=length,previous_kvs=tuple(23 if (i+1)%6==0 else 22 for i in range(24,42)),shared={22:None,23:(v,v)})
 monkeypatch.setattr(attention,'capture_with_prefix',capture);monkeypatch.setattr(attention,'replay',lambda base,state:np.ones((1,1,2560),np.float32))
 z=np.zeros((1,19),np.float32);z[0,0]=.5
 parent=types.SimpleNamespace(mx=mx,tokenizer=FakeTokenizer(),make_cache=lambda base:[],model=types.SimpleNamespace(base=types.SimpleNamespace(model=lambda *a,**k:None),heads={'ro':lambda q:z.copy()}))
 parent.scores=lambda p,s:[.5]*len(p)
 weights,_=fixtures();weights['last.weight'][:]=0;weights['last.bias'][:]=0;res=m.ResidualArrays(weights)
 spec=dict(language='ro',mode='coupled_policy',threshold=.3,couplings={'scope':[[1]*3 for _ in range(3)],'assertion_relation':[[1]*2 for _ in range(4)]})
 oldspec=dict(spec,threshold=.8);a=tmp_path/'75.json';b=tmp_path/'v9.json';a.write_text(json.dumps(spec,indent=2));b.write_text(json.dumps(oldspec,indent=2))
 spec75=m.BoundSpec(a,hashlib.sha256(a.read_bytes()).hexdigest());spec9=m.BoundSpec(b,hashlib.sha256(b.read_bytes()).hexdigest())
 res._manifest=manifest('1'*64);res._manifest['parent']['spec_sha256']=spec75.sha256;res._manifest['parent']['original_v9_spec_sha256']=spec9.sha256;res.identity='a'*64
 alignmentfile=tmp_path/'alignment.py';alignmentfile.write_text('# SYNTHETIC ONLY\n');monkeypatch.setattr(m,'ALIGNMENT_SHA',hashlib.sha256(alignmentfile.read_bytes()).hexdigest())
 def align(p,t):return dict(classifier_token_ids=ids(t,p),classifier_ids_unchanged=True,token_truncation_applied=False,role_pool_positions=[[388,389],[390,391],[392,393]],shared_structural_role_presence=[True]*3)
 alignment=types.SimpleNamespace(__file__=str(alignmentfile),align=align)
 engine=m.OptionalResidualEngine(parent,enabled=True,residual=res,alignment=alignment,parent_spec=spec75,original_spec=spec9,original_head=lambda q:z.copy())
 return engine,calls,review,spec9

def test_mixed_route_full_outputs_match_original_public_model(tmp_path,monkeypatch):
 engine,calls,review,spec9=fake_enabled(tmp_path,monkeypatch)
 payloads=[dict(language='ro',turns=[dict(role='minor',text=str(n))]) for n in (613,614,1024,1025)]
 result=engine.predict_many(payloads);assert calls==[613,614,1024]
 assert result[0]['predicted']==1 and result[0]['model_identity']=='a'*64
 import types
 old=review.Gemma4ReviewModel.__new__(review.Gemma4ReviewModel);old._spec=spec9.value;old.identity=spec9.sha256;old.language='ro';old._engine=types.SimpleNamespace(scores=lambda prefixes,spec:[.5 if int(p['turns'][-1]['text'])<=1024 else None for p in prefixes])
 expected=old.predict_many(payloads[1:]);assert result[1:]==expected
 with pytest.raises(ValueError,match='threshold'):engine.scores(payloads,spec9.value)

def test_native_error_does_not_fallback_and_cleans(tmp_path,monkeypatch):
 engine,calls,review,spec9=fake_enabled(tmp_path,monkeypatch)
 def broken(q):raise RuntimeError('synthetic native failure')
 engine.parent.model.heads['ro']=broken
 with pytest.raises(RuntimeError,match='synthetic native failure'):engine.predict_many([dict(language='ro',turns=[dict(role='minor',text='613')])])
 assert calls==[613] and engine.parent.mx.clears>=1

def test_all_unsupported_never_forward(tmp_path,monkeypatch):
 engine,calls,_,_=fake_enabled(tmp_path,monkeypatch);result=engine.predict_many([dict(language='ro',turns=[dict(role='minor',text='1025')])]);assert calls==[] and result[0]['predicted'] is None

def test_entire_batch_preflight_before_native(tmp_path,monkeypatch):
 engine,calls,_,_=fake_enabled(tmp_path,monkeypatch)
 with pytest.raises(ValueError):engine.predict_many([dict(language='ro',turns=[dict(role='minor',text='613')]),dict(language='uk',turns=[dict(role='minor',text='613')])])
 assert calls==[]

def flat(d,prefix=''):
 out=[]
 for k,v in d.items():
  key=prefix+'.'+k if prefix else k
  out.extend(flat(v,key) if isinstance(v,dict) else [(key,v)])
 return out

def test_single_backbone_head_install_and_original_snapshot():
 import types
 class Head:
  def __init__(self):self.p={'first':{'weight':np.ones((2,3),np.float32),'bias':np.zeros(2,np.float32)},'last':{'weight':np.ones((19,2),np.float32),'bias':np.zeros(19,np.float32)}}
  def parameters(self):return self.p
  def load_weights(self,values,strict):
   for k,v in values:a,b=k.split('.');self.p[a][b]=v
 class Model:
  def __init__(self):self.heads={'ro':Head(),'uk':Head()};self.base={'fixed':np.ones(3,np.float32)}
  def parameters(self):return dict(heads={k:v.parameters() for k,v in self.heads.items()},base=self.base)
  trainable_parameters=parameters
  def eval(self):pass
 model=Model();before=copy.deepcopy(model.heads['ro'].parameters());tensors={'heads.ro.'+k:np.full(v.shape,2,np.float32) for k,v in flat(before)}
 original,proof=m.install75_and_retain_v9(model,'ro',tensors,np,FakeMX(),flat)
 assert all(v.tobytes()==dict(flat(before))[k].tobytes() for k,v in flat(original.parameters()))
 assert all(np.all(v==2) for k,v in flat(model.heads['ro'].parameters()))
 assert proof['frozen_base_values_hashed'] is False


def test_cleanup_does_not_mask_native_failure(tmp_path,monkeypatch):
 engine,calls,_,_=fake_enabled(tmp_path,monkeypatch)
 def broken(q):raise RuntimeError('original synthetic forward error')
 def bad_cleanup():raise ValueError('synthetic cleanup error')
 engine.parent.model.heads['ro']=broken;engine.parent.mx.clear_cache=bad_cleanup
 with pytest.raises(RuntimeError,match='original synthetic forward error') as caught:engine.predict_many([dict(language='ro',turns=[dict(role='minor',text='613')])])
 assert 'cleanup' in ' '.join(caught.value.__notes__)


def test_detailed_same_forward_route_response_and_immutable_snapshot(tmp_path, monkeypatch):
    engine, calls, _, spec9 = fake_enabled(tmp_path, monkeypatch)
    z = np.zeros((1, 19), np.float32)
    z[0, 0] = .5
    z[0, 17] = np.float32(-0.)
    engine.parent.model.heads['ro'] = lambda q: z.copy()
    engine.original_head = lambda q: z.copy()
    payloads = [dict(language='ro', turns=[dict(role='minor', text=str(n))])
                for n in (613, 614, 1024, 1025)]
    normal = engine.predict_many(payloads)
    calls.clear()
    detailed = engine.predict_detailed_many(payloads)
    assert calls == [613, 614, 1024]
    assert [d.response for d in detailed] == normal
    assert [d.route for d in detailed] == ['residual', 'original_v9', 'original_v9', 'original_v9_unsupported']
    for d in detailed[:3]:
        assert type(d.logits) is tuple and len(d.logits) == 19
        assert all(type(v) is float for v in d.logits)
        assert np.asarray(d.logits, np.float32).tobytes() == z[0].tobytes()
        with pytest.raises(AttributeError):
            d.route = 'changed'
        with pytest.raises(TypeError):
            d.logits[0] = 0
    assert detailed[0].spec_sha256 == engine.residual.manifest['parent']['spec_sha256']
    assert all(d.spec_sha256 == spec9.sha256 for d in detailed[1:])
    assert detailed[3].logits is None
    # Normal response dictionaries deliberately retain their familiar mutability.
    detailed[0].response['local_note'] = 'caller-owned'
    assert 'local_note' not in normal[0]


def test_default_path_never_builds_detailed_snapshot(tmp_path, monkeypatch):
    engine, calls, _, _ = fake_enabled(tmp_path, monkeypatch)
    def forbidden(*args):
        raise AssertionError('default path built detail record')
    monkeypatch.setattr(m, 'DetailedPrediction', forbidden)
    result = engine.predict_many([dict(language='ro', turns=[dict(role='minor', text='613')])])
    assert result[0]['predicted'] == 1 and calls == [613]


def test_detailed_native_exception_propagates_without_fallback(tmp_path, monkeypatch):
    engine, calls, _, _ = fake_enabled(tmp_path, monkeypatch)
    def broken(q):
        raise RuntimeError('same native failure')
    engine.parent.model.heads['ro'] = broken
    with pytest.raises(RuntimeError, match='same native failure'):
        engine.predict_detailed_many([dict(language='ro', turns=[dict(role='minor', text='613')])])
    assert calls == [613] and engine.parent.mx.clears >= 1


def test_detailed_public_convenience_and_lock(tmp_path, monkeypatch):
    engine, calls, review, _ = fake_enabled(tmp_path, monkeypatch)
    class Lock:
        active = False
        entries = 0
        def __enter__(self):
            assert not self.active
            self.active = True
            self.entries += 1
        def __exit__(self, *args):
            self.active = False
    lock = Lock()
    monkeypatch.setattr(review, '_LOCK', lock)
    original_head = engine.parent.model.heads['ro']
    def checked_head(q):
        assert lock.active
        return original_head(q)
    engine.parent.model.heads['ro'] = checked_head
    model = m.ExperimentalRoleResidualModel(engine, enabled=True)
    result = model.predict_detailed(dict(language='ro', turns=[dict(role='minor', text='613')]))
    assert result.route == 'residual' and lock.entries == 1 and not lock.active
    assert calls == [613]
