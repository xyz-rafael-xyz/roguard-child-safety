import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

np = pytest.importorskip('numpy')
save = pytest.importorskip('safetensors.numpy').save

P = Path(__file__).parent
from roguard import native_tail as r
from roguard import gemma4_review as g
from roguard.gemma4_contract import BASE_MODEL, BASE_REVISION, POLICY_SHA256
from roguard.assurance import policy_digest

class Closure:
    def __init__(self, paths):
        self.paths = paths; self.identity = 'a'*64; self.calls = 0
        self.hashes = {k:r.digest(v.read_bytes()) for k,v in paths.items()}
    def verify(self):
        self.calls += 1
        for k,p in self.paths.items():
            assert r.digest(p.read_bytes()) == self.hashes[k], 'Changed closure asset'
    def asset_path(self, key): return self.paths[key]
    def asset_sha(self, key): return self.hashes[key]
    def base_path(self): return P/'nonexistent-no-model-fixture'

@pytest.fixture
def fixture(tmp_path, monkeypatch):
    original = {k:np.zeros(v,np.float32) for k,v in r.tensor_schema().items()}
    candidate = {k:x.copy() for k,x in original.items()}
    candidate['base.model.layers.24.self_attn.q_proj.lora_a'][0,0] = 1
    candidate['heads.ro.last.bias'][0] = 3
    original['heads.ro.last.bias'][0] = -0.0
    paths = {k:tmp_path/(k+'.json' if k not in ('native_adapter','adapter9') else k+'.safetensors') for k in r.ASSETS}
    paths['adapter9'].write_bytes(save(original)); paths['native_adapter'].write_bytes(save(candidate))
    h9 = r.digest(paths['adapter9'].read_bytes()); hn = r.digest(paths['native_adapter'].read_bytes())
    spec9 = dict(format=g.FORMAT,language='ro',base_model=BASE_MODEL,base_revision=BASE_REVISION,
                 policy_sha256=POLICY_SHA256,policy_digest=policy_digest(),adapter_sha256=h9,
                 mode='policy',couplings=None,threshold=.8659504636108948,boundary=24,lora=g.LORA)
    paths['spec9'].write_text(json.dumps(spec9)); paths['native_spec'].write_text(json.dumps(dict(spec9,adapter_sha256=hn)))
    m = dict(format=r.FORMAT,qualified=False,language='ro',native_spec_sha256=r.digest(paths['native_spec'].read_bytes()),
             adapter_sha256=hn,original_v9_spec_sha256=r.digest(paths['spec9'].read_bytes()),original_v9_adapter_sha256=h9,
             tensor_inventory={k:dict(dtype='float32',shape=list(v.shape),sha256=r.digest(v.tobytes())) for k,v in candidate.items()},
             runtime_sources=r.source_digests(),routing=copy.deepcopy(r.ROUTING),provenance={k:'1'*64 for k in r.PROVENANCE})
    m['provenance']['selected_checkpoint_sha256']=hn
    paths['manifest'].write_text(json.dumps(m))
    monkeypatch.setitem(r.SPEC9,'ro',m['original_v9_spec_sha256']); monkeypatch.setitem(r.WEIGHTS9,'ro',h9)
    monkeypatch.setattr(r,'_OWNER',None)
    engines=[]; loads=[]; heads=[]
    class Head:
        def __init__(self,z): self.z=z
        def __call__(self,x):
            result=self.z[None,:].copy(); heads.append(result); return result
    class Engine:
        def __init__(self,*args):
            engines.append(self)
            self.model=SimpleNamespace(heads={'ro':Head(original['heads.ro.last.bias'])})
            self.mx=SimpleNamespace(eval=lambda x:None,clear_cache=lambda:None)
            self.tokenizer=object(); self.make_cache=object(); self.calls=0
        def scores(self,prefixes,spec):
            self.calls += 1
            out=[]
            for p in prefixes:
                if p['tokens']>1024: out.append(None); continue
                z=self.model.heads['ro'](None)
                assert z is heads[-1]
                out.append(g._score(z[0],spec))
            return out
    def weights(model,tensors):
        assert set(tensors)==set(r.tensor_schema())
        loads.append(float(tensors['base.model.layers.24.self_attn.q_proj.lora_a'][0,0]))
        model.heads={'ro':Head(tensors['heads.ro.last.bias'])}
    monkeypatch.setattr(g,'_Engine',Engine); monkeypatch.setattr(g,'_load_weights',weights)
    monkeypatch.setattr(g,'classifier_tokens',lambda tokenizer,prefix:list(range(prefix['tokens'])))
    monkeypatch.setattr(g,'prepare_input',lambda p,f:(dict(p),0,r.digest(json.dumps(p,sort_keys=True).encode())))
    c=Closure(paths)
    yield SimpleNamespace(c=c,m=m,paths=paths,original=original,candidate=candidate,engines=engines,loads=loads,heads=heads)
    if r._OWNER is not None:
        r._OWNER._poisoned=False
        r._OWNER.close()

def load(f): return r.load(enabled=True,closure=f.c,expected_closure_identity=f.c.identity)
def payload(n): return dict(language='ro',tokens=n)
def rewrite(f,key,value):
    f.paths[key].write_text(json.dumps(value)); f.c.hashes[key]=r.digest(f.paths[key].read_bytes())


def test_exact80_geometry():
    x=r.tensor_schema(); assert len(x)==80
    assert x['base.model.layers.29.self_attn.q_proj.lora_b']==[8,4096]
    assert x['base.model.layers.28.self_attn.q_proj.lora_b']==[8,2048]
    assert sum(k.startswith('heads.') for k in x)==8


def test_disabled_no_asset_or_constructor_access():
    parent=object()
    class Bomb:
        def verify(self):raise AssertionError('access')
    assert r.load(enabled=False,configured_parent=parent,closure=Bomb()) is parent


@pytest.mark.parametrize('mutation',['unknown','threshold','source','tensor','inactive','closure'])
def test_bad_contract_before_engine(fixture,mutation):
    f=fixture
    if mutation=='unknown':f.m['extra']=1;rewrite(f,'manifest',f.m)
    elif mutation=='source':f.m['runtime_sources']['runtime']='0'*64;rewrite(f,'manifest',f.m)
    elif mutation=='tensor':f.m['tensor_inventory']['heads.ro.last.bias']['sha256']='0'*64;rewrite(f,'manifest',f.m)
    elif mutation=='threshold':
        s=json.loads(f.paths['native_spec'].read_bytes());s['threshold']=.5;rewrite(f,'native_spec',s)
        f.m['native_spec_sha256']=f.c.hashes['native_spec'];rewrite(f,'manifest',f.m)
    elif mutation=='inactive':
        f.candidate['heads.uk.last.bias'][0]=1
        f.paths['native_adapter'].write_bytes(save(f.candidate));f.c.hashes['native_adapter']=r.digest(f.paths['native_adapter'].read_bytes())
        f.m['adapter_sha256']=f.c.hashes['native_adapter']
        f.m['provenance']['selected_checkpoint_sha256']=f.m['adapter_sha256']
        f.m['tensor_inventory']['heads.uk.last.bias']['sha256']=r.digest(f.candidate['heads.uk.last.bias'].tobytes())
        s=json.loads(f.paths['native_spec'].read_bytes());s['adapter_sha256']=f.m['adapter_sha256'];rewrite(f,'native_spec',s)
        f.m['native_spec_sha256']=f.c.hashes['native_spec'];rewrite(f,'manifest',f.m)
    else:f.paths['manifest'].write_text('{}')
    with pytest.raises((ValueError,AssertionError)):load(f)
    assert not f.engines


def test_boundaries_single_engine_full80_and_default_detailed(fixture):
    f=fixture;model=load(f); assert not f.engines
    results=model.predict_detailed_many([payload(n) for n in [636,637,1024,1025]])
    assert [x.route for x in results]==['native_tail','original_v9','original_v9','original_v9_unsupported']
    assert len(f.engines)==1 and len(f.heads)==3 and f.engines[0].calls==2
    assert f.loads==[1.,0.,0.]  # full80 per nonempty route group, closing original restoration
    assert results[-1].logits is None and results[-1].response['decision']=='clarify'
    assert np.signbit(np.float32(results[1].logits[0]))
    assert model.predict(payload(636))==model.predict_detailed(payload(636)).response
    assert results[1].response['model_identity']==f.m['original_v9_spec_sha256']
    for detail in results:
        spec=model._parents['native_tail' if detail.route=='native_tail' else 'original_v9']._spec
        fields=r.structured_fields(detail,spec)
        assert fields['qualified'] is False
        if detail.logits is not None:
            assert len(fields['probabilities'])==432 and abs(sum(fields['probabilities'])-1)<1e-12
    model.close(); assert r._OWNER is None and f.engines[0].model is None


def test_full_public_fallback_equals_original(fixture):
    f=fixture;model=load(f)
    got=model.predict(payload(637))
    expected=model._parents['original_v9'].predict(payload(637))
    assert got==expected


def test_single_strong_lease(fixture):
    one=load(fixture);two=load(fixture)
    one.predict(payload(636))
    with pytest.raises(ValueError,match='lease'):two.predict(payload(636))
    assert len(fixture.engines)==1
    assert two._closed
    one.close();three=load(fixture);three.predict(payload(636));assert len(fixture.engines)==2;three.close()


def test_error_no_fallback_and_head_restored(fixture,monkeypatch):
    f=fixture;model=load(f);model.predict(payload(636));eng=model._engine
    def fail(*a):raise RuntimeError('native failure')
    monkeypatch.setattr(eng,'scores',fail)
    before=len(f.loads)
    with pytest.raises(RuntimeError,match='native failure'):model.predict(payload(636))
    assert f.loads[before:]==[1.] and model._closed and eng.model is None
    assert all(p._engine is None for p in model._parents.values())


def test_closing_asset_mutation_discards_return_and_cleans(fixture,monkeypatch):
    f=fixture;model=load(f);model.predict(payload(636));eng=model._engine;old=eng.scores
    def altered(*a):
        result=old(*a);f.paths['manifest'].write_text('{}');return result
    monkeypatch.setattr(eng,'scores',altered)
    with pytest.raises(AssertionError,match='Changed'):model.predict(payload(636))
    assert model._closed and eng.model is None


def test_cleanup_failure_retains_both_causes(fixture,monkeypatch):
    model=load(fixture);model.predict(payload(636));eng=model._engine
    def fail(*a):raise RuntimeError('forward')
    def cleanup():raise RuntimeError('cleanup')
    monkeypatch.setattr(eng,'scores',fail);eng.mx.clear_cache=cleanup
    with pytest.raises(BaseExceptionGroup) as exc:model.predict(payload(636))
    assert [str(e) for e in exc.value.exceptions]==['forward','cleanup']
    assert eng.model is None and r._OWNER is model
    model.close(); assert r._OWNER is model  # Cleanup failure cannot be waived by a second close.
    eng.mx.clear_cache=lambda:None


def test_constructor_failure_poisoned_lease(fixture,monkeypatch):
    model=load(fixture)
    def fail(*a):raise RuntimeError('partial constructor')
    monkeypatch.setattr(g,'_Engine',fail)
    with pytest.raises(RuntimeError,match='partial'):model.predict(payload(636))
    model.close();assert r._OWNER is model
    with pytest.raises(ValueError,match='lease'):load(fixture).predict(payload(636))

@pytest.mark.parametrize('mutation',['asset','spec','tensor'])
def test_entry_mutation_releases_active_engine(fixture,mutation):
    f=fixture;model=load(f);model.predict(payload(636));eng=model._engine
    if mutation=='asset': f.paths['manifest'].write_text('{}')
    elif mutation=='spec': model._parents['native_tail']._spec['threshold']=.5
    else: model._candidate['heads.ro.last.bias']=np.ones(19,np.float32)
    with pytest.raises((ValueError,AssertionError)):model.predict(payload(636))
    assert eng.model is None and model._closed and r._OWNER is None


def test_own_spec_structured_binding(fixture):
    model=load(fixture);detail=model.predict_detailed(payload(636))
    assert model.structured_fields(detail)['field_argmax']['concern'] == 0
    with pytest.raises(ValueError,match='identity'):
        model.structured_fields(detail._replace(spec_sha256='f'*64))
    model._parents['native_tail']._spec['threshold']=.5
    with pytest.raises(ValueError,match='mutated'):model.structured_fields(detail)


def test_uk_legacy_binary_is_separate_from_semantic_mass():
    from roguard.trajectory_sets import WORLDS
    from roguard.evidence import _complete_action
    z=np.zeros(19,np.float32);z[17]=-10;z[18]=10
    spec=dict(mode='binary',threshold=.6608994271813364,couplings=None)
    score=g._score(z,spec)
    detail=r.DetailedPrediction(dict(score=score,predicted=1),tuple(float(x) for x in z),'native_tail','f'*64)
    result=r.structured_fields(detail,spec)
    mass=sum(p for p,w in zip(result['probabilities'],WORLDS) if _complete_action(w)!='CONTINUE')
    assert abs(mass-score)>.1 and result['semantic_probability_is_legacy_score'] is False
    assert result['binary_decision']==1
    with pytest.raises(ValueError,match='decision'):
        r.structured_fields(detail._replace(response=dict(score=score,predicted=0)),spec)


def test_tensor_missing_extra_dtype_and_nonfinite():
    arrays={k:np.zeros(v,np.float32) for k,v in r.tensor_schema().items()}
    for changed in [dict(arrays,extra=np.zeros(1,np.float32)),dict(list(arrays.items())[1:])]:
        with pytest.raises(ValueError,match='80'):r.checked_tensors(save(changed))
    for bad in [np.zeros(19,np.float64),np.full(19,np.nan,np.float32)]:
        changed=dict(arrays);changed['heads.ro.last.bias']=bad
        with pytest.raises(ValueError,match='Invalid tensor'):r.checked_tensors(save(changed))


def test_interleaved_routes_restore_input_order_and_capture_once(fixture):
    f=fixture;model=load(f)
    numbers=[1025,636,637,635,1024,1026,634]
    details=model.predict_detailed_many([payload(n) for n in numbers])
    assert f.engines[0].calls==2 and len(f.heads)==5 and f.loads==[1.,0.,0.]
    for n,detail in zip(numbers,details):
        assert detail.response['input_digest']==r.digest(json.dumps(payload(n),sort_keys=True).encode())
        assert detail.route==('native_tail' if n<=636 else 'original_v9' if n<=1024 else 'original_v9_unsupported')
        assert (detail.logits is None)==(n>1024)
    # Compare complete public metadata to individual calls after preserving initial grouped counts.
    assert [d.response for d in details]==[model.predict(payload(n)) for n in numbers]


def test_batch128_one_public_call_one_capture_per_row(fixture):
    f=fixture;model=load(f)
    details=model.predict_detailed_many([payload(500+i) for i in range(128)])
    assert len(details)==128 and f.engines[0].calls==1 and len(f.heads)==128
    assert f.loads==[1.,0.]
    with pytest.raises(ValueError,match='128'):model.predict_many([payload(500)]*129)
    assert f.engines[0].calls==1


def test_second_group_failure_discards_first_and_cleans(fixture,monkeypatch):
    f=fixture;model=load(f);model.predict(payload(636));eng=model._engine;old=eng.scores
    calls=[]
    def fail_second(prefixes,spec):
        calls.append([p['tokens'] for p in prefixes])
        if len(calls)==2:raise RuntimeError('second route failed')
        return old(prefixes,spec)
    monkeypatch.setattr(eng,'scores',fail_second)
    with pytest.raises(RuntimeError,match='second route'):model.predict_detailed_many([payload(637),payload(636)])
    assert calls==[[636],[637]] and eng.model is None and model._closed


def test_incomplete_public_result_fails_batch(fixture,monkeypatch):
    f=fixture;model=load(f);model.predict(payload(636));eng=model._engine
    monkeypatch.setattr(eng,'scores',lambda *a:[])
    with pytest.raises(ValueError,match='Incomplete'):model.predict_many([payload(630),payload(631)])
    assert eng.model is None and model._closed


def test_failed_native_traceback_does_not_retain_capture_state(fixture,monkeypatch):
    import gc,weakref
    model=load(fixture);model.predict(payload(636));eng=model._engine;refs=[]
    class NativeState: pass
    def fail(*a):
        capture_state=NativeState();refs.append(weakref.ref(capture_state))
        raise RuntimeError('captured state failure')
    monkeypatch.setattr(eng,'scores',fail)
    with pytest.raises(RuntimeError,match='captured state') as exc:model.predict(payload(636))
    gc.collect()
    assert exc.value.__traceback__ is not None and refs[0]() is None
    assert eng.model is None


def test_acyclic_exact_manifest_provenance(fixture):
    assert r.PROVENANCE=={'selected_checkpoint_sha256','training_plan_sha256','training_report_sha256'}
    f=fixture;f.m['provenance']['native_validation_report_sha256']='0'*64;rewrite(f,'manifest',f.m)
    with pytest.raises(ValueError,match='provenance'):load(f)
    assert not f.engines


def test_selected_checkpoint_must_be_actual_full80_adapter(fixture):
    f=fixture;f.m['provenance']['selected_checkpoint_sha256']='b'*64;rewrite(f,'manifest',f.m)
    with pytest.raises(ValueError,match='Selected full80'):load(f)
    assert not f.engines


@pytest.mark.parametrize('kind',['extra','tuple'])
def test_exact_public_response_batch_shape(fixture,monkeypatch,kind):
    model=load(fixture);parent=model._parents['native_tail'];old=parent.predict_many
    def changed(*args,**kwargs):
        values=old(*args,**kwargs)
        return values+[copy.deepcopy(values[0])] if kind=='extra' else tuple(values)
    monkeypatch.setattr(parent,'predict_many',changed)
    with pytest.raises(ValueError,match='cardinality'):model.predict_many([payload(600)])
    assert model._closed and fixture.engines[0].model is None


# Independent regression cases retained from the prototype review.
@pytest.fixture
def valid(fixture):
 fixture.m['provenance']['selected_checkpoint_sha256']=fixture.m['adapter_sha256'];rewrite(fixture,'manifest',fixture.m);return fixture

def test_selected_checkpoint_must_equal_loaded_adapter(valid):
 f=valid;f.m['provenance']['selected_checkpoint_sha256']='e'*64;rewrite(f,'manifest',f.m)
 with pytest.raises(ValueError,match='checkpoint'):load(f)
 assert not f.engines

def test_extra_public_result_rejects_complete_batch(valid,monkeypatch):
 f=valid;m=load(f);parent=m._parents['native_tail'];old=parent.predict_many
 def extra(*args,**kw):
  results=old(*args,**kw);return results+[results[0].copy()]
 monkeypatch.setattr(parent,'predict_many',extra)
 with pytest.raises(ValueError,match='[Bb]atch|[Cc]ount|[Oo]utput|[Rr]esponse|[Rr]esult'):m.predict_many([payload(500),payload(501)])
 assert m._closed and f.engines[0].model is None

def test_grouped_zero_raw_bytes_and_fallback_order(valid):
 f=valid;m=load(f);out=m.predict_detailed_many([payload(n) for n in (1025,637,500,1024,636)])
 assert [x.route for x in out]==['original_v9_unsupported','original_v9','native_tail','original_v9','native_tail']
 assert out[0].logits is None and f.engines[0].calls==2 and len(f.heads)==4
 import numpy as np
 assert np.asarray(out[1].logits,np.float32).tobytes()==f.original['heads.ro.last.bias'].tobytes()
 assert np.asarray(out[2].logits,np.float32).tobytes()==f.candidate['heads.ro.last.bias'].tobytes()
 m.close()

def test_input_deepcopy_prevents_caller_mutation(valid,monkeypatch):
 f=valid;m=load(f);payloads=[payload(500),payload(637)];old=m._acquire
 def mutate():
  payloads[0]['tokens']=1025;old()
 monkeypatch.setattr(m,'_acquire',mutate);out=m.predict_detailed_many(payloads)
 assert [x.route for x in out]==['native_tail','original_v9'];m.close()
