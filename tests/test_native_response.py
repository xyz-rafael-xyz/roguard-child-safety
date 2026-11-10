import copy
import json
import sys
from contextlib import contextmanager
from types import SimpleNamespace
import pytest
np = pytest.importorskip('numpy')
save = pytest.importorskip('safetensors.numpy').save
from test_native_tail import fixture as tail_fixture
from roguard import native_response as r, gemma4_response_attention as a, v10_closure


@pytest.fixture
def response_fixture(tail_fixture, tmp_path, monkeypatch):
    f = tail_fixture
    candidate = {k: np.zeros(shape, np.float32) for k, shape in r.tensor_schema().items()}
    candidate['head.last.bias'][:] = [0., -0., 2., -1., 0.5]
    paths = {'reference_spec': f.paths['spec9'], 'reference_weights': f.paths['adapter9'],
             **{k: tmp_path / ('response-' + k) for k in ('native_spec', 'native_adapter', 'manifest')}}
    paths['native_adapter'].write_bytes(save(candidate))
    spec = dict(format=a.FORMAT, base_model=r.BASE_MODEL, base_revision=r.BASE_REVISION,
                query_sha256=r.QUERY_SHA256, adapter_sha256=r.sha(paths['native_adapter'].read_bytes()),
                boundary=24, hidden=0, labels=list(r.LABELS), qualified=False)
    paths['native_spec'].write_text(json.dumps(spec))
    m = dict(format=r.FORMAT, qualified=False, selected_arm='native_attention_primary',
             native_spec_sha256=r.sha(paths['native_spec'].read_bytes()), adapter_sha256=spec['adapter_sha256'],
             reference_spec_sha256=r.sha(paths['reference_spec'].read_bytes()),
             reference_adapter_sha256=r.sha(paths['reference_weights'].read_bytes()),
             original59_spec_sha256=r.PARENT_SPEC, original59_adapter_sha256=r.PARENT_WEIGHTS,
             tensor_inventory={k: dict(dtype='float32', shape=list(v.shape), sha256=r.sha(v.tobytes())) for k,v in candidate.items()},
             runtime_sources=r.source_digests(), provenance={k:'1'*64 for k in r.PROVENANCE})
    m['provenance']['selected_checkpoint_sha256'] = m['adapter_sha256']
    paths['manifest'].write_text(json.dumps(m))
    class Closure:
        identity='a'*64
        def __init__(self): self.full=0; self.cheap=0; self.hashes={k:r.sha(p.read_bytes()) for k,p in paths.items()}
        def verify(self):
            self.full += 1
            for k,p in paths.items(): assert r.sha(p.read_bytes()) == self.hashes[k], 'Changed asset'
        def cheap_verify(self): self.cheap += 1
        def asset_path(self,k): return paths[k]
        def asset_sha(self,k): return self.hashes[k]
        def base_path(self): return tmp_path/'no-model'
    monkeypatch.setattr(v10_closure,'NativeResponseClosure',Closure)
    engines=[]; calls=[]; outputs=[]
    class Engine:
        def __init__(self,*args):
            engines.append(self); self.arrays={k:v.copy() for k,v in f.original.items()}
            self.model=SimpleNamespace(base=self.arrays, trainable_parameters=lambda:self.arrays)
            self.mx=SimpleNamespace(eval=lambda x:None, clear_cache=lambda:None)
            self.tokenizer=None; self.make_cache=None
    class Readout:
        def __init__(self,*args):
            self.adapters={k:v.copy() for k,v in candidate.items() if k.startswith('base.')}
            self.head=SimpleNamespace(weight=candidate['head.last.weight'].copy(),bias=candidate['head.last.bias'].copy())
        def logits(self,features):
            value=self.head.bias.tolist(); outputs.append(value); return value
    @contextmanager
    def replace(base,adapters):
        old={k:base[k] for k in adapters}
        base.update(adapters)
        try: yield
        finally: base.update(old)
    def features(engine,payload):
        calls.append(payload)
        draft=payload['turns'][-1]['text']
        if draft=='FAIL': raise RuntimeError('synthetic forward failure')
        return None if draft=='1025' else np.ones((1,2560),np.float32)
    monkeypatch.setattr(a,'_Engine',Engine); monkeypatch.setattr(a,'_AdapterReadout',Readout)
    monkeypatch.setattr(a,'_replace_adapters',replace); monkeypatch.setattr(a,'_features',features)
    monkeypatch.setitem(sys.modules,'mlx.utils',SimpleNamespace(tree_flatten=lambda x:list(x.items())))
    c=Closure()
    return SimpleNamespace(c=c,m=m,paths=paths,candidate=candidate,engines=engines,calls=calls,outputs=outputs)


def load(f): return r.load(enabled=True,closure=f.c,expected_closure_identity=f.c.identity)
def request(d='hello',lang='ro'): return dict(draft=d,language=lang,context='explicit')
def rewrite(f):
    f.paths['manifest'].write_text(json.dumps(f.m)); f.c.hashes['manifest']=r.sha(f.paths['manifest'].read_bytes())


def test_disabled_exact_parent_without_access():
    p=object()
    assert r.load(enabled=False,configured_parent=p,closure=object()) is p


def test_exact74_geometry():
    schema=r.tensor_schema(); assert len(schema)==74
    assert schema['head.last.weight']==[5,2560] and sum(k.startswith('base.') for k in schema)==72


@pytest.mark.parametrize('mutation',['unknown','source','selected','reference','tensor','spec'])
def test_contract_rejects_before_lazy_constructor(response_fixture,monkeypatch,mutation):
    f=response_fixture
    if mutation=='unknown': f.m['extra']=0
    elif mutation=='source': f.m['runtime_sources']['runtime']='0'*64
    elif mutation=='selected': f.m['provenance']['selected_checkpoint_sha256']='0'*64
    elif mutation=='reference': f.m['reference_spec_sha256']='0'*64
    elif mutation=='tensor': f.m['tensor_inventory']['head.last.bias']['sha256']='0'*64
    else:
        spec=json.loads(f.paths['native_spec'].read_text());spec['hidden']=32
        f.paths['native_spec'].write_text(json.dumps(spec));f.c.hashes['native_spec']=r.sha(f.paths['native_spec'].read_bytes())
        f.m['native_spec_sha256']=f.c.hashes['native_spec']
    rewrite(f)
    monkeypatch.setattr(a.Gemma4ResponseAttentionModel,'__init__',lambda *a,**kw:pytest.fail('constructor before validation'))
    with pytest.raises(ValueError):load(f)
    assert not f.engines


def test_real_public_method_mixed_batch_and_boundaries(response_fixture):
    f=response_fixture; model=load(f); start=f.c.full
    assert not f.engines
    rows=[request('636'),request('637','uk'),request('1024'),request('1025','uk')]
    results=model.review_response_detailed_many(rows)
    assert f.c.full-start==2 and len(f.engines)==1 and len(f.calls)==4
    assert [x.response['language'] for x in results]==['ro','uk','ro','uk']
    assert [x.response['abstained'] for x in results]==[False,False,False,True]
    assert results[0].logits==tuple(f.outputs[0]) and np.signbit(results[0].logits[1])
    assert results[-1].logits is None and all(x.response['qualified'] is False for x in results)
    assert model.review_response('636','ro','explicit')==results[0].response
    assert len(f.engines)==1


@pytest.mark.parametrize('rows',[[request('')],[dict(draft='ok',language='ro')],[request()]*97,[request('\ud800')]])
def test_invalid_request_before_engine(response_fixture,rows):
    f=response_fixture; model=load(f)
    with pytest.raises((ValueError,UnicodeError)): model.review_response_many(rows)
    assert not f.engines


def test_failure_restores_all_arrays_and_poisoned_batch(response_fixture):
    f=response_fixture; model=load(f)
    with pytest.raises(RuntimeError,match='forward failure'):model.review_response_many([request(),request('FAIL'),request()])
    assert len(f.calls)==2 and model._closed and type(model.parent._readout) is a._AdapterReadout
    assert model._native_state()
    with pytest.raises(ValueError):model.review_response('hello','ro')


@pytest.mark.parametrize('mutation',['proof','original_proof','asset','parent'])
def test_bound_state_mutation_rejects(response_fixture,mutation):
    f=response_fixture; model=load(f)
    if mutation=='proof': model._proof['head.last.bias']['sha256']='0'*64
    elif mutation=='original_proof': model._original_proof.clear()
    elif mutation=='asset': f.paths['manifest'].write_text('{}')
    else: model.parent._spec['hidden']=1
    with pytest.raises((ValueError,AssertionError)):model.review_response('hello','ro')
    assert not f.engines


def test_closing_failure_no_success(response_fixture,monkeypatch):
    f=response_fixture;model=load(f); original=model._one
    def row(rq):
        out=original(rq);f.paths['manifest'].write_text('{}');return out
    monkeypatch.setattr(model,'_one',row)
    with pytest.raises(AssertionError):model.review_response_many([request()])
    assert model._closed


def test_original_public_probabilities_get_identical_readout_object(response_fixture,monkeypatch):
    f=response_fixture;model=load(f);original=a._probabilities;seen=[]
    def probabilities(values):
        assert values is f.outputs[-1];seen.append(values);return original(values)
    monkeypatch.setattr(a,'_probabilities',probabilities)
    observed=model.review_response_detailed('draft','uk','context')
    ordinary=model.parent.review_response('draft','uk','context')
    assert observed.response==ordinary and len(seen)==2 and len(f.calls)==2


def test_unused_original_head_corruption_rejected_after_forward(response_fixture,monkeypatch):
    f=response_fixture;model=load(f);original=a._features
    def features(engine,payload):
        out=original(engine,payload)
        engine.arrays['heads.uk.last.bias'][0]=1
        return out
    monkeypatch.setattr(a,'_features',features)
    with pytest.raises(ValueError,match='original80'):model.review_response('draft','ro')
    assert model._closed
