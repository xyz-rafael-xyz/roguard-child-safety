import copy
import types
import pytest
from test_v10_bundle import fixture, write, payload, Factory, reset_lease
from test_v10_portable_closure import fixture as closure_fixture, record
from roguard import v10_bundle as b, v10_closure as c, gemma4_contract as g


def native_spec(tmp_path):
    path,spec=fixture(tmp_path)
    spec['format']=b.NATIVE_RESPONSE_FORMAT;spec['structured']['backend']='native_tail19'
    spec['response']['backend']='native_attention74'
    spec['response']['model_identity']=spec['response'].pop('parent_identity')
    write(path,spec)
    return path,spec


def test_exact_response_closure_rebase(closure_fixture,tmp_path,monkeypatch):
    roots,bundle,make,freeze=closure_fixture;value=make('native_response')
    base=bundle/'base';base.mkdir()
    for name in c.BASE:
        (base/name).write_bytes(b'{}\n');value['assets']['base:'+name]['location']['relative_path']='base/'+name
    monkeypatch.setattr(g,'FILES',{name:record(b'{}\n') for name in c.BASE})
    monkeypatch.setattr(g,'VERSIONS',{name:'1.0' for name in c.PACKAGES})
    path,sha=freeze(value)
    bound=c.rebase_closure(path,expected_sha256=sha,bundle_root=bundle,output_path=tmp_path/'registration.json')
    closure=c.NativeResponseClosure(bound.path,bound.sha256)
    assert len(closure._registration['assets'])==14 and closure.identity==bound.receipt_sha256
    assert closure.base_path()==base and bound.kind=='native_response'
    with pytest.raises(ValueError):closure.language_view('ro')
    closure.asset_path('manifest').write_text('changed')
    with pytest.raises(ValueError):closure.verify()


@pytest.mark.parametrize('mutation',['backend','parent_key','old_format'])
def test_explicit_bundle3_rejects_inferred_backend(tmp_path,mutation):
    path,spec=native_spec(tmp_path)
    if mutation=='backend':spec['response']['backend']='offset'
    elif mutation=='parent_key':spec['response']['parent_identity']=spec['response'].pop('model_identity')
    else:spec['format']=b.NATIVE_FORMAT
    write(path,spec);factory=Factory()
    with pytest.raises(ValueError):b.V10Bundle(path,b._sha(path),_factory=factory)
    assert not factory.events


def test_actual_prepare_selects_separate_native74_closure(tmp_path,monkeypatch):
    path,spec=native_spec(tmp_path);directory=tmp_path/'local';directory.mkdir();calls=[]
    def rebase(path,**kw):
        kind='native_tail' if 'structured' in path.name else 'native_response';calls.append(kind)
        return types.SimpleNamespace(path=directory/kind,sha256='a'*64,receipt_sha256='b'*64,kind=kind)
    monkeypatch.setattr(c,'rebase_closure',rebase)
    for name in ['NativeTailClosure','NativeResponseClosure']:
        monkeypatch.setattr(c,name,lambda *a:types.SimpleNamespace(identity='b'*64))
    closures=b._ActualFactory().prepare(tmp_path,spec,directory)
    assert calls==['native_tail','native_response'] and set(closures)=={'structured','response'}


def test_actual_response_activation_and_cleanup(tmp_path,monkeypatch):
    pytest.importorskip('numpy')
    from roguard import native_response as r
    path,spec=native_spec(tmp_path);events=[]
    class Closure:
        identity='c'*64
        def verify(self):events.append('verify')
        def asset_path(self,key):return tmp_path/key.removeprefix('base:')
    closure=Closure();engine=types.SimpleNamespace(model=object(),tokenizer=object(),make_cache=object(),_response_prefix=object(),mx=types.SimpleNamespace(clear_cache=lambda:events.append('clear')))
    reference=types.SimpleNamespace(identity=spec['response']['reference_identity'],_engine=engine)
    parent=types.SimpleNamespace(identity=spec['response']['model_identity'],reference=reference,_readout=object())
    wrapper=types.SimpleNamespace(parent=parent,artifact_identity=spec['response']['artifact_identity'])
    def load(**kw):
        assert kw==dict(enabled=True,closure=closure,expected_closure_identity=closure.identity)
        return wrapper
    monkeypatch.setattr(r,'load',load)
    monkeypatch.setattr(b.shutil,'disk_usage',lambda p:types.SimpleNamespace(free=64*1024**3))
    backend=b._ActualFactory().activate('response','uk',spec,{'structured':closure,'response':closure})
    assert backend.parent is parent and backend.reference is reference
    backend.close()
    assert wrapper.parent is None and parent.reference is None and reference._engine is None
    assert engine.model is None and engine._response_prefix is None and 'clear' in events


def test_public_batch_single_lease_order_and_legacy_guard(tmp_path):
    pytest.importorskip('numpy')
    path,spec=native_spec(tmp_path);factory=Factory(); original=factory.activate
    def activate(*args):
        backend=original(*args)
        backend.offset=types.SimpleNamespace(review_response_many=lambda rows:[dict(row) for row in rows])
        return backend
    factory.activate=activate
    obj=b.V10Bundle(path,b._sha(path),_factory=factory)
    rows=[dict(draft='a',language='uk',context='x'),dict(draft='b',language='ro',context='y')]
    try:
        assert obj.review_response_many(rows)==rows and factory.live==1
        assert [e for e in factory.events if e[0]=='activate']==[('activate','response','uk')]
        obj.predict(payload('ro','main'))
        assert factory.live==1
    finally:obj.close()
    assert factory.live==0
    path2,spec2=fixture(tmp_path/'legacy')
    obj=b.V10Bundle(path2,b._sha(path2),_factory=Factory())
    try:
        with pytest.raises(ValueError,match='bundle3'):obj.review_response_many(rows)
    finally:obj.close()


@pytest.mark.parametrize('failure',['short','long','raises'])
def test_public_batch_failure_releases_single_lease(tmp_path,failure):
    pytest.importorskip('numpy')
    path,spec=native_spec(tmp_path);factory=Factory();original=factory.activate
    def activate(*args):
        backend=original(*args)
        def call(rows):
            if failure=='raises':raise RuntimeError('row failure')
            return [] if failure=='short' else rows*2
        backend.offset=types.SimpleNamespace(review_response_many=call)
        return backend
    factory.activate=activate;obj=b.V10Bundle(path,b._sha(path),_factory=factory)
    with pytest.raises((ValueError,RuntimeError)):
        obj.review_response_many([dict(draft='ok',language='ro',context='')])
    assert factory.live==0 and b._POISONED
    obj.close()
