import copy
import types
import pytest
from test_v10_bundle import fixture, write, payload, Factory, reset_lease
from roguard import v10_bundle as b


def native_spec(tmp_path):
    path, spec = fixture(tmp_path)
    spec['format'] = b.NATIVE_FORMAT; spec['structured']['backend'] = 'native_tail19'
    write(path, spec)
    return path, spec


def test_explicit_bundle2_and_shared_single_encoder_lease(tmp_path):
    path, spec = native_spec(tmp_path); factory = Factory()
    obj = b.V10Bundle(path, b._sha(path), _factory=factory)
    try:
        assert obj.profile['format'] == b.NATIVE_FORMAT and not obj.profile['qualified']
        result = obj.predict_many([payload('uk', 'one'), payload('ro', 'two'), payload('uk', 'three')])
        assert [r['text'] for r in result] == ['one', 'two', 'three']
        obj.review_response('Draft', 'ro', context='Explicit context')
        assert factory.live == 1 and factory.events[-2:] == [('close', 'structured', 'uk'), ('activate', 'response', 'ro')]
    finally: obj.close()
    assert factory.live == 0


@pytest.mark.parametrize('change', ['missing_backend', 'wrong_backend', 'backend_in_bundle1'])
def test_backend_never_inferred_from_assets(tmp_path, change):
    path, spec = native_spec(tmp_path)
    if change == 'missing_backend': del spec['structured']['backend']
    elif change == 'wrong_backend': spec['structured']['backend'] = 'residual17'
    else: spec['format'] = b.FORMAT
    write(path, spec); factory = Factory()
    with pytest.raises(ValueError): b.V10Bundle(path, b._sha(path), _factory=factory)
    assert not factory.events


def test_actual_prepare_requires_native_kind_and_receipt(tmp_path, monkeypatch):
    from roguard import v10_closure
    from roguard.response_offset import closure as response_closure
    path, spec = native_spec(tmp_path); directory = tmp_path/'local'; directory.mkdir(); calls = []
    def rebase(path, **kwargs):
        kind = 'native_tail' if 'structured' in path.name else 'response'; calls.append(kind)
        return types.SimpleNamespace(path=directory/(kind+'.json'), sha256='a'*64, receipt_sha256='b'*64, kind=kind)
    monkeypatch.setattr(v10_closure, 'rebase_closure', rebase)
    monkeypatch.setattr(v10_closure, 'NativeTailClosure', lambda *a: types.SimpleNamespace(identity='b'*64))
    monkeypatch.setattr(response_closure, 'Closure', lambda *a: types.SimpleNamespace(identity='b'*64))
    result = b._ActualFactory().prepare(tmp_path, spec, directory)
    assert calls == ['native_tail', 'response'] and set(result) == {'structured', 'response'}
    monkeypatch.setattr(v10_closure, 'NativeTailClosure', lambda *a: types.SimpleNamespace(identity='c'*64))
    with pytest.raises(ValueError, match='receipt'): b._ActualFactory().prepare(tmp_path, spec, directory)


def test_actual_native_activation_interface_and_nested_cleanup(tmp_path, monkeypatch):
    pytest.importorskip('numpy')
    from roguard import native_tail, native_tail_structured, trajectory_model, gemma4_contract
    path, spec = native_spec(tmp_path); events=[]
    class Closure:
        def verify(self): events.append('verify')
        def asset_path(self, name): return tmp_path / name.removeprefix('base:')
        def asset_sha(self, name): return 'a'*64
        def language_view(self, lang):
            events.append(('language_view', lang)); return types.SimpleNamespace(identity='b'*64)
    closure=Closure(); component=object.__new__(native_tail.NativeTail)
    def load(**kwargs):
        assert kwargs['enabled'] is True and kwargs['expected_closure_identity'] == kwargs['closure'].identity
        events.append('load'); return component
    def close(self): events.append('native_close')
    def wrapper(components, selector, closure, **kwargs):
        assert components == {'ro':component} and kwargs == dict(active_language='ro', mode='direct-first', budget=1200)
        events.append('wrapper'); return types.SimpleNamespace(identity=spec['structured']['pipeline_digest'], components=dict(components))
    monkeypatch.setattr(native_tail, 'load', load); monkeypatch.setattr(native_tail.NativeTail, 'close', close)
    monkeypatch.setattr(native_tail_structured, 'NativeTailStructuredWrapper', wrapper)
    monkeypatch.setattr(trajectory_model, 'TrajectoryModel', lambda *a, **k: object())
    monkeypatch.setattr(b.shutil, 'disk_usage', lambda p: types.SimpleNamespace(free=64*1024**3))
    backend=b._ActualFactory().activate('structured', 'ro', spec, {'structured':closure,'response':closure})
    assert events.index('load') < events.index('wrapper'); backend.close(); assert 'native_close' in events
    assert backend.component is None and backend.wrapper is None


def test_actual_cleanup_still_releases_native_after_closure_failure(tmp_path, monkeypatch):
    from roguard import native_tail
    events=[]; component=object.__new__(native_tail.NativeTail)
    def bad_verify(): raise ValueError('closing closure failed')
    backend=b._Backend(types.SimpleNamespace(verify=bad_verify)); backend.component=component
    monkeypatch.setattr(native_tail.NativeTail, 'close', lambda self: events.append('closed'))
    with pytest.raises(BaseExceptionGroup): backend.close()
    assert events == ['closed'] and backend.component is None
