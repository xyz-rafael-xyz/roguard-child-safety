import types

import pytest

from test_native_response_bundle import native_spec
from test_v10_bundle import Factory, payload, reset_lease, write
from roguard import v10_bundle as b, v10_closure as c


def focal_spec(tmp_path):
    path, spec = native_spec(tmp_path)
    spec['format'] = b.FOCAL_READER_FORMAT
    spec['structured']['backend'] = 'focal_reader19'
    write(path, spec)
    return path, spec


@pytest.mark.parametrize('mutation', ['tail_backend', 'legacy_response', 'qualified'])
def test_focal_bundle_rejects_wrong_backend_and_qualification(tmp_path, mutation):
    path, spec = focal_spec(tmp_path)
    if mutation == 'tail_backend':
        spec['structured']['backend'] = 'native_tail19'
    elif mutation == 'legacy_response':
        spec['response']['parent_identity'] = spec['response'].pop('model_identity')
    else:
        spec['qualified'] = True
    write(path, spec)
    factory = Factory()
    with pytest.raises(ValueError):
        b.V10Bundle(path, b._sha(path), _factory=factory)
    assert not factory.events


def test_prepare_selects_focal_and_native_response_inventories(tmp_path, monkeypatch):
    _, spec = focal_spec(tmp_path)
    directory = tmp_path / 'local'; directory.mkdir()
    calls = []

    def rebase(path, **kwargs):
        kind = 'focal_reader' if 'structured' in path.name else 'native_response'
        calls.append(kind)
        return types.SimpleNamespace(path=directory / kind, sha256='a' * 64,
                                     receipt_sha256='b' * 64, kind=kind)

    monkeypatch.setattr(c, 'rebase_closure', rebase)
    for name in ('FocalReaderClosure', 'NativeResponseClosure'):
        monkeypatch.setattr(c, name, lambda *args: types.SimpleNamespace(identity='b' * 64))
    closures = b._ActualFactory().prepare(tmp_path, spec, directory)
    assert calls == ['focal_reader', 'native_response']
    assert set(closures) == {'structured', 'response'}


@pytest.mark.parametrize('bad_pipeline', [False, True])
def test_actual_focal_activation_binds_language_assets_and_closes(tmp_path, monkeypatch, bad_pipeline):
    from roguard import focal_reader_residual as runtime
    from roguard import focal_reader_structured as structured
    from roguard import trajectory_model

    _, spec = focal_spec(tmp_path)
    events = []; components = []

    class Closure:
        def verify(self): events.append('verify')
        def asset_path(self, key): return tmp_path / key.removeprefix('base:')
        def asset_sha(self, key): return 'a' * 64

    class Component:
        def __init__(self, **kwargs):
            self.arguments = kwargs
            self.closed = False
            components.append(self)

        def close(self):
            self.closed = True
            events.append('component_close')

    def wrapper(parts, selector, closure, **kwargs):
        assert set(parts) == {'uk'} and kwargs['active_language'] == 'uk'
        identity = 'f' * 64 if bad_pipeline else spec['structured']['pipeline_digest']
        return types.SimpleNamespace(identity=identity, components=dict(parts))

    monkeypatch.setattr(runtime, 'FocalReaderResidualModel', Component)
    monkeypatch.setattr(structured, 'FocalReaderStructuredWrapper', wrapper)
    monkeypatch.setattr(trajectory_model, 'TrajectoryModel', lambda *args, **kwargs: object())
    monkeypatch.setattr(b.shutil, 'disk_usage', lambda _: types.SimpleNamespace(free=64 * 1024**3))
    closure = Closure()
    if bad_pipeline:
        with pytest.raises(ValueError, match='focal structured pipeline'):
            b._ActualFactory().activate('structured', 'uk', spec,
                                       {'structured': closure, 'response': closure})
    else:
        backend = b._ActualFactory().activate('structured', 'uk', spec,
                                             {'structured': closure, 'response': closure})
        assert backend.component is components[0]
        backend.close()
        backend.close()
    assert components[0].closed and events.count('component_close') == 1
    args = components[0].arguments
    assert args['manifest'] == tmp_path / 'uk_manifest'
    assert args['parent_spec'] == tmp_path / 'uk_spec9'
    assert args['parent_adapter'] == tmp_path / 'uk_adapter9'
    for role in ('reader_weights', 'reader_stats', 'residual_weights', 'residual_stats'):
        assert args[role] == tmp_path / ('uk_' + role)


def test_focal_bundle_serializes_structured_and_response_tasks(tmp_path):
    path, _ = focal_spec(tmp_path)
    factory = Factory(); original = factory.activate

    def activate(*args):
        backend = original(*args)
        backend.offset = types.SimpleNamespace(review_response_many=lambda rows: [dict(r) for r in rows])
        return backend

    factory.activate = activate
    bundle = b.V10Bundle(path, b._sha(path), _factory=factory)
    rows = [dict(draft='Example.', language='uk', context='Context.')]
    try:
        bundle.predict(payload('ro', 'Example.'))
        assert factory.live == 1
        assert bundle.review_response_many(rows) == rows
        assert factory.live == 1 and bundle.profile['qualified'] is False
    finally:
        bundle.close()
    assert factory.live == 0
