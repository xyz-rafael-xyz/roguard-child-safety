import copy
import hashlib
import json
from pathlib import Path
import pytest
from test_v10_portable_closure import fixture, record
from roguard import v10_closure as c
from roguard import gemma4_contract as g


def native_inventory(fixture, monkeypatch):
    roots, bundle, make, freeze = fixture
    value = make('native_tail')
    base = bundle / 'base'; base.mkdir()
    for name in c.BASE:
        path = base / name; path.write_bytes(b'{}\n')
        value['assets']['base:' + name]['location']['relative_path'] = 'base/' + name
    monkeypatch.setattr(g, 'FILES', {name: record(b'{}\n') for name in c.BASE})
    monkeypatch.setattr(g, 'VERSIONS', {name: '1.0' for name in c.PACKAGES})
    return roots, bundle, value, freeze


def test_native_kind_real_rebase_verification_and_language_view(fixture, tmp_path, monkeypatch):
    roots, bundle, value, freeze = native_inventory(fixture, monkeypatch)
    path, sha = freeze(value)
    binding = c.rebase_closure(path, expected_sha256=sha, bundle_root=bundle, output_path=tmp_path/'registration.json')
    closure = c.NativeTailClosure(binding.path, binding.sha256)
    assert closure.identity == binding.receipt_sha256 and binding.kind == 'native_tail'
    assert len(closure._registration['assets']) == 20
    ro, uk = closure.language_view('ro'), closure.language_view('uk')
    assert ro.identity != uk.identity and ro.parent is uk.parent is closure
    assert ro.asset_sha('native_spec') == value['assets']['ro_native_spec']['sha256']
    assert ro.base_path() == bundle/'base'
    ro.verify(); uk.verify()
    moved = copy.deepcopy(closure._registration); moved['bundle_root'] = '/different/root'
    for asset in moved['assets'].values(): asset['path'] = '/different/root/x'
    assert c.digest(c.receipt(moved, 'native_tail')) == closure.identity


def test_native_kind_rejects_research_asset_before_discovery(fixture, tmp_path, monkeypatch):
    roots, bundle, value, freeze = native_inventory(fixture, monkeypatch)
    value['assets']['training_plan'] = copy.deepcopy(value['assets']['ro_manifest']); path, sha = freeze(value)
    monkeypatch.setattr(c, '_roots', lambda x: pytest.fail('Target discovery before finite schema'))
    with pytest.raises(ValueError, match='Exact runtime assets'):
        c.rebase_closure(path, expected_sha256=sha, bundle_root=bundle, output_path=tmp_path/'registration.json')


@pytest.mark.parametrize('change', ['asset', 'extra_native', 'registration', 'derived', 'outside_asset'])
def test_native_closing_rechecks_approved_bytes_and_boundaries(fixture, tmp_path, monkeypatch, change):
    roots, bundle, value, freeze = native_inventory(fixture, monkeypatch)
    path, sha = freeze(value); binding = c.rebase_closure(path, expected_sha256=sha, bundle_root=bundle, output_path=tmp_path/'registration.json')
    closure = c.NativeTailClosure(binding.path, binding.sha256)
    if change == 'asset': closure.asset_path('ro_manifest').write_bytes(b'changed')
    elif change == 'extra_native': (roots['mlx']/'extra.metallib').write_bytes(b'extra')
    elif change == 'registration': Path(binding.path).write_bytes(b'{}')
    elif change == 'derived': closure._approved['packages']['numpy']['version'] = '9'
    else:
        target = closure.asset_path('ro_manifest'); target.unlink(); target.symlink_to(tmp_path/'unopened')
    with pytest.raises((ValueError, FileNotFoundError)): closure.verify()


def test_native_fixed_base_contract_rejects_even_rehashed_inventory(fixture, tmp_path, monkeypatch):
    roots, bundle, value, freeze = native_inventory(fixture, monkeypatch)
    target = bundle/value['assets']['base:config.json']['location']['relative_path']; target.write_bytes(b'changed')
    value['assets']['base:config.json'].update(record(b'changed')); path, sha = freeze(value)
    binding = c.rebase_closure(path, expected_sha256=sha, bundle_root=bundle, output_path=tmp_path/'registration.json')
    with pytest.raises(ValueError, match='base asset'): c.NativeTailClosure(binding.path, binding.sha256)
