import copy
import json
from pathlib import Path

import pytest

from test_v10_portable_closure import fixture, record
from roguard import gemma4_contract as g
from roguard import v10_closure as c


def focal_inventory(fixture, monkeypatch):
    roots, bundle, make, freeze = fixture
    value = make('focal_reader')
    monkeypatch.setattr(g, 'FILES', {name: record(b'{}\n') for name in c.BASE})
    monkeypatch.setattr(g, 'VERSIONS', {name: '1.0' for name in c.PACKAGES})
    return roots, bundle, value, freeze


def test_focal_inventory_rebases_and_detects_learned_asset_mutation(fixture, tmp_path, monkeypatch):
    _, bundle, value, freeze = focal_inventory(fixture, monkeypatch)
    path, expected = freeze(value)
    bound = c.rebase_closure(path, expected_sha256=expected, bundle_root=bundle,
                             output_path=tmp_path / 'local.json')
    closure = c.FocalReaderClosure(bound.path, bound.sha256)
    assert bound.kind == 'focal_reader' and closure.identity == bound.receipt_sha256
    assert closure._registration['qualified'] is False
    for language in ('ro', 'uk'):
        for role in ('reader_weights', 'reader_stats', 'residual_weights', 'residual_stats'):
            assert closure.asset_sha(language + '_' + role) == record(b'{}\n')['sha256']
    with pytest.raises(ValueError, match='explicit language assets'):
        closure.language_view('ro')
    closure.asset_path('uk_residual_stats').write_bytes(b'changed')
    with pytest.raises(ValueError):
        closure.verify()


@pytest.mark.parametrize('change', ['missing_stats', 'legacy_weights', 'research_asset'])
def test_wrong_focal_assets_reject_before_target_discovery(fixture, tmp_path, monkeypatch, change):
    _, bundle, value, freeze = focal_inventory(fixture, monkeypatch)
    if change == 'missing_stats':
        del value['assets']['ro_reader_stats']
    elif change == 'legacy_weights':
        value['assets']['ro_native_adapter'] = value['assets'].pop('ro_residual_weights')
    else:
        value['assets']['training_rows'] = copy.deepcopy(value['assets']['ro_manifest'])
    path, expected = freeze(value)
    monkeypatch.setattr(c, '_roots', lambda _: pytest.fail('Discovery before schema rejection'))
    with pytest.raises(ValueError, match='Exact runtime assets'):
        c.rebase_closure(path, expected_sha256=expected, bundle_root=bundle,
                         output_path=tmp_path / 'local.json')


def test_focal_registration_cannot_assert_qualification(fixture, tmp_path, monkeypatch):
    _, bundle, value, freeze = focal_inventory(fixture, monkeypatch)
    path, expected = freeze(value)
    bound = c.rebase_closure(path, expected_sha256=expected, bundle_root=bundle,
                             output_path=tmp_path / 'local.json')
    registration = json.loads(Path(bound.path).read_bytes())
    registration['qualified'] = True
    raw = (c.canonical(registration) + '\n').encode()
    Path(bound.path).write_bytes(raw)
    with pytest.raises(ValueError, match='registration required'):
        c.FocalReaderClosure(bound.path, record(raw)['sha256'])
