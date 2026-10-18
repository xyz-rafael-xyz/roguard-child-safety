import hashlib
import json

import pytest

from roguard import native_field_bundle as bundle


def manifest(tmp_path, **updates):
    spec = dict(format='roguard-native-field-bundle-v1', heads_sha256='a' * 64,
                assets=dict(heads='heads.json.gz', artifact='review.json.gz',
                            checkpoint='model.safetensors', base='base', manifest='base.json'))
    spec.update(updates)
    path = tmp_path / 'bundle.json'
    path.write_text(json.dumps(spec))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_bundle_resolves_portable_assets_without_loading_neural_model(tmp_path, monkeypatch):
    path, digest = manifest(tmp_path)
    calls = []
    monkeypatch.setattr(bundle, 'NativeFieldsModel', lambda *args, **kwargs: calls.append((args, kwargs)))
    bundle.load_bundle(path, expected_sha256=digest)
    args, kwargs = calls[0]
    assert args == tuple(tmp_path / p for p in ('heads.json.gz', 'review.json.gz', 'model.safetensors', 'base', 'base.json'))
    assert kwargs == dict(expected_heads_sha256='a' * 64, device='cpu')


def test_tampered_manifest_rejected_before_model_construction(tmp_path, monkeypatch):
    path, digest = manifest(tmp_path)
    path.write_bytes(path.read_bytes() + b' ')
    monkeypatch.setattr(bundle, 'NativeFieldsModel', lambda *a, **kw: pytest.fail('Model constructed'))
    with pytest.raises(ValueError, match='identity'):
        bundle.load_bundle(path, expected_sha256=digest)


@pytest.mark.parametrize('value', ['../outside', '/outside'])
def test_manifest_cannot_reference_external_checkpoint(tmp_path, value):
    path, _ = manifest(tmp_path)
    spec = json.loads(path.read_text())
    spec['assets']['checkpoint'] = value
    path, digest = manifest(tmp_path, **spec)
    with pytest.raises(ValueError, match='relative paths'):
        bundle.load_bundle(path, expected_sha256=digest)


def test_manifest_cannot_follow_external_base_symlink(tmp_path):
    path, digest = manifest(tmp_path)
    (tmp_path / 'base').symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match='escapes'):
        bundle.load_bundle(path, expected_sha256=digest)
