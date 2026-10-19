from io import BytesIO
import hashlib
import json
from pathlib import Path

import pytest

from scripts import prepare_portable_model_download as build
from roguard import model_install


URL = 'https://github.com/xyz-rafael-xyz/roguard-child-safety/releases/download/models-test'


def small_payload(tmp_path):
    source = tmp_path / 'weights'
    source.write_bytes(b'fixed-test-weights')
    value = build.Payload()
    value.json('bundle.json', {'format': 'test-only'})
    value.add('ro/model.safetensors', source)
    value.add('uk/model.safetensors', source)
    return value, source


def test_staged_artifacts_install_and_resume_without_original_sources(tmp_path, monkeypatch):
    payload, source = small_payload(tmp_path)
    output = tmp_path / 'staged'
    spec = build.stage(payload, output, 'test-model', URL, ['https://example.org/terms'], validate_bundle=False)
    source.unlink()
    assert len(spec['files']) == 3
    assert len(list((output / 'assets').iterdir())) == 2
    raw = (output / 'model-download.json').read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    calls = []

    def download(url):
        calls.append(url)
        assert url.startswith(URL + '/')
        return BytesIO((output / 'assets' / url.rsplit('/', 1)[1]).read_bytes())

    monkeypatch.setattr(model_install, '_download', download)
    bundle, identity = model_install.install(raw, sha, tmp_path / 'installed', download=True)
    assert identity == spec['bundle']['sha256']
    assert (bundle.parent / 'ro/model.safetensors').read_bytes() == b'fixed-test-weights'
    assert (bundle.parent / 'uk/model.safetensors').read_bytes() == b'fixed-test-weights'
    model_install.install(raw, sha, tmp_path / 'installed', download=True)
    model_install.install(raw, sha, tmp_path / 'installed')
    assert len(calls) == 2
    assert str(tmp_path) not in raw.decode()
    report = json.loads((output / 'staging-report.json').read_text())
    assert report['uploaded'] is False and report['qualifies_for_release'] is False


def test_changed_input_cannot_leave_a_publishable_staging_directory(tmp_path):
    payload, source = small_payload(tmp_path)
    source.write_bytes(b'changed-after-identity-check')
    output = tmp_path / 'staged'
    with pytest.raises(ValueError, match='changed during staging'):
        build.stage(payload, output, 'test-model', URL, ['https://example.org/terms'], validate_bundle=False)
    assert not output.exists()
    assert not list(tmp_path.glob('roguard-assets-*'))


def test_unsafe_or_colliding_paths_are_rejected_before_staging(tmp_path):
    payload, _ = small_payload(tmp_path)
    payload.add('bundle.json/nested', b'collision')
    with pytest.raises(ValueError, match='conflict'):
        build.stage(payload, tmp_path / 'staged', 'test-model', URL, ['https://example.org/terms'], validate_bundle=False)
    assert not (tmp_path / 'staged').exists()
    with pytest.raises(ValueError, match='relative paths'):
        build.Payload().add('../outside', b'bad')


def test_incomplete_or_changed_candidate_is_not_packaged(tmp_path):
    (tmp_path / 'review.json').write_bytes(b'{}')
    identity = build.digest(tmp_path / 'review.json')
    for record in ({'complete': False, 'artifact_sha256': identity},
                   {'complete': True, 'artifact_sha256': '0' * 64}):
        (tmp_path / 'development.json').write_text(json.dumps(record))
        with pytest.raises(ValueError, match='complete and bound'):
            build.completed(tmp_path, 'review.json')


def test_staging_validates_bundles_without_neural_inference(tmp_path, monkeypatch):
    payload, _ = small_payload(tmp_path)
    calls = []
    monkeypatch.setattr(build, 'runtime_extra', lambda *args: 'semantic')
    monkeypatch.setattr(build, 'load_bundle', lambda *args, **kwargs: calls.append((args, kwargs)))
    build.stage(payload, tmp_path / 'staged', 'test-model', URL, ['https://example.org/terms'])
    assert len(calls) == 1


def test_pinned_licenses_are_included_unchanged():
    root = Path(__file__).resolve().parents[1] / 'models/v9/licenses'
    for kind in ('native', 'entailment'):
        payload = build.Payload()
        terms = build.add_terms(payload, root, kind)
        assert terms and 'NOTICE.txt' in payload.files
        assert 'licenses/sources.json' in payload.files
        for name, (source, _, identity) in payload.files.items():
            if isinstance(source, Path):
                assert build.digest(source) == identity
