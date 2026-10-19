from io import BytesIO
import hashlib
import json
from pathlib import Path

import pytest

from roguard import model_install
from scripts import prepare_gemma4_model_download as build

URL = 'https://github.com/xyz-rafael-xyz/roguard-child-safety/releases/download/models-test'


def test_split_upstream_and_release_assets_install_offline_after_fetch(tmp_path, monkeypatch):
    upstream = b'unchanged-upstream-base'
    monkeypatch.setattr(build, 'FILES', {'model.safetensors': dict(bytes=len(upstream), sha256=hashlib.sha256(upstream).hexdigest())})
    payload = build.Payload()
    payload.json('bundle.json', {'fixture': True})
    payload.add('ro-adapter.safetensors', b'adapter')
    payload.add('uk-adapter.safetensors', b'adapter')
    output = tmp_path/'staged'
    spec = build.stage(payload, output, 'test', URL, ['https://example.org/terms'])
    raw = (output/'model-download-test.json').read_bytes()
    identity = hashlib.sha256(raw).hexdigest()
    calls = []
    def download(url):
        calls.append(url)
        if url.startswith('https://huggingface.co/'):
            assert '/resolve/'+build.BASE_REVISION+'/' in url
            return BytesIO(upstream)
        assert url.startswith(URL+'/')
        return BytesIO((output/'assets'/url.rsplit('/', 1)[1]).read_bytes())
    monkeypatch.setattr(model_install, '_download', download)
    bundle, _ = model_install.install(raw, identity, tmp_path/'cache', download=True)
    assert (bundle.parent/'base/model.safetensors').read_bytes() == upstream
    assert len(calls) == 3  # Identical adapter assets are reused locally.
    monkeypatch.setattr(model_install, '_download', lambda _: pytest.fail('Offline installation used network'))
    model_install.install(raw, identity, tmp_path/'cache')
    assert len(list((output/'assets').iterdir())) == 2
    assert spec['bundle']['sha256'] == hashlib.sha256(bundle.read_bytes()).hexdigest()
    assert json.loads((output/'model-profiles.json').read_text())['profiles']['test']['runtime'] == 'apple-silicon'
    assert json.loads((output/'staging-report.json').read_text())['qualifies_for_release'] is False


def test_tampered_assets_and_base_overrides_leave_no_staging_output(tmp_path):
    source = tmp_path/'adapter'
    source.write_bytes(b'original')
    payload = build.Payload()
    payload.json('bundle.json', {})
    payload.add('adapter', source)
    source.write_bytes(b'altered')
    with pytest.raises(ValueError, match='changed during staging'):
        build.stage(payload, tmp_path/'staged', 'test', URL, ['https://example.org/terms'])
    assert not (tmp_path/'staged').exists()
    assert not list(tmp_path.glob('roguard-gemma4-assets-*'))
    payload.add('base/model.safetensors', b'override')
    with pytest.raises(ValueError, match='only from the pinned'):
        build.stage(payload, tmp_path/'staged', 'test', URL, ['https://example.org/terms'])


def test_gemma4_attribution_preserves_both_pinned_model_cards():
    payload = build.Payload()
    terms = build.add_terms(payload, Path(__file__).resolve().parents[1]/'models/v9/licenses')
    assert 'https://ai.google.dev/gemma/docs/gemma_4_license' in terms
    for name in ('gemma4-google-README.md', 'gemma4-mlx-README.md', 'Apache-2.0.txt'):
        source, size, identity = payload.files['licenses/'+name]
        assert source.stat().st_size == size and build.digest(source) == identity


def test_bundle_inventory_binds_adapters_and_rejects_changed_base(tmp_path, monkeypatch):
    from roguard.gemma4_review import LORA
    from training.stage_shared_tail_bundle_v9 import review_metadata
    base = tmp_path/'base'
    base.mkdir()
    (base/'model.safetensors').write_bytes(b'fixture base')
    monkeypatch.setattr(build, 'FILES', {'model.safetensors': dict(bytes=12, sha256=build.digest(base/'model.safetensors'))})
    languages = {}
    for language in ('ro', 'uk'):
        adapter = tmp_path/(language+'-adapter.safetensors')
        adapter.write_bytes((language+' fixture adapter').encode())
        artifact = tmp_path/(language+'-review.json')
        artifact.write_bytes(build.encoded(review_metadata(dict(base_revision=build.BASE_REVISION,
            boundary=24, lora=LORA, language=language, checkpoint_sha256=build.digest(adapter),
            mode='binary', couplings=None, threshold=.25))))
        child = tmp_path/(language+'-bundle.json')
        child.write_bytes(build.encoded(dict(format='roguard-gemma4-tail-review-bundle-v1',
            artifact_sha256=build.digest(artifact), assets=dict(artifact=artifact.name, adapter=adapter.name, base='base'))))
        languages[language] = dict(path=child.name, sha256=build.digest(child), device='mps')
    bundle = tmp_path/'bundle.json'
    bundle.write_bytes(build.encoded(dict(format='roguard-language-review-bundle-v1', languages=languages)))
    payload = build.bundle_payload(bundle, build.digest(bundle))
    assert len(payload.files) == 8 and 'runtime-constraints.txt' in payload.files
    assert not any(name.startswith('base/') for name in payload.files)
    (base/'model.safetensors').write_bytes(b'altered base')
    with pytest.raises(ValueError, match='Pinned base differs'):
        build.bundle_payload(bundle, build.digest(bundle))
