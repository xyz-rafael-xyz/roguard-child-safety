import hashlib
import json
from types import SimpleNamespace

import pytest

from roguard import review_pipeline as module


def components():
    fusion = SimpleNamespace(identity='fusion', _spec=dict(language='ro', native_artifact_sha256='native',
        interpreter_archive_sha256='interpreter'), predict=lambda payload, left, right, focus=None: (left, right, focus))
    native = SimpleNamespace(identity='native', predict_many=lambda payloads, focuses=None: ['direct'] * len(payloads))
    interpreter = SimpleNamespace(identity='interpreter', predict=lambda payload, focus=None: 'semantic')
    return fusion, native, interpreter


def test_components_must_match_selected_model():
    fusion, native, interpreter = components()
    interpreter.identity = 'other'
    with pytest.raises(ValueError, match='components'):
        module.ReviewFusionPipeline(fusion, native, interpreter)


def test_batch_preserves_focus_and_rejects_mixed_language_before_inference():
    fusion, native, interpreter = components()
    model = module.ReviewFusionPipeline(fusion, native, interpreter)
    payload = dict(language='ro', turns=[dict(role='minor', text='Exemplu fictiv.')])
    assert model.predict(payload, focus=0) == ('direct', 'semantic', 0)
    native.predict_many = lambda *args, **kwargs: pytest.fail('Inference on invalid language')
    with pytest.raises(ValueError, match='language'):
        model.predict(dict(payload, language='uk'))


def test_incomplete_native_batch_cannot_reach_fusion():
    fusion, native, interpreter = components()
    native.predict_many = lambda *args, **kwargs: []
    interpreter.predict = lambda *args, **kwargs: pytest.fail('Interpreter should not run')
    model = module.ReviewFusionPipeline(fusion, native, interpreter)
    with pytest.raises(ValueError, match='Incomplete'):
        model.predict(dict(language='ro', turns=[dict(role='minor', text='Test fictiv.')]))


@pytest.mark.parametrize('asset', ['../outside', '/outside'])
def test_bundle_rejects_escaping_paths_before_model_construction(tmp_path, monkeypatch, asset):
    path = tmp_path / 'bundle.json'
    spec = dict(format='roguard-review-fusion-bundle-v1', fusion=dict(path='fusion.json', sha256='a'*64),
        components=dict(native=dict(artifact='review.json.gz', checkpoint='model.safetensors', base='base', manifest='base.json'),
                        interpreter=dict(archive='adapter.json.gz', base=asset)))
    path.write_text(json.dumps(spec))
    monkeypatch.setattr(module, 'ReviewFusion', lambda *a, **kw: pytest.fail('Constructed unsafe bundle'))
    with pytest.raises(ValueError, match='relative paths'):
        module.ReviewFusionPipeline.from_bundle(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_tampered_manifest_rejected(tmp_path):
    path = tmp_path / 'bundle.json'
    path.write_text('{}')
    with pytest.raises(ValueError, match='identity'):
        module.ReviewFusionPipeline.from_bundle(path, expected_sha256='0'*64)
