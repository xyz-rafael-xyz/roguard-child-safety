import gzip
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
import time

import pytest
import roguard.native_review as runtime


def sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def assets(tmp_path):
    base = tmp_path/'base'
    base.mkdir()
    files = {'config.json': b'{"model_type":"bert"}',
             'tokenizer_config.json': b'{}', 'vocab.txt': b'[PAD]\n[UNK]\n'}
    for name, data in files.items():
        (base/name).write_bytes(data)
    inventory = {name: dict(bytes=len(data), sha256=sha(data)) for name, data in files.items()}
    inventory['model.safetensors'] = dict(bytes=1, sha256='0'*64)
    # The original pretrained weights are deliberately absent; this runtime
    # loads the full selected checkpoint, not a second copy of the base model.
    model, revision, _, _ = runtime.BASES['ro']
    manifest = tmp_path/'base-manifest.json'
    manifest.write_text(json.dumps({'models': {'ro': dict(model=model, revision=revision, files=inventory)}}))
    checkpoint = tmp_path/'selected.safetensors'
    checkpoint.write_bytes(b'fixture only; never used for neural inference')
    spec = dict(format='roguard-native-full-review-experiment-v1', base=model, revision=revision,
                language='ro', thresholds={'ro': .5}, checkpoint_sha256=sha(checkpoint.read_bytes()),
                base_manifest_sha256=sha(manifest.read_bytes()), preprocessing=runtime.PREPROCESSING)
    artifact = tmp_path/'review.json.gz'
    artifact.write_bytes(gzip.compress(json.dumps(spec).encode()))
    return dict(artifact=artifact, checkpoint=checkpoint, base=base, base_manifest=manifest,
                expected_artifact_sha256=sha(artifact.read_bytes()))


def payload(text='Ştiu să citesc.'):
    return dict(language='ro', turns=[dict(role='minor', text=text)])


def test_preprocessing_ignores_future_content_and_preserves_sentence_case():
    original = payload()
    pair, focus, identity = runtime.prepare_input(original)
    extended = dict(original, turns=original['turns']+[{'invalid': 'future data'}])
    assert runtime.prepare_input(extended, 0) == (pair, focus, identity)
    assert pair == ('FOCUS minor: Știu să citesc.', 'Știu să citesc.')
    with pytest.raises(ValueError, match='focal turn index'):
        runtime.prepare_input(original, True)
    with pytest.raises(ValueError, match='minor message'):
        runtime.prepare_input(dict(language='ro', turns=[dict(role='assistant', text='Text')]))


def test_tampered_artifact_and_unregistered_tokenizer_are_rejected(assets):
    runtime.NativeReviewModel(**assets)
    with pytest.raises(ValueError, match='artifact hash'):
        runtime.NativeReviewModel(**dict(assets, expected_artifact_sha256='0'*64))
    (assets['base']/'tokenizer.json').write_text('{}')
    with pytest.raises(ValueError, match='Unregistered native tokenizer'):
        runtime.NativeReviewModel(**assets)


def test_invalid_prefix_is_rejected_before_copying_caller_objects():
    class MustNotCopy:
        def __deepcopy__(self, memo):
            raise AssertionError('Invalid text must be rejected before copying')
    with pytest.raises(ValueError, match='characters'):
        runtime.prepare_input(payload(MustNotCopy()))
    with pytest.raises(ValueError, match='12 turns'):
        runtime.prepare_input(dict(language='ro', turns=payload()['turns']*13))


def test_verified_tokenizer_snapshot_does_not_follow_later_cache_changes(assets, monkeypatch):
    model = runtime.NativeReviewModel(**assets)
    (assets['base']/'vocab.txt').write_text('changed after verification')
    seen = []
    class Engine:
        def __init__(self, files, *args):
            seen.append(files['vocab.txt'])
        def scores(self, pairs):
            return [.8]*len(pairs)
    monkeypatch.setattr(runtime, '_TorchEngine', Engine)
    result = model.predict(payload())
    assert seen == [b'[PAD]\n[UNK]\n']
    assert result['predicted'] == 1
    assert result['permits_external_action'] is False
    assert 'Ştiu' not in json.dumps(result, ensure_ascii=False)


def test_overlength_abstention_stays_distinct_from_a_negative_prediction(assets, monkeypatch):
    class Engine:
        def __init__(self, *args): pass
        def scores(self, pairs): return [None, .49, .5]
    monkeypatch.setattr(runtime, '_TorchEngine', Engine)
    values = runtime.NativeReviewModel(**assets).predict_many([payload()]*3)
    assert [v['predicted'] for v in values] == [None, 0, 1]
    assert [v['decision'] for v in values] == ['clarify', 'no_signal', 'review']
    assert values[0]['review_requested'] is True
    assert all(v['establishes_safety'] is False for v in values)


@pytest.mark.parametrize('scores', [[float('nan')], [True], [1.1], [], [.5, .6]])
def test_bad_model_outputs_cannot_become_no_signal(assets, monkeypatch, scores):
    class Engine:
        def __init__(self, *args): pass
        def scores(self, pairs): return scores
    monkeypatch.setattr(runtime, '_TorchEngine', Engine)
    with pytest.raises(ValueError, match='native model'):
        runtime.NativeReviewModel(**assets).predict(payload())


def test_language_mismatch_is_rejected_before_loading_weights(assets, monkeypatch):
    def forbidden(*args):
        raise AssertionError('Must not load a model for an unsupported request')
    monkeypatch.setattr(runtime, '_TorchEngine', forbidden)
    with pytest.raises(ValueError, match='language differs'):
        runtime.NativeReviewModel(**assets).predict(dict(payload(), language='uk'))


def test_concurrent_requests_load_once_and_serialize_inference(assets, monkeypatch):
    events = dict(loads=0, active=0, maximum=0)
    class Engine:
        def __init__(self, *args):
            events['loads'] += 1
            time.sleep(.01)
        def scores(self, pairs):
            events['active'] += 1
            events['maximum'] = max(events['maximum'], events['active'])
            time.sleep(.005)
            events['active'] -= 1
            return [.8]*len(pairs)
    monkeypatch.setattr(runtime, '_TorchEngine', Engine)
    model = runtime.NativeReviewModel(**assets)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: model.predict(payload()), range(8)))
    assert events == dict(loads=1, active=0, maximum=1)
    assert all(r['predicted'] == 1 for r in results)
