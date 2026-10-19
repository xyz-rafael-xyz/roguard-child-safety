from copy import deepcopy
import hashlib
import json
import weakref

import pytest

from roguard.language_bundle import LanguageReviewModel
from roguard.native_review import prepare_input


def write(path, value):
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup(tmp_path, monkeypatch):
    entries = {}
    for language in ('ro', 'uk'):
        path = tmp_path / (language + '.json')
        digest = write(path, dict(format='test-local-model', language=language))
        entries[language] = dict(path=path.name, sha256=digest, device='cpu')
    path = tmp_path / 'languages.json'
    digest = write(path, dict(format='roguard-language-review-bundle-v1', languages=entries))
    calls, refs = [], []

    class Model:
        def __init__(self, language):
            self.language = language
            self.identity = language * 32

        def predict_many(self, payloads, *, focuses):
            calls.append(('predict', self.language, deepcopy(payloads), focuses))
            values = []
            for payload, focus in zip(payloads, focuses):
                _, selected, identity = prepare_input(payload, focus)
                values.append(dict(language=self.language, focus=selected, input_digest=identity,
                    model_identity=self.identity, score=.25, predicted=0, decision='no_signal',
                    text_in_report=False, establishes_safety=False,
                    permits_external_action=False, permits_notification=False))
            return values

    def load(path, *, expected_sha256, device):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256
        assert all(ref() is None for ref in refs), 'Previous model still retained during load'
        language = json.loads(path.read_text())['language']
        model = Model(language)
        refs.append(weakref.ref(model))
        calls.append(('load', language, device))
        return model

    monkeypatch.setattr('roguard.model_bundle.load_bundle', load)
    return LanguageReviewModel(path, expected_sha256=digest), calls, path


def payload(language='ro', text='Bună.'):
    return dict(language=language, turns=[dict(role='minor', text=text)])


def test_mixed_batch_preserves_order_and_loads_one_model_per_language(tmp_path, monkeypatch):
    model, calls, _ = setup(tmp_path, monkeypatch)
    rows = [payload('uk', 'Вітаю.'), payload(), payload('uk', 'Добрий день.')]
    actual = model.predict_many(rows)
    assert [r['language'] for r in actual] == ['uk', 'ro', 'uk']
    assert [r['input_digest'] for r in actual] == [prepare_input(r)[2] for r in rows]
    assert [c[:2] for c in calls if c[0] == 'load'] == [('load', 'uk'), ('load', 'ro')]
    assert all(r['score'] == .25 and r['predicted'] == 0 for r in actual)
    model.predict(payload())
    assert len([c for c in calls if c[0] == 'load']) == 2


@pytest.mark.parametrize('bad', [payload('en'), {}, dict(language='uk', turns=[]), None])
def test_entire_batch_checked_before_any_model_load(tmp_path, monkeypatch, bad):
    model, calls, _ = setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        model.predict_many([payload(), bad])
    assert not calls


def test_future_turns_never_copied_or_forwarded(tmp_path, monkeypatch):
    model, calls, _ = setup(tmp_path, monkeypatch)
    class Forbidden:
        def __deepcopy__(self, memo):
            pytest.fail('Future turn was accessed')
    row = payload()
    row['turns'].append(Forbidden())
    result = model.predict(row, focus=0)
    assert result['focus'] == 0
    assert len(calls[-1][2][0]['turns']) == 1


@pytest.mark.parametrize('field,value', [
    ('language', 'uk'), ('input_digest', '0' * 64), ('model_identity', '0' * 64),
    ('focus', 5), ('permits_notification', True), ('establishes_safety', True),
])
def test_wrong_output_binding_or_claim_is_rejected(tmp_path, monkeypatch, field, value):
    model, _, _ = setup(tmp_path, monkeypatch)
    engine = model._load('ro')
    original = engine.predict_many
    def corrupt(*args, **kwargs):
        rows = original(*args, **kwargs)
        rows[0][field] = value
        return rows
    monkeypatch.setattr(engine, 'predict_many', corrupt)
    with pytest.raises(ValueError):
        model.predict(payload())


def test_incomplete_output_rejected(tmp_path, monkeypatch):
    model, _, _ = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(model._load('ro'), 'predict_many', lambda *a, **k: [])
    with pytest.raises(ValueError, match='Incomplete'):
        model.predict(payload())


def test_child_manifest_changed_after_configuration_is_rejected(tmp_path, monkeypatch):
    model, _, _ = setup(tmp_path, monkeypatch)
    (tmp_path / 'ro.json').write_text('{}')
    # The real bundle loader rejects the hash before loading any model.
    from importlib import reload
    import roguard.model_bundle
    reload(roguard.model_bundle)
    with pytest.raises(ValueError, match='identity'):
        model.predict(payload())


def test_symlink_swap_cannot_escape_bundle_root(tmp_path, monkeypatch):
    model, calls, _ = setup(tmp_path, monkeypatch)
    child = tmp_path / 'ro.json'
    child.unlink()
    child.symlink_to(tmp_path.parent / 'elsewhere.json')
    with pytest.raises(ValueError, match='escapes'):
        model.predict(payload())
    assert not calls


@pytest.mark.parametrize('mutation', ['missing_language', 'nested', 'escape', 'duplicate', 'tamper'])
def test_invalid_manifests_rejected_before_loading(tmp_path, monkeypatch, mutation):
    _, calls, path = setup(tmp_path, monkeypatch)
    spec = json.loads(path.read_text())
    if mutation == 'missing_language':
        del spec['languages']['uk']
    elif mutation == 'nested':
        spec['languages']['uk']['sha256'] = write(tmp_path / 'uk.json', dict(format='roguard-language-review-bundle-v1'))
    elif mutation == 'escape':
        spec['languages']['uk']['path'] = '../uk.json'
    digest = write(path, spec)
    if mutation == 'duplicate':
        path.write_text('{"format":"a","format":"b"}')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    elif mutation == 'tamper':
        path.write_text('{}')
    with pytest.raises(ValueError):
        LanguageReviewModel(path, expected_sha256=digest)
    assert not calls
