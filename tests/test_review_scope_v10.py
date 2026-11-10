import hashlib
import json

import pytest

np = pytest.importorskip('numpy')

from roguard.assurance import policy_digest
from roguard.gemma4_contract import BASE_MODEL, BASE_REVISION, POLICY_SHA256
from roguard.gemma4_review import FORMAT, LORA
from roguard.gemma4_review_v10 import Gemma4StructuredV10Model
from roguard.review_scope_v10 import correct_logits, cues


def test_scope_reader_separates_outer_claim_from_quoted_words():
    text = 'Poate că este așa: «Un prieten spune că nu știu.» Asta mă privește pe mine. Asta e situația de azi.'
    assert cues(text, 'ro') == {'uncertain': 'poate că', 'role': 'self', 'time': 0}
    assert cues('Așa se simte: «Poate că nu știu.»', 'ro')['uncertain'] is None
    assert cues('Nu știu dacă se numește amenințare, dar am auzit o amenințare.', 'ro')['uncertain'] is None
    assert cues('Nu știu dacă este adevărat: «x»', 'ro')['uncertain'] == 'nu știu dacă'
    assert cues('Nu pot stabili dacă această descriere este adevărată despre mine: «x»', 'ro')['uncertain'] is not None
    assert cues('Я не можу визначити, чи цей опис відповідає дійсності щодо мене: «x»', 'uk')['uncertain'] is not None
    assert cues('Un prieten nu știe dacă este adevărat despre el: «x»', 'ro')['role'] == 'other'
    assert cues('Un prieten nu știe dacă este adevărat despre el: «x»', 'ro')['uncertain'] is not None
    assert cues('Друг не знає, чи це правда про нього: «x»', 'uk')['role'] == 'other'
    assert cues('Друг не знає, чи це правда про нього: «x»', 'uk')['uncertain'] is not None
    assert cues('Я замислююся, що було б, якби це стало правдою: «x»', 'uk')['time'] == 2
    assert cues('Citez: «Nu pot stabili dacă această descriere este adevărată despre mine.»', 'ro')['uncertain'] is None
    assert cues('Compun un dialog pentru un roman: «Asta mă privește pe mine.»', 'ro')['role'] == 'fiction'
    assert cues('Asta mă privește pe mine. «citat neterminat', 'ro')['role'] is None


def test_counterfactual_guard_keeps_parent_binary_decision():
    raw = np.zeros(19, dtype=np.float32)
    raw[6] = 3
    spec = {'language': 'ro', 'threshold': .5}
    scorer = lambda values, _: float(values[8] > values[6])
    unchanged, evidence = correct_logits(raw, text='Poate că este așa.', language='ro', spec=spec, score_fn=scorer)
    assert np.array_equal(unchanged, raw)
    assert evidence['binary_guard_rejected'] is True
    assert evidence['fields'][2] == 0
    scorer = lambda values, _: .25
    changed, evidence = correct_logits(raw, text='Poate că este așa.', language='ro', spec=spec, score_fn=scorer)
    assert changed[8] > changed[6]
    assert evidence['binary_guard_rejected'] is False
    assert evidence['fields'][2] == 2


def test_opt_in_structured_review_preserves_default_review(tmp_path, monkeypatch):
    import roguard.gemma4_review as review
    import roguard.gemma4_review_v10 as v10_review

    spec = dict(format=FORMAT, language='ro', base_model=BASE_MODEL, base_revision=BASE_REVISION,
                policy_sha256=POLICY_SHA256, policy_digest=policy_digest(), adapter_sha256='0'*64,
                mode='binary', couplings=None, threshold=.5, boundary=24, lora=LORA)
    artifact = tmp_path / 'review.json'
    artifact.write_text(json.dumps(spec))
    identity = hashlib.sha256(artifact.read_bytes()).hexdigest()
    raw = np.zeros(19, dtype=np.float32)
    raw[6] = 4
    raw[17:] = [0, 2]

    class Engine:
        def __init__(self, *args):
            pass

        def scores(self, prefixes, spec, *, return_logits=False):
            return [raw.copy() if return_logits else review._score(raw, spec) for _ in prefixes]

    monkeypatch.setattr(review, '_Engine', Engine)
    monkeypatch.setattr(v10_review, 'V10FeatureEngine', Engine)
    model = Gemma4StructuredV10Model(artifact, tmp_path / 'adapter', tmp_path / 'base', expected_sha256=identity)
    payload = {'language': 'ro', 'turns': [{'role': 'minor', 'text': 'Poate că este așa: «Nu sunt sigur.» Asta mă privește pe mine. Asta e situația de azi.'}]}
    baseline = model.predict(payload)
    structured = model.interpret(payload)
    assert baseline['predicted'] == structured['predicted'] == 1
    assert structured['fields']['assertion'] == 'UNCERTAIN'
    assert structured['fields']['subject'] == 'SELF'
    assert structured['fields']['time'] == 'CURRENT'
    assert structured['scope_evidence']['binary_guard_rejected'] is False
    assert structured['input_digest'] == baseline['input_digest']
    assert payload['turns'][0]['text'] not in json.dumps(structured, ensure_ascii=False)
