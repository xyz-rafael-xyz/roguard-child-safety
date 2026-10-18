from copy import deepcopy
import gzip
import hashlib
import json

import pytest
import roguard.interpreter as runtime
from roguard.interpreter_schema import FIELDS, POLICY, aligned_tokens, chat
from roguard.evidence import ONTOLOGY


def payload():
    return {'language': 'ro', 'turns': [{'role': 'minor', 'text': 'Temă pentru școală.'}]}


def decoded(concern='0', valid=True):
    values = {field: ONTOLOGY[field][0] for field in FIELDS}
    values['concern'] = concern
    return dict(interpretation=values, conditional_field_probabilities={
        field: [float(label == values[field]) for label in ONTOLOGY[field]] for field in FIELDS},
        valid_label_tokens=valid)


@pytest.fixture
def engine(monkeypatch):
    state = {'result': decoded(), 'calls': [], 'loads': 0}
    monkeypatch.setattr(runtime, '_read_archive', lambda *args: {})
    class Engine:
        def __init__(self, *args):
            state['loads'] += 1
        def decode(self, turns):
            state['calls'].append(deepcopy(turns))
            return deepcopy(state['result'])
    monkeypatch.setattr(runtime, '_MLXEngine', Engine)
    state['model'] = runtime.InterpreterModel('unused', 'unused', expected_archive_sha256='a'*64)
    return state


def test_runtime_contract_matches_frozen_training_prompt_and_alignment():
    from training.semantic_interpreter_v9 import FIELDS as training_fields, POLICY as training_policy
    from training.semantic_interpreter_v9 import aligned_tokens as training_align, chat as training_chat
    assert FIELDS == training_fields
    assert POLICY == training_policy
    turns = [{'role': 'assistant', 'text': 'Text'}, {'role': 'minor', 'text': 's\u0326coală'}]
    row = dict(turns=turns+[{'invalid': 'ignored future'}], focus=1)
    assert chat(turns) == training_chat(row)
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return list(map(ord, json.dumps(messages)))
        def encode(self, text, **kwargs):
            return list(map(ord, text))
    values = {f: 0 for f in FIELDS}
    assert aligned_tokens(Tokenizer(), chat(turns), values) == training_align(Tokenizer(), chat(turns), values)


def test_future_content_cannot_change_interpretation_or_digest(engine):
    original = payload()
    a = engine['model'].predict(original)
    original['turns'].append({'invalid': 'future'})
    b = engine['model'].predict(original, focus=0)
    assert a == b
    assert engine['calls'] == [payload()['turns']]*2
    assert engine['loads'] == 1
    assert a['predicted'] == 0
    assert 'Temă' not in json.dumps(a, ensure_ascii=False)
    assert not a['permits_external_action'] and not a['permits_notification']


def test_message_key_order_cannot_change_prompt_or_input_binding(engine):
    a = payload()
    b = dict(language=a['language'], turns=[{'text': t['text'], 'role': t['role']} for t in a['turns']])
    assert chat(a['turns']) == chat(b['turns'])
    assert engine['model'].predict(a) == engine['model'].predict(b)


@pytest.mark.parametrize('concern', ['0', '1'])
def test_invalid_full_vocabulary_token_is_always_an_abstention(engine, concern):
    engine['result'] = decoded(concern, valid=False)
    result = engine['model'].predict(payload())
    assert result['predicted'] is None
    assert result['decision'] == 'clarify'
    assert result['review_requested'] is True
    assert result['reason'] == 'INTERPRETER_INVALID_LABEL_TOKEN'


def test_overlength_abstention_has_no_invented_interpretation(engine):
    engine['result'] = {'abstention': 'INTERPRETER_CONTEXT_LIMIT'}
    result = engine['model'].predict(payload())
    assert result['predicted'] is None and result['interpretation'] is None
    assert result['conditional_field_probabilities'] is None


@pytest.mark.parametrize('fault', ['missing_field', 'unknown_label', 'nan', 'sum', 'argmax', 'boolean_valid', 'extra_output'])
def test_malformed_decoding_cannot_become_a_decision(engine, fault):
    value = engine['result']
    if fault == 'missing_field': del value['interpretation']['subject']
    if fault == 'unknown_label': value['interpretation']['subject'] = 'UNKNOWN'
    if fault == 'nan': value['conditional_field_probabilities']['concern'] = [float('nan'), 1.]
    if fault == 'sum': value['conditional_field_probabilities']['concern'] = [.2, .2]
    if fault == 'argmax': value['conditional_field_probabilities']['concern'] = [0., 1.]
    if fault == 'boolean_valid': value['valid_label_tokens'] = 1
    if fault == 'extra_output': value['unexpected'] = 'text'
    with pytest.raises(ValueError, match='[Ii]nterpreter'):
        engine['model'].predict(payload())


def test_prefix_validation_precedes_engine_loading(engine):
    value = payload()
    value['turns'][0]['role'] = 'assistant'
    with pytest.raises(ValueError, match='minor message'):
        engine['model'].predict(value)
    assert engine['loads'] == 0


def test_archive_hash_and_bounded_decompression(tmp_path, monkeypatch):
    path = tmp_path/'bad.gz'
    path.write_bytes(gzip.compress(b'x'*301))
    monkeypatch.setattr(runtime, 'MAX_BYTES', 100)
    with pytest.raises(ValueError, match='hash'):
        runtime._read_archive(path, '0'*64)
    with pytest.raises(ValueError, match='Oversized'):
        runtime._read_archive(path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_snapshot_copies_only_pinned_bytes_and_rejects_changed_asset(tmp_path, monkeypatch):
    base = tmp_path/'base'; base.mkdir()
    config = b'{"model_type":"gemma3"}'
    (base/'config.json').write_bytes(config)
    (base/'untrusted.py').write_text('raise RuntimeError("must never be loaded")')
    monkeypatch.setattr(runtime, 'FILES', {'config.json': dict(bytes=len(config), sha256=hashlib.sha256(config).hexdigest())})
    destination = tmp_path/'snapshot'
    runtime._snapshot(base, destination)
    assert [p.name for p in destination.iterdir()] == ['config.json']
    (base/'config.json').write_bytes(config+b' ')
    with pytest.raises(ValueError, match='pinned length'):
        runtime._snapshot(base, tmp_path/'changed')
