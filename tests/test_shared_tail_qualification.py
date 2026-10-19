import hashlib
import json

import pytest

from training.shared_tail_qualification_v9 import candidate_bindings, compare_replay, scores_from_logits


def test_qualification_requires_complete_bilingual_gates_and_exact_weights(tmp_path):
    actual = tmp_path/'actual'
    actual.mkdir()
    (tmp_path/'source-freeze.json').write_text('{}')
    (tmp_path/'protocol.json').write_text('{}')
    freeze = hashlib.sha256(b'{}').hexdigest()
    selections = {}
    for lang in ('ro', 'uk'):
        checkpoint = actual/f'{lang}-selected.safetensors'
        checkpoint.write_bytes(lang.encode())
        expected = hashlib.sha256(lang.encode()).hexdigest()
        selections[lang] = dict(frontier=dict(counts_pass=True, threshold=.5), checkpoint_sha256=expected, mode='binary')
        (actual/f'{lang}-review.json').write_text(json.dumps(dict(format='roguard-shared-tail-adapter-experiment-v1',
            language=lang, source_freeze_sha256=freeze, checkpoint_sha256=expected, threshold=.5, mode='binary')))
        (actual/f'{lang}-dev-predictions.jsonl').write_text('')
    report = dict(complete=True, development_passes=True, selected=selections, source_freeze_sha256=freeze)
    report_path = actual/'report.json'
    report_path.write_text(json.dumps(report))
    specs, binding = candidate_bindings(tmp_path)
    assert set(specs) == {'ro', 'uk'}
    assert binding[str(actual/'ro-selected.safetensors')] == hashlib.sha256(b'ro').hexdigest()
    for key in ('complete', 'development_passes'):
        report_path.write_text(json.dumps(dict(report, **{key: False})))
        with pytest.raises(ValueError, match='Completed bilingual'):
            candidate_bindings(tmp_path)
    report['selected']['uk']['frontier']['counts_pass'] = False
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='Every selected language'):
        candidate_bindings(tmp_path)
    report['selected']['uk']['frontier']['counts_pass'] = True
    report_path.write_text(json.dumps(report))
    (actual/'uk-selected.safetensors').write_bytes(b'altered')
    with pytest.raises(ValueError, match='checkpoint differs'):
        candidate_bindings(tmp_path)


def test_selected_scores_are_stable_and_follow_declared_world_policy():
    np = pytest.importorskip('numpy')
    from roguard.evidence import ONTOLOGY
    from training.context_transition_v9 import generate
    from training.native_fields_v9 import FIELDS, SIZES
    from training.prepare_instruction_sft_v8 import target
    rows = generate('ro', 'dev')
    logits = np.full((len(rows), sum(SIZES)+2), -1000.)
    offset = 0
    for field, size in zip(FIELDS, SIZES):
        for i, row in enumerate(rows):
            logits[i, offset+ONTOLOGY[field].index(row['attributes'][field])] = 1000.
        offset += size
    for i, row in enumerate(rows):
        logits[i, -2+target(row)] = 1000.
    expected = np.array([target(row) for row in rows])
    couplings = dict(scope=np.ones((len(ONTOLOGY['subject']), len(ONTOLOGY['source']))).tolist(),
        assertion_relation=np.ones((len(ONTOLOGY['assertion']), len(ONTOLOGY['relation']))).tolist())
    for mode in ('binary', 'policy', 'coupled_policy'):
        assert np.array_equal(scores_from_logits(logits, dict(mode=mode, couplings=couplings)), expected)
    for bad in (logits[:, :-1], logits[:0], np.full_like(logits, np.nan)):
        with pytest.raises(ValueError, match='finite semantic'):
            scores_from_logits(bad, dict(mode='binary'))
    with pytest.raises(ValueError, match='Unknown'):
        scores_from_logits(logits, dict(mode='unknown'))


def test_development_replay_rejects_changed_identity_label_or_decision():
    expected = [dict(case_id='a', language='ro', group='formal', gold=0, score=.2, predicted=0),
                dict(case_id='b', language='uk', group='transition', gold=1, score=.8, predicted=1)]
    actual = [dict(row) for row in reversed(expected)]
    assert compare_replay(actual, expected, .5, 1e-5)['decision_changes'] == 0
    for changed in ([actual[0], actual[0]], [dict(actual[0], gold=0), actual[1]],
                    [dict(actual[0], score=.1), actual[1]], [dict(actual[0], score=float('nan')), actual[1]]):
        with pytest.raises(ValueError):
            compare_replay(changed, expected, .5, 1e-5)
