"""Paired source alignment and serialization guard for the next encoder job."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from roguard.assurance import digest
from roguard.gemma4_contract import PREFIX_TOKENS
from roguard.trajectory_data import prefix, file_sha
from training.assertion_variation_features_v10 import validate_inputs
from scripts.wait_completed_sequence_exit_v10 import require_exit
from training.extract_assertion_variation_v10 import require_dependency


def inputs():
    treatment = []; controls = []; runtime = []; metadata = {}
    for li, lang in enumerate(('ro', 'uk')):
        fields = [0, 0, 2, 0, 0, 1]
        old = {'case_id': 'train-'+lang, 'language': lang, 'partition': 'train', 'gold': 1,
               'fields': fields, 'token_sha256': digest(PREFIX_TOKENS+[999+li])}
        metadata[lang] = [old, dict(old, case_id='dev-'+lang, partition='dev', token_sha256='f'*64)]
        parent = {k: old[k] for k in ('case_id', 'language', 'gold', 'fields', 'token_sha256')}
        parent['feature_index'] = 0
        for op in range(4):
            for pos in ('before', 'after'):
                pair = lang+str(op)+pos
                factors = {'operator_index': op, 'quote_placement': pos, 'operator_negation_marker': op >= 2}
                common = {'language': lang, 'partition': 'train', 'review': 1, 'focus': 1, 'pair_id': pair,
                          'parent': deepcopy(parent), 'factors': factors,
                          'attributes': {'subject': 'SELF', 'time': 'CURRENT', 'assertion': 'UNCERTAIN',
                                         'source': 'DIRECT', 'relation': 'NEW', 'concern': '1'}}
                changed = dict(common, case_id=pair+':treatment', arm='treatment',
                               turns=[{'role': 'assistant', 'text': 'context'}, {'role': 'minor', 'text': pair+' «core»'}])
                control = dict(common, case_id=pair+':control', arm='control',
                               turns=[{'role': 'assistant', 'text': 'context'}, {'role': 'minor', 'text': 'old «core»'}])
                tokens = PREFIX_TOKENS+[100*li+op*2+int(pos == 'after')]
                record = {'language': lang, 'partition': 'train', 'case_id': changed['case_id'], 'pair_id': pair,
                          'parent': deepcopy(parent), 'factors': factors, 'gold': 1, 'fields': fields[:],
                          'payload': deepcopy(prefix(changed, changed['focus'])), 'tokens': tokens,
                          'token_count': len(tokens), 'token_sha256': digest(tokens)}
                treatment.append(changed); controls.append(control); runtime.append(record)
    return deepcopy((treatment, controls, runtime, metadata))


def test_repeated_control_positions_are_intentional_and_pairing_is_exact():
    result = validate_inputs(*inputs())
    assert all(len(rows) == 8 and {r['parent_index'] for r in rows} == {0} for rows in result.values())


@pytest.mark.parametrize('fault', ['missing', 'dev_parent', 'position', 'parent_label', 'row_label', 'runtime_label',
                                 'partition', 'token_hash', 'prefix', 'payload', 'identity', 'context', 'core', 'factor', 'duplicate'])
def test_alignment_faults_fail_closed(fault):
    t, c, r, m = inputs()
    if fault == 'missing': c.pop()
    elif fault == 'dev_parent': m['ro'][0]['partition'] = 'dev'
    elif fault == 'position': t[0]['parent']['feature_index'] = 999
    elif fault == 'parent_label': t[0]['parent']['gold'] = 0
    elif fault == 'row_label': c[0]['review'] = 0
    elif fault == 'runtime_label': r[0]['gold'] = 0
    elif fault == 'partition': t[0]['partition'] = 'dev'
    elif fault == 'token_hash': r[0]['tokens'][-1] += 20
    elif fault == 'prefix': r[0]['tokens'][0] += 20; r[0]['token_sha256'] = digest(r[0]['tokens'])
    elif fault == 'payload': r[0]['payload']['turns'][-1]['text'] += '!'
    elif fault == 'identity': c[0]['pair_id'] = 'other'
    elif fault == 'context': c[0]['turns'][0]['text'] = 'different'
    elif fault == 'core': c[0]['turns'][-1]['text'] = 'old «changed core»'
    elif fault == 'factor':
        for row in (t[0], c[0], r[0]): row['factors'] = {'operator_index': 1, 'quote_placement': 'before', 'operator_negation_marker': False}
    elif fault == 'duplicate': t.append(deepcopy(t[0])); c.append(deepcopy(c[0])); r.append(deepcopy(r[0]))
    with pytest.raises(ValueError): validate_inputs(t, c, r, m)


def terminal():
    return {'execution_complete': True, 'phase': 'finished', 'completed_steps': [{'exit_code': 0}]}


def test_process_guard_accepts_completed_execution_even_when_model_rejected(monkeypatch):
    def absent(pid, signal): raise ProcessLookupError()
    monkeypatch.setattr('scripts.wait_completed_sequence_exit_v10.os.kill', absent)
    require_exit(dict(terminal(), decision='REJECT'), 123, timeout=0)


def test_process_guard_never_accepts_a_live_pid(monkeypatch):
    monkeypatch.setattr('scripts.wait_completed_sequence_exit_v10.os.kill', lambda pid, signal: None)
    with pytest.raises(ValueError, match='remains live'): require_exit(terminal(), 123, timeout=0)


@pytest.mark.parametrize('fault', ['incomplete', 'failed', 'empty_steps', 'wrong_phase'])
def test_process_guard_rejects_bad_terminal_reports(fault):
    r = terminal()
    if fault == 'incomplete': r['execution_complete'] = False
    elif fault == 'failed': r['completed_steps'][0]['exit_code'] = 1
    elif fault == 'empty_steps': r['completed_steps'] = []
    else: r['phase'] = 'running'
    with pytest.raises(ValueError): require_exit(r, 123, timeout=0)


@pytest.mark.parametrize('fault', [None, 'stale_report', 'wrong_source', 'incomplete_recount', 'missing_pass'])
def test_actual73_proof_schema_and_current_output_binding(tmp_path, monkeypatch, fault):
    root = tmp_path/'root'; private = tmp_path/'private'; private.mkdir()
    paths = ['research/v10-evidence/large-review-attention-sequence-73/report.json',
             'research/v10-evidence/large-review-attention-73/report.json',
             'research/v10-evidence/large-review-attention-73/protocol.json',
             'eval/verify_large_review_attention_v10.py']
    for name in paths:
        p = root/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('{}')
    (root/paths[0]).write_text(json.dumps(terminal()))
    proof = {'verification': 'PASS', 'checkpoint_count': 8, 'checkpoint_dev_rows_recounted': 35348,
             'report_sha256': file_sha(root/paths[1]), 'protocol_sha256': file_sha(root/paths[2]),
             'source_sha256': file_sha(root/paths[3])}
    if fault == 'stale_report': (root/paths[1]).write_text('{"changed": true}')
    elif fault == 'wrong_source': proof['source_sha256'] = '0'*64
    elif fault == 'incomplete_recount': proof['checkpoint_dev_rows_recounted'] = 8837
    elif fault == 'missing_pass': proof['verification'] = 'FAIL'
    (private/'proof.json').write_text(json.dumps(proof))
    def absent(pid, signal): raise ProcessLookupError()
    monkeypatch.setattr('training.extract_assertion_variation_v10.os.kill', absent)
    args = SimpleNamespace(root=root, private=private)
    plan = {'dependency_pid': 123, 'dependency_verification': 'proof.json'}
    if fault:
        with pytest.raises(ValueError): require_dependency(args, plan)
    else:
        assert require_dependency(args, plan)['verification_sha256'] == file_sha(private/'proof.json')
