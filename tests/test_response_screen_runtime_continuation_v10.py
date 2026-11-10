"""Exercise the continuation auditor's CLI, provenance, and concurrent-change guards."""
import json
import sys
from types import SimpleNamespace

import pytest

from eval import verify_response_screen_runtime_continuation_v10 as audit
from roguard.assurance import digest
from roguard.trajectory_data import file_sha
from tests.test_response_diversity_screen_v10 import fixture, raw
from training.screen_response_diversity_v10 import CODES, SYSTEM, result


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def setup_case(tmp_path, monkeypatch, rejected=False):
    root = tmp_path/'root'; private = tmp_path/'private'; base = tmp_path/'revision'
    evidence = root/'research/v10-evidence'; study = evidence/'response-diversity-screen-97'
    plan_path = evidence/'protocol-amendment-78.json'
    queue_path = private/'response-diversity-screen-sequence-97-plan.json'
    sequence_path = evidence/'response-diversity-screen-sequence-97/report.json'
    dependency = 'research/v10-evidence/dependency/report.json'
    write(root/dependency, {'execution_complete': True})
    for name in ('training/screen_response_diversity_v10.py', 'scripts/run_report_sequence_v10.py'):
        write(root/name, {'source': name})
    for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'config.json'):
        write(base/name, {'file': name})
    rows = fixture()
    rows_path = private/'response-diversity-pilot-69/train-draft.jsonl'
    rows_path.parent.mkdir(parents=True)
    rows_path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    tokens = {row['case_id']: [1, 2] for row in rows}
    codes = {value: key for key, value in CODES.items()}
    records = [result(row, raw(codes[row['gold']]), tokens[row['case_id']]) for row in rows]
    if rejected:
        records[0] = result(rows[0], raw('unclear'), tokens[rows[0]['case_id']])
    raw_path = private/'response-diversity-screen-97/judgments.jsonl'
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text(''.join(json.dumps(row)+'\n' for row in records))
    public = {name: file_sha(root/name) for name in (
        'training/screen_response_diversity_v10.py', 'scripts/run_report_sequence_v10.py')}
    plan = {'public': public, 'private': {str(rows_path.relative_to(private)): file_sha(rows_path)},
            'dependency_report': dependency, 'judge': {'revision': 'revision'},
            'model_files': {p.name: file_sha(p) for p in base.iterdir()},
            'software': {'transformers': 'fixture', 'tokenizers': 'fixture'},
            'task_prompt_sha256': 'task', 'screen_prompt_sha256': digest(SYSTEM)}
    write(plan_path, plan)
    queue = {'source_hashes': public, 'steps': [{'name': 'wait'}, {'name': 'screen'}]}
    write(queue_path, queue)
    sequence = {'execution_complete': True, 'phase': 'finished', 'plan_sha256': file_sha(queue_path),
                'source_hashes': public, 'source_sha256': public['scripts/run_report_sequence_v10.py'],
                'dependency_report_sha256': file_sha(root/dependency),
                'stable_release_ready': False, 'promotion_performed': False,
                'completed_steps': [{'name': 'wait', 'exit_code': 0}, {'name': 'screen', 'exit_code': 0}]}
    write(sequence_path, sequence)
    (private/'response-diversity-screen-sequence-97.pid').write_text('12345\n')
    protocol = {key: plan[key] for key in ('public', 'private', 'judge', 'model_files', 'software', 'task_prompt_sha256')}
    protocol.update(format='roguard-v10-single-response-screen-1', plan_sha256=file_sha(plan_path),
                    source_sha256=public['training/screen_response_diversity_v10.py'], screen_prompt=SYSTEM,
                    rows=128, ordered_inputs_sha256=digest(rows), query_tokens_sha256=digest(tokens),
                    minimum_query_tokens=2, maximum_query_tokens=2, **audit.SCOPE)
    write(study/'protocol.json', protocol)
    (study/'source.py').write_bytes((root/'training/screen_response_diversity_v10.py').read_bytes())
    counts = audit.recount_raw(rows, records, tokens)
    report = {'complete': True, **counts, **audit.SCOPE, 'seconds': 2.0,
              'judgments_sha256': file_sha(raw_path), 'protocol_sha256': file_sha(study/'protocol.json')}
    write(study/'report.json', report)
    write(study/'progress.json', {'completed_rows': 128, 'seconds': 1.0, **audit.SCOPE})
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return [1, 2]
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(AutoTokenizer=SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: Tokenizer())))
    monkeypatch.setattr(audit, 'version', lambda name: 'fixture')
    def exited(pid, signal):
        raise ProcessLookupError
    monkeypatch.setattr(audit.os, 'kill', exited)
    output = private/'verification.json'
    argv = ['audit', '--root', str(root), '--private', str(private), '--base', str(base),
            '--output', str(output), '--plan-sha256', file_sha(plan_path),
            '--queue-plan-sha256', file_sha(queue_path), '--queue-pid', '12345']
    monkeypatch.setattr(sys, 'argv', argv)
    return SimpleNamespace(root=root, private=private, base=base, study=study, output=output,
                           plan=plan_path, queue=queue_path, sequence=sequence_path, argv=argv, raw=raw_path)


@pytest.mark.parametrize('rejected', [False, True])
def test_complete_audit_preserves_machine_decision_and_never_qualifies(tmp_path, monkeypatch, rejected):
    case = setup_case(tmp_path, monkeypatch, rejected)
    audit.main()
    proof = json.loads(case.output.read_text())
    assert proof['verification'] == 'PASS' and proof['recount']['accepted_rows'] == 128-int(rejected)
    assert proof['model_loaded'] is proof['generations_repeated'] is proof['qualified'] is False
    assert all(proof[key] is False for key in audit.SCOPE)
    with pytest.raises(ValueError, match='overwrite'):
        audit.main()


@pytest.mark.parametrize('fault', ['plan', 'queue', 'sequence', 'dependency', 'pid', 'live_pid',
                                  'public', 'private', 'tokenizer', 'software', 'protocol', 'raw', 'scope'])
def test_invalid_or_unfinished_evidence_never_writes_pass(tmp_path, monkeypatch, fault):
    c = setup_case(tmp_path, monkeypatch)
    if fault in ('plan', 'queue'):
        with getattr(c, fault).open('a') as output: output.write(' ')
    elif fault == 'sequence':
        value = json.loads(c.sequence.read_text()); value['execution_complete'] = False; write(c.sequence, value)
    elif fault == 'dependency': write(c.root/'research/v10-evidence/dependency/report.json', {})
    elif fault == 'pid': (c.private/'response-diversity-screen-sequence-97.pid').write_text('54321')
    elif fault == 'live_pid': monkeypatch.setattr(audit.os, 'kill', lambda *args: None)
    elif fault == 'public': write(c.root/'training/screen_response_diversity_v10.py', 'changed')
    elif fault == 'private': (c.private/'response-diversity-pilot-69/train-draft.jsonl').write_text('')
    elif fault == 'tokenizer': write(c.base/'tokenizer.json', {})
    elif fault == 'software': monkeypatch.setattr(audit, 'version', lambda name: 'changed')
    elif fault == 'protocol':
        value = json.loads((c.study/'protocol.json').read_text()); value['screen_prompt'] = 'changed'
        write(c.study/'protocol.json', value)
    elif fault == 'raw': c.raw.write_text('')
    else:
        value = json.loads((c.study/'report.json').read_text()); value['training_ready'] = True
        write(c.study/'report.json', value)
    with pytest.raises(ValueError): audit.main()
    assert not c.output.exists()


@pytest.mark.parametrize('target', ['report', 'sequence', 'plan', 'queue', 'drafts'])
def test_changes_during_audit_cannot_receive_pass(tmp_path, monkeypatch, target):
    c = setup_case(tmp_path, monkeypatch)
    original = audit.verify_report
    path = {'report': c.study/'report.json', 'sequence': c.sequence, 'plan': c.plan, 'queue': c.queue,
            'drafts': c.private/'response-diversity-pilot-69/train-draft.jsonl'}[target]
    def changing(*args):
        original(*args)
        with path.open('a') as output: output.write(' ')
    monkeypatch.setattr(audit, 'verify_report', changing)
    with pytest.raises(ValueError, match='changed during audit'): audit.main()
    assert not c.output.exists()
