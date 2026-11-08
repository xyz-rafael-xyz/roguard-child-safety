"""The queue must freeze completed plans before either production training arm."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from roguard.trajectory_data import file_sha
from scripts import run_matched_role_pipeline_v10 as pipeline
from training.matched_role_attention_inputs_v10 import OPTIMIZATION, POPULATIONS
from training.register_matched_role_attention_v10 import OUTPUTS


def fixture(tmp_path, monkeypatch, *, fault=None):
    root = tmp_path/'repo'; private = tmp_path/'private'; root.mkdir(); private.mkdir()
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value)+'\n')
    sentinel = root/'source.py'; sentinel.write_text('frozen-source')
    plan = dict(format='roguard-matched-role-pipeline-registration-1', optimization=OPTIMIZATION,
        populations=POPULATIONS, outputs=OUTPUTS, markers=pipeline.MARKERS, training_performed=False,
        held_out_test_opened=False, qualified=False, public={'source.py': file_sha(sentinel)}, private={},
        dependency_sequence='dependency.json', dependency_pid=123, queue_plan='queue.json',
        sequence='sequence.json', log='training.log', training_plan='plan104.json',
        audit_plan='audit107/plan.json', registration_directory='registration104')
    prospective = root/'prospective.json'; write(prospective, plan)
    write(root/'dependency.json', dict(execution_complete=fault != 'dependency', phase='finished' if fault != 'dependency' else 'running',
                                     completed_steps=[dict(name='audit103', exit_code=0)]))
    write(private/'queue.json', dict(source_hashes={**plan['public'], 'prospective.json': file_sha(prospective)}))
    a = SimpleNamespace(root=root, private_root=private, base=tmp_path/'base', prospective=prospective,
                        prospective_sha256=file_sha(prospective))
    def require_exit(report, pid):
        assert pid == 123
        if report['execution_complete'] is not True: raise ValueError('Incomplete dependency')
    monkeypatch.setattr('scripts.wait_completed_sequence_exit_v10.require_exit', require_exit)
    calls = []
    def execute(command, *, cwd, check):
        assert cwd == root and check is False
        calls.append(command)
        args = dict(zip(command[2::2], command[3::2]))
        if command[1] == 'training/register_matched_role_attention_v10.py':
            write(Path(args['--plan']), {'fixture': True})
        else:
            assert command[1] == 'training/train_matched_role_attention_v10.py'
            arm = args['--arm']; training = root/plan['training_plan']; audit = root/plan['audit_plan']
            assert args['--plan-sha256'] == file_sha(training)
            if '--preflight-report' in args:
                assert not audit.exists()
                if fault == 'preflight_exit': return SimpleNamespace(returncode=7)
                write(Path(args['--preflight-report']), dict(complete=True, arm=arm, plan_sha256=file_sha(training),
                    encoder_loaded=False, training_launched=fault == 'preflight_scope', held_out_test_opened=False, qualified=False))
            else:
                assert audit.exists(), 'No optimizer before frozen audit plan'
                audit_plan = json.loads(audit.read_text())
                assert audit_plan['training_plan_sha256'] == file_sha(training)
                record = json.loads((root/plan['registration_directory']/'report.json').read_text())
                assert record['audit_frozen_before_training'] and record['audit_plan_sha256'] == file_sha(audit)
                if arm == 'treatment':
                    assert (root/OUTPUTS['control']['study']/'report.json').exists()
                if fault == 'control_exit' and arm == 'control': return SimpleNamespace(returncode=9)
                write(root/OUTPUTS[arm]['study']/'report.json', dict(complete=True, arm=arm,
                    languages={'ro': {'complete': True}, 'uk': {'complete': True}}, held_out_test_opened=False, stable_release_ready=False))
                if fault == 'source_after_control' and arm == 'control': sentinel.write_text('changed')
                if fault == 'plan_after_control' and arm == 'control': write(training, {'tampered': True})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(pipeline.subprocess, 'run', execute)
    return a, plan, calls


def test_register_preflight_audit_freeze_then_serial_training(tmp_path, monkeypatch, capsys):
    a, plan, calls = fixture(tmp_path, monkeypatch)
    pipeline.run(a)
    record = json.loads((a.root/plan['registration_directory']/'report.json').read_text())
    assert record['complete'] and record['phase'] == 'finished' and not record['qualified']
    assert len(calls) == 5 and record['audit_frozen_before_training']
    markers = [json.loads(line)['starting'] for line in capsys.readouterr().out.splitlines()]
    assert markers == ['register-matched-role104', 'preflight-matched-role-control', 'preflight-matched-role-treatment',
                       pipeline.MARKERS['control'], pipeline.MARKERS['treatment']]
    assert [stage['name'] for stage in record['completed_stages']] == markers
    with pytest.raises(ValueError, match='overwrite'):
        pipeline.run(a)
    assert len(calls) == 5


@pytest.mark.parametrize('fault,expected_calls', [('dependency', 0), ('preflight_exit', 2), ('preflight_scope', 2),
    ('control_exit', 4), ('source_after_control', 4), ('plan_after_control', 4)])
def test_incomplete_or_changed_dependency_cannot_advance(fault, expected_calls, tmp_path, monkeypatch):
    a, plan, calls = fixture(tmp_path, monkeypatch, fault=fault)
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        pipeline.run(a)
    assert len(calls) == expected_calls
    assert not (a.root/OUTPUTS['treatment']['study']).exists()
    directory = a.root/plan['registration_directory']
    if fault == 'dependency':
        assert not directory.exists() and not (a.root/plan['training_plan']).exists()
    else:
        record = json.loads((directory/'report.json').read_text())
        assert not record['complete'] and record['phase'] == 'failed'
    if expected_calls <= 2: assert not (a.root/plan['audit_plan']).exists()


@pytest.mark.parametrize('fault', ['scope', 'source', 'queue_binding', 'prospective_hash'])
def test_static_registration_changes_rejected_before_commands(fault, tmp_path, monkeypatch):
    a, plan, calls = fixture(tmp_path, monkeypatch)
    if fault == 'source': (a.root/'source.py').write_text('changed')
    elif fault == 'scope':
        plan['qualified'] = True; a.prospective.write_text(json.dumps(plan)); a.prospective_sha256 = file_sha(a.prospective)
    elif fault == 'queue_binding':
        queue = json.loads((a.private_root/'queue.json').read_text()); queue['source_hashes'].pop('source.py')
        (a.private_root/'queue.json').write_text(json.dumps(queue))
    else: a.prospective_sha256 = '0'*64
    with pytest.raises(ValueError): pipeline.run(a)
    assert not calls and not (a.root/plan['registration_directory']).exists()


def test_registered_audit_wrapper_requires_terminal_frozen_pipeline(tmp_path, monkeypatch):
    from scripts import audit_registered_matched_role_v10 as wrapper
    a, plan, _ = fixture(tmp_path, monkeypatch); pipeline.run(a)
    calls = []
    monkeypatch.setattr(wrapper.subprocess, 'run', lambda command, **kwargs: calls.append((command, kwargs)))
    output = a.private_root/'verification.json'
    monkeypatch.setattr(sys, 'argv', ['audit', '--root', str(a.root), '--private-root', str(a.private_root),
        '--prospective', str(a.prospective), '--prospective-sha256', a.prospective_sha256,
        '--producer-pid', '123', '--output', str(output)])
    wrapper.main()
    assert len(calls) == 1 and calls[0][1] == {'cwd': a.root, 'check': True}
    assert calls[0][0][1] == 'eval/verify_matched_role_attention_v10.py'
    path = a.root/plan['registration_directory']/'report.json'; original = json.loads(path.read_text())
    for key, value in [('complete', False), ('audit_frozen_before_training', False), ('held_out_test_opened', True),
                       ('training_plan_sha256', '0'*64), ('audit_plan_sha256', '0'*64)]:
        record = deepcopy(original); record[key] = value; path.write_text(json.dumps(record))
        with pytest.raises(ValueError): wrapper.main()
        assert len(calls) == 1
