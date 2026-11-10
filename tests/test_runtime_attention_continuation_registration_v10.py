"""Future-input registration cannot change the frozen experiment or start early."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from roguard.trajectory_data import file_sha
import training.register_runtime_attention_continuation_v10 as registration


def fixture():
    rows = {lang: [{'partition': 'train'}]*counts['train']+[{'partition': 'dev'}]*counts['dev']
            for lang, counts in registration.POPULATIONS.items()}
    populations, schedules = registration.population(rows)
    original = {'immutable_original_inputs': 'a'*64}; helpers = {'helper.py': 'b'*64}
    bindings = dict(original=original, runtime_helpers=helpers, runtime={'actual_completed_inputs': 'c'*64})
    plan = dict(source_sha256='d'*64, trainer_source_sha256='e'*64,
        optimization=deepcopy(registration.OPTIMIZATION), populations=populations, schedules=schedules,
        original_input_bindings=original, runtime_helpers=helpers, dependency_pid=123,
        completion_audit_sources={'auditor.py': 'f'*64})
    warm = dict(complete=True, decision='WARM_START_REFERENCE_ONLY_NO_MODEL_QUALIFICATION', languages={})
    for lang, counts in populations.items():
        warm['languages'][lang] = dict(
            runtime75=dict(rows=counts['dev'], binary_errors=14 if lang == 'ro' else 2, joint_argmax_errors=699 if lang == 'ro' else 645),
            runtime71=dict(rows=counts['dev'], binary_errors=17 if lang == 'ro' else 2, joint_argmax_errors=746 if lang == 'ro' else 645),
            original75={'retained': True}, runtime75_vs_original75={}, runtime75_vs_runtime71={}, historical_logit_difference={})
    return plan, rows, bindings, warm


def test_completed_registration_freezes_actual_hashes_and_original_training_scope():
    plan, rows, bindings, warm = fixture()
    result = registration.compose(plan, rows, bindings, warm, '0'*64)
    assert result['input_bindings'] == bindings and result['preregistration_sha256'] == '0'*64
    assert result['optimization'] == registration.OPTIMIZATION
    assert sum(v['total_updates'] for v in result['schedules'].values()) == 10851
    assert sum(v['total_presentations'] for v in result['schedules'].values()) == 86787
    assert result['qualified'] is result['held_out_test_opened'] is False
    assert result['warm_start_observation']['ro']['runtime75']['binary_errors'] == 14


@pytest.mark.parametrize('changed', [False, True])
def test_actual_future_report_hash_is_bound_before_composition(tmp_path, monkeypatch, changed):
    import training.train_runtime_review_attention_continuation_v10 as trainer
    plan, rows, bindings, warm = fixture()
    prereg = tmp_path/'registration.json'; prereg.write_text('{}')
    report = tmp_path/'research/v10-evidence/runtime-warm-start-90/report.json'
    registration.write_new(report, warm)
    bindings['runtime']['warm_report_sha256'] = file_sha(report)
    monkeypatch.setattr(registration, 'static_inputs', lambda a: (plan, rows))
    monkeypatch.setattr(trainer, 'load_inputs', lambda a: (rows, bindings))
    a = SimpleNamespace(root=tmp_path, private_root=tmp_path, registration=prereg)
    if changed:
        report.write_text(json.dumps(warm)+'\n\n')
        with pytest.raises(ValueError, match='report changed'): registration.completed_plan(a)
    else:
        result = registration.completed_plan(a)
        assert result['input_bindings']['runtime']['warm_report_sha256'] == file_sha(report)


def test_static_source_corruption_rejected_before_loading_inputs(tmp_path, monkeypatch):
    import training.train_large_review_attention_v10 as original
    source = tmp_path/'source.py'; source.write_text('changed')
    prereg = tmp_path/'registration.json'
    registration.write_new(prereg, dict(source_sha256=file_sha(registration.__file__),
        optimization=registration.OPTIMIZATION, public={'source.py': '0'*64}, private={}))
    monkeypatch.setattr(original, 'load_inputs', lambda a: pytest.fail('Changed source admitted'))
    with pytest.raises(ValueError, match='Preregistered source'):
        registration.static_inputs(SimpleNamespace(root=tmp_path, private_root=tmp_path, registration=prereg))


@pytest.mark.parametrize('fault', ['epochs', 'original', 'helpers', 'rows', 'partition', 'schedule',
    'incomplete', 'decision', 'worse_binary', 'worse_joint', 'wrong_dev', 'negative', 'float_count'])
def test_changed_scope_or_weaker_runtime_initialization_rejected(fault):
    plan, rows, bindings, warm = fixture()
    if fault == 'epochs': plan['optimization']['epochs'] = 4
    elif fault == 'original': bindings['original'] = {}
    elif fault == 'helpers': bindings['runtime_helpers'] = {}
    elif fault == 'rows': rows['ro'].pop()
    elif fault == 'partition': rows['ro'][0] = {'partition': 'test'}
    elif fault == 'schedule': plan['schedules']['ro']['total_updates'] += 1
    elif fault == 'incomplete': warm['complete'] = False
    elif fault == 'decision': warm['decision'] = 'PASS'
    elif fault == 'worse_binary': warm['languages']['uk']['runtime75']['binary_errors'] = 3
    elif fault == 'worse_joint': warm['languages']['ro']['runtime75']['joint_argmax_errors'] = 747
    elif fault == 'wrong_dev': warm['languages']['ro']['runtime75']['rows'] -= 1
    elif fault == 'negative': warm['languages']['ro']['runtime75']['binary_errors'] = -1
    else: warm['languages']['ro']['runtime75']['binary_errors'] = 14.
    with pytest.raises(ValueError): registration.compose(plan, rows, bindings, warm, '0'*64)


def completed_fixture(tmp_path, monkeypatch):
    plan, rows, bindings, warm = fixture()
    prereg = tmp_path/'registration.json'; prereg.write_text('{}')
    a = SimpleNamespace(root=tmp_path, private_root=tmp_path/'private', registration=prereg)
    expected = registration.compose(plan, rows, bindings, warm, file_sha(prereg))
    monkeypatch.setattr(registration, 'completed_plan', lambda a: deepcopy(expected))
    amendment, proof, preflight = registration.paths(tmp_path)
    registration.write_new(amendment, expected)
    registration.write_new(proof, dict(complete=True, amendment_sha256=file_sha(amendment),
        preregistration_sha256=file_sha(prereg), source_sha256=file_sha(registration.__file__),
        encoder_loaded=False, training_launched=False, qualified=False))
    registration.write_new(preflight, dict(complete=True, source_sha256=expected['source_sha256'],
        amendment_sha256=file_sha(amendment), bindings=bindings, schedules=expected['schedules'],
        populations=expected['populations'], encoder_loaded=False, training_launched=False, held_out_test_opened=False, qualified=False))
    return a, amendment, proof, preflight


def test_registration_proof_and_actual_training_preflight_verified(tmp_path, monkeypatch):
    a, amendment, _, _ = completed_fixture(tmp_path, monkeypatch)
    assert registration.verify_completed_registration(a, require_preflight=True) == file_sha(amendment)


@pytest.mark.parametrize('fault', ['amendment', 'proof_hash', 'preflight_bindings', 'preflight_scope', 'preflight_complete'])
def test_binding_proof_cannot_admit_changed_future_plan(tmp_path, monkeypatch, fault):
    a, amendment, proof, preflight = completed_fixture(tmp_path, monkeypatch)
    path = amendment if fault == 'amendment' else proof if fault == 'proof_hash' else preflight
    value = json.loads(path.read_text())
    if fault == 'amendment': value['optimization']['epochs'] += 1
    elif fault == 'proof_hash': value['amendment_sha256'] = 'f'*64
    elif fault == 'preflight_bindings': value['bindings'] = {}
    elif fault == 'preflight_scope': value['training_launched'] = True
    else: value['complete'] = False
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError): registration.verify_completed_registration(a, require_preflight=True)


def test_registration_waits_successful_dependency_before_loading_actual_inputs(tmp_path, monkeypatch):
    prereg = tmp_path/'registration.json'; prereg.write_text('{"dependency_pid": 123}')
    dependency = tmp_path/'research/v10-evidence/runtime-warm-start-sequence-90/report.json'
    registration.write_new(dependency, dict(execution_complete=False, phase='waiting_for_encoder'))
    monkeypatch.setattr(registration, 'completed_plan', lambda a: pytest.fail('Incomplete dependency loaded'))
    monkeypatch.setattr(sys, 'argv', ['register', '--root', str(tmp_path), '--private-root', str(tmp_path), '--registration', str(prereg)])
    with pytest.raises(ValueError, match='not successfully complete'): registration.main()
    assert all(not p.exists() for p in registration.paths(tmp_path))


def test_existing_registration_is_never_overwritten(tmp_path, monkeypatch):
    amendment, _, _ = registration.paths(tmp_path); registration.write_new(amendment, {'retain': True})
    prereg = tmp_path/'registration.json'; prereg.write_text('{}')
    monkeypatch.setattr(sys, 'argv', ['register', '--root', str(tmp_path), '--private-root', str(tmp_path), '--registration', str(prereg)])
    with pytest.raises(ValueError, match='overwrite'): registration.main()
    assert json.loads(amendment.read_text()) == {'retain': True}


def test_registration_writes_then_reverifies_before_training(tmp_path, monkeypatch):
    plan, rows, bindings, warm = fixture()
    prereg = tmp_path/'registration.json'; prereg.write_text('{"dependency_pid": 1073741824}')
    expected = registration.compose(plan, rows, bindings, warm, file_sha(prereg))
    dependency = tmp_path/'research/v10-evidence/runtime-warm-start-sequence-90/report.json'
    registration.write_new(dependency, dict(execution_complete=True, phase='finished', completed_steps=[{'exit_code': 0}]))
    monkeypatch.setattr(registration, 'completed_plan', lambda a: deepcopy(expected))
    monkeypatch.setattr(sys, 'argv', ['register', '--root', str(tmp_path), '--private-root', str(tmp_path), '--registration', str(prereg)])
    registration.main()
    amendment, proof, preflight = registration.paths(tmp_path)
    assert json.loads(amendment.read_text()) == expected and json.loads(proof.read_text())['complete']
    assert not preflight.exists()


def test_audit_wrapper_passes_only_verified_registration_hash(tmp_path, monkeypatch):
    import scripts.audit_registered_runtime_attention_continuation_v10 as wrapper
    calls = []
    monkeypatch.setattr(wrapper, 'verify_completed_registration', lambda a, **kw: 'a'*64 if kw['require_preflight'] else pytest.fail('Missing preflight'))
    monkeypatch.setattr(wrapper.subprocess, 'run', lambda command, **kw: calls.append((command, kw)))
    monkeypatch.setattr(sys, 'argv', ['audit', '--root', str(tmp_path), '--private-root', str(tmp_path),
        '--registration', str(tmp_path/'registration.json'), '--sequence-pid', '456', '--output', str(tmp_path/'audit.json')])
    wrapper.main()
    command, options = calls[0]
    assert command[command.index('--amendment-sha256')+1] == 'a'*64
    assert command[command.index('--sequence-pid')+1] == '456' and options == dict(cwd=tmp_path, check=True)


def test_audit_wrapper_does_not_invoke_auditor_after_failed_binding(tmp_path, monkeypatch):
    import scripts.audit_registered_runtime_attention_continuation_v10 as wrapper
    def fail(*args, **kwargs): raise ValueError('binding changed')
    monkeypatch.setattr(wrapper, 'verify_completed_registration', fail)
    monkeypatch.setattr(wrapper.subprocess, 'run', lambda *a, **k: pytest.fail('Audit must not run'))
    monkeypatch.setattr(sys, 'argv', ['audit', '--root', str(tmp_path), '--private-root', str(tmp_path),
        '--registration', str(tmp_path/'registration.json'), '--sequence-pid', '456', '--output', str(tmp_path/'audit.json')])
    with pytest.raises(ValueError, match='binding changed'): wrapper.main()
