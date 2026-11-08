from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.matched_role_attention_inputs_v10 import (
    OPTIMIZATION, POPULATIONS, aligned_teacher, require_audit, validate_initial_weights)


def completed_fixture():
    proof = dict(verification='PASS', artifact_count=16, cpu_head_rows=6336,
        development_rows_compared=8837, ordered_progress_events=154, device='cpu',
        encoder_loaded=False, training_performed=False, training_admitted=False,
        held_out_test_opened=False, qualified=False)
    report = dict(complete=True, decision='REFERENCE_EXTRACTION_ONLY_REQUIRES_INDEPENDENT_AUDIT',
        training_performed=False, held_out_test_opened=False, qualified=False,
        languages={lang: dict(complete=True, modes={m: dict(complete=True) for m in ('v9', 'selected95')})
                   for lang in ('ro', 'uk')})
    sequence = dict(execution_complete=True, phase='finished', completed_steps=[dict(exit_code=0)])
    return proof, report, sequence


def test_completed_dependency_scope():
    require_audit(*completed_fixture())


@pytest.mark.parametrize('target,key,value', [
    (0, 'verification', 'FAIL'), (0, 'artifact_count', 15), (0, 'cpu_head_rows', 6335),
    (0, 'development_rows_compared', 8836), (0, 'ordered_progress_events', 153),
    (0, 'device', 'gpu'), (0, 'encoder_loaded', True), (0, 'training_performed', True),
    (0, 'training_admitted', True), (0, 'held_out_test_opened', True), (0, 'qualified', True),
    (1, 'complete', False), (2, 'execution_complete', False), (2, 'phase', 'running'),
    (2, 'completed_steps', []), (2, 'completed_steps', [dict(exit_code=1)])])
def test_incomplete_or_changed_audit_rejected(target, key, value):
    records = completed_fixture(); records[target][key] = value
    with pytest.raises(ValueError):
        require_audit(*records)


def test_teacher_lookup_keeps_occurrence_alignment_and_repeated_controls():
    np = pytest.importorskip('numpy')
    base = np.arange(4*19, dtype=np.float32).reshape(4, 19)
    new = np.arange(2*19, dtype=np.float32).reshape(2, 19)+1000
    inputs = dict(base_teacher=base, new_teacher=new,
        control=[None]*6, treatment=[None]*6,
        references=dict(control=[('base', i) for i in [0, 1, 2, 3, 1, 1]],
                        treatment=[('base', i) for i in range(4)]+[('new', 0), ('new', 1)]))
    control, treatment = (aligned_teacher(inputs, arm) for arm in ('control', 'treatment'))
    assert np.array_equal(control[:4], treatment[:4])
    assert np.array_equal(control[4:], np.stack([base[1], base[1]]))
    assert np.array_equal(treatment[4:], new)
    inputs['new_teacher'] = new.astype(np.float64)
    with pytest.raises(ValueError):
        aligned_teacher(inputs, 'treatment')


def inventory(np, language):
    names = {f'base.model.layers.{i}.self_attn.{p}.lora_{part}'
             for i in range(24, 42) for p in ('q_proj', 'o_proj') for part in ('a', 'b')}
    names |= {f'heads.{language}.{layer}.{part}' for layer in ('first', 'last') for part in ('weight', 'bias')}
    return {name: np.zeros((2, 2), np.float32) for name in names | {'frozen.fixture.weight'}}


@pytest.mark.parametrize('language', ['ro', 'uk'])
def test_selected95_can_change_registered_tail_and_head_only(language):
    np = pytest.importorskip('numpy')
    original = inventory(np, language); initial = deepcopy(original)
    initial['base.model.layers.24.self_attn.q_proj.lora_b'] += .25
    initial[f'heads.{language}.first.weight'] += .5
    assert len(validate_initial_weights(original, initial, language)) == 76
    for fault in ('frozen', 'shape', 'dtype', 'nan', 'missing'):
        bad = deepcopy(initial); name = f'heads.{language}.last.bias'
        if fault == 'frozen': bad['frozen.fixture.weight'] += 1
        elif fault == 'shape': bad[name] = np.zeros((1, 2), np.float32)
        elif fault == 'dtype': bad[name] = bad[name].astype(np.float64)
        elif fault == 'nan': bad[name][0, 0] = np.nan
        else: del bad[name]
        with pytest.raises(ValueError):
            validate_initial_weights(original, bad, language)


def plan_fixture(tmp_path, monkeypatch):
    from roguard.trajectory_data import file_sha
    from training import train_matched_role_attention_v10 as trainer
    root = tmp_path/'root'; private = tmp_path/'private'; root.mkdir(); private.mkdir()
    sentinel = root/'sentinel.txt'; sentinel.write_text('bound')
    a = SimpleNamespace(root=root, private_root=private, arm='control',
        plan=root/'plan.json', study=root/'study-control', private=private/'control')
    binding = {'fixture': True}
    plan = dict(format='roguard-matched-role-attention-plan-1', source_sha256=file_sha(trainer.__file__),
        optimization=OPTIMIZATION, populations=POPULATIONS, initialization='selected95',
        held_out_test_opened=False, qualified=False, public={'sentinel.txt': file_sha(sentinel)}, private={},
        outputs={'control': {'study': 'study-control', 'private': 'control'}}, input_bindings=binding)
    def write():
        a.plan.write_text(json.dumps(plan)); a.plan_sha256 = file_sha(a.plan)
    write()
    monkeypatch.setattr('training.matched_role_attention_inputs_v10.completed_inputs',
                        lambda *args: ({'fixture_inputs': True}, binding))
    return trainer, a, plan, write


def test_trainer_input_gate_accepts_bound_plan(tmp_path, monkeypatch):
    trainer, a, plan, _ = plan_fixture(tmp_path, monkeypatch)
    assert trainer.load_inputs(a)[2] == plan


@pytest.mark.parametrize('fault', ['hash', 'source', 'initialization', 'optimization', 'scope', 'public', 'destination', 'bindings'])
def test_trainer_input_gate_rejects_changed_plan_or_sources(tmp_path, monkeypatch, fault):
    trainer, a, plan, write = plan_fixture(tmp_path, monkeypatch)
    if fault == 'hash': a.plan_sha256 = '0'*64
    elif fault == 'source': plan['source_sha256'] = '0'*64
    elif fault == 'initialization': plan['initialization'] = 'v9'
    elif fault == 'optimization': plan['optimization'] = dict(OPTIMIZATION, epochs=4)
    elif fault == 'scope': plan['held_out_test_opened'] = True
    elif fault == 'public': (a.root/'sentinel.txt').write_text('changed')
    elif fault == 'destination': a.private = a.private_root/'wrong-arm'
    elif fault == 'bindings': plan['input_bindings'] = {'fixture': False}
    if fault != 'hash': write()
    with pytest.raises(ValueError):
        trainer.load_inputs(a)
