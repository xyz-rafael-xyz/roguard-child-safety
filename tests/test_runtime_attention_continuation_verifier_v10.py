"""Independent audit of a real tiny fit and deliberately corrupted evidence."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

from eval.verify_runtime_review_attention_continuation_v10 import audit_language, audit_training_log, parity
from roguard.trajectory_data import file_sha, read_jsonl, write_jsonl


@pytest.fixture(scope='module')
def actual_fit(tmp_path_factory):
    pytest.importorskip('mlx.core'); np = pytest.importorskip('numpy')
    import training.train_runtime_review_attention_continuation_v10 as trainer
    from test_runtime_attention_continuation_v10 import test_full_training_loop_on_small_real_42_layer_model
    path = tmp_path_factory.mktemp('actual-attention-fit')
    captured = {}; stream = io.StringIO(); original_main = trainer.main
    def run():
        captured.update(dict(zip(sys.argv[1::2], sys.argv[2::2])))
        return original_main()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(trainer, 'main', run)
        with redirect_stdout(stream):
            test_full_training_loop_on_small_real_42_layer_model(path, patch)
    args = {key[2:].replace('-', '_'): Path(value) for key, value in captured.items()}
    rows = {}
    for language in ('ro', 'uk'):
        rows[language] = [dict(case_id=language+str(i), language=language, group='fixture',
            partition='train' if i < 2 else 'dev', fields=[i % 2]*6, gold=i % 2) for i in range(4)]
    return SimpleNamespace(args=SimpleNamespace(**args), rows=rows,
        report=json.loads((args['study']/'report.json').read_text()),
        plan=json.loads(args['amendment'].read_text()),
        log=b'{"starting": "train-runtime-review-attention95"}\n'+stream.getvalue().encode())


def audit_fixture(fit, study=None, weights=None, record=None, rows=None, teacher=None, initial_logits=None):
    import numpy as np
    from safetensors.numpy import load_file
    a = fit.args
    return audit_language(study or a.study, weights or a.private, 'ro', record or fit.report['languages']['ro'],
        rows or fit.rows['ro'], load_file(a.candidate/'actual/ro-selected.safetensors'),
        load_file(a.initial_private/'ro-selected.safetensors'),
        teacher if teacher is not None else np.load(a.private_root/'runtime-review-baseline-88/ro-logits.npy'),
        json.loads((a.initial_study/'ro-review.json').read_text()),
        json.loads((a.initial_study/'ro-review.json').read_text()),
        initial_logits if initial_logits is not None else np.load(a.private_root/'runtime-warm-start-90/ro-selected-dev-logits.npy'))


def test_actual_fit_all_checkpoints_reload_and_event_history_recount(actual_fit):
    result = audit_fixture(actual_fit)
    assert result['checkpoint_count'] == 4 and result['checkpoint_dev_rows_recounted'] == 8
    assert result['reload_rows_recounted'] == 2 and len(result['artifacts']) == 11
    assert result['maximum_policy_enumeration_error'] < 1e-12
    log = audit_training_log(actual_fit.log, actual_fit.report, actual_fit.plan['schedules'], {'ro': 1, 'uk': 1})
    assert log['training_events'] == 27


@pytest.mark.parametrize('fault', ['missing_epoch', 'frozen_tensor', 'initial_tensor', 'nonfinite_tensor',
    'metrics', 'counter', 'selection', 'identity', 'spec', 'warm_logits', 'initial_parity_report',
    'reload_values', 'reload_hash', 'reload_report', 'teacher', 'partition', 'incomplete',
    'no_attention_update', 'no_head_update'])
def test_changed_checkpoint_or_evidence_rejected(actual_fit, tmp_path, fault):
    import numpy as np
    from safetensors.numpy import load_file, save_file
    weights = tmp_path/'weights'; study = tmp_path/'study'
    shutil.copytree(actual_fit.args.private, weights); shutil.copytree(actual_fit.args.study, study)
    record = deepcopy(actual_fit.report['languages']['ro']); rows = deepcopy(actual_fit.rows['ro'])
    teacher = np.load(actual_fit.args.private_root/'runtime-review-baseline-88/ro-logits.npy')
    warm = np.load(actual_fit.args.private_root/'runtime-warm-start-90/ro-selected-dev-logits.npy')
    if fault == 'missing_epoch': record['checkpoints'].pop(1)
    elif fault in ('frozen_tensor', 'initial_tensor', 'nonfinite_tensor'):
        epoch = 0 if fault == 'initial_tensor' else 1
        path = weights/'ro'/f'epoch-{epoch}.safetensors'; values = load_file(path)
        name = 'heads.uk.first.bias' if fault == 'frozen_tensor' else 'heads.ro.first.bias'
        values[name][0] += 1.
        if fault == 'nonfinite_tensor': values[name][0] = np.nan
        save_file(values, str(path)); record['checkpoints'][epoch]['adapter_sha256'] = file_sha(path)
    elif fault in ('no_attention_update', 'no_head_update'):
        path = weights/'ro/epoch-1.safetensors'; values = load_file(path)
        initial = load_file(actual_fit.args.initial_private/'ro-selected.safetensors')
        prefix = 'base.' if fault == 'no_attention_update' else 'heads.ro.'
        values.update({name: value for name, value in initial.items() if name.startswith(prefix)})
        save_file(values, str(path)); record['checkpoints'][1]['adapter_sha256'] = file_sha(path)
    elif fault == 'metrics': record['checkpoints'][1]['binary_errors'] += 1
    elif fault == 'counter': record['checkpoints'][2]['completed_training_presentations'] += 1
    elif fault == 'selection': record['selected']['epoch'] = 99
    elif fault == 'identity':
        path = study/'ro-development-predictions.jsonl'; values = read_jsonl(path)
        values[0]['case_id'] = 'changed'; write_jsonl(path, values)
    elif fault == 'spec':
        path = study/'ro-review.json'; values = json.loads(path.read_text()); values['threshold'] = -1
        path.write_text(json.dumps(values))
    elif fault == 'warm_logits': warm[0, 0] += .1
    elif fault == 'initial_parity_report': record['maximum_initial_logit_error'] = 9e-6
    elif fault in ('reload_values', 'reload_hash'):
        path = weights/'ro-selected-reload-dev-logits.npy'; values = np.load(path); values[0, 0] += .1
        np.save(path, values)
        if fault == 'reload_values': record['selected_reload_logits_sha256'] = file_sha(path)
    elif fault == 'reload_report': record['selected_reload_decision_parity']['argmax_rows_differing'] = 1
    elif fault == 'teacher': teacher = teacher.astype(np.float64)
    elif fault == 'partition': rows[0]['partition'] = 'test'
    else: record['complete'] = False
    with pytest.raises(ValueError):
        audit_fixture(actual_fit, study, weights, record, rows, teacher, warm)


@pytest.mark.parametrize('head', range(7))
def test_subtolerance_argmax_change_in_every_head_is_rejected(head):
    np = pytest.importorskip('numpy')
    before = np.zeros((2, 19), np.float32); after = before.copy()
    after[0, [0, 3, 6, 10, 13, 15, 17][head]+1] = 1e-6
    with pytest.raises(ValueError): parity(after, before, dict(mode='binary', threshold=.5))


def test_served_threshold_change_without_argmax_change_is_rejected():
    np = pytest.importorskip('numpy')
    from roguard.gemma4_review import _score
    before = np.zeros((1, 19), np.float32); before[0, -1] = 1.
    after = before.copy(); after[0, -1] += 1e-6
    spec = dict(mode='binary'); spec['threshold'] = (_score(before[0], spec)+_score(after[0], spec))/2
    with pytest.raises(ValueError): parity(after, before, spec)


def history_fixture():
    # Includes partial final batches, update25 boundaries and >100 DEV rows.
    events = []; report = dict(decision='REJECT', languages={}); schedules = {}; groups = {'ro': 43, 'uk': 42}
    def append(event):
        events.append(dict(event, seconds=float(len(events)+1)))
    def progress(lang, stage, n):
        for checked in list(range(100, n, 100))+[n]:
            append(dict(language=lang, inference=dict(stage=stage, checked=checked, rows=n)))
    for lang, n, dev in (('ro', 73, 311), ('uk', 121, 209)):
        per = (n+7)//8; schedules[lang] = dict(updates_per_epoch=per, total_updates=3*per, total_presentations=3*n)
        checkpoints = [dict(epoch=epoch, binary_errors=4 if epoch == 0 else 1, joint_argmax_errors=8,
            introduced_v9=1, selection_rank=[4 if epoch == 0 else 1, 8, .5, 0.]) for epoch in range(4)]
        report['languages'][lang] = dict(training_rows=n, development_rows=dev, checkpoints=checkpoints)
        progress(lang, 'train-source-probes', groups[lang])
        for epoch in range(4):
            if epoch:
                for step in range((epoch-1)*per+1, epoch*per+1):
                    if step % 25 and step != epoch*per: continue
                    append(dict(language=lang, epoch=epoch, step=step, total_steps=3*per,
                        presentations=(epoch-1)*n+min((step-(epoch-1)*per)*8, n), loss=.1))
            progress(lang, f'epoch-{epoch}', dev)
            append(dict(language=lang, epoch=epoch, binary_errors=checkpoints[epoch]['binary_errors'],
                joint_argmax_errors=8, introduced_v9=1, selected_epoch=0 if epoch == 0 else 1))
        progress(lang, 'selected-reload', dev)
    append(dict(complete=True, decision='REJECT')); report['seconds'] = events[-1]['seconds']
    return report, schedules, groups, events


def raw(events):
    return b'earlier queue step\n{"starting": "train-runtime-review-attention95"}\n'+b''.join(
        (json.dumps(e)+'\n').encode() for e in events)


def test_full_history_partial_batches_and_progress():
    report, schedules, groups, events = history_fixture()
    assert audit_training_log(raw(events), report, schedules, groups)['training_events'] == len(events)


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'counter', 'epoch', 'selection', 'time', 'loss',
    'terminal', 'progress_stage', 'progress_rows', 'progress_count', 'progress_order', 'unknown', 'elapsed'])
def test_incomplete_or_changed_event_history_rejected(fault):
    report, schedules, groups, events = history_fixture()
    update = next(i for i, e in enumerate(events) if 'step' in e)
    inference = next(i for i, e in enumerate(events) if e.get('inference', {}).get('stage') == 'epoch-0')
    if fault == 'missing': events.pop(update)
    elif fault == 'duplicate': events.insert(update, dict(events[update]))
    elif fault == 'counter': events[update]['presentations'] += 1
    elif fault == 'epoch': events[update]['epoch'] += 1
    elif fault == 'selection': next(e for e in events if 'selected_epoch' in e)['selected_epoch'] = 3
    elif fault == 'time': events[update]['seconds'] = 0
    elif fault == 'loss': events[update]['loss'] = float('nan')
    elif fault == 'terminal': events[-1]['complete'] = False
    elif fault == 'progress_stage': events[inference]['inference']['stage'] = 'epoch-1'
    elif fault == 'progress_rows': events[inference]['inference']['rows'] += 1
    elif fault == 'progress_count': events[inference]['inference']['checked'] += 1
    elif fault == 'progress_order':
        events[inference]['inference'], events[inference+1]['inference'] = events[inference+1]['inference'], events[inference]['inference']
    elif fault == 'unknown': events[inference]['extra'] = True
    else: report['seconds'] += .1
    with pytest.raises(ValueError): audit_training_log(raw(events), report, schedules, groups)


def test_incomplete_training_rejected_before_output(tmp_path, monkeypatch):
    import eval.verify_runtime_review_attention_continuation_v10 as verifier
    plan = tmp_path/'plan.json'; plan.write_text('{}')
    study = tmp_path/'research/v10-evidence/runtime-review-attention-95'; study.mkdir(parents=True)
    (study/'protocol.json').write_text('{}'); (study/'report.json').write_text('{"complete": false}')
    output = tmp_path/'output.json'
    monkeypatch.setattr(sys, 'argv', ['verify', '--root', str(tmp_path), '--private', str(tmp_path),
        '--amendment', str(plan), '--amendment-sha256', file_sha(plan), '--sequence-pid', '123', '--output', str(output)])
    with pytest.raises(ValueError, match='Training is incomplete'): verifier.main()
    assert not output.exists()
