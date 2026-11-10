"""Audit actual tiny paired fits and reject corrupted retained evidence."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

from eval.verify_matched_role_attention_v10 import (
    audit_class_weights, audit_language, audit_occurrences, audit_training_log,
    paired_outcomes, require_complete)
from roguard.trajectory_data import file_sha, read_jsonl, write_jsonl


@pytest.fixture(scope='module')
def actual_pair(tmp_path_factory):
    pytest.importorskip('mlx.core'); pytest.importorskip('numpy')
    from training import train_matched_role_attention_v10 as trainer
    from test_matched_role_attention_training_v10 import test_driver_trains_both_languages_and_reloads_selected_checkpoint
    runs = {}
    for arm in ('control', 'treatment'):
        path = tmp_path_factory.mktemp('actual-matched-'+arm)
        stream = io.StringIO(); captured = {}; original_main = trainer.main
        def run():
            captured['args'] = dict(zip(sys.argv[1::2], sys.argv[2::2]))
            captured['inputs'], captured['bindings'], captured['plan'] = trainer.load_inputs(None)
            return original_main()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(trainer, 'main', run)
            with redirect_stdout(stream):
                test_driver_trains_both_languages_and_reloads_selected_checkpoint(path, patch, arm)
        args = captured['args']; study = Path(args['--study']); weights = Path(args['--private'])
        runs[arm] = SimpleNamespace(study=study, weights=weights, inputs=captured['inputs'],
            bindings=captured['bindings'], plan=captured['plan'], report=json.loads((study/'report.json').read_text()),
            log=stream.getvalue().encode())
    raw = b'{"starting": "control"}\n'+runs['control'].log+b'{"starting": "treatment"}\n'+runs['treatment'].log
    return runs, raw


def audit_one(fit, language='ro', *, study=None, weights=None, record=None, teacher=None):
    import numpy as np
    from safetensors.numpy import load_file
    current = fit.inputs[language]; arm = fit.report['arm']
    refs = current['references'][arm]
    original_teacher = np.stack([(current['base_teacher'] if source == 'base' else current['new_teacher'])[index]
                                 for source, index in refs])
    return audit_language(study or fit.study, weights or fit.weights, language,
        record or fit.report['languages'][language], current[arm],
        load_file(current['weights']['v9']), load_file(current['weights']['selected95']),
        original_teacher if teacher is None else teacher, current['specs']['v9'], current['specs']['selected95'],
        current['initial_dev_logits'], epochs=1, batch_size=2)


def test_actual_paired_checkpoints_occurrences_weights_and_logs(actual_pair):
    runs, raw = actual_pair
    for arm, fit in runs.items():
        require_complete(fit.report, arm)
        for lang, current in fit.inputs.items():
            result = audit_one(fit, lang)
            assert result['checkpoint_count'] == 2 and result['checkpoint_dev_rows_recounted'] == 4
            assert result['reload_rows_recounted'] == 2 and result['maximum_policy_enumeration_error'] < 1e-12
            observed = read_jsonl(fit.weights/(lang+'-occurrences.jsonl'))
            assert audit_occurrences(current[arm], current['base'], current['new'], observed, arm, 1073) == current['references'][arm]
            audit_class_weights(current[arm], fit.report['languages'][lang]['field_weights'])
        log = audit_training_log(raw, fit.report, fit.bindings['schedules'], {'ro': 1, 'uk': 1}, {'ro': 2, 'uk': 2},
            marker=arm, next_marker='treatment' if arm == 'control' else None, epochs=1, batch_size=2)
        assert log['events'] == 17


@pytest.mark.parametrize('fault', ['frozen', 'initial', 'tensor_nan', 'missing_epoch', 'metrics', 'counter', 'selection',
    'identity', 'spec', 'initial_parity', 'reload_values', 'reload_report', 'no_tail_update', 'no_head_update'])
def test_corrupted_checkpoint_evidence_rejected(actual_pair, tmp_path, fault):
    import numpy as np
    from safetensors.numpy import load_file, save_file
    fit = actual_pair[0]['treatment']; study = tmp_path/'study'; weights = tmp_path/'weights'
    shutil.copytree(fit.study, study); shutil.copytree(fit.weights, weights)
    record = deepcopy(fit.report['languages']['ro'])
    if fault in ('frozen', 'initial', 'tensor_nan', 'no_tail_update', 'no_head_update'):
        epoch = 0 if fault == 'initial' else 1
        path = weights/'ro'/f'epoch-{epoch}.safetensors'; tensor = load_file(path)
        if fault.startswith('no_'):
            initial = load_file(fit.inputs['ro']['weights']['selected95'])
            prefix = 'base.' if fault == 'no_tail_update' else 'heads.ro.'
            tensor.update({k: v for k, v in initial.items() if k.startswith(prefix)})
        else:
            name = 'heads.uk.first.bias' if fault == 'frozen' else 'heads.ro.first.bias'
            tensor[name][0] += 1
            if fault == 'tensor_nan': tensor[name][0] = np.nan
        save_file(tensor, str(path)); record['checkpoints'][epoch]['adapter_sha256'] = file_sha(path)
    elif fault == 'missing_epoch': record['checkpoints'].pop()
    elif fault == 'metrics': record['checkpoints'][1]['joint_argmax_errors'] += 1
    elif fault == 'counter': record['checkpoints'][1]['completed_training_updates'] += 1
    elif fault == 'selection': record['selected']['epoch'] = 9
    elif fault == 'identity':
        path = study/'ro-development-predictions.jsonl'; rows = read_jsonl(path)
        rows.reverse(); write_jsonl(path, rows)
    elif fault == 'spec':
        path = study/'ro-review.json'; spec = json.loads(path.read_text()); spec['threshold'] = -1
        path.write_text(json.dumps(spec))
    elif fault == 'initial_parity': record['maximum_initial_logit_error'] = 9e-6
    elif fault == 'reload_values':
        path = weights/'ro-selected-reload-dev-logits.npy'; z = np.load(path); z[0, 0] += .1
        np.save(path, z); record['selected_reload_logits_sha256'] = file_sha(path)
    else: record['selected_reload_decision_parity']['argmax_rows_differing'] = 1
    with pytest.raises(ValueError):
        audit_one(fit, study=study, weights=weights, record=record)


@pytest.mark.parametrize('fault', ['source', 'index', 'identity', 'partition', 'slot', 'order', 'boolean'])
def test_occurrence_corruption_rejected(actual_pair, fault):
    fit = actual_pair[0]['control']; current = fit.inputs['ro']
    observed = read_jsonl(fit.weights/'ro-occurrences.jsonl')
    if fault == 'source': observed[-1]['source'] = 'new'
    elif fault == 'index': observed[-1]['source_index'] = 2  # DEV must never supply an extra control.
    elif fault == 'identity': observed[-1]['case_id'] = 'different'
    elif fault == 'partition': observed[-1]['partition'] = 'dev'
    elif fault == 'slot': observed[-1]['slot'] -= 1
    elif fault == 'order': observed.reverse()
    else: observed[0]['slot'] = False
    with pytest.raises(ValueError):
        audit_occurrences(current['control'], current['base'], current['new'], observed, 'control', 1073)


def test_class_weights_and_paired_identity_corruption(actual_pair):
    runs, _ = actual_pair; fit = runs['control']; current = fit.inputs['ro']
    bad = deepcopy(fit.report['languages']['ro']['field_weights']); bad[0][0] += .1
    with pytest.raises(ValueError): audit_class_weights(current['control'], bad)
    c = read_jsonl(fit.study/'ro-development-predictions.jsonl')
    t = read_jsonl(runs['treatment'].study/'ro-development-predictions.jsonl')
    rows = [r for r in current['base'] if r['partition'] == 'dev']
    value = paired_outcomes(rows, c, t, current['initial_dev_logits'], current['specs']['selected95'])
    assert value['treatment_vs_control']['rows'] == 2
    with pytest.raises(ValueError):
        paired_outcomes(rows, c, list(reversed(t)), current['initial_dev_logits'], current['specs']['selected95'])
    t[0]['baseline_v9'] = 1-c[0]['baseline_v9']
    with pytest.raises(ValueError):
        paired_outcomes(rows, c, t, current['initial_dev_logits'], current['specs']['selected95'])


def history():
    from test_runtime_attention_continuation_verifier_v10 import history_fixture
    report, schedules, groups, old = history_fixture()
    new_counts = {'ro': 1584, 'uk': 1584}; events = []
    for event in old:
        event = deepcopy(event); events.append(event)
        progress = event.get('inference', {})
        if progress.get('stage') == 'train-source-probes' and progress['checked'] == progress['rows']:
            for n in list(range(100, 1584, 100))+[1584]:
                events.append(dict(language=event['language'], inference=dict(stage='new-initial-reference', checked=n, rows=1584)))
    for index, event in enumerate(events): event['seconds'] = float(index+1)
    report['seconds'] = events[-1]['seconds']
    return report, schedules, groups, new_counts, events


def encode(events, second=True):
    result = b'{"starting": "control"}\n'+b''.join((json.dumps(e)+'\n').encode() for e in events)
    if second: result += b'{"starting": "treatment"}\n'+b''.join((json.dumps(e)+'\n').encode() for e in events)
    return result


def test_paired_log_boundaries_and_full_partial_batch_schedule():
    report, schedules, groups, counts, events = history()
    for arm, next_marker in [('control', 'treatment'), ('treatment', None)]:
        assert audit_training_log(encode(events), report, schedules, groups, counts,
                                  marker=arm, next_marker=next_marker)['events'] == len(events)


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'counter', 'loss', 'time', 'bool_time', 'stage', 'reference_count',
                                  'unknown', 'terminal', 'wrong_boundary', 'elapsed'])
def test_corrupted_paired_log_rejected(fault):
    report, schedules, groups, counts, events = history()
    update = next(i for i, e in enumerate(events) if 'step' in e)
    reference = next(i for i, e in enumerate(events) if e.get('inference', {}).get('stage') == 'new-initial-reference')
    if fault == 'missing': events.pop(reference)
    elif fault == 'duplicate': events.insert(update, deepcopy(events[update]))
    elif fault == 'counter': events[update]['presentations'] += 1
    elif fault == 'loss': events[update]['loss'] = float('nan')
    elif fault == 'time': events[update]['seconds'] = 0
    elif fault == 'bool_time': events[0]['seconds'] = True
    elif fault == 'stage': events[reference]['inference']['stage'] = 'epoch-0'
    elif fault == 'reference_count': events[reference]['inference']['rows'] -= 1
    elif fault == 'unknown': events[reference]['extra'] = True
    elif fault == 'terminal': events[-1]['complete'] = False
    elif fault == 'elapsed': report['seconds'] += 1
    with pytest.raises(ValueError):
        audit_training_log(encode(events, second=fault != 'wrong_boundary'), report, schedules, groups, counts,
                           marker='control', next_marker='treatment')


def test_incomplete_arm_rejected_before_numeric_imports_or_output(tmp_path, monkeypatch):
    from eval import verify_matched_role_attention_v10 as auditor
    training = tmp_path/'training.json'
    training.write_text(json.dumps({'outputs': {a: {'study': a} for a in ('control', 'treatment')}}))
    for arm in ('control', 'treatment'):
        (tmp_path/arm).mkdir(); (tmp_path/arm/'report.json').write_text('{"complete": false}')
    plan = tmp_path/'audit.json'
    plan.write_text(json.dumps(dict(format='roguard-matched-role-attention-audit-plan-1', training_plan='training.json')))
    output = tmp_path/'verification.json'
    monkeypatch.setattr(sys, 'argv', ['audit', '--root', str(tmp_path), '--private-root', str(tmp_path),
        '--audit-plan', str(plan), '--audit-plan-sha256', file_sha(plan), '--producer-pid', '123', '--output', str(output)])
    with pytest.raises(ValueError, match='incomplete'):
        auditor.main()
    assert not output.exists()
