"""Verify every completed study73 checkpoint and saved DEV logit population."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from eval.audit_attention_development_v10 import verify_checkpoint


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def audit_training_log(raw, report, schedules):
    records = raw.splitlines()
    marker = b'{"starting": "train-large-review-attention73"}'
    require(records.count(marker) == 1, 'Expected one registered training start')
    events = [json.loads(line) for line in records[records.index(marker)+1:] if line.strip()]
    require(events[-1]['complete'] is True and events[-1]['decision'] == report['decision'], 'Terminal training log differs')
    require(all(math.isfinite(e['seconds']) and e['seconds'] >= 0 for e in events), 'Invalid training event time')
    require(all(a['seconds'] < b['seconds'] for a, b in zip(events, events[1:])), 'Nonmonotone training event time')
    summaries = [e for e in events if 'epoch' in e and 'step' not in e]
    require([(e['language'], e['epoch']) for e in summaries] == [(lang, ep) for lang in ('ro', 'uk') for ep in range(4)],
            'Missing or repeated epoch event')
    for lang in ('ro', 'uk'):
        schedule = schedules[lang]; record = report['languages'][lang]
        updates = [e for e in events if e.get('language') == lang and 'step' in e]
        per_epoch = schedule['updates_per_epoch']; total = schedule['total_updates']
        expected = sorted(set(range(25, total+1, 25)) | {per_epoch*epoch for epoch in (1, 2, 3)})
        require([e['step'] for e in updates] == expected, 'Missing or repeated update event')
        for event in updates:
            epoch = (event['step']-1)//per_epoch+1
            in_epoch = event['step']-(epoch-1)*per_epoch
            presentations = (epoch-1)*record['training_rows']+min(in_epoch*8, record['training_rows'])
            require(event['epoch'] == epoch and event['total_steps'] == total and event['presentations'] == presentations,
                    'Logged schedule or presentations differ')
            require(math.isfinite(event['loss']), 'Nonfinite loss')
        for event in [e for e in summaries if e['language'] == lang]:
            epoch = event['epoch']; checkpoint = record['checkpoints'][epoch]
            require(all(event[k] == checkpoint[k] for k in ('binary_errors', 'joint_argmax_errors', 'introduced_v9')),
                    'Logged DEV errors differ')
            selected = min(record['checkpoints'][:epoch+1], key=lambda c: c['selection_rank'])
            require(event['selected_epoch'] == selected['epoch'], 'Logged selection differs')
    return {'training_events': len(events), 'complete_log_sha256': hashlib.sha256(raw).hexdigest()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--private', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    require(not a.output.exists(), 'Refusing to overwrite verification')
    import numpy as np
    from safetensors.numpy import load_file
    from roguard.gemma4_review import _score
    e = a.root/'research/v10-evidence'; study = e/'large-review-attention-73'; weights = a.private/'large-review-attention-73'
    amendment_path = e/'protocol-amendment-66.json'
    require(sha(amendment_path) == 'd980769ca2df00d35bba6e4fc25085765040516e3eed8a56da503909adcfa16e', 'Registration changed')
    amendment = read(amendment_path); protocol = read(study/'protocol.json'); report = read(study/'report.json')
    sequence = read(e/'large-review-attention-sequence-73/report.json')
    queue_path = a.private/'large-review-attention-sequence-73-plan.json'; queue = read(queue_path)
    require(sequence['execution_complete'] is True and sequence['phase'] == 'finished', 'Sequence is incomplete')
    require([step['name'] for step in sequence['completed_steps']] == [step['name'] for step in queue['steps']]
            and all(step['exit_code'] == 0 for step in sequence['completed_steps']), 'Sequence steps failed or differ')
    require(sequence['plan_sha256'] == sha(queue_path) and sequence['source_hashes'] == queue['source_hashes'], 'Queue changed')
    for path, value in queue['source_hashes'].items():
        require(sha(a.root/path) == value, 'Bound queue source changed: '+path)
    pid = int((a.private/'large-review-attention-sequence-73.pid').read_text())
    try: os.kill(pid, 0)
    except ProcessLookupError: pass
    else: raise ValueError('Sequence process has not exited')
    require(protocol['source_sha256'] == amendment['source_sha256'] == sha(study/'source.py')
            == sha(a.root/'training/train_large_review_attention_v10.py'), 'Trainer source changed')
    require(protocol['amendment_sha256'] == sha(amendment_path), 'Protocol registration changed')
    require(protocol['bindings'] == amendment['input_bindings'], 'Protocol data bindings changed')
    for key in ('optimization', 'schedules', 'populations'):
        require(protocol[key] == amendment[key], 'Protocol configuration changed: '+key)
    bindings = protocol['bindings']
    require(sha(study/'helper.py') == bindings['helpers']['training/large_review_attention_v10.py'], 'Helper snapshot changed')
    require(sha(a.private/'large-review-input-recovery-72/actual/inputs.jsonl') == bindings['inputs_sha256'], 'Recovered inputs changed')
    for path, key in ((e/'large-review-input-recovery-72/report.json', 'input_report_sha256'),
                      (e/'large-review-input-recovery-72/verification.json', 'input_verification_sha256'),
                      (e/'crossed-review-features-62/report.json', 'feature_report_sha256'),
                      (e/'policy-consistent-readout-71/report.json', 'initial_report_sha256')):
        require(sha(path) == bindings[key], 'Input report changed')
    require(report['complete'] is True and set(report['languages']) == {'ro', 'uk'}, 'Training is incomplete')
    require(report['training_rows'] == 28929 and report['development_rows'] == 8837, 'Population totals differ')
    require(report['held_out_test_opened'] is protocol['held_out_test_opened'] is False
            and report['stable_release_ready'] is protocol['stable_release_ready'] is False, 'Unsupported qualification claim')
    summaries = {}; checkpoint_count = 0; maximum_policy_error = 0.
    for lang in ('ro', 'uk'):
        bound = bindings[lang]; features = a.private/'crossed-review-features-62'
        for suffix, value in bound['feature_artifacts'].items():
            require(sha(features/(lang+'-'+suffix)) == value, 'Feature store changed')
        metadata = lines(features/(lang+'-rows.jsonl'))
        dev = [i for i, row in enumerate(metadata) if row['partition'] == 'dev']; rows = [metadata[i] for i in dev]
        train_count = sum(row['partition'] == 'train' for row in metadata)
        original_path = a.root/'research/v9-stabilization/reported-tail-fit-196/actual'/(lang+'-selected.safetensors')
        initial_path = a.private/'policy-consistent-readout-71'/(lang+'-selected.safetensors')
        initial_logits_path = a.private/'policy-consistent-readout-71'/(lang+'-selected-dev-logits.npy')
        original_spec_path = e/'crossed-review-features-62/v9-binding'/(lang+'-review.json')
        require(sha(original_path) == bound['original_weights_sha256'] and sha(initial_path) == bound['initial_weights_sha256']
                and sha(original_spec_path) == bound['original_spec_sha256']
                and sha(initial_logits_path) == bound['initial_dev_logits_sha256']
                and sha(e/'policy-consistent-readout-71'/(lang+'-review.json')) == bound['initial_spec_sha256'],
                'Initial checkpoint changed')
        original = load_file(original_path); initial = load_file(initial_path); original_spec = read(original_spec_path)
        teacher = np.load(features/(lang+'-logits.npy'), mmap_mode='r', allow_pickle=False)
        baseline = [int(_score(teacher[i], original_spec) >= original_spec['threshold']) for i in dev]
        record = report['languages'][lang]
        require(record['complete'] is True and record['training_rows'] == train_count
                and record['development_rows'] == len(rows) and record['trainable_tensors'] == 76, 'Language counters differ')
        for key, limit in (('maximum_initial_feature_error', 1e-3), ('maximum_initial_logit_error', 1e-3),
                           ('maximum_full_replay_error', 1e-3), ('selected_reload_maximum_logit_error', 1e-5)):
            require(math.isfinite(record[key]) and 0 <= record[key] <= limit, 'Reported numerical check failed')
        history = record['checkpoints']
        require([c['epoch'] for c in history] == [0, 1, 2, 3], 'Checkpoint sequence differs')
        trainable = {f'base.model.layers.{i}.self_attn.{p}.lora_{f}' for i in range(24, 42)
                     for p in ('q_proj', 'o_proj') for f in ('a', 'b')}
        head_names = {f'heads.{lang}.{layer}.{part}' for layer in ('first', 'last') for part in ('weight', 'bias')}
        trainable |= head_names
        require(len(trainable) == 76 and trainable <= set(original) and set(initial) == set(original), 'Parameter scope differs')
        require(all(np.array_equal(v, original[k]) for k, v in initial.items() if k not in head_names), 'Warm-start scope differs')
        for epoch, checkpoint in enumerate(history):
            path = weights/lang/f'epoch-{epoch}.safetensors'; tensor = load_file(path)
            require(sha(path) == checkpoint['adapter_sha256'] and set(tensor) == set(original), 'Retained checkpoint changed')
            for name, value in tensor.items():
                require(value.shape == original[name].shape and value.dtype == np.float32 and np.isfinite(value).all(), 'Invalid tensor')
                if name not in trainable:
                    require(np.array_equal(value, original[name]), 'Frozen checkpoint tensor changed')
                if epoch == 0:
                    require(np.array_equal(value, initial[name]), 'Epoch-zero warm start differs')
            logit_path = weights/lang/f'epoch-{epoch}-dev-logits.npy'
            require(sha(logit_path) == checkpoint['development_logits_sha256'], 'Checkpoint logits changed')
            z = np.load(logit_path, allow_pickle=False)
            if epoch == 0:
                initial_logits = np.load(initial_logits_path, allow_pickle=False)
                require(z.shape == initial_logits.shape
                        and float(np.max(np.abs(z-initial_logits))) <= 1e-3, 'Saved initial logits differ from warm start')
            _, _, error = verify_checkpoint(rows, z, original_spec['couplings'], baseline, checkpoint,
                                            epoch=epoch, train_rows=train_count, batch_size=8)
            maximum_policy_error = max(maximum_policy_error, error); checkpoint_count += 1
        selected = min(history, key=lambda c: c['selection_rank'])
        require(record['selected'] == selected, 'Selected checkpoint violates the earliest-tie rule')
        require(sha(weights/(lang+'-selected.safetensors')) == selected['adapter_sha256'], 'Selected weights differ')
        selected_logits_path = weights/(lang+'-selected-dev-logits.npy')
        require(sha(selected_logits_path) == selected['development_logits_sha256'], 'Selected logits differ')
        z = np.load(selected_logits_path, allow_pickle=False)
        predictions, scores, _ = verify_checkpoint(rows, z, original_spec['couplings'], baseline, selected,
                                                   epoch=selected['epoch'], train_rows=train_count, batch_size=8)
        saved = lines(study/(lang+'-development-predictions.jsonl'))
        expected = [{'case_id': row['case_id'], 'language': lang, 'group': row['group'], 'gold': row['gold'],
                     'predicted': int(pred), 'baseline_v9': int(old), 'score': float(score)}
                    for row, pred, old, score in zip(rows, predictions, baseline, scores)]
        require(saved == expected, 'Selected per-case identities or decisions differ')
        spec = read(study/(lang+'-review.json'))
        require(spec == dict(original_spec, adapter_sha256=selected['adapter_sha256'], mode=selected['mode'],
                             threshold=selected['threshold']), 'Selected runtime specification changed')
        summaries[lang] = {'epoch': selected['epoch'], 'rows': len(rows), 'binary_errors': selected['binary_errors'],
                           'joint_argmax_errors': selected['joint_argmax_errors'], 'fixed_v9': selected['fixed_v9'],
                           'introduced_v9': selected['introduced_v9']}
    decision = ('REJECT' if any(s['binary_errors'] or s['joint_argmax_errors'] for s in summaries.values())
                else 'DEVELOPMENT_PASS_REQUIRES_FULL_REGRESSION')
    require(report['decision'] == decision, 'Qualification prerequisite decision differs')
    log = audit_training_log((a.private/'large-review-attention-sequence-73.log').read_bytes(), report, protocol['schedules'])
    result = {'verification': 'PASS', 'source_sha256': sha(Path(__file__)),
              'audit_source_sha256': sha(a.root/'eval/audit_attention_development_v10.py'),
              'report_sha256': sha(study/'report.json'), 'protocol_sha256': sha(study/'protocol.json'),
              'checkpoint_count': checkpoint_count, 'checkpoint_dev_rows_recounted': 4*8837,
              'maximum_policy_enumeration_error': maximum_policy_error, 'selected': summaries,
              'decision': decision, **log, 'encoder_loaded': False, 'model_qualified': False,
              'scope': 'Stored checkpoint tensors, sources, all epoch DEV logits/statistics, independent threshold sweep, '
                       'world-enumeration policy probability, paired decisions, selected spec and complete update log. '
                       'This does not rerun model inference or independently validate source gold labels.'}
    a.output.write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
