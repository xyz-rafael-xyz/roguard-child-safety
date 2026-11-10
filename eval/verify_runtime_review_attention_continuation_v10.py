"""Recount all runtime-attention checkpoints, reload outputs and training events."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

from eval.audit_attention_development_v10 import verify_checkpoint
from roguard.trajectory_data import file_sha, read_jsonl


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def parity(actual, reference, spec):
    """Independently compare all seven heads and the unchanged served threshold."""
    import numpy as np
    from roguard.gemma4_review import _score
    require(actual.shape == reference.shape and actual.ndim == 2 and actual.shape[1] == 19
            and len(actual) > 0 and actual.dtype == reference.dtype == np.float32
            and np.isfinite(actual).all() and np.isfinite(reference).all(), 'Invalid reload population')
    error = float(np.abs(actual-reference).max())
    a = np.column_stack([z.argmax(axis=1) for z in np.split(actual, [3, 6, 10, 13, 15, 17], axis=1)])
    b = np.column_stack([z.argmax(axis=1) for z in np.split(reference, [3, 6, 10, 13, 15, 17], axis=1)])
    changed = int(np.any(a != b, axis=1).sum())
    binary = sum((_score(left, spec) >= spec['threshold']) != (_score(right, spec) >= spec['threshold'])
                 for left, right in zip(actual, reference))
    require(error <= 1e-5 and changed == binary == 0, 'Reload or initialization decisions differ')
    return dict(maximum_logit_error=error, argmax_rows_differing=changed,
                fixed_threshold_decision_differences=binary)


def audit_language(study, weights, language, record, rows, original, initial, teacher,
                   original_spec, initial_spec, initial_logits, *, epochs=3, batch_size=8):
    """Check every retained checkpoint, including nonselected epochs."""
    import numpy as np
    from safetensors.numpy import load_file
    from roguard.gemma4_review import _score
    dev = [i for i, row in enumerate(rows) if row['partition'] == 'dev']
    train_count = sum(row['partition'] == 'train' for row in rows)
    require(train_count + len(dev) == len(rows) and train_count and dev, 'Unexpected training population')
    dev_rows = [rows[i] for i in dev]
    require(teacher.shape == (len(rows), 19) and teacher.dtype == np.float32
            and np.isfinite(teacher).all(), 'Invalid original-v9 teacher logits')
    baseline = [int(_score(teacher[i], original_spec) >= original_spec['threshold']) for i in dev]
    require(record.get('complete') is True and record['training_rows'] == train_count
            and record['development_rows'] == len(dev) and record['trainable_tensors'] == 76,
            'Language completion counters differ')
    for key, limit in (('maximum_initial_feature_error', 1e-3), ('maximum_initial_logit_error', 1e-5),
                       ('maximum_full_replay_error', 1e-3), ('selected_reload_maximum_logit_error', 1e-5)):
        require(math.isfinite(record[key]) and 0 <= record[key] <= limit, 'Reported numerical check failed')
    history = record['checkpoints']
    require([c['epoch'] for c in history] == list(range(epochs+1)), 'Checkpoint sequence differs')
    trainable = {f'base.model.layers.{i}.self_attn.{projection}.lora_{part}'
                 for i in range(24, 42) for projection in ('q_proj', 'o_proj') for part in ('a', 'b')}
    head_names = {f'heads.{language}.{layer}.{part}' for layer in ('first', 'last') for part in ('weight', 'bias')}
    trainable |= head_names
    require(len(trainable) == 76 and trainable <= set(original) and set(initial) == set(original), 'Parameter scope differs')
    require(all(np.array_equal(value, original[name]) for name, value in initial.items() if name not in head_names),
            'Warm start changes frozen v9 tensors')
    maximum_policy_error = 0.
    artifacts = {}
    def bind(path):
        value = file_sha(path); artifacts[str(path.relative_to(weights))] = value
        return value
    for epoch, checkpoint in enumerate(history):
        path = weights/language/f'epoch-{epoch}.safetensors'; tensor = load_file(path)
        require(bind(path) == checkpoint['adapter_sha256'] and set(tensor) == set(original), 'Checkpoint weights changed')
        for name, value in tensor.items():
            require(value.shape == original[name].shape and value.dtype == np.float32 and np.isfinite(value).all(),
                    'Invalid checkpoint tensor')
            if name not in trainable:
                require(np.array_equal(value, original[name]), 'Frozen checkpoint tensor changed')
            if epoch == 0:
                require(np.array_equal(value, initial[name]), 'Epoch-zero checkpoint differs from initialization')
        if epoch:
            for scope in (trainable-head_names, head_names):
                require(any(not np.array_equal(tensor[name], initial[name]) for name in scope),
                        'Trained checkpoint leaves an entire trainable component unchanged')
        path = weights/language/f'epoch-{epoch}-dev-logits.npy'
        require(bind(path) == checkpoint['development_logits_sha256'], 'Checkpoint logits changed')
        z = np.load(path, allow_pickle=False)
        require(z.dtype == np.float32, 'Checkpoint logit dtype differs')
        if epoch == 0:
            initial_parity = parity(z, initial_logits, initial_spec)
            require(initial_parity == record['initial_runtime_decision_parity']
                    and initial_parity['maximum_logit_error'] == record['maximum_initial_logit_error'],
                    'Initial runtime parity report differs')
        _, _, error = verify_checkpoint(dev_rows, z, original_spec['couplings'], baseline, checkpoint,
                                        epoch=epoch, train_rows=train_count, batch_size=batch_size)
        maximum_policy_error = max(maximum_policy_error, error)
    selected = min(history, key=lambda c: c['selection_rank'])
    require(record['selected'] == selected, 'Selected checkpoint violates earliest-tie selection')
    require(bind(weights/(language+'-selected.safetensors')) == selected['adapter_sha256'], 'Selected weights changed')
    path = weights/(language+'-selected-dev-logits.npy')
    require(bind(path) == selected['development_logits_sha256'], 'Selected logits changed')
    z = np.load(path, allow_pickle=False)
    predictions, scores, _ = verify_checkpoint(dev_rows, z, original_spec['couplings'], baseline, selected,
                                               epoch=selected['epoch'], train_rows=train_count, batch_size=batch_size)
    expected = [dict(case_id=row['case_id'], language=language, group=row['group'], gold=row['gold'],
                     predicted=int(pred), baseline_v9=int(old), score=float(score))
                for row, pred, old, score in zip(dev_rows, predictions, baseline, scores)]
    require(read_jsonl(study/(language+'-development-predictions.jsonl')) == expected, 'Selected identities or decisions differ')
    spec = read(study/(language+'-review.json'))
    require(spec == dict(original_spec, adapter_sha256=selected['adapter_sha256'], mode=selected['mode'],
                         threshold=selected['threshold']), 'Selected runtime specification changed')
    path = weights/(language+'-selected-reload-dev-logits.npy')
    require(bind(path) == record['selected_reload_logits_sha256'], 'Retained reload outputs changed')
    reload_parity = parity(np.load(path, allow_pickle=False), z, spec)
    require(reload_parity == record['selected_reload_decision_parity']
            and reload_parity['maximum_logit_error'] == record['selected_reload_maximum_logit_error'],
            'Selected reload parity report differs')
    return dict(epoch=selected['epoch'], rows=len(dev), binary_errors=selected['binary_errors'],
        joint_argmax_errors=selected['joint_argmax_errors'], fixed_v9=selected['fixed_v9'],
        introduced_v9=selected['introduced_v9'], checkpoint_count=len(history),
        checkpoint_dev_rows_recounted=len(history)*len(dev), reload_rows_recounted=len(dev),
        maximum_policy_enumeration_error=maximum_policy_error, artifacts=artifacts)


def audit_training_log(raw, report, schedules, train_groups, *, marker='train-runtime-review-attention95', batch_size=8):
    """Verify the ordered update, inference-progress, selection and reload history."""
    records = raw.splitlines(); start = json.dumps({'starting': marker}).encode()
    require(records.count(start) == 1, 'Expected one registered training start')
    events = [json.loads(line) for line in records[records.index(start)+1:] if line.strip()]
    require(bool(events) and all(isinstance(e, dict) and math.isfinite(e.get('seconds', float('nan')))
            and e['seconds'] >= 0 for e in events), 'Invalid training event time')
    require(all(a['seconds'] < b['seconds'] for a, b in zip(events, events[1:])), 'Nonmonotone training event time')
    expected = []
    def progress(language, stage, n):
        require(n > 0, 'Empty inference population')
        for checked in sorted(set(range(100, n+1, 100)) | {n}):
            expected.append(dict(language=language, inference=dict(stage=stage, checked=checked, rows=n)))
    for language in ('ro', 'uk'):
        record = report['languages'][language]; schedule = schedules[language]
        n = record['training_rows']; per = (n+batch_size-1)//batch_size
        require(schedule == dict(updates_per_epoch=per, total_updates=3*per, total_presentations=3*n), 'Log schedule differs')
        progress(language, 'train-source-probes', train_groups[language])
        for epoch in range(4):
            if epoch:
                for step in range((epoch-1)*per+1, epoch*per+1):
                    if step % 25 and step != epoch*per:
                        continue
                    expected.append(dict(language=language, epoch=epoch, step=step, total_steps=3*per,
                        presentations=(epoch-1)*n+min((step-(epoch-1)*per)*batch_size, n)))
            progress(language, f'epoch-{epoch}', record['development_rows'])
            selected = min(record['checkpoints'][:epoch+1], key=lambda c: c['selection_rank'])
            checkpoint = record['checkpoints'][epoch]
            expected.append(dict(language=language, epoch=epoch, selected_epoch=selected['epoch'],
                **{key: checkpoint[key] for key in ('binary_errors', 'joint_argmax_errors', 'introduced_v9')}))
        progress(language, 'selected-reload', record['development_rows'])
    expected.append(dict(complete=True, decision=report['decision']))
    require(len(events) == len(expected), 'Missing or extra training events')
    for event, wanted in zip(events, expected):
        trimmed = {key: value for key, value in event.items() if key != 'seconds'}
        if 'step' in wanted:
            require('loss' in trimmed and math.isfinite(trimmed.pop('loss')), 'Nonfinite or missing loss')
        require(trimmed == wanted, 'Training event order, population, counters or selection differ')
    require(events[-1]['seconds'] == report['seconds'], 'Terminal elapsed time differs')
    return dict(training_events=len(events), complete_log_sha256=hashlib.sha256(raw).hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'private', 'amendment', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--amendment-sha256', required=True)
    p.add_argument('--sequence-pid', type=int, required=True)
    a = p.parse_args()
    require(not a.output.exists(), 'Refusing to overwrite verification')
    require(file_sha(a.amendment) == a.amendment_sha256, 'Registration changed')
    e = a.root/'research/v10-evidence'; study = e/'runtime-review-attention-95'; weights = a.private/'runtime-review-attention-95'
    report = read(study/'report.json'); protocol = read(study/'protocol.json'); plan = read(a.amendment)
    require(report.get('complete') is True and set(report['languages']) == {'ro', 'uk'}, 'Training is incomplete')
    sequence_path = e/'runtime-review-attention-sequence-95/report.json'; sequence = read(sequence_path)
    queue_path = a.private/'runtime-review-attention-sequence-95-plan.json'; queue = read(queue_path)
    require(sequence['plan_sha256'] == file_sha(queue_path) and sequence['source_hashes'] == queue['source_hashes']
            and [s['name'] for s in sequence['completed_steps']] == [s['name'] for s in queue['steps']], 'Training queue changed')
    from scripts.wait_completed_sequence_exit_v10 import require_exit
    require_exit(sequence, a.sequence_pid)
    for name, checksum in queue['source_hashes'].items():
        require(file_sha(a.root/name) == checksum, 'Training queue source changed')
    require(protocol['source_sha256'] == plan['source_sha256'] == file_sha(study/'source.py')
            == file_sha(a.root/'training/train_runtime_review_attention_continuation_v10.py'), 'Trainer source changed')
    require(protocol['amendment_sha256'] == file_sha(a.amendment), 'Protocol registration changed')
    audit_sources = plan['completion_audit_sources']
    require(set(audit_sources) == {'eval/verify_runtime_review_attention_continuation_v10.py',
            'eval/audit_attention_development_v10.py', 'tests/test_runtime_attention_continuation_verifier_v10.py'},
            'Completion audit source inventory differs')
    fixed = {a.root/name: checksum for name, checksum in queue['source_hashes'].items()}
    for name, checksum in audit_sources.items():
        require(file_sha(a.root/name) == checksum, 'Completion audit source changed')
        fixed[a.root/name] = checksum
    for path in (a.amendment, study/'report.json', study/'protocol.json', study/'source.py',
                 study/'helper.py', sequence_path, queue_path,
                 a.private/'runtime-review-attention-sequence-95.log'):
        fixed[path] = file_sha(path)
    for lang in ('ro', 'uk'):
        for suffix in ('review.json', 'development-predictions.jsonl'):
            path = study/(lang+'-'+suffix); fixed[path] = file_sha(path)
    from training.train_runtime_review_attention_continuation_v10 import load_inputs
    args = SimpleNamespace(root=a.root, private_root=a.private,
        input_study=e/'large-review-input-recovery-72', inputs=a.private/'large-review-input-recovery-72/actual',
        feature_study=e/'crossed-review-features-62', features=a.private/'crossed-review-features-62',
        initial_study=e/'dropout-readout-75', initial_private=a.private/'dropout-readout-75',
        candidate=a.root/'research/v9-stabilization/reported-tail-fit-196')
    rows, bindings = load_inputs(args)
    require(bindings == protocol['bindings'] == plan['input_bindings'], 'Completed training inputs changed')
    require(file_sha(study/'helper.py') == bindings['runtime_helpers']['training/runtime_attention_continuation_v10.py'], 'Helper snapshot changed')
    for key in ('optimization', 'schedules', 'populations'):
        require(protocol[key] == plan[key], 'Registered configuration differs')
    require(protocol['optimization'] == dict(epochs=3, batch_size=8, seed=1073, learning_rate=1e-5,
        head_learning_rate=1e-4, weight_decay=.01, gradient_norm_clip=1., teacher_kl=.2,
        policy_ce={'ro': .25, 'uk': 1.}, head_dropout={'ro': .1, 'uk': 0.}), 'Unregistered optimization')
    require(report['training_rows'] == 28929 and report['development_rows'] == 8837, 'Population totals differ')
    require(report['held_out_test_opened'] is protocol['held_out_test_opened'] is False
            and report['stable_release_ready'] is protocol['stable_release_ready'] is False, 'Unsupported qualification claim')
    require(protocol['native_comparison'] == 'cached suffix with a copied fixed387-token prefix'
            and protocol['initialization'] == 'selected75 with training-only head dropout'
            and protocol['persistent_body_cache'] is False, 'Runtime training contract differs')
    import numpy as np
    from safetensors.numpy import load_file
    summaries = {}; groups = {}
    for lang, counts in (('ro', {'train': 13174, 'dev': 4202}), ('uk', {'train': 15755, 'dev': 4635})):
        require({split: sum(r['partition'] == split for r in rows[lang]) for split in counts}
                == counts == protocol['populations'][lang], 'Language populations differ')
        groups[lang] = len({r['group'] for r in rows[lang] if r['partition'] == 'train'})
        summaries[lang] = audit_language(study, weights, lang, report['languages'][lang], rows[lang],
            load_file(args.candidate/'actual'/(lang+'-selected.safetensors')),
            load_file(args.initial_private/(lang+'-selected.safetensors')),
            np.load(a.private/'runtime-review-baseline-88'/(lang+'-logits.npy'), mmap_mode='r', allow_pickle=False),
            read(e/'crossed-review-features-62/v9-binding'/(lang+'-review.json')),
            read(args.initial_study/(lang+'-review.json')),
            np.load(a.private/'runtime-warm-start-90'/(lang+'-selected-dev-logits.npy'), allow_pickle=False))
    decision = ('REJECT' if any(s['binary_errors'] or s['joint_argmax_errors'] for s in summaries.values())
                else 'DEVELOPMENT_PASS_REQUIRES_FULL_REGRESSION')
    require(report['decision'] == decision, 'Qualification prerequisite decision differs')
    log = audit_training_log((a.private/'runtime-review-attention-sequence-95.log').read_bytes(), report, protocol['schedules'], groups)
    require(load_inputs(args)[1] == bindings, 'Inputs changed during audit')
    for summary in summaries.values():
        for name, checksum in summary['artifacts'].items():
            fixed[weights/name] = checksum
    require(all(file_sha(path) == checksum for path, checksum in fixed.items()), 'Audited evidence changed during verification')
    result = dict(verification='PASS', source_sha256=file_sha(__file__),
        audit_source_sha256=file_sha(a.root/'eval/audit_attention_development_v10.py'),
        report_sha256=file_sha(study/'report.json'), protocol_sha256=file_sha(study/'protocol.json'),
        amendment_sha256=file_sha(a.amendment), sequence_sha256=file_sha(sequence_path),
        checkpoint_count=sum(s['checkpoint_count'] for s in summaries.values()),
        checkpoint_dev_rows_recounted=sum(s['checkpoint_dev_rows_recounted'] for s in summaries.values()),
        reload_rows_recounted=sum(s['reload_rows_recounted'] for s in summaries.values()),
        selected=summaries, decision=decision, **log, encoder_loaded=False, model_qualified=False,
        scope='All checkpoint tensors and saved DEV outputs, independently enumerated policy probabilities and threshold '
              'selection, exact initialization, retained reload logits, source/queue bindings and full event history. '
              'Native encoder feature/replay maxima remain producer measurements. No encoder rerun or independent gold validation.')
    a.output.write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
