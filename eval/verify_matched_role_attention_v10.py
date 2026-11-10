"""Audit both matched attention arms without loading the production encoder."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from eval.audit_attention_development_v10 import verify_checkpoint
from eval.verify_runtime_review_attention_continuation_v10 import parity, require, read
from roguard.assurance import digest
from roguard.trajectory_data import file_sha, read_jsonl


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
        require(type(record[key]) in (int, float) and math.isfinite(record[key]) and 0 <= record[key] <= limit, 'Reported numerical check failed')
    history = record['checkpoints']
    require([c['epoch'] for c in history] == list(range(epochs+1)), 'Checkpoint sequence differs')
    trainable = {f'base.model.layers.{i}.self_attn.{projection}.lora_{part}'
                 for i in range(24, 42) for projection in ('q_proj', 'o_proj') for part in ('a', 'b')}
    head_names = {f'heads.{language}.{layer}.{part}' for layer in ('first', 'last') for part in ('weight', 'bias')}
    trainable |= head_names
    require(len(trainable) == 76 and trainable <= set(original) and set(initial) == set(original), 'Parameter scope differs')
    require(all(np.array_equal(value, original[name]) for name, value in initial.items() if name not in trainable),
            'Selected95 changes frozen v9 tensors')
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


def require_complete(report, arm):
    require(report.get('complete') is True and report.get('arm') == arm
            and set(report.get('languages', {})) == {'ro', 'uk'}
            and report.get('held_out_test_opened') is False
            and report.get('stable_release_ready') is False
            and all(v.get('complete') is True for v in report['languages'].values()),
            'Matched training arm is incomplete or has invalid scope')


def audit_occurrences(rows, base, new, observed, arm, seed):
    """Reconstruct matching from original TRAIN cells and source identities."""
    require(arm in ('control', 'treatment') and len(rows) == len(base)+len(new)
            and len(observed) == len(rows), 'Occurrence population differs')
    pools = {}
    for index, row in enumerate(base):
        if row['partition'] == 'train':
            pools.setdefault((tuple(row['fields']), row['gold']), []).append(index)
    language = base[0]['language']
    for key, pool in pools.items():
        pool.sort(key=lambda i: (digest([seed, language, key, base[i]['case_id']]), base[i]['case_id']))
    seen = Counter(); resolved = []
    for slot, (actual, occurrence) in enumerate(zip(rows, observed)):
        if slot < len(base):
            source, index, expected = 'base', slot, base[slot]
        else:
            addition = new[slot-len(base)]
            require(addition['partition'] == 'train', 'Reserved input used as an addition')
            key = tuple(addition['fields']), addition['gold']
            require(key in pools and bool(pools[key]), 'No original TRAIN control cell')
            index = pools[key][seen[key] % len(pools[key])]
            seen[key] += 1
            if arm == 'control':
                source, expected = 'base', base[index]
            else:
                source, index, expected = 'new', slot-len(base), addition
            require(expected['partition'] == 'train' and expected['fields'] == addition['fields']
                    and expected['gold'] == addition['gold'], 'Control label or partition differs')
        wanted = dict(slot=slot, case_id=expected['case_id'], partition=expected['partition'],
                      source=source, source_index=index, token_sha256=expected['token_sha256'])
        require(type(occurrence.get('slot')) is int and type(occurrence.get('source_index')) is int
                and occurrence == wanted and actual == expected, 'Retained occurrence mapping differs')
        resolved.append((source, index))
    return resolved


def audit_class_weights(rows, reported):
    """Recount TRAIN class weights without calling the trainer's helper."""
    import numpy as np
    from roguard.evidence import ONTOLOGY
    labels = [r['fields'] for r in rows if r['partition'] == 'train']
    require(bool(labels) and len(reported) == len(ONTOLOGY), 'Class-weight population differs')
    expected = []
    for column, values in enumerate(ONTOLOGY.values()):
        counts = Counter(row[column] for row in labels)
        weights = np.array([min(4., math.sqrt(len(labels)/max(1, counts[j])/len(values)))
                            for j in range(len(values))], dtype=np.float64)
        weights /= np.mean(weights[[row[column] for row in labels]])
        expected.append(weights.astype(np.float32).tolist())
    require(reported == expected, 'Reported TRAIN class weights differ')
    return expected


def audit_training_log(raw, report, schedules, train_groups, new_counts, *, marker, next_marker=None,
                       epochs=3, batch_size=8):
    lines = raw.splitlines(); start = json.dumps({'starting': marker}).encode()
    require(lines.count(start) == 1, 'Expected one registered arm start')
    first = lines.index(start)+1; last = len(lines)
    if next_marker is not None:
        end = json.dumps({'starting': next_marker}).encode()
        require(lines.count(end) == 1 and lines.index(end) >= first, 'Arm start order differs')
        last = lines.index(end)
    events = [json.loads(line) for line in lines[first:last] if line.strip()]
    require(bool(events) and all(isinstance(e, dict) and type(e.get('seconds')) in (float, int)
            and math.isfinite(e['seconds']) and e['seconds'] >= 0 for e in events), 'Invalid event time')
    require(all(a['seconds'] < b['seconds'] for a, b in zip(events, events[1:])), 'Nonmonotone event time')
    expected = []
    def progress(lang, stage, count):
        require(type(count) is int and count > 0, 'Empty inference population')
        for checked in sorted(set(range(100, count+1, 100)) | {count}):
            expected.append(dict(language=lang, inference=dict(stage=stage, checked=checked, rows=count)))
    for lang in ('ro', 'uk'):
        record = report['languages'][lang]; n = record['training_rows']; per = (n+batch_size-1)//batch_size
        require(schedules[lang] == dict(updates_per_epoch=per, total_updates=epochs*per,
                                       total_presentations=epochs*n), 'Log schedule differs')
        progress(lang, 'train-source-probes', train_groups[lang])
        progress(lang, 'new-initial-reference', new_counts[lang])
        for epoch in range(epochs+1):
            if epoch:
                for step in range((epoch-1)*per+1, epoch*per+1):
                    if step % 25 and step != epoch*per:
                        continue
                    expected.append(dict(language=lang, epoch=epoch, step=step, total_steps=epochs*per,
                        presentations=(epoch-1)*n+min((step-(epoch-1)*per)*batch_size, n)))
            progress(lang, f'epoch-{epoch}', record['development_rows'])
            selected = min(record['checkpoints'][:epoch+1], key=lambda c: c['selection_rank'])
            checkpoint = record['checkpoints'][epoch]
            expected.append(dict(language=lang, epoch=epoch, selected_epoch=selected['epoch'],
                **{k: checkpoint[k] for k in ('binary_errors', 'joint_argmax_errors', 'introduced_v9')}))
        progress(lang, 'selected-reload', record['development_rows'])
    expected.append(dict(complete=True, decision=report['decision']))
    require(len(events) == len(expected), 'Missing or extra arm events')
    for actual, wanted in zip(events, expected):
        value = {k: v for k, v in actual.items() if k != 'seconds'}
        if 'step' in wanted:
            loss = value.pop('loss', None)
            require(type(loss) in (int, float) and math.isfinite(loss), 'Missing or nonfinite loss')
        require(value == wanted, 'Arm event order, population, counters or selection differs')
    require(events[-1]['seconds'] == report['seconds'], 'Terminal arm time differs')
    return dict(events=len(events), segment_sha256=hashlib.sha256(b'\n'.join(lines[first:last])).hexdigest())


def paired_outcomes(rows, control, treatment, initial_logits, initial_spec):
    import numpy as np
    from roguard.gemma4_review import _score
    from eval.verify_matched_assertion_readout_v10 import paired_counts
    expected = [(r['case_id'], r['language'], r['group'], r['gold']) for r in rows]
    for predictions in (control, treatment):
        require([(r['case_id'], r['language'], r['group'], r['gold']) for r in predictions] == expected,
                'Paired development identities/order differ')
    require([r['baseline_v9'] for r in control] == [r['baseline_v9'] for r in treatment],
            'Paired original-v9 baseline decisions differ')
    require(initial_logits.shape == (len(rows), 19) and initial_logits.dtype == np.float32
            and np.isfinite(initial_logits).all(), 'Invalid selected95 paired reference')
    gold = [r['gold'] for r in rows]
    c = [r['predicted'] for r in control]; t = [r['predicted'] for r in treatment]
    v9 = [r['baseline_v9'] for r in control]
    initial = [int(_score(z, initial_spec) >= initial_spec['threshold']) for z in initial_logits]
    return dict(treatment_vs_control=paired_counts(gold, c, t),
        control_vs_selected95=paired_counts(gold, initial, c),
        treatment_vs_selected95=paired_counts(gold, initial, t),
        control_vs_v9=paired_counts(gold, v9, c), treatment_vs_v9=paired_counts(gold, v9, t))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'private_root', 'audit_plan', 'output'):
        parser.add_argument('--'+name.replace('_', '-'), dest=name, type=Path, required=True)
    parser.add_argument('--audit-plan-sha256', required=True)
    parser.add_argument('--producer-pid', type=int, required=True)
    a = parser.parse_args()
    require(not a.output.exists(), 'Refusing to overwrite paired verification')
    require(file_sha(a.audit_plan) == a.audit_plan_sha256, 'Paired audit plan differs')
    audit = read(a.audit_plan)
    require(audit['format'] == 'roguard-matched-role-attention-audit-plan-1', 'Wrong paired audit format')
    plan_path = a.root/audit['training_plan']; plan = read(plan_path)
    reports = {arm: read(a.root/plan['outputs'][arm]['study']/'report.json') for arm in ('control', 'treatment')}
    for arm, report in reports.items():
        require_complete(report, arm)
    from scripts.wait_completed_sequence_exit_v10 import require_exit
    from training.role_diversity_runtime_v10 import check_bindings
    from training.matched_role_attention_inputs_v10 import completed_inputs, OPTIMIZATION, POPULATIONS
    sequence_path = a.root/audit['sequence']; sequence = read(sequence_path)
    queue_path = a.private_root/audit['queue_plan']; queue = read(queue_path)
    require_exit(sequence, a.producer_pid)
    check_bindings(a.root, a.private_root, audit); check_bindings(a.root, a.private_root, plan)
    require(file_sha(plan_path) == audit['training_plan_sha256']
            and plan['format'] == 'roguard-matched-role-attention-plan-1'
            and plan['optimization'] == OPTIMIZATION and plan['populations'] == POPULATIONS
            and plan['initialization'] == 'selected95' and plan['training_admitted'] is True
            and plan['held_out_test_opened'] is False and plan['qualified'] is False
            and plan['arms'] == ['control', 'treatment'], 'Registered paired training scope differs')
    require(sequence['plan_sha256'] == file_sha(queue_path) and sequence['source_hashes'] == queue['source_hashes']
            and [s['name'] for s in sequence['completed_steps']] == [s['name'] for s in queue['steps']],
            'Paired training queue differs')
    require(audit['public'].get('eval/verify_matched_role_attention_v10.py') == file_sha(__file__)
            and audit['public'].get('eval/audit_attention_development_v10.py') == file_sha(a.root/'eval/audit_attention_development_v10.py')
            and audit['public'].get('eval/verify_runtime_review_attention_continuation_v10.py')
                == file_sha(a.root/'eval/verify_runtime_review_attention_continuation_v10.py'), 'Audit sources not bound')
    fixed = {a.audit_plan: a.audit_plan_sha256, plan_path: file_sha(plan_path),
             sequence_path: file_sha(sequence_path), queue_path: file_sha(queue_path)}
    for name, sha in queue['source_hashes'].items():
        require(file_sha(a.root/name) == sha, 'Queue source changed'); fixed[a.root/name] = sha
    log_path = a.private_root/audit['log']; raw = log_path.read_bytes(); fixed[log_path] = file_sha(log_path)
    inputs, bindings = completed_inputs(a.root, a.private_root)
    require(bindings == plan['input_bindings'], 'Paired input bindings changed')
    import numpy as np
    from safetensors.numpy import load_file
    summaries, selected_predictions = {}, {}
    for arm, report in reports.items():
        study = a.root/plan['outputs'][arm]['study']; weights = a.private_root/plan['outputs'][arm]['private']
        protocol = read(study/'protocol.json')
        expected_protocol = dict(format='roguard-matched-role-attention-fit-1', arm=arm,
            source_sha256=plan['source_sha256'], plan_sha256=file_sha(plan_path), optimization=OPTIMIZATION,
            bindings=bindings, schedules=bindings['schedules'], populations=POPULATIONS,
            native_comparison='cached suffix with a copied fixed387-token prefix',
            initialization='selected95 with training-only head dropout', persistent_body_cache=False,
            held_out_test_opened=False, stable_release_ready=False)
        require(protocol == expected_protocol
                and file_sha(study/'source.py') == plan['source_sha256'] == file_sha(a.root/'training/train_matched_role_attention_v10.py')
                and file_sha(study/'helper.py') == file_sha(a.root/'training/matched_role_attention_inputs_v10.py'),
                'Arm protocol or source snapshot differs')
        require(report['training_rows'] == 32097 and report['development_rows'] == 8837, 'Arm population totals differ')
        for name in ('report.json', 'protocol.json', 'source.py', 'helper.py'):
            fixed[study/name] = file_sha(study/name)
        languages, groups, new_counts = {}, {}, {}
        selected_predictions[arm] = {}
        for lang, current in inputs.items():
            rows = current[arm]; record = report['languages'][lang]
            path = weights/(lang+'-occurrences.jsonl')
            refs = audit_occurrences(rows, current['base'], current['new'], read_jsonl(path), arm, OPTIMIZATION['seed'])
            fixed[path] = file_sha(path)
            teacher = np.stack([(current['base_teacher'] if source == 'base' else current['new_teacher'])[index]
                                for source, index in refs])
            summary = audit_language(study, weights, lang, record, rows,
                load_file(current['weights']['v9']), load_file(current['weights']['selected95']), teacher,
                current['specs']['v9'], current['specs']['selected95'], current['initial_dev_logits'])
            summary['class_weights'] = audit_class_weights(rows, record['field_weights'])
            path = weights/(lang+'-initial-new-logits.npy'); fixed[path] = file_sha(path)
            require(fixed[path] == record['initial_new_logits_sha256'], 'New-input initialization output changed')
            fresh = np.load(path, allow_pickle=False)
            require(parity(fresh, current['new_initial_logits'], current['specs']['selected95']) == record['initial_new_parity'],
                    'New-input initialization parity differs')
            summary['initial_new_rows_compared'] = len(fresh)
            for name, sha in summary['artifacts'].items():
                fixed[weights/name] = sha
            summary['artifacts'][path.name] = fixed[path]
            summary['artifacts'][lang+'-occurrences.jsonl'] = fixed[weights/(lang+'-occurrences.jsonl')]
            for suffix in ('review.json', 'development-predictions.jsonl'):
                path = study/(lang+'-'+suffix); fixed[path] = file_sha(path)
            selected_predictions[arm][lang] = read_jsonl(study/(lang+'-development-predictions.jsonl'))
            languages[lang] = summary
            groups[lang] = len({r['group'] for r in rows if r['partition'] == 'train'})
            new_counts[lang] = len(current['new'])
        inventory = {name for summary in languages.values() for name in summary['artifacts']}
        retained = {str(path.relative_to(weights)) for path in weights.rglob('*')
                    if path.is_file() and path != weights/'writer.lock'}
        require(retained == inventory, 'Unexpected or missing retained arm artifact inventory')
        decision = ('REJECT' if any(s['binary_errors'] or s['joint_argmax_errors'] or s['introduced_v9']
                                   for s in languages.values()) else 'DEVELOPMENT_PASS_REQUIRES_FULL_REGRESSION')
        require(report['decision'] == decision, 'Arm engineering prerequisite decision differs')
        log = audit_training_log(raw, report, bindings['schedules'], groups, new_counts,
            marker=audit['markers'][arm], next_marker=audit['markers']['treatment'] if arm == 'control' else None)
        summaries[arm] = dict(languages=languages, decision=decision, log=log)
    paired = {}
    for lang, current in inputs.items():
        require(summaries['control']['languages'][lang]['class_weights'] == summaries['treatment']['languages'][lang]['class_weights'],
                'Paired class weights differ')
        rows = [r for r in current['base'] if r['partition'] == 'dev']
        paired[lang] = paired_outcomes(rows, selected_predictions['control'][lang], selected_predictions['treatment'][lang],
                                       current['initial_dev_logits'], current['specs']['selected95'])
    totals = {key: sum(s[key] for arm in summaries.values() for s in arm['languages'].values())
              for key in ('checkpoint_count', 'checkpoint_dev_rows_recounted', 'reload_rows_recounted', 'initial_new_rows_compared')}
    require(totals == dict(checkpoint_count=16, checkpoint_dev_rows_recounted=70696,
                          reload_rows_recounted=17674, initial_new_rows_compared=6336), 'Paired audit population incomplete')
    require(completed_inputs(a.root, a.private_root)[1] == bindings, 'Inputs changed during paired audit')
    check_bindings(a.root, a.private_root, audit); check_bindings(a.root, a.private_root, plan)
    require(all(file_sha(path) == sha for path, sha in fixed.items()), 'Evidence changed during paired audit')
    passed = [arm for arm, summary in summaries.items() if summary['decision'] != 'REJECT']
    result = dict(verification='PASS', source_sha256=file_sha(__file__), audit_plan_sha256=a.audit_plan_sha256,
        training_plan_sha256=file_sha(plan_path), sequence_sha256=file_sha(sequence_path), log_sha256=fixed[log_path],
        arms=summaries, paired=paired, **totals, development_pass_arms=passed,
        decision='DEVELOPMENT_PASS_REQUIRES_FULL_REGRESSION' if passed else 'REJECT',
        encoder_loaded=False, model_qualified=False, held_out_test_opened=False,
        scope='Retained tensors, prediction arrays, initial references, occurrence mappings, class weights, source bindings '
              'and ordered training logs audited. Policy probabilities and threshold selection independently recounted. '
              'Native encoder maxima remain producer observations. Paired comparisons concern exposed, correlated DEV '
              'inputs; no new encoder inference, independent gold validation, causal generalization or release claim.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as handle:
        json.dump(result, handle, indent=2); handle.write('\n')
    print(json.dumps({k: result[k] for k in ('verification', 'decision', 'checkpoint_count',
                                          'checkpoint_dev_rows_recounted', 'development_pass_arms', 'model_qualified')}))


if __name__ == '__main__':
    main()


