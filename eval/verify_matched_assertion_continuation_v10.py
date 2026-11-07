"""Replay completed86 heads, independently recount selection and paired outcomes."""
import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

from eval.audit_attention_development_v10 import recount
from roguard.assurance import digest
from roguard.trajectory_data import file_sha, read_jsonl, write_jsonl

ARMS = ('matched-control', 'wording-treatment')


def check(value, message):
    if not value: raise ValueError(message)


def verify_selections(history, selected, arm_selected):
    check([(r['candidate'], r['epoch']) for r in history] == [(arm, ep) for arm in ARMS for ep in range(13)],
          'Incomplete or reordered checkpoint history')
    check(set(arm_selected) == set(ARMS), 'Missing or extra arm selection')
    best = min(history, key=lambda r: r['selection_rank'])
    check(selected == {k: v for k, v in best.items() if k != 'training_loss'}, 'Wrong global earliest-tie selection')
    by_arm = {}
    for arm in ARMS:
        entry = min((r for r in history if r['candidate'] == arm), key=lambda r: r['selection_rank'])
        check(arm_selected[arm] == {k: v for k, v in entry.items() if k not in ('training_loss', 'adapter_sha256')},
              'Wrong per-arm earliest-tie selection')
        by_arm[arm] = entry
    return best, by_arm


def paired_counts(gold, baseline, candidate):
    check(len(gold) == len(baseline) == len(candidate) and bool(len(gold)), 'Incomplete paired population')
    check(all(v in (0, 1) for values in (gold, baseline, candidate) for v in values), 'Nonbinary paired decisions')
    fixes = introductions = both_wrong = both_correct = 0
    for y, before, after in zip(gold, baseline, candidate):
        fixes += before != y and after == y
        introductions += before == y and after != y
        both_wrong += before != y and after != y
        both_correct += before == y and after == y
    return {'rows': len(gold), 'baseline_errors': fixes+both_wrong, 'candidate_errors': introductions+both_wrong,
            'fixes': fixes, 'introductions': introductions, 'both_wrong': both_wrong, 'both_correct': both_correct}


def audit_log(raw, report):
    marker = b'{"starting": "train-matched-assertion-readout86"}'
    lines = raw.splitlines(); check(lines.count(marker) == 1, 'Missing or repeated training start')
    body = b'\n'.join(lines[lines.index(marker)+1:])
    prefix, separator, terminal = body.partition(b'\n{\n')
    check(bool(separator), 'Missing terminal selection object')
    events = [json.loads(line) for line in prefix.splitlines() if line.strip()]
    expected = [{'language': lang, **{k: v for k, v in r.items() if k != 'training_loss'}}
                for lang in ('ro', 'uk') for r in report['languages'][lang]['epochs']]
    check(events == expected and len(events) == 52, 'Checkpoint log differs from retained history')
    check(json.loads(b'{\n'+terminal) == {lang: report['languages'][lang]['selected'] for lang in ('ro', 'uk')},
          'Terminal selection log differs')
    return {'checkpoint_events': len(events), 'per_minibatch_log_available': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'private', 'output'): parser.add_argument('--'+name, type=Path, required=True)
    a = parser.parse_args(); paired_path = a.output.with_name(a.output.stem+'-paired-dev.jsonl')
    check(not a.output.exists() and not paired_path.exists(), 'Refusing to overwrite verification')
    import numpy as np
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    from safetensors.numpy import load_file
    from roguard.gemma4_review import _score
    from roguard.evidence import ONTOLOGY
    from training.shared_tail_training_v9 import make_head
    from training.matched_assertion_continuation_v10 import validate_bindings
    read = lambda path: json.loads(path.read_text())
    e = a.root/'research/v10-evidence'; study = e/'matched-assertion-readout-86'; weights = a.private/'matched-assertion-readout-86'
    amendment = e/'protocol-amendment-72.json'; plan = read(amendment)
    check(file_sha(amendment) == '215e3a455c1c9f9fb14b98d4c490c7a5d17d8646d59c57ed9670c89e49d8d7d3', 'Registered86 plan changed')
    queue_path = a.private/'matched-assertion-readout-sequence-86-plan.json'; queue = read(queue_path)
    sequence_path = e/'matched-assertion-readout-sequence-86/report.json'; sequence = read(sequence_path)
    check(sequence.get('execution_complete') is True and sequence.get('phase') == 'finished'
          and [s['name'] for s in sequence['completed_steps']] == [s['name'] for s in queue['steps']]
          and all(s['exit_code'] == 0 for s in sequence['completed_steps']), 'Training sequence incomplete or failed')
    check(sequence['plan_sha256'] == file_sha(queue_path) and sequence['source_hashes'] == queue['source_hashes'], 'Queue changed')
    for name, expected in queue['source_hashes'].items(): check(file_sha(a.root/name) == expected, 'Queued source changed: '+name)
    pid = int((a.private/'matched-assertion-readout-sequence-86.pid').read_text())
    try: os.kill(pid, 0)
    except ProcessLookupError: pass
    else: raise ValueError('Training queue has not exited')
    protocol = read(study/'protocol.json'); report = read(study/'report.json')
    source = a.root/'training/train_matched_assertion_continuation_v10.py'
    check(protocol['source_sha256'] == file_sha(source) == file_sha(study/'source.py')
          and protocol['amendment_sha256'] == file_sha(amendment), 'Trainer source or registration changed')
    for key in ('device', 'epochs', 'seed', 'batch_size', 'registered_candidates', 'policy_loss_epsilon'):
        check(protocol[key] == plan[key], 'Registered setting changed: '+key)
    check(report.get('complete') is True and set(report['languages']) == {'ro', 'uk'}
          and report['training_rows'] == 29793 and report['development_rows'] == 8837,
          'Incomplete or different training population')
    for record in (protocol, report):
        check(record['device'] == 'cpu' and record['encoder_loaded'] is False
              and record['calibration_or_test_loaded'] is False and record['stable_release_ready'] is False,
              'Unsupported scope or qualification claim')
    args = SimpleNamespace(root=a.root, private_root=a.private, features=a.private/'crossed-review-features-62',
                           feature_study=e/'crossed-review-features-62', candidate=a.root/'research/v9-stabilization/reported-tail-fit-196')
    extraction = read(args.feature_study/'report.json')
    bindings = validate_bindings(args, extraction, plan, ready=True)
    check(bindings == protocol['bindings'] and bindings['future_features_verified'] is True, 'Actual input bindings changed')
    runtime = read_jsonl(a.private/'assertion-variation-77/curated/runtime-inputs.jsonl')
    mapping = {'0.weight': 'first.weight', '0.bias': 'first.bias', '3.weight': 'last.weight', '3.bias': 'last.bias'}
    sizes = tuple(map(len, ONTOLOGY.values()))
    summaries = {}; pairs = []; checkpoints = predictions_replayed = updates = presentations = 0
    maximum_reload = maximum_policy_error = 0.
    for lang in ('ro', 'uk'):
        metadata = read_jsonl(args.features/(lang+'-rows.jsonl'))
        dev = [i for i, r in enumerate(metadata) if r['partition'] == 'dev']; rows = [metadata[i] for i in dev]
        # Reconstruct the common labels/order independently of paired_rows().
        combined = list(metadata)
        for r in [row for row in runtime if row['language'] == lang]:
            p = r['parent']; original_row = metadata[p['feature_index']]
            check(original_row['partition'] == 'train' and all(original_row[k] == p[k] for k in ('case_id', 'gold', 'fields', 'token_sha256'))
                  and r['gold'] == p['gold'] and r['fields'] == p['fields'], 'Reconstructed parent labels differ')
            combined.append({'case_id': 'matched86:'+r['pair_id'], 'language': lang, 'partition': 'train',
                'group': 'assertion_variation_77', 'gold': r['gold'], 'fields': r['fields'], 'pair_id': r['pair_id'],
                'parent_feature_index': p['feature_index'], 'treatment_token_sha256': r['token_sha256'],
                'control_token_sha256': p['token_sha256']})
        check(digest(combined) == bindings['schedules'][lang]['paired_rows_sha256'], 'Common input order/labels differ')
        train_count = sum(r['partition'] == 'train' for r in combined)
        check(train_count == plan['languages'][lang]['training_rows'] and len(rows) == plan['languages'][lang]['development_rows'],
              'Registered language population differs')
        x = np.load(args.features/(lang+'-features.npy'), mmap_mode='r', allow_pickle=False)
        teacher = np.load(args.features/(lang+'-logits.npy'), mmap_mode='r', allow_pickle=False)
        original = load_file(args.candidate/'actual'/(lang+'-selected.safetensors'))
        spec = read(args.feature_study/'v9-binding'/(lang+'-review.json'))
        baseline = [int(_score(teacher[i], spec) >= spec['threshold']) for i in dev]
        record = report['languages'][lang]; history = record['epochs']
        check(record['complete'] is True and record['training_rows'] == train_count and record['development_rows'] == len(dev), 'Language report differs')
        selected, by_arm = verify_selections(history, record['selected'], record['arm_selected'])
        arm_results = {}; selected_result = None; initial_logits = None; initial_error = 0.
        for entry in history:
            arm, epoch = entry['candidate'], entry['epoch']; folder = weights/lang/arm
            checkpoint = folder/f'epoch-{epoch}.safetensors'; logits_path = folder/f'epoch-{epoch}-dev-logits.npy'
            check(file_sha(checkpoint) == entry['checkpoint_sha256'] and file_sha(logits_path) == entry['development_logits_sha256'],
                  'Checkpoint or saved logit identity differs')
            tensors = load_file(checkpoint); check(set(tensors) == set(mapping.values()), 'Head tensor inventory differs')
            for name, value in tensors.items():
                old = original['heads.'+lang+'.'+name]
                check(value.shape == old.shape and value.dtype == np.float32 and np.isfinite(value).all(), 'Invalid head tensor')
                if epoch == 0: check(np.array_equal(value, old), 'Original-v9 initialization differs')
            check(entry['completed_training_updates'] == epoch*math.ceil(train_count/64)
                  and entry['completed_training_presentations'] == epoch*train_count, 'Checkpoint counters differ')
            if epoch: check(math.isfinite(entry['training_loss']), 'Nonfinite recorded training loss')
            saved = np.load(logits_path, allow_pickle=False)
            check(saved.shape == (len(dev), 19) and saved.dtype == np.float64 and np.isfinite(saved).all(), 'Invalid saved DEV logits')
            head = make_head({key: tensors[value] for key, value in mapping.items()}); head.eval(); chunks = []
            for start in range(0, len(dev), 256):
                values = head(mx.array(x[dev[start:start+256]])); mx.eval(values); chunks.append(np.asarray(values).copy())
            replayed = np.concatenate(chunks).astype(np.float64)
            error = float(np.max(np.abs(replayed-saved))); maximum_reload = max(maximum_reload, error)
            check(error <= 1e-5 and np.isfinite(replayed).all(), 'Head replay differs from saved DEV logits')
            if epoch == 0:
                initial_error = max(initial_error, float(np.max(np.abs(saved-teacher[dev]))))
                check(initial_error <= 1e-3, 'Initial outputs differ from cached original-v9')
                if initial_logits is None: initial_logits = saved.copy()
                else: check(np.array_equal(initial_logits, saved), 'Initial arms differ')
            summary, predicted, scores, policy_error = recount(rows, saved, spec['couplings'], baseline)
            maximum_policy_error = max(maximum_policy_error, policy_error)
            for key in ('binary_errors', 'joint_argmax_errors', 'field_errors', 'mean_field_cross_entropy', 'mode',
                        'threshold', 'score_margin', 'selection_rank'):
                check(summary[key] == entry[key], 'Independent checkpoint recount differs: '+key)
            replay_predictions = [int(_score(z, dict(spec, mode=entry['mode'])) >= entry['threshold']) for z in replayed]
            check(np.array_equal(replay_predictions, predicted), 'Replayed served decisions differ')
            for left, right in zip(np.split(replayed[:, :17], np.cumsum(sizes)[:-1], axis=1),
                                   np.split(saved[:, :17], np.cumsum(sizes)[:-1], axis=1)):
                check(np.array_equal(left.argmax(axis=1), right.argmax(axis=1)), 'Replayed field decisions differ')
            if entry == by_arm[arm]: arm_results[arm] = {'summary': summary, 'predicted': predicted.tolist(), 'scores': scores.tolist()}
            if entry == selected:
                selected_result = {'summary': summary, 'predicted': predicted.tolist(), 'scores': scores.tolist()}
                global_logits = np.load(weights/(lang+'-selected-dev-logits.npy'), allow_pickle=False)
                check(np.array_equal(global_logits, saved), 'Selected DEV array differs from selected checkpoint')
                check(record['selected_reload_maximum_logit_error'] == error, 'Reported selected reload error differs')
                full_path = weights/(lang+'-selected.safetensors'); full = load_file(full_path)
                check(set(full) == set(original) and file_sha(full_path) == selected['adapter_sha256'], 'Selected full checkpoint differs')
                for name, value in full.items():
                    expected = tensors[name[len('heads.'+lang+'.'):]] if name.startswith('heads.'+lang+'.') else original[name]
                    check(value.dtype == expected.dtype and np.array_equal(value, expected), 'Selected or frozen tensor differs')
                check(read(study/(lang+'-review.json')) == dict(spec, adapter_sha256=file_sha(full_path),
                      mode=entry['mode'], threshold=entry['threshold']), 'Exported specification differs')
                expected_rows = [{'case_id': row['case_id'], 'language': lang, 'group': row['group'], 'gold': row['gold'],
                                  'score': float(score), 'predicted': int(pred)} for row, score, pred in zip(rows, scores, predicted)]
                check(read_jsonl(study/(lang+'-development-predictions.jsonl')) == expected_rows, 'Selected prediction records differ')
            checkpoints += 1; predictions_replayed += len(rows)
            del head; mx.clear_cache()
        check(record['maximum_initial_logit_error'] == initial_error, 'Reported initial parity error differs')
        check(selected_result is not None and set(arm_results) == set(ARMS), 'Missing selected checkpoint replay')
        gold = [row['gold'] for row in rows]; control = arm_results[ARMS[0]]; treatment = arm_results[ARMS[1]]
        summaries[lang] = {'selected': record['selected'], 'selected_recount': selected_result['summary'],
            'arm_selected': record['arm_selected'], 'arm_recounts': {arm: r['summary'] for arm, r in arm_results.items()},
            'selected_vs_v9': paired_counts(gold, baseline, selected_result['predicted']),
            'control_vs_v9': paired_counts(gold, baseline, control['predicted']),
            'treatment_vs_v9': paired_counts(gold, baseline, treatment['predicted']),
            'treatment_vs_control': paired_counts(gold, control['predicted'], treatment['predicted'])}
        for i, row in enumerate(rows):
            pairs.append({'case_id': row['case_id'], 'language': lang, 'group': row['group'], 'gold': row['gold'],
                          'v9': baseline[i], 'control': control['predicted'][i], 'treatment': treatment['predicted'][i],
                          'selected': selected_result['predicted'][i], 'selected_arm': selected['candidate'],
                          'control_score': control['scores'][i], 'treatment_score': treatment['scores'][i],
                          'selected_score': selected_result['scores'][i]})
        updates += 2*12*math.ceil(train_count/64); presentations += 2*12*train_count
    check((checkpoints, predictions_replayed, updates, presentations) == (52, 229762, 11184, 715032), 'Total training scope differs')
    decision = ('REJECT' if any(r['selected']['binary_errors'] or r['selected']['joint_argmax_errors'] for r in summaries.values())
                else 'DEVELOPMENT_PASS_REQUIRES_FULL_REGRESSION')
    check(report['decision'] == decision, 'Decision differs from registered DEV prerequisites')
    log_path = a.private/'matched-assertion-readout-sequence-86.log'; log = audit_log(log_path.read_bytes(), report)
    check(validate_bindings(args, extraction, plan, ready=True) == bindings, 'Bound inputs changed during verification')
    a.output.parent.mkdir(parents=True, exist_ok=True); write_jsonl(paired_path, pairs)
    proof = {'verification': 'PASS', 'source_sha256': file_sha(__file__),
             'audit_source_sha256': file_sha(a.root/'eval/audit_attention_development_v10.py'),
             'report_sha256': file_sha(study/'report.json'), 'protocol_sha256': file_sha(study/'protocol.json'),
             'sequence_sha256': file_sha(sequence_path), 'training_log_sha256': file_sha(log_path),
             'checkpoints_replayed': checkpoints, 'development_predictions_replayed': predictions_replayed,
             'optimizer_updates_from_registered_checkpoint_counters': updates, 'training_presentations_from_registered_counters': presentations,
             'maximum_checkpoint_replay_error': maximum_reload, 'maximum_policy_enumeration_error': maximum_policy_error,
             'languages': summaries, 'paired_predictions_sha256': file_sha(paired_path), 'paired_rows': len(pairs),
             'decision': decision, **log, 'device': 'cpu', 'encoder_loaded': False, 'head_inference_performed': True,
             'held_out_test_opened': False, 'qualified': False,
             'scope': 'All52 retained heads and DEV arrays, independent threshold/world recount, earliest selections, per-arm/v9 comparisons, '
                      'full exports and frozen tensors, registered counters and52 checkpoint log events. No per-minibatch trace, new encoder inference or independent gold validation.'}
    a.output.write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps({'verification': 'PASS', 'checkpoints': checkpoints, 'decision': decision,
                      'selected_vs_v9': {lang: r['selected_vs_v9'] for lang, r in summaries.items()}}, indent=2))


if __name__ == '__main__': main()
