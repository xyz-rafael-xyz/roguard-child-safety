"""Replay every CPU dropout checkpoint and independently audit its exposed DEV results."""
import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

from eval.audit_attention_development_v10 import recount
from roguard.trajectory_data import file_sha, read_jsonl


def check(value, message):
    if not value: raise ValueError(message)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--private', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    check(not a.output.exists(), 'Refusing to overwrite verification')
    import numpy as np
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    from safetensors.numpy import load_file
    from roguard.gemma4_review import _score
    from training.shared_tail_training_v9 import make_head
    from training.train_dropout_readout_v10 import validate_inputs
    e = a.root/'research/v10-evidence'; study = e/'dropout-readout-75'; weights = a.private/'dropout-readout-75'
    def read(path): return json.loads(path.read_text())
    amendment_path = e/'protocol-amendment-67.json'; plan = read(amendment_path)
    check(file_sha(amendment_path) == '91652e0b1a6b1f89430c00925113ee3b790a058500e3133e2752d6e7e8e2906e', 'Registration changed')
    report = read(study/'report.json'); protocol = read(study/'protocol.json')
    queue_path = a.private/'dropout-readout-sequence-75-plan.json'; queue = read(queue_path)
    sequence = read(e/'dropout-readout-sequence-75/report.json')
    check(sequence['execution_complete'] is True and sequence['phase'] == 'finished'
          and len(sequence['completed_steps']) == 1 and sequence['completed_steps'][0]['exit_code'] == 0,
          'Training sequence is incomplete')
    check(sequence['plan_sha256'] == file_sha(queue_path) and sequence['source_hashes'] == queue['source_hashes'], 'Queue changed')
    for path, value in queue['source_hashes'].items(): check(file_sha(a.root/path) == value, 'Queued source changed')
    pid = int((a.private/'dropout-readout-sequence-75.pid').read_text())
    try: os.kill(pid, 0)
    except ProcessLookupError: pass
    else: raise ValueError('Training sequence process still exists')
    check(report['complete'] is True and set(report['languages']) == {'ro', 'uk'}, 'Incomplete training report')
    check(protocol['source_sha256'] == plan['source_sha256'] == file_sha(study/'source.py')
          == file_sha(a.root/'training/train_dropout_readout_v10.py'), 'Trainer source changed')
    check(protocol['amendment_sha256'] == file_sha(amendment_path), 'Protocol registration changed')
    for key in ('device', 'epochs', 'seed', 'batch_size', 'registered_candidates', 'policy_loss_epsilon'):
        check(protocol[key] == plan[key], 'Protocol setting changed: '+key)
    feature_study = e/'crossed-review-features-62'; features = a.private/'crossed-review-features-62'
    candidate = a.root/'research/v9-stabilization/reported-tail-fit-196'
    bindings = validate_inputs(SimpleNamespace(feature_study=feature_study, features=features, candidate=candidate),
                               read(feature_study/'report.json'), plan)
    check(bindings == protocol['bindings'], 'Data bindings changed')
    check(report['device'] == 'cpu' and report['encoder_loaded'] is False
          and report['calibration_or_test_loaded'] is False and report['stable_release_ready'] is False,
          'Unsupported qualification claim')
    checkpoint_count = control_count = updates = presentations = 0
    maximum_policy_error = maximum_reload_error = 0.; summaries = {}
    mapping = {'0.weight': 'first.weight', '0.bias': 'first.bias', '3.weight': 'last.weight', '3.bias': 'last.bias'}
    for lang in ('ro', 'uk'):
        metadata = read_jsonl(features/(lang+'-rows.jsonl')); dev = [i for i, row in enumerate(metadata) if row['partition'] == 'dev']
        rows = [metadata[i] for i in dev]; train_count = sum(row['partition'] == 'train' for row in metadata)
        x = np.load(features/(lang+'-features.npy'), mmap_mode='r', allow_pickle=False)
        teacher = np.load(features/(lang+'-logits.npy'), mmap_mode='r', allow_pickle=False)
        original = load_file(candidate/'actual'/(lang+'-selected.safetensors'))
        original_spec = read(feature_study/'v9-binding'/(lang+'-review.json'))
        baseline = [int(_score(teacher[i], original_spec) >= original_spec['threshold']) for i in dev]
        record = report['languages'][lang]; history = record['epochs']; names = [c['name'] for c in plan['registered_candidates']]
        check(record['complete'] is True and record['training_rows'] == train_count and record['development_rows'] == len(dev), 'Population changed')
        check([(c['candidate'], c['epoch']) for c in history] == [(name, ep) for name in names for ep in range(13)], 'Incomplete checkpoint history')
        for key, limit in (('maximum_initial_logit_error', 1e-3), ('selected_reload_maximum_logit_error', 1e-5)):
            check(math.isfinite(record[key]) and 0 <= record[key] <= limit, 'Reported parity check failed')
        selected = min(history, key=lambda c: c['selection_rank'])
        check(record['selected'] == {k: v for k, v in selected.items() if k != 'training_loss'},
              'Selection violates the earliest-tie rule')
        selected_summary = None
        for entry in history:
            epoch = entry['epoch']; path = weights/lang/entry['candidate']/f'epoch-{epoch}.safetensors'
            check(file_sha(path) == entry['checkpoint_sha256'], 'Checkpoint hash changed')
            tensors = load_file(path)
            check(set(tensors) == set(mapping.values()), 'Head tensor inventory changed')
            for name, value in tensors.items():
                check(value.dtype == np.float32 and value.shape == original['heads.'+lang+'.'+name].shape
                      and np.isfinite(value).all(), 'Malformed head tensor')
                if epoch == 0: check(np.array_equal(value, original['heads.'+lang+'.'+name]), 'Initialization changed')
            check(entry['completed_training_updates'] == epoch*math.ceil(train_count/64)
                  and entry['completed_training_presentations'] == epoch*train_count, 'Training counters changed')
            if entry['candidate'] == 'dropout-control':
                prior_name = 'policy-quarter' if lang == 'ro' else 'policy-one'
                prior_path = a.private/'policy-consistent-readout-71'/lang/prior_name/f'epoch-{epoch}.safetensors'
                old = load_file(prior_path)
                check(all(np.array_equal(value, old[name]) for name, value in tensors.items()), 'Matched control differs from study71')
                control_count += 1
            head = make_head({key: tensors[value] for key, value in mapping.items()}); head.eval(); parts = []
            for start in range(0, len(dev), 256):
                z = head(mx.array(x[dev[start:start+256]])); mx.eval(z); parts.append(np.asarray(z).copy())
            z = np.concatenate(parts).astype(np.float64)
            summary, predicted, scores, error = recount(rows, z, original_spec['couplings'], baseline)
            maximum_policy_error = max(maximum_policy_error, error)
            for key in ('binary_errors', 'joint_argmax_errors', 'field_errors', 'mean_field_cross_entropy',
                        'mode', 'threshold', 'score_margin', 'selection_rank'):
                check(summary[key] == entry[key], 'Independent epoch recount differs: '+key)
            if entry == selected:
                selected_summary = summary
                saved = np.load(weights/(lang+'-selected-dev-logits.npy'), allow_pickle=False)
                delta = float(np.max(np.abs(z-saved))); maximum_reload_error = max(maximum_reload_error, delta)
                check(delta <= 1e-5 and np.array_equal(z.argmax(axis=1), saved.argmax(axis=1)), 'Selected reload differs')
                expected = [{'case_id': row['case_id'], 'language': lang, 'group': row['group'], 'gold': row['gold'],
                             'score': float(score), 'predicted': int(pred)} for row, score, pred in zip(rows, scores, predicted)]
                check(read_jsonl(study/(lang+'-development-predictions.jsonl')) == expected, 'Selected prediction records differ')
                selected_path = weights/(lang+'-selected.safetensors'); full = load_file(selected_path)
                check(set(full) == set(original) and file_sha(selected_path) == selected['adapter_sha256'], 'Selected full checkpoint changed')
                for name, value in full.items():
                    expected_tensor = tensors[name[len('heads.'+lang+'.'):]] if name.startswith('heads.'+lang+'.') else original[name]
                    check(np.array_equal(value, expected_tensor), 'Selected tensor or frozen weight differs')
                check(read(study/(lang+'-review.json')) == dict(original_spec, adapter_sha256=file_sha(selected_path),
                      mode=entry['mode'], threshold=entry['threshold']), 'Export specification changed')
            checkpoint_count += 1
        check(selected_summary is not None, 'Selected checkpoint was not replayed')
        summaries[lang] = {'selected': selected, 'recount': selected_summary,
                           'best_by_configuration': {name: min((c for c in history if c['candidate'] == name), key=lambda c: c['selection_rank']) for name in names}}
        updates += 3*12*math.ceil(train_count/64); presentations += 3*12*train_count
    check((checkpoint_count, control_count, updates, presentations) == (78, 26, 16308, 1041444), 'Total training scope differs')
    decision = ('REJECT' if any(r['selected']['binary_errors'] or r['selected']['joint_argmax_errors'] for r in summaries.values())
                else 'DEVELOPMENT_PASS_REQUIRES_FULL_REGRESSION')
    check(report['decision'] == decision, 'Decision differs from the registered gate')
    proof = {'complete': True, 'verified': True, 'source_sha256': file_sha(__file__),
             'audit_source_sha256': file_sha(a.root/'eval/audit_attention_development_v10.py'),
             'report_sha256': file_sha(study/'report.json'), 'protocol_sha256': file_sha(study/'protocol.json'),
             'checkpoints_replayed': checkpoint_count, 'development_predictions_replayed': 39*8837,
             'matched_control_checkpoints_bit_identical_to71': control_count, 'optimizer_updates': updates,
             'training_presentations': presentations, 'languages': summaries,
             'maximum_policy_enumeration_error': maximum_policy_error, 'maximum_selected_reload_error': maximum_reload_error,
             'decision': decision, 'device': 'cpu', 'encoder_loaded': False, 'head_inference_performed': True,
             'held_out_test_opened': False, 'qualified': False,
             'scope': 'Every retained head checkpoint replayed on exposedDEV and independently recounted; exact matched-control tensors, '
                      'source bindings, counters, selected predictions/specification and frozen full-checkpoint weights. No independent gold validation.'}
    a.output.write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps({'verified': True, 'checkpoints_replayed': checkpoint_count, 'control_exact': control_count,
                      'selected': {lang: r['selected'] for lang, r in summaries.items()}, 'decision': decision}, indent=2))


if __name__ == '__main__': main()
