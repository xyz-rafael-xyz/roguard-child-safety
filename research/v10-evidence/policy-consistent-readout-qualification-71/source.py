"""Verify completed CPU head fitting, retained checkpoints and paired DEV output."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--private', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists(), 'Refuse overwrite'
    import numpy as np
    from safetensors.numpy import load_file
    from training.native_fields_v9 import SIZES
    from roguard.gemma4_review import _score
    evidence = a.root / 'research/v10-evidence'
    study = evidence / 'policy-consistent-readout-71'
    weights_root = a.private / 'policy-consistent-readout-71'
    assessment = evidence / 'policy-consistent-readout-assessment-71'
    report = json.loads((study / 'report.json').read_text())
    protocol = json.loads((study / 'protocol.json').read_text())
    amendment = json.loads((evidence / 'protocol-amendment-65.json').read_text())
    preflight = json.loads((evidence / 'policy-consistent-readout-preflight-71/report.json').read_text())
    sequence = json.loads((evidence / 'policy-consistent-readout-sequence-71/report.json').read_text())
    runner_plan_path = a.private / 'policy-consistent-readout-sequence-71-plan.json'
    runner_plan = json.loads(runner_plan_path.read_text())
    assert sequence['execution_complete'] and sequence['phase'] == 'finished'
    assert len(sequence['completed_steps']) == 1 and sequence['completed_steps'][0]['exit_code'] == 0
    assert sequence['plan_sha256'] == sha(runner_plan_path)
    assert sequence['source_hashes'] == runner_plan['source_hashes']
    for path, value in runner_plan['source_hashes'].items():
        assert sha(a.root / path) == value, path
    pid = int((a.private / 'policy-consistent-readout-sequence-71.pid').read_text())
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError('Completed sequence process must have exited')
    assert report['complete'] and set(report['languages']) == {'ro', 'uk'}
    assert report['device'] == protocol['device'] == amendment['device'] == 'cpu'
    assert report['encoder_loaded'] is protocol['encoder_loaded'] is False
    assert report['calibration_or_test_loaded'] is protocol['calibration_or_test_loaded'] is False
    assert report['stable_release_ready'] is protocol['stable_release_ready'] is False
    assert protocol['source_sha256'] == sha(a.root / 'training/train_policy_consistent_readout_v10.py') == sha(study / 'source.py')
    assert protocol['amendment_sha256'] == sha(evidence / 'protocol-amendment-65.json')
    assert protocol['bindings'] == preflight['bindings'] == amendment['input_bindings']
    for key in ('epochs', 'seed', 'batch_size', 'registered_candidates', 'policy_loss_epsilon'):
        assert protocol[key] == amendment[key]
    feature_study = evidence / 'crossed-review-features-62'
    feature_report = json.loads((feature_study / 'report.json').read_text())
    assert sha(feature_study / 'report.json') == protocol['feature_report_sha256']
    paired_report = json.loads((assessment / 'report.json').read_text())
    pred_path = a.private / 'policy-consistent-readout-assessment-71/predictions.jsonl'
    assert paired_report['complete'] and paired_report['predictions_sha256'] == sha(pred_path)
    assert paired_report['source_sha256'] == sha(a.root / 'eval/assess_crossed_review_v10.py')
    assert paired_report['selection_report_sha256'] == sha(study / 'report.json')
    assert paired_report['feature_report_sha256'] == sha(feature_study / 'report.json')
    assert paired_report['maximum_batch_single_logit_error'] <= 1e-3
    assert paired_report['batch_single_decision_disagreements'] == 0
    paired = read(pred_path)
    assert len(paired) == len({(r['language'], r['case_id']) for r in paired}) == 8837
    by_id = {(r['language'], r['case_id']): r for r in paired}
    mapping = {'first.weight': 'first.weight', 'first.bias': 'first.bias',
               'last.weight': 'last.weight', 'last.bias': 'last.bias'}
    checkpoint_count = updates = presentations = 0
    summaries, recounted = {}, {}
    max_score_delta = 0.
    for lang in ('ro', 'uk'):
        info = feature_report['languages'][lang]
        features = a.private / 'crossed-review-features-62'
        for suffix, value in info['artifacts'].items():
            assert sha(features / (lang + '-' + suffix)) == value
        rows = read(features / (lang + '-rows.jsonl'))
        dev_indices = [i for i, r in enumerate(rows) if r['partition'] == 'dev']
        dev_rows = [rows[i] for i in dev_indices]
        train_count = sum(r['partition'] == 'train' for r in rows)
        original_path = a.root / 'research/v9-stabilization/reported-tail-fit-196/actual' / (lang + '-selected.safetensors')
        original = load_file(original_path)
        original_spec = json.loads((feature_study / 'v9-binding' / (lang + '-review.json')).read_text())
        assert sha(original_path) == original_spec['adapter_sha256'] == protocol['bindings'][lang]['v9_weights_sha256']
        history = report['languages'][lang]['epochs']
        names = [c['name'] for c in amendment['registered_candidates']]
        assert [(r['candidate'], r['epoch']) for r in history] == [(name, epoch) for name in names for epoch in range(13)]
        assert report['languages'][lang]['complete']
        assert report['languages'][lang]['training_rows'] == train_count
        assert report['languages'][lang]['development_rows'] == len(dev_rows)
        assert report['languages'][lang]['maximum_initial_logit_error'] <= 1e-3
        assert report['languages'][lang]['selected_reload_maximum_logit_error'] <= 1e-5
        for entry in history:
            path = weights_root / lang / entry['candidate'] / ('epoch-' + str(entry['epoch']) + '.safetensors')
            assert sha(path) == entry['checkpoint_sha256']
            tensors = load_file(path)
            assert set(tensors) == set(mapping)
            for key, value in tensors.items():
                old = original['heads.' + lang + '.' + key]
                assert value.shape == old.shape and value.dtype == old.dtype == np.float32
                assert np.isfinite(value).all()
                if entry['epoch'] == 0:
                    assert np.array_equal(value, old)
            assert entry['completed_training_updates'] == entry['epoch'] * math.ceil(train_count / 64)
            assert entry['completed_training_presentations'] == entry['epoch'] * train_count
            assert entry['selection_rank'] == [entry['binary_errors'], entry['joint_argmax_errors'],
                                                entry['mean_field_cross_entropy'], -entry['score_margin']]
            assert all(math.isfinite(v) for v in entry['selection_rank'])
            assert 0 <= entry['binary_errors'] <= len(dev_rows)
            assert 0 <= entry['joint_argmax_errors'] <= len(dev_rows)
            assert max(entry['field_errors']) <= entry['joint_argmax_errors'] <= sum(entry['field_errors'])
            checkpoint_count += 1
        selected = report['languages'][lang]['selected']
        minimum = min(history, key=lambda r: r['selection_rank'])
        assert (selected['candidate'], selected['epoch'], selected['selection_rank']) == (minimum['candidate'], minimum['epoch'], minimum['selection_rank'])
        actual_path = weights_root / (lang + '-selected.safetensors')
        spec = json.loads((study / (lang + '-review.json')).read_text())
        assert sha(actual_path) == spec['adapter_sha256'] == selected['adapter_sha256']
        assert spec['mode'] == selected['mode'] and spec['threshold'] == selected['threshold']
        assert spec['couplings'] == original_spec['couplings']
        combined = load_file(actual_path)
        selected_head = load_file(weights_root / lang / selected['candidate'] / ('epoch-' + str(selected['epoch']) + '.safetensors'))
        assert set(combined) == set(original)
        for key, value in combined.items():
            if key.startswith('heads.' + lang + '.'):
                assert np.array_equal(value, selected_head[key.removeprefix('heads.' + lang + '.')])
            else:
                assert np.array_equal(value, original[key])
        logits = np.load(weights_root / (lang + '-selected-dev-logits.npy'), allow_pickle=False)
        teacher = np.load(features / (lang + '-logits.npy'), mmap_mode='r', allow_pickle=False)
        assert logits.shape == (len(dev_rows), sum(SIZES) + 2) and np.isfinite(logits).all()
        parts = np.split(logits[:, :sum(SIZES)], np.cumsum(SIZES)[:-1], axis=-1)
        field_pred = np.column_stack([part.argmax(axis=1) for part in parts])
        gold_fields = np.array([row['fields'] for row in dev_rows])
        assert int(np.any(field_pred != gold_fields, axis=1).sum()) == selected['joint_argmax_errors']
        assert (field_pred != gold_fields).sum(axis=0).tolist() == selected['field_errors']
        losses = []
        for j, part in enumerate(parts):
            probability = np.exp(part - part.max(axis=1, keepdims=True))
            probability /= probability.sum(axis=1, keepdims=True)
            losses.append(-np.log(np.maximum(probability[np.arange(len(dev_rows)), gold_fields[:, j]], 1e-30)).mean())
        assert abs(float(np.mean(losses)) - selected['mean_field_cross_entropy']) < 1e-12
        binary_errors = 0
        for j, (i, row) in enumerate(zip(dev_indices, dev_rows)):
            pred = by_id[(lang, row['case_id'])]
            assert pred['gold'] == row['gold'] and pred['token_sha256'] == row['token_sha256']
            score = _score(logits[j], spec)
            baseline = _score(teacher[i], original_spec)
            assert int(score >= spec['threshold']) == pred['candidate']
            assert int(baseline >= original_spec['threshold']) == pred['v9']
            assert abs(baseline - pred['v9_score']) < 1e-12
            max_score_delta = max(max_score_delta, abs(score - pred['candidate_score']))
            assert abs(score - pred['candidate_score']) <= 1e-5
            origin = 'crossed' if row['case_id'].startswith('composition61:') else 'generated' if row.get('family', '').startswith('composition56:') else 'historical'
            assert pred['origin'] == origin
            group = recounted.setdefault(origin + ':' + lang, {'rows': 0, 'candidate_errors': 0, 'v9_errors': 0, 'fixed': 0, 'introduced': 0})
            bad, old_bad = pred['candidate'] != row['gold'], pred['v9'] != row['gold']
            group['rows'] += 1
            group['candidate_errors'] += bad
            group['v9_errors'] += old_bad
            group['fixed'] += old_bad and not bad
            group['introduced'] += bad and not old_bad
            binary_errors += bad
        assert binary_errors == selected['binary_errors']
        best_by_config = {name: min((v for v in history if v['candidate'] == name), key=lambda v: v['selection_rank']) for name in names}
        summaries[lang] = {'selected': selected, 'best_by_configuration': best_by_config,
                           'all_epoch0_tensors_equal_original_v9': True, 'frozen_other_parameters_exact': True}
        updates += len(names) * 12 * math.ceil(train_count / 64)
        presentations += len(names) * 12 * train_count
    assert recounted == paired_report['groups']
    assert checkpoint_count == amendment['schedule']['saved_checkpoints'] == 78
    assert updates == amendment['schedule']['total_optimizer_updates'] == 16308
    assert presentations == amendment['schedule']['total_training_presentations'] == 1041444
    assert report['decision'] == 'REJECT' and paired_report['decision'] == 'REJECT'
    proof = {'complete': True, 'verified': True, 'source_sha256': sha(Path(__file__)),
             'protocol_sha256': sha(study / 'protocol.json'), 'report_sha256': sha(study / 'report.json'),
             'assessment_sha256': sha(assessment / 'report.json'), 'predictions_sha256': sha(pred_path),
             'checkpoints_verified': checkpoint_count, 'schedule_updates': updates,
             'schedule_presentations': presentations, 'languages': summaries, 'paired_groups': recounted,
             'maximum_batch_single_score_difference': max_score_delta,
             'selection_rule': 'Earliest lexicographic minimum of all39 retained epoch summaries per language.',
             'scope': 'All checkpoint identities/finiteness/initialization and reported schedule counters; frozen non-head tensors; selected saved logits, source gold, field errors/CE, paired scores/decisions and aggregate counts. Nonselected DEV logits are not replayed by this verifier.',
             'model_inference_performed': False, 'encoder_loaded': False,
             'held_out_test_opened': False, 'qualification': 'REJECT', 'stable_release_ready': False}
    with a.output.open('x') as stream:
        stream.write(json.dumps(proof, indent=2) + '\n')
    print(json.dumps({'verified': True, 'checkpoints': checkpoint_count, 'groups': recounted,
                      'maximum_batch_single_score_difference': max_score_delta, 'qualification': 'REJECT'}, indent=2))


if __name__ == '__main__':
    main()
