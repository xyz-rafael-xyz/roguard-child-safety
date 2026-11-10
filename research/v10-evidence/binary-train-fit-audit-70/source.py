"""Evaluate retained TRAIN features with two frozen small heads on the CPU."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--private', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Refuse overwrite')
    if shutil.disk_usage(a.private).free < 8 * 1024**3:
        raise ValueError('Disk reserve reached')
    plan_path = a.private / 'binary-train-fit-audit-70/plan.json'
    plan = json.loads(plan_path.read_text())
    evidence = a.root / 'research/v10-evidence'
    feature_study = evidence / 'crossed-review-features-62'
    candidate_study = evidence / 'crossed-review-readout-62'
    feature_private = a.private / 'crossed-review-features-62'
    candidate_private = a.private / 'crossed-review-readout-62'
    original_private = a.root / 'research/v9-stabilization/reported-tail-fit-196/actual'
    paths = {
        'feature_report': feature_study / 'report.json',
        'selected_report': candidate_study / 'report.json',
        'fit_protocol': candidate_study / 'protocol.json',
        'head_source': a.root / 'training/shared_tail_training_v9.py',
        'field_source': a.root / 'training/native_fields_v9.py',
        'runtime_score_source': a.root / 'src/roguard/gemma4_review.py',
    }
    for lang in ('ro', 'uk'):
        paths[lang + '_candidate_spec'] = candidate_study / (lang + '-review.json')
        paths[lang + '_v9_spec'] = feature_study / 'v9-binding' / (lang + '-review.json')
        paths[lang + '_candidate_weights'] = candidate_private / (lang + '-selected.safetensors')
        paths[lang + '_v9_weights'] = original_private / (lang + '-selected.safetensors')
    assert {k: sha(v) for k, v in paths.items()} == plan['bindings']
    extraction = json.loads(paths['feature_report'].read_text())
    selection = json.loads(paths['selected_report'].read_text())
    assert extraction['complete'] and selection['complete']
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    import numpy as np
    from safetensors.numpy import load_file
    from training.shared_tail_training_v9 import make_head
    from training.native_fields_v9 import FIELDS, SIZES
    from roguard.gemma4_review import _score
    assert mx.default_device() == mx.cpu
    languages = {}
    all_predictions = []
    for lang in ('ro', 'uk'):
        info = extraction['languages'][lang]
        for suffix, expected in info['artifacts'].items():
            assert sha(feature_private / (lang + '-' + suffix)) == expected
        rows = [json.loads(line) for line in (feature_private / (lang + '-rows.jsonl')).read_text().splitlines()]
        x = np.load(feature_private / (lang + '-features.npy'), mmap_mode='r', allow_pickle=False)
        cached = np.load(feature_private / (lang + '-logits.npy'), mmap_mode='r', allow_pickle=False)
        assert x.shape == (len(rows), info['dimensions'])
        assert cached.shape == (len(rows), sum(SIZES) + 2)
        chosen = np.array([i for i, r in enumerate(rows) if r['partition'] == 'train'])
        assert len(chosen) == info['partitions']['train']
        train_rows = [rows[i] for i in chosen]
        gold = np.array([r['gold'] for r in train_rows])
        field_gold = np.array([r['fields'] for r in train_rows])
        actual = {}
        for model in ('v9', 'candidate'):
            weights = load_file(paths[lang + '_' + model + '_weights'])
            spec = json.loads(paths[lang + '_' + model + '_spec'].read_text())
            assert sha(paths[lang + '_' + model + '_weights']) == spec['adapter_sha256']
            if model == 'candidate':
                assert spec['adapter_sha256'] == selection['languages'][lang]['selected']['adapter_sha256']
            initial = {k: weights['heads.' + lang + '.' + v] for k, v in {
                '0.weight': 'first.weight', '0.bias': 'first.bias',
                '3.weight': 'last.weight', '3.bias': 'last.bias'}.items()}
            head = make_head(initial)
            head.eval()
            parts = []
            for start in range(0, len(chosen), 128):
                z = head(mx.array(np.asarray(x[chosen[start:start + 128]])))
                mx.eval(z)
                parts.append(np.asarray(z).copy())
            logits = np.concatenate(parts)
            assert np.isfinite(logits).all()
            scores = np.array([_score(z, spec) for z in logits])
            binary = (scores >= spec['threshold']).astype(int)
            fields = np.column_stack([v.argmax(axis=1) for v in np.split(logits[:, :sum(SIZES)], np.cumsum(SIZES)[:-1], axis=1)])
            actual[model] = {'logits': logits, 'scores': scores, 'binary': binary, 'fields': fields,
                             'mode': spec['mode'], 'threshold': spec['threshold']}
            del head, weights, initial
        old_spec = json.loads(paths[lang + '_v9_spec'].read_text())
        reference = np.asarray(cached[chosen])
        parity = float(np.max(np.abs(actual['v9']['logits'] - reference)))
        assert parity <= 1e-3, parity
        reference_binary = np.array([_score(z, old_spec) >= old_spec['threshold'] for z in reference])
        disagreements = int(np.sum(reference_binary != actual['v9']['binary']))
        assert disagreements == 0
        origins = {'controlled-synthetic-scope-composition': 'generated56',
                   'controlled-crossed-scope-composition': 'crossed61'}
        groups = defaultdict(list)
        for j, row in enumerate(train_rows):
            groups['all'].append(j)
            groups['origin:' + origins.get(row['group'], 'historical')].append(j)
            groups['source:' + row['group']].append(j)
            all_predictions.append({
                'case_id': row['case_id'], 'language': lang, 'partition': 'train',
                'group': row['group'], 'gold': int(gold[j]), 'field_gold': field_gold[j].tolist(),
                **{model: {'score': float(actual[model]['scores'][j]), 'binary': int(actual[model]['binary'][j]),
                            'fields': actual[model]['fields'][j].tolist()} for model in actual}})
        summaries = {}
        for group, indices in sorted(groups.items()):
            idx = np.array(indices)
            metrics = {}
            for model, values in actual.items():
                wrong = values['binary'][idx] != gold[idx]
                different = values['fields'][idx] != field_gold[idx]
                metrics[model] = {
                    'binary_errors': int(wrong.sum()),
                    'false_positive': int(np.sum(wrong & (gold[idx] == 0))),
                    'false_negative': int(np.sum(wrong & (gold[idx] == 1))),
                    'joint_field_errors': int(np.any(different, axis=1).sum()),
                    'field_errors': dict(zip(FIELDS, map(int, different.sum(axis=0)))),
                }
            old_bad = actual['v9']['binary'][idx] != gold[idx]
            new_bad = actual['candidate']['binary'][idx] != gold[idx]
            summaries[group] = {'rows': len(indices), **metrics,
                                'fixed': int(np.sum(old_bad & ~new_bad)),
                                'introduced': int(np.sum(~old_bad & new_bad))}
        languages[lang] = {
            'training_rows': len(chosen), 'cached_v9_maximum_logit_error': parity,
            'cached_v9_binary_disagreements': disagreements,
            'modes': {model: {'mode': values['mode'], 'threshold': values['threshold']} for model, values in actual.items()},
            'groups': summaries,
        }
        print(json.dumps({'language': lang, 'all': summaries['all'], 'cached_v9_logit_error': parity}), flush=True)
        del x, cached, actual, reference
    assert sum(v['training_rows'] for v in languages.values()) == 28929
    assert {k: sha(v) for k, v in paths.items()} == plan['bindings']
    a.output.mkdir()
    predictions_path = a.output / 'train-predictions.jsonl'
    predictions_path.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in all_predictions))
    report = {
        'complete': True, 'plan_sha256': sha(plan_path), 'source_sha256': sha(Path(__file__)),
        'bindings': plan['bindings'], 'languages': languages,
        'prediction_rows': len(all_predictions), 'predictions_sha256': sha(predictions_path),
        'scope': 'Observed TRAIN fit of fixed selected62 and original-v9 heads, with saved thresholds. No fitted or selected parameters.',
        'cpu_only': True, 'encoder_loaded': False, 'training_launched': False,
        'development_logits_evaluated': False, 'held_out_accuracy_claimed': False,
        'reserved_test_opened': False, 'stable_release_ready': False,
    }
    (a.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
