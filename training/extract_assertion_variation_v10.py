"""Extract curated TRAIN wording features and replay exact matched parent anchors."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

from roguard.assurance import digest
from roguard.trajectory_data import file_sha, prefix, read_jsonl, write_jsonl
from training.assertion_variation_features_v10 import validate_inputs


def load(args):
    plan = json.loads(args.plan.read_text())
    for base, bindings in ((args.root, plan['public']), (args.private, plan['private'])):
        for name, expected in bindings.items():
            if file_sha(base/name) != expected:
                raise ValueError('Bound source changed: '+name)
    if plan['format'] != 'roguard-assertion-variation-features-1':
        raise ValueError('Wrong feature plan')
    from roguard.gemma4_contract import BASE_REVISION
    if args.base.name != BASE_REVISION:
        raise ValueError('Wrong base model revision')
    parent_study = args.root/'research/v10-evidence/crossed-review-features-62'
    parent_report = json.loads((parent_study/'report.json').read_text())
    features = args.private/'crossed-review-features-62'
    if not parent_report['complete']:
        raise ValueError('Original features are incomplete')
    import numpy as np
    metadata = {}; originals = {}
    for lang, expected in plan['languages'].items():
        metadata[lang] = read_jsonl(features/(lang+'-rows.jsonl'))
        for suffix, checksum in parent_report['languages'][lang]['artifacts'].items():
            if file_sha(features/(lang+'-'+suffix)) != checksum:
                raise ValueError('Original feature artifacts differ')
        x = np.load(features/(lang+'-features.npy'), mmap_mode='r', allow_pickle=False)
        z = np.load(features/(lang+'-logits.npy'), mmap_mode='r', allow_pickle=False)
        if (x.shape != (len(metadata[lang]), 2560) or z.shape != (len(metadata[lang]), 19)
                or x.dtype != np.float32 or z.dtype != np.float32
                or not np.isfinite(x).all() or not np.isfinite(z).all()):
            raise ValueError('Original feature shape, dtype or finiteness differs')
        originals[lang] = (x, z)
        for suffix, key in (('selected.safetensors', 'weights_sha256'), ('review.json', 'candidate_spec_sha256')):
            if file_sha(args.candidate/'actual'/(lang+'-'+suffix)) != expected[key]:
                raise ValueError('Original-v9 candidate binding differs')
        if file_sha(parent_study/'v9-binding'/(lang+'-review.json')) != expected['runtime_spec_sha256']:
            raise ValueError('Original-v9 runtime specification differs')
    data = args.private/'assertion-variation-77/curated'
    rows = validate_inputs(read_jsonl(data/'additional-train.jsonl'), read_jsonl(data/'matched-control.jsonl'),
                           read_jsonl(data/'runtime-inputs.jsonl'), metadata)
    for lang, expected in plan['languages'].items():
        if (len(rows[lang]) != expected['treatment_rows']
                or len({r['parent_index'] for r in rows[lang]}) != expected['unique_control_parents']):
            raise ValueError('Registered new population differs')
    return plan, rows, metadata, originals


def require_dependency(args, plan):
    report_path = args.root/'research/v10-evidence/large-review-attention-sequence-73/report.json'
    sequence = json.loads(report_path.read_text())
    if (sequence.get('execution_complete') is not True or sequence.get('phase') != 'finished'
            or not sequence.get('completed_steps')
            or any(s.get('exit_code') != 0 for s in sequence.get('completed_steps', []))):
        raise ValueError('Preceding encoder sequence is incomplete')
    try:
        os.kill(plan['dependency_pid'], 0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError('Preceding encoder queue is still live')
    proof_path = args.private/plan['dependency_verification']
    proof = json.loads(proof_path.read_text())
    # The independent73 verifier has its own exact source and checkpoint checks.
    study = args.root/'research/v10-evidence/large-review-attention-73'
    if (proof.get('verification') != 'PASS' or proof.get('checkpoint_count') != 8
            or proof.get('checkpoint_dev_rows_recounted') != 35348
            or proof.get('source_sha256') != file_sha(args.root/'eval/verify_large_review_attention_v10.py')
            or proof.get('report_sha256') != file_sha(study/'report.json')
            or proof.get('protocol_sha256') != file_sha(study/'protocol.json')):
        raise ValueError('Preceding encoder outputs are not verified')
    return {'sequence_sha256': file_sha(report_path), 'verification_sha256': file_sha(proof_path)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'private', 'base', 'candidate', 'plan', 'study', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--preflight-report', type=Path)
    a = p.parse_args()
    if a.study.exists() or a.output.exists():
        raise ValueError('Refusing to overwrite extraction')
    plan, rows, metadata, originals = load(a)
    protocol = {'format': plan['format'], 'plan_sha256': file_sha(a.plan), 'source_sha256': file_sha(__file__),
                'treatment_rows': sum(map(len, rows.values())), 'languages': plan['languages'],
                'new_partitions': ['train'], 'training_performed': False, 'held_out_test_opened': False, 'qualified': False}
    if a.preflight_report:
        if a.preflight_report.exists(): raise ValueError('Preflight already exists')
        a.preflight_report.parent.mkdir(parents=True, exist_ok=True)
        a.preflight_report.write_text(json.dumps({**protocol, 'complete': True, 'encoder_loaded': False,
            'all_parent_positions_labels_tokens_and_factor_cells_verified': True}, indent=2)+'\n')
        print(json.dumps({'preflight_verified': True, 'rows': protocol['treatment_rows'], 'encoder_loaded': False}))
        return
    dependency = require_dependency(a, plan)
    if shutil.disk_usage(a.private).free < 8*1024**3: raise RuntimeError('Disk reserve reached')
    a.study.mkdir(parents=True); a.output.mkdir(parents=True)
    import fcntl
    lock = (a.output/'writer.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    (a.study/'protocol.json').write_text(json.dumps({**protocol, 'dependency': dependency}, indent=2)+'\n')
    (a.study/'source.py').write_bytes(Path(__file__).read_bytes())
    import gc
    import numpy as np
    import mlx.nn as nn
    from eval.run_trajectory_v10 import LocalFrozenV9
    from training.extract_semantic_augmentation_v10 import CachedEngine
    from roguard.gemma4_contract import classifier_tokens
    from roguard.gemma4_review import _score
    frozen = LocalFrozenV9(a.base, a.candidate, a.study/'v9-binding')
    started = time.monotonic()
    report = {'complete': False, 'languages': {}, 'plan_sha256': file_sha(a.plan),
              'source_sha256': file_sha(__file__), 'dependency': dependency,
              'training_performed': False, 'held_out_test_opened': False, 'qualified': False}

    def persist():
        report['seconds'] = time.monotonic()-started
        temporary = a.study/'report.tmp'; temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(a.study/'report.json')

    for lang in ('ro', 'uk'):
        component = frozen.models[lang]; engine = CachedEngine(*component._args); mx = engine.mx

        class Capture(nn.Module):
            def __init__(self, inner): super().__init__(); self.inner = inner
            def __call__(self, hidden):
                logits = self.inner(hidden); mx.eval(hidden, logits)
                self.hidden = np.asarray(hidden).copy(); self.logits = np.asarray(logits).copy()
                return logits

        tap = Capture(engine.model.heads[lang]); engine.model.heads[lang] = tap
        if file_sha(a.study/'v9-binding'/(lang+'-review.json')) != plan['languages'][lang]['runtime_spec_sha256']:
            raise ValueError('Loaded runtime specification differs')
        old_x, old_z = originals[lang]; parent_values = {}; maximum_x = maximum_z = 0.

        def infer(payload, expected_tokens=None):
            tokens = classifier_tokens(engine.tokenizer, payload)
            if expected_tokens is not None and tokens != expected_tokens:
                raise ValueError('Actual runtime tokenization differs from audited tokens')
            score = engine.scores([payload], component._spec)[0]
            if score is None: raise ValueError('Input exceeded runtime envelope')
            x, z = tap.hidden[0].copy(), tap.logits[0].copy()
            if x.shape != (2560,) or z.shape != (19,) or not np.isfinite(x).all() or not np.isfinite(z).all():
                raise ValueError('Invalid extracted representation')
            mx.clear_cache()
            return x, z, tokens

        for item in rows[lang]:
            index = item['parent_index']
            if index in parent_values: continue
            control = item['control']; payload = prefix(control, control['focus'])
            x, z, tokens = infer(payload)
            if digest(tokens) != metadata[lang][index]['token_sha256']:
                raise ValueError('Replayed control token identity differs')
            dx = float(np.max(np.abs(x-old_x[index]))); dz = float(np.max(np.abs(z-old_z[index])))
            maximum_x = max(maximum_x, dx); maximum_z = max(maximum_z, dz)
            if dx > plan['maximum_parent_feature_error'] or dz > plan['maximum_parent_logit_error']:
                raise ValueError('Replayed control differs from bound original feature store')
            spec = component._spec
            if (_score(z, spec) >= spec['threshold']) != (_score(old_z[index], spec) >= spec['threshold']):
                raise ValueError('Replayed control binary decision differs')
            parent_values[index] = (x, z)
        xs = []; zs = []; records = []
        for number, item in enumerate(rows[lang], 1):
            if shutil.disk_usage(a.output).free < 8*1024**3: raise RuntimeError('Disk reserve reached')
            record = item['input']; x, z, tokens = infer(record['payload'], record['tokens'])
            xs.append(x); zs.append(z)
            records.append({'case_id': record['case_id'], 'language': lang, 'partition': 'train',
                'group': 'assertion_variation_77', 'gold': record['gold'], 'fields': record['fields'],
                'token_sha256': digest(tokens), 'pair_id': record['pair_id'], 'parent_feature_index': item['parent_index'],
                'parent_case_id': record['parent']['case_id'], 'parent_token_sha256': record['parent']['token_sha256'],
                'factors': record['factors']})
            if number % 32 == 0 or number == len(rows[lang]):
                event = {'language': lang, 'treatment_completed': number, 'treatment_total': len(rows[lang]),
                         'control_parents_replayed': len(parent_values), 'seconds': time.monotonic()-started}
                (a.study/'progress.json').write_text(json.dumps(event, indent=2)+'\n')
                print(json.dumps(event), flush=True)
        parents = sorted(parent_values)
        arrays = {'treatment-features.npy': np.asarray(xs, dtype=np.float32),
                  'treatment-logits.npy': np.asarray(zs, dtype=np.float32),
                  'control-parent-features.npy': np.asarray([parent_values[i][0] for i in parents], dtype=np.float32),
                  'control-parent-logits.npy': np.asarray([parent_values[i][1] for i in parents], dtype=np.float32)}
        for suffix, array in arrays.items(): np.save(a.output/(lang+'-'+suffix), array, allow_pickle=False)
        write_jsonl(a.output/(lang+'-rows.jsonl'), records)
        write_jsonl(a.output/(lang+'-control-parents.jsonl'), [{'feature_index': i, **metadata[lang][i]} for i in parents])
        report['languages'][lang] = {'treatment_rows': len(records), 'control_parent_rows': len(parents),
            'dimensions': 2560, 'maximum_parent_feature_error': maximum_x, 'maximum_parent_logit_error': maximum_z,
            'parent_binary_decision_disagreements': 0,
            'artifacts': {path.name: file_sha(path) for path in sorted(a.output.glob(lang+'-*'))}}
        persist()
        del infer, engine, tap, x, z, xs, zs, arrays, parent_values
        gc.collect(); mx.clear_cache()
    report['complete'] = True; persist(); print(json.dumps({'complete': True, 'seconds': report['seconds']}), flush=True)


if __name__ == '__main__': main()
