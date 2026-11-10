"""Compare cached and uncached execution with retained features; never fit weights."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import time

from roguard.trajectory_data import file_sha
from training.train_large_review_attention_v10 import load_inputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'candidate', 'input_study', 'inputs', 'feature_study', 'features',
                 'initial_study', 'initial_private', 'failed_study', 'study', 'private'):
        p.add_argument('--'+name.replace('_', '-'), dest=name, type=Path, required=True)
    a = p.parse_args()
    if a.study.exists() or a.private.exists():
        raise ValueError('Refusing to overwrite a diagnostic')
    import numpy as np
    import mlx.core as mx
    from safetensors.numpy import load_file
    from eval.run_trajectory_v10 import LocalFrozenV9
    from roguard.gemma4_attention import capture, capture_with_prefix, replay
    from roguard.gemma4_contract import PREFIX_TOKENS, classifier_tokens
    from roguard.gemma4_review import _Engine, _load_weights, _score
    from training.large_review_attention_v10 import summarize

    rows_by_language, bindings = load_inputs(a)
    failed = json.loads((a.failed_study/'protocol.json').read_text())
    if (bindings != failed['bindings'] or
            file_sha('training/train_large_review_attention_v10.py') != failed['source_sha256']):
        raise ValueError('Failed run source or input binding changed')
    if shutil.disk_usage(a.private.parent).free < 8*1024**3:
        raise ValueError('Less than 8 GiB free')
    a.study.mkdir(parents=True); a.private.mkdir(parents=True)
    started = time.monotonic()
    protocol = {'format': 'roguard-replay-parity-diagnostic-1',
        'source_sha256': file_sha(__file__), 'failed_protocol_sha256': file_sha(a.failed_study/'protocol.json'),
        'bindings': bindings, 'paths': ['prefix', 'uncached'],
        'population': 'All DEV rows and the first TRAIN row of every source group, in retained order',
        'full_forward_population': 'First row of each partition/group and every feature or logit tolerance failure',
        'feature_tolerance': 1e-3, 'logit_tolerance': 1e-3, 'full_replay_tolerance': 1e-3,
        'optimizer_updates': 0, 'training_changed': False, 'held_out_test_opened': False,
        'qualification_possible': False}
    (a.study/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (a.study/'source.py').write_bytes(Path(__file__).read_bytes())
    frozen = LocalFrozenV9(a.base, a.candidate, a.study/'v9-binding')
    report = {'complete': False, 'languages': {}, 'optimizer_updates': 0,
              'training_changed': False, 'held_out_test_opened': False, 'qualified': False}

    def persist():
        report['seconds'] = time.monotonic()-started
        tmp = a.study/'report.tmp'
        tmp.write_text(json.dumps(report, indent=2)+'\n'); tmp.replace(a.study/'report.json')

    def field_argmax(z):
        offsets = (0, 3, 6, 10, 13, 15, 17, 19)
        return [int(z[lo:hi].argmax()) for lo, hi in zip(offsets, offsets[1:])]

    for lang, rows in rows_by_language.items():
        features = np.load(a.features/(lang+'-features.npy'), mmap_mode='r', allow_pickle=False)
        teacher = np.load(a.features/(lang+'-logits.npy'), mmap_mode='r', allow_pickle=False)
        reference = np.load(a.initial_private/(lang+'-selected-dev-logits.npy'), allow_pickle=False)
        spec = json.loads((a.initial_study/(lang+'-review.json')).read_text())
        engine = _Engine(*frozen.models[lang]._args); model = engine.model
        _load_weights(model, load_file(a.initial_private/(lang+'-selected.safetensors')))
        model.eval()
        prefix = engine.make_cache(model.base)
        mx.eval(model.base.model(mx.array([PREFIX_TOKENS]), cache=prefix))
        dev = [i for i, r in enumerate(rows) if r['partition'] == 'dev']
        probes = {}
        for i, r in enumerate(rows):
            if r['partition'] == 'train':
                probes.setdefault(r['group'], i)
        dev_positions = {i: j for j, i in enumerate(dev)}
        record = {'complete': False, 'development_rows': len(dev), 'training_probes': len(probes),
                  'checked': 0, 'full_forward_checks': 0, 'paths': {}}
        for name in protocol['paths']:
            record['paths'][name] = {'maximum_feature_error': 0., 'feature_tolerance_failures': 0,
                'maximum_dev_logit_error': 0., 'dev_logit_tolerance_failures': 0,
                'dev_field_or_binary_argmax_changes': 0, 'dev_fixed_threshold_decision_changes': 0,
                'maximum_full_replay_error': 0., 'full_replay_tolerance_failures': 0}
        report['languages'][lang] = record; persist()
        logits = {name: np.empty_like(reference) for name in protocol['paths']}
        checked_groups = set()
        row_path = a.private/(lang+'-observations.jsonl')
        with row_path.open('x') as output:
            for i in list(probes.values())+dev:
                row = rows[i]
                if classifier_tokens(engine.tokenizer, row['payload']) != row['tokens']:
                    raise ValueError('Recovered tokens differ from actual runtime tokenizer')
                variants = {}; observation = {
                    'feature_index': i, 'case_id': row['case_id'], 'partition': row['partition'],
                    'group': row['group'], 'paths': {}}
                needs_full = (row['partition'], row['group']) not in checked_groups
                for name in protocol['paths']:
                    if name == 'prefix':
                        state = capture_with_prefix(model.base,
                            mx.array([row['tokens'][len(PREFIX_TOKENS):]]), prefix, start=24)
                    else:
                        state = capture(model.base, mx.array([row['tokens']]), start=24)
                    mx.eval(state)
                    hidden = replay(model.base, state)[:, 0, :]
                    z = model.heads[lang](hidden); mx.eval(hidden, z)
                    h, v = np.asarray(hidden)[0].copy(), np.asarray(z)[0].copy()
                    if not np.isfinite(h).all() or not np.isfinite(v).all():
                        raise ValueError('Nonfinite diagnostic inference')
                    variants[name] = h
                    delta = np.abs(h-features[i]); error = float(delta.max())
                    stats = record['paths'][name]
                    stats['maximum_feature_error'] = max(stats['maximum_feature_error'], error)
                    stats['feature_tolerance_failures'] += int(error > protocol['feature_tolerance'])
                    obs = {'feature_error': error, 'maximum_error_dimension': int(delta.argmax())}
                    needs_full |= error > protocol['feature_tolerance']
                    if i in dev_positions:
                        position = dev_positions[i]; old = reference[position]; logits[name][position] = v
                        error = float(np.max(np.abs(v-old)))
                        stats['maximum_dev_logit_error'] = max(stats['maximum_dev_logit_error'], error)
                        stats['dev_logit_tolerance_failures'] += int(error > protocol['logit_tolerance'])
                        args, old_args = field_argmax(v), field_argmax(old)
                        score, old_score = _score(v, spec), _score(old, spec)
                        decision, old_decision = int(score >= spec['threshold']), int(old_score >= spec['threshold'])
                        stats['dev_field_or_binary_argmax_changes'] += int(args != old_args)
                        stats['dev_fixed_threshold_decision_changes'] += int(decision != old_decision)
                        obs.update(logit_error=error, argmax=args, reference_argmax=old_args,
                                   score=score, reference_score=old_score, decision=decision,
                                   reference_decision=old_decision)
                        needs_full |= error > protocol['logit_tolerance']
                    observation['paths'][name] = obs
                    del state, hidden, z; mx.clear_cache()
                if needs_full:
                    full = model.base.model(mx.array([row['tokens']]))[:, -1, :]; mx.eval(full)
                    value = np.asarray(full)[0].copy()
                    if not np.isfinite(value).all():
                        raise ValueError('Nonfinite full-forward inference')
                    observation['full_feature_error'] = float(np.max(np.abs(value-features[i])))
                    for name, h in variants.items():
                        error = float(np.max(np.abs(value-h)))
                        observation['paths'][name]['full_replay_error'] = error
                        stats = record['paths'][name]
                        stats['maximum_full_replay_error'] = max(stats['maximum_full_replay_error'], error)
                        stats['full_replay_tolerance_failures'] += int(error > protocol['full_replay_tolerance'])
                    record['full_forward_checks'] += 1
                    checked_groups.add((row['partition'], row['group']))
                    del full; mx.clear_cache()
                output.write(json.dumps(observation)+'\n'); output.flush()
                record['checked'] += 1
                if record['checked'] % 50 == 0:
                    persist()
                    print(json.dumps({'language': lang, 'checked': record['checked'],
                        'total': len(probes)+len(dev), 'paths': record['paths'],
                        'seconds': time.monotonic()-started}), flush=True)
        original_spec = frozen.models[lang]._spec
        baseline = np.array([int(_score(teacher[i], original_spec) >= original_spec['threshold']) for i in dev])
        dev_rows = [rows[i] for i in dev]
        record['reference_summary'], _ = summarize(dev_rows, reference, original_spec['couplings'], baseline)
        record['artifacts'] = {row_path.name: file_sha(row_path)}
        for name, values in logits.items():
            path = a.private/(lang+'-'+name+'-dev-logits.npy')
            np.save(path, values, allow_pickle=False)
            record['artifacts'][path.name] = file_sha(path)
            summary, _ = summarize(dev_rows, values, original_spec['couplings'], baseline)
            record['paths'][name]['summary'] = summary
            record['paths'][name]['initial_error_counts_equal'] = all(
                summary[key] == record['reference_summary'][key]
                for key in ('binary_errors', 'joint_argmax_errors', 'field_errors'))
        record['complete'] = True; persist()
        del model, engine, prefix, features, teacher, reference, logits
        gc.collect(); mx.clear_cache()
    _, final_bindings = load_inputs(a)
    if final_bindings != bindings or file_sha(__file__) != protocol['source_sha256']:
        raise ValueError('Diagnostic source or inputs changed')
    report['complete'] = True
    report['decision'] = 'DIAGNOSTIC_ONLY_NO_MODEL_QUALIFICATION'
    persist()
    print(json.dumps({'complete': True, 'decision': report['decision'], 'seconds': report['seconds']}), flush=True)


if __name__ == '__main__':
    main()
