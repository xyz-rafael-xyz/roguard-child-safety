"""Rebuild TRAIN/DEV references through one fixed-prefix runtime execution path."""
import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import shutil
import time

from roguard.trajectory_data import file_sha, read_jsonl, write_jsonl
from training.train_large_review_attention_v10 import load_inputs


def bounded_error(left, right, limit):
    import numpy as np
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('Invalid numerical comparison')
    value = float(np.max(np.abs(left-right)))
    if value > limit:
        raise ValueError('Runtime numerical parity failed: '+str(value))
    return value


def load(a):
    plan = json.loads(a.amendment.read_text())
    if plan['source_sha256'] != file_sha(__file__):
        raise ValueError('Registered extractor changed')
    for name, expected in plan['public'].items():
        if file_sha(name) != expected:
            raise ValueError('Bound evidence changed: '+name)
    rows, bindings = load_inputs(a)
    if bindings != plan['input_bindings']:
        raise ValueError('Original inputs changed')
    diagnostic = json.loads((a.diagnostic_study/'report.json').read_text())
    if diagnostic.get('complete') is not True:
        raise ValueError('Replay diagnostic incomplete')
    for lang in ('ro', 'uk'):
        name = lang+'-prefix-dev-logits.npy'
        if file_sha(a.diagnostic_private/name) != plan['runtime_reference_hashes'][lang]:
            raise ValueError('Runtime reference changed')
        if diagnostic['languages'][lang]['artifacts'][name] != plan['runtime_reference_hashes'][lang]:
            raise ValueError('Runtime reference differs from diagnostic')
    if (plan['rows'] != 37766 or plan['native_replay_tolerance'] != 1e-3
            or plan['warm_start_logit_tolerance'] != 1e-5 or plan['runtime_score_tolerance'] != 1e-10
            or sum(map(len, rows.values())) != plan['rows']):
        raise ValueError('Registered population or tolerances differ')
    return plan, rows, bindings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'candidate', 'input_study', 'inputs', 'feature_study', 'features',
                 'initial_study', 'initial_private', 'diagnostic_study', 'diagnostic_private',
                 'study', 'private', 'amendment'):
        p.add_argument('--'+name.replace('_', '-'), dest=name, type=Path, required=True)
    p.add_argument('--preflight-report', type=Path)
    a = p.parse_args()
    if a.study.exists() or a.private.exists():
        raise ValueError('Refusing to overwrite extraction')
    plan, languages, bindings = load(a)
    if a.preflight_report:
        if a.preflight_report.exists():
            raise ValueError('Refusing to overwrite preflight')
        a.preflight_report.parent.mkdir(parents=True, exist_ok=True)
        a.preflight_report.write_text(json.dumps(dict(complete=True, rows=37766, training_rows=28929,
            development_rows=8837, source_sha256=file_sha(__file__), amendment_sha256=file_sha(a.amendment),
            encoder_loaded=False, training_performed=False, held_out_test_opened=False), indent=2)+'\n')
        return
    if shutil.disk_usage(a.private.parent).free < 8*1024**3:
        raise ValueError('Disk reserve reached')
    import numpy as np
    import mlx.core as mx
    from safetensors.numpy import load_file
    from eval.run_trajectory_v10 import LocalFrozenV9
    from roguard.gemma4_review import _Engine, _score
    from roguard.gemma4_attention import capture_with_prefix, replay
    from roguard.gemma4_contract import PREFIX_TOKENS, classifier_tokens
    from training.shared_tail_training_v9 import make_head
    a.study.mkdir(parents=True); a.private.mkdir(parents=True)
    import fcntl
    lock = (a.private/'writer.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol = dict(format='roguard-runtime-review-baseline-1', source_sha256=file_sha(__file__),
        amendment_sha256=file_sha(a.amendment), bindings=bindings,
        runtime_reference_hashes=plan['runtime_reference_hashes'], native_replay_tolerance=1e-3,
        warm_start_logit_tolerance=1e-5, runtime_score_tolerance=1e-10,
        training_performed=False, held_out_test_opened=False, qualified=False)
    (a.study/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (a.study/'source.py').write_bytes(Path(__file__).read_bytes())
    frozen = LocalFrozenV9(a.base, a.candidate, a.study/'v9-binding')
    report = dict(complete=False, languages={}, training_performed=False, held_out_test_opened=False, qualified=False)
    started = time.monotonic()

    def persist():
        report['seconds'] = time.monotonic()-started
        tmp = a.study/'report.tmp'; tmp.write_text(json.dumps(report, indent=2)+'\n'); tmp.replace(a.study/'report.json')

    for lang, rows in languages.items():
        engine = _Engine(*frozen.models[lang]._args); model = engine.model; model.eval()
        prefix = engine.make_cache(model.base)
        mx.eval(model.base.model(mx.array([PREFIX_TOKENS]), cache=prefix))
        spec = frozen.models[lang]._spec
        initial = load_file(a.initial_private/(lang+'-selected.safetensors'))
        mapping = {'0.weight':'first.weight', '0.bias':'first.bias', '3.weight':'last.weight', '3.bias':'last.bias'}
        warm = make_head({k:initial['heads.'+lang+'.'+v] for k,v in mapping.items()}); warm.eval()
        references = np.load(a.diagnostic_private/(lang+'-prefix-dev-logits.npy'), allow_pickle=False)
        old_x = np.load(a.features/(lang+'-features.npy'), mmap_mode='r', allow_pickle=False)
        old_z = np.load(a.features/(lang+'-logits.npy'), mmap_mode='r', allow_pickle=False)
        xs = np.lib.format.open_memmap(a.private/(lang+'-features.npy'), mode='w+', dtype=np.float32, shape=old_x.shape)
        zs = np.lib.format.open_memmap(a.private/(lang+'-logits.npy'), mode='w+', dtype=np.float32, shape=old_z.shape)
        warm_dev = np.empty(references.shape, dtype=np.float32)
        record = dict(complete=False, rows=len(rows), checked=0, development_checked=0,
            maximum_native_replay_error=0., maximum_warm_start_logit_error=0., maximum_runtime_score_error=0.,
            maximum_historical_feature_difference=0., historical_feature_differences_above_001=0,
            historical_binary_decision_changes=0, runtime_source_groups_checked=0)
        report['languages'][lang] = record; persist(); groups = set(); dev_position = 0
        with (a.private/(lang+'-observations.jsonl')).open('x') as observations:
            for i, row in enumerate(rows):
                if shutil.disk_usage(a.private).free < 8*1024**3:
                    raise ValueError('Disk reserve reached')
                if classifier_tokens(engine.tokenizer, row['payload']) != row['tokens']:
                    raise ValueError('Actual runtime tokens differ')
                suffix = mx.array([row['tokens'][len(PREFIX_TOKENS):]])
                state = capture_with_prefix(model.base, suffix, prefix, start=24); mx.eval(state)
                hidden = replay(model.base, state)[:, 0, :]
                z = model.heads[lang](hidden); w = warm(hidden); mx.eval(hidden, z, w)
                x, logits, warm_logits = np.asarray(hidden)[0].copy(), np.asarray(z)[0].copy(), np.asarray(w)[0].copy()
                native = model.base.model(suffix, cache=deepcopy(prefix))[:, -1, :]; mx.eval(native)
                native_error = bounded_error(x, np.asarray(native)[0], plan['native_replay_tolerance'])
                historical_error = float(np.max(np.abs(x-old_x[i])))
                score = _score(logits, spec); old_score = _score(old_z[i], spec)
                decision_change = int((score >= spec['threshold']) != (old_score >= spec['threshold']))
                group = (row['partition'], row['group'])
                if group not in groups:
                    runtime = engine.scores([row['payload']], spec)[0]
                    runtime_error = bounded_error(np.array([score]), np.array([runtime]), plan['runtime_score_tolerance'])
                    if (score >= spec['threshold']) != (runtime >= spec['threshold']):
                        raise ValueError('Served runtime decision differs')
                    groups.add(group); record['runtime_source_groups_checked'] = len(groups)
                    record['maximum_runtime_score_error'] = max(record['maximum_runtime_score_error'], runtime_error)
                if row['partition'] == 'dev':
                    warm_error = bounded_error(warm_logits, references[dev_position], plan['warm_start_logit_tolerance'])
                    record['maximum_warm_start_logit_error'] = max(record['maximum_warm_start_logit_error'], warm_error)
                    warm_dev[dev_position] = warm_logits; dev_position += 1
                xs[i] = x; zs[i] = logits
                record.update(checked=i+1, development_checked=dev_position,
                    maximum_native_replay_error=max(record['maximum_native_replay_error'], native_error),
                    maximum_historical_feature_difference=max(record['maximum_historical_feature_difference'], historical_error),
                    historical_feature_differences_above_001=record['historical_feature_differences_above_001']+int(historical_error>1e-3),
                    historical_binary_decision_changes=record['historical_binary_decision_changes']+decision_change)
                observations.write(json.dumps(dict(feature_index=i, case_id=row['case_id'], partition=row['partition'],
                    native_replay_error=native_error, historical_feature_difference=historical_error,
                    historical_binary_decision_change=decision_change))+'\n')
                if (i+1)%100 == 0 or i+1 == len(rows):
                    xs.flush(); zs.flush(); observations.flush(); persist()
                    print(json.dumps(dict(language=lang, checked=i+1, total=len(rows),
                        maximum_native_replay_error=record['maximum_native_replay_error'], seconds=report['seconds'])), flush=True)
                del suffix, state, hidden, z, w, native; mx.clear_cache()
        if dev_position != len(references):
            raise ValueError('Development reference inventory differs')
        xs.flush(); zs.flush(); del xs, zs
        np.save(a.private/(lang+'-warm-start-dev-logits.npy'), warm_dev, allow_pickle=False)
        metadata = read_jsonl(a.features/(lang+'-rows.jsonl'))
        write_jsonl(a.private/(lang+'-rows.jsonl'), metadata)
        record.update(complete=True, artifacts={p.name:file_sha(p) for p in sorted(a.private.glob(lang+'-*'))})
        persist()
        del engine, model, prefix, warm, initial, references, old_x, old_z, warm_dev
        gc.collect(); mx.clear_cache()
    _, _, final_bindings = load(a)
    if final_bindings != bindings:
        raise ValueError('Original inputs changed during extraction')
    report.update(complete=True, decision='REFERENCE_EXTRACTION_ONLY_NO_MODEL_QUALIFICATION'); persist()


if __name__ == '__main__':
    main()
