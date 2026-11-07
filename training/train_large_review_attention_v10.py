"""Stream the recovered larger TRAIN corpus through trainable binary attention tails."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import time

from roguard.trajectory_data import file_sha, read_jsonl, write_jsonl
from training.large_review_attention_v10 import (check_trainable, field_weights, make_loss,
                                                make_optimizer, summarize, training_schedule, validate_rows)
from training.native_fields_v9 import SIZES


HELPERS = ('training/large_review_attention_v10.py', 'training/coupled_policy_loss_v10.py',
           'training/native_fields_v9.py', 'training/separation_frontier_v9.py',
           'training/balanced_headroom_frontier_v9.py', 'src/roguard/gemma4_attention.py',
           'src/roguard/gemma4_review.py', 'src/roguard/gemma4_contract.py',
           'src/roguard/native_fields.py', 'src/roguard/evidence.py', 'eval/run_trajectory_v10.py')


def load_inputs(a):
    """No encoder import or reserved data access; usable as a preflight."""
    import numpy as np
    from safetensors.numpy import load_file
    recovery = json.loads((a.input_study/'report.json').read_text())
    verification = json.loads((a.input_study/'verification.json').read_text())
    feature_report = json.loads((a.feature_study/'report.json').read_text())
    parent_report = json.loads((a.initial_study/'report.json').read_text())
    input_path = a.inputs/'inputs.jsonl'
    if (recovery.get('complete') is not True or verification.get('verified') is not True
            or verification['report_sha256'] != file_sha(a.input_study/'report.json')
            or verification['inputs_sha256'] != recovery['inputs_sha256']
            or file_sha(input_path) != recovery['inputs_sha256']
            or feature_report.get('complete') is not True or parent_report.get('complete') is not True
            or recovery['feature_report_sha256'] != file_sha(a.feature_study/'report.json')):
        raise ValueError('Input recovery or preceding studies are incomplete or changed')
    bindings = {'input_report_sha256': file_sha(a.input_study/'report.json'),
                'input_verification_sha256': file_sha(a.input_study/'verification.json'),
                'inputs_sha256': file_sha(input_path),
                'feature_report_sha256': file_sha(a.feature_study/'report.json'),
                'initial_report_sha256': file_sha(a.initial_study/'report.json'),
                'helpers': {name: file_sha(name) for name in HELPERS}}
    metadata = {}
    for lang in ('ro', 'uk'):
        info = feature_report['languages'][lang]
        hashes = {suffix: file_sha(a.features/(lang+'-'+suffix)) for suffix in info['artifacts']}
        if hashes != info['artifacts']:
            raise ValueError('Feature store changed')
        metadata[lang] = read_jsonl(a.features/(lang+'-rows.jsonl'))
        for suffix, width in (('features.npy', 2560), ('logits.npy', sum(SIZES)+2)):
            values = np.load(a.features/(lang+'-'+suffix), mmap_mode='r', allow_pickle=False)
            if (values.shape != (len(metadata[lang]), width) or values.dtype != np.float32
                    or not np.isfinite(values).all()):
                raise ValueError('Invalid feature store array')
        original_path = a.candidate/'actual'/(lang+'-selected.safetensors')
        initial_path = a.initial_private/(lang+'-selected.safetensors')
        original = load_file(original_path)
        initial = load_file(initial_path)
        head_names = {f'heads.{lang}.{layer}.{part}' for layer in ('first', 'last') for part in ('weight', 'bias')}
        if (set(original) != set(initial) or not head_names <= set(initial)
                or any(v.dtype != np.float32 or not np.isfinite(v).all() or v.shape != original[k].shape
                       for k, v in initial.items())
                or any(not np.array_equal(v, original[k]) for k, v in initial.items() if k not in head_names)):
            raise ValueError('Warm start changes more than the registered language head')
        original_spec_path = a.feature_study/'v9-binding'/(lang+'-review.json')
        initial_spec_path = a.initial_study/(lang+'-review.json')
        original_spec = json.loads(original_spec_path.read_text())
        initial_spec = json.loads(initial_spec_path.read_text())
        if (original_spec['adapter_sha256'] != file_sha(original_path)
                or initial_spec['adapter_sha256'] != file_sha(initial_path)
                or parent_report['languages'][lang]['selected']['adapter_sha256'] != file_sha(initial_path)
                or {k: v for k, v in initial_spec.items() if k not in ('adapter_sha256', 'mode', 'threshold')}
                   != {k: v for k, v in original_spec.items() if k not in ('adapter_sha256', 'mode', 'threshold')}):
            raise ValueError('Warm-start specification differs from the unchanged runtime contract')
        initial_logits = np.load(a.initial_private/(lang+'-selected-dev-logits.npy'), allow_pickle=False)
        ndev = sum(row['partition'] == 'dev' for row in metadata[lang])
        if initial_logits.shape != (ndev, sum(SIZES)+2) or not np.isfinite(initial_logits).all():
            raise ValueError('Invalid warm-start reference logits')
        bindings[lang] = {'feature_artifacts': hashes, 'original_weights_sha256': file_sha(original_path),
                         'original_spec_sha256': file_sha(original_spec_path),
                         'initial_weights_sha256': file_sha(initial_path),
                         'initial_spec_sha256': file_sha(initial_spec_path),
                         'initial_dev_logits_sha256': file_sha(a.initial_private/(lang+'-selected-dev-logits.npy'))}
    rows = validate_rows(read_jsonl(input_path), metadata)
    return rows, bindings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'candidate', 'input_study', 'inputs', 'feature_study', 'features',
                 'initial_study', 'initial_private', 'study', 'private', 'amendment'):
        parser.add_argument('--'+name.replace('_', '-'), dest=name, type=Path, required=True)
    parser.add_argument('--preflight-report', type=Path)
    a = parser.parse_args()
    if a.study.exists() or a.private.exists():
        raise ValueError('Refusing to overwrite an attention experiment')
    plan = json.loads(a.amendment.read_text())
    config = plan['optimization']
    expected = {'epochs': 3, 'batch_size': 8, 'seed': 1073, 'learning_rate': 1e-5,
                'head_learning_rate': 1e-4, 'weight_decay': .01, 'gradient_norm_clip': 1.,
                'teacher_kl': .2, 'policy_ce': {'ro': .25, 'uk': 1.}}
    if config != expected or plan['source_sha256'] != file_sha(__file__):
        raise ValueError('Registered source or optimization differs')
    rows_by_language, bindings = load_inputs(a)
    schedules = {lang: training_schedule(rows, config['epochs'], config['batch_size'])
                 for lang, rows in rows_by_language.items()}
    counts = {lang: {split: sum(r['partition'] == split for r in rows) for split in ('train', 'dev')}
              for lang, rows in rows_by_language.items()}
    if bindings != plan['input_bindings'] or schedules != plan['schedules'] or counts != plan['populations']:
        raise ValueError('Registered inputs or schedule changed')
    if a.preflight_report:
        if a.preflight_report.exists():
            raise ValueError('Refusing to overwrite preflight')
        record = {'complete': True, 'source_sha256': file_sha(__file__),
                  'amendment_sha256': file_sha(a.amendment), 'bindings': bindings,
                  'schedules': schedules, 'populations': counts, 'encoder_loaded': False,
                  'training_launched': False, 'held_out_test_opened': False, 'qualified': False}
        a.preflight_report.parent.mkdir(parents=True, exist_ok=True)
        a.preflight_report.write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps({'complete': True, 'populations': counts, 'schedules': schedules})); return
    def disk_check():
        if shutil.disk_usage(a.private.parent).free < 8*1024**3:
            raise RuntimeError('Disk reserve reached')
    disk_check()
    a.study.mkdir(parents=True); a.private.mkdir(parents=True)
    import fcntl
    lock = (a.private/'writer.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = {'format': 'roguard-large-review-attention-fit-1', 'source_sha256': file_sha(__file__),
                'amendment_sha256': file_sha(a.amendment), 'optimization': config,
                'bindings': bindings, 'schedules': schedules, 'populations': counts,
                'persistent_body_cache': False, 'held_out_test_opened': False, 'stable_release_ready': False}
    (a.study/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (a.study/'source.py').write_bytes(Path(__file__).read_bytes())
    (a.study/'helper.py').write_bytes(Path('training/large_review_attention_v10.py').read_bytes())
    import numpy as np
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from safetensors.numpy import load_file, save_file
    from eval.run_trajectory_v10 import LocalFrozenV9
    from roguard.gemma4_attention import capture_with_prefix, replay
    from roguard.gemma4_contract import PREFIX_TOKENS, classifier_tokens
    from roguard.gemma4_review import _Engine, _load_weights, _score
    frozen = LocalFrozenV9(a.base, a.candidate, a.study/'v9-binding')
    report = {'complete': False, 'languages': {}, 'held_out_test_opened': False,
              'stable_release_ready': False, 'training_rows': 28929, 'development_rows': 8837}
    started = time.monotonic()
    def persist():
        report['seconds'] = time.monotonic()-started
        temporary = a.study/'report.tmp'
        temporary.write_text(json.dumps(report, indent=2)+'\n'); temporary.replace(a.study/'report.json')
    for lang in ('ro', 'uk'):
        rows = rows_by_language[lang]
        train = np.array([i for i, r in enumerate(rows) if r['partition'] == 'train'])
        dev = np.array([i for i, r in enumerate(rows) if r['partition'] == 'dev'])
        dev_rows = [rows[i] for i in dev]
        labels = np.array([r['fields'] for r in rows], dtype=np.int32)
        gold = np.array([r['gold'] for r in rows], dtype=np.int32)
        features = np.load(a.features/(lang+'-features.npy'), mmap_mode='r', allow_pickle=False)
        teacher = np.load(a.features/(lang+'-logits.npy'), mmap_mode='r', allow_pickle=False)
        initial_logits = np.load(a.initial_private/(lang+'-selected-dev-logits.npy'), allow_pickle=False)
        original_spec = frozen.models[lang]._spec
        original = load_file(a.candidate/'actual'/(lang+'-selected.safetensors'))
        initial = load_file(a.initial_private/(lang+'-selected.safetensors'))
        engine = _Engine(*frozen.models[lang]._args)
        model = engine.model
        _load_weights(model, initial)
        model.heads['uk' if lang == 'ro' else 'ro'].freeze()
        names = check_trainable(model, lang)
        if len(names) != 76:
            raise ValueError('Expected 72 attention and four head tensors')
        baseline = np.array([int(_score(teacher[i], original_spec) >= original_spec['threshold']) for i in dev])
        prefix = engine.make_cache(model.base)
        mx.eval(model.base.model(mx.array([PREFIX_TOKENS]), cache=prefix))
        record = {'training_rows': len(train), 'development_rows': len(dev), 'checkpoints': [],
                  'selected': None, 'complete': False, 'trainable_tensors': len(names),
                  'maximum_initial_feature_error': 0., 'maximum_initial_logit_error': 0.,
                  'maximum_full_replay_error': 0.}
        report['languages'][lang] = record
        step = presentations = 0
        def state_for(i):
            state = capture_with_prefix(model.base, mx.array([rows[i]['tokens'][len(PREFIX_TOKENS):]]), prefix, start=24)
            mx.eval(state)
            return state
        def infer(indices, *, initial_check=False, check_full=False):
            model.eval(); values = []
            checked_groups = set()
            for i in indices:
                state = state_for(i)
                hidden = replay(model.base, state)[:, 0, :]
                z = model.heads[lang](hidden); mx.eval(hidden, z)
                if initial_check:
                    error = float(np.max(np.abs(np.asarray(hidden)[0]-features[i])))
                    record['maximum_initial_feature_error'] = max(record['maximum_initial_feature_error'], error)
                    if error > 1e-3:
                        raise ValueError('Initial encoder differs from the bound larger feature store')
                if check_full and rows[i]['group'] not in checked_groups:
                    if classifier_tokens(engine.tokenizer, rows[i]['payload']) != rows[i]['tokens']:
                        raise ValueError('Actual runtime tokenizer differs from recovered tokens')
                    full = model.base.model(mx.array([rows[i]['tokens']]))[:, -1, :]; mx.eval(full)
                    error = float(np.max(np.abs(np.asarray(full)-np.asarray(hidden))))
                    record['maximum_full_replay_error'] = max(record['maximum_full_replay_error'], error)
                    if error > 1e-3:
                        raise ValueError('Full encoder and streaming tail replay differ')
                    checked_groups.add(rows[i]['group'])
                values.append(np.asarray(z)[0].copy())
                del state, hidden, z; mx.clear_cache()
            return np.stack(values)
        # Verify the runtime tokenization and representation of every TRAIN source group.
        probe_by_group = {}
        for i in train:
            probe_by_group.setdefault(rows[i]['group'], i)
        infer(list(probe_by_group.values()), initial_check=True, check_full=True)
        def evaluate(epoch):
            z = infer(dev, initial_check=epoch == 0, check_full=True)
            if epoch == 0:
                error = float(np.max(np.abs(z-initial_logits)))
                record['maximum_initial_logit_error'] = error
                if error > 1e-3:
                    raise ValueError('Initial logits differ from the selected71 head')
            summary, predictions = summarize(dev_rows, z, original_spec['couplings'], baseline)
            if epoch == 0:
                reference, _ = summarize(dev_rows, initial_logits, original_spec['couplings'], baseline)
                if any(summary[k] != reference[k] for k in ('binary_errors', 'joint_argmax_errors', 'field_errors')):
                    raise ValueError('Initial warm-start errors differ from the retained logits')
            tensors = dict(original)
            parameters = dict(tree_flatten(model.trainable_parameters()))
            if set(parameters) != names:
                raise ValueError('Trainable parameter inventory changed')
            for name, value in parameters.items():
                v = np.asarray(value).copy()
                if v.dtype != np.float32 or not np.isfinite(v).all() or v.shape != original[name].shape:
                    raise ValueError('Invalid attention checkpoint tensor')
                tensors[name] = v
            if epoch == 0 and any(not np.array_equal(tensors[k], initial[k]) for k in tensors):
                raise ValueError('Epoch-zero checkpoint differs from registered initialization')
            disk_check()
            destination = a.private/lang; destination.mkdir(exist_ok=True)
            checkpoint = destination/f'epoch-{epoch}.safetensors'
            save_file(tensors, str(checkpoint))
            np.save(destination/f'epoch-{epoch}-dev-logits.npy', z, allow_pickle=False)
            summary.update(epoch=epoch, adapter_sha256=file_sha(checkpoint),
                           development_logits_sha256=file_sha(destination/f'epoch-{epoch}-dev-logits.npy'),
                           completed_training_updates=step, completed_training_presentations=presentations)
            record['checkpoints'].append(summary)
            if record['selected'] is None or summary['selection_rank'] < record['selected']['selection_rank']:
                record['selected'] = dict(summary)
                selected = a.private/(lang+'-selected.safetensors'); selected.write_bytes(checkpoint.read_bytes())
                np.save(a.private/(lang+'-selected-dev-logits.npy'), z, allow_pickle=False)
                spec = dict(original_spec, adapter_sha256=file_sha(selected), mode=summary['mode'], threshold=summary['threshold'])
                (a.study/(lang+'-review.json')).write_text(json.dumps(spec, indent=2)+'\n')
                write_jsonl(a.study/(lang+'-development-predictions.jsonl'), predictions)
            persist()
            print(json.dumps({'language': lang, 'epoch': epoch, 'binary_errors': summary['binary_errors'],
                              'joint_argmax_errors': summary['joint_argmax_errors'], 'introduced_v9': summary['introduced_v9'],
                              'selected_epoch': record['selected']['epoch'], 'seconds': time.monotonic()-started}), flush=True)
        weights = [mx.array(v) for v in field_weights(labels[train])]
        objective = make_loss(mx, nn, original_spec['couplings'], weights,
                              teacher_kl=config['teacher_kl'], policy_ce=config['policy_ce'][lang])
        def loss_fn(current, states, y, fields, old):
            z = mx.concatenate([current.heads[lang](replay(current.base, state)[:, 0, :]) for state in states])
            return objective(z, y, fields, old)
        value_grad = nn.value_and_grad(model, loss_fn)
        optimizer, rate_groups = make_optimizer(config, lang)
        mx.random.seed(config['seed']); rng = np.random.default_rng(config['seed'])
        evaluate(0)
        total = schedules[lang]['total_updates']; warm = max(1, int(.05*total))
        for epoch in range(1, config['epochs']+1):
            model.train(); order = rng.permutation(train)
            for offset in range(0, len(order), config['batch_size']):
                disk_check()
                chosen = order[offset:offset+config['batch_size']]
                states = [state_for(i) for i in chosen]
                step += 1
                scale = min(step/warm, max(.1, (total-step)/max(1, total-warm)))
                for group, rate in rate_groups:
                    group.learning_rate = rate*scale
                loss, gradient = value_grad(model, states, mx.array(gold[chosen]), mx.array(labels[chosen]), mx.array(teacher[chosen]))
                gradient, norm = optim.clip_grad_norm(gradient, config['gradient_norm_clip']); mx.eval(loss, norm, gradient)
                if not bool(mx.isfinite(loss).item()) or not bool(mx.isfinite(norm).item()):
                    raise ValueError('Nonfinite attention update')
                optimizer.update(model, gradient); mx.eval(model.trainable_parameters(), optimizer.state)
                presentations += len(chosen)
                if step % 25 == 0 or offset+len(chosen) == len(order):
                    print(json.dumps({'language': lang, 'epoch': epoch, 'step': step, 'total_steps': total,
                                      'presentations': presentations, 'loss': float(loss.item()),
                                      'seconds': time.monotonic()-started}), flush=True)
                del states, gradient; mx.clear_cache()
            evaluate(epoch)
        if step != total or presentations != schedules[lang]['total_presentations']:
            raise ValueError('Incomplete registered training schedule')
        selected = load_file(a.private/(lang+'-selected.safetensors'))
        model.load_weights([(k, mx.array(selected[k])) for k in names], strict=False); mx.eval(model.trainable_parameters())
        saved = np.load(a.private/(lang+'-selected-dev-logits.npy'), allow_pickle=False)
        reloaded = infer(dev, check_full=True)
        error = float(np.max(np.abs(reloaded-saved)))
        if error > 1e-5 or not np.array_equal(reloaded.argmax(axis=1), saved.argmax(axis=1)):
            raise ValueError('Selected attention checkpoint reload differs')
        record.update(complete=True, selected_reload_maximum_logit_error=error)
        persist()
        del engine, model, prefix, optimizer, rate_groups, value_grad, objective, selected, features, teacher
        gc.collect(); mx.clear_cache()
    _, final_bindings = load_inputs(a)
    if final_bindings != bindings or file_sha(__file__) != plan['source_sha256']:
        raise ValueError('Source or inputs changed during the registered fit')
    report.update(complete=True, decision='REJECT' if any(
        r['selected']['binary_errors'] or r['selected']['joint_argmax_errors']
        for r in report['languages'].values()) else 'DEVELOPMENT_PASS_REQUIRES_FULL_REGRESSION')
    persist()
    print(json.dumps({'complete': True, 'decision': report['decision'], 'seconds': report['seconds']}), flush=True)


if __name__ == '__main__':
    main()
