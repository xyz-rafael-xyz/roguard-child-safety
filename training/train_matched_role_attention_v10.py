"""Train one registered arm of the matched role-diversity attention comparison."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import time

from roguard.trajectory_data import file_sha, write_jsonl
from training.large_review_attention_v10 import (check_trainable, field_weights, make_loss,
                                                make_optimizer, summarize)
from training.runtime_attention_continuation_v10 import (check_decisions, install_head,
                                                       native_cached_features)
from training.matched_role_attention_inputs_v10 import aligned_teacher


def load_inputs(a):
    from training.matched_role_attention_inputs_v10 import completed_inputs, OPTIMIZATION, POPULATIONS
    from training.role_diversity_runtime_v10 import check_bindings
    if file_sha(a.plan) != a.plan_sha256:
        raise ValueError('Registered paired training plan differs')
    plan = json.loads(a.plan.read_text())
    if (plan['format'] != 'roguard-matched-role-attention-plan-1'
            or plan['source_sha256'] != file_sha(__file__)
            or plan['optimization'] != OPTIMIZATION or plan['populations'] != POPULATIONS
            or plan['initialization'] != 'selected95' or plan['held_out_test_opened'] is not False
            or plan['qualified'] is not False):
        raise ValueError('Registered source, scope or optimization differs')
    check_bindings(a.root, a.private_root, plan)
    destination = plan['outputs'][a.arm]
    if (a.study.resolve() != (a.root/destination['study']).resolve()
            or a.private.resolve() != (a.private_root/destination['private']).resolve()):
        raise ValueError('Matched arm output location differs')
    inputs, bindings = completed_inputs(a.root, a.private_root)
    if bindings != plan['input_bindings']:
        raise ValueError('Completed paired inputs differ from registration')
    return inputs, bindings, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'study', 'private', 'plan', 'root', 'private_root'):
        parser.add_argument('--'+name.replace('_', '-'), dest=name, type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--arm', choices=('control', 'treatment'), required=True)
    parser.add_argument('--preflight-report', type=Path)
    a = parser.parse_args()
    if a.study.exists() or a.private.exists():
        raise ValueError('Refusing to overwrite a paired attention experiment')
    inputs_by_language, bindings, plan = load_inputs(a)
    config = plan['optimization']; schedules = bindings['schedules']; counts = plan['populations']
    if a.preflight_report:
        record = dict(complete=True, source_sha256=file_sha(__file__), plan_sha256=file_sha(a.plan),
            arm=a.arm, bindings=bindings, schedules=schedules, populations=counts,
            encoder_loaded=False, training_launched=False, held_out_test_opened=False, qualified=False)
        a.preflight_report.parent.mkdir(parents=True, exist_ok=True)
        with a.preflight_report.open('x') as handle:
            json.dump(record, handle, indent=2); handle.write('\n')
        print(json.dumps(dict(complete=True, arm=a.arm, populations=counts, schedules=schedules))); return
    def disk_check():
        if shutil.disk_usage(a.private.parent).free < 8*1024**3:
            raise RuntimeError('Disk reserve reached')
    disk_check()
    a.study.mkdir(parents=True); a.private.mkdir(parents=True)
    import fcntl
    lock = (a.private/'writer.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = {'format': 'roguard-matched-role-attention-fit-1', 'arm': a.arm, 'source_sha256': file_sha(__file__),
                'plan_sha256': file_sha(a.plan), 'optimization': config,
                'bindings': bindings, 'schedules': schedules, 'populations': counts,
                'native_comparison': 'cached suffix with a copied fixed387-token prefix',
                'initialization': 'selected95 with training-only head dropout',
                'persistent_body_cache': False, 'held_out_test_opened': False, 'stable_release_ready': False}
    (a.study/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (a.study/'source.py').write_bytes(Path(__file__).read_bytes())
    (a.study/'helper.py').write_bytes((a.root/'training/matched_role_attention_inputs_v10.py').read_bytes())
    import numpy as np
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from safetensors.numpy import load_file, save_file
    from roguard.gemma4_attention import capture_with_prefix, replay
    from roguard.gemma4_contract import PREFIX_TOKENS, classifier_tokens
    from roguard.gemma4_review import _Engine, _score
    report = {'complete': False, 'languages': {}, 'held_out_test_opened': False,
              'stable_release_ready': False, 'arm': a.arm, 'training_rows': 32097, 'development_rows': 8837}
    started = time.monotonic()
    def persist():
        report['seconds'] = time.monotonic()-started
        temporary = a.study/'report.tmp'
        temporary.write_text(json.dumps(report, indent=2)+'\n'); temporary.replace(a.study/'report.json')
    for lang in ('ro', 'uk'):
        inputs = inputs_by_language[lang]
        rows = inputs[a.arm]
        write_jsonl(a.private/(lang+'-occurrences.jsonl'), [dict(slot=i, case_id=r['case_id'],
            partition=r['partition'], source=ref[0], source_index=ref[1], token_sha256=r['token_sha256'])
            for i, (r, ref) in enumerate(zip(rows, inputs['references'][a.arm]))])
        train = np.array([i for i, r in enumerate(rows) if r['partition'] == 'train'])
        dev = np.array([i for i, r in enumerate(rows) if r['partition'] == 'dev'])
        dev_rows = [rows[i] for i in dev]
        labels = np.array([r['fields'] for r in rows], dtype=np.int32)
        gold = np.array([r['gold'] for r in rows], dtype=np.int32)
        teacher = aligned_teacher(inputs, a.arm)
        initial_logits = inputs['initial_dev_logits']
        original_spec = inputs['specs']['v9']
        initial_spec = inputs['specs']['selected95']
        original = load_file(inputs['weights']['v9'])
        initial = load_file(inputs['weights']['selected95'])
        engine = _Engine(a.base, inputs['weights']['selected95'], file_sha(inputs['weights']['selected95']))
        model = engine.model
        install_head(model, initial, lang, config['head_dropout'][lang])
        names = check_trainable(model, lang)
        if sorted(names) != bindings['trainable_inventory'][lang] or len(names) != 76:
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
        def state_for_row(row):
            state = capture_with_prefix(model.base, mx.array([row['tokens'][len(PREFIX_TOKENS):]]), prefix, start=24)
            mx.eval(state)
            return state
        def state_for(i):
            return state_for_row(rows[i])
        def infer(indices, *, new_reference=False, check_full=False, stage='development'):
            model.eval(); values = []
            checked_groups = set()
            for position, i in enumerate(indices, 1):
                row = inputs['new'][i] if new_reference else rows[i]
                state = state_for_row(row)
                hidden = replay(model.base, state)[:, 0, :]
                z = model.heads[lang](hidden); mx.eval(hidden, z)
                if new_reference:
                    error = float(np.max(np.abs(np.asarray(hidden)[0]-inputs['new_initial_features'][i])))
                    record['maximum_initial_feature_error'] = max(record['maximum_initial_feature_error'], error)
                    if not np.isfinite(error) or error > 1e-3:
                        raise ValueError('Initial encoder differs from the selected95 new-input reference')
                if check_full and row['group'] not in checked_groups:
                    if classifier_tokens(engine.tokenizer, row['payload']) != row['tokens']:
                        raise ValueError('Actual runtime tokenizer differs from recovered tokens')
                    full = native_cached_features(model, mx.array([row['tokens'][len(PREFIX_TOKENS):]]), prefix); mx.eval(full)
                    error = float(np.max(np.abs(np.asarray(full)-np.asarray(hidden))))
                    record['maximum_full_replay_error'] = max(record['maximum_full_replay_error'], error)
                    if not np.isfinite(error) or error > 1e-3:
                        raise ValueError('Native cached encoder and streaming tail replay differ')
                    checked_groups.add(row['group'])
                values.append(np.asarray(z)[0].copy())
                del state, hidden, z; mx.clear_cache()
                if position % 100 == 0 or position == len(indices):
                    record['inference_progress'] = dict(stage=stage, checked=position, rows=len(indices))
                    persist()
                    print(json.dumps(dict(language=lang, inference=record['inference_progress'],
                                          seconds=time.monotonic()-started)), flush=True)
            return np.stack(values)
        # Verify the runtime tokenization and representation of every TRAIN source group.
        probe_by_group = {}
        for i in train:
            probe_by_group.setdefault(rows[i]['group'], i)
        infer(list(probe_by_group.values()), check_full=True, stage='train-source-probes')
        # Both arms repeat the same initial representation check on the new inputs.
        initial_new = infer(list(range(len(inputs['new']))), new_reference=True, check_full=True, stage='new-initial-reference')
        record['initial_new_parity'] = check_decisions(initial_new, inputs['new_initial_logits'], initial_spec)
        initial_new_path = a.private/(lang+'-initial-new-logits.npy')
        np.save(initial_new_path, initial_new, allow_pickle=False)
        record['initial_new_logits_sha256'] = file_sha(initial_new_path)
        del initial_new
        def evaluate(epoch):
            z = infer(dev, check_full=True, stage=f'epoch-{epoch}')
            if epoch == 0:
                parity = check_decisions(z, initial_logits, initial_spec)
                record['maximum_initial_logit_error'] = parity['maximum_logit_error']
                record['initial_runtime_decision_parity'] = parity
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
        computed_weights = field_weights(labels[train])
        record['field_weights'] = [v.tolist() for v in computed_weights]
        weights = [mx.array(v) for v in computed_weights]
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
        reloaded = infer(dev, check_full=True, stage='selected-reload')
        selected_spec = json.loads((a.study/(lang+'-review.json')).read_text())
        parity = check_decisions(reloaded, saved, selected_spec)
        reload_path = a.private/(lang+'-selected-reload-dev-logits.npy')
        np.save(reload_path, reloaded, allow_pickle=False)
        record.update(complete=True, selected_reload_maximum_logit_error=parity['maximum_logit_error'],
                      selected_reload_decision_parity=parity,
                      selected_reload_logits_sha256=file_sha(reload_path))
        persist()
        del engine, model, prefix, optimizer, rate_groups, value_grad, objective, selected, teacher
        gc.collect(); mx.clear_cache()
    _, final_bindings, final_plan = load_inputs(a)
    if final_bindings != bindings or final_plan != plan or file_sha(__file__) != plan['source_sha256']:
        raise ValueError('Source or inputs changed during the registered fit')
    report.update(complete=True, decision='REJECT' if any(
        r['selected']['binary_errors'] or r['selected']['joint_argmax_errors'] or r['selected']['introduced_v9']
        for r in report['languages'].values()) else 'DEVELOPMENT_PASS_REQUIRES_FULL_REGRESSION')
    persist()
    print(json.dumps({'complete': True, 'decision': report['decision'], 'seconds': report['seconds']}), flush=True)


if __name__ == '__main__':
    main()
