"""Run the complete paired audit on a production-size synthetic artifact fixture."""
from collections import Counter
import json
from pathlib import Path
import sys

import pytest


def test_complete_paired_audit_and_late_corruption(tmp_path, monkeypatch):
    np = pytest.importorskip('numpy')
    from safetensors.numpy import save_file
    from roguard.assurance import digest
    from roguard.trajectory_data import file_sha, read_jsonl, write_jsonl
    from training.large_review_attention_v10 import field_weights, summarize
    from training.matched_role_attention_inputs_v10 import OPTIMIZATION, POPULATIONS
    from training.register_matched_role_attention_v10 import OUTPUTS
    from eval.verify_runtime_review_attention_continuation_v10 import parity
    from eval import verify_matched_role_attention_v10 as auditor
    actual_root = Path(auditor.__file__).resolve().parents[1]
    root = tmp_path/'repo'; private = tmp_path/'private'; private.mkdir()
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2)+'\n')
    source_names = ['eval/verify_matched_role_attention_v10.py', 'eval/audit_attention_development_v10.py',
                   'eval/verify_runtime_review_attention_continuation_v10.py',
                   'training/train_matched_role_attention_v10.py', 'training/matched_role_attention_inputs_v10.py']
    for name in source_names:
        (root/name).parent.mkdir(parents=True, exist_ok=True); (root/name).write_bytes((actual_root/name).read_bytes())
    sources = {name: file_sha(root/name) for name in source_names}
    def logits(rows):
        z = np.zeros((len(rows), 19), np.float32)
        for j, offset in enumerate([0, 3, 6, 10, 13, 15]):
            z[np.arange(len(rows)), offset+np.array([r['fields'][j] for r in rows])] = 2
        z[np.arange(len(rows)), 17+np.array([r['gold'] for r in rows])] = 2
        return z
    inputs = {}; schedules = {}; reports = {}; events_by_arm = {}
    for lang, count in POPULATIONS.items():
        def row(i, split):
            return dict(case_id=f'{lang}-{split}-{i}', language=lang, partition=split, group='fixture',
                fields=[0, 0, 0, 0, 0, i % 2], gold=i % 2, token_sha256=digest([lang, split, i]))
        base = [row(i, 'train') for i in range(count['base_train'])]+[row(i, 'dev') for i in range(count['dev'])]
        new = [row(count['base_train']+i, 'train') for i in range(1584)]
        pools = {}
        for i, r in enumerate(base):
            if r['partition'] == 'train': pools.setdefault((tuple(r['fields']), r['gold']), []).append(i)
        for key, pool in pools.items():
            pool.sort(key=lambda i: (digest([1073, lang, key, base[i]['case_id']]), base[i]['case_id']))
        used = Counter(); extra = []
        for r in new:
            key = tuple(r['fields']), r['gold']; extra.append(pools[key][used[key] % len(pools[key])]); used[key] += 1
        control = base+[base[i] for i in extra]; treatment = base+new
        refs = dict(control=[('base', i) for i in range(len(base))]+[('base', i) for i in extra],
                    treatment=[('base', i) for i in range(len(base))]+[('new', i) for i in range(len(new))])
        trainable = {f'base.model.layers.{i}.self_attn.{p}.lora_{part}'
                     for i in range(24, 42) for p in ('q_proj', 'o_proj') for part in ('a', 'b')}
        names = trainable | {f'heads.{language}.{layer}.{part}' for language in ('ro', 'uk')
                            for layer in ('first', 'last') for part in ('weight', 'bias')}
        original = {name: np.zeros((2, 2), np.float32) for name in names}
        initial = {name: v.copy() for name, v in original.items()}
        initial[f'heads.{lang}.first.bias'] += .01
        initial['base.model.layers.24.self_attn.q_proj.lora_b'] += .005
        v9_path = private/(lang+'-original.safetensors'); initial_path = private/(lang+'-initial.safetensors')
        save_file(original, str(v9_path)); save_file(initial, str(initial_path))
        spec = dict(mode='binary', threshold=.5, couplings=dict(scope=[[1.]*3]*3, assertion_relation=[[1.]*2]*4))
        dev = [r for r in base if r['partition'] == 'dev']
        inputs[lang] = dict(base=base, new=new, control=control, treatment=treatment, references=refs,
            base_teacher=np.zeros((len(base), 19), np.float32), new_teacher=np.zeros((1584, 19), np.float32),
            initial_dev_logits=logits(dev), new_initial_logits=logits(new),
            weights={'v9': v9_path, 'selected95': initial_path},
            specs={'v9': dict(spec, adapter_sha256=file_sha(v9_path)),
                   'selected95': dict(spec, adapter_sha256=file_sha(initial_path))}, initial_tensors=initial)
        per = (count['train']+7)//8
        schedules[lang] = dict(updates_per_epoch=per, total_updates=3*per, total_presentations=3*count['train'])
    bindings = dict(schedules=schedules, fixture=True)
    plan = dict(format='roguard-matched-role-attention-plan-1', source_sha256=sources['training/train_matched_role_attention_v10.py'],
        optimization=OPTIMIZATION, populations=POPULATIONS, initialization='selected95', training_admitted=True,
        held_out_test_opened=False, qualified=False, arms=['control', 'treatment'], outputs=OUTPUTS,
        input_bindings=bindings, public=sources, private={})
    plan_path = root/'plan104.json'; write(plan_path, plan)
    for arm in ('control', 'treatment'):
        study = root/OUTPUTS[arm]['study']; weights = private/OUTPUTS[arm]['private']; weights.mkdir()
        protocol = dict(format='roguard-matched-role-attention-fit-1', arm=arm, source_sha256=plan['source_sha256'],
            plan_sha256=file_sha(plan_path), optimization=OPTIMIZATION, bindings=bindings, schedules=schedules,
            populations=POPULATIONS, native_comparison='cached suffix with a copied fixed387-token prefix',
            initialization='selected95 with training-only head dropout', persistent_body_cache=False,
            held_out_test_opened=False, stable_release_ready=False)
        write(study/'protocol.json', protocol)
        (study/'source.py').write_bytes((root/'training/train_matched_role_attention_v10.py').read_bytes())
        (study/'helper.py').write_bytes((root/'training/matched_role_attention_inputs_v10.py').read_bytes())
        report = dict(complete=True, arm=arm, languages={}, training_rows=32097, development_rows=8837,
                      held_out_test_opened=False, stable_release_ready=False, decision='DEVELOPMENT_PASS_REQUIRES_FULL_REGRESSION')
        events = []
        def emit(value): events.append(dict(value, seconds=float(len(events)+1)))
        def progress(lang, stage, n):
            for checked in list(range(100, n, 100))+[n]:
                emit(dict(language=lang, inference=dict(stage=stage, checked=checked, rows=n)))
        for lang, current in inputs.items():
            rows = current[arm]; dev = [r for r in rows if r['partition'] == 'dev']; n = POPULATIONS[lang]['train']
            z = current['initial_dev_logits']; spec = current['specs']['v9']; initial_spec = current['specs']['selected95']
            write_jsonl(weights/(lang+'-occurrences.jsonl'), [dict(slot=i, case_id=r['case_id'], partition=r['partition'],
                source=ref[0], source_index=ref[1], token_sha256=r['token_sha256']) for i, (r, ref) in enumerate(zip(rows, current['references'][arm]))])
            native = weights/(lang+'-initial-new-logits.npy'); np.save(native, current['new_initial_logits'])
            record = dict(complete=True, training_rows=n, development_rows=len(dev), trainable_tensors=76, checkpoints=[],
                maximum_initial_feature_error=0., maximum_initial_logit_error=0., maximum_full_replay_error=0.,
                selected_reload_maximum_logit_error=0., initial_runtime_decision_parity=parity(z, z, initial_spec),
                initial_new_parity=parity(current['new_initial_logits'], current['new_initial_logits'], initial_spec),
                initial_new_logits_sha256=file_sha(native),
                field_weights=[v.tolist() for v in field_weights(np.array([r['fields'] for r in rows if r['partition'] == 'train']))])
            progress(lang, 'train-source-probes', 1); progress(lang, 'new-initial-reference', 1584)
            for epoch in range(4):
                tensor = {name: value.copy() for name, value in current['initial_tensors'].items()}
                if epoch:
                    tensor[f'heads.{lang}.first.bias'] += .01*epoch
                    tensor['base.model.layers.24.self_attn.q_proj.lora_b'] += .01*epoch
                folder = weights/lang; folder.mkdir(exist_ok=True)
                w = folder/f'epoch-{epoch}.safetensors'; save_file(tensor, str(w))
                values = folder/f'epoch-{epoch}-dev-logits.npy'; np.save(values, z)
                summary, predictions = summarize(dev, z, spec['couplings'], np.ones(len(dev), dtype=int))
                summary.update(epoch=epoch, adapter_sha256=file_sha(w), development_logits_sha256=file_sha(values),
                    completed_training_updates=epoch*schedules[lang]['updates_per_epoch'], completed_training_presentations=epoch*n)
                record['checkpoints'].append(summary)
                if epoch == 0:
                    record['selected'] = dict(summary)
                    (weights/(lang+'-selected.safetensors')).write_bytes(w.read_bytes())
                    np.save(weights/(lang+'-selected-dev-logits.npy'), z)
                    selected_spec = dict(spec, adapter_sha256=file_sha(w), mode=summary['mode'], threshold=summary['threshold'])
                    write(study/(lang+'-review.json'), selected_spec)
                    write_jsonl(study/(lang+'-development-predictions.jsonl'), predictions)
                if epoch:
                    per = schedules[lang]['updates_per_epoch']
                    for step in range((epoch-1)*per+1, epoch*per+1):
                        if step % 25 == 0 or step == epoch*per:
                            emit(dict(language=lang, epoch=epoch, step=step, total_steps=3*per,
                                presentations=(epoch-1)*n+min((step-(epoch-1)*per)*8, n), loss=.1))
                progress(lang, f'epoch-{epoch}', len(dev))
                emit(dict(language=lang, epoch=epoch, selected_epoch=0,
                    **{k: summary[k] for k in ('binary_errors', 'joint_argmax_errors', 'introduced_v9')}))
            reload = weights/(lang+'-selected-reload-dev-logits.npy'); np.save(reload, z)
            record['selected_reload_logits_sha256'] = file_sha(reload)
            record['selected_reload_decision_parity'] = parity(z, z, selected_spec)
            progress(lang, 'selected-reload', len(dev)); report['languages'][lang] = record
        emit(dict(complete=True, decision=report['decision'])); report['seconds'] = events[-1]['seconds']
        write(study/'report.json', report); reports[arm] = report; events_by_arm[arm] = events
    queue_path = private/'queue.json'; sequence_path = root/'sequence.json'
    queue = dict(source_hashes=sources, steps=[dict(name=arm) for arm in ('control', 'treatment')]); write(queue_path, queue)
    write(sequence_path, dict(execution_complete=True, phase='finished', plan_sha256=file_sha(queue_path),
        source_hashes=sources, completed_steps=[dict(name=arm, exit_code=0) for arm in ('control', 'treatment')]))
    raw = b''.join((json.dumps(dict(starting=arm))+'\n'+''.join(json.dumps(e)+'\n' for e in events_by_arm[arm])).encode()
                   for arm in ('control', 'treatment'))
    (private/'train.log').write_bytes(raw)
    audit = dict(format='roguard-matched-role-attention-audit-plan-1', training_plan='plan104.json',
        training_plan_sha256=file_sha(plan_path), sequence='sequence.json', queue_plan='queue.json', log='train.log',
        public=sources, private={'queue.json': file_sha(queue_path)}, markers={arm: arm for arm in ('control', 'treatment')})
    audit_path = root/'audit107.json'; write(audit_path, audit); output = private/'verification.json'
    monkeypatch.setattr('training.matched_role_attention_inputs_v10.completed_inputs', lambda *args: (inputs, bindings))
    monkeypatch.setattr('scripts.wait_completed_sequence_exit_v10.require_exit', lambda *args: None)
    args = ['audit', '--root', str(root), '--private-root', str(private), '--audit-plan', str(audit_path),
            '--audit-plan-sha256', file_sha(audit_path), '--producer-pid', '123', '--output', str(output)]
    monkeypatch.setattr(sys, 'argv', args); auditor.main()
    proof = json.loads(output.read_text())
    assert proof['verification'] == 'PASS' and proof['model_qualified'] is False
    assert proof['checkpoint_count'] == 16 and proof['checkpoint_dev_rows_recounted'] == 70696
    assert proof['reload_rows_recounted'] == 17674 and proof['initial_new_rows_compared'] == 6336
    assert proof['development_pass_arms'] == ['control', 'treatment']
    assert [v['log']['events'] for v in proof['arms'].values()] == [980, 980]
    assert sum(len(language['artifacts']) for arm in proof['arms'].values() for language in arm['languages'].values()) == 52
    extra = private/OUTPUTS['control']['private']/'ro/epoch-4-dev-logits.npy'
    np.save(extra, inputs['ro']['initial_dev_logits'])
    args[-1] = str(private/'must-not-exist.json')
    with pytest.raises(ValueError, match='artifact inventory'):
        auditor.main()
    assert not (private/'must-not-exist.json').exists()
    extra.unlink()
    path = root/OUTPUTS['treatment']['study']/'uk-development-predictions.jsonl'
    predictions = read_jsonl(path); predictions[-1]['case_id'] = 'corrupted-final-identity'; write_jsonl(path, predictions)
    args[-1] = str(private/'must-not-exist.json')
    with pytest.raises(ValueError, match='identities'):
        auditor.main()
    assert not (private/'must-not-exist.json').exists()
