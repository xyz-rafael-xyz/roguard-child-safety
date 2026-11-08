"""Project the selected75 heads on audited runtime features without fitting."""
import argparse
import json
import os
from pathlib import Path
import shutil

from roguard.trajectory_data import file_sha, read_jsonl, write_jsonl
from eval.verify_runtime_review_baseline_v10 import argmaxes


def check_weights(original, candidate, language):
    import numpy as np
    heads = {f'heads.{language}.{layer}.{part}' for layer in ('first', 'last') for part in ('weight', 'bias')}
    if set(original) != set(candidate) or not heads <= set(candidate):
        raise ValueError('Warm-start tensor inventory differs')
    for name, value in candidate.items():
        if value.shape != original[name].shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError('Warm-start tensor shape or value differs')
        if name not in heads and not np.array_equal(value, original[name]):
            raise ValueError('Warm start modifies a frozen v9 tensor')


def compare_logits(left, right, spec, *, limit=None, exact_decisions=False):
    import numpy as np
    from roguard.gemma4_review import _score
    if (left.shape != right.shape or left.ndim != 2 or left.shape[1] != 19 or not len(left)
            or not np.isfinite(left).all() or not np.isfinite(right).all()):
        raise ValueError('Invalid comparison logits')
    error = float(np.max(np.abs(left-right)))
    if limit is not None and error > limit:
        raise ValueError('Head replay exceeds tolerance')
    argmax_changes = int(np.any(argmaxes(left) != argmaxes(right), axis=1).sum())
    decision_changes = sum((_score(a, spec) >= spec['threshold']) != (_score(b, spec) >= spec['threshold'])
                           for a,b in zip(left, right))
    if exact_decisions and (argmax_changes or decision_changes):
        raise ValueError('Reloaded head decisions differ')
    return dict(maximum_logit_error=error, argmax_rows_differing=argmax_changes,
                fixed_threshold_decision_differences=decision_changes)


def summarize_fixed(rows, logits, spec):
    import numpy as np
    from roguard.gemma4_review import _score
    if (len(rows) != len(logits) or logits.shape != (len(rows), 19)
            or any(r['partition'] != 'dev' for r in rows) or not np.isfinite(logits).all()):
        raise ValueError('Expected aligned development logits')
    fields = np.array([r['fields'] for r in rows]); predicted = argmaxes(logits)[:, :6]
    decisions = [int(_score(z, spec) >= spec['threshold']) for z in logits]
    return dict(rows=len(rows), binary_errors=sum(p != r['gold'] for p,r in zip(decisions, rows)),
        joint_argmax_errors=int(np.any(fields != predicted, axis=1).sum()),
        field_errors=(fields != predicted).sum(axis=0).tolist(), mode=spec['mode'], threshold=spec['threshold']), decisions


def project_rows(head, features, indices, mx, np, batch_size):
    pieces = []
    for start in range(0, len(indices), batch_size):
        z = head(mx.array(features[indices[start:start+batch_size]]))
        mx.eval(z); pieces.append(np.asarray(z).copy())
    return np.concatenate(pieces)


def load(a, *, ready):
    import numpy as np
    from safetensors.numpy import load_file
    plan = json.loads(a.plan.read_text())
    if plan['source_sha256'] != file_sha(__file__):
        raise ValueError('Registered source differs')
    for base, mapping in ((a.root, plan['public']), (a.private_root, plan['private'])):
        for name, expected in mapping.items():
            if file_sha(base/name) != expected:
                raise ValueError('Bound source or input differs: '+name)
    e = a.root/'research/v10-evidence'
    report75 = json.loads((e/'dropout-readout-75/report.json').read_text())
    proof75 = json.loads((e/'dropout-readout-qualification-75/verification.json').read_text())
    if (report75.get('complete') is not True or proof75.get('verified') is not True
            or proof75['report_sha256'] != file_sha(e/'dropout-readout-75/report.json')
            or proof75['checkpoints_replayed'] != 78 or proof75['development_predictions_replayed'] != 344643):
        raise ValueError('Selected75 verification differs')
    for lang, expected in plan['languages'].items():
        path = a.private_root/'dropout-readout-75'/(lang+'-selected.safetensors')
        original_path = a.root/'research/v9-stabilization/reported-tail-fit-196/actual'/(lang+'-selected.safetensors')
        if file_sha(path) != expected['selected_weights_sha256'] or file_sha(original_path) != expected['v9_weights_sha256']:
            raise ValueError('Registered initialization differs')
        if report75['languages'][lang]['selected']['adapter_sha256'] != file_sha(path):
            raise ValueError('Selected75 report does not bind initialization')
        check_weights(load_file(original_path), load_file(path), lang)
        spec = json.loads((e/'dropout-readout-75'/(lang+'-review.json')).read_text())
        original_spec = json.loads((e/'crossed-review-features-62/v9-binding'/(lang+'-review.json')).read_text())
        excluded = {'adapter_sha256', 'mode', 'threshold'}
        if (spec['adapter_sha256'] != file_sha(path)
                or {k:v for k,v in spec.items() if k not in excluded} != {k:v for k,v in original_spec.items() if k not in excluded}):
            raise ValueError('Warm-start runtime contract differs')
        old_logits = np.load(a.private_root/'dropout-readout-75'/(lang+'-selected-dev-logits.npy'), allow_pickle=False)
        if old_logits.shape != (expected['development_rows'], 19) or not np.isfinite(old_logits).all():
            raise ValueError('Selected75 reference logits differ')
        metadata = read_jsonl(a.private_root/'crossed-review-features-62'/(lang+'-rows.jsonl'))
        dev_rows = [r for r in metadata if r['partition'] == 'dev']
        previous, _ = summarize_fixed(dev_rows, old_logits, spec)
        selected = report75['languages'][lang]['selected']
        if any(previous[k] != selected[k] for k in ('binary_errors', 'joint_argmax_errors', 'field_errors')):
            raise ValueError('Selected75 fixed-threshold reference metrics differ')
    bindings = dict(public=plan['public'], private=plan['private'], languages=plan['languages'], baseline_verified=False)
    if not ready:
        return plan, bindings
    study = e/'runtime-review-baseline-88'; data = a.private_root/'runtime-review-baseline-88'
    report = json.loads((study/'report.json').read_text()); protocol = json.loads((study/'protocol.json').read_text())
    from eval.verify_runtime_review_baseline_v10 import require_complete
    require_complete(report)
    proof_path = a.private_root/'runtime-review-baseline88-verification-for89.json'
    proof = json.loads(proof_path.read_text())
    queue_path = a.private_root/'runtime-review-baseline-audit-sequence-89-plan.json'
    queue = json.loads(queue_path.read_text())
    sequence_path = e/'runtime-review-baseline-audit-sequence-89/report.json'
    sequence = json.loads(sequence_path.read_text())
    if (sequence.get('execution_complete') is not True or sequence.get('phase') != 'finished'
            or sequence['plan_sha256'] != file_sha(queue_path) or sequence['source_hashes'] != queue['source_hashes']
            or [s['name'] for s in sequence['completed_steps']] != [s['name'] for s in queue['steps']]
            or any(s['exit_code'] != 0 for s in sequence['completed_steps'])):
        raise ValueError('Baseline audit sequence is incomplete or changed')
    if (proof.get('verification') != 'PASS' or proof.get('encoder_loaded') is not False
            or proof.get('qualified') is not False
            or proof['source_sha256'] != file_sha(a.root/'eval/verify_runtime_review_baseline_v10.py')
            or proof['report_sha256'] != file_sha(study/'report.json')
            or proof['protocol_sha256'] != file_sha(study/'protocol.json')
            or proof['amendment_sha256'] != file_sha(e/'protocol-amendment-73.json')
            or proof['original_input_bindings'] != protocol['bindings']):
        raise ValueError('Completed baseline proof differs')
    actual_artifacts = {}
    for lang, expected in plan['languages'].items():
        record = report['languages'][lang]; audited = proof['languages'][lang]
        suffixes = ('features.npy', 'logits.npy', 'warm-start-dev-logits.npy', 'rows.jsonl', 'observations.jsonl')
        if set(record['artifacts']) != {lang+'-'+s for s in suffixes}:
            raise ValueError('Baseline language artifact inventory differs')
        if (record.get('complete') is not True or audited.get('complete') is not True
                or record['checked'] != expected['rows'] or audited['checked'] != expected['rows']
                or audited['development_checked'] != expected['development_rows']
                or audited['cpu_head_replay']['v9']['rows'] != expected['rows']
                or audited['cpu_head_replay']['warm_start']['rows'] != expected['development_rows']):
            raise ValueError('Audited baseline population differs')
        for name, checksum in record['artifacts'].items():
            if Path(name).name != name or file_sha(data/name) != checksum or proof['artifacts'].get(name) != checksum:
                raise ValueError('Audited baseline artifact differs')
            actual_artifacts[name] = checksum
        rows = read_jsonl(data/(lang+'-rows.jsonl'))
        if (rows != read_jsonl(a.private_root/'crossed-review-features-62'/(lang+'-rows.jsonl'))
                or len(rows) != expected['rows'] or sum(r['partition'] == 'dev' for r in rows) != expected['development_rows']):
            raise ValueError('Baseline metadata differs')
    if actual_artifacts != proof['artifacts'] or len(actual_artifacts) != 10:
        raise ValueError('Baseline artifact inventory differs')
    bindings.update(baseline_verified=True, baseline_report_sha256=file_sha(study/'report.json'),
        baseline_protocol_sha256=file_sha(study/'protocol.json'), baseline_verification_sha256=file_sha(proof_path),
        baseline_audit_sequence_sha256=file_sha(sequence_path), baseline_artifacts=actual_artifacts)
    return plan, bindings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'private_root', 'study', 'private', 'plan'):
        p.add_argument('--'+name.replace('_', '-'), dest=name, type=Path, required=True)
    p.add_argument('--preflight-report', type=Path)
    a = p.parse_args()
    if a.study.exists() or a.private.exists():
        raise ValueError('Refusing to overwrite warm-start preparation')
    plan, bindings = load(a, ready=not bool(a.preflight_report))
    if a.preflight_report:
        if a.preflight_report.exists(): raise ValueError('Refusing to overwrite preflight')
        a.preflight_report.parent.mkdir(parents=True, exist_ok=True)
        a.preflight_report.write_text(json.dumps(dict(complete=True, source_sha256=file_sha(__file__),
            plan_sha256=file_sha(a.plan), bindings=bindings, encoder_loaded=False,
            training_performed=False, gpu_inference_performed=False, qualified=False), indent=2)+'\n')
        return
    try: os.kill(plan['dependency_pid'], 0)
    except ProcessLookupError: pass
    else: raise ValueError('Baseline audit queue has not exited')
    if shutil.disk_usage(a.private_root).free < 8*1024**3: raise ValueError('Disk reserve reached')
    import numpy as np
    import mlx.core as mx
    from safetensors.numpy import load_file
    from training.shared_tail_training_v9 import make_head
    from eval.verify_matched_assertion_readout_v10 import paired_counts
    if not mx.metal.is_available(): raise ValueError('Registered GPU head projection requires Metal')
    a.study.mkdir(parents=True); a.private.mkdir(parents=True)
    protocol = dict(format='roguard-runtime-warm-start-1', source_sha256=file_sha(__file__),
        plan_sha256=file_sha(a.plan), bindings=bindings, initialization='selected75',
        gpu_batch_size=1, cpu_batch_size=256, cpu_tolerance=1e-3, reload_tolerance=1e-5,
        encoder_loaded=False, training_performed=False, held_out_test_opened=False, qualified=False)
    (a.study/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (a.study/'source.py').write_bytes(Path(__file__).read_bytes())
    report = dict(complete=False, languages={}, encoder_loaded=False, training_performed=False,
                  held_out_test_opened=False, qualified=False)
    mapping = {'0.weight':'first.weight', '0.bias':'first.bias', '3.weight':'last.weight', '3.bias':'last.bias'}
    for lang in ('ro', 'uk'):
        weights_path = a.private_root/'dropout-readout-75'/(lang+'-selected.safetensors')
        destination = a.private/(lang+'-selected.safetensors'); shutil.copyfile(weights_path, destination)
        spec_path = a.root/'research/v10-evidence/dropout-readout-75'/(lang+'-review.json')
        shutil.copyfile(spec_path, a.study/(lang+'-review.json')); spec = json.loads(spec_path.read_text())
        metadata = read_jsonl(a.private_root/'runtime-review-baseline-88'/(lang+'-rows.jsonl'))
        indices = np.array([i for i,r in enumerate(metadata) if r['partition'] == 'dev']); rows = [metadata[i] for i in indices]
        x = np.load(a.private_root/'runtime-review-baseline-88'/(lang+'-features.npy'), mmap_mode='r', allow_pickle=False)

        def project(device, batch):
            mx.set_default_device(device)
            tensors = load_file(destination)
            head = make_head({k:tensors['heads.'+lang+'.'+v] for k,v in mapping.items()}); head.eval()
            return project_rows(head, x, indices, mx, np, batch)

        gpu = project(mx.gpu, 1); mx.clear_cache()
        cpu = project(mx.cpu, 256); mx.clear_cache()
        reloaded = project(mx.gpu, 1); mx.clear_cache()
        cpu_check = compare_logits(cpu, gpu, spec, limit=1e-3)
        reload_check = compare_logits(reloaded, gpu, spec, limit=1e-5, exact_decisions=True)
        previous = np.load(a.private_root/'dropout-readout-75'/(lang+'-selected-dev-logits.npy'), allow_pickle=False)
        old_summary, old_decisions = summarize_fixed(rows, previous, spec)
        fresh_summary, fresh_decisions = summarize_fixed(rows, gpu, spec)
        initial71 = np.load(a.private_root/'runtime-review-baseline-88'/(lang+'-warm-start-dev-logits.npy'), allow_pickle=False)
        spec71 = json.loads((a.root/'research/v10-evidence/policy-consistent-readout-71'/(lang+'-review.json')).read_text())
        summary71, decisions71 = summarize_fixed(rows, initial71, spec71)
        for name, values in (('selected-dev-logits', gpu), ('cpu-dev-logits', cpu)):
            np.save(a.private/(lang+'-'+name+'.npy'), values, allow_pickle=False)
        write_jsonl(a.private/(lang+'-dev-rows.jsonl'), rows)
        gold = [r['gold'] for r in rows]
        report['languages'][lang] = dict(original75=old_summary, runtime75=fresh_summary, runtime71=summary71,
            cpu_replay=cpu_check, gpu_reload=reload_check, historical_logit_difference=compare_logits(gpu, previous, spec),
            runtime75_vs_original75=paired_counts(gold, old_decisions, fresh_decisions),
            runtime75_vs_runtime71=paired_counts(gold, decisions71, fresh_decisions),
            artifacts={path.name:file_sha(path) for path in sorted(a.private.glob(lang+'-*'))})
        tmp = a.study/'report.tmp'; tmp.write_text(json.dumps(report, indent=2)+'\n'); tmp.replace(a.study/'report.json')
        del x, gpu, cpu, reloaded, previous, initial71; mx.clear_cache()
    if load(a, ready=True)[1] != bindings: raise ValueError('Preparation inputs changed')
    report.update(complete=True, decision='WARM_START_REFERENCE_ONLY_NO_MODEL_QUALIFICATION')
    tmp = a.study/'report.tmp'; tmp.write_text(json.dumps(report, indent=2)+'\n'); tmp.replace(a.study/'report.json')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
