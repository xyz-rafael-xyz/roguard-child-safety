"""Runtime-reference binding and training-only dropout for the next attention fit."""
from copy import deepcopy
import json
from types import SimpleNamespace

from roguard.trajectory_data import file_sha, read_jsonl


def install_head(model, tensors, language, probability):
    """Keep the exported tensor names; dropout adds no persistent parameters."""
    from training.dropout_readout_v10 import make_head
    mapping = {'0.weight': 'first.weight', '0.bias': 'first.bias',
               '3.weight': 'last.weight', '3.bias': 'last.bias'}
    model.heads[language] = make_head({key: tensors['heads.'+language+'.'+value]
                                     for key, value in mapping.items()}, probability)
    model.heads['uk' if language == 'ro' else 'ro'].freeze()
    model.eval()


def native_cached_features(model, suffix, prefix):
    """Use the serving cache path, with independent mutable cache state per call."""
    return model.base.model(suffix, cache=deepcopy(prefix))[:, -1, :]


def check_decisions(actual, expected, spec):
    from training.prepare_runtime_warm_start_v10 import compare_logits
    return compare_logits(actual, expected, spec, limit=1e-5, exact_decisions=True)


def runtime_inputs(root, private_root):
    """Revalidate completed88/89 and bind completed90 before any encoder import.

    CPU outputs are independently retained in90. Its GPU reload measurements
    remain producer observations until epoch-zero inference repeats them.
    """
    import numpy as np
    from training.prepare_runtime_warm_start_v10 import load, summarize_fixed, compare_logits
    from eval.verify_matched_assertion_readout_v10 import paired_counts
    evidence = root/'research/v10-evidence'
    warm = evidence/'runtime-warm-start-90'; data = private_root/'runtime-warm-start-90'
    plan_path = evidence/'protocol-amendment-74.json'
    _, expected = load(SimpleNamespace(root=root, private_root=private_root, plan=plan_path), ready=True)
    report_path = warm/'report.json'; protocol_path = warm/'protocol.json'
    report = json.loads(report_path.read_text()); protocol = json.loads(protocol_path.read_text())
    if (report.get('complete') is not True
            or report.get('decision') != 'WARM_START_REFERENCE_ONLY_NO_MODEL_QUALIFICATION'
            or set(report['languages']) != {'ro', 'uk'}
            or protocol['source_sha256'] != file_sha(root/'training/prepare_runtime_warm_start_v10.py')
            or file_sha(warm/'source.py') != protocol['source_sha256']
            or protocol['plan_sha256'] != file_sha(plan_path)
            or protocol['bindings'] != expected or protocol['initialization'] != 'selected75'):
        raise ValueError('Completed runtime warm-start binding differs')
    queue_path = private_root/'runtime-warm-start-sequence-90-plan.json'
    sequence_path = evidence/'runtime-warm-start-sequence-90/report.json'
    queue = json.loads(queue_path.read_text()); sequence = json.loads(sequence_path.read_text())
    if (sequence.get('execution_complete') is not True or sequence.get('phase') != 'finished'
            or sequence['plan_sha256'] != file_sha(queue_path)
            or sequence['source_hashes'] != queue['source_hashes']
            or not sequence['completed_steps']
            or [s['name'] for s in sequence['completed_steps']] != [s['name'] for s in queue['steps']]
            or any(s['exit_code'] != 0 for s in sequence['completed_steps'])):
        raise ValueError('Warm-start preparation sequence is incomplete or changed')
    for name, checksum in queue['source_hashes'].items():
        if file_sha(root/name) != checksum:
            raise ValueError('Warm-start queue source changed')
    for record in (report, protocol):
        if any(record.get(name) is not False for name in
               ('encoder_loaded', 'training_performed', 'held_out_test_opened', 'qualified')):
            raise ValueError('Warm-start scope differs')
    artifacts = {}; specifications = {}
    for lang in ('ro', 'uk'):
        record = report['languages'][lang]
        suffixes = ('selected.safetensors', 'selected-dev-logits.npy', 'cpu-dev-logits.npy', 'dev-rows.jsonl')
        names = {lang+'-'+suffix for suffix in suffixes}
        if set(record['artifacts']) != names:
            raise ValueError('Warm-start artifact inventory differs')
        for name, checksum in record['artifacts'].items():
            if file_sha(data/name) != checksum:
                raise ValueError('Warm-start artifact changed')
            artifacts[name] = checksum
        old_data = private_root/'dropout-readout-75'
        old_study = evidence/'dropout-readout-75'
        spec_path = warm/(lang+'-review.json')
        if (file_sha(data/(lang+'-selected.safetensors')) != file_sha(old_data/(lang+'-selected.safetensors'))
                or file_sha(spec_path) != file_sha(old_study/(lang+'-review.json'))):
            raise ValueError('Warm start differs from the fixed selected75 initialization')
        spec = json.loads(spec_path.read_text()); specifications[lang] = file_sha(spec_path)
        rows = read_jsonl(data/(lang+'-dev-rows.jsonl'))
        baseline_rows = read_jsonl(private_root/'runtime-review-baseline-88'/(lang+'-rows.jsonl'))
        if rows != [r for r in baseline_rows if r['partition'] == 'dev']:
            raise ValueError('Warm-start development alignment changed')
        gpu = np.load(data/(lang+'-selected-dev-logits.npy'), allow_pickle=False)
        cpu = np.load(data/(lang+'-cpu-dev-logits.npy'), allow_pickle=False)
        old = np.load(old_data/(lang+'-selected-dev-logits.npy'), allow_pickle=False)
        initial71 = np.load(private_root/'runtime-review-baseline-88'/(lang+'-warm-start-dev-logits.npy'), allow_pickle=False)
        spec71 = json.loads((evidence/'policy-consistent-readout-71'/(lang+'-review.json')).read_text())
        # Historical study75 retained float64 logits. Preserve their original
        # precision; fresh runtime references remain strictly float32.
        for label, values, dtype in (('gpu', gpu, np.float32), ('cpu', cpu, np.float32),
                                     ('historical75', old, np.float64), ('initial71', initial71, np.float32)):
            if values.dtype != dtype or values.shape != (len(rows), 19) or not np.isfinite(values).all():
                raise ValueError('Invalid runtime initialization logits: '+label)
        fresh_summary, fresh_decisions = summarize_fixed(rows, gpu, spec)
        old_summary, old_decisions = summarize_fixed(rows, old, spec)
        summary71, decisions71 = summarize_fixed(rows, initial71, spec71)
        gold = [r['gold'] for r in rows]
        recounted = dict(runtime75=fresh_summary, original75=old_summary, runtime71=summary71,
            cpu_replay=compare_logits(cpu, gpu, spec, limit=1e-3),
            historical_logit_difference=compare_logits(gpu, old, spec),
            runtime75_vs_original75=paired_counts(gold, old_decisions, fresh_decisions),
            runtime75_vs_runtime71=paired_counts(gold, decisions71, fresh_decisions))
        if any(record[name] != value for name, value in recounted.items()):
            raise ValueError('Warm-start metrics differ from retained outputs')
        reload = record['gpu_reload']
        if (not 0 <= reload['maximum_logit_error'] <= 1e-5
                or reload['argmax_rows_differing'] != 0
                or reload['fixed_threshold_decision_differences'] != 0):
            raise ValueError('Reported GPU warm-start reload failed')
    return dict(baseline=expected, warm_report_sha256=file_sha(report_path),
                warm_protocol_sha256=file_sha(protocol_path), warm_artifacts=artifacts,
                warm_specifications=specifications, warm_sequence_sha256=file_sha(sequence_path),
                warm_sequence_plan_sha256=file_sha(queue_path))
