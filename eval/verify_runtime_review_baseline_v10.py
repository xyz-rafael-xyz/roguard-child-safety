"""Audit completed runtime references, retained observations and both heads on CPU."""
import argparse
import json
import math
import os
from pathlib import Path
import time

from roguard.trajectory_data import file_sha, read_jsonl


def require_complete(report):
    if (report.get('complete') is not True
            or report.get('decision') != 'REFERENCE_EXTRACTION_ONLY_NO_MODEL_QUALIFICATION'
            or report.get('training_performed') is not False
            or report.get('held_out_test_opened') is not False or report.get('qualified') is not False
            or set(report.get('languages', {})) != {'ro', 'uk'}):
        raise ValueError('Runtime reference extraction is incomplete or has invalid scope')


def argmaxes(values):
    import numpy as np
    return np.column_stack([part.argmax(axis=1) for part in
                            np.split(values, [3, 6, 10, 13, 15, 17], axis=1)])


def check_population(rows, observations, x, z, warm, reference, old_x, old_z, spec, warm_spec, reported):
    """Recount array-backed metrics; native differences remain recorded observations."""
    import numpy as np
    from roguard.gemma4_review import _score
    n = len(rows); dev = [i for i, row in enumerate(rows) if row['partition'] == 'dev']
    if not n or not dev or any(row['partition'] not in ('train', 'dev') for row in rows):
        raise ValueError('Invalid reference population')
    for value, shape in ((x, (n, 2560)), (z, (n, 19)), (warm, (len(dev), 19))):
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError('Invalid retained reference array')
    if (old_x.shape != x.shape or old_z.shape != z.shape or reference.shape != warm.shape
            or not all(np.isfinite(v).all() for v in (old_x, old_z, reference))):
        raise ValueError('Invalid historical reference array')
    if len(observations) != n:
        raise ValueError('Observation inventory differs')
    max_native = max_historical = 0.; differences = changes = 0
    before = []; after = []
    for i, (row, obs) in enumerate(zip(rows, observations)):
        if (type(obs['feature_index']) is not int or obs['feature_index'] != i
                or obs['case_id'] != row['case_id'] or obs['partition'] != row['partition']):
            raise ValueError('Observation identity or order differs')
        native = obs['native_replay_error']
        if not math.isfinite(native) or not 0 <= native <= 1e-3:
            raise ValueError('Native replay observation exceeds tolerance')
        historical = float(np.max(np.abs(x[i]-old_x[i])))
        a = int(_score(old_z[i], spec) >= spec['threshold'])
        b = int(_score(z[i], spec) >= spec['threshold'])
        if obs['historical_feature_difference'] != historical or obs['historical_binary_decision_change'] != int(a != b):
            raise ValueError('Historical difference observation disagrees with arrays')
        max_native = max(max_native, native); max_historical = max(max_historical, historical)
        differences += int(historical > 1e-3); changes += int(a != b)
        before.append(a); after.append(b)
    warm_error = float(np.max(np.abs(warm-reference)))
    if warm_error > 1e-5 or not np.array_equal(argmaxes(warm), argmaxes(reference)):
        raise ValueError('Warm-start runtime reference differs')
    expected_decisions = [int(_score(v, warm_spec) >= warm_spec['threshold']) for v in reference]
    actual_decisions = [int(_score(v, warm_spec) >= warm_spec['threshold']) for v in warm]
    if actual_decisions != expected_decisions:
        raise ValueError('Warm-start fixed-threshold decisions differ')
    expected = dict(complete=True, rows=n, checked=n, development_checked=len(dev),
        maximum_native_replay_error=max_native, maximum_warm_start_logit_error=warm_error,
        maximum_historical_feature_difference=max_historical,
        historical_feature_differences_above_001=differences, historical_binary_decision_changes=changes,
        runtime_source_groups_checked=len({(r['partition'], r['group']) for r in rows}))
    if any(reported.get(k) != v for k, v in expected.items()):
        raise ValueError('Reported reference metrics differ')
    score_error = reported['maximum_runtime_score_error']
    if not math.isfinite(score_error) or not 0 <= score_error <= 1e-10:
        raise ValueError('Reported serving score error exceeds tolerance')
    paired = {}
    for split in ('train', 'dev'):
        indices = [i for i, r in enumerate(rows) if r['partition'] == split]
        paired[split] = dict(rows=len(indices),
            historical_errors=sum(before[i] != rows[i]['gold'] for i in indices),
            runtime_errors=sum(after[i] != rows[i]['gold'] for i in indices),
            fixes=sum(before[i] != rows[i]['gold'] and after[i] == rows[i]['gold'] for i in indices),
            introductions=sum(before[i] == rows[i]['gold'] and after[i] != rows[i]['gold'] for i in indices))
    return dict(**expected, paired_v9=paired,
        warm_start_development_binary_errors=sum(p != rows[i]['gold'] for p, i in zip(actual_decisions, dev)),
        reported_maximum_runtime_score_error=score_error)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'candidate', 'input_study', 'inputs', 'feature_study', 'features',
                 'initial_study', 'initial_private', 'diagnostic_study', 'diagnostic_private',
                 'study', 'private', 'amendment', 'output'):
        p.add_argument('--'+name.replace('_', '-'), dest=name, type=Path, required=True)
    p.add_argument('--producer-pid', type=int, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Refusing to overwrite verification')
    report = json.loads((a.study/'report.json').read_text()); require_complete(report)
    deadline = time.monotonic()+30
    while True:
        try: os.kill(a.producer_pid, 0)
        except ProcessLookupError: break
        if time.monotonic() >= deadline:
            raise ValueError('Reference producer has not exited')
        time.sleep(.2)
    from training.extract_runtime_review_baseline_v10 import load
    plan, recovered, bindings = load(a)
    protocol = json.loads((a.study/'protocol.json').read_text())
    expected_protocol = dict(format='roguard-runtime-review-baseline-1', source_sha256=plan['source_sha256'],
        amendment_sha256=file_sha(a.amendment), bindings=bindings,
        runtime_reference_hashes=plan['runtime_reference_hashes'], native_replay_tolerance=1e-3,
        warm_start_logit_tolerance=1e-5, runtime_score_tolerance=1e-10,
        training_performed=False, held_out_test_opened=False, qualified=False)
    if protocol != expected_protocol or file_sha(a.study/'source.py') != plan['source_sha256']:
        raise ValueError('Runtime reference protocol or source differs')
    import numpy as np
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    from safetensors.numpy import load_file
    from training.shared_tail_training_v9 import make_head
    from roguard.gemma4_review import _score
    mapping = {'0.weight':'first.weight', '0.bias':'first.bias', '3.weight':'last.weight', '3.bias':'last.bias'}
    summaries = {}; artifacts = {}
    snapshot = {path:file_sha(path) for path in (a.study/'report.json', a.study/'protocol.json', a.study/'source.py')}
    for lang in ('ro', 'uk'):
        record = report['languages'][lang]
        suffixes = ('features.npy', 'logits.npy', 'warm-start-dev-logits.npy', 'rows.jsonl', 'observations.jsonl')
        if set(record['artifacts']) != {lang+'-'+s for s in suffixes}:
            raise ValueError('Retained artifact inventory differs')
        for name, expected in record['artifacts'].items():
            path = a.private/name
            if file_sha(path) != expected:
                raise ValueError('Retained artifact changed: '+name)
            artifacts[name] = expected; snapshot[path] = expected
        rows = read_jsonl(a.private/(lang+'-rows.jsonl'))
        if rows != read_jsonl(a.features/(lang+'-rows.jsonl')) or len(rows) != len(recovered[lang]):
            raise ValueError('Original metadata differs')
        observations = read_jsonl(a.private/(lang+'-observations.jsonl'))
        x, z, warm = [np.load(a.private/(lang+'-'+s), mmap_mode='r', allow_pickle=False) for s in suffixes[:3]]
        old_x = np.load(a.features/(lang+'-features.npy'), mmap_mode='r', allow_pickle=False)
        old_z = np.load(a.features/(lang+'-logits.npy'), mmap_mode='r', allow_pickle=False)
        reference = np.load(a.diagnostic_private/(lang+'-prefix-dev-logits.npy'), allow_pickle=False)
        spec_path = a.study/'v9-binding'/(lang+'-review.json')
        if file_sha(spec_path) != bindings[lang]['original_spec_sha256']:
            raise ValueError('Original-v9 runtime specification differs')
        spec = json.loads(spec_path.read_text()); warm_spec = json.loads((a.initial_study/(lang+'-review.json')).read_text())
        summary = check_population(rows, observations, x, z, warm, reference, old_x, old_z, spec, warm_spec, record)
        dev = np.array([i for i, row in enumerate(rows) if row['partition'] == 'dev'])
        head_checks = {}
        for name, weights_path, indices, expected, head_spec in (
                ('v9', a.candidate/'actual'/(lang+'-selected.safetensors'), np.arange(len(rows)), z, spec),
                ('warm_start', a.initial_private/(lang+'-selected.safetensors'), dev, warm, warm_spec)):
            tensors = load_file(weights_path)
            head = make_head({k:tensors['heads.'+lang+'.'+v] for k,v in mapping.items()}); head.eval()
            pieces = []
            for start in range(0, len(indices), 256):
                values = head(mx.array(x[indices[start:start+256]])); mx.eval(values); pieces.append(np.asarray(values).copy())
            actual = np.concatenate(pieces); error = float(np.max(np.abs(actual-expected)))
            if not np.isfinite(actual).all() or error > 1e-3:
                raise ValueError('CPU head replay exceeds tolerance')
            head_checks[name] = dict(rows=len(actual), maximum_logit_error=error,
                argmax_rows_differing=int(np.any(argmaxes(actual) != argmaxes(expected), axis=1).sum()),
                fixed_threshold_decision_differences=sum(
                    (_score(v, head_spec) >= head_spec['threshold']) != (_score(w, head_spec) >= head_spec['threshold'])
                    for v,w in zip(actual, expected)))
            del head, tensors, pieces, actual; mx.clear_cache()
        summaries[lang] = dict(**summary, cpu_head_replay=head_checks)
    if load(a)[2] != bindings or any(file_sha(path) != expected for path,expected in snapshot.items()):
        raise ValueError('Audit inputs changed')
    proof = dict(verification='PASS', source_sha256=file_sha(__file__),
        report_sha256=file_sha(a.study/'report.json'), protocol_sha256=file_sha(a.study/'protocol.json'),
        amendment_sha256=file_sha(a.amendment), artifacts=artifacts, languages=summaries,
        original_input_bindings=bindings, device='cpu', encoder_loaded=False, training_performed=False,
        held_out_test_opened=False, qualified=False,
        scope='All retained arrays, metadata, historical feature/decision differences and warm-start DEV references; original-v9 heads on37766 rows and warm heads on8837 DEV rows replayed on CPU. Native encoder differences are recounted from observations; direct-engine score differences are reported maxima, not independently rerun. CPU threshold/argmax differences are reported separately and do not establish runtime qualification.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps(dict(verification='PASS', languages=summaries, qualified=False), indent=2))


if __name__ == '__main__':
    main()
