"""Audit completed85 metadata/parent anchors and replay frozen original-v9 heads."""
import argparse
import json
import os
from pathlib import Path

from roguard.assurance import digest
from roguard.trajectory_data import file_sha, read_jsonl


def check_arrays(old_rows, old_x, old_z, runtime, saved_rows, parents, arrays, reported, spec, feature_limit, logit_limit):
    import numpy as np
    from roguard.gemma4_review import _score
    expected_rows = []
    for r in runtime:
        p = r['parent']; index = p['feature_index']
        if type(index) is not int or not 0 <= index < len(old_rows): raise ValueError('Bad parent index')
        parent = old_rows[index]
        if (parent['partition'] != 'train' or r['partition'] != 'train'
                or any(parent[k] != p[k] for k in ('case_id', 'language', 'gold', 'fields', 'token_sha256'))
                or r['gold'] != parent['gold'] or r['fields'] != parent['fields']
                or digest(r['tokens']) != r['token_sha256']):
            raise ValueError('Runtime labels, parent or tokens differ')
        expected_rows.append({'case_id': r['case_id'], 'language': r['language'], 'partition': 'train',
            'group': 'assertion_variation_77', 'gold': r['gold'], 'fields': r['fields'], 'token_sha256': r['token_sha256'],
            'pair_id': r['pair_id'], 'parent_feature_index': index, 'parent_case_id': p['case_id'],
            'parent_token_sha256': p['token_sha256'], 'factors': r['factors']})
    if saved_rows != expected_rows: raise ValueError('Extracted metadata differs')
    indices = sorted({r['parent_feature_index'] for r in expected_rows})
    if parents != [{'feature_index': i, **old_rows[i]} for i in indices]:
        raise ValueError('Parent replay ordering differs')
    shapes = {'treatment-features.npy': (len(runtime), 2560), 'treatment-logits.npy': (len(runtime), 19),
              'control-parent-features.npy': (len(indices), 2560), 'control-parent-logits.npy': (len(indices), 19)}
    if set(arrays) != set(shapes): raise ValueError('Wrong artifact array inventory')
    for name, value in arrays.items():
        if value.shape != shapes[name] or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError('Invalid extracted array: '+name)
    dx = float(np.max(np.abs(arrays['control-parent-features.npy']-old_x[indices])))
    dz = float(np.max(np.abs(arrays['control-parent-logits.npy']-old_z[indices])))
    disagreements = sum((_score(a, spec) >= spec['threshold']) != (_score(b, spec) >= spec['threshold'])
                        for a, b in zip(arrays['control-parent-logits.npy'], old_z[indices]))
    if dx > feature_limit or dz > logit_limit or disagreements:
        raise ValueError('Replayed original anchors differ')
    if (reported['maximum_parent_feature_error'] != dx or reported['maximum_parent_logit_error'] != dz
            or reported['parent_binary_decision_disagreements'] != disagreements
            or reported['treatment_rows'] != len(runtime) or reported['control_parent_rows'] != len(indices)):
        raise ValueError('Reported anchor metrics or counts differ')
    return {'treatment_rows': len(runtime), 'control_parents': len(indices),
            'maximum_parent_feature_error': dx, 'maximum_parent_logit_error': dz,
            'parent_binary_decision_disagreements': disagreements}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'private', 'output'): p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise ValueError('Refusing to overwrite verification')
    import numpy as np
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    from safetensors.numpy import load_file
    from training.shared_tail_training_v9 import make_head
    from roguard.gemma4_review import _score
    e = a.root/'research/v10-evidence'; study = e/'assertion-variation-features-85'
    data = a.private/'assertion-variation-features-85'
    plan_path = e/'protocol-amendment-71.json'; plan = json.loads(plan_path.read_text())
    for base, bindings in ((a.root, plan['public']), (a.private, plan['private'])):
        for name, expected in bindings.items():
            if file_sha(base/name) != expected: raise ValueError('Bound85 source differs: '+name)
    queue_path = a.private/'assertion-variation-features-sequence-85-plan.json'; queue = json.loads(queue_path.read_text())
    sequence_path = e/'assertion-variation-features-sequence-85/report.json'; sequence = json.loads(sequence_path.read_text())
    if (sequence.get('execution_complete') is not True or sequence.get('phase') != 'finished'
            or [s['name'] for s in sequence['completed_steps']] != [s['name'] for s in queue['steps']]
            or any(s['exit_code'] != 0 for s in sequence['completed_steps'])
            or sequence['plan_sha256'] != file_sha(queue_path) or sequence['source_hashes'] != queue['source_hashes']):
        raise ValueError('Feature sequence did not complete correctly')
    for name, expected in queue['source_hashes'].items():
        if file_sha(a.root/name) != expected: raise ValueError('Queued85 source differs: '+name)
    pid = int((a.private/'assertion-variation-features-sequence-85.pid').read_text())
    try: os.kill(pid, 0)
    except ProcessLookupError: pass
    else: raise ValueError('Extraction queue has not exited')
    report = json.loads((study/'report.json').read_text()); protocol = json.loads((study/'protocol.json').read_text())
    source = a.root/'training/extract_assertion_continuation_v10.py'
    if (report.get('complete') is not True or report['source_sha256'] != file_sha(source)
            or report['source_sha256'] != file_sha(study/'source.py')
            or report['plan_sha256'] != file_sha(plan_path) or protocol['plan_sha256'] != file_sha(plan_path)
            or protocol['source_sha256'] != file_sha(source) or protocol['languages'] != plan['languages']
            or protocol['treatment_rows'] != 864 or set(report['languages']) != {'ro', 'uk'}
            or report['training_performed'] is not False or report['held_out_test_opened'] is not False):
        raise ValueError('Feature report or protocol differs')
    from training.extract_assertion_continuation_v10 import require_dependency
    dep = require_dependency(a, plan)
    if report['dependency'] != dep or protocol['dependency'] != dep:
        raise ValueError('Preceding84 verification differs')
    inputs = read_jsonl(a.private/'assertion-variation-77/curated/runtime-inputs.jsonl')
    summaries = {}; artifact_hashes = {}
    for lang in ('ro', 'uk'):
        record = report['languages'][lang]; old = a.private/'crossed-review-features-62'
        suffixes = ('treatment-features.npy', 'treatment-logits.npy', 'control-parent-features.npy',
                    'control-parent-logits.npy', 'rows.jsonl', 'control-parents.jsonl')
        expected_names = {lang+'-'+suffix for suffix in suffixes}
        if set(record['artifacts']) != expected_names: raise ValueError('Wrong retained artifact inventory')
        for name, expected in record['artifacts'].items():
            if file_sha(data/name) != expected: raise ValueError('Feature artifact differs: '+name)
            artifact_hashes[name] = expected
        old_rows = read_jsonl(old/(lang+'-rows.jsonl'))
        old_x = np.load(old/(lang+'-features.npy'), mmap_mode='r', allow_pickle=False)
        old_z = np.load(old/(lang+'-logits.npy'), mmap_mode='r', allow_pickle=False)
        runtime = [r for r in inputs if r['language'] == lang]
        arrays = {suffix: np.load(data/(lang+'-'+suffix), allow_pickle=False) for suffix in suffixes[:4]}
        spec_path = study/'v9-binding'/(lang+'-review.json'); spec = json.loads(spec_path.read_text())
        if file_sha(spec_path) != plan['languages'][lang]['runtime_spec_sha256']:
            raise ValueError('Frozen specification changed')
        summary = check_arrays(old_rows, old_x, old_z, runtime, read_jsonl(data/(lang+'-rows.jsonl')),
            read_jsonl(data/(lang+'-control-parents.jsonl')), arrays, record, spec,
            plan['maximum_parent_feature_error'], plan['maximum_parent_logit_error'])
        if (summary['treatment_rows'] != plan['languages'][lang]['treatment_rows']
                or summary['control_parents'] != plan['languages'][lang]['unique_control_parents']):
            raise ValueError('Registered population differs')
        path = a.root/'research/v9-stabilization/reported-tail-fit-196/actual'/(lang+'-selected.safetensors')
        if file_sha(path) != plan['languages'][lang]['weights_sha256']:
            raise ValueError('Original head weights changed')
        weights = load_file(path); mapping = {'0.weight': 'first.weight', '0.bias': 'first.bias',
                                             '3.weight': 'last.weight', '3.bias': 'last.bias'}
        head = make_head({key: weights['heads.'+lang+'.'+value] for key, value in mapping.items()}); head.eval()
        replay = {}
        for group in ('treatment', 'control-parent'):
            x = arrays[group+'-features.npy']; expected = arrays[group+'-logits.npy']; pieces = []
            for start in range(0, len(x), 256):
                values = head(mx.array(x[start:start+256])); mx.eval(values); pieces.append(np.asarray(values).copy())
            actual = np.concatenate(pieces); error = float(np.max(np.abs(actual-expected)))
            if not np.isfinite(actual).all() or error > 1e-3: raise ValueError('Original head replay differs')
            replay[group] = {'rows': len(x), 'maximum_logit_error': error,
                'threshold_decision_differences': sum((_score(x, spec) >= spec['threshold']) != (_score(y, spec) >= spec['threshold'])
                                                    for x, y in zip(actual, expected))}
        summary['original_head_cpu_replay'] = replay; summaries[lang] = summary
        del head, weights; mx.clear_cache()
    proof = {'verification': 'PASS', 'source_sha256': file_sha(__file__),
             'report_sha256': file_sha(study/'report.json'), 'protocol_sha256': file_sha(study/'protocol.json'),
             'plan_sha256': file_sha(plan_path), 'sequence_sha256': file_sha(sequence_path),
             'artifacts': artifact_hashes, 'languages': summaries,
             'runtime_inputs_sha256': file_sha(a.private/'assertion-variation-77/curated/runtime-inputs.jsonl'),
             'device': 'cpu', 'encoder_loaded': False, 'training_performed': False, 'qualified': False,
             'scope': 'Exact metadata, source and token identities;108 saved control anchors recounted;972 original-head outputs replayed on CPU. No repeated encoder inference or independent gold validation.'}
    a.output.write_text(json.dumps(proof, indent=2)+'\n'); print(json.dumps(proof, indent=2))


if __name__ == '__main__': main()
