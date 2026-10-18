"""Repair a boundary cutoff using unchanged, previously installed neural outputs.

This is a deterministic threshold replay, not a new encoder inference run.
"""
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

from roguard.native_fields import factorized_review_mass
from training.threshold_center_v9 import center_threshold
from training.reserved_metrics_v9 import counts


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path('research/v9-stabilization')
    parent = root / 'native-fields-43/uk'
    installed = root / 'installed-native-fields-45/uk'
    output = root / 'field-threshold-portability-50'
    report = json.loads((installed / 'report.json').read_text())
    original = parent / 'selected-heads.json.gz'
    if (report.get('complete') is not True or report['artifact_sha256'] != sha(original)
            or report['runtime_source_sha256'] != sha(Path('src/roguard/native_fields.py'))):
        raise ValueError('Installed neural evidence does not match original heads')
    spec = json.loads(gzip.decompress(original.read_bytes()))
    development = [json.loads(line) for line in (parent / 'dev-predictions.jsonl').read_text().splitlines()]
    old = spec['threshold']
    spec['threshold'] = center_threshold([r['score'] for r in development], old)
    if any(int(r['score'] >= spec['threshold']) != r['predicted'] for r in development):
        raise ValueError('Centered threshold changed development decisions')
    output.mkdir(exist_ok=False)
    artifact = output / 'selected-heads.json.gz'
    artifact.write_bytes(gzip.compress(json.dumps(spec, separators=(',', ':')).encode(), mtime=0))
    original_hash, artifact_hash = sha(original), sha(artifact)
    selection = dict(created_utc=datetime.now(timezone.utc).isoformat(), language='uk',
        original_artifact_sha256=sha(original), artifact_sha256=sha(artifact),
        old_threshold=old, threshold=spec['threshold'],
        minimum_development_distance=min(abs(r['score']-spec['threshold']) for r in development),
        development_decisions_changed=0, head_weights_and_encoder_unchanged=True,
        installed_report_sha256=sha(installed / 'report.json'),
        threshold_selection_uses_calibration=False, reserved_probes_used=False)
    (output / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    result = dict(format='roguard-field-threshold-portability-v1', language='uk',
        artifact_sha256=sha(artifact), selection_sha256=sha(output / 'selection.json'),
        cached_actual_installed_neural_outputs_used=True, fresh_neural_inference=False,
        synthetic_only=True, stable_release_ready=False, splits={})
    for split in ('dev', 'cal'):
        reference_path = parent / (split + '-predictions.jsonl')
        actual_path = installed / (split + '-predictions.jsonl')
        evidence = report['splits'][split]
        if (evidence['reference_sha256'] != sha(reference_path)
                or evidence['predictions_sha256'] != sha(actual_path)):
            raise ValueError('Cached predictions changed')
        reference = [json.loads(line) for line in reference_path.read_text().splitlines()]
        actual = [json.loads(line) for line in actual_path.read_text().splitlines()]
        if [r['case_id'] for r in reference] != [r['case_id'] for r in actual]:
            raise ValueError('Prediction coverage differs')
        results = []
        for ref, value in zip(reference, actual):
            if ref['gold'] != value['gold'] or value['model_identity'] != original_hash:
                raise ValueError('Cached model or labels differ')
            score = factorized_review_mass(value['field_probabilities'], couplings=spec['couplings'])
            if abs(score-value['score']) > 1e-12:
                raise ValueError('Unchanged neural scores do not replay')
            predicted = int(score >= spec['threshold'])
            results.append(dict(case_id=value['case_id'], input_digest=value['input_digest'],
                gold=value['gold'], kind=value['kind'], score=score, predicted=predicted,
                model_identity=artifact_hash, reference_predicted=int(ref['score'] >= spec['threshold'])))
        groups = {kind: counts([r for r in results if r['kind'] == kind]) for kind in ('formal', 'conversational')}
        result['splits'][split] = dict(decision_changes=sum(r['predicted'] != r['reference_predicted'] for r in results),
            counts=groups, source_predictions_sha256=sha(actual_path))
        (output / (split + '-predictions.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in results))
    result['parity_pass'] = all(v['decision_changes'] == 0 for v in result['splits'].values())
    result['accuracy_gates_pass'] = all(g['passes'] for v in result['splits'].values() for g in v['counts'].values())
    (output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    if not result['parity_pass']:
        raise SystemExit('Centered threshold still differs between devices')


if __name__ == '__main__':
    main()
