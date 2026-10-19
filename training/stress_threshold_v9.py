"""Development/stress-only threshold repair of the fixed study97 tree."""
import json
from pathlib import Path
from training.review_fusion_v9 import read, sha, metrics
from training.conservative_fusion_v9 import full_coverage_gap
from training.reserved_metrics_v9 import counts


def main():
    root = Path('research/v9-stabilization')
    study = root/'stress-threshold-101'
    for path, expected in json.loads((study/'source-freeze.json').read_text()).items():
        if sha(path) != expected:
            raise ValueError('Frozen threshold recipe differs')
    parent = root/'repaired-three-expert-97/95'
    selection = json.loads((parent/'selection.json').read_text())
    rows = read(root/'bare-distress-95/uk/dev.jsonl.gz')
    dev = read(parent/'dev-predictions.jsonl')
    oof = read(parent/'oof-predictions.jsonl')
    stress = read(root/'installed-repaired-99/actual/stress-predictions.jsonl')
    if [r['case_id'] for r in rows] != [r['case_id'] for r in dev] or [r['case_id'] for r in dev] != [r['case_id'] for r in oof]:
        raise ValueError('Development prediction coverage differs')
    all_values = dev + oof + stress
    if any(r['score'] is None for r in all_values):
        raise ValueError('Threshold repair cannot recover an abstention')
    threshold = min(selection['threshold'], full_coverage_gap(
        [r['gold'] for r in all_values], [r['score'] for r in all_values]))
    development = {name: metrics(rows, [r['gold'] for r in values], [r['score'] for r in values], threshold)
                   for name, values in [('refit', dev), ('oof', oof)]}
    stress_counts = counts([dict(gold=r['gold'], predicted=int(r['score'] >= threshold)) for r in stress])
    passed = all(v['passes'] for group in development.values() for v in group.values()) and stress_counts['passes']
    record = dict(parent_selection_sha256=sha(parent/'selection.json'),
        actual_stress_sha256=sha(root/'installed-repaired-99/actual/stress-predictions.jsonl'),
        model=selection['model'], old_threshold=selection['threshold'], threshold=threshold,
        development=development, stress=stress_counts, development_prerequisites_pass=passed,
        calibration_used_for_threshold=False, reserved_probes_used=False, stable_release_ready=False)
    # Freeze the proposed cutoff before opening calibration predictions.
    (study/'selection.json').write_text(json.dumps(record, indent=2)+'\n')
    if passed:
        cal = read(parent/'cal-predictions.jsonl')
        cal_rows = read(root/'bare-distress-95/uk/cal.jsonl.gz')
        if [r['case_id'] for r in cal_rows] != [r['case_id'] for r in cal]:
            raise ValueError('Calibration coverage differs')
        calibration = metrics(cal_rows, [r['gold'] for r in cal], [r['score'] for r in cal], threshold)
        report = dict(selection_sha256=sha(study/'selection.json'), calibration=calibration,
            counts_pass=all(v['passes'] for v in calibration.values()), calibration_audit_skipped=False,
            reserved_probes_used=False, stable_release_ready=False)
    else:
        report = dict(selection_sha256=sha(study/'selection.json'), counts_pass=False,
            calibration_audit_skipped=True, reason='Development/stress counts already fail',
            reserved_probes_used=False, stable_release_ready=False)
    (study/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in record.items() if k != 'model'}))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
