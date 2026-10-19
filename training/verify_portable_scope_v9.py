"""Replay selected sparse gates on every development input without NumPy/sklearn."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import time

from training.portable_scope_gate_v9 import ScopeGate


def read(path):
    data = path.read_bytes()
    if path.suffix == '.gz':
        data = gzip.decompress(data)
    return [json.loads(line) for line in data.decode().splitlines()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path('research/v9-stabilization')
    inputs = root/'gemma4-readout-141/inputs.jsonl.gz'
    rows = read(inputs)
    record = dict(complete=False, new_neural_inference=False, calibration_inferred=False,
                  fresh_test_inferred=False, installed_execution_verified=False,
                  implementation_sha256=digest(Path('training/portable_scope_gate_v9.py')),
                  verification_source_sha256=digest(Path(__file__)), inputs_sha256=digest(inputs), languages={})
    report = json.loads((root/'expanded-scope-gate-147/actual/report.json').read_text())
    for language in ('ro', 'uk'):
        path = root/'expanded-scope-gate-147/actual'/(language+'-scope-gate.json.gz')
        if digest(path) != report['languages'][language]['selected']['artifact_sha256']:
            raise ValueError('Selected scope artifact differs')
        spec = json.loads(gzip.decompress(path.read_bytes()))
        if spec['selected']['baseline'] != 'blend146':
            raise ValueError('This replay requires the selected146 baseline')
        model = ScopeGate(spec)
        dev = [r['row'] for r in rows if r['partition'] == 'dev' and r['row']['language'] == language]
        baseline_path = root/'contextual-complementarity-146/actual'/(language+'-dev-predictions.jsonl')
        expected_path = root/'expanded-scope-gate-147/actual'/(language+'-dev-predictions.jsonl')
        baseline = {r['case_id']: r for r in read(baseline_path)}
        expected = {r['case_id']: r for r in read(expected_path)}
        if set(baseline) != set(expected) or set(expected) != {r['case_id'] for r in dev}:
            raise ValueError('Development population differs')
        start = time.perf_counter()
        values = model.score(dev, [baseline[r['case_id']]['score'] for r in dev])
        seconds = time.perf_counter()-start
        error = max(abs(v-expected[r['case_id']]['score']) for r, v in zip(dev, values))
        threshold = spec['selected']['threshold']
        changed = sum(int(v >= threshold) != expected[r['case_id']]['predicted'] for r, v in zip(dev, values))
        if error > 1e-12 or changed:
            raise ValueError('Portable scope scores or decisions differ')
        record['languages'][language] = dict(rows=len(dev), score_max_error=error, decision_changes=changed,
            seconds=seconds, milliseconds_per_case=seconds*1000/len(dev), artifact_sha256=digest(path),
            baseline_predictions_sha256=digest(baseline_path), reference_predictions_sha256=digest(expected_path))
    record.update(complete=True, finished_utc=datetime.now(timezone.utc).isoformat())
    (root/'portable-scope-replay-20261019.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
