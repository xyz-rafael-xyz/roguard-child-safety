"""Replay score-bound exits against the fixed study147 development decisions."""
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

from training.contextual_bounds_v9 import early_decision
from training.portable_scope_gate_v9 import ScopeGate
from training.verify_portable_scope_v9 import read


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path('research/v9-stabilization')
    input_path = root/'gemma4-readout-141/inputs.jsonl.gz'
    entries = read(input_path)
    blend = json.loads((root/'contextual-complementarity-146/actual/report.json').read_text())
    original = json.loads((root/'expanded-scope-gate-147/actual/report.json').read_text())
    report = dict(complete=False, languages={}, new_neural_inference=False, fresh_test_inferred=False,
        calibration_inferred=False, installed_execution_verified=False, synthetic_only=True,
        scope='Decision-equivalence and avoided-call counts only; no accuracy or end-to-end latency claim',
        implementation_sha256=digest('training/contextual_bounds_v9.py'),
        scope_implementation_sha256=digest('training/portable_scope_gate_v9.py'),
        verification_source_sha256=digest(__file__), inputs_sha256=digest(input_path))
    for language in ('ro', 'uk'):
        gate_path = root/'expanded-scope-gate-147/actual'/(language+'-scope-gate.json.gz')
        selection = original['languages'][language]['selected']
        if digest(gate_path) != selection['artifact_sha256']:
            raise ValueError('Selected gate differs')
        spec = json.loads(gzip.decompress(gate_path.read_bytes()))
        if spec['selected']['baseline'] != 'blend146' or blend['languages'][language]['selected']['mode'] != 'logit':
            raise ValueError('Bounds require the frozen logit composition')
        gate = ScopeGate(spec)
        threshold = spec['selected']['threshold']
        weight = blend['languages'][language]['selected']['native_weight']
        native = {}
        native_paths = []
        for population in ('dev', 'scope_dev')+(('stress',) if language == 'uk' else ()):
            path = root/'native-scope-operators-131'/language/(population+'-predictions.jsonl')
            native_paths.append(path)
            native.update({r['case_id']: r for r in read(path)})
        expected_path = root/'expanded-scope-gate-147/actual'/(language+'-dev-predictions.jsonl')
        expected = {r['case_id']: r for r in read(expected_path)}
        rows = [r for r in entries if r['partition'] == 'dev' and r['row']['language'] == language]
        if set(native) != set(expected) or set(expected) != {r['row']['case_id'] for r in rows}:
            raise ValueError('Development population differs')
        counts = Counter()
        groups = {}
        for entry in rows:
            row = entry['row']
            multiplier = gate.probability(row)**gate.power
            answer = early_decision(multiplier, weight, threshold)
            stage = 'scope_only' if answer is not None else 'after_native'
            if answer is None:
                answer = early_decision(multiplier, weight, threshold, native[row['case_id']]['score'])
                if answer is None:
                    stage = 'contextual_required'
            if answer is not None and answer != expected[row['case_id']]['predicted']:
                raise ValueError('An early bound changed the selected decision')
            counts[stage] += 1
            groups.setdefault(entry['group'], Counter())[stage] += 1
        report['languages'][language] = dict(rows=len(rows), stages=dict(counts), groups={k: dict(v) for k, v in groups.items()},
            contextual_calls_avoided=counts['scope_only']+counts['after_native'], native_calls_avoided=counts['scope_only'],
            early_decision_changes=0, threshold=threshold, native_weight=weight, scope_power=gate.power,
            gate_sha256=digest(gate_path), reference_predictions_sha256=digest(expected_path),
            native_predictions_sha256={str(path): digest(path) for path in native_paths})
    report.update(complete=True, finished_utc=datetime.now(timezone.utc).isoformat())
    (root/'contextual-bounds-replay-20261019.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
