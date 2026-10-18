"""Select by the shared validation loss, then evaluate complete regression sets."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    args = parser.parse_args()
    root = Path('research/v9-stabilization')
    study = root/'interpreter-full-21'
    deadline = datetime.fromisoformat(json.loads((root/'work-window.json').read_text())['deadline_utc'].replace('Z', '+00:00')).timestamp()
    while True:
        if time.time() >= deadline:
            raise TimeoutError('Authorized work window ended')
        path = root/'interpreter-policy-27/run-status.json'
        if path.exists():
            previous = json.loads(path.read_text())
            jobs = previous['jobs']
            if jobs and (previous.get('complete', False) or jobs[-1]['exit_code'] != 0):
                if time.time() > datetime.fromisoformat(jobs[-1]['finished_utc']).timestamp()+15:
                    break
        time.sleep(10)
    for path, expected in json.loads((study/'source-freeze.json').read_text()).items():
        if digest(Path(path)) != expected:
            raise ValueError('Frozen source changed: '+path)
    candidates = []
    validation_hash = digest(root/'interpreter-14/valid.jsonl')
    for number, directory in ((14, 'interpreter-14'), (27, 'interpreter-policy-27')):
        candidate = root/directory
        if digest(candidate/'valid.jsonl') != validation_hash:
            raise ValueError('Candidates do not share the same validation population')
        if not (candidate/'run-status.json').exists():
            continue
        jobs = json.loads((candidate/'run-status.json').read_text())['jobs']
        if not all(any(j['name'] == name and j['exit_code'] == 0 for j in jobs) for name in ('train','adapted','archive')):
            continue
        verification = candidate/'archive-verification.json'
        if not verification.exists():
            continue
        checked = json.loads(verification.read_text())
        if (checked.get('exact_tensor_roundtrip') is not True
                or checked.get('archive_sha256') != digest(candidate/'selected-adapter.json.gz')
                or checked.get('source_safetensors_sha256') != digest(candidate/'selected-adapter/adapters.safetensors')):
            raise ValueError('Candidate archive verification differs')
        metrics = json.loads((candidate/'training-metrics.json').read_text())
        if not math.isfinite(metrics['selected']['val_loss']):
            raise ValueError('Nonfinite selected validation loss')
        candidates.append((metrics['selected']['val_loss'], number, candidate))
    if not candidates:
        raise ValueError('No completed structured candidate')
    loss, number, selected = min(candidates)
    selection = dict(method='Minimum selected loss on the shared 80-row validation set; earlier study breaks ties.',
                     candidates=[dict(study=n, validation_loss=v) for v,n,p in candidates],
                     selected_study=number, validation_loss=loss, validation_sha256=validation_hash,
                     archive_sha256=digest(selected/'selected-adapter.json.gz'),
                     raw_adapter_sha256=digest(selected/'selected-adapter/adapters.safetensors'),
                     reserved_probes_used=False, stable_release_ready=False)
    (study/'selection.json').write_text(json.dumps(selection, indent=2)+'\n')
    logs = Path(tempfile.gettempdir())/'roguard-v9-window'
    logs.mkdir(exist_ok=True)
    status = dict(stable_release_ready=False, reserved_probes_used=False, jobs=[])
    for split in ('dev', 'cal'):
        remaining = deadline-time.time()
        if remaining <= 0:
            break
        source = Path('research/v8-stabilization/review-09-ro')/(split+'.jsonl.gz')
        data = study/(split+'.jsonl')
        data.write_bytes(gzip.decompress(source.read_bytes()))
        command = [sys.executable, '-m', 'training.evaluate_interpreter_v9', '--base', args.base,
                   '--adapter', str(selected/'selected-adapter'), '--data', str(data), '--output', str(study/split)]
        entry = dict(name=split, started_utc=datetime.now(timezone.utc).isoformat())
        print('START full interpreter', split, flush=True)
        with (logs/('interpreter-full-21-'+split+'.log')).open('w') as log:
            try:
                entry['exit_code'] = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=remaining).returncode
            except subprocess.TimeoutExpired:
                entry.update(exit_code=-1, timed_out=True)
        if entry['exit_code'] == 0:
            from training.diagnose_interpreter_v9 import diagnose
            rows = [json.loads(line) for line in data.read_text().splitlines()]
            predictions = [json.loads(line) for line in (study/split/'predictions.jsonl').read_text().splitlines()]
            report = diagnose(rows, predictions)
            (study/split/'diagnosis.json').write_text(json.dumps(report, indent=2)+'\n')
            entry['observed_targets_pass'] = not report['target_failures']
        entry['finished_utc'] = datetime.now(timezone.utc).isoformat()
        status['jobs'].append(entry)
        (study/'run-status.json').write_text(json.dumps(status, indent=2)+'\n')
        print('FINISHED', json.dumps(entry), flush=True)
        if entry['exit_code'] != 0:
            break


if __name__ == '__main__':
    main()
