"""Serialize two frozen entailment-transfer trials within the work deadline."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def main():
    p = argparse.ArgumentParser(); p.add_argument('--base', required=True); a = p.parse_args()
    root = Path('research/v9-stabilization')
    deadline = datetime.fromisoformat(json.loads((root/'work-window.json').read_text())['deadline_utc'].replace('Z', '+00:00')).timestamp()
    while True:
        if time.time() >= deadline: raise TimeoutError('Authorized work window ended')
        path = root/'policy-decode-16/run-status.json'
        if path.exists():
            previous = json.loads(path.read_text()); jobs = previous['jobs']
            if jobs and (len(jobs) == 2 or jobs[-1]['exit_code'] != 0 or previous.get('adapted_skipped')):
                if time.time() > datetime.fromisoformat(jobs[-1]['finished_utc']).timestamp()+15: break
        time.sleep(10)
    for path, digest in json.loads((root/'nli-transfer-18-source-freeze.json').read_text()).items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest: raise ValueError('Frozen source changed: '+path)
    logs = Path(tempfile.gettempdir())/'roguard-v9-window'; logs.mkdir(exist_ok=True)
    status = dict(stable_release_ready=False, reserved_probes_used=False, jobs=[])
    for label, rate in [('low', '0.00002'), ('high', '0.00005')]:
        name = 'nli-transfer-18-'+label
        command = [sys.executable, '-m', 'training.adapt_nli_review_v9', '--base', a.base,
                   '--output', str(root/name), '--device', 'mps', '--epochs', '2', '--learning-rate', rate]
        remaining = deadline-time.time()
        if remaining <= 0: break
        print('START', name, flush=True)
        entry = dict(name=name, started_utc=datetime.now(timezone.utc).isoformat())
        with (logs/(name+'.log')).open('w') as log:
            try: entry['exit_code'] = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=remaining).returncode
            except subprocess.TimeoutExpired: entry['exit_code'] = -1; entry['timed_out'] = True
        if entry['exit_code'] == 0:
            from training.check_review_candidate_v8 import inspect
            check = inspect(root/name)
            (root/name/'promotion-check.json').write_text(json.dumps(check, indent=2)+'\n')
            entry['prerequisites_pass'] = check['study_prerequisites_pass']
        entry['finished_utc'] = datetime.now(timezone.utc).isoformat(); status['jobs'].append(entry)
        (root/'nli-transfer-18-run-status.json').write_text(json.dumps(status, indent=2)+'\n')
        print('FINISHED', json.dumps(entry), flush=True)
        if entry['exit_code'] != 0: break


if __name__ == '__main__': main()
