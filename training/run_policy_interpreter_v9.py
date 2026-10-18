"""Run the policy-sensitive continuation after adaptive policy decoding."""
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    args = parser.parse_args()
    root = Path('research/v9-stabilization')
    study = root/'interpreter-policy-27'
    deadline = datetime.fromisoformat(json.loads((root/'work-window.json').read_text())['deadline_utc'].replace('Z', '+00:00')).timestamp()
    while True:
        if time.time() >= deadline:
            raise TimeoutError('Authorized work window ended')
        previous = root/'policy-decode-16/run-status.json'
        if previous.exists():
            previous_status = json.loads(previous.read_text())
            jobs = previous_status['jobs']
            if jobs and (len(jobs) == 2 or jobs[-1]['exit_code'] != 0 or previous_status.get('adapted_skipped')):
                if time.time() > datetime.fromisoformat(jobs[-1]['finished_utc']).timestamp()+15:
                    break
        time.sleep(10)
    for path, digest in json.loads((study/'source-freeze.json').read_text()).items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen source changed: '+path)
    screen = 'research/v8-stabilization/instruction-sft-01/screen.jsonl'
    jobs = [
        ('train', [sys.executable, '-m', 'training.train_policy_interpreter_v9', '--base', args.base,
                   '--config', str(study/'config.json')]),
        ('validation-selection', [sys.executable, '-m', 'training.select_policy_interpreter_v9', '--base', args.base]),
        ('adapted', [sys.executable, '-m', 'training.evaluate_interpreter_v9', '--base', args.base,
                     '--adapter', str(study/'selected-adapter'), '--data', screen, '--output', str(study/'adapted')]),
        ('archive', [sys.executable, '-m', 'training.archive_interpreter_v9', 'export',
                     '--input', str(study/'selected-adapter'), '--output', str(study/'selected-adapter.json.gz')]),
        ('diagnosis', [sys.executable, '-m', 'training.diagnose_interpreter_v9', '--data', screen,
                       '--predictions', str(study/'adapted/predictions.jsonl'),
                       '--reference', str(root/'interpreter-14/adapted/predictions.jsonl'),
                       '--output', str(study/'adapted/diagnosis.json')]),
        ('archive-verification', [sys.executable, '-m', 'training.verify_interpreter_archive_v9',
                     '--archive', str(study/'selected-adapter.json.gz'), '--source', str(study/'selected-adapter'),
                     '--output', str(study/'archive-verification.json')]),
    ]
    logs = Path(tempfile.gettempdir())/'roguard-v9-window'
    logs.mkdir(exist_ok=True)
    status = dict(stable_release_ready=False, reserved_probes_used=False, jobs=[])
    for name, command in jobs:
        remaining = deadline-time.time()
        if remaining <= 0:
            break
        print('START interpreter-policy-27', name, flush=True)
        entry = dict(name=name, started_utc=datetime.now(timezone.utc).isoformat())
        with (logs/('interpreter-policy-27-'+name+'.log')).open('w') as log:
            try:
                entry['exit_code'] = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=remaining).returncode
            except subprocess.TimeoutExpired:
                entry.update(exit_code=-1, timed_out=True)
        entry['finished_utc'] = datetime.now(timezone.utc).isoformat()
        status['jobs'].append(entry)
        status['complete'] = len(status['jobs']) == len(jobs) or entry['exit_code'] != 0
        pending = study/'run-status.tmp'
        pending.write_text(json.dumps(status, indent=2)+'\n')
        pending.replace(study/'run-status.json')
        print('FINISHED', json.dumps(entry), flush=True)
        if entry['exit_code'] != 0:
            break


if __name__ == '__main__':
    from training.coordinator_status_v9 import supervise
    supervise(main,'research/v9-stabilization/interpreter-policy-27')
