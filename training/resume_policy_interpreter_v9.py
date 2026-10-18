"""Resume completed policy training at checkpoint selection after semantic runs."""
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    args = parser.parse_args()
    root = Path('research/v9-stabilization'); study = root/'interpreter-policy-27'
    recovery = root/'policy-selection-recovery-60'
    deadline = datetime.fromisoformat(json.loads((root/'work-window.json').read_text())[
        'deadline_utc'].replace('Z', '+00:00')).timestamp()-5100
    while time.time() < deadline:
        previous = root/'multitask-review-61/run-status.json'
        if previous.exists():
            status = json.loads(previous.read_text())
            if ('finished_utc' in status and time.time() >
                    datetime.fromisoformat(status['finished_utc']).timestamp()+15): break
        time.sleep(10)
    else: raise TimeoutError('Recovery budget exhausted before model loading')
    for name, expected in json.loads((recovery/'source-freeze.json').read_text()).items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen recovery source or training artifact changed: '+name)
    status = json.loads((study/'run-status.json').read_text())
    if not any(j['name']=='train' and j['exit_code']==0 for j in status['jobs']):
        raise ValueError('Completed training is required for selection-only recovery')
    if (study/'validation-selection.json').exists(): raise ValueError('Selection already completed')
    screen = 'research/v8-stabilization/instruction-sft-01/screen.jsonl'
    jobs = [
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
    result = dict(jobs=[], stable_release_ready=False, reserved_probes_used=False)
    for index, (name, command) in enumerate(jobs):
        if time.time() >= deadline: raise TimeoutError('Recovery budget exhausted')
        entry = dict(name=name, recovery=60, started_utc=datetime.now(timezone.utc).isoformat())
        print('START recovery60', name, flush=True)
        with (logs/('policy60-'+name+'.log')).open('w') as log:
            try:
                entry['exit_code'] = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                    timeout=max(1, deadline-time.time())).returncode
            except subprocess.TimeoutExpired: entry.update(exit_code=-1, timed_out=True)
        entry['finished_utc'] = datetime.now(timezone.utc).isoformat()
        status['jobs'].append(entry); result['jobs'].append(entry)
        status['complete'] = index == len(jobs)-1 or entry['exit_code'] != 0
        for path, value in ((study/'run-status.json', status), (recovery/'run-status.json', result)):
            temporary = path.with_suffix('.tmp'); temporary.write_text(json.dumps(value, indent=2)+'\n'); temporary.replace(path)
        print(json.dumps(entry), flush=True)
        if entry['exit_code'] != 0: break
    result['finished_utc'] = datetime.now(timezone.utc).isoformat()
    (recovery/'run-status.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    from training.coordinator_status_v9 import supervise
    supervise(main, 'research/v9-stabilization/policy-selection-recovery-60')
