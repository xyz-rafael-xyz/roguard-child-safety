"""Run the frozen adaptive-decoding comparison after capacity experiments."""
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
    root = Path('research/v9-stabilization'); study = root/'policy-decode-16'
    deadline = datetime.fromisoformat(json.loads((root/'work-window.json').read_text())['deadline_utc'].replace('Z', '+00:00')).timestamp()
    while True:
        if time.time() >= deadline: raise TimeoutError('Authorized work window ended')
        path = root/'capacity-15-run-status.json'
        if path.exists():
            jobs = json.loads(path.read_text())['jobs']
            if len(jobs) == 4 and time.time() > datetime.fromisoformat(jobs[-1]['finished_utc']).timestamp()+15:
                break
        time.sleep(10)
    for path, digest in json.loads((study/'source-freeze.json').read_text()).items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen source changed: '+path)
    logs = Path(tempfile.gettempdir())/'roguard-v9-window'; logs.mkdir(exist_ok=True)
    status = dict(stable_release_ready=False, reserved_probes_used=False, jobs=[])
    for name in ('baseline', 'adapted'):
        if name == 'adapted' and not (root/'interpreter-14/selected-adapter/adapters.safetensors').exists():
            status['adapted_skipped'] = 'No completed selected adapter'; break
        command = [sys.executable, '-m', 'training.evaluate_policy_decode_v9', '--base', a.base,
                   '--data', 'research/v8-stabilization/instruction-sft-01/screen.jsonl',
                   '--output', str(study/name), '--max-queries', '32']
        if name == 'adapted': command += ['--adapter', str(root/'interpreter-14/selected-adapter')]
        remaining = deadline-time.time()
        if remaining <= 0: break
        print('START policy-decode-16', name, flush=True)
        entry = dict(name=name, started_utc=datetime.now(timezone.utc).isoformat())
        with (logs/('policy-decode-16-'+name+'.log')).open('w') as log:
            try: entry['exit_code'] = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=remaining).returncode
            except subprocess.TimeoutExpired: entry['exit_code'] = -1; entry['timed_out'] = True
        entry['finished_utc'] = datetime.now(timezone.utc).isoformat(); status['jobs'].append(entry)
        (study/'run-status.json').write_text(json.dumps(status, indent=2)+'\n')
        print('FINISHED', json.dumps(entry), flush=True)
        if entry['exit_code'] != 0: break
    (study/'run-status.json').write_text(json.dumps(status, indent=2)+'\n')


if __name__ == '__main__': main()
