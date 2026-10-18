"""Wait for installed neural replay, then verify the installed review bridge."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def main():
    repo = Path.cwd().resolve(); root = repo/'research/v9-stabilization'
    study = root/'installed-interpreter-assurance-37'
    deadline = datetime.fromisoformat(json.loads((root/'work-window.json').read_text())['deadline_utc'].replace('Z', '+00:00')).timestamp()
    status = dict(stable_release_ready=False, reserved_probes_used=False)
    while True:
        if time.time() >= deadline:
            raise TimeoutError('Authorized work window ended')
        path = root/'installed-interpreter-30/run-status.json'
        if path.exists():
            previous = json.loads(path.read_text())
            if 'finished_utc' in previous:
                break
        time.sleep(10)
    for path, expected in json.loads((study/'source-freeze.json').read_text()).items():
        if hashlib.sha256((repo/path).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen bridge source changed: '+path)
    report = root/'installed-interpreter-30/screen/report.json'
    if not report.exists() or json.loads(report.read_text()).get('parity_pass') is not True:
        status['skipped'] = 'Installed neural replay did not pass; not a pass'
    else:
        evidence = json.loads(report.read_text())
        installed = Path(evidence['installed_module']).parent.parent
        if not installed.is_dir():
            raise ValueError('Isolated wheel installation is unavailable')
        command = [sys.executable, str(repo/'training/verify_interpreter_assurance_v9.py'),
            '--installed-root', str(installed), '--data', str(repo/'research/v8-stabilization/instruction-sft-01/screen.jsonl'),
            '--runtime-report', str(report), '--predictions', str(report.parent/'predictions.jsonl'),
            '--output', str(study/'replay')]
        log = Path(tempfile.gettempdir())/'roguard-v9-window/installed-interpreter-assurance-37.log'
        with log.open('w') as stream:
            try:
                status['exit_code'] = subprocess.run(command, cwd=installed.parent,
                    env=dict(os.environ, PYTHONPATH=str(installed)), stdout=stream, stderr=subprocess.STDOUT,
                    timeout=max(1, deadline-time.time())).returncode
            except subprocess.TimeoutExpired:
                status['exit_code'] = -1
    status['finished_utc'] = datetime.now(timezone.utc).isoformat()
    (study/'run-status.json').write_text(json.dumps(status, indent=2)+'\n')
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
