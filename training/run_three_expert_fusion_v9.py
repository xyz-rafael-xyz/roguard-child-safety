"""Run the frozen lightweight fusion experiment after full neural predictions."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def main():
    root = Path('research/v9-stabilization'); study = root/'three-expert-fusion-76'
    deadline = datetime.fromisoformat(json.loads((root/'work-window.json').read_text())['deadline_utc'].replace('Z', '+00:00')).timestamp()
    status = dict(jobs=[], reserved_probes_used=False, stable_release_ready=False)
    while True:
        if time.time() >= deadline:
            raise TimeoutError('Authorized work window ended')
        path = root/'interpreter-full-21/run-status.json'
        if path.exists():
            try:
                jobs = json.loads(path.read_text())['jobs']
            except json.JSONDecodeError:
                time.sleep(1); continue
            if jobs and (len(jobs) == 2 or jobs[-1]['exit_code'] != 0):
                break
        time.sleep(10)
    if len(jobs) != 2 or any(job['exit_code'] != 0 for job in jobs):
        status['skipped'] = 'Full interpreter predictions unavailable; not a pass'
    else:
        for path, expected in json.loads((study/'source-freeze.json').read_text()).items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise ValueError('Frozen fusion source changed: '+path)
        logs = Path(tempfile.gettempdir())/'roguard-v9-window'; logs.mkdir(exist_ok=True)
        for language in ('ro', 'uk'):
            parent = root/('native-full-22-'+language)/'development.json'
            if not parent.exists() or 'calibration_audit' not in json.loads(parent.read_text()):
                status.setdefault('skipped_languages', []).append(language); continue
            for stage in ('prepare', 'audit'):
                command = [sys.executable, '-m', 'training.three_expert_fusion_v9', stage, '--language', language]
                entry = dict(language=language, stage=stage, started_utc=datetime.now(timezone.utc).isoformat())
                with (logs/('three-expert-fusion-76-'+language+'-'+stage+'.log')).open('w') as log:
                    try:
                        entry['exit_code'] = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                            timeout=max(1, deadline-time.time())).returncode
                    except subprocess.TimeoutExpired:
                        entry['exit_code'] = -1
                entry['finished_utc'] = datetime.now(timezone.utc).isoformat(); status['jobs'].append(entry)
                temporary = study/'run-status.tmp'
                temporary.write_text(json.dumps(status, indent=2)+'\n'); temporary.replace(study/'run-status.json')
                print(json.dumps(entry), flush=True)
                if entry['exit_code'] != 0:
                    break
    status['finished_utc'] = datetime.now(timezone.utc).isoformat()
    temporary = study/'run-status.tmp'
    temporary.write_text(json.dumps(status, indent=2)+'\n'); temporary.replace(study/'run-status.json')


if __name__ == '__main__':
    main()
