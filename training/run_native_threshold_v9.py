"""Run a predeclared CPU-only threshold audit after each native fit completes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def main():
    root = Path('research/v9-stabilization')
    study = root/'native-threshold-34'
    deadline = datetime.fromisoformat(json.loads((root/'work-window.json').read_text())['deadline_utc'].replace('Z', '+00:00')).timestamp()
    status = dict(jobs=[], stable_release_ready=False, reserved_probes_used=False)
    handled = set()
    logs = Path(tempfile.gettempdir())/'roguard-v9-window'
    logs.mkdir(exist_ok=True)
    while len(handled) < 2:
        if time.time() >= deadline:
            raise TimeoutError('Authorized work window ended')
        parent_status = root/'native-full-22-run-status.json'
        if not parent_status.exists():
            time.sleep(10); continue
        try:
            completed = json.loads(parent_status.read_text())['jobs']
        except json.JSONDecodeError:
            time.sleep(1); continue
        for job in completed:
            language = job['name'].rsplit('-', 1)[-1]
            if language in handled:
                continue
            if language not in ('ro', 'uk'):
                raise ValueError('Unexpected native language')
            for path, expected in json.loads((study/'source-freeze.json').read_text()).items():
                if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                    raise ValueError('Frozen threshold source changed: '+path)
            entry = dict(language=language, started_utc=datetime.now(timezone.utc).isoformat(), stages=[])
            if job['exit_code'] != 0:
                entry['skipped'] = 'Parent fit did not complete; not a pass'
            else:
                for stage in ('prepare', 'audit'):
                    command = [sys.executable, '-m', 'training.native_threshold_v9', stage,
                        '--candidate', str(root/('native-full-22-'+language)), '--language', language,
                        '--output', str(study/language)]
                    with (logs/('native-threshold-34-'+language+'-'+stage+'.log')).open('w') as log:
                        try:
                            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                                  timeout=max(1, deadline-time.time())).returncode
                        except subprocess.TimeoutExpired:
                            code = -1
                    entry['stages'].append(dict(stage=stage, exit_code=code))
                    if code != 0:
                        break
            entry['finished_utc'] = datetime.now(timezone.utc).isoformat()
            status['jobs'].append(entry); handled.add(language)
            temporary = study/'run-status.tmp'
            temporary.write_text(json.dumps(status, indent=2)+'\n'); temporary.replace(study/'run-status.json')
            print(json.dumps(entry), flush=True)
        if len(handled) < 2:
            time.sleep(10)


if __name__ == '__main__':
    main()
