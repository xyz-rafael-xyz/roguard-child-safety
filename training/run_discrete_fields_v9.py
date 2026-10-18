"""Check fixed categorical decisions after the focused semantic model replay."""
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
    root = Path.cwd().resolve()
    research = root / 'research/v9-stabilization'
    study = research / 'discrete-policy-69'
    deadline = datetime.fromisoformat(json.loads((research / 'work-window.json').read_text())[
        'deadline_utc'].replace('Z', '+00:00')).timestamp()
    while time.time() < deadline:
        path = research / 'scope-repair-66/run-status.json'
        if path.exists() and 'finished_utc' in json.loads(path.read_text()):
            break
        time.sleep(10)
    else:
        raise TimeoutError('Authorized work window ended')
    for name, expected in json.loads((study / 'source-freeze.json').read_text()).items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Declared categorical comparison changed: ' + name)
    status = dict(jobs=[], synthetic_only=True, reserved_probes_used=False, stable_release_ready=False)
    working = Path(tempfile.mkdtemp(prefix='roguard-discrete-policy-69-'))
    for language in ('ro', 'uk'):
        if not (research / 'scope-repair-66' / language / 'audit.json').exists():
            status.setdefault('skipped_languages', []).append(language)
            continue
        for stage in ('research', 'installed'):
            command = [sys.executable, str(root / 'training/evaluate_discrete_fields_v9.py'),
                       stage, '--root', str(research), '--language', language]
            if stage == 'installed':
                path = research / 'installed-scope-repair-67' / language / 'report.json'
                if not path.exists():
                    status.setdefault('installed_missing', []).append(language)
                    break
                report = json.loads(path.read_text())
                installed = Path(report['installed_module']).parent.parent
                command.extend(['--installed-root', str(installed)])
                env = dict(os.environ, PYTHONPATH=str(installed))
            else:
                env = dict(os.environ, PYTHONPATH=str(root / 'src'))
            entry = dict(language=language, stage=stage, started_utc=datetime.now(timezone.utc).isoformat())
            entry['exit_code'] = subprocess.run(command, cwd=working, env=env,
                timeout=max(1, deadline-time.time())).returncode
            entry['finished_utc'] = datetime.now(timezone.utc).isoformat()
            status['jobs'].append(entry)
            (study / 'run-status.json').write_text(json.dumps(status, indent=2) + '\n')
            if entry['exit_code']:
                break
    status['finished_utc'] = datetime.now(timezone.utc).isoformat()
    (study / 'run-status.json').write_text(json.dumps(status, indent=2) + '\n')


if __name__ == '__main__':
    from training.coordinator_status_v9 import supervise
    supervise(main, 'research/v9-stabilization/discrete-policy-69')
