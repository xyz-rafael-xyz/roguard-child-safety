"""Fit native expert combinations without loading additional neural models."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    root = Path('research/v9-stabilization')
    study = root / 'native-expert-fusion-75'
    deadline = datetime.fromisoformat(json.loads((root / 'work-window.json').read_text())[
        'deadline_utc'].replace('Z', '+00:00')).timestamp()
    for name, expected in json.loads((study / 'source-freeze.json').read_text()).items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError('Native expert fusion recipe changed')
    status = dict(jobs=[], stable_release_ready=False, reserved_probes_used=False)
    for language, parent in (('uk', 'scope-repair-66'), ('ro', 'semantic-selection-71')):
        while not (root / parent / language / 'audit.json').exists():
            if time.time() >= deadline:
                raise TimeoutError('Authorized work window ended')
            path = root / parent / 'run-status.json'
            if path.exists() and 'finished_utc' in json.loads(path.read_text()):
                status.setdefault('skipped_languages', []).append(language)
                break
            time.sleep(10)
        else:
            for stage in ('prepare', 'audit'):
                entry = dict(language=language, stage=stage, started_utc=datetime.now(timezone.utc).isoformat())
                entry['exit_code'] = subprocess.run([sys.executable, '-m', 'training.native_expert_fusion_v9',
                    stage, '--language', language], timeout=max(1, deadline-time.time())).returncode
                entry['finished_utc'] = datetime.now(timezone.utc).isoformat()
                status['jobs'].append(entry)
                (study / 'run-status.json').write_text(json.dumps(status, indent=2) + '\n')
                if entry['exit_code']:
                    break
    status['finished_utc'] = datetime.now(timezone.utc).isoformat()
    (study / 'run-status.json').write_text(json.dumps(status, indent=2) + '\n')


if __name__ == '__main__':
    from training.coordinator_status_v9 import supervise
    supervise(main, 'research/v9-stabilization/native-expert-fusion-75')
