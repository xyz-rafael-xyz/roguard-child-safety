"""Evaluate fixed auxiliary fields as each parent language completes training."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    root = Path('research/v9-stabilization')
    study = root / 'multitask-policy-65'
    deadline = datetime.fromisoformat(json.loads((root / 'work-window.json').read_text())[
        'deadline_utc'].replace('Z', '+00:00')).timestamp()
    status = dict(jobs=[], reserved_probes_used=False, stable_release_ready=False)
    for language in ('ro', 'uk'):
        while not (root / 'multitask-review-61' / language / 'development.json').exists():
            if time.time() >= deadline:
                raise TimeoutError('Authorized work window ended')
            parent_status = root / 'multitask-review-61/run-status.json'
            if parent_status.exists() and 'finished_utc' in json.loads(parent_status.read_text()):
                raise ValueError('Parent ended without complete language evidence')
            time.sleep(10)
        for stage in ('fit', 'audit'):
            entry = dict(language=language, stage=stage, started_utc=datetime.now(timezone.utc).isoformat())
            entry['exit_code'] = subprocess.run([sys.executable, '-m',
                'training.select_multitask_fields_v9', stage, '--language', language],
                timeout=max(1, deadline-time.time())).returncode
            entry['finished_utc'] = datetime.now(timezone.utc).isoformat()
            status['jobs'].append(entry)
            (study / 'run-status.json').write_text(json.dumps(status, indent=2) + '\n')
            if entry['exit_code']:
                raise ValueError('Auxiliary policy selection failed')
    status['finished_utc'] = datetime.now(timezone.utc).isoformat()
    (study / 'run-status.json').write_text(json.dumps(status, indent=2) + '\n')


if __name__ == '__main__':
    from training.coordinator_status_v9 import supervise
    supervise(main, 'research/v9-stabilization/multitask-policy-65')
