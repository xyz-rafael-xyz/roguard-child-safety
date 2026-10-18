"""Evaluate reserved inputs only after both installed categorical models qualify."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('torch-python', 'ro-base', 'uk-base'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    repo = Path.cwd().resolve()
    root = repo / 'research/v9-stabilization'
    study = root / 'reserved-discrete-policy-70'
    deadline = datetime.fromisoformat(json.loads((root / 'work-window.json').read_text())[
        'deadline_utc'].replace('Z', '+00:00')).timestamp() - 5100
    while time.time() < deadline:
        path = root / 'discrete-policy-69/run-status.json'
        if path.exists() and 'finished_utc' in json.loads(path.read_text()):
            break
        time.sleep(10)
    status = dict(jobs=[], stable_release_ready=False, reserved_probes_used=False)

    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def save():
        temporary = study / 'run-status.tmp'
        temporary.write_text(json.dumps(status, indent=2) + '\n')
        temporary.replace(study / 'run-status.json')

    for name, expected in json.loads((study / 'source-freeze.json').read_text()).items():
        if sha(repo / name) != expected:
            raise ValueError('Frozen categorical verifier changed')
    from training.verify_reserved_discrete_fields_v9 import checked_report
    candidates = {}
    installed_roots = set()
    for language in ('ro', 'uk'):
        report_path = root / 'discrete-policy-69' / language / 'installed/report.json'
        if not report_path.exists():
            continue
        parent = root / 'scope-repair-66'
        data = parent / 'encoders' / language
        heads = parent / language / 'selected-heads.json.gz'
        selection = root / 'discrete-policy-69' / language / 'selection.json'
        entry = dict(runtime_report=str(report_path), runtime_report_sha256=sha(report_path),
            artifact=str(heads), artifact_sha256=sha(heads), native_artifact=str(data / 'review.json.gz'),
            native_artifact_sha256=sha(data / 'review.json.gz'), selection=str(selection),
            selection_sha256=sha(selection), checkpoint=str(data / 'selected-model.safetensors'),
            base=getattr(args, language + '_base'), manifest=str(root / 'native-full-22-base-manifest.json'))
        try:
            report = checked_report(entry)
            if report['language'] != language:
                raise ValueError('Language differs')
        except ValueError:
            continue
        installed_roots.add(str(Path(report['installed_module']).parent.parent))
        candidates[language] = entry
    if len(candidates) != 2:
        status['skipped'] = 'Both categorical models must pass installed accuracy and parity gates.'
    elif len(installed_roots) != 1:
        raise ValueError('Categorical models use different installed packages')
    elif deadline - time.time() < 180:
        status['skipped'] = 'Insufficient reserved inference time before interpreter recovery.'
    else:
        installed = next(iter(installed_roots))
        plan = dict(candidates=candidates,
            verifier_source_sha256={name: sha(repo / 'training' / name) for name in
                ('verify_reserved_discrete_fields_v9.py', 'reserved_metrics_v9.py')},
            runtime_source_sha256={'roguard.' + name: sha(repo / 'src/roguard' / (name + '.py')) for name in
                ('discrete_fields', 'native_fields', 'native_review', 'native_assets', 'assurance', 'evidence')})
        working = Path(tempfile.mkdtemp(prefix='roguard-reserved-discrete-70-'))
        path = working / 'plan.json'
        path.write_text(json.dumps(plan, indent=2) + '\n')
        status.update(reserved_evaluation_attempted=True, plan_sha256=sha(path))
        save()
        entry = dict(name='reserved', started_utc=datetime.now(timezone.utc).isoformat())
        command = [args.torch_python, str(repo / 'training/verify_reserved_discrete_fields_v9.py'),
            '--plan', str(path), '--plan-sha256', sha(path), '--installed-root', installed,
            '--probe-root', str(root.parent / 'v8-stabilization'), '--output', str(study / 'results')]
        with (working / 'reserved.log').open('w') as log:
            try:
                entry['exit_code'] = subprocess.run(command, cwd=working,
                    env=dict(os.environ, PYTHONPATH=installed, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                             TOKENIZERS_PARALLELISM='false'), stdout=log, stderr=subprocess.STDOUT,
                    timeout=max(1, deadline-time.time())).returncode
            except subprocess.TimeoutExpired:
                entry.update(exit_code=-1, timed_out=True)
        entry['finished_utc'] = datetime.now(timezone.utc).isoformat()
        status['jobs'].append(entry)
        status['reserved_probes_used'] = (study / 'results/selection.json').exists()
        status['reserved_predictions_complete'] = (study / 'results/predictions.jsonl').exists()
    status['finished_utc'] = datetime.now(timezone.utc).isoformat()
    save()


if __name__ == '__main__':
    from training.coordinator_status_v9 import supervise
    supervise(main, 'research/v9-stabilization/reserved-discrete-policy-70')
