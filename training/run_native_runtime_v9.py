"""Build an isolated wheel and replay native checkpoints after neural training."""
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
    p = argparse.ArgumentParser()
    for name in ('builder-python', 'torch-python', 'ro-base', 'uk-base'):
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    root = Path.cwd().resolve()
    research = root/'research/v9-stabilization'
    study = research/'installed-native-29'
    deadline = datetime.fromisoformat(json.loads((research/'work-window.json').read_text())['deadline_utc'].replace('Z', '+00:00')).timestamp()
    while True:
        if time.time() >= deadline:
            raise TimeoutError('Authorized work window ended')
        path = research/'interpreter-full-21/run-status.json'
        if path.exists():
            jobs = json.loads(path.read_text())['jobs']
            if jobs and (len(jobs) == 2 or jobs[-1]['exit_code'] != 0):
                if time.time() > datetime.fromisoformat(jobs[-1]['finished_utc']).timestamp()+15:
                    break
        time.sleep(10)
    for path, expected in json.loads((study/'source-freeze.json').read_text()).items():
        if hashlib.sha256((root/path).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen runtime source changed: '+path)
    working = Path(tempfile.mkdtemp(prefix='roguard-installed-native-29-'))
    wheels = working/'wheels'
    installed = working/'installed'
    logs = Path(tempfile.gettempdir())/'roguard-v9-window'
    logs.mkdir(exist_ok=True)
    status = dict(stable_release_ready=False, reserved_probes_used=False, jobs=[])

    def run(name, command, cwd, env=None):
        remaining = deadline-time.time()
        if remaining <= 0:
            raise TimeoutError('Authorized work window ended')
        entry = dict(name=name, started_utc=datetime.now(timezone.utc).isoformat())
        print('START installed-native-29', name, flush=True)
        with (logs/('installed-native-29-'+name+'.log')).open('w') as log:
            try:
                entry['exit_code'] = subprocess.run(command, cwd=cwd, env=env, stdout=log,
                    stderr=subprocess.STDOUT, timeout=remaining).returncode
            except subprocess.TimeoutExpired:
                entry.update(exit_code=-1, timed_out=True)
        entry['finished_utc'] = datetime.now(timezone.utc).isoformat()
        status['jobs'].append(entry)
        temporary = study/'run-status.tmp'
        temporary.write_text(json.dumps(status, indent=2)+'\n')
        temporary.replace(study/'run-status.json')
        print('FINISHED', json.dumps(entry), flush=True)
        return entry['exit_code'] == 0

    if not run('build', [a.builder_python, '-m', 'build', '--wheel', '--no-isolation', '--outdir', str(wheels)], root):
        return
    artifacts = list(wheels.glob('*.whl'))
    if len(artifacts) != 1:
        raise ValueError('Expected exactly one built wheel')
    wheel = artifacts[0]
    if not run('install', [a.builder_python, '-m', 'pip', 'install', '--no-deps', '--no-index', '--target', str(installed), str(wheel)], working):
        return
    env = dict(os.environ, PYTHONPATH=str(installed), TOKENIZERS_PARALLELISM='false',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    for language, base in [('ro', a.ro_base), ('uk', a.uk_base)]:
        candidate = research/('native-full-22-'+language)
        record = candidate/'development.json'
        if not record.exists() or 'calibration_audit' not in json.loads(record.read_text()):
            status.setdefault('skipped', {})[language] = 'No completed native checkpoint; not a pass'
            continue
        run(language, [a.torch_python, str(root/'training/verify_native_runtime_v9.py'),
            '--candidate', str(candidate), '--base', base, '--manifest', str(research/'native-full-22-base-manifest.json'),
            '--installed-root', str(installed), '--wheel', str(wheel), '--output', str(study/language)], working, env)
    status['finished_utc'] = datetime.now(timezone.utc).isoformat()
    (study/'run-status.json').write_text(json.dumps(status, indent=2)+'\n')


if __name__ == '__main__':
    main()
