"""Verify the selected structured interpreter after the native wheel replay."""
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
    for name in ('builder-python', 'mlx-python', 'base'):
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    root = Path.cwd().resolve()
    research = root/'research/v9-stabilization'
    study = research/'installed-interpreter-30'
    deadline = datetime.fromisoformat(json.loads((research/'work-window.json').read_text())['deadline_utc'].replace('Z', '+00:00')).timestamp()
    while True:
        if time.time() >= deadline:
            raise TimeoutError('Authorized work window ended')
        path = research/'installed-native-29/run-status.json'
        if path.exists():
            prior = json.loads(path.read_text())
            if 'finished_utc' in prior:
                if time.time() > datetime.fromisoformat(prior['finished_utc']).timestamp()+15:
                    break
        time.sleep(10)
    for path, expected in json.loads((study/'source-freeze.json').read_text()).items():
        if hashlib.sha256((root/path).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen interpreter runtime source changed: '+path)
    selection = json.loads((research/'interpreter-full-21/selection.json').read_text())
    number = selection['selected_study']
    if number not in (14, 27):
        raise ValueError('Unexpected selected study')
    candidate = research/('interpreter-14' if number == 14 else 'interpreter-policy-27')
    archive = candidate/'selected-adapter.json.gz'
    if hashlib.sha256(archive.read_bytes()).hexdigest() != selection['archive_sha256']:
        raise ValueError('Selected adapter differs from full evaluation')
    working = Path(tempfile.mkdtemp(prefix='roguard-installed-interpreter-30-'))
    wheels, installed = working/'wheels', working/'installed'
    logs = Path(tempfile.gettempdir())/'roguard-v9-window'
    logs.mkdir(exist_ok=True)
    status = dict(stable_release_ready=False, reserved_probes_used=False, selected_study=number, jobs=[])

    def save():
        temporary = study/'run-status.tmp'
        temporary.write_text(json.dumps(status, indent=2)+'\n')
        temporary.replace(study/'run-status.json')

    def run(name, command, cwd, env=None):
        remaining = deadline-time.time()
        if remaining <= 0:
            raise TimeoutError('Authorized work window ended')
        entry = dict(name=name, started_utc=datetime.now(timezone.utc).isoformat())
        print('START installed-interpreter-30', name, flush=True)
        with (logs/('installed-interpreter-30-'+name+'.log')).open('w') as log:
            try:
                entry['exit_code'] = subprocess.run(command, cwd=cwd, env=env, stdout=log,
                    stderr=subprocess.STDOUT, timeout=remaining).returncode
            except subprocess.TimeoutExpired:
                entry.update(exit_code=-1, timed_out=True)
        entry['finished_utc'] = datetime.now(timezone.utc).isoformat()
        status['jobs'].append(entry); save()
        print('FINISHED', json.dumps(entry), flush=True)
        return entry['exit_code'] == 0

    if run('build', [a.builder_python, '-m', 'build', '--wheel', '--no-isolation', '--outdir', str(wheels)], root):
        artifacts = list(wheels.glob('*.whl'))
        if len(artifacts) != 1:
            raise ValueError('Expected exactly one built wheel')
        wheel = artifacts[0]
        if run('install', [a.builder_python, '-m', 'pip', 'install', '--no-deps', '--no-index', '--target', str(installed), str(wheel)], working):
            env = dict(os.environ, PYTHONPATH=str(installed), TOKENIZERS_PARALLELISM='false',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
            run('screen', [a.mlx_python, str(root/'training/verify_interpreter_runtime_v9.py'),
                '--archive', str(archive), '--archive-sha256', selection['archive_sha256'], '--base', a.base,
                '--installed-root', str(installed), '--wheel', str(wheel),
                '--data', str(root/'research/v8-stabilization/instruction-sft-01/screen.jsonl'),
                '--reference', str(candidate/'adapted/predictions.jsonl'), '--output', str(study/'screen')], working, env)
    status['finished_utc'] = datetime.now(timezone.utc).isoformat(); save()


if __name__ == '__main__':
    main()
