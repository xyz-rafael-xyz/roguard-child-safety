"""Fit field mixtures and replay them using already installed neural outputs."""
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
    for name in ('builder-python', 'torch-python'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    research = root / 'research/v9-stabilization'
    study = research / 'field-fusion-51'
    deadline = datetime.fromisoformat(json.loads((research / 'work-window.json').read_text())[
        'deadline_utc'].replace('Z', '+00:00')).timestamp()
    while time.time() < deadline:
        path = research / 'pretrained-fields-47/run-status.json'
        if path.exists():
            prior = json.loads(path.read_text())
            if 'finished_utc' in prior and time.time() > datetime.fromisoformat(prior['finished_utc']).timestamp() + 15:
                break
        time.sleep(10)
    else:
        raise TimeoutError('Authorized work window ended')
    for path, expected in json.loads((study / 'source-freeze.json').read_text()).items():
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen installed semantic-head source changed: ' + path)
    status = dict(jobs=[], stable_release_ready=False, reserved_probes_used=False)

    def save():
        temporary = study / 'run-status.tmp'
        temporary.write_text(json.dumps(status, indent=2) + '\n')
        temporary.replace(study / 'run-status.json')

    languages = ['ro', 'uk']
    if deadline-time.time() < 120:
        status['skipped'] = 'Less than two minutes remain for mixture fitting and replay.'
    else:
        working = Path(tempfile.mkdtemp(prefix='roguard-field-fusion-51-'))
        wheels, installed = working / 'wheels', working / 'installed'
        logs = Path(tempfile.gettempdir()) / 'roguard-v9-window'
        logs.mkdir(exist_ok=True)

        def run(name, command, cwd, env=None):
            if time.time() >= deadline:
                raise TimeoutError('Authorized work window ended')
            entry = dict(name=name, started_utc=datetime.now(timezone.utc).isoformat())
            print('START field-fusion-51', name, flush=True)
            with (logs / ('field-fusion-51-' + name + '.log')).open('w') as stream:
                try:
                    entry['exit_code'] = subprocess.run(command, cwd=cwd, env=env, stdout=stream,
                        stderr=subprocess.STDOUT, timeout=max(1, deadline-time.time())).returncode
                except subprocess.TimeoutExpired:
                    entry.update(exit_code=-1, timed_out=True)
            entry['finished_utc'] = datetime.now(timezone.utc).isoformat()
            status['jobs'].append(entry)
            save()
            print(json.dumps(entry), flush=True)
            return entry['exit_code'] == 0

        env = dict(os.environ, PYTHONPATH=str(root / 'src') + os.pathsep + str(root),
                   TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   PYTORCH_MPS_HIGH_WATERMARK_RATIO='.7', PYTORCH_MPS_LOW_WATERMARK_RATIO='.4')
        completed = []
        for language in languages:
            stages = [('fit', ['training.fit_field_fusion_v9', 'fit']),
                      ('audit', ['training.fit_field_fusion_v9', 'audit'])]
            for name, command in stages:
                if not run(language + '-' + name, [args.torch_python, '-m', *command, '--language', language], root, env):
                    break
            else:
                completed.append(language)
        languages = completed
        if run('build', [args.builder_python, '-m', 'build', '--wheel', '--no-isolation', '--outdir', str(wheels)], root):
            artifacts = list(wheels.glob('*.whl'))
            if len(artifacts) != 1:
                raise ValueError('Expected exactly one wheel')
            wheel = artifacts[0]
            if run('install', [args.builder_python, '-m', 'pip', 'install', '--no-deps', '--no-index',
                              '--target', str(installed), str(wheel)], working):
                env = dict(os.environ, PYTHONPATH=str(installed), TOKENIZERS_PARALLELISM='false',
                           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
                for language in languages:
                    run(language, [args.torch_python, str(root / 'training/verify_field_fusion_runtime_v9.py'),
                        '--root', str(research),
                        '--installed-root', str(installed), '--wheel', str(wheel),
                        '--output', str(research / 'installed-field-fusion-52' / language), '--language', language], working, env)
    status['finished_utc'] = datetime.now(timezone.utc).isoformat()
    save()


if __name__ == '__main__':
    from training.coordinator_status_v9 import supervise
    supervise(main, 'research/v9-stabilization/field-fusion-51')
