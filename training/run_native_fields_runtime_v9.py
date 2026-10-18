"""Serialize installed semantic-head replay after all feature extraction ends."""
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
    for name in ('builder-python', 'torch-python', 'ro-base', 'uk-base'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    research = root / 'research/v9-stabilization'
    study = research / 'installed-native-fields-45'
    deadline = datetime.fromisoformat(json.loads((research / 'work-window.json').read_text())[
        'deadline_utc'].replace('Z', '+00:00')).timestamp()
    while time.time() < deadline:
        path = research / 'native-fields-43/run-status.json'
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

    languages = [language for language in ('ro', 'uk')
                 if (research / 'native-fields-43' / language / 'audit.json').exists()]
    if not languages:
        status['skipped'] = 'No completed semantic-head audit is available.'
    else:
        working = Path(tempfile.mkdtemp(prefix='roguard-installed-native-fields-45-'))
        wheels, installed = working / 'wheels', working / 'installed'
        logs = Path(tempfile.gettempdir()) / 'roguard-v9-window'
        logs.mkdir(exist_ok=True)

        def run(name, command, cwd, env=None):
            if time.time() >= deadline:
                raise TimeoutError('Authorized work window ended')
            entry = dict(name=name, started_utc=datetime.now(timezone.utc).isoformat())
            print('START installed-native-fields-45', name, flush=True)
            with (logs / ('installed-native-fields-45-' + name + '.log')).open('w') as stream:
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
                    run(language, [args.torch_python, str(root / 'training/verify_native_fields_runtime_v9.py'),
                        '--root', str(research), '--base', getattr(args, language + '_base'),
                        '--installed-root', str(installed), '--wheel', str(wheel),
                        '--output', str(study / language), '--language', language], working, env)
    status['finished_utc'] = datetime.now(timezone.utc).isoformat()
    save()


if __name__ == '__main__':
    from training.coordinator_status_v9 import supervise
    supervise(main, 'research/v9-stabilization/installed-native-fields-45')
