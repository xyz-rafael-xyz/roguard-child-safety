"""Train joint semantic models, verify installed execution, then evaluate qualified reserved inputs."""
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
    study = research / 'scope-repair-66'
    deadline = datetime.fromisoformat(json.loads((research / 'work-window.json').read_text())[
        'deadline_utc'].replace('Z', '+00:00')).timestamp() - 5400
    while time.time() < deadline:
        path = research / 'multitask-review-61/run-status.json'
        if path.exists():
            prior = json.loads(path.read_text())
            if ('finished_utc' in prior and time.time() >
                    datetime.fromisoformat(prior['finished_utc']).timestamp() + 15):
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
    if deadline-time.time() < 1200:
        status['skipped'] = 'Less than20minutes remain for representation comparison and replay.'
    else:
        working = Path(tempfile.mkdtemp(prefix='roguard-scope-repair-66-'))
        wheels, installed = working / 'wheels', working / 'installed'
        logs = Path(tempfile.gettempdir()) / 'roguard-v9-window'
        logs.mkdir(exist_ok=True)

        def run(name, command, cwd, env=None):
            if time.time() >= deadline:
                raise TimeoutError('Authorized work window ended')
            entry = dict(name=name, started_utc=datetime.now(timezone.utc).isoformat())
            print('START scope-repair-66', name, flush=True)
            with (logs / ('scope-repair-66-' + name + '.log')).open('w') as stream:
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
            base = getattr(args, language + '_base')
            stages = [('train', ['training.train_scope_repair_v9', '--base', base])]
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
                    run(language, [args.torch_python, str(root / 'training/verify_scope_repair_runtime_v9.py'),
                        '--root', str(research), '--base', getattr(args, language + '_base'),
                        '--installed-root', str(installed), '--wheel', str(wheel),
                        '--output', str(research / 'installed-scope-repair-67' / language), '--language', language], working, env)
        from training.verify_reserved_native_fields_v9 import checked_report
        def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
        candidates = {}
        for language in ('ro', 'uk'):
            report = research / 'installed-scope-repair-67' / language / 'report.json'
            heads = study / language / 'selected-heads.json.gz'
            selection = study / language / 'selection.json'
            parent = study / 'encoders' / language
            if not all(p.exists() for p in (report, heads, selection, parent / 'review.json.gz')):
                continue
            entry = dict(runtime_report=str(report), runtime_report_sha256=digest(report),
                artifact=str(heads), artifact_sha256=digest(heads),
                native_artifact=str(parent / 'review.json.gz'), native_artifact_sha256=digest(parent / 'review.json.gz'),
                selection=str(selection), selection_sha256=digest(selection),
                checkpoint=str(parent / 'selected-model.safetensors'), base=getattr(args, language + '_base'),
                manifest=str(research / 'native-full-22-base-manifest.json'))
            try: checked_report(entry)
            except ValueError: continue
            candidates[language] = entry
        if len(candidates) == 2:
            plan = dict(candidates=candidates,
                verifier_source_sha256={name:digest(root / 'training' / name) for name in
                    ('verify_reserved_native_fields_v9.py','reserved_metrics_v9.py')},
                runtime_source_sha256={'roguard.'+name:digest(root / 'src/roguard' / (name+'.py')) for name in
                    ('native_fields','native_review','native_assets','assurance','evidence')})
            plan_path = working / 'reserved-plan.json'
            plan_path.write_text(json.dumps(plan, indent=2)+'\n')
            status['reserved_plan_sha256'] = digest(plan_path); save()
            status['reserved_evaluation_attempted'] = True; save()
            run('reserved', [args.torch_python, str(root / 'training/verify_reserved_native_fields_v9.py'),
                '--plan', str(plan_path), '--plan-sha256', digest(plan_path), '--installed-root', str(installed),
                '--probe-root', str(research.parent / 'v8-stabilization'),
                '--output', str(research / 'reserved-scope-repair-68/results')], working,
                dict(os.environ, PYTHONPATH=str(installed), TOKENIZERS_PARALLELISM='false',
                     HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1'))
            status['reserved_probes_used'] = (research / 'reserved-scope-repair-68/results/selection.json').exists()
            status['reserved_predictions_complete'] = (research / 'reserved-scope-repair-68/results/predictions.jsonl').exists()
        else:
            status['reserved_skipped'] = 'Both installed candidates must pass unchanged accuracy and parity gates.'
    status['finished_utc'] = datetime.now(timezone.utc).isoformat()
    save()


if __name__ == '__main__':
    from training.coordinator_status_v9 import supervise
    supervise(main, 'research/v9-stabilization/scope-repair-66')
