"""Run reserved pipeline checks only after both languages qualify."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('torch-python', 'ro-base', 'uk-base'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve(); research = root/'research/v9-stabilization'
    study = research/'reserved-joint-field-fusion-59'
    deadline = datetime.fromisoformat(json.loads((research/'work-window.json').read_text())[
        'deadline_utc'].replace('Z', '+00:00')).timestamp()-5700
    while time.time() < deadline:
        prior = research/'joint-field-fusion-57/run-status.json'
        if prior.exists() and 'finished_utc' in json.loads(prior.read_text()): break
        time.sleep(10)
    status = dict(jobs=[], stable_release_ready=False, reserved_probes_used=False)
    def save():
        temporary = study/'run-status.tmp'
        temporary.write_text(json.dumps(status, indent=2)+'\n')
        temporary.replace(study/'run-status.json')
    for name, expected in json.loads((study/'source-freeze.json').read_text()).items():
        if sha(root/name) != expected: raise ValueError('Reserved pipeline source changed')
    from training.verify_reserved_joint_field_fusion_v9 import checked_report
    candidates = {}; installed_roots = set()
    for language in ('ro', 'uk'):
        parent = research/'joint-field-fusion-57'/language
        report = research/'installed-joint-field-fusion-58'/language/'report.json'
        if not report.exists() or not (parent/'selection.json').exists(): continue
        entry = dict(runtime_report=str(report), runtime_report_sha256=sha(report),
            artifact=str(parent/'fusion.json'), artifact_sha256=sha(parent/'fusion.json'),
            selection=str(parent/'selection.json'), selection_sha256=sha(parent/'selection.json'))
        try: checked = checked_report(entry)
        except ValueError: continue
        if checked['language'] != language: raise ValueError('Candidate language differs')
        installed_roots.add(Path(checked['installed_module']).parents[1])
        entry['components'] = {}
        for name, heads, encoder in (
                ('native', research/'native-fields-43'/language, research/('native-full-22-'+language)),
                ('joint', research/'multitask-fields-54'/language, research/'multitask-fields-54/encoders'/language)):
            entry['components'][name] = dict(heads=str(heads/'selected-heads.json.gz'),
                artifact=str(encoder/'review.json.gz'), checkpoint=str(encoder/'selected-model.safetensors'),
                base=getattr(args, language+'_base'), manifest=str(research/'native-full-22-base-manifest.json'))
        candidates[language] = entry
    if len(candidates) != 2:
        status['skipped'] = 'Both installed language candidates must pass unchanged accuracy and parity gates.'
    elif time.time() >= deadline:
        status['skipped'] = 'Reserved evaluation time budget exhausted; inputs remain unopened.'
    else:
        if len(installed_roots) != 1: raise ValueError('Candidates use different installed packages')
        installed = installed_roots.pop()
        plan = dict(candidates=candidates,
            verifier_source_sha256={name:sha(root/'training'/name) for name in
                ('verify_reserved_joint_field_fusion_v9.py', 'reserved_metrics_v9.py')},
            runtime_source_sha256={'roguard.'+name:sha(root/'src/roguard'/(name+'.py')) for name in
                ('field_pipeline', 'field_fusion', 'native_fields', 'native_review', 'native_assets', 'assurance', 'evidence')})
        working = Path(tempfile.mkdtemp(prefix='roguard-reserved-joint-fusion-59-'))
        path = working/'plan.json'; path.write_text(json.dumps(plan, indent=2)+'\n')
        status['plan_sha256'] = sha(path); save()
        command = [args.torch_python, str(root/'training/verify_reserved_joint_field_fusion_v9.py'),
            '--plan', str(path), '--plan-sha256', sha(path), '--installed-root', str(installed),
            '--probe-root', str(research.parent/'v8-stabilization'), '--output', str(study/'results')]
        env = dict(os.environ, PYTHONPATH=str(installed), HF_HUB_OFFLINE='1',
                   TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
        entry = dict(name='reserved', started_utc=datetime.now(timezone.utc).isoformat())
        status['reserved_evaluation_attempted'] = True; save()
        print('START reserved-joint-field-fusion-59', flush=True)
        with (working/'evaluation.log').open('w') as log:
            try:
                entry['exit_code'] = subprocess.run(command, cwd=working, env=env,
                    stdout=log, stderr=subprocess.STDOUT, timeout=max(1, deadline-time.time())).returncode
            except subprocess.TimeoutExpired: entry.update(exit_code=-1, timed_out=True)
        entry['finished_utc'] = datetime.now(timezone.utc).isoformat()
        status['jobs'].append(entry)
        status['reserved_probes_used'] = (study/'results/selection.json').exists()
        status['reserved_predictions_complete'] = (study/'results/predictions.jsonl').exists()
        print(json.dumps(entry), flush=True)
    status['finished_utc'] = datetime.now(timezone.utc).isoformat(); save()


if __name__ == '__main__':
    from training.coordinator_status_v9 import supervise
    supervise(main, 'research/v9-stabilization/reserved-joint-field-fusion-59')
