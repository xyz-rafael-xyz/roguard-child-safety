"""Export the fixed fusion fit, then pair actual installed neural reports."""
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
    repo = Path.cwd().resolve(); root = repo/'research/v9-stabilization'; study = root/'review-fusion-38'
    deadline = datetime.fromisoformat(json.loads((root/'work-window.json').read_text())['deadline_utc'].replace('Z','+00:00')).timestamp()
    status = dict(jobs=[],stable_release_ready=False,reserved_probes_used=False)
    def wait(name):
        while time.time() < deadline:
            path = root/name/'run-status.json'
            if path.exists():
                prior = json.loads(path.read_text())
                if 'finished_utc' in prior:return prior
            time.sleep(10)
        raise TimeoutError('Authorized work window ended')
    def save():
        temporary = study/'run-status.tmp'
        temporary.write_text(json.dumps(status,indent=2)+'\n'); temporary.replace(study/'run-status.json')
    def run(name,command,cwd,env):
        entry = dict(name=name,started_utc=datetime.now(timezone.utc).isoformat())
        log = Path(tempfile.gettempdir())/'roguard-v9-window'/('review-fusion-38-'+name+'.log')
        with log.open('w') as stream:
            try:entry['exit_code'] = subprocess.run(command,cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=max(1,deadline-time.time())).returncode
            except subprocess.TimeoutExpired:entry['exit_code'] = -1
        entry['finished_utc'] = datetime.now(timezone.utc).isoformat(); status['jobs'].append(entry);save()
        print(json.dumps(entry),flush=True)
        return entry['exit_code'] == 0
    wait('review-fusion-36')
    for path,expected in json.loads((study/'source-freeze.json').read_text()).items():
        if hashlib.sha256((repo/path).read_bytes()).hexdigest()!=expected:raise ValueError('Frozen fusion runtime source changed: '+path)
    if run('export',[sys.executable,'-m','training.export_review_fusion_v9'],repo,os.environ.copy()):
        wait('installed-interpreter-30')
        report = root/'installed-interpreter-30/screen/report.json'
        if not report.exists() or json.loads(report.read_text()).get('parity_pass') is not True:
            status['skipped'] = 'Installed interpreter parity did not pass; not a pass'
        else:
            installed = Path(json.loads(report.read_text())['installed_module']).parent.parent
            for language in ('ro','uk'):
                if not (study/language/'export.json').exists():
                    status.setdefault('skipped_languages',[]).append(language);continue
                command = [sys.executable,str(repo/'training/verify_review_fusion_runtime_v9.py'),
                    '--root',str(root),'--installed-root',str(installed),'--language',language]
                run(language,command,installed.parent,dict(os.environ,PYTHONPATH=str(installed)))
    status['finished_utc'] = datetime.now(timezone.utc).isoformat();save()


if __name__ == '__main__':
    main()
