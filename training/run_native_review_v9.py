"""Serialize native full-fine-tuning comparisons after capacity experiments."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--ro-base',required=True);p.add_argument('--uk-base',required=True);args=p.parse_args()
    root=Path('research/v9-stabilization')
    deadline=datetime.fromisoformat(json.loads((root/'work-window.json').read_text())['deadline_utc'].replace('Z','+00:00')).timestamp()
    while True:
        if time.time()>=deadline:raise TimeoutError('Authorized work window ended')
        path=root/'capacity-15-run-status.json'
        if path.exists():
            jobs=json.loads(path.read_text())['jobs']
            if len(jobs)==4:
                if time.time()>datetime.fromisoformat(jobs[-1]['finished_utc']).timestamp()+15:break
        time.sleep(10)
    for path,digest in json.loads((root/'native-full-22-source-freeze.json').read_text()).items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen source changed: '+path)
    status=dict(stable_release_ready=False,reserved_probes_used=False,jobs=[])
    logs=Path(tempfile.gettempdir())/'roguard-v9-window';logs.mkdir(exist_ok=True)
    for language,base in [('ro',args.ro_base),('uk',args.uk_base)]:
        remaining=deadline-time.time()
        if remaining<=0:break
        name='native-full-22-'+language
        command=[sys.executable,'-m','training.train_native_review_v9','--language',language,'--base',base,'--output',str(root/name),'--device','mps']
        entry=dict(name=name,started_utc=datetime.now(timezone.utc).isoformat());print('START',name,flush=True)
        with (logs/(name+'.log')).open('w') as log:
            try:entry['exit_code']=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=remaining).returncode
            except subprocess.TimeoutExpired:entry.update(exit_code=-1,timed_out=True)
        if entry['exit_code']==0:
            entry['prerequisites_pass']=json.loads((root/name/'promotion-check.json').read_text())['study_prerequisites_pass']
        entry['finished_utc']=datetime.now(timezone.utc).isoformat();status['jobs'].append(entry)
        (root/'native-full-22-run-status.json').write_text(json.dumps(status,indent=2)+'\n');print('FINISHED',json.dumps(entry),flush=True)


if __name__=='__main__':main()
