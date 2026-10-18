"""Serialize the structured-interpreter study after the earlier GPU queue."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);a=p.parse_args()
    root=Path('research/v9-stabilization');study=root/'interpreter-14'
    deadline=datetime.fromisoformat(json.loads((root/'work-window.json').read_text())['deadline_utc'].replace('Z','+00:00')).timestamp()
    while True:
        if time.time()>=deadline:raise TimeoutError('Authorized work window ended')
        state=root/'window-run-status.json'
        if state.exists() and len(json.loads(state.read_text())['jobs'])==3:break
        time.sleep(10)
    for path,digest in json.loads((study/'source-freeze.json').read_text()).items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen source changed: '+path)
    screen='research/v8-stabilization/instruction-sft-01/screen.jsonl'
    jobs=[('baseline',[sys.executable,'-m','training.evaluate_interpreter_v9','--base',a.base,'--data',screen,'--output',str(study/'baseline')]),
          ('train',[sys.executable,'-m','training.train_interpreter_v9','--base',a.base,'--config',str(study/'config-v3.json')]),
          ('adapted',[sys.executable,'-m','training.evaluate_interpreter_v9','--base',a.base,'--adapter',str(study/'selected-adapter'),'--data',screen,'--output',str(study/'adapted')]),
          ('archive',[sys.executable,'-m','training.archive_interpreter_v9','export','--input',str(study/'selected-adapter'),'--output',str(study/'selected-adapter.json.gz')])]
    logs=Path(tempfile.gettempdir())/'roguard-v9-window';logs.mkdir(exist_ok=True)
    status={'stable_release_ready':False,'reserved_probes_used':False,'jobs':[]}
    for name,command in jobs:
        remaining=deadline-time.time()
        if remaining<=0:break
        print('START interpreter-14',name,flush=True)
        entry=dict(name=name,started_utc=datetime.now(timezone.utc).isoformat())
        with (logs/('interpreter-14-'+name+'.log')).open('w') as log:
            try:entry['exit_code']=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=remaining).returncode
            except subprocess.TimeoutExpired:entry['exit_code']=-1;entry['timed_out']=True
        entry['finished_utc']=datetime.now(timezone.utc).isoformat();status['jobs'].append(entry)
        (study/'run-status.json').write_text(json.dumps(status,indent=2)+'\n')
        print('FINISHED',json.dumps(entry),flush=True)
        if entry['exit_code']!=0:break


if __name__=='__main__':main()
