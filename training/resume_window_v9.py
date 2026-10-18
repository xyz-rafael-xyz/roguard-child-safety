"""Replace the interrupted slow screen with a separately recorded bounded run."""
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
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);a=p.parse_args()
    root=Path('research/v9-stabilization')
    status=json.loads((root/'window-run-status.json').read_text())
    if [j['name'] for j in status['jobs']]!=['attention-12-uk','attention-12-ro']:
        raise ValueError('Unexpected resume state')
    for path,digest in json.loads((root/'gemma12-screen-13b-source-freeze.json').read_text()).items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('Source changed: '+path)
    deadline=datetime.fromisoformat(json.loads((root/'work-window.json').read_text())['deadline_utc'].replace('Z','+00:00')).timestamp()
    name='gemma12-screen-13b';remaining=deadline-time.time()
    if remaining<=0:raise TimeoutError('Work window ended')
    command=[sys.executable,'-m','training.evaluate_gemma_review_v9','--base',a.base,
             '--data','research/v8-stabilization/instruction-sft-01/screen.jsonl','--output',str(root/name)]
    logs=Path(tempfile.gettempdir())/'roguard-v9-window';logs.mkdir(exist_ok=True)
    entry=dict(name=name,started_utc=datetime.now(timezone.utc).isoformat());print('START',name,flush=True)
    with (logs/(name+'.log')).open('w') as log:
        try:entry['exit_code']=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=remaining).returncode
        except subprocess.TimeoutExpired:entry['exit_code']=-1;entry['timed_out']=True
    entry['finished_utc']=datetime.now(timezone.utc).isoformat();status['jobs'].append(entry)
    status['interrupted_attempts']=[dict(name='gemma12-screen-13',reason='slow full-vocabulary projection and unbounded allocation cache; partial results retained')]
    (root/'window-run-status.json').write_text(json.dumps(status,indent=2)+'\n');print('FINISHED',json.dumps(entry),flush=True)


if __name__=='__main__':main()
