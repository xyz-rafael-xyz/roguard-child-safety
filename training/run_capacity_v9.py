"""Run the predeclared capacity comparisons after the structured-model queue."""
import argparse
from datetime import datetime,timezone
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
    deadline=datetime.fromisoformat(json.loads((root/'work-window.json').read_text())['deadline_utc'].replace('Z','+00:00')).timestamp()
    while True:
        if time.time()>=deadline:raise TimeoutError('Authorized work window ended')
        state=root/'interpreter-14/run-status.json'
        if state.exists():
            jobs=json.loads(state.read_text())['jobs']
            if jobs and (len(jobs)==4 or jobs[-1]['exit_code']!=0):
                finished=datetime.fromisoformat(jobs[-1]['finished_utc']).timestamp()
                if time.time()>finished+15:break
        time.sleep(10)
    for path,digest in json.loads((root/'capacity-15-source-freeze.json').read_text()).items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen source changed: '+path)
    logs=Path(tempfile.gettempdir())/'roguard-v9-window';logs.mkdir(exist_ok=True)
    status={'stable_release_ready':False,'reserved_probes_used':False,'jobs':[]}
    for rate,label in (('0.00001','low'),('0.00003','high')):
        for language,baseline in (('ro','research/v8-stabilization/review-09-ro'),('uk','research/v9-stabilization/replay-04-uk')):
            remaining=deadline-time.time()
            if remaining<=0:return
            name='capacity-15-'+label+'-'+language
            if (root/name).exists():raise ValueError('Refusing to overwrite '+name)
            command=[sys.executable,'-m','training.adapt_capacity_v9','--base',a.base,'--device','mps',
                '--output',str(root/name),'--initial-review',baseline+'/review.json.gz','--train-language',language,
                '--natural-contrasts','--temporal-contrasts','--replay-rows','8000','--structured-rows','0',
                '--conversational-rows','0','--epochs','2','--learning-rate',rate,'--pooling-learning-rate','0.001','--label-smoothing','0.02']
            entry=dict(name=name,started_utc=datetime.now(timezone.utc).isoformat());print('START',name,flush=True)
            with (logs/(name+'.log')).open('w') as log:
                try:entry['exit_code']=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=remaining).returncode
                except subprocess.TimeoutExpired:entry['exit_code']=-1;entry['timed_out']=True
            if entry['exit_code']==0:
                from training.check_review_candidate_v8 import inspect
                from training.compare_review_candidates_v9 import compare
                from training.mix_review_v9 import evaluate
                reports={'promotion-check':inspect(root/name,language),'comparison':compare(baseline,root/name,language),
                    'mixture':evaluate(baseline,root/name,language)}
                for key,value in reports.items():(root/name/(key+'.json')).write_text(json.dumps(value,indent=2)+'\n')
                entry['prerequisites_pass']=reports['promotion-check']['study_prerequisites_pass']
            entry['finished_utc']=datetime.now(timezone.utc).isoformat();status['jobs'].append(entry)
            (root/'capacity-15-run-status.json').write_text(json.dumps(status,indent=2)+'\n');print('FINISHED',json.dumps(entry),flush=True)


if __name__=='__main__':main()
