"""Run declared local experiments serially within the authorized work window.

Model qualification remains separate from release publication. No training or
evaluation process is allowed to run beyond the configured deadline.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--encoder-base',required=True)
    p.add_argument('--gemma-base',required=True)
    p.add_argument('--wait-for-pid',type=int)
    a=p.parse_args()
    root=Path('research/v9-stabilization')
    deadline=datetime.fromisoformat(json.loads((root/'work-window.json').read_text())['deadline_utc'].replace('Z','+00:00')).timestamp()
    while a.wait_for_pid:
        if time.time()>=deadline:raise TimeoutError('Authorized window ended')
        try:os.kill(a.wait_for_pid,0)
        except ProcessLookupError:break
        time.sleep(10)
    frozen=json.loads((root/'attention-12-source-freeze.json').read_text())
    for path,digest in frozen.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen source changed: '+path)
    logs=Path(tempfile.gettempdir())/'roguard-v9-window';logs.mkdir(exist_ok=True)
    jobs=[]
    for language,baseline in (('uk','research/v9-stabilization/replay-04-uk'),('ro','research/v8-stabilization/review-09-ro')):
        name='attention-12-'+language
        jobs.append((name,[sys.executable,'-m','training.adapt_attention_v9','--base',a.encoder_base,
            '--device','mps','--output',str(root/name),'--initial-review',baseline+'/review.json.gz',
            '--train-language',language,'--natural-contrasts','--temporal-contrasts','--replay-rows','8000',
            '--structured-rows','0','--conversational-rows','0','--epochs','2','--learning-rate','0.00001',
            '--pooling-learning-rate','0.001','--label-smoothing','0.02'],language,baseline))
    # The MLX interpreter is deliberately supplied through the environment;
    # private environment paths are never stored in the public result record.
    mlx_python=os.environ.get('ROGUARD_MLX_PYTHON')
    if mlx_python:
        jobs.append(('gemma12-screen-13',[mlx_python,'-m','training.evaluate_instruction_generic_v9',
            '--base',a.gemma_base,'--data','research/v8-stabilization/instruction-sft-01/screen.jsonl',
            '--output',str(root/'gemma12-screen-13')],None,None))
    status={'stable_release_ready':False,'reserved_probes_used':False,'jobs':[]}
    for name,command,language,baseline in jobs:
        remaining=deadline-time.time()
        if remaining<=0:break
        if (root/name).exists():raise ValueError('Refusing to overwrite existing experiment: '+name)
        print('START',name,flush=True)
        entry={'name':name,'started_utc':datetime.now(timezone.utc).isoformat()}
        with (logs/(name+'.log')).open('w') as log:
            try:
                run=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=remaining)
                entry['exit_code']=run.returncode
            except subprocess.TimeoutExpired:
                entry['timed_out']=True;entry['exit_code']=-1
        if entry['exit_code']==0 and language:
            from training.check_review_candidate_v8 import inspect
            from training.compare_review_candidates_v9 import compare
            from training.mix_review_v9 import evaluate
            reports={'promotion-check':inspect(root/name,language),'comparison':compare(baseline,root/name,language),
                     'mixture':evaluate(baseline,root/name,language)}
            for key,value in reports.items():(root/name/(key+'.json')).write_text(json.dumps(value,indent=2)+'\n')
            entry['prerequisites_pass']=reports['promotion-check']['study_prerequisites_pass']
        entry['finished_utc']=datetime.now(timezone.utc).isoformat();status['jobs'].append(entry)
        (root/'window-run-status.json').write_text(json.dumps(status,indent=2)+'\n')
        print('FINISHED',json.dumps(entry),flush=True)


if __name__=='__main__':main()
