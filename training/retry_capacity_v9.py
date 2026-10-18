"""Retry the interrupted Ukrainian capacity trial without changing its math."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',required=True)
    args=parser.parse_args()
    root=Path('research/v9-stabilization')
    for path,expected in json.loads((root/'capacity-15-attempt3-source-freeze.json').read_text()).items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
            raise ValueError('Frozen experiment source changed: '+path)
    output=root/'capacity-15-high-uk'
    if output.exists():raise ValueError('Refusing overwrite')
    deadline=datetime.fromisoformat(json.loads((root/'work-window.json').read_text())['deadline_utc'].replace('Z','+00:00')).timestamp()
    if time.time()>=deadline:raise TimeoutError('Authorized work window ended')
    env=dict(os.environ,PYTORCH_MPS_HIGH_WATERMARK_RATIO='0.7',PYTORCH_MPS_LOW_WATERMARK_RATIO='0.4',TOKENIZERS_PARALLELISM='false')
    status=dict(attempt=3,started_utc=datetime.now(timezone.utc).isoformat(),stable_release_ready=False,mps_cache_interval=25,
                environment={k:env[k] for k in ('PYTORCH_MPS_HIGH_WATERMARK_RATIO','PYTORCH_MPS_LOW_WATERMARK_RATIO','TOKENIZERS_PARALLELISM')},
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),reserved_probes_used=False)
    def save():
        pending=root/'capacity-15-high-uk-retry.tmp'
        pending.write_text(json.dumps(status,indent=2)+'\n')
        pending.replace(root/'capacity-15-high-uk-retry.json')
    save()
    baseline='research/v9-stabilization/replay-04-uk'
    command=[sys.executable,'-m','training.adapt_capacity_v9','--base',args.base,'--device','mps',
             '--output',str(output),'--initial-review',baseline+'/review.json.gz','--train-language','uk',
             '--natural-contrasts','--temporal-contrasts','--replay-rows','8000','--structured-rows','0',
             '--conversational-rows','0','--epochs','2','--learning-rate','0.00003',
             '--pooling-learning-rate','0.001','--label-smoothing','0.02','--mps-cache-interval','25']
    log=Path(tempfile.gettempdir())/'roguard-v9-window/capacity-15-high-uk-attempt3.log'
    with log.open('x') as stream:
        try:status['exit_code']=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=deadline-time.time()).returncode
        except subprocess.TimeoutExpired:status.update(exit_code=-1,timed_out=True)
    if status['exit_code']==0:
        from training.check_review_candidate_v8 import inspect
        from training.compare_review_candidates_v9 import compare
        from training.mix_review_v9 import evaluate
        reports={'promotion-check':inspect(output,'uk'),'comparison':compare(baseline,output,'uk'),'mixture':evaluate(baseline,output,'uk')}
        for name,value in reports.items():(output/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
        status['prerequisites_pass']=reports['promotion-check']['study_prerequisites_pass']
    status['finished_utc']=datetime.now(timezone.utc).isoformat();save()
    print(json.dumps(status,indent=2),flush=True)


if __name__=='__main__':main()
