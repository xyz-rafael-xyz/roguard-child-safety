"""Automatically evaluate a qualifying installed native/interpreter ensemble."""
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
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('torch-python','mlx-python','ro-base','uk-base','interpreter-base'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args();repo=Path.cwd().resolve();root=repo/'research/v9-stabilization'
    study=root/'reserved-fusion-42'
    deadline=datetime.fromisoformat(json.loads((root/'work-window.json').read_text())['deadline_utc'].replace('Z','+00:00')).timestamp()
    while time.time()<deadline:
        path=root/'review-fusion-38/run-status.json'
        if path.exists():
            previous=json.loads(path.read_text())
            if ('finished_utc' in previous and time.time()>
                    datetime.fromisoformat(previous['finished_utc']).timestamp()+15):break
        time.sleep(10)
    status=dict(jobs=[],stable_release_ready=False,reserved_probes_used=False)
    def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    def save():
        temporary=study/'run-status.tmp';temporary.write_text(json.dumps(status,indent=2)+'\n')
        temporary.replace(study/'run-status.json')
    for name,expected in json.loads((study/'automation-source-freeze.json').read_text()).items():
        if sha(repo/name)!=expected:raise ValueError('Frozen ensemble verifier source changed')
    from training.reserved_fusion_v9 import qualify
    try:plan=qualify(root)
    except (ValueError,FileNotFoundError) as error:
        status['skipped']='Installed ensemble qualification did not pass'
        status['qualification_error_type']=type(error).__name__
    else:
        if deadline-time.time()<360:
            status['skipped']='Insufficient time for serial reserved inference'
        else:
            path=study/'selection-plan.json'
            with path.open('x') as stream:stream.write(json.dumps(plan,indent=2)+'\n')
            report=json.loads((root/'installed-interpreter-30/screen/report.json').read_text())
            installed=Path(report['installed_module']).parent.parent
            working=Path(tempfile.mkdtemp(prefix='roguard-reserved-fusion-42-'))
            status['plan_sha256']=sha(path);status['reserved_evaluation_attempted']=True;save()
            for stage,python,base in [('native-ro',args.torch_python,args.ro_base),
                    ('native-uk',args.torch_python,args.uk_base),
                    ('interpreter',args.mlx_python,args.interpreter_base),('fusion',args.torch_python,None)]:
                command=[python,str(repo/'training/evaluate_reserved_fusion_v9.py'),stage,
                    '--root',str(root),'--installed-root',str(installed),'--plan',str(path),
                    '--plan-sha256',status['plan_sha256'],'--output',str(study/'results')]
                if base is not None:command.extend(['--base',base])
                entry=dict(name=stage,started_utc=datetime.now(timezone.utc).isoformat())
                print('START reserved-fusion42',stage,flush=True)
                with (working/(stage+'.log')).open('w') as log:
                    try:
                        entry['exit_code']=subprocess.run(command,cwd=working,
                            env=dict(os.environ,PYTHONPATH=str(installed),HF_HUB_OFFLINE='1',
                                     TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false'),
                            stdout=log,stderr=subprocess.STDOUT,timeout=max(1,deadline-time.time())).returncode
                    except subprocess.TimeoutExpired:entry.update(exit_code=-1,timed_out=True)
                entry['finished_utc']=datetime.now(timezone.utc).isoformat();status['jobs'].append(entry)
                status['reserved_probes_used']=any((study/'results').glob('*-predictions.jsonl'));save()
                print(json.dumps(entry),flush=True)
                if entry['exit_code']!=0:break
            status['reserved_predictions_complete']=(study/'results/fusion-report.json').exists()
    status['finished_utc']=datetime.now(timezone.utc).isoformat();save()


if __name__=='__main__':
    from training.coordinator_status_v9 import supervise
    supervise(main,'research/v9-stabilization/reserved-fusion-42')
