"""Run frozen interpretation heads after adaptive decoding, before continuation."""
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
    p=argparse.ArgumentParser();p.add_argument('--ro-base',required=True);p.add_argument('--uk-base',required=True);a=p.parse_args()
    root=Path('research/v9-stabilization');study=root/'native-fields-43'
    deadline=datetime.fromisoformat(json.loads((root/'work-window.json').read_text())['deadline_utc'].replace('Z','+00:00')).timestamp()
    def wait_for_decoding():
        while time.time()<deadline:
            path=root/'policy-decode-16/run-status.json'
            if path.exists():
                try:value=json.loads(path.read_text())
                except json.JSONDecodeError:time.sleep(1);continue
                jobs=value.get('jobs',[])
                if jobs and (len(jobs)==2 or jobs[-1]['exit_code']!=0 or value.get('adapted_skipped')):
                    if time.time()>datetime.fromisoformat(jobs[-1]['finished_utc']).timestamp()+15:return value
            time.sleep(10)
        raise TimeoutError('Authorized work window ended')
    wait_for_decoding()
    for path,expected in json.loads((study/'source-freeze.json').read_text()).items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise ValueError('Frozen native-field source changed: '+path)
    status=dict(jobs=[],stable_release_ready=False,reserved_probes_used=False)
    def save():
        temp=study/'run-status.tmp';temp.write_text(json.dumps(status,indent=2)+'\n');temp.replace(study/'run-status.json')
    if deadline-time.time()<3600:
        status['skipped']='Less than one hour remains; preserve time for active qualification and final reporting.'
    else:
        logs=Path(tempfile.gettempdir())/'roguard-v9-window';logs.mkdir(exist_ok=True)
        env=dict(os.environ,PYTORCH_MPS_HIGH_WATERMARK_RATIO='.7',PYTORCH_MPS_LOW_WATERMARK_RATIO='.4',
                 TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        for language,base in (('ro',a.ro_base),('uk',a.uk_base)):
            report=root/('native-full-22-'+language)/'development.json'
            if (not report.exists() or not 0<=json.loads(report.read_text()).get(
                    'selected_checkpoint_dev_replay_max_error',float('inf'))<=1e-5):
                status.setdefault('skipped_languages',[]).append(language);continue
            for stage,command in (
                ('features-fit',['training.extract_native_fields_v9','--stage','fit','--base',base]),
                ('fit',['training.train_native_fields_v9','fit']),
                ('features-cal',['training.extract_native_fields_v9','--stage','cal','--base',base]),
                ('audit',['training.train_native_fields_v9','audit'])):
                entry=dict(language=language,stage=stage,started_utc=datetime.now(timezone.utc).isoformat())
                with (logs/('native-fields-43-'+language+'-'+stage+'.log')).open('w') as stream:
                    try:entry['exit_code']=subprocess.run([sys.executable,'-m',*command,'--language',language],env=env,
                        stdout=stream,stderr=subprocess.STDOUT,timeout=max(1,deadline-time.time())).returncode
                    except subprocess.TimeoutExpired:entry.update(exit_code=-1,timed_out=True)
                entry['finished_utc']=datetime.now(timezone.utc).isoformat();status['jobs'].append(entry);save()
                print(json.dumps(entry),flush=True)
                if entry['exit_code']!=0:break
    status['finished_utc']=datetime.now(timezone.utc).isoformat();save()


if __name__=='__main__':
    from training.coordinator_status_v9 import supervise
    supervise(main,'research/v9-stabilization/native-fields-43')
