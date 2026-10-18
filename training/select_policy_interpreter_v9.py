"""Choose saved policy checkpoints using only the existing 80 validation rows."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from training.interpreter_validation_v9 import summarize


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);a=p.parse_args()
    root=Path('research/v9-stabilization');study=root/'interpreter-policy-27'
    deadline=datetime.fromisoformat(json.loads((root/'work-window.json').read_text())['deadline_utc'].replace('Z','+00:00')).timestamp()
    validation=study/'valid.jsonl';rows=read(validation)
    parent=root/'interpreter-14'
    if sha(validation)!=sha(parent/'valid.jsonl'):raise ValueError('Validation population changed')
    metrics=json.loads((study/'training-metrics.json').read_text())
    checkpoints=metrics['validation_checkpoints'];results=[];seen={}
    initial_sha=json.loads((study/'config.json').read_text())['initial_adapter_sha256']
    if not checkpoints or checkpoints[0]['iteration']!=0 or checkpoints[0]['sha256']!=initial_sha:
        raise ValueError('Initial checkpoint is not the verified parent')
    parent_predictions=parent/'adapted/predictions.jsonl'
    for checkpoint in checkpoints:
        folder=Path(checkpoint['path']);weights=folder/'adapters.safetensors'
        if sha(weights)!=checkpoint['sha256']:raise ValueError('Saved validation checkpoint changed')
        identity=checkpoint['sha256']
        if identity==initial_sha:
            prediction_path=parent_predictions
        elif identity in seen:
            prediction_path=seen[identity]
        else:
            output=study/'validation-decodes'/str(checkpoint['iteration'])
            command=[sys.executable,'-m','training.evaluate_interpreter_v9','--base',a.base,
                '--adapter',str(folder),'--data',str(validation),'--output',str(output)]
            log=Path(tempfile.gettempdir())/'roguard-v9-window'/('policy27-validation-'+str(checkpoint['iteration'])+'.log')
            with log.open('w') as stream:
                subprocess.run(command,check=True,stdout=stream,stderr=subprocess.STDOUT,timeout=max(1,deadline-time.time()))
            prediction_path=output/'predictions.jsonl'
        seen[identity]=prediction_path
        summary=summarize(rows,read(prediction_path),checkpoint['val_loss'])
        entry=dict(checkpoint,**summary,predictions=str(prediction_path),predictions_sha256=sha(prediction_path))
        results.append(entry);print(json.dumps(entry),flush=True)
    selected=min(results,key=lambda r:r['rank']+[r['iteration']])
    record=dict(created_utc=datetime.now(timezone.utc).isoformat(),validation_sha256=sha(validation),
        method='Passing original gates first, then smallest worst class error rate across four strata, mean class error rate, unweighted six-field loss, earlier iteration.',
        candidates=results,selected=selected,calibration_used=False,reserved_probes_used=False,stable_release_ready=False,
        source_sha256=sha(__file__),ranking_source_sha256=sha(Path('training/interpreter_validation_v9.py')))
    # Keep the original loss-selected checkpoint, with its original metrics.
    shutil.copytree(study/'selected-adapter',study/'loss-selected-adapter')
    for filename in ('adapters.safetensors','adapter_config.json'):
        shutil.copyfile(Path(selected['path'])/filename,study/'selected-adapter'/filename)
    if sha(study/'selected-adapter/adapters.safetensors')!=selected['sha256']:raise ValueError('Selected checkpoint export differs')
    (study/'validation-selection.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
