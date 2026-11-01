"""Check portable installed inference without optional numerical packages."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from roguard.trajectory_model import TrajectoryModel


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True)
    a=p.parse_args();sha=hashlib.sha256(a.artifact.read_bytes()).hexdigest()
    model=TrajectoryModel(a.artifact,expected_sha256=sha)
    calibration=json.loads(a.calibration.read_text())
    for language,text in [('ro','Mă ajuți cu exercițiul?'),('uk','Допоможеш мені з вправою?')]:
        payload={'language':language,'turns':[{'role':'minor','text':text}]}
        result=model.predict(payload,calibration=calibration,plan=True)
        assert result['qualified'] is False and result['permits_external_action'] is False
        assert result['predicted'] in (0,1,None)
        assert 1<=len(result['interpretation_set']['hypotheses'])<=432
        assert result['assurance']['input']['context_digest']==result['input_digest']
        assert json.loads(json.dumps(result,allow_nan=False))==result
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'conversation.json'
            path.write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8')
            environment=dict(os.environ,PYTHONIOENCODING='ascii:strict',PYTHONUTF8='0')
            completed=subprocess.run([sys.executable,'-m','roguard.trajectory_cli',str(path),
                '--model',str(a.artifact.resolve()),'--model-sha256',sha,
                '--calibration',str(a.calibration.resolve()),'--plan'],
                env=environment,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
            actual=json.loads(completed.stdout.decode('ascii'))
            assert actual=={k:v for k,v in result.items() if k!='probabilities'}
    print('Portable bilingual inference, planner and legacy-encoding CLI smoke passed; no model qualification implied.')


if __name__=='__main__':main()
