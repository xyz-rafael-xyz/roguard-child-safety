"""Check portable installed inference without optional numerical packages."""
import argparse
import hashlib
import json
from pathlib import Path

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
    print('Portable bilingual research inference, calibration and planner smoke passed; no model qualification implied.')


if __name__=='__main__':main()
