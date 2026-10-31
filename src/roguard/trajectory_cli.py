"""Inspect an experimental v10 local evidence proposal. No external actions."""
import argparse
import json
from pathlib import Path

from .trajectory_model import TrajectoryModel


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--model-sha256',required=True)
    p.add_argument('--calibration',type=Path)
    p.add_argument('--plan',action='store_true')
    p.add_argument('--neural',action='store_true',help='Use the installed pinned v9 Gemma4 components with this artifact as evidence selector')
    p.add_argument('--draft',help='Optional assistant reply for the experimental response head')
    a=p.parse_args()
    try:
        selector=TrajectoryModel(a.model,expected_sha256=a.model_sha256)
        if a.neural:
            from .trajectory_neural import NeuralTrajectoryModel
            model=NeuralTrajectoryModel.from_profile(selector)
        else:model=selector
        with a.input.open('rb') as source:raw=source.read(4_000_001)
        if len(raw)>4_000_000:raise ValueError('Conversation JSON exceeds file-size limit')
        payload=json.loads(raw)
        calibration=json.loads(a.calibration.read_text()) if a.calibration else None
        if a.plan and calibration is None:raise ValueError('Planning requires a pipeline-bound calibration artifact')
        result=model.predict(payload,calibration=calibration,plan=a.plan)
        result.pop('probabilities')
        if a.draft is not None:
            result['response_review']=selector.review_response(a.draft,payload['language'],payload['turns'][-1]['text'])
        print(json.dumps(result,indent=2,ensure_ascii=False))
    except (ValueError,KeyError,TypeError,OSError,ImportError) as error:
        p.exit(2,'Invalid v10 research input: '+str(error)+'\n')


if __name__=='__main__':main()
