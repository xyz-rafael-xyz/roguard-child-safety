"""Resume an auditable development loop; never touch a sealed final test.

Each registered configuration is run once. Exhaustion exits rejected instead of
silently retraining on test failures or claiming success. Add a prospective
amendment and new configurations before the next research iteration.
"""
import argparse
import json
from pathlib import Path

from eval.run_trajectory_v10 import run as predict
from roguard.trajectory_data import read_jsonl,write_jsonl
from roguard.trajectory_evaluation import evaluate
from roguard.trajectory_model import TrajectoryModel
from roguard.trajectory_sets import WORLDS
from training.train_trajectory_v10 import run as train

CONFIGURATIONS=[
    {'name':'balanced-pooled-policy','min_relevance':.5,'joint':True,'balanced_relevance':True,'review_mode':'joint-policy'},
    {'name':'balanced-focus-binary','min_relevance':.5,'joint':True,'balanced_relevance':True,'representation':'focus-context'},
    {'name':'balanced-pooled-binary','min_relevance':.5,'joint':True,'balanced_relevance':True},
]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    dev=read_jsonl(a.data/'dev.jsonl');ledger=[]
    for config in CONFIGURATIONS:
        config=dict(config);name=config.pop('name');destination=a.output/name
        if not destination.exists():train(a.data,destination,**config)
        model=TrajectoryModel(destination/'student.json.gz')
        # Candidate selection must not depend on calibration quantiles. Compare
        # raw development decisions and joint argmax labels only.
        predictions=predict(dev,model=model)
        report=evaluate(dev,predictions)
        write_jsonl(destination/'development-predictions.jsonl',predictions)
        (destination/'development-report.json').write_text(json.dumps(report,indent=2)+'\n')
        expected={c['case_id']+':'+str(p['focus']):p for c in dev for p in c['checkpoints']}
        field_errors=sum(WORLDS[max(range(len(WORLDS)),key=lambda i:r['probabilities'][i])]!=expected[r['checkpoint_id']]['world'] for r in predictions)
        numerical_pass=field_errors==0 and set(report['groups'])=={'ro','uk'} and all(
            g['strict_accuracy']==1 and g['annotated_evidence_retention']==1 for g in report['groups'].values())
        record={'candidate':name,'model_sha256':model.identity,'development_pass':numerical_pass,
                'release_pass':False,'joint_argmax_errors':field_errors,'groups':report['groups']}
        ledger.append(record)
        (a.output/'ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
        print(json.dumps({'candidate':name,'development_pass':numerical_pass}),flush=True)
        if numerical_pass:
            print('Development prerequisite passed. Independent evaluation, regressions, response and reviewer gates remain.')
            return
    raise SystemExit('REJECTED: registered candidates exhausted. Preserve the failures; register the next experiment before proceeding.')


if __name__=='__main__':main()
