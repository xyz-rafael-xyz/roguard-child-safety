"""Frozen neural bridge study with field parity, calibration and regressions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from eval.run_trajectory_v10 import LocalFrozenV9
from roguard.assurance import assure,digest
from roguard.gemma4_review import _Engine
from roguard.interpreter import prepare_input
from roguard.trajectory_data import read_jsonl,write_jsonl,prefix,file_sha
from roguard.trajectory_evaluation import checkpoint_id,evaluate
from roguard.trajectory_model import TrajectoryModel
from roguard.trajectory_neural import NeuralTrajectoryModel,FieldEngine
from roguard.trajectory_sets import calibrate,predict_set


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--selector',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise SystemExit('Refusing to overwrite a neural study')
    a.output.mkdir(parents=True)
    frozen=LocalFrozenV9(a.base,a.candidate,a.output/'v9-binding')
    selector=TrajectoryModel(a.selector)
    model=NeuralTrajectoryModel(frozen.models,selector)
    cases={name:read_jsonl(a.data/(name+'.jsonl')) for name in ('calibration','dev','challenge','v9-regression')}
    source_paths=[Path('training/run_neural_trajectory_v10.py'),Path('src/roguard/trajectory_neural.py'),Path('src/roguard/trajectory_memory.py'),
                  Path('src/roguard/trajectory_sets.py'),Path('src/roguard/trajectory_model.py')]
    protocol={'pipeline_digest':model.pipeline_digest(),'selector_sha256':selector.identity,
              'inputs':{k:digest(v) for k,v in cases.items()},'source':{str(x):file_sha(x) for x in source_paths},
              'alpha':.1,'selection_performed':False,'synthetic_only':True,'stable_release_ready':False,
              'calibration_groups':'scenario family maximum, separate per language',
              'existing_v9_model_unchanged':True}
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    raw={k:[] for k in cases};checks=[];done=0;total=sum(len(c['checkpoints']) for rows in cases.values() for c in rows)
    started=time.monotonic()
    for lang in ('ro','uk'):
        for name,rows in cases.items():
            for c in rows:
                if c['language']!=lang:continue
                for point in c['checkpoints']:
                    payload=prefix(c,point['focus']);tick=time.perf_counter()
                    r=model.predict(payload)
                    r['latency_seconds']=time.perf_counter()-tick;r['checkpoint_id']=checkpoint_id(c,point)
                    raw[name].append(r);done+=1
                    if name=='v9-regression' and len([x for x in checks if x['language']==lang])<4:
                        component=model.components[lang]
                        prepared,_,_=prepare_input(payload)
                        reference=_Engine.scores(component._engine,[prepared],component._spec)[0]
                        error=abs(reference-r['score'])
                        if error>1e-5 or int(reference>=component._spec['threshold'])!=r['predicted']:
                            raise ValueError('Neural bridge differs from v9 scores')
                        checks.append({'language':lang,'checkpoint_id':r['checkpoint_id'],'score_error':error})
                    if done%20==0:
                        progress={'done':done,'total':total,'seconds':time.monotonic()-started,'language':lang,'population':name}
                        (a.output/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
                        print(json.dumps(progress),flush=True)
        for name in raw:write_jsonl(a.output/(name+'-raw.jsonl'),raw[name])
    calibration_rows=[]
    lookup={checkpoint_id(c,p):(c,p) for c in cases['calibration'] for p in c['checkpoints']}
    for r in raw['calibration']:
        c,point=lookup[r['checkpoint_id']]
        calibration_rows.append({'case_id':r['checkpoint_id'],'split':'calibration','language':c['language'],'family':c['family'],
            'world':point['world'],'pipeline_digest':model.pipeline_digest(),'probabilities':r['probabilities']})
    calibration=calibrate(calibration_rows,pipeline_digest=model.pipeline_digest(),alpha=.1)
    (a.output/'calibration.json').write_text(json.dumps(calibration,indent=2)+'\n')
    reports={}
    for name in ('dev','challenge','v9-regression'):
        for r in raw[name]:
            sets=predict_set(r['probabilities'],calibration,language=r['language'],pipeline_digest=model.pipeline_digest(),supported=r['predicted'] is not None)
            r['interpretation_set']=sets
            r['assurance']=assure({'language':r['language'],'context_digest':r['input_digest'],
                'hypotheses':sets['hypotheses'],'model_digests':[model.identity],'review_floor':r['predicted']==1})
        write_jsonl(a.output/(name+'-predictions.jsonl'),raw[name])
        reports[name]=evaluate(cases[name],raw[name])
    report={'format':'roguard-v10-neural-study1','reports':reports,'v9_parity':checks,
            'seconds':time.monotonic()-started,'calibration':calibration,'stable_release_ready':False}
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:{l:{m:g[m] for m in ('strict_accuracy','joint_set_coverage','mean_set_size')} for l,g in v['groups'].items()} for k,v in reports.items()},indent=2))


if __name__=='__main__':main()
