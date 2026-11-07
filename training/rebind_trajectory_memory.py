"""Verify extraction equivalence before reusing frozen neural forward results.

This is a source-contract migration, not a new model evaluation or improvement.
Any changed selected text or overflow requires fresh neural inference instead.
"""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

from eval.run_trajectory_v10 import LocalFrozenV9
from roguard.assurance import assure
from roguard.trajectory_data import file_sha, prefix, read_jsonl, write_jsonl
from roguard.trajectory_evaluation import checkpoint_id, evaluate
from roguard.trajectory_memory import select, units
from roguard.trajectory_model import TrajectoryModel
from roguard.trajectory_neural import NeuralTrajectoryModel, view_payload
from roguard.trajectory_sets import calibrate, predict_set


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('data','reference','selector','student','base','candidate','output'):
        p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise SystemExit('Refusing to overwrite a migration audit')
    a.output.mkdir(parents=True)
    source=a.reference/'source/src/roguard/trajectory_memory.py'
    spec=importlib.util.spec_from_file_location('roguard._reference_memory_v10',source)
    legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    selector=TrajectoryModel(a.selector)
    frozen=LocalFrozenV9(a.base,a.candidate,a.output/'v9-binding')
    neural=NeuralTrajectoryModel(frozen.models,selector)
    cases={name:read_jsonl(a.data/(name+'.jsonl')) for name in ('calibration','dev','challenge','v9-regression')}
    checked=0
    for name,rows in cases.items():
        for c in rows:
            for point in c['checkpoints']:
                payload=prefix(c,point['focus'])
                kwargs={'budget':neural.budget,'strategy':'learned',
                        'min_relevance':selector.spec.get('min_relevance')}
                old=legacy.select(payload,selector.relevance,**kwargs)
                new=select(payload,selector.relevance,**kwargs)
                if (view_payload(payload,old)!=view_payload(payload,new)
                    or old['mandatory_overflow']!=new['mandatory_overflow']
                    or old['spans']!=new['spans']):
                    raise ValueError('Changed neural input: rerun inference for '+checkpoint_id(c,point))
                if legacy.units(payload)!=units(payload):
                    raise ValueError('Changed full-prefix evidence: rerun evaluation')
                checked+=1
    raw={name:read_jsonl(a.reference/(name+'-raw.jsonl')) for name in cases}
    cal_lookup={checkpoint_id(c,p):(c,p) for c in cases['calibration'] for p in c['checkpoints']}
    cal_rows=[]
    for r in raw['calibration']:
        c,point=cal_lookup[r['checkpoint_id']]
        cal_rows.append({'case_id':r['checkpoint_id'],'split':'calibration','language':c['language'],
                        'family':c['family'],'world':point['world'],'pipeline_digest':neural.pipeline_digest(),
                        'probabilities':r['probabilities']})
    calibration=calibrate(cal_rows,pipeline_digest=neural.pipeline_digest(),alpha=.1)
    (a.output/'neural-calibration.json').write_text(json.dumps(calibration,indent=2)+'\n')
    reports={}
    for name in ('dev','challenge','v9-regression'):
        predictions=deepcopy(raw[name])
        for r in predictions:
            r['reference_model_identity']=r['model_identity']
            r.update(model_identity=neural.identity,pipeline_digest=neural.pipeline_digest(),
                     forward_result_reused_after_input_equivalence_check=True)
            r['interpretation_set']=predict_set(r['probabilities'],calibration,language=r['language'],
                pipeline_digest=neural.pipeline_digest(),supported=r['predicted'] is not None)
            r['assurance']=assure({'language':r['language'],'context_digest':r['input_digest'],
                'hypotheses':r['interpretation_set']['hypotheses'],'model_digests':[neural.identity],
                'review_floor':r['predicted']==1})
        write_jsonl(a.output/(name+'-predictions.jsonl'),predictions)
        reports[name]=evaluate(cases[name],predictions)
    student=TrajectoryModel(a.student);student_rows=[]
    for c in cases['calibration']:
        for point in c['checkpoints']:
            r=student.predict(prefix(c,point['focus']))
            student_rows.append({'case_id':checkpoint_id(c,point),'split':'calibration','language':c['language'],
                'family':c['family'],'world':point['world'],'pipeline_digest':student.pipeline_digest(),
                'probabilities':r['probabilities']})
    student_cal=calibrate(student_rows,pipeline_digest=student.pipeline_digest(),alpha=.1)
    (a.output/'portable-calibration.json').write_text(json.dumps(student_cal,indent=2)+'\n')
    audit={'format':'roguard-memory-contract-migration-1','checked_prefixes':checked,
           'neural_selected_text_changed':False,'reference_source_sha256':file_sha(source),
           'neural_forward_passes_reused':True,'portable_calibration_recomputed':True,
           'source_sha256':{path:file_sha(path) for path in ('training/rebind_trajectory_memory.py',
                'src/roguard/trajectory_memory.py','src/roguard/trajectory_neural.py','src/roguard/trajectory_model.py')},
           'data_sha256':{name:file_sha(a.data/(name+'.jsonl')) for name in cases},
           'reference_predictions_sha256':{name:file_sha(a.reference/(name+'-raw.jsonl')) for name in cases},
           'reference_protocol_sha256':file_sha(a.reference/'protocol.json'),
           'selector_sha256':selector.identity,'portable_sha256':student.identity,
           'reports':reports,'independent_evaluation':False,'stable_release_ready':False}
    (a.output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({'checked_prefixes':checked,'neural_inputs_changed':False,'release_pass':False}))


if __name__=='__main__':main()
