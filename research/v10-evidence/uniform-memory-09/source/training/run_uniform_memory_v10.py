"""Evaluate registered uniform selection with input-bound neural-result reuse."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

from eval.run_trajectory_v10 import LocalFrozenV9
from roguard.assurance import assure
from roguard.trajectory_data import file_sha, input_identity, prefix, read_jsonl, write_jsonl
from roguard.trajectory_evaluation import checkpoint_id, evaluate
from roguard.trajectory_model import TrajectoryModel
from roguard.trajectory_neural import NeuralTrajectoryModel, prepare_memory
from roguard.trajectory_sets import calibrate, predict_set


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('base','candidate','selector','data','reference','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise SystemExit('Refusing to overwrite an experiment')
    args.output.mkdir(parents=True)
    frozen=LocalFrozenV9(args.base,args.candidate,args.output/'v9-binding')
    selector=TrajectoryModel(args.selector)
    model=NeuralTrajectoryModel(frozen.models,selector,selection_mode='uniform')
    cases={name:read_jsonl(args.data/(name+'.jsonl')) for name in ('calibration','dev','challenge','v9-regression')}
    reference_protocol=json.loads((args.reference/'protocol.json').read_text())
    if reference_protocol['selector_sha256']!=selector.identity:
        raise ValueError('Reference selector differs')
    for lang in ('ro','uk'):
        if file_sha(args.output/'v9-binding'/(lang+'-review.json'))!=file_sha(args.reference/'v9-binding'/(lang+'-review.json')):
            raise ValueError('Reference neural component differs')
    cache={}
    for name in cases:
        for row in read_jsonl(args.reference/(name+'-raw.jsonl')):
            if row['predicted'] is None:continue
            key=(row['language'],row['inference_input_digest'])
            if key in cache and (abs(cache[key]['score']-row['score'])>1e-5
                                 or max(abs(a-b) for a,b in zip(cache[key]['probabilities'],row['probabilities']))>1e-5):
                raise ValueError('Reference outputs disagree on the same input')
            cache.setdefault(key,row)
    source_paths=['training/run_uniform_memory_v10.py','src/roguard/trajectory_neural.py',
                  'src/roguard/trajectory_memory.py','src/roguard/trajectory_sets.py','src/roguard/trajectory_model.py']
    protocol={'pipeline_digest':model.pipeline_digest(),'selector_sha256':selector.identity,
              'reference_protocol_sha256':file_sha(args.reference/'protocol.json'),
              'reference_raw_sha256':{name:file_sha(args.reference/(name+'-raw.jsonl')) for name in cases},
              'data_sha256':{name:file_sha(args.data/(name+'.jsonl')) for name in cases},
              'source_sha256':{path:file_sha(path) for path in source_paths},
              'amendment_sha256':file_sha('research/v10-evidence/protocol-amendment-06.json'),
              'alpha':.1,'independent_evaluation':False,'stable_release_ready':False}
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for path in source_paths:
        dest=args.output/'source'/path;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(Path(path).read_bytes())
    raw={name:[] for name in cases};computed=0;reused=0;started=time.monotonic()
    for lang in ('ro','uk'):
        for name,rows in cases.items():
            for case in rows:
                if case['language']!=lang:continue
                for point in case['checkpoints']:
                    payload=prefix(case,point['focus'])
                    view,inference=prepare_memory(payload,selector,model.budget,preserve_complete=True)
                    inference_hash=input_identity(inference) if inference['turns'] else None
                    prior=cache.get((lang,inference_hash))
                    if prior is not None and not view['mandatory_overflow'] and len(inference['turns'])<=12:
                        result={'predicted':prior['predicted'],'score':prior['score'],
                                'probabilities':deepcopy(prior['probabilities']),
                                'input_digest':input_identity(payload),'inference_input_digest':inference_hash,
                                'language':lang,'focus':point['focus'],'model_identity':model.identity,
                                'pipeline_digest':model.pipeline_digest(),'used_memory':True,
                                'evidence':[{k:s[k] for k in ('turn','start','end')} for s in view['spans']],
                                'qualified':False,'establishes_safety':False,'permits_external_action':False}
                        result['reference_checkpoint_id']=prior['checkpoint_id'];reused+=1
                    else:
                        tick=time.perf_counter();result=model.predict(payload)
                        result['new_forward_seconds']=time.perf_counter()-tick
                        if result['inference_input_digest']!=inference_hash:
                            raise ValueError('Actual inference input differs from reuse audit')
                        computed+=1
                    result['latency_seconds']=None
                    result['checkpoint_id']=checkpoint_id(case,point)
                    result['forward_result_reused']=prior is not None and 'reference_checkpoint_id' in result
                    raw[name].append(result)
                    if (computed+reused)%20==0:
                        progress={'done':computed+reused,'computed':computed,'reused':reused,
                                  'language':lang,'population':name,'seconds':time.monotonic()-started}
                        (args.output/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
                        print(json.dumps(progress),flush=True)
        for name,rows in raw.items():write_jsonl(args.output/(name+'-raw.jsonl'),rows)
    lookup={checkpoint_id(c,p):(c,p) for c in cases['calibration'] for p in c['checkpoints']}
    calibration_rows=[]
    for result in raw['calibration']:
        case,point=lookup[result['checkpoint_id']]
        calibration_rows.append({'case_id':result['checkpoint_id'],'split':'calibration','language':case['language'],
            'family':case['family'],'world':point['world'],'pipeline_digest':model.pipeline_digest(),
            'probabilities':result['probabilities']})
    calibration=calibrate(calibration_rows,pipeline_digest=model.pipeline_digest(),alpha=.1)
    (args.output/'calibration.json').write_text(json.dumps(calibration,indent=2)+'\n')
    reports={}
    for name in ('dev','challenge','v9-regression'):
        for r in raw[name]:
            r['interpretation_set']=predict_set(r['probabilities'],calibration,language=r['language'],
                pipeline_digest=model.pipeline_digest(),supported=r['predicted'] is not None)
            r['assurance']=assure({'language':r['language'],'context_digest':r['input_digest'],
                'hypotheses':r['interpretation_set']['hypotheses'],'model_digests':[model.identity],
                'review_floor':r['predicted']==1})
        write_jsonl(args.output/(name+'-predictions.jsonl'),raw[name]);reports[name]=evaluate(cases[name],raw[name])
    report={'reports':reports,'computed':computed,'reused':reused,'seconds':time.monotonic()-started,
            'stable_release_ready':False,'latency_note':'Reused forward results; no end-to-end latency claim.'}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({name:{lang:{k:g[k] for k in ('strict_accuracy','joint_set_coverage','mean_set_size')}
                           for lang,g in r['groups'].items()} for name,r in reports.items()},indent=2))


if __name__=='__main__':main()
