"""Compare a registered score combination with the exposed v9 prior audit."""
import argparse
import json
import math
from pathlib import Path

from roguard.assurance import digest
from roguard.trajectory_data import file_sha, prefix, read_jsonl, write_jsonl
from roguard.trajectory_evaluation import checkpoint_id, evaluate, paired_compare
from roguard.trajectory_model import TrajectoryModel
from roguard.trajectory_neural import prepare_memory
from training.combine_trajectory_v10 import logit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('cases','reference','combination','student','selector','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists():raise SystemExit('Refusing to overwrite broad comparison')
    a.output.mkdir(parents=True)
    spec=json.loads(a.combination.read_text());student=TrajectoryModel(a.student)
    selector=TrajectoryModel(a.selector)
    if student.identity!=spec['bindings']['student_sha256']:raise ValueError('Student binding differs')
    protocol=json.loads((a.reference/'protocol.json').read_text())
    if protocol['thresholds']!=spec['thresholds']:raise ValueError('Neural thresholds differ')
    original={}
    for lang in ('ro','uk'):
        for r in read_jsonl(a.reference/'actual'/(lang+'-prior-predictions.jsonl')):
            key=(lang,r['group'],r['case_id'])
            if key in original:raise ValueError('Duplicate prior prediction')
            original[key]=r
    cases=read_jsonl(a.cases);baseline=[];combined=[];identity=digest(spec)
    for case in cases:
        point,=case['checkpoints'];payload=prefix(case,point['focus'])
        view,inference=prepare_memory(payload,selector,1200,preserve_complete=True)
        if inference!=payload or view['mandatory_overflow']:
            raise ValueError('Input changed; cannot reuse the v9 prior score')
        old=original.pop((case['language'],case['original_group'],case['original_case_id']))
        if old['gold']!=point['review']:raise ValueError('Gold label changed')
        s=student.predict(payload)
        if s['predicted'] is None:raise ValueError('Component abstention requires a separate registered policy')
        vector=[logit(old['score'])-logit(spec['thresholds'][case['language']]),
                logit(s['score'])-logit(spec['student_threshold']),1.]
        if spec['bindings']['feature_contract']=='bounded-interactions-v1':
            x,y=math.tanh(vector[0]/4),math.tanh(vector[1]/4);vector += [x,y,x*y]
        elif spec['bindings']['feature_contract']!='two-centered-logits-v1':
            raise ValueError('Unknown combination contract')
        if len(vector)!=len(spec['weights'][case['language']]):raise ValueError('Combination shape differs')
        predicted=int(sum(x*w for x,w in zip(vector,spec['weights'][case['language']]))>=0)
        core={'checkpoint_id':checkpoint_id(case,point),'language':case['language'],'focus':point['focus'],
              'input_digest':s['input_digest'],'latency_seconds':None,'qualified':False,
              'establishes_safety':False,'permits_external_action':False}
        baseline.append({**core,'predicted':old['predicted'],'score':old['score']})
        combined.append({**core,'predicted':predicted,'combination_identity':identity,
                         'neural_score':old['score'],'student_score':s['score']})
    if original:raise ValueError('Missing prior cases')
    write_jsonl(a.output/'baseline-predictions.jsonl',baseline)
    write_jsonl(a.output/'candidate-predictions.jsonl',combined)
    report={'baseline':evaluate(cases,baseline),'candidate':evaluate(cases,combined),
            'comparison':paired_compare(cases,baseline,combined),
            'cases_sha256':file_sha(a.cases),'combination_sha256':file_sha(a.combination),
            'selector_sha256':selector.identity,'student_sha256':student.identity,
            'reference_protocol_sha256':file_sha(a.reference/'protocol.json'),
            'reference_predictions_sha256':{lang:file_sha(a.reference/'actual'/(lang+'-prior-predictions.jsonl')) for lang in ('ro','uk')},
            'input_equivalence_checks':len(cases),'stable_release_ready':False,
            'latency_note':'Reused unchanged-input v9 scores; no end-to-end latency measurement.'}
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({name:{lang:g['strict_accuracy'] for lang,g in report[name]['groups'].items()}
                      for name in ('baseline','candidate')},indent=2))


if __name__=='__main__':main()
