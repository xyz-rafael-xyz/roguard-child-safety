"""Grouped development-only fusion of native and Qwen review scores."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from training.review_fusion_v9 import read, sha, folds, fit, probability, metrics, strata
from training.conservative_fusion_v9 import full_coverage_gap
from training.prepare_instruction_sft_v8 import target

ROOT=Path('research/v9-stabilization')
STUDY=ROOT/'native-qwen-fusion-92'
REGULARIZATION=(.1,1.,10.,100.)


def features(native,qwen):
    if type(qwen['valid_label_token']) is not bool:
        raise ValueError('Invalid token-validity flag')
    values=[native['score'],qwen['score']]
    if any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in values):
        raise ValueError('Invalid component score')
    if not qwen['valid_label_token']:return None
    n,q=values
    def logit(v):
        v=min(1-1e-6,max(1e-6,v))
        return math.log(v/(1-v))
    return [n,logit(n),q,logit(q),n*q,n*(1-q),(1-n)*q]


def load(language,split):
    native=ROOT/('native-full-22-ro' if language=='ro' else 'needs-repair-85/uk')
    paths=dict(rows=native/(split+'.jsonl.gz'),native=native/(split+'-predictions.jsonl'),
               qwen=ROOT/'qwen-qualification-90/inference/predictions.jsonl')
    rows=read(paths['rows']);n=read(paths['native']);qrows=read(paths['qwen'])
    q={r['case_id']:r for r in qrows}
    if len(q)!=len(qrows) or len(rows)!=len(n) or len({r['case_id'] for r in rows})!=len(rows):
        raise ValueError('Duplicate or incomplete component outputs')
    x,y=[],[]
    for r,a in zip(rows,n):
        b=q[r['case_id']];label=target(r)
        if a['case_id']!=r['case_id'] or a['gold']!=label or b['gold']!=label or any(v['language']!=language for v in (r,a,b)):
            raise ValueError('Component identity or label differs')
        x.append(features(a,b));y.append(label)
    return rows,x,y,{k:sha(p) for k,p in paths.items()}


def cutoff(truth,scores):
    valid=[(y,s) for y,s in zip(truth,scores) if s is not None]
    return full_coverage_gap([y for y,s in valid],[s for y,s in valid])


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('fit','audit'))
    p.add_argument('--language',choices=('ro','uk'),required=True);a=p.parse_args()
    for name,digest in json.loads((STUDY/'source-freeze.json').read_text()).items():
        if sha(name)!=digest:raise ValueError('Source freeze changed: '+name)
    out=STUDY/a.language
    if a.stage=='fit':
        rows,x,y,hashes=load(a.language,'dev');assignment=folds(rows);kinds=[strata(r) for r in rows]
        if set(assignment)!=set(range(5)):raise ValueError('Missing connected development fold')
        trials=[]
        for regularization in REGULARIZATION:
            scores=[None]*len(rows)
            for fold in range(5):
                indices=[i for i in range(len(rows)) if assignment[i]!=fold and x[i] is not None]
                model=fit([x[i] for i in indices],[y[i] for i in indices],[kinds[i] for i in indices],regularization)
                for i in range(len(rows)):
                    if assignment[i]==fold:scores[i]=probability(model,x[i])
            limit=cutoff(y,scores);counts=metrics(rows,y,scores,limit)
            rank=(not all(v['passes'] for v in counts.values()),max(v['false_review_rate'] for v in counts.values()),
                  sum(v['false_review_rate'] for v in counts.values()),-regularization)
            trials.append(dict(regularization=regularization,threshold=limit,metrics=counts,rank=rank,scores=scores))
        selected=min(trials,key=lambda r:r['rank'])
        valid=[i for i in range(len(rows)) if x[i] is not None]
        model=fit([x[i] for i in valid],[y[i] for i in valid],[kinds[i] for i in valid],selected['regularization'])
        fitted=[probability(model,v) for v in x]
        limit=cutoff(y+y,selected['scores']+fitted)
        record=dict(language=a.language,model=model,threshold=limit,
            development=metrics(rows,y,fitted,limit),oof=metrics(rows,y,selected['scores'],limit),
            trials=[{k:v for k,v in t.items() if k!='scores'} for t in trials],
            selected_regularization=selected['regularization'],development_hashes=hashes,
            fold_counts=dict(Counter(assignment)),reserved_probes_used=False,stable_release_ready=False,
            selection='Grouped development only; final threshold covers valid OOF and refit positives',
            synthetic_only=True,independent_accuracy_evaluation=False)
        out.mkdir(exist_ok=False);(out/'selection.json').write_text(json.dumps(record,indent=2)+'\n')
        for name,scores in [('dev',fitted),('oof',selected['scores'])]:
            (out/(name+'-predictions.jsonl')).write_text(''.join(json.dumps(dict(case_id=r['case_id'],language=a.language,gold=t,score=s,
                 predicted=None if s is None else int(s>=limit)))+'\n' for r,t,s in zip(rows,y,scores)))
        print(json.dumps({k:record[k] for k in ('threshold','development','oof')}))
    else:
        selection=out/'selection.json';record=json.loads(selection.read_text())
        _,_,_,hashes=load(a.language,'dev')
        if hashes!=record['development_hashes']:raise ValueError('Component evidence changed')
        rows,x,y,hashes=load(a.language,'cal');scores=[probability(record['model'],v) for v in x]
        counts=metrics(rows,y,scores,record['threshold'])
        report=dict(calibration=counts,calibration_hashes=hashes,selection_sha256=sha(selection),
            counts_pass=all(v['passes'] for group in (counts,record['development'],record['oof']) for v in group.values()),
            reserved_probes_used=False,stable_release_ready=False)
        (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
        (out/'cal-predictions.jsonl').write_text(''.join(json.dumps(dict(case_id=r['case_id'],language=a.language,gold=t,score=s,
             predicted=None if s is None else int(s>=record['threshold'])))+'\n' for r,t,s in zip(rows,y,scores)))
        print(json.dumps(report))


if __name__=='__main__':main()
