"""Describe paired errors without changing gold, fitting models or choosing thresholds."""
import argparse
from collections import Counter
import json
from pathlib import Path

from roguard.trajectory_data import file_sha,read_jsonl,write_jsonl
from roguard.trajectory_evaluation import checkpoint_id


def analyze(cases,baseline,candidate):
    expected={}
    for case in cases:
        for point in case['checkpoints']:
            key=checkpoint_id(case,point)
            if key in expected:raise ValueError('Duplicate expected checkpoint')
            expected[key]=(case,point)
    def index(rows):
        result={row['checkpoint_id']:row for row in rows}
        if len(result)!=len(rows) or set(result)!=set(expected):raise ValueError('Incomplete or duplicate predictions')
        return result
    old,new=index(baseline),index(candidate);records=[];groups=Counter()
    for key,(case,point) in expected.items():
        left,right=old[key],new[key]
        if (left['input_digest']!=right['input_digest']
                or left['inference_input_digest']!=right['inference_input_digest']):
            raise ValueError('Paired inference inputs differ')
        a,b=left['predicted'],right['predicted'];gold=point['review']
        if a==gold and b==gold:continue
        outcome='introduced' if a==gold else 'fixed' if b==gold else 'shared_error'
        world=point.get('world');fields=world or {}
        label=':'.join([outcome,case['language'],str(gold)]+[fields.get(field,'UNANNOTATED') for field in ('assertion','subject','source','time')])
        groups[label]+=1
        records.append({'checkpoint_id':key,'language':case['language'],'family':case['family'],
                        'gold':gold,'world':world,'baseline':a,'candidate':b,'outcome':outcome,
                        'baseline_score':left['score'],'candidate_score':right['score']})
    return records,dict(sorted(groups.items()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cases','baseline','candidate','comparison_report','study'):p.add_argument('--'+name.replace('_','-'),dest=name,type=Path,required=True)
    a=p.parse_args()
    if a.study.exists():raise ValueError('Refusing to overwrite error analysis')
    comparison=json.loads(a.comparison_report.read_text())
    if comparison.get('complete') is not True:raise ValueError('Incomplete comparison')
    rows,groups=analyze(read_jsonl(a.cases),read_jsonl(a.baseline),read_jsonl(a.candidate))
    counts=Counter(row['outcome'] for row in rows)
    broad=comparison['populations']['broad']['comparison']
    if counts['introduced']!=broad['baseline_only_correct'] or counts['fixed']!=broad['candidate_only_correct']:
        raise ValueError('Paired error counts differ from completed evaluation')
    a.study.mkdir(parents=True);write_jsonl(a.study/'paired-errors.jsonl',rows)
    result={'complete':True,'source_sha256':file_sha(__file__),
            'inputs':{key:file_sha(getattr(a,key)) for key in ('cases','baseline','candidate','comparison_report')},
            'counts':dict(counts),'groups':groups,'fitting_performed':False,'labels_changed':False,
            'causal_mechanism_established':False,'independent_validation':False,'stable_release_ready':False}
    (a.study/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
