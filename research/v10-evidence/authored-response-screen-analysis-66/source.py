"""Describe retained machine-screen disagreements without changing any labels."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--private',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Refusing to overwrite audit')
    sys.path[:0]=[str(a.root/'src'),str(a.root)]
    from roguard.assurance import digest
    ev=a.root/'research/v10-evidence'
    source=a.private/'authored-response-pilot-66/families-revision1.jsonl'
    proof_path=a.private/'authored-response-screen-66-verification.json'
    proof=json.loads(proof_path.read_text())
    protocol_path=ev/'authored-response-screen-66/protocol.json'
    protocol=json.loads(protocol_path.read_text())
    assert sha(source)==protocol['corpus_sha256']
    assert proof['protocol_sha256']==sha(protocol_path)
    assert proof['report_sha256']==sha(ev/'authored-response-screen-66/report.json')
    assert proof['judgments_sha256']==sha(a.private/'authored-response-screen-66/judgments.jsonl')
    rows={r['request']['family']:r for r in map(json.loads,source.read_text().splitlines())}
    assert len(rows)==16 and len(proof['outcomes'])==32
    individual=[];role_counts=defaultdict(Counter);confusion=Counter();family_results=defaultdict(dict)
    for outcome in proof['outcomes']:
        family,language=outcome['key'].rsplit(':',1);row=rows[family]
        roles=['problematic','appropriate']
        if int(digest({'family':family,'language':language})[-1],16)%2:roles.reverse()
        expected=[row['request']['category'] if role=='problematic' else 'appropriate' for role in roles]
        assert expected==outcome['expected_labels']
        judgment=outcome['judgment'];assert judgment is not None
        normalized={}
        for role,gold,predicted,natural in zip(roles,expected,judgment['labels'],judgment['natural'],strict=True):
            role_counts[role]['drafts']+=1;role_counts[role]['label_agreement']+=gold==predicted
            role_counts[role]['fluency_rejection']+=not natural
            confusion[(gold,predicted)]+=1
            normalized[role]={'label':predicted,'natural':natural}
            individual.append({'key':outcome['key']+':'+role,'expected_label':gold,
                               'judge_label':predicted,'judge_natural':natural,
                               'label_agreement':gold==predicted,'partition':'train'})
        family_results[family][language]=normalized
    differences=[family for family,value in family_results.items() if value['ro']!=value['uk']]
    assert len(individual)==64 and len({r['key'] for r in individual})==64
    missed=sum(r['expected_label']!='appropriate' and r['judge_label']=='appropriate' for r in individual)
    wrong_category=sum(r['expected_label']!='appropriate' and r['judge_label'] not in ('appropriate',r['expected_label']) for r in individual)
    result={'complete':True,'scope':'Machine judgments against intended synthetic labels, not judge accuracy or independent semantic validation.',
      'source_sha256':sha(Path(__file__)),'source_corpus_sha256':sha(source),'verification_sha256':sha(proof_path),
      'role_counts':{role:dict(counts) for role,counts in role_counts.items()},
      'intended_problematic_judged_appropriate':missed,'intended_problematic_judged_other_problem_category':wrong_category,
      'label_comparisons':[{'intended':key[0],'judged':key[1],'drafts':value} for key,value in sorted(confusion.items())],
      'bilingual_families_with_different_label_or_fluency_outcomes':differences,
      'individual_outcomes':individual,'labels_changed':False,'judge_rerun':False,'training_ready':False,
      'independent_annotation':False,'stable_release_ready':False}
    with a.output.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('individual_outcomes','label_comparisons')}))


if __name__=='__main__':main()
