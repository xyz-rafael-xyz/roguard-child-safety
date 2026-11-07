"""Adapt retained assistant-authored TRAIN drafts for the existing blinded screen.

The generator.model marker describes authorship for the legacy screen interface;
this was not a pinned local-generator run and no model-generation trace is claimed.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from roguard.trajectory_data import file_sha
from training.generate_large_corpus_v10 import parse
from training.screen_large_corpus_v10 import judge_prompt, ordered_items, validated_rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private',type=Path,required=True)
    parser.add_argument('--study',type=Path,required=True)
    args=parser.parse_args();pilot=args.private/'authored-response-pilot-66'
    inspection=json.loads((pilot/'revision1-inspection.json').read_text())
    source=pilot/'train-revision1.jsonl'
    if file_sha(source)!=inspection['revised_draft_sha256']:
        raise ValueError('Revised pilot changed')
    rows=[json.loads(s) for s in source.read_text().splitlines()]
    if len(rows)!=64 or len({r['case_id'] for r in rows})!=64:
        raise ValueError('Wrong pilot population')
    families=defaultdict(list)
    for row in rows:
        if row['partition']!='train' or row['context']!='':
            raise ValueError('Expected TRAIN-only context-free drafts')
        families[row['family']].append(row)
    records=[];manifest=[];order_counts=defaultdict(int)
    for family,group in sorted(families.items()):
        category=family.split(':')[1]
        if category not in ('boundary','privacy','pressure','unsupported') or len(group)!=4:
            raise ValueError('Invalid paired family')
        expected={(lang,gold) for lang in ('ro','uk') for gold in ('appropriate',category)}
        if {(r['language'],r['gold']) for r in group}!=expected:
            raise ValueError('Incomplete bilingual contrast')
        request={'family':family,'partition':'train','task':'response_behavior','category':category}
        texts={r['language']+('_appropriate' if r['gold']=='appropriate' else '_problematic'):r['draft'] for r in group}
        if parse(json.dumps(texts),request)!=texts:
            raise ValueError('Screen normalization changes a draft')
        record={'request':request,'accepted':True,'acceptance_scope':'Structural schema only; labels remain authored and unverified.',
                'texts':texts,'provenance':'assistant-authored-TRAIN-pilot-revision1'}
        for language in ('ro','uk'):
            query=judge_prompt(record,language)
            if family in query or 'authored66' in query:
                raise ValueError('Family/category metadata leaked into judge prompt')
            order_counts[ordered_items(record,language)[0][0]]+=1
        records.append(record);manifest.append(request)
    if len(records)!=16 or set(order_counts)!={'appropriate','problematic'}:
        raise ValueError('Incorrect family coverage or fixed label position')
    out=pilot/'families-revision1.jsonl'
    targets=[out,args.study/'protocol.json',args.study/'manifest.json',args.study/'report.json']
    if any(p.exists() for p in targets):raise ValueError('Refusing to overwrite authored screen preparation')
    out.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in records))
    if validated_rows(out)!=records:raise ValueError('Legacy screen rejects source rows')
    protocol={'format':'roguard-authored-response-pilot-1','source_sha256':file_sha(__file__),
              'generator':{'model':'assistant-authored','pinned_model_identity_available':False},
              'provenance':'Assistant-authored drafting and explicit two-row revision; not a pinned local-generator run.',
              'initial_draft_sha256':inspection['initial_draft_sha256'],'revised_draft_sha256':file_sha(source),
              'initial_inspection_sha256':inspection['initial_inspection_sha256'],'revision_inspection_sha256':file_sha(pilot/'revision1-inspection.json'),
              'families_sha256':file_sha(out),'training_ready':False,'semantic_labels_verified':False,'stable_release_ready':False}
    (args.study/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (args.study/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report={'complete':True,'families':16,'core_drafts':64,'partition':'train','families_sha256':file_sha(out),
            'protocol_sha256':file_sha(args.study/'protocol.json'),'manifest_sha256':file_sha(args.study/'manifest.json'),
            'screen_first_position_counts':dict(order_counts),'structural_acceptance_only':True,
            'native_human_validation':False,'training_ready':False,'semantic_labels_verified':False,
            'screen_launched':False,'reserved_test_opened':False,'stable_release_ready':False}
    (args.study/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
