"""Expose the already-published v9 test as a regression, never as a fresh test."""
import argparse
from pathlib import Path
from roguard.trajectory_data import read_jsonl,write_jsonl,FORMAT


def build(root):
    rows=[]
    for lang in ('ro','uk'):
        for r in read_jsonl(root/f'research/v9-stabilization/paired-scope-117/data/{lang}-test.jsonl'):
            rows.append({'format':FORMAT,'case_id':'regression-'+r['case_id'],'language':lang,
                'turns':r['turns'][:r['focus']+1],'split':'challenge','family':'v9-regression-'+r.get('content_family','unknown'),
                'author':'project-v9','generator':'paired-scope-117','synthetic':True,
                'checkpoints':[{'focus':r['focus'],'review':r['review'],'world':r['attributes']}]})
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise SystemExit('Refusing to overwrite regression data')
    write_jsonl(a.output,build(a.root))
