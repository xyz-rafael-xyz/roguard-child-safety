"""Development-only spelling and code-switch challenges from fictional families."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

TYPO={'ro':{'singur':'singuur','sprijin':'sprijinn','vorbesc':'vorbeesc','trist':'triist'},
      'uk':{'самотньо':'самоотньо','підтримки':'підтриимки','поговорити':'поговориити','сумно':'суумно'}}


def challenges(rows):
    result=[]
    for source in rows:
        if source['variant']!='original':continue
        for variant in ('original','lowercase','romanian_without_diacritics','typing_error','mixed_marker','english_prior'):
            row=deepcopy(source);text=row['turns'][row['focus']]['text'];new=text
            if variant=='lowercase':new=text.lower()
            elif variant=='romanian_without_diacritics':
                if row['language']!='ro':continue
                new=text.translate(str.maketrans('ăâîșțĂÂÎȘȚ','aaistAAIST'))
            elif variant=='typing_error':
                for word,replacement in TYPO[row['language']].items():
                    if word in new:
                        new=new.replace(word,replacement,1);break
            elif variant=='mixed_marker':new='btw, '+text
            elif variant=='english_prior':
                row['turns'].insert(0,{'role':'minor','text':'Can you help me with a geography exercise?'})
                row['focus']+=1
            if variant not in ('original','english_prior') and new==text:continue
            row['turns'][row['focus']]['text']=new
            row['case_id']=source['case_id']+'-robust-'+variant
            row['variant']=variant;row['source']='authored_robustness_challenge_v8'
            result.append(row)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);manifest={'synthetic_only':True,'reserved_test_read':False,'independent_review':False,'splits':{}}
    for split in ('train','dev'):
        path=Path('research/v8-stabilization/context-families-01')/(split+'.jsonl')
        rows=challenges([json.loads(l) for l in path.read_text(encoding='utf-8').splitlines()])
        raw=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
        (a.output/(split+'.jsonl')).write_text(raw,encoding='utf-8')
        manifest['splits'][split]=dict(rows=len(rows),sha256=hashlib.sha256(raw.encode()).hexdigest(),families=sorted({r['family'] for r in rows}))
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8');print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
