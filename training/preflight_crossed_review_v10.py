"""Check new binary token inputs and family partitions without loading an encoder."""
import argparse
from collections import Counter
import json
from pathlib import Path

from roguard.assurance import digest
from roguard.gemma4_contract import classifier_tokens
from roguard.trajectory_data import file_sha,read_jsonl


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','data_study','data_private','feature_study','features','output'):
        p.add_argument('--'+name.replace('_','-'),dest=name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Refusing to overwrite token preflight')
    data=json.loads((a.data_study/'report.json').read_text());parent=json.loads((a.feature_study/'report.json').read_text())
    if not data['complete'] or not parent['complete']:raise ValueError('Incomplete sources')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(a.base,local_files_only=True)
    seen=set();families={};counts=Counter();maximum=0
    for lang in ('ro','uk'):
        path=a.features/(lang+'-rows.jsonl')
        if file_sha(path)!=parent['languages'][lang]['artifacts']['rows.jsonl']:raise ValueError('Parent metadata binding differs')
        for row in read_jsonl(path):
            seen.add((lang,row['token_sha256']))
            if row.get('family'):
                if families.setdefault(row['family'],row['partition'])!=row['partition']:raise ValueError('Parent family crosses partitions')
    for split in ('train','dev'):
        path=a.data_private/(split+'.jsonl')
        if file_sha(path)!=data['artifacts'][split]:raise ValueError('New data binding differs')
        for row in read_jsonl(path):
            if row['partition']!=split or families.setdefault(row['family'],split)!=split:
                raise ValueError('New family crosses parent or new partitions')
            tokens=classifier_tokens(tokenizer,{'language':row['language'],'turns':row['turns']})
            if not tokens or len(tokens)>1024:raise ValueError('New input exceeds model envelope')
            key=(row['language'],digest(tokens))
            if key in seen:raise ValueError('New token input duplicates existing input')
            seen.add(key);maximum=max(maximum,len(tokens));counts[split+':'+row['language']]+=1
    result={'complete':True,'source_sha256':file_sha(__file__),'data_report_sha256':file_sha(a.data_study/'report.json'),
            'parent_feature_report_sha256':file_sha(a.feature_study/'report.json'),
            'counts':dict(counts),'maximum_tokens':maximum,'new_token_duplicates':0,'parent_family_partitions_preserved':True,
            'test_opened':False,'encoder_loaded':False,'stable_release_ready':False}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
