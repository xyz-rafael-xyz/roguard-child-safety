"""Tokenize and bind the authored pilot screen without loading model weights."""
import argparse
from collections import Counter
import json
from pathlib import Path

from roguard.assurance import digest
from roguard.trajectory_data import file_sha
from training.screen_large_corpus_v10 import SYSTEM, judge_prompt, ordered_items, validated_rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--private',type=Path,required=True)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists():raise ValueError('Refusing to overwrite preflight')
    amendment=a.root/'research/v10-evidence/protocol-amendment-64.json'
    plan=json.loads(amendment.read_text())
    if a.base.name!=plan['judge']['revision']:raise ValueError('Judge revision differs')
    for name,expected in plan['bindings'].items():
        if file_sha(a.root/name)!=expected:raise ValueError('Screen binding changed: '+name)
    inventory={p.name:file_sha(p) for p in sorted(a.base.iterdir()) if p.is_file()}
    if inventory!=plan['model_files']:raise ValueError('Judge snapshot changed')
    source=a.private/'authored-response-pilot-66/families-revision1.jsonl'
    report=json.loads((a.root/'research/v10-evidence/authored-response-pilot-66/report.json').read_text())
    if file_sha(source)!=report['families_sha256']:raise ValueError('Pilot source changed')
    rows=validated_rows(source)
    if len(rows)!=16 or any(r['request']['partition']!='train' or not r['accepted'] for r in rows):
        raise ValueError('Expected exactly16 structurally accepted TRAIN families')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(a.base),trust_remote_code=False,local_files_only=True)
    queries=[];orders=Counter()
    for row in rows:
        for language in ('ro','uk'):
            prompt=judge_prompt(row,language)
            changed={**row,'request':{**row['request'],'category':'withheld-label-placeholder'}}
            if judge_prompt(changed,language)!=prompt:raise ValueError('Expected category changes judge input')
            messages=[{'role':'system','content':SYSTEM},{'role':'user','content':prompt}]
            tokens=tokenizer.apply_chat_template(messages,tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False)
            if not 1<=len(tokens)<=2048:raise ValueError('Prompt exceeds screen envelope')
            orders[ordered_items(row,language)[0][0]]+=1
            queries.append({'key':row['request']['family']+':'+language,'tokens':len(tokens),'query_token_sha256':digest(tokens),'expected_category_withheld':True})
    if dict(orders)!=report['screen_first_position_counts']:raise ValueError('Blinded ordering differs')
    output={'complete':True,'verification':'PASS','source_sha256':file_sha(__file__),'amendment_sha256':file_sha(amendment),
            'corpus_sha256':file_sha(source),'model_files_verified':len(inventory),'model_files':inventory,
            'queries':queries,'maximum_query_tokens':max(q['tokens'] for q in queries),'first_position_counts':dict(orders),
            'tokenizer_loaded':True,'model_loaded':False,'screen_launched':False,'training_ready':False,
            'semantic_labels_verified':False,'independent_annotation':False,'reserved_test_opened':False,'stable_release_ready':False}
    a.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'verification':'PASS','queries':len(queries),'maximum_query_tokens':output['maximum_query_tokens'],'model_loaded':False}))


if __name__=='__main__':main()
