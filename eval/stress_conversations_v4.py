"""Development-only perturbation diagnostic; not an independent accuracy test."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import subprocess
import unicodedata
from pathlib import Path
from roguard.conversation import ConversationScorer, analyze_batch
from training.conversations_v4b import generate
from roguard.conversation import render
from roguard.semantic import normalize_semantic_text


def variants(payload):
    def change(name, transform):
        value=copy.deepcopy(payload)
        for turn in value['turns']:
            turn['text']=transform(turn['text'])
        return name,value
    yield change('upper_case',str.upper)
    yield change('extra_spaces',lambda text:text.replace(' ','  '))
    yield change('unicode_decomposed',lambda text:unicodedata.normalize('NFD',text))
    yield change('punctuation',lambda text:text+' ...')
    if payload['language']=='ro':
        yield change('without_diacritics',lambda text:''.join(c for c in unicodedata.normalize('NFD',text) if not unicodedata.combining(c)))
    else:
        yield change('mixed_latin_i',lambda text:text.replace('і','i'))
    for name in ['english_mixture','unknown_speaker','token_overflow']:
        value=copy.deepcopy(payload)
        if name=='unknown_speaker':value['turns'][0]['role']='unknown'
        elif name=='english_mixture':value['turns'][-1]['text']+=' Please help with this.'
        else:value['turns']=[{'role':'minor','text':('Am nevoie de sprijin. ' if payload['language']=='ro' else 'Мені потрібна підтримка. ')*70}]
        yield name,value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Preserve prior diagnostic output')
    import torch
    torch.set_num_threads(4)
    selection=json.loads(Path('research/v4b-conversation-selection/selection.json').read_text())
    for name,digest in selection['artifact_sha256'].items():
        assert hashlib.sha256((Path('src/roguard')/name).read_bytes()).hexdigest()==digest
    holdout_texts=[render(r['turns']) for r in generate('test')]
    assert all(normalize_semantic_text(text)==text for text in holdout_texts), 'Re-evaluation required: holdout encoder inputs changed'
    rows=generate('dev')
    payloads=[]
    for language in ['ro','uk']:
        payloads.extend({'language':r['language'],'turns':r['turns']} for r in [r for r in rows if r['language']==language][:10])
    diagnostic={}
    for backend in ['fast','semantic']:
        scorer=ConversationScorer(backend,args.base)
        baseline=analyze_batch(payloads,scorer)
        changes=[(i,name,value) for i,p in enumerate(payloads) for name,value in variants(p)]
        reports=analyze_batch([value for _,_,value in changes],scorer)
        summary={}
        for (index,name,_),report in zip(changes,reports):
            key=payloads[index]['language']+'/'+name
            item=summary.setdefault(key,{'rows':0,'decision_changes':0,'abstentions':0,'max_score_change':0.0})
            item['rows']+=1
            item['decision_changes']+=report['signal']['decision']!=baseline[index]['signal']['decision']
            item['abstentions']+=report['signal']['decision']=='INSUFFICIENT_EVIDENCE'
            before=baseline[index]['signal']['synthetic_score'];after=report['signal']['synthetic_score']
            if before is not None and after is not None:item['max_score_change']=max(item['max_score_change'],abs(before-after))
            if name in ['english_mixture','unknown_speaker'] or (name=='token_overflow' and backend=='semantic'):
                assert report['signal']['decision']=='INSUFFICIENT_EVIDENCE'
            assert not report['authority_from_model'] and not report['external_action_taken']
        diagnostic[backend]=summary
    args.output.write_text(json.dumps({'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'runtime_sha256':{name:hashlib.sha256((Path('src/roguard')/name).read_bytes()).hexdigest() for name in ['semantic.py','conversation.py']},
        'unchanged_holdout_encoder_inputs':len(holdout_texts),'development_only':True,'base_rows':len(payloads),'variants_per_backend':len(changes),
        'synthetic_only':True,'artifact_sha256':selection['artifact_sha256'],'results':diagnostic},indent=2)+'\n')
    print('Diagnostic complete:',args.output)

if __name__=='__main__':main()
